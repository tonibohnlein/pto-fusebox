#include "search/verbose.h"
#include "search/fm_pass.h"
#include <algorithm>
#include <cassert>
#include <iostream>

// ============================================================================
// Collect all activatable ops (border + internal from large groups)
// ============================================================================

static std::vector<size_t> all_activatable_ops(const Partition& part) {
    FlatSet<size_t> ops;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        for (auto op : part.border_ops(gi))
            ops.insert(op);
        if (part.groups[gi].ops.size() >= 3) {
            for (auto op : part.internal_ops(gi))
                ops.insert(op);
        }
    }
    return {ops.begin(), ops.end()};
}

// ============================================================================
// Select N random border ops
// ============================================================================

static std::vector<size_t> random_subset_n(const std::vector<size_t>& ops,
                                            int count, unsigned seed) {
    if (count >= (int)ops.size() || ops.size() <= 3) return ops;

    std::vector<size_t> shuffled = ops;
    std::mt19937 rng(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);

    size_t n = std::max<size_t>(1, (size_t)count);
    shuffled.resize(n);
    return shuffled;
}

// ============================================================================
// Lock ops involved in a merge: the initiating op + the boundary ops from
// the partner group that connect to the initiating op.
// ============================================================================

static std::vector<size_t> merge_lock_ops(const Partition& part, const FMMove& m) {
    std::vector<size_t> to_lock;
    if (m.ga < part.groups.size() && part.groups[m.ga].alive)
        for (auto op : part.groups[m.ga].ops) to_lock.push_back(op);
    if (m.gb < part.groups.size() && part.groups[m.gb].alive)
        for (auto op : part.groups[m.gb].ops) to_lock.push_back(op);
    return to_lock;
}

// ============================================================================
// FM inner pass
// ============================================================================

FMPassResult fm_inner_pass(Partition part, const FMConfig& cfg) {
    FMPassResult result;
    result.start_cost = part.total_cost();
    result.best_cost = result.start_cost;

    // Lightweight checkpoint: store only groups (ops + cost + alive).
    // Avoids copying op_to_groups_, group DAG, etc. on every improvement.
    // Restored into a full Partition at the end.
    auto best_groups = part.groups;

    double max_drift = result.start_cost * cfg.max_drift_fraction;
    double floor = result.start_cost * cfg.floor_fraction;

    // Step 1: activate random subset of border ops
    auto candidates = all_activatable_ops(part);
    auto initial = random_subset_n(candidates, cfg.init_count, cfg.seed);

    ActiveSet active(part, floor);
    for (auto op : initial)
        active.activate(op);

    // Step 2: main loop
    double cumulative_gain = 0;
    double best_cumulative_gain = 0;
    int fm_iters = 0;
    int consecutive_non_improving = 0;
    using SteadyClock = std::chrono::steady_clock;

    // Per-type counters: [type] → {applied, failed}
    constexpr int NUM_TYPES = 10;
    int type_applied[NUM_TYPES] = {};
    int type_failed[NUM_TYPES] = {};
    const char* stop_reason = "max_iters";

    while (true) {
        fm_iters++;

        // Check deadline every 32 iterations (clock syscall amortization)
        if ((fm_iters & 31) == 0 && SteadyClock::now() >= cfg.deadline) {
            stop_reason = "deadline";
            break;
        }

        auto move_opt = active.pop_best();
        if (!move_opt.has_value()) {
            stop_reason = "exhausted";
            break;
        }

        FMMove move = *move_opt;

        // Additional locking: collect ops that should be locked BEFORE apply
        std::vector<size_t> extra_locks;
        if (move.type == FMMove::MERGE) {
            extra_locks = merge_lock_ops(part, move);
        } else if (move.type == FMMove::TENSOR_MERGE
                || move.type == FMMove::TENSOR_EXTRACT) {
            for (auto cg : move.tensor_groups)
                if (part.groups[cg].alive)
                    for (auto cop : part.groups[cg].ops)
                        extra_locks.push_back(cop);
        } else if (move.type == FMMove::SPLIT) {
            if (move.ga < part.groups.size() && part.groups[move.ga].alive)
                for (auto op : part.groups[move.ga].ops) extra_locks.push_back(op);
        } else if (move.type == FMMove::STEAL) {
            extra_locks.push_back(move.op);
            // Lock DAG neighbors of the moved op in both groups
            for (auto nbr : part.dag->op_neighbors[move.op]) {
                if (part.groups[move.ga].ops.count(nbr) || part.groups[move.gb].ops.count(nbr))
                    extra_locks.push_back(nbr);
            }
        } else if (move.type == FMMove::EJECT || move.type == FMMove::INTERNAL_EJECT) {
            extra_locks.push_back(move.op);
            // Lock DAG neighbors of the ejected op in ga (prevent immediate re-merge)
            for (auto nbr : part.dag->op_neighbors[move.op]) {
                if (part.groups[move.ga].ops.count(nbr))
                    extra_locks.push_back(nbr);
            }
        } else if (move.type == FMMove::RECOMPUTE) {
            extra_locks.push_back(move.op);
        } else if (move.type == FMMove::DE_RECOMPUTE) {
            extra_locks.push_back(move.op);
        } else if (move.type == FMMove::FORCE_RECOMPUTE) {
            for (auto cop : move.tensor_consumer_ops)
                extra_locks.push_back(cop);
        }

        double total_before = part.total_cost();
        auto affected = apply_fm_move(part, move);
        if (affected.empty()) {
            if (move.type >= 0 && move.type < NUM_TYPES) type_failed[move.type]++;
            continue;
        }
        if (move.type >= 0 && move.type < NUM_TYPES) type_applied[move.type]++;

#ifndef NDEBUG
        {
            double actual_gain = total_before - part.total_cost();
            double discrepancy = move.saving - actual_gain;
            if (std::abs(discrepancy) > 0.1 * std::max(1.0, std::abs(move.saving)) + 1.0) {
                std::cerr << "    PARTITION FM GAIN MISMATCH: predicted=" << move.saving
                          << " actual=" << actual_gain
                          << " type=" << (int)move.type
                          << " op=" << move.op
                          << " ga=" << move.ga << " gb=" << move.gb << "\n";
            }
        }
#endif

        result.moves_applied++;
        if (move.saving > 0.001)
            result.moves_positive++;
        else if (move.saving < -0.001)
            result.moves_negative++;

        double new_cost = part.total_cost();
        cumulative_gain = result.start_cost - new_cost;

        if (new_cost < result.best_cost - 0.001) {
            result.best_cost = new_cost;
            best_groups = part.groups;
            best_cumulative_gain = cumulative_gain;
            consecutive_non_improving = 0;
        } else {
            consecutive_non_improving++;
        }

        if (best_cumulative_gain - cumulative_gain > max_drift) {
            stop_reason = "drift";
            break;
        }
        if (consecutive_non_improving >= cfg.max_consecutive_non_improving) {
            stop_reason = "no_improve";
            break;
        }

        active.lock_all(extra_locks);
        active.refresh_after_move(affected);
    }

    result.end_cost = part.total_cost();
    result.end_partition = std::move(part);

    // Restore best partition from lightweight checkpoint.
    // Only the groups snapshot was saved — reconstruct the full Partition
    // from pointers + groups + rebuild_index (avoids copying op_to_groups_).
    result.best_partition.prob = result.end_partition.prob;
    result.best_partition.dag = result.end_partition.dag;
    result.best_partition.cache = result.end_partition.cache;
    result.best_partition.groups = std::move(best_groups);
    result.best_partition.rebuild_index();

    if (g_verbose) {
        static const char* type_names[] = {
            "STEAL", "EJECT", "RECOMP", "MERGE", "INT_EJECT",
            "SPLIT", "T_MERGE", "T_EXTRACT", "DE_RECOMP", "FORCE_RECOMP"
        };
        std::cerr << "      FM pass: " << fm_iters << " iters, "
                  << result.moves_applied << " applied ("
                  << result.moves_positive << "+, " << result.moves_negative << "-), "
                  << "best=" << result.best_cost
                  << " end=" << result.end_cost
                  << " [" << stop_reason << "]\n";
        std::cerr << "        moves:";
        for (int t = 0; t < NUM_TYPES; t++) {
            if (type_applied[t] || type_failed[t])
                std::cerr << " " << type_names[t] << "=" << type_applied[t]
                          << "(" << type_failed[t] << "f)";
        }
        std::cerr << "\n";
    }

    return result;
}