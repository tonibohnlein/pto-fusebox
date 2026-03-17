#include "search/verbose.h"
#include "search/fm_pass.h"
#include "search/local_search.h"  // partition_has_gap
#include <algorithm>
#include <iostream>

// ============================================================================
// Collect all activatable ops (border + internal from large groups)
// ============================================================================

static std::vector<size_t> all_activatable_ops(const Partition& part) {
    std::set<size_t> ops;
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
    std::vector<size_t> to_lock = {m.op};

    // Find ops in gb that are neighbors of m.op (DAG edges + co-consumers)
    if (m.gb < part.groups.size() && part.groups[m.gb].alive) {
        for (auto nbr : part.dag->op_neighbors[m.op]) {
            if (part.groups[m.gb].ops.count(nbr))
                to_lock.push_back(nbr);
        }
    }
    return to_lock;
}

// ============================================================================
// FM inner pass
// ============================================================================

FMPassResult fm_inner_pass(Partition part, const FMConfig& cfg) {
    FMPassResult result;
    result.start_cost = part.total_cost();
    result.best_cost = result.start_cost;
    result.best_partition = part;  // initial snapshot

    double floor = result.start_cost * cfg.floor_fraction;
    double max_drift = result.start_cost * cfg.max_drift_fraction;

    // Step 1: activate random subset of border ops
    auto candidates = all_activatable_ops(part);
    auto initial = random_subset_n(candidates, cfg.init_count, cfg.seed);

    ActiveSet active(part, floor);
    for (auto op : initial)
        active.activate(op);

    // Step 2: main loop
    double cumulative_gain = 0;
    double best_cumulative_gain = 0;

    while (true) {
        // Pop the best unlocked move
        auto move_opt = active.pop_best();
        if (!move_opt.has_value()) break;

        FMMove move = *move_opt;

        // Additional locking: collect ops that should be locked BEFORE apply
        // (since apply changes the groups)
        std::vector<size_t> extra_locks;
        if (move.type == FMMove::MERGE) {
            extra_locks = merge_lock_ops(part, move);
        } else if (move.type == FMMove::TENSOR_MERGE
                || move.type == FMMove::TENSOR_EXTRACT) {
            // Lock ALL ops in ALL groups involved in the tensor move.
            // These ops are all being relocated — they must not initiate
            // further moves this pass.
            for (auto cg : move.tensor_groups)
                if (part.groups[cg].alive)
                    for (auto cop : part.groups[cg].ops)
                        extra_locks.push_back(cop);
        }

        // Apply the move — snapshot for gap-revert
        Partition snapshot = part;
        double total_before = part.total_cost();
        auto affected = apply_fm_move(part, move);
        if (affected.empty()) {
            part = std::move(snapshot);
            continue;
        }
        if (partition_has_gap(part)) {
            part = std::move(snapshot);
            continue;
        }

#ifndef NDEBUG
        {
            double total_after = part.total_cost();
            double actual_gain = total_before - total_after;
            double discrepancy = move.saving - actual_gain;
            if (std::abs(discrepancy) > 0.1 * std::max(1.0, std::abs(move.saving)) + 1.0) {
                if (g_verbose)
                    std::cerr << "    FM GAIN MISMATCH: predicted=" << move.saving
                              << " actual=" << actual_gain
                              << " Δ=" << discrepancy
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

        // Update cumulative gain
        double new_cost = part.total_cost();
        cumulative_gain = result.start_cost - new_cost;

        // Check for new best in this pass
        if (new_cost < result.best_cost - 0.001) {
            result.best_cost = new_cost;
            result.best_partition = part;  // snapshot
            best_cumulative_gain = cumulative_gain;
        }

        // Check max drift: if we've dropped too far below the pass-best, abort
        if (best_cumulative_gain - cumulative_gain > max_drift) break;

        // Lock extra ops (merge/tensor partners).
        // active.pop_best() already locked move.op.
        active.lock_all(extra_locks);

        // Update affected ops + activate new border ops (combined)
        active.refresh_after_move(affected);
    }

    // Capture the final (maximally perturbed) state
    result.end_partition = part;
    result.end_cost = part.total_cost();

    return result;
}