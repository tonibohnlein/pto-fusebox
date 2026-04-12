#include "search/coupled_fm_pass.h"
#include "search/verbose.h"
#include <algorithm>
#include <iostream>

// ============================================================================
// Collect all activatable ops (border + internal from larger groups)
// ============================================================================

static std::vector<size_t> all_activatable_ops_coupled(const CoupledPartition& cp) {
    FlatSet<size_t> ops;
    const auto& part = cp.part;
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

static std::vector<size_t> random_subset_n_coupled(const std::vector<size_t>& ops,
                                                     int count, unsigned seed) {
    if (count >= (int)ops.size() || ops.size() <= 3) return ops;
    std::vector<size_t> shuffled = ops;
    std::mt19937 rng(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    shuffled.resize(std::max<size_t>(1, (size_t)count));
    return shuffled;
}

// Lock ops involved in a MERGE move (same as partition fm_pass)
static std::vector<size_t> merge_lock_ops_coupled(const CoupledPartition& cp,
                                                    const CoupledFMMove& m) {
    std::vector<size_t> to_lock = {m.op};
    const auto& part = cp.part;
    if (m.gb < part.groups.size() && part.groups[m.gb].alive) {
        for (auto nbr : part.dag->op_neighbors[m.op]) {
            if (part.groups[m.gb].ops.count(nbr))
                to_lock.push_back(nbr);
        }
    }
    return to_lock;
}

// ============================================================================
// coupled_fm_inner_pass
// ============================================================================

CoupledFMPassResult coupled_fm_inner_pass(CoupledPartition cp,
                                           const FlatSet<size_t>& feasibly_ret,
                                           const FMConfig& cfg) {
    CoupledFMPassResult result;
    result.start_cost = cp.total_cost();
    result.best_cost  = result.start_cost;

    // Lightweight checkpoint: groups + coupling state
    auto best_groups   = cp.part.groups;
    auto best_next     = cp.next_group;
    auto best_prev     = cp.prev_group;
    auto best_retained = cp.retained;

    double max_drift = result.start_cost * cfg.max_drift_fraction;
    double floor     = result.start_cost * cfg.floor_fraction;

    auto candidates = all_activatable_ops_coupled(cp);
    auto initial    = random_subset_n_coupled(candidates, cfg.init_count, cfg.seed);

    CoupledActiveSet active(cp, feasibly_ret, floor);
    for (auto op : initial)
        active.activate(op);

    double cumulative_gain           = 0;
    double best_cumulative_gain      = 0;
    int    fm_iters                  = 0;
    int    consecutive_non_improving = 0;
    using SteadyClock = std::chrono::steady_clock;

    constexpr int NUM_TYPES = 14;
    int type_applied[NUM_TYPES] = {};
    int type_failed[NUM_TYPES]  = {};
    const char* stop_reason = "max_iters";

    while (true) {
        fm_iters++;

        if ((fm_iters & 31) == 0 && SteadyClock::now() >= cfg.deadline) {
            stop_reason = "deadline";
            break;
        }

        auto move_opt = active.pop_best();
        if (!move_opt.has_value()) {
            stop_reason = "exhausted";
            break;
        }

        CoupledFMMove move = *move_opt;

        // Extra locks before apply
        std::vector<size_t> extra_locks;
        if (move.type == CoupledFMMove::MERGE) {
            extra_locks = merge_lock_ops_coupled(cp, move);
        } else if (move.type == CoupledFMMove::TENSOR_MERGE ||
                   move.type == CoupledFMMove::TENSOR_EXTRACT) {
            for (auto cg : move.tensor_groups)
                if (cp.part.groups[cg].alive)
                    for (auto cop : cp.part.groups[cg].ops)
                        extra_locks.push_back(cop);
        } else if (move.type == CoupledFMMove::RETAIN_FORCE_SPLIT) {
            // Lock the op that is being split around
            extra_locks.push_back(move.op);
            extra_locks.push_back(move.op2);
        } else if (move.type == CoupledFMMove::FORCE_RETAIN) {
            // Lock split boundary ops + tensor endpoints
            extra_locks.push_back(move.op2);  // op_a_dst (consumer of t)
            extra_locks.push_back(move.op3);  // op_b_dst (split partner)
            if (move.tensor < cp.part.dag->tensor_consumers.size()) {
                int prod = cp.part.dag->tensor_producer[move.tensor];
                if (prod >= 0) extra_locks.push_back((size_t)prod);
            }
        } else if (move.type == CoupledFMMove::EPHEMERAL_FUSE) {
            // Lock the extracted ops (P, C1) and consumers of T in g_c2
            // to prevent the FM pass from immediately undoing the fuse.
            extra_locks.push_back(move.op);   // P
            extra_locks.push_back(move.op2);  // C1
            if (move.tensor < cp.part.dag->tensor_consumers.size())
                for (auto cop : cp.part.dag->tensor_consumers[move.tensor])
                    if (move.ga < cp.part.groups.size() &&
                        cp.part.groups[move.ga].ops.count(cop))
                        extra_locks.push_back(cop);
        } else if (move.type == CoupledFMMove::COUPLE ||
                   move.type == CoupledFMMove::UNCOUPLE) {
            // Lock ops at the coupling boundary to prevent immediate reversal.
            // The tensor that's being coupled/uncoupled flows from ga to gb —
            // lock its producer in ga and consumers in gb.
            if (move.tensor < cp.part.dag->tensor_consumers.size()) {
                int prod = cp.part.dag->tensor_producer[move.tensor];
                if (prod >= 0) extra_locks.push_back((size_t)prod);
                for (auto cop : cp.part.dag->tensor_consumers[move.tensor])
                    extra_locks.push_back(cop);
            }
        }

        [[maybe_unused]] double total_before = 0;
#ifndef NDEBUG
        total_before = cp.total_cost();
#endif
        auto affected = apply_coupled_fm_move(cp, move);
        if (affected.empty()) {
            int t = (int)move.type;
            if (t >= 0 && t < NUM_TYPES) type_failed[t]++;
            continue;
        }

        int t = (int)move.type;
        if (t >= 0 && t < NUM_TYPES) type_applied[t]++;

#ifndef NDEBUG
        {
            double actual_gain  = total_before - cp.total_cost();
            double discrepancy  = move.saving - actual_gain;
            if (std::abs(discrepancy) > 0.1 * std::max(1.0, std::abs(move.saving)) + 1.0) {
                std::cerr << "    COUPLED FM GAIN MISMATCH: predicted=" << move.saving
                          << " actual=" << actual_gain
                          << " type=" << (int)move.type << " op=" << move.op
                          << " ga=" << move.ga << " gb=" << move.gb << "\n";
            }

            // Verify acyclicity invariant after every move.
            static bool first_violation_dumped = false;
            cp.part.rebuild_index();
            if (!cp.part.is_acyclic() && !first_violation_dumped) {
                first_violation_dumped = true;
                std::cerr << "    ACYCLICITY VIOLATED after move type="
                          << (int)move.type << " op=" << move.op
                          << " ga=" << move.ga << " gb=" << move.gb
                          << " saving=" << move.saving << "\n";
                std::cerr << "    POST-MOVE groups:\n";
                for (size_t gi = 0; gi < cp.part.groups.size(); gi++) {
                    if (!cp.part.groups[gi].alive) continue;
                    std::cerr << "      G" << gi << " ops={";
                    for (auto op2 : cp.part.groups[gi].ops) std::cerr << op2 << ",";
                    std::cerr << "}";
                    // Mark recomputed ops
                    for (auto op2 : cp.part.groups[gi].ops) {
                        auto gs = cp.part.groups_of(op2);
                        int alive_count = 0;
                        for (auto g : gs) if (cp.part.groups[g].alive) alive_count++;
                        if (alive_count > 1) std::cerr << " [op" << op2 << " recomp x" << alive_count << "]";
                    }
                    std::cerr << "\n";
                }
                // Also verify: was it acyclic BEFORE this move?
                // (We can't undo, but we can check the previous assertion.)
                std::cerr << "    NOTE: this is the first violation detected in this pass\n";
            }
        }
#endif

        result.moves_applied++;
        if (move.saving > 0.001)  result.moves_positive++;
        else if (move.saving < -0.001) result.moves_negative++;

        double new_cost  = cp.total_cost();
        cumulative_gain  = result.start_cost - new_cost;

        if (new_cost < result.best_cost - 0.001) {
            result.best_cost       = new_cost;
            best_groups            = cp.part.groups;
            best_next              = cp.next_group;
            best_prev              = cp.prev_group;
            best_retained          = cp.retained;
            best_cumulative_gain   = cumulative_gain;
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

    result.end_cost = cp.total_cost();
    result.end_cp   = std::move(cp);

    // Restore best state from checkpoint
    result.best_cp.part.prob    = result.end_cp.part.prob;
    result.best_cp.part.dag     = result.end_cp.part.dag;
    result.best_cp.part.cache   = result.end_cp.part.cache;
    result.best_cp.part.groups  = std::move(best_groups);
    result.best_cp.part.rebuild_index();
    result.best_cp.next_group   = std::move(best_next);
    result.best_cp.prev_group   = std::move(best_prev);
    result.best_cp.retained     = std::move(best_retained);

    if (g_verbose) {
        static const char* type_names[] = {
            "STEAL","EJECT","RECOMP","MERGE","INT_EJECT",
            "SPLIT","T_MERGE","T_EXTRACT","DE_RECOMP","FORCE_RECOMP",
            "COUPLE","UNCOUPLE","RFS","FORCE_RET","EPH_FUSE"
        };
        std::cerr << "      Coupled FM pass: " << fm_iters << " iters, "
                  << result.moves_applied << " applied ("
                  << result.moves_positive << "+, " << result.moves_negative << "-), "
                  << "best=" << result.best_cost
                  << " end=" << result.end_cost
                  << " [" << stop_reason << "]\n";
        std::cerr << "        moves:";
        for (int i = 0; i < NUM_TYPES; i++)
            if (type_applied[i] || type_failed[i])
                std::cerr << " " << type_names[i] << "=" << type_applied[i]
                          << "(" << type_failed[i] << "f)";
        std::cerr << "\n";
    }

    return result;
}
