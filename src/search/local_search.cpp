#include "search/verbose.h"
#include "search/local_search.h"
#include "search/fm_search.h"
#include "search/feasibility.h"
#include "util/pairing_heap.h"
#include <iostream>
#include <cassert>

// ============================================================================
// Greedy descent with pairing heap — true steepest descent
// Uses best_move_for() and apply_fm_move() from fm_search (unified evaluator).
// ============================================================================

Partition greedy_descent(Partition part) {
    const size_t num_ops = part.prob->num_ops();
    PairingHeap<FMMove> heap(num_ops);

    // Initialize: best move per op (no locked ops for greedy)
    for (size_t op = 0; op < num_ops; op++) {
        auto m = best_move_for(part, op);
        if (m.valid() && m.saving > 0.001)
            heap.push_or_update(op, m);
    }

    int applied = 0;
    int rejected = 0;
    int iterations = 0;
    const int MAX_ITERATIONS = 1000;  // safety cap
    while (!heap.empty() && iterations < MAX_ITERATIONS) {
        iterations++;
        auto m_opt = heap.pop_best();
        if (!m_opt || m_opt->saving <= 0.001) break;
        FMMove& m = *m_opt;

#ifndef NDEBUG
        double old_total = part.total_cost();
#endif

        auto affected = apply_fm_move(part, m);
        if (affected.empty()) {
            rejected++;
            continue;
        }

#ifndef NDEBUG
        {
            double actual_gain = old_total - part.total_cost();
            double discrepancy = m.saving - actual_gain;
            if (std::abs(discrepancy) > 0.1 * std::max(1.0, std::abs(m.saving)) + 1.0) {
                std::cerr << "    GREEDY GAIN MISMATCH: predicted=" << m.saving
                          << " actual=" << actual_gain
                          << " type=" << (int)m.type
                          << " op=" << m.op << "\n";
            }
        }
#endif

        applied++;

        // Collect affected ops: ops in affected groups + their DAG neighbors
        FlatSet<size_t> affected_ops;
        affected_ops.insert(m.op);
        for (auto nbr : part.dag->op_neighbors[m.op])
            affected_ops.insert(nbr);
        for (auto gi : affected) {
            if (!part.groups[gi].alive) continue;
            for (auto op : part.groups[gi].ops) {
                affected_ops.insert(op);
                for (auto nbr : part.dag->op_neighbors[op])
                    affected_ops.insert(nbr);
            }
        }

        // Update each affected op in the heap
        for (auto op : affected_ops) {
            auto fresh = best_move_for(part, op);
            if (fresh.valid() && fresh.saving > 0.001)
                heap.push_or_update(op, fresh);
            else
                heap.remove(op);
        }
    }

    if (g_verbose) {
        std::cerr << "    greedy: " << iterations << " iters, "
                  << applied << " applied, " << rejected << " rejected, cost="
                  << part.total_cost() << "\n";
    }
    return part;
}

bool partition_has_gap(const Partition& part, std::function<bool(size_t)> is_retained) {
    if (!part.is_acyclic()) return true;
    if (!part.prob || !part.dag) return false;
    const auto& prob = *part.prob;
    const auto& dag = *part.dag;

    // Coverage check: every op must be in at least one alive group.
    // A missing op makes total_cost() artificially low (zero contribution)
    // and will fail Solution::validate ("Op N not covered").
    for (size_t op = 0; op < prob.num_ops(); op++) {
        bool covered = false;
        for (auto gi : part.groups_of(op))
            if (part.groups[gi].alive) { covered = true; break; }
        if (!covered) return true;
    }

    // Mixed-consumer check: if a tensor T is produced AND consumed inside a
    // group, T is ephemeral (never written to slow memory).  Every group that
    // needs T from slow memory must be covered:
    //   (a) external consumers: ops outside gi that consume T
    //   (b) recomputed internal consumers: ops inside gi that consume T but
    //       also exist in OTHER groups — those other copies need T from slow
    //       memory too
    //
    // Coverage means either:
    //   (i)  the consumer's group also contains the producer (recomputed), OR
    //   (ii) another alive group exports T as boundary output
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        for (auto op : part.groups[gi].ops) {
            {
                size_t t = prob.ops[op].output();
                bool consumed_internal = false;
                for (auto cop : dag.tensor_consumers[t])
                    if (cop != op && part.groups[gi].ops.count(cop))
                        { consumed_internal = true; break; }
                if (!consumed_internal) continue;

                // T is ephemeral in gi.  Check all groups that need T from
                // slow memory: external consumers + recomputed internal consumers.

                // Helper: is T available from slow memory?  True if some alive
                // group exports T as boundary output (produced, not consumed).
                auto t_available_from_slow = [&]() -> bool {
                    for (auto gj : part.groups_of(op)) {
                        if (gj == gi || !part.groups[gj].alive) continue;
                        if (!is_boundary_output_of(part.groups[gj].ops, t, dag))
                            continue;
                        return true;
                    }
                    return false;
                };

                // If T is retained across a coupling edge, the next step
                // gets it from fast memory — no gap.
                if (is_retained && is_retained(t)) continue;

                // (a) External consumers
                for (auto cop : dag.tensor_consumers[t]) {
                    if (part.groups[gi].ops.count(cop)) continue;
                    bool covered = false;
                    // consumer's group recomputes producer?
                    for (auto gj : part.groups_of(cop)) {
                        if (!part.groups[gj].alive) continue;
                        if (part.groups[gj].ops.count(op)) { covered = true; break; }
                    }
                    if (!covered) covered = t_available_from_slow();
                    if (!covered) return true;
                }

                // (b) Internal consumers that are recomputed in other groups:
                // those other copies need T from slow memory.
                for (auto cop : dag.tensor_consumers[t]) {
                    if (!part.groups[gi].ops.count(cop)) continue;  // external, handled above
                    // Is cop recomputed in another group?
                    for (auto gj : part.groups_of(cop)) {
                        if (gj == gi || !part.groups[gj].alive) continue;
                        // gj has a copy of cop that needs T.
                        // Covered if gj also has the producer.
                        if (part.groups[gj].ops.count(op)) continue;
                        // Not covered by recomputation — need T from slow memory.
                        if (!t_available_from_slow()) return true;
                    }
                }
            }
        }
    }

    return false;
}
