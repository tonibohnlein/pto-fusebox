#include "search/fm_search.h"
#include <algorithm>
#include <iostream>
#include <cassert>

// ============================================================================
// Local cycle check for SPLIT: are the two new sides in a mutual dependency?
// This is correct for splits because both sides come from the same parent
// group — no third-party groups can create indirect cycles.
// ============================================================================

static bool split_creates_topo_cycle(const std::set<size_t>& side_a,
                                      const std::set<size_t>& side_b,
                                      const DAG& dag) {
    bool a_to_b = false, b_to_a = false;
    for (auto u : side_a) {
        if (a_to_b) break;
        for (auto v : dag.op_succs[u])
            if (side_b.count(v)) { a_to_b = true; break; }
    }
    if (!a_to_b) return false;
    for (auto u : side_b) {
        if (b_to_a) break;
        for (auto v : dag.op_succs[u])
            if (side_a.count(v)) { b_to_a = true; break; }
    }
    return b_to_a;
}

// ============================================================================
// best_move_for: evaluate all candidate moves for op
// ============================================================================

FMMove best_move_for(const Partition& part, size_t op,
                     double floor, const std::set<size_t>& locked) {
    if (locked.count(op)) return {};

    constexpr double EPSILON = 1.0;

    FMMove best;
    auto groups_of_x = part.groups_of(op);
    if (groups_of_x.empty()) return {};

    auto accept = [&](double saving) {
        return (saving > EPSILON || saving < -EPSILON) &&
               saving > -floor && saving > best.saving;
    };

    bool op_in_multiple = groups_of_x.size() > 1;

    // --- DE_RECOMPUTE: if op is in multiple groups, evaluate removing
    // singleton groups or fully-redundant groups ---
    if (op_in_multiple) {
        for (auto gx : groups_of_x) {
            if (!part.groups[gx].alive) continue;
            if (part.groups[gx].ops.size() == 1) {
                double saving = part.groups[gx].cost;
                if (accept(saving))
                    best = FMMove{FMMove::DE_RECOMPUTE, op, gx, SIZE_MAX, SIZE_MAX, saving};
            } else {
                bool all_covered = true;
                for (auto o : part.groups[gx].ops) {
                    bool in_other = false;
                    for (auto gj : part.groups_of(o))
                        if (gj != gx && part.groups[gj].alive) { in_other = true; break; }
                    if (!in_other) { all_covered = false; break; }
                }
                if (all_covered) {
                    double saving = part.groups[gx].cost;
                    if (accept(saving))
                        best = FMMove{FMMove::DE_RECOMPUTE, op, gx, SIZE_MAX, SIZE_MAX, saving};
                }
            }
        }
    }

    // --- Border moves: MERGE, STEAL, RECOMPUTE, EJECT ---
    for (auto gx : groups_of_x) {
        if (!part.is_border_op(op, gx)) continue;

        std::set<size_t> neighbor_groups;
        for (auto nbr : part.dag->op_neighbors[op])
            if (!part.groups[gx].ops.count(nbr))
                for (auto gy : part.groups_of(nbr))
                    if (gy != gx) neighbor_groups.insert(gy);

        for (auto gy : neighbor_groups) {
            bool x_in_gy = part.groups[gy].ops.count(op);

            // MERGE: use hypothetical acyclicity check
            if (part.is_acyclic_after_merge(gx, gy)) {
                std::set<size_t> merged = part.groups[gx].ops;
                merged.insert(part.groups[gy].ops.begin(), part.groups[gy].ops.end());
                double merged_cost = part.eval_set(merged);
                if (merged_cost < 1e17) {
                    double saving = (part.groups[gx].cost + part.groups[gy].cost) - merged_cost;
                    if (accept(saving))
                        best = FMMove{FMMove::MERGE, op, gx, gy, SIZE_MAX, saving};
                }
            }

            // STEAL + RECOMPUTE
            if (!x_in_gy) {
                std::set<size_t> new_gy = part.groups[gy].ops;
                new_gy.insert(op);
                double new_gy_cost = part.eval_set(new_gy);

                if (new_gy_cost < 1e17) {
                    // STEAL: use hypothetical acyclicity check
                    if (part.is_acyclic_after_steal(op, gx, gy)) {
                        std::set<size_t> new_gx = part.groups[gx].ops;
                        new_gx.erase(op);
                        double new_gx_cost = 0;
                        bool gx_valid = true;
                        if (new_gx.empty()) {
                            if (groups_of_x.size() <= 1) gx_valid = false;
                        } else {
                            new_gx_cost = part.eval_set(new_gx);
                            if (new_gx_cost >= 1e17) gx_valid = false;
                        }
                        if (gx_valid) {
                            double saving = (part.groups[gx].cost + part.groups[gy].cost)
                                            - (new_gx_cost + new_gy_cost);
                            if (accept(saving))
                                best = FMMove{FMMove::STEAL, op, gx, gy, SIZE_MAX, saving};
                        }
                    }

                    // RECOMPUTE: use hypothetical acyclicity check
                    if (part.is_acyclic_after_recompute(op, gy)) {
                        double saving = part.groups[gy].cost - new_gy_cost;
                        if (accept(saving))
                            best = FMMove{FMMove::RECOMPUTE, op, gx, gy, SIZE_MAX, saving};
                    }
                }
            }
        }

        // EJECT: cannot create cycles
        {
            auto er = part.eval_eject(op, gx);
            if (er.feasible && accept(er.saving))
                best = FMMove{FMMove::EJECT, op, gx, SIZE_MAX, SIZE_MAX, er.saving};
        }
    }

    // --- Internal moves: INTERNAL_EJECT and SPLIT ---
    for (auto gx : groups_of_x) {
        if (part.is_border_op(op, gx)) continue;
        if (part.groups[gx].ops.size() < 3 || part.groups[gx].ops.size() > 15) continue;

        {
            auto er = part.eval_eject(op, gx);
            if (er.feasible && accept(er.saving)) {
                FMMove candidate;
                candidate.type = FMMove::INTERNAL_EJECT;
                candidate.op = op; candidate.ga = gx;
                candidate.saving = er.saving;
                if (candidate.saving > best.saving) best = candidate;
            }
        }

        {

            for (auto v : part.dag->op_neighbors[op]) {
                if (!part.groups[gx].ops.count(v)) continue;
                auto edge = std::make_pair(std::min(op, v), std::max(op, v));

                auto sr = part.eval_split(edge.first, edge.second, gx);
                if (sr.feasible && accept(sr.saving) && sr.saving > best.saving) {
                    if (!split_creates_topo_cycle(sr.side_a, sr.side_b, *part.dag))
                        best = FMMove{FMMove::SPLIT, op, gx, SIZE_MAX, edge.second, sr.saving};
                }
            }
        }
    }

    // --- Tensor-centric moves: TENSOR_MERGE and TENSOR_EXTRACT ---
    {
        std::set<size_t> tensors_checked;
        for (auto gx : groups_of_x) {
            for (auto t : part.prob->ops[op].inputs) {
                if (tensors_checked.count(t)) continue;
                tensors_checked.insert(t);

                auto& consumers = part.dag->tensor_consumers[t];
                if (consumers.size() < 2) continue;

                std::set<size_t> consumer_groups;
                std::vector<size_t> consumer_ops_vec;
                for (auto cop : consumers) {
                    for (auto cg : part.groups_of(cop))
                        consumer_groups.insert(cg);
                    consumer_ops_vec.push_back(cop);
                }

                int prod = part.dag->tensor_producer[t];
                if (prod >= 0) {
                    auto& pg = part.groups_of((size_t)prod);
                    if (!pg.empty())
                        consumer_groups.insert(pg[0]);
                }

                if (consumer_groups.size() < 2) continue;

                bool has_locked = false;
                for (auto cg : consumer_groups) {
                    if (!part.groups[cg].alive) { has_locked = true; break; }
                    for (auto cop : part.groups[cg].ops) {
                        if (locked.count(cop)) { has_locked = true; break; }
                    }
                    if (has_locked) break;
                }
                if (has_locked) continue;

                std::vector<size_t> group_list(consumer_groups.begin(),
                                               consumer_groups.end());

                // TENSOR_MERGE: use hypothetical multi-group merge check
                {
                    std::set<size_t> merged_ops;
                    double old_cost = 0;

                    for (auto cg : group_list) {
                        old_cost += part.groups[cg].cost;
                        merged_ops.insert(part.groups[cg].ops.begin(),
                                          part.groups[cg].ops.end());
                    }

                    if (part.is_acyclic_after_merge(group_list)) {
                        double new_cost = part.eval_set(merged_ops);
                        if (new_cost < 1e17) {
                            double saving = old_cost - new_cost;
                            if (accept(saving)) {
                                FMMove candidate;
                                candidate.type = FMMove::TENSOR_MERGE;
                                candidate.op = op;
                                candidate.op2 = t;
                                candidate.saving = saving;
                                candidate.tensor_groups = group_list;
                                if (candidate.saving > best.saving) best = candidate;
                            }
                        }
                    }
                }

                // TENSOR_EXTRACT: no cheap hypothetical check available.
                // Use post-hoc is_acyclic() in apply_fm_move as safety net.
                {
                    std::set<size_t> extract_ops(consumer_ops_vec.begin(),
                                                 consumer_ops_vec.end());
                    if (prod >= 0) extract_ops.insert((size_t)prod);

                    double old_cost = 0;
                    double remainder_cost = 0;
                    bool feasible = true;

                    for (auto cg : group_list) {
                        old_cost += part.groups[cg].cost;
                        std::set<size_t> remainder;
                        for (auto rop : part.groups[cg].ops)
                            if (!extract_ops.count(rop))
                                remainder.insert(rop);
                        if (!remainder.empty()) {
                            double rc = part.eval_set(remainder);
                            if (rc >= 1e17) { feasible = false; break; }
                            remainder_cost += rc;
                        }
                    }

                    if (feasible) {
                        double extract_cost = part.eval_set(extract_ops);
                        if (extract_cost < 1e17) {
                            double saving = old_cost - (extract_cost + remainder_cost);
                            if (accept(saving)) {
                                FMMove candidate;
                                candidate.type = FMMove::TENSOR_EXTRACT;
                                candidate.op = op;
                                candidate.op2 = t;
                                candidate.saving = saving;
                                candidate.tensor_groups = group_list;
                                candidate.tensor_consumer_ops.assign(
                                    extract_ops.begin(), extract_ops.end());
                                if (candidate.saving > best.saving) best = candidate;
                            }
                        }
                    }
                }
            }
        }
    }

    return best;
}

// ============================================================================
// apply_fm_move
//
// Pre-mutation acyclicity checks for MERGE/STEAL/RECOMPUTE/TENSOR_MERGE.
// TENSOR_EXTRACT: all remainder costs validated before any mutation.
// No post-hoc is_acyclic() needed — no partial mutation on failure.
// ============================================================================

std::set<size_t> apply_fm_move(Partition& part, const FMMove& m) {
    if (!m.valid()) return {};
    std::set<size_t> affected;

    switch (m.type) {
        case FMMove::STEAL: {
            // FMMove::STEAL: op moves FROM m.ga INTO m.gb
            if (!part.is_acyclic_after_steal(m.op, m.ga, m.gb)) return {};

            std::set<size_t> new_ga = part.groups[m.ga].ops;
            new_ga.erase(m.op);
            std::set<size_t> new_gb = part.groups[m.gb].ops;
            new_gb.insert(m.op);

            double new_gb_cost = part.eval_set(new_gb);
            if (new_gb_cost >= 1e17) return {};

            if (new_ga.empty()) {
                part.groups[m.ga].alive = false;
            } else {
                double new_ga_cost = part.eval_set(new_ga);
                if (new_ga_cost >= 1e17) return {};
                part.groups[m.ga].ops = std::move(new_ga);
                part.groups[m.ga].cost = new_ga_cost;
            }
            part.groups[m.ga].gen++;
            affected.insert(m.ga);

            part.groups[m.gb].ops = std::move(new_gb);
            part.groups[m.gb].cost = new_gb_cost;
            part.groups[m.gb].gen++;
            affected.insert(m.gb);
            break;
        }
        case FMMove::MERGE: {
            if (!part.is_acyclic_after_merge(m.ga, m.gb)) return {};

            std::set<size_t> merged = part.groups[m.ga].ops;
            merged.insert(part.groups[m.gb].ops.begin(), part.groups[m.gb].ops.end());
            double merged_cost = part.eval_set(merged);
            if (merged_cost >= 1e17) return {};

            part.groups[m.ga].ops = std::move(merged);
            part.groups[m.ga].cost = merged_cost;
            part.groups[m.ga].gen++;
            affected.insert(m.ga);

            part.groups[m.gb].alive = false;
            part.groups[m.gb].gen++;
            affected.insert(m.gb);
            break;
        }
        case FMMove::RECOMPUTE: {
            if (!part.is_acyclic_after_recompute(m.op, m.gb)) return {};

            std::set<size_t> new_gb = part.groups[m.gb].ops;
            new_gb.insert(m.op);
            double new_cost = part.eval_set(new_gb);
            if (new_cost >= 1e17) return {};

            part.groups[m.gb].ops = std::move(new_gb);
            part.groups[m.gb].cost = new_cost;
            part.groups[m.gb].gen++;
            affected.insert(m.gb);
            break;
        }
        case FMMove::EJECT:
        case FMMove::INTERNAL_EJECT: {
            auto er = part.eval_eject(m.op, m.ga);
            if (!er.feasible) return {};

            if (er.remainder_components.empty()) {
                part.groups[m.ga].alive = false;
            } else {
                part.groups[m.ga].ops = std::move(er.remainder_components[0]);
                part.groups[m.ga].cost = er.component_costs[0];
                for (size_t i = 1; i < er.remainder_components.size(); i++) {
                    size_t new_gi = part.add_group(
                        std::move(er.remainder_components[i]), er.component_costs[i]);
                    affected.insert(new_gi);
                }
            }
            part.groups[m.ga].gen++;
            affected.insert(m.ga);

            if (er.singleton_cost > 0) {
                size_t new_gi = part.add_group({m.op}, er.singleton_cost);
                affected.insert(new_gi);
            }
            break;
        }
        case FMMove::SPLIT: {
            auto sr = part.eval_split(m.op, m.op2, m.ga);
            if (!sr.feasible) return {};
            if (split_creates_topo_cycle(sr.side_a, sr.side_b, *part.dag))
                return {};

            part.groups[m.ga].ops = std::move(sr.side_a);
            part.groups[m.ga].cost = sr.cost_a;
            part.groups[m.ga].gen++;
            affected.insert(m.ga);

            size_t gb = part.add_group(std::move(sr.side_b), sr.cost_b);
            affected.insert(gb);
            break;
        }
        case FMMove::TENSOR_MERGE: {
            for (auto cg : m.tensor_groups)
                if (!part.groups[cg].alive) return {};
            if (!part.is_acyclic_after_merge(m.tensor_groups)) return {};

            std::set<size_t> merged_ops;
            for (auto cg : m.tensor_groups)
                merged_ops.insert(part.groups[cg].ops.begin(),
                                  part.groups[cg].ops.end());

            double merged_cost = part.eval_set(merged_ops);
            if (merged_cost >= 1e17) return {};

            size_t survivor = m.tensor_groups[0];
            part.groups[survivor].ops = std::move(merged_ops);
            part.groups[survivor].cost = merged_cost;
            part.groups[survivor].gen++;
            affected.insert(survivor);

            for (size_t i = 1; i < m.tensor_groups.size(); i++) {
                part.groups[m.tensor_groups[i]].alive = false;
                part.groups[m.tensor_groups[i]].gen++;
                affected.insert(m.tensor_groups[i]);
            }
            break;
        }
        case FMMove::TENSOR_EXTRACT: {
            for (auto cg : m.tensor_groups)
                if (!part.groups[cg].alive) return {};

            std::set<size_t> extract_ops(m.tensor_consumer_ops.begin(),
                                         m.tensor_consumer_ops.end());

            double extract_cost = part.eval_set(extract_ops);
            if (extract_cost >= 1e17) return {};

            // Pre-validate ALL remainder costs before mutating anything
            struct RemInfo { size_t gi; std::set<size_t> rem; double cost; };
            std::vector<RemInfo> remainders;
            for (auto cg : m.tensor_groups) {
                std::set<size_t> rem;
                for (auto rop : part.groups[cg].ops)
                    if (!extract_ops.count(rop)) rem.insert(rop);
                double rc = 0;
                if (!rem.empty()) {
                    rc = part.eval_set(rem);
                    if (rc >= 1e17) return {};  // reject before any mutation
                }
                remainders.push_back({cg, std::move(rem), rc});
            }

            // All costs validated — now mutate
            for (auto& r : remainders) {
                if (r.rem.empty()) {
                    part.groups[r.gi].alive = false;
                } else {
                    part.groups[r.gi].ops = std::move(r.rem);
                    part.groups[r.gi].cost = r.cost;
                }
                part.groups[r.gi].gen++;
                affected.insert(r.gi);
            }

            size_t new_gi = part.add_group(std::move(extract_ops), extract_cost);
            affected.insert(new_gi);

            // Post-hoc acyclicity check: TENSOR_EXTRACT has no cheap
            // hypothetical check (unlike MERGE/STEAL/RECOMPUTE), so we
            // verify the result after mutation. The caller holds a snapshot
            // and reverts on empty return.
            part.rebuild_index();
            if (!part.is_acyclic()) return {};

            break;
        }
        case FMMove::DE_RECOMPUTE: {
            if (!part.groups[m.ga].alive) return {};
            for (auto o : part.groups[m.ga].ops) {
                bool in_other = false;
                for (auto gj : part.groups_of(o))
                    if (gj != m.ga && part.groups[gj].alive) { in_other = true; break; }
                if (!in_other) return {};
            }
            auto adj = part.adjacent_groups(m.ga);
            part.groups[m.ga].alive = false;
            part.groups[m.ga].gen++;
            affected.insert(adj.begin(), adj.end());
            affected.insert(m.ga);
            break;
        }
        default:
            return {};
    }

    part.rebuild_index();
    return affected;
}