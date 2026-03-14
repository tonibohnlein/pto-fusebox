#include "search/fm_search.h"
#include <algorithm>

// ============================================================================
// best_move_for: evaluate all candidate moves for op x
// ============================================================================

FMMove best_move_for(const Partition& part, size_t op,
                     double floor, const std::set<size_t>& locked) {
    if (locked.count(op)) return {};

    // Minimum meaningful saving — below this is numerical noise
    constexpr double EPSILON = 1.0;

    FMMove best;
    auto groups_of_x = part.groups_of(op);
    if (groups_of_x.empty()) return {};

    auto accept = [&](double saving) {
        return (saving > EPSILON || saving < -EPSILON) &&
               saving > -floor && saving > best.saving;
    };

    // Collect neighbor groups via DAG edges + shared inputs
    for (auto gx : groups_of_x) {
        if (!part.is_border_op(op, gx)) continue;

        // Neighbor groups reachable via op_neighbors (DAG edges + co-consumers)
        std::set<size_t> neighbor_groups;
        for (auto nbr : part.dag->op_neighbors[op])
            if (!part.groups[gx].ops.count(nbr))
                for (auto gy : part.groups_of(nbr))
                    if (gy != gx) neighbor_groups.insert(gy);

        for (auto gy : neighbor_groups) {

            bool x_in_gy = part.groups[gy].ops.count(op);

            // --- Merge: combine gx and gy ---
            if (!part.dag->merge_creates_cycle(part.groups[gx].ops, part.groups[gy].ops)) {
                std::set<size_t> merged = part.groups[gx].ops;
                merged.insert(part.groups[gy].ops.begin(), part.groups[gy].ops.end());
                if (!part.creates_ephemeral_gap(merged, gx, gy)) {
                    double merged_cost = part.eval_set(merged);
                    if (merged_cost < 1e17) {
                        double saving = (part.groups[gx].cost + part.groups[gy].cost)
                                        - merged_cost;
                        if (accept(saving))
                            best = FMMove{FMMove::MERGE, op, gx, gy, SIZE_MAX, saving};
                    }
                }
            }

            // --- Steal + Recompute: both add op to gy ---
            // Evaluate new_gy once — shared by both move types.
            if (!x_in_gy && !part.dag->merge_creates_cycle({op}, part.groups[gy].ops)) {
                std::set<size_t> new_gy = part.groups[gy].ops;
                new_gy.insert(op);
                double new_gy_cost = part.eval_set(new_gy);

                if (new_gy_cost < 1e17) {
                    // STEAL: op leaves gx. Exclude gx (losing op, no longer a
                    // backup source for op's outputs) AND gy (replaced by new_gy).
                    if (!part.creates_ephemeral_gap(new_gy, gx, gy)) {
                        std::set<size_t> new_gx = part.groups[gx].ops;
                        new_gx.erase(op);
                        double new_gx_cost = 0;
                        bool gx_valid = true;
                        if (new_gx.empty()) {
                            // gx becomes empty — only valid if op is in other groups
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

                    // RECOMPUTE: op is copied to gy; gx keeps it. No gap check
                    // needed — RECOMPUTE cannot create an ephemeral gap: new_gy
                    // still exports any newly-ephemeral tensor as a boundary
                    // output, and gx retains op as an additional backup source.
                    {
                        double saving = part.groups[gy].cost - new_gy_cost;
                        if (accept(saving))
                            best = FMMove{FMMove::RECOMPUTE, op, gx, gy, SIZE_MAX, saving};
                    }
                }
            }
        }  // end for gy

        // --- Eject: remove op from gx (one per gx, not per gy) ---
        auto er = part.eval_eject(op, gx);
        if (er.feasible && accept(er.saving))
            best = FMMove{FMMove::EJECT, op, gx, SIZE_MAX, SIZE_MAX, er.saving};
    }  // end for gx

    // --- Internal moves: INTERNAL_EJECT and SPLIT ---
    // These apply when op is internal (no DAG neighbors outside its group)
    // Cap at 15 ops to avoid expensive eval_set calls on large groups
    for (auto gx : groups_of_x) {
        if (part.is_border_op(op, gx)) continue;  // border ops handled above
        if (part.groups[gx].ops.size() < 3 || part.groups[gx].ops.size() > 15) continue;

        // INTERNAL_EJECT: remove op, remainder may split into components
        auto er = part.eval_eject(op, gx);
        if (er.feasible && accept(er.saving)) {
            FMMove candidate;
            candidate.type = FMMove::INTERNAL_EJECT;
            candidate.op = op; candidate.ga = gx;
            candidate.saving = er.saving;
            if (candidate.saving > best.saving) best = candidate;
        }

        // SPLIT at each bridge edge incident to op.
        // Use op_neighbors (DAG + co-consumer edges) so co-consumer bridges
        // are also proposed, not just directed op_succs/op_preds edges.
        // Canonicalise as (lo, hi) to avoid evaluating the same edge twice.
        {
            std::set<std::pair<size_t,size_t>> split_checked;
            for (auto v : part.dag->op_neighbors[op]) {
                if (!part.groups[gx].ops.count(v)) continue;
                auto edge = std::make_pair(std::min(op, v), std::max(op, v));
                if (!split_checked.insert(edge).second) continue;
                auto sr = part.eval_split(edge.first, edge.second, gx);
                if (sr.feasible && accept(sr.saving) && sr.saving > best.saving) {
                    best = FMMove{FMMove::SPLIT, op, gx, SIZE_MAX, edge.second, sr.saving};
                }
            }
        }
    }

    // --- Tensor-centric moves: TENSOR_MERGE and TENSOR_EXTRACT ---
    // For each boundary input tensor of op's groups, check if merging all
    // consumer groups (and optionally the producer group) is profitable.
    {
        std::set<size_t> tensors_checked;
        for (auto gx : groups_of_x) {
            for (auto t : part.prob->ops[op].inputs) {
                if (tensors_checked.count(t)) continue;
                tensors_checked.insert(t);

                // Collect all consumer groups of tensor t
                auto& consumers = part.dag->tensor_consumers[t];
                if (consumers.size() < 2) continue;

                std::set<size_t> consumer_groups;
                std::vector<size_t> consumer_ops_vec;
                for (auto cop : consumers) {
                    for (auto cg : part.groups_of(cop)) {
                        consumer_groups.insert(cg);
                    }
                    consumer_ops_vec.push_back(cop);
                }

                // Optionally include producer group (tensor becomes ephemeral)
                int prod = part.dag->tensor_producer[t];
                size_t producer_group = SIZE_MAX;
                if (prod >= 0) {
                    auto& pg = part.groups_of((size_t)prod);
                    if (!pg.empty()) {
                        producer_group = pg[0];
                        consumer_groups.insert(producer_group);
                    }
                }

                if (consumer_groups.size() < 2) continue;

                // Check no locked ops in any of these groups
                bool has_locked = false;
                for (auto cg : consumer_groups) {
                    if (!part.groups[cg].alive) { has_locked = true; break; }
                    for (auto cop : part.groups[cg].ops) {
                        if (locked.count(cop)) { has_locked = true; break; }
                    }
                    if (has_locked) break;
                }
                if (has_locked) continue;

                // TENSOR_MERGE: merge all consumer (+ producer) groups
                {
                    std::set<size_t> merged_ops;
                    double old_cost = 0;
                    bool cycle = false;
                    std::vector<size_t> group_list(consumer_groups.begin(),
                                                   consumer_groups.end());

                    for (auto cg : group_list) {
                        old_cost += part.groups[cg].cost;
                        merged_ops.insert(part.groups[cg].ops.begin(),
                                          part.groups[cg].ops.end());
                    }

                    // Pairwise cycle check (conservative but correct)
                    for (size_t a = 0; a < group_list.size() && !cycle; a++)
                        for (size_t b = a + 1; b < group_list.size() && !cycle; b++)
                            if (part.dag->merge_creates_cycle(
                                    part.groups[group_list[a]].ops,
                                    part.groups[group_list[b]].ops))
                                cycle = true;

                    if (!cycle) {
                        std::vector<size_t> excl(group_list.begin(), group_list.end());
                        if (!part.creates_ephemeral_gap(merged_ops, excl)) {
                            double new_cost = part.eval_set(merged_ops);
                            if (new_cost < 1e17) {
                                double saving = old_cost - new_cost;
                                if (accept(saving)) {
                                    FMMove candidate;
                                    candidate.type = FMMove::TENSOR_MERGE;
                                    candidate.op = op;
                                    candidate.op2 = t;  // tensor_id
                                    candidate.saving = saving;
                                    candidate.tensor_groups = group_list;
                                    if (candidate.saving > best.saving) best = candidate;
                                }
                            }
                        }
                    }
                }

                // TENSOR_EXTRACT: pull just the consumer ops (+ producer) into
                // a new group, leaving remainders in their original groups.
                {
                    std::set<size_t> extract_ops(consumer_ops_vec.begin(),
                                                 consumer_ops_vec.end());
                    if (prod >= 0) extract_ops.insert((size_t)prod);

                    double old_cost = 0;
                    double remainder_cost = 0;
                    bool feasible = true;
                    std::vector<size_t> group_list(consumer_groups.begin(),
                                                   consumer_groups.end());

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
                        std::vector<size_t> excl(group_list.begin(), group_list.end());
                        // Cycle check: the new group N=extract_ops must not form a
                        // cycle in the condensed DAG with the remainder groups.
                        // merge_creates_cycle(extract_ops, {}) checks whether N,
                        // treated as a super-node, has any external op that can
                        // reach back into N — covering chains of any length.
                        // Cycle check: verify the extracted group doesn't form
                        // a cycle with any non-empty remainder group.
                        // merge_creates_cycle(A, {}) is always false — empty B
                        // has no ops to reach into — so we check pairwise.
                        bool extract_cycle = false;
                        for (auto cg : group_list) {
                            std::set<size_t> rem;
                            for (auto rop : part.groups[cg].ops)
                                if (!extract_ops.count(rop)) rem.insert(rop);
                            if (!rem.empty() &&
                                part.dag->merge_creates_cycle(extract_ops, rem)) {
                                extract_cycle = true; break;
                            }
                        }
                        if (!extract_cycle &&
                            !part.creates_ephemeral_gap(extract_ops, excl)) {
                            double extract_cost = part.eval_set(extract_ops);
                            if (extract_cost < 1e17) {
                                double saving = old_cost - (extract_cost + remainder_cost);
                                if (accept(saving)) {
                                    FMMove candidate;
                                    candidate.type = FMMove::TENSOR_EXTRACT;
                                    candidate.op = op;
                                    candidate.op2 = t;  // tensor_id
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
    }

    return best;
}

// ============================================================================
// apply_fm_move: execute the move, return affected group indices
// ============================================================================

std::set<size_t> apply_fm_move(Partition& part, const FMMove& m) {
    if (!m.valid()) return {};
    std::set<size_t> affected;

    switch (m.type) {
        case FMMove::STEAL: {
            // Remove op from ga, add to gb
            std::set<size_t> new_ga = part.groups[m.ga].ops;
            new_ga.erase(m.op);

            std::set<size_t> new_gb = part.groups[m.gb].ops;
            new_gb.insert(m.op);

            // Cycle re-check at apply time: groups may have changed since evaluation
            if (part.dag->merge_creates_cycle({m.op}, part.groups[m.gb].ops)) return {};
            // Gap check: exclude ga (losing m.op, no longer a backup source for
            // m.op's outputs) AND gb (being replaced by new_gb).
            if (part.creates_ephemeral_gap(new_gb, m.ga, m.gb)) return {};
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
            std::set<size_t> merged = part.groups[m.ga].ops;
            merged.insert(part.groups[m.gb].ops.begin(), part.groups[m.gb].ops.end());
            // Cycle re-check at apply time: groups may have changed since evaluation
            if (part.dag->merge_creates_cycle(part.groups[m.ga].ops,
                                               part.groups[m.gb].ops)) return {};
            if (part.creates_ephemeral_gap(merged, m.ga, m.gb)) return {};
            double cost = part.eval_set(merged);
            if (cost >= 1e17) return {};

            part.groups[m.ga].ops = std::move(merged);
            part.groups[m.ga].cost = cost;
            part.groups[m.ga].gen++;
            part.groups[m.gb].alive = false;
            part.groups[m.gb].gen++;
            affected.insert(m.ga);
            affected.insert(m.gb);
            break;
        }
        case FMMove::RECOMPUTE: {
            std::set<size_t> new_gb = part.groups[m.gb].ops;
            new_gb.insert(m.op);
            // Cycle re-check at apply time: groups may have changed
            if (part.dag->merge_creates_cycle({m.op}, part.groups[m.gb].ops)) return {};
            // No ephemeral gap check: RECOMPUTE cannot create a gap.
            // new_gb still exports any newly-ephemeral tensor as boundary output;
            // ga retains m.op as an additional backup source.
            double cost = part.eval_set(new_gb);
            if (cost >= 1e17) return {};

            part.groups[m.gb].ops = std::move(new_gb);
            part.groups[m.gb].cost = cost;
            part.groups[m.gb].gen++;
            affected.insert(m.gb);
            break;
        }
        case FMMove::EJECT: {
            auto er = part.eval_eject(m.op, m.ga);
            if (!er.feasible) return {};

            // Replace ga with the components safely!
            if (er.remainder_components.empty()) {
                part.groups[m.ga].alive = false;
            } else if (er.remainder_components.size() == 1) {
                part.groups[m.ga].ops = std::move(er.remainder_components[0]);
                part.groups[m.ga].cost = er.component_costs[0];
            } else {
                // Multiple components: first takes ga's slot, rest get new groups
                part.groups[m.ga].ops = std::move(er.remainder_components[0]);
                part.groups[m.ga].cost = er.component_costs[0];
                for (size_t i = 1; i < er.remainder_components.size(); i++) {
                    size_t new_gi = part.add_group(
                        std::move(er.remainder_components[i]),
                        er.component_costs[i]);
                    affected.insert(new_gi);
                }
            }
            part.groups[m.ga].gen++;
            affected.insert(m.ga);

            // Singleton for op if not in other groups
            if (er.singleton_cost > 0) {
                size_t new_gi = part.add_group({m.op}, er.singleton_cost);
                affected.insert(new_gi);
            }
            break;
        }
        case FMMove::INTERNAL_EJECT: {
            auto er = part.eval_eject(m.op, m.ga);
            if (!er.feasible) return {};

            // Replace ga with the components safely!
            if (er.remainder_components.empty()) {
                part.groups[m.ga].alive = false;
            } else if (er.remainder_components.size() == 1) {
                part.groups[m.ga].ops = std::move(er.remainder_components[0]);
                part.groups[m.ga].cost = er.component_costs[0];
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

            part.groups[m.ga].ops = std::move(sr.side_a);
            part.groups[m.ga].cost = sr.cost_a;
            part.groups[m.ga].gen++;
            affected.insert(m.ga);

            size_t gb = part.add_group(std::move(sr.side_b), sr.cost_b);
            affected.insert(gb);
            break;
        }
        case FMMove::TENSOR_MERGE: {
            // Re-verify: all groups must still be alive
            for (auto cg : m.tensor_groups)
                if (!part.groups[cg].alive) return {};

            // Merge all ops from all groups into one
            std::set<size_t> merged_ops;
            for (auto cg : m.tensor_groups)
                merged_ops.insert(part.groups[cg].ops.begin(),
                                  part.groups[cg].ops.end());

            if (part.creates_ephemeral_gap(merged_ops,
                    std::vector<size_t>(m.tensor_groups.begin(),
                                        m.tensor_groups.end()))) return {};
            double merged_cost = part.eval_set(merged_ops);
            if (merged_cost >= 1e17) return {};

            // Re-verify saving is still positive
            double old_cost = 0;
            for (auto cg : m.tensor_groups) old_cost += part.groups[cg].cost;
            if (merged_cost >= old_cost - 0.001) return {};

            // First group absorbs everything, rest are killed
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
            // Re-verify: all groups must still be alive
            for (auto cg : m.tensor_groups)
                if (!part.groups[cg].alive) return {};

            std::set<size_t> extract_ops(m.tensor_consumer_ops.begin(),
                                         m.tensor_consumer_ops.end());

            // Evaluate the extracted group.
            // Cycle check: verify extract_ops as a new group doesn't form a
            // cycle with any non-empty remainder group. merge_creates_cycle(A,{})
            // is always false (empty B has no ops to reach into), so check pairwise.
            for (auto cg : m.tensor_groups) {
                std::set<size_t> rem;
                for (auto rop : part.groups[cg].ops)
                    if (!extract_ops.count(rop)) rem.insert(rop);
                if (!rem.empty() && part.dag->merge_creates_cycle(extract_ops, rem))
                    return {};
            }
            if (part.creates_ephemeral_gap(extract_ops,
                    std::vector<size_t>(m.tensor_groups.begin(),
                                        m.tensor_groups.end()))) return {};
            double extract_cost = part.eval_set(extract_ops);
            if (extract_cost >= 1e17) return {};

            // Evaluate all remainders
            double old_cost = 0;
            double remainder_total = 0;
            struct RemainderInfo { size_t gi; std::set<size_t> ops; double cost; };
            std::vector<RemainderInfo> remainders;

            for (auto cg : m.tensor_groups) {
                old_cost += part.groups[cg].cost;
                std::set<size_t> rem;
                for (auto rop : part.groups[cg].ops)
                    if (!extract_ops.count(rop))
                        rem.insert(rop);
                double rc = 0;
                if (!rem.empty()) {
                    rc = part.eval_set(rem);
                    if (rc >= 1e17) return {};
                }
                remainder_total += rc;
                remainders.push_back({cg, std::move(rem), rc});
            }

            if (extract_cost + remainder_total >= old_cost - 0.001) return {};

            // Apply: update remainder groups, create extracted group
            for (auto& ri : remainders) {
                if (ri.ops.empty()) {
                    part.groups[ri.gi].alive = false;
                } else {
                    part.groups[ri.gi].ops = std::move(ri.ops);
                    part.groups[ri.gi].cost = ri.cost;
                }
                part.groups[ri.gi].gen++;
                affected.insert(ri.gi);
            }

            size_t new_gi = part.add_group(std::move(extract_ops), extract_cost);
            affected.insert(new_gi);
            break;
        }
        default:
            return {};
    }

    part.rebuild_index();
    return affected;
}