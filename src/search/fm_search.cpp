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

    // Collect neighbor groups via DAG edges
    // For each group gx containing op, find groups of DAG neighbors
    for (auto gx : groups_of_x) {
        if (!part.is_border_op(op, gx)) continue;

        // Neighbor groups reachable via op's DAG edges
        std::set<size_t> neighbor_groups;
        for (auto pred : part.dag->op_preds[op])
            if (!part.groups[gx].ops.count(pred))
                for (auto gy : part.groups_of(pred))
                    if (gy != gx) neighbor_groups.insert(gy);
        for (auto succ : part.dag->op_succs[op])
            if (!part.groups[gx].ops.count(succ))
                for (auto gy : part.groups_of(succ))
                    if (gy != gx) neighbor_groups.insert(gy);

        for (auto gy : neighbor_groups) {
            bool x_in_gy = part.groups[gy].ops.count(op);

            // --- Steal: move op from gx to gy ---
            if (!x_in_gy && !part.dag->merge_creates_cycle({op}, part.groups[gy].ops)) {
                std::set<size_t> new_gx = part.groups[gx].ops;
                new_gx.erase(op);

                // Check remainder validity (may disconnect)
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
                    std::set<size_t> new_gy = part.groups[gy].ops;
                    new_gy.insert(op);
                    double new_gy_cost = part.eval_set(new_gy);
                    if (new_gy_cost < 1e17) {
                        double saving = (part.groups[gx].cost + part.groups[gy].cost)
                                        - (new_gx_cost + new_gy_cost);
                        if (accept(saving))
                            best = FMMove{FMMove::STEAL, op, gx, gy, SIZE_MAX, saving};
                    }
                }
            }

            // --- Merge: combine gx and gy ---
            if (!part.dag->merge_creates_cycle(part.groups[gx].ops, part.groups[gy].ops)) {
                std::set<size_t> merged = part.groups[gx].ops;
                merged.insert(part.groups[gy].ops.begin(), part.groups[gy].ops.end());
                double merged_cost = part.eval_set(merged);
                if (merged_cost < 1e17) {
                    double saving = (part.groups[gx].cost + part.groups[gy].cost)
                                    - merged_cost;
                    if (accept(saving))
                        best = FMMove{FMMove::MERGE, op, gx, gy, SIZE_MAX, saving};
                }
            }

            // --- Recompute: add op to gy (keep in gx) ---
            if (!x_in_gy && !part.dag->merge_creates_cycle({op}, part.groups[gy].ops)) {
                std::set<size_t> new_gy = part.groups[gy].ops;
                new_gy.insert(op);
                double new_gy_cost = part.eval_set(new_gy);
                if (new_gy_cost < 1e17) {
                    double saving = part.groups[gy].cost - new_gy_cost;
                    if (accept(saving))
                        best = FMMove{FMMove::RECOMPUTE, op, gx, gy, SIZE_MAX, saving};
                }
            }
        }

        // --- Eject: remove op from gx ---
        auto er = part.eval_eject(op, gx);
        if (er.feasible && accept(er.saving))
            best = FMMove{FMMove::EJECT, op, gx, SIZE_MAX, SIZE_MAX, er.saving};
    }

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

        // SPLIT at each bridge edge incident to op
        for (auto succ : part.dag->op_succs[op]) {
            if (!part.groups[gx].ops.count(succ)) continue;
            auto sr = part.eval_split(op, succ, gx);
            if (sr.feasible && accept(sr.saving) && sr.saving > best.saving) {
                best = FMMove{FMMove::SPLIT, op, gx, SIZE_MAX, succ, sr.saving};
            }
        }
        for (auto pred : part.dag->op_preds[op]) {
            if (!part.groups[gx].ops.count(pred)) continue;
            auto sr = part.eval_split(pred, op, gx);
            if (sr.feasible && accept(sr.saving) && sr.saving > best.saving) {
                best = FMMove{FMMove::SPLIT, op, gx, SIZE_MAX, pred, sr.saving};
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

            // Replace ga with the components
            if (er.remainder_components.size() == 1) {
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

            part.groups[m.ga].ops = std::move(er.remainder_components[0]);
            part.groups[m.ga].cost = er.component_costs[0];
            for (size_t i = 1; i < er.remainder_components.size(); i++) {
                size_t new_gi = part.add_group(
                    std::move(er.remainder_components[i]), er.component_costs[i]);
                affected.insert(new_gi);
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
        default:
            return {};
    }

    return affected;
}