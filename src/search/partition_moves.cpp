#include "search/partition_moves.h"
#include "search/structural_ops.h"
#include <cassert>
#include <iostream>

namespace partition_moves {

// ============================================================================
// STEAL: move op from group `from` to group `to`
// ============================================================================

EvalResult eval_steal(const Partition& p, size_t op, size_t from, size_t to) {
    EvalResult r;

    if (!p.groups[from].alive || !p.groups[to].alive) return r;
    if (!p.groups[from].ops.count(op)) return r;

    // Compute new `to` cost
    FlatSet<size_t> new_to = p.groups[to].ops;
    new_to.insert(op);
    double new_to_cost = p.eval_set(new_to);
    if (new_to_cost >= 1e17) return r;

    // Compute new `from` cost — account for possible disconnected components
    FlatSet<size_t> new_from = p.groups[from].ops;
    new_from.erase(op);
    double new_from_cost = 0;
    std::vector<FlatSet<size_t>> from_comps;

    if (new_from.empty()) {
        // `from` dies — its cost becomes 0.
    } else {
        from_comps = structural_ops::connected_components(new_from, *p.dag);
        for (auto& comp : from_comps) {
            double c = p.eval_set(comp);
            if (c >= 1e17) return r;
            new_from_cost += c;
        }
    }

    // Ephemeral gap check using actual components (after connectivity split).
    {
        std::vector<FlatSet<size_t>> gap_components;
        gap_components.push_back(new_to);
        for (auto& c : from_comps) gap_components.push_back(c);
        if (p.split_creates_ephemeral_gap(gap_components, {from, to})) return r;
    }

    r.feasible = true;
    r.saving = (p.groups[from].cost + p.groups[to].cost)
             - (new_from_cost + new_to_cost);
    return r;
}

FlatSet<size_t> apply_steal(Partition& p, size_t op, size_t from, size_t to,
                              double precomputed_from_cost,
                              double precomputed_to_cost) {
    if (!p.groups[from].alive || !p.groups[to].alive) return {};
    if (!p.groups[from].ops.count(op)) return {};
    if (!p.acyclic_steal_local(op, from, to)) {
#ifndef NDEBUG
        std::cerr << "  DBG apply_steal: cyclic move blocked (op=" << op
                  << " from=" << from << " to=" << to << ")\n";
#endif
        return {};
    }

    // Build new op sets
    FlatSet<size_t> new_to = p.groups[to].ops;
    new_to.insert(op);

    FlatSet<size_t> new_from = p.groups[from].ops;
    new_from.erase(op);

    // Validate costs — use precomputed values if both are provided
    bool use_precomputed = (precomputed_from_cost >= 0.0 && precomputed_to_cost >= 0.0);
    bool from_dies = new_from.empty();

    double new_to_cost;
    if (use_precomputed) {
        new_to_cost = precomputed_to_cost;
    } else {
        new_to_cost = p.eval_set(new_to);
        if (new_to_cost >= 1e17) return {};
    }

    double new_from_cost = 0;
    if (from_dies) {
        // `from` dies — op moves to `to`, always covered.
    } else if (use_precomputed) {
        new_from_cost = precomputed_from_cost;
    } else {
        new_from_cost = p.eval_set(new_from);
        if (new_from_cost >= 1e17) return {};
    }

    // Mutate
    FlatSet<size_t> affected;

    if (from_dies) {
        p.groups[from].alive = false;
    } else {
        // Split remainder into connected components if needed.
        auto comps = structural_ops::connected_components(new_from, *p.dag);
        p.groups[from].ops = std::move(comps[0]);
        p.groups[from].cost = (comps.size() == 1) ? new_from_cost : p.eval_set(p.groups[from].ops);
        for (size_t i = 1; i < comps.size(); i++) {
            double c = p.eval_set(comps[i]);
            size_t gi = p.add_group(std::move(comps[i]), c);
            affected.insert(gi);
        }
    }
    p.groups[from].gen++;
    affected.insert(from);

    p.groups[to].ops = std::move(new_to);
    p.groups[to].cost = new_to_cost;
    p.groups[to].gen++;
    affected.insert(to);

    return affected;
}

// ============================================================================
// MERGE: fuse gb into ga (ga survives, gb killed)
// ============================================================================

EvalResult eval_merge(const Partition& p, size_t ga, size_t gb) {
    EvalResult r;

    if (!p.groups[ga].alive || !p.groups[gb].alive) return r;
    if (ga == gb) return r;

    FlatSet<size_t> merged = p.groups[ga].ops;
    merged.insert(p.groups[gb].ops.begin(), p.groups[gb].ops.end());
    double merged_cost = p.eval_set(merged);
    if (merged_cost >= 1e17) return r;

    r.feasible = true;
    r.saving = (p.groups[ga].cost + p.groups[gb].cost) - merged_cost;
    return r;
}

FlatSet<size_t> apply_merge(Partition& p, size_t ga, size_t gb, double precomputed_cost) {
    if (!p.groups[ga].alive || !p.groups[gb].alive) return {};
    if (ga == gb) return {};
    assert(p.acyclic_merge_local(ga, gb) && "apply_merge: cyclic — eval-time check missed it");

    // Build merged ops set (always needed to assign to ga.ops)
    FlatSet<size_t> merged = p.groups[ga].ops;
    merged.insert(p.groups[gb].ops.begin(), p.groups[gb].ops.end());

    // Use precomputed cost if provided, otherwise evaluate
    double merged_cost;
    if (precomputed_cost >= 0.0) {
        merged_cost = precomputed_cost;
    } else {
        merged_cost = p.eval_set(merged);
        if (merged_cost >= 1e17) return {};
    }

    // Mutate
    FlatSet<size_t> affected;

    p.groups[ga].ops = std::move(merged);
    p.groups[ga].cost = merged_cost;
    p.groups[ga].gen++;
    affected.insert(ga);

    p.groups[gb].alive = false;
    p.groups[gb].gen++;
    affected.insert(gb);

    return affected;
}

// ============================================================================
// EJECT: remove op from group ga → singleton {op} + remainder(s)
// ============================================================================

FlatSet<size_t> apply_eject(Partition& p, size_t op, size_t ga,
                              const Partition::EjectResult* precomputed) {
    if (!p.groups[ga].alive) return {};
    if (!p.groups[ga].ops.count(op)) return {};
    if (p.groups[ga].ops.size() < 2) return {};

    // Use precomputed result if provided and feasible, otherwise evaluate now.
    Partition::EjectResult er;
    if (precomputed && precomputed->feasible) {
        er = *precomputed;
    } else {
        er = p.eval_eject(op, ga);
        if (!er.feasible) return {};
    }

    // Acyclicity safety net (debug-only, matching MERGE/STEAL pattern).
    {
        bool in_other = false;
        for (auto gi : p.groups_of(op))
            if (gi != ga && p.groups[gi].alive) { in_other = true; break; }
        assert((!in_other || p.acyclic_de_recompute_local(op, ga))
               && "apply_eject: cyclic move — eval-time check missed it");
    }

    FlatSet<size_t> affected;

    // Replace ga with first remainder component (or kill if empty)
    if (er.remainder_components.empty()) {
        p.groups[ga].alive = false;
    } else {
        p.groups[ga].ops = std::move(er.remainder_components[0]);
        p.groups[ga].cost = er.component_costs[0];
        // Additional remainder components become new groups
        for (size_t i = 1; i < er.remainder_components.size(); i++) {
            size_t new_gi = p.add_group(
                std::move(er.remainder_components[i]), er.component_costs[i]);
            affected.insert(new_gi);
        }
    }
    p.groups[ga].gen++;
    affected.insert(ga);

    // Add singleton group for ejected op
    if (er.singleton_cost > 0) {
        size_t new_gi = p.add_group({op}, er.singleton_cost);
        affected.insert(new_gi);
    }

    return affected;
}

// ============================================================================
// SPLIT: split group ga at edge (op_a, op_b) into two groups
// ============================================================================

FlatSet<size_t> apply_split(Partition& p, size_t op_a, size_t op_b, size_t ga,
                              const Partition::SplitResult* precomputed) {
    if (!p.groups[ga].alive) return {};

    Partition::SplitResult sr;
    if (precomputed && precomputed->feasible) {
        sr = *precomputed;
    } else {
        sr = p.eval_split(op_a, op_b, ga);
        if (!sr.feasible) return {};
    }

    assert(p.acyclic_split_local(sr.side_a, sr.side_b, ga)
           && "apply_split: acyclic_split_local rejected");

    FlatSet<size_t> affected;

    p.groups[ga].ops = std::move(sr.side_a);
    p.groups[ga].cost = sr.cost_a;
    p.groups[ga].gen++;
    affected.insert(ga);

    size_t gb = p.add_group(std::move(sr.side_b), sr.cost_b);
    affected.insert(gb);

    return affected;
}

// ============================================================================
// DE_RECOMPUTE: remove a recomputed op from a group.
// The op must exist in at least one other alive group. The source group
// survives with the remaining ops (or dies if op was the only one).
// ============================================================================

EvalResult eval_de_recompute(const Partition& p, size_t ga, size_t op) {
    EvalResult r;
    if (!p.groups[ga].alive) return r;
    if (!p.groups[ga].ops.count(op)) return r;

    // Op must exist in at least one other alive group.
    bool in_other = false;
    for (auto gj : p.groups_of(op))
        if (gj != ga && p.groups[gj].alive) { in_other = true; break; }
    if (!in_other) return r;

    bool ga_dies = (p.groups[ga].ops.size() == 1);
    double new_ga_cost = 0;
    if (!ga_dies) {
        FlatSet<size_t> remaining = p.groups[ga].ops;
        remaining.erase(op);
        // Account for possible disconnected components.
        auto comps = structural_ops::connected_components(remaining, *p.dag);
        for (auto& comp : comps) {
            double c = p.eval_set(comp);
            if (c >= 1e17) return r;
            new_ga_cost += c;
        }
    }

    // Ephemeral gap check: for each output tensor of op, verify it's still
    // available as a boundary output from at least one other source.
    {
        size_t t = p.prob->ops[op].output();
        // Is T still produced as boundary output by some other group?
        bool still_available = false;
        for (auto gj : p.groups_of(op)) {
            if (gj == ga || !p.groups[gj].alive) continue;
            bool consumed_in_gj = false;
            for (auto cop : p.dag->tensor_consumers[t])
                if (p.groups[gj].ops.count(cop)) { consumed_in_gj = true; break; }
            if (!consumed_in_gj) { still_available = true; break; }
        }
        if (!still_available) {
            // T lost its only boundary-output source. Check every external consumer
            // has op recomputed in its group.
            for (auto cop : p.dag->tensor_consumers[t]) {
                bool served = false;
                for (auto gj : p.groups_of(cop)) {
                    if (gj == ga || !p.groups[gj].alive) continue;
                    if (p.groups[gj].ops.count(op)) { served = true; break; }
                }
                if (!served) return r;
            }
        }
    }

    r.feasible = true;
    r.saving = p.groups[ga].cost - new_ga_cost;
    return r;
}

FlatSet<size_t> apply_de_recompute(Partition& p, size_t ga, size_t op, double precomputed_cost) {
    if (!p.groups[ga].alive) return {};
    if (!p.groups[ga].ops.count(op)) return {};

    assert(p.acyclic_de_recompute_local(op, ga)
           && "apply_de_recompute: cyclic — acyclicity check missed it");

    bool ga_dies = (p.groups[ga].ops.size() == 1);
    double new_ga_cost = 0;
    if (!ga_dies) {
        if (precomputed_cost >= 0.0) {
            new_ga_cost = precomputed_cost;
        } else {
            auto check = eval_de_recompute(p, ga, op);
            if (!check.feasible) return {};
            FlatSet<size_t> remaining = p.groups[ga].ops;
            remaining.erase(op);
            new_ga_cost = p.eval_set(remaining);
            if (new_ga_cost >= 1e17) return {};
        }
    }

    FlatSet<size_t> affected;

    if (ga_dies) {
        p.groups[ga].alive = false;
    } else {
        FlatSet<size_t> remaining = p.groups[ga].ops;
        remaining.erase(op);
        // Split remainder into connected components if needed.
        auto comps = structural_ops::connected_components(remaining, *p.dag);
        p.groups[ga].ops = std::move(comps[0]);
        p.groups[ga].cost = (comps.size() == 1) ? new_ga_cost : p.eval_set(p.groups[ga].ops);
        for (size_t i = 1; i < comps.size(); i++) {
            double c = p.eval_set(comps[i]);
            size_t gi = p.add_group(std::move(comps[i]), c);
            affected.insert(gi);
        }
    }
    p.groups[ga].gen++;
    affected.insert(ga);

    return affected;
}

// ============================================================================
// RECOMPUTE: copy op into group `into` (op stays in its original group)
// ============================================================================

EvalResult eval_recompute(const Partition& p, size_t op, size_t into) {
    EvalResult r;
    if (!p.groups[into].alive) return r;

    FlatSet<size_t> new_into = p.groups[into].ops;
    new_into.insert(op);
    double new_cost = p.eval_set(new_into);
    if (new_cost >= 1e17) return r;

    if (p.creates_ephemeral_gap(new_into, into)) return r;

    // Recomputing op into `into` makes the new group consume op's input
    // tensors.  If any input tensor T is ephemeral in ALL groups containing
    // its producer (produced+consumed internally, never written to slow memory),
    // then T is not available as a boundary output from any group — gap.
    for (auto t : p.prob->ops[op].inputs) {
        int prod = p.dag->tensor_producer[t];
        if (prod < 0) continue;
        // Check if the new group itself produces T (would be internal, not a gap)
        if (new_into.count((size_t)prod)) continue;
        // T must be a boundary output of at least one alive group
        bool available = false;
        for (auto gj : p.groups_of((size_t)prod)) {
            if (!p.groups[gj].alive) continue;
            if (is_boundary_output_of(p.groups[gj].ops, t, *p.dag))
                { available = true; break; }
        }
        if (!available) return r;
    }

    r.feasible = true;
    r.saving = p.groups[into].cost - new_cost;
    return r;
}

FlatSet<size_t> apply_recompute(Partition& p, size_t op, size_t into, double precomputed_cost) {
    if (!p.groups[into].alive) return {};
    assert(p.acyclic_recompute_local(op, into) && "apply_recompute: cyclic — eval-time check missed it");

    FlatSet<size_t> new_into = p.groups[into].ops;
    new_into.insert(op);

    double new_cost;
    if (precomputed_cost >= 0.0) {
        new_cost = precomputed_cost;
    } else {
        new_cost = p.eval_set(new_into);
        if (new_cost >= 1e17) return {};
    }

    FlatSet<size_t> affected;
    p.groups[into].ops = std::move(new_into);
    p.groups[into].cost = new_cost;
    p.groups[into].gen++;
    affected.insert(into);

    return affected;
}

// ============================================================================
// TENSOR_MERGE: merge all groups in group_list into group_list[0]
// ============================================================================

EvalResult eval_tensor_merge(const Partition& p,
                              const std::vector<size_t>& group_list) {
    EvalResult r;
    if (group_list.size() < 2) return r;

    FlatSet<size_t> merged_ops;
    double old_cost = 0;
    for (auto gi : group_list) {
        if (!p.groups[gi].alive) return r;
        old_cost += p.groups[gi].cost;
        merged_ops.insert(p.groups[gi].ops.begin(), p.groups[gi].ops.end());
    }

    double merged_cost = p.eval_set(merged_ops);
    if (merged_cost >= 1e17) return r;

    if (p.creates_ephemeral_gap(merged_ops, group_list)) return r;

    r.feasible = true;
    r.saving = old_cost - merged_cost;
    return r;
}

FlatSet<size_t> apply_tensor_merge(Partition& p,
                                     const std::vector<size_t>& group_list,
                                     double precomputed_cost) {
    if (group_list.size() < 2) return {};
    for (auto gi : group_list)
        if (!p.groups[gi].alive) return {};

    assert(p.acyclic_merge_local(group_list) && "apply_tensor_merge: cyclic move");

    FlatSet<size_t> merged_ops;
    for (auto gi : group_list)
        merged_ops.insert(p.groups[gi].ops.begin(), p.groups[gi].ops.end());

    double merged_cost = (precomputed_cost >= 0.0) ? precomputed_cost : p.eval_set(merged_ops);
    if (merged_cost >= 1e17) return {};

    FlatSet<size_t> affected;
    size_t survivor = group_list[0];
    p.groups[survivor].ops = std::move(merged_ops);
    p.groups[survivor].cost = merged_cost;
    p.groups[survivor].gen++;
    affected.insert(survivor);

    for (size_t i = 1; i < group_list.size(); i++) {
        p.groups[group_list[i]].alive = false;
        p.groups[group_list[i]].gen++;
        affected.insert(group_list[i]);
    }

    return affected;
}

// ============================================================================
// TENSOR_EXTRACT: extract ops from source groups into a new group
// ============================================================================

EvalResult eval_tensor_extract(const Partition& p,
                                const FlatSet<size_t>& extract_ops,
                                const std::vector<size_t>& source_groups) {
    EvalResult r;
    if (extract_ops.empty() || source_groups.empty()) return r;

    for (auto gi : source_groups)
        if (!p.groups[gi].alive) return r;

    double old_cost = 0;
    double remainder_cost = 0;
    std::vector<FlatSet<size_t>> remainder_sets;

    for (auto gi : source_groups) {
        old_cost += p.groups[gi].cost;
        FlatSet<size_t> rem;
        for (auto op : p.groups[gi].ops)
            if (!extract_ops.count(op)) rem.insert(op);
        if (!rem.empty()) {
            // Account for possible disconnected components in remainder.
            auto comps = structural_ops::connected_components(rem, *p.dag);
            for (auto& comp : comps) {
                double c = p.eval_set(comp);
                if (c >= 1e17) return r;
                remainder_cost += c;
                remainder_sets.push_back(std::move(comp));
            }
        }
    }

    double extract_cost = p.eval_set(extract_ops);
    if (extract_cost >= 1e17) return r;

    // Ephemeral gap check at eval time (uses actual components)
    {
        std::vector<FlatSet<size_t>> gap_components;
        gap_components.push_back(extract_ops);
        for (auto& rem : remainder_sets) gap_components.push_back(rem);
        FlatSet<size_t> excluded(source_groups.begin(), source_groups.end());
        if (p.split_creates_ephemeral_gap(gap_components, excluded)) return r;
    }

    r.feasible = true;
    r.saving = old_cost - (extract_cost + remainder_cost);
    return r;
}

FlatSet<size_t> apply_tensor_extract(Partition& p,
                                       const FlatSet<size_t>& extract_ops,
                                       const std::vector<size_t>& source_groups,
                                       double precomputed_extract_cost) {
    assert(p.acyclic_extract_local(extract_ops) && "apply_tensor_extract: cyclic");

    if (extract_ops.empty() || source_groups.empty()) return {};
    for (auto gi : source_groups)
        if (!p.groups[gi].alive) return {};

    double extract_cost;
    if (precomputed_extract_cost >= 0.0) {
        extract_cost = precomputed_extract_cost;
    } else {
        extract_cost = p.eval_set(extract_ops);
    }
    if (extract_cost >= 1e17) return {};

    // Pre-validate ALL remainder costs before mutating
    struct RemInfo { size_t gi; FlatSet<size_t> rem; double cost; };
    std::vector<RemInfo> remainders;
    for (auto gi : source_groups) {
        FlatSet<size_t> rem;
        for (auto op : p.groups[gi].ops)
            if (!extract_ops.count(op)) rem.insert(op);
        double rc = 0;
        if (!rem.empty()) {
            rc = p.eval_set(rem);
            if (rc >= 1e17) return {};
        }
        remainders.push_back({gi, std::move(rem), rc});
    }

    // All validated — mutate. Split disconnected remainders into components.
    FlatSet<size_t> affected;
    for (auto& ri : remainders) {
        if (ri.rem.empty()) {
            p.groups[ri.gi].alive = false;
        } else {
            auto comps = structural_ops::connected_components(ri.rem, *p.dag);
            p.groups[ri.gi].ops = std::move(comps[0]);
            p.groups[ri.gi].cost = (comps.size() == 1) ? ri.cost : p.eval_set(p.groups[ri.gi].ops);
            for (size_t i = 1; i < comps.size(); i++) {
                double c = p.eval_set(comps[i]);
                size_t gi = p.add_group(std::move(comps[i]), c);
                affected.insert(gi);
            }
        }
        p.groups[ri.gi].gen++;
        affected.insert(ri.gi);
    }

    size_t new_gi = p.add_group(extract_ops, extract_cost);
    affected.insert(new_gi);

    return affected;
}

// ============================================================================
// TENSOR_EXTRACT_SPLIT: split consumers into k balanced sub-groups,
// each containing the producer (recomputed).
// ============================================================================

SplitExtractResult eval_tensor_extract_split(
    const Partition& p,
    size_t tensor_id,
    const std::vector<size_t>& consumer_ops,
    const std::vector<size_t>& source_groups)
{
    SplitExtractResult best;
    const auto& dag = *p.dag;

    int prod = dag.tensor_producer[tensor_id];
    if (prod < 0) return best;
    size_t prod_op = (size_t)prod;

    size_t n = consumer_ops.size();
    if (n < 2) return best;

    // Compute old cost of all source groups (once).
    double old_cost = 0;
    for (auto gi : source_groups) {
        if (!p.groups[gi].alive) return best;
        old_cost += p.groups[gi].cost;
    }

    // Compute remainder cost (source groups minus all extract ops).
    FlatSet<size_t> all_extract_ops(consumer_ops.begin(), consumer_ops.end());
    all_extract_ops.insert(prod_op);
    double remainder_cost = 0;
    for (auto gi : source_groups) {
        FlatSet<size_t> rem;
        for (auto op : p.groups[gi].ops)
            if (!all_extract_ops.count(op)) rem.insert(op);
        if (!rem.empty()) {
            auto comps = structural_ops::connected_components(rem, *p.dag);
            for (auto& comp : comps) {
                double c = p.eval_set(comp);
                if (c >= 1e17) return best;
                remainder_cost += c;
            }
        }
    }

    // Try k = 2, 4, 8, ... up to n.
    for (size_t k = 2; k <= n; k *= 2) {
        size_t per_group = (n + k - 1) / k;  // ceil(n / k)
        std::vector<FlatSet<size_t>> sub_groups;
        std::vector<double> sub_costs;
        double total_sub_cost = 0;
        bool feasible = true;

        for (size_t g = 0; g < k && feasible; g++) {
            FlatSet<size_t> sg;
            sg.insert(prod_op);
            for (size_t i = g * per_group; i < std::min((g + 1) * per_group, n); i++)
                sg.insert(consumer_ops[i]);

            double c = p.eval_set(sg);
            if (c >= 1e17) { feasible = false; break; }
            total_sub_cost += c;
            sub_groups.push_back(std::move(sg));
            sub_costs.push_back(c);
        }

        if (!feasible) continue;

        double saving = old_cost - (total_sub_cost + remainder_cost);
        if (saving > best.saving) {
            best.feasible = true;
            best.saving = saving;
            best.prod_op = prod;
            best.sub_groups = std::move(sub_groups);
            best.sub_costs = std::move(sub_costs);
        }
        // Found a feasible split — smaller k is always better if feasible,
        // so we could break here, but trying larger k as well in case
        // the cost model favors more splits.
    }

    return best;
}

FlatSet<size_t> apply_tensor_extract_split(
    Partition& p,
    const SplitExtractResult& result,
    const std::vector<size_t>& source_groups)
{
    if (!result.feasible || result.sub_groups.empty()) return {};

    // Collect all ops being extracted.
    FlatSet<size_t> all_extract_ops;
    for (auto& sg : result.sub_groups)
        for (auto op : sg)
            all_extract_ops.insert(op);

    // Remove extract ops from source groups, handle remainders.
    FlatSet<size_t> affected;
    for (auto gi : source_groups) {
        if (!p.groups[gi].alive) continue;
        FlatSet<size_t> rem;
        for (auto op : p.groups[gi].ops)
            if (!all_extract_ops.count(op)) rem.insert(op);

        if (rem.empty()) {
            p.groups[gi].alive = false;
        } else {
            auto comps = structural_ops::connected_components(rem, *p.dag);
            p.groups[gi].ops = std::move(comps[0]);
            p.groups[gi].cost = p.eval_set(p.groups[gi].ops);
            for (size_t i = 1; i < comps.size(); i++) {
                double c = p.eval_set(comps[i]);
                size_t ngi = p.add_group(std::move(comps[i]), c);
                affected.insert(ngi);
            }
        }
        p.groups[gi].gen++;
        affected.insert(gi);
    }

    // Create the sub-groups.
    for (size_t i = 0; i < result.sub_groups.size(); i++) {
        size_t ngi = p.add_group(result.sub_groups[i], result.sub_costs[i]);
        affected.insert(ngi);
    }

    return affected;
}

// ============================================================================
// FORCE_RECOMPUTE
// ============================================================================

ForceRecomputeResult eval_force_recompute(const Partition& p, size_t tensor_id) {
    ForceRecomputeResult r;
    const auto& dag = *p.dag;
    const auto& prob = *p.prob;

    int prod = dag.tensor_producer[tensor_id];
    if (prod < 0) return r;  // model input
    size_t prod_op = (size_t)prod;

    const auto& consumers = dag.tensor_consumers[tensor_id];
    if (consumers.size() < 2) return r;

    r.prod_op = prod_op;

    // Evaluate each {P, C_i} pair
    double old_cost_sum = 0.0;
    double new_cost_sum = 0.0;

    // Cost of groups that will lose ops (producer's groups + each consumer's groups)
    // Use a set to avoid double-counting groups that contain both P and a consumer
    FlatSet<size_t> affected_groups;
    for (auto gi : p.groups_of(prod_op))
        if (p.groups[gi].alive) affected_groups.insert(gi);
    for (auto cop : consumers)
        for (auto gi : p.groups_of(cop))
            if (p.groups[gi].alive) affected_groups.insert(gi);

    for (auto gi : affected_groups)
        old_cost_sum += p.groups[gi].cost;

    // For each consumer, check if {P, C_i} is feasible
    for (auto cop : consumers) {
        FlatSet<size_t> pair_ops = {prod_op, cop};
        double pair_cost = p.eval_set(pair_ops);
        if (pair_cost >= 1e17) continue;  // infeasible
        r.consumer_ops.push_back(cop);
        r.pair_costs.push_back(pair_cost);
        new_cost_sum += pair_cost;
    }

    if (r.consumer_ops.empty()) return r;

    // Estimate cost of damaged groups (after removing ops).
    // For each affected group, compute remainder cost.
    FlatSet<size_t> extracted_ops;
    extracted_ops.insert(prod_op);
    for (auto cop : r.consumer_ops) extracted_ops.insert(cop);

    for (auto gi : affected_groups) {
        FlatSet<size_t> remainder;
        for (auto op : p.groups[gi].ops)
            if (!extracted_ops.count(op)) remainder.insert(op);
        if (remainder.empty()) continue;  // group dies → 0 cost (already subtracted)
        auto comps = structural_ops::connected_components(remainder, *p.dag);
        for (auto& comp : comps) {
            double cc = p.eval_set(comp);
            if (cc >= 1e17) cc = 1e18;
            new_cost_sum += cc;
        }
    }

    r.saving = old_cost_sum - new_cost_sum;
    r.feasible = true;
    return r;
}

FlatSet<size_t> apply_force_recompute(Partition& p, size_t tensor_id,
                                        const ForceRecomputeResult& eval) {
    if (!eval.feasible) return {};

    const auto& dag = *p.dag;
    size_t prod_op = eval.prod_op;
    FlatSet<size_t> affected;

    // Collect all ops being extracted
    FlatSet<size_t> extracted_ops;
    extracted_ops.insert(prod_op);
    for (auto cop : eval.consumer_ops) extracted_ops.insert(cop);

    // Collect affected groups
    FlatSet<size_t> affected_groups;
    for (auto op : extracted_ops)
        for (auto gi : p.groups_of(op))
            if (p.groups[gi].alive) affected_groups.insert(gi);

    // Remove extracted ops from their current groups
    for (auto gi : affected_groups) {
        for (auto op : extracted_ops)
            p.groups[gi].ops.erase(op);
        affected.insert(gi);
    }

    // Fix damaged groups
    p.rebuild_index();
    for (auto gi : affected_groups) {
        if (!p.groups[gi].alive) continue;
        if (p.groups[gi].ops.empty()) {
            p.groups[gi].alive = false;
            continue;
        }
        auto comps = structural_ops::connected_components(p.groups[gi].ops, dag);
        if (comps.size() <= 1) {
            double nc = p.eval_set(p.groups[gi].ops);
            p.groups[gi].cost = (nc < 1e17) ? nc : 1e18;
        } else {
            p.groups[gi].ops = comps[0];
            double nc = p.eval_set(comps[0]);
            p.groups[gi].cost = (nc < 1e17) ? nc : 1e18;
            for (size_t c = 1; c < comps.size(); c++) {
                double cc = p.eval_set(comps[c]);
                size_t ngi = p.add_group(std::move(comps[c]), (cc < 1e17) ? cc : 1e18);
                affected.insert(ngi);
            }
        }
    }

    // Create new {P, C_i} groups
    for (size_t i = 0; i < eval.consumer_ops.size(); i++) {
        size_t ngi = p.add_group({prod_op, eval.consumer_ops[i]}, eval.pair_costs[i]);
        affected.insert(ngi);
    }

    p.rebuild_index();
    return affected;
}

} // namespace partition_moves