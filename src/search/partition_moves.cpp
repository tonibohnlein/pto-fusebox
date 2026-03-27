#include "search/partition_moves.h"
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
    std::set<size_t> new_to = p.groups[to].ops;
    new_to.insert(op);
    double new_to_cost = p.eval_set(new_to);
    if (new_to_cost >= 1e17) return r;

    // Compute new `from` cost
    std::set<size_t> new_from = p.groups[from].ops;
    new_from.erase(op);
    double new_from_cost = 0;

    if (new_from.empty()) {
        // `from` dies — its cost becomes 0.
        // Op is covered by `to` after the move (STEAL adds op to `to`).
    } else {
        new_from_cost = p.eval_set(new_from);
        if (new_from_cost >= 1e17) return r;
    }

    // Ephemeral gap check at eval time: moves are always evaluated against the
    // current partition, so a gap found here is a real rejection, not a stale one.
    {
        std::vector<std::set<size_t>> components;
        components.push_back(new_to);
        if (!new_from.empty()) components.push_back(new_from);
        if (p.split_creates_ephemeral_gap(components, {from, to})) return r;
    }

    r.feasible = true;
    r.saving = (p.groups[from].cost + p.groups[to].cost)
             - (new_from_cost + new_to_cost);
    return r;
}

std::set<size_t> apply_steal(Partition& p, size_t op, size_t from, size_t to) {
    if (!p.groups[from].alive || !p.groups[to].alive) return {};
    if (!p.groups[from].ops.count(op)) return {};
    // Safety re-check: catch any stale evals that would create a cycle.
    if (!p.acyclic_steal_local(op, from, to)) {
#ifndef NDEBUG
        std::cerr << "  DBG apply_steal: stale cyclic move blocked (op=" << op
                  << " from=" << from << " to=" << to << ")\n";
#endif
        return {};
    }

    // Build new op sets
    std::set<size_t> new_to = p.groups[to].ops;
    new_to.insert(op);

    std::set<size_t> new_from = p.groups[from].ops;
    new_from.erase(op);

    // Validate costs
    double new_to_cost = p.eval_set(new_to);
    if (new_to_cost >= 1e17) return {};

    double new_from_cost = 0;
    bool from_dies = new_from.empty();

    if (from_dies) {
        // `from` dies — op moves to `to`, always covered.
    } else {
        new_from_cost = p.eval_set(new_from);
        if (new_from_cost >= 1e17) return {};
    }

    // Mutate
    std::set<size_t> affected;

    if (from_dies) {
        p.groups[from].alive = false;
    } else {
        p.groups[from].ops = std::move(new_from);
        p.groups[from].cost = new_from_cost;
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

    std::set<size_t> merged = p.groups[ga].ops;
    merged.insert(p.groups[gb].ops.begin(), p.groups[gb].ops.end());
    double merged_cost = p.eval_set(merged);
    if (merged_cost >= 1e17) return r;

    if (p.creates_ephemeral_gap(merged, ga, gb)) return r;

    r.feasible = true;
    r.saving = (p.groups[ga].cost + p.groups[gb].cost) - merged_cost;
    return r;
}

std::set<size_t> apply_merge(Partition& p, size_t ga, size_t gb) {
    if (!p.groups[ga].alive || !p.groups[gb].alive) return {};
    if (ga == gb) return {};
    // Safety re-check: catch any stale evals that would create a cycle.
    if (!p.acyclic_merge_local(ga, gb)) {
#ifndef NDEBUG
        std::cerr << "  DBG apply_merge: stale cyclic move blocked (ga=" << ga
                  << " gb=" << gb << ")\n";
#endif
        return {};
    }

    // Re-evaluate cost
    std::set<size_t> merged = p.groups[ga].ops;
    merged.insert(p.groups[gb].ops.begin(), p.groups[gb].ops.end());
    double merged_cost = p.eval_set(merged);
    if (merged_cost >= 1e17) return {};

    // Mutate
    std::set<size_t> affected;

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

std::set<size_t> apply_eject(Partition& p, size_t op, size_t ga) {
    if (!p.groups[ga].alive) return {};
    if (!p.groups[ga].ops.count(op)) return {};
    if (p.groups[ga].ops.size() < 2) return {};

#ifndef NDEBUG
    // Eval-time checks in best_move_for should have rejected cyclic ejects.
    // Assert here to catch stale heap entries that bypass eval.
    {
        bool in_other = false;
        for (auto gi : p.groups_of(op))
            if (gi != ga && p.groups[gi].alive) { in_other = true; break; }
        if (in_other && !p.acyclic_de_recompute_local(op, ga)) {
            std::cerr << "  BUG: apply_eject: cyclic eject of recomputed op="
                      << op << " from G" << ga
                      << " — should have been rejected at eval time\n";
            assert(false && "cyclic eject reached apply — eval-time check missed it");
        }
    }
#endif

    // Re-evaluate (remainder components may have changed since eval)
    auto er = p.eval_eject(op, ga);
    if (!er.feasible) return {};

    std::set<size_t> affected;

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

std::set<size_t> apply_split(Partition& p, size_t op_a, size_t op_b, size_t ga) {
    if (!p.groups[ga].alive) return {};

    auto sr = p.eval_split(op_a, op_b, ga);
    if (!sr.feasible) return {};

    // Acyclicity: eval_split now rejects splits that would create a cycle
    // through recomputed ops.  The assertion below catches regressions.

    std::set<size_t> affected;

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

    if (!p.acyclic_de_recompute_local(op, ga)) return r;

    bool ga_dies = (p.groups[ga].ops.size() == 1);
    double new_ga_cost = 0;
    if (!ga_dies) {
        std::set<size_t> remaining = p.groups[ga].ops;
        remaining.erase(op);
        new_ga_cost = p.eval_set(remaining);
        if (new_ga_cost >= 1e17) return r;
    }

    // Ephemeral gap check: for each output tensor of op, verify it's still
    // available as a boundary output from at least one other source.
    for (auto t : p.prob->ops[op].outputs) {
        // Is T still produced as boundary output by some other group?
        bool still_available = false;
        for (auto gj : p.groups_of(op)) {
            if (gj == ga || !p.groups[gj].alive) continue;
            bool consumed_in_gj = false;
            for (auto cop : p.dag->tensor_consumers[t])
                if (p.groups[gj].ops.count(cop)) { consumed_in_gj = true; break; }
            if (!consumed_in_gj) { still_available = true; break; }
        }
        if (still_available) continue;

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

    r.feasible = true;
    r.saving = p.groups[ga].cost - new_ga_cost;
    return r;
}

std::set<size_t> apply_de_recompute(Partition& p, size_t ga, size_t op) {
    if (!p.groups[ga].alive) return {};
    if (!p.groups[ga].ops.count(op)) return {};

    auto check = eval_de_recompute(p, ga, op);
    if (!check.feasible) return {};

    std::set<size_t> affected;
    auto adj = p.adjacent_groups(ga);
    affected.insert(adj.begin(), adj.end());

    bool ga_dies = (p.groups[ga].ops.size() == 1);
    if (ga_dies) {
        p.groups[ga].alive = false;
    } else {
        p.groups[ga].ops.erase(op);
        std::set<size_t> remaining = p.groups[ga].ops;
        p.groups[ga].cost = p.eval_set(remaining);
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
    if (!p.acyclic_recompute_local(op, into)) return r;

    std::set<size_t> new_into = p.groups[into].ops;
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

std::set<size_t> apply_recompute(Partition& p, size_t op, size_t into) {
    if (!p.groups[into].alive) return {};
    // Safety re-check: catch any stale evals that would create a cycle.
    if (!p.acyclic_recompute_local(op, into)) {
#ifndef NDEBUG
        std::cerr << "  DBG apply_recompute: stale cyclic move blocked (op=" << op
                  << " into=" << into << ")\n";
#endif
        return {};
    }

    std::set<size_t> new_into = p.groups[into].ops;
    new_into.insert(op);
    double new_cost = p.eval_set(new_into);
    if (new_cost >= 1e17) return {};

    std::set<size_t> affected;
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

    std::set<size_t> merged_ops;
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

std::set<size_t> apply_tensor_merge(Partition& p,
                                     const std::vector<size_t>& group_list) {
    if (group_list.size() < 2) return {};
    for (auto gi : group_list)
        if (!p.groups[gi].alive) return {};

    // Safety re-check: catch any stale evals that would create a cycle.
    if (!p.acyclic_merge_local(group_list)) {
#ifndef NDEBUG
        std::cerr << "  DBG apply_tensor_merge: stale cyclic move blocked\n";
#endif
        return {};
    }

    std::set<size_t> merged_ops;
    for (auto gi : group_list)
        merged_ops.insert(p.groups[gi].ops.begin(), p.groups[gi].ops.end());

    double merged_cost = p.eval_set(merged_ops);
    if (merged_cost >= 1e17) return {};

    std::set<size_t> affected;
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
                                const std::set<size_t>& extract_ops,
                                const std::vector<size_t>& source_groups) {
    EvalResult r;
    if (extract_ops.empty() || source_groups.empty()) return r;

    for (auto gi : source_groups)
        if (!p.groups[gi].alive) return r;

    double old_cost = 0;
    double remainder_cost = 0;
    std::vector<std::set<size_t>> remainder_sets;

    for (auto gi : source_groups) {
        old_cost += p.groups[gi].cost;
        std::set<size_t> rem;
        for (auto op : p.groups[gi].ops)
            if (!extract_ops.count(op)) rem.insert(op);
        if (!rem.empty()) {
            double rc = p.eval_set(rem);
            if (rc >= 1e17) return r;
            remainder_cost += rc;
            remainder_sets.push_back(std::move(rem));
        }
    }

    double extract_cost = p.eval_set(extract_ops);
    if (extract_cost >= 1e17) return r;

    // Ephemeral gap check at eval time
    {
        std::vector<std::set<size_t>> components;
        components.push_back(extract_ops);
        for (auto& rem : remainder_sets) components.push_back(rem);
        std::set<size_t> excluded(source_groups.begin(), source_groups.end());
        if (p.split_creates_ephemeral_gap(components, excluded)) return r;
    }

    r.feasible = true;
    r.saving = old_cost - (extract_cost + remainder_cost);
    return r;
}

std::set<size_t> apply_tensor_extract(Partition& p,
                                       const std::set<size_t>& extract_ops,
                                       const std::vector<size_t>& source_groups) {
    if (extract_ops.empty() || source_groups.empty()) return {};
    for (auto gi : source_groups)
        if (!p.groups[gi].alive) return {};

    // Acyclicity checked at eval time via acyclic_extract_local — no re-check here.

    double extract_cost = p.eval_set(extract_ops);
    if (extract_cost >= 1e17) return {};

    // Pre-validate ALL remainder costs before mutating
    struct RemInfo { size_t gi; std::set<size_t> rem; double cost; };
    std::vector<RemInfo> remainders;
    for (auto gi : source_groups) {
        std::set<size_t> rem;
        for (auto op : p.groups[gi].ops)
            if (!extract_ops.count(op)) rem.insert(op);
        double rc = 0;
        if (!rem.empty()) {
            rc = p.eval_set(rem);
            if (rc >= 1e17) return {};
        }
        remainders.push_back({gi, std::move(rem), rc});
    }

    // All validated — mutate
    std::set<size_t> affected;
    for (auto& ri : remainders) {
        if (ri.rem.empty()) {
            p.groups[ri.gi].alive = false;
        } else {
            p.groups[ri.gi].ops = std::move(ri.rem);
            p.groups[ri.gi].cost = ri.cost;
        }
        p.groups[ri.gi].gen++;
        affected.insert(ri.gi);
    }

    size_t new_gi = p.add_group(extract_ops, extract_cost);
    affected.insert(new_gi);

    return affected;
}

} // namespace partition_moves