#include "search/feasibility.h"
#include <algorithm>

namespace feasibility {

// ============================================================================
// Layer 1: O(degree) pre-filter for SPLIT
// ============================================================================

bool split_creates_topo_cycle(const FlatSet<size_t>& side_a,
                               const FlatSet<size_t>& side_b,
                               const DAG& dag) {
    bool a_to_b = false, b_to_a = false;
    for (auto op : side_a) {
        for (auto succ : dag.op_succs[op]) {
            if (side_b.count(succ)) { a_to_b = true; break; }
        }
        if (a_to_b) break;
    }
    if (!a_to_b) return false;
    for (auto op : side_b) {
        for (auto succ : dag.op_succs[op]) {
            if (side_a.count(succ)) { b_to_a = true; break; }
        }
        if (b_to_a) break;
    }
    return b_to_a;
}

// ============================================================================
// Layer 2: Partition-level Kahn's algorithm
//
// Zero-allocation Kahn's with virtual group mapping via lambda.
// Instead of copying op_to_groups and mutating it, we use for_virtual_groups
// to iterate over an op's groups AS IF the move had been applied.
//
// OR-node semantics: when a tensor is produced by multiple groups (recompute),
// any ONE producer satisfying the dependency unlocks the consumer.  Tracked
// via dep_met[] booleans and unsatisfied_deps[] counters.
// ============================================================================

bool kahn_with_delta(
    const Problem& prob,
    const DAG& dag,
    const std::vector<std::vector<size_t>>& op_to_groups,
    const std::vector<bool>& alive,
    size_t num_alive_groups,
    const MoveDelta& delta)
{
    size_t ng = alive.size();

    // For MERGE_MULTI: build killed lookup once
    std::vector<bool> is_killed;
    size_t survivor = 0;
    if (delta.type == MoveDelta::MERGE_MULTI && delta.merge_list) {
        is_killed.resize(ng, false);
        survivor = (*delta.merge_list)[0];
        for (size_t k = 1; k < delta.merge_list->size(); k++)
            is_killed[(*delta.merge_list)[k]] = true;
    }

    // Lambda: iterate over an op's groups as if the move had been applied
    auto for_virtual_groups = [&](size_t op_idx, auto&& callback) {
        switch (delta.type) {
        case MoveDelta::NONE:
            for (size_t g : op_to_groups[op_idx]) callback(g);
            return;

        case MoveDelta::MERGE_PAIR: {
            bool yielded_survivor = false;
            for (size_t g : op_to_groups[op_idx]) {
                if (g == delta.ga || g == delta.gb) {
                    if (!yielded_survivor) { callback(delta.ga); yielded_survivor = true; }
                } else {
                    callback(g);
                }
            }
            return;
        }
        case MoveDelta::MERGE_MULTI: {
            bool yielded_survivor = false;
            for (size_t g : op_to_groups[op_idx]) {
                if (g == survivor || (g < ng && is_killed[g])) {
                    if (!yielded_survivor) { callback(survivor); yielded_survivor = true; }
                } else {
                    callback(g);
                }
            }
            return;
        }
        case MoveDelta::STEAL:
            if (op_idx == delta.op) {
                bool gb_added = false;
                for (size_t g : op_to_groups[op_idx]) {
                    if (g == delta.ga) continue;
                    if (g == delta.gb) gb_added = true;
                    callback(g);
                }
                if (!gb_added) callback(delta.gb);
                return;
            }
            for (size_t g : op_to_groups[op_idx]) callback(g);
            return;

        case MoveDelta::RECOMPUTE:
            if (op_idx == delta.op) {
                bool gb_added = false;
                for (size_t g : op_to_groups[op_idx]) {
                    if (g == delta.gb) gb_added = true;
                    callback(g);
                }
                if (!gb_added) callback(delta.gb);
                return;
            }
            for (size_t g : op_to_groups[op_idx]) callback(g);
            return;

        case MoveDelta::SPLIT_MOVE:
            if (delta.split_ops && delta.split_ops->count(op_idx)) {
                bool gb_added = false;
                for (size_t g : op_to_groups[op_idx]) {
                    if (g == delta.ga) continue;
                    if (g == delta.gb) gb_added = true;
                    callback(g);
                }
                if (!gb_added) callback(delta.gb);
                return;
            }
            for (size_t g : op_to_groups[op_idx]) callback(g);
            return;

        case MoveDelta::EXTRACT_MOVE:
            if (delta.split_ops && delta.split_ops->count(op_idx)) {
                bool gb_added = false;
                for (size_t g : op_to_groups[op_idx]) {
                    bool is_source = false;
                    if (delta.merge_list)
                        for (auto sg : *delta.merge_list)
                            if (sg == g) { is_source = true; break; }
                    if (is_source) {
                        if (!gb_added) { callback(delta.gb); gb_added = true; }
                    } else {
                        if (g == delta.gb) gb_added = true;
                        callback(g);
                    }
                }
                if (!gb_added) callback(delta.gb);
                return;
            }
            for (size_t g : op_to_groups[op_idx]) callback(g);
            return;
        }
    };

    // 1. Build dependencies using flat vectors
    std::vector<int> unsatisfied_deps(ng, 0);
    std::vector<bool> dep_met;
    std::vector<std::vector<std::pair<size_t, size_t>>> frees(ng);

    for (size_t op_idx = 0; op_idx < prob.ops.size(); op_idx++) {
        for (auto t : prob.ops[op_idx].inputs) {
            int prod = dag.tensor_producer[t];
            if (prod < 0) continue;

            for_virtual_groups(op_idx, [&](size_t target_gi) {
                if (!alive[target_gi]) return;

                bool prod_internal = false;
                for_virtual_groups((size_t)prod, [&](size_t g) {
                    if (g == target_gi) prod_internal = true;
                });
                if (prod_internal) return;

                // Check if ANY alive source exists
                bool has_source = false;
                for_virtual_groups((size_t)prod, [&](size_t source_gj) {
                    if (alive[source_gj]) has_source = true;
                });
                if (!has_source) return;

                size_t dep_id = dep_met.size();
                dep_met.push_back(false);
                unsatisfied_deps[target_gi]++;

                for_virtual_groups((size_t)prod, [&](size_t source_gj) {
                    if (alive[source_gj])
                        frees[source_gj].push_back({target_gi, dep_id});
                });
            });
        }
    }

    // 2. Kahn's traversal with flat vector queue
    std::vector<size_t> q;
    q.reserve(ng);
    std::vector<bool> enqueued(ng, false);
    size_t visited = 0;

    for (size_t i = 0; i < ng; i++) {
        if (alive[i] && unsatisfied_deps[i] == 0) {
            q.push_back(i);
            enqueued[i] = true;
        }
    }

    size_t head = 0;
    while (head < q.size()) {
        size_t u = q[head++];
        visited++;

        for (auto [gi, dep_id] : frees[u]) {
            if (!dep_met[dep_id]) {
                dep_met[dep_id] = true;
                unsatisfied_deps[gi]--;

                if (unsatisfied_deps[gi] == 0 && !enqueued[gi]) {
                    q.push_back(gi);
                    enqueued[gi] = true;
                }
            }
        }
    }

    return visited >= num_alive_groups;
}

} // namespace feasibility