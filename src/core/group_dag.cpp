#include "core/group_dag.h"
#include "partition/partition.h"
#include <algorithm>
#include <climits>

// ============================================================================
// Build from scratch
// ============================================================================

void GroupDAG::build(const Partition& part) {
    size_t ng = part.groups.size();
    succs_.assign(ng, {});
    preds_.assign(ng, {});

    for (size_t gi = 0; gi < ng; gi++) {
        if (!part.groups[gi].alive) continue;
        rebuild_edges_for(part, gi);
    }

    compute_topo_order(part);
}

// ============================================================================
// Incremental update for affected groups
// ============================================================================

void GroupDAG::update(const Partition& part, const FlatSet<size_t>& affected) {
    ensure_size(part.groups.size());

    // Collect groups to rebuild: affected + their current neighbors.
    FlatSet<size_t> to_rebuild;
    for (auto gi : affected) {
        if (gi >= succs_.size()) continue;
        to_rebuild.insert(gi);
        for (auto gj : succs_[gi]) to_rebuild.insert(gj);
        for (auto gj : preds_[gi]) to_rebuild.insert(gj);
    }

    // Clear and rebuild edges for all affected groups
    for (auto gi : to_rebuild)
        clear_edges_for(gi);
    for (auto gi : to_rebuild) {
        if (gi >= part.groups.size() || !part.groups[gi].alive) continue;
        rebuild_edges_for(part, gi);
    }

    // Repair topological order for affected region
    repair_topo_order(part, to_rebuild);
}

void GroupDAG::ensure_size(size_t n) {
    if (succs_.size() < n) {
        succs_.resize(n);
        preds_.resize(n);
        topo_pos_.resize(n, -1);
    }
}

// ============================================================================
// Queries
// ============================================================================

bool GroupDAG::can_reach(size_t from, size_t to) const {
    if (from >= succs_.size() || to >= succs_.size()) return false;
    if (from == to) return true;

    // Fast path: topo order rules out reachability in O(1).
    // If from comes after to in topo order, from cannot reach to.
    if (topo_pos_[from] >= 0 && topo_pos_[to] >= 0 &&
        topo_pos_[from] >= topo_pos_[to])
        return false;

    // Slow path: BFS bounded by topo order.
    // Only visit nodes with topo_pos in (topo_pos_[from], ...].
    int from_pos = topo_pos_[from];

    thread_local std::vector<bool> visited;
    visited.assign(succs_.size(), false);
    thread_local std::vector<size_t> q;
    q.clear();

    visited[from] = true;
    q.push_back(from);
    size_t front = 0;

    while (front < q.size()) {
        size_t cur = q[front++];
        for (auto gj : succs_[cur]) {
            if (gj == to) return true;
            if (gj < visited.size() && !visited[gj]) {
                // Prune: don't explore nodes that are before `from` in topo order
                // (they can't be on a forward path from `from`).
                if (topo_pos_[gj] >= 0 && topo_pos_[gj] <= from_pos) continue;
                visited[gj] = true;
                q.push_back(gj);
            }
        }
    }
    return false;
}

bool GroupDAG::merge_creates_cycle(const std::vector<size_t>& merge_set) const {
    if (merge_set.size() <= 1) return false;
    size_t n = succs_.size();

    // Fast path: if all groups in merge_set are in a contiguous topo region
    // with no external group in between that has edges to/from the set,
    // there's no cycle.  Check: find min/max topo_pos of the merge set.
    // If no external successor of the set has topo_pos <= max_topo, no cycle.
    int min_pos = INT_MAX, max_pos = -1;
    for (auto g : merge_set) {
        if (g < n && topo_pos_[g] >= 0) {
            min_pos = std::min(min_pos, topo_pos_[g]);
            max_pos = std::max(max_pos, topo_pos_[g]);
        }
    }

    if (min_pos != INT_MAX) {
        // Quick check: all external successors must have topo_pos > max_pos.
        // If any external successor has topo_pos <= max_pos, it MIGHT reach
        // back into the merge set — fall through to BFS.
        thread_local std::vector<bool> in_merge;
        in_merge.assign(n, false);
        for (auto g : merge_set)
            if (g < n) in_merge[g] = true;

        bool need_bfs = false;
        for (auto gi : merge_set) {
            if (gi >= n) continue;
            for (auto gj : succs_[gi]) {
                if (in_merge[gj]) continue;
                if (topo_pos_[gj] >= 0 && topo_pos_[gj] <= max_pos) {
                    need_bfs = true;
                    break;
                }
            }
            if (need_bfs) break;
        }

        if (!need_bfs) return false;  // all successors are after merge set — safe
    }

    // Slow path: BFS from external successors
    thread_local std::vector<bool> in_merge_slow;
    in_merge_slow.assign(n, false);
    for (auto g : merge_set)
        if (g < n) in_merge_slow[g] = true;

    thread_local std::vector<bool> visited;
    visited.assign(n, false);
    thread_local std::vector<size_t> q;
    q.clear();
    size_t front = 0;

    for (auto gi : merge_set) {
        if (gi >= n) continue;
        for (auto gj : succs_[gi]) {
            if (in_merge_slow[gj] || visited[gj]) continue;
            visited[gj] = true;
            q.push_back(gj);
        }
    }

    while (front < q.size()) {
        size_t cur = q[front++];
        for (auto gj : succs_[cur]) {
            if (in_merge_slow[gj]) return true;  // cycle
            if (!visited[gj]) {
                visited[gj] = true;
                q.push_back(gj);
            }
        }
    }
    return false;
}

// ============================================================================
// Move-specific eval/apply
// ============================================================================

bool GroupDAG::eval_merge(size_t ga, size_t gb) const {
    return merge_creates_cycle({ga, gb});
}

void GroupDAG::apply_merge(const Partition& part, size_t ga, size_t gb,
                            const FlatSet<size_t>& affected) {
    apply_generic(part, affected);
}

bool GroupDAG::eval_steal(const Partition& part, size_t op,
                           size_t from, size_t to) const {
    // After stealing, `from` loses edges related to op's tensors,
    // `to` gains them. Check if the new edges create a cycle.
    // Quick check: if `to` cannot currently reach `from`, no new cycle
    // is possible (the steal only adds edges from `to`'s predecessors
    // to `to`, not backward edges).
    // But stealing can also change `from`'s edges — a tensor that was
    // internal to `from` might become a boundary output.
    // Conservative: just check if `to` can reach `from` via the current DAG.
    // If yes, adding more edges from `to` might tighten the constraint.
    // For now, defer to apply_generic + repair_topo_order for correctness.
    // The fast path is: if topo_pos_[to] > topo_pos_[from], no cycle.
    if (topo_pos_[to] >= 0 && topo_pos_[from] >= 0 &&
        topo_pos_[to] > topo_pos_[from])
        return false;
    // Can't rule it out cheaply — caller should apply and check.
    return true;  // conservative: might create cycle
}

void GroupDAG::apply_steal(const Partition& part,
                            const FlatSet<size_t>& affected) {
    apply_generic(part, affected);
}

void GroupDAG::apply_split(const Partition& part,
                            const FlatSet<size_t>& affected) {
    apply_generic(part, affected);
}

void GroupDAG::apply_eject(const Partition& part,
                            const FlatSet<size_t>& affected) {
    apply_generic(part, affected);
}

void GroupDAG::apply_generic(const Partition& part,
                              const FlatSet<size_t>& affected) {
    update(part, affected);
}

// ============================================================================
// Topological order
// ============================================================================

void GroupDAG::compute_topo_order(const Partition& part) {
    size_t ng = succs_.size();
    topo_pos_.assign(ng, -1);
    topo_order_.clear();
    topo_order_.reserve(ng);

    // Kahn's algorithm
    std::vector<int> in_deg(ng, 0);
    for (size_t gi = 0; gi < ng; gi++)
        if (part.groups[gi].alive)
            in_deg[gi] = (int)preds_[gi].size();

    std::vector<size_t> q;
    for (size_t gi = 0; gi < ng; gi++)
        if (part.groups[gi].alive && in_deg[gi] == 0)
            q.push_back(gi);

    size_t front = 0;
    while (front < q.size()) {
        size_t g = q[front++];
        topo_pos_[g] = (int)topo_order_.size();
        topo_order_.push_back(g);
        for (auto gj : succs_[g]) {
            if (--in_deg[gj] == 0)
                q.push_back(gj);
        }
    }
}

void GroupDAG::repair_topo_order(const Partition& part, const FlatSet<size_t>& affected) {
    // Simple approach: re-sort only the affected groups + their forward
    // reachable set within the current topo range.
    //
    // For correctness, we do a full recompute.  This is still fast because
    // Kahn's on the group DAG is O(groups + edges), not O(ops × consumers).
    // TODO: implement MNR-style local repair for better performance.
    compute_topo_order(part);
}

// ============================================================================
// Internal helpers
// ============================================================================

void GroupDAG::rebuild_edges_for(const Partition& part, size_t gi) {
    if (gi >= part.groups.size() || !part.groups[gi].alive) return;

    const auto& prob = *part.prob;
    const auto& dag = *part.dag;

    for (auto op : part.groups[gi].ops) {
        // Outgoing: op produces tensor consumed by another group
        size_t t_out = prob.ops[op].output();
        for (auto cop : dag.tensor_consumers[t_out]) {
            if (part.groups[gi].ops.count(cop)) continue;
            for (auto gj : part.groups_of(cop)) {
                if (gj == gi || !part.groups[gj].alive) continue;
                succs_[gi].insert(gj);
                preds_[gj].insert(gi);
            }
        }
        // Incoming: op consumes tensor produced by another group
        for (auto t_in : prob.ops[op].inputs) {
            int prod = dag.tensor_producer[t_in];
            if (prod < 0 || part.groups[gi].ops.count((size_t)prod)) continue;
            for (auto gj : part.groups_of((size_t)prod)) {
                if (gj == gi || !part.groups[gj].alive) continue;
                succs_[gj].insert(gi);
                preds_[gi].insert(gj);
            }
        }
    }
}

void GroupDAG::clear_edges_for(size_t gi) {
    if (gi >= succs_.size()) return;
    for (auto gj : succs_[gi])
        if (gj < preds_.size()) preds_[gj].erase(gi);
    for (auto gj : preds_[gi])
        if (gj < succs_.size()) succs_[gj].erase(gi);
    succs_[gi].clear();
    preds_[gi].clear();
}
