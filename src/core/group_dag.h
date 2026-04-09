#pragma once

#include "core/types.h"
#include "core/dag.h"
#include "core/flat_set.h"
#include <vector>

class Partition;  // forward declaration

// ============================================================================
// GroupDAG: maintains group-level directed adjacency + topological order
// incrementally.
//
// An edge gi→gj exists when some op in gi produces a tensor consumed by
// an op in gj (and the producer is not also in gj, i.e., not internal).
//
// The topological order enables O(1) "definitely no path" checks:
//   if topo_pos_[from] >= topo_pos_[to], then from cannot reach to.
//
// When edges change, the topo order is repaired locally using the
// MNR-style approach: only re-sort the affected region between the
// backward edge endpoints.
//
// Supports:
//   - Full rebuild from a Partition (adjacency + topo order)
//   - Incremental update after moves (given affected group indices)
//   - O(1) "definitely unreachable" via topo order
//   - O(reachable groups) full reachability BFS (when topo can't rule it out)
//   - O(succs of merge set) cycle detection for merge
// ============================================================================

class GroupDAG {
public:
    GroupDAG() = default;

    // Build from scratch using current op_to_groups_ (must call rebuild_index first).
    // Computes adjacency + topological order via Kahn's algorithm.
    void build(const Partition& part);

    // Incremental update: recompute edges for affected groups and their
    // neighbors, then repair the topological order.
    // Call after rebuild_index() on the partition.
    void update(const Partition& part, const FlatSet<size_t>& affected);

    // Resize to accommodate new groups (e.g., after split/add_group).
    void ensure_size(size_t n);

    // --- Reachability queries ---

    // Can group `from` reach group `to`?
    // Fast path: if topo_pos_[from] >= topo_pos_[to], return false (O(1)).
    // Slow path: BFS on succs_ (O(reachable groups)).
    bool can_reach(size_t from, size_t to) const;

    // Would merging all groups in `merge_set` create a cycle?
    // Uses topo order for fast rejection, falls back to BFS.
    bool merge_creates_cycle(const std::vector<size_t>& merge_set) const;

    // --- Direct access (read-only) ---

    const FlatSet<size_t>& succs(size_t g) const {
        static const FlatSet<size_t> empty;
        return g < succs_.size() ? succs_[g] : empty;
    }
    const FlatSet<size_t>& preds(size_t g) const {
        static const FlatSet<size_t> empty;
        return g < preds_.size() ? preds_[g] : empty;
    }

    // Topological position of group g (lower = earlier in schedule).
    // Dead/nonexistent groups have position -1.
    int topo_pos(size_t g) const {
        return g < topo_pos_.size() ? topo_pos_[g] : -1;
    }

    // --- Move-specific eval/apply ---
    //
    // eval_* returns true if the move would create a cycle.
    // All eval methods use OR-semantics for recomputation: when an op exists
    // in multiple groups, a dependency on its output is satisfied if AT LEAST
    // ONE producer group can be scheduled before the consumer.
    //
    // apply_generic recomputes edges for affected groups and repairs topo order.
    // Call after rebuild_index() on the partition.

    // MERGE(ga absorbs gb): contract two nodes.
    // Cycle iff external successors of {ga,gb} reach back to {ga,gb}.
    bool eval_merge(size_t ga, size_t gb) const;

    // STEAL(op from ga to gb): move op between groups.
    // Three checks with OR-semantics:
    //   1. gb gains op → gb's new input dependencies create a cycle?
    //   2. ga loses op → if op's output was consumed internally, ga now
    //      depends on remaining copies. Cycle iff ALL copies reachable from ga.
    //   3. External consumers of op's output lose ga as source. Cycle iff
    //      ALL remaining sources reachable from each consumer.
    bool eval_steal(const Partition& part, size_t op, size_t from, size_t to) const;

    // RECOMPUTE(op into gb): add op to gb (op stays in other groups).
    // gb gains new input dependencies. Cycle iff ALL producer groups of each
    // input are reachable from gb (no source can precede gb).
    bool eval_recompute(const Partition& part, size_t op, size_t gb) const;

    // DE_RECOMPUTE(op from ga): remove op's copy from ga.
    // ga loses op's output as internal source. Cycle iff ALL remaining
    // copies are reachable from ga / from external consumers.
    bool eval_de_recompute(const Partition& part, size_t op, size_t ga) const;

    // Apply: recompute edges for affected groups and repair topo order.
    void apply_generic(const Partition& part, const FlatSet<size_t>& affected);

    size_t size() const { return succs_.size(); }
    bool empty() const { return succs_.empty(); }

private:
    std::vector<FlatSet<size_t>> succs_;
    std::vector<FlatSet<size_t>> preds_;
    std::vector<int> topo_pos_;     // group → position in topo order
    std::vector<size_t> topo_order_; // position → group (inverse of topo_pos_)

    // Recompute edges for a single group gi.
    void rebuild_edges_for(const Partition& part, size_t gi);

    // Remove all edges involving gi (from both succs_ and preds_).
    void clear_edges_for(size_t gi);

    // Full topological sort via Kahn's algorithm. Called by build().
    void compute_topo_order(const Partition& part);

    // Repair topo order after incremental edge changes.
    // Only re-sorts the region affected by backward edges.
    void repair_topo_order(const Partition& part, const FlatSet<size_t>& affected);
};
