#pragma once

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

class CostCache;

// ============================================================================
// Partition: mutable representation of op-to-subgraph assignments.
//
// Each Group is a set of op indices. Groups may overlap (recomputation).
// Every op must be in at least one group.
//
// Each Group caches its Subgraph and best TileConfig so that the solver and
// ordering algorithms can use them without calling Subgraph::create again.
//
// The group-level DAG (group_preds / group_succs) is derived from tensor
// dependencies between alive groups and must be rebuilt (rebuild_group_dag)
// after any structural mutation that changes group membership.
// ============================================================================

struct Partition {
    const Problem* prob = nullptr;
    const DAG*     dag  = nullptr;
    CostCache*     cache = nullptr;

    // -------------------------------------------------------------------------
    struct Group {
        std::set<size_t> ops;
        double           cost     = 1e18;   // latency from best_cost(), no retention
        bool             alive    = true;
        int              gen      = 0;

        // Cached evaluation result — populated whenever cost is set.
        // Ordering algorithms read sg and best_cfg directly, avoiding redundant
        // Subgraph::create calls.
        std::optional<Subgraph> sg;
        TileConfig              best_cfg;
    };

    std::vector<Group> groups;

    // -------------------------------------------------------------------------
    // Group-level DAG.
    // Indices into groups[] (sparse — dead groups have empty pred/succ sets).
    // Rebuilt by rebuild_group_dag() after any structural mutation.
    // -------------------------------------------------------------------------
    std::vector<std::set<size_t>> group_preds;   // group → predecessor groups
    std::vector<std::set<size_t>> group_succs;   // group → successor groups
    std::vector<int>              group_in_deg;   // in-degree per group
    // Boundary output tensor → alive group index.  Used by ordering algorithms
    // for retain scoring without scanning all groups.
    std::unordered_map<size_t, size_t> tensor_to_group;

    // --- Construction ---

    static Partition trivial(const Problem& prob, const DAG& dag);

    // --- Index maintenance ---

    // Rebuild op_to_groups_ from scratch.  Call after any mutation.
    void rebuild_index();

    // Rebuild group-level DAG (group_preds/succs/in_deg/tensor_to_group).
    // Must be called after rebuild_index() since it relies on op_to_groups_.
    void rebuild_group_dag();

    // Finalize for use by ordering algorithms.
    //
    // Phase 1 search (greedy descent + FM) mutates ops/cost but never updates
    // Group::sg, Group::best_cfg, or the group-level DAG — those fields are only
    // needed when converting a Partition to a Solution.  Call finalize() once
    // before passing a post-search Partition to the ordering algorithms.
    //
    // What it does:
    //   1. Re-evaluates every alive group via CostCache::evaluate_entry()
    //      (almost always a cache hit — the same op-sets were evaluated during search).
    //   2. Populates Group::sg and Group::best_cfg from the cache entry.
    //   3. Calls rebuild_index() + rebuild_group_dag().
    void finalize();

    // --- Queries ---

    double total_cost() const;
    size_t num_alive()  const;

    // All alive groups containing op — O(1) via index.
    const std::vector<size_t>& groups_of(size_t op) const {
        static const std::vector<size_t> empty;
        if (op_to_groups_.empty() || op >= op_to_groups_.size()) return empty;
        return op_to_groups_[op];
    }

    bool                     is_border_op(size_t op, size_t gi) const;
    std::vector<size_t>      border_ops(size_t gi) const;
    std::set<size_t>         boundary_neighbors(size_t gi) const;
    std::set<size_t>         adjacent_groups(size_t gi) const;

    // Connected components of an op set using DAG op_neighbors.
    std::vector<std::set<size_t>> connected_components(const std::set<size_t>& ops) const;

    // Does any alive, not-yet-scheduled group need tensor t as a boundary input?
    // scheduled[i] == true means group i has already been placed in the order.
    bool future_needs(size_t t, const std::vector<bool>& scheduled) const;

    // Check if the group DAG has a valid topological ordering.
    // Uses the reference approach: for each (consumer_op, input_tensor,
    // producer_op), if the producer is not in the consumer's group, that
    // group depends on any group containing the producer. Kahn's algorithm
    // verifies all dependencies can be resolved.
    // O(ops * inputs * groups). Does NOT check tensor materialization —
    // that's the cost model's job (ephemeral classification at finalization).
    bool is_acyclic() const;

    // Check acyclicity of a hypothetical partition state after a move.
    // These are called by best_move_for_op / best_move_for to guarantee
    // that ONLY feasible moves go on the heap.
    //
    // Each builds a temporary op→groups mapping reflecting the move, then
    // runs the reference Kahn's algorithm. O(ops * inputs * groups_per_op).

    // After MERGE: ga absorbs gb's ops, gb dies.
    bool is_acyclic_after_merge(size_t ga, size_t gb) const;

    // After multi-group MERGE: first group absorbs all, rest die.
    bool is_acyclic_after_merge(const std::vector<size_t>& group_list) const;

    // After STEAL: op removed from ga, added to gb.
    bool is_acyclic_after_steal(size_t op, size_t ga, size_t gb) const;

    // After RECOMPUTE: op added to gb (stays in all existing groups too).
    bool is_acyclic_after_recompute(size_t op, size_t gb) const;

    // --- Evaluation ---

    double eval_set(const std::set<size_t>& ops) const;

    // Partition-aware evaluation: computes which tensors can be treated as
    // ephemeral because all external consumers are served by recomputing groups.
    // Returns the set of tensors that should be force-ephemeral for group gi.
    std::set<size_t> compute_force_ephemeral(size_t gi) const;

    // Evaluate group gi with partition-aware ephemeral classification.
    // Uses force_ephemeral to avoid unnecessary eviction costs.
    // More accurate than eval_set but cannot use CostCache (result depends
    // on what other groups contain, not just the op-set).
    double eval_group_in_context(size_t gi) const;

    struct EjectResult {
        bool   feasible = false;
        double saving   = -1e18;
        double singleton_cost = 1e18;
        std::vector<std::set<size_t>> remainder_components;
        std::vector<double>           component_costs;
    };

    EjectResult eval_eject(size_t op, size_t gi) const;

    // --- Mutation ---

    // Add a new group.  sg and cfg are optional — when provided (e.g. retrieved
    // from CostCache::evaluate_entry) the values are stored directly, avoiding a
    // redundant Subgraph::create on the caller side.
    size_t add_group(std::set<size_t> ops, double cost,
                     std::optional<Subgraph> sg = std::nullopt,
                     TileConfig cfg = {});

    std::vector<size_t> ejectable_ops(size_t gi) const;
    std::vector<size_t> internal_ops(size_t gi)  const;

    struct SplitResult {
        bool             feasible = false;
        double           saving   = -1e18;
        std::set<size_t> side_a, side_b;
        double           cost_a = 1e18, cost_b = 1e18;
    };

    SplitResult                          eval_split(size_t op_a, size_t op_b, size_t gi) const;
    std::vector<std::pair<size_t,size_t>> bridge_edges(size_t gi) const;

    // Would merging groups gi and gj create a cycle in the condensed DAG?
    // Must be checked before any MERGE move in addition to creates_ephemeral_gap.
    // Thin wrapper around DAG::merge_creates_cycle operating on group op-sets.
    bool would_create_cycle(size_t gi, size_t gj) const {
        if (gi >= groups.size() || gj >= groups.size()) return true;
        return dag->merge_creates_cycle(groups[gi].ops, groups[gj].ops);
    }

    // --- Ephemeral gap check ---

    bool creates_ephemeral_gap(const std::set<size_t>& proposed_ops,
                               size_t exclude_ga = SIZE_MAX,
                               size_t exclude_gb = SIZE_MAX) const;

    bool creates_ephemeral_gap(const std::set<size_t>& proposed_ops,
                               const std::vector<size_t>& exclude_groups) const;

    // Check whether splitting group `exclude_gi` into `components` creates a gap.
    // Unlike creates_ephemeral_gap (which checks ONE proposed group against
    // existing groups), this checks MULTIPLE new components simultaneously —
    // required because a tensor might be available from one sibling component
    // but not from any existing group.
    bool split_creates_ephemeral_gap(
        const std::vector<std::set<size_t>>& components,
        size_t exclude_gi) const {
        return split_creates_ephemeral_gap(components, std::set<size_t>{exclude_gi});
    }

    // General form: exclude multiple groups (e.g., for STEAL replacing ga+gb).
    bool split_creates_ephemeral_gap(
        const std::vector<std::set<size_t>>& components,
        const std::set<size_t>& excluded_groups) const;

    // Merge ops into an existing group, updating op_to_groups_ incrementally.
    // Does NOT recompute cost — caller must set groups[gi].cost.
    void merge_ops_into(size_t gi, const std::set<size_t>& ops) {
        if (op_to_groups_.size() < prob->num_ops())
            op_to_groups_.resize(prob->num_ops());
        for (auto op : ops) {
            groups[gi].ops.insert(op);
            op_to_groups_[op].push_back(gi);
        }
    }

private:
    std::vector<std::vector<size_t>> op_to_groups_;
};