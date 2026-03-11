#pragma once

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include <set>
#include <vector>

class CostCache;  // forward declaration

// ============================================================================
// Partition: mutable representation of op-to-subgraph assignments.
//
// Each Group is a set of op indices. Groups may overlap (recomputation).
// Every op must be in at least one group. Groups carry a generation counter
// for lazy heap invalidation in the local search.
//
// Validity: Subgraph::create is the authoritative check. It verifies
// connectivity (including shared-input edges), ephemeral fan-out, and
// boundary output dimension consistency.
// ============================================================================

struct Partition {
    const Problem* prob = nullptr;
    const DAG* dag = nullptr;
    CostCache* cache = nullptr;  // shared cost cache (optional, speeds up eval_set)

    struct Group {
        std::set<size_t> ops;
        double cost = 1e18;
        bool alive = true;
        int gen = 0;
    };

    std::vector<Group> groups;

    // --- Construction ---

    // One op per group, each evaluated with best_cost()
    static Partition trivial(const Problem& prob, const DAG& dag);

    // --- Queries ---

    double total_cost() const;
    size_t num_alive() const;

    // All alive groups containing op
    std::vector<size_t> groups_of(size_t op) const;

    // Is op on the boundary of group gi? (has DAG neighbor outside gi)
    bool is_border_op(size_t op, size_t gi) const;

    // All border ops of group gi
    std::vector<size_t> border_ops(size_t gi) const;

    // Ops adjacent to group gi but NOT in gi
    std::set<size_t> boundary_neighbors(size_t gi) const;

    // Alive groups adjacent to gi (share a tensor boundary)
    std::set<size_t> adjacent_groups(size_t gi) const;

    // Find connected components of an op set using the DAG.
    // Returns a vector of op sets, each forming a connected component.
    std::vector<std::set<size_t>> connected_components(const std::set<size_t>& ops) const;

    // --- Evaluation ---

    // Evaluate a candidate op set: create Subgraph + best_cost().
    // Returns cost, or 1e18 if invalid (disconnected, infeasible, etc.).
    double eval_set(const std::set<size_t>& ops) const;

    // Result of ejecting an op from a group
    struct EjectResult {
        bool feasible = false;
        double saving = -1e18;          // old_cost - new_total_cost
        double singleton_cost = 1e18;   // cost of {op} (0 if op in other groups)
        std::vector<std::set<size_t>> remainder_components;  // may be >1 if disconnected
        std::vector<double> component_costs;
    };

    // Evaluate ejecting op from group gi. Handles disconnection.
    // If op is in other groups, no singleton needed.
    EjectResult eval_eject(size_t op, size_t gi) const;

    // --- Mutation ---

    // Add a new group, returns its index
    size_t add_group(std::set<size_t> ops, double cost);

    // Ops in group gi that are on the boundary (have neighbors outside gi)
    // and whose removal leaves a non-empty group. These are eject candidates.
    std::vector<size_t> ejectable_ops(size_t gi) const;

    // All internal ops in group gi (no DAG neighbors outside gi)
    std::vector<size_t> internal_ops(size_t gi) const;

    // Result of splitting a group at a DAG edge
    struct SplitResult {
        bool feasible = false;
        double saving = -1e18;
        std::set<size_t> side_a, side_b;
        double cost_a = 1e18, cost_b = 1e18;
    };

    // Evaluate splitting group gi at DAG edge (op_a → op_b).
    // Both must be in gi. If the edge is a bridge, returns the two sides.
    SplitResult eval_split(size_t op_a, size_t op_b, size_t gi) const;

    // Find all bridge edges within group gi. Returns (op_a, op_b) pairs.
    std::vector<std::pair<size_t,size_t>> bridge_edges(size_t gi) const;
};