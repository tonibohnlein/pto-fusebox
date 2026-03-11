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
//
// Performance: maintains an op_to_groups_ index for O(1) lookups of
// which groups contain a given op. Must be rebuilt after mutations.
// ============================================================================

struct Partition {
    const Problem* prob = nullptr;
    const DAG* dag = nullptr;
    CostCache* cache = nullptr;

    struct Group {
        std::set<size_t> ops;
        double cost = 1e18;
        bool alive = true;
        int gen = 0;
    };

    std::vector<Group> groups;

    // --- Construction ---

    static Partition trivial(const Problem& prob, const DAG& dag);

    // --- Index maintenance ---

    // Rebuild op_to_groups_ from scratch. Call after any mutation.
    void rebuild_index();

    // --- Queries ---

    double total_cost() const;
    size_t num_alive() const;

    // All alive groups containing op (O(1) via index)
    const std::vector<size_t>& groups_of(size_t op) const { return op_to_groups_[op]; }

    // Is op on the boundary of group gi?
    bool is_border_op(size_t op, size_t gi) const;

    // All border ops of group gi
    std::vector<size_t> border_ops(size_t gi) const;

    // Ops adjacent to group gi but NOT in gi
    std::set<size_t> boundary_neighbors(size_t gi) const;

    // Alive groups adjacent to gi (O(neighbors) via index)
    std::set<size_t> adjacent_groups(size_t gi) const;

    // Connected components of an op set (using DAG op_neighbors)
    std::vector<std::set<size_t>> connected_components(const std::set<size_t>& ops) const;

    // --- Evaluation ---

    double eval_set(const std::set<size_t>& ops) const;

    struct EjectResult {
        bool feasible = false;
        double saving = -1e18;
        double singleton_cost = 1e18;
        std::vector<std::set<size_t>> remainder_components;
        std::vector<double> component_costs;
    };

    EjectResult eval_eject(size_t op, size_t gi) const;

    // --- Mutation ---

    size_t add_group(std::set<size_t> ops, double cost);

    std::vector<size_t> ejectable_ops(size_t gi) const;
    std::vector<size_t> internal_ops(size_t gi) const;

    struct SplitResult {
        bool feasible = false;
        double saving = -1e18;
        std::set<size_t> side_a, side_b;
        double cost_a = 1e18, cost_b = 1e18;
    };

    SplitResult eval_split(size_t op_a, size_t op_b, size_t gi) const;
    std::vector<std::pair<size_t,size_t>> bridge_edges(size_t gi) const;

private:
    // Index: op -> list of alive group indices containing it.
    // Rebuilt by rebuild_index().
    std::vector<std::vector<size_t>> op_to_groups_;
};