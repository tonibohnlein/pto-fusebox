#pragma once

#include "partition/partition.h"
#include "core/cost_cache.h"
#include <string>
#include <vector>
#include <functional>

// ============================================================================
// Partition initialization strategies.
//
// Each strategy produces a valid Partition from the problem: every op is
// covered by at least one group, each group forms a valid Subgraph
// (connected, feasible), and costs are evaluated via best_cost().
//
// Strategies differ in how they discover fusion opportunities:
//   - trivial:       no fusion (baseline)
//   - chain+edge:    structural chain detection + greedy by tensor size
//   - seed+grow:     cluster around expensive ops
//   - rev-topo:      bottom-up greedy merging
//   - random:        random feasible merges (diversity for multi-start)
//
// CostCache:
//   All strategies accept an optional CostCache* (default nullptr).
//   Wiring in a cache makes repeated eval_set calls on the same op-set
//   O(1) after the first evaluation.  The warm cache carries into Phase 1
//   FM search when the same CostCache instance is reused — init evaluations
//   become free hits in the search.
//
// Feasibility invariants enforced by every strategy:
//   1. Memory: every alive group has eval_set() < 1e18 (valid tiling exists).
//   2. No cycles: merge_creates_cycle checked before every merge.
//   3. No ephemeral gap: creates_ephemeral_gap checked before every merge.
//   (PW-sink k=1 constraint is enforced inside Subgraph::is_valid_tiling,
//    automatically respected by best_cost/eval_set.)
// ============================================================================

struct InitStrategy {
    std::string name;
    std::function<Partition(const Problem&, const DAG&, CostCache*)> init;
};

// One op per group (baseline, no fusion)
Partition init_trivial(const Problem& prob, const DAG& dag,
                       CostCache* cache = nullptr);

// Maximal chain detection + greedy merge by intermediate tensor size
Partition init_chain_then_edge(const Problem& prob, const DAG& dag,
                               CostCache* cache = nullptr);

// Seed on expensive ops, greedily grow by adding best neighbor
Partition init_seed_and_grow(const Problem& prob, const DAG& dag,
                             CostCache* cache = nullptr);

// Process ops in reverse topological order, merge into successor groups
Partition init_reverse_topo(const Problem& prob, const DAG& dag,
                            CostCache* cache = nullptr);

// Random partition: start from singletons, apply random merges.
// Produces different results each call (internal atomic counter for seeds).
Partition init_random(const Problem& prob, const DAG& dag,
                      CostCache* cache = nullptr);

// All registered strategies
std::vector<InitStrategy> all_init_strategies();

// Run every strategy once, return the partition with lowest total_cost.
// All strategies share `cache` — evaluations from early strategies benefit
// later ones, and the warm cache carries into Phase 1 FM search when the
// caller reuses the same CostCache instance.
Partition best_initial(const Problem& prob, const DAG& dag,
                       CostCache* cache = nullptr);

// ============================================================================
// Feasibility validator
//
// Checks the two meaningful invariants for a partition produced by any init strategy:
//   1. Memory: every alive group has a feasible tiling (eval_set < 1e18).
//   2. No ephemeral gap: no tensor would be ephemeral in its producing group
//      while an external consumer group has no other source.
//
// Note: a "no cycles in condensed DAG" check is NOT included here because the
// condensed group DAG of any partition of a DAG is always acyclic by definition.
// merge_creates_cycle is a pre-condition on MERGE moves in the search, not a
// partition invariant.
//
// Returns an empty string on success, or a description of the first violation.
// Primarily for testing and debug assertions; not called on the hot path.
// ============================================================================
std::string verify_partition_feasibility(const Partition& part);