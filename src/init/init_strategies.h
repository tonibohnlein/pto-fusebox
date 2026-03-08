#pragma once

#include "partition/partition.h"
#include <string>
#include <vector>
#include <functional>

// ============================================================================
// Partition initialization strategies.
//
// Each strategy produces a valid Partition from the problem: every op is
// covered by at least one group, each group forms a valid Subgraph
// (single-sink, connected), and costs are evaluated via best_cost().
//
// Strategies differ in how they discover fusion opportunities:
//   - trivial: no fusion (baseline)
//   - chain_then_edge: structural chain detection + greedy by tensor size
//   - seed_and_grow: cluster around expensive ops
//   - reverse_topo: bottom-up greedy merging
// ============================================================================

struct InitStrategy {
    std::string name;
    std::function<Partition(const Problem&, const DAG&)> init;
};

// One op per group (baseline, no fusion)
Partition init_trivial(const Problem& prob, const DAG& dag);

// Maximal chain detection + greedy merge by intermediate tensor size
Partition init_chain_then_edge(const Problem& prob, const DAG& dag);

// Seed on expensive ops, greedily grow by adding best neighbor
Partition init_seed_and_grow(const Problem& prob, const DAG& dag);

// Process ops in reverse topological order, merge into successor groups
Partition init_reverse_topo(const Problem& prob, const DAG& dag);

// All registered strategies
std::vector<InitStrategy> all_init_strategies();

// Run all strategies, return the one with lowest total_cost
Partition best_initial(const Problem& prob, const DAG& dag);
