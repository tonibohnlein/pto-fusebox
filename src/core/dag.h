#pragma once

#include "core/types.h"
#include <set>
#include <vector>

// ============================================================================
// DAG: op-level dependency graph derived from the problem
// ============================================================================

struct DAG {
    size_t num_ops;

    std::vector<int> tensor_producer;                  // tensor -> producing op (-1 = graph input)
    std::vector<std::vector<size_t>> tensor_consumers; // tensor -> consuming ops
    std::vector<std::set<size_t>> op_preds;            // op -> predecessor ops
    std::vector<std::set<size_t>> op_succs;            // op -> successor ops
    std::vector<size_t> graph_inputs;                  // tensors with no producer
    std::vector<size_t> graph_outputs;                 // tensors with no consumer

    // Build the DAG from a problem specification
    static DAG build(const Problem& prob);

    // Topological sort of all ops
    std::vector<size_t> topo_sort() const;

    // Would merging two op-sets create a cycle in the condensed DAG?
    // True iff there exists a directed path from one set to the other
    // through ops outside both sets.
    bool merge_creates_cycle(const std::set<size_t>& a,
                             const std::set<size_t>& b) const;
};
