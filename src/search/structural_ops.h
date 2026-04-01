#pragma once

#include "core/dag.h"
#include <set>
#include <vector>
#include <utility>

// ============================================================================
// structural_ops: pure graph topology operations on op sets.
//
// These are container-agnostic: they take a set of ops + a DAG reference
// and return structural information (components, sides, bridge edges).
// No cost model, no Partition, no SolState.
//
// Used by:
//   - partition_moves (partition-level eval/apply)
//   - solution_search (solution-level eval/apply)
//   - evolution (mutation candidate selection)
//   - Partition::eval_eject/eval_split (delegates here for BFS)
// ============================================================================

namespace structural_ops {

// ============================================================================
// Connected components of an op set using DAG neighbor edges.
// Returns vector of disjoint op sets whose union equals `ops`.
// ============================================================================

std::vector<FlatSet<size_t>> connected_components(
    const FlatSet<size_t>& ops, const DAG& dag);

// ============================================================================
// Is the op set still connected after removing `rm`?
// Equivalent to: connected_components(ops \ {rm}).size() == 1
// but faster (single BFS, early exit).
// Returns false if ops.size() <= 1 (nothing left after removal).
// ============================================================================

bool is_connected_without(const FlatSet<size_t>& ops, size_t rm,
                           const DAG& dag);

// ============================================================================
// Eject analysis: structural result of removing `op` from `ops`.
//
// Returns the connected components of (ops \ {op}).
// Does NOT evaluate costs — caller does that.
// ============================================================================

struct EjectAnalysis {
    std::vector<FlatSet<size_t>> remainder_components;
    bool connected;  // true if remainder is a single connected component
};

EjectAnalysis analyze_eject(size_t op, const FlatSet<size_t>& ops,
                             const DAG& dag);

// ============================================================================
// Split analysis: structural result of cutting edge (op_a, op_b) in `ops`.
//
// BFS from op_a through all neighbors EXCEPT the direct (op_a↔op_b) edge.
// If op_b is still reachable, the edge is not a bridge → split fails.
// Otherwise, ops reachable from op_a = side_a, rest = side_b.
// ============================================================================

struct SplitAnalysis {
    FlatSet<size_t> side_a;
    FlatSet<size_t> side_b;
    bool is_bridge;  // true if cutting the edge disconnects the graph
};

SplitAnalysis analyze_split(size_t op_a, size_t op_b,
                             const FlatSet<size_t>& ops,
                             const DAG& dag);

// ============================================================================
// Find all bridge edges in an op set.
// A bridge edge (u,v) is one whose removal disconnects the subgraph
// induced by `ops` in the DAG neighbor graph.
// Returns pairs (min(u,v), max(u,v)) — canonicalized.
// ============================================================================

std::vector<std::pair<size_t, size_t>> bridge_edges(
    const FlatSet<size_t>& ops, const DAG& dag);

} // namespace structural_ops