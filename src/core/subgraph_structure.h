#pragma once

#include "core/dag.h"
#include "core/types.h"
#include <vector>

// ============================================================================
// SubgraphStructure — the ARCHITECTURE-INDEPENDENT structural facts of a
// subgraph (a connected op set), computed once from (Problem, DAG, ops).
//
// This is pure op-DAG analysis: which tensors cross the subgraph boundary,
// which stay internal, and the fixed pebbling order. It carries NO tiling, NO
// feasibility, NO cost — those are the cost model's job (see cost_model.h).
//
// Both consumers read only this type:
//   * the generic PARTITION layer — acyclicity / ephemeral-gap / coverage /
//     connectivity are all architecture-independent and decided here-or-above,
//     never inside a cost model;
//   * every architecture's COST MODEL — it needs the boundary/ephemeral split
//     and the execution order to size bands, IO, and the working-set peak.
//
// Why factored out (vs. each cost model re-deriving it): the classification is
// identical across architectures (it's a DAG fact — there is no arch where
// "ephemeral" or "boundary output" means something different), so sharing it
// avoids duplication, keeps the partition layer free of the arch template, and
// gives a single source of truth the ephemeral-gap constraint can rely on.
// ============================================================================
class SubgraphStructure {
 public:
  // Classify the op set against the DAG. Always constructs; structural
  // degeneracy is reported via valid() rather than thrown.
  SubgraphStructure(const Problem& prob, const DAG& dag, std::vector<size_t> ops);

  // Structurally usable = non-empty AND produces at least one boundary output.
  // NOTE: this is NOT architecture admissibility (unit-homogeneity, same-shape
  // sinks, tiling-representability) — that is the cost model's CM::create().
  // NOTE: connectivity is NOT rejected here; the partition layer enforces it by
  // construction (connected_components over DAG::op_neighbors, which already
  // includes co-consumer / shared-input edges). This type only classifies.
  [[nodiscard]] bool valid() const { return valid_; }

  [[nodiscard]] const Problem& problem() const { return *prob_; }
  [[nodiscard]] const DAG& dag() const { return *dag_; }
  [[nodiscard]] const std::vector<size_t>& ops() const { return ops_; }
  [[nodiscard]] size_t num_ops() const { return ops_.size(); }

  // Boundary tensors — cross the subgraph edge, so they MUST be DDR-materialized:
  //   inputs  : consumed inside, produced outside (or a graph input)
  //   outputs : produced inside, consumed outside (a sink / externally needed)
  [[nodiscard]] const FlatSet<size_t>& boundary_inputs() const { return boundary_inputs_; }
  [[nodiscard]] const FlatSet<size_t>& boundary_outputs() const { return boundary_outputs_; }

  // Ephemeral tensors — produced AND consumed inside (never touch DDR). The
  // ephemeral-gap constraint forbids one subgraph's ephemeral from being a
  // boundary INPUT of another; that check lives in the partition layer and
  // reads exactly this set.
  [[nodiscard]] const FlatSet<size_t>& ephemeral() const { return ephemeral_; }

  // Sink ops — no in-subgraph consumer; they produce the boundary outputs.
  [[nodiscard]] const std::vector<size_t>& sinks() const { return sinks_; }

  // Fixed depth-first (post-order) execution order over ops: the pebbling order
  // the cost model evaluates the working-set peak along, and the order emitted
  // with the solution. Deterministic (topo-position tie-break) so the cost
  // cache keys on it.
  [[nodiscard]] const std::vector<size_t>& execution_order() const { return dfs_order_; }

 private:
  const Problem* prob_ = nullptr;
  const DAG* dag_ = nullptr;
  std::vector<size_t> ops_;
  FlatSet<size_t> boundary_inputs_;
  FlatSet<size_t> boundary_outputs_;
  FlatSet<size_t> ephemeral_;
  std::vector<size_t> sinks_;
  std::vector<size_t> dfs_order_;
  bool valid_ = false;
};
