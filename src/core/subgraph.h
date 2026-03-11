#pragma once

#include "dag.h"
#include "types.h"
#include <optional>
#include <set>
#include <vector>

// ============================================================================
// Subgraph: a connected group of ops that share a unified tiling grid.
//
// Invariants enforced at construction:
//   1. At least one boundary output tensor
//   2. All boundary outputs have the same (W, H) dimensions
//   3. Connected: ops form a connected sub-DAG (undirected)
//
// Tiling validity constraints (derived from chain execution model):
//   - w divides every output width (boundary AND ephemeral — the unified
//     execution grid requires every op to tile at w × h)
//   - h divides every output height (boundary AND ephemeral)
//   - k divides every MatMul's K dimension (= LHS tensor width)
//   - If any boundary output is produced by a PW op, k must be 1.
//     (PW runs once per tile; with k>1 the cost model for mixed MM→PW
//      chains is ambiguous, so we enforce k=1 for safety.)
//
// Note: this means ops with different output widths CAN share a subgraph
// only if there exists a w that divides all of them.
// ============================================================================

class Subgraph {
public:
  // Factory: returns nullopt if ops don't form a valid subgraph.
  static std::optional<Subgraph> create(const Problem &prob, const DAG &dag,
                                        std::vector<size_t> op_indices);

  // --- Accessors ---

  const Problem &problem() const { return *prob_; }
  const DAG &dag() const { return *dag_; }
  const std::vector<size_t> &ops() const { return ops_; }
  size_t num_ops() const { return ops_.size(); }

  const std::set<size_t> &boundary_inputs() const { return boundary_inputs_; }
  const std::set<size_t> &boundary_outputs() const { return boundary_outputs_; }
  const std::set<size_t> &ephemeral() const { return ephemeral_; }

  // Dimensions of boundary outputs (all must match).
  int64_t output_width() const { return out_W_; }
  int64_t output_height() const { return out_H_; }

  bool has_matmul() const { return has_matmul_; }

  // Reduction dimension for a specific MatMul op (= LHS tensor width).
  int64_t op_K(size_t op_idx) const {
    return prob_->tensors[prob_->ops[op_idx].inputs[0]].width;
  }

  // Maximum K across all MatMul ops in this subgraph.
  int64_t max_K() const { return max_K_; }

  // --- Tiling validity ---

  bool is_valid_tiling(const TileConfig &cfg) const;

  // --- Feasibility ---

  int64_t working_set(const TileConfig &cfg,
                      const std::set<size_t> &retained_from_prev = {},
                      const std::set<size_t> &retain_these = {}) const;

  bool is_feasible(const TileConfig &cfg,
                   const std::set<size_t> &retained_from_prev = {},
                   const std::set<size_t> &retain_these = {}) const;

  // --- Cost evaluation ---

  CostResult compute_cost(const TileConfig &cfg,
                          const std::set<size_t> &retained_from_prev = {},
                          const std::set<size_t> &retain_these = {}) const;

  // --- Parameter enumeration ---

  CostResult best_cost(const std::set<size_t> &retained_from_prev = {},
                       const std::set<size_t> &retain_these = {}) const;

  Subgraph() = default;

private:
  const Problem *prob_ = nullptr;
  const DAG *dag_ = nullptr;
  std::vector<size_t> ops_;

  std::set<size_t> boundary_inputs_;
  std::set<size_t> boundary_outputs_;
  std::set<size_t> ephemeral_;

  int64_t out_W_ = 0, out_H_ = 0;
  bool has_matmul_ = false;
  bool has_pw_sink_ = false;  // true if any boundary output is produced by a PW op
  int64_t max_K_ = 1;

  // Tiling constraints derived from per-tensor roles.
  // w must divide every value in w_divides_, etc.
  // Built at create() time by analyzing how each tensor is consumed.
  std::vector<int64_t> w_divides_;  // w must divide all these
  std::vector<int64_t> h_divides_;  // h must divide all these
  std::vector<int64_t> k_divides_;  // k must divide all these

  // Precomputed per-boundary-tensor info for fast working_set/compute_cost.
  struct BoundaryTensorInfo {
    size_t id;
    int64_t max_lhs_K = 0;    // max K among MM ops using this as LHS (0 = not LHS)
    bool is_mm_rhs = false;    // used as MatMul RHS
    bool is_mm_out = false;    // MatMul boundary output (accumulator — counts in WS)
    bool is_boundary_out = false; // is a boundary output tensor (for eviction tracking)
    bool is_pw_in = false;     // used as Pointwise input
  };
  std::vector<BoundaryTensorInfo> boundary_tensor_info_;

  std::vector<int64_t> ws_cand_; // Divisor candidates for 'w' (must divide all output widths)
  std::vector<int64_t> hs_cand_; // Divisor candidates for 'h' (must divide all output heights)
  std::vector<int64_t> ks_cand_; // Divisor candidates for 'k' (must divide all K values)
};