#pragma once

#include "core/dag.h"
#include "core/types.h"
#include <optional>
#include <set>
#include <vector>

// ============================================================================
// Subgraph: a connected group of ops that share a unified tiling grid.
//
// Invariants enforced at construction:
//   1. At least one boundary output tensor
//   2. All boundary outputs have the same (W, H) dimensions
//   3. Connected: ops form a connected sub-DAG (via DAG edges or shared inputs)
//   4. Ephemeral tensors: produced + consumed internally + ALL DAG consumers
//      are inside the subgraph.  Tensors with external consumers are boundary
//      outputs (materialized to slow memory for external access).
//   5. Ephemeral tensors may have multiple internal consumers (fan-out OK).
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

  // Reduction dimension of the output-producing MatMul(s).
  // This is the K that drives the temporal loop (nk = output_K / k).
  // For PW-sink subgraphs, output_K_ == 1 (k is forced to 1).
  int64_t output_K() const { return output_K_; }

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
  bool has_pw_sink_ = false;
  int64_t max_K_ = 1;
  int64_t output_K_ = 1;   // K of the boundary-output-producing MatMul

  // Role-based tiling constraints.
  std::vector<int64_t> w_divides_;
  std::vector<int64_t> h_divides_;
  std::vector<int64_t> k_divides_;

  // Precomputed per-boundary-tensor tiling info for working_set/compute_cost.
  //
  // Each boundary tensor's slice dimensions are determined by backward
  // propagation through the chain, exactly matching the reference evaluator's
  // h_tiles/v_tiles system:
  //
  //   h_tiles divides tensor.width  → slice_w = tensor.width / h_tiles
  //   v_tiles divides tensor.height → slice_h = tensor.height / v_tiles
  //   slice_size = slice_w × slice_h
  //
  // h_tiles and v_tiles depend on the tiling config (w, h, k). We store
  // their SOURCE so they can be evaluated at runtime for any config.
  //
  // Propagation rules (from reference evaluator):
  //   Output boundary: h = FROM_NTW (= W_out/w), v = FROM_NTH (= H_out/h)
  //   PW:              inputs inherit output's (h, v)
  //   Non-sink MM:     LHS: h = FIXED_1, v = output.v
  //                    RHS: h = output.h, v = FIXED_1
  //   Sink MM (nk>1):  LHS: h = FROM_NK (= K/k), v = output.v
  //                    RHS: h = output.h, v = FROM_NK
  //
  // Transfer timing is derived from the sources:
  //   FIXED_1 → position never changes in that dimension
  //   FROM_NTW → position changes with output column (h_pos)
  //   FROM_NTH → position changes with output row (v_pos)
  //   FROM_NK → position changes every k-step
  struct BoundaryTensorInfo {
    size_t id;
    int64_t full_size;               // width * height (precomputed)

    // How this tensor's tiling is determined
    enum TileSource : uint8_t {
      FIXED_1  = 0,  // h_tiles or v_tiles = 1 (full extent in this dim)
      FROM_NTW = 1,  // = W_out / w (output column tiling)
      FROM_NTH = 2,  // = H_out / h (output row tiling)
      FROM_NK  = 3   // = output_K / k (depth tiling for sink split-K)
    };
    TileSource h_source = FROM_NTW;  // determines h_tiles
    TileSource v_source = FROM_NTH;  // determines v_tiles

    // Output roles
    bool is_boundary_out = false;    // h×w evicted per tile
    bool is_mm_out = false;          // MatMul accumulator (h×w, resident)

    // Internal production flag (no mem_in cost)
    bool is_internally_produced = false;

    // Evaluate h_tiles for a given config
    int64_t eval_h_tiles(int64_t ntw, int64_t nk) const {
      switch (h_source) {
        case FIXED_1:  return 1;
        case FROM_NTW: return ntw;
        case FROM_NTH: return 1; // shouldn't happen, but safe
        case FROM_NK:  return nk;
      }
      return 1;
    }
    int64_t eval_v_tiles(int64_t nth, int64_t nk) const {
      switch (v_source) {
        case FIXED_1:  return 1;
        case FROM_NTW: return 1; // shouldn't happen, but safe
        case FROM_NTH: return nth;
        case FROM_NK:  return nk;
      }
      return 1;
    }
  };
  std::vector<BoundaryTensorInfo> boundary_tensor_info_;

  std::vector<int64_t> ws_cand_;
  std::vector<int64_t> hs_cand_;
  std::vector<int64_t> ks_cand_;

  // Flat lookup: tensor index → index into boundary_tensor_info_, or -1.
  // Enables O(1) lookup in working_set for the retained_from_prev scan.
  // Sized to prob_->num_tensors() and populated by create().
  std::vector<int> tensor_id_to_info_;
};