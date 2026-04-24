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

  const FlatSet<size_t> &boundary_inputs() const { return boundary_inputs_; }
  const FlatSet<size_t> &boundary_outputs() const { return boundary_outputs_; }
  const FlatSet<size_t> &ephemeral() const { return ephemeral_; }

  // Dimensions of boundary outputs (all must match).
  int64_t output_width() const { return out_W_; }
  int64_t output_height() const { return out_H_; }

  bool has_matmul() const { return has_matmul_; }

  // Number of distinct role-signature entries for a tensor in this subgraph.
  // Returns 0 for non-boundary tensors. Multi-role tensors (per #70 + #73)
  // return >1. Mainly for tests and diagnostics.
  size_t boundary_entries_for(size_t tensor_id) const {
    if (tensor_id >= tensor_id_to_infos_.size()) return 0;
    return tensor_id_to_infos_[tensor_id].size();
  }

  // Reduction dimension for a specific MatMul op (= LHS tensor width).
  int64_t op_K(size_t op_idx) const {
    return prob_->tensors[prob_->ops[op_idx].inputs[0]].width;
  }

  // Maximum K across all MatMul ops in this subgraph.
  int64_t max_K() const { return max_K_; }

  // Reduction dimension used in the temporal loop (nk = output_K / k).
  // For MatMul-sink subgraphs: K of the boundary-output-producing MatMul.
  // For PW-sink + MatMul subgraphs: max_K_ of the internal MatMul(s).
  //   nk is still enforced to 1 (PW constraint), but k = output_K in the
  //   solution file tells the runtime to do the full K in one pass.
  // For pure-PW subgraphs: 1 (no K dimension).
  int64_t output_K() const { return output_K_; }

  // --- Tiling validity ---

  bool is_valid_tiling(const TileConfig &cfg) const;

  // --- Feasibility ---

  int64_t working_set(const TileConfig &cfg,
                      const FlatSet<size_t> &retained_from_prev = {},
                      const FlatSet<size_t> &retain_these = {}) const;

  bool is_feasible(const TileConfig &cfg,
                   const FlatSet<size_t> &retained_from_prev = {},
                   const FlatSet<size_t> &retain_these = {}) const;

  // --- Cost evaluation ---

  CostResult compute_cost(const TileConfig &cfg,
                          const FlatSet<size_t> &retained_from_prev = {},
                          const FlatSet<size_t> &retain_these = {}) const;

  // --- Parameter enumeration ---

  CostResult best_cost(const FlatSet<size_t> &retained_from_prev = {},
                       const FlatSet<size_t> &retain_these = {}) const;


  Subgraph() = default;

private:
  // working_set without the is_valid_tiling() guard — caller must ensure
  // validity before calling.  Used by compute_cost() and is_feasible() to
  // avoid redundant validation when best_cost() already checked.
  int64_t working_set_unchecked(const TileConfig &cfg,
                                const FlatSet<size_t> &retained_from_prev,
                                const FlatSet<size_t> &retain_these) const;

  const Problem *prob_ = nullptr;
  const DAG *dag_ = nullptr;
  std::vector<size_t> ops_;

  FlatSet<size_t> boundary_inputs_;
  FlatSet<size_t> boundary_outputs_;
  FlatSet<size_t> ephemeral_;
  // Ephemerals split by producer-op type. Used for the granule-fit check in
  // is_valid_tiling:
  //   PW producer: slice ≤ (cfg.w, cfg.h) — PW has no k-loop, one granule
  //                per execution.
  //   MM producer: slice ≤ native — MM's internal k-loop covers multiple
  //                native-sized steps; the bound is the hardware limit.
  // Defensive: mostly subsumed by cfg ≤ native + role propagation, but cheap
  // and catches role-propagation surprises in long op chains.
  std::vector<size_t> pw_produced_ephemerals_;
  std::vector<size_t> mm_produced_ephemerals_;

  int64_t out_W_ = 0, out_H_ = 0;
  bool has_matmul_ = false;
  bool has_pw_sink_ = false;
  int64_t max_K_ = 1;
  int64_t output_K_ = 1;   // K of the boundary-output-producing MatMul

  // Prologue-PW geometric condition (issue #71 Rules 2/3):
  //   Pointwise feeding a matmul's LHS → require cfg.w ≥ matmul.K
  //   Pointwise feeding a matmul's RHS → require cfg.h ≥ matmul.K
  // Without this, the matmul's k-loop would ask for LHS/RHS data a single
  // PW tile hasn't produced yet — violating per-tile execution.
  // Populated per-subgraph as the max K across all such PW→MM pairs.
  // Zero means "no constraint" (no prologue PW of the relevant orientation).
  int64_t prologue_cfg_w_min_ = 0;
  int64_t prologue_cfg_h_min_ = 0;

  // Role-based tiling constraints.
  std::vector<int64_t> w_divides_;
  std::vector<int64_t> h_divides_;
  std::vector<int64_t> k_divides_;

  std::vector<size_t> reverse_topo_ops_;
  std::vector<bool> is_sink_op_vec_;

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

  // Per-tensor tiling sources (for ALL tensors, not just boundary).
  // Used by compute_cost to compute per-op scale factors.
  // Indexed by tensor id; only entries for tensors in this subgraph are valid.
  struct TilePair {
    BoundaryTensorInfo::TileSource h = BoundaryTensorInfo::FROM_NTW;
    BoundaryTensorInfo::TileSource v = BoundaryTensorInfo::FROM_NTH;
  };
  std::vector<TilePair> tensor_tiling_;

  std::vector<int64_t> ws_cand_;
  std::vector<int64_t> hs_cand_;
  std::vector<int64_t> ks_cand_;

  // Flat lookup: tensor index → list of indices into boundary_tensor_info_.
  // Empty list means the tensor isn't a boundary tensor of this subgraph.
  // A tensor can have multiple entries when it's used under multiple distinct
  // role signatures (per issues #70 + #73 — working set sums distinct slices).
  // Currently populated 1:1 per tensor; multi-entry materialization is
  // introduced in a follow-up step.
  std::vector<std::vector<int>> tensor_id_to_infos_;
};