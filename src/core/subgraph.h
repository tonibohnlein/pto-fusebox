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

  // Fixed depth-first execution order over this subgraph's ops (post-order DFS
  // from the sinks, each op after its in-subgraph producers). This is the
  // pebbling order: peak working set is evaluated along it, and it is emitted
  // with the solution because the peak depends on it. (Optimizing the order is
  // black-pebbling = PSPACE-complete; one DFS post-order is the accepted
  // heuristic — optimal for chains/trees, good for diamonds.) Empty until
  // create() populates it.
  const std::vector<size_t> &execution_order() const { return dfs_order_; }

  // Op id of the boundary-output MatMul (the sink whose contraction may be
  // parallel-split across cores), or -1 if the subgraph has none. Its emitted
  // per-core k is the composed config.k (L1-fit capped by the split-K share).
  int64_t sink_matmul_op() const { return sink_mm_op_; }

  // --- Tiling validity ---

  bool is_valid_tiling(const TileConfig &cfg) const;

  // --- Feasibility ---

  int64_t working_set(const TileConfig &cfg,
                      const FlatSet<size_t> &retained_from_prev = {},
                      const FlatSet<size_t> &retain_these = {}) const;

  // 910B cube peak L1 working set (bytes) along the fixed execution order: the
  // red-blue pebble peak over the per-output-tile schedule, with each matmul's
  // single-core k-tile derived greedily (largest 16-aligned divisor of its K
  // whose boundary operand strip fits the headroom left by the live intermediate
  // bands). Returns INT64_MAX if infeasible at cfg. Optionally returns the
  // derived per-op k (indexed by op id; 0 for non-matmul / non-participating).
  // This is the dynamic peak that replaces the old static operand-strip SUM.
  int64_t cube_peak_l1(const TileConfig &cfg,
                       std::vector<int64_t> *perop_k = nullptr) const;

  // Vector (UB) pebble peak — the dynamic on-chip working set of a vector
  // subgraph, the analog of cube_peak_l1 for the cube. Each tensor's tile
  // footprint is [min(cfg.w,W), min(cfg.h,H)]; the reduced axis is coupled to its
  // full extent (cfg forces it), so a reduction input spans the whole row/col
  // band. Ephemeral tensors are resident across [producer, last consumer];
  // boundary tiles are transient. Returns the peak UB bytes. (No W-axis streaming
  // yet — a reduction band spans its FULL reduced extent; the streaming
  // single-core reduction that would shrink it is the next increment.)
  // reduce_chunk caps the reduced axis at a single-core STREAMING granularity:
  // the reduction is accumulated chunk-by-chunk on one core, so a reused ephemeral
  // (softmax's e) is held only a chunk at a time (recomputed past the reduction)
  // rather than as a full [reduced_extent, h] band. INT64_MAX = no streaming
  // (materialize the full band). Lets a large-W reduction fit UB.
  // stream_axis selects which axis reduce_chunk caps: 0 => the reduced axis (a
  // reduction's coupled online accumulation); 1/2 => width/height, used to stream
  // a pure-pointwise tile (which has no coupled axis — the single-core k-stream).
  int64_t vector_peak_ub(const TileConfig &cfg,
                         const FlatSet<size_t> &retained_from_prev = {},
                         const FlatSet<size_t> &retain_these = {},
                         int64_t reduce_chunk = INT64_MAX,
                         int stream_axis = 0) const;

  // The derived single-core k-stream of a vector subgraph at this tiling — the
  // analog of the matmul per-op seq-k. axis: 0 = materialized (whole tile fits
  // UB, no sub-streaming); 1/2 = the width/height axis streamed in UB-chunks.
  // chunk: the streaming granularity along that axis (INT64_MAX when materialized,
  // 0 when infeasible). For a reduction the axis is the coupled reduced one; for
  // a pure pointwise it's the larger tile axis (shrinks the footprint most).
  struct VecStream { int axis = 0; int64_t chunk = 0; };
  VecStream vector_stream(const TileConfig &cfg,
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

  // 910B per-core, byte-based, two-pool feasibility. Forks on cube-vs-vector:
  //   cube  : operand strips fit L1 (l1_capacity), output fits L0c (cube_capacity)
  //   vector: tile + ephemerals fit UB (vec_capacity)
  // double_buffer halves the streaming pools (L1/UB). When the relevant 910B
  // pool budgets are 0 (legacy/competition), falls back to the single
  // fast_memory_capacity element-count check.
  bool fits_on_chip(const TileConfig &cfg,
                    const FlatSet<size_t> &retained_from_prev,
                    const FlatSet<size_t> &retain_these) const;

  // Engine behind cube_peak_l1(): sweep the execution order, accumulate live
  // intermediate-band bytes, derive each matmul's per-op k against the headroom,
  // return the peak L1 bytes (INT64_MAX if infeasible). sink_K_eff is the sink
  // matmul's per-core contraction share (= output_K_ for S=1 feasibility; =
  // output_K_/S when emitting the schedule for a known parallel split).
  int64_t derive_exec(const TileConfig &cfg, int64_t sink_K_eff,
                      const FlatSet<size_t> &retained_from_prev,
                      const FlatSet<size_t> &retain_these,
                      std::vector<int64_t> *perop_k_out) const;

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

  // Ephemeral MM outputs that serve as accumulators in MM→PW epilogue patterns.
  // These are physically resident in fast memory across all k-steps (w×h each).
  // Only populated when has_simple_epilogue_ is true.
  std::vector<size_t> ephemeral_accumulators_;

  int64_t out_W_ = 0, out_H_ = 0;
  bool has_matmul_ = false;
  // 910B reduction (vector): a Reduction couples its reduced axis (the whole
  // row/col must be present to reduce it), so the tile spans the FULL reduced
  // dim and only the non-reduced dim is tiled for spatial parallelism. The
  // reduced axis may still be split ACROSS cores (override), paying a thin
  // per-partial merge. reduced_axis_: 0=none, 1=width, 2=height. Competition
  // never emits Reduction, so these stay default on the single-context path.
  bool has_reduction_ = false;
  int reduced_axis_ = 0;
  // Full extent of the reduced axis (the un-reduced data width/height the tile
  // must span). May exceed out_W_/out_H_ when the reduction output IS the sink
  // (e.g. a bare rowmax: out is [1,H] but the tile must cover the full W).
  int64_t reduced_extent_ = 0;
  bool has_pw_sink_ = false;
  bool reduction_is_sink_ = false;  // a sink op is a Reduction — the ONLY case
                                    // where the reduced axis may be parallel-split
                                    // across cores (cross-core merge = DDR, so an
                                    // internal reduction split must be a subgraph cut)
  bool has_simple_epilogue_ = false;  // MM→PW(chain) epilogue pattern detected
  int64_t max_K_ = 1;
  int64_t output_K_ = 1;   // K of the boundary-output-producing MatMul
  int64_t sink_mm_op_ = -1;  // op id of the boundary-output MatMul (-1 if none);
                             // its derived per-op k is the displayed config.k

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

  // Depth-first topological execution order (see execution_order()). Fixed at
  // construction; drives the peak-working-set sweep and the emitted schedule.
  std::vector<size_t> dfs_order_;

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