#pragma once

#include "core/cost_model.h"
#include "core/dag.h"
#include "core/types.h"
#include <optional>
#include <set>
#include <vector>

// ============================================================================
// Ascend910BCost: the 910B cost model. A connected group of ops that share a
// unified tiling grid, together with the 910B-specific tiling/feasibility/cost
// model over it. Models the CostModel concept (see cost_model.h); the pipeline
// reaches it through the `Subgraph` alias (subgraph.h). Structural
// classification is delegated to SubgraphStructure inside create().
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

class Ascend910BCost {
public:
  // Factory: returns nullopt if ops don't form a valid subgraph.
  //
  // allow_mixed relaxes unit-homogeneity to permit a fused CUBE+VECTOR group.
  // The base model leaves it false (homogeneous); Ascend910BMixed passes true.
  // It is the ONLY behavioural difference between the two models — the cost of a
  // mixed group, when one is built, is the shared compute_cost mixed branch.
  static std::optional<Ascend910BCost> create(const Problem &prob, const DAG &dag,
                                              std::vector<size_t> op_indices,
                                              bool allow_mixed = false);

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
  // Returns 0 for non-boundary tensors. Multi-role tensors
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

  // Solver-owned vector sub-stream specification. Candidate costing derives it
  // as a stack-local value; final-solution/forced-plan consumers call this again
  // for the winning config. It is intentionally absent from CostResult so the
  // local-search cache stays compact. vector_stream() remains a compatibility
  // axis/chunk view for existing callers.
  VectorStreamPlan vector_stream_plan(const TileConfig &cfg,
                                      const FlatSet<size_t> &retained_from_prev = {},
                                      const FlatSet<size_t> &retain_these = {}) const;

  // Compatibility view: axis 0 + chunk INT64_MAX means materialized; chunk 0
  // means infeasible, matching the historical API.
  struct VecStream { int axis = 0; int64_t chunk = 0; };
  VecStream vector_stream(const TileConfig &cfg,
                          const FlatSet<size_t> &retained_from_prev = {},
                          const FlatSet<size_t> &retain_these = {}) const;

  bool is_feasible(const TileConfig &cfg,
                   const FlatSet<size_t> &retained_from_prev = {},
                   const FlatSet<size_t> &retain_these = {}) const;

  // --- Cost evaluation ---

  // Cost of one (cube-only or vector-only) subgraph. virtual so Ascend910BMixed
  // can add the third (mixed cube+vector) type; best_cost dispatches through it.
  virtual CostResult compute_cost(const TileConfig &cfg,
                                  const FlatSet<size_t> &retained_from_prev = {},
                                  const FlatSet<size_t> &retain_these = {}) const;

  // --- Parameter enumeration ---

  CostResult best_cost(const FlatSet<size_t> &retained_from_prev = {},
                       const FlatSet<size_t> &retain_these = {}) const;

  // Enumerate every FEASIBLE (config, cost) candidate for this subgraph (the same grid the
  // argmin best_cost picks from). Used by the cost-vs-wall-time validation: dump the plans + their
  // modeled costs, then force one for the device emit and measure its latency. Not on the solver
  // hot path.
  std::vector<std::pair<TileConfig, CostResult>> enumerate_plans() const;


  Ascend910BCost() = default;

  // Polymorphic base (Ascend910BMixed derives). Cost models are value types
  // (stored by value in ScheduleStep, never deleted via a base pointer), so no
  // virtual destructor is needed; the implicit copy/move carry the vptr.

protected:  // Ascend910BMixed::compute_cost reads these to cost the mixed type.
  // 910B per-core, byte-based, two-pool feasibility. Forks on cube-vs-vector:
  //   cube  : operand strips fit L1 (l1_capacity), output fits L0c (cube_capacity)
  //   vector: tile + ephemerals fit UB (vec_capacity)
  // Always double-buffered, but the pools are NOT halved: the two ping-pong
  // buffers together ARE the L1/UB, so feasibility uses the full capacity and
  // the emit halves the per-load k instead. When the relevant 910B pool budgets
  // are 0 (legacy/competition), falls back to the single fast_memory_capacity
  // element-count check.
  bool fits_on_chip(const TileConfig &cfg,
                    const FlatSet<size_t> &retained_from_prev,
                    const FlatSet<size_t> &retain_these) const;

  // Two-pool feasibility for a MIXED cube+vector kernel (used by
  // Ascend910BMixed). A fused mixed kernel needs BOTH on-chip pools live at once:
  // the cube stage's matmul operand strips fit L1 and its output tile fits L0c,
  // AND the vector stage's tile working set fits UB. (The crossing intermediate
  // lives in L0c on the cube side and UB on the vector side, joined by the
  // off-chip GM ring — it is not an L1/UB resident band.) This is the constraint
  // that can make a fusion infeasible at a large shared tile where the separate
  // kernels — each tiling for its own single pool — both fit.
  bool mixed_fits_on_chip(const TileConfig &cfg,
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

  // Matmul boundary-operand reload (BYTES) at this tiling: the distribution-aware
  // M*N*K*(1/w + 1/h) term, deduped per (tensor, role). Shared by the cube cost
  // and the mixed cost so fusion does not silently drop the operand reload.
  // matmul_at_output_grid=true treats every matmul as tiled at the output grid
  // (w_i = min(cfg.w, N)) — correct for a feed-forward mixed kernel whose matmul
  // output is consumed elementwise by the vector stage (it is the effective sink),
  // vs the cube path's chained-intermediate default (full-width band, w_i = N).
  // Optionally splits the reload BYTES per port (lhs via L0A, rhs via L0B) into the
  // out-params — used to derive the L1->L0 extract (MTE1) tiebreaker in best_cost.
  double cube_operand_reload(const TileConfig &cfg,
                             bool matmul_at_output_grid = false,
                             double *lhs_bytes_out = nullptr,
                             double *rhs_bytes_out = nullptr) const;

  const Problem *prob_ = nullptr;
  const DAG *dag_ = nullptr;
  std::vector<size_t> ops_;

  FlatSet<size_t> boundary_inputs_;
  FlatSet<size_t> boundary_outputs_;
  FlatSet<size_t> ephemeral_;
  // PW-produced ephemerals, for the granule-fit check in is_valid_tiling: a PW
  // op has no k-loop, so its output slice must fit within one (cfg.w, cfg.h)
  // granule. Defensive — mostly subsumed by cfg ≤ native + role propagation,
  // but cheap and catches role-propagation surprises in long op chains.
  std::vector<size_t> pw_produced_ephemerals_;

  int64_t out_W_ = 0, out_H_ = 0;
  bool has_matmul_ = false;   // group has ≥1 CUBE (matmul) op
  bool has_vector_ = false;   // group has ≥1 VECTOR (pointwise/reduction) op.
                              // has_matmul_ && has_vector_ ⇒ a MIXED kernel
                              // (allowed only when Problem::fuse_cube_vector).
  // Max cube↔vector unit ALTERNATIONS along any dependency path in the group.
  // The mixed model's `max` overlap is the SKEWED cost, valid only for a single
  // round-trip: depth ≤ 2 (the 4 canonical shapes c→v / v→c / v→c→v / c→v→c;
  // #1900's depth-2 buffers cap the validated skew). Deeper multi-round-trips
  // demote to Sequential, so create() REJECTS them (depth > 2) and compute_cost
  // asserts the survivor is ≤ 2. 0 for a homogeneous group.
  int mixed_round_trip_depth_ = 0;
  // 910B reduction (vector): a Reduction couples its reduced axis (the whole
  // row/col must be present to reduce it), so the tile spans the FULL reduced
  // dim and only the non-reduced dim is tiled for spatial parallelism. The
  // reduced axis may still be split ACROSS cores (override), paying a thin
  // per-partial merge. reduced_axis_: 0=none, 1=width, 2=height. Competition
  // never emits Reduction, so these stay default on the single-context path.
  bool has_reduction_ = false;
  int reduced_axis_ = 0;
  int reduction_count_ = 0;
  bool reduction_spans_output_ = false;
  // Candidate-invariant vector layout facts. Cached once in create() because
  // vector_peak_ub() is called repeatedly while evaluating tile candidates.
  int64_t vector_min_dtype_bytes_ = 4;
  int64_t vector_emit_granule_ = 1;
  // Exact P4 algorithm implemented for this complete candidate op set. None means a streamed
  // multi-reduction is buildable only under the analytic model-ahead override.
  P4PatternKind p4_pattern_kind_ = P4PatternKind::None;
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

  // Prologue-PW geometric condition (Rules 2/3):
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

  // Precomputed per-boundary-tensor tiling info for compute_cost.
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

  std::vector<int64_t> ks_cand_;
  // SpatialSchedule (parts_m, parts_n, split_k) TRIPLES for the cube path. Each
  // lands parts_m*parts_n*split_k WORK UNITS targeting the core count (n_cores) or
  // 2*n_cores (a 2-wave grid): P*Q is the balanced 16-aligned spatial partition,
  // S=split_k splits the sink K across cores to fill what the spatial grid alone
  // can't (a power-of-two shape that can't form C spatial regions still fills the
  // cores via split-K). One enumeration of all three core-fill levers; the
  // single-core seq-k is NOT here (derive_exec sets it greedily). Empty for
  // vector / legacy subgraphs (the uniform divisor tiles cover those).
  struct SpatialTriple { int64_t parts_m, parts_n, split_k; };
  std::vector<SpatialTriple> grid_cand_;
  // Per-axis region-extent granularity for the grid (partition_axis). Cube: 16 on
  // both (the 16x16 MAC fractal). Vector: 1 along the free (row/height) axis and
  // 16 (the 32-byte DMA block) along the contiguous (width) axis -- no fractal
  // constraint, so a few-row reduction can still tile finely enough to fill C.
  int64_t grid_gran_h_ = 16;
  int64_t grid_gran_w_ = 16;

  // Flat lookup: tensor index → list of indices into boundary_tensor_info_.
  // Empty list means the tensor isn't a boundary tensor of this subgraph.
  // A tensor can have multiple entries when it's used under multiple distinct
  // role signatures (working set sums distinct slices).
  // Currently populated 1:1 per tensor; multi-entry materialization is
  // introduced in a follow-up step.
  std::vector<std::vector<int>> tensor_id_to_infos_;
};

// Compile-time check that the 910B cost model satisfies the architecture
// interface the solver pipeline depends on. A new backend that fails this
// assertion is missing part of the contract.
static_assert(CostModel<Ascend910BCost>,
              "Ascend910BCost must model the CostModel interface");

// ============================================================================
// Ascend910BMixed — the 910B cost model that ALSO permits fused cube+vector
// kernels (mixed groups), streaming the cube↔vector handoff through DDR so the
// roundtrip latency is hidden behind the cube∥vector overlap.
//
// It differs from the homogeneous Ascend910BCost in exactly ONE thing:
// admissibility. create() allows a cube+vector group instead of rejecting it.
// Everything else — tiling, feasibility, and the cost of any given group
// (including the mixed branch of compute_cost) — is inherited unchanged. The
// base model is therefore left exactly as it is; selecting this model (via the
// `Subgraph` alias in subgraph.h) is the compile-time opt-in to fusion.
// ============================================================================
class Ascend910BMixed : public Ascend910BCost {
 public:
  static std::optional<Ascend910BMixed> create(const Problem &prob, const DAG &dag,
                                               std::vector<size_t> op_indices) {
    auto base = Ascend910BCost::create(prob, dag, std::move(op_indices),
                                       /*allow_mixed=*/true);
    if (!base) return std::nullopt;
    return Ascend910BMixed(std::move(*base));
  }

  // Default-constructible like the base (ScheduleStep default-inits its subgraph
  // member); the user-declared private ctor below otherwise suppresses it.
  Ascend910BMixed() = default;

  // The third subgraph type: a fused cube+vector mixed kernel. Cube-only and
  // vector-only groups delegate to the (shared, unchanged) base cost.
  CostResult compute_cost(const TileConfig &cfg,
                          const FlatSet<size_t> &retained_from_prev = {},
                          const FlatSet<size_t> &retain_these = {}) const override;

 private:
  // Adds no state of its own — it is an Ascend910BCost built with mixed groups
  // admitted, so it slices up from a fully-built base.
  explicit Ascend910BMixed(Ascend910BCost base) : Ascend910BCost(std::move(base)) {}
};

static_assert(CostModel<Ascend910BMixed>,
              "Ascend910BMixed must model the CostModel interface");
