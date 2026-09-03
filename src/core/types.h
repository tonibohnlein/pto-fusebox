#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/flat_set.h"
#include "core/l0_matmul_plan.h"

// ============================================================================
// Tensor & Op
// ============================================================================

// Element data type — lets the cost model size tensors in BYTES (Ascend
// capacities and bandwidth are byte-based), not just element counts.
enum class DType { FP32, FP16, BF16, INT32, INT16, INT8, BOOL };

inline int dtype_bytes(DType dt) {
    switch (dt) {
        case DType::FP32:
        case DType::INT32: return 4;
        case DType::FP16:
        case DType::BF16:
        case DType::INT16: return 2;
        case DType::INT8:
        case DType::BOOL:  return 1;
    }
    return 4;
}

inline DType cube_accumulator_dtype(DType operand_dtype) {
  switch (operand_dtype) {
    case DType::FP32:
    case DType::FP16:
    case DType::BF16:
      return DType::FP32;
    case DType::INT32:
    case DType::INT16:
    case DType::INT8:
    case DType::BOOL:
      return DType::INT32;
  }
  return DType::FP32;
}

struct Tensor {
    int64_t width;
    int64_t height;
    DType dtype = DType::FP32;  // default keeps legacy element-count instances valid

    int64_t size() const { return width * height; }                            // elements
    int64_t size_bytes() const { return width * height * dtype_bytes(dtype); }  // bytes
};

// Coarse op category. Broadcast and the reduction AXIS are inferred from
// input/output shapes (a Reduction collapses a dim [H,W]->[H,1]; a broadcast
// input has extent 1 in a dim it is consumed along). The enum only carries what
// shapes cannot disambiguate (reduction vs matmul-contraction vs elementwise).
// Opaque = data-dependent (gather/scatter/sort) or relayout (transpose/reshape/
// concat) ops the unified-grid tiling can't fuse through — a barrier in the DAG.
enum class OpType { MatMul, Pointwise, Reduction, Opaque };

// Exact pto-isa VECTOR instruction family and tile geometry emitted for one
// source-DAG vector op.  `Generic` preserves the historical vec_slope /
// vec_fixed fallback for standalone benchmark instances that do not carry
// lowering semantics.  The PyPTO adapter fills both fields after tensor shapes
// are known, so candidate costing can replay the op at the strip/chunk geometry
// selected by VectorStreamPlan rather than scaling a full-tensor estimate.
enum class VectorPrimitiveFamily : uint8_t {
    Generic,
    Add,
    Mul,
    Div,
    Cast,
    Recip,
    Exp,
    Log,
    Abs,
    Sqrt,
    Rsqrt,
    ScalarAdd,
    ScalarMul,
    ScalarMax,
    ScalarMin,
    RowSum,
    RowExtrema,
    ColSum,
    ColExtrema,
    // Backward-compatible descriptor for JSON/research instances that identify
    // a reduction without its exact PTO instruction. Production PyPTO graphs
    // use one of the four axis/op-specific families above.
    Reduction
};

enum class VectorOpGeometry : uint8_t {
    Generic,
    Flat,
    RowExpand,
    ColExpand
};

// Buildability contract for a source-DAG vector operation.  PrimitiveFamily
// describes cost; capability describes which tiled algorithm the compiler can
// actually replay.  Production adapters must never rely on the historical
// Generic fallback for admission: an operation is either an elementwise strip,
// one of the two implemented streaming reductions, or explicitly unsupported.
enum class VectorOpCapability : uint8_t {
    Generic,
    Elementwise,
    ReductionSum,
    ReductionMax,
    Unsupported
};

enum class VectorCoordinateTransform : uint8_t {
    None,
    SingletonColumnToRow
};

// Exact source-op identity needed by a mixed algorithm recognizer. This is
// deliberately separate from VectorPrimitiveFamily: the latter is a grounded
// cost class, while Recip and Cast are also load-bearing semantic nodes for
// mixed-algorithm legality and emission.
enum class MixedVectorSemantic : uint8_t { None, Neg, Exp, ScalarAdd, Recip, Mul, Cast };

// Exact streamed multi-reduction algorithms implemented by the AutoFuse emit.
// A P4 pattern identifies the complete op set whose semantics were proven to
// match the emitted online algorithm, not merely a reduction family.
enum class P4PatternKind { None, SoftmaxFlash, LayerNormWelford };

enum class P4SubstitutionValue {
    RunningMax,
    RunningSum,
    Mean,
    Variance
};

struct P4ApplyBinding {
    size_t op = std::numeric_limits<size_t>::max();
    P4SubstitutionValue value = P4SubstitutionValue::RunningMax;
};

struct P4Pattern {
    P4PatternKind kind = P4PatternKind::None;
    FlatSet<size_t> ops;
    // Ops whose values are supplied by the online algorithm to the apply pass.
    // Softmax substitutes max/sum; Welford substitutes mean/variance.  Keeping
    // these semantic cut points in the descriptor lets the model and emitter
    // derive the same apply cone without independently recognizing P4 again.
    FlatSet<size_t> apply_substitutions;
    // Semantic bindings are separate from the set above on purpose. Legacy
    // analytic problem instances may carry only the cut-point set; only a
    // frontend-owned, named binding is strong enough to become an executable
    // source recipe.
    std::vector<P4ApplyBinding> apply_bindings;
};

struct Op {
  OpType type;
  std::vector<size_t> inputs;   // tensor indices
  std::vector<size_t> outputs;  // tensor indices (always exactly one)

  // Per-op VECTOR compute slope (cycles per SIMD repeat) for a Pointwise op, when it differs
  // from the group default `vec_slope_pw` (~2). pto-isa vec_tile_study: most pointwise ops are
  // slope 2, but `vdiv` is 4 and the scalar/activation ops (`vrsqrt`/`vrelu`/`vmuls`) are 1.
  // 0.0 => use the problem default. Set by the PyPTO adapter from the op name; JSON instances
  // that omit it fall back to the default (so the benchmark suite is unaffected).
  double vec_slope = 0.0;

  // Per-op VECTOR fixed cost (head+tail cycles), charged ONCE per fused chain (the stream-start
  // op). pto-isa device-calibrated per-op: vadd/vsub/vmax/vmin 24, vmul 25, vexp 31, vdiv 30,
  // vrsqrt/vrelu/vmuls ~24. 0.0 => use the problem default `vec_op_head+vec_op_tail` (~32).
  double vec_fixed = 0.0;

  // Grounded source-op semantics.  These two bytes are candidate-invariant
  // metadata; they are intentionally stored on Op rather than in CostResult
  // or VectorStreamPlan, so local-search cache entries stay compact.
  VectorPrimitiveFamily vector_primitive = VectorPrimitiveFamily::Generic;
  VectorOpGeometry vector_geometry = VectorOpGeometry::Generic;
  VectorOpCapability vector_capability = VectorOpCapability::Generic;

  // True only when the production mixed emitter can replay this source op
  // without losing semantic attributes.  The first C->V increment accepts a
  // default-orientation tensor.matmul; transposed/NZ variants remain a clean
  // partition boundary until their semantics are carried by the plan.
  bool mixed_emit_compatible = true;
  MixedVectorSemantic mixed_vector_semantic = MixedVectorSemantic::None;

  // Convenience: each op produces exactly one output tensor.
  size_t output() const { return outputs[0]; }
};

// ============================================================================
// Problem specification
// ============================================================================

struct Problem {
    std::vector<Tensor> tensors;
    std::vector<Op> ops;
    int64_t fast_memory_capacity;
    // Candidate-invariant zero-copy coordinate normalization selected by the
    // frontend. Every vector subgraph and its emitted plan must preserve it.
    VectorCoordinateTransform vector_coordinate_transform = VectorCoordinateTransform::None;

    // --- Ascend 910B machine model -------------------------------------------
    // Compute parallelizes across these cores; DDR bandwidth is shared (one HBM).
    // The 910B path (adapter / tests) sets the real counts (cube 24, vector 48).
    int num_cube_cores = 24;     // AIC cores — matmul
    int num_vector_cores = 48;   // AIV cores — pointwise / reduction
    // Per-core local-memory budgets in BYTES. 0 = fall back to the single
    // fast_memory_capacity pool (so legacy element-count instances are
    // unchanged). The 910B feasibility check forks on cube-vs-vector:
    //   cube subgraph : operand strips -> L1, output accumulator -> L0c
    //   vector subgraph: tile + ephemerals -> UB
    int64_t vec_capacity = 0;    // per-vector-core UB (910B: 192*1024)
    int64_t cube_capacity = 0;   // per-cube-core L0c accumulator (910B: 128*1024)
    int64_t l1_capacity = 0;     // per-cube-core L1/Mat operand pool (910B: 512*1024)
    // Grounded cube-compute CALIBRATION MULTIPLIER on the pto-isa fractal-cycle
    // model (CubeMacCycles = repeats * cyc * cube_compute_cost). The pto-isa model
    // is already in real cycles, so the realistic value is 1; tests dial it up to
    // force a compute-bound regime. 0 => treated as 1.
    int64_t cube_compute_cost = 1;
    // Per-kernel pipeline fill/drain latency. A subgraph's tiling produces
    // num_tiles "kernels"; each core runs ceil(num_tiles/cores) of them in
    // sequence, paying one fill per pass. Charged as rounds*kernel_fill_cost,
    // the DUAL of the eff core-fill incentive: eff penalizes under-tiling
    // (fewer tiles than cores), this penalizes over-tiling (more) — so the
    // optimum sits at ~one kernel per core. 0 => off (legacy/competition).
    int64_t kernel_fill_cost = 0;
    // Per-TASK launch overhead (cycles) for homogeneous vector schedules,
    // charged as num_tiles*split. The kernel_fill term above is per-WAVE, so
    // this term distinguishes otherwise tied grids with different task counts.
    // It remains separate from the heavier 1-AIC + 2-AIV mixed-group launch.
    int64_t per_task_overhead_cycles = 0;
    // Per active 1-AIC + 2-AIV mixed group (cycles). A two-device 910B2 block
    // sweep measured 0.2579 us/group (95% CI [0.2545, 0.2619]); at 1.85 GHz
    // that is 477 cycles, rounded to 480 in the production target profile.
    // Keeping this separate avoids perturbing the already-grounded vector-only
    // task term. 0 => off for legacy/research problem descriptors.
    int64_t mixed_group_overhead_cycles = 0;
    // Additional ordered-task synchronization cost for either cube split-K
    // merge: AIC first-partial -> AIC atomic-rest, or AIV zero-seed -> AIC
    // atomic-all. The task walls themselves are always serialized; this is only
    // their dependency-boundary overhead. Zero keeps the term explicit without
    // inventing an ungrounded device constant.
    int64_t cube_split_sync_cycles = 0;
    // Double-buffering is ALWAYS assumed on 910B but does NOT reserve half the
    // pool: the two ping-pong buffers together ARE the L1/UB, so feasibility uses
    // the FULL l1_capacity / vec_capacity. The overlap is realized in the emit by
    // streaming each seq-K strip (or tile) as >=2 sub-strips -- it halves the
    // per-load k, not the resident operand. (Hence no double_buffer flag.)
    // Split-K first publishes share zero with an ordinary full-tile store, then
    // the remaining S-1 partials accumulate via SetAtomicAdd after an ordered
    // phase boundary. There is no read-back + serial re-sum. This is an
    // unconditional 910B capability; the old `ddr_atomic_add` toggle (and the
    // no-atomic-add serial-reduction cost) was removed with the legacy strip.

    // --- Grounded pto-isa machine model (Ascend 910B / A2A3) -----------------
    // The cube/vector costs use pto-isa's measured coefficients
    // (pto-isa include/pto/costmodel/arch_config.hpp + cce_costmodel_cube.hpp):
    //
    //   * WORK IS IN CYCLES. A matmul costs `repeats * cyc_per_repeat` cycles,
    //     repeats = ceil(M/16)*ceil(N/16)*ceil(K/kF), kF = 32/dtype_bytes
    //     (fp32:8, fp16/bf16:16), cyc_per_repeat = 2 (fp32) else 1.
    //   * A byte transfer costs (bytes/2^30)/bw_GiBps * cube_freq_hz cycles
    //     (EstimateBandwidthCycles): bandwidths below are GiB/s, PER DIRECTION.
    //   * The cube core's WORK is hierarchical and double-buffered: producing a
    //     tile needs both the cube MACs AND the L1->L0A/L0B operand extract,
    //     which OVERLAP, so per-core work = max(cube_cycles, extract_cycles).
    //     The L1->L0 reload reuses the L0 GEMM base tile (l0_tile_m/n) the same
    //     way cube_operand_reload reuses the L1 tile (w/h) one level up.
    double cube_freq_hz = 1.85e9;  // core clock (A2A3 1.85e9)
    double bw_gm_l1   = 0.0;     // GM->L1 operand reload      (GiB/s; pto-isa 135.0)
    double bw_l0c_gm  = 0.0;     // L0C->GM output store       (GiB/s; pto-isa 70.0)
    double bw_l1_l0a  = 0.0;     // L1->L0A lhs extract        (GiB/s; pto-isa 441.0)
    double bw_l1_l0b  = 0.0;     // L1->L0B rhs extract        (GiB/s; pto-isa 220.5)
    double bw_gm_ub   = 0.0;     // GM->UB vector load         (GiB/s; pto-isa 100.9)
    double bw_ub_gm   = 0.0;     // UB->GM vector store        (GiB/s; pto-isa 188.46)
    // Aggregate HBM bandwidth (GiB/s) shared by ALL cores: the cap on the SUM of
    // the per-core GM pipes. Each core has its own MTE2 (GM->L1/UB) + FixPipe
    // (L0C/UB->GM), so DDR traffic divides across active cores up to this ceiling:
    //   effective aggregate bw = min(active * per_core_peak, hbm_aggregate_gibps)
    // (exactly pto-isa BwEff). Source: pto-isa docs/coding/performance-best-
    // practices.md — A3 (24-core) hardware bandwidth ~900 GB/s (theoretical;
    // achievable ~70-80%); treated as GiB/s to match the per-core peaks above.
    // 0 => uncapped (pure per-core divide). The live 910B config sets this to the
    // realistic A3 aggregate ~900, so par() binds at ~900/135 = 6.7 cores: beyond
    // the knee, reload-bound matmuls saturate HBM instead of scaling linearly to 24.
    // Perf-sim VALIDATED in the saturation regime (pto-isa gml1_multicore: per-core
    // bw = min(135, 900/B) to <=0.4%). The exact aggregate is DEVICE-EVAL-PENDING.
    double hbm_aggregate_gibps = 0.0;
    int64_t l0_tile_m = 0;       // L0 GEMM base M (pto-isa oracle 128) — L1->L0 reuse
    int64_t l0_tile_n = 0;       // L0 GEMM base N (pto-isa oracle 256) — L1->L0 reuse
    // Shared lower-level planner configuration. PTO Fusebox fills the request-
    // specific M/N/K/dtype fields before calling choose_l0_matmul_plan();
    // AutoTileMatmulL0 calls the same chooser and realizes the selected plan.
    L0MatmulConfig l0_matmul_config;

    // Grounded vector (AIV) compute, in CYCLES (pto-isa A2A3
    // cce_costmodel_vector_compute.hpp). Active in grounded mode (cube_freq_hz>0).
    // Each vector op costs head + slope*repeat + tail, repeat = ceil(elems /
    // (vec_reg_bytes / dtype_bytes)) -- the 256-byte vector register holds
    // 64 fp32 / 128 fp16 elements (fp16 = 2x throughput). slope picks elementwise
    // (vadd/vmul ~1-2) vs reduction (vreducev2 ~14), so a reduction is ~14x an add
    // and the per-op head/tail charges the pipeline fill of a multi-op chain
    // (e.g. softmax = exp + reduce + div).
    int64_t vec_reg_bytes = 256;      // vector register size in bytes (pto-isa 256)
    // DMA-block granule in BYTES (Ascend 910B: 32). A vector tile's contiguous-axis byte extent
    // must be a multiple of this, so the emit allocates AlignUp(extent, dma_block/dtype_bytes)
    // tiles. vector_peak_ub() pads its band footprint to match — a thin free axis (e.g. an M-tile
    // of 3 -> 8 for fp32) is otherwise under-counted ~2.7x, making a group look UB-feasible that
    // the emit overflows. Set by the AutoFuse adapter from BackendHandler::GetVectorDmaAlignment.
    int64_t vec_dma_align_bytes = 32;
    double vec_op_head = 0.0;         // per-op pipeline startup cycles (~14)
    double vec_op_tail = 0.0;         // per-op drain cycles (~18)
    double vec_slope_pw = 0.0;        // cycles/repeat, elementwise (~2)
    double vec_slope_reduce = 0.0;    // DEPRECATED by Fix 1 (VecOpCompute reduction tree):
                                      // a reduction is no longer one slope*repeat op. Field
                                      // kept for ABI/serialization; unused by the cost path.
    // Split-K is MODEL-AHEAD of the AutoFuse emit (both the base lone-matmul and the mixed
    // cube-sink path — see grounded_cost_model.md §12). This flag gates it: true (default) =
    // analytic/research mode, the cost model may credit parallel_split > 1; false = BUILDABLE
    // mode, split-K is forced to S=1 so best_cost never selects a config the emit can't yet
    // lower. Flip to false in any harness that treats the winning config as emittable;
    // CostResult::uses_model_ahead_split_k flags when a chosen config used a split.
    bool allow_model_ahead_split_k = true;

    // Analytic override for streamed MULTI-reduction groups. AutoFuse sets this FALSE and instead
    // populates p4_patterns with exact candidate op sets whose semantics match a real emitted
    // algorithm. A different candidate is infeasible and is cut into P1/P2-buildable groups.
    bool allow_model_ahead_multi_reduction_stream = true;

    // Runtime architecture policy. False keeps the production solver strictly
    // unit-homogeneous; true lets the same monomorphized Ascend910BCost admit
    // and price mixed cube+vector groups. The dedicated Ascend910BMixed test/
    // research model still opts in directly.
    bool fuse_cube_vector = false;

    // Compiler/buildability policy for mixed candidates.  `fuse_cube_vector`
    // controls admission; this stricter bit restricts admitted groups to the
    // exact topology the production emitter can replay.  Analytic users may
    // keep it false while studying broader mixed schedules.
    bool require_buildable_mixed = false;

    // External PyPTO-source policy. This leaves the analytic mixed model broad
    // by default, but when requested it filters candidate tiles through the
    // public-source topology, stage-kind, and physical FIFO capacity contract.
    // It is separate from require_buildable_mixed, which targets the older
    // in-compiler AutoFuse lowering surface.
    bool require_source_codegen = false;

    // Analytic/research permission for a mixed topology whose FIFO contains
    // more than one round trip (for example full C->V->C->V attention). The
    // existing PyPTO skew pass demotes it to sequential, so the model prices a
    // serial stage sum. Compiler/buildable mode sets this false until a whole-
    // FIFO wavefront emitter exists.
    bool allow_model_ahead_mixed_multi_roundtrip = true;

    // Buildability gate for homogeneous cube DAGs. Multi-matmul groups enumerate
    // only exact uniform M/N partitions. Lone matmuls retain the full grid, but
    // a non-uniform candidate is specified as split=1 ClampedOverlap; ragged
    // split-K is rejected because its atomic edge regions would overlap.
    bool require_uniform_cube_dag_grid = false;
    // Price the explicit output-tile/K-window hierarchy reconstructed by
    // CubeSchedulePlan. PyPTO enables this for buildable solver decisions.
    bool use_hierarchical_cube_cost = false;

    // Exact buildable P4 candidates found once by the IR adapter. The cost model compares the
    // candidate subgraph's complete op set against these entries; the emitter consumes the same
    // analysis result.
    std::vector<P4Pattern> p4_patterns;

    // Tensor values that must be materialized as function results.  A required
    // output can also feed another op in the same candidate subgraph; in that
    // case it is simultaneously an on-chip intermediate and a DDR live-out.
    // Keeping this fact in Problem prevents both the cost model and emitter
    // from swallowing a returned-and-consumed SSA value.
    FlatSet<size_t> required_outputs;

    size_t num_ops() const { return ops.size(); }
    size_t num_tensors() const { return tensors.size(); }

    // Precomputed by read_problem(). A tensor is retainable if:
    //   1. Its full size (width × height) fits in fast_memory_capacity.
    //   2. It has at least one consuming op (graph outputs are excluded).
    // Graph inputs with a single consumer are included — recomputation
    // schedules may load such a tensor in two separate subgraphs, and
    // retaining it after the first eliminates the second load.
    // This set is the permissive upper bound; the ordering layer decides
    // whether retention is beneficial for any given schedule.
    FlatSet<size_t> retainable_tensors;
};

// ============================================================================
// Tile configuration
// ============================================================================

struct TileConfig {
    int64_t w = 0, h = 0, k = 0;
    // Non-uniform spatial grid (SpatialSchedule). 0 => legacy uniform-divisor
    // tile (w,h are an exact-divisor tile). >0 => the output is partitioned into
    // parts_m x parts_n regions whose extents are an even split of the 16-fractal
    // counts (regions differ by <=1 fractal per axis); w,h then carry the
    // PHYSICAL (max) region extent so the L1-fit / reload machinery is unchanged.
    // See partition_axis().
    int64_t parts_m = 0, parts_n = 0;
    // Parallel split-K: the sink contraction is split into `split_k` equal
    // 16-aligned partials across cores (atomic-add merge), so the work units are
    // parts_m * parts_n * split_k. 0 => UNSET — compute_cost sweeps S internally
    // (the uniform/legacy path). >=1 => a FIXED factor from the SpatialSchedule
    // (P,Q,S) triple enumeration, evaluated as-is (no internal sweep).
    int64_t split_k = 0;
};

// Even distribution of an axis of `dim` elements into `parts` regions, in units
// of `granule`-element blocks. F = ceil(dim/granule) blocks split as evenly as
// possible: `num_big` regions get (base+1) blocks, the rest get `base`. Since
// regions differ by at most one block, an axis has at most two distinct extents
// -- so a P x Q grid has at most four distinct region shapes.
struct AxisPartition {
    int64_t big = 16, small = 16;  // element extents of the two region sizes
    int64_t num_big = 0;           // first num_big of `parts` regions use `big`
    int64_t parts = 1;
    int64_t offset(int64_t i) const {
        return i < num_big ? i * big : num_big * big + (i - num_big) * small;
    }
    int64_t extent(int64_t i, int64_t dim) const {  // valid extent (final region clamped)
        const int64_t e = (i < num_big) ? big : small;
        const int64_t off = offset(i);
        return (off + e <= dim) ? e : (dim - off);
    }
};

// `granule` is the alignment of each region extent: 16 for the cube (the 16x16
// MAC fractal); finer for the vector (1 along free rows, the 32-byte DMA block
// along the contiguous axis), which has no fractal constraint.
inline AxisPartition partition_axis(int64_t dim, int64_t parts, int64_t granule = 16) {
    const int64_t g = (granule < 1) ? 1 : granule;
    const int64_t F = (dim + g - 1) / g;                     // granules on the axis
    parts = (parts < 1) ? 1 : ((parts > F) ? F : parts);     // never more parts than granules
    const int64_t base = F / parts, rem = F % parts;
    AxisPartition p;
    p.parts = parts;
    p.num_big = rem;
    p.big = (base + (rem > 0 ? 1 : 0)) * g;
    p.small = base * g;
    return p;
}

// ============================================================================
// Derived vector streaming plan
// ============================================================================

// The algorithm realized when a vector tile does not materialize in UB. This is
// derived from (subgraph, TileConfig, retention context) by Ascend910BCost. Candidate
// pricing uses it as a stack-local value; downstream consumers re-derive it only
// for a final or explicitly forced configuration. Materialized means no UB
// sub-stream is required.
enum class VectorStreamKind {
    Materialized,
    Pointwise,
    ReductionFolded,          // P1: bare reduction or thin folded finalize
    ReductionSpanning,        // P2: stats pass + spanning apply pass
    SoftmaxFlash,             // P4: online (m,l) + apply
    LayerNormWelford,         // P4: online (mean,M2,count) + apply
    ModelAheadMultiReduction, // legacy analytic-only multi-reduction estimate
    MultiPass                 // generic ordered reduction-barrier replay
};

// Spatial ownership and executable replay are related but distinct.  The
// balanced partition records which logical elements belong to each work unit;
// the clamped-overlap policy replays the partition's maximum physical tile at
// every work unit and clamps its origin at a ragged edge.  That keeps tile
// shapes compile-time static while harmlessly recomputing the overlap.
enum class VectorSpatialPolicy {
  ExactBalanced,
  ClampedOverlap,
};

// Cross-core reduced-axis algorithm attached to a materialized vector plan.
// The 910B backend currently lowers atomic ADD only, and the generic emitter
// can slice the reduced M axis without changing semantics only for a terminal
// col_sum whose upstream cone is pointwise. Unsupported reductions remain
// single-core along their reduced axis instead of borrowing fictional split
// parallelism from the cost model.
enum class VectorReductionSplitKind {
    None,
    ColSumAtomicAdd
};

struct VectorLoopPlan {
    int64_t first_chunk = 0;
    int64_t trip_count = 0;
    int pipeline_stages = 1;  // 1 = sequential/absent, 2 = stage-2 ping-pong
};

struct VectorSerialPhasePlan {
  bool present = false;
  int64_t chunk_index = 0;
  int64_t extent = 0;
};

// Separate zero-initialization launch required before an atomic cross-core
// reduction. Each work unit emits one row-major TEXPANDS tile followed by one
// non-atomic UB->GM store. It is a serial phase: neither operation is hidden by
// the main reduction body's pipeline.
struct VectorReductionSeedPlan {
    bool present = false;
    int64_t work_units = 0;
    int64_t valid_rows = 0;
    int64_t valid_cols = 0;
};

// Grounded VECTOR primitives used by an emitter-generated online algorithm.
// The source DAG remains authoritative for ordinary body/apply work, but P4
// statistics replace the source reductions with softmax (m,l) or Welford/Chan
// updates. A compact fixed-size tally keeps that replacement explicit without
// adding heap allocation to candidate enumeration.
enum class VectorPrimitiveKind : size_t {
    Add,
    Mul,
    Div,
    Exp,
    RowExpandSub,
    ScalarAdd,
    ScalarMul,
    RowSum,
    RowMax,
    Count
};

struct VectorPrimitiveWork {
    // Wide instances operate on one [free_tile, reduced_chunk] tile; thin
    // instances operate on one persistent [free_tile, 1] statistic/carry.
    uint8_t wide = 0;
    uint8_t thin = 0;
    // Number of barrier-separated pointwise streams whose first instruction is
    // this primitive. Only those instances pay the primitive's grounded fixed
    // head+tail cost; back-to-back operations share it.
    uint8_t stream_starts = 0;

    bool operator==(const VectorPrimitiveWork& other) const {
        return wide == other.wide && thin == other.thin && stream_starts == other.stream_starts;
    }
};

struct VectorPhaseWorkPlan {
    bool generated = false;
    std::array<VectorPrimitiveWork, static_cast<size_t>(VectorPrimitiveKind::Count)> primitives{};

    void add(VectorPrimitiveKind kind, int64_t wide, int64_t thin, int64_t stream_starts = 0) {
        auto& work = primitives[static_cast<size_t>(kind)];
        work.wide = static_cast<uint8_t>(work.wide + wide);
        work.thin = static_cast<uint8_t>(work.thin + thin);
        work.stream_starts = static_cast<uint8_t>(work.stream_starts + stream_starts);
    }

    const VectorPrimitiveWork& get(VectorPrimitiveKind kind) const {
        return primitives[static_cast<size_t>(kind)];
    }

    bool operator==(const VectorPhaseWorkPlan& other) const {
        return generated == other.generated && primitives == other.primitives;
    }
};

struct VectorP4WorkPlan {
    bool generated = false;
    VectorPhaseWorkPlan stats_init;
    // One online update. The rolled stats loop repeats it trip_count times and
    // a ragged stats tail executes the same update once at its shorter extent.
    VectorPhaseWorkPlan stats_update;
    VectorPhaseWorkPlan finalize;

    bool operator==(const VectorP4WorkPlan& other) const {
        return generated == other.generated && stats_init == other.stats_init &&
               stats_update == other.stats_update && finalize == other.finalize;
    }
};

struct VectorP4RecipePlan {
  // Versioned semantic recipe selected by the C++ planner. Python renders this
  // recipe mechanically; it never rediscovers a softmax/normalization pattern
  // from the source DAG.
  P4PatternKind kind = P4PatternKind::None;
  size_t input_tensor = std::numeric_limits<size_t>::max();
  std::vector<P4ApplyBinding> apply_bindings;
};

// This is the single algorithm-work description shared by candidate costing
// and final-plan emission checks. It describes exactly the primitive sequence
// emitted in auto_fuse_pass.cpp; dependency semantics remain in the exact
// P4Pattern descriptor and emitter builders.
inline VectorP4WorkPlan make_vector_p4_work_plan(P4PatternKind kind) {
  VectorP4WorkPlan plan;
  if (kind == P4PatternKind::None) return plan;

  plan.generated = true;
  plan.stats_init.generated = true;
  plan.stats_update.generated = true;

  if (kind == P4PatternKind::SoftmaxFlash) {
    // Init: row_max; sub -> exp; row_sum.
    plan.stats_init.add(VectorPrimitiveKind::RowMax, 1, 0);
    plan.stats_init.add(VectorPrimitiveKind::RowSum, 1, 0);
    plan.stats_init.add(VectorPrimitiveKind::RowExpandSub, 1, 0, 1);
    plan.stats_init.add(VectorPrimitiveKind::Exp, 1, 0);

    // Update: row_max; max(carry,cmax) -> sub -> exp; row_sum;
    // sub(carry,m) -> exp -> mul -> add.
    plan.stats_update.add(VectorPrimitiveKind::RowMax, 1, 0);
    plan.stats_update.add(VectorPrimitiveKind::RowSum, 1, 0);
    plan.stats_update.add(VectorPrimitiveKind::Add, 0, 3, 2);
    plan.stats_update.add(VectorPrimitiveKind::Exp, 1, 1);
    plan.stats_update.add(VectorPrimitiveKind::Mul, 0, 1);
    plan.stats_update.add(VectorPrimitiveKind::RowExpandSub, 1, 0, 1);
    return plan;
  }

  // Welford init: row_sum; muls(mean) -> sub -> mul; row_sum;
  // muls(zero-count) -> adds(count).
  plan.stats_init.add(VectorPrimitiveKind::RowSum, 2, 0);
  plan.stats_init.add(VectorPrimitiveKind::Mul, 1, 0);
  plan.stats_init.add(VectorPrimitiveKind::RowExpandSub, 1, 0, 1);
  plan.stats_init.add(VectorPrimitiveKind::ScalarAdd, 0, 1);
  plan.stats_init.add(VectorPrimitiveKind::ScalarMul, 0, 2, 2);

  // Welford update repeats the chunk mean/M2 stream, then performs Chan's
  // thin (mean,M2,count) merge after the second reduction barrier.
  plan.stats_update.add(VectorPrimitiveKind::RowSum, 2, 0);
  plan.stats_update.add(VectorPrimitiveKind::Add, 0, 4, 1);
  plan.stats_update.add(VectorPrimitiveKind::Mul, 1, 2);
  plan.stats_update.add(VectorPrimitiveKind::Div, 0, 2);
  plan.stats_update.add(VectorPrimitiveKind::RowExpandSub, 1, 0, 1);
  plan.stats_update.add(VectorPrimitiveKind::ScalarAdd, 0, 1);
  plan.stats_update.add(VectorPrimitiveKind::ScalarMul, 0, 3, 1);

  // Population variance = final M2 / N.
  plan.finalize.generated = true;
  plan.finalize.add(VectorPrimitiveKind::ScalarMul, 0, 1, 1);
  return plan;
}

static_assert(sizeof(VectorP4WorkPlan) <= 96,
              "P4 work descriptors are rebuilt in candidate enumeration and must stay compact");

// One vector kernel can replay the source DAG in several barrier-separated
// phases. A boundary tensor has one compatible UB representation within a
// phase, independent of which vector operand position consumes it. Crossing a
// phase boundary is different: P2/P4 deliberately reload spanning inputs, so
// each phase owns a separate lifetime instance.
enum class VectorReplayPhase : uint8_t {
  Body = 0,
  Stats = 1,
  Apply = 2,
  Finalize = 3,
};

inline constexpr size_t vector_replay_phase_index(VectorReplayPhase phase) {
  return static_cast<size_t>(phase);
}

struct VectorInputUsePlan {
  size_t op = 0;
  size_t arg = 0;
};

struct VectorInputLifetimePlan {
  size_t tensor = 0;
  VectorReplayPhase phase = VectorReplayPhase::Body;
  size_t first_use_step = 0;
  size_t last_use_step = 0;
  size_t use_count = 0;
  std::vector<VectorInputUsePlan> uses;

  [[nodiscard]] bool spans_steps() const { return first_use_step < last_use_step; }
};

// Candidate-invariant topology shared by every tiling of one subgraph. Keeping
// it behind a shared pointer avoids rebuilding/copying use lists in the local-
// search hot path; candidate-dependent tile bytes remain in VectorStreamPlan.
struct VectorInputLifetimeTopology {
  // Exact source-DAG replay order for each barrier-separated phase. Keeping
  // this beside the phase-local lifetimes prevents a downstream emitter from
  // reconstructing cones or filtering the global execution order.
  std::array<std::vector<size_t>, 4> ops;
  std::array<std::vector<VectorInputLifetimePlan>, 4> phases;
};

// A generic replay pass is derived only from graph dependencies. Reduction
// passes accumulate one or more same-depth reductions over the streamed axis;
// the final apply pass replays the remaining sink cone. Values crossing a
// pass boundary are named explicitly and must be thin reduction results.
enum class VectorReplayPassKind : uint8_t { Reduction, Apply };

struct VectorReplayPassTopology {
  VectorReplayPassKind kind = VectorReplayPassKind::Apply;
  std::vector<size_t> ops;
  std::vector<size_t> state_inputs;
  std::vector<size_t> state_outputs;
  std::vector<size_t> output_tensors;
  std::vector<VectorInputLifetimePlan> input_lifetimes;
};

struct VectorReplayTopology {
  std::vector<VectorReplayPassTopology> passes;
};

struct VectorReplayPassExecutionPlan {
  size_t index = 0;
  VectorLoopPlan loop;
  VectorSerialPhasePlan init;
  VectorSerialPhasePlan tail;
};

struct VectorTensorFramePlan {
  // Concrete candidate-local tile frame for one tensor in one replay phase.
  // Logical extents drive valid_shape; physical extents drive allocation and
  // DMA legality. Source backends consume this verbatim rather than
  // re-implementing singleton/broadcast/reduction padding policy.
  size_t tensor = 0;
  VectorReplayPhase phase = VectorReplayPhase::Body;
  int64_t logical_rows = 0;
  int64_t logical_cols = 0;
  int64_t physical_rows = 0;
  int64_t physical_cols = 0;
};

struct VectorWorkspaceFramePlan {
  // Tile-level reduction APIs require a distinct source-shaped scratch tile.
  // It is allocation-owning and therefore part of both UB accounting and the
  // executable source contract.
  size_t op = 0;
  size_t source_tensor = 0;
  VectorReplayPhase phase = VectorReplayPhase::Body;
  int64_t logical_rows = 0;
  int64_t logical_cols = 0;
  int64_t physical_rows = 0;
  int64_t physical_cols = 0;
};

struct VectorStreamPlan {
  bool feasible = false;
  VectorStreamKind kind = VectorStreamKind::Materialized;
  VectorCoordinateTransform coordinate_transform = VectorCoordinateTransform::None;
  VectorSpatialPolicy spatial_policy = VectorSpatialPolicy::ClampedOverlap;
  // One logical region is one runtime work unit / SPMD block. The physical
  // UB/DMA allocation used while replaying a region may be padded or split
  // into strips, but it must never change this solver-owned launch grid.
  AxisPartition m_partition;
  AxisPartition n_partition;
  int64_t work_units = 0;
  // Peak UB footprint before streaming and at the selected chunk. Keeping
  // both in the derived plan lets compute_cost reuse the feasibility work
  // instead of rescanning the pebbling order for every cost term.
  int64_t full_peak_ub_bytes = 0;
  // Full-region peak with row-reduction scratch transients excluded. Source
  // backends that allocate every reduction workspace eagerly add their exact
  // phase-local workspace sum to this value; unlike subtracting an arbitrary
  // workspace from full_peak_ub_bytes, that remains sound when workspaces have
  // heterogeneous shapes or dtypes.
  int64_t workspace_free_peak_ub_bytes = 0;
  int64_t chunk_peak_ub_bytes = 0;
  int64_t stream_band_count = 0;
  // Mechanical physical-frame contract shared by pricing and emission. These
  // are not search choices: they record the exact legalization policy used by
  // VectorAllocatedFrame so a source backend does not rediscover padding,
  // broadcast, or thin-reduction rules.
  int64_t physical_element_granule = 1;
  // Same-shaped elementwise values share one physical box. Native cast hops
  // therefore use the least common DMA element granule of their connected
  // class, while unrelated tensors retain their own dtype-sized granule. The
  // immutable vector is shared by all candidates of one subgraph.
  std::shared_ptr<const std::vector<int64_t>> tensor_element_granules;
  int64_t iteration_rows = 1;
  int64_t iteration_cols = 1;
  int reduced_axis = 0;  // 0 = none, 1 = width, 2 = height
  bool align_rows = false;
  int axis = 0;  // 0 = materialized, 1 = width, 2 = height
  // Maximum logical free-axis extent owned by one work unit. Its physical UB
  // allocation is `free_tile_alloc`, rounded to the DMA granule. Keeping the
  // two separate prevents alignment from silently coarsening the SPMD grid.
  int64_t free_tile = 0;
  int64_t free_tile_alloc = 0;
  int64_t extent = 0;
  int64_t chunk = 0;
  int64_t full_chunks = 0;
  int64_t tail = 0;
  int stream_passes = 0;
  // Exact boundary-input identities and phase-local first/last uses replayed
  // by the emitter. Traffic is one GM->UB slice per tensor, phase and emitted
  // strip/chunk; the tile stays live through the descriptor's last use.
  std::shared_ptr<const VectorInputLifetimeTopology> input_lifetimes;
  // Ordered generic replay contract. Empty for the four compatibility phases
  // above; populated for MultiPass. Topology is candidate-invariant while the
  // loop descriptors carry the selected chunk geometry.
  std::shared_ptr<const VectorReplayTopology> replay_topology;
  std::vector<VectorReplayPassExecutionPlan> replay_passes;
  // Solver-owned materialized/pointwise strip geometry.  The emitter must not
  // independently choose a row/width strip count: these fields are the exact
  // uniform (clamp-overlap on a ragged edge) loop it builds.
  int64_t tile_h = 0;
  int64_t tile_w = 0;
  int64_t strip_h = 0;
  int64_t strip_w = 0;
  int64_t row_strips = 1;
  int64_t width_strips = 1;
  VectorLoopPlan body;

  // Optional cross-core split of a materialized reduction. `work_units`
  // remains the spatial grid size; the main launch has
  // work_units*reduction_split_factor tasks and a separate spatial seed grid.
  // The factor/partial extent are populated only when this exact candidate is
  // realizable by the emitter.
  VectorReductionSplitKind reduction_split_kind = VectorReductionSplitKind::None;
  int64_t reduction_split_factor = 1;
  int64_t reduction_partial_extent = 0;
  VectorReductionSeedPlan reduction_seed;

  // A streamed reduction is a sequence of barrier-separated phases.  Peeled
  // init/tail/finalize work is always serial; only stats/apply rolled loops
  // may receive stage-2 overlap.
  VectorSerialPhasePlan stats_init;
  VectorLoopPlan stats;  // P1/P2/P4 online statistics
  VectorLoopPlan apply;  // P2/P4 spanning output
  VectorSerialPhasePlan stats_tail;
  VectorSerialPhasePlan apply_tail;
  VectorSerialPhasePlan finalize;

  // P4 stats are emitter-generated work, not a scaled replay of the source
  // DAG. Apply remains a source-DAG cone with the online stats substituted.
  VectorP4WorkPlan p4_work;
  std::shared_ptr<const VectorP4RecipePlan> p4_recipe;

  // Diagnostic compatibility bit: true when every rolled data-moving loop in
  // this plan is stage-2.  Costing is phase-local and never uses this to hide
  // serial init/tail/finalize work behind another phase.
  bool overlap_granted = false;

  bool streamed() const { return feasible && axis != 0; }
};

// ============================================================================
// Derived cube schedule plan
// ============================================================================

// Symbolic source of one axis of a fixed tensor region. Bindings are propagated
// backwards from each sink request, so an emitter can recover both the concrete
// extent and offset for a spatial tile, parallel K share, or sequential K chunk
// without reclassifying the matmul DAG.
enum class CubeAxisBinding {
    Full,
    SpatialM,
    SpatialN,
    ParallelK,
    SequentialK
};

struct CubeTensorRegionPlan {
    size_t tensor = std::numeric_limits<size_t>::max();
    CubeAxisBinding height_binding = CubeAxisBinding::Full;
    CubeAxisBinding width_binding = CubeAxisBinding::Full;
    int64_t height = 0;
    int64_t width = 0;
};

struct CubeKLoopPlan {
    // `l1_window_k` is the total contraction extent whose two ping-pong buffers
    // fit in the L1 headroom derived by the pebble sweep. `chunk` is the actual
    // per-iteration slice the emitter must load. They intentionally differ:
    // the historical cost comments assume the emitter halves the resident
    // window when overlap is granted.
    int64_t l1_window_k = 0;
    int64_t chunk = 0;
    int64_t full_chunks = 0;
    int64_t tail = 0;
    int pipeline_stages = 1;
};

struct CubeFinalDrainPlan {
  // One explicit drain after the complete K-window loop. The partial C remains
  // in L0C throughout the loop: A2/A3 supports Acc->Mat/GM but no Mat->Acc.
  // For FirstPartialThenAtomic, `atomic` describes the atomic-rest variant;
  // share zero overrides it with the ordinary first-partial store.
  bool required = false;
  bool target_l1 = false;
  bool atomic = false;
  int64_t valid_rows = 0;
  int64_t valid_cols = 0;
  int64_t tile_count = 0;
  int64_t bytes = 0;
  double cycles = 0.0;
};

struct CubeOutputTileVariant {
  // At most four variants exist: full/full, tail/full, full/tail, tail/tail.
  // `count` lets the cost path remain O(1) in the number of L0 output tiles.
  int64_t height = 0;
  int64_t width = 0;
  int64_t count = 0;
  L0MatmulPlan l0_init;
  L0MatmulPlan l0_rolled;
  L0MatmulPlan l0_tail;
};

struct CubeRetainedPanelPlan {
  // Load a boundary panel from GM to L1 once before this request's output-tile
  // loop, then take local L1 slices for every child L0 tile. Retention is
  // deliberately scoped to one work unit and one matmul request; it assumes no
  // cross-core affinity or cross-kernel persistence.
  bool lhs = false;
  bool rhs = false;
  int64_t lhs_bytes = 0;
  int64_t rhs_bytes = 0;

  bool present() const { return lhs || rhs; }
  int64_t bytes() const { return lhs_bytes + rhs_bytes; }
};

enum class CubeOperandRole {
  Lhs,
  Rhs,
};

struct CubeResidentBoundaryPlan {
  // A canonical boundary request shared by multiple matmul instances in one
  // work unit. Identity includes the tensor region and the compatible operand
  // representation, so LHS and RHS requests never alias accidentally.
  size_t id = std::numeric_limits<size_t>::max();
  CubeTensorRegionPlan region;
  CubeOperandRole role = CubeOperandRole::Lhs;
  size_t first_use = 0;
  size_t last_use = 0;
  size_t use_count = 0;
  int64_t bytes = 0;
};

struct CubeMatmulSchedule {
  size_t instance = std::numeric_limits<size_t>::max();
  size_t op = std::numeric_limits<size_t>::max();
  int64_t lhs_producer = -1;  // schedule-instance index, -1 = boundary input
  int64_t rhs_producer = -1;
  // Index into CubeSchedulePlan::resident_boundaries, or -1 when this operand
  // is produced internally or streamed as a single-use boundary request.
  int64_t lhs_resident_boundary = -1;
  int64_t rhs_resident_boundary = -1;
  bool is_sink = false;
  bool lhs_ephemeral = false;
  bool rhs_ephemeral = false;
  bool output_ephemeral = false;
  int64_t contraction = 0;
  int64_t effective_contraction = 0;  // sink K/split; full K for internal ops
  CubeTensorRegionPlan lhs;
  CubeTensorRegionPlan rhs;
  CubeTensorRegionPlan output;
  // TMATMUL accumulates float operands in FP32 (integer operands in INT32),
  // independently of the tensor dtype used when the result drains to GM/L1.
  // A fused A2/A3 producer may drain FP32 Acc to BF16/FP16 Mat, but cannot
  // carry an FP32 Mat value back into Acc.
  DType accumulator_dtype = DType::FP32;
  DType storage_dtype = DType::FP32;
  CubeKLoopPlan k_loop;
  // Output/L0C subtiles are outer to the GM->L1 K-window loop. Every child
  // plan targets Acc, so its C stays resident across serial-init, rolled, and
  // tail phases. The explicit final_drain then publishes that completed tile.
  int64_t output_tile_m = 0;
  int64_t output_tile_n = 0;
  int64_t output_tiles_m = 0;
  int64_t output_tiles_n = 0;
  std::vector<CubeOutputTileVariant> output_variants;
  CubeRetainedPanelPlan retained_panels;
  CubeFinalDrainPlan final_drain;
};

enum class CubeSplitMergePolicy {
    None,
    FirstPartialThenAtomic,
    AivZeroSeedThenAtomic,
};

// FirstPartialThenAtomic emits two ordered AIC phases. The first phase assigns
// K share zero to every spatial output region and publishes it with an ordinary
// store. Only after that phase completes may the remaining shares atomic-add
// into the initialized output. The alternative AivZeroSeedThenAtomic policy
// emits an ordered AIV seed phase followed by one AIC atomic phase.
struct CubeFirstPartialThenAtomicPlan {
  bool present = false;
  int64_t first_work_units = 0;
  int64_t atomic_work_units = 0;
  // Explicit model hook for the phase boundary. It defaults to zero until a
  // device experiment isolates synchronization from the two launch walls.
  double synchronization_cycles = 0.0;
};

// Alternative split-K merge using the ordinary PyPTO AIV-seed + dependent
// AIC-atomic pattern. One vector work unit owns each spatial output region and
// zeroes all of its output subtiles. After that complete SPMD dispatch, every K
// share is runnable in one atomic AIC dispatch. The dependency is a runtime
// TaskId edge; workers never spin on a device flag.
struct CubeAivZeroSeedThenAtomicPlan {
  bool present = false;
  int64_t seed_work_units = 0;
  int64_t atomic_work_units = 0;
  int64_t seed_bytes = 0;
  double synchronization_cycles = 0.0;
};

// How one static per-work-unit tensor shape covers the spatial output grid.
// Uniform regions are disjoint. ClampedOverlap is the lone-matmul split=1
// fallback: every task computes the maximum region shape and the last task on
// a ragged axis clamps backward, recomputing an identical overlapping edge.
enum class CubeSpatialPolicy {
    Uniform,
    ClampedOverlap,
};

// The solver-owned algorithm descriptor for a homogeneous cube subgraph at one
// fixed TileConfig and split-K factor. Like VectorStreamPlan, this is rebuilt
// only for a winning/forced solution; it is deliberately absent from
// CostResult, the million-entry local-search cache value.
struct CubeSchedulePlan {
    bool feasible = false;
    // True when every request instance has a role-aware region and dependency
    // representation that the plan-driven emitter can replay.
    bool emit_compatible = false;
    TileConfig config;
    AxisPartition m_partition;
    AxisPartition n_partition;
    CubeSpatialPolicy spatial_policy = CubeSpatialPolicy::Uniform;
    int64_t spatial_tiles = 0;
    int64_t split_k = 1;
    int64_t work_units = 0;
    int64_t peak_l1_bytes = 0;
    CubeSplitMergePolicy split_merge_policy = CubeSplitMergePolicy::None;
    CubeFirstPartialThenAtomicPlan first_partial_then_atomic;
    CubeAivZeroSeedThenAtomicPlan aiv_zero_seed_then_atomic;
    // The cost and emitter both derive this from the concrete per-request K
    // loops. Both fields are retained so validation can fail loudly if a future
    // model change grants overlap that the reconstructed loop cannot realize.
    bool model_overlap_granted = false;
    bool overlap_implementable = false;
    std::vector<CubeResidentBoundaryPlan> resident_boundaries;
    std::vector<size_t> execution_order;
    std::vector<CubeMatmulSchedule> matmuls;
};

// ============================================================================
// Derived mixed cube/vector schedule plan
// ============================================================================

// One logical 910B mixed group contains one cube core and two vector cores.
// Same-engine connected ops form stages; every edge between unlike stages is a
// GM FIFO transfer because 910B has no direct UB<->Mat/L1 path.
enum class MixedEngine { Cube, Vector };

enum class MixedPipelineMode {
    Serial,
    OneWay,
    SingleRoundTripSkew,
    MultiRoundTripSequential
};

// Descriptor-level protocols shared by the mixed planner and PyPTO's public
// cross-core pipe contract. SingleRoundTripBundle is:
//
//   producer_bundle[0..N] -> one peer stage -> one reply -> sink stage
//
// where every producer bundle transfer has the same direction.
// BranchedRoundTripBundle is a three-phase fan-in/fan-out protocol:
//
//   producer stages --bundle 0--> peer stages --bundle 1--> one sink stage
//
// with one reverse-direction reply per peer. MultiRoundTripSequential is
// initially the exact linear C->V->C->V chain; it
// preserves all three FIFO descriptors but deliberately receives no skew
// overlap. Adjacent same-engine operations have already been collapsed into
// one stage; the two physical AIV lanes are represented by MixedVectorSplit,
// not by two stages.
enum class MixedCrossCoreProtocol {
  Unsupported,
  OneWay,
  SingleRoundTripBundle,
  BranchedRoundTripBundle,
  MultiRoundTripSequential
};

enum class MixedPipelineAxis {
  SpatialRegion,
  VectorWidthChunk,
  VectorHeightChunk,
  AttentionKeyChunk,
  IntermediateFeatureChunk
};

// Source algorithm selected by the mixed planner. FeatureChunkRoundTrip is a
// topology/geometry mechanism, not a workload recognizer: multiple cube
// projections feed an arbitrary supported vector DAG, whose result is consumed
// by a sink matmul one feature chunk at a time.
enum class MixedAlgorithmKind { Generic, FeatureChunkRoundTrip };

// How the vector stage uses the two AIV lanes in one 910B mixed group.  This
// is an algorithmic fact, not merely a hardware count: an unsplit mixed kernel
// executes real vector work on lane 0 while lane 1 only replays FIFO
// handshakes.  The cost may divide vector work by two only for Rows/Columns.
enum class MixedVectorSplit {
    None,
    Rows,
    Columns
};

enum class MixedTransferDirection {
    CubeToVector,
    VectorToCube
};

struct MixedStageTopology {
    MixedEngine engine = MixedEngine::Vector;
    std::vector<size_t> ops;
};

struct MixedTransferTopology {
    size_t tensor = std::numeric_limits<size_t>::max();
    size_t producer_stage = std::numeric_limits<size_t>::max();
    size_t consumer_stage = std::numeric_limits<size_t>::max();
    MixedEngine producer_engine = MixedEngine::Vector;
    MixedEngine consumer_engine = MixedEngine::Vector;
};

struct MixedCrossCoreProtocolTopology {
  MixedCrossCoreProtocol kind = MixedCrossCoreProtocol::Unsupported;
  MixedEngine producer_engine = MixedEngine::Cube;
  MixedEngine peer_engine = MixedEngine::Vector;
  std::vector<size_t> producer_stages;
  size_t peer_stage = std::numeric_limits<size_t>::max();
  size_t sink_stage = std::numeric_limits<size_t>::max();
  // Indices into MixedScheduleTopology::transfers. The producer bundle is
  // advanced together; a single-round-trip bundle has exactly one reply while
  // the branched protocol carries one reply from each peer stage.
  std::vector<size_t> producer_bundle_transfers;
  std::vector<size_t> reply_bundle_transfers;
  bool skew_pass_compatible = false;
};

struct MixedFeatureRoundTripTopology {
  bool present = false;
  std::vector<size_t> producer_matmuls;
  std::vector<size_t> producer_tensors;
  size_t peer_stage = std::numeric_limits<size_t>::max();
  size_t sink_matmul = std::numeric_limits<size_t>::max();
  size_t reply_tensor = std::numeric_limits<size_t>::max();
  std::vector<int64_t> producer_input_extents;
  std::vector<DType> producer_operand_dtypes;
  int64_t intermediate_extent = 0;
  int64_t output_extent = 0;
  DType sink_operand_dtype = DType::FP32;
  // Candidate K values partition every producer output, the complete vector
  // DAG, and the sink-matmul contraction. They are built once with the
  // topology, then crossed with the final output grid.
  std::vector<int64_t> feature_chunks;
};

// Candidate-invariant portion of a mixed schedule. It is built once by create;
// hot candidates read the owning cost model directly, while an explicitly
// requested final plan receives a shared owner. CostResult does not retain it.
struct MixedScheduleTopology {
  std::vector<MixedStageTopology> stages;
  std::vector<MixedTransferTopology> transfers;
  int max_alternations = 0;
  bool output_is_cube = false;
  bool output_engines_uniform = true;
  bool sink_runs_early_stage = false;
  MixedPipelineMode mode = MixedPipelineMode::Serial;
  bool emit_compatible = false;
  // Strict subset that production AutoFuse can currently replay.  The more
  // permissive emit_compatible bit remains useful to analytic/research
  // callers studying the generic cross-core passes.
  bool compiler_emit_compatible = false;
  MixedCrossCoreProtocolTopology protocol;
  MixedAlgorithmKind algorithm = MixedAlgorithmKind::Generic;
  MixedFeatureRoundTripTopology feature_round_trip;
};

struct MixedFifoPlan {
  size_t tensor = std::numeric_limits<size_t>::max();
  MixedTransferDirection direction = MixedTransferDirection::CubeToVector;
  // Bind a spatial-region FIFO frame to the unified output grid without
  // inferring axis identity from equal numeric extents. Protocol-local loops
  // such as generic feature chunks retain their own serialized frame geometry.
  bool spatial_m = false;
  bool spatial_n = false;
  int64_t valid_rows = 0;
  int64_t valid_cols = 0;
  int64_t slot_bytes = 0;
  int64_t slot_count = 0;
  // Vec memory reserved by ExpandMixedKernel for this FIFO.  Keeping this in
  // the winning descriptor makes the plan's feasibility proof inspectable
  // and lets emission fail closed if the downstream default changes.
  int64_t reserved_bytes = 0;
  // Stable protocol identity. Pipes in one bundle are advanced together by
  // SkewCrossCorePipeline; order inside a bundle is the topology order.
  int pipe_id = -1;
  int bundle = -1;
};

struct MixedPipelineLoopPlan {
    MixedPipelineAxis axis = MixedPipelineAxis::SpatialRegion;
    int64_t extent = 0;
    int64_t chunk = 1;
    int64_t items_per_spatial_tile = 1;
    int64_t work_items = 0;
    int64_t active_groups = 0;
    int64_t min_trips_per_group = 0;
    int64_t max_trips_per_group = 0;
    int pipeline_stages = 1;
    int64_t requested_skew_depth = 0;
};

// Final-plan-only stage descriptor. Candidate-hot costing leaves this vector
// empty; mixed_schedule_plan() reconstructs it once for a selected/forced
// candidate so model/emit validation does not rediscover stage geometry.
struct MixedStagePlan {
  size_t topology_stage = std::numeric_limits<size_t>::max();
  MixedEngine engine = MixedEngine::Vector;
  std::vector<size_t> ops;
  int64_t valid_rows = 0;
  int64_t valid_cols = 0;
  std::vector<int64_t> cube_window_k;
  VectorStreamPlan vector_stream;
};

struct MixedFeatureRoundTripPlan {
  bool present = false;
  int64_t intermediate_extent = 0;
  int64_t intermediate_chunk = 0;
  int64_t intermediate_chunks = 0;
  int64_t output_extent = 0;
  std::vector<int64_t> producer_window_k;
  int64_t persistent_accumulator_bytes = 0;
  bool first_chunk_initializes = false;
  bool later_chunks_accumulate = false;
};

// Solver-owned cross-engine algorithm for one fixed mixed candidate. The hot
// path constructs only its scalar fields for costing; final/forced consumers
// re-derive it once and attach the immutable topology, like the homogeneous
// vector and cube plans.
struct MixedSchedulePlan {
  bool feasible = false;
  bool emit_compatible = false;
  // True only when this selected analytic plan carries a complete stage/FIFO
  // contract that a source backend can consume without re-planning. Analytic
  // solutions may remain feasible when this is false.
  bool source_codegen_ready = false;
  TileConfig config;
  AxisPartition m_partition;
  AxisPartition n_partition;
  int64_t spatial_tiles = 0;
  int64_t split_k = 1;
  int64_t work_units = 0;
  int64_t group_capacity = 0;
  MixedCrossCoreProtocol protocol = MixedCrossCoreProtocol::Unsupported;
  MixedAlgorithmKind algorithm = MixedAlgorithmKind::Generic;
  // Stage-local facts consumed by the first compiler emitter.  These are
  // populated only when reconstructing a winning/forced plan, not retained
  // in the hot CostResult cache.
  int64_t cube_window_k = 0;
  // Peak cube-side L1 operand/resident footprint before V2C consumer rings.
  // Source readiness combines this with every V2C FIFO reservation.
  int64_t cube_stage_peak_l1_bytes = 0;
  VectorStreamKind vector_stage_kind = VectorStreamKind::Materialized;
  int64_t vector_stage_peak_ub_bytes = 0;
  MixedVectorSplit vector_split = MixedVectorSplit::None;
  int64_t vector_lanes = 1;
  MixedPipelineMode mode = MixedPipelineMode::Serial;
  MixedPipelineLoopPlan loop;
  std::vector<MixedFifoPlan> fifos;
  std::vector<MixedStagePlan> stages;
  MixedFeatureRoundTripPlan feature_round_trip;
  // Kept separate during migration: the first bit records what the legacy
  // scalar cost grants, while the second mirrors what the existing PyPTO
  // pipeline passes can actually construct for this exact topology/loop.
  bool model_overlap_granted = false;
  bool overlap_implementable = false;
  bool pipeline_fill_absorbed = false;
  std::shared_ptr<const MixedScheduleTopology> topology;
};

// ============================================================================
// Cost evaluation result
// ============================================================================

struct CostResult {
    bool feasible = false;
    double latency = std::numeric_limits<double>::infinity();
    int num_spatial_tiles = 0;
    int num_k_passes = 0;
    TileConfig config;
    // 910B parallel-roofline introspection (set by the cores>1 override). Lets
    // tests visualize the chosen strategy and the eventual emit act on it.
    int parallel_split = 1;      // cores ganged per spatial tile: matmul split-K S /
                                 // reduction split S. 1 = pure spatial parallelism.
    CubeSplitMergePolicy cube_split_merge_policy = CubeSplitMergePolicy::None;
    int cores_used = 1;          // effective cores busy (<= total unit cores)
    int mixed_active_groups = 0; // selected 1-AIC + 2-AIV groups for a mixed schedule;
                                 // zero for homogeneous schedules
    bool compute_bound = false;  // latency limited by compute (vs the shared DDR floor)
    double ddr_traffic = 0.0;    // DDR bytes/BW for this tile (reload + merge). Secondary
                                 // objective: among equal-latency tiles prefer less DDR
                                 // (=> larger, balanced tiles with better L1/L0 reuse).
    double l1l0_extract = 0.0;   // L1->L0 extract cycles (MTE1): lhs via L0A, rhs via L0B.
                                 // TIEBREAKER ONLY (feed dominates the wall): the L0A/L0B
                                 // ports are asymmetric (441 vs 220.5), so among reload-equal
                                 // transposed tiles the lower-extract (TALL) tile wins.
                                 // Perf-sim-driven (pto-isa gml1_decision); device eval pending.
    bool pipeline_fill_absorbed = false;  // MIXED kernels only: is the cube||vector fill/drain
                                 // ACTUALLY absorbed into the max for THIS config. true iff the
                                 // topology is an exact replayable 3-stage chain (v->c->v or
                                 // c->v->c), the sink unit runs an early stage, and num_tiles>=2.
                                 // Multi-message/round-trip plans are serial and never absorb it.
    bool uses_model_ahead_split_k = false;  // this config's parallel_split > 1 came from the
                                 // model-ahead split-K path (Problem::allow_model_ahead_split_k;
                                 // base lone-matmul or mixed cube-sink) — NOT yet emittable.
};

// Developer-facing mixed-candidate decomposition. This stays out of
// CostResult because that value occupies the million-entry local-search cache.
// Every field is reconstructed by the production cost path; the diagnostic
// sweep never re-prices a candidate in Python.
struct MixedCostBreakdown {
    bool feasible = false;
    int64_t active_groups = 0;
    int64_t trips_per_group = 0;
    int64_t pipeline_stages = 0;
    bool overlap_implementable = false;
    double cube_phase_cycles = 0.0;
    double vector_phase_cycles = 0.0;
    double gm_l1_bytes = 0.0;
    double gm_ub_bytes = 0.0;
    double l0c_gm_bytes = 0.0;
    double ub_gm_bytes = 0.0;
    double gm_l1_effective_parallelism = 0.0;
    double gm_ub_effective_parallelism = 0.0;
    double l0c_gm_effective_parallelism = 0.0;
    double ub_gm_effective_parallelism = 0.0;
    double gm_l1_cycles = 0.0;
    double gm_ub_cycles = 0.0;
    double l0c_gm_cycles = 0.0;
    double ub_gm_cycles = 0.0;
    double ddr_wall_cycles = 0.0;
    double pipeline_wall_cycles = 0.0;
    double kernel_fill_cycles = 0.0;
    double group_overhead_cycles = 0.0;
    double total_cycles = std::numeric_limits<double>::infinity();
};

struct MixedGroupCostCandidate {
    CostResult cost;
    MixedCostBreakdown breakdown;
};
