#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/flat_set.h"

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

struct Op {
    OpType type;
    std::vector<size_t> inputs;   // tensor indices
    std::vector<size_t> outputs;  // tensor indices (always exactly one per competition rules)
    int64_t base_cost;

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
    int64_t native_w, native_h;

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
    // Double-buffering is ALWAYS assumed on 910B but does NOT reserve half the
    // pool: the two ping-pong buffers together ARE the L1/UB, so feasibility uses
    // the FULL l1_capacity / vec_capacity. The overlap is realized in the emit by
    // streaming each seq-K strip (or tile) as >=2 sub-strips -- it halves the
    // per-load k, not the resident operand. (Hence no double_buffer flag.)
    // Parallel split-K aggregation path (a target capability — the scheduler's
    // analog of a compiler flag). When TRUE, the hardware can write back to DDR
    // with SetAtomicAdd, so the S split-K partials are atomically accumulated
    // into the DDR output DURING write-back — no separate read-back + sum. The
    // merge barrier is then just the S partial writes (sat-discounted), so
    // split-K stays cheap and max-fill is optimal. When FALSE (default), the
    // partials are written, then read back and summed serially per output tile
    // (one DDR accumulator/tile) — the merge grows ~linearly with S, punishing
    // split-K. The Ascend 910B HAS SetAtomicAdd, so set_910b enables it.
    bool ddr_atomic_add = false;

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
    int64_t l0_tile_m = 0;       // L0 GEMM base M (pto-isa oracle 128) — L1->L0 reuse
    int64_t l0_tile_n = 0;       // L0 GEMM base N (pto-isa oracle 256) — L1->L0 reuse

    // Grounded vector (AIV) compute, in CYCLES (pto-isa A2A3
    // cce_costmodel_vector_compute.hpp). Active in grounded mode (cube_freq_hz>0).
    // Each vector op costs head + slope*repeat + tail, repeat = ceil(elems /
    // (vec_reg_bytes / dtype_bytes)) -- the 256-byte vector register holds
    // 64 fp32 / 128 fp16 elements (fp16 = 2x throughput). slope picks elementwise
    // (vadd/vmul ~1-2) vs reduction (vreducev2 ~14), so a reduction is ~14x an add
    // and the per-op head/tail charges the pipeline fill of a multi-op chain
    // (e.g. softmax = exp + reduce + div).
    int64_t vec_reg_bytes = 256;      // vector register size in bytes (pto-isa 256)
    double vec_op_head = 0.0;         // per-op pipeline startup cycles (~14)
    double vec_op_tail = 0.0;         // per-op drain cycles (~18)
    double vec_slope_pw = 0.0;        // cycles/repeat, elementwise (~2)
    double vec_slope_reduce = 0.0;    // cycles/repeat, reduction (~14)

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
    int cores_used = 1;          // effective cores busy (<= total unit cores)
    bool compute_bound = false;  // latency limited by compute (vs the shared DDR floor)
    double ddr_traffic = 0.0;    // DDR bytes/BW for this tile (reload + merge). Secondary
                                 // objective: among equal-latency tiles prefer less DDR
                                 // (=> larger, balanced tiles with better L1/L0 reuse).
};