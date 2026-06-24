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
    int64_t slow_memory_bandwidth;
    int64_t native_w, native_h;

    // --- Ascend 910B extensions ----------------------------------------------
    // Compute parallelizes across these cores; DDR bandwidth is shared (one HBM).
    // Default 1 = SINGLE-CONTEXT: the parallel roofline + unit-homogeneity
    // constraint are inert, so legacy/competition instances are unchanged. The
    // 910B path (adapter / tests) sets the real counts (cube 24, vector 48).
    int num_cube_cores = 1;      // AIC cores — matmul
    int num_vector_cores = 1;    // AIV cores — pointwise / reduction
    // Per-core local-memory budgets in BYTES. 0 = fall back to the single
    // fast_memory_capacity pool (so legacy element-count instances are
    // unchanged). The 910B feasibility check forks on cube-vs-vector:
    //   cube subgraph : operand strips -> L1, output accumulator -> L0c
    //   vector subgraph: tile + ephemerals -> UB
    int64_t vec_capacity = 0;    // per-vector-core UB (910B: 192*1024)
    int64_t cube_capacity = 0;   // per-cube-core L0c accumulator (910B: 128*1024)
    int64_t l1_capacity = 0;     // per-cube-core L1/Mat operand pool (910B: 512*1024)
    // 910B compute is derived from TILE GEOMETRY x a machine cost, NOT a per-op
    // padded-native-tile base_cost (the competition model). cube_compute_cost is
    // the time for one 16x16x16 cube fractal; vector_compute_cost is the time per
    // vector element. 0 => fall back to the per-op base_cost (legacy/competition).
    // cube does one 16x16x16 fractal per step; vector processes vector_lanes
    // elements per SIMD step. compute = (#steps) x (per-step cost). vector_lanes
    // is a machine parameter (SIMD width) to be calibrated later; 0 => treat as 1
    // (per-element). The cube step is fixed at the 16x16x16 fractal (hardware).
    int64_t cube_compute_cost = 0;    // cost per 16x16x16 cube fractal step
    int64_t vector_compute_cost = 0;  // cost per vector SIMD step
    int64_t vector_lanes = 0;         // elements per vector SIMD step (0 => 1)
    // Per-kernel pipeline fill/drain latency. A subgraph's tiling produces
    // num_tiles "kernels"; each core runs ceil(num_tiles/cores) of them in
    // sequence, paying one fill per pass. Charged as rounds*kernel_fill_cost,
    // the DUAL of the eff core-fill incentive: eff penalizes under-tiling
    // (fewer tiles than cores), this penalizes over-tiling (more) — so the
    // optimum sits at ~one kernel per core. 0 => off (legacy/competition).
    int64_t kernel_fill_cost = 0;
    // Ping-pong double-buffering: reserve half of each STREAMING pool (L1 / UB)
    // for prefetch of the next tile while the current one computes. Halves the
    // effective L1 and UB budgets. Off => single-buffer (full budget).
    bool double_buffer = false;
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
    // The cost terms above (cube_compute_cost per fractal, the single
    // slow_memory_bandwidth) are coarse placeholders. These fields, when set,
    // switch the cube path to pto-isa's measured coefficients
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
    //
    // 0 / unset => fall back to slow_memory_bandwidth + cube_compute_cost (the
    // competition / legacy instances are byte-for-byte unchanged).
    double cube_freq_hz = 0.0;   // core clock (1.85e9). >0 activates the grounded path.
    double bw_gm_l1   = 0.0;     // GM->L1 operand reload      (GiB/s; pto-isa 135.0)
    double bw_l0c_gm  = 0.0;     // L0C->GM output store       (GiB/s; pto-isa 70.0)
    double bw_l1_l0a  = 0.0;     // L1->L0A lhs extract        (GiB/s; pto-isa 441.0)
    double bw_l1_l0b  = 0.0;     // L1->L0B rhs extract        (GiB/s; pto-isa 220.5)
    double bw_gm_ub   = 0.0;     // GM->UB vector load         (GiB/s; pto-isa 100.9)
    double bw_ub_gm   = 0.0;     // UB->GM vector store        (GiB/s; pto-isa 188.46)
    int64_t l0_tile_m = 0;       // L0 GEMM base M (pto-isa oracle 128) — L1->L0 reuse
    int64_t l0_tile_n = 0;       // L0 GEMM base N (pto-isa oracle 256) — L1->L0 reuse

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
    int cores_used = 1;          // effective cores busy (<= total unit cores)
    bool compute_bound = false;  // latency limited by compute (vs the shared DDR floor)
    double ddr_traffic = 0.0;    // DDR bytes/BW for this tile (reload + merge). Secondary
                                 // objective: among equal-latency tiles prefer less DDR
                                 // (=> larger, balanced tiles with better L1/L0 reuse).
};