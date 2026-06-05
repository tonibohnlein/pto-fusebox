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
enum class OpType { MatMul, Pointwise, Reduction };

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

    // --- Ascend 910B extensions (defaults = one 910B die) ---------------------
    // Compute parallelizes across these cores; DDR bandwidth is shared (one HBM).
    int num_cube_cores = 24;     // AIC cores — matmul
    int num_vector_cores = 48;   // AIV cores — pointwise / reduction
    // Per-core local-memory budgets in BYTES. 0 = fall back to
    // fast_memory_capacity (so legacy element-count instances are unchanged).
    int64_t vec_capacity = 0;    // per-vector-core UB (910B: 192*1024)
    int64_t cube_capacity = 0;   // per-cube-core L0c accumulator budget (910B: 128*1024)

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
// Tile configuration & traversal direction
// ============================================================================

enum class SnakeDir { None, RowMajor, ColMajor };

struct TileConfig {
    int64_t w = 0, h = 0, k = 0;
    SnakeDir snake = SnakeDir::None;
};

// ============================================================================
// Cost evaluation result
// ============================================================================

struct CostResult {
    bool feasible = false;
    double latency = std::numeric_limits<double>::infinity();
    int64_t working_set = 0;
    int num_spatial_tiles = 0;
    int num_k_passes = 0;
    double compute_per_step = 0;
    TileConfig config;
};