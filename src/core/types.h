#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/flat_set.h"

// ============================================================================
// Tensor & Op
// ============================================================================

struct Tensor {
    int64_t width;
    int64_t height;
    int64_t size() const { return width * height; }
};

enum class OpType { MatMul, Pointwise };

struct Op {
    OpType type;
    std::vector<size_t> inputs;   // tensor indices
    std::vector<size_t> outputs;  // tensor indices
    int64_t base_cost;
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