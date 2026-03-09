#pragma once

#include "core/types.h"
#include "core/dag.h"
#include <optional>
#include <set>
#include <vector>

// ============================================================================
// Subgraph: a connected group of ops with exactly one output tensor.
//
// Invariants enforced at construction:
//   1. Single-sink: exactly one boundary output tensor
//   2. Connected: ops form a connected sub-DAG (undirected)
//
// The Subgraph caches boundary analysis and provides cost evaluation
// for any given TileConfig. It does NOT own a TileConfig — that's the
// caller's responsibility (or the Solution class).
// ============================================================================

class Subgraph {
public:
    // Factory: returns nullopt if ops don't form a valid subgraph.
    // Checks: non-empty, single-sink, connected.
    static std::optional<Subgraph> create(const Problem& prob,
                                          const DAG& dag,
                                          std::vector<size_t> op_indices);

    // --- Accessors ---

    const Problem& problem() const { return *prob_; }
    const DAG& dag() const { return *dag_; }
    const std::vector<size_t>& ops() const { return ops_; }
    size_t num_ops() const { return ops_.size(); }

    const std::set<size_t>& boundary_inputs() const { return boundary_inputs_; }
    const std::set<size_t>& boundary_outputs() const { return boundary_outputs_; }
    const std::set<size_t>& ephemeral() const { return ephemeral_; }

    size_t sink_tensor() const { return sink_tensor_; }
    int64_t output_width() const { return out_W_; }
    int64_t output_height() const { return out_H_; }

    bool has_matmul() const { return has_matmul_; }

    // Reduction dimension for a specific MatMul op (= LHS tensor width).
    // Undefined for Pointwise ops.
    int64_t op_K(size_t op_idx) const {
        return prob_->tensors[prob_->ops[op_idx].inputs[0]].width;
    }

    // Maximum K across all MatMul ops in this subgraph (determines num_k_passes).
    int64_t max_K() const { return max_K_; }

    // --- Tiling validity ---

    // Check that [w, h, k] is compatible with ALL ops in the subgraph:
    //   - w divides every op's output width (including ephemeral intermediates)
    //   - h divides every op's output height
    //   - k divides every MatMul's K dimension
    // This is required because the hardware applies one unified execution grid.
    bool is_valid_tiling(const TileConfig& cfg) const;

    // --- Feasibility ---

    // Peak working set for one tile-step (worst case across all tiles).
    int64_t working_set(const TileConfig& cfg,
                        const std::set<size_t>& retained_from_prev = {},
                        const std::set<size_t>& retain_these = {}) const;

    // Does the working set fit in fast memory?
    bool is_feasible(const TileConfig& cfg,
                     const std::set<size_t>& retained_from_prev = {},
                     const std::set<size_t>& retain_these = {}) const;

    // --- Cost evaluation ---

    // Total latency at the given configuration.
    CostResult compute_cost(const TileConfig& cfg,
                            const std::set<size_t>& retained_from_prev = {},
                            const std::set<size_t>& retain_these = {}) const;

    // --- Parameter enumeration ---

    // Find the TileConfig with minimum latency (exhaustive over power-of-2 divisors).
    CostResult best_cost(const std::set<size_t>& retained_from_prev = {},
                         const std::set<size_t>& retain_these = {}) const;

    Subgraph() = default;  // needed for ScheduleStep in vector ops

private:

    const Problem* prob_ = nullptr;
    const DAG* dag_ = nullptr;
    std::vector<size_t> ops_;

    std::set<size_t> boundary_inputs_;
    std::set<size_t> boundary_outputs_;
    std::set<size_t> ephemeral_;

    size_t sink_tensor_ = 0;
    int64_t out_W_ = 0, out_H_ = 0;
    bool has_matmul_ = false;
    int64_t max_K_ = 1;
    std::vector<int64_t> all_K_values_;      // distinct K values (for k enumeration)
    std::vector<int64_t> all_out_widths_;    // all op output widths (for w validation)
    std::vector<int64_t> all_out_heights_;   // all op output heights (for h validation)

    // Precomputed per-boundary-tensor info for fast working_set/compute_cost.
    // Avoids std::map allocation on every call.
    struct BoundaryTensorInfo {
        size_t id;
        int64_t max_lhs_K = 0;    // max K among MM ops using this as LHS (0 = not LHS)
        bool is_mm_rhs = false;
        bool is_mm_out = false;
        bool is_pw_in = false;     // used as PW input (or MM out for ws purposes)
        bool is_boundary_out = false;
    };
    std::vector<BoundaryTensorInfo> boundary_tensor_info_;  // built at create() time
};