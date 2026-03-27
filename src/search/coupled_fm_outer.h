#pragma once

#include "search/coupled_fm_pass.h"
#include "search/fm_outer.h"    // FMOuterConfig (reused as-is)
#include "core/cost_cache.h"
#include <set>

// ============================================================================
// Coupled FM outer loop result
// ============================================================================

struct CoupledFMOuterResult {
    CoupledPartition best_cp;
    CoupledPartition end_cp;
    double best_cost      = 1e18;
    double end_cost       = 1e18;
    int    total_passes   = 0;
    int    improving_passes = 0;
    int    total_moves    = 0;
    int    elapsed_ms     = 0;
};

// ============================================================================
// Run the FM outer loop on a CoupledPartition.
//
// Mirrors fm_outer_loop:
//   1. For each pass: run coupled_fm_inner_pass from the input cp
//   2. If pass found a new best, update best_overall
//   3. Run coupling_greedy_descent on the end state (explore different basin)
//   4. Stop on max_passes, max_no_improve, or deadline
//
// cache: used to finalize groups that are missing .sg after mutation.
//        May be null (finalize() uses its own cache).
// ============================================================================

CoupledFMOuterResult coupled_fm_outer_loop(
    CoupledPartition        cp,
    const std::set<size_t>& feasibly_ret,
    const FMOuterConfig&    cfg   = {},
    CostCache*              cache = nullptr);
