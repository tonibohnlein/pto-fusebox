#pragma once

#include "partition/partition.h"
#include "search/fm_pass.h"

// ============================================================================
// FM outer loop configuration
// ============================================================================

struct FMOuterConfig {
    int max_passes = 50;              // enough to find late improvements (was 100)
    int max_no_improve = 15;          // transformer-large found improvements at pass 25,32,37
    FMConfig pass_config;             // per-pass configuration (floor, drift, init_count)
};

// ============================================================================
// FM outer loop result
// ============================================================================

struct FMOuterResult {
    Partition best_partition;
    double best_cost = 1e18;
    int total_passes = 0;
    int improving_passes = 0;
    int total_moves = 0;
};

// ============================================================================
// Run the full FM outer loop.
//
// Starting from the given partition:
//   1. Run fm_inner_pass with a different seed each pass
//   2. If the pass found a new best, update best_overall
//   3. Always restart from best_overall (rollback)
//   4. Stop after max_passes or max_no_improve consecutive non-improving passes
// ============================================================================

FMOuterResult fm_outer_loop(Partition part, const FMOuterConfig& cfg = {});