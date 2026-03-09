#pragma once

#include "partition/partition.h"
#include "search/fm_pass.h"
#include <chrono>

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;

// ============================================================================
// FM outer loop configuration
// ============================================================================

struct FMOuterConfig {
    int max_passes = 100;             // quality-focused: allow many passes
    int max_no_improve = 30;          // experiment showed improvements at pass 25,32,37
    FMConfig pass_config;             // per-pass configuration (floor, drift, init_count)
    TimePoint deadline = TimePoint::max();  // wall-clock cutoff
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
//   3. If not, run greedy on the perturbed result (escape mechanism)
//   4. Stop after max_passes, max_no_improve, or deadline
// ============================================================================

FMOuterResult fm_outer_loop(Partition part, const FMOuterConfig& cfg = {});