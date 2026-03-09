#pragma once

#include "partition/partition.h"
#include "search/active_set.h"
#include <random>

// ============================================================================
// FM pass configuration
// ============================================================================

struct FMConfig {
    double floor_fraction = 0.30;     // allow moves worsening by up to this % of cost
    double max_drift_fraction = 0.50; // abort if cumulative gain drops this % below best
    int init_count = 3;               // number of border ops to activate initially
    unsigned seed = 42;               // RNG seed for random initial subset
};

// ============================================================================
// FM pass result
// ============================================================================

struct FMPassResult {
    Partition best_partition;    // best partition seen during this pass
    Partition end_partition;     // final state after all moves (maximally perturbed)
    double best_cost = 1e18;    // cost of best_partition
    double end_cost = 1e18;     // cost of end_partition
    double start_cost = 1e18;   // cost at start of pass
    int moves_applied = 0;      // total moves applied
    int moves_positive = 0;     // moves with positive saving
    int moves_negative = 0;     // moves with negative saving
};

// ============================================================================
// Run one FM inner iteration (one pass).
//
// Starting from the given partition:
//   1. Activate a random subset of border ops
//   2. Pop best move, apply, lock initiating op (+ merge partner)
//   3. Update affected ops, activate new border ops of affected groups
//   4. Track cumulative gain, snapshot at each new best
//   5. Stop on max_drift or exhaustion
//   6. Return best partition seen during this pass
// ============================================================================

FMPassResult fm_inner_pass(Partition part, const FMConfig& cfg = {});