#pragma once

#include "solution/solution.h"
#include <chrono>
#include <set>
#include <vector>
#include <optional>

// ============================================================================
// Solution-level FM search: unified partition + retain moves.
//
// Operates on a mutable solution state. Each move modifies step structure
// (ops, tiling) and/or retain decisions, then re-evaluates affected steps.
//
// Move vocabulary:
//   PARTITION moves (change subgraph structure):
//     STEAL:  move border op from step i to adjacent step j
//     SPLIT:  split step at bridge edge into two sub-steps
//     MERGE:  fuse two adjacent steps into one
//
//   RETAIN moves (change data residency):
//     RETAIN_ADD:    retain tensor T at step i (for step i+1)
//     RETAIN_REMOVE: stop retaining tensor T at step i
//
//   COMBINED moves (ordering + partition + retain):
//     RELOCATE: move step to different position for better retain
//
// FM-style execution: evaluate all moves, pick best, lock affected
// steps/tensors, re-evaluate neighborhood, repeat.
// ============================================================================

struct SolutionMove {
    enum Type {
        NONE = -1,
        STEAL = 0,       // move op from step_a to step_b
        SPLIT = 1,       // split step_a at bridge edge
        MERGE = 2,       // merge step_a and step_a+1
        RETAIN_ADD = 3,  // add tensor to step_a's retain set
        RETAIN_REMOVE = 4, // remove tensor from step_a's retain set
        RELOCATE = 5,    // move step_a to position step_b
    };

    Type type = NONE;
    size_t step_a = SIZE_MAX;  // primary step
    size_t step_b = SIZE_MAX;  // secondary step (target for steal, merge partner, destination)
    size_t op = SIZE_MAX;      // op to move (STEAL)
    size_t tensor = SIZE_MAX;  // tensor (RETAIN_ADD/REMOVE, SPLIT bridge)
    size_t op2 = SIZE_MAX;     // second bridge op (SPLIT)
    double saving = -1e18;     // positive = improvement

    bool valid() const { return type != NONE; }
};

// ============================================================================
// Solution FM configuration
// ============================================================================

struct SolutionFMConfig {
    int max_rounds = 30;
    int max_no_improve = 10;
    double floor_fraction = 0.20;
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
};

// ============================================================================
// Run solution-level FM search.
// Returns improved solution (never worse than input).
// ============================================================================

Solution solution_fm_search(const Problem& prob, const DAG& dag,
                             Solution sol, const SolutionFMConfig& cfg = {});