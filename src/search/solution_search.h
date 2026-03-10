#pragma once

#include "solution/solution.h"
#include <chrono>
#include <set>

// ============================================================================
// Solution-level FM search
//
// Mirrors the partition FM architecture with the full move vocabulary:
//   Partition moves: STEAL, MERGE, RECOMPUTE, EJECT, INTERNAL_EJECT, SPLIT
//   Retain moves:    RETAIN_ADD, RETAIN_REMOVE
// ============================================================================

struct SolutionMove {
    enum Type { NONE=-1, STEAL=0, SPLIT=1, MERGE=2, RETAIN_ADD=3, RETAIN_REMOVE=4,
                RECOMPUTE=5, EJECT=6, INTERNAL_EJECT=7 };
    Type type = NONE;
    size_t step_a = SIZE_MAX;  // primary step
    size_t step_b = SIZE_MAX;  // secondary step
    size_t op = SIZE_MAX;      // op involved
    size_t op2 = SIZE_MAX;     // second op (SPLIT bridge)
    size_t tensor = SIZE_MAX;  // tensor (RETAIN_ADD/REMOVE)
    double saving = -1e18;
    int gen_a = -1, gen_b = -1;  // generation counters for heap staleness

    bool valid() const { return type != NONE; }
    bool operator<(const SolutionMove& o) const { return saving < o.saving; }
};

struct SolutionFMPassConfig {
    double floor_fraction = 0.30;
    double max_drift_fraction = 0.50;
    unsigned seed = 42;
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
};

struct SolutionFMPassResult {
    std::vector<ScheduleStep> best_steps;
    std::vector<ScheduleStep> end_steps;
    double best_cost = 1e18;
    double end_cost = 1e18;
    double start_cost = 1e18;
    int moves_applied = 0;
};

struct SolutionFMConfig {
    int max_passes = 100;
    int max_no_improve = 30;
    SolutionFMPassConfig pass_config;
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
};

// One FM pass (active set + op/tensor locking + drift)
SolutionFMPassResult solution_fm_pass(const Problem& prob, const DAG& dag,
                                       std::vector<ScheduleStep> steps,
                                       const SolutionFMPassConfig& cfg = {});

// Greedy hill climb with heap-based lazy move selection
std::vector<ScheduleStep> solution_greedy_descent(const Problem& prob, const DAG& dag,
                                                    std::vector<ScheduleStep> steps,
                                                    std::chrono::steady_clock::time_point deadline 
                                                    = std::chrono::steady_clock::time_point::max());

// Parallel FM search: N threads with different seeds, adaptive cooling
Solution solution_fm_search(const Problem& prob, const DAG& dag,
                             Solution sol, const SolutionFMConfig& cfg = {});