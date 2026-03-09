#pragma once

#include "solution/solution.h"
#include <chrono>
#include <set>

// ============================================================================
// Solution-level FM search
//
// Mirrors the partition FM architecture:
//   Inner pass: active set + locking + drift control
//   Outer loop: repeated passes + greedy-kick + adaptive heat/cooling
//
// Move vocabulary: STEAL, SPLIT, MERGE, RETAIN_ADD, RETAIN_REMOVE
// ============================================================================

struct SolutionMove {
    enum Type { NONE=-1, STEAL=0, SPLIT=1, MERGE=2, RETAIN_ADD=3, RETAIN_REMOVE=4 };
    Type type = NONE;
    size_t step_a = SIZE_MAX;
    size_t step_b = SIZE_MAX;
    size_t op = SIZE_MAX;
    size_t op2 = SIZE_MAX;
    size_t tensor = SIZE_MAX;
    double saving = -1e18;
    bool valid() const { return type != NONE; }
};

struct SolutionFMPassConfig {
    double floor_fraction = 0.30;
    double max_drift_fraction = 0.50;
    unsigned seed = 42;
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

// One FM pass on a solution (active set + locking + drift)
SolutionFMPassResult solution_fm_pass(const Problem& prob, const DAG& dag,
                                       std::vector<ScheduleStep> steps,
                                       const SolutionFMPassConfig& cfg = {});

// Greedy hill climb: solution FM pass with floor=0, no locking
std::vector<ScheduleStep> solution_greedy_descent(const Problem& prob, const DAG& dag,
                                                    std::vector<ScheduleStep> steps);

// Full outer loop: passes + greedy-kick + cooling
Solution solution_fm_search(const Problem& prob, const DAG& dag,
                             Solution sol, const SolutionFMConfig& cfg = {});