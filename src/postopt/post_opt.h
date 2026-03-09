#pragma once

#include "solution/solution.h"
#include <chrono>

// Retain optimization: try retaining tensors between consecutive steps
Solution optimize_retain(const Problem& prob, const DAG& dag, Solution sol);

// Recompute optimization: try adding predecessor ops to eliminate boundary loads
Solution optimize_recompute(const Problem& prob, const DAG& dag, Solution sol);

// Solution-level FM search: unified partition + retain moves
// (delegates to solution_search.h)
Solution optimize_solution(const Problem& prob, const DAG& dag, Solution sol,
                            std::chrono::steady_clock::time_point deadline 
                            = std::chrono::steady_clock::time_point::max());