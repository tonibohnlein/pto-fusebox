#pragma once

#include "solution/solution.h"
#include "search/fm_outer.h"   // provides SteadyClock + TimePoint

// ============================================================================
// Top-level solver entry point.
//
// Pipeline (unchanged from pre-refactor):
//
//   Phase 1 (35% budget, or 95% if no retainable tensors):
//     parallel_search — per-strategy greedy+FM gen0, then evolutionary
//     mutation/crossover generations until deadline.
//
//   Phase 2 (5% budget):
//     Build solution pool from every partition in the Phase 1 pool.
//     For each partition: DFS ordering + beam search ordering (via
//     Solution::from_partition), plus a random-retain variant when tensors
//     are retainable.  The two ordering algorithms are now in ordering.h/cpp
//     and no longer inlined in solver.cpp.
//
//   Phase 3 (60% budget, skipped if no retainable tensors):
//     solution_evo_search — parallel mutation/crossover + FM polish on the
//     solution pool.
//
// deadline = TimePoint::max() → use a default 5-second internal budget.
// ============================================================================

Solution solve(const Problem& prob, const DAG& dag,
               TimePoint deadline = TimePoint::max());

// V2 pipeline: init-only seeding → evo loop (partition or coupled).
// Skips Phase 2 (ordering+greedy) and does no FM in the init phase.
Solution solve_v2(const Problem& prob, const DAG& dag,
                  TimePoint deadline = TimePoint::max());