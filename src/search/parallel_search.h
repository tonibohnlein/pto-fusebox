#pragma once

#include "partition/partition.h"
#include "search/fm_outer.h"

// ============================================================================
// Parallel multi-start search.
//
// Spawns multiple threads, each running a different (initialization, seed)
// combination through greedy local search + FM refinement. Returns the best
// partition found across all threads.
//
// Thread-safe by construction: each thread works on its own Partition copy.
// The Problem and DAG are read-only shared state.
// ============================================================================

struct ParallelConfig {
    int num_threads = 0;     // 0 = auto-detect (hardware_concurrency)
    int tasks_per_init = 0;  // 0 = auto (~1 task per thread)
    FMOuterConfig fm;        // FM settings propagated to every task
    double time_budget = 0;  // 0 = no limit. If set, scales FM passes to fit.
};

// Scale FM config to fit within a time budget for a given problem size.
// Heuristic: each FM pass costs roughly O(n_ops * n_groups) eval_set calls.
inline void adapt_fm_budget(ParallelConfig& cfg, size_t n_ops) {
    if (n_ops <= 20) {
        // Small: FM is cheap, be generous
        cfg.fm.max_passes = 50;
        cfg.fm.max_no_improve = 15;
    } else if (n_ops <= 50) {
        // Medium
        cfg.fm.max_passes = 30;
        cfg.fm.max_no_improve = 10;
    } else if (n_ops <= 100) {
        // Large: each pass is expensive
        cfg.fm.max_passes = 15;
        cfg.fm.max_no_improve = 6;
    } else {
        // Very large: minimal FM
        cfg.fm.max_passes = 8;
        cfg.fm.max_no_improve = 4;
    }
}

Partition parallel_search(const Problem& prob, const DAG& dag,
                          const ParallelConfig& cfg = {});