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
        cfg.fm.max_passes = 50;
        cfg.fm.max_no_improve = 15;
        cfg.fm.pass_config.floor_fraction = 0.30;
        cfg.fm.pass_config.max_drift_fraction = 0.50;
    } else if (n_ops <= 40) {
        cfg.fm.max_passes = 30;
        cfg.fm.max_no_improve = 10;
        cfg.fm.pass_config.floor_fraction = 0.30;
        cfg.fm.pass_config.max_drift_fraction = 0.40;
    } else if (n_ops <= 70) {
        cfg.fm.max_passes = 3;
        cfg.fm.max_no_improve = 2;
        cfg.fm.pass_config.floor_fraction = 0.20;
        cfg.fm.pass_config.max_drift_fraction = 0.20;
    } else {
        cfg.fm.max_passes = 5;
        cfg.fm.max_no_improve = 3;
        cfg.fm.pass_config.floor_fraction = 0.15;
        cfg.fm.pass_config.max_drift_fraction = 0.15;
    }
}

Partition parallel_search(const Problem& prob, const DAG& dag,
                          const ParallelConfig& cfg = {});