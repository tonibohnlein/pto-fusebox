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
};

Partition parallel_search(const Problem& prob, const DAG& dag,
                          const ParallelConfig& cfg = {});