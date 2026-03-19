#pragma once

#include "partition/partition.h"
#include "search/fm_outer.h"

// Forward declarations for symmetry patterns
struct SymmetricPattern;
struct SeriesPattern;

// ============================================================================
// Parallel multi-start search.
//
// Spawns multiple threads, each running a different (initialization, seed)
// combination through greedy local search + FM refinement.
// Returns the full diverse pool of partitions (sorted best-first).
//
// Thread-safe by construction: each thread works on its own Partition copy.
// The Problem and DAG are read-only shared state.
// ============================================================================

struct ParallelConfig {
    int num_threads = 0;     // 0 = auto-detect (hardware_concurrency)
    int tasks_per_init = 0;  // 0 = auto (~1 task per thread)
    int pool_size = 8;       // max partitions to keep in pool
    FMOuterConfig fm;        // FM settings propagated to every task
    bool early_stop = true;  // stop after 25 non-improving generations
                             // set false when Phase 1 owns the full budget

    // Output: discovered symmetry patterns (populated by parallel_search).
    // Caller can use these for symmetry-aware mutations in later phases.
    // If null, patterns are still used internally but not exported.
    std::vector<SymmetricPattern>* out_parallel_patterns = nullptr;
    std::vector<SeriesPattern>* out_series_patterns = nullptr;
};

std::vector<Partition> parallel_search(const Problem& prob, const DAG& dag,
                                       const ParallelConfig& cfg = {});