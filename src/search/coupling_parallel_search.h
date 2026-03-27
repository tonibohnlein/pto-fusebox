#pragma once

#include "search/coupling_search.h"
#include "search/fm_outer.h"
#include <set>
#include <vector>

// Forward declaration
class CostCache;

// ============================================================================
// Parallel coupling evo search.
//
// Takes an already-initialized pool of CoupledPartitions (built in Phase 2)
// and runs an evolutionary evo loop:
//
//   select → mutate_compound(cp.part) → finalize → fm_outer_loop →
//   init_from(best_partition) → coupling_greedy_descent → insert
//
// The partition FM re-optimizes structure after mutation; coupling greedy
// then discovers coupling edges on the improved partition.
//
// Returns the best CoupledPartition's to_solution().
// ============================================================================

struct CouplingParallelConfig {
    int  num_threads = 0;    // 0 = hardware_concurrency
    int  pool_size   = 16;   // max CoupledPartitions in diversity pool
    bool early_stop  = true; // stop after num_threads*25 non-improving tasks
    FMOuterConfig fm;        // fm_outer_loop config for evo workers
    CostCache* cache = nullptr;  // shared cost cache (optional)
};

Solution coupling_parallel_search(
    std::vector<CoupledPartition> coupled_pool,  // initialized by Phase 2
    const std::set<size_t>&       feasibly_ret,
    CouplingTimePoint             deadline = CouplingTimePoint::max(),
    const CouplingParallelConfig& cfg      = {});
