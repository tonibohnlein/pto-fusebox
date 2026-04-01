#pragma once

#include "search/coupling_search.h"
#include "search/fm_outer.h"
#include <set>
#include <vector>

// Forward declarations
class CostCache;
struct MerkleHashes;
namespace symm_mutations { struct PatternContext; }

// ============================================================================
// Parallel coupling evo search.
//
// Takes an already-initialized pool of CoupledPartitions (built in Phase 2)
// and runs an evolutionary evo loop with three operators:
//
//   1. Crossover: combine partition structure from two parents, rebuild coupling
//   2. Symmetry mutation: inject/align symmetric representatives
//   3. Compound mutation: random partition moves + coupling moves
//
// After mutation: coupled_fm_outer_loop (partition + coupling moves) → insert.
//
// Returns the best CoupledPartition's to_solution().
// ============================================================================

struct CouplingParallelConfig {
    int  num_threads = 0;    // 0 = hardware_concurrency
    int  pool_size   = 16;   // max CoupledPartitions in diversity pool
    bool early_stop  = true; // stop after num_threads*25 non-improving tasks
    FMOuterConfig fm;        // fm_outer_loop config for evo workers
    CostCache* cache = nullptr;  // shared cost cache (optional)

    // Symmetry context (optional): enables crossover with Merkle canonicalisation
    // and symmetry-guided mutations.  Null = disabled.
    const MerkleHashes* merkle = nullptr;
    const symm_mutations::PatternContext* symm_ctx = nullptr;
};

Solution coupling_parallel_search(
    std::vector<CoupledPartition> coupled_pool,  // initialized by Phase 2
    const std::set<size_t>&       feasibly_ret,
    CouplingTimePoint             deadline = CouplingTimePoint::max(),
    const CouplingParallelConfig& cfg      = {});
