#pragma once

#include "partition/partition.h"
#include "core/cost_cache.h"
#include <vector>

struct SymmetricPattern;
struct SeriesPattern;

// ============================================================================
// Symmetry-aware partition initialization.
//
// Discovers parallel and series patterns in the DAG, solves a
// representative of each pattern via restricted greedy descent,
// replicates the grouping to all isomorphic copies, then runs
// global greedy descent on the assembled partition.
//
// Returns one candidate partition per pattern found.  The caller
// picks the best and feeds it into the full search pipeline.
//
// If no patterns are found, returns an empty vector (caller should
// fall back to standard init strategies).
// ============================================================================

std::vector<Partition> init_from_patterns(
    const Problem& prob, const DAG& dag, CostCache* cache = nullptr);