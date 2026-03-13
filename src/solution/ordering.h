#pragma once

#include "partition/partition.h"
#include <random>
#include <set>
#include <vector>

// ============================================================================
// Ordering algorithms: Partition → execution order + retention decisions.
//
// Both algorithms work directly on a Partition that has been fully built
// (rebuild_group_dag called, Group::sg cached).  They produce an
// OrderingResult — a topological ordering of alive groups plus a per-step
// retain set — which Solution::from_partition converts into ScheduleSteps.
//
// DFS ordering:   fast, deterministic, O(n log n).
//                 Greedily prefers successors of the last scheduled group and
//                 breaks ties by retainable boundary tensor size.
//
// Beam ordering:  slower but higher quality, O(n * beam_width * n).
//                 Tracks fast-memory residency through the schedule and
//                 selects the globally cheapest state at each step.
//
// Random ordering: randomised retention choices + ordering, for diversity.
//                  Takes a seeded mt19937 so the caller controls reproducibility.
// ============================================================================

struct OrderingResult {
    std::vector<size_t>             order;            // group indices in execution order
    std::vector<std::set<size_t>>   retain_per_step;  // what each step should retain
    double                          total_latency = 0;
};

// Fast greedy DFS ordering.
OrderingResult dfs_ordering(const Partition& part);

// Beam search ordering with explicit residency tracking.
OrderingResult beam_search_ordering(const Partition& part, int beam_width);

// Randomised ordering for solution-pool diversity.
// feasibly_ret: tensors that can physically be retained (subset of
//   prob.retainable_tensors filtered by working-set feasibility).
OrderingResult random_ordering(const Partition& part,
                               const std::set<size_t>& feasibly_ret,
                               std::mt19937& rng);