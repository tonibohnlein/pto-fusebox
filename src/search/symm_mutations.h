#pragma once

#include "partition/partition.h"
#include "symmetry/merkle_hash.h"
#include "symmetry/symmetry.h"
#include "symmetry/series.h"
#include <optional>
#include <random>
#include <vector>

struct ScheduleStep;  // forward decl (solution/solution.h)

// ============================================================================
// Symmetry-guided partition mutations.
//
// Two mutations that exploit discovered symmetry patterns:
//
// 1. inject_representative_solution:
//    Takes a cached local solution (from symm_init) and surgically injects
//    it into an existing partition by extracting symmetric ops from their
//    current groups and creating new groups matching the local solution.
//
// 2. align_symmetric_reps:
//    Picks a random donor representative from a symmetry class and
//    propagates its current grouping config to all other representatives
//    in the class.
//
// Both mutations produce valid partitions that may have boundary damage
// from the extraction. The caller should run greedy + FM after mutation.
//
// Usage:
//   auto child = align_symmetric_reps(partition, patterns, prob, dag, merkle, rng);
//   if (child) { child = greedy_descent(*child); /* → FM → pool */ }
// ============================================================================

namespace symm_mutations {

// Cached representative solution: the grouping found by symm_init
// for a pattern's representative component.
struct RepSolution {
    std::vector<std::set<size_t>> groups;  // groups of ops in rep component
};

// Precomputed pattern context: pattern + bijections + cached solutions.
// Built once after symm_init, passed to mutations.
struct PatternContext {
    // Parallel patterns
    std::vector<SymmetricPattern> parallel;
    std::vector<std::vector<RepSolution>> parallel_solutions;  // [pattern][solution_variant]

    // Series patterns
    std::vector<SeriesPattern> series;
    std::vector<std::vector<RepSolution>> series_solutions;

    // Shared
    MerkleHashes merkle;

    bool empty() const { return parallel.empty() && series.empty(); }
};

// Build pattern context from symm_init results.
// Call once during solver setup; pass to mutations.
PatternContext build_context(
    const Problem& prob, const DAG& dag,
    const std::vector<SymmetricPattern>& parallel,
    const std::vector<SeriesPattern>& series,
    const MerkleHashes& merkle);

// ============================================================================
// Mutation 1: Inject representative solution
//
// Takes a cached local solution and replicates it across all
// representatives in a symmetry class.
//
// Returns nullopt if no usable patterns exist.
// ============================================================================

std::optional<Partition> inject_representative_solution(
    Partition part,
    const PatternContext& ctx,
    const Problem& prob, const DAG& dag,
    std::mt19937& rng);

// ============================================================================
// Mutation 2: Cross-representative alignment
//
// Picks a random donor representative, extracts its current grouping
// from the partition, and replicates it to all other representatives
// in the same symmetry class.
//
// Returns nullopt if no usable patterns exist.
// ============================================================================

std::optional<Partition> align_symmetric_reps(
    Partition part,
    const PatternContext& ctx,
    const Problem& prob, const DAG& dag,
    std::mt19937& rng);

// ============================================================================
// Utility: extract the current grouping config for a set of ops from a
// partition.  Returns multi-op groups whose ops are a subset of `ops`.
// Used to build RepSolution from gen0 results.
// ============================================================================

std::vector<std::set<size_t>> extract_config_from_partition(
    const Partition& part, const std::set<size_t>& ops);

// Extract grouping config from solution steps (for solution-level mutations)
std::vector<std::set<size_t>> extract_config_from_steps(
    const std::vector<ScheduleStep>& steps, const std::set<size_t>& ops);

// ============================================================================
// Solution-level variants
//
// Operate on vector<ScheduleStep> instead of Partition. Same logic:
// extract symmetric ops from current steps, inject mapped configuration.
// New steps are topologically ordered via DAG edges.
// ============================================================================

std::optional<std::vector<ScheduleStep>> inject_representative_solution_steps(
    std::vector<ScheduleStep> steps,
    const PatternContext& ctx,
    const Problem& prob, const DAG& dag,
    std::mt19937& rng);

std::optional<std::vector<ScheduleStep>> align_symmetric_reps_steps(
    std::vector<ScheduleStep> steps,
    const PatternContext& ctx,
    const Problem& prob, const DAG& dag,
    std::mt19937& rng);

} // namespace symm_mutations