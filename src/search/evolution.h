#pragma once

#include "partition/partition.h"
#include <random>
#include <vector>

// ============================================================================
// Evolutionary operators for partition optimization.
//
// Mutations: structurally random perturbations (NOT gain-guided).
// Crossover: combines structural insights from two parents.
// ============================================================================

// --- Mutations ---

// Random merge: pick two random adjacent groups, fuse them.
Partition mutate_merge(Partition part, std::mt19937& rng);

// Random split: pick a random group with ≥3 ops, cut at a random bridge edge.
Partition mutate_split(Partition part, std::mt19937& rng);

// Random reassign: pick a random border op, move to a random neighbor group.
// Unlike FM's best_move_for, this ignores cost — pure structural perturbation.
Partition mutate_reassign(Partition part, std::mt19937& rng);

// Block move: move an entire DAG chain (connected subpath) between groups.
Partition mutate_block_move(Partition part, std::mt19937& rng);

// Compound mutation: apply N random mutations (mix of all types).
Partition mutate_compound(Partition part, int num_mutations, std::mt19937& rng);

// --- Crossover ---

// Agreement-based crossover (Sanders & Schulz style):
// Ops that are in the same group in BOTH parents stay together.
// Ops that disagree are reassigned greedily.
Partition crossover(const Partition& parent_a, const Partition& parent_b,
                    std::mt19937& rng);