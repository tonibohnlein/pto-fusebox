#pragma once
#include "symmetry/merkle_hash.h"

#include "partition/partition.h"
#include "search/coupling_search.h"
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

// Random eject: pick a random op from a random group, eject it as a singleton.
// If this disconnects the group, the components become separate groups.
Partition mutate_eject(Partition part, std::mt19937& rng);

// Tensor-centric merge: pick a random tensor with ≥2 consumer ops in different
// groups, merge all consumer groups (+ optionally the producer group).
// If full merge is infeasible, try extracting just the consumer ops.
Partition mutate_tensor_merge(Partition part, std::mt19937& rng);

// Remove a recomputation group where every op is covered by another group.
Partition mutate_de_recompute(Partition part, std::mt19937& rng);

// Compound mutation: apply N random mutations (mix of all types).
Partition mutate_compound(Partition part, int num_mutations, std::mt19937& rng);

// Coupled compound mutation: same mutations applied to cp.part, then
// invalidate_couplings() to remove any edges broken by the mutation.
CoupledPartition mutate_compound_coupled(CoupledPartition cp,
                                          int num_mutations,
                                          std::mt19937& rng);

// --- Crossover ---

// Agreement-based crossover (Sanders & Schulz style):
// Ops that are in the same group in BOTH parents stay together.
// Ops that disagree are reassigned greedily.
Partition crossover(const Partition& parent_a, const Partition& parent_b,
                    std::mt19937& rng,
                    const MerkleHashes* mh = nullptr);