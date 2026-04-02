#pragma once

#include "partition/partition.h"
#include <set>

// ============================================================================
// partition_moves: canonical evaluation and application of partition moves.
//
// Convention: all functions use a uniform naming for groups:
//   STEAL:  op moves FROM `from` INTO `to`
//   MERGE:  `ga` survives, `gb` is killed (absorbed into ga)
//
// eval_* functions are pure — no mutation, return costs and saving.
// apply_* functions validate + mutate. Return affected group indices,
// or empty set on validation failure (no mutation occurred).
//
// Search files (fm_search, local_search, evolution) call these functions.
// They NEVER implement move logic inline.
// ============================================================================

namespace partition_moves {

struct EvalResult {
    bool feasible = false;
    double saving = 0.0;
};

// ============================================================================
// STEAL: move a single op from one group to another
//
// Validates:
//   - `to` group can absorb op (eval_set < 1e17)
//   - `from` group remains feasible after losing op (eval_set < 1e17 or empty)
//   - If `from` becomes empty, op must exist in another alive group
//
// Does NOT check acyclicity (caller uses cheap pre-filter at eval time,
// apply_steal runs full Kahn's).
// ============================================================================

EvalResult eval_steal(const Partition& p, size_t op, size_t from, size_t to);

// Apply: Kahn's check → mutate → return affected groups.
// Returns {} if Kahn's fails or cost re-evaluation fails.
// When both precomputed costs are >= 0, skip re-evaluation and use them directly.
FlatSet<size_t> apply_steal(Partition& p, size_t op, size_t from, size_t to,
                              double precomputed_from_cost = -1.0,
                              double precomputed_to_cost = -1.0);

// ============================================================================
// MERGE: fuse two groups (gb absorbed into ga, gb killed)
//
// Validates:
//   - merged set is feasible (eval_set < 1e17)
//
// Does NOT check acyclicity (caller uses merge_creates_cycle at eval time,
// apply_merge runs full Kahn's).
// ============================================================================

EvalResult eval_merge(const Partition& p, size_t ga, size_t gb);

// Apply: Kahn's check → mutate → return affected groups.
// Returns {} if Kahn's fails or cost re-evaluation fails.
// When precomputed_cost >= 0, skip re-evaluation and use that value directly.
FlatSet<size_t> apply_merge(Partition& p, size_t ga, size_t gb, double precomputed_cost = -1.0);

// ============================================================================
// EJECT: remove a single op from a group, creating a singleton + remainder(s)
//
// eval_eject lives on Partition (returns EjectResult with remainder components).
// We provide only apply_eject here as the canonical mutation path.
//
// Validates:
//   - Ephemeral gap check (eject never creates a cycle — singleton always fits)
//   - Re-evaluates eject to get current remainder components
//
// Does NOT check feasibility (caller already called part.eval_eject).
// ============================================================================

FlatSet<size_t> apply_eject(Partition& p, size_t op, size_t ga,
                              const Partition::EjectResult* precomputed = nullptr);

// ============================================================================
// SPLIT: split a group at a bridge edge into two groups
//
// eval_split lives on Partition (returns SplitResult with side_a/side_b).
// We provide only apply_split here as the canonical mutation path.
//
// Validates:
//   - Acyclicity via acyclic_split_local (eval-time, local)
//   - Re-evaluates split to get current sides
// ============================================================================

FlatSet<size_t> apply_split(Partition& p, size_t op_a, size_t op_b, size_t ga,
                              const Partition::SplitResult* precomputed = nullptr);

// ============================================================================
// DE_RECOMPUTE: remove a recomputed op from a group.
// The op must exist in at least one other alive group. The source group
// survives with the remaining ops (or dies if op was the only one).
//
// Validates:
//   - Op exists in ga and in at least one other alive group
//   - No ephemeral gap (output tensors still available as boundary outputs)
//
// Does NOT check acyclicity (caller's responsibility — call
// acyclic_de_recompute_local before eval_de_recompute).
// ============================================================================

// Eval: check if op can be removed and compute saving.
EvalResult eval_de_recompute(const Partition& p, size_t ga, size_t op);

// Apply: validate + remove op from group. Returns affected groups.
// When precomputed_cost >= 0, skip re-evaluation and use that value directly.
FlatSet<size_t> apply_de_recompute(Partition& p, size_t ga, size_t op,
                                     double precomputed_cost = -1.0);

// ============================================================================
// RECOMPUTE: copy op into a second group (op stays in its original group too)
//
// Validates:
//   - Target group can absorb op (eval_set < 1e17)
//   - Acyclicity via acyclic_recompute_local (eval-time)
// ============================================================================

EvalResult eval_recompute(const Partition& p, size_t op, size_t into);

FlatSet<size_t> apply_recompute(Partition& p, size_t op, size_t into,
                                  double precomputed_cost = -1.0);

// ============================================================================
// TENSOR_MERGE: merge all groups that touch a tensor into one
//
// Validates:
//   - All groups alive
//   - Merged set feasible (eval_set < 1e17)
//   - Acyclicity via acyclic_merge_local (eval-time, in best_move_for)
// ============================================================================

EvalResult eval_tensor_merge(const Partition& p,
                              const std::vector<size_t>& group_list);

FlatSet<size_t> apply_tensor_merge(Partition& p,
                                     const std::vector<size_t>& group_list,
                                     double precomputed_cost = -1.0);

// ============================================================================
// TENSOR_EXTRACT: pull ops out of multiple groups into a new group
//
// Validates:
//   - All source groups alive
//   - Extract set and all remainders feasible
//   - Acyclicity via acyclic_extract_local (eval-time, in best_move_for)
// ============================================================================

EvalResult eval_tensor_extract(const Partition& p,
                                const FlatSet<size_t>& extract_ops,
                                const std::vector<size_t>& source_groups);

FlatSet<size_t> apply_tensor_extract(Partition& p,
                                       const FlatSet<size_t>& extract_ops,
                                       const std::vector<size_t>& source_groups,
                                       double precomputed_extract_cost = -1.0);

// ============================================================================
// TENSOR_EXTRACT_SPLIT: when full extract fails (too many ops for one group),
// split consumer ops into k balanced sub-groups, each containing the producer
// (recomputed).  The tensor becomes ephemeral in every sub-group.
//
// Tries k = 2, 4, ... up to n (one consumer per group = force_recompute).
// Picks the k that minimises total cost.
// ============================================================================

struct SplitExtractResult {
    bool   feasible = false;
    double saving   = -1e18;
    int    prod_op  = -1;
    std::vector<FlatSet<size_t>> sub_groups;  // each contains producer + consumer subset
    std::vector<double>          sub_costs;
};

SplitExtractResult eval_tensor_extract_split(
    const Partition& p,
    size_t tensor_id,
    const std::vector<size_t>& consumer_ops,
    const std::vector<size_t>& source_groups);

FlatSet<size_t> apply_tensor_extract_split(
    Partition& p,
    const SplitExtractResult& result,
    const std::vector<size_t>& source_groups);

// FORCE_RECOMPUTE: for tensor t with producer P and consumers [C_i],
// extract P and each C_i from their current groups, create {P, C_i} pairs.
// T becomes ephemeral in each new group.
struct ForceRecomputeResult {
    bool   feasible = false;
    double saving   = -1e18;
    size_t prod_op  = SIZE_MAX;
    std::vector<size_t> consumer_ops;   // feasible consumers
    std::vector<double> pair_costs;     // cost of each {P, C_i}
};

ForceRecomputeResult eval_force_recompute(const Partition& p, size_t tensor_id);

FlatSet<size_t> apply_force_recompute(Partition& p, size_t tensor_id,
                                        const ForceRecomputeResult& eval);

} // namespace partition_moves