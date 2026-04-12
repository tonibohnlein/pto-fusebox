#pragma once

#include "partition/partition.h"
#include "solution/solution.h"
#include <chrono>
#include <map>
#include <set>
#include <vector>


// ============================================================================
// CoupledPartition: Partition + retainment coupling between adjacent groups.
//
// A coupling edge ga→gb means ga immediately precedes gb in the final schedule
// and retains a set of tensors across the boundary.  Each group has at most
// one outgoing (next_group) and one incoming (prev_group) coupling edge, so
// coupling edges form disjoint linear chains.
//
// Feasibility: coupling ga→gb requires acyclic_merge_local on both chains
// (treated as a super-node in the group DAG).  This reuses the exact same
// BFS infrastructure as the partition merge check.
//
// Converting to a Solution: topological sort of chains in the group DAG →
// walk each chain head-to-tail → emit ScheduleSteps with retain_these set
// from the coupling retained map.
// ============================================================================

struct CoupledPartition {
    Partition part;

    // next_group[g] = h  ↔  g immediately precedes h (coupling edge g→h).
    // SIZE_MAX = no coupling in that direction.
    std::vector<size_t> next_group;
    std::vector<size_t> prev_group;

    // Tensors retained across each coupling edge (ga → gb).
    std::map<std::pair<size_t,size_t>, FlatSet<size_t>> retained;

    // --- Construction ---

    // Initialize from a partition. Calls finalize(cache) internally if
    // groups do not yet have subgraphs built (.sg is nullopt).
    // All coupling arrays start empty (SIZE_MAX).
    void init_from(Partition p, class CostCache* cache = nullptr);

    // Remove coupling edges that have become invalid after partition mutations:
    // group died, or retained tensors are no longer boundary outputs of ga.
    void invalidate_couplings();

    // After a mutation that may have introduced new group DAG edges, remove any
    // coupling links whose chains now form a cycle in the chain-level DAG.
    // (eval_couple prevents new cycles, but mutations can re-create them without
    // going through eval_couple.)
    void fix_chain_couplings();

    // --- Chain helpers ---

    size_t chain_head(size_t g) const;           // follow prev_group to start
    size_t chain_head_cached(size_t g) const;    // cached version (lazy, invalidated on coupling change)
    size_t chain_tail(size_t g) const;           // follow next_group to end
    std::vector<size_t> chain_of(size_t g) const; // head-to-tail group list
    void invalidate_chain_cache() const {
        chain_head_cache_.clear();
        coupled_gdag_built_ = false;
    }

private:
    mutable std::vector<size_t> chain_head_cache_;
public:

    // Tensors entering group g from its chain predecessor's retained set.
    const FlatSet<size_t>& entering_for(size_t g) const;
    // Tensors group g must retain for its chain successor.
    const FlatSet<size_t>& retain_for(size_t g) const;

    // Check whether a specific tensor is retained across any coupling edge.
    bool is_retained(size_t t) const {
        for (auto& [edge, tensors] : retained)
            if (tensors.count(t)) return true;
        return false;
    }

    // --- Coupled GroupDAG ---
    // Group adjacency (tensor edges) + coupling edges (next_group).
    // Used for fast cycle detection, replacing acyclic_chain_merge BFS.
    GroupDAG& coupled_dag();
    const GroupDAG& coupled_dag() const;
    void rebuild_coupled_dag();

private:
    mutable GroupDAG coupled_gdag_;
    mutable bool coupled_gdag_built_ = false;
public:

    // --- Cost ---

    // Effective latency of group g, accounting for entering and retained tensors.
    double group_cost(size_t g) const;
    double total_cost() const;

    // --- Conversion to output ---

    Solution to_solution() const;
};

// ============================================================================
// Coupling move evaluation
//
// COUPLE(ga, gb, t):  ga and gb are currently free endpoints (ga has no
//   outgoing coupling, gb has no incoming coupling).  Adding retained tensor t
//   from ga to gb forms or extends a chain.
//
// UNCOUPLE(ga, gb, t):  remove tensor t from retained[(ga,gb)].
//   If the retained set becomes empty the coupling edge is dissolved.
// ============================================================================

struct CouplingEvalResult {
    bool   feasible = false;
    double saving   = 0.0;
};

CouplingEvalResult eval_couple(const CoupledPartition& cp,
                                size_t ga, size_t gb, size_t t);

CouplingEvalResult eval_uncouple(const CoupledPartition& cp,
                                  size_t ga, size_t gb, size_t t);

// ============================================================================
// RETAIN_FORCE_SPLIT: split group g at edge (op_a, op_b) so that tensor t
// becomes a boundary output of side_a (op_a's side) and a boundary input of
// side_b (op_b's side), then immediately couple side_a → side_b via t.
//
// op_a must be the producer of t; op_b must be a consumer of t; both must
// currently reside in group g.  t must NOT already be a boundary output of g
// (otherwise COUPLE alone suffices).
//
// This unblocks coupling for tensors that are currently internal to a group.
// The split is only attempted when (op_a, op_b) is a bridge edge in g.
// ============================================================================

CouplingEvalResult eval_retain_force_split(const CoupledPartition& cp,
                                            size_t g,
                                            size_t op_a, size_t op_b,
                                            size_t t);

// Apply: split g and couple the two halves via t.
// Returns affected group indices, or {} on failure.
// After a successful apply: call cp.part.rebuild_index() + rebuild_group_dag().
FlatSet<size_t> apply_retain_force_split(CoupledPartition& cp,
                                           size_t g,
                                           size_t op_a, size_t op_b,
                                           size_t t);

// ============================================================================
// FORCE_RETAIN: t is already a boundary output of ga.  Couple ga → side_a
// via t, where side_a is obtained by splitting the destination group g_dst
// at bridge (op_a_dst, op_b_dst).
//
// Motivation: COUPLE(ga, g_dst) may be infeasible because g_dst is too large
// to hold t in fast memory.  Splitting g_dst isolates the consumer of t into
// a smaller side_a, making the coupling feasible.
//
// Constraints (same as COUPLE):
//   ga must be a free chain tail (no next_group).
//   g_dst must be a free chain head (no prev_group).
//   acyclic_chain_merge(chain_of(ga), chain_of(g_dst)) must hold.
//
// op_a_dst: consumer of t inside g_dst — ends up in side_a (coupled to ga).
// op_b_dst: neighbor of op_a_dst in g_dst — ends up in side_b (standalone).
// ============================================================================

CouplingEvalResult eval_force_retain(const CoupledPartition& cp,
                                      size_t ga, size_t g_dst,
                                      size_t op_a_dst, size_t op_b_dst,
                                      size_t t);

// Apply: split g_dst, couple ga → side_a via t, side_b inherits g_dst's
// old outgoing chain link (if any).
// Returns affected group indices (including ga), or {} on failure.
FlatSet<size_t> apply_force_retain(CoupledPartition& cp,
                                     size_t ga, size_t g_dst,
                                     size_t op_a_dst, size_t op_b_dst,
                                     size_t t);

// ============================================================================
// EPHEMERAL_FUSE: extract producer P (= op_p) and consumer C1 (= op_c1) of
// tensor t from their current groups into a new group {P, C1} where t is
// ephemeral, then couple t to consumer C2's group (g_c2).
//
// t must have ≥2 consumers. P and C1 are extracted from their source groups
// (which may be the same or different groups). The source groups' remainders
// are re-evaluated; empty groups are killed.
//
// After the move:
//   new_group = {P, C1}  (t ephemeral inside)
//   new_group → g_c2 via retained {t}
//   C2 in g_c2 gets t from fast memory (no slow memory round-trip)
// ============================================================================

CouplingEvalResult eval_ephemeral_fuse(const CoupledPartition& cp,
                                        size_t op_p, size_t op_c1,
                                        size_t g_c2, size_t t);

FlatSet<size_t> apply_ephemeral_fuse(CoupledPartition& cp,
                                       size_t op_p, size_t op_c1,
                                       size_t g_c2, size_t t);

// ============================================================================
// Chain-level acyclicity check for partition moves on a CoupledPartition.
//
// Returns true if treating all chains containing the given groups as a single
// merged super-node does NOT create a cycle in the chain-level DAG.
// Used by best_coupled_move_for_op to filter MERGE / TENSOR_MERGE moves that
// would create chain-level cycles even when acyclic_merge_local passes.
// ============================================================================

bool acyclic_chain_merge_groups(const CoupledPartition& cp,
                                  const std::vector<size_t>& groups);

// ============================================================================
// Coupling move application — returns affected group indices.
// Callers must have verified feasibility via eval_* first.
// ============================================================================

FlatSet<size_t> apply_couple(CoupledPartition& cp,
                               size_t ga, size_t gb, size_t t);

FlatSet<size_t> apply_uncouple(CoupledPartition& cp,
                                 size_t ga, size_t gb, size_t t);

// ============================================================================
// Main search entry point
//
// Takes a finalized Partition, adds and removes coupling edges via greedy
// descent, returns the best Solution found before the deadline.
//
// feasibly_ret: tensors that can physically be retained (from
//   compute_feasibly_retainable).  Only tensors in this set are considered
//   for coupling.
// ============================================================================

using CouplingTimePoint = std::chrono::steady_clock::time_point;

// Run greedy descent on an already-initialized CoupledPartition in-place.
// Returns final total_cost(). Used by the parallel search workers.
double coupling_greedy_descent(CoupledPartition& cp,
                                const FlatSet<size_t>& feasibly_ret,
                                CouplingTimePoint deadline = CouplingTimePoint::max());

Solution coupling_search(const Problem& prob, const DAG& dag,
                          Partition part,
                          const FlatSet<size_t>& feasibly_ret,
                          CouplingTimePoint deadline = CouplingTimePoint::max());
