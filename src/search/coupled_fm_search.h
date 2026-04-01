#pragma once

#include "search/coupling_search.h"
#include "search/fm_search.h"
#include "util/pairing_heap.h"
#include <optional>
#include <set>
#include <vector>

// ============================================================================
// CoupledFMMove: FMMove extended with 4 coupling move types.
//
// Types 0-8 are identical to FMMove (partition moves).
// Types 9-12 are coupling moves operating on a CoupledPartition.
// ============================================================================

struct CoupledFMMove {
    enum Type {
        NONE = -1,
        // Partition moves (same semantics as FMMove::Type)
        STEAL = 0, EJECT = 1, RECOMPUTE = 2, MERGE = 3,
        INTERNAL_EJECT = 4, SPLIT = 5,
        TENSOR_MERGE = 6, TENSOR_EXTRACT = 7, DE_RECOMPUTE = 8,
        FORCE_RECOMPUTE = 9,
        // Coupling moves
        COUPLE = 10, UNCOUPLE = 11, RETAIN_FORCE_SPLIT = 12,
        // FORCE_RETAIN: t is already a boundary output of ga; split the
        // destination group at a bridge adjacent to t's consumer so the
        // smaller side_a can be coupled to ga via t.
        FORCE_RETAIN = 13
    } type = NONE;

    size_t op     = SIZE_MAX;   // initiating op (heap key; for coupling moves: any op in ga)
    size_t ga     = SIZE_MAX;   // source group
    size_t gb     = SIZE_MAX;   // target / coupled-partner group (FORCE_RETAIN: g_dst to split)
    size_t op2    = SIZE_MAX;   // second op (SPLIT: split partner; RETAIN_FORCE_SPLIT: op_b;
                                //            FORCE_RETAIN: consumer of t in g_dst → side_a)
    size_t op3    = SIZE_MAX;   // third op  (FORCE_RETAIN: split-from op in g_dst → side_b)
    size_t tensor = SIZE_MAX;   // retained tensor (COUPLE / UNCOUPLE / RETAIN_FORCE_SPLIT / FORCE_RETAIN)
    double saving = -1e18;

    // For TENSOR_MERGE / TENSOR_EXTRACT
    std::vector<size_t> tensor_groups;
    std::vector<size_t> tensor_consumer_ops;

    bool valid()             const { return type != NONE; }
    bool is_partition_move() const { return type >= 0 && type <= 9; }
    bool is_coupling_move()  const { return type >= 10 && type <= 13; }

    // Build an FMMove from this (only valid for partition move types)
    FMMove as_fm_move() const {
        FMMove fm;
        fm.type               = static_cast<FMMove::Type>(static_cast<int>(type));
        fm.op                 = op;
        fm.ga                 = ga;
        fm.gb                 = gb;
        fm.op2                = op2;
        fm.saving             = saving;
        fm.tensor_groups      = tensor_groups;
        fm.tensor_consumer_ops = tensor_consumer_ops;
        return fm;
    }
};

// ============================================================================
// best_coupled_move_for_op
//
// Evaluates all candidate moves for op on a CoupledPartition:
//   - All partition moves (STEAL, EJECT, MERGE, SPLIT, …) via best_move_for
//   - COUPLE / UNCOUPLE / RETAIN_FORCE_SPLIT / FORCE_RETAIN for tensors adjacent to op's groups
//
// Returns the single best move (highest saving).
// ============================================================================

CoupledFMMove best_coupled_move_for_op(const CoupledPartition& cp,
                                        size_t op,
                                        const std::set<size_t>& feasibly_ret,
                                        const std::set<size_t>& locked = {});

// ============================================================================
// apply_coupled_fm_move
//
// Applies a CoupledFMMove to a CoupledPartition.
// For partition moves (types 0-8):
//   - Delegates to apply_fm_move(cp.part, …)
//   - Fixes up coupling edges: proper chain transfer for MERGE, sole-consumer
//     transfer for STEAL, invalidate_couplings() for the rest.
// For coupling moves (types 9-12):
//   - Delegates to apply_couple / apply_uncouple / apply_retain_force_split /
//     apply_force_retain.
//
// Returns affected group indices (for active-set refresh).
// Empty return = move was not applied.
// ============================================================================

std::set<size_t> apply_coupled_fm_move(CoupledPartition& cp,
                                        const CoupledFMMove& m);

// ============================================================================
// CoupledActiveSet
//
// PairingHeap-backed active set for CoupledFMMove, mirroring ActiveSet
// (active_set.h) but operating on a CoupledPartition and evaluating
// coupling moves in addition to partition moves.
// ============================================================================

class CoupledActiveSet {
public:
    CoupledActiveSet(const CoupledPartition& cp,
                     const std::set<size_t>& feasibly_ret,
                     double floor = 0.0);

    // --- Activation ---
    void activate(size_t op);
    void activate_group_ops(size_t gi);
    void activate_border(size_t gi);

    // --- Selection ---
    std::optional<CoupledFMMove> pop_best();
    void lock(size_t op);
    void lock_all(const std::vector<size_t>& ops);

    // --- Update ---
    void refresh_after_move(const std::set<size_t>& affected_groups);

    // --- Queries ---
    bool is_active(size_t op) const { return heap_.contains(op); }
    bool is_locked(size_t op) const { return locked_.count(op) > 0; }

private:
    const CoupledPartition*   cp_;
    const std::set<size_t>*   feasibly_ret_;
    PairingHeap<CoupledFMMove> heap_;
    std::set<size_t>           locked_;
    double                     floor_ = 0.0;

    void recompute_and_update(size_t op);
};
