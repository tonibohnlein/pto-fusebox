#pragma once

#include "partition/partition.h"
#include <set>
#include <vector>

// ============================================================================
// FM move: a single-op (or merge) partition modification.
//
// Unlike the heap-based Move, this is always associated with one "initiating"
// op and represents that op's current best action.
// ============================================================================

struct FMMove {
    enum Type { NONE = -1, STEAL = 0, EJECT = 1, RECOMPUTE = 2, MERGE = 3 } type = NONE;
    size_t op = SIZE_MAX;     // initiating op
    size_t ga = SIZE_MAX;     // source group (for steal/eject/merge)
    size_t gb = SIZE_MAX;     // target group (for steal/recompute/merge)
    double saving = -1e18;    // positive = improvement

    bool valid() const { return type != NONE; }
};

// ============================================================================
// Compute the best move for a single op across all its groups.
//
// Evaluates steal, eject, recompute, merge for each (group, neighbor-group)
// pair reachable via the op's DAG edges. Returns the single best move.
//
// floor: minimum saving to consider (negative = allow worsening moves)
// locked: set of ops that cannot initiate moves (skip if op is locked)
// ============================================================================

FMMove best_move_for(const Partition& part, size_t op,
                     double floor = 0.0,
                     const std::set<size_t>& locked = {});

// ============================================================================
// Apply an FM move to the partition. Returns the set of affected group indices
// (groups whose ops or costs changed). Empty if move could not be applied.
// ============================================================================

std::set<size_t> apply_fm_move(Partition& part, const FMMove& m);
