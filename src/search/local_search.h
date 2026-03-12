#pragma once

#include "partition/partition.h"
#include "search/fm_search.h"  // reuse best_move_for / apply_fm_move
#include <cmath>
#include <queue>

// ============================================================================
// OpMove: one best move per op, for the greedy descent heap.
//
// Unlike the old Move struct (one entry per (group, neighbor, move_type)
// triple), this produces at most one heap entry per op. This keeps the
// heap size O(num_ops) instead of O(ops × neighbors × move_types).
//
// Staleness is detected by storing the generation of the primary group
// at evaluation time. If the group's gen changes, the entry is stale.
// ============================================================================

struct OpMove {
    size_t op = SIZE_MAX;     // initiating op
    FMMove move;              // the best move for this op
    size_t primary_group = SIZE_MAX;
    int gen_at_eval = -1;     // gen of primary_group when evaluated

    double saving() const { return move.saving; }
    bool valid() const { return move.valid(); }

    bool operator<(const OpMove& o) const {
        if (std::abs(move.saving - o.move.saving) > 0.001)
            return move.saving < o.move.saving;
        return op > o.op;
    }
};

using MoveHeap = std::priority_queue<OpMove>;

// ============================================================================
// Push the best move for a single op into the heap.
// ============================================================================

void push_op_move(const Partition& part, size_t op, MoveHeap& heap);

// ============================================================================
// Greedy descent: repeatedly apply the best positive move until no
// improving move exists. Returns a local optimum.
// ============================================================================

Partition greedy_descent(Partition part);

// ============================================================================
// Full search pipeline:
//   1. Try multiple initialization strategies
//   2. Greedy descent on each
//   3. FM exploration from the best
// Returns the overall best partition found.
// ============================================================================

Partition local_search(const Problem& prob, const DAG& dag);