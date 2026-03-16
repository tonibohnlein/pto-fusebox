#pragma once

#include "partition/partition.h"
#include <cmath>
#include <queue>

// ============================================================================
// Move: a candidate partition modification for greedy descent
// ============================================================================

struct Move {
    enum Type { MERGE = 0, STEAL = 1, RECOMPUTE = 2, EJECT = 3,
                INTERNAL_EJECT = 4, SPLIT = 5, DE_RECOMPUTE = 6 } type;
    size_t ga, gb;       // groups involved
    size_t op;           // op involved (for steal/recompute/eject/internal_eject/split)
    double saving;       // positive = improvement
    int gen_a, gen_b;    // generation of ga, gb when evaluated
    size_t op2 = 0;      // second op (for SPLIT: the neighbor)

    bool operator<(const Move& o) const {
        if (std::abs(saving - o.saving) > 0.001) return saving < o.saving;
        if (type != o.type) return type > o.type;
        return ga > o.ga;
    }
};

using MoveHeap = std::priority_queue<Move>;

// ============================================================================
// Generate moves involving group gi with saving > -floor.
// floor=0 (default) means only positive-saving moves.
// ============================================================================

void generate_moves(const Partition& part, size_t gi, MoveHeap& heap,
                    double floor = 0.0);

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

// ============================================================================
// Post-search validation (debug only)
// ============================================================================

// Quick full gap check: returns true if ANY ephemeral gap exists
// OR the group DAG has a cycle. Builds fresh Subgraphs for all groups
// and runs Kahn's algorithm on the resulting group DAG.
// O(groups * tensors) — used as post-move safety check.
bool partition_has_gap(const Partition& part);