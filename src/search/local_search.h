#pragma once

#include "partition/partition.h"
#include <cmath>
#include <map>
#include <queue>

// ============================================================================
// Move: a candidate partition modification
// ============================================================================

struct Move {
    enum Type { MERGE = 0, STEAL = 1, RECOMPUTE = 2, EJECT = 3,
                INTERNAL_EJECT = 4, SPLIT = 5 } type;
    size_t ga, gb;       // groups involved
    size_t op;           // op involved (for steal/recompute/eject/internal_eject/split)
    double saving;       // positive = improvement, negative = worsening
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
// Tabu list: tracks (op, group) pairs that cannot be reversed for TTL steps.
//
// After moving op into/out-of a group, the reverse is forbidden for ttl_
// iterations. Each tick() call decrements all TTLs and removes expired ones.
// ============================================================================

class TabuList {
public:
    explicit TabuList(int default_ttl = 7) : ttl_(default_ttl) {}

    // Mark (op, group) as tabu
    void add(size_t op, size_t group);

    // Is (op, group) currently tabu?
    bool is_tabu(size_t op, size_t group) const;

    // Advance one iteration: decrement TTLs, remove expired
    void tick();

    size_t size() const { return entries_.size(); }

private:
    int ttl_;
    std::map<std::pair<size_t,size_t>, int> entries_;  // (op,group) -> remaining TTL
};

// ============================================================================
// Generate moves involving group gi, including negative-gain moves
// down to -floor. Filters tabu moves.
// ============================================================================

void generate_moves(const Partition& part, size_t gi, MoveHeap& heap,
                    double floor = 0.0, const TabuList* tabu = nullptr);

// ============================================================================
// Run local search from a given partition:
//   Phase 1: greedy descent (positive moves only)
//   Phase 2: tabu exploration (allows negative moves to escape local optima)
// Returns the best partition seen during the search.
// ============================================================================

Partition local_search_from(Partition part);

// Greedy-only descent (no tabu). Faster than local_search_from.
Partition greedy_descent(Partition part);

// ============================================================================
// Multi-start: run local_search_from each initialization, return best.
// ============================================================================

Partition local_search(const Problem& prob, const DAG& dag);