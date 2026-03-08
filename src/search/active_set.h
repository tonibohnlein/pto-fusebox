#pragma once

#include "search/fm_search.h"
#include <vector>
#include <set>
#include <optional>

// ============================================================================
// ActiveSet: indexed container of (op, best_move) entries for FM search.
//
// Supports:
//   - Activate: add an op with its computed best move
//   - Pop best: find the unlocked op with highest saving, lock it
//   - Update: recompute best moves for ops in affected groups
//   - Activate neighbors: add new border ops from affected groups
//
// Implementation: flat vector with O(N) scan for max. N ≤ 200 ops in
// practice, so this is faster than a heap with stale-entry management.
// ============================================================================

class ActiveSet {
public:
    ActiveSet(const Partition& part, double floor);

    // --- Activation ---

    // Activate a single op: compute its best move, add to set.
    // No-op if already active or locked.
    void activate(size_t op);

    // Activate all border ops of a group.
    void activate_border(size_t gi);

    // Activate border ops of groups adjacent to the given set of groups.
    void activate_neighbors_of(const std::set<size_t>& affected_groups);

    // --- Selection ---

    // Pop the unlocked op with the highest saving. Locks it.
    // Returns nullopt if no unlocked ops remain or all have saving < -floor.
    std::optional<FMMove> pop_best();

    // Explicitly lock an op (prevents it from initiating moves).
    // Used for merge partner locking.
    void lock(size_t op);

    // --- Update ---

    // Recompute best moves for all active, unlocked ops that belong to
    // any of the given groups (or adjacent groups). Called after a move.
    void update_affected(const std::set<size_t>& affected_groups);

    // --- Queries ---

    bool is_active(size_t op) const;
    bool is_locked(size_t op) const;
    size_t num_active() const;
    size_t num_unlocked() const;
    const std::set<size_t>& locked_ops() const { return locked_; }

private:
    const Partition* part_;
    double floor_;

    struct Entry {
        size_t op;
        FMMove move;
    };

    std::vector<Entry> entries_;
    std::set<size_t> active_ops_;   // ops currently in the active set
    std::set<size_t> locked_;       // ops that have been moved (can't initiate)

    // Find groups that an op belongs to or is adjacent to
    std::set<size_t> op_relevant_groups(size_t op) const;
};
