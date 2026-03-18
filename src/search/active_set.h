#pragma once

#include "search/fm_search.h"
#include "util/pairing_heap.h"
#include <set>
#include <optional>

// ============================================================================
// ActiveSet: PairingHeap-backed container of (op, best_move) for FM search.
//
// Supports:
//   - Activate: compute op's best move, insert/update in heap
//   - Pop best: O(log N) extract-max of unlocked ops
//   - Lock: prevent op from initiating further moves
//   - Refresh: recompute moves for ops in affected groups
// ============================================================================

class ActiveSet {
public:
    ActiveSet(const Partition& part, double floor);

    // --- Activation ---
    void activate(size_t op);
    void activate_group_ops(size_t gi);
    void activate_border(size_t gi);

    // --- Selection ---
    std::optional<FMMove> pop_best();
    void lock(size_t op);
    void lock_all(const std::vector<size_t>& ops);

    // --- Update ---
    void refresh_after_move(const std::set<size_t>& affected_groups);

    // --- Queries ---
    bool is_active(size_t op) const { return heap_.contains(op); }
    bool is_locked(size_t op) const { return locked_.count(op) > 0; }
    size_t num_active() const;
    const std::set<size_t>& locked_ops() const { return locked_; }

private:
    const Partition* part_;
    double floor_;
    PairingHeap<FMMove> heap_;
    std::set<size_t> locked_;

    void recompute_and_update(size_t op);
};