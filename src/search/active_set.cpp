#include "search/active_set.h"
#include <iostream>
#include <queue>

ActiveSet::ActiveSet(const Partition& part, double floor, const GroupDAG* gdag)
    : part_(&part), gdag_(gdag), heap_(part.prob->num_ops()), floor_(floor) {}

// ============================================================================
// Activation
// ============================================================================

void ActiveSet::recompute_and_update(size_t op) {
    if (locked_.count(op)) return;
    auto move = best_move_for(*part_, op, locked_, gdag_);
    if (move.valid() && move.saving > -floor_)
        heap_.push_or_update(op, move);
    else
        heap_.remove(op);
}

void ActiveSet::activate(size_t op) {
    if (locked_.count(op)) return;
    recompute_and_update(op);
}

void ActiveSet::activate_group_ops(size_t gi) {
    if (!part_->groups[gi].alive) return;
    for (auto op : part_->border_ops(gi))
        recompute_and_update(op);
    if (part_->groups[gi].ops.size() >= 3 && part_->groups[gi].ops.size() <= 15) {
        for (auto op : part_->internal_ops(gi))
            recompute_and_update(op);
    }
}

void ActiveSet::activate_border(size_t gi) {
    if (!part_->groups[gi].alive) return;
    for (auto op : part_->border_ops(gi))
        recompute_and_update(op);
}

// ============================================================================
// Selection
// ============================================================================

std::optional<FMMove> ActiveSet::pop_best() {
    // Pop until we find an unlocked op or heap is empty
    while (!heap_.empty()) {
        auto m = heap_.pop_best();
        if (!m) return std::nullopt;
        if (!locked_.count(m->op)) {
            locked_.insert(m->op);
            return m;
        }
        // Locked op leaked into heap — discard and continue
    }
    return std::nullopt;
}

void ActiveSet::lock(size_t op) {
    locked_.insert(op);
    heap_.remove(op);
}

void ActiveSet::lock_all(const std::vector<size_t>& ops) {
    for (auto op : ops) {
        locked_.insert(op);
        heap_.remove(op);
    }
}

// ============================================================================
// Queries
// ============================================================================

size_t ActiveSet::num_active() const {
    size_t count = 0;
    for (size_t i = 0; i < part_->prob->num_ops(); i++)
        if (heap_.contains(i)) count++;
    return count;
}

// ============================================================================
// Update
// ============================================================================

void ActiveSet::refresh_after_move(const FlatSet<size_t>& affected_groups) {
    // Collect relevant groups: affected + their adjacents
    FlatSet<size_t> relevant;
    for (auto gi : affected_groups) {
        relevant.insert(gi);
        if (part_->groups[gi].alive) {
            auto adj = part_->adjacent_groups(gi);
            relevant.insert(adj.begin(), adj.end());
        }
    }

    // Collect all ops in relevant groups + their DAG neighbors
    FlatSet<size_t> ops_to_refresh;
    for (auto gi : relevant) {
        if (!part_->groups[gi].alive) continue;
        for (auto op : part_->groups[gi].ops) {
            ops_to_refresh.insert(op);
            for (auto nbr : part_->dag->op_neighbors[op])
                ops_to_refresh.insert(nbr);
        }
    }

    // Recompute each affected op's best move
    for (auto op : ops_to_refresh)
        recompute_and_update(op);

    // Activate new border/internal ops of relevant groups
    for (auto gi : relevant)
        activate_group_ops(gi);
}