#include "search/active_set.h"

ActiveSet::ActiveSet(const Partition& part, double floor)
    : part_(&part), floor_(floor) {}

// ============================================================================
// Activation
// ============================================================================

void ActiveSet::activate(size_t op) {
    if (active_ops_.count(op) || locked_.count(op)) return;

    auto move = best_move_for(*part_, op, floor_, locked_);
    // Add even if move is invalid — it might become valid after updates
    active_ops_.insert(op);
    entries_.push_back({op, move});
}

void ActiveSet::activate_border(size_t gi) {
    if (!part_->groups[gi].alive) return;
    for (auto op : part_->border_ops(gi))
        activate(op);
}

void ActiveSet::activate_group_ops(size_t gi) {
    if (!part_->groups[gi].alive) return;
    // Activate border ops (for merge/steal/recompute/eject)
    for (auto op : part_->border_ops(gi))
        activate(op);
    // Also activate internal ops (for internal_eject/split) if group size is 3-15
    if (part_->groups[gi].ops.size() >= 3 && part_->groups[gi].ops.size() <= 15) {
        for (auto op : part_->internal_ops(gi))
            activate(op);
    }
}

void ActiveSet::activate_neighbors_of(const std::set<size_t>& affected_groups) {
    // Activate all ops of affected groups and their neighbors
    for (auto gi : affected_groups) {
        if (!part_->groups[gi].alive) continue;
        activate_group_ops(gi);
        auto adj = part_->adjacent_groups(gi);
        for (auto gj : adj)
            activate_group_ops(gj);
    }
}

// ============================================================================
// Selection
// ============================================================================

std::optional<FMMove> ActiveSet::pop_best() {
    int best_idx = -1;
    double best_saving = -1e18;

    for (size_t i = 0; i < entries_.size(); i++) {
        auto& e = entries_[i];
        if (locked_.count(e.op)) continue;
        if (!e.move.valid()) continue;
        if (e.move.saving > best_saving) {
            best_saving = e.move.saving;
            best_idx = (int)i;
        }
    }

    if (best_idx < 0) return std::nullopt;

    auto& entry = entries_[best_idx];
    FMMove result = entry.move;
    locked_.insert(entry.op);
    return result;
}

void ActiveSet::lock(size_t op) {
    locked_.insert(op);
}

// ============================================================================
// Update
// ============================================================================

std::set<size_t> ActiveSet::op_relevant_groups(size_t op) const {
    std::set<size_t> groups;
    for (auto gi : part_->groups_of(op))
        groups.insert(gi);
    // Also add groups of DAG neighbors
    for (auto pred : part_->dag->op_preds[op])
        for (auto gi : part_->groups_of(pred))
            groups.insert(gi);
    for (auto succ : part_->dag->op_succs[op])
        for (auto gi : part_->groups_of(succ))
            groups.insert(gi);
    return groups;
}

void ActiveSet::update_affected(const std::set<size_t>& affected_groups) {
    // Collect the full set of affected + adjacent groups
    std::set<size_t> relevant;
    for (auto gi : affected_groups) {
        relevant.insert(gi);
        if (part_->groups[gi].alive) {
            auto adj = part_->adjacent_groups(gi);
            relevant.insert(adj.begin(), adj.end());
        }
    }

    // Recompute best moves for active, unlocked ops that touch relevant groups
    for (auto& entry : entries_) {
        if (locked_.count(entry.op)) continue;

        // Check if this op touches any relevant group
        auto op_groups = op_relevant_groups(entry.op);
        bool touches = false;
        for (auto gi : op_groups) {
            if (relevant.count(gi)) { touches = true; break; }
        }

        if (touches) {
            entry.move = best_move_for(*part_, entry.op, floor_, locked_);
        }
    }
}

// ============================================================================
// Queries
// ============================================================================

bool ActiveSet::is_active(size_t op) const {
    return active_ops_.count(op) > 0;
}

bool ActiveSet::is_locked(size_t op) const {
    return locked_.count(op) > 0;
}

size_t ActiveSet::num_active() const {
    return active_ops_.size();
}

size_t ActiveSet::num_unlocked() const {
    size_t n = 0;
    for (auto op : active_ops_)
        if (!locked_.count(op)) n++;
    return n;
}