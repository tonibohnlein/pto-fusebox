#include "search/fm_pass.h"
#include <algorithm>
#include <iostream>

// ============================================================================
// Collect all border ops across all alive groups
// ============================================================================

static std::vector<size_t> all_border_ops(const Partition& part) {
    std::set<size_t> border_set;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        for (auto op : part.border_ops(gi))
            border_set.insert(op);
    }
    return {border_set.begin(), border_set.end()};
}

// ============================================================================
// Select N random border ops
// ============================================================================

static std::vector<size_t> random_subset_n(const std::vector<size_t>& ops,
                                            int count, unsigned seed) {
    if (count >= (int)ops.size() || ops.size() <= 3) return ops;

    std::vector<size_t> shuffled = ops;
    std::mt19937 rng(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);

    size_t n = std::max<size_t>(1, (size_t)count);
    shuffled.resize(n);
    return shuffled;
}

// ============================================================================
// Lock ops involved in a merge: the initiating op + the boundary ops from
// the partner group that connect to the initiating op.
// ============================================================================

static std::vector<size_t> merge_lock_ops(const Partition& part, const FMMove& m) {
    std::vector<size_t> to_lock = {m.op};

    // Find ops in gb that are DAG neighbors of m.op
    for (auto pred : part.dag->op_preds[m.op]) {
        if (m.gb < part.groups.size() && part.groups[m.gb].alive &&
            part.groups[m.gb].ops.count(pred))
            to_lock.push_back(pred);
    }
    for (auto succ : part.dag->op_succs[m.op]) {
        if (m.gb < part.groups.size() && part.groups[m.gb].alive &&
            part.groups[m.gb].ops.count(succ))
            to_lock.push_back(succ);
    }
    return to_lock;
}

// ============================================================================
// FM inner pass
// ============================================================================

FMPassResult fm_inner_pass(Partition part, const FMConfig& cfg) {
    FMPassResult result;
    result.start_cost = part.total_cost();
    result.best_cost = result.start_cost;
    result.best_partition = part;  // initial snapshot

    double floor = result.start_cost * cfg.floor_fraction;
    double max_drift = result.start_cost * cfg.max_drift_fraction;

    // Step 1: activate random subset of border ops
    auto borders = all_border_ops(part);
    auto initial = random_subset_n(borders, cfg.init_count, cfg.seed);

    ActiveSet active(part, floor);
    for (auto op : initial)
        active.activate(op);

    // Step 2: main loop
    double cumulative_gain = 0;
    double best_cumulative_gain = 0;

    while (true) {
        // Pop the best unlocked move
        auto move_opt = active.pop_best();
        if (!move_opt.has_value()) break;

        FMMove move = *move_opt;

        // Additional locking for merge: lock boundary ops from partner group
        // (must do BEFORE apply, since apply changes the group)
        std::vector<size_t> extra_locks;
        if (move.type == FMMove::MERGE) {
            extra_locks = merge_lock_ops(part, move);
        }

        // Apply the move
        auto affected = apply_fm_move(part, move);
        if (affected.empty()) {
            // Move failed on re-verify — just continue
            continue;
        }

        result.moves_applied++;
        if (move.saving > 0.001)
            result.moves_positive++;
        else if (move.saving < -0.001)
            result.moves_negative++;

        // Update cumulative gain
        double new_cost = part.total_cost();
        cumulative_gain = result.start_cost - new_cost;

        // Check for new best in this pass
        if (new_cost < result.best_cost - 0.001) {
            result.best_cost = new_cost;
            result.best_partition = part;  // snapshot
            best_cumulative_gain = cumulative_gain;
        }

        // Check max drift: if we've dropped too far below the pass-best, abort
        if (best_cumulative_gain - cumulative_gain > max_drift) break;

        // Lock extra ops (merge partners)
        // active.pop_best() already locked move.op.
        // For merge: also lock the boundary ops from the partner group.
        for (auto op : extra_locks)
            active.lock(op);

        // Step 3: Update existing active ops affected by the move
        active.update_affected(affected);

        // Step 4: Activate new border ops of affected and adjacent groups
        active.activate_neighbors_of(affected);
    }

    return result;
}