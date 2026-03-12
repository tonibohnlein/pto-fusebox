#include "search/verbose.h"
#include "search/local_search.h"
#include "init/init_strategies.h"
#include "search/fm_outer.h"
#include <iostream>

// ============================================================================
// Per-op move generation: compute best move for one op, push to heap
// ============================================================================

void push_op_move(const Partition& part, size_t op, MoveHeap& heap) {
    auto move = best_move_for(part, op, /*floor=*/0.0);
    if (!move.valid() || move.saving < 0.001) return;

    // Find primary group for staleness tracking
    auto& groups = part.groups_of(op);
    if (groups.empty()) return;
    size_t primary = groups[0];

    OpMove entry;
    entry.op = op;
    entry.move = move;
    entry.primary_group = primary;
    entry.gen_at_eval = part.groups[primary].gen;
    heap.push(entry);
}

// ============================================================================
// Collect all ops that need re-evaluation after affected groups change
// ============================================================================

static std::set<size_t> dirty_ops(const Partition& part,
                                   const std::set<size_t>& affected_groups) {
    std::set<size_t> ops;
    // Collect affected + adjacent groups
    std::set<size_t> relevant = affected_groups;
    for (auto gi : affected_groups) {
        if (!part.groups[gi].alive) continue;
        auto adj = part.adjacent_groups(gi);
        relevant.insert(adj.begin(), adj.end());
    }
    // All ops in relevant groups
    for (auto gi : relevant) {
        if (!part.groups[gi].alive) continue;
        for (auto op : part.groups[gi].ops)
            ops.insert(op);
    }
    return ops;
}

// ============================================================================
// Greedy descent
// ============================================================================

Partition greedy_descent(Partition part) {
    MoveHeap heap;

    // Seed: one best move per op
    for (size_t op = 0; op < part.prob->num_ops(); op++)
        push_op_move(part, op, heap);

    int applied = 0;
    while (!heap.empty()) {
        OpMove entry = heap.top();
        heap.pop();

        // Staleness check: has the primary group changed since evaluation?
        if (entry.primary_group >= part.groups.size() ||
            !part.groups[entry.primary_group].alive ||
            part.groups[entry.primary_group].gen != entry.gen_at_eval)
            continue;

        // Also check secondary group for merge/steal/recompute
        if (entry.move.type == FMMove::MERGE ||
            entry.move.type == FMMove::STEAL ||
            entry.move.type == FMMove::RECOMPUTE) {
            if (entry.move.gb >= part.groups.size() ||
                !part.groups[entry.move.gb].alive)
                continue;
        }

        // Apply the move (re-verifies internally)
        auto affected = apply_fm_move(part, entry.move);
        if (affected.empty()) continue;

        applied++;

        // Regenerate: push new best move for each affected op
        auto ops_to_update = dirty_ops(part, affected);
        for (auto op : ops_to_update)
            push_op_move(part, op, heap);
    }

    if (g_verbose && applied > 0)
        std::cerr << "    greedy: " << applied << " moves, cost="
                  << part.total_cost() << "\n";
    return part;
}

// ============================================================================
// Full search pipeline
// ============================================================================

Partition local_search(const Problem& prob, const DAG& dag) {
    // Phase 1: try all initialization strategies + greedy descent
    auto strategies = all_init_strategies();
    Partition best;
    double best_cost = 1e18;

    for (auto& s : strategies) {
        std::cerr << "  Init " << s.name << "...\n";
        auto init = s.init(prob, dag);
        if (g_verbose) std::cerr << "    " << init.num_alive() << " groups, cost="
                  << init.total_cost() << "\n";

        auto result = greedy_descent(std::move(init));

        if (result.total_cost() < best_cost - 0.001) {
            best_cost = result.total_cost();
            best = std::move(result);
            if (g_verbose) std::cerr << "    * new best\n";
        }
    }

    std::cerr << "  After greedy: " << best.num_alive() << " groups, cost="
              << best.total_cost() << "\n";

    // Phase 2: FM exploration from the best greedy result
    FMOuterConfig fm_cfg;
    fm_cfg.pass_config.floor_fraction = 0.30;
    fm_cfg.pass_config.max_drift_fraction = 0.50;
    fm_cfg.pass_config.init_count = 3;
    fm_cfg.max_passes = 1000;
    fm_cfg.max_no_improve = 15;

    auto fm_result = fm_outer_loop(best, fm_cfg);

    if (fm_result.best_cost < best_cost - 0.001) {
        best = std::move(fm_result.best_partition);
        best_cost = fm_result.best_cost;
    }

    std::cerr << "  Final: " << best.num_alive() << " groups, cost="
              << best.total_cost() << "\n";
    return best;
}