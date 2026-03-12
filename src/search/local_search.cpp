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

    auto& groups = part.groups_of(op);
    if (groups.empty()) return;
    size_t primary = groups[0];

    OpMove entry;
    entry.op = op;
    entry.move = move;
    entry.primary_group = primary;
    entry.gen_a = part.groups[primary].gen;

    // Track secondary group gen for two-group moves
    if (move.gb < part.groups.size() && part.groups[move.gb].alive)
        entry.gen_b = part.groups[move.gb].gen;

    heap.push(entry);
}

// ============================================================================
// Collect all ops that need re-evaluation after affected groups change
// ============================================================================

static std::set<size_t> dirty_ops(const Partition& part,
                                   const std::set<size_t>& affected_groups) {
    std::set<size_t> ops;
    std::set<size_t> relevant = affected_groups;
    for (auto gi : affected_groups) {
        if (!part.groups[gi].alive) continue;
        auto adj = part.adjacent_groups(gi);
        relevant.insert(adj.begin(), adj.end());
    }
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

    for (size_t op = 0; op < part.prob->num_ops(); op++)
        push_op_move(part, op, heap);

    int applied = 0;
    while (!heap.empty()) {
        OpMove entry = heap.top();
        heap.pop();

        // Stop when no positive moves remain
        if (entry.move.saving <= 0.001) break;

        // Staleness: primary group gen must match
        if (entry.primary_group >= part.groups.size() ||
            !part.groups[entry.primary_group].alive ||
            part.groups[entry.primary_group].gen != entry.gen_a)
            continue;

        // Staleness: secondary group gen must match (for two-group moves)
        if (entry.move.gb < part.groups.size()) {
            if (!part.groups[entry.move.gb].alive)
                continue;
            if (entry.gen_b >= 0 &&
                part.groups[entry.move.gb].gen != entry.gen_b)
                continue;
        }

        // Snapshot cost, apply, then verify improvement.
        // apply_fm_move is designed for FM (accepts negative moves), so we
        // must guard against stale entries whose true gain is now ≤ 0.
        double cost_before = part.total_cost();

        auto affected = apply_fm_move(part, entry.move);
        if (affected.empty()) continue;

        double cost_after = part.total_cost();
        if (cost_after >= cost_before - 0.001) continue;  // no improvement → don't regenerate

        applied++;

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