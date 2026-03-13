#include "search/verbose.h"
#include "search/local_search.h"
#include "init/init_strategies.h"
#include "search/fm_outer.h"
#include <iostream>

// ============================================================================
// Move generation (positive-saving only)
// ============================================================================

void generate_moves(const Partition& part, size_t gi, MoveHeap& heap,
                    double floor) {
    if (!part.groups[gi].alive) return;
    int gen_i = part.groups[gi].gen;

    // Helper: track best candidate and push if good enough
    auto try_better = [](Move& best, Move candidate) {
        if (candidate.saving > best.saving)
            best = candidate;
    };

    // --- Border moves: one best per neighbor op ---
    // adj_op is outside gi; evaluate MERGE/STEAL/RECOMPUTE with each neighbor group
    auto neighbors = part.boundary_neighbors(gi);
    std::set<size_t> merge_checked;  // groups already evaluated for MERGE with gi
    for (auto adj_op : neighbors) {
        Move best;
        best.saving = -floor;

        for (auto gj : part.groups_of(adj_op)) {
            if (gj == gi) continue;
            int gen_j = part.groups[gj].gen;

            // MERGE (once per gj — result is the same regardless of which adj_op)
            if (!merge_checked.count(gj)) {
                merge_checked.insert(gj);
                if (!part.dag->merge_creates_cycle(part.groups[gi].ops, part.groups[gj].ops)) {
                    std::set<size_t> merged = part.groups[gi].ops;
                    merged.insert(part.groups[gj].ops.begin(), part.groups[gj].ops.end());
                    if (!part.creates_ephemeral_gap(merged, gi, gj)) {
                        double new_cost = part.eval_set(merged);
                        double saving = (part.groups[gi].cost + part.groups[gj].cost) - new_cost;
                        try_better(best, {Move::MERGE, gi, gj, 0, saving, gen_i, gen_j});
                    }
                }
            }

            // STEAL + RECOMPUTE (share the gi-expansion eval)
            if (!part.dag->merge_creates_cycle({adj_op}, part.groups[gi].ops)) {
                std::set<size_t> new_gi = part.groups[gi].ops;
                new_gi.insert(adj_op);
                // For STEAL, adj_op is removed from gj after the move, so gj must
                // be excluded from the recompute-exemption check: gj can no longer
                // serve as a backup producer of tensors that adj_op emits.
                // For RECOMPUTE, adj_op stays in gj so excluding gj is equally
                // safe (gj's own consumers are fine; only cross-group consumers matter).
                if (!part.creates_ephemeral_gap(new_gi, gi, gj)) {
                    double new_gi_cost = part.eval_set(new_gi);
                    if (new_gi_cost < 1e17) {
                        // STEAL: move adj_op from gj to gi
                        std::set<size_t> new_gj = part.groups[gj].ops;
                        new_gj.erase(adj_op);
                        double new_gj_cost = 0;
                        bool valid = true;
                        if (!new_gj.empty()) {
                            new_gj_cost = part.eval_set(new_gj);
                            if (new_gj_cost >= 1e17) valid = false;
                        }
                        if (valid) {
                            double saving = (part.groups[gi].cost + part.groups[gj].cost)
                                            - (new_gi_cost + new_gj_cost);
                            try_better(best, {Move::STEAL, gi, gj, adj_op, saving, gen_i, gen_j});
                        }

                        // RECOMPUTE: add adj_op to gi, keep in gj
                        double rsaving = part.groups[gi].cost - new_gi_cost;
                        try_better(best, {Move::RECOMPUTE, gi, gj, adj_op, rsaving, gen_i, gen_j});
                    }
                }
            }
        }

        if (best.saving > -floor)
            heap.push(best);
    }

    // --- Eject moves: one per ejectable border op ---
    if (part.groups[gi].ops.size() >= 2) {
        auto ejectable = part.ejectable_ops(gi);
        for (auto op : ejectable) {
            auto er = part.eval_eject(op, gi);
            if (!er.feasible) continue;
            if (er.saving > -floor)
                heap.push({Move::EJECT, gi, 0, op, er.saving, gen_i, 0});
        }
    }

    // --- Internal moves: one best per internal op (INTERNAL_EJECT or SPLIT) ---
    if (part.groups[gi].ops.size() >= 3 && part.groups[gi].ops.size() <= 15) {
        auto internals = part.internal_ops(gi);
        for (auto op : internals) {
            Move best;
            best.saving = -floor;

            auto er = part.eval_eject(op, gi);
            if (er.feasible)
                try_better(best, {Move::INTERNAL_EJECT, gi, 0, op, er.saving, gen_i, 0});

            // SPLIT at bridge edges incident to this op
            for (auto succ : part.dag->op_succs[op]) {
                if (!part.groups[gi].ops.count(succ)) continue;
                auto sr = part.eval_split(op, succ, gi);
                if (sr.feasible) {
                    Move m;
                    m.type = Move::SPLIT; m.ga = gi; m.gb = 0;
                    m.op = op; m.saving = sr.saving;
                    m.gen_a = gen_i; m.gen_b = 0; m.op2 = succ;
                    try_better(best, m);
                }
            }
            for (auto pred : part.dag->op_preds[op]) {
                if (!part.groups[gi].ops.count(pred)) continue;
                auto sr = part.eval_split(pred, op, gi);
                if (sr.feasible) {
                    Move m;
                    m.type = Move::SPLIT; m.ga = gi; m.gb = 0;
                    m.op = pred; m.saving = sr.saving;
                    m.gen_a = gen_i; m.gen_b = 0; m.op2 = op;
                    try_better(best, m);
                }
            }

            if (best.saving > -floor)
                heap.push(best);
        }
    }
}

// ============================================================================
// Apply a move. Returns dirty group set (empty if move rejected on re-verify).
// ============================================================================

static std::set<size_t> apply_move(Partition& part, const Move& m) {
    std::set<size_t> dirty;

    switch (m.type) {
        case Move::MERGE: {
            std::set<size_t> merged = part.groups[m.ga].ops;
            merged.insert(part.groups[m.gb].ops.begin(),
                          part.groups[m.gb].ops.end());
            if (part.creates_ephemeral_gap(merged, m.ga, m.gb)) return {};
            double new_cost = part.eval_set(merged);
            if (new_cost >= part.groups[m.ga].cost + part.groups[m.gb].cost - 0.001)
                return {};

            dirty = part.adjacent_groups(m.ga);
            auto nb = part.adjacent_groups(m.gb);
            dirty.insert(nb.begin(), nb.end());
            dirty.erase(m.ga); dirty.erase(m.gb);

            part.groups[m.ga].ops = std::move(merged);
            part.groups[m.ga].cost = new_cost;
            part.groups[m.ga].gen++;
            part.groups[m.gb].alive = false;
            part.groups[m.gb].gen++;
            dirty.insert(m.ga);
            break;
        }
        case Move::STEAL: {
            std::set<size_t> new_ga = part.groups[m.ga].ops;
            new_ga.insert(m.op);
            // m.gb loses m.op after STEAL — exclude it so it is not incorrectly
            // treated as a recompute source for tensors that m.op produces.
            if (part.creates_ephemeral_gap(new_ga, m.ga, m.gb)) return {};
            double new_ga_cost = part.eval_set(new_ga);
            if (new_ga_cost >= 1e17) return {};

            std::set<size_t> new_gb = part.groups[m.gb].ops;
            new_gb.erase(m.op);
            double new_gb_cost = 0;
            if (!new_gb.empty()) {
                new_gb_cost = part.eval_set(new_gb);
                if (new_gb_cost >= 1e17) return {};
            }
            double actual_saving = (part.groups[m.ga].cost + part.groups[m.gb].cost)
                                   - (new_ga_cost + new_gb_cost);
            if (actual_saving < -0.001) return {};

            dirty = part.adjacent_groups(m.ga);
            auto nb = part.adjacent_groups(m.gb);
            dirty.insert(nb.begin(), nb.end());
            dirty.erase(m.ga); dirty.erase(m.gb);

            part.groups[m.ga].ops = std::move(new_ga);
            part.groups[m.ga].cost = new_ga_cost;
            part.groups[m.ga].gen++;
            if (new_gb.empty()) {
                part.groups[m.gb].alive = false;
            } else {
                part.groups[m.gb].ops = std::move(new_gb);
                part.groups[m.gb].cost = new_gb_cost;
            }
            part.groups[m.gb].gen++;
            dirty.insert(m.ga);
            if (part.groups[m.gb].alive) dirty.insert(m.gb);
            break;
        }
        case Move::RECOMPUTE: {
            std::set<size_t> new_ga = part.groups[m.ga].ops;
            new_ga.insert(m.op);
            if (part.creates_ephemeral_gap(new_ga, m.ga, SIZE_MAX)) return {};
            double new_cost = part.eval_set(new_ga);
            double actual_saving = part.groups[m.ga].cost - new_cost;
            if (actual_saving < -0.001) return {};

            dirty = part.adjacent_groups(m.ga);
            dirty.erase(m.ga);

            part.groups[m.ga].ops = std::move(new_ga);
            part.groups[m.ga].cost = new_cost;
            part.groups[m.ga].gen++;
            dirty.insert(m.ga);
            break;
        }
        case Move::EJECT: {
            auto er = part.eval_eject(m.op, m.ga);
            if (!er.feasible || er.saving < -0.001) return {};

            dirty = part.adjacent_groups(m.ga);
            dirty.erase(m.ga);

            part.groups[m.ga].ops = er.remainder_components[0];
            part.groups[m.ga].cost = er.component_costs[0];
            part.groups[m.ga].gen++;
            dirty.insert(m.ga);

            for (size_t c = 1; c < er.remainder_components.size(); c++) {
                size_t ng = part.add_group(er.remainder_components[c], er.component_costs[c]);
                dirty.insert(ng);
            }

            if (er.singleton_cost > 0) {
                size_t sg = part.add_group({m.op}, er.singleton_cost);
                dirty.insert(sg);
            }
            break;
        }
        case Move::INTERNAL_EJECT: {
            auto er = part.eval_eject(m.op, m.ga);
            if (!er.feasible || er.saving < -0.001) return {};

            dirty = part.adjacent_groups(m.ga);
            dirty.erase(m.ga);

            part.groups[m.ga].ops = er.remainder_components[0];
            part.groups[m.ga].cost = er.component_costs[0];
            part.groups[m.ga].gen++;
            dirty.insert(m.ga);

            for (size_t c = 1; c < er.remainder_components.size(); c++) {
                size_t ng = part.add_group(er.remainder_components[c], er.component_costs[c]);
                dirty.insert(ng);
            }

            if (er.singleton_cost > 0) {
                size_t sg = part.add_group({m.op}, er.singleton_cost);
                dirty.insert(sg);
            }
            break;
        }
        case Move::SPLIT: {
            auto sr = part.eval_split(m.op, m.op2, m.ga);
            if (!sr.feasible || sr.saving < -0.001) return {};

            dirty = part.adjacent_groups(m.ga);
            dirty.erase(m.ga);

            part.groups[m.ga].ops = sr.side_a;
            part.groups[m.ga].cost = sr.cost_a;
            part.groups[m.ga].gen++;
            dirty.insert(m.ga);

            size_t gb = part.add_group(sr.side_b, sr.cost_b);
            dirty.insert(gb);
            break;
        }
    }

    part.rebuild_index();
    return dirty;
}

// ============================================================================
// Greedy descent
// ============================================================================

Partition greedy_descent(Partition part) {
    MoveHeap heap;
    for (size_t gi = 0; gi < part.groups.size(); gi++)
        generate_moves(part, gi, heap);

    int applied = 0;
    while (!heap.empty()) {
        Move m = heap.top();
        heap.pop();

        // Stale check
        if (!part.groups[m.ga].alive || part.groups[m.ga].gen != m.gen_a)
            continue;
        if (m.type == Move::MERGE || m.type == Move::STEAL || m.type == Move::RECOMPUTE) {
            if (!part.groups[m.gb].alive || part.groups[m.gb].gen != m.gen_b)
                continue;
        }

        auto dirty = apply_move(part, m);
        if (dirty.empty()) continue;

        applied++;
        for (auto gi : dirty)
            generate_moves(part, gi, heap);
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