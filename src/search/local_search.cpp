#include "search/verbose.h"
#include "search/local_search.h"
#include "init/init_strategies.h"
#include "core/cost_cache.h"
#include "search/fm_outer.h"
#include "util/pairing_heap.h"
#include <iostream>
#include <cassert>

// Under the new ephemeral rule, gap checks are unnecessary for acyclicity.
// Tensor materialization is the cost model's job (at finalization via
// compute_force_ephemeral / eval_group_in_context).

// Check if removing op from group_ops would create topological straddling:
// op has both predecessors AND successors in the remainder. If so, the
// remainder must execute both before AND after the removed op → group-DAG cycle.
static bool creates_topo_cycle(size_t op, const std::set<size_t>& group_ops,
                                const DAG& dag) {
    bool has_pred = false, has_succ = false;
    for (auto p : dag.op_preds[op])
        if (p != op && group_ops.count(p)) { has_pred = true; break; }
    if (!has_pred) return false;
    for (auto s : dag.op_succs[op])
        if (s != op && group_ops.count(s)) { has_succ = true; break; }
    return has_succ;
}

static bool split_creates_topo_cycle(const std::set<size_t>& side_a,
                                      const std::set<size_t>& side_b,
                                      const DAG& dag) {
    bool a_to_b = false, b_to_a = false;
    for (auto u : side_a) {
        if (a_to_b) break;
        for (auto v : dag.op_succs[u])
            if (side_b.count(v)) { a_to_b = true; break; }
    }
    if (!a_to_b) return false;
    for (auto u : side_b) {
        if (b_to_a) break;
        for (auto v : dag.op_succs[u])
            if (side_a.count(v)) { b_to_a = true; break; }
    }
    return b_to_a;
}

// ============================================================================
// Move generation (positive-saving only)
// ============================================================================

void generate_moves(const Partition& part, size_t gi, MoveHeap& heap,
                    double floor) {
    if (!part.groups[gi].alive) return;
    int gen_i = part.groups[gi].gen;

    auto try_better = [](Move& best, Move candidate) {
        if (candidate.saving > best.saving)
            best = candidate;
    };

    auto neighbors = part.boundary_neighbors(gi);
    std::set<size_t> merge_checked;
    for (auto adj_op : neighbors) {
        Move best;
        best.saving = -floor;

        for (auto gj : part.groups_of(adj_op)) {
            if (gj == gi) continue;
            int gen_j = part.groups[gj].gen;

            // MERGE
            if (!merge_checked.count(gj)) {
                merge_checked.insert(gj);
                if (!part.dag->merge_creates_cycle(part.groups[gi].ops, part.groups[gj].ops)) {
                    std::set<size_t> merged = part.groups[gi].ops;
                    merged.insert(part.groups[gj].ops.begin(), part.groups[gj].ops.end());
                    double new_cost = part.eval_set(merged);
                    double saving = (part.groups[gi].cost + part.groups[gj].cost) - new_cost;
                    try_better(best, {Move::MERGE, gi, gj, 0, saving, gen_i, gen_j});
                }
            }

            // STEAL + RECOMPUTE
            if (!part.dag->merge_creates_cycle({adj_op}, part.groups[gi].ops)) {
                std::set<size_t> new_gi = part.groups[gi].ops;
                new_gi.insert(adj_op);
                double new_gi_cost = part.eval_set(new_gi);
                if (new_gi_cost < 1e17) {
                    // STEAL
                    if (!creates_topo_cycle(adj_op, part.groups[gj].ops, *part.dag)) {
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
                    }

                    // RECOMPUTE
                    double rsaving = part.groups[gi].cost - new_gi_cost;
                    try_better(best, {Move::RECOMPUTE, gi, gj, adj_op, rsaving, gen_i, gen_j});
                }
            }
        }

        if (best.saving > -floor)
            heap.push(best);
    }

    // EJECT
    if (part.groups[gi].ops.size() >= 2) {
        auto ejectable = part.ejectable_ops(gi);
        for (auto op : ejectable) {
            if (creates_topo_cycle(op, part.groups[gi].ops, *part.dag)) continue;
            auto er = part.eval_eject(op, gi);
            if (!er.feasible) continue;
            if (er.saving > -floor)
                heap.push({Move::EJECT, gi, 0, op, er.saving, gen_i, 0});
        }
    }

    // INTERNAL_EJECT + SPLIT
    if (part.groups[gi].ops.size() >= 3 && part.groups[gi].ops.size() <= 15) {
        auto internals = part.internal_ops(gi);
        for (auto op : internals) {
            Move best;
            best.saving = -floor;

            if (!creates_topo_cycle(op, part.groups[gi].ops, *part.dag)) {
                auto er = part.eval_eject(op, gi);
                if (er.feasible)
                    try_better(best, {Move::INTERNAL_EJECT, gi, 0, op, er.saving, gen_i, 0});
            }

            for (auto v : part.dag->op_neighbors[op]) {
                if (!part.groups[gi].ops.count(v)) continue;
                size_t u_lo = std::min(op, v);
                size_t u_hi = std::max(op, v);
                auto sr = part.eval_split(u_lo, u_hi, gi);
                if (sr.feasible &&
                    !split_creates_topo_cycle(sr.side_a, sr.side_b, *part.dag)) {
                    Move m;
                    m.type = Move::SPLIT; m.ga = gi; m.gb = 0;
                    m.op = u_lo; m.saving = sr.saving;
                    m.gen_a = gen_i; m.gen_b = 0; m.op2 = u_hi;
                    try_better(best, m);
                }
            }

            if (best.saving > -floor)
                heap.push(best);
        }
    }

    // DE_RECOMPUTE
    {
        bool all_covered = true;
        for (auto op : part.groups[gi].ops) {
            bool in_other = false;
            for (auto gj : part.groups_of(op))
                if (gj != gi && part.groups[gj].alive) { in_other = true; break; }
            if (!in_other) { all_covered = false; break; }
        }
        if (all_covered) {
            double saving = part.groups[gi].cost;
            if (saving > -floor)
                heap.push({Move::DE_RECOMPUTE, gi, 0, 0, saving, gen_i, 0});
        }
    }
}

// ============================================================================
// Apply a move. Returns dirty group set (empty if move rejected on re-verify).
// ============================================================================

static std::set<size_t> apply_move(Partition& part, const Move& m) {
    // Moves on the heap are guaranteed feasible (acyclic) by best_move_for_op.
    // apply_move only re-verifies cost (which may have changed due to
    // concurrent mutations in the greedy loop).
    std::set<size_t> dirty;

    switch (m.type) {
        case Move::MERGE: {
            std::set<size_t> merged = part.groups[m.ga].ops;
            merged.insert(part.groups[m.gb].ops.begin(),
                          part.groups[m.gb].ops.end());
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
            std::set<size_t> new_gb = part.groups[m.gb].ops;
            new_gb.erase(m.op);

            double new_ga_cost = part.eval_set(new_ga);
            if (new_ga_cost >= 1e17) return {};
            double new_gb_cost = 0;
            if (!new_gb.empty()) {
                new_gb_cost = part.eval_set(new_gb);
                if (new_gb_cost >= 1e17) return {};
            }
            double actual_saving = (part.groups[m.ga].cost + part.groups[m.gb].cost)
                                   - (new_ga_cost + new_gb_cost);
            if (actual_saving < 0.001) return {};

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
            double new_cost = part.eval_set(new_ga);
            double actual_saving = part.groups[m.ga].cost - new_cost;
            if (actual_saving < 0.001) return {};

            dirty = part.adjacent_groups(m.ga);
            dirty.erase(m.ga);

            part.groups[m.ga].ops = std::move(new_ga);
            part.groups[m.ga].cost = new_cost;
            part.groups[m.ga].gen++;
            dirty.insert(m.ga);
            break;
        }
        case Move::EJECT:
        case Move::INTERNAL_EJECT: {
            auto er = part.eval_eject(m.op, m.ga);
            if (!er.feasible || er.saving < 0.001) return {};

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
            if (!sr.feasible || sr.saving < 0.001) return {};

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
        case Move::DE_RECOMPUTE: {
            for (auto op : part.groups[m.ga].ops) {
                bool in_other = false;
                for (auto gj : part.groups_of(op))
                    if (gj != m.ga && part.groups[gj].alive) { in_other = true; break; }
                if (!in_other) return {};
            }

            dirty = part.adjacent_groups(m.ga);
            dirty.erase(m.ga);
            part.groups[m.ga].alive = false;
            part.groups[m.ga].gen++;
            break;
        }
    }

    part.rebuild_index();

#ifndef NDEBUG
    if (!part.is_acyclic()) {
        std::cerr << "    ASSERT FAIL: apply_move produced cyclic partition! type="
                  << (int)m.type << " op=" << m.op
                  << " ga=" << m.ga << " gb=" << m.gb << "\n";
        assert(false && "apply_move produced cyclic partition");
    }
#endif

    return dirty;
}

// ============================================================================
// best_move_for_op: evaluate ALL possible moves involving op across all groups.
// Returns Move with saving > 0, or saving=0 if nothing positive found.
// ============================================================================

static Move best_move_for_op(const Partition& part, size_t op) {
    Move best;
    best.saving = 0;
    const auto& dag = *part.dag;
    auto groups_of_op = part.groups_of(op);
    bool op_in_multiple = groups_of_op.size() > 1;

    // --- DE_RECOMPUTE: if op is in multiple groups, evaluate removing it
    // from each group. For singletons {op}, removing = killing the group
    // (saving = group cost). For multi-op groups, this is EJECT with
    // singleton_cost=0 (handled below).
    if (op_in_multiple) {
        for (auto gi : groups_of_op) {
            if (!part.groups[gi].alive) continue;
            if (part.groups[gi].ops.size() == 1) {
                // Singleton group: removing it is pure savings
                double saving = part.groups[gi].cost;
                if (saving > best.saving)
                    best = {Move::DE_RECOMPUTE, gi, 0, op, saving, 0, 0};
            }
        }
    }

    // --- EJECT / INTERNAL_EJECT: remove op from a multi-op group ---
    for (auto gi : groups_of_op) {
        if (!part.groups[gi].alive || part.groups[gi].ops.size() < 2) continue;
        if (creates_topo_cycle(op, part.groups[gi].ops, dag)) continue;
        auto er = part.eval_eject(op, gi);
        if (!er.feasible) continue;

        bool is_border = false;
        for (auto nbr : dag.op_neighbors[op])
            if (!part.groups[gi].ops.count(nbr)) { is_border = true; break; }

        Move::Type mtype = is_border ? Move::EJECT : Move::INTERNAL_EJECT;
        if (er.saving > best.saving)
            best = {mtype, gi, 0, op, er.saving, 0, 0};
    }

    // --- SPLIT: split op's group at edge incident to op ---
    for (auto gi : groups_of_op) {
        if (!part.groups[gi].alive) continue;
        if (part.groups[gi].ops.size() < 3 || part.groups[gi].ops.size() > 15) continue;
        for (auto v : dag.op_neighbors[op]) {
            if (!part.groups[gi].ops.count(v)) continue;
            size_t u_lo = std::min(op, v), u_hi = std::max(op, v);
            auto sr = part.eval_split(u_lo, u_hi, gi);
            if (!sr.feasible) continue;
            if (split_creates_topo_cycle(sr.side_a, sr.side_b, dag)) continue;
            if (sr.saving > best.saving) {
                best.type = Move::SPLIT; best.ga = gi; best.gb = 0;
                best.op = u_lo; best.op2 = u_hi;
                best.saving = sr.saving; best.gen_a = 0; best.gen_b = 0;
            }
        }
    }

    // --- STEAL / RECOMPUTE / MERGE: op pulled into adjacent group ---
    std::set<size_t> adj_groups;
    for (auto nbr : dag.op_neighbors[op])
        for (auto gi : part.groups_of(nbr))
            if (part.groups[gi].alive && !part.groups[gi].ops.count(op))
                adj_groups.insert(gi);

    std::set<std::pair<size_t,size_t>> merge_checked;
    for (auto gi : adj_groups) {
        if (part.dag->merge_creates_cycle({op}, part.groups[gi].ops)) continue;

        std::set<size_t> new_gi = part.groups[gi].ops;
        new_gi.insert(op);
        double new_gi_cost = part.eval_set(new_gi);
        if (new_gi_cost >= 1e17) continue;

        for (auto gj : groups_of_op) {
            if (gj == gi || !part.groups[gj].alive) continue;

            // STEAL: move op from gj into gi
            if (!creates_topo_cycle(op, part.groups[gj].ops, dag)) {
                std::set<size_t> new_gj = part.groups[gj].ops;
                new_gj.erase(op);
                double new_gj_cost = new_gj.empty() ? 0 : part.eval_set(new_gj);
                if (new_gj.empty() || new_gj_cost < 1e17) {
                    double saving = (part.groups[gi].cost + part.groups[gj].cost)
                                    - (new_gi_cost + new_gj_cost);
                    if (saving > best.saving)
                        best = {Move::STEAL, gi, gj, op, saving, 0, 0};
                }
            }

            // MERGE: merge gi with gj (once per pair)
            auto pair_key = std::make_pair(std::min(gi, gj), std::max(gi, gj));
            if (!merge_checked.count(pair_key)) {
                merge_checked.insert(pair_key);
                if (!part.dag->merge_creates_cycle(part.groups[gi].ops, part.groups[gj].ops)) {
                    std::set<size_t> merged = part.groups[gi].ops;
                    merged.insert(part.groups[gj].ops.begin(), part.groups[gj].ops.end());
                    double mc = part.eval_set(merged);
                    double saving = (part.groups[gi].cost + part.groups[gj].cost) - mc;
                    if (saving > best.saving)
                        best = {Move::MERGE, gi, gj, op, saving, 0, 0};
                }
            }
        }

        // RECOMPUTE: copy op into gi
        {
            double rsaving = part.groups[gi].cost - new_gi_cost;
            size_t gb = groups_of_op.empty() ? 0 : groups_of_op[0];
            if (rsaving > best.saving)
                best = {Move::RECOMPUTE, gi, gb, op, rsaving, 0, 0};
        }
    }

    return best;
}

// ============================================================================
// Greedy descent with pairing heap — true steepest descent
// ============================================================================

Partition greedy_descent(Partition part) {
    const size_t num_ops = part.prob->num_ops();
    PairingHeap<Move> heap(num_ops);

    // Initialize: best move per op
    for (size_t op = 0; op < num_ops; op++) {
        auto m = best_move_for_op(part, op);
        if (m.saving > 0.001)
            heap.push_or_update(op, m);
    }

    int applied = 0;
    int rejected = 0;
    int iterations = 0;
    const int MAX_ITERATIONS = 100000;  // safety cap
    while (!heap.empty() && iterations < MAX_ITERATIONS) {
        iterations++;
        if (iterations % 500 == 0) {
            std::cerr << "    greedy iter=" << iterations
                      << " applied=" << applied << " rejected=" << rejected
                      << " cost=" << part.total_cost() << "\n";
        }
        auto m_opt = heap.pop_best();
        if (!m_opt || m_opt->saving <= 0.001) break;
        Move m = *m_opt;

        if (iterations <= 50 || iterations % 100 == 0) {
            std::cerr << "    [" << iterations << "] pop op=" << m.op
                      << " type=" << (int)m.type
                      << " ga=" << m.ga << " gb=" << m.gb
                      << " saving=" << m.saving << "\n";
        }

        // Apply move — heap guarantees feasibility (acyclicity checked in
        // best_move_for_op). Cost may have changed, so apply_move re-verifies.
        double old_total = part.total_cost();
        auto dirty = apply_move(part, m);
        if (dirty.empty()) {
            rejected++;
            if (iterations <= 50 || rejected % 100 == 0) {
                std::cerr << "    [" << iterations << "] REJECTED op=" << m.op
                          << " type=" << (int)m.type
                          << " ga=" << m.ga << " gb=" << m.gb
                          << " saving=" << m.saving
                          << " cost=" << part.total_cost() << "\n";
            }
            continue;
        }

#ifndef NDEBUG
        {
            double actual_gain = old_total - part.total_cost();
            double discrepancy = m.saving - actual_gain;
            if (std::abs(discrepancy) > 0.1 * std::max(1.0, std::abs(m.saving)) + 1.0) {
                std::cerr << "    GREEDY GAIN MISMATCH: predicted=" << m.saving
                          << " actual=" << actual_gain
                          << " Δ=" << discrepancy
                          << " type=" << (int)m.type
                          << " ga=" << m.ga << " gb=" << m.gb
                          << " op=" << m.op << "\n";
            }
        }
#endif

        applied++;

        if (g_verbose || applied <= 20 || applied % 100 == 0) {
            std::cerr << "    greedy #" << applied
                      << " type=" << (int)m.type
                      << " op=" << m.op << " ga=" << m.ga << " gb=" << m.gb
                      << " predicted=" << m.saving
                      << " actual=" << (old_total - part.total_cost())
                      << " cost=" << part.total_cost()
                      << " affected=" << 0 // placeholder
                      << "\n";
        }

        // Collect affected ops: ops in dirty groups + their DAG neighbors
        // Always include the moved op itself (may be in a killed group)
        std::set<size_t> affected_ops;
        affected_ops.insert(m.op);
        for (auto nbr : part.dag->op_neighbors[m.op])
            affected_ops.insert(nbr);
        for (auto gi : dirty) {
            if (!part.groups[gi].alive) continue;
            for (auto op : part.groups[gi].ops) {
                affected_ops.insert(op);
                for (auto nbr : part.dag->op_neighbors[op])
                    affected_ops.insert(nbr);
            }
        }

        // Update each affected op in the heap
        for (auto op : affected_ops) {
            auto fresh = best_move_for_op(part, op);
            if (fresh.saving > 0.001)
                heap.push_or_update(op, fresh);
            else
                heap.remove(op);
        }

        if (iterations <= 50 || iterations % 100 == 0) {
            std::cerr << "    [" << iterations << "] APPLIED #" << applied
                      << " actual_gain=" << (old_total - part.total_cost())
                      << " cost=" << part.total_cost()
                      << " refreshed=" << affected_ops.size() << "\n";
        }
    }

    if (iterations >= MAX_ITERATIONS) {
        std::cerr << "    greedy: HIT ITERATION CAP at " << MAX_ITERATIONS
                  << " applied=" << applied << " rejected=" << rejected
                  << " cost=" << part.total_cost() << "\n";
    } else {
        std::cerr << "    greedy: " << iterations << " iters, "
                  << applied << " applied, " << rejected << " rejected, cost="
                  << part.total_cost() << "\n";
    }
    return part;
}

// cleanup_redundant_recomputation and repair_ephemeral_gaps removed.
// Under the new ephemeral rule (tensor is ephemeral only if ALL DAG consumers
// are internal), external consumers force boundary output materialization,
// Cycle check used by FM pass as safety net.
bool partition_has_gap(const Partition& part) {
    return !part.is_acyclic();
}

// ============================================================================
// Full search pipeline
// ============================================================================

Partition local_search(const Problem& prob, const DAG& dag) {
    // Shared cache across all strategies and the greedy/FM passes.
    // Warm evaluations from init carry into the search immediately.
    CostCache cache;

    // Phase 1: try all initialization strategies + greedy descent
    auto strategies = all_init_strategies();
    Partition best;
    double best_cost = 1e18;

    for (auto& s : strategies) {
        std::cerr << "  Init " << s.name << "...\n";
        auto init = s.init(prob, dag, &cache);
        
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

    // Phase 3: done
    std::cerr << "  Final: " << best.num_alive() << " groups, cost="
              << best.total_cost() << "\n";
    return best;
}