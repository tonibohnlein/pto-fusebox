#include "search/verbose.h"
#include "search/local_search.h"
#include "init/init_strategies.h"
#include "core/cost_cache.h"
#include "search/fm_outer.h"
#include <iostream>

// ============================================================================
// Standalone inline ephemeral gap check for merges.
// Checks: would any tensor PRODUCED by ops in `merged` become ephemeral
// with unserved external consumers?
// ============================================================================

static bool inline_gap_check(const Partition& p, const std::set<size_t>& merged,
                              size_t ga, size_t gb) {
    const Problem& prob = *p.prob;
    const DAG& dag = *p.dag;
    for (auto op : merged) {
        for (auto t : prob.ops[op].outputs) {
            bool consumed_in = false;
            for (auto cop : dag.tensor_consumers[t])
                if (merged.count(cop)) { consumed_in = true; break; }
            if (!consumed_in) continue;

            int prod_op = dag.tensor_producer[t];
            if (prod_op < 0) continue;

            bool available = false;
            for (auto gj : p.groups_of((size_t)prod_op)) {
                if (gj == ga || gj == gb || !p.groups[gj].alive) continue;
                bool gj_consumes = false;
                for (auto cop : dag.tensor_consumers[t])
                    if (p.groups[gj].ops.count(cop)) { gj_consumes = true; break; }
                if (!gj_consumes) { available = true; break; }
            }
            if (available) continue;

            for (auto cop : dag.tensor_consumers[t]) {
                for (auto gj : p.groups_of(cop)) {
                    if (gj == ga || gj == gb || !p.groups[gj].alive) continue;
                    if (!p.groups[gj].ops.count((size_t)prod_op))
                        return true;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Check that all boundary inputs of `proposed` are available from slow memory.
//
// inline_gap_check checks the OUTPUT direction: "does this group make a tensor
// ephemeral that strands external consumers?"
//
// This checks the INPUT direction: "does this group NEED a tensor that's
// ephemeral everywhere it's produced?"
//
// Required for RECOMPUTE and STEAL (ga side), where adding an op to a group
// creates new boundary input demands. NOT needed for MERGE (merged group's
// inputs ⊆ ga's inputs ∪ gb's inputs, and ga+gb are excluded).
// ============================================================================

static bool boundary_inputs_unavailable(const Partition& p,
                                         const std::set<size_t>& proposed) {
    const Problem& prob = *p.prob;
    const DAG& dag = *p.dag;

    for (auto op : proposed) {
        for (auto t : prob.ops[op].inputs) {
            int prod = dag.tensor_producer[t];
            if (prod < 0) continue;            // graph input → always available
            if (proposed.count((size_t)prod)) continue;  // produced internally → fine

            // T is a boundary input of proposed. Is T available from slow memory?
            // T is available if some alive group writes it as a boundary output
            // (= has the producer, does NOT consume T internally).
            bool available = false;
            for (auto gj : p.groups_of((size_t)prod)) {
                if (!p.groups[gj].alive) continue;
                bool gj_consumes = false;
                for (auto cop : dag.tensor_consumers[t])
                    if (p.groups[gj].ops.count(cop)) { gj_consumes = true; break; }
                if (!gj_consumes) { available = true; break; }
            }
            if (!available) return true;  // T needed but not available → gap!
        }
    }
    return false;
}

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
                    if (!inline_gap_check(part, merged, gi, gj)) {
                        double new_cost = part.eval_set(merged);
                        double saving = (part.groups[gi].cost + part.groups[gj].cost) - new_cost;
                        try_better(best, {Move::MERGE, gi, gj, 0, saving, gen_i, gen_j});
                    }
                }
            }

            // STEAL + RECOMPUTE
            if (!part.dag->merge_creates_cycle({adj_op}, part.groups[gi].ops)) {
                std::set<size_t> new_gi = part.groups[gi].ops;
                new_gi.insert(adj_op);

                // Evaluate new_gi cost once — shared by STEAL and RECOMPUTE.
                double new_gi_cost = part.eval_set(new_gi);
                if (new_gi_cost < 1e17) {
                    // STEAL: adj_op leaves gj. Gap check: both the expanded gi
                    // and the shrunk gj may create ephemeral tensors.
                    {
                        std::set<size_t> new_gj = part.groups[gj].ops;
                        new_gj.erase(adj_op);
                        std::vector<std::set<size_t>> components = {new_gi};
                        if (!new_gj.empty()) components.push_back(new_gj);
                        if (!part.split_creates_ephemeral_gap(components, {gi, gj}) &&
                            !boundary_inputs_unavailable(part, new_gi)) {
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
                    }

                    // RECOMPUTE: adj_op is copied into gi; gj keeps it.
                    // Two checks needed:
                    //   1. Output direction: does new_gi make a tensor ephemeral
                    //      that strands external consumers? (inline_gap_check)
                    //   2. Input direction: does new_gi NEED a tensor that's
                    //      ephemeral everywhere? (boundary_inputs_unavailable)
                    //      E.g. adj_op consumes T35 which is ephemeral in gj.
                    if (!inline_gap_check(part, new_gi, gi, SIZE_MAX) &&
                        !boundary_inputs_unavailable(part, new_gi)) {
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

            // Gap check: would the split components strand any consumer?
            std::vector<std::set<size_t>> components = er.remainder_components;
            bool op_in_other = false;
            for (auto gj : part.groups_of(op))
                if (gj != gi) { op_in_other = true; break; }
            if (!op_in_other)
                components.push_back({op});
            if (part.split_creates_ephemeral_gap(components, gi)) continue;

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
            if (er.feasible) {
                std::vector<std::set<size_t>> components = er.remainder_components;
                bool op_in_other = false;
                for (auto gj : part.groups_of(op))
                    if (gj != gi) { op_in_other = true; break; }
                if (!op_in_other)
                    components.push_back({op});
                if (!part.split_creates_ephemeral_gap(components, gi))
                    try_better(best, {Move::INTERNAL_EJECT, gi, 0, op, er.saving, gen_i, 0});
            }

            for (auto v : part.dag->op_neighbors[op]) {
                if (!part.groups[gi].ops.count(v)) continue;
                size_t u_lo = std::min(op, v);
                size_t u_hi = std::max(op, v);
                auto sr = part.eval_split(u_lo, u_hi, gi);
                if (sr.feasible) {
                    if (!part.split_creates_ephemeral_gap({sr.side_a, sr.side_b}, gi))
                    {
                        Move m;
                        m.type = Move::SPLIT; m.ga = gi; m.gb = 0;
                        m.op = u_lo; m.saving = sr.saving;
                        m.gen_a = gen_i; m.gen_b = 0; m.op2 = u_hi;
                        try_better(best, m);
                    }
                }
            }

            if (best.saving > -floor)
                heap.push(best);
        }
    }

    // --- DE_RECOMPUTE: remove group gi if all ops covered elsewhere ---
    {
        bool all_covered = true;
        for (auto op : part.groups[gi].ops) {
            bool in_other = false;
            for (auto gj : part.groups_of(op))
                if (gj != gi && part.groups[gj].alive) { in_other = true; break; }
            if (!in_other) { all_covered = false; break; }
        }
        if (all_covered) {
            // Check: every boundary output of gi is available from some other group
            auto sg = Subgraph::create(*part.prob, *part.dag,
                          std::vector<size_t>(part.groups[gi].ops.begin(),
                                              part.groups[gi].ops.end()));
            bool safe = true;
            if (sg) {
                for (auto t : sg->boundary_outputs()) {
                    int prod = part.dag->tensor_producer[t];
                    if (prod < 0) continue;
                    bool available = false;
                    for (auto gj : part.groups_of((size_t)prod)) {
                        if (gj == gi || !part.groups[gj].alive) continue;
                        auto sg_j = Subgraph::create(*part.prob, *part.dag,
                                        std::vector<size_t>(part.groups[gj].ops.begin(),
                                                            part.groups[gj].ops.end()));
                        if (sg_j && sg_j->boundary_outputs().count(t)) {
                            available = true; break;
                        }
                    }
                    if (!available) { safe = false; break; }
                }
            }
            if (safe) {
                double saving = part.groups[gi].cost;
                if (saving > -floor)
                    heap.push({Move::DE_RECOMPUTE, gi, 0, 0, saving, gen_i, 0});
            }
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
            if (inline_gap_check(part, merged, m.ga, m.gb)) return {};
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

            // Ephemeral gap check: both the growing ga and shrinking gb.
            // Shrinking gb can make a tensor ephemeral that was boundary before.
            {
                std::vector<std::set<size_t>> components = {new_ga};
                if (!new_gb.empty()) components.push_back(new_gb);
                if (part.split_creates_ephemeral_gap(components, {m.ga, m.gb}))
                    return {};
            }
            // Input direction: new_ga gains m.op whose inputs might be ephemeral
            if (boundary_inputs_unavailable(part, new_ga))
                return {};

            double new_ga_cost = part.eval_set(new_ga);
            if (new_ga_cost >= 1e17) return {};

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
            if (part.dag->merge_creates_cycle({m.op}, part.groups[m.ga].ops)) return {};
            // Output direction: ephemeral tensors stranding external consumers
            if (inline_gap_check(part, new_ga, m.ga, SIZE_MAX)) return {};
            // Input direction: new boundary inputs unavailable from slow memory
            if (boundary_inputs_unavailable(part, new_ga)) return {};
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
        case Move::EJECT:
        case Move::INTERNAL_EJECT: {
            auto er = part.eval_eject(m.op, m.ga);
            if (!er.feasible || er.saving < -0.001) return {};

            // Ephemeral gap check: splitting the group may make a tensor
            // ephemeral in a remainder component, stranding external consumers.
            {
                std::vector<std::set<size_t>> components = er.remainder_components;
                // Add singleton if it will become a new group
                bool op_in_other = false;
                for (auto gj : part.groups_of(m.op))
                    if (gj != m.ga) { op_in_other = true; break; }
                if (!op_in_other)
                    components.push_back({m.op});
                if (part.split_creates_ephemeral_gap(components, m.ga))
                    return {};
            }

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

            // Ephemeral gap check on the two halves.
            if (part.split_creates_ephemeral_gap({sr.side_a, sr.side_b}, m.ga))
                return {};

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
            // Re-verify: all ops still covered elsewhere
            for (auto op : part.groups[m.ga].ops) {
                bool in_other = false;
                for (auto gj : part.groups_of(op))
                    if (gj != m.ga && part.groups[gj].alive) { in_other = true; break; }
                if (!in_other) return {};
            }
            // Re-verify: boundary outputs still available
            auto sg = Subgraph::create(*part.prob, *part.dag,
                          std::vector<size_t>(part.groups[m.ga].ops.begin(),
                                              part.groups[m.ga].ops.end()));
            if (sg) {
                for (auto t : sg->boundary_outputs()) {
                    int prod = part.dag->tensor_producer[t];
                    if (prod < 0) continue;
                    bool available = false;
                    for (auto gj : part.groups_of((size_t)prod)) {
                        if (gj == m.ga || !part.groups[gj].alive) continue;
                        auto sg_j = Subgraph::create(*part.prob, *part.dag,
                                        std::vector<size_t>(part.groups[gj].ops.begin(),
                                                            part.groups[gj].ops.end()));
                        if (sg_j && sg_j->boundary_outputs().count(t)) {
                            available = true; break;
                        }
                    }
                    if (!available) return {};
                }
            }

            dirty = part.adjacent_groups(m.ga);
            dirty.erase(m.ga);
            part.groups[m.ga].alive = false;
            part.groups[m.ga].gen++;
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
    // Cache contract: if the caller set part.cache before passing in, all
    // eval_set calls during descent route through that cache. The cache
    // pointer is preserved through std::move, so callers that do:
    //   auto init = s.init(prob, dag, &cache);   // sets init.cache
    //   greedy_descent(std::move(init));          // cache preserved
    // get full cache benefits. A direct call like:
    //   greedy_descent(Partition::trivial(p, d))  // cache is nullptr
    // works correctly but is uncached — wire a cache at the call site.
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
// Cleanup: remove redundant recomputation groups
//
// A group is redundant if every op in it also appears in at least one other
// alive group, AND removing it doesn't create an ephemeral gap.
// ============================================================================

void cleanup_redundant_recomputation(Partition& part) {
    bool changed = true;
    int removed = 0;

    while (changed) {
        changed = false;
        part.rebuild_index();

        for (size_t gi = 0; gi < part.groups.size(); gi++) {
            if (!part.groups[gi].alive) continue;

            // Check: every op covered by another alive group
            bool all_covered = true;
            for (auto op : part.groups[gi].ops) {
                bool in_other = false;
                for (auto gj : part.groups_of(op))
                    if (gj != gi && part.groups[gj].alive) { in_other = true; break; }
                if (!in_other) { all_covered = false; break; }
            }
            if (!all_covered) continue;

            // Check: removing gi doesn't create an ephemeral gap.
            // For each tensor T that is a boundary output of gi (produced but
            // NOT consumed internally), verify some other group also writes T.
            bool safe = true;
            auto sg = Subgraph::create(*part.prob, *part.dag,
                          std::vector<size_t>(part.groups[gi].ops.begin(),
                                              part.groups[gi].ops.end()));
            if (sg) {
                for (auto t : sg->boundary_outputs()) {
                    int prod = part.dag->tensor_producer[t];
                    if (prod < 0) continue;
                    bool available_elsewhere = false;
                    for (auto gj : part.groups_of((size_t)prod)) {
                        if (gj == gi || !part.groups[gj].alive) continue;
                        auto sg_j = Subgraph::create(*part.prob, *part.dag,
                                        std::vector<size_t>(part.groups[gj].ops.begin(),
                                                            part.groups[gj].ops.end()));
                        if (sg_j && sg_j->boundary_outputs().count(t)) {
                            available_elsewhere = true; break;
                        }
                    }
                    if (!available_elsewhere) { safe = false; break; }
                }
            }
            if (!safe) continue;

            part.groups[gi].alive = false;
            part.groups[gi].gen++;
            changed = true;
            removed++;
        }
    }
    if (removed > 0) {
        part.rebuild_index();
        std::cerr << "    cleanup: removed " << removed
                  << " redundant recomputation group(s), cost="
                  << part.total_cost() << "\n";
    }
}

// ============================================================================
// Repair: fix any ephemeral gaps by splitting offending groups
//
// For each alive group, check if any ephemeral tensor T has an external
// consumer that isn't served. If so, eject the producer op from the group
// so that T becomes a boundary output instead of ephemeral.
//
// This is a safety net — with correct gap checks during search, this
// should never fire. But it guarantees a valid partition regardless.
// ============================================================================

void repair_ephemeral_gaps(Partition& part) {
    int repairs = 0;
    bool changed = true;

    while (changed) {
        changed = false;
        part.rebuild_index();

        // Collect all boundary outputs across alive groups
        std::set<size_t> all_boundary_outputs;
        for (size_t gi = 0; gi < part.groups.size(); gi++) {
            if (!part.groups[gi].alive) continue;
            auto sg = Subgraph::create(*part.prob, *part.dag,
                          std::vector<size_t>(part.groups[gi].ops.begin(),
                                              part.groups[gi].ops.end()));
            if (sg) for (auto t : sg->boundary_outputs())
                all_boundary_outputs.insert(t);
        }

        for (size_t gi = 0; gi < part.groups.size(); gi++) {
            if (!part.groups[gi].alive) continue;
            auto sg = Subgraph::create(*part.prob, *part.dag,
                          std::vector<size_t>(part.groups[gi].ops.begin(),
                                              part.groups[gi].ops.end()));
            if (!sg) continue;

            for (auto t : sg->boundary_inputs()) {
                if (part.dag->tensor_producer[t] < 0) continue;
                if (all_boundary_outputs.count(t)) continue;



                // T is needed but not available. Find which group makes it
                // ephemeral and split it.
                int prod_op = part.dag->tensor_producer[t];
                for (auto gj : part.groups_of((size_t)prod_op)) {
                    if (!part.groups[gj].alive) continue;
                    // Check if T is ephemeral in gj
                    auto sg_j = Subgraph::create(*part.prob, *part.dag,
                                    std::vector<size_t>(part.groups[gj].ops.begin(),
                                                        part.groups[gj].ops.end()));
                    if (!sg_j || !sg_j->ephemeral().count(t)) continue;


                    std::cerr << "    REPAIR: T" << t << " needed by G" << gi 
                            << ", ephemeral in G" << gj
                            << " (ops:";
                    for (auto o : part.groups[gj].ops) std::cerr << " " << o;
                    std::cerr << ")\n";

                    // T is ephemeral in gj. Eject the producer to force T
                    // to become a boundary output. Split gj into:
                    //   {prod_op} (singleton) + remainder
                    std::set<size_t> remainder = part.groups[gj].ops;
                    remainder.erase((size_t)prod_op);

                    double prod_cost = part.eval_set({(size_t)prod_op});

                    if (!remainder.empty()) {
                        auto comps = part.connected_components(remainder);
                        part.groups[gj].ops = comps[0];
                        part.groups[gj].cost = part.eval_set(comps[0]);
                        part.groups[gj].gen++;
                        for (size_t c = 1; c < comps.size(); c++)
                            part.add_group(comps[c], part.eval_set(comps[c]));
                    } else {
                        part.groups[gj].alive = false;
                        part.groups[gj].gen++;
                    }
                    part.add_group({(size_t)prod_op}, prod_cost);

                    repairs++;
                    changed = true;
                    break;
                }
                if (changed) break;
            }
            if (changed) break;
        }
    }

    if (repairs > 0) {
        part.rebuild_index();
        std::cerr << "    repair: fixed " << repairs
                  << " ephemeral gap(s), cost="
                  << part.total_cost() << "\n";
    }
}

// ============================================================================
// Quick full gap check
// ============================================================================

bool partition_has_gap(const Partition& part) {
    if (!part.prob || !part.dag) return false;

    // Collect all boundary outputs across alive groups
    std::set<size_t> all_boundary_outputs;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        auto sg = Subgraph::create(*part.prob, *part.dag,
                      std::vector<size_t>(part.groups[gi].ops.begin(),
                                          part.groups[gi].ops.end()));
        if (sg) for (auto t : sg->boundary_outputs())
            all_boundary_outputs.insert(t);
    }

    // Check every group's boundary inputs
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        auto sg = Subgraph::create(*part.prob, *part.dag,
                      std::vector<size_t>(part.groups[gi].ops.begin(),
                                          part.groups[gi].ops.end()));
        if (!sg) continue;
        for (auto t : sg->boundary_inputs()) {
            if (part.dag->tensor_producer[t] < 0) continue;
            if (!all_boundary_outputs.count(t)) return true;
        }
    }
    return false;
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
        cleanup_redundant_recomputation(result);
        // Prevention-based gap checks in all move types should prevent gaps.
        // repair_ephemeral_gaps is a final safety net — fires a warning if hit.
        repair_ephemeral_gaps(result);

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

    // Phase 3: post-search cleanup
    cleanup_redundant_recomputation(best);
    repair_ephemeral_gaps(best);

    std::cerr << "  Final: " << best.num_alive() << " groups, cost="
              << best.total_cost() << "\n";
    return best;
}