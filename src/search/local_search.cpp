#include "search/verbose.h"
#include "search/local_search.h"
#include "init/init_strategies.h"
#include <iostream>

// ============================================================================
// TabuList
// ============================================================================

void TabuList::add(size_t op, size_t group) {
    entries_[{op, group}] = ttl_;
}

bool TabuList::is_tabu(size_t op, size_t group) const {
    return entries_.count({op, group}) > 0;
}

void TabuList::tick() {
    for (auto it = entries_.begin(); it != entries_.end(); ) {
        if (--it->second <= 0)
            it = entries_.erase(it);
        else
            ++it;
    }
}

// ============================================================================
// Move generation
//
// Generates moves involving group gi with saving > -floor.
// If tabu is non-null, filters moves that would reverse a tabu entry.
// ============================================================================

void generate_moves(const Partition& part, size_t gi, MoveHeap& heap,
                    double floor, const TabuList* tabu) {
    if (!part.groups[gi].alive) return;

    auto neighbors = part.boundary_neighbors(gi);
    for (auto adj_op : neighbors) {
        auto adj_groups = part.groups_of(adj_op);
        for (auto gj : adj_groups) {
            if (gj == gi) continue;
            int gen_i = part.groups[gi].gen;
            int gen_j = part.groups[gj].gen;

            // Move 1: Merge gi and gj
            {
                std::set<size_t> merged = part.groups[gi].ops;
                merged.insert(part.groups[gj].ops.begin(),
                              part.groups[gj].ops.end());
                double new_cost = part.eval_set(merged);
                double saving = (part.groups[gi].cost + part.groups[gj].cost)
                                - new_cost;
                if (saving > -floor) {
                    // Check tabu: after merge, ops from gb are in ga
                    // Tabu if any op from gb is tabu for ga
                    bool blocked = false;
                    if (tabu) {
                        for (auto op : part.groups[gj].ops)
                            if (tabu->is_tabu(op, gi)) { blocked = true; break; }
                    }
                    if (!blocked)
                        heap.push({Move::MERGE, gi, gj, 0, saving, gen_i, gen_j});
                }
            }

            // Move 2: Steal adj_op from gj into gi
            {
                // Tabu check: adj_op moving into gi
                if (tabu && tabu->is_tabu(adj_op, gi)) continue;

                std::set<size_t> new_gi = part.groups[gi].ops;
                new_gi.insert(adj_op);
                double new_gi_cost = part.eval_set(new_gi);
                if (new_gi_cost < 1e17) {
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
                        if (saving > -floor)
                            heap.push({Move::STEAL, gi, gj, adj_op, saving, gen_i, gen_j});
                    }
                }
            }

            // Move 3: Recompute adj_op in gi (keep in gj too)
            {
                if (tabu && tabu->is_tabu(adj_op, gi)) continue;

                std::set<size_t> new_gi = part.groups[gi].ops;
                new_gi.insert(adj_op);
                double new_gi_cost = part.eval_set(new_gi);
                double saving = part.groups[gi].cost - new_gi_cost;
                if (saving > -floor)
                    heap.push({Move::RECOMPUTE, gi, gj, adj_op, saving, gen_i, gen_j});
            }
        }
    }

    // Move 4: Eject a border op from gi into a new singleton
    if (part.groups[gi].ops.size() >= 2) {
        auto ejectable = part.ejectable_ops(gi);
        int gen_i = part.groups[gi].gen;
        for (auto op : ejectable) {
            // Tabu check: op leaving gi
            if (tabu && tabu->is_tabu(op, gi)) continue;

            std::set<size_t> remainder = part.groups[gi].ops;
            remainder.erase(op);
            double remainder_cost = part.eval_set(remainder);
            if (remainder_cost >= 1e17) continue;

            double singleton_cost = part.eval_set({op});
            if (singleton_cost >= 1e17) continue;

            double saving = part.groups[gi].cost - (remainder_cost + singleton_cost);
            if (saving > -floor)
                heap.push({Move::EJECT, gi, 0, op, saving, gen_i, 0});
        }
    }

    // Move 5: Internal eject — eject an internal op (may split into 3+ components)
    if (part.groups[gi].ops.size() >= 3 && part.groups[gi].ops.size() <= 15) {
        auto internals = part.internal_ops(gi);
        int gen_i = part.groups[gi].gen;
        for (auto op : internals) {
            if (tabu && tabu->is_tabu(op, gi)) continue;

            auto er = part.eval_eject(op, gi);
            if (!er.feasible) continue;
            if (er.saving > -floor)
                heap.push({Move::INTERNAL_EJECT, gi, 0, op, er.saving, gen_i, 0});
        }
    }

    // Move 6: Split group at a bridge edge into two non-trivial parts
    if (part.groups[gi].ops.size() >= 3 && part.groups[gi].ops.size() <= 15) {
        auto bridges = part.bridge_edges(gi);
        int gen_i = part.groups[gi].gen;
        for (auto& [a, b] : bridges) {
            if (tabu && (tabu->is_tabu(a, gi) || tabu->is_tabu(b, gi))) continue;

            auto sr = part.eval_split(a, b, gi);
            if (!sr.feasible) continue;
            if (sr.saving > -floor) {
                Move m;
                m.type = Move::SPLIT;
                m.ga = gi; m.gb = 0; m.op = a;
                m.saving = sr.saving;
                m.gen_a = gen_i; m.gen_b = 0;
                m.op2 = b;
                heap.push(m);
            }
        }
    }
}

// ============================================================================
// Apply a move. Returns dirty set (empty if move rejected on re-verify).
// Also populates tabu_entries with (op, group) pairs to mark tabu.
// ============================================================================

struct ApplyResult {
    std::set<size_t> dirty;
    std::vector<std::pair<size_t, size_t>> tabu_entries;  // (op, group) to forbid
};

static ApplyResult apply_move(Partition& part, const Move& m) {
    ApplyResult result;

    switch (m.type) {
        case Move::MERGE: {
            std::set<size_t> merged = part.groups[m.ga].ops;
            merged.insert(part.groups[m.gb].ops.begin(),
                          part.groups[m.gb].ops.end());
            double new_cost = part.eval_set(merged);
            if (new_cost >= part.groups[m.ga].cost + part.groups[m.gb].cost - 0.001)
                if (m.saving > 0) return result;  // positive move no longer profitable

            result.dirty = part.adjacent_groups(m.ga);
            auto nb = part.adjacent_groups(m.gb);
            result.dirty.insert(nb.begin(), nb.end());
            result.dirty.erase(m.ga); result.dirty.erase(m.gb);

            // Tabu: ops from gb can't be ejected from ga
            for (auto op : part.groups[m.gb].ops)
                result.tabu_entries.push_back({op, m.ga});

            part.groups[m.ga].ops = std::move(merged);
            part.groups[m.ga].cost = new_cost;
            part.groups[m.ga].gen++;
            part.groups[m.gb].alive = false;
            part.groups[m.gb].gen++;
            result.dirty.insert(m.ga);
            break;
        }
        case Move::STEAL: {
            std::set<size_t> new_ga = part.groups[m.ga].ops;
            new_ga.insert(m.op);
            double new_ga_cost = part.eval_set(new_ga);
            if (new_ga_cost >= 1e17) return result;

            std::set<size_t> new_gb = part.groups[m.gb].ops;
            new_gb.erase(m.op);
            double new_gb_cost = 0;
            if (!new_gb.empty()) {
                new_gb_cost = part.eval_set(new_gb);
                if (new_gb_cost >= 1e17) return result;
            }
            double actual_saving = (part.groups[m.ga].cost + part.groups[m.gb].cost)
                                   - (new_ga_cost + new_gb_cost);
            if (actual_saving < -0.001 && m.saving > 0) return result;

            result.dirty = part.adjacent_groups(m.ga);
            auto nb = part.adjacent_groups(m.gb);
            result.dirty.insert(nb.begin(), nb.end());
            result.dirty.erase(m.ga); result.dirty.erase(m.gb);

            // Tabu: op can't be stolen back from ga to gb
            result.tabu_entries.push_back({m.op, m.gb});

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
            result.dirty.insert(m.ga);
            if (part.groups[m.gb].alive) result.dirty.insert(m.gb);
            break;
        }
        case Move::RECOMPUTE: {
            std::set<size_t> new_ga = part.groups[m.ga].ops;
            new_ga.insert(m.op);
            double new_cost = part.eval_set(new_ga);
            double actual_saving = part.groups[m.ga].cost - new_cost;
            if (actual_saving < -0.001 && m.saving > 0) return result;

            result.dirty = part.adjacent_groups(m.ga);
            result.dirty.erase(m.ga);

            // Tabu: op can't be removed from ga
            result.tabu_entries.push_back({m.op, m.ga});

            part.groups[m.ga].ops = std::move(new_ga);
            part.groups[m.ga].cost = new_cost;
            part.groups[m.ga].gen++;
            result.dirty.insert(m.ga);
            break;
        }
        case Move::EJECT: {
            std::set<size_t> remainder = part.groups[m.ga].ops;
            remainder.erase(m.op);
            double remainder_cost = part.eval_set(remainder);
            if (remainder_cost >= 1e17) return result;

            double singleton_cost = part.eval_set({m.op});
            if (singleton_cost >= 1e17) return result;

            double actual_saving = part.groups[m.ga].cost
                                   - (remainder_cost + singleton_cost);
            if (actual_saving < -0.001 && m.saving > 0) return result;

            result.dirty = part.adjacent_groups(m.ga);
            result.dirty.erase(m.ga);

            part.groups[m.ga].ops = std::move(remainder);
            part.groups[m.ga].cost = remainder_cost;
            part.groups[m.ga].gen++;

            size_t new_gi = part.add_group({m.op}, singleton_cost);

            // Tabu: op can't be merged back into ga
            result.tabu_entries.push_back({m.op, m.ga});

            result.dirty.insert(m.ga);
            result.dirty.insert(new_gi);
            break;
        }
        case Move::INTERNAL_EJECT: {
            auto er = part.eval_eject(m.op, m.ga);
            if (!er.feasible) return result;
            if (er.saving < -0.001 && m.saving > 0) return result;

            result.dirty = part.adjacent_groups(m.ga);
            result.dirty.erase(m.ga);

            // Replace ga with first component, create new groups for rest
            part.groups[m.ga].ops = er.remainder_components[0];
            part.groups[m.ga].cost = er.component_costs[0];
            part.groups[m.ga].gen++;
            result.dirty.insert(m.ga);

            for (size_t c = 1; c < er.remainder_components.size(); c++) {
                size_t ng = part.add_group(er.remainder_components[c], er.component_costs[c]);
                result.dirty.insert(ng);
            }

            if (er.singleton_cost > 0) {
                size_t sg = part.add_group({m.op}, er.singleton_cost);
                result.dirty.insert(sg);
            }

            result.tabu_entries.push_back({m.op, m.ga});
            break;
        }
        case Move::SPLIT: {
            auto sr = part.eval_split(m.op, m.op2, m.ga);
            if (!sr.feasible) return result;
            if (sr.saving < -0.001 && m.saving > 0) return result;

            result.dirty = part.adjacent_groups(m.ga);
            result.dirty.erase(m.ga);

            // Replace ga with side_a, create new group for side_b
            part.groups[m.ga].ops = sr.side_a;
            part.groups[m.ga].cost = sr.cost_a;
            part.groups[m.ga].gen++;
            result.dirty.insert(m.ga);

            size_t gb = part.add_group(sr.side_b, sr.cost_b);
            result.dirty.insert(gb);

            // Tabu: don't immediately re-merge
            result.tabu_entries.push_back({m.op, gb});
            result.tabu_entries.push_back({m.op2, m.ga});
            break;
        }
    }

    return result;
}

// ============================================================================
// Phase 1: Greedy descent (positive moves only)
// ============================================================================

static int greedy_phase(Partition& part) {
    MoveHeap heap;
    for (size_t gi = 0; gi < part.groups.size(); gi++)
        generate_moves(part, gi, heap, 0.0, nullptr);

    int applied = 0;
    while (!heap.empty()) {
        Move m = heap.top();
        heap.pop();

        if (!part.groups[m.ga].alive || part.groups[m.ga].gen != m.gen_a)
            continue;
        if (m.type == Move::MERGE || m.type == Move::STEAL || m.type == Move::RECOMPUTE) {
            if (!part.groups[m.gb].alive || part.groups[m.gb].gen != m.gen_b)
                continue;
        }

        auto result = apply_move(part, m);
        if (result.dirty.empty()) continue;

        applied++;
        for (auto gi : result.dirty)
            generate_moves(part, gi, heap, 0.0, nullptr);
    }
    return applied;
}

// ============================================================================
// Phase 2: Tabu exploration (allows worsening moves)
// ============================================================================

static int tabu_phase(Partition& part, Partition& best_seen, double& best_cost) {
    int tabu_ttl = std::max(5, (int)part.num_alive() / 3);
    TabuList tabu(tabu_ttl);

    double neg_floor = best_cost * 0.10;

    int max_no_improve = std::max(10, (int)part.prob->num_ops() / 3);
    int no_improve = 0;
    int applied = 0;

    // Build initial heap with negative moves allowed
    MoveHeap heap;
    for (size_t gi = 0; gi < part.groups.size(); gi++)
        generate_moves(part, gi, heap, neg_floor, &tabu);

    while (!heap.empty() && no_improve < max_no_improve) {
        Move m = heap.top();
        heap.pop();

        // Stale check
        if (!part.groups[m.ga].alive || part.groups[m.ga].gen != m.gen_a)
            continue;
        if (m.type == Move::MERGE || m.type == Move::STEAL || m.type == Move::RECOMPUTE) {
            if (!part.groups[m.gb].alive || part.groups[m.gb].gen != m.gen_b)
                continue;
        }

        // Snapshot before worsening move if we're at best
        if (m.saving < 0) {
            double current_cost = part.total_cost();
            if (current_cost <= best_cost + 0.001) {
                best_seen = part;
                best_cost = current_cost;
            }
        }

        auto result = apply_move(part, m);
        if (result.dirty.empty()) continue;

        // Register tabu entries
        for (auto& [op, grp] : result.tabu_entries)
            tabu.add(op, grp);

        applied++;

        // Check for new best
        double new_cost = part.total_cost();
        if (new_cost < best_cost - 0.001) {
            best_cost = new_cost;
            best_seen = part;
            no_improve = 0;
            if (g_verbose) std::cerr << "      tabu: new best " << best_cost << "\n";
        } else {
            no_improve++;
        }

        // Regenerate moves only for dirty groups
        for (auto gi : result.dirty)
            generate_moves(part, gi, heap, neg_floor, &tabu);

        tabu.tick();
    }

    return applied;
}

Partition greedy_descent(Partition part) {
    greedy_phase(part);
    return part;
}

// ============================================================================
// Two-phase local search from a given partition
// ============================================================================

Partition local_search_from(Partition part) {
    // Phase 1: greedy descent
    int greedy_moves = greedy_phase(part);
    if (g_verbose) std::cerr << "    greedy: " << greedy_moves << " moves, cost="
              << part.total_cost() << "\n";

    // Snapshot best after greedy
    Partition best_seen = part;
    double best_cost = part.total_cost();

    // Phase 2: tabu exploration — allows worsening moves to escape local optima
    {
        int tabu_moves = tabu_phase(part, best_seen, best_cost);
        if (tabu_moves > 0) {
            if (g_verbose) std::cerr << "    tabu: " << tabu_moves << " moves, best="
                      << best_cost << "\n";
        }
    }

    if (g_verbose) std::cerr << "    -> " << best_seen.num_alive() << " groups, cost="
              << best_seen.total_cost() << "\n";
    return best_seen;
}

// ============================================================================
// Multi-start: run from each initialization, return overall best
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

        auto result = local_search_from(std::move(init));

        if (result.total_cost() < best_cost - 0.001) {
            best_cost = result.total_cost();
            best = std::move(result);
            if (g_verbose) std::cerr << "    * new best\n";
        }
    }

    std::cerr << "  Best: " << best.num_alive() << " groups, cost="
              << best.total_cost() << "\n";
    return best;
}