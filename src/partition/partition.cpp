#include "partition/partition.h"
#include "core/cost_cache.h"
#include <iostream>
#include <queue>
#include <functional>
#include <deque>
#include <algorithm>

// ============================================================================
// finalize()
// ============================================================================

void Partition::finalize() {
    // Rebuild index first so compute_force_ephemeral can look up other groups.
    rebuild_index();

    for (size_t gi = 0; gi < groups.size(); gi++) {
        auto& g = groups[gi];
        if (!g.alive) continue;
        g.sg = std::nullopt;

        // Compute partition-aware ephemeral set: tensors whose external
        // consumers are all served by recomputing groups.
        auto fe = compute_force_ephemeral(gi);

        auto sg_opt = Subgraph::create(*prob, *dag,
                          std::vector<size_t>(g.ops.begin(), g.ops.end()), fe);
        if (sg_opt) {
            auto c = sg_opt->best_cost();
            if (c.feasible) {
                g.cost     = c.latency;
                g.best_cfg = c.config;
            } else {
                g.cost = 1e18;
            }
            g.sg = std::move(*sg_opt);
        } else {
            g.cost = 1e18;
            std::cerr << "    WARNING: finalize: Subgraph::create failed for group with "
                      << g.ops.size() << " ops:";
            for (auto op : g.ops) std::cerr << " " << op;
            std::cerr << "\n";
        }
    }

    rebuild_group_dag();
}

void Partition::rebuild_index() {
    op_to_groups_.assign(prob->num_ops(), {});
    for (size_t i = 0; i < groups.size(); i++)
        if (groups[i].alive)
            for (auto op : groups[i].ops)
                op_to_groups_[op].push_back(i);
}

void Partition::rebuild_group_dag() {
    size_t ng = groups.size();
    group_preds.assign(ng, {});
    group_succs.assign(ng, {});
    group_in_deg.assign(ng, 0);
    tensor_to_group.clear();

    for (size_t i = 0; i < ng; i++) {
        if (!groups[i].alive || !groups[i].sg) continue;
        for (auto t : groups[i].sg->boundary_outputs())
            tensor_to_group[t] = i;
    }

    // Step 1: Get a valid topological order using OP-BASED OR-node Kahn's.
    // Used to disambiguate which source group to draw an edge from when a
    // producer op is in multiple alive groups (recomputation).
    std::vector<int> topo_pos(ng, -1);
    bool kahn_complete = false;
    {
        std::vector<int> unsatisfied(ng, 0);
        std::vector<bool> dep_met;
        std::vector<std::vector<std::pair<size_t, size_t>>> frees(ng);

        for (size_t op_idx = 0; op_idx < prob->ops.size(); op_idx++) {
            for (auto t : prob->ops[op_idx].inputs) {
                int prod = dag->tensor_producer[t];
                if (prod < 0) continue;
                for (auto target_gi : op_to_groups_[op_idx]) {
                    if (!groups[target_gi].alive) continue;
                    bool prod_internal = false;
                    for (auto g : op_to_groups_[(size_t)prod])
                        if (g == target_gi) { prod_internal = true; break; }
                    if (prod_internal) continue;

                    // Check if ANY alive source exists for this dep.
                    // If not, the dep is permanently unsatisfied — skip it
                    // to avoid poisoning the Kahn's traversal.
                    bool has_source = false;
                    for (auto source_gj : op_to_groups_[(size_t)prod])
                        if (groups[source_gj].alive) { has_source = true; break; }
                    if (!has_source) continue;

                    size_t dep_id = dep_met.size();
                    dep_met.push_back(false);
                    unsatisfied[target_gi]++;
                    for (auto source_gj : op_to_groups_[(size_t)prod]) {
                        if (groups[source_gj].alive)
                            frees[source_gj].push_back({target_gi, dep_id});
                    }
                }
            }
        }

        std::vector<size_t> q;
        q.reserve(ng);
        std::vector<bool> enqueued(ng, false);
        for (size_t i = 0; i < ng; i++)
            if (groups[i].alive && unsatisfied[i] == 0) {
                q.push_back(i);
                enqueued[i] = true;
            }
        int pos = 0;
        size_t head = 0;
        while (head < q.size()) {
            size_t u = q[head++];
            topo_pos[u] = pos++;
            for (auto [gi, dep_id] : frees[u]) {
                if (!dep_met[dep_id]) {
                    dep_met[dep_id] = true;
                    unsatisfied[gi]--;
                    if (unsatisfied[gi] == 0 && !enqueued[gi]) {
                        q.push_back(gi);
                        enqueued[gi] = true;
                    }
                }
            }
        }

        size_t n_resolved = (size_t)pos;
        size_t n_alive = num_alive();
        kahn_complete = (n_resolved >= n_alive);
        if (!kahn_complete) {
            std::cerr << "WARNING: rebuild_group_dag: Kahn's resolved "
                      << n_resolved << "/" << n_alive
                      << " groups — falling back to DAG topo order\n";
        }
    }

    // Step 2: Build group DAG edges.
    //
    // For each group i, for each op's input tensor T, if T's producer is
    // external to i, find the best source group and draw an edge.
    //
    // Source priority (best to worst):
    //   1. Kahn's topo_pos (available when Step 1 resolves the group)
    //   2. DAG-level topo position of the producer op within the group
    //      (always available — the op DAG is guaranteed acyclic)
    //   3. Arbitrary (smallest group index)
    for (size_t i = 0; i < ng; i++) {
        if (!groups[i].alive) continue;
        for (auto op : groups[i].ops) {
            for (auto t : prob->ops[op].inputs) {
                int prod = dag->tensor_producer[t];
                if (prod < 0) continue;
                if (groups[i].ops.count((size_t)prod)) continue;  // internal

                size_t best_gj = SIZE_MAX;
                int best_pos = INT_MAX;
                for (auto gj : op_to_groups_[(size_t)prod]) {
                    if (!groups[gj].alive || gj == i) continue;
                    // Use Kahn's topo_pos if available; otherwise fall back
                    // to DAG topo position of the producer op (deterministic,
                    // always valid). The offset by ng ensures resolved groups
                    // are always preferred over unresolved ones.
                    int pos;
                    if (topo_pos[gj] >= 0) {
                        pos = topo_pos[gj];
                    } else {
                        pos = (int)ng + (int)dag->topo_position((size_t)prod);
                    }
                    if (pos < best_pos) {
                        best_gj = gj;
                        best_pos = pos;
                    }
                }
                if (best_gj != SIZE_MAX) {
                    if (group_preds[i].insert(best_gj).second)
                        group_succs[best_gj].insert(i);
                }
            }
        }
    }

    for (size_t i = 0; i < ng; i++)
        group_in_deg[i] = (int)group_preds[i].size();

    // =========================================================================
    // Validation: cross-check group DAG edges against subgraph boundary inputs.
    // For every alive group i with a subgraph, every boundary input tensor T
    // whose producer op P is in some other alive group gj should have an edge
    // gj → i in the group DAG. If not, the DFS ordering will be wrong.
    // =========================================================================
    for (size_t i = 0; i < ng; i++) {
        if (!groups[i].alive || !groups[i].sg) continue;
        for (auto t : groups[i].sg->boundary_inputs()) {
            int prod = dag->tensor_producer[t];
            if (prod < 0) continue;  // graph input — no edge needed

            // Find any alive group that exports T as a boundary output
            bool has_edge_from_producer = false;
            size_t exporting_gj = SIZE_MAX;
            for (size_t gj = 0; gj < ng; gj++) {
                if (!groups[gj].alive || gj == i || !groups[gj].sg) continue;
                if (groups[gj].sg->boundary_outputs().count(t)) {
                    exporting_gj = gj;
                    if (group_preds[i].count(gj)) {
                        has_edge_from_producer = true;
                        break;
                    }
                }
            }

            if (!has_edge_from_producer && exporting_gj != SIZE_MAX) {
                // Edge is missing! Dump diagnostic info.
                std::cerr << "  EDGE_MISSING: G" << i << " needs T" << t
                          << " (prod=op" << prod << "), G" << exporting_gj
                          << " exports it, but no edge G" << exporting_gj
                          << "→G" << i << " exists.\n";

                // Why was the edge not created? Check Step 2 conditions:
                bool prod_in_i = groups[i].ops.count((size_t)prod);
                std::cerr << "    op" << prod << " in G" << i << ".ops? "
                          << (prod_in_i ? "YES (skipped as internal)" : "no") << "\n";

                std::cerr << "    op_to_groups_[" << prod << "] = {";
                for (auto gx : op_to_groups_[(size_t)prod])
                    std::cerr << " G" << gx
                              << (groups[gx].alive ? "" : "(dead)");
                std::cerr << " }\n";

                // Check if any op in group i actually consumes T in its raw inputs
                bool any_op_consumes_t = false;
                for (auto op : groups[i].ops) {
                    for (auto inp : prob->ops[op].inputs) {
                        if (inp == t) {
                            any_op_consumes_t = true;
                            std::cerr << "    op" << op << " in G" << i
                                      << " consumes T" << t << " as raw input\n";
                        }
                    }
                }
                if (!any_op_consumes_t) {
                    std::cerr << "    NO op in G" << i
                              << " consumes T" << t << " as raw input!"
                              << " (subgraph reports it as boundary input — "
                              << "possible force_ephemeral mismatch)\n";
                }
            }
        }
    }
}


// ============================================================================
// Construction
// ============================================================================

Partition Partition::trivial(const Problem& prob, const DAG& dag) {
    Partition p;
    p.prob = &prob;
    p.dag  = &dag;
    p.groups.resize(prob.num_ops());

    for (size_t i = 0; i < prob.num_ops(); i++) {
        p.groups[i].ops   = {i};
        p.groups[i].alive = true;
        auto sg = Subgraph::create(prob, dag, {i});
        if (sg) {
            auto c = sg->best_cost();
            p.groups[i].cost     = c.feasible ? c.latency : 1e18;
            p.groups[i].best_cfg = c.config;
            p.groups[i].sg       = std::move(*sg);
        }
    }

    p.rebuild_index();
    p.rebuild_group_dag();
    return p;
}

// ============================================================================
// Queries
// ============================================================================

double Partition::total_cost() const {
    double t = 0;
    for (auto& g : groups) if (g.alive) t += g.cost;
    return t;
}

size_t Partition::num_alive() const {
    size_t n = 0;
    for (auto& g : groups) if (g.alive) n++;
    return n;
}

bool Partition::is_border_op(size_t op, size_t gi) const {
    for (auto v : dag->op_neighbors[op])
        if (!groups[gi].ops.count(v)) return true;
    return false;
}

std::vector<size_t> Partition::border_ops(size_t gi) const {
    std::vector<size_t> result;
    for (auto op : groups[gi].ops)
        if (is_border_op(op, gi))
            result.push_back(op);
    return result;
}

std::set<size_t> Partition::boundary_neighbors(size_t gi) const {
    std::set<size_t> result;
    for (auto op : groups[gi].ops)
        for (auto v : dag->op_neighbors[op])
            if (!groups[gi].ops.count(v))
                result.insert(v);
    return result;
}

std::set<size_t> Partition::adjacent_groups(size_t gi) const {
    std::set<size_t> result;
    for (auto op : boundary_neighbors(gi))
        for (auto gj : op_to_groups_[op])
            if (gj != gi) result.insert(gj);
    return result;
}

bool Partition::future_needs(size_t t, const std::vector<bool>& scheduled) const {
    for (size_t i = 0; i < groups.size(); i++) {
        if (!groups[i].alive || scheduled[i]) continue;
        if (groups[i].sg && groups[i].sg->boundary_inputs().count(t)) return true;
    }
    return false;
}

// ============================================================================
// MoveDelta: describes a hypothetical partition move for acyclicity checking.
// Passed into kahn_with_delta so we can check acyclicity WITHOUT copying
// op_to_groups_.
// ============================================================================

struct MoveDelta {
    enum Type { NONE, MERGE_PAIR, MERGE_MULTI, STEAL, RECOMPUTE } type = NONE;
    size_t op = SIZE_MAX;
    size_t ga = SIZE_MAX;  // source group (dies in MERGE_PAIR, loses op in STEAL)
    size_t gb = SIZE_MAX;  // target group (absorbs in MERGE_PAIR, gains op in STEAL/RECOMPUTE)
    const std::vector<size_t>* merge_list = nullptr;  // for MERGE_MULTI
};

// ============================================================================
// kahn_with_delta: zero-allocation Kahn's algorithm with virtual group mapping.
//
// Instead of copying op_to_groups_ and mutating it, we use a lambda that
// iterates over an op's groups AS IF the move had been applied. Dependencies
// are tracked with flat vectors (no std::set, no std::deque).
//
// OR-node semantics: when a tensor is produced by multiple groups (recompute),
// any ONE producer satisfying the dependency unlocks the consumer. This is
// tracked via dep_met[] booleans and unsatisfied_deps[] counters.
// ============================================================================

static bool kahn_with_delta(
    const Problem& prob,
    const DAG& dag,
    const std::vector<std::vector<size_t>>& op_to_groups,
    const std::vector<bool>& alive,
    size_t num_alive_groups,
    const MoveDelta& delta)
{
    size_t ng = alive.size();

    // For MERGE_MULTI: build killed lookup once
    std::vector<bool> is_killed;
    size_t survivor = 0;
    if (delta.type == MoveDelta::MERGE_MULTI && delta.merge_list) {
        is_killed.resize(ng, false);
        survivor = (*delta.merge_list)[0];
        for (size_t k = 1; k < delta.merge_list->size(); k++)
            is_killed[(*delta.merge_list)[k]] = true;
    }

    // Lambda: iterate over an op's groups as if the move had been applied
    auto for_virtual_groups = [&](size_t op_idx, auto&& callback) {
        switch (delta.type) {
        case MoveDelta::NONE:
            for (size_t g : op_to_groups[op_idx]) callback(g);
            return;

        case MoveDelta::MERGE_PAIR: {
            bool yielded_survivor = false;
            for (size_t g : op_to_groups[op_idx]) {
                if (g == delta.ga || g == delta.gb) {
                    if (!yielded_survivor) { callback(delta.ga); yielded_survivor = true; }
                } else {
                    callback(g);
                }
            }
            return;
        }
        case MoveDelta::MERGE_MULTI: {
            bool yielded_survivor = false;
            for (size_t g : op_to_groups[op_idx]) {
                if (g == survivor || (g < ng && is_killed[g])) {
                    if (!yielded_survivor) { callback(survivor); yielded_survivor = true; }
                } else {
                    callback(g);
                }
            }
            return;
        }
        case MoveDelta::STEAL:
            if (op_idx == delta.op) {
                bool gb_added = false;
                for (size_t g : op_to_groups[op_idx]) {
                    if (g == delta.ga) continue;  // op removed from ga
                    if (g == delta.gb) gb_added = true;
                    callback(g);
                }
                if (!gb_added) callback(delta.gb);  // op added to gb
                return;
            }
            for (size_t g : op_to_groups[op_idx]) callback(g);
            return;

        case MoveDelta::RECOMPUTE:
            if (op_idx == delta.op) {
                bool gb_added = false;
                for (size_t g : op_to_groups[op_idx]) {
                    if (g == delta.gb) gb_added = true;
                    callback(g);
                }
                if (!gb_added) callback(delta.gb);  // op copied to gb
                return;
            }
            for (size_t g : op_to_groups[op_idx]) callback(g);
            return;
        }
    };

    // 1. Build dependencies using flat vectors
    std::vector<int> unsatisfied_deps(ng, 0);
    std::vector<bool> dep_met;
    std::vector<std::vector<std::pair<size_t, size_t>>> frees(ng);

    for (size_t op_idx = 0; op_idx < prob.ops.size(); op_idx++) {
        for (auto t : prob.ops[op_idx].inputs) {
            int prod = dag.tensor_producer[t];
            if (prod < 0) continue;  // graph input

            for_virtual_groups(op_idx, [&](size_t target_gi) {
                if (!alive[target_gi]) return;

                // Is producer internal to target group?
                bool prod_internal = false;
                for_virtual_groups((size_t)prod, [&](size_t g) {
                    if (g == target_gi) prod_internal = true;
                });
                if (prod_internal) return;

                // External dependency: target_gi needs prod from outside
                size_t dep_id = dep_met.size();
                dep_met.push_back(false);
                unsatisfied_deps[target_gi]++;

                for_virtual_groups((size_t)prod, [&](size_t source_gj) {
                    if (alive[source_gj])
                        frees[source_gj].push_back({target_gi, dep_id});
                });
            });
        }
    }

    // 2. Kahn's traversal with flat vector queue
    std::vector<size_t> q;
    q.reserve(ng);
    std::vector<bool> enqueued(ng, false);
    size_t visited = 0;

    for (size_t i = 0; i < ng; i++) {
        if (alive[i] && unsatisfied_deps[i] == 0) {
            q.push_back(i);
            enqueued[i] = true;
        }
    }

    size_t head = 0;
    while (head < q.size()) {
        size_t u = q[head++];
        visited++;

        for (auto [gi, dep_id] : frees[u]) {
            if (!dep_met[dep_id]) {
                dep_met[dep_id] = true;
                unsatisfied_deps[gi]--;

                if (unsatisfied_deps[gi] == 0 && !enqueued[gi]) {
                    q.push_back(gi);
                    enqueued[gi] = true;
                }
            }
        }
    }

    return visited >= num_alive_groups;
}

// ============================================================================
// Public acyclicity methods — thin wrappers around kahn_with_delta
// ============================================================================

bool Partition::is_acyclic() const {
    if (!prob || !dag) return true;

    std::vector<bool> alive(groups.size(), false);
    size_t na = 0;
    for (size_t i = 0; i < groups.size(); i++)
        if (groups[i].alive) { alive[i] = true; na++; }

    MoveDelta delta;  // NONE
    return kahn_with_delta(*prob, *dag, op_to_groups_, alive, na, delta);
}

bool Partition::is_acyclic_after_merge(size_t ga, size_t gb) const {
    if (!prob || !dag) return true;

    std::vector<bool> alive(groups.size(), false);
    size_t na = 0;
    for (size_t i = 0; i < groups.size(); i++)
        if (groups[i].alive && i != gb) { alive[i] = true; na++; }

    MoveDelta delta{MoveDelta::MERGE_PAIR, SIZE_MAX, ga, gb, nullptr};
    return kahn_with_delta(*prob, *dag, op_to_groups_, alive, na, delta);
}

bool Partition::is_acyclic_after_merge(const std::vector<size_t>& group_list) const {
    if (!prob || !dag || group_list.size() < 2) return true;

    std::vector<bool> alive(groups.size(), false);
    size_t na = 0;
    std::vector<bool> killed(groups.size(), false);
    for (size_t k = 1; k < group_list.size(); k++)
        killed[group_list[k]] = true;
    for (size_t i = 0; i < groups.size(); i++)
        if (groups[i].alive && !killed[i]) { alive[i] = true; na++; }

    MoveDelta delta{MoveDelta::MERGE_MULTI, SIZE_MAX, SIZE_MAX, SIZE_MAX, &group_list};
    return kahn_with_delta(*prob, *dag, op_to_groups_, alive, na, delta);
}

bool Partition::is_acyclic_after_steal(size_t op, size_t ga, size_t gb) const {
    if (!prob || !dag) return true;

    std::vector<bool> alive(groups.size(), false);
    size_t na = 0;
    for (size_t i = 0; i < groups.size(); i++)
        if (groups[i].alive) { alive[i] = true; na++; }

    // If ga becomes empty after losing op, it virtually dies
    if (groups[ga].ops.size() == 1 && groups[ga].ops.count(op)) {
        alive[ga] = false;
        na--;
    }

    MoveDelta delta{MoveDelta::STEAL, op, ga, gb, nullptr};
    return kahn_with_delta(*prob, *dag, op_to_groups_, alive, na, delta);
}

bool Partition::is_acyclic_after_recompute(size_t op, size_t gb) const {
    if (!prob || !dag) return true;

    std::vector<bool> alive(groups.size(), false);
    size_t na = 0;
    for (size_t i = 0; i < groups.size(); i++)
        if (groups[i].alive) { alive[i] = true; na++; }

    MoveDelta delta{MoveDelta::RECOMPUTE, op, SIZE_MAX, gb, nullptr};
    return kahn_with_delta(*prob, *dag, op_to_groups_, alive, na, delta);
}

double Partition::eval_set(const std::set<size_t>& ops) const {
    if (ops.empty()) return 1e18;
    if (cache) return cache->evaluate(ops, *prob, *dag);
    auto sg = Subgraph::create(*prob, *dag, {ops.begin(), ops.end()});
    if (!sg) return 1e18;
    auto c = sg->best_cost();
    return c.feasible ? c.latency : 1e18;
}

std::set<size_t> Partition::compute_force_ephemeral(size_t gi) const {
    std::set<size_t> result;
    if (!prob || !dag) return result;

    for (auto op : groups[gi].ops) {
        for (auto t : prob->ops[op].outputs) {
            bool consumed_internally = false;
            for (auto cop : dag->tensor_consumers[t])
                if (groups[gi].ops.count(cop)) { consumed_internally = true; break; }
            if (!consumed_internally) continue;

            bool has_external = false;
            for (auto cop : dag->tensor_consumers[t])
                if (!groups[gi].ops.count(cop)) { has_external = true; break; }
            if (!has_external) continue;

            int prod = dag->tensor_producer[t];
            if (prod < 0) continue;

            bool has_stranded = false;
            for (auto cop : dag->tensor_consumers[t]) {
                for (auto gj : groups_of(cop)) {
                    if (gj == gi || !groups[gj].alive) continue;
                    if (!groups[gj].ops.count((size_t)prod)) {
                        has_stranded = true;
                        break;
                    }
                }
                if (has_stranded) break;
            }
            if (!has_stranded) result.insert(t);
        }
    }
    return result;
}

double Partition::eval_group_in_context(size_t gi) const {
    if (!groups[gi].alive) return 0;
    auto fe = compute_force_ephemeral(gi);
    auto sg = Subgraph::create(*prob, *dag,
                  std::vector<size_t>(groups[gi].ops.begin(), groups[gi].ops.end()),
                  fe);
    if (!sg) return 1e18;
    auto c = sg->best_cost();
    return c.feasible ? c.latency : 1e18;
}

// ============================================================================
// Mutation
// ============================================================================

size_t Partition::add_group(std::set<size_t> ops, double cost,
                             std::optional<Subgraph> sg, TileConfig cfg) {
    size_t idx = groups.size();
    Group g;
    g.ops      = std::move(ops);
    g.cost     = cost;
    g.alive    = true;
    g.gen      = 0;
    g.best_cfg = cfg;
    g.sg       = std::move(sg);
    groups.push_back(std::move(g));

    // Update op_to_groups_ incrementally
    if (op_to_groups_.size() < prob->num_ops())
        op_to_groups_.resize(prob->num_ops());
    for (auto op : groups[idx].ops)
        op_to_groups_[op].push_back(idx);

    return idx;
}

// ============================================================================
// Eject / split helpers
// ============================================================================

std::vector<size_t> Partition::ejectable_ops(size_t gi) const {
    if (groups[gi].ops.size() <= 1) return {};
    std::vector<size_t> result;
    for (auto op : groups[gi].ops)
        if (is_border_op(op, gi))
            result.push_back(op);
    return result;
}

std::vector<size_t> Partition::internal_ops(size_t gi) const {
    if (!groups[gi].alive) return {};
    std::vector<size_t> result;
    for (auto op : groups[gi].ops)
        if (!is_border_op(op, gi))
            result.push_back(op);
    return result;
}

// ============================================================================
// Connected components
// ============================================================================

std::vector<std::set<size_t>> Partition::connected_components(
        const std::set<size_t>& ops) const {
    std::vector<std::set<size_t>> components;
    std::vector<bool> in_set(prob->num_ops(), false);
    std::vector<bool> visited(prob->num_ops(), false);
    for (auto op : ops) in_set[op] = true;

    for (auto seed : ops) {
        if (visited[seed]) continue;
        std::set<size_t> comp;
        std::vector<size_t> queue = {seed};
        visited[seed] = true;
        while (!queue.empty()) {
            size_t u = queue.back(); queue.pop_back();
            comp.insert(u);
            for (auto v : dag->op_neighbors[u])
                if (in_set[v] && !visited[v]) { visited[v] = true; queue.push_back(v); }
        }
        components.push_back(std::move(comp));
    }
    return components;
}

// ============================================================================
// Eject evaluation
// ============================================================================

Partition::EjectResult Partition::eval_eject(size_t op, size_t gi) const {
    EjectResult result;
    if (!groups[gi].alive || !groups[gi].ops.count(op)) return result;
    if (groups[gi].ops.size() <= 1) return result;

    std::set<size_t> remainder = groups[gi].ops;
    remainder.erase(op);
    result.remainder_components = connected_components(remainder);

    double total_remainder = 0;
    result.component_costs.reserve(result.remainder_components.size());
    for (auto& comp : result.remainder_components) {
        double c = eval_set(comp);
        if (c >= 1e17) return result;
        result.component_costs.push_back(c);
        total_remainder += c;
    }

    bool op_in_other = false;
    for (auto gj : op_to_groups_[op])
        if (gj != gi) { op_in_other = true; break; }

    result.singleton_cost = op_in_other ? 0.0 : eval_set({op});
    if (!op_in_other && result.singleton_cost >= 1e17) return result;

    result.feasible = true;
    result.saving   = groups[gi].cost - (total_remainder + result.singleton_cost);
    return result;
}

// ============================================================================
// Split evaluation
// ============================================================================

Partition::SplitResult Partition::eval_split(size_t op_a, size_t op_b, size_t gi) const {
    SplitResult result;
    if (!groups[gi].alive) return result;
    if (!groups[gi].ops.count(op_a) || !groups[gi].ops.count(op_b)) return result;
    // No size guard: a 2-op group {A,B} where the single edge is a bridge is
    // a valid split into {A} and {B}. Let the BFS determine feasibility.

    const auto& ops = groups[gi].ops;
    std::vector<bool> visited(prob->num_ops(), false);
    std::vector<size_t> queue = {op_a};
    visited[op_a] = true;

    while (!queue.empty()) {
        size_t u = queue.back(); queue.pop_back();
        for (auto v : dag->op_neighbors[u]) {
            if (!ops.count(v) || visited[v]) continue;
            if ((u == op_a && v == op_b) || (u == op_b && v == op_a)) continue;
            visited[v] = true;
            queue.push_back(v);
        }
    }
    if (visited[op_b]) return result;

    for (auto op : ops) {
        if (visited[op]) result.side_a.insert(op);
        else             result.side_b.insert(op);
    }
    if (result.side_a.empty() || result.side_b.empty()) return result;

    result.cost_a = eval_set(result.side_a);
    if (result.cost_a >= 1e17) return result;
    result.cost_b = eval_set(result.side_b);
    if (result.cost_b >= 1e17) return result;

    result.feasible = true;
    result.saving   = groups[gi].cost - (result.cost_a + result.cost_b);
    return result;
}

// ============================================================================
// Bridge edges
// ============================================================================

std::vector<std::pair<size_t,size_t>> Partition::bridge_edges(size_t gi) const {
    if (!groups[gi].alive || groups[gi].ops.size() < 2) return {};

    std::vector<std::pair<size_t,size_t>> bridges;
    const auto& ops  = groups[gi].ops;
    size_t       nops = prob->num_ops();

    std::vector<bool>   visited(nops, false);
    std::vector<size_t> to_clear;

    // Iterate all edges in op_neighbors (DAG edges + co-consumer edges).
    // The old code only iterated op_succs, which missed co-consumer bridges.
    // A co-consumer bridge is an undirected edge u-v where both ops share a
    // common boundary input tensor, and severing that connection would split
    // the group into two disconnected components.
    // We canonicalise as (min,max) to avoid double-reporting each undirected
    // edge as both (u,v) and (v,u).
    std::set<std::pair<size_t,size_t>> seen_edges;

    for (auto u : ops) {
        for (auto v : dag->op_neighbors[u]) {
            if (!ops.count(v)) continue;
            // Canonicalise undirected edge to avoid duplicates
            auto edge = std::make_pair(std::min(u,v), std::max(u,v));
            if (!seen_edges.insert(edge).second) continue;

            for (auto idx : to_clear) visited[idx] = false;
            to_clear.clear();

            std::vector<size_t> queue = {u};
            visited[u] = true;
            to_clear.push_back(u);
            bool found = false;

            while (!queue.empty() && !found) {
                size_t w = queue.back(); queue.pop_back();
                for (auto x : dag->op_neighbors[w]) {
                    if (!ops.count(x) || visited[x]) continue;
                    if ((w == u && x == v) || (w == v && x == u)) continue;
                    if (x == v) { found = true; break; }
                    visited[x] = true;
                    to_clear.push_back(x);
                    queue.push_back(x);
                }
            }
            if (!found) bridges.push_back({u, v});
        }
    }
    return bridges;
}

// ============================================================================
// Ephemeral gap check
// ============================================================================

bool Partition::creates_ephemeral_gap(const std::set<size_t>& proposed_ops,
                                      size_t exclude_ga,
                                      size_t exclude_gb) const {
    if (!prob || !dag) return false;

    for (auto op : proposed_ops) {
        for (auto t : prob->ops[op].outputs) {
            // Under the new ephemeral rule: T is ephemeral in proposed_ops
            // iff produced internally AND ALL DAG consumers are internal.
            // If ANY consumer is external to proposed_ops, T is a boundary
            // output (materialized) → no gap concern.
            bool all_consumers_internal = true;
            bool any_consumer_internal = false;
            for (auto cop : dag->tensor_consumers[t]) {
                if (proposed_ops.count(cop))
                    any_consumer_internal = true;
                else
                    all_consumers_internal = false;
            }
            if (!any_consumer_internal) continue;  // pure boundary output → no gap
            if (!all_consumers_internal) continue;  // has external consumer → boundary output

            // T would be purely ephemeral → never written to slow memory.
            // Check if T is available as a boundary output from some other group.
            int prod_op = dag->tensor_producer[t];
            if (prod_op < 0) continue;

            bool available_from_other = false;
            for (auto gj : groups_of((size_t)prod_op)) {
                if (!groups[gj].alive) continue;
                if (gj == exclude_ga || gj == exclude_gb) continue;
                // T is a boundary output of gj if gj produces T and
                // NOT all of T's consumers are in gj (new ephemeral rule).
                bool all_in_gj = true;
                for (auto cop : dag->tensor_consumers[t])
                    if (!groups[gj].ops.count(cop)) { all_in_gj = false; break; }
                if (!all_in_gj) { available_from_other = true; break; }
            }
            if (available_from_other) continue;

            // T is ephemeral in every group that produces it.
            // Any consumer in a non-excluded alive group that does NOT
            // recompute the producer is stranded.
            for (auto cop : dag->tensor_consumers[t]) {
                for (auto gj : groups_of(cop)) {
                    if (!groups[gj].alive) continue;
                    if (gj == exclude_ga || gj == exclude_gb) continue;
                    if (groups[gj].ops.count((size_t)prod_op)) continue;
                    return true;
                }
            }
        }
    }
    return false;
}

bool Partition::creates_ephemeral_gap(const std::set<size_t>& proposed_ops,
                                      const std::vector<size_t>& exclude_groups) const {
    if (!prob || !dag) return false;
    std::set<size_t> excluded(exclude_groups.begin(), exclude_groups.end());

    for (auto op : proposed_ops) {
        for (auto t : prob->ops[op].outputs) {
            bool all_consumers_internal = true;
            bool any_consumer_internal = false;
            for (auto cop : dag->tensor_consumers[t]) {
                if (proposed_ops.count(cop))
                    any_consumer_internal = true;
                else
                    all_consumers_internal = false;
            }
            if (!any_consumer_internal) continue;
            if (!all_consumers_internal) continue;  // external consumer → boundary output

            int prod_op = dag->tensor_producer[t];
            if (prod_op < 0) continue;

            bool available_from_other = false;
            for (auto gj : groups_of((size_t)prod_op)) {
                if (!groups[gj].alive || excluded.count(gj)) continue;
                bool all_in_gj = true;
                for (auto cop : dag->tensor_consumers[t])
                    if (!groups[gj].ops.count(cop)) { all_in_gj = false; break; }
                if (!all_in_gj) { available_from_other = true; break; }
            }
            if (available_from_other) continue;

            for (auto cop : dag->tensor_consumers[t]) {
                for (auto gj : groups_of(cop)) {
                    if (!groups[gj].alive || excluded.count(gj)) continue;
                    if (groups[gj].ops.count((size_t)prod_op)) continue;
                    return true;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// split_creates_ephemeral_gap
//
// Checks whether replacing excluded groups with a set of new component
// op-sets creates an ephemeral gap.
//
// Unlike creates_ephemeral_gap (single proposed group vs existing groups),
// this checks multiple sibling components simultaneously.  A tensor might be
// available as a boundary output from a sibling component even if no existing
// group provides it.
// ============================================================================

bool Partition::split_creates_ephemeral_gap(
        const std::vector<std::set<size_t>>& components,
        const std::set<size_t>& excluded) const {
    if (!prob || !dag) return false;

    for (size_t ci = 0; ci < components.size(); ci++) {
        const auto& comp = components[ci];

        for (auto op : comp) {
            for (auto t : prob->ops[op].outputs) {
                // New ephemeral rule: T is ephemeral only if ALL DAG
                // consumers are inside this component.
                bool all_consumers_in_comp = true;
                bool any_consumer_in_comp = false;
                for (auto cop : dag->tensor_consumers[t]) {
                    if (comp.count(cop))
                        any_consumer_in_comp = true;
                    else
                        all_consumers_in_comp = false;
                }
                if (!any_consumer_in_comp) continue;
                if (!all_consumers_in_comp) continue;  // external → boundary output

                int prod_op = dag->tensor_producer[t];
                if (prod_op < 0) continue;

                bool t_available = false;

                // Check sibling components: T is boundary output of cj if
                // cj has the producer and NOT all consumers are in cj.
                for (size_t cj = 0; cj < components.size() && !t_available; cj++) {
                    if (cj == ci) continue;
                    if (!components[cj].count((size_t)prod_op)) continue;
                    bool all_in_cj = true;
                    for (auto cop : dag->tensor_consumers[t])
                        if (!components[cj].count(cop)) { all_in_cj = false; break; }
                    if (!all_in_cj) t_available = true;
                }
                // Check existing non-excluded groups
                for (auto gj : groups_of((size_t)prod_op)) {
                    if (t_available) break;
                    if (!groups[gj].alive || excluded.count(gj)) continue;
                    bool all_in_gj = true;
                    for (auto cop : dag->tensor_consumers[t])
                        if (!groups[gj].ops.count(cop)) { all_in_gj = false; break; }
                    if (!all_in_gj) t_available = true;
                }

                if (t_available) continue;

                for (auto cop : dag->tensor_consumers[t]) {
                    bool served = false;
                    for (size_t cj = 0; cj < components.size() && !served; cj++)
                        if (components[cj].count(cop) && components[cj].count((size_t)prod_op))
                            served = true;
                    if (served) continue;

                    for (auto gj : groups_of(cop)) {
                        if (served) break;
                        if (!groups[gj].alive || excluded.count(gj)) continue;
                        if (groups[gj].ops.count((size_t)prod_op)) served = true;
                    }
                    if (served) continue;

                    return true;
                }
            }
        }
    }
    return false;
}