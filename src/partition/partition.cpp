#include "partition/partition.h"
#include "core/cost_cache.h"
#include <iostream>
#include <queue>

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

    // Map: boundary output tensor → group index (alive groups only)
    for (size_t i = 0; i < ng; i++) {
        if (!groups[i].alive || !groups[i].sg) continue;
        for (auto t : groups[i].sg->boundary_outputs())
            tensor_to_group[t] = i;
    }

    // For each alive group i, for each boundary input tensor t, find which
    // alive group produces t AS A BOUNDARY OUTPUT and add the directed edge
    // producer → i.
    //
    // Critical: only add edges from groups where t is a boundary output, NOT
    // from groups where t is ephemeral (produced + consumed internally).
    // With recomputation, an op can be in multiple groups — t might be
    // ephemeral in one and a boundary output in another. Adding edges from
    // the ephemeral group creates spurious dependencies that can form cycles,
    // causing the topological sort to miss groups ("Op X not covered").
    for (size_t i = 0; i < ng; i++) {
        if (!groups[i].alive || !groups[i].sg) continue;
        for (auto t : groups[i].sg->boundary_inputs()) {
            int prod_op = dag->tensor_producer[t];
            if (prod_op < 0) continue;  // graph input — no producing group
            for (auto gj : op_to_groups_[(size_t)prod_op]) {
                if (!groups[gj].alive || gj == i) continue;
                // Only add edge if gj has t as a boundary output
                if (!groups[gj].sg || !groups[gj].sg->boundary_outputs().count(t))
                    continue;
                if (group_preds[i].insert(gj).second)
                    group_succs[gj].insert(i);
            }
        }
    }

    for (size_t i = 0; i < ng; i++)
        group_in_deg[i] = (int)group_preds[i].size();

    // Diagnostic: detect cycles in the group DAG
    {
        std::vector<int> deg = group_in_deg;
        std::queue<size_t> q;
        size_t visited = 0;
        for (size_t i = 0; i < ng; i++)
            if (groups[i].alive && deg[i] == 0) q.push(i);
        while (!q.empty()) {
            size_t u = q.front(); q.pop();
            visited++;
            for (auto v : group_succs[u])
                if (--deg[v] == 0) q.push(v);
        }
        size_t alive = num_alive();
        if (visited < alive) {
            std::cerr << "    WARNING: group DAG has cycle! "
                      << visited << "/" << alive << " groups reachable\n";
            for (size_t i = 0; i < ng; i++) {
                if (!groups[i].alive || deg[i] == 0) continue;
                std::cerr << "      stuck G" << i << " (in_deg=" << deg[i]
                          << ", ops:";
                for (auto op : groups[i].ops) std::cerr << " " << op;
                std::cerr << ") preds:";
                for (auto p : group_preds[i]) std::cerr << " G" << p;
                std::cerr << "\n";
                // Show which tensors created the edges
                if (groups[i].sg) {
                    for (auto t : groups[i].sg->boundary_inputs()) {
                        int prod_op = dag->tensor_producer[t];
                        if (prod_op < 0) continue;
                        for (auto gj : op_to_groups_[(size_t)prod_op]) {
                            if (!groups[gj].alive || gj == i) continue;
                            if (groups[gj].sg && groups[gj].sg->boundary_outputs().count(t))
                                std::cerr << "        T" << t << " (prod op" << prod_op
                                          << "): G" << gj << " → G" << i << "\n";
                        }
                    }
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

bool Partition::is_acyclic() const {
    if (!prob || !dag) return true;
    size_t ng = groups.size();

    // Build Subgraphs for all alive groups
    std::vector<std::optional<Subgraph>> sgs(ng);
    for (size_t gi = 0; gi < ng; gi++) {
        if (!groups[gi].alive) continue;
        sgs[gi] = Subgraph::create(*prob, *dag,
                      std::vector<size_t>(groups[gi].ops.begin(),
                                          groups[gi].ops.end()));
    }

    // Build group DAG: edge gj→gi if gi needs a boundary input that gj produces
    std::vector<int> in_deg(ng, 0);
    std::vector<std::set<size_t>> succs(ng);

    for (size_t i = 0; i < ng; i++) {
        if (!groups[i].alive || !sgs[i]) continue;
        for (auto t : sgs[i]->boundary_inputs()) {
            int prod_op = dag->tensor_producer[t];
            if (prod_op < 0) continue;
            for (auto gj : groups_of((size_t)prod_op)) {
                if (!groups[gj].alive || gj == i) continue;
                if (!sgs[gj] || !sgs[gj]->boundary_outputs().count(t)) continue;
                if (succs[gj].insert(i).second)
                    in_deg[i]++;
            }
        }
    }

    // Kahn's algorithm
    std::vector<size_t> q;
    for (size_t i = 0; i < ng; i++)
        if (groups[i].alive && in_deg[i] == 0) q.push_back(i);
    size_t visited = 0, qi = 0;
    while (qi < q.size()) {
        size_t u = q[qi++];
        visited++;
        for (auto v : succs[u])
            if (--in_deg[v] == 0) q.push_back(v);
    }
    return visited >= num_alive();
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
            // Only consider tensors produced AND consumed inside this group
            bool consumed_internally = false;
            for (auto cop : dag->tensor_consumers[t])
                if (groups[gi].ops.count(cop)) { consumed_internally = true; break; }
            if (!consumed_internally) continue;

            // Skip if already all-internal (no external consumers)
            bool has_external = false;
            for (auto cop : dag->tensor_consumers[t])
                if (!groups[gi].ops.count(cop)) { has_external = true; break; }
            if (!has_external) continue;

            // T has external consumers. Check if EVERY external consumer is
            // in a group that also contains the producer (recomputes it).
            int prod = dag->tensor_producer[t];
            if (prod < 0) continue;

            bool all_served = true;
            for (auto cop : dag->tensor_consumers[t]) {
                if (groups[gi].ops.count(cop)) continue; // internal
                bool served = false;
                for (auto gj : groups_of(cop)) {
                    if (gj == gi || !groups[gj].alive) continue;
                    if (groups[gj].ops.count((size_t)prod)) { served = true; break; }
                }
                if (!served) { all_served = false; break; }
            }
            if (all_served) result.insert(t);
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