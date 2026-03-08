#include "partition/partition.h"

Partition Partition::trivial(const Problem& prob, const DAG& dag) {
    Partition p;
    p.prob = &prob;
    p.dag = &dag;
    p.groups.resize(prob.num_ops());
    for (size_t i = 0; i < prob.num_ops(); i++) {
        p.groups[i].ops = {i};
        p.groups[i].alive = true;
        auto sg = Subgraph::create(prob, dag, {i});
        if (sg) {
            auto c = sg->best_cost();
            p.groups[i].cost = c.feasible ? c.latency : 1e18;
        }
    }
    return p;
}

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

std::vector<size_t> Partition::groups_of(size_t op) const {
    std::vector<size_t> result;
    for (size_t i = 0; i < groups.size(); i++)
        if (groups[i].alive && groups[i].ops.count(op))
            result.push_back(i);
    return result;
}

bool Partition::is_border_op(size_t op, size_t gi) const {
    for (auto p : dag->op_preds[op])
        if (!groups[gi].ops.count(p)) return true;
    for (auto s : dag->op_succs[op])
        if (!groups[gi].ops.count(s)) return true;
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
    for (auto op : groups[gi].ops) {
        for (auto p : dag->op_preds[op])
            if (!groups[gi].ops.count(p)) result.insert(p);
        for (auto s : dag->op_succs[op])
            if (!groups[gi].ops.count(s)) result.insert(s);
    }
    return result;
}

std::set<size_t> Partition::adjacent_groups(size_t gi) const {
    std::set<size_t> result;
    auto neighbors = boundary_neighbors(gi);
    for (auto op : neighbors)
        for (size_t gj = 0; gj < groups.size(); gj++)
            if (gj != gi && groups[gj].alive && groups[gj].ops.count(op))
                result.insert(gj);
    return result;
}

double Partition::eval_set(const std::set<size_t>& ops) const {
    auto sg = Subgraph::create(*prob, *dag, {ops.begin(), ops.end()});
    if (!sg) return 1e18;
    auto c = sg->best_cost();
    return c.feasible ? c.latency : 1e18;
}

size_t Partition::add_group(std::set<size_t> ops, double cost) {
    size_t idx = groups.size();
    groups.push_back({std::move(ops), cost, true, 0});
    return idx;
}

std::vector<size_t> Partition::ejectable_ops(size_t gi) const {
    if (groups[gi].ops.size() <= 1) return {};

    std::vector<size_t> result;
    for (auto op : groups[gi].ops) {
        bool on_boundary = false;
        for (auto p : dag->op_preds[op])
            if (!groups[gi].ops.count(p)) { on_boundary = true; break; }
        if (!on_boundary)
            for (auto s : dag->op_succs[op])
                if (!groups[gi].ops.count(s)) { on_boundary = true; break; }
        if (on_boundary)
            result.push_back(op);
    }
    return result;
}

std::vector<std::set<size_t>> Partition::connected_components(
        const std::set<size_t>& ops) const {
    std::vector<std::set<size_t>> components;
    std::set<size_t> visited;

    for (auto seed : ops) {
        if (visited.count(seed)) continue;

        // BFS from seed within ops
        std::set<size_t> comp;
        std::vector<size_t> queue = {seed};
        visited.insert(seed);

        while (!queue.empty()) {
            size_t u = queue.back(); queue.pop_back();
            comp.insert(u);
            for (auto v : dag->op_preds[u]) {
                if (ops.count(v) && !visited.count(v)) {
                    visited.insert(v);
                    queue.push_back(v);
                }
            }
            for (auto v : dag->op_succs[u]) {
                if (ops.count(v) && !visited.count(v)) {
                    visited.insert(v);
                    queue.push_back(v);
                }
            }
        }
        components.push_back(std::move(comp));
    }
    return components;
}

Partition::EjectResult Partition::eval_eject(size_t op, size_t gi) const {
    EjectResult result;

    if (!groups[gi].alive || !groups[gi].ops.count(op)) return result;
    if (groups[gi].ops.size() <= 1) return result;  // can't eject from singleton

    std::set<size_t> remainder = groups[gi].ops;
    remainder.erase(op);

    // Find connected components of remainder
    result.remainder_components = connected_components(remainder);

    // Evaluate each component
    result.component_costs.reserve(result.remainder_components.size());
    double total_remainder_cost = 0;
    for (auto& comp : result.remainder_components) {
        double cost = eval_set(comp);
        if (cost >= 1e17) return result;  // component is invalid (multi-sink etc.)
        result.component_costs.push_back(cost);
        total_remainder_cost += cost;
    }

    // Singleton cost: only needed if op is in exactly this group
    bool op_in_other_groups = false;
    for (size_t i = 0; i < groups.size(); i++) {
        if (i != gi && groups[i].alive && groups[i].ops.count(op)) {
            op_in_other_groups = true;
            break;
        }
    }

    if (op_in_other_groups) {
        result.singleton_cost = 0;  // no singleton needed
    } else {
        result.singleton_cost = eval_set({op});
        if (result.singleton_cost >= 1e17) return result;
    }

    result.feasible = true;
    result.saving = groups[gi].cost - (total_remainder_cost + result.singleton_cost);
    return result;
}

// ============================================================================
// Internal ops: ops with ALL DAG neighbors inside the group
// ============================================================================

std::vector<size_t> Partition::internal_ops(size_t gi) const {
    if (!groups[gi].alive) return {};
    std::vector<size_t> result;
    for (auto op : groups[gi].ops) {
        bool internal = true;
        for (auto p : dag->op_preds[op])
            if (!groups[gi].ops.count(p)) { internal = false; break; }
        if (internal)
            for (auto s : dag->op_succs[op])
                if (!groups[gi].ops.count(s)) { internal = false; break; }
        if (internal)
            result.push_back(op);
    }
    return result;
}

// ============================================================================
// Split evaluation: split group at a DAG edge (bridge detection)
// ============================================================================

Partition::SplitResult Partition::eval_split(size_t op_a, size_t op_b, size_t gi) const {
    SplitResult result;
    if (!groups[gi].alive) return result;
    if (!groups[gi].ops.count(op_a) || !groups[gi].ops.count(op_b)) return result;
    if (groups[gi].ops.size() < 3) return result;  // need at least 3 ops to split non-trivially

    const auto& ops = groups[gi].ops;

    // BFS from op_a in the undirected DAG of ops, skipping edge a↔b
    std::set<size_t> visited;
    std::vector<size_t> queue = {op_a};
    visited.insert(op_a);

    while (!queue.empty()) {
        size_t u = queue.back(); queue.pop_back();
        auto try_add = [&](size_t v) {
            if (!ops.count(v) || visited.count(v)) return;
            // Skip the specific edge a↔b
            if ((u == op_a && v == op_b) || (u == op_b && v == op_a)) return;
            visited.insert(v);
            queue.push_back(v);
        };
        for (auto v : dag->op_preds[u]) try_add(v);
        for (auto v : dag->op_succs[u]) try_add(v);
    }

    // If b is still reachable, edge is not a bridge → no split
    if (visited.count(op_b)) return result;

    // Split into two sides
    result.side_a = visited;
    for (auto op : ops)
        if (!visited.count(op)) result.side_b.insert(op);

    // Both sides must be non-empty and have at least 1 op
    if (result.side_a.empty() || result.side_b.empty()) return result;

    // Evaluate both sides
    result.cost_a = eval_set(result.side_a);
    if (result.cost_a >= 1e17) return result;
    result.cost_b = eval_set(result.side_b);
    if (result.cost_b >= 1e17) return result;

    result.feasible = true;
    result.saving = groups[gi].cost - (result.cost_a + result.cost_b);
    return result;
}

// ============================================================================
// Find all bridge edges within a group
// ============================================================================

std::vector<std::pair<size_t,size_t>> Partition::bridge_edges(size_t gi) const {
    if (!groups[gi].alive || groups[gi].ops.size() < 3) return {};

    std::vector<std::pair<size_t,size_t>> bridges;
    const auto& ops = groups[gi].ops;

    // For each DAG edge within the group, check if it's a bridge
    for (auto u : ops) {
        for (auto v : dag->op_succs[u]) {
            if (!ops.count(v)) continue;

            // Quick BFS: can we reach v from u without using edge u→v?
            std::set<size_t> visited;
            std::vector<size_t> queue = {u};
            visited.insert(u);
            bool found = false;

            while (!queue.empty() && !found) {
                size_t w = queue.back(); queue.pop_back();
                auto try_add = [&](size_t x) {
                    if (!ops.count(x) || visited.count(x)) return;
                    if ((w == u && x == v) || (w == v && x == u)) return;
                    if (x == v) { found = true; return; }
                    visited.insert(x);
                    queue.push_back(x);
                };
                for (auto x : dag->op_preds[w]) try_add(x);
                for (auto x : dag->op_succs[w]) try_add(x);
            }

            if (!found)
                bridges.push_back({u, v});
        }
    }
    return bridges;
}