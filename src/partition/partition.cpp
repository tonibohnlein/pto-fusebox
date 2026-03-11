#include "partition/partition.h"
#include "core/cost_cache.h"

// ============================================================================
// Index maintenance
// ============================================================================

void Partition::rebuild_index() {
    op_to_groups_.assign(prob->num_ops(), {});
    for (size_t i = 0; i < groups.size(); i++)
        if (groups[i].alive)
            for (auto op : groups[i].ops)
                op_to_groups_[op].push_back(i);
}

// ============================================================================
// Construction
// ============================================================================

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
    p.rebuild_index();
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
    auto neighbors = boundary_neighbors(gi);
    // Use op_to_groups_ index: O(neighbors) instead of O(neighbors × groups)
    for (auto op : neighbors)
        for (auto gj : op_to_groups_[op])
            if (gj != gi)
                result.insert(gj);
    return result;
}

double Partition::eval_set(const std::set<size_t>& ops) const {
    if (ops.empty()) return 1e18;
    if (cache) return cache->evaluate(ops, *prob, *dag);
    auto sg = Subgraph::create(*prob, *dag, {ops.begin(), ops.end()});
    if (!sg) return 1e18;
    auto c = sg->best_cost();
    return c.feasible ? c.latency : 1e18;
}

size_t Partition::add_group(std::set<size_t> ops, double cost) {
    size_t idx = groups.size();
    groups.push_back({std::move(ops), cost, true, 0});
    // Update index incrementally (avoid full rebuild)
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
// Connected components using precomputed op_neighbors
// ============================================================================

std::vector<std::set<size_t>> Partition::connected_components(
        const std::set<size_t>& ops) const {
    std::vector<std::set<size_t>> components;
    // Use a vector<bool> for O(1) visited checks
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
            for (auto v : dag->op_neighbors[u]) {
                if (in_set[v] && !visited[v]) {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
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

    result.component_costs.reserve(result.remainder_components.size());
    double total_remainder_cost = 0;
    for (auto& comp : result.remainder_components) {
        double cost = eval_set(comp);
        if (cost >= 1e17) return result;
        result.component_costs.push_back(cost);
        total_remainder_cost += cost;
    }

    // Use index to check if op is in other groups
    bool op_in_other_groups = false;
    for (auto gj : op_to_groups_[op]) {
        if (gj != gi) {
            op_in_other_groups = true;
            break;
        }
    }

    if (op_in_other_groups) {
        result.singleton_cost = 0;
    } else {
        result.singleton_cost = eval_set({op});
        if (result.singleton_cost >= 1e17) return result;
    }

    result.feasible = true;
    result.saving = groups[gi].cost - (total_remainder_cost + result.singleton_cost);
    return result;
}

// ============================================================================
// Split evaluation
// ============================================================================

Partition::SplitResult Partition::eval_split(size_t op_a, size_t op_b, size_t gi) const {
    SplitResult result;
    if (!groups[gi].alive) return result;
    if (!groups[gi].ops.count(op_a) || !groups[gi].ops.count(op_b)) return result;
    if (groups[gi].ops.size() < 3) return result;

    const auto& ops = groups[gi].ops;

    // BFS from op_a skipping edge a↔b, using op_neighbors for connectivity
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
        else result.side_b.insert(op);
    }

    if (result.side_a.empty() || result.side_b.empty()) return result;

    result.cost_a = eval_set(result.side_a);
    if (result.cost_a >= 1e17) return result;
    result.cost_b = eval_set(result.side_b);
    if (result.cost_b >= 1e17) return result;

    result.feasible = true;
    result.saving = groups[gi].cost - (result.cost_a + result.cost_b);
    return result;
}

// ============================================================================
// Bridge edges
// ============================================================================

std::vector<std::pair<size_t,size_t>> Partition::bridge_edges(size_t gi) const {
    if (!groups[gi].alive || groups[gi].ops.size() < 3) return {};

    std::vector<std::pair<size_t,size_t>> bridges;
    const auto& ops = groups[gi].ops;
    size_t nops = prob->num_ops();

    std::vector<bool> visited(nops, false);
    std::vector<size_t> to_clear;

    // For each DAG edge within the group, check if it's a bridge
    for (auto u : ops) {
        for (auto v : dag->op_succs[u]) {
            if (!ops.count(v)) continue;

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

            if (!found)
                bridges.push_back({u, v});
        }
    }
    return bridges;
}