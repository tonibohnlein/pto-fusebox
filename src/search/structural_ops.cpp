#include "search/structural_ops.h"

namespace structural_ops {

// ============================================================================
// Connected components via BFS on DAG neighbor edges
// ============================================================================

std::vector<FlatSet<size_t>> connected_components(
    const FlatSet<size_t>& ops, const DAG& dag)
{
    std::vector<FlatSet<size_t>> components;
    size_t n = dag.num_ops;

    // Thread-local scratch — reused across calls, no heap allocation after first.
    thread_local std::vector<bool> in_set, visited;
    thread_local std::vector<size_t> to_clear;
    in_set.assign(n, false);
    visited.assign(n, false);
    to_clear.clear();

    for (auto op : ops) { in_set[op] = true; to_clear.push_back(op); }

    for (auto seed : ops) {
        if (visited[seed]) continue;
        FlatSet<size_t> comp;
        std::vector<size_t> queue = {seed};
        visited[seed] = true;
        while (!queue.empty()) {
            size_t u = queue.back(); queue.pop_back();
            comp.insert(u);
            for (auto v : dag.op_neighbors[u])
                if (in_set[v] && !visited[v]) {
                    visited[v] = true;
                    queue.push_back(v);
                }
        }
        components.push_back(std::move(comp));
    }

    // Reset only touched entries
    for (auto op : to_clear) { in_set[op] = false; visited[op] = false; }
    return components;
}

// ============================================================================
// Connectivity check after removing one op
// ============================================================================

bool is_connected_without(const FlatSet<size_t>& ops, size_t rm,
                           const DAG& dag)
{
    if (ops.size() <= 1) return false;
    if (ops.size() == 2) return true;

    size_t seed = SIZE_MAX;
    for (auto op : ops)
        if (op != rm) { seed = op; break; }
    if (seed == SIZE_MAX) return false;

    size_t n = dag.num_ops;
    thread_local std::vector<bool> visited;
    thread_local std::vector<size_t> to_clear;
    visited.assign(n, false);
    to_clear.clear();

    std::vector<size_t> queue = {seed};
    visited[seed] = true;
    to_clear.push_back(seed);
    int count = 1;

    while (!queue.empty()) {
        size_t u = queue.back(); queue.pop_back();
        for (auto v : dag.op_neighbors[u]) {
            if (v != rm && ops.count(v) && !visited[v]) {
                visited[v] = true;
                to_clear.push_back(v);
                queue.push_back(v);
                count++;
            }
        }
    }

    for (auto idx : to_clear) visited[idx] = false;
    return count == (int)ops.size() - 1;
}

// ============================================================================
// Eject analysis
// ============================================================================

EjectAnalysis analyze_eject(size_t op, const FlatSet<size_t>& ops,
                             const DAG& dag)
{
    EjectAnalysis result;
    if (!ops.count(op) || ops.size() <= 1) return result;

    FlatSet<size_t> remainder = ops;
    remainder.erase(op);
    result.remainder_components = connected_components(remainder, dag);
    result.connected = (result.remainder_components.size() == 1);
    return result;
}

// ============================================================================
// Split analysis
// ============================================================================

SplitAnalysis analyze_split(size_t op_a, size_t op_b,
                             const FlatSet<size_t>& ops,
                             const DAG& dag)
{
    SplitAnalysis result;
    result.is_bridge = false;
    if (!ops.count(op_a) || !ops.count(op_b)) return result;

    // BFS from op_a, excluding the direct edge op_a↔op_b
    size_t n = dag.num_ops;
    std::vector<bool> visited(n, false);
    std::vector<size_t> queue = {op_a};
    visited[op_a] = true;

    while (!queue.empty()) {
        size_t u = queue.back(); queue.pop_back();
        for (auto v : dag.op_neighbors[u]) {
            if (!ops.count(v) || visited[v]) continue;
            // Skip the cut edge in both directions
            if ((u == op_a && v == op_b) || (u == op_b && v == op_a)) continue;
            visited[v] = true;
            queue.push_back(v);
        }
    }

    // If op_b is still reachable, the edge is not a bridge
    if (visited[op_b]) return result;

    // Edge is a bridge — partition into two sides
    result.is_bridge = true;
    for (auto op : ops) {
        if (visited[op]) result.side_a.insert(op);
        else             result.side_b.insert(op);
    }
    return result;
}

// ============================================================================
// Bridge edges
// ============================================================================

std::vector<std::pair<size_t, size_t>> bridge_edges(
    const FlatSet<size_t>& ops, const DAG& dag)
{
    std::vector<std::pair<size_t, size_t>> bridges;
    if (ops.size() < 2) return bridges;

    size_t n = dag.num_ops;
    thread_local std::vector<bool> visited;
    if (visited.size() < n) visited.assign(n, false);
    std::vector<size_t> to_clear;
    std::set<std::pair<size_t, size_t>> seen_edges;

    for (auto u : ops) {
        for (auto v : dag.op_neighbors[u]) {
            if (!ops.count(v)) continue;
            auto edge = std::make_pair(std::min(u, v), std::max(u, v));
            if (!seen_edges.insert(edge).second) continue;

            // Reset visited
            for (auto idx : to_clear) visited[idx] = false;
            to_clear.clear();

            // BFS from u, excluding edge u↔v
            std::vector<size_t> queue = {u};
            visited[u] = true;
            to_clear.push_back(u);
            bool found = false;

            while (!queue.empty() && !found) {
                size_t w = queue.back(); queue.pop_back();
                for (auto x : dag.op_neighbors[w]) {
                    if (!ops.count(x) || visited[x]) continue;
                    if ((w == u && x == v) || (w == v && x == u)) continue;
                    if (x == v) { found = true; break; }
                    visited[x] = true;
                    to_clear.push_back(x);
                    queue.push_back(x);
                }
            }
            if (!found) bridges.push_back(edge);
        }
    }
    // Clean up thread-local state
    for (auto idx : to_clear) visited[idx] = false;
    return bridges;
}

} // namespace structural_ops