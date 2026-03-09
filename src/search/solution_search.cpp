#include "search/solution_search.h"
#include "partition/partition.h"
#include "postopt/post_opt.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <chrono>

using Clock = std::chrono::steady_clock;

// ============================================================================
// Helpers
// ============================================================================

static Partition step_as_partition(const Problem& prob, const DAG& dag,
                                    const ScheduleStep& step) {
    Partition p;
    p.prob = &prob;
    p.dag = &dag;
    std::set<size_t> ops(step.subgraph.ops().begin(), step.subgraph.ops().end());
    p.groups.push_back({ops, 0, true, 0});
    return p;
}

static bool depends_on(const ScheduleStep& si, const ScheduleStep& sj,
                        const DAG& dag) {
    for (auto t : sj.subgraph.boundary_inputs()) {
        int prod = dag.tensor_producer[t];
        if (prod < 0) continue;
        for (auto op : si.subgraph.ops())
            if ((size_t)prod == op) return true;
    }
    return false;
}

// ============================================================================
// Mutable solution state for FM search
//
// Maintains incremental data structures to avoid full rebuilds:
//   - retained_entering[i]: what's retained entering step i
//   - step_cost[i]: current latency of step i
//   - bridge_cache[i]: cached bridge edges (invalidated on op changes)
//   - moves[i]: cached best move per step (invalidated on changes)
// ============================================================================

struct SolutionState {
    const Problem* prob;
    const DAG* dag;
    
    std::vector<ScheduleStep> steps;
    std::vector<std::set<size_t>> retained_entering;
    std::vector<double> step_cost;
    double total_cost = 0;
    
    // Lazy caches
    std::vector<std::vector<std::pair<size_t,size_t>>> bridge_cache;
    std::vector<bool> bridge_valid;
    
    // Locked elements
    std::set<size_t> locked_ops;
    std::set<size_t> locked_tensors;
    std::set<size_t> locked_steps;
    
    // Ensure all caches match steps.size()
    void sync_caches() {
        size_t n = steps.size();
        if (bridge_cache.size() < n) bridge_cache.resize(n);
        if (bridge_valid.size() < n) bridge_valid.resize(n, false);
        if (retained_entering.size() < n) retained_entering.resize(n);
        if (step_cost.size() < n) step_cost.resize(n, 0);
    }
    
    void init(const Problem& p, const DAG& d, std::vector<ScheduleStep> s) {
        prob = &p;
        dag = &d;
        steps = std::move(s);
        size_t n = steps.size();
        retained_entering.resize(n);
        step_cost.resize(n);
        bridge_cache.resize(n);
        bridge_valid.assign(n, false);
        rebuild_all();
    }
    
    void rebuild_all() {
        size_t n = steps.size();
        retained_entering.resize(n);
        step_cost.resize(n);
        bridge_cache.resize(n);
        bridge_valid.assign(n, false);
        total_cost = 0;
        
        std::set<size_t> currently_retained;
        for (size_t i = 0; i < n; i++) {
            retained_entering[i] = currently_retained;
            auto c = steps[i].subgraph.compute_cost(
                steps[i].config, currently_retained, steps[i].retain_these);
            step_cost[i] = c.latency;
            total_cost += c.latency;
            currently_retained = steps[i].retain_these;
        }
    }
    
    // Rebuild retained_entering and costs from step idx onwards
    void rebuild_from(size_t idx) {
        sync_caches();
        size_t n = steps.size();
        std::set<size_t> currently_retained;
        if (idx > 0) currently_retained = steps[idx - 1].retain_these;
        
        total_cost = 0;
        for (size_t i = 0; i < idx; i++) total_cost += step_cost[i];
        
        for (size_t i = idx; i < n; i++) {
            retained_entering[i] = currently_retained;
            auto c = steps[i].subgraph.compute_cost(
                steps[i].config, currently_retained, steps[i].retain_these);
            step_cost[i] = c.latency;
            total_cost += c.latency;
            currently_retained = steps[i].retain_these;
        }
    }
    
    // Get bridge edges for a step (cached)
    const std::vector<std::pair<size_t,size_t>>& get_bridges(size_t i) {
        if (i >= bridge_valid.size() || !bridge_valid[i]) {
            if (i >= bridge_cache.size()) {
                bridge_cache.resize(i + 1);
                bridge_valid.resize(i + 1, false);
            }
            bridge_cache[i].clear();
            if (steps[i].subgraph.ops().size() >= 3) {
                Partition tmp;
                tmp.prob = prob;
                tmp.dag = dag;
                auto ops = steps[i].subgraph.ops();
                std::set<size_t> op_set(ops.begin(), ops.end());
                tmp.groups.push_back({op_set, 0, true, 0});
                bridge_cache[i] = tmp.bridge_edges(0);
            }
            bridge_valid[i] = true;
        }
        return bridge_cache[i];
    }
    
    // Invalidate caches for affected steps
    void invalidate_bridges(size_t i) {
        if (i < bridge_valid.size()) bridge_valid[i] = false;
    }
    
    size_t size() const { return steps.size(); }
};

// ============================================================================
// Quick connectivity check: can we remove op from a set and stay connected?
// Uses vector<bool> — no allocations.
// ============================================================================

static bool is_connected_without(const std::set<size_t>& ops, size_t remove_op,
                                  const DAG& dag, size_t num_ops) {
    if (ops.size() <= 2) return false;  // removing leaves 1 or 0
    
    // Find seed (first op that isn't remove_op)
    size_t seed = SIZE_MAX;
    for (auto op : ops)
        if (op != remove_op) { seed = op; break; }
    if (seed == SIZE_MAX) return false;
    
    // BFS
    thread_local std::vector<bool> visited;
    thread_local std::vector<size_t> to_clear;
    if (visited.size() < num_ops) visited.resize(num_ops, false);
    
    to_clear.clear();
    std::vector<size_t> queue = {seed};
    visited[seed] = true;
    to_clear.push_back(seed);
    int count = 1;
    
    while (!queue.empty()) {
        size_t u = queue.back(); queue.pop_back();
        for (auto v : dag.op_preds[u]) {
            if (v != remove_op && ops.count(v) && !visited[v]) {
                visited[v] = true;
                to_clear.push_back(v);
                queue.push_back(v);
                count++;
            }
        }
        for (auto v : dag.op_succs[u]) {
            if (v != remove_op && ops.count(v) && !visited[v]) {
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
// Move generators — operate on SolutionState
// ============================================================================

static SolutionMove gen_steal(SolutionState& state, size_t src_idx, size_t dst_idx) {
    SolutionMove best;
    const auto& prob = *state.prob;
    const auto& dag = *state.dag;
    if (src_idx >= state.size() || dst_idx >= state.size()) return best;
    if (src_idx == dst_idx) return best;
    if (src_idx + 1 != dst_idx && dst_idx + 1 != src_idx) return best;

    auto& src = state.steps[src_idx];
    auto& dst = state.steps[dst_idx];
    auto src_ops = src.subgraph.ops();
    if (src_ops.size() <= 1) return best;

    std::set<size_t> dst_op_set(dst.subgraph.ops().begin(), dst.subgraph.ops().end());
    std::set<size_t> src_op_set(src_ops.begin(), src_ops.end());

    for (auto op : src_ops) {
        if (state.locked_ops.count(op)) continue;

        // Quick check: is op adjacent to dst?
        bool adj = false;
        for (auto p : dag.op_preds[op])
            if (dst_op_set.count(p)) { adj = true; break; }
        if (!adj) for (auto s : dag.op_succs[op])
            if (dst_op_set.count(s)) { adj = true; break; }
        if (!adj) continue;

        // Quick-reject: connectivity check WITHOUT Subgraph::create
        if (!is_connected_without(src_op_set, op, dag, prob.num_ops()))
            continue;

        // Now do the expensive evaluation
        std::set<size_t> new_src = src_op_set;
        new_src.erase(op);
        std::set<size_t> new_dst = dst_op_set;
        new_dst.insert(op);

        auto sg_src = Subgraph::create(prob, dag, {new_src.begin(), new_src.end()});
        auto sg_dst = Subgraph::create(prob, dag, {new_dst.begin(), new_dst.end()});
        if (!sg_src || !sg_dst) continue;

        auto c_src = sg_src->best_cost(state.retained_entering[src_idx], src.retain_these);
        if (!c_src.feasible) continue;
        auto c_dst = sg_dst->best_cost(state.retained_entering[dst_idx], dst.retain_these);
        if (!c_dst.feasible) continue;

        double old_cost = state.step_cost[src_idx] + state.step_cost[dst_idx];
        double new_cost = c_src.latency + c_dst.latency;
        double saving = old_cost - new_cost;

        if (saving > best.saving) {
            best.type = SolutionMove::STEAL; best.step_a = src_idx; best.step_b = dst_idx; best.op = op; best.saving = saving;
        }
    }
    return best;
}

static SolutionMove gen_split(SolutionState& state, size_t step_idx) {
    SolutionMove best;
    const auto& prob = *state.prob;
    const auto& dag = *state.dag;
    if (state.locked_steps.count(step_idx)) return best;
    if (step_idx >= state.size()) return best;
    if (state.steps[step_idx].subgraph.ops().size() < 3) return best;

    auto& bridges = state.get_bridges(step_idx);
    auto& step = state.steps[step_idx];

    for (auto& [op_a, op_b] : bridges) {
        Partition tmp;
        tmp.prob = &prob;
        tmp.dag = &dag;
        auto ops = step.subgraph.ops();
        std::set<size_t> op_set(ops.begin(), ops.end());
        tmp.groups.push_back({op_set, 0, true, 0});
        auto sr = tmp.eval_split(op_a, op_b, 0);
        if (!sr.feasible) continue;

        auto sg_a = Subgraph::create(prob, dag, {sr.side_a.begin(), sr.side_a.end()});
        auto sg_b = Subgraph::create(prob, dag, {sr.side_b.begin(), sr.side_b.end()});
        if (!sg_a || !sg_b) continue;

        for (int order = 0; order < 2; order++) {
            auto& sg_first = (order == 0) ? sg_a : sg_b;
            auto& sg_second = (order == 0) ? sg_b : sg_a;

            std::set<size_t> bridge_ret;
            for (auto t : sg_first->boundary_outputs())
                if (sg_second->boundary_inputs().count(t) &&
                    prob.retainable_tensors.count(t))
                    bridge_ret.insert(t);

            auto c1 = sg_first->best_cost(state.retained_entering[step_idx], bridge_ret);
            if (!c1.feasible) { bridge_ret.clear(); c1 = sg_first->best_cost(state.retained_entering[step_idx], {}); }
            if (!c1.feasible) continue;

            auto c2 = sg_second->best_cost(bridge_ret, step.retain_these);
            if (!c2.feasible) { c2 = sg_second->best_cost(bridge_ret, {}); }
            if (!c2.feasible) continue;

            double saving = state.step_cost[step_idx] - (c1.latency + c2.latency);
            if (saving > best.saving) {
                best.type = SolutionMove::SPLIT; best.step_a = step_idx; best.op = op_a; best.op2 = op_b; best.saving = saving;
            }
        }
    }
    return best;
}

static SolutionMove gen_merge(SolutionState& state, size_t step_idx) {
    SolutionMove best;
    const auto& prob = *state.prob;
    const auto& dag = *state.dag;
    if (step_idx + 1 >= state.size()) return best;
    if (state.locked_steps.count(step_idx) || state.locked_steps.count(step_idx + 1)) return best;
    if (!depends_on(state.steps[step_idx], state.steps[step_idx + 1], dag)) return best;

    auto ops_i = state.steps[step_idx].subgraph.ops();
    auto ops_j = state.steps[step_idx + 1].subgraph.ops();
    std::vector<size_t> merged;
    std::set<size_t> seen;
    for (auto op : ops_i) if (seen.insert(op).second) merged.push_back(op);
    for (auto op : ops_j) if (seen.insert(op).second) merged.push_back(op);

    auto sg = Subgraph::create(prob, dag, merged);
    if (!sg) return best;

    auto retain = state.steps[step_idx + 1].retain_these;
    auto cm = sg->best_cost(state.retained_entering[step_idx], retain);
    if (!cm.feasible) { retain.clear(); cm = sg->best_cost(state.retained_entering[step_idx], {}); }
    if (!cm.feasible) return best;

    double old_cost = state.step_cost[step_idx] + state.step_cost[step_idx + 1];
    best.type = SolutionMove::MERGE; best.step_a = step_idx; best.step_b = step_idx + 1; best.saving = old_cost - cm.latency;
    return best;
}

static SolutionMove gen_retain_add(SolutionState& state, size_t step_idx) {
    SolutionMove best;
    const auto& prob = *state.prob;
    if (step_idx + 1 >= state.size()) return best;
    auto& si = state.steps[step_idx];
    auto& sj = state.steps[step_idx + 1];

    std::vector<size_t> candidates;
    for (auto t : si.subgraph.boundary_inputs())
        if (sj.subgraph.boundary_inputs().count(t)) candidates.push_back(t);
    size_t sink = si.subgraph.sink_tensor();
    if (sj.subgraph.boundary_inputs().count(sink)) candidates.push_back(sink);
    // Extend existing retain chains
    for (auto t : state.retained_entering[step_idx])
        if (!si.retain_these.count(t) && sj.subgraph.boundary_inputs().count(t))
            candidates.push_back(t);

    for (auto t : candidates) {
        if (si.retain_these.count(t)) continue;
        if (state.locked_tensors.count(t)) continue;
        if (!prob.retainable_tensors.count(t)) continue;

        auto new_ret = si.retain_these;
        new_ret.insert(t);
        auto new_ent_j = state.retained_entering[step_idx + 1];
        new_ent_j.insert(t);

        auto ci = si.subgraph.best_cost(state.retained_entering[step_idx], new_ret);
        if (!ci.feasible) continue;
        auto cj = sj.subgraph.best_cost(new_ent_j, sj.retain_these);
        if (!cj.feasible) continue;

        double old_cost = state.step_cost[step_idx] + state.step_cost[step_idx + 1];
        double saving = old_cost - (ci.latency + cj.latency);
        if (saving > best.saving) {
            best.type = SolutionMove::RETAIN_ADD; best.step_a = step_idx; best.tensor = t; best.saving = saving;
        }
    }
    return best;
}

static SolutionMove gen_retain_remove(SolutionState& state, size_t step_idx) {
    SolutionMove best;
    const auto& prob = *state.prob;
    if (step_idx + 1 >= state.size()) return best;
    auto& si = state.steps[step_idx];
    auto& sj = state.steps[step_idx + 1];

    for (auto t : si.retain_these) {
        if (state.locked_tensors.count(t)) continue;

        auto new_ret = si.retain_these;
        new_ret.erase(t);
        auto new_ent_j = state.retained_entering[step_idx + 1];
        new_ent_j.erase(t);

        auto ci = si.subgraph.best_cost(state.retained_entering[step_idx], new_ret);
        if (!ci.feasible) continue;
        auto cj = sj.subgraph.best_cost(new_ent_j, sj.retain_these);
        if (!cj.feasible) continue;

        double old_cost = state.step_cost[step_idx] + state.step_cost[step_idx + 1];
        double saving = old_cost - (ci.latency + cj.latency);
        if (saving > best.saving) {
            best.type = SolutionMove::RETAIN_REMOVE; best.step_a = step_idx; best.tensor = t; best.saving = saving;
        }
    }
    return best;
}

// ============================================================================
// Apply move and return affected step range [lo, hi] for incremental update
// ============================================================================

static std::pair<size_t, size_t> apply_move(SolutionState& state, const SolutionMove& m) {
    const auto& prob = *state.prob;
    const auto& dag = *state.dag;

    switch (m.type) {
    case SolutionMove::STEAL: {
        auto& src = state.steps[m.step_a];
        auto& dst = state.steps[m.step_b];
        std::set<size_t> new_src(src.subgraph.ops().begin(), src.subgraph.ops().end());
        new_src.erase(m.op);
        std::set<size_t> new_dst(dst.subgraph.ops().begin(), dst.subgraph.ops().end());
        new_dst.insert(m.op);

        auto sg_src = Subgraph::create(prob, dag, {new_src.begin(), new_src.end()});
        auto sg_dst = Subgraph::create(prob, dag, {new_dst.begin(), new_dst.end()});
        if (!sg_src || !sg_dst) return {SIZE_MAX, 0};

        auto c_src = sg_src->best_cost(state.retained_entering[m.step_a], src.retain_these);
        auto c_dst = sg_dst->best_cost(state.retained_entering[m.step_b], dst.retain_these);
        if (!c_src.feasible || !c_dst.feasible) return {SIZE_MAX, 0};

        state.steps[m.step_a].subgraph = std::move(*sg_src); state.steps[m.step_a].config = c_src.config; state.steps[m.step_a].retain_these = src.retain_these;
        state.steps[m.step_b].subgraph = std::move(*sg_dst); state.steps[m.step_b].config = c_dst.config; state.steps[m.step_b].retain_these = dst.retain_these;
        state.invalidate_bridges(m.step_a);
        state.invalidate_bridges(m.step_b);
        size_t lo = std::min(m.step_a, m.step_b);
        return {lo, lo + 2};
    }
    case SolutionMove::SPLIT: {
        auto tmp = step_as_partition(prob, dag, state.steps[m.step_a]);
        auto sr = tmp.eval_split(m.op, m.op2, 0);
        if (!sr.feasible) return {SIZE_MAX, 0};

        auto sg_a = Subgraph::create(prob, dag, {sr.side_a.begin(), sr.side_a.end()});
        auto sg_b = Subgraph::create(prob, dag, {sr.side_b.begin(), sr.side_b.end()});
        if (!sg_a || !sg_b) return {SIZE_MAX, 0};

        std::set<size_t> bridge_ret;
        for (auto t : sg_a->boundary_outputs())
            if (sg_b->boundary_inputs().count(t) && prob.retainable_tensors.count(t))
                bridge_ret.insert(t);

        auto c_a = sg_a->best_cost(state.retained_entering[m.step_a], bridge_ret);
        if (!c_a.feasible) { bridge_ret.clear(); c_a = sg_a->best_cost(state.retained_entering[m.step_a], {}); }
        auto c_b = sg_b->best_cost(bridge_ret, state.steps[m.step_a].retain_these);
        if (!c_b.feasible) c_b = sg_b->best_cost(bridge_ret, {});
        if (!c_a.feasible || !c_b.feasible) return {SIZE_MAX, 0};

        auto old_retain = state.steps[m.step_a].retain_these;
        state.steps[m.step_a].subgraph = std::move(*sg_a);
        state.steps[m.step_a].config = c_a.config;
        state.steps[m.step_a].retain_these = bridge_ret;
        
        ScheduleStep new_step;
        new_step.subgraph = std::move(*sg_b);
        new_step.config = c_b.config;
        new_step.retain_these = old_retain;
        state.steps.insert(state.steps.begin() + m.step_a + 1, std::move(new_step));
        
        // Resize caches
        state.bridge_cache.insert(state.bridge_cache.begin() + m.step_a + 1, {});
        state.bridge_valid.insert(state.bridge_valid.begin() + m.step_a + 1, false);
        state.retained_entering.insert(state.retained_entering.begin() + m.step_a + 1, {});
        state.step_cost.insert(state.step_cost.begin() + m.step_a + 1, 0);
        state.invalidate_bridges(m.step_a);
        state.invalidate_bridges(m.step_a + 1);
        return {m.step_a, state.size()};
    }
    case SolutionMove::MERGE: {
        if (m.step_a >= state.steps.size() || m.step_b >= state.steps.size()) return {SIZE_MAX, 0};
        if (m.step_b >= state.bridge_cache.size()) return {SIZE_MAX, 0};
        
        auto ops_i = state.steps[m.step_a].subgraph.ops();
        auto ops_j = state.steps[m.step_b].subgraph.ops();
        std::vector<size_t> merged;
        std::set<size_t> seen;
        for (auto op : ops_i) if (seen.insert(op).second) merged.push_back(op);
        for (auto op : ops_j) if (seen.insert(op).second) merged.push_back(op);

        auto sg = Subgraph::create(prob, dag, merged);
        if (!sg) return {SIZE_MAX, 0};
        auto retain = state.steps[m.step_b].retain_these;
        auto cm = sg->best_cost(state.retained_entering[m.step_a], retain);
        if (!cm.feasible) { retain.clear(); cm = sg->best_cost(state.retained_entering[m.step_a], {}); }
        if (!cm.feasible) return {SIZE_MAX, 0};

        state.steps[m.step_a].subgraph = std::move(*sg); state.steps[m.step_a].config = cm.config; state.steps[m.step_a].retain_these = retain;
        state.steps.erase(state.steps.begin() + m.step_b);
        state.bridge_cache.erase(state.bridge_cache.begin() + m.step_b);
        state.bridge_valid.erase(state.bridge_valid.begin() + m.step_b);
        state.retained_entering.erase(state.retained_entering.begin() + m.step_b);
        state.step_cost.erase(state.step_cost.begin() + m.step_b);
        state.invalidate_bridges(m.step_a);
        return {m.step_a, state.size()};
    }
    case SolutionMove::RETAIN_ADD: {
        auto& si = state.steps[m.step_a];
        auto new_ret = si.retain_these;
        new_ret.insert(m.tensor);
        auto ci = si.subgraph.best_cost(state.retained_entering[m.step_a], new_ret);
        if (!ci.feasible) return {SIZE_MAX, 0};
        si.retain_these = new_ret;
        si.config = ci.config;
        // Rebuild from step_a onwards (retain chain changed)
        return {m.step_a, state.size()};
    }
    case SolutionMove::RETAIN_REMOVE: {
        auto& si = state.steps[m.step_a];
        auto new_ret = si.retain_these;
        new_ret.erase(m.tensor);
        auto ci = si.subgraph.best_cost(state.retained_entering[m.step_a], new_ret);
        if (!ci.feasible) return {SIZE_MAX, 0};
        si.retain_these = new_ret;
        si.config = ci.config;
        return {m.step_a, state.size()};
    }
    default: return {SIZE_MAX, 0};
    }
}

// ============================================================================
// Solution FM search — main loop
// ============================================================================

Solution solution_fm_search(const Problem& prob, const DAG& dag,
                             Solution sol, const SolutionFMConfig& cfg) {
    SolutionState state;
    state.init(prob, dag, sol.steps());
    
    auto best_steps = state.steps;
    double best_cost = state.total_cost;
    int no_improve = 0;

    for (int round = 0; round < cfg.max_rounds; round++) {
        if (no_improve >= cfg.max_no_improve) break;
        if (Clock::now() >= cfg.deadline) break;

        // Reset state to best known
        state.steps = best_steps;
        state.bridge_valid.assign(state.size(), false);
        state.rebuild_all();
        state.locked_ops.clear();
        state.locked_tensors.clear();
        state.locked_steps.clear();

        double round_start = state.total_cost;
        double floor = round_start * cfg.floor_fraction;
        bool round_improved = false;

        for (int iter = 0; iter < 100; iter++) {
            if (Clock::now() >= cfg.deadline) break;

            // Generate best move across all steps
            SolutionMove best_move;

            for (size_t i = 0; i < state.size(); i++) {
                // Steal both directions
                if (i > 0) {
                    auto m = gen_steal(state, i, i - 1);
                    if (m.valid() && m.saving > best_move.saving) best_move = m;
                }
                if (i + 1 < state.size()) {
                    auto m = gen_steal(state, i, i + 1);
                    if (m.valid() && m.saving > best_move.saving) best_move = m;
                }

                // Split
                auto ms = gen_split(state, i);
                if (ms.valid() && ms.saving > best_move.saving) best_move = ms;

                // Merge
                auto mm = gen_merge(state, i);
                if (mm.valid() && mm.saving > best_move.saving) best_move = mm;

                // Retain add/remove
                auto mra = gen_retain_add(state, i);
                if (mra.valid() && mra.saving > best_move.saving) best_move = mra;

                auto mrr = gen_retain_remove(state, i);
                if (mrr.valid() && mrr.saving > best_move.saving) best_move = mrr;
            }

            if (!best_move.valid() || best_move.saving < -floor) break;

            // Apply move
            auto [lo, hi] = apply_move(state, best_move);
            if (lo == SIZE_MAX) break;

            // Incremental cost update
            state.rebuild_from(lo);

            // Lock affected elements
            if (best_move.op != SIZE_MAX) state.locked_ops.insert(best_move.op);
            if (best_move.op2 != SIZE_MAX) state.locked_ops.insert(best_move.op2);
            if (best_move.tensor != SIZE_MAX) state.locked_tensors.insert(best_move.tensor);
            for (size_t s = lo; s < std::min(hi, state.size()); s++)
                state.locked_steps.insert(s);

            // Track best
            if (state.total_cost < best_cost - 0.01) {
                best_steps = state.steps;
                best_cost = state.total_cost;
                round_improved = true;
            }
        }

        if (round_improved) no_improve = 0;
        else no_improve++;
    }

    return Solution(prob, dag, std::move(best_steps));
}