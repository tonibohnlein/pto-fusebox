#include "search/solution_search.h"
#include "partition/partition.h"
#include "postopt/post_opt.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>

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

static std::set<size_t> filter_retain(const std::set<size_t>& retain, const Subgraph& sg) {
    std::set<size_t> result;
    for (auto t : retain)
        if (sg.boundary_inputs().count(t) || sg.boundary_outputs().count(t) ||
            t == sg.sink_tensor())
            result.insert(t);
    return result;
}

// Fast connectivity check
static bool is_connected_without(const std::set<size_t>& ops, size_t remove_op,
                                  const DAG& dag, size_t num_ops) {
    if (ops.size() <= 2) return false;
    size_t seed = SIZE_MAX;
    for (auto op : ops) if (op != remove_op) { seed = op; break; }
    if (seed == SIZE_MAX) return false;
    
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
        for (auto v : dag.op_preds[u])
            if (v != remove_op && ops.count(v) && !visited[v]) {
                visited[v] = true; to_clear.push_back(v); queue.push_back(v); count++;
            }
        for (auto v : dag.op_succs[u])
            if (v != remove_op && ops.count(v) && !visited[v]) {
                visited[v] = true; to_clear.push_back(v); queue.push_back(v); count++;
            }
    }
    for (auto idx : to_clear) visited[idx] = false;
    return count == (int)ops.size() - 1;
}

// ============================================================================
// Mutable solution state with incremental updates
// ============================================================================

struct SolState {
    const Problem* prob;
    const DAG* dag;
    std::vector<ScheduleStep> steps;
    std::vector<std::set<size_t>> ret_entering;
    std::vector<double> cost;
    double total = 0;

    void init(const Problem& p, const DAG& d, std::vector<ScheduleStep> s) {
        prob = &p; dag = &d; steps = std::move(s);
        rebuild_all();
    }

    void rebuild_all() {
        size_t n = steps.size();
        ret_entering.resize(n);
        cost.resize(n);
        total = 0;
        std::set<size_t> cur;
        for (size_t i = 0; i < n; i++) {
            ret_entering[i] = cur;
            auto c = steps[i].subgraph.compute_cost(steps[i].config, cur, steps[i].retain_these);
            cost[i] = c.latency;
            total += c.latency;
            cur = steps[i].retain_these;
        }
    }

    void rebuild_from(size_t idx) {
        size_t n = steps.size();
        ret_entering.resize(n);
        cost.resize(n, 0);
        std::set<size_t> cur;
        if (idx > 0) cur = steps[idx - 1].retain_these;
        total = 0;
        for (size_t i = 0; i < idx; i++) total += cost[i];
        for (size_t i = idx; i < n; i++) {
            ret_entering[i] = cur;
            auto c = steps[i].subgraph.compute_cost(steps[i].config, cur, steps[i].retain_these);
            cost[i] = c.latency;
            total += c.latency;
            cur = steps[i].retain_these;
        }
    }
    
    size_t size() const { return steps.size(); }
};

// ============================================================================
// Per-step best move computation
// ============================================================================

static SolutionMove best_move_for_step(SolState& state, size_t i,
                                        const std::set<size_t>& locked_ops,
                                        const std::set<size_t>& locked_tensors,
                                        const std::set<size_t>& locked_steps,
                                        double floor) {
    SolutionMove best;
    const auto& prob = *state.prob;
    const auto& dag = *state.dag;
    
    if (locked_steps.count(i)) return best;

    // --- STEAL: try moving border ops to/from adjacent steps ---
    for (int dir = -1; dir <= 1; dir += 2) {
        size_t j = (dir == -1) ? (i > 0 ? i - 1 : SIZE_MAX) : i + 1;
        if (j == SIZE_MAX || j >= state.size()) continue;
        
        auto& src = state.steps[i];
        auto& dst = state.steps[j];
        auto src_ops = src.subgraph.ops();
        if (src_ops.size() <= 1) continue;
        
        std::set<size_t> src_set(src_ops.begin(), src_ops.end());
        std::set<size_t> dst_set(dst.subgraph.ops().begin(), dst.subgraph.ops().end());
        
        for (auto op : src_ops) {
            if (locked_ops.count(op)) continue;
            
            bool adj = false;
            for (auto p : dag.op_preds[op]) if (dst_set.count(p)) { adj = true; break; }
            if (!adj) for (auto s : dag.op_succs[op]) if (dst_set.count(s)) { adj = true; break; }
            if (!adj) continue;
            
            if (!is_connected_without(src_set, op, dag, prob.num_ops())) continue;
            
            std::set<size_t> new_src = src_set; new_src.erase(op);
            std::set<size_t> new_dst = dst_set; new_dst.insert(op);
            
            auto sg_src = Subgraph::create(prob, dag, {new_src.begin(), new_src.end()});
            auto sg_dst = Subgraph::create(prob, dag, {new_dst.begin(), new_dst.end()});
            if (!sg_src || !sg_dst) continue;
            
            auto src_ret = filter_retain(src.retain_these, *sg_src);
            auto dst_ret = filter_retain(dst.retain_these, *sg_dst);
            auto c_src = sg_src->best_cost(state.ret_entering[i], src_ret);
            if (!c_src.feasible) continue;
            auto c_dst = sg_dst->best_cost(state.ret_entering[j], dst_ret);
            if (!c_dst.feasible) continue;
            
            double saving = (state.cost[i] + state.cost[j]) - (c_src.latency + c_dst.latency);
            if (saving > best.saving && saving > -floor) {
                best.type = SolutionMove::STEAL;
                best.step_a = i; best.step_b = j; best.op = op;
                best.saving = saving;
            }
        }
    }
    
    // --- SPLIT: try bridge edges ---
    if (state.steps[i].subgraph.ops().size() >= 3 && !locked_steps.count(i)) {
        auto tmp = step_as_partition(prob, dag, state.steps[i]);
        auto bridges = tmp.bridge_edges(0);
        
        for (auto& [op_a, op_b] : bridges) {
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
                
                auto c1 = sg_first->best_cost(state.ret_entering[i], bridge_ret);
                if (!c1.feasible) { bridge_ret.clear(); c1 = sg_first->best_cost(state.ret_entering[i], {}); }
                if (!c1.feasible) continue;
                
                auto second_ret = filter_retain(state.steps[i].retain_these, *sg_second);
                auto c2 = sg_second->best_cost(bridge_ret, second_ret);
                if (!c2.feasible) c2 = sg_second->best_cost(bridge_ret, {});
                if (!c2.feasible) continue;
                
                double saving = state.cost[i] - (c1.latency + c2.latency);
                if (saving > best.saving && saving > -floor) {
                    best.type = SolutionMove::SPLIT;
                    best.step_a = i; best.op = op_a; best.op2 = op_b;
                    best.saving = saving;
                }
            }
        }
    }
    
    // --- MERGE: fuse with next step ---
    if (i + 1 < state.size() && !locked_steps.count(i) && !locked_steps.count(i + 1) &&
        depends_on(state.steps[i], state.steps[i + 1], dag)) {
        auto ops_i = state.steps[i].subgraph.ops();
        auto ops_j = state.steps[i + 1].subgraph.ops();
        std::vector<size_t> merged;
        std::set<size_t> seen;
        for (auto op : ops_i) if (seen.insert(op).second) merged.push_back(op);
        for (auto op : ops_j) if (seen.insert(op).second) merged.push_back(op);
        
        auto sg = Subgraph::create(prob, dag, merged);
        if (sg) {
            auto retain = filter_retain(state.steps[i + 1].retain_these, *sg);
            auto cm = sg->best_cost(state.ret_entering[i], retain);
            if (!cm.feasible) { retain.clear(); cm = sg->best_cost(state.ret_entering[i], {}); }
            if (cm.feasible) {
                double saving = (state.cost[i] + state.cost[i + 1]) - cm.latency;
                if (saving > best.saving && saving > -floor) {
                    best.type = SolutionMove::MERGE;
                    best.step_a = i; best.step_b = i + 1;
                    best.saving = saving;
                }
            }
        }
    }
    
    // --- RETAIN_ADD ---
    if (i + 1 < state.size()) {
        auto& si = state.steps[i];
        auto& sj = state.steps[i + 1];
        std::set<size_t> cands;
        for (auto t : si.subgraph.boundary_inputs())
            if (sj.subgraph.boundary_inputs().count(t)) cands.insert(t);
        size_t sink = si.subgraph.sink_tensor();
        if (sj.subgraph.boundary_inputs().count(sink)) cands.insert(sink);
        for (auto t : state.ret_entering[i])
            if (!si.retain_these.count(t) && sj.subgraph.boundary_inputs().count(t))
                if (si.subgraph.boundary_inputs().count(t) ||
                    si.subgraph.boundary_outputs().count(t) || t == sink)
                    cands.insert(t);
        
        for (auto t : cands) {
            if (si.retain_these.count(t) || locked_tensors.count(t)) continue;
            if (!prob.retainable_tensors.count(t)) continue;
            auto new_ret = si.retain_these; new_ret.insert(t);
            auto new_ent_j = state.ret_entering[i + 1]; new_ent_j.insert(t);
            auto ci = si.subgraph.best_cost(state.ret_entering[i], new_ret);
            if (!ci.feasible) continue;
            auto cj = sj.subgraph.best_cost(new_ent_j, sj.retain_these);
            if (!cj.feasible) continue;
            double saving = (state.cost[i] + state.cost[i + 1]) - (ci.latency + cj.latency);
            if (saving > best.saving && saving > -floor) {
                best.type = SolutionMove::RETAIN_ADD;
                best.step_a = i; best.tensor = t; best.saving = saving;
            }
        }
    }
    
    // --- RETAIN_REMOVE ---
    if (i + 1 < state.size()) {
        auto& si = state.steps[i];
        auto& sj = state.steps[i + 1];
        for (auto t : si.retain_these) {
            if (locked_tensors.count(t)) continue;
            auto new_ret = si.retain_these; new_ret.erase(t);
            auto new_ent_j = state.ret_entering[i + 1]; new_ent_j.erase(t);
            auto ci = si.subgraph.best_cost(state.ret_entering[i], new_ret);
            if (!ci.feasible) continue;
            auto cj = sj.subgraph.best_cost(new_ent_j, sj.retain_these);
            if (!cj.feasible) continue;
            double saving = (state.cost[i] + state.cost[i + 1]) - (ci.latency + cj.latency);
            if (saving > best.saving && saving > -floor) {
                best.type = SolutionMove::RETAIN_REMOVE;
                best.step_a = i; best.tensor = t; best.saving = saving;
            }
        }
    }
    
    return best;
}

// ============================================================================
// Apply a move, return affected step range [lo, hi) for incremental update
// ============================================================================

static std::pair<size_t,size_t> apply_move(SolState& state, const SolutionMove& m) {
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
        auto src_ret = filter_retain(src.retain_these, *sg_src);
        auto dst_ret = filter_retain(dst.retain_these, *sg_dst);
        auto c_src = sg_src->best_cost(state.ret_entering[m.step_a], src_ret);
        auto c_dst = sg_dst->best_cost(state.ret_entering[m.step_b], dst_ret);
        if (!c_src.feasible || !c_dst.feasible) return {SIZE_MAX, 0};
        src.subgraph = std::move(*sg_src); src.config = c_src.config; src.retain_these = src_ret;
        dst.subgraph = std::move(*sg_dst); dst.config = c_dst.config; dst.retain_these = dst_ret;
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
        auto c_a = sg_a->best_cost(state.ret_entering[m.step_a], bridge_ret);
        if (!c_a.feasible) { bridge_ret.clear(); c_a = sg_a->best_cost(state.ret_entering[m.step_a], {}); }
        auto valid_ret = filter_retain(state.steps[m.step_a].retain_these, *sg_b);
        auto c_b = sg_b->best_cost(bridge_ret, valid_ret);
        if (!c_b.feasible) c_b = sg_b->best_cost(bridge_ret, {});
        if (!c_a.feasible || !c_b.feasible) return {SIZE_MAX, 0};
        
        state.steps[m.step_a].subgraph = std::move(*sg_a);
        state.steps[m.step_a].config = c_a.config;
        state.steps[m.step_a].retain_these = bridge_ret;
        ScheduleStep ns; ns.subgraph = std::move(*sg_b); ns.config = c_b.config; ns.retain_these = valid_ret;
        state.steps.insert(state.steps.begin() + m.step_a + 1, std::move(ns));
        return {m.step_a, state.size()};
    }
    case SolutionMove::MERGE: {
        if (m.step_b >= state.size()) return {SIZE_MAX, 0};
        auto ops_i = state.steps[m.step_a].subgraph.ops();
        auto ops_j = state.steps[m.step_b].subgraph.ops();
        std::vector<size_t> merged; std::set<size_t> seen;
        for (auto op : ops_i) if (seen.insert(op).second) merged.push_back(op);
        for (auto op : ops_j) if (seen.insert(op).second) merged.push_back(op);
        auto sg = Subgraph::create(prob, dag, merged);
        if (!sg) return {SIZE_MAX, 0};
        auto retain = filter_retain(state.steps[m.step_b].retain_these, *sg);
        auto cm = sg->best_cost(state.ret_entering[m.step_a], retain);
        if (!cm.feasible) { retain.clear(); cm = sg->best_cost(state.ret_entering[m.step_a], {}); }
        if (!cm.feasible) return {SIZE_MAX, 0};
        state.steps[m.step_a].subgraph = std::move(*sg);
        state.steps[m.step_a].config = cm.config;
        state.steps[m.step_a].retain_these = retain;
        state.steps.erase(state.steps.begin() + m.step_b);
        return {m.step_a, state.size()};
    }
    case SolutionMove::RETAIN_ADD: {
        auto& si = state.steps[m.step_a];
        auto new_ret = si.retain_these; new_ret.insert(m.tensor);
        auto ci = si.subgraph.best_cost(state.ret_entering[m.step_a], new_ret);
        if (!ci.feasible) return {SIZE_MAX, 0};
        si.retain_these = new_ret; si.config = ci.config;
        return {m.step_a, state.size()};
    }
    case SolutionMove::RETAIN_REMOVE: {
        auto& si = state.steps[m.step_a];
        auto new_ret = si.retain_these; new_ret.erase(m.tensor);
        auto ci = si.subgraph.best_cost(state.ret_entering[m.step_a], new_ret);
        if (!ci.feasible) return {SIZE_MAX, 0};
        si.retain_these = new_ret; si.config = ci.config;
        return {m.step_a, state.size()};
    }
    default: return {SIZE_MAX, 0};
    }
}

// ============================================================================
// Inner FM pass: active set + locking + drift control
// ============================================================================

SolutionFMPassResult solution_fm_pass(const Problem& prob, const DAG& dag,
                                       std::vector<ScheduleStep> steps,
                                       const SolutionFMPassConfig& cfg) {
    SolState state;
    state.init(prob, dag, std::move(steps));
    
    SolutionFMPassResult result;
    result.start_cost = state.total;
    result.best_cost = state.total;
    result.best_steps = state.steps;
    
    double floor = result.start_cost * cfg.floor_fraction;
    double max_drift = result.start_cost * cfg.max_drift_fraction;
    
    // Active set: best move per step, indexed by step
    std::vector<SolutionMove> active(state.size());
    std::vector<bool> active_valid(state.size(), false);
    std::set<size_t> locked_ops, locked_tensors, locked_steps;
    
    // Initialize: compute best move for random subset of steps
    std::vector<size_t> all_steps;
    for (size_t i = 0; i < state.size(); i++) all_steps.push_back(i);
    std::mt19937 rng(cfg.seed);
    std::shuffle(all_steps.begin(), all_steps.end(), rng);
    int init_count = std::max(1, (int)state.size() / 2);
    for (int k = 0; k < init_count && k < (int)all_steps.size(); k++) {
        size_t i = all_steps[k];
        active[i] = best_move_for_step(state, i, locked_ops, locked_tensors, locked_steps, floor);
        active_valid[i] = true;
    }
    
    double cumul_gain = 0, best_cumul_gain = 0;
    
    for (int iter = 0; iter < 200; iter++) {
        // Pop best move from active set
        SolutionMove best;
        size_t best_idx = SIZE_MAX;
        for (size_t i = 0; i < state.size(); i++) {
            if (!active_valid[i]) continue;
            if (active[i].valid() && active[i].saving > best.saving) {
                best = active[i];
                best_idx = i;
            }
        }
        
        if (!best.valid() || best.saving < -floor) break;
        
        // Apply
        auto [lo, hi] = apply_move(state, best);
        if (lo == SIZE_MAX) {
            active_valid[best_idx] = false;
            continue;
        }
        
        result.moves_applied++;
        state.rebuild_from(lo);
        
        cumul_gain = result.start_cost - state.total;
        if (state.total < result.best_cost - 0.01) {
            result.best_cost = state.total;
            result.best_steps = state.steps;
            best_cumul_gain = cumul_gain;
        }
        
        if (best_cumul_gain - cumul_gain > max_drift) break;
        
        // Lock
        if (best.op != SIZE_MAX) locked_ops.insert(best.op);
        if (best.op2 != SIZE_MAX) locked_ops.insert(best.op2);
        if (best.tensor != SIZE_MAX) locked_tensors.insert(best.tensor);
        locked_steps.insert(best.step_a);
        if (best.step_b != SIZE_MAX && best.step_b < state.size())
            locked_steps.insert(best.step_b);
        
        // Invalidate and re-activate affected neighborhood
        size_t act_lo = (lo > 0) ? lo - 1 : 0;
        size_t act_hi = std::min(hi + 1, state.size());
        // Resize active set if steps were added/removed
        active.resize(state.size());
        active_valid.resize(state.size(), false);
        
        for (size_t i = act_lo; i < act_hi; i++) {
            active[i] = best_move_for_step(state, i, locked_ops, locked_tensors, locked_steps, floor);
            active_valid[i] = true;
        }
    }
    
    result.end_steps = state.steps;
    result.end_cost = state.total;
    return result;
}

// ============================================================================
// Greedy hill climb: no locking, floor=0, run until no improvement
// ============================================================================

std::vector<ScheduleStep> solution_greedy_descent(const Problem& prob, const DAG& dag,
                                                    std::vector<ScheduleStep> steps) {
    SolState state;
    state.init(prob, dag, std::move(steps));
    
    std::set<size_t> empty_set;
    
    for (int iter = 0; iter < 200; iter++) {
        SolutionMove best;
        for (size_t i = 0; i < state.size(); i++) {
            auto m = best_move_for_step(state, i, empty_set, empty_set, empty_set, 0.0);
            if (m.valid() && m.saving > best.saving) best = m;
        }
        
        if (!best.valid() || best.saving <= 0.01) break;
        
        auto [lo, hi] = apply_move(state, best);
        if (lo == SIZE_MAX) break;
        state.rebuild_from(lo);
    }
    
    return state.steps;
}

// ============================================================================
// Outer loop: passes + greedy-kick + adaptive cooling
// ============================================================================

Solution solution_fm_search(const Problem& prob, const DAG& dag,
                             Solution sol, const SolutionFMConfig& cfg) {
    auto best_steps = sol.steps();
    double best_cost = sol.total_latency();
    
    int no_improve = 0;
    double base_floor = cfg.pass_config.floor_fraction;
    double base_drift = cfg.pass_config.max_drift_fraction;
    double heat = 1.0;
    
    for (int pass = 0; pass < cfg.max_passes; pass++) {
        if (Clock::now() >= cfg.deadline) break;
        if (no_improve >= cfg.max_no_improve) break;
        
        double progress = (double)pass / std::max(1, cfg.max_passes - 1);
        double temperature = 0.1 + 0.9 * 0.5 * (1.0 + std::cos(progress * M_PI));
        
        double eff_floor = std::clamp(base_floor * temperature * heat, 0.02, 1.0);
        double eff_drift = std::clamp(base_drift * temperature * heat, 0.05, 2.0);
        
        SolutionFMPassConfig pc = cfg.pass_config;
        pc.seed = (unsigned)(pc.seed + pass * 7);
        pc.floor_fraction = eff_floor;
        pc.max_drift_fraction = eff_drift;
        
        auto pr = solution_fm_pass(prob, dag, best_steps, pc);
        
        if (pr.best_cost < best_cost - 0.01) {
            best_cost = pr.best_cost;
            best_steps = std::move(pr.best_steps);
            no_improve = 0;
            heat = std::clamp(heat * 0.7, 0.1, 3.0);
        } else {
            // Greedy-kick on perturbed end state
            if (pr.moves_applied > 0) {
                auto kicked = solution_greedy_descent(prob, dag, std::move(pr.end_steps));
                Solution kicked_sol(prob, dag, kicked);
                kicked_sol = optimize_retain(prob, dag, std::move(kicked_sol));
                if (kicked_sol.total_latency() < best_cost - 0.01) {
                    best_cost = kicked_sol.total_latency();
                    best_steps = kicked_sol.steps();
                    no_improve = 0;
                    heat = std::clamp(heat * 0.9, 0.1, 3.0);
                    continue;
                }
            }
            no_improve++;
            heat = std::clamp(heat * 1.3, 0.1, 3.0);
        }
    }
    
    return Solution(prob, dag, std::move(best_steps));
}