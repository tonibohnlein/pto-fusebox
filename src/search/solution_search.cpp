#include "search/solution_search.h"
#include "partition/partition.h"
#include "postopt/post_opt.h"
#include "search/verbose.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using SolMoveHeap = std::priority_queue<SolutionMove>;

// ============================================================================
// Helpers
// ============================================================================

static bool step_depends_on(const ScheduleStep &si, const ScheduleStep &sj,
                            const DAG &dag) {
  for (auto t : sj.subgraph.boundary_inputs()) {
    int prod = dag.tensor_producer[t];
    if (prod < 0)
      continue;
    for (auto op : si.subgraph.ops())
      if ((size_t)prod == op)
        return true;
  }
  return false;
}

static std::set<size_t> filter_retain(const std::set<size_t> &retain,
                                      const Subgraph &sg) {
  std::set<size_t> r;
  for (auto t : retain)
    if (sg.boundary_inputs().count(t) || sg.boundary_outputs().count(t))
      r.insert(t);
  return r;
}

static bool is_connected_without(const std::set<size_t> &ops, size_t rm,
                                 const DAG &dag, size_t n) {
  if (ops.size() <= 2)
    return false;
  size_t seed = SIZE_MAX;
  for (auto op : ops)
    if (op != rm) {
      seed = op;
      break;
    }
  if (seed == SIZE_MAX)
    return false;
  thread_local std::vector<bool> vis;
  thread_local std::vector<size_t> clr;
  if (vis.size() < n)
    vis.assign(n, false);
  clr.clear();
  std::vector<size_t> q = {seed};
  vis[seed] = true;
  clr.push_back(seed);
  int cnt = 1;
  while (!q.empty()) {
    size_t u = q.back();
    q.pop_back();
    for (auto v : dag.op_preds[u])
      if (v != rm && ops.count(v) && !vis[v]) {
        vis[v] = true;
        clr.push_back(v);
        q.push_back(v);
        cnt++;
      }
    for (auto v : dag.op_succs[u])
      if (v != rm && ops.count(v) && !vis[v]) {
        vis[v] = true;
        clr.push_back(v);
        q.push_back(v);
        cnt++;
      }
  }
  for (auto i : clr)
    vis[i] = false;
  return cnt == (int)ops.size() - 1;
}

// Find which step contains an op
static size_t find_step_of(const std::vector<ScheduleStep> &steps, size_t op) {
  for (size_t i = 0; i < steps.size(); i++)
    for (auto o : steps[i].subgraph.ops())
      if (o == op)
        return i;
  return SIZE_MAX;
}

// Find which step(s) have tensor t as boundary input/output
static std::vector<size_t>
find_steps_of_tensor(const std::vector<ScheduleStep> &steps, size_t t) {
  std::vector<size_t> r;
  for (size_t i = 0; i < steps.size(); i++)
    if (steps[i].subgraph.boundary_inputs().count(t) ||
        steps[i].subgraph.boundary_outputs().count(t) ||
        steps[i].retain_these.count(t))
      r.push_back(i);
  return r;
}

// ============================================================================
// Mutable solution state
// ============================================================================

struct SolState {
  const Problem *prob;
  const DAG *dag;
  std::vector<ScheduleStep> steps;
  std::vector<std::set<size_t>> ret_entering;
  std::vector<double> cost;
  std::vector<int> gen; // per-step generation counter
  double total = 0;

  void init(const Problem &p, const DAG &d, std::vector<ScheduleStep> s) {
    prob = &p;
    dag = &d;
    steps = std::move(s);
    gen.assign(steps.size(), 0);
    rebuild_all();
  }
  
  void rebuild_all() {
    size_t n = steps.size();
    ret_entering.resize(n);
    cost.resize(n);
    gen.resize(n, 0);
    total = 0;
    std::set<size_t> cur;
    for (size_t i = 0; i < n; i++) {
      ret_entering[i] = cur;

      // PRUNE useless retentions: only keep if the NEXT step actually needs it
      std::set<size_t> useful_retain;
      for (auto t : steps[i].retain_these) {
        bool valid = steps[i].subgraph.boundary_inputs().count(t) ||
                     steps[i].subgraph.boundary_outputs().count(t);
        bool useful = (i + 1 < n) && steps[i + 1].subgraph.boundary_inputs().count(t);
        if (valid && useful) {
          useful_retain.insert(t);
        }
      }
      steps[i].retain_these = useful_retain;

      auto c = steps[i].subgraph.compute_cost(steps[i].config, cur,
                                              steps[i].retain_these);
      if (c.latency >= 1e17) {
        auto bc = steps[i].subgraph.best_cost(cur, steps[i].retain_these);
        if (bc.feasible) {
          steps[i].config = bc.config;
          c.latency = bc.latency;
        } else {
          auto bc2 = steps[i].subgraph.best_cost(cur, {});
          if (bc2.feasible) {
            steps[i].config = bc2.config;
            steps[i].retain_these.clear();
            c.latency = bc2.latency;
          }
        }
      }
      cost[i] = c.latency;
      total += c.latency;
      cur = steps[i].retain_these;
    }
  }

  void rebuild_from(size_t idx) {
    size_t n = steps.size();
    ret_entering.resize(n);
    cost.resize(n, 0);
    gen.resize(n, 0);
    std::set<size_t> cur;
    if (idx > 0)
      cur = steps[idx - 1].retain_these;
    total = 0;
    for (size_t i = 0; i < idx; i++)
      total += cost[i];
    for (size_t i = idx; i < n; i++) {
      ret_entering[i] = cur;

      // PRUNE useless retentions: only keep if the NEXT step actually needs it
      std::set<size_t> useful_retain;
      for (auto t : steps[i].retain_these) {
        bool valid = steps[i].subgraph.boundary_inputs().count(t) ||
                     steps[i].subgraph.boundary_outputs().count(t);
        bool useful = (i + 1 < n) && steps[i + 1].subgraph.boundary_inputs().count(t);
        if (valid && useful) {
          useful_retain.insert(t);
        }
      }
      steps[i].retain_these = useful_retain;

      auto c = steps[i].subgraph.compute_cost(steps[i].config, cur,
                                              steps[i].retain_these);
      if (c.latency >= 1e17) {
        // Config infeasible with new entering set — find new optimal config
        auto bc = steps[i].subgraph.best_cost(cur, steps[i].retain_these);
        if (bc.feasible) {
          steps[i].config = bc.config;
          c.latency = bc.latency;
        } else {
          // Try without retain
          auto bc2 = steps[i].subgraph.best_cost(cur, {});
          if (bc2.feasible) {
            steps[i].config = bc2.config;
            steps[i].retain_these.clear();
            c.latency = bc2.latency;
          }
          // else: leave as infeasible, will be caught by validation
        }
      }
      cost[i] = c.latency;
      total += c.latency;
      cur = steps[i].retain_these;
    }
  }

  size_t size() const { return steps.size(); }
  void bump(size_t i) {
    if (i < gen.size())
      gen[i]++;
  }
  // Bump all gens — invalidates all stale heap moves (for structural changes)
  void bump_all() {
    for (auto &g : gen)
      g++;
  }
};

// ============================================================================
// Best move for a single OP (partition-style moves)
// ============================================================================

static SolutionMove best_move_for_op(SolState &state, size_t op,
                                     const std::set<size_t> &locked_ops,
                                     double floor) {
  SolutionMove best;
  const auto &prob = *state.prob;
  const auto &dag = *state.dag;
  if (locked_ops.count(op))
    return best;

  size_t si = find_step_of(state.steps, op);
  if (si == SIZE_MAX)
    return best;

  auto si_ops = state.steps[si].subgraph.ops();
  std::set<size_t> si_set(si_ops.begin(), si_ops.end());
  bool is_border = false;
  for (auto p : dag.op_preds[op])
    if (!si_set.count(p)) {
      is_border = true;
      break;
    }
  if (!is_border)
    for (auto s : dag.op_succs[op])
      if (!si_set.count(s)) {
        is_border = true;
        break;
      }

  // --- STEAL: move op to adjacent step ---
  if (is_border && si_set.size() > 1) {
    for (int dir = -1; dir <= 1; dir += 2) {
      size_t sj = (dir == -1) ? (si > 0 ? si - 1 : SIZE_MAX) : si + 1;
      if (sj == SIZE_MAX || sj >= state.size())
        continue;
      auto &dst = state.steps[sj];
      std::set<size_t> dj_set(dst.subgraph.ops().begin(),
                              dst.subgraph.ops().end());
      bool adj = false;
      for (auto p : dag.op_preds[op])
        if (dj_set.count(p)) {
          adj = true;
          break;
        }
      if (!adj)
        for (auto s : dag.op_succs[op])
          if (dj_set.count(s)) {
            adj = true;
            break;
          }
      if (!adj)
        continue;
      if (!is_connected_without(si_set, op, dag, prob.num_ops()))
        continue;


      std::set<size_t> new_src = si_set;
      new_src.erase(op);
      
      bool topo_violation = false;
      if (dir == 1) { 
        // Moving RIGHT (Delaying op): 'op' cannot be a dependency for anything left behind
        for (auto succ : dag.op_succs[op]) {
          if (new_src.count(succ)) { topo_violation = true; break; }
        }
      } else { 
        // Moving LEFT (Advancing op): 'op' cannot depend on anything left behind
        for (auto pred : dag.op_preds[op]) {
          if (new_src.count(pred)) { topo_violation = true; break; }
        }
      }
      if (topo_violation) continue;
      
      std::set<size_t> new_dst = dj_set;
      new_dst.insert(op);
      auto sg_s = Subgraph::create(prob, dag, {new_src.begin(), new_src.end()});
      auto sg_d = Subgraph::create(prob, dag, {new_dst.begin(), new_dst.end()});
      if (!sg_s || !sg_d)
        continue;
      auto sr = filter_retain(state.steps[si].retain_these, *sg_s);
      auto dr = filter_retain(dst.retain_these, *sg_d);
      auto cs = sg_s->best_cost(state.ret_entering[si], sr);
      if (!cs.feasible)
        continue;
      auto cd = sg_d->best_cost(state.ret_entering[sj], dr);
      if (!cd.feasible)
        continue;
      double saving =
          (state.cost[si] + state.cost[sj]) - (cs.latency + cd.latency);
      if (saving > -floor && saving > best.saving) {
        best.type = SolutionMove::STEAL;
        best.step_a = si;
        best.step_b = sj;
        best.op = op;
        best.saving = saving;
      }
    }
  }

  // --- MERGE: fuse step si with adjacent step that op connects to ---
  if (is_border) {
    for (int dir = -1; dir <= 1; dir += 2) {
      size_t sj = (dir == -1) ? (si > 0 ? si - 1 : SIZE_MAX) : si + 1;
      if (sj == SIZE_MAX || sj >= state.size())
        continue;
      if (dir == 1 && !step_depends_on(state.steps[si], state.steps[sj], dag))
        continue;
      if (dir == -1 && !step_depends_on(state.steps[sj], state.steps[si], dag))
        continue;

      auto ops_j = state.steps[sj].subgraph.ops();


      std::vector<size_t> merged;
      std::set<size_t> seen;
      for (auto o : si_ops)
        if (seen.insert(o).second)
          merged.push_back(o);
      for (auto o : ops_j)
        if (seen.insert(o).second)
          merged.push_back(o);
      auto sg = Subgraph::create(prob, dag, merged);
      if (!sg)
        continue;
      size_t lo = std::min(si, sj), hi = std::max(si, sj);
      auto retain = filter_retain(state.steps[hi].retain_these, *sg);
      auto cm = sg->best_cost(state.ret_entering[lo], retain);
      if (!cm.feasible) {
        retain.clear();
        cm = sg->best_cost(state.ret_entering[lo], {});
      }
      if (!cm.feasible)
        continue;
      double saving = (state.cost[si] + state.cost[sj]) - cm.latency;
      if (saving > -floor && saving > best.saving) {
        best.type = SolutionMove::MERGE;
        best.step_a = lo;
        best.step_b = hi;
        best.op = op;
        best.saving = saving;
      }
    }
  }

  // --- RECOMPUTE: duplicate op into adjacent step ---
  if (is_border) {
    for (int dir = -1; dir <= 1; dir += 2) {
      size_t sj = (dir == -1) ? (si > 0 ? si - 1 : SIZE_MAX) : si + 1;
      if (sj == SIZE_MAX || sj >= state.size())
        continue;
      auto sj_ops = state.steps[sj].subgraph.ops();
      bool already_in = false;
      for (auto o : sj_ops)
        if (o == op) {
          already_in = true;
          break;
        }
      if (already_in)
        continue;
      // Check op produces a tensor consumed by step sj
      bool produces_for_sj = false;
      for (auto t : state.steps[sj].subgraph.boundary_inputs())
        if (dag.tensor_producer[t] == (int)op) {
          produces_for_sj = true;
          break;
        }
      if (!produces_for_sj)
        continue;


      std::vector<size_t> expanded(sj_ops.begin(), sj_ops.end());
      expanded.push_back(op);
      auto sg = Subgraph::create(prob, dag, expanded);
      if (!sg)
        continue;
      auto ret = filter_retain(state.steps[sj].retain_these, *sg);
      auto ce = sg->best_cost(state.ret_entering[sj], ret);
      if (!ce.feasible)
        continue;
      double saving = state.cost[sj] - ce.latency;
      if (saving > -floor && saving > best.saving) {
        best.type = SolutionMove::RECOMPUTE;
        best.step_a = sj;
        best.step_b = si;
        best.op = op;
        best.saving = saving;
      }
    }
  }

  // --- EJECT: extract border op to new singleton ---
  if (is_border && si_set.size() >= 2 &&
      is_connected_without(si_set, op, dag, prob.num_ops())) {
    std::set<size_t> remainder = si_set;
    remainder.erase(op);
    auto sg_r =
        Subgraph::create(prob, dag, {remainder.begin(), remainder.end()});
    auto sg_s = Subgraph::create(prob, dag, {op});
    if (sg_r && sg_s) {
      auto rr = filter_retain(state.steps[si].retain_these, *sg_r);
      auto cr = sg_r->best_cost(state.ret_entering[si], rr);
      auto cs = sg_s->best_cost({}, {});
      if (cr.feasible && cs.feasible) {
        double saving = state.cost[si] - (cr.latency + cs.latency);
        if (saving > -floor && saving > best.saving) {
          best.type = SolutionMove::EJECT;
          best.step_a = si;
          best.op = op;
          best.saving = saving;
        }
      }
    }
  }

  // --- INTERNAL_EJECT: extract internal op ---
  if (!is_border && si_set.size() >= 3 && si_set.size() <= 15) {
    Partition tmp;
    tmp.prob = &prob;
    tmp.dag = &dag;
    tmp.groups.push_back({si_set, state.cost[si], true, 0});
    tmp.rebuild_index();
    auto er = tmp.eval_eject(op, 0);
    if (er.feasible && er.saving > -floor && er.saving > best.saving) {
      best.type = SolutionMove::INTERNAL_EJECT;
      best.step_a = si;
      best.op = op;
      best.saving = er.saving;
    }
  }

  // --- SPLIT: at bridge edges incident to op ---
  if (si_set.size() >= 3 && si_set.size() <= 15) {
    auto try_split = [&](size_t a, size_t b) {
      Partition tmp;
      tmp.prob = &prob;
      tmp.dag = &dag;
      tmp.groups.push_back({si_set, 0, true, 0});
      tmp.rebuild_index();
      auto sr = tmp.eval_split(a, b, 0);
      if (!sr.feasible)
        return;
      auto sg_a =
          Subgraph::create(prob, dag, {sr.side_a.begin(), sr.side_a.end()});
      auto sg_b =
          Subgraph::create(prob, dag, {sr.side_b.begin(), sr.side_b.end()});
      if (!sg_a || !sg_b)
        return;
      for (int order = 0; order < 2; order++) {
        auto &first = (order == 0) ? sg_a : sg_b;
        auto &second = (order == 0) ? sg_b : sg_a;
        std::set<size_t> br;
        for (auto t : first->boundary_outputs())
          if (second->boundary_inputs().count(t) &&
              prob.retainable_tensors.count(t))
            br.insert(t);
        auto c1 = first->best_cost(state.ret_entering[si], br);
        if (!c1.feasible) {
          br.clear();
          c1 = first->best_cost(state.ret_entering[si], {});
        }
        if (!c1.feasible)
          continue;
        auto sr2 = filter_retain(state.steps[si].retain_these, *second);
        auto c2 = second->best_cost(br, sr2);
        if (!c2.feasible)
          c2 = second->best_cost(br, {});
        if (!c2.feasible)
          continue;
        double saving = state.cost[si] - (c1.latency + c2.latency);
        if (saving > -floor && saving > best.saving) {
          best.type = SolutionMove::SPLIT;
          best.step_a = si;
          best.op = a;
          best.op2 = b;
          best.saving = saving;
        }
      }
    };
    for (auto s : dag.op_succs[op])
      if (si_set.count(s))
        try_split(op, s);
    for (auto p : dag.op_preds[op])
      if (si_set.count(p))
        try_split(p, op);
  }

  return best;
}

// ============================================================================
// Best move for a single TENSOR (retain-style moves)
// ============================================================================

static SolutionMove best_move_for_tensor(SolState &state, size_t t,
                                         const std::set<size_t> &locked_tensors,
                                         double floor) {
  SolutionMove best;
  const auto &prob = *state.prob;
  if (locked_tensors.count(t))
    return best;
  if (!prob.retainable_tensors.count(t))
    return best;

  // RETAIN_ADD: find step pairs where adding t to retain helps
  for (size_t i = 0; i + 1 < state.size(); i++) {
    auto &si = state.steps[i];
    auto &sj = state.steps[i + 1];
    if (si.retain_these.count(t))
      continue; // already retained

    // Step i must have t as a boundary tensor to retain it
    bool si_boundary = si.subgraph.boundary_inputs().count(t) ||
                       si.subgraph.boundary_outputs().count(t);
    if (!si_boundary)
      continue;
    // Step i+1 must benefit from having t
    if (!sj.subgraph.boundary_inputs().count(t))
      continue;

    auto new_ret = si.retain_these;
    new_ret.insert(t);
    auto new_ent = state.ret_entering[i + 1];
    new_ent.insert(t);
    auto ci = si.subgraph.best_cost(state.ret_entering[i], new_ret);
    if (!ci.feasible)
      continue;
    auto cj = sj.subgraph.best_cost(new_ent, sj.retain_these);
    if (!cj.feasible)
      continue;
    double saving =
        (state.cost[i] + state.cost[i + 1]) - (ci.latency + cj.latency);
    if (saving > -floor && saving > best.saving) {
      best.type = SolutionMove::RETAIN_ADD;
      best.step_a = i;
      best.tensor = t;
      best.saving = saving;
    }
  }

  // RETAIN_REMOVE: find steps where removing t from retain helps
  for (size_t i = 0; i + 1 < state.size(); i++) {
    if (!state.steps[i].retain_these.count(t))
      continue;
    auto &si = state.steps[i];
    auto &sj = state.steps[i + 1];
    auto new_ret = si.retain_these;
    new_ret.erase(t);
    auto new_ent = state.ret_entering[i + 1];
    new_ent.erase(t);
    auto ci = si.subgraph.best_cost(state.ret_entering[i], new_ret);
    if (!ci.feasible)
      continue;
    auto cj = sj.subgraph.best_cost(new_ent, sj.retain_these);
    if (!cj.feasible)
      continue;
    double saving =
        (state.cost[i] + state.cost[i + 1]) - (ci.latency + cj.latency);
    if (saving > -floor && saving > best.saving) {
      best.type = SolutionMove::RETAIN_REMOVE;
      best.step_a = i;
      best.tensor = t;
      best.saving = saving;
    }
  }

  return best;
}

// ============================================================================
// Apply a move, return affected step range [lo, hi)
// ============================================================================

static std::pair<size_t, size_t> apply_move(SolState &state,
                                            const SolutionMove &m) {
  const auto &prob = *state.prob;
  const auto &dag = *state.dag;

  switch (m.type) {
  case SolutionMove::STEAL: {
    if (m.step_a >= state.size() || m.step_b >= state.size())
      return {SIZE_MAX, 0};
    auto &src = state.steps[m.step_a];
    auto &dst = state.steps[m.step_b];
    std::set<size_t> ns(src.subgraph.ops().begin(), src.subgraph.ops().end());
    ns.erase(m.op);
    std::set<size_t> nd(dst.subgraph.ops().begin(), dst.subgraph.ops().end());
    nd.insert(m.op);
    auto ss = Subgraph::create(prob, dag, {ns.begin(), ns.end()});
    auto sd = Subgraph::create(prob, dag, {nd.begin(), nd.end()});
    if (!ss || !sd)
      return {SIZE_MAX, 0};
    auto sr = filter_retain(src.retain_these, *ss);
    auto dr = filter_retain(dst.retain_these, *sd);
    auto cs = ss->best_cost(state.ret_entering[m.step_a], sr);
    auto cd = sd->best_cost(state.ret_entering[m.step_b], dr);
    if (!cs.feasible || !cd.feasible)
      return {SIZE_MAX, 0};
    src.subgraph = std::move(*ss);
    src.config = cs.config;
    src.retain_these = sr;
    dst.subgraph = std::move(*sd);
    dst.config = cd.config;
    dst.retain_these = dr;
    size_t lo = std::min(m.step_a, m.step_b);
    return {lo, lo + 2};
  }
  case SolutionMove::MERGE: {
    if (m.step_a >= state.size() || m.step_b >= state.size())
      return {SIZE_MAX, 0};
    auto oi = state.steps[m.step_a].subgraph.ops();
    auto oj = state.steps[m.step_b].subgraph.ops();
    std::vector<size_t> merged;
    std::set<size_t> seen;
    for (auto o : oi)
      if (seen.insert(o).second)
        merged.push_back(o);
    for (auto o : oj)
      if (seen.insert(o).second)
        merged.push_back(o);
    auto sg = Subgraph::create(prob, dag, merged);
    if (!sg)
      return {SIZE_MAX, 0};
    auto ret = filter_retain(state.steps[m.step_b].retain_these, *sg);
    auto cm = sg->best_cost(state.ret_entering[m.step_a], ret);
    if (!cm.feasible) {
      ret.clear();
      cm = sg->best_cost(state.ret_entering[m.step_a], {});
    }
    if (!cm.feasible)
      return {SIZE_MAX, 0};
    state.steps[m.step_a].subgraph = std::move(*sg);
    state.steps[m.step_a].config = cm.config;
    state.steps[m.step_a].retain_these = ret;
    state.steps.erase(state.steps.begin() + m.step_b);
    return {m.step_a, state.size()};
  }
  case SolutionMove::RECOMPUTE: {
    if (m.step_a >= state.size())
      return {SIZE_MAX, 0};
    auto ops = state.steps[m.step_a].subgraph.ops();
    std::vector<size_t> expanded(ops.begin(), ops.end());
    expanded.push_back(m.op);
    auto sg = Subgraph::create(prob, dag, expanded);
    if (!sg)
      return {SIZE_MAX, 0};
    auto ret = filter_retain(state.steps[m.step_a].retain_these, *sg);
    auto ce = sg->best_cost(state.ret_entering[m.step_a], ret);
    if (!ce.feasible)
      return {SIZE_MAX, 0};
    state.steps[m.step_a].subgraph = std::move(*sg);
    state.steps[m.step_a].config = ce.config;
    state.steps[m.step_a].retain_these = ret;
    return {m.step_a, m.step_a + 1};
  }
  case SolutionMove::EJECT: {
    if (m.step_a >= state.size()) return {SIZE_MAX, 0};
    auto &si = state.steps[m.step_a];
    std::set<size_t> si_set(si.subgraph.ops().begin(), si.subgraph.ops().end());
    std::set<size_t> rem = si_set;
    rem.erase(m.op);

    // ====================================================================
    // TOPOLOGICAL TIMELINE CHECK
    // We cannot blindly put the ejected op after the remainder!
    // ====================================================================
    bool must_be_before = false;
    for (auto succ : dag.op_succs[m.op]) {
        if (rem.count(succ)) must_be_before = true;
    }
    bool must_be_after = false;
    for (auto pred : dag.op_preds[m.op]) {
        if (rem.count(pred)) must_be_after = true;
    }
    if (must_be_before && must_be_after) return {SIZE_MAX, 0}; // Paradox!
    // ====================================================================

    auto sg_r = Subgraph::create(prob, dag, {rem.begin(), rem.end()});
    auto sg_s = Subgraph::create(prob, dag, {m.op});
    if (!sg_r || !sg_s) return {SIZE_MAX, 0};
    
    auto rr = filter_retain(si.retain_these, *sg_r);
    auto cr = sg_r->best_cost(state.ret_entering[m.step_a], rr);
    auto cs = sg_s->best_cost({}, {});
    if (!cr.feasible || !cs.feasible) return {SIZE_MAX, 0};
    
    ScheduleStep step_r;
    step_r.subgraph = std::move(*sg_r);
    step_r.config = cr.config;
    step_r.retain_these = rr;

    ScheduleStep step_s;
    step_s.subgraph = std::move(*sg_s);
    step_s.config = cs.config;

    // Apply strict chronological order
    if (must_be_before) {
        state.steps[m.step_a] = std::move(step_s);
        state.steps.insert(state.steps.begin() + m.step_a + 1, std::move(step_r));
    } else {
        state.steps[m.step_a] = std::move(step_r);
        state.steps.insert(state.steps.begin() + m.step_a + 1, std::move(step_s));
    }
    return {m.step_a, state.size()};
  }
  case SolutionMove::INTERNAL_EJECT: {
    if (m.step_a >= state.size()) return {SIZE_MAX, 0};
    auto &si = state.steps[m.step_a];
    std::set<size_t> si_set(si.subgraph.ops().begin(), si.subgraph.ops().end());
    Partition tmp;
    tmp.prob = &prob;
    tmp.dag = &dag;
    tmp.groups.push_back({si_set, state.cost[m.step_a], true, 0});
    tmp.rebuild_index();
    auto er = tmp.eval_eject(m.op, 0);
    if (!er.feasible) return {SIZE_MAX, 0};

    // ====================================================================
    // TOPOLOGICAL TIMELINE SORT
    // We must rebuild the local DAG of fragments to sort them chronologically!
    // ====================================================================
    std::vector<std::set<size_t>> new_groups;
    new_groups.push_back({m.op});
    for (auto& comp : er.remainder_components) new_groups.push_back(comp);

    int k = new_groups.size();
    std::vector<int> in_deg(k, 0);
    std::vector<std::vector<int>> adj(k);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (i == j) continue;
            bool has_edge = false;
            for (auto u : new_groups[i]) {
                for (auto v : dag.op_succs[u]) {
                    if (new_groups[j].count(v)) has_edge = true;
                }
            }
            if (has_edge) { adj[i].push_back(j); in_deg[j]++; }
        }
    }
    
    std::vector<int> order;
    std::vector<int> q;
    for (int i = 0; i < k; i++) if (in_deg[i] == 0) q.push_back(i);
    while(!q.empty()) {
        int u = q.back(); q.pop_back();
        order.push_back(u);
        for (int v : adj[u]) if (--in_deg[v] == 0) q.push_back(v);
    }
    if (order.size() != k) return {SIZE_MAX, 0}; // Impossible cycle!
    // ====================================================================

    std::vector<ScheduleStep> new_steps;
    for (int idx : order) {
        auto sg = Subgraph::create(prob, dag, {new_groups[idx].begin(), new_groups[idx].end()});
        if (!sg) return {SIZE_MAX, 0};
        auto cc = sg->best_cost({}, {});
        if (!cc.feasible) return {SIZE_MAX, 0};
        ScheduleStep ns;
        ns.subgraph = std::move(*sg);
        ns.config = cc.config;
        new_steps.push_back(std::move(ns));
    }

    state.steps.erase(state.steps.begin() + m.step_a);
    for (size_t i = 0; i < new_steps.size(); i++)
      state.steps.insert(state.steps.begin() + m.step_a + i, std::move(new_steps[i]));
      
    return {m.step_a, state.size()};
  }
  case SolutionMove::SPLIT: {
    if (m.step_a >= state.size()) return {SIZE_MAX, 0};
    Partition tmp;
    tmp.prob = &prob;
    tmp.dag = &dag;
    auto ops = state.steps[m.step_a].subgraph.ops();
    std::set<size_t> op_set(ops.begin(), ops.end());
    tmp.groups.push_back({op_set, 0, true, 0});
    tmp.rebuild_index();
    auto sr = tmp.eval_split(m.op, m.op2, 0);
    if (!sr.feasible) return {SIZE_MAX, 0};

    // ====================================================================
    // TOPOLOGICAL TIMELINE CHECK
    // Undirected BFS doesn't guarantee side_a precedes side_b!
    // ====================================================================
    bool a_before_b = false, b_before_a = false;
    for (auto op_a : sr.side_a) {
        for (auto succ : dag.op_succs[op_a]) if (sr.side_b.count(succ)) a_before_b = true;
        for (auto pred : dag.op_preds[op_a]) if (sr.side_b.count(pred)) b_before_a = true;
    }
    if (a_before_b && b_before_a) return {SIZE_MAX, 0};
    
    if (b_before_a) {
        std::swap(sr.side_a, sr.side_b);
        // Cost swapping is unnecessary as we recalculate below
    }
    // ====================================================================

    auto sg_a = Subgraph::create(prob, dag, {sr.side_a.begin(), sr.side_a.end()});
    auto sg_b = Subgraph::create(prob, dag, {sr.side_b.begin(), sr.side_b.end()});
    if (!sg_a || !sg_b) return {SIZE_MAX, 0};

    std::set<size_t> br;
    for (auto t : sg_a->boundary_outputs())
      if (sg_b->boundary_inputs().count(t) && prob.retainable_tensors.count(t))
        br.insert(t);
        
    auto c_a = sg_a->best_cost(state.ret_entering[m.step_a], br);
    if (!c_a.feasible) {
      br.clear();
      c_a = sg_a->best_cost(state.ret_entering[m.step_a], {});
    }
    
    auto vr = filter_retain(state.steps[m.step_a].retain_these, *sg_b);
    auto c_b = sg_b->best_cost(br, vr);
    if (!c_b.feasible) c_b = sg_b->best_cost(br, {});
    
    if (!c_a.feasible || !c_b.feasible) return {SIZE_MAX, 0};
    
    state.steps[m.step_a].subgraph = std::move(*sg_a);
    state.steps[m.step_a].config = c_a.config;
    state.steps[m.step_a].retain_these = br;
    
    ScheduleStep ns;
    ns.subgraph = std::move(*sg_b);
    ns.config = c_b.config;
    ns.retain_these = vr;
    state.steps.insert(state.steps.begin() + m.step_a + 1, std::move(ns));
    return {m.step_a, state.size()};
  }
  case SolutionMove::RETAIN_ADD: {
    if (m.step_a >= state.size())
      return {SIZE_MAX, 0};
    auto &si = state.steps[m.step_a];
    auto nr = si.retain_these;
    nr.insert(m.tensor);
    auto ci = si.subgraph.best_cost(state.ret_entering[m.step_a], nr);
    if (!ci.feasible)
      return {SIZE_MAX, 0};
    // Must also verify and update step_a+1 with new entering set
    if (m.step_a + 1 < state.size()) {
      auto new_ent = nr; // what exits step_a = its retain set
      auto cj = state.steps[m.step_a + 1].subgraph.best_cost(
          new_ent, state.steps[m.step_a + 1].retain_these);
      if (!cj.feasible)
        return {SIZE_MAX, 0};
      state.steps[m.step_a + 1].config = cj.config;
    }
    si.retain_these = nr;
    si.config = ci.config;
    return {m.step_a, state.size()};
  }
  case SolutionMove::RETAIN_REMOVE: {
    if (m.step_a >= state.size())
      return {SIZE_MAX, 0};
    auto &si = state.steps[m.step_a];
    auto nr = si.retain_these;
    nr.erase(m.tensor);
    auto ci = si.subgraph.best_cost(state.ret_entering[m.step_a], nr);
    if (!ci.feasible)
      return {SIZE_MAX, 0};
    // Must also verify and update step_a+1 with new entering set
    if (m.step_a + 1 < state.size()) {
      auto new_ent = nr;
      auto cj = state.steps[m.step_a + 1].subgraph.best_cost(
          new_ent, state.steps[m.step_a + 1].retain_these);
      if (!cj.feasible)
        return {SIZE_MAX, 0};
      state.steps[m.step_a + 1].config = cj.config;
    }
    si.retain_these = nr;
    si.config = ci.config;
    return {m.step_a, state.size()};
  }
  default:
    return {SIZE_MAX, 0};
  }
}

// ============================================================================
// Generate moves for a step and push to heap (for greedy/heap variant)
// Each move stores gen_a/gen_b for staleness detection.
// ============================================================================

static void
generate_step_moves(SolState &state, size_t si, SolMoveHeap &heap, double floor,
                    Clock::time_point deadline = Clock::time_point::max()) {
  if (si >= state.size())
    return;
  if (Clock::now() >= deadline)
    return;
  const auto &prob = *state.prob;
  int gen_i = state.gen[si];
  std::set<size_t> empty;

  // Per-op moves
  for (auto op : state.steps[si].subgraph.ops()) {
    if (Clock::now() >= deadline)
      return;
    auto m = best_move_for_op(state, op, empty, floor);
    if (m.valid() && m.saving > -floor) {
      m.gen_a = gen_i;
      m.gen_b = (m.step_b != SIZE_MAX && m.step_b < state.size())
                    ? state.gen[m.step_b]
                    : -1;
      heap.push(m);
    }
  }

  if (Clock::now() >= deadline)
    return;

  // Per-tensor moves
  for (auto t : state.steps[si].subgraph.boundary_inputs()) {
    auto m = best_move_for_tensor(state, t, empty, floor);
    if (m.valid() && m.saving > -floor) {
      m.gen_a = (m.step_a < state.size()) ? state.gen[m.step_a] : -1;
      m.gen_b = (m.step_a + 1 < state.size()) ? state.gen[m.step_a + 1] : -1;
      heap.push(m);
    }
  }

  if (Clock::now() >= deadline)
    return;

  for (auto t : state.steps[si].subgraph.boundary_outputs()) {
    auto m = best_move_for_tensor(state, t, empty, floor);
    if (m.valid() && m.saving > -floor) {
      m.gen_a = (m.step_a < state.size()) ? state.gen[m.step_a] : -1;
      m.gen_b = (m.step_a + 1 < state.size()) ? state.gen[m.step_a + 1] : -1;
      heap.push(m);
    }
  }
  // Retained tensors
  for (auto t : state.steps[si].retain_these) {
    auto mr = best_move_for_tensor(state, t, empty, floor);
    if (mr.valid() && mr.saving > -floor) {
      mr.gen_a = (mr.step_a < state.size()) ? state.gen[mr.step_a] : -1;
      mr.gen_b = (mr.step_a + 1 < state.size()) ? state.gen[mr.step_a + 1] : -1;
      heap.push(mr);
    }
  }
}

// Helper: bump gens for affected range, or all if structural change
static void bump_affected(SolState &state, size_t lo, size_t hi,
                          bool structural) {
  if (structural) {
    // Step count changed — all step indices may have shifted
    state.bump_all();
  } else {
    for (size_t i = lo; i < std::min(hi, state.size()); i++)
      state.bump(i);
    // Also bump neighbors
    if (lo > 0)
      state.bump(lo - 1);
    if (hi < state.size())
      state.bump(hi);
  }
}

// Check if a move type changes step count
static bool is_structural(SolutionMove::Type t) {
  return t == SolutionMove::SPLIT || t == SolutionMove::MERGE ||
         t == SolutionMove::EJECT || t == SolutionMove::INTERNAL_EJECT;
}

// Staleness check for a heap move
static bool is_stale(const SolutionMove &m, const SolState &state) {
  if (m.step_a >= state.size())
    return true;
  if (m.gen_a != -1 && m.gen_a != state.gen[m.step_a])
    return true;
  if (m.step_b != SIZE_MAX) {
    if (m.step_b >= state.size())
      return true;
    if (m.gen_b != -1 && m.gen_b != state.gen[m.step_b])
      return true;
  }
  // For RETAIN moves, check gen_b (next step)
  if ((m.type == SolutionMove::RETAIN_ADD ||
       m.type == SolutionMove::RETAIN_REMOVE) &&
      m.step_a + 1 < state.size() && m.gen_b != -1 &&
      m.gen_b != state.gen[m.step_a + 1])
    return true;
  return false;
}

// ============================================================================
// Active set: per-op and per-tensor entries (for FM pass)
// ============================================================================

struct SolActiveSet {
  struct Entry {
    bool is_tensor;
    size_t id;
    SolutionMove move;
  };

  std::vector<Entry> entries;
  std::set<size_t> active_ops, active_tensors;
  std::set<size_t> locked_ops, locked_tensors;
  double floor = 0;
  Clock::time_point deadline = Clock::time_point::max();

  void activate_op(SolState &state, size_t op) {
    if (active_ops.count(op) || locked_ops.count(op))
      return;
    if (Clock::now() >= deadline)
      return;
    active_ops.insert(op);
    auto m = best_move_for_op(state, op, locked_ops, floor);
    entries.push_back({false, op, m});
  }

  void activate_tensor(SolState &state, size_t t) {
    if (active_tensors.count(t) || locked_tensors.count(t))
      return;
    if (!state.prob->retainable_tensors.count(t))
      return;
    if (Clock::now() >= deadline)
      return;
    active_tensors.insert(t);
    auto m = best_move_for_tensor(state, t, locked_tensors, floor);
    entries.push_back({true, t, m});
  }

  // Activate ops and tensors associated with a step
  void activate_step(SolState &state, size_t si) {
    if (si >= state.size())
      return;
    auto &step = state.steps[si];
    for (auto op : step.subgraph.ops())
      activate_op(state, op);
    for (auto t : step.subgraph.boundary_inputs())
      activate_tensor(state, t);
    for (auto t : step.subgraph.boundary_outputs())
      activate_tensor(state, t);
    for (auto t : step.retain_these)
      activate_tensor(state, t);
  }

  // Re-evaluate entries that touch affected steps
  void update_affected(SolState &state, size_t lo, size_t hi) {
    // Collect ops and tensors in the affected range
    std::set<size_t> affected_ops, affected_tensors;
    for (size_t i = lo; i < std::min(hi, state.size()); i++) {
      for (auto op : state.steps[i].subgraph.ops())
        affected_ops.insert(op);
      for (auto t : state.steps[i].subgraph.boundary_inputs())
        affected_tensors.insert(t);
      for (auto t : state.steps[i].subgraph.boundary_outputs())
        affected_tensors.insert(t);
    }
    // Neighbors: one step before lo and after hi
    if (lo > 0) {
      for (auto op : state.steps[lo - 1].subgraph.ops())
        affected_ops.insert(op);
      for (auto t : state.steps[lo - 1].subgraph.boundary_outputs())
        affected_tensors.insert(t);
    }
    if (hi < state.size()) {
      for (auto op : state.steps[hi].subgraph.ops())
        affected_ops.insert(op);
      for (auto t : state.steps[hi].subgraph.boundary_inputs())
        affected_tensors.insert(t);
    }

    for (auto &e : entries) {
      if (Clock::now() >= deadline)
        break;
      if (e.is_tensor) {
        if (locked_tensors.count(e.id))
          continue;
        if (affected_tensors.count(e.id))
          e.move = best_move_for_tensor(state, e.id, locked_tensors, floor);
      } else {
        if (locked_ops.count(e.id))
          continue;
        if (affected_ops.count(e.id))
          e.move = best_move_for_op(state, e.id, locked_ops, floor);
      }
    }

    // Activate new ops/tensors in affected range
    for (size_t i = lo; i < std::min(hi + 1, state.size()); i++)
      activate_step(state, i);
  }

  std::optional<SolutionMove> pop_best() {
    int best_idx = -1;
    double best_saving = -1e18;
    for (size_t i = 0; i < entries.size(); i++) {
      auto &e = entries[i];
      if (e.is_tensor && locked_tensors.count(e.id))
        continue;
      if (!e.is_tensor && locked_ops.count(e.id))
        continue;
      if (!e.move.valid())
        continue;
      if (e.move.saving > best_saving) {
        best_saving = e.move.saving;
        best_idx = (int)i;
      }
    }
    if (best_idx < 0)
      return std::nullopt;
    auto &e = entries[best_idx];
    auto result = e.move;
    if (e.is_tensor)
      locked_tensors.insert(e.id);
    else
      locked_ops.insert(e.id);
    return result;
  }
};

// ============================================================================
// Greedy hill climb with active set
// ============================================================================

std::vector<ScheduleStep>
solution_greedy_descent(const Problem &prob, const DAG &dag,
                        std::vector<ScheduleStep> steps,
                        Clock::time_point deadline) {
  SolState state;
  state.init(prob, dag, std::move(steps));
  SolMoveHeap heap;

  // Generate initial moves — for large solutions, limit initial scope
  size_t max_init = state.size();
  if (state.size() > 30)
    max_init = state.size() / 2;
  for (size_t i = 0; i < max_init; i++) {
    if (Clock::now() >= deadline)
      break;
    generate_step_moves(state, i, heap, 0.0, deadline);
  }

  int applied = 0;
  while (!heap.empty()) {
    if (Clock::now() >= deadline)
      break;
    auto m = heap.top();
    heap.pop();

    // Only accept improving moves in greedy
    if (m.saving <= 0.01)
      break;

    // Staleness check
    if (is_stale(m, state))
      continue;

    auto [lo, hi] = apply_move(state, m);
    if (lo == SIZE_MAX)
      continue;

    bool structural = is_structural(m.type);
    bump_affected(state, lo, hi, structural);
    state.rebuild_from(lo);
    applied++;

    // Regenerate moves for affected + neighbor steps
    size_t regen_lo = (lo > 0) ? lo - 1 : 0;
    size_t regen_hi = std::min(hi + 1, state.size());
    for (size_t i = regen_lo; i < regen_hi; i++)
      generate_step_moves(state, i, heap, 0.0, deadline);
  }
  return state.steps;
}

// ============================================================================
// FM inner pass
// ============================================================================

SolutionFMPassResult solution_fm_pass(const Problem &prob, const DAG &dag,
                                      std::vector<ScheduleStep> steps,
                                      const SolutionFMPassConfig &cfg) {
  SolState state;
  state.init(prob, dag, std::move(steps));

  SolutionFMPassResult result;
  result.start_cost = state.total;
  result.best_cost = state.total;
  result.best_steps = state.steps;

  double floor = result.start_cost * cfg.floor_fraction;
  double max_drift = result.start_cost * cfg.max_drift_fraction;

  SolActiveSet active;
  active.floor = floor;
  active.deadline = cfg.deadline;

  // Initialize: 1 random entity (tensor preferred, like partition FM's
  // init_count=1) The active set grows locally via update_affected +
  // activate_step after each move.
  std::mt19937 rng(cfg.seed);
  std::vector<size_t> all_tensors;
  for (size_t t = 0; t < prob.num_tensors(); t++)
    if (prob.retainable_tensors.count(t))
      all_tensors.push_back(t);
  std::vector<size_t> all_ops;
  for (size_t i = 0; i < state.size(); i++)
    for (auto op : state.steps[i].subgraph.ops())
      all_ops.push_back(op);
  std::sort(all_ops.begin(), all_ops.end());
  all_ops.erase(std::unique(all_ops.begin(), all_ops.end()), all_ops.end());

  std::shuffle(all_tensors.begin(), all_tensors.end(), rng);
  std::shuffle(all_ops.begin(), all_ops.end(), rng);

  if (!all_tensors.empty()) {
    active.activate_tensor(state, all_tensors[0]);
    // Activate seed step + adjacent steps (like partition FM's
    // activate_neighbors_of)
    auto steps_of = find_steps_of_tensor(state.steps, all_tensors[0]);
    for (auto si : steps_of) {
      active.activate_step(state, si);
      if (si > 0)
        active.activate_step(state, si - 1);
      if (si + 1 < state.size())
        active.activate_step(state, si + 1);
    }
  } else if (!all_ops.empty()) {
    active.activate_op(state, all_ops[0]);
    size_t si = find_step_of(state.steps, all_ops[0]);
    if (si != SIZE_MAX) {
      active.activate_step(state, si);
      if (si > 0)
        active.activate_step(state, si - 1);
      if (si + 1 < state.size())
        active.activate_step(state, si + 1);
    }
  }

  double cumul_gain = 0, best_cumul_gain = 0;
  static const char *move_names[] = {"STEAL",   "SPLIT",  "MERGE", "RET_ADD",
                                     "RET_REM", "RECOMP", "EJECT", "INT_EJECT"};

  for (int iter = 0; iter < 200; iter++) {
    if (Clock::now() >= cfg.deadline)
      break;

    auto m_opt = active.pop_best();
    if (!m_opt || m_opt->saving < -floor)
      break;

    auto [lo, hi] = apply_move(state, *m_opt);
    if (lo == SIZE_MAX)
      continue;
    result.moves_applied++;
    state.rebuild_from(lo);

    if (g_verbose) {
      auto &m = *m_opt;
      std::cerr << "      [pass] iter=" << iter << " " << move_names[m.type]
                << " step=" << m.step_a;
      if (m.op != SIZE_MAX)
        std::cerr << " op=" << m.op;
      if (m.tensor != SIZE_MAX)
        std::cerr << " t=" << m.tensor;
      std::cerr << " saving=" << m.saving << " total=" << state.total
                << " active=" << active.entries.size()
                << " locked_ops=" << active.locked_ops.size()
                << " locked_t=" << active.locked_tensors.size() << "\n";
    }

    cumul_gain = result.start_cost - state.total;
    if (state.total < result.best_cost - 0.01) {
      result.best_cost = state.total;
      result.best_steps = state.steps;
      best_cumul_gain = cumul_gain;
    }
    if (best_cumul_gain - cumul_gain > max_drift)
      break;

    // --- Post-move locking ---
    // Lock the initiating entity (already done by pop_best).
    // Additional locks to prevent reversal:
    auto &m = *m_opt;
    if (m.type == SolutionMove::RETAIN_ADD && m.tensor != SIZE_MAX) {
      // Lock producer op of tensor (prevents STEAL/EJECT breaking boundary)
      int prod = dag.tensor_producer[m.tensor];
      if (prod >= 0)
        active.locked_ops.insert((size_t)prod);
      // Lock consumer ops in the beneficiary step (step_a + 1)
      if (m.step_a + 1 < state.size()) {
        for (auto op : state.steps[m.step_a + 1].subgraph.ops()) {
          for (auto t : prob.ops[op].inputs) {
            if (t == m.tensor) {
              active.locked_ops.insert(op);
              break;
            }
          }
        }
      }
    } else if (m.type == SolutionMove::SPLIT) {
      // Lock bridge ops + all boundary ops between the two new steps
      // to prevent immediate re-merge
      if (m.op != SIZE_MAX)
        active.locked_ops.insert(m.op);
      if (m.op2 != SIZE_MAX)
        active.locked_ops.insert(m.op2);
      if (m.step_a + 1 < state.size()) {
        std::set<size_t> sb(state.steps[m.step_a + 1].subgraph.ops().begin(),
                            state.steps[m.step_a + 1].subgraph.ops().end());
        for (auto op : state.steps[m.step_a].subgraph.ops()) {
          for (auto s : dag.op_succs[op])
            if (sb.count(s)) {
              active.locked_ops.insert(op);
              break;
            }
        }
        std::set<size_t> sa(state.steps[m.step_a].subgraph.ops().begin(),
                            state.steps[m.step_a].subgraph.ops().end());
        for (auto op : state.steps[m.step_a + 1].subgraph.ops()) {
          for (auto p : dag.op_preds[op])
            if (sa.count(p)) {
              active.locked_ops.insert(op);
              break;
            }
        }
      }
    } else if (m.type == SolutionMove::MERGE) {
      // Lock initiating op + its DAG neighbors in the merged step
      // (prevents SPLIT/EJECT from undoing the merge — same as partition FM)
      if (m.op != SIZE_MAX) {
        active.locked_ops.insert(m.op);
        std::set<size_t> step_ops(state.steps[m.step_a].subgraph.ops().begin(),
                                  state.steps[m.step_a].subgraph.ops().end());
        for (auto p : dag.op_preds[m.op])
          if (step_ops.count(p))
            active.locked_ops.insert(p);
        for (auto s : dag.op_succs[m.op])
          if (step_ops.count(s))
            active.locked_ops.insert(s);
      }
    } else if (m.type == SolutionMove::EJECT ||
               m.type == SolutionMove::INTERNAL_EJECT) {
      // Lock the ejected op + its DAG neighbors in the remainder
      if (m.op != SIZE_MAX) {
        active.locked_ops.insert(m.op);
        for (auto p : dag.op_preds[m.op])
          active.locked_ops.insert(p);
        for (auto s : dag.op_succs[m.op])
          active.locked_ops.insert(s);
      }
    } else if (m.type == SolutionMove::STEAL) {
      if (m.op != SIZE_MAX)
        active.locked_ops.insert(m.op);
    }

    active.update_affected(state, lo, hi);
  }

  result.end_steps = state.steps;
  result.end_cost = state.total;
  return result;
}

// ============================================================================
// Parallel solution FM: N threads with different seeds, logging
// ============================================================================

#include <atomic>
#include <mutex>
#include <thread>

Solution solution_fm_search(const Problem &prob, const DAG &dag, Solution sol,
                            const SolutionFMConfig &cfg) {
  // Delegate to multi-start with a single-element pool
  std::vector<Solution> pool;
  pool.push_back(std::move(sol));
  return solution_fm_search(prob, dag, std::move(pool), cfg);
}

Solution solution_fm_search(const Problem &prob, const DAG &dag,
                            std::vector<Solution> pool,
                            const SolutionFMConfig &cfg) {
  if (pool.empty()) return Solution(prob, dag, {});

  int hw_threads = (int)std::thread::hardware_concurrency();
  if (hw_threads <= 0)
    hw_threads = 4;
  int num_threads = hw_threads;

  // Precompute per-pool-entry steps and costs
  struct PoolEntry {
    std::vector<ScheduleStep> steps;
    double cost;
  };
  std::vector<PoolEntry> pool_entries;
  for (auto& s : pool)
    pool_entries.push_back({s.steps(), s.total_latency()});

  double global_best_init = pool_entries[0].cost;
  for (auto& pe : pool_entries)
    global_best_init = std::min(global_best_init, pe.cost);

  struct ThreadResult {
    std::vector<ScheduleStep> steps;
    double cost = 1e18;
    int passes = 0;
    int improving = 0;
    int total_moves = 0;
  };

  std::vector<ThreadResult> results(num_threads);
  bool main_verbose = g_verbose;

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back([&, t]() {
      g_verbose = (t == 0) && main_verbose;

      // Each thread starts from a different pool entry (round-robin)
      auto& start = pool_entries[t % pool_entries.size()];
      auto best_steps = start.steps;
      double best_cost = start.cost;
      int no_improve = 0;
      int total_passes = 0, improving_passes = 0, total_moves = 0;

      double base_floor = cfg.pass_config.floor_fraction;
      double base_drift = cfg.pass_config.max_drift_fraction;
      double heat = 1.0;

      for (int pass = 0; pass < cfg.max_passes; pass++) {
        if (Clock::now() >= cfg.deadline)
          break;

        double progress = (double)pass / std::max(1, cfg.max_passes - 1);
        double temp = 0.1 + 0.9 * 0.5 * (1.0 + std::cos(progress * M_PI));
        double eff_floor = std::clamp(base_floor * temp * heat, 0.02, 1.0);
        double eff_drift = std::clamp(base_drift * temp * heat, 0.05, 2.0);

        if (no_improve >= cfg.max_no_improve)
          break;

        SolutionFMPassConfig pc = cfg.pass_config;
        pc.seed = (unsigned)(cfg.pass_config.seed + t * 1000 + pass * 7);
        pc.floor_fraction = eff_floor;
        pc.max_drift_fraction = eff_drift;
        pc.deadline = cfg.deadline;

        auto pr = solution_fm_pass(prob, dag, best_steps, pc);
        total_passes++;
        total_moves += pr.moves_applied;

        if (g_verbose) {
          std::cerr << "    [t" << t << " pass " << pass
                    << "] moves=" << pr.moves_applied
                    << " best=" << pr.best_cost << " end=" << pr.end_cost
                    << " floor=" << std::fixed << std::setprecision(3)
                    << eff_floor << " heat=" << heat << " no_imp=" << no_improve
                    << "\n";
        }

        if (pr.best_cost < best_cost - 0.01) {
          best_cost = pr.best_cost;
          best_steps = std::move(pr.best_steps);
          no_improve = 0;
          improving_passes++;
          heat = std::clamp(heat * 0.7, 0.1, 3.0);
        } else {
          if (pr.moves_applied > 0) {
            auto kicked = solution_greedy_descent(
                prob, dag, std::move(pr.end_steps), cfg.deadline);
            Solution kicked_sol(prob, dag, kicked);
            kicked_sol = optimize_retain(prob, dag, std::move(kicked_sol));
            if (kicked_sol.total_latency() < best_cost - 0.01) {
              best_cost = kicked_sol.total_latency();
              best_steps = kicked_sol.steps();
              no_improve = 0;
              improving_passes++;
              heat = std::clamp(heat * 0.9, 0.1, 3.0);
              continue;
            }
          }
          no_improve++;
          heat = std::clamp(heat * 1.3, 0.1, 3.0);
        }
      }

      results[t] = {std::move(best_steps), best_cost, total_passes,
                    improving_passes, total_moves};
    });
  }
  for (auto &t : threads)
    t.join();

  // Find best across threads and log
  int best_t = 0;
  for (int t = 1; t < num_threads; t++)
    if (results[t].cost < results[best_t].cost)
      best_t = t;

  std::cerr << "  Sol-FM: " << num_threads << " threads from "
            << pool_entries.size() << " starting solutions";
  int total_passes = 0, total_improving = 0, total_moves = 0;
  for (int t = 0; t < num_threads; t++) {
    total_passes += results[t].passes;
    total_improving += results[t].improving;
    total_moves += results[t].total_moves;
  }
  std::cerr << ", " << total_passes << " passes"
            << " (" << total_improving << " improving)"
            << ", " << total_moves << " moves";
  if (results[best_t].cost < global_best_init - 0.01) {
    std::cerr << ", best=" << results[best_t].cost << " (thread " << best_t
              << ")";
  }
  std::cerr << "\n";

  return Solution(prob, dag, std::move(results[best_t].steps));
}