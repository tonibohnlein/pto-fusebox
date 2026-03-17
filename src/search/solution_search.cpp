#include "search/solution_search.h"
#include "search/merkle_hash.h"
#include "partition/partition.h"
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

// ============================================================================
// Merkle-aware ARI canonicalisation
//
// Within each Merkle equivalence class (structurally symmetric ops), sort ops
// by their assignment in map_a, then match them rank-for-rank to ops sorted
// by their assignment in map_b.  This makes symmetric variants have distance 0
// instead of a spurious non-zero ARI distance.
//
// Time: O(sum_over_classes(k log k)) — negligible for typical ML graphs.
// ============================================================================
static void merkle_canonicalise(
        const MerkleHashes& mh,
        const std::vector<int>& map_a,
        std::vector<int>& map_b)   // modified in-place
{
    for (auto& [hash, ops] : mh.equiv_classes) {
        if (ops.size() <= 1) continue;

        // Sort ops by their assignment in A (break ties by op index for stability)
        std::vector<size_t> by_a(ops.begin(), ops.end());
        std::sort(by_a.begin(), by_a.end(), [&](size_t x, size_t y){
            int gx = (x < map_a.size()) ? map_a[x] : -1;
            int gy = (y < map_a.size()) ? map_a[y] : -1;
            return gx != gy ? gx < gy : x < y;
        });

        // Sort ops by their assignment in B
        std::vector<size_t> by_b(ops.begin(), ops.end());
        std::sort(by_b.begin(), by_b.end(), [&](size_t x, size_t y){
            int gx = (x < map_b.size()) ? map_b[x] : -1;
            int gy = (y < map_b.size()) ? map_b[y] : -1;
            return gx != gy ? gx < gy : x < y;
        });

        // Match rank-for-rank: A's i-th op gets B's i-th op's assignment
        std::vector<int> new_b(ops.size());
        for (size_t i = 0; i < ops.size(); i++)
            new_b[i] = (by_b[i] < map_b.size()) ? map_b[by_b[i]] : -1;
        for (size_t i = 0; i < ops.size(); i++)
            if (by_a[i] < map_b.size()) map_b[by_a[i]] = new_b[i];
    }
}
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
                                      const Subgraph &sg,
                                      const std::set<size_t> &entering = {}) {
  (void)entering;
  // Only boundary OUTPUTS can be retained. The organizer confirmed:
  // "By specifying output tensors in tensors_to_retain, it is possible to
  // keep output tensors in the fast memory."
  // Boundary inputs cannot be retained — they are consumed and freed.
  std::set<size_t> r;
  for (auto t : retain)
    if (sg.boundary_outputs().count(t))
      r.insert(t);
  return r;
}

// Quick check: does a solution have any ephemeral gaps?
static bool has_ephemeral_gaps(const DAG &dag,
                                const std::vector<ScheduleStep> &steps) {
  for (auto &step : steps)
    for (auto t : step.subgraph.boundary_inputs()) {
      if (dag.tensor_producer[t] < 0) continue;
      bool found = false;
      for (auto &s : steps)
        if (s.subgraph.boundary_outputs().count(t)) { found = true; break; }
      if (!found) return true;
    }
  return false;
}

// Check INPUT direction: does a proposed op-set need a tensor that's
// ephemeral everywhere it's produced?
// exclude_step = step being replaced (not yet updated in `steps` array)
static bool sol_inputs_unavailable(const Problem &prob, const DAG &dag,
                                    const std::set<size_t> &proposed,
                                    const std::vector<ScheduleStep> &steps,
                                    size_t exclude_step = SIZE_MAX,
                                    size_t exclude_step2 = SIZE_MAX) {
  for (auto op : proposed) {
    for (auto t : prob.ops[op].inputs) {
      int prod = dag.tensor_producer[t];
      if (prod < 0) continue;                    // graph input
      if (proposed.count((size_t)prod)) continue; // produced internally
      // T is a boundary input. Is it available from some step?
      bool available = false;
      for (size_t si = 0; si < steps.size(); si++) {
        if (si == exclude_step || si == exclude_step2) continue;
        if (steps[si].subgraph.boundary_outputs().count(t)) {
          available = true; break;
        }
      }
      if (!available) return true;
    }
  }
  return false;
}

static bool is_connected_without(const std::set<size_t> &ops, size_t rm,
                                 const DAG &dag, size_t n) {
  if (ops.size() <= 1)
    return false;  // nothing to remove from, or already empty
  if (ops.size() == 2)
    return true;   // removing rm leaves a singleton — trivially connected
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
    // Use op_neighbors (DAG edges + co-consumer edges) for full connectivity,
    // matching Subgraph::create's connectivity check.
    for (auto v : dag.op_neighbors[u])
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

  // O(1) op → step lookup (rebuilt on every structural change)
  std::vector<size_t> op_to_step;

  // Precomputed: tensors that CAN be retained (singleton feasibility check).
  // Points to externally-owned set (shared across all SolStates for an instance).
  // If null, all retainable_tensors are considered feasible (fallback).
  const std::set<size_t> *feasibly_retainable = nullptr;

  // Convenience: check if tensor is feasibly retainable
  bool is_feasibly_retainable(size_t t) const {
    if (feasibly_retainable) return feasibly_retainable->count(t);
    return prob->retainable_tensors.count(t);
  }

  size_t step_of(size_t op) const {
    return op < op_to_step.size() ? op_to_step[op] : SIZE_MAX;
  }

  void rebuild_op_index() {
    op_to_step.assign(prob->num_ops(), SIZE_MAX);
    for (size_t i = 0; i < steps.size(); i++)
      for (auto op : steps[i].subgraph.ops())
        op_to_step[op] = i;
  }

  void init(const Problem &p, const DAG &d, std::vector<ScheduleStep> s,
            const std::set<size_t> *fr = nullptr) {
    prob = &p;
    dag = &d;
    steps = std::move(s);
    gen.assign(steps.size(), 0);
    feasibly_retainable = fr;
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

      // Only boundary OUTPUTS that the next step needs can be retained.
      std::set<size_t> useful_retain;
      for (auto t : steps[i].retain_these) {
        bool is_output = steps[i].subgraph.boundary_outputs().count(t);
        bool useful = (i + 1 < n) && steps[i + 1].subgraph.boundary_inputs().count(t);
        if (is_output && useful)
          useful_retain.insert(t);
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
    rebuild_op_index();
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

      // Only boundary OUTPUTS that the next step needs can be retained.
      std::set<size_t> useful_retain;
      for (auto t : steps[i].retain_these) {
        bool is_output = steps[i].subgraph.boundary_outputs().count(t);
        bool useful = (i + 1 < n) && steps[i + 1].subgraph.boundary_inputs().count(t);
        if (is_output && useful)
          useful_retain.insert(t);
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
    rebuild_op_index();
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
// One-time feasibility check: which tensors can actually be retained?
// Depends only on Problem + DAG, not on the current solution.
// ============================================================================
std::set<size_t> compute_feasibly_retainable(const Problem &prob, const DAG &dag) {
    std::set<size_t> result;
    for (auto t : prob.retainable_tensors) {
        if (prob.tensors[t].size() > prob.fast_memory_capacity)
            continue;

        bool ok = true;

        // Producer singleton must be feasible while retaining t
        int prod = dag.tensor_producer[t];
        if (prod >= 0) {
            auto sg = Subgraph::create(prob, dag, {(size_t)prod});
            if (!sg) { ok = false; }
            else {
                auto bc = sg->best_cost({}, {t});
                if (!bc.feasible) ok = false;
            }
        }
        if (!ok) continue;

        // Each consumer singleton must be feasible with t entering
        for (size_t op = 0; op < prob.num_ops() && ok; op++) {
            bool consumes = false;
            for (auto inp : prob.ops[op].inputs)
                if (inp == t) { consumes = true; break; }
            if (!consumes) continue;

            auto sg = Subgraph::create(prob, dag, {op});
            if (!sg) { ok = false; break; }
            auto bc = sg->best_cost({t}, {});
            if (!bc.feasible) { ok = false; break; }
        }
        if (ok) result.insert(t);
    }
    return result;
}

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

  size_t si = state.step_of(op);
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
      // Check ephemeral gap: would new_dst create an ephemeral tensor
      // that some other step needs but can't get?
      if (Solution::creates_ephemeral_gap(prob, dag, new_dst, state.steps, sj, si))
        continue;
      if (sol_inputs_unavailable(prob, dag, new_dst, state.steps, sj, si))
        continue;
      auto sg_s = Subgraph::create(prob, dag, {new_src.begin(), new_src.end()});
      auto sg_d = Subgraph::create(prob, dag, {new_dst.begin(), new_dst.end()});
      if (!sg_s || !sg_d)
        continue;
      auto sr = filter_retain(state.steps[si].retain_these, *sg_s, state.ret_entering[si]);
      auto dr = filter_retain(dst.retain_these, *sg_d, state.ret_entering[sj]);
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
      // Check ephemeral gap for the merged set (exclude both merging steps)
      if (Solution::creates_ephemeral_gap(prob, dag, seen, state.steps, si, sj))
        continue;
      auto sg = Subgraph::create(prob, dag, merged);
      if (!sg)
        continue;
      size_t lo = std::min(si, sj), hi = std::max(si, sj);
      auto retain = filter_retain(state.steps[hi].retain_these, *sg, state.ret_entering[lo]);
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
      std::set<size_t> expanded_set(expanded.begin(), expanded.end());
      if (Solution::creates_ephemeral_gap(prob, dag, expanded_set, state.steps, sj))
        continue;
      // Input direction: does expanded set need a tensor ephemeral elsewhere?
      if (sol_inputs_unavailable(prob, dag, expanded_set, state.steps, sj))
        continue;
      auto sg = Subgraph::create(prob, dag, expanded);
      if (!sg)
        continue;
      auto ret = filter_retain(state.steps[sj].retain_these, *sg, state.ret_entering[sj]);
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
    if (sg_r && sg_s) do {
      // Determine placement: singleton before or after remainder
      bool must_be_before = false, must_be_after = false;
      for (auto succ : dag.op_succs[op])
          if (remainder.count(succ)) must_be_before = true;
      for (auto pred : dag.op_preds[op])
          if (remainder.count(pred)) must_be_after = true;
      if (must_be_before && must_be_after) break;

      auto rr = filter_retain(state.steps[si].retain_these, *sg_r, state.ret_entering[si]);

      // Compute costs based on placement order
      CostResult cr, cs;
      if (must_be_before) {
        // singleton first, remainder second
        cs = sg_s->best_cost(state.ret_entering[si], {});
        cr = sg_r->best_cost({}, rr);  // remainder loses entering
      } else {
        // remainder first (common case), singleton second
        cr = sg_r->best_cost(state.ret_entering[si], rr);
        cs = sg_s->best_cost({}, {});  // singleton has no entering
      }
      if (!cr.feasible || !cs.feasible) break;

      double saving = 0;
      // Build temporary step vector with the insertion and cascade forward
      // to get accurate total cost including downstream retention effects.
      {
        auto tmp_steps = state.steps;
        ScheduleStep step_r; step_r.subgraph = *sg_r; step_r.config = cr.config; step_r.retain_these = rr;
        ScheduleStep step_s; step_s.subgraph = *sg_s; step_s.config = cs.config;
        if (must_be_before) {
          tmp_steps[si] = std::move(step_s);
          tmp_steps.insert(tmp_steps.begin() + si + 1, std::move(step_r));
        } else {
          tmp_steps[si] = std::move(step_r);
          tmp_steps.insert(tmp_steps.begin() + si + 1, std::move(step_s));
        }
        double old_cost_from_si = 0;
        for (size_t j = si; j < state.size(); j++)
          old_cost_from_si += state.cost[j];

        double new_cost_from_si = 0;
        std::set<size_t> cur;
        if (si > 0) cur = tmp_steps[si - 1].retain_these;
        for (size_t j = si; j < tmp_steps.size(); j++) {
          std::set<size_t> useful;
          for (auto tt : tmp_steps[j].retain_these) {
            bool is_out = tmp_steps[j].subgraph.boundary_outputs().count(tt);
            bool needed = (j + 1 < tmp_steps.size()) &&
                          tmp_steps[j + 1].subgraph.boundary_inputs().count(tt);
            if (is_out && needed) useful.insert(tt);
          }
          auto c = tmp_steps[j].subgraph.compute_cost(tmp_steps[j].config, cur, useful);
          double cost_j;
          if (c.latency < 1e17) {
            cost_j = c.latency;
          } else {
            auto bc = tmp_steps[j].subgraph.best_cost(cur, useful);
            if (bc.feasible) cost_j = bc.latency;
            else { useful.clear(); auto bc2 = tmp_steps[j].subgraph.best_cost(cur, {}); cost_j = bc2.feasible ? bc2.latency : 1e18; }
          }
          new_cost_from_si += cost_j;
          cur = useful;
          // Early exit: past insertion, entering converges with original
          if (j >= si + 2 && (j - 1) < state.size() &&
              cur == state.ret_entering[j - 1]) {
            for (size_t k = j - 1; k < state.size(); k++)
              new_cost_from_si += state.cost[k];
            break;
          }
        }
        saving = old_cost_from_si - new_cost_from_si;
      }

      if (saving > -floor && saving > best.saving) {
        best.type = SolutionMove::EJECT;
        best.step_a = si;
        best.op = op;
        best.saving = saving;
      }
    } while (false);
  }

  // --- INTERNAL_EJECT: extract internal op ---
  if (!is_border && si_set.size() >= 3 && si_set.size() <= 15) {
    Partition tmp;
    tmp.prob = &prob;
    tmp.dag = &dag;
    tmp.groups.push_back({si_set, state.cost[si], true, 0});
    tmp.rebuild_index();
    auto er = tmp.eval_eject(op, 0);
    if (er.feasible && er.saving > -floor) {
      double saving = er.saving;

      // Account for downstream step losing its entering set.
      // eval_eject computes all components with best_cost({}, {}).
      // After insertion, the last component retains nothing → old si+1
      // loses its entering.
      if (si + 1 < state.size() && !state.ret_entering[si + 1].empty()) {
        auto &sj = state.steps[si + 1];
        // The last component has no retain → old si+1 enters with {}
        auto cj_new = sj.subgraph.best_cost({}, sj.retain_these);
        if (!cj_new.feasible)
          cj_new = sj.subgraph.best_cost({}, {});
        if (cj_new.feasible)
          saving -= (cj_new.latency - state.cost[si + 1]);
        else
          saving = -1e18;  // can't evaluate → skip
      }

      if (saving > -floor && saving > best.saving) {
        best.type = SolutionMove::INTERNAL_EJECT;
        best.step_a = si;
        best.op = op;
        best.saving = saving;
      }
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
        auto sr2 = filter_retain(state.steps[si].retain_these, *second, br);
        auto c2 = second->best_cost(br, sr2);
        if (!c2.feasible)
          c2 = second->best_cost(br, {});
        if (!c2.feasible)
          continue;
        double saving = state.cost[si] - (c1.latency + c2.latency);

        // Account for downstream step (si+1) if the second half's retain
        // differs from what the original step was passing.
        if (si + 1 < state.size() && !state.ret_entering[si + 1].empty()) {
          // What will old si+1 actually enter with?
          std::set<size_t> new_sj_entering;
          for (auto tt : sr2)
            if (second->boundary_outputs().count(tt) &&
                state.steps[si + 1].subgraph.boundary_inputs().count(tt))
              new_sj_entering.insert(tt);
          if (new_sj_entering != state.ret_entering[si + 1]) {
            auto &sj = state.steps[si + 1];
            auto cj_new = sj.subgraph.best_cost(new_sj_entering, sj.retain_these);
            if (!cj_new.feasible)
              cj_new = sj.subgraph.best_cost(new_sj_entering, {});
            if (!cj_new.feasible)
              continue;
            saving -= (cj_new.latency - state.cost[si + 1]);
          }
        }

        if (saving > -floor && saving > best.saving) {
          best.type = SolutionMove::SPLIT;
          best.step_a = si;
          best.op = a;
          best.op2 = b;
          best.saving = saving;
        }
      }
    };
    // SPLIT at bridge edges incident to op.
    // Use op_neighbors (DAG edges + co-consumer edges) so co-consumer bridges
    // are also proposed. Canonicalise as (min, max) to avoid evaluating
    // the same edge twice.
    std::set<std::pair<size_t,size_t>> split_checked;
    for (auto v : dag.op_neighbors[op]) {
      if (!si_set.count(v)) continue;
      auto edge = std::make_pair(std::min(op, v), std::max(op, v));
      if (!split_checked.insert(edge).second) continue;
      try_split(edge.first, edge.second);
    }
  }

  // --- DE_RECOMPUTE: remove a recomputed op from this step ---
  // Op must appear in at least one other step. Removing it here saves compute
  // and possibly memory. If this step becomes empty, it's removed entirely.
  {
    // Check if op appears in another step
    bool in_other_step = false;
    for (size_t sj = 0; sj < state.size(); sj++) {
      if (sj == si) continue;
      for (auto o : state.steps[sj].subgraph.ops())
        if (o == op) { in_other_step = true; break; }
      if (in_other_step) break;
    }
    if (in_other_step) {
      if (si_set.size() == 1) {
        // Singleton step: removing it saves the entire step cost.
        // Verify all boundary outputs are still available from other steps.
        bool safe = true;
        for (auto t : state.steps[si].subgraph.boundary_outputs()) {
          bool avail = false;
          for (size_t sj = 0; sj < state.size(); sj++) {
            if (sj == si) continue;
            if (state.steps[sj].subgraph.boundary_outputs().count(t)) {
              avail = true; break;
            }
          }
          if (!avail) { safe = false; break; }
        }
        if (safe) {
          double saving = state.cost[si];
          if (saving > -floor && saving > best.saving) {
            best.type = SolutionMove::DE_RECOMPUTE;
            best.step_a = si;
            best.op = op;
            best.saving = saving;
          }
        }
      } else if (is_border &&
                 is_connected_without(si_set, op, dag, prob.num_ops())) {
        // Multi-op step: remove op, remainder must be connected + feasible.
        // Verify all boundary outputs produced by op are available elsewhere.
        std::set<size_t> remainder = si_set;
        remainder.erase(op);
        bool safe = true;
        for (auto t : prob.ops[op].outputs) {
          if (!state.steps[si].subgraph.boundary_outputs().count(t)) continue;
          bool avail = false;
          for (size_t sj = 0; sj < state.size(); sj++) {
            if (sj == si) continue;
            if (state.steps[sj].subgraph.boundary_outputs().count(t)) {
              avail = true; break;
            }
          }
          if (!avail) { safe = false; break; }
        }
        if (safe) {
          auto sg_r = Subgraph::create(prob, dag, {remainder.begin(), remainder.end()});
          if (sg_r) {
            auto rr = filter_retain(state.steps[si].retain_these, *sg_r, state.ret_entering[si]);
            auto cr = sg_r->best_cost(state.ret_entering[si], rr);
            if (cr.feasible) {
              double saving = state.cost[si] - cr.latency;
              if (saving > -floor && saving > best.saving) {
                best.type = SolutionMove::DE_RECOMPUTE;
                best.step_a = si;
                best.op = op;
                best.saving = saving;
              }
            }
          }
        }
      }
    }
  }

  return best;
}

// ============================================================================
// Cascade cost delta: simulate rebuild_from for steps [start..end) and return
// the total cost change relative to state.cost[].
//
// This propagates the retention chain forward from `start`, using the modified
// steps (caller should have already changed retain_these/config on the steps
// in the temporary copy). Early-exits when entering set converges with state.
//
// Returns the sum of (new_cost[j] - state.cost[j]) for all affected steps.
// ============================================================================
static double cascade_cost_delta(const SolState &state,
                                 const std::vector<ScheduleStep> &steps,
                                 size_t start) {
  double delta = 0;
  std::set<size_t> cur;
  if (start > 0)
    cur = steps[start - 1].retain_these;

  for (size_t j = start; j < steps.size(); j++) {
    // Filter useful_retain (same logic as rebuild_from)
    std::set<size_t> useful;
    for (auto t : steps[j].retain_these) {
      bool is_out = steps[j].subgraph.boundary_outputs().count(t);
      bool needed = (j + 1 < steps.size()) &&
                    steps[j + 1].subgraph.boundary_inputs().count(t);
      if (is_out && needed) useful.insert(t);
    }

    auto c = steps[j].subgraph.compute_cost(steps[j].config, cur, useful);
    double new_cost;
    if (c.latency < 1e17) {
      new_cost = c.latency;
    } else {
      auto bc = steps[j].subgraph.best_cost(cur, useful);
      if (bc.feasible)
        new_cost = bc.latency;
      else {
        auto bc2 = steps[j].subgraph.best_cost(cur, {});
        useful.clear();
        new_cost = bc2.feasible ? bc2.latency : 1e18;
      }
    }

    delta += new_cost - state.cost[j];
    cur = useful;

    // Early exit: if entering matches the stored entering for next step,
    // no further cascade (all downstream costs stay the same)
    if (j + 1 < steps.size() && cur == state.ret_entering[j + 1])
      break;
  }
  return delta;
}

static SolutionMove best_move_for_tensor(SolState &state, size_t t,
                                         const std::set<size_t> &locked_tensors,
                                         double floor) {
  SolutionMove best;
  const auto &prob = *state.prob;
  if (locked_tensors.count(t))
    return best;
  if (!state.is_feasibly_retainable(t))
    return best;

  // RETAIN_ADD: find step pairs where adding t to retain helps
  for (size_t i = 0; i + 1 < state.size(); i++) {
    auto &si = state.steps[i];
    auto &sj = state.steps[i + 1];
    if (si.retain_these.count(t))
      continue; // already retained
    if (state.ret_entering[i].count(t))
      continue;
    if (!si.subgraph.boundary_outputs().count(t))
      continue;
    if (!sj.subgraph.boundary_inputs().count(t))
      continue;

    // Simulate: add t to step i's retain, then cascade forward.
    auto tmp_steps = state.steps; // shallow copy
    tmp_steps[i].retain_these.insert(t);
    double delta = cascade_cost_delta(state, tmp_steps, i);
    double saving = -delta;
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
    bool is_output = state.steps[i].subgraph.boundary_outputs().count(t);
    bool useful_next = state.steps[i + 1].subgraph.boundary_inputs().count(t);
    if (!is_output || !useful_next)
      continue;  // already pruned by rebuild_from → saving=0

    // Simulate: remove t from step i's retain, then cascade forward.
    auto tmp_steps = state.steps; // shallow copy
    tmp_steps[i].retain_these.erase(t);
    double delta = cascade_cost_delta(state, tmp_steps, i);
    double saving = -delta;
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
    if (Solution::creates_ephemeral_gap(prob, dag, nd, state.steps, m.step_b, m.step_a))
      return {SIZE_MAX, 0};
    if (sol_inputs_unavailable(prob, dag, nd, state.steps, m.step_b, m.step_a))
      return {SIZE_MAX, 0};
    auto ss = Subgraph::create(prob, dag, {ns.begin(), ns.end()});
    auto sd = Subgraph::create(prob, dag, {nd.begin(), nd.end()});
    if (!ss || !sd)
      return {SIZE_MAX, 0};
    auto sr = filter_retain(src.retain_these, *ss, state.ret_entering[m.step_a]);
    auto dr = filter_retain(dst.retain_these, *sd, state.ret_entering[m.step_b]);
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
    if (Solution::creates_ephemeral_gap(prob, dag, seen, state.steps, m.step_a, m.step_b))
      return {SIZE_MAX, 0};
    auto sg = Subgraph::create(prob, dag, merged);
    if (!sg)
      return {SIZE_MAX, 0};
    auto ret = filter_retain(state.steps[m.step_b].retain_these, *sg, state.ret_entering[m.step_a]);
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
    std::set<size_t> expanded_set(expanded.begin(), expanded.end());
    // Output direction: does expanded set make a tensor ephemeral that strands others?
    if (Solution::creates_ephemeral_gap(prob, dag, expanded_set, state.steps, m.step_a))
      return {SIZE_MAX, 0};
    // Input direction: does expanded set need a tensor that's ephemeral elsewhere?
    if (sol_inputs_unavailable(prob, dag, expanded_set, state.steps, m.step_a))
      return {SIZE_MAX, 0};
    auto sg = Subgraph::create(prob, dag, expanded);
    if (!sg)
      return {SIZE_MAX, 0};
    auto ret = filter_retain(state.steps[m.step_a].retain_these, *sg, state.ret_entering[m.step_a]);
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
    
    auto rr = filter_retain(si.retain_these, *sg_r, state.ret_entering[m.step_a]);
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
    
    // sg_b's entering is br (retained from sg_a); exclude those from its retain_these
    auto vr = filter_retain(state.steps[m.step_a].retain_these, *sg_b, br);
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
    // Verify and update step_a+1 with new entering set.
    // Overlap between new entering and next step's retain is now safe.
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
  case SolutionMove::RETAIN_REMOVE: {
    if (m.step_a >= state.size())
      return {SIZE_MAX, 0};
    auto &si = state.steps[m.step_a];
    auto nr = si.retain_these;
    nr.erase(m.tensor);
    auto ci = si.subgraph.best_cost(state.ret_entering[m.step_a], nr);
    if (!ci.feasible)
      return {SIZE_MAX, 0};
    // Verify and update step_a+1 with new entering set.
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
  case SolutionMove::DE_RECOMPUTE: {
    if (m.step_a >= state.size())
      return {SIZE_MAX, 0};
    auto &si = state.steps[m.step_a];
    auto si_ops = si.subgraph.ops();
    std::set<size_t> si_set(si_ops.begin(), si_ops.end());

    // Re-verify: op is still in another step
    bool in_other = false;
    for (size_t sj = 0; sj < state.size(); sj++) {
      if (sj == m.step_a) continue;
      for (auto o : state.steps[sj].subgraph.ops())
        if (o == m.op) { in_other = true; break; }
      if (in_other) break;
    }
    if (!in_other) return {SIZE_MAX, 0};

    // Re-verify: boundary outputs produced by op are available elsewhere
    for (auto t : prob.ops[m.op].outputs) {
      if (!si.subgraph.boundary_outputs().count(t)) continue;
      bool avail = false;
      for (size_t sj = 0; sj < state.size(); sj++) {
        if (sj == m.step_a) continue;
        if (state.steps[sj].subgraph.boundary_outputs().count(t)) {
          avail = true; break;
        }
      }
      if (!avail) return {SIZE_MAX, 0};
    }

    if (si_set.size() == 1) {
      // Singleton: remove entire step
      state.steps.erase(state.steps.begin() + m.step_a);
      return {m.step_a > 0 ? m.step_a - 1 : 0, state.size()};
    } else {
      // Multi-op: remove op, rebuild remainder
      std::set<size_t> remainder = si_set;
      remainder.erase(m.op);
      auto sg_r = Subgraph::create(prob, dag, {remainder.begin(), remainder.end()});
      if (!sg_r) return {SIZE_MAX, 0};
      auto rr = filter_retain(si.retain_these, *sg_r, state.ret_entering[m.step_a]);
      auto cr = sg_r->best_cost(state.ret_entering[m.step_a], rr);
      if (!cr.feasible) return {SIZE_MAX, 0};
      si.subgraph = std::move(*sg_r);
      si.config = cr.config;
      si.retain_these = rr;
      return {m.step_a, state.size()};
    }
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
    if (!state.is_feasibly_retainable(t))
      return;  // precomputed: can never be retained
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
                        Clock::time_point deadline,
                        const std::set<size_t> *fr) {
  SolState state;
  state.init(prob, dag, std::move(steps), fr);
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

    double total_before = state.total;
    auto [lo, hi] = apply_move(state, m);
    if (lo == SIZE_MAX)
      continue;

    bool structural = is_structural(m.type);
    bump_affected(state, lo, hi, structural);
    state.rebuild_from(lo);
    applied++;

#ifndef NDEBUG
    {
      double actual_gain = total_before - state.total;
      double discrepancy = m.saving - actual_gain;
      if (std::abs(discrepancy) > 0.1 * std::max(1.0, std::abs(m.saving)) + 1.0) {
        std::cerr << "    GAIN MISMATCH: predicted=" << m.saving
                    << " actual=" << actual_gain
                    << " Δ=" << discrepancy
                    << " type=" << (int)m.type
                    << " steps=" << m.step_a << "," << m.step_b << "\n";
      }
    }
#endif

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
                                      const SolutionFMPassConfig &cfg,
                                      const std::set<size_t> *fr) {
  SolState state;
  state.init(prob, dag, std::move(steps), fr);

  SolutionFMPassResult result;
  result.start_cost = state.total;
  result.best_cost = state.total;
  result.best_steps = state.steps;

  double floor = result.start_cost * cfg.floor_fraction;
  double max_drift = result.start_cost * cfg.max_drift_fraction;

  SolActiveSet active;
  active.floor = floor;
  active.deadline = cfg.deadline;

  // Initialize: seed init_count entities (biased 2:1 toward tensors)
  std::mt19937 rng(cfg.seed);
  std::vector<size_t> all_tensors;
  for (auto t : (state.feasibly_retainable ? *state.feasibly_retainable : prob.retainable_tensors))
    all_tensors.push_back(t);
  std::vector<size_t> all_ops;
  for (size_t i = 0; i < state.size(); i++)
    for (auto op : state.steps[i].subgraph.ops())
      all_ops.push_back(op);
  std::sort(all_ops.begin(), all_ops.end());
  all_ops.erase(std::unique(all_ops.begin(), all_ops.end()), all_ops.end());

  std::shuffle(all_tensors.begin(), all_tensors.end(), rng);
  std::shuffle(all_ops.begin(), all_ops.end(), rng);

  int seeds_added = 0;
  size_t t_idx = 0, o_idx = 0;
  while (seeds_added < cfg.init_count && Clock::now() < cfg.deadline) {
    // 2:1 bias toward tensors
    bool pick_tensor = (rng() % 3 != 0) && t_idx < all_tensors.size();
    if (pick_tensor) {
      active.activate_tensor(state, all_tensors[t_idx]);
      auto steps_of = find_steps_of_tensor(state.steps, all_tensors[t_idx]);
      for (auto si : steps_of) {
        active.activate_step(state, si);
        if (si > 0) active.activate_step(state, si - 1);
        if (si + 1 < state.size()) active.activate_step(state, si + 1);
      }
      t_idx++;
      seeds_added++;
    } else if (o_idx < all_ops.size()) {
      active.activate_op(state, all_ops[o_idx]);
      size_t si = state.step_of(all_ops[o_idx]);
      if (si != SIZE_MAX) {
        active.activate_step(state, si);
        if (si > 0) active.activate_step(state, si - 1);
        if (si + 1 < state.size()) active.activate_step(state, si + 1);
      }
      o_idx++;
      seeds_added++;
    } else if (t_idx < all_tensors.size()) {
      // Fallback: use tensor even if we wanted op
      active.activate_tensor(state, all_tensors[t_idx]);
      auto steps_of = find_steps_of_tensor(state.steps, all_tensors[t_idx]);
      for (auto si : steps_of) {
        active.activate_step(state, si);
        if (si > 0) active.activate_step(state, si - 1);
        if (si + 1 < state.size()) active.activate_step(state, si + 1);
      }
      t_idx++;
      seeds_added++;
    } else {
      break; // exhausted both
    }
  }

  double cumul_gain = 0, best_cumul_gain = 0;
  int no_improve = 0;
  int max_no_improve = std::max(30, cfg.max_no_improve);

  for (int iter = 0; ; iter++) {
    // --- Stopping criteria ---
    if (Clock::now() >= cfg.deadline) break;
    if (no_improve >= max_no_improve) break;

    auto m_opt = active.pop_best();
    // Active set empty or best move below floor → stop
    if (!m_opt) break;
    if (m_opt->saving < -floor) break;

    double total_before = state.total;
    auto [lo, hi] = apply_move(state, *m_opt);
    if (lo == SIZE_MAX)
      continue;
    result.moves_applied++;
    state.rebuild_from(lo);

#ifndef NDEBUG
    {
      double actual_gain = total_before - state.total;
      double discrepancy = m_opt->saving - actual_gain;
      if (std::abs(discrepancy) > 0.1 * std::max(1.0, std::abs(m_opt->saving)) + 1.0) {
        std::cerr << "    FM GAIN MISMATCH: predicted=" << m_opt->saving
                    << " actual=" << actual_gain
                    << " Δ=" << discrepancy
                    << " type=" << (int)m_opt->type
                    << " steps=" << m_opt->step_a << "," << m_opt->step_b << "\n";
      }
    }
#endif

    cumul_gain = result.start_cost - state.total;
    if (state.total < result.best_cost - 0.01) {
      if (!has_ephemeral_gaps(dag, state.steps)) {
        result.best_cost = state.total;
        result.best_steps = state.steps;
        best_cumul_gain = cumul_gain;
      }
      no_improve = 0;
    } else {
      no_improve++;
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
// Solution FM outer loop: multiple passes with fresh seeds
// Each pass explores a different neighborhood. Best-seen is tracked globally.
// Stops after max_no_improve_passes consecutive passes without improvement.
// ============================================================================

static SolutionFMPassResult solution_fm_outer(const Problem &prob, const DAG &dag,
                                               std::vector<ScheduleStep> steps,
                                               const SolutionFMPassConfig &base_cfg,
                                               const std::set<size_t> *fr,
                                               int max_passes = 0,
                                               int max_no_improve_passes = 0) {
  if (max_passes <= 0) max_passes = 20;
  if (max_no_improve_passes <= 0) max_no_improve_passes = 5;

  SolutionFMPassResult best;
  best.start_cost = Solution(prob, dag, steps).total_latency();
  best.best_cost = best.start_cost;
  best.best_steps = steps;
  best.end_steps = steps;
  best.end_cost = best.start_cost;

  std::mt19937 seed_rng(base_cfg.seed);
  int no_improve_passes = 0;

  for (int pass = 0; pass < max_passes; pass++) {
    if (Clock::now() >= base_cfg.deadline) break;
    if (no_improve_passes >= max_no_improve_passes) break;

    SolutionFMPassConfig pc = base_cfg;
    pc.seed = (unsigned)(seed_rng());

    // Inner FM pass: seed → lock-based exploration → drift → stop
    auto pr = solution_fm_pass(prob, dag, best.best_steps, pc, fr);
    best.moves_applied += pr.moves_applied;

    // Greedy descent on end state: no locking, only improving moves.
    // The FM pass may have drifted to a worse state that has a different
    // local basin — greedy descent finds the bottom of that basin.
    if (pr.moves_applied > 0 && Clock::now() < base_cfg.deadline) {
      auto end_copy = pr.end_steps;  // preserve end state for diversity
      auto descended = solution_greedy_descent(prob, dag,
          std::move(end_copy), base_cfg.deadline, fr);
      Solution desc_sol(prob, dag, descended);
      if (desc_sol.total_latency() < pr.best_cost - 0.01) {
        pr.best_cost = desc_sol.total_latency();
        pr.best_steps = std::move(descended);
      }
    }

    if (pr.best_cost < best.best_cost - 0.01) {
      best.best_cost = pr.best_cost;
      best.best_steps = std::move(pr.best_steps);
      no_improve_passes = 0;
    } else {
      no_improve_passes++;
    }
    best.end_steps = pr.end_steps;
    best.end_cost = pr.end_cost;
  }
  return best;
}

// ============================================================================
// Parallel solution FM: N threads with different seeds, logging
// ============================================================================

#include <atomic>
#include <mutex>
#include <thread>

Solution solution_fm_search(const Problem &prob, const DAG &dag, Solution sol,
                            const SolutionFMConfig &cfg) {
  std::vector<Solution> pool;
  pool.push_back(std::move(sol));
  return solution_evo_search(prob, dag, std::move(pool), cfg);
}

// ============================================================================
// Solution crossover: agreement-based (like partition crossover)
//
// 1. Map ops to steps in both parents
// 2. Cluster ops that share the same step in BOTH parents (O(n) via hash)
// 3. Each cluster → validate as subgraph, split invalid ones to singletons
// 4. Topologically sort clusters
// 5. Greedily merge adjacent small clusters where profitable
// 6. Apply retain decisions from both parents
// ============================================================================

// Forward declaration (defined below with mutation helpers)
static void recompute_costs(const Problem& prob, const DAG& dag,
                            std::vector<ScheduleStep>& steps);

static std::vector<ScheduleStep> solution_crossover(
    const Problem& prob, const DAG& dag,
    const std::vector<ScheduleStep>& parent_a,
    const std::vector<ScheduleStep>& parent_b,
    std::mt19937& rng) {

    size_t n_ops = prob.num_ops();

    // Step 1: Map ops to step index in each parent
    std::vector<int> map_a(n_ops, -1), map_b(n_ops, -1);
    for (size_t i = 0; i < parent_a.size(); i++)
        for (auto op : parent_a[i].subgraph.ops()) map_a[op] = (int)i;
    for (size_t i = 0; i < parent_b.size(); i++)
        for (auto op : parent_b[i].subgraph.ops()) map_b[op] = (int)i;

    // Step 2: Agreement clusters — ops with same (step_a, step_b) pair
    std::map<std::pair<int,int>, std::vector<size_t>> clusters;
    for (size_t op = 0; op < n_ops; op++) {
        if (map_a[op] < 0 && map_b[op] < 0) continue;
        int a = map_a[op] >= 0 ? map_a[op] : -1;
        int b = map_b[op] >= 0 ? map_b[op] : -1;
        clusters[{a, b}].push_back(op);
    }

    // Step 3: Validate each cluster as a subgraph. Invalid → split to singletons.
    std::vector<std::vector<size_t>> valid_groups;
    for (auto& [key, ops] : clusters) {
        auto sg = Subgraph::create(prob, dag, ops);
        if (sg) {
            valid_groups.push_back(std::move(ops));
        } else {
            for (auto op : ops)
                valid_groups.push_back({op});
        }
    }

    // Step 4: Topological sort groups by inter-group DAG edges
    size_t ng = valid_groups.size();
    // Map each op to its group index
    std::vector<size_t> op_group(n_ops, SIZE_MAX);
    for (size_t g = 0; g < ng; g++)
        for (auto op : valid_groups[g])
            op_group[op] = g;

    std::vector<int> in_deg(ng, 0);
    std::vector<std::set<size_t>> adj(ng);
    for (size_t g = 0; g < ng; g++)
        for (auto op : valid_groups[g])
            for (auto succ : dag.op_succs[op]) {
                size_t sg = op_group[succ];
                if (sg != SIZE_MAX && sg != g && !adj[g].count(sg)) {
                    adj[g].insert(sg);
                    in_deg[sg]++;
                }
            }

    std::vector<size_t> order;
    std::vector<size_t> q;
    for (size_t g = 0; g < ng; g++)
        if (in_deg[g] == 0) q.push_back(g);
    while (!q.empty()) {
        // Random tie-breaking for diversity
        size_t pick = rng() % q.size();
        std::swap(q[pick], q.back());
        size_t u = q.back(); q.pop_back();
        order.push_back(u);
        for (auto v : adj[u])
            if (--in_deg[v] == 0) q.push_back(v);
    }
    if (order.size() != ng) return {};  // cycle (shouldn't happen)

    // Step 5: Build child steps in topological order.
    // Try merging each cluster with the previous step if profitable.
    std::vector<ScheduleStep> child_steps;
    for (auto gi : order) {
        auto& ops = valid_groups[gi];
        auto sg = Subgraph::create(prob, dag, ops);
        if (!sg) {
            // Shouldn't happen (validated above), but add singletons as fallback
            for (auto op : ops) {
                auto ss = Subgraph::create(prob, dag, {op});
                if (!ss) continue;
                auto bc = ss->best_cost({}, {});
                if (!bc.feasible) continue;
                child_steps.push_back({std::move(*ss), bc.config, {}});
            }
            continue;
        }

        // Try merging with previous step
        bool merged = false;
        if (!child_steps.empty()) {
            auto& prev = child_steps.back();
            auto prev_ops = prev.subgraph.ops();
            std::vector<size_t> combined(prev_ops.begin(), prev_ops.end());
            combined.insert(combined.end(), ops.begin(), ops.end());
            // Only merge if no cycle and Subgraph is valid
            auto sg_m = Subgraph::create(prob, dag, combined);
            if (sg_m) {
                auto bc_m = sg_m->best_cost({}, {});
                auto bc_sep = sg->best_cost({}, {});
                if (bc_m.feasible && bc_sep.feasible) {
                    double merged_cost = bc_m.latency;
                    double separate_cost = prev.subgraph.best_cost({}, {}).latency + bc_sep.latency;
                    if (merged_cost < separate_cost + 100) {
                        prev.subgraph = std::move(*sg_m);
                        prev.config = bc_m.config;
                        prev.retain_these.clear();
                        merged = true;
                    }
                }
            }
        }

        if (!merged) {
            auto bc = sg->best_cost({}, {});
            if (bc.feasible)
                child_steps.push_back({std::move(*sg), bc.config, {}});
        }
    }

    if (child_steps.empty()) return {};

    // Step 6: Apply retain decisions from both parents.
    // Collect all retain decisions — recompute_costs will prune invalid ones.
    std::set<size_t> retained_tensors;
    for (auto& s : parent_a)
        for (auto t : s.retain_these)
            retained_tensors.insert(t);
    for (auto& s : parent_b)
        for (auto t : s.retain_these)
            retained_tensors.insert(t);
    // Apply where structurally valid
    for (size_t i = 0; i + 1 < child_steps.size(); i++) {
        for (auto t : retained_tensors) {
            bool is_output = child_steps[i].subgraph.boundary_outputs().count(t);
            bool useful_next = child_steps[i+1].subgraph.boundary_inputs().count(t);
            if (is_output && useful_next && prob.retainable_tensors.count(t))
                child_steps[i].retain_these.insert(t);
        }
    }

    recompute_costs(prob, dag, child_steps);
    if (has_ephemeral_gaps(dag, child_steps)) return {};
    return child_steps;
}

// ============================================================================
// Solution mutation: random FM-style moves
// ============================================================================

// Helper: check if step i must come before step j (data dependency)
static bool steps_depend(const Problem& prob, const DAG& dag,
                         const ScheduleStep& a, const ScheduleStep& b) {
    std::set<size_t> b_ops(b.subgraph.ops().begin(), b.subgraph.ops().end());
    for (auto op_a : a.subgraph.ops())
        for (auto succ : dag.op_succs[op_a])
            if (b_ops.count(succ)) return true;
    return false;
}

// Helper: recompute costs for a step sequence (fixing retain entering sets)
static void recompute_costs(const Problem& prob, const DAG& dag,
                            std::vector<ScheduleStep>& steps) {
    std::set<size_t> entering;
    for (size_t i = 0; i < steps.size(); i++) {
        // Only boundary OUTPUTS that the NEXT step needs can be retained.
        // (One-step retention: retain at step i → available at step i+1 only.)
        std::set<size_t> valid_retain;
        for (auto t : steps[i].retain_these) {
            bool is_output = steps[i].subgraph.boundary_outputs().count(t);
            bool useful_next = (i + 1 < steps.size()) &&
                               steps[i + 1].subgraph.boundary_inputs().count(t);
            if (is_output && useful_next)
                valid_retain.insert(t);
        }
        steps[i].retain_these = valid_retain;

        auto bc = steps[i].subgraph.best_cost(entering, steps[i].retain_these);
        if (!bc.feasible) {
            steps[i].retain_these.clear();
            bc = steps[i].subgraph.best_cost(entering, {});
        }
        if (!bc.feasible) {
            entering.clear();
            steps[i].retain_these.clear();
            bc = steps[i].subgraph.best_cost({}, {});
        }
        steps[i].config = bc.config;
        entering = steps[i].retain_these;
    }
}

// Apply one random FM-style move to steps. Returns true if a move was applied.
static bool apply_one_random_move(const Problem& prob, const DAG& dag,
                                   std::vector<ScheduleStep>& steps, std::mt19937& rng) {
    if (steps.empty()) return false;

    // Weighted move types: SWAP(15) STEAL(25) MERGE(15) EJECT(15) SPLIT(10) RETAIN_ADD(10) RETAIN_REMOVE(10)
    int roll = rng() % 100;
    int move_type;
    if (roll < 15) move_type = 0;       // SWAP adjacent steps
    else if (roll < 40) move_type = 1;  // STEAL op between adjacent steps
    else if (roll < 55) move_type = 2;  // MERGE adjacent steps
    else if (roll < 70) move_type = 3;  // EJECT op to singleton
    else if (roll < 80) move_type = 4;  // SPLIT step
    else if (roll < 90) move_type = 5;  // RETAIN_ADD
    else move_type = 6;                 // RETAIN_REMOVE

    switch (move_type) {

    case 0: { // SWAP: swap two adjacent independent steps
        if (steps.size() < 2) return false;
        std::vector<size_t> swappable;
        for (size_t i = 0; i + 1 < steps.size(); i++)
            if (!steps_depend(prob, dag, steps[i], steps[i+1]) &&
                !steps_depend(prob, dag, steps[i+1], steps[i]))
                swappable.push_back(i);
        if (swappable.empty()) return false;
        size_t idx = swappable[rng() % swappable.size()];
        std::swap(steps[idx], steps[idx + 1]);
        // Retains may become invalid at the swap boundary — fix via recompute
        recompute_costs(prob, dag, steps);
        return true;
    }

    case 1: { // STEAL: move a random op from step_a to adjacent step_b
        if (steps.size() < 2) return false;
        // Collect all (step, op, neighbor_step) candidates
        struct StealCandidate { size_t src; size_t op; size_t dst; };
        std::vector<StealCandidate> cands;
        for (size_t i = 0; i < steps.size(); i++) {
            if (steps[i].subgraph.ops().size() < 2) continue; // don't empty a step
            for (auto op : steps[i].subgraph.ops()) {
                // Can steal to previous step?
                if (i > 0) cands.push_back({i, op, i - 1});
                // Can steal to next step?
                if (i + 1 < steps.size()) cands.push_back({i, op, i + 1});
            }
        }
        if (cands.empty()) return false;
        std::shuffle(cands.begin(), cands.end(), rng);

        for (auto& c : cands) {
            auto& src = steps[c.src];
            auto& dst = steps[c.dst];
            std::set<size_t> ns(src.subgraph.ops().begin(), src.subgraph.ops().end());
            ns.erase(c.op);
            std::set<size_t> nd(dst.subgraph.ops().begin(), dst.subgraph.ops().end());
            nd.insert(c.op);
            auto ss = Subgraph::create(prob, dag, {ns.begin(), ns.end()});
            auto sd = Subgraph::create(prob, dag, {nd.begin(), nd.end()});
            if (!ss || !sd) continue;
            if (Solution::creates_ephemeral_gap(prob, dag, nd, steps, c.dst, c.src)) continue;
            if (sol_inputs_unavailable(prob, dag, nd, steps, c.dst, c.src)) continue;
            // Topological check: op must not cross a data dependency boundary.
            bool topo_ok = true;
            if (c.dst < c.src) {
                // Moving LEFT (op goes to earlier step):
                // op must not depend on anything left behind in src.
                for (auto pred : dag.op_preds[c.op])
                    if (ns.count(pred)) { topo_ok = false; break; }
            } else {
                // Moving RIGHT (op goes to later step):
                // nothing left behind in src may depend on op.
                for (auto succ : dag.op_succs[c.op])
                    if (ns.count(succ)) { topo_ok = false; break; }
            }
            if (!topo_ok) continue;
            auto sr = filter_retain(src.retain_these, *ss);
            auto dr = filter_retain(dst.retain_these, *sd);
            // Check feasibility with empty entering (conservative — recompute fixes later)
            auto cs = ss->best_cost({}, sr);
            auto cd = sd->best_cost({}, dr);
            if (!cs.feasible || !cd.feasible) continue;
            src.subgraph = std::move(*ss);
            src.config = cs.config;
            src.retain_these = sr;
            dst.subgraph = std::move(*sd);
            dst.config = cd.config;
            dst.retain_these = dr;
            recompute_costs(prob, dag, steps);
            return true;
        }
        return false;
    }

    case 2: { // MERGE: merge two adjacent steps
        if (steps.size() < 2) return false;
        std::vector<size_t> cands;
        for (size_t i = 0; i + 1 < steps.size(); i++) {
            auto& ops_a = steps[i].subgraph.ops();
            auto& ops_b = steps[i+1].subgraph.ops();
            if (!dag.merge_creates_cycle({ops_a.begin(), ops_a.end()},
                                          {ops_b.begin(), ops_b.end()}))
                cands.push_back(i);
        }
        if (cands.empty()) return false;
        size_t si = cands[rng() % cands.size()];
        auto& ops_a = steps[si].subgraph.ops();
        auto& ops_b = steps[si+1].subgraph.ops();
        std::vector<size_t> merged(ops_a.begin(), ops_a.end());
        merged.insert(merged.end(), ops_b.begin(), ops_b.end());
        std::set<size_t> merged_set(merged.begin(), merged.end());
        if (Solution::creates_ephemeral_gap(prob, dag, merged_set, steps, si, si+1))
            return false;
        auto sg = Subgraph::create(prob, dag, merged);
        if (!sg) return false;
        auto ret = filter_retain(steps[si+1].retain_these, *sg);
        auto bc = sg->best_cost({}, ret);
        if (!bc.feasible) { ret.clear(); bc = sg->best_cost({}, {}); }
        if (!bc.feasible) return false;
        steps[si].subgraph = std::move(*sg);
        steps[si].config = bc.config;
        steps[si].retain_these = ret;
        steps.erase(steps.begin() + si + 1);
        recompute_costs(prob, dag, steps);
        return true;
    }

    case 3: { // EJECT: remove a random op from a multi-op step, create singleton
        std::vector<std::pair<size_t, size_t>> cands; // (step, op)
        for (size_t i = 0; i < steps.size(); i++)
            if (steps[i].subgraph.ops().size() >= 2)
                for (auto op : steps[i].subgraph.ops())
                    cands.push_back({i, op});
        if (cands.empty()) return false;
        std::shuffle(cands.begin(), cands.end(), rng);

        for (auto& [si, op] : cands) {
            auto& s = steps[si];
            std::set<size_t> rem(s.subgraph.ops().begin(), s.subgraph.ops().end());
            rem.erase(op);

            // Check ordering constraints
            bool must_before = false, must_after = false;
            for (auto succ : dag.op_succs[op]) if (rem.count(succ)) must_before = true;
            for (auto pred : dag.op_preds[op]) if (rem.count(pred)) must_after = true;
            if (must_before && must_after) continue;

            auto sg_r = Subgraph::create(prob, dag, {rem.begin(), rem.end()});
            auto sg_s = Subgraph::create(prob, dag, {op});
            if (!sg_r || !sg_s) continue;
            auto rr = filter_retain(s.retain_these, *sg_r);
            auto cr = sg_r->best_cost({}, rr);
            auto cs = sg_s->best_cost({}, {});
            if (!cr.feasible || !cs.feasible) continue;

            ScheduleStep step_r{std::move(*sg_r), cr.config, rr};
            ScheduleStep step_s{std::move(*sg_s), cs.config, {}};

            if (must_before) {
                steps[si] = std::move(step_s);
                steps.insert(steps.begin() + si + 1, std::move(step_r));
            } else {
                steps[si] = std::move(step_r);
                steps.insert(steps.begin() + si + 1, std::move(step_s));
            }
            recompute_costs(prob, dag, steps);
            return true;
        }
        return false;
    }

    case 4: { // SPLIT: pick a step with ≥3 ops, split randomly into two
        std::vector<size_t> cands;
        for (size_t i = 0; i < steps.size(); i++)
            if (steps[i].subgraph.ops().size() >= 3) cands.push_back(i);
        if (cands.empty()) return false;
        std::shuffle(cands.begin(), cands.end(), rng);

        for (auto si : cands) {
            auto& ops = steps[si].subgraph.ops();
            std::vector<size_t> ops_vec(ops.begin(), ops.end());
            std::shuffle(ops_vec.begin(), ops_vec.end(), rng);
            size_t split = 1 + rng() % (ops_vec.size() - 1);
            std::vector<size_t> side_a(ops_vec.begin(), ops_vec.begin() + split);
            std::vector<size_t> side_b(ops_vec.begin() + split, ops_vec.end());

            auto sg_a = Subgraph::create(prob, dag, side_a);
            auto sg_b = Subgraph::create(prob, dag, side_b);
            if (!sg_a || !sg_b) continue;

            // Preserve retains where possible
            auto ret_a = filter_retain(steps[si].retain_these, *sg_a);
            auto ret_b = filter_retain(steps[si].retain_these, *sg_b);
            auto ca = sg_a->best_cost({}, ret_a);
            auto cb = sg_b->best_cost({}, ret_b);
            if (!ca.feasible) { ret_a.clear(); ca = sg_a->best_cost({}, {}); }
            if (!cb.feasible) { ret_b.clear(); cb = sg_b->best_cost({}, {}); }
            if (!ca.feasible || !cb.feasible) continue;

            // Determine ordering
            std::set<size_t> set_a(side_a.begin(), side_a.end());
            std::set<size_t> set_b(side_b.begin(), side_b.end());
            bool a_before_b = false, b_before_a = false;
            for (auto op : side_a)
                for (auto succ : dag.op_succs[op])
                    if (set_b.count(succ)) a_before_b = true;
            for (auto op : side_b)
                for (auto succ : dag.op_succs[op])
                    if (set_a.count(succ)) b_before_a = true;
            if (a_before_b && b_before_a) continue; // cycle

            if (b_before_a) {
                std::swap(sg_a, sg_b); std::swap(ca, cb);
                std::swap(ret_a, ret_b);
            }

            std::vector<ScheduleStep> result;
            for (size_t i = 0; i < si; i++) result.push_back(steps[i]);
            result.push_back({std::move(*sg_a), ca.config, ret_a});
            result.push_back({std::move(*sg_b), cb.config, ret_b});
            for (size_t i = si + 1; i < steps.size(); i++) result.push_back(steps[i]);
            recompute_costs(prob, dag, result);
            steps = std::move(result);
            return true;
        }
        return false;
    }

    case 5: { // RETAIN_ADD: randomly add a retain decision
        if (steps.size() < 2) return false;
        std::vector<std::pair<size_t, size_t>> cands; // (step, tensor)
        for (size_t i = 0; i + 1 < steps.size(); i++) {
            for (auto t : steps[i].subgraph.boundary_outputs()) {
                if (prob.retainable_tensors.count(t) &&
                    steps[i+1].subgraph.boundary_inputs().count(t) &&
                    !steps[i].retain_these.count(t))
                    cands.push_back({i, t});
            }
            for (auto t : steps[i].subgraph.boundary_inputs()) {
                if (prob.retainable_tensors.count(t) &&
                    steps[i+1].subgraph.boundary_inputs().count(t) &&
                    !steps[i].retain_these.count(t))
                    cands.push_back({i, t});
            }
        }
        if (cands.empty()) return false;
        auto [si, t] = cands[rng() % cands.size()];
        steps[si].retain_these.insert(t);
        recompute_costs(prob, dag, steps);
        return true;
    }

    case 6: { // RETAIN_REMOVE: randomly remove a retain decision
        std::vector<std::pair<size_t, size_t>> cands;
        for (size_t i = 0; i < steps.size(); i++)
            for (auto t : steps[i].retain_these)
                cands.push_back({i, t});
        if (cands.empty()) return false;
        auto [si, t] = cands[rng() % cands.size()];
        steps[si].retain_these.erase(t);
        recompute_costs(prob, dag, steps);
        return true;
    }

    } // switch
    return false;
}

std::vector<ScheduleStep> mutate_random(const Problem& prob, const DAG& dag,
                                         const std::vector<ScheduleStep>& steps,
                                         std::mt19937& rng, int n_moves) {
    if (steps.empty()) return {};
    if (n_moves <= 0) n_moves = 3 + (int)(rng() % 4); // default: 3-6 moves

    auto result = steps;
    int applied  = 0;
    int attempts = 0;
    // 10x budget: enough retries even when many move types have no candidates.
    // Each apply_one_random_move internally shuffles candidates so repeated
    // attempts on the same type explore different ops/tensors each time.
    const int max_attempts = n_moves * 10;
    while (applied < n_moves && attempts < max_attempts) {
        attempts++;
        // Take a snapshot to detect whether the move actually changed the solution.
        // apply_one_random_move returns true if it applied any structural/retain
        // change — but we track this independently for accurate counting.
        size_t before_steps = result.size();
        std::set<size_t> before_retain;
        for (auto& s : result) before_retain.insert(s.retain_these.begin(), s.retain_these.end());

        if (apply_one_random_move(prob, dag, result, rng)) {
            // Verify the state actually changed (not a degenerate no-op)
            bool changed = (result.size() != before_steps);
            if (!changed) {
                std::set<size_t> after_retain;
                for (auto& s : result) after_retain.insert(s.retain_these.begin(), s.retain_these.end());
                changed = (after_retain != before_retain);
            }
            if (changed) applied++;
        }
    }
    if (applied == 0) return {};
    if (has_ephemeral_gaps(dag, result)) return {};
    return result;
}

// ============================================================================
// Solution evolutionary search
// ============================================================================

// Solution distance: ARI for grouping + ordering + retain disagreement.
// Uses contingency table for O(n + sa*sb) grouping instead of O(n²) pairwise.
static double solution_distance(const Problem& prob,
                                const std::vector<ScheduleStep>& a,
                                const std::vector<ScheduleStep>& b,
                                const MerkleHashes* mh = nullptr) {
    size_t n = prob.num_ops();
    // Map op → step index
    std::vector<int> map_a(n, -1), map_b(n, -1);
    int num_sa = (int)a.size(), num_sb = (int)b.size();
    for (size_t i = 0; i < a.size(); i++)
        for (auto op : a[i].subgraph.ops()) map_a[op] = (int)i;
    for (size_t i = 0; i < b.size(); i++)
        for (auto op : b[i].subgraph.ops()) map_b[op] = (int)i;

    // Apply Merkle canonicalisation before distance computation
    if (mh) merkle_canonicalise(*mh, map_a, map_b);

    // Component 1: grouping via Adjusted Rand Index (contingency table)
    double d_group = 0.0;
    if (num_sa > 0 && num_sb > 0) {
        std::vector<std::vector<int>> table(num_sa, std::vector<int>(num_sb, 0));
        std::vector<int> row_sum(num_sa, 0), col_sum(num_sb, 0);
        int total = 0;
        for (size_t op = 0; op < n; op++) {
            if (map_a[op] < 0 || map_b[op] < 0) continue;
            table[map_a[op]][map_b[op]]++;
            row_sum[map_a[op]]++;
            col_sum[map_b[op]]++;
            total++;
        }
        if (total > 1) {
            auto choose2 = [](int64_t x) -> int64_t { return x * (x - 1) / 2; };
            int64_t same_a = 0, same_b = 0, agree = 0;
            for (int i = 0; i < num_sa; i++) same_a += choose2(row_sum[i]);
            for (int j = 0; j < num_sb; j++) same_b += choose2(col_sum[j]);
            for (int i = 0; i < num_sa; i++)
                for (int j = 0; j < num_sb; j++)
                    agree += choose2(table[i][j]);
            int64_t total_pairs = choose2(total);
            double expected = (double)same_a * same_b / total_pairs;
            double max_agree = (double)(same_a + same_b) / 2.0;
            double denom = max_agree - expected;
            if (std::abs(denom) > 1e-12) {
                double ari = std::clamp(((double)agree - expected) / denom, 0.0, 1.0);
                d_group = 1.0 - ari;
            }
        }
    }

    // Component 2: ordering disagreement via Kendall tau-like metric.
    // Only count pairs in different steps in at least one solution.
    // Use step indices directly — O(n²) but n ≤ ~100 so ~5000 pairs.
    int ord_disagree = 0, ord_total = 0;
    for (size_t i = 0; i < n; i++) {
        if (map_a[i] < 0 || map_b[i] < 0) continue;
        for (size_t j = i + 1; j < n; j++) {
            if (map_a[j] < 0 || map_b[j] < 0) continue;
            if (map_a[i] == map_a[j] && map_b[i] == map_b[j]) continue;
            ord_total++;
            bool a_i_first = (map_a[i] < map_a[j]);
            bool b_i_first = (map_b[i] < map_b[j]);
            if (a_i_first != b_i_first) ord_disagree++;
        }
    }
    double d_order = ord_total > 0 ? (double)ord_disagree / ord_total : 0.0;

    // Component 3: retain disagreement (symmetric difference)
    std::set<size_t> ret_a, ret_b;
    for (auto& s : a) for (auto t : s.retain_these) ret_a.insert(t);
    for (auto& s : b) for (auto t : s.retain_these) ret_b.insert(t);
    int ret_union = 0, ret_diff = 0;
    std::set<size_t> all_ret;
    all_ret.insert(ret_a.begin(), ret_a.end());
    all_ret.insert(ret_b.begin(), ret_b.end());
    ret_union = (int)all_ret.size();
    for (auto t : all_ret) {
        bool in_a = ret_a.count(t), in_b = ret_b.count(t);
        if (in_a != in_b) ret_diff++;
    }
    double d_retain = ret_union > 0 ? (double)ret_diff / ret_union : 0.0;

    // Weighted combination
    return 0.5 * d_group + 0.3 * d_order + 0.2 * d_retain;
}

Solution solution_evo_search(const Problem &prob, const DAG &dag,
                              std::vector<Solution> init_pool,
                              const SolutionFMConfig &cfg) {
    if (init_pool.empty()) return Solution(prob, dag, {});

    using Clock = std::chrono::steady_clock;

    int hw_threads = (int)std::thread::hardware_concurrency();
    if (hw_threads <= 0) hw_threads = 4;
    int num_threads = hw_threads;

    // --- Shared pool (thread-safe) ---
    struct SolPoolEntry {
        std::vector<ScheduleStep> steps;
        double cost;
    };
    const size_t MAX_POOL = 16;

    std::vector<SolPoolEntry> pool;
    std::mutex pool_mutex;
    double global_best_cost = 1e18;

    // Symmetry-aware pool deduplication
    MerkleHashes sol_mh = MerkleHashes::compute(prob, dag);
    const MerkleHashes* sol_mhp = sol_mh.symmetric_ops() > 0 ? &sol_mh : nullptr;

    // Pool insert is defined before initialization so we can use it
    // for diversity-aware seeding too.
    auto pool_insert = [&](std::vector<ScheduleStep> steps, double cost) -> bool {
        if (has_ephemeral_gaps(dag, steps)) return false;

        std::lock_guard<std::mutex> lock(pool_mutex);

        // Compute min distance to any existing pool entry
        double min_dist = 1.0;
        size_t closest_idx = 0;
        for (size_t i = 0; i < pool.size(); i++) {
            double dist = solution_distance(prob, pool[i].steps, steps, sol_mhp);
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = i;
            }
        }

        // Near-duplicate: only replace if strictly better
        // ARI-based distance: 0 = identical, ~1 = maximally different
        if (min_dist < 0.05) {
            if (cost < pool[closest_idx].cost - 0.01) {
                pool[closest_idx].steps = std::move(steps);
                pool[closest_idx].cost = cost;
                if (cost < global_best_cost) global_best_cost = cost;
                return true;
            }
            return false;
        }

        // Pool not full: always add
        if (pool.size() < MAX_POOL) {
            pool.push_back({std::move(steps), cost});
            if (cost < global_best_cost) global_best_cost = cost;
            return true;
        }

        // Pool full: diversity-aware eviction.
        size_t best_cost_idx = 0;
        for (size_t i = 1; i < pool.size(); i++)
            if (pool[i].cost < pool[best_cost_idx].cost) best_cost_idx = i;

        // Find the least-unique entry (smallest nearest-neighbor distance), excluding best
        size_t least_unique = SIZE_MAX;
        double least_unique_dist = 2.0;
        for (size_t i = 0; i < pool.size(); i++) {
            if (i == best_cost_idx) continue;
            double nn_dist = 1.0;
            for (size_t j = 0; j < pool.size(); j++) {
                if (i == j) continue;
                double d = solution_distance(prob, pool[i].steps, pool[j].steps, sol_mhp);
                nn_dist = std::min(nn_dist, d);
            }
            if (nn_dist < least_unique_dist) {
                least_unique_dist = nn_dist;
                least_unique = i;
            }
        }

        if (least_unique == SIZE_MAX) return false;

        bool more_diverse = (min_dist > least_unique_dist + 0.05);
        bool better_cost = (cost < pool[least_unique].cost - 0.01);
        bool decent_diversity = (min_dist > 0.10);

        if (more_diverse || (better_cost && decent_diversity)) {
            pool[least_unique] = {std::move(steps), cost};
            if (cost < global_best_cost) global_best_cost = cost;
            return true;
        }
        return false;
    };

    // Initialize pool from input solutions — diversity-filtered
    // Sort by cost first so the best entry enters first and is protected
    std::sort(init_pool.begin(), init_pool.end(),
              [](const Solution& a, const Solution& b) {
                  return a.total_latency() < b.total_latency();
              });
    for (auto& sol : init_pool)
        pool_insert(sol.steps(), sol.total_latency());

    double starting_best = global_best_cost;

    // One-time feasibility check: depends only on Problem + DAG
    auto fr = compute_feasibly_retainable(prob, dag);
    std::cerr << "  Feasibly retainable: " << fr.size()
              << "/" << prob.retainable_tensors.size() << " tensors\n";

    std::cerr << "  Init pool: " << pool.size() << " diverse entries from "
              << init_pool.size() << " candidates\n";

    // --- Stats ---
    std::atomic<int> total_mutations{0}, total_fm_passes{0};
    std::atomic<int> total_improvements{0}, total_fm_improvements{0};
    std::atomic<int> total_generations{0};

    // --- Worker threads ---
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(cfg.pass_config.seed + t * 997);
            auto thread_start = Clock::now();
            int local_gen = 0;
            int local_no_improve = 0;
            double heat = 1.0;

            double base_floor = cfg.pass_config.floor_fraction;
            double base_drift = cfg.pass_config.max_drift_fraction;
            double thread_best_cost;
            {
                std::lock_guard<std::mutex> lock(pool_mutex);
                thread_best_cost = global_best_cost;
            }

            while (Clock::now() < cfg.deadline) {
                local_gen++;
                total_generations++;

                // --- Temperature: cosine annealing based on wall-clock progress ---
                auto elapsed = std::chrono::duration<double>(Clock::now() - thread_start).count();
                auto budget = std::chrono::duration<double>(cfg.deadline - thread_start).count();
                double progress = std::clamp(elapsed / std::max(budget, 0.1), 0.0, 1.0);
                double temp = 0.1 + 0.9 * 0.5 * (1.0 + std::cos(progress * M_PI));
                double eff_floor = std::clamp(base_floor * temp * heat, 0.02, 1.0);
                double eff_drift = std::clamp(base_drift * temp * heat, 0.05, 2.0);

                // --- Select operator: Crossover (1/3) or Mutation (2/3) ---
                std::vector<ScheduleStep> child;
                bool do_crossover = (rng() % 3 == 0);

                std::vector<ScheduleStep> parent, parent2;
                {
                    std::lock_guard<std::mutex> lock(pool_mutex);
                    if (pool.empty()) break;
                    size_t idx1 = rng() % pool.size();
                    parent = pool[idx1].steps;
                    if (do_crossover && pool.size() >= 2) {
                        size_t idx2 = rng() % pool.size();
                        while (idx2 == idx1) idx2 = rng() % pool.size();
                        parent2 = pool[idx2].steps;
                    }
                }

                if (do_crossover && !parent2.empty())
                    child = solution_crossover(prob, dag, parent, parent2, rng);
                if (child.empty()) {
                    // Mutation path (2/3 of the time, or crossover fallback)
                    int n_moves = 3 + (int)(rng() % 4);  // base: 3..6
                    // Scale with heat: stagnation → more mutations
                    n_moves = std::max(2, (int)(n_moves * heat));
                    child = mutate_random(prob, dag, parent, rng, n_moves);
                }
                if (child.empty()) continue;
                total_mutations++;

                // Validate
                Solution child_sol(prob, dag, child);
                if (!child_sol.validate().valid) continue;

                double child_cost = child_sol.total_latency();

                // Try inserting mutant (diversity-aware pool handles filtering)
                pool_insert(child_sol.steps(), child_cost);

                bool improved = false;
                if (child_cost < thread_best_cost - 0.01) {
                    thread_best_cost = child_cost;
                    total_improvements++;
                    improved = true;
                }

                // --- Polish: FM outer loop (multiple passes with fresh seeds) ---
                if (Clock::now() < cfg.deadline) {
                    SolutionFMPassConfig pc = cfg.pass_config;
                    pc.seed = (unsigned)(rng());
                    pc.deadline = cfg.deadline;
                    pc.floor_fraction = eff_floor;
                    pc.max_drift_fraction = eff_drift;

                    // Outer loop: more passes early (temp high), fewer late (temp low)
                    int max_passes = std::clamp((int)(200 * temp * heat), 100, 250);
                    int max_no_imp_passes = std::clamp((int)(50 * temp * heat), 25, 50);

                    auto pr = solution_fm_outer(prob, dag, child_sol.steps(), pc, &fr,
                                                max_passes, max_no_imp_passes);
                    total_fm_passes++;

                    if (pr.best_cost < child_cost - 0.01) {
                        total_fm_improvements++;
                        Solution fm_sol(prob, dag, pr.best_steps);
                        pool_insert(fm_sol.steps(), fm_sol.total_latency());
                        if (fm_sol.total_latency() < thread_best_cost - 0.01) {
                            thread_best_cost = fm_sol.total_latency();
                            improved = true;
                        }
                    }

                    // Greedy-kick on the perturbed end state — may escape to a
                    // different basin than what FM found as best.
                    if (pr.end_cost < 1e17 && pr.moves_applied > 0 &&
                        Clock::now() < cfg.deadline) {
                        auto kicked = solution_greedy_descent(
                            prob, dag, std::move(pr.end_steps), cfg.deadline, &fr);
                        Solution kick_sol(prob, dag, kicked);
                        if (kick_sol.validate().valid) {
                            pool_insert(kick_sol.steps(), kick_sol.total_latency());
                            if (kick_sol.total_latency() < thread_best_cost - 0.01) {
                                thread_best_cost = kick_sol.total_latency();
                                improved = true;
                            }
                        }
                    }
                }

                // Adapt heat based on improvement
                if (improved) {
                    local_no_improve = 0;
                    heat = std::clamp(heat * 0.7, 0.1, 3.0);
                } else {
                    local_no_improve++;
                    heat = std::clamp(heat * 1.1, 0.1, 3.0);
                }
            }
        });
    }

    for (auto& t : threads)
        t.join();

    // Sort pool, return best
    std::sort(pool.begin(), pool.end(),
              [](const SolPoolEntry& a, const SolPoolEntry& b) { return a.cost < b.cost; });

    std::cerr << "  Sol-Evo: " << num_threads << " threads, "
              << total_generations.load() << " generations, "
              << total_mutations.load() << " mutations ("
              << total_improvements.load() << " improving), "
              << total_fm_passes.load() << " FM passes ("
              << total_fm_improvements.load() << " improving)";
    if (!pool.empty() && pool[0].cost < starting_best - 0.01) {
        std::cerr << ", improved " << starting_best << " → " << pool[0].cost
                  << " (-" << std::fixed << std::setprecision(2)
                  << 100.0 * (starting_best - pool[0].cost) / starting_best << "%)";
    }
    std::cerr << ", pool=" << pool.size() << " entries\n";

    if (pool.empty()) return init_pool.empty()
        ? Solution(prob, dag, {})
        : Solution(prob, dag, init_pool[0].steps());
    return Solution(prob, dag, std::move(pool[0].steps));
}