#include "core/subgraph.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>

// ============================================================================
// Utility
// ============================================================================

static std::vector<int64_t> all_divisors(int64_t n) {
  std::vector<int64_t> result;
  for (int64_t i = 1; i * i <= n; i++) {
    if (n % i == 0) {
      result.push_back(i);
      if (i != n / i)
        result.push_back(n / i);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

// ============================================================================
// Factory
// ============================================================================

std::optional<Subgraph> Subgraph::create(const Problem &prob, const DAG &dag,
                                         std::vector<size_t> op_indices,
                                         const std::set<size_t> &force_ephemeral) {
  if (op_indices.empty())
    return std::nullopt;

  Subgraph sg;
  sg.prob_ = &prob;
  sg.dag_ = &dag;
  sg.ops_ = std::move(op_indices);

  const size_t num_tensors = prob.num_tensors();
  const size_t num_ops = prob.num_ops();

  // Use vectors indexed by ID instead of maps/sets for O(1) lookups
  std::vector<bool> is_in_sg(num_ops, false);
  std::vector<bool> is_produced(num_tensors, false);
  std::vector<bool> is_consumed(num_tensors, false);

  for (auto i : sg.ops_) {
    is_in_sg[i] = true;
    for (auto t : prob.ops[i].outputs)
      is_produced[t] = true;
    for (auto t : prob.ops[i].inputs)
      is_consumed[t] = true;
    if (prob.ops[i].type == OpType::MatMul) {
      sg.has_matmul_ = true;
      int64_t Ki = prob.tensors[prob.ops[i].inputs[0]].width;
      sg.max_K_ = std::max(sg.max_K_, Ki);
    }
  }

  // Classify tensors.
  //
  // A tensor produced AND consumed within the subgraph is ephemeral ONLY IF
  // all of its DAG consumers are inside the subgraph.  It lives only in the
  // tile pipeline and never touches fast or slow memory.  Zero capacity,
  // zero transfer cost.
  //
  // If ANY consumer is external, the tensor is a boundary output — it must
  // be materialized to slow memory so external subgraphs can read it.
  // Internally it still occupies fast memory for the consuming op (tracked
  // via is_internally_produced in BoundaryTensorInfo).
  std::vector<bool> is_ephemeral(num_tensors, false);

  for (size_t t = 0; t < num_tensors; t++) {
    if (is_consumed[t] && !is_produced[t])
      sg.boundary_inputs_.insert(t);
    if (is_produced[t] && is_consumed[t]) {
      // Ephemeral if ALL DAG consumers are inside, OR if force_ephemeral
      // says all external consumers are served by recomputing groups.
      bool all_internal = force_ephemeral.count(t) > 0;
      if (!all_internal) {
        all_internal = true;
        for (auto cop : dag.tensor_consumers[t])
          if (!is_in_sg[cop]) { all_internal = false; break; }
      }
      if (all_internal)
        is_ephemeral[t] = true;
    }
  }
  for (size_t t = 0; t < num_tensors; t++) {
    if (is_produced[t] && !is_ephemeral[t])
      sg.boundary_outputs_.insert(t);
    if (is_ephemeral[t])
      sg.ephemeral_.insert(t);
  }

  // Note: ephemeral tensors MAY have multiple internal consumers (fan-out
  // within the subgraph is permitted). All consumers add tiling constraints;
  // the GCD-based candidate generation ensures compatibility.

  // Must have at least one boundary output
  if (sg.boundary_outputs_.empty())
    return std::nullopt;

  // Detect PW sinks (using vector for producer lookup, not map)
  // Slice-role enums and resolved roles are declared here (outer scope) so
  // they remain accessible during BoundaryTensorInfo population below.
  enum class SliceW : uint8_t { W_param, K_param };
  enum class SliceH : uint8_t { H_param, K_param };
  std::vector<SliceW> eph_sw(num_tensors, SliceW::W_param);
  std::vector<SliceH> eph_sh(num_tensors, SliceH::H_param);

  {
    std::vector<int> tensor_producer_in_sg(num_tensors, -1);
    for (auto i : sg.ops_)
      for (auto t : prob.ops[i].outputs)
        tensor_producer_in_sg[t] = (int)i;

    // All boundary outputs must have the same dimensions
    auto it = sg.boundary_outputs_.begin();
    sg.out_W_ = prob.tensors[*it].width;
    sg.out_H_ = prob.tensors[*it].height;
    for (++it; it != sg.boundary_outputs_.end(); ++it) {
      if (prob.tensors[*it].width != sg.out_W_ ||
          prob.tensors[*it].height != sg.out_H_)
        return std::nullopt;
    }

    for (auto t : sg.boundary_outputs_) {
      int prod = tensor_producer_in_sg[t];
      if (prod >= 0 && prob.ops[prod].type == OpType::Pointwise) {
        sg.has_pw_sink_ = true;
        break;
      }
    }

    // Compute output_K_: the reduction dimension that drives the temporal
    // loop (nk = output_K / k).
    //
    // When the boundary output is produced by a MatMul, output_K_ is that
    // MatMul's reduction dimension.  nk = output_K_ / k splits the
    // reduction into temporal steps.
    //
    // When a Pointwise produces the boundary output (has_pw_sink_), k is
    // forced to 1 and output_K_ = 1, giving nk = 1.  This means NO
    // temporal splitting: the MatMul performs its full reduction in a
    // single step, and the PW fires immediately after.
    // Confirmed by organizer in issue #32: "With split-K, the MatMul
    // takes 4 passes to accumulate its result, and the Pointwise cannot
    // participate in those k-steps."
    if (!sg.has_pw_sink_) {
      for (auto t : sg.boundary_outputs_) {
        int prod = tensor_producer_in_sg[t];
        if (prod >= 0 && prob.ops[prod].type == OpType::MatMul) {
          sg.output_K_ = sg.op_K(prod);
          break;  // all boundary outputs have same dims; first suffices
        }
      }
    }

    // ---- Collect role-based tiling constraints ----
    //
    // For each tensor (boundary or ephemeral), determine what tiling axes
    // its dimensions map to, based on how it's consumed.  With multi-consumer
    // ephemerals, a tensor may have MULTIPLE roles — each adds constraints.
    //
    // Role resolution for an ephemeral tensor: for each internal consumer,
    // determine the demanded role.  If the consumer is a PW, propagate
    // through the PW output (which may itself be ephemeral or boundary).

    // Build consumer lists for ephemeral tensors (may have multiple)
    std::vector<std::vector<size_t>> eph_consumers(num_tensors);
    for (auto i : sg.ops_)
      for (auto t : prob.ops[i].inputs)
        if (is_ephemeral[t])
          eph_consumers[t].push_back(i);

    // Resolve all (sw, sh) roles for an ephemeral tensor via DFS through
    // PW chains.  Returns a vector of (SliceW, SliceH) pairs.
    // Uses a visited set to prevent infinite loops in pathological cases.
    std::vector<bool> resolving(num_tensors, false);

    // Forward declaration for mutual recursion (lambda captures itself)
    struct RolePair { SliceW sw; SliceH sh; };
    std::vector<std::vector<RolePair>> eph_roles(num_tensors);
    std::vector<bool> eph_roles_computed(num_tensors, false);

    // Iterative role resolution using a worklist
    // Process in reverse topological order of the subgraph ops so that
    // downstream ephemeral roles are resolved before upstream ones.
    {
      // Collect all ephemeral tensors in reverse topo order of their consumers
      std::vector<size_t> eph_order;
      for (auto op_idx : dag.topological_order())
        if (is_in_sg[op_idx])
          for (auto t : prob.ops[op_idx].outputs)
            if (is_ephemeral[t])
              eph_order.push_back(t);

      // Process in reverse: sinks first, sources last
      for (int ei = (int)eph_order.size() - 1; ei >= 0; ei--) {
        size_t t = eph_order[ei];
        if (eph_roles_computed[t]) continue;

        for (auto cop : eph_consumers[t]) {
          const auto &op = prob.ops[cop];
          if (op.type == OpType::MatMul) {
            if (op.inputs[0] == t)
              eph_roles[t].push_back({SliceW::K_param, SliceH::H_param});
            else
              eph_roles[t].push_back({SliceW::W_param, SliceH::K_param});
          } else {
            // PW: role is same as PW output's role
            size_t pw_out = op.outputs[0];
            if (sg.boundary_outputs_.count(pw_out)) {
              eph_roles[t].push_back({SliceW::W_param, SliceH::H_param});
            } else if (is_ephemeral[pw_out] && eph_roles_computed[pw_out]) {
              // Propagate all roles from the PW output
              for (auto &r : eph_roles[pw_out])
                eph_roles[t].push_back(r);
            } else {
              // Fallback: boundary or unresolved → w×h
              eph_roles[t].push_back({SliceW::W_param, SliceH::H_param});
            }
          }
        }
        eph_roles_computed[t] = true;
      }
    }

    // For backward compatibility, store the first resolved role in eph_sw/eph_sh
    // (used by BoundaryTensorInfo classification for PW inputs)
    for (size_t t = 0; t < num_tensors; t++) {
      if (is_ephemeral[t] && !eph_roles[t].empty()) {
        eph_sw[t] = eph_roles[t][0].sw;
        eph_sh[t] = eph_roles[t][0].sh;
      }
    }

    // Collect constraints into sets
    std::set<int64_t> w_set, h_set, k_set;

    auto add_constraint = [&](size_t t, SliceW sw, SliceH sh) {
      int64_t W = prob.tensors[t].width;
      int64_t H = prob.tensors[t].height;
      if (sw == SliceW::W_param) w_set.insert(W); else k_set.insert(W);
      if (sh == SliceH::H_param) h_set.insert(H); else k_set.insert(H);
    };

    for (auto i : sg.ops_) {
      const auto &op = prob.ops[i];
      if (op.type == OpType::MatMul) {
        add_constraint(op.inputs[0], SliceW::K_param, SliceH::H_param);
        add_constraint(op.inputs[1], SliceW::W_param, SliceH::K_param);
        size_t out = op.outputs[0];
        if (sg.boundary_outputs_.count(out)) {
          add_constraint(out, SliceW::W_param, SliceH::H_param);
        } else if (is_ephemeral[out]) {
          // Add constraints for ALL roles this ephemeral serves
          for (auto &r : eph_roles[out])
            add_constraint(out, r.sw, r.sh);
        }
      } else {
        size_t out = op.outputs[0];
        if (sg.boundary_outputs_.count(out)) {
          add_constraint(out, SliceW::W_param, SliceH::H_param);
          for (auto t : op.inputs)
            add_constraint(t, SliceW::W_param, SliceH::H_param);
        } else if (is_ephemeral[out]) {
          // PW with ephemeral output: add constraints for ALL roles
          for (auto &r : eph_roles[out]) {
            add_constraint(out, r.sw, r.sh);
            for (auto t : op.inputs)
              add_constraint(t, r.sw, r.sh);
          }
          // Fallback: if no roles resolved, use w×h
          if (eph_roles[out].empty()) {
            add_constraint(out, SliceW::W_param, SliceH::H_param);
            for (auto t : op.inputs)
              add_constraint(t, SliceW::W_param, SliceH::H_param);
          }
        }
      }
    }

    sg.w_divides_.assign(w_set.begin(), w_set.end());
    sg.h_divides_.assign(h_set.begin(), h_set.end());
    sg.k_divides_.assign(k_set.begin(), k_set.end());
  }

  // Validate: ops form a connected group using precomputed op_neighbors
  if (sg.ops_.size() > 1) {
    std::vector<bool> visited(num_ops, false);
    std::vector<size_t> bfs = {sg.ops_[0]};
    visited[sg.ops_[0]] = true;
    size_t visit_count = 1;

    while (!bfs.empty()) {
      size_t u = bfs.back();
      bfs.pop_back();
      for (auto v : dag.op_neighbors[u]) {
        if (is_in_sg[v] && !visited[v]) {
          visited[v] = true;
          visit_count++;
          bfs.push_back(v);
        }
      }
    }
    if (visit_count != sg.ops_.size())
      return std::nullopt;
  }

  // ---- Precompute per-boundary-tensor tiling via backward propagation ----
  //
  // Matches the reference evaluator's h_tiles/v_tiles system exactly.
  // Process ops in reverse topological order; for each op, propagate
  // tiling from its output to its inputs.
  {
    using TS = BoundaryTensorInfo::TileSource;

    // Per-tensor tiling sources (for ALL tensors, not just boundary)
    struct TilePair { TS h; TS v; bool assigned = false; };
    std::vector<TilePair> tsrc(num_tensors);

    // Determine is_sink per op (no internal consumer of output)
    std::vector<bool> is_sink_op(num_ops, false);
    for (auto i : sg.ops_) {
      is_sink_op[i] = true;
      for (auto t : prob.ops[i].outputs)
        for (auto cop : dag.tensor_consumers[t])
          if (is_in_sg[cop]) { is_sink_op[i] = false; break; }
    }

    // Seed: boundary output tensors get FROM_NTW × FROM_NTH
    for (auto t : sg.boundary_outputs_)
      tsrc[t] = {TS::FROM_NTW, TS::FROM_NTH, true};

    // Backward propagation through ops in reverse topological order.
    // If a tensor gets conflicting tiling requirements from two consumers,
    // the subgraph is invalid (reference: SHAPES_MISALIGNED).
    std::vector<size_t> sg_topo;
    for (auto op_idx : dag.topological_order())
      if (is_in_sg[op_idx]) sg_topo.push_back(op_idx);

    bool tiling_conflict = false;

    auto assign_or_check = [&](size_t t, TS new_h, TS new_v) {
      if (!tsrc[t].assigned) {
        tsrc[t] = {new_h, new_v, true};
      } else if (tsrc[t].h != new_h || tsrc[t].v != new_v) {
        // Conflict: this tensor is consumed by two ops that need different
        // tile shapes. The reference evaluator rejects this as SHAPES_MISALIGNED.
        tiling_conflict = true;
      }
    };

    for (int ri = (int)sg_topo.size() - 1; ri >= 0 && !tiling_conflict; ri--) {
      size_t op_idx = sg_topo[ri];
      const auto &op = prob.ops[op_idx];
      size_t out = op.outputs[0];

      if (!tsrc[out].assigned)
        tsrc[out] = {TS::FROM_NTW, TS::FROM_NTH, true};

      TS out_h = tsrc[out].h;
      TS out_v = tsrc[out].v;

      if (op.type == OpType::Pointwise) {
        for (auto t : op.inputs)
          assign_or_check(t, out_h, out_v);
      } else {
        size_t lhs = op.inputs[0], rhs = op.inputs[1];

        if (is_sink_op[op_idx]) {
          assign_or_check(lhs, TS::FROM_NK, out_v);
          assign_or_check(rhs, out_h, TS::FROM_NK);
        } else {
          assign_or_check(lhs, TS::FIXED_1, out_v);
          assign_or_check(rhs, out_h, TS::FIXED_1);
        }
      }
    }

    if (tiling_conflict)
      return std::nullopt;

    // Build BoundaryTensorInfo from the propagated sources
    std::vector<int> tensor_in_info(num_tensors, -1);
    auto ensure = [&](size_t t) -> size_t {
      if (tensor_in_info[t] < 0) {
        tensor_in_info[t] = (int)sg.boundary_tensor_info_.size();
        BoundaryTensorInfo info;
        info.id = t;
        info.full_size = prob.tensors[t].width * prob.tensors[t].height;
        if (tsrc[t].assigned) {
          info.h_source = tsrc[t].h;
          info.v_source = tsrc[t].v;
        }
        sg.boundary_tensor_info_.push_back(info);
      }
      return (size_t)tensor_in_info[t];
    };

    // Register all boundary inputs
    for (auto t : sg.boundary_inputs_) {
      size_t idx = ensure(t);
      // Mark internally-produced boundary inputs (produced inside + external consumers)
      if (is_produced[t])
        sg.boundary_tensor_info_[idx].is_internally_produced = true;
    }

    // Register all boundary outputs
    for (auto t : sg.boundary_outputs_) {
      size_t idx = ensure(t);
      sg.boundary_tensor_info_[idx].is_boundary_out = true;
      if (is_produced[t])
        sg.boundary_tensor_info_[idx].is_internally_produced =
            sg.boundary_inputs_.count(t) > 0 ? false : false;
      // Also: if produced internally AND is boundary input too (fan-out),
      // the internally_produced flag was set above
    }

    // Mark MM accumulators (sink MM outputs that are boundary)
    for (auto op_idx : sg.ops_) {
      const auto &op = prob.ops[op_idx];
      if (op.type == OpType::MatMul) {
        size_t out = op.outputs[0];
        if (sg.boundary_outputs_.count(out)) {
          size_t idx = ensure(out);
          sg.boundary_tensor_info_[idx].is_mm_out = true;
        }
      }
    }

    // Mark internally-produced boundary outputs (produced inside, has external consumers)
    for (auto op_idx : sg.ops_) {
      for (auto t : prob.ops[op_idx].outputs) {
        if (sg.boundary_outputs_.count(t) && is_produced[t]) {
          size_t idx = ensure(t);
          // It's internally produced if it's also used as an input to some op
          // in this subgraph (otherwise it's just a pure output)
          bool used_internally = false;
          for (auto cop : dag.tensor_consumers[t])
            if (is_in_sg[cop]) { used_internally = true; break; }
          if (used_internally)
            sg.boundary_tensor_info_[idx].is_internally_produced = true;
        }
      }
    }
  }

  // ---- Populate tensor_id_to_info_ for O(1) lookup in working_set ----
  sg.tensor_id_to_info_.assign(num_tensors, -1);
  for (size_t idx = 0; idx < sg.boundary_tensor_info_.size(); idx++)
    sg.tensor_id_to_info_[sg.boundary_tensor_info_[idx].id] = (int)idx;

  // ---- Build tiling candidates ----
  auto gcd_of = [](const std::vector<int64_t>& vals) -> int64_t {
    if (vals.empty()) return 0;
    int64_t g = vals[0];
    for (size_t i = 1; i < vals.size(); i++)
      g = std::gcd(g, vals[i]);
    return g;
  };

  int64_t w_gcd = gcd_of(sg.w_divides_);
  int64_t h_gcd = gcd_of(sg.h_divides_);
  int64_t k_gcd = gcd_of(sg.k_divides_);

  sg.ws_cand_ = w_gcd > 0 ? all_divisors(w_gcd) : std::vector<int64_t>{1};
  sg.hs_cand_ = h_gcd > 0 ? all_divisors(h_gcd) : std::vector<int64_t>{1};
  sg.ks_cand_ = sg.has_pw_sink_ ? std::vector<int64_t>{1}
              : k_gcd > 0       ? all_divisors(k_gcd)
                                : std::vector<int64_t>{1};

  return sg;
}

// ============================================================================
// Tiling validity
// ============================================================================

bool Subgraph::is_valid_tiling(const TileConfig &cfg) const {
  if (cfg.w <= 0 || cfg.h <= 0 || cfg.k <= 0)
    return false;

  if (has_pw_sink_ && cfg.k > 1)
    return false;

  for (int64_t v : w_divides_)
    if (v % cfg.w != 0) return false;
  for (int64_t v : h_divides_)
    if (v % cfg.h != 0) return false;
  for (int64_t v : k_divides_)
    if (v % cfg.k != 0) return false;

  return true;
}

// ============================================================================
// Working set — physical peak memory across all (tile, d) steps.
//
// All data that's simultaneously resident in fast memory during execution:
//   - Input slices: each boundary input occupies its tile slice (W/h_tiles × H/v_tiles)
//     regardless of whether it was freshly loaded or reused from a previous step.
//     Resident data (FIXED_1, once-loaded) stays in memory across d-steps.
//   - Output accumulator: the boundary output tile (w × h) is always resident
//     during accumulation across d-steps.
//   - Retained tensors: full size, always present.
//
// This is CONSTANT across d-steps: the same data sits in memory at d=0 and d=nk-1.
// Matches PROBLEM.md Example 5: T0(16384) + T1_strip(4096) + T2_strip(4096) +
// T4_accum(16384) = 40960.
//
// Note: the reference code (main.cpp) checks bytes_transferred per step, which
// is more permissive (24576 for Example 5). Our model matches the physical
// description in PROBLEM.md — solutions that pass our check always pass the
// reference evaluator too.
// ============================================================================
int64_t Subgraph::working_set(const TileConfig &cfg,
                              const std::set<size_t> &retained_from_prev,
                              const std::set<size_t> &retain_these) const {
  int64_t ntw = out_W_ / cfg.w;
  int64_t nth = out_H_ / cfg.h;
  int64_t nk = has_matmul_ ? (output_K_ / cfg.k) : 1;

  int64_t ws = 0;

  for (auto &info : boundary_tensor_info_) {
    // Retained tensors from prev step: full size always resident
    if (retained_from_prev.count(info.id)) {
      ws += info.full_size;
      continue;
    }
    // Tensors being retained for next step: handled in post-pass
    if (retain_these.count(info.id)) {
      continue;
    }

    // Tile slice from propagated tiling
    int64_t ht = info.eval_h_tiles(ntw, nk);
    int64_t vt = info.eval_v_tiles(nth, nk);
    int64_t W = prob_->tensors[info.id].width;
    int64_t H = prob_->tensors[info.id].height;
    int64_t slice = (ht > 0 && vt > 0) ? (W / ht) * (H / vt) : info.full_size;
    ws += slice;
  }

  // Retained-from-prev tensors not in boundary_tensor_info_
  for (auto t : retained_from_prev) {
    if (t < tensor_id_to_info_.size() && tensor_id_to_info_[t] >= 0)
      continue;  // already counted above
    ws += prob_->tensors[t].size();
  }

  // Retained-for-next tensors: full size (accumulate all tiles)
  for (auto t : retain_these)
    if (!retained_from_prev.count(t))
      ws += prob_->tensors[t].size();

  return ws;
}

bool Subgraph::is_feasible(const TileConfig &cfg,
                           const std::set<size_t> &retained_from_prev,
                           const std::set<size_t> &retain_these) const {
  return is_valid_tiling(cfg) &&
         working_set(cfg, retained_from_prev, retain_these) <=
             prob_->fast_memory_capacity;
}

// ============================================================================
// Cost computation
// ============================================================================

CostResult Subgraph::compute_cost(const TileConfig &cfg,
                                  const std::set<size_t> &retained_from_prev,
                                  const std::set<size_t> &retain_these) const {
  CostResult result;
  result.config = cfg;

  if (!is_valid_tiling(cfg))
    return result;

  result.working_set = working_set(cfg, retained_from_prev, retain_these);

  if (result.working_set > prob_->fast_memory_capacity)
    return result;
  result.feasible = true;

  double B = (double)prob_->slow_memory_bandwidth;
  int num_tw = (int)(out_W_ / cfg.w);
  int num_th = (int)(out_H_ / cfg.h);
  int num_tiles = num_tw * num_th;
  result.num_spatial_tiles = num_tiles;
  result.num_k_passes = has_matmul_ ? (int)(output_K_ / cfg.k) : 1;
  int nk = result.num_k_passes;

  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  int64_t scale =
      ceil_div(cfg.w, prob_->native_w) * ceil_div(cfg.h, prob_->native_h);

  // Compute: separate MM (per k-step) from PW (once per tile)
  //
  // All MatMul ops in a fused chain share the same temporal fraction per
  // k-step: k / output_K_.  This is because the unified temporal loop is
  // driven by the output-producing MatMul's reduction dimension.  Each
  // k-step, every op in the chain produces k / output_K_ of its output
  // columns — upstream ops do full internal reduction but for fewer columns.
  //
  // Example: chain (T0 @ T1) @ T2, output_K_ = K1 = T3.width.
  //   Per step, Op0 produces k columns of T3 (fraction k/K1).
  //   Per step, Op1 does rank-k update of T4 (fraction k/K1).
  //   Both ops: base_cost × k / output_K_ per step.
  double mm_comp = 0.0;
  double pw_comp = 0.0;
  for (auto i : ops_) {
    double c = (double)prob_->ops[i].base_cost;
    if (prob_->ops[i].type == OpType::MatMul) {
      mm_comp += c * ((double)cfg.k / output_K_);
    } else {
      pw_comp += c;
    }
  }
  mm_comp *= (double)scale;
  pw_comp *= (double)scale;
  result.compute_per_step = mm_comp;

  // Memory transfer costs: 5 categories matching the reference evaluator's
  // position-based reuse model.
  //
  //   once_load   : FIXED_1 × FIXED_1  — loaded once (first tile only).
  //   row_load    : FIXED_1 × FROM_NTH — loaded when output row changes.
  //   col_load    : FROM_NTW × FIXED_1 — loaded when output column changes.
  //   tile_load   : FROM_NTW × FROM_NTH — loaded every tile.
  //   stream_load : FROM_NK in either   — loaded every k-step.
  //   out_evict   : boundary output evicted on last k-step per tile.
  double once_load = 0, row_load = 0, col_load = 0;
  double tile_load = 0, stream_load = 0, out_evict = 0;

  for (auto &info : boundary_tensor_info_) {
    bool retained_in = retained_from_prev.count(info.id);
    bool retained_out = retain_these.count(info.id);

    if (!retained_in && !info.is_internally_produced) {
      int64_t ht = info.eval_h_tiles(num_tw, nk);
      int64_t vt = info.eval_v_tiles(num_th, nk);
      int64_t W = prob_->tensors[info.id].width;
      int64_t H = prob_->tensors[info.id].height;
      double slice_io = (double)((W / std::max(ht, (int64_t)1)) *
                                 (H / std::max(vt, (int64_t)1))) / B;

      bool k_dep = (info.h_source == BoundaryTensorInfo::FROM_NK ||
                    info.v_source == BoundaryTensorInfo::FROM_NK);
      bool h_fixed = (info.h_source == BoundaryTensorInfo::FIXED_1);
      bool v_fixed = (info.v_source == BoundaryTensorInfo::FIXED_1);

      if (k_dep)                    stream_load += slice_io;
      else if (h_fixed && v_fixed)  once_load   += slice_io;
      else if (h_fixed)             row_load    += slice_io;
      else if (v_fixed)             col_load    += slice_io;
      else                          tile_load   += slice_io;
    }

    if (info.is_boundary_out && !retained_out)
      out_evict += (double)(cfg.h * cfg.w) / B;
  }

  // Per-tile cost with 3 freshness flags:
  //   once_fresh: true only for the very first tile
  //   row_fresh:  true when output row changes
  //   col_fresh:  true when output column changes
  //
  // stream_load is always fresh when nk > 1 (d changes every step).
  auto tile_cost = [&](bool once_fresh, bool row_fresh, bool col_fresh) -> double {
    double per_tile_io = tile_load;
    if (once_fresh) per_tile_io += once_load;
    if (row_fresh)  per_tile_io += row_load;
    if (col_fresh)  per_tile_io += col_load;

    if (nk == 1) {
      return std::max(mm_comp + pw_comp, per_tile_io + stream_load + out_evict);
    }

    // Step 0: all per-tile IO + first stream strip
    double step0 = std::max(mm_comp, per_tile_io + stream_load);
    // Middle steps: stream only
    double mid = (nk >= 3) ? (double)(nk - 2) * std::max(mm_comp, stream_load) : 0.0;
    // Last step: stream + evict + PW compute
    double last = std::max(mm_comp + pw_comp, stream_load + out_evict);

    return step0 + mid + last;
  };

  // Traversal-dependent latency.
  //
  // Raster (row-major): row 0 cols 0..ntw-1, row 1 cols 0..ntw-1, ...
  //   First tile of each row: once_fresh (row 0 only), row_fresh, col_fresh
  //   Remaining tiles in row: col_fresh only
  //
  // Snake row-major: row 0 L→R, row 1 R→L, ...
  //   First tile ever: once + row + col
  //   Row transitions: same column → row_fresh only (col reused)
  //   Within-row tiles: col_fresh only
  //
  // Snake col-major: col 0 T→B, col 1 B→T, ...
  //   First tile ever: once + row + col
  //   Column transitions: same row → col_fresh only (row reused)
  //   Within-column tiles: row_fresh only
  if (cfg.snake == SnakeDir::None) {
    if (has_matmul_ && num_tw > 1) {
      // First row, first tile: once + row + col
      double first_tile = tile_cost(true, true, true);
      // First tile of subsequent rows: row + col (once reused)
      double row_start = tile_cost(false, true, true);
      // Remaining tiles in any row: col only
      double within_row = tile_cost(false, false, true);
      result.latency = first_tile +
                       (double)(num_th - 1) * row_start +
                       (double)(num_tw - 1) * num_th * within_row;
    } else {
      result.latency = (double)num_tiles * tile_cost(true, true, true);
    }
  } else if (cfg.snake == SnakeDir::RowMajor) {
    double first = tile_cost(true, true, true);         // first tile
    double row_trans = tile_cost(false, true, false);    // row change, col reused
    double within = tile_cost(false, false, true);       // col change
    int n_row_trans = num_th - 1;
    int n_within = (num_tw - 1) * num_th;
    result.latency = first +
                     (double)n_row_trans * row_trans +
                     (double)n_within * within;
  } else { // ColMajor
    double first = tile_cost(true, true, true);          // first tile
    double col_trans = tile_cost(false, false, true);     // col change, row reused
    double within = tile_cost(false, true, false);        // row change
    int n_col_trans = num_tw - 1;
    int n_within = (num_th - 1) * num_tw;
    result.latency = first +
                     (double)n_col_trans * col_trans +
                     (double)n_within * within;
  }

  return result;
}

// ============================================================================
// Granularity enumeration
// ============================================================================

CostResult Subgraph::best_cost(const std::set<size_t> &retained_from_prev,
                               const std::set<size_t> &retain_these) const {
  int64_t min_w = std::max<int64_t>(1, prob_->native_w / 4);
  int64_t min_h = std::max<int64_t>(1, prob_->native_h / 4);

  // For MatMul: snake always ties or beats None. For PW-only: snake has no effect.
  std::vector<SnakeDir> snakes;
  if (has_matmul_) {
    snakes = {SnakeDir::RowMajor, SnakeDir::ColMajor};
  } else {
    snakes = {SnakeDir::None};
  }

  // FIX #1: call compute_cost directly (it already checks validity + feasibility
  // internally). Avoids redundant working_set computation from is_feasible.
  auto search = [&](const std::vector<int64_t> &ws,
                    const std::vector<int64_t> &hs,
                    const std::vector<int64_t> &ks, int64_t mw, int64_t mh) {
    CostResult best;
    for (int64_t ww : ws) {
      if (ww < mw)
        continue;
      for (int64_t hh : hs) {
        if (hh < mh)
          continue;
        for (int64_t kk : ks) {
          for (auto sd : snakes) {
            TileConfig cfg{ww, hh, kk, sd};
            auto r = compute_cost(cfg, retained_from_prev, retain_these);
            if (r.feasible && r.latency < best.latency)
              best = r;
          }
        }
      }
    }
    return best;
  };

  CostResult best = search(ws_cand_, hs_cand_, ks_cand_, min_w, min_h);
  if (!best.feasible)
    best = search(ws_cand_, hs_cand_, ks_cand_, 1, 1);

  return best;
}