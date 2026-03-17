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

    // Determine sink ops: ops with no internal successor.
    // The reference evaluator seeds tiling from the sink's output tensor.
    // Only sink outputs define the subgraph's spatial dimensions (out_W, out_H).
    // Non-sink ops may produce boundary outputs with different dimensions
    // (e.g., intermediate tensors consumed externally via recomputation).
    std::vector<size_t> sink_ops;
    for (auto i : sg.ops_) {
      bool has_internal_succ = false;
      for (auto t : prob.ops[i].outputs)
        for (auto cop : dag.tensor_consumers[t])
          if (is_in_sg[cop]) { has_internal_succ = true; break; }
      if (!has_internal_succ)
        sink_ops.push_back(i);
    }

    // out_W, out_H from the first sink's output. All sinks must agree.
    if (sink_ops.empty()) return std::nullopt;
    size_t first_sink_out = prob.ops[sink_ops[0]].outputs[0];
    sg.out_W_ = prob.tensors[first_sink_out].width;
    sg.out_H_ = prob.tensors[first_sink_out].height;
    for (size_t si = 1; si < sink_ops.size(); si++) {
      size_t out = prob.ops[sink_ops[si]].outputs[0];
      if (prob.tensors[out].width != sg.out_W_ ||
          prob.tensors[out].height != sg.out_H_)
        return std::nullopt;
      // Also check K dimension agreement for MatMul sinks (ref line 277)
      if (prob.ops[sink_ops[si]].type == OpType::MatMul &&
          prob.ops[sink_ops[0]].type == OpType::MatMul) {
        if (prob.tensors[prob.ops[sink_ops[si]].inputs[0]].width !=
            prob.tensors[prob.ops[sink_ops[0]].inputs[0]].width)
          return std::nullopt;
      }
    }

    // Check for PW sinks
    for (auto s : sink_ops) {
      if (prob.ops[s].type == OpType::Pointwise) {
        sg.has_pw_sink_ = true;
        break;
      }
    }

    // Compute output_K_: the reduction dimension that drives the temporal
    // loop (nk = output_K / k).
    if (!sg.has_pw_sink_) {
      for (auto s : sink_ops) {
        if (prob.ops[s].type == OpType::MatMul) {
          sg.output_K_ = sg.op_K(s);
          break;
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

    // Seed: only SINK ops' output tensors get FROM_NTW × FROM_NTH.
    // Non-sink boundary outputs get their tiling from backward propagation
    // (they're produced by internal ops whose tiling is set by the chain).
    for (auto i : sg.ops_) {
      if (!is_sink_op[i]) continue;
      for (auto t : prob.ops[i].outputs)
        tsrc[t] = {TS::FROM_NTW, TS::FROM_NTH, true};
    }

    // Backward propagation through ops in reverse topological order.
    // If a tensor gets conflicting tiling requirements from two consumers,
    // the subgraph is invalid (reference: SHAPES_MISALIGNED).
    std::vector<size_t> sg_topo;
    for (auto op_idx : dag.topological_order())
      if (is_in_sg[op_idx]) sg_topo.push_back(op_idx);

    bool tiling_conflict = false;

    // Check if two tile sources are compatible.
    // FROM_NK degenerates to FIXED_1 when nk=1. Both mean "full extent"
    // in that dimension. They only differ when nk>1 (split-K), and configs
    // with conflicting nk>1 tiling are rejected at eval time.
    auto compatible = [](TS a, TS b) -> bool {
      if (a == b) return true;
      if ((a == TS::FROM_NK && b == TS::FIXED_1) ||
          (a == TS::FIXED_1 && b == TS::FROM_NK))
        return true;
      return false;
    };

    // When merging FROM_NK with FIXED_1, keep FROM_NK (more general:
    // evaluates to 1 when nk=1, to nk when nk>1).
    auto merge_source = [](TS existing, TS incoming) -> TS {
      if (existing == TS::FROM_NK || incoming == TS::FROM_NK)
        return TS::FROM_NK;
      return existing;
    };

    auto assign_or_check = [&](size_t t, TS new_h, TS new_v) {
      if (!tsrc[t].assigned) {
        tsrc[t] = {new_h, new_v, true};
      } else if (!compatible(tsrc[t].h, new_h) || !compatible(tsrc[t].v, new_v)) {
        tiling_conflict = true;
      } else {
        // Merge compatible sources (prefer FROM_NK over FIXED_1)
        tsrc[t].h = merge_source(tsrc[t].h, new_h);
        tsrc[t].v = merge_source(tsrc[t].v, new_v);
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

    // Save per-tensor tiling for compute_cost scale factors
    sg.tensor_tiling_.resize(num_tensors);
    for (size_t t = 0; t < num_tensors; t++) {
      if (tsrc[t].assigned) {
        sg.tensor_tiling_[t] = {tsrc[t].h, tsrc[t].v};
      }
    }

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
      sg.boundary_tensor_info_[idx].is_internally_produced = is_produced[t];
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

  // Generate valid tiling candidates for each dimension.
  //
  // A candidate c is valid for dimension set S if for all v in S:
  //   if c < v then v % c == 0   (divisibility when tiling)
  //   if c >= v then skip         (granularity exceeds dim → 1 tile)
  //
  // The old GCD-based approach only generated divisors of GCD(S), missing
  // values like 1024 for S={128, 2048} where GCD=128 but 1024 divides 2048
  // and exceeds 128. The reference evaluator uses max(W/w, 1) to allow this.
  //
  // Fix: generate divisors of max(S), then filter to those valid for all of S.
  auto valid_candidates = [](const std::vector<int64_t> &dims) -> std::vector<int64_t> {
    if (dims.empty()) return {1};
    int64_t mx = *std::max_element(dims.begin(), dims.end());
    if (mx <= 0) return {1};
    auto divs = all_divisors(mx);
    std::vector<int64_t> result;
    for (auto c : divs) {
      bool ok = true;
      for (auto v : dims) {
        if (c < v && v % c != 0) { ok = false; break; }
      }
      if (ok) result.push_back(c);
    }
    return result;
  };

  sg.ws_cand_ = valid_candidates(sg.w_divides_);
  sg.hs_cand_ = valid_candidates(sg.h_divides_);
  sg.ks_cand_ = sg.has_pw_sink_ ? std::vector<int64_t>{1}
                                 : valid_candidates(sg.k_divides_);

  return sg;
}

// ============================================================================
// Tiling validity
// ============================================================================

bool Subgraph::is_valid_tiling(const TileConfig &cfg) const {
  if (cfg.w <= 0 || cfg.h <= 0 || cfg.k <= 0)
    return false;

  // When sink is PW, k is irrelevant (d_tiles=1 regardless).
  // The reference doesn't validate k at all for PW sinks.
  for (int64_t v : w_divides_)
    if (cfg.w < v && v % cfg.w != 0) return false;
  for (int64_t v : h_divides_)
    if (cfg.h < v && v % cfg.h != 0) return false;
  if (!has_pw_sink_) {
    for (int64_t v : k_divides_)
      if (cfg.k < v && v % cfg.k != 0) return false;
  }

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
  if (!is_valid_tiling(cfg))
    return INT64_MAX;

  int64_t ntw = std::max(out_W_ / cfg.w, (int64_t)1);
  int64_t nth = std::max(out_H_ / cfg.h, (int64_t)1);
  int64_t nk = has_matmul_ ? std::max(output_K_ / cfg.k, (int64_t)1) : 1;

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
  int num_tw = std::max((int)(out_W_ / cfg.w), 1);
  int num_th = std::max((int)(out_H_ / cfg.h), 1);
  int num_tiles = num_tw * num_th;
  result.num_spatial_tiles = num_tiles;
  result.num_k_passes = has_matmul_ ? std::max((int)(output_K_ / cfg.k), 1) : 1;
  int nk = result.num_k_passes;

  // Compute: per-op scale factor based on output tile size vs granularity.
  //
  // Reference evaluator:
  //   For EVERY op (MM and PW), at EVERY d-step:
  //     compute_time += base_cost / d_tiles
  //                   × max(output_slice_w / gran_w, 1.0)
  //                   × max(output_slice_h / gran_h, 1.0)
  //
  // There is NO distinction between MM and PW in compute timing.
  // Both contribute equally to every d-step.
  //
  // comp_per_step = sum over all ops of (base_cost / nk × op_scale)
  // Total compute per tile = nk × comp_per_step
  double comp_per_step = 0.0;
  for (auto i : ops_) {
    double c = (double)prob_->ops[i].base_cost;
    size_t out_t = prob_->ops[i].outputs[0];

    // Per-op scale: output tile slice relative to granularity
    double op_scale = 1.0;
    if (out_t < tensor_tiling_.size()) {
      auto &tp = tensor_tiling_[out_t];
      BoundaryTensorInfo tmpinfo;
      tmpinfo.h_source = tp.h;
      tmpinfo.v_source = tp.v;
      int64_t ht = tmpinfo.eval_h_tiles(num_tw, nk);
      int64_t vt = tmpinfo.eval_v_tiles(num_th, nk);
      double slice_w = (double)prob_->tensors[out_t].width / std::max(ht, (int64_t)1);
      double slice_h = (double)prob_->tensors[out_t].height / std::max(vt, (int64_t)1);
      op_scale = std::max(slice_w / (double)cfg.w, 1.0) *
                 std::max(slice_h / (double)cfg.h, 1.0);
    }

    comp_per_step += c / (double)nk * op_scale;
  }
  result.compute_per_step = comp_per_step;

  // Memory transfer costs: 5 categories matching the reference evaluator's
  // position-based reuse model.
  //
  //   once_load   : FIXED_1 × FIXED_1  — loaded once (first tile only).
  //   row_load    : FROM_NTW × FIXED_1 — position depends on row → loaded when row changes.
  //   col_load    : FIXED_1 × FROM_NTH — position depends on column → loaded when column changes.
  //   tile_load   : FROM_NTW × FROM_NTH — loaded every tile.
  //   stream_load : FROM_NK in either   — loaded every k-step.
  //   out_evict   : boundary output evicted on last k-step per tile.
  //
  //   Reference semantics: h_pos = tile_idx // ntw = ROW, v_pos = tile_idx % ntw = COL.
  //   FIXED_1 in h → h_pos constant → doesn't depend on row → depends on column (v_pos) → col_load.
  //   FIXED_1 in v → v_pos constant → doesn't depend on column → depends on row (h_pos) → row_load.
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

      // For categorization: when nk=1, FROM_NK evaluates to 1 = FIXED_1.
      // The reference uses the non-sink path (h_tiles=1) when d_tiles=1,
      // so FROM_NK degenerates and the tensor gets reused like FIXED_1.
      auto eff_h = info.h_source;
      auto eff_v = info.v_source;
      if (nk <= 1) {
        if (eff_h == BoundaryTensorInfo::FROM_NK) eff_h = BoundaryTensorInfo::FIXED_1;
        if (eff_v == BoundaryTensorInfo::FROM_NK) eff_v = BoundaryTensorInfo::FIXED_1;
      }

      bool k_dep = (eff_h == BoundaryTensorInfo::FROM_NK ||
                    eff_v == BoundaryTensorInfo::FROM_NK);
      bool h_fixed = (eff_h == BoundaryTensorInfo::FIXED_1);
      bool v_fixed = (eff_v == BoundaryTensorInfo::FIXED_1);

      if (k_dep)                    stream_load += slice_io;
      else if (h_fixed && v_fixed)  once_load   += slice_io;
      else if (h_fixed)             col_load    += slice_io;
      else if (v_fixed)             row_load    += slice_io;
      else                          tile_load   += slice_io;
    }

    if (info.is_boundary_out && !retained_out) {
      int64_t ht = info.eval_h_tiles(num_tw, nk);
      int64_t vt = info.eval_v_tiles(num_th, nk);
      int64_t W = prob_->tensors[info.id].width;
      int64_t H = prob_->tensors[info.id].height;
      out_evict += (double)((W / std::max(ht, (int64_t)1)) *
                            (H / std::max(vt, (int64_t)1))) / B;
    }
  }

  // Per-tile cost with 3 freshness flags:
  //   once_fresh: true only for the very first tile
  //   row_fresh:  true when output row changes
  //   col_fresh:  true when output column changes
  //
  // compute_time = comp_per_step at EVERY d-step (reference: no MM/PW distinction).
  // io varies by step: d=0 loads fresh data, d=1..nk-2 only stream, d=nk-1 adds eviction.
  auto tile_cost = [&](bool once_fresh, bool row_fresh, bool col_fresh) -> double {
    double per_tile_io = tile_load;
    if (once_fresh) per_tile_io += once_load;
    if (row_fresh)  per_tile_io += row_load;
    if (col_fresh)  per_tile_io += col_load;

    if (nk == 1) {
      return std::max(comp_per_step, per_tile_io + stream_load + out_evict);
    }

    // d=0: all per-tile IO + first stream strip
    double step0 = std::max(comp_per_step, per_tile_io + stream_load);
    // d=1..nk-2: stream only (nk-2 copies)
    double mid = (nk >= 3) ? (double)(nk - 2) * std::max(comp_per_step, stream_load) : 0.0;
    // d=nk-1: stream + evict
    double last = std::max(comp_per_step, stream_load + out_evict);

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