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
                                         std::vector<size_t> op_indices) {
  if (op_indices.empty())
    return std::nullopt;

  Subgraph sg;
  sg.prob_ = &prob;
  sg.dag_ = &dag;
  sg.ops_ = std::move(op_indices);

  const size_t num_tensors = prob.num_tensors();
  const size_t num_ops = prob.num_ops();

  std::vector<bool> is_in_sg(num_ops, false);
  std::vector<bool> is_produced(num_tensors, false);
  std::vector<bool> is_consumed(num_tensors, false);

  for (auto i : sg.ops_) {
    is_in_sg[i] = true;
    { size_t t = prob.ops[i].output();
      is_produced[t] = true; }
    for (auto t : prob.ops[i].inputs)
      is_consumed[t] = true;
    if (prob.ops[i].type == OpType::MatMul) {
      sg.has_matmul_ = true;
      int64_t Ki = prob.tensors[prob.ops[i].inputs[0]].width;
      sg.max_K_ = std::max(sg.max_K_, Ki);
    }
  }

  // Classify ephemerals
  //
  // Rule: a tensor produced AND consumed inside this subgraph is ephemeral.
  // It passes directly between ops without consuming fast memory or slow
  // memory bandwidth. Zero memory footprint, zero IO cost.
  //
  // Whether external consumers can access this tensor is NOT the subgraph's
  // concern — that's validated at the partition/solution level by
  // partition_has_gap() and Solution::validate().
  std::vector<bool> is_ephemeral(num_tensors, false);
  for (size_t t = 0; t < num_tensors; t++) {
    if (is_consumed[t] && !is_produced[t])
      sg.boundary_inputs_.insert(t);
    if (is_produced[t] && is_consumed[t])
      is_ephemeral[t] = true;
  }
  for (size_t t = 0; t < num_tensors; t++) {
    if (is_produced[t] && !is_ephemeral[t])
      sg.boundary_outputs_.insert(t);
    if (is_ephemeral[t])
      sg.ephemeral_.insert(t);
  }

  // Partition ephemerals by producer op-type for the granule-fit check.
  // PW-produced ephemerals have a stricter bound (cfg) since PW has no
  // k-loop; MM-produced ephemerals just need to fit native.
  for (auto i : sg.ops_) {
    size_t out_t = prob.ops[i].output();
    if (!is_ephemeral[out_t]) continue;
    if (prob.ops[i].type == OpType::Pointwise)
      sg.pw_produced_ephemerals_.push_back(out_t);
    else
      sg.mm_produced_ephemerals_.push_back(out_t);
  }

  // Issue #71 Rules 2/3 (prologue-PW geometric condition):
  //   PW feeds a downstream MM's LHS (directly or through PW chain) →
  //     require cfg.w ≥ matmul.K
  //   PW feeds a downstream MM's RHS → require cfg.h ≥ matmul.K
  // For each PW op in the subgraph, forward-BFS through PW-only chains to
  // any MM it reaches; note the role (LHS/RHS) and record the MM's K.
  // Subgraph-level thresholds = max K across all such pairs.
  for (auto i : sg.ops_) {
    if (prob.ops[i].type != OpType::Pointwise) continue;
    // BFS forward through subgraph, crossing only PW→PW edges. A PW that
    // reaches an MM via PW-only chain is a "prologue" of that MM.
    std::vector<size_t> stack = {i};
    std::vector<bool> visited(num_ops, false);
    visited[i] = true;
    while (!stack.empty()) {
      size_t op = stack.back(); stack.pop_back();
      size_t out_t = prob.ops[op].output();
      for (auto cop : dag.tensor_consumers[out_t]) {
        if (!is_in_sg[cop] || visited[cop]) continue;
        visited[cop] = true;
        const auto &cop_op = prob.ops[cop];
        if (cop_op.type == OpType::MatMul) {
          int64_t K = prob.tensors[cop_op.inputs[0]].width;
          // Determine LHS vs RHS by which input slot the propagated tensor
          // occupies. `out_t` here is the *immediate* downstream tensor we
          // arrived at via the BFS edge; for PW-chain intermediates this
          // still lands on the MM's LHS/RHS slot correctly since PW
          // preserves shape and chain identity.
          if (cop_op.inputs[0] == out_t)
            sg.prologue_cfg_w_min_ = std::max(sg.prologue_cfg_w_min_, K);
          if (cop_op.inputs.size() > 1 && cop_op.inputs[1] == out_t)
            sg.prologue_cfg_h_min_ = std::max(sg.prologue_cfg_h_min_, K);
          // Stop — MM materializes its output, so downstream-of-MM is a
          // separate concern (that MM's epilogue, not this PW's prologue).
        } else if (cop_op.type == OpType::Pointwise) {
          stack.push_back(cop);
        }
      }
    }
  }

  if (sg.boundary_outputs_.empty())
    return std::nullopt;

  enum class SliceW : uint8_t { W_param, K_param };
  enum class SliceH : uint8_t { H_param, K_param };

  {
    sg.is_sink_op_vec_.assign(num_ops, false);
    std::vector<size_t> sink_ops;
    for (auto i : sg.ops_) {
      bool has_internal_succ = false;
      { size_t t = prob.ops[i].output();
        for (auto cop : dag.tensor_consumers[t])
          if (is_in_sg[cop]) { has_internal_succ = true; break; } }
      
      if (!has_internal_succ) {
        sink_ops.push_back(i);
        sg.is_sink_op_vec_[i] = true;
      }
    }

    if (sink_ops.empty()) return std::nullopt;
    size_t first_sink_out = prob.ops[sink_ops[0]].output();
    sg.out_W_ = prob.tensors[first_sink_out].width;
    sg.out_H_ = prob.tensors[first_sink_out].height;
    
    for (size_t si = 1; si < sink_ops.size(); si++) {
      size_t out = prob.ops[sink_ops[si]].output();
      if (prob.tensors[out].width != sg.out_W_ ||
          prob.tensors[out].height != sg.out_H_)
        return std::nullopt;
      if (prob.ops[sink_ops[si]].type == OpType::MatMul &&
          prob.ops[sink_ops[0]].type == OpType::MatMul) {
        if (prob.tensors[prob.ops[sink_ops[si]].inputs[0]].width !=
            prob.tensors[prob.ops[sink_ops[0]].inputs[0]].width)
          return std::nullopt;
      }
    }

    for (auto s : sink_ops) {
      if (prob.ops[s].type == OpType::Pointwise) {
        sg.has_pw_sink_ = true;
        break;
      }
    }

    // Set output_K_ from the sink matmul (if any).
    //   MM-only sinks:  output_K_ = op_K(sink_mm) — standard temporal tiling.
    //   Mixed MM+PW sinks: output_K_ = op_K(sink_mm) — the MM sink determines
    //     K; has_pw_sink_ enforcement below ensures nk == 1.
    //   PW-only sinks:  output_K_ stays 1 — no temporal dimension.
    for (auto s : sink_ops) {
      if (prob.ops[s].type == OpType::MatMul) {
        sg.output_K_ = sg.op_K(s);
        break;
      }
    }

    std::vector<std::vector<size_t>> eph_consumers(num_tensors);
    for (auto i : sg.ops_)
      for (auto t : prob.ops[i].inputs)
        if (is_ephemeral[t])
          eph_consumers[t].push_back(i);

    struct RolePair { SliceW sw; SliceH sh; };
    std::vector<std::vector<RolePair>> eph_roles(num_tensors);
    std::vector<bool> eph_roles_computed(num_tensors, false);

    {
      std::vector<size_t> eph_order;
      for (auto op_idx : dag.topological_order())
        if (is_in_sg[op_idx])
          { size_t t = prob.ops[op_idx].output();
            if (is_ephemeral[t])
              eph_order.push_back(t); }

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
            size_t pw_out = op.output();
            if (sg.boundary_outputs_.count(pw_out)) {
              eph_roles[t].push_back({SliceW::W_param, SliceH::H_param});
            } else if (is_ephemeral[pw_out] && eph_roles_computed[pw_out]) {
              for (auto &r : eph_roles[pw_out]) eph_roles[t].push_back(r);
            } else {
              eph_roles[t].push_back({SliceW::W_param, SliceH::H_param});
            }
          }
        }
        eph_roles_computed[t] = true;
      }
    }

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
        size_t out = op.output();
        if (sg.boundary_outputs_.count(out)) {
          add_constraint(out, SliceW::W_param, SliceH::H_param);
        } else if (is_ephemeral[out]) {
          for (auto &r : eph_roles[out]) add_constraint(out, r.sw, r.sh);
        }
      } else {
        size_t out = op.output();
        if (sg.boundary_outputs_.count(out)) {
          add_constraint(out, SliceW::W_param, SliceH::H_param);
          for (auto t : op.inputs) add_constraint(t, SliceW::W_param, SliceH::H_param);
        } else if (is_ephemeral[out]) {
          for (auto &r : eph_roles[out]) {
            add_constraint(out, r.sw, r.sh);
            for (auto t : op.inputs) add_constraint(t, r.sw, r.sh);
          }
          if (eph_roles[out].empty()) {
            add_constraint(out, SliceW::W_param, SliceH::H_param);
            for (auto t : op.inputs) add_constraint(t, SliceW::W_param, SliceH::H_param);
          }
        }
      }
    }

    sg.w_divides_.assign(w_set.begin(), w_set.end());
    sg.h_divides_.assign(h_set.begin(), h_set.end());
    sg.k_divides_.assign(k_set.begin(), k_set.end());
  }

  // Precompute reverse topo ops using DAG order
  const auto& topo = dag.topological_order();
  for (int ri = (int)topo.size() - 1; ri >= 0; ri--) {
    if (is_in_sg[topo[ri]]) {
      sg.reverse_topo_ops_.push_back(topo[ri]);
    }
  }

  {
    using TS = BoundaryTensorInfo::TileSource;
    struct TilePair { TS h; TS v; bool assigned = false; };
    std::vector<TilePair> tsrc(num_tensors);

    // Per-tensor set of distinct role signatures. Powers the multi-entry
    // boundary_tensor_info_ materialization per issues #70 + #73:
    //   FULL signature (FIXED_1, FIXED_1) dominates — collapse to one full
    //     entry, drop partials (#70).
    //   Multiple distinct partial signatures → one entry each, working set
    //     sums them (#73). Capped at 2 partials per the row/col simplification.
    struct RoleSig { TS h; TS v; };
    std::vector<std::vector<RoleSig>> roles_per_tensor(num_tensors);
    auto push_role = [&](size_t t, TS h, TS v) {
      for (auto &r : roles_per_tensor[t])
        if (r.h == h && r.v == v) return;  // dedup identical signatures
      roles_per_tensor[t].push_back({h, v});
    };

    for (auto i : sg.ops_) {
      if (!sg.is_sink_op_vec_[i]) continue;
      { size_t t = prob.ops[i].output();
        tsrc[t] = {TS::FROM_NTW, TS::FROM_NTH, true};
        push_role(t, TS::FROM_NTW, TS::FROM_NTH); }
    }

    auto merge_source = [](TS existing, TS incoming) -> TS {
      if (existing == TS::FROM_NK || incoming == TS::FROM_NK) return TS::FROM_NK;
      return existing;
    };

    auto assign_or_check = [&](size_t t, TS new_h, TS new_v) {
      push_role(t, new_h, new_v);
      if (!tsrc[t].assigned) {
        tsrc[t] = {new_h, new_v, true};
      } else {
        // Merge for the tsrc-based tensor_tiling_ (used by op_scale in
        // compute_cost). Multi-role is tracked separately in
        // roles_per_tensor and materialized as distinct entries in
        // boundary_tensor_info_ per #70 + #73.
        tsrc[t].h = merge_source(tsrc[t].h, new_h);
        tsrc[t].v = merge_source(tsrc[t].v, new_v);
      }
    };

    for (auto op_idx : sg.reverse_topo_ops_) {
      const auto &op = prob.ops[op_idx];
      size_t out = op.output();

      if (!tsrc[out].assigned) tsrc[out] = {TS::FROM_NTW, TS::FROM_NTH, true};

      TS out_h = tsrc[out].h;
      TS out_v = tsrc[out].v;

      if (op.type == OpType::Pointwise) {
        for (auto t : op.inputs) assign_or_check(t, out_h, out_v);
      } else {
        size_t lhs = op.inputs[0], rhs = op.inputs[1];
        if (sg.is_sink_op_vec_[op_idx]) {
          assign_or_check(lhs, TS::FROM_NK, out_v);
          assign_or_check(rhs, out_h, TS::FROM_NK);
        } else {
          assign_or_check(lhs, TS::FIXED_1, out_v);
          assign_or_check(rhs, out_h, TS::FIXED_1);
        }
      }
    }

    sg.tensor_tiling_.resize(num_tensors);
    for (size_t t = 0; t < num_tensors; t++) {
      if (tsrc[t].assigned) {
        sg.tensor_tiling_[t] = {tsrc[t].h, tsrc[t].v};
      }
    }

    // Tensor-dim bounds (formerly min_*_dim_) are now checked per-entry in
    // is_valid_tiling rather than precomputed from the merged tsrc role.
    // Necessary for correctness under multi-role: each entry has its own
    // (h_source, v_source) signature, and each imposes its own bound.

    // Materialize boundary_tensor_info_ entries, one per distinct retained
    // role signature per tensor. Rules:
    //   #70 collapse: if any signature is FULL (FIXED_1, FIXED_1), it
    //     subsumes all partials — keep only the full entry. Full move covers
    //     any partial access.
    //   2-partial limit: if no full and >2 distinct partial signatures
    //     remain, reject the subgraph. Keeps the data structure bounded.
    std::vector<std::vector<int>> tensor_in_info(num_tensors);
    bool too_many_partials = false;
    auto ensure = [&](size_t t) -> const std::vector<int>& {
      if (!tensor_in_info[t].empty()) return tensor_in_info[t];

      // Fallback if no consumer pushed a role (shouldn't happen for tensors
      // we actually process, but be safe).
      if (roles_per_tensor[t].empty() && tsrc[t].assigned)
        push_role(t, tsrc[t].h, tsrc[t].v);

      auto &roles = roles_per_tensor[t];

      // #70 collapse: full role subsumes all others.
      bool has_full = false;
      for (auto &r : roles)
        if (r.h == TS::FIXED_1 && r.v == TS::FIXED_1) { has_full = true; break; }

      std::vector<RoleSig> retained;
      if (has_full) {
        retained.push_back({TS::FIXED_1, TS::FIXED_1});
      } else {
        retained = roles;
        if (retained.size() > 2) too_many_partials = true;
      }

      for (auto &r : retained) {
        BoundaryTensorInfo info;
        info.id = t;
        info.full_size = prob.tensors[t].width * prob.tensors[t].height;
        info.h_source = r.h;
        info.v_source = r.v;
        tensor_in_info[t].push_back((int)sg.boundary_tensor_info_.size());
        sg.boundary_tensor_info_.push_back(info);
      }
      return tensor_in_info[t];
    };

    for (auto t : sg.boundary_inputs_) {
      const auto &indices = ensure(t);
      if (is_produced[t])
        for (int idx : indices)
          sg.boundary_tensor_info_[idx].is_internally_produced = true;
    }

    for (auto t : sg.boundary_outputs_) {
      const auto &indices = ensure(t);
      // Eviction is per-tensor, priced at the producer's output tile size.
      // Flag exactly one entry to avoid N× eviction.
      if (!indices.empty())
        sg.boundary_tensor_info_[indices[0]].is_boundary_out = true;
      for (int idx : indices)
        sg.boundary_tensor_info_[idx].is_internally_produced = is_produced[t];
    }

    for (auto op_idx : sg.ops_) {
      const auto &op = prob.ops[op_idx];
      if (op.type == OpType::MatMul) {
        size_t out = op.output();
        if (sg.boundary_outputs_.count(out)) {
          const auto &indices = ensure(out);
          if (!indices.empty())
            sg.boundary_tensor_info_[indices[0]].is_mm_out = true;
        }
      }
    }

    for (auto op_idx : sg.ops_) {
      {
        size_t t = prob.ops[op_idx].output();
        if (sg.boundary_outputs_.count(t) && is_produced[t]) {
          const auto &indices = ensure(t);
          bool used_internally = false;
          for (auto cop : dag.tensor_consumers[t])
            if (is_in_sg[cop]) { used_internally = true; break; }
          if (used_internally)
            for (int idx : indices)
              sg.boundary_tensor_info_[idx].is_internally_produced = true;
        }
      }
    }

    if (too_many_partials) return std::nullopt;
  }

  sg.tensor_id_to_infos_.assign(num_tensors, std::vector<int>{});
  for (size_t idx = 0; idx < sg.boundary_tensor_info_.size(); idx++)
    sg.tensor_id_to_infos_[sg.boundary_tensor_info_[idx].id].push_back((int)idx);

  // Super-native granule forbidden on all three axes per issues #74 Q1, #78
  // Q3, #80 Q3, #81 Q1. Per #80 Q1 native is a single value across w/h/k;
  // we use native_w as the uniform cap (benchmarks always have native_w ==
  // native_h, and native_k follows the same value).
  const int64_t native_cap = prob.native_w;
  auto valid_candidates = [native_cap](const std::vector<int64_t> &dims)
      -> std::vector<int64_t> {
    if (dims.empty()) return {1};
    int64_t mx = *std::max_element(dims.begin(), dims.end());
    if (mx <= 0) return {1};
    auto divs = all_divisors(mx);
    std::vector<int64_t> result;
    for (auto c : divs) {
      if (native_cap > 0 && c > native_cap) continue;  // super-native invalid
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
  // PW-sink subgraphs: force k = output_K_ so nk == 1 (no temporal tiling).
  //   PW-only sinks:  output_K_ == 1 → k == 1 in solution.
  //   Mixed MM+PW sinks: output_K_ == op_K(mm) → k == K in solution
  //     (full reduction in one pass).
  sg.ks_cand_ = sg.has_pw_sink_ ? std::vector<int64_t>{sg.output_K_}
                                 : valid_candidates(sg.k_divides_);

  return sg;
}

// ============================================================================
// Tiling validity (Replicates `evaluator.cpp` SHAPES_MISALIGNED EXACTLY)
// ============================================================================

bool Subgraph::is_valid_tiling(const TileConfig &cfg) const {
  if (cfg.w <= 0 || cfg.h <= 0 || cfg.k <= 0)
    return false;

  // Super-native granule forbidden (issues #74 Q1, #78 Q3, #80 Q3, #81 Q1).
  // native_w == native_h == native_k per #80 Q1; use native_w as the cap.
  const int64_t native_cap = prob_->native_w;
  if (native_cap > 0) {
    if (cfg.w > native_cap) return false;
    if (cfg.h > native_cap) return false;
    if (cfg.k > native_cap) return false;
  }

  for (int64_t v : w_divides_)
    if (cfg.w < v && v % cfg.w != 0) return false;
  for (int64_t v : h_divides_)
    if (cfg.h < v && v % cfg.h != 0) return false;
  if (!has_pw_sink_) {
    for (int64_t v : k_divides_) {
      // Per #75: cfg.k > op_K is physically undefined — there's nothing
      // to stream into the back half of the k-granule, and whatever gets
      // summed in would corrupt the accumulator.
      if (cfg.k > v) return false;
      if (cfg.k < v && v % cfg.k != 0) return false;
    }
  }
  // For PW-sink subgraphs k is irrelevant (nk is always 1): skip k
  // divisibility but enforce nk == 1 explicitly below.

  // Derived tile-count bounds: reject if ntw/nth/nk would exceed any tensor's
  // dimension in the corresponding direction. Without this, slice computation
  // produces zero-size slices (integer division W/h_tiles = 0 when h_tiles > W).
  int64_t ntw = std::max(out_W_ / cfg.w, (int64_t)1);
  int64_t nth = std::max(out_H_ / cfg.h, (int64_t)1);
  int64_t nk = has_matmul_ ? std::max(output_K_ / cfg.k, (int64_t)1) : 1;

  // PW-sink: no temporal tiling allowed.
  if (has_pw_sink_ && nk > 1) return false;

  // Issue #71 Rules 2/3: prologue-PW geometric condition. A PW that feeds
  // an MM's LHS (via PW-only chain) requires cfg.w ≥ matmul.K so a single
  // PW tile spans the full LHS K-axis; feeding RHS requires cfg.h ≥ matmul.K.
  // Applies at all nk — the constraint is geometric (PW tile shape vs
  // K-axis), not conditional on split-K. No prologue PW → thresholds are 0
  // and these checks are no-ops.
  if (cfg.w < prologue_cfg_w_min_) return false;
  if (cfg.h < prologue_cfg_h_min_) return false;

  // Per-entry tensor-dim bounds (per #70 + #73 multi-role). For each
  // distinct role signature of each boundary tensor, ensure the derived
  // tile count doesn't exceed the tensor's dim in that direction (otherwise
  // slice < 1). Replaces the old min_*_dim_ bounds which were computed from
  // the single merged role — incorrect when a tensor's roles disagree on
  // which axis is tiled how.
  // Divisibility isn't enforced here; the existing w_divides_ / h_divides_ /
  // k_divides_ checks cover the cfg-granule divisibility, and non-integer
  // slices are tolerated by compute_cost's floating-point accounting.
  for (const auto &info : boundary_tensor_info_) {
    int64_t ht = info.eval_h_tiles(ntw, nk);
    int64_t vt = info.eval_v_tiles(nth, nk);
    int64_t W = prob_->tensors[info.id].width;
    int64_t H = prob_->tensors[info.id].height;
    if (ht > W || vt > H) return false;
  }
  // Ephemerals aren't in boundary_tensor_info_; use tensor_tiling_ (single
  // merged role — ephemerals aren't materialized per-role since they're
  // zero-cost per #77 and don't affect WS/IO accounting).
  for (size_t t : ephemeral_) {
    const auto &tp = tensor_tiling_[t];
    BoundaryTensorInfo tmp;
    tmp.h_source = tp.h;
    tmp.v_source = tp.v;
    int64_t ht = tmp.eval_h_tiles(ntw, nk);
    int64_t vt = tmp.eval_v_tiles(nth, nk);
    int64_t W = prob_->tensors[t].width;
    int64_t H = prob_->tensors[t].height;
    if (ht > W || vt > H) return false;
  }

  // Granule-fit check on ephemerals. Every op in the subgraph runs at the
  // subgraph's (cfg.w, cfg.h) granule; the slice its producer writes per
  // execution must be representable within that granule:
  //   PW producer: slice ≤ (cfg.w, cfg.h) — PW has no k-loop, must produce
  //                the whole slice in one granule execution.
  //   MM producer: slice ≤ native — MM's internal k-loop allows slices that
  //                exceed cfg as long as hardware-native-sized.
  // Mostly redundant with cfg ≤ native + role propagation, kept as a
  // defensive check against role-propagation surprises in long op chains.
  auto slice_for = [&](size_t t, int64_t &slice_w, int64_t &slice_h) {
    const auto &tp = tensor_tiling_[t];
    BoundaryTensorInfo tmp;
    tmp.h_source = tp.h;
    tmp.v_source = tp.v;
    int64_t ht = tmp.eval_h_tiles(ntw, nk);
    int64_t vt = tmp.eval_v_tiles(nth, nk);
    int64_t W = prob_->tensors[t].width;
    int64_t H = prob_->tensors[t].height;
    slice_w = W / std::max(ht, (int64_t)1);
    slice_h = H / std::max(vt, (int64_t)1);
  };
  for (size_t t : pw_produced_ephemerals_) {
    int64_t sw, sh;
    slice_for(t, sw, sh);
    if (sw > cfg.w || sh > cfg.h) return false;
  }
  for (size_t t : mm_produced_ephemerals_) {
    int64_t sw, sh;
    slice_for(t, sw, sh);
    if (native_cap > 0 && (sw > native_cap || sh > native_cap)) return false;
  }

  // Multi-role tensors are now modeled explicitly via multi-entry
  // boundary_tensor_info_ (per #70 + #73); divisibility checks above cover
  // shape constraints across all role orientations, and the 2-partial limit
  // is enforced at Subgraph::create. No symbolic-propagation conflict check
  // is needed — the former slow path rejected exactly the multi-role configs
  // #73 now accepts.
  return true;
}

// ============================================================================
// Working set
// ============================================================================

// Internal: compute working set assuming cfg is already validated.
int64_t Subgraph::working_set_unchecked(const TileConfig &cfg,
                              const FlatSet<size_t> &retained_from_prev,
                              const FlatSet<size_t> &retain_these) const {
  int64_t ntw = std::max(out_W_ / cfg.w, (int64_t)1);
  int64_t nth = std::max(out_H_ / cfg.h, (int64_t)1);
  int64_t nk = has_matmul_ ? std::max(output_K_ / cfg.k, (int64_t)1) : 1;

  int64_t ws = 0;

  for (auto &info : boundary_tensor_info_) {
    if (retained_from_prev.count(info.id)) {
      ws += info.full_size;
      continue;
    }
    if (retain_these.count(info.id)) {
      continue;
    }

    int64_t ht = info.eval_h_tiles(ntw, nk);
    int64_t vt = info.eval_v_tiles(nth, nk);
    int64_t W = prob_->tensors[info.id].width;
    int64_t H = prob_->tensors[info.id].height;
    // Clamp: tile count cannot exceed tensor dimension (safety net;
    // is_valid_tiling should already reject such configs).
    ht = std::max(std::min(ht, W), (int64_t)1);
    vt = std::max(std::min(vt, H), (int64_t)1);
    int64_t slice = (W / ht) * (H / vt);
    ws += slice;
  }

  for (auto t : retained_from_prev) {
    if (t < tensor_id_to_infos_.size() && !tensor_id_to_infos_[t].empty())
      continue;
    ws += prob_->tensors[t].size();
  }

  for (auto t : retain_these)
    if (!retained_from_prev.count(t))
      ws += prob_->tensors[t].size();

  return ws;
}

int64_t Subgraph::working_set(const TileConfig &cfg,
                              const FlatSet<size_t> &retained_from_prev,
                              const FlatSet<size_t> &retain_these) const {
  if (!is_valid_tiling(cfg))
    return INT64_MAX;
  return working_set_unchecked(cfg, retained_from_prev, retain_these);
}

bool Subgraph::is_feasible(const TileConfig &cfg,
                           const FlatSet<size_t> &retained_from_prev,
                           const FlatSet<size_t> &retain_these) const {
  return is_valid_tiling(cfg) &&
         working_set_unchecked(cfg, retained_from_prev, retain_these) <=
             prob_->fast_memory_capacity;
}

// ============================================================================
// Cost computation
// ============================================================================

CostResult Subgraph::compute_cost(const TileConfig &cfg,
                                  const FlatSet<size_t> &retained_from_prev,
                                  const FlatSet<size_t> &retain_these) const {
  CostResult result;
  result.config = cfg;

  result.working_set = working_set(cfg, retained_from_prev, retain_these);

  if (result.working_set > prob_->fast_memory_capacity)
    return result;
  result.feasible = true;

  const double inv_B = 1.0 / (double)prob_->slow_memory_bandwidth;
  int num_tw = std::max((int)(out_W_ / cfg.w), 1);
  int num_th = std::max((int)(out_H_ / cfg.h), 1);
  int num_tiles = num_tw * num_th;
  result.num_spatial_tiles = num_tiles;
  result.num_k_passes = has_matmul_ ? std::max((int)(output_K_ / cfg.k), 1) : 1;
  const int nk = result.num_k_passes;

  double comp_per_step = 0.0;
  double native_w = (double)prob_->native_w;
  double native_h = (double)prob_->native_h;

  for (auto i : ops_) {
    double c = (double)prob_->ops[i].base_cost;
    size_t out_t = prob_->ops[i].output();

    double op_scale = 1.0;
    if (out_t < tensor_tiling_.size()) {
      auto &tp = tensor_tiling_[out_t];
      BoundaryTensorInfo tmpinfo;
      tmpinfo.h_source = tp.h;
      tmpinfo.v_source = tp.v;
      int64_t ht = tmpinfo.eval_h_tiles(num_tw, nk);
      int64_t vt = tmpinfo.eval_v_tiles(num_th, nk);
      int64_t tW = prob_->tensors[out_t].width;
      int64_t tH = prob_->tensors[out_t].height;
      ht = std::max(std::min(ht, tW), (int64_t)1);
      vt = std::max(std::min(vt, tH), (int64_t)1);
      double slice_w = (double)tW / ht;
      double slice_h = (double)tH / vt;
      op_scale = std::max(slice_w / native_w, 1.0) *
                 std::max(slice_h / native_h, 1.0);
    }
    // Per PROBLEM.md Example 5: when the unified execution grid has k<K,
    // every op's compute is amortized across the nk k-steps (the whole
    // subgraph co-pipelines partial k-slices — no op materializes its
    // output before the sink). Divide by nk uniformly.
    comp_per_step += c / (double)nk * op_scale;
  }
  result.compute_per_step = comp_per_step;

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
      ht = std::max(std::min(ht, W), (int64_t)1);
      vt = std::max(std::min(vt, H), (int64_t)1);
      double slice_io = (double)((W / ht) * (H / vt)) * inv_B;

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
      else if (h_fixed)             row_load    += slice_io; // FIX: Fixed horizontally (LHS). Paid on row change.
      else if (v_fixed)             col_load    += slice_io; // FIX: Fixed vertically (RHS). Paid on col change.
      else                          tile_load   += slice_io;
    }

    if (info.is_boundary_out && !retained_out) {
      int64_t ht = info.eval_h_tiles(num_tw, nk);
      int64_t vt = info.eval_v_tiles(num_th, nk);
      int64_t W = prob_->tensors[info.id].width;
      int64_t H = prob_->tensors[info.id].height;
      ht = std::max(std::min(ht, W), (int64_t)1);
      vt = std::max(std::min(vt, H), (int64_t)1);
      out_evict += (double)((W / ht) * (H / vt)) * inv_B;
    }
  }

  auto tile_cost = [&](bool once_fresh, bool row_fresh, bool col_fresh) -> double {
    double per_tile_io = tile_load;
    if (once_fresh) per_tile_io += once_load;
    if (row_fresh)  per_tile_io += row_load;
    if (col_fresh)  per_tile_io += col_load;

    if (nk == 1) {
      return std::max(comp_per_step, per_tile_io + stream_load + out_evict);
    }

    double step0 = std::max(comp_per_step, per_tile_io + stream_load);
    double mid = (nk >= 3) ? (double)(nk - 2) * std::max(comp_per_step, stream_load) : 0.0;
    double last = std::max(comp_per_step, stream_load + out_evict);

    return step0 + mid + last;
  };

  if (cfg.snake == SnakeDir::None) {
    if (has_matmul_ && num_tw > 1) {
      // Row-major scan, no snake. Rows go left-to-right, then reset.
      // Within row: h changes → RHS (col_load) reloads; LHS stays.
      // Row start: both h resets and v changes → both reload.
      double first_tile = tile_cost(true, true, true);
      double row_start = tile_cost(false, true, true);
      double within_row = tile_cost(false, false, true);
      result.latency = first_tile +
                       (double)(num_th - 1) * row_start +
                       (double)(num_tw - 1) * num_th * within_row;
    } else if (has_matmul_ && num_th > 1) {
      // Single column of tiles (num_tw=1): only v changes between tiles.
      // LHS (row_load) reloads every tile; RHS (col_load/once_load) loads once.
      double first_tile = tile_cost(true, true, true);
      double rest = tile_cost(false, true, false);
      result.latency = first_tile + (double)(num_th - 1) * rest;
    } else {
      result.latency = (double)num_tiles * tile_cost(true, true, true);
    }
  } else if (cfg.snake == SnakeDir::RowMajor) {
    // hsnake: sweep rows, alternate direction each row.
    // Within row: h changes → RHS (col_load) reloads.
    // Row transition: v changes, h stays (snake) → LHS (row_load) reloads.
    double first = tile_cost(true, true, true);
    double row_trans = tile_cost(false, true, false);
    double within = tile_cost(false, false, true);
    int n_row_trans = num_th - 1;
    int n_within = (num_tw - 1) * num_th;
    result.latency = first +
                     (double)n_row_trans * row_trans +
                     (double)n_within * within;
  } else { // ColMajor
    double first = tile_cost(true, true, true);
    double col_trans = tile_cost(false, false, true);
    double within = tile_cost(false, true, false);
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

CostResult Subgraph::best_cost(const FlatSet<size_t> &retained_from_prev,
                               const FlatSet<size_t> &retain_these) const {
  std::vector<SnakeDir> snakes;
  if (has_matmul_) {
    snakes = {SnakeDir::RowMajor, SnakeDir::ColMajor};
  } else {
    snakes = {SnakeDir::None};
  }

  CostResult best;

  for (int64_t ww : ws_cand_) {
    for (int64_t hh : hs_cand_) {
      for (int64_t kk : ks_cand_) {
        // is_valid_tiling depends on (w,h,k) only, not snake direction.
        // Check once before iterating snakes to avoid redundant work inside
        // compute_cost → working_set → is_valid_tiling.
        TileConfig base_cfg{ww, hh, kk, SnakeDir::None};
        if (!is_valid_tiling(base_cfg)) continue;
        // Working set depends on (w,h,k) only, not snake direction.
        // Check once here to skip all snake variants for infeasible tiles.
        int64_t ws = working_set_unchecked(base_cfg, retained_from_prev, retain_these);
        if (ws > prob_->fast_memory_capacity) continue;
        for (auto sd : snakes) {
          TileConfig cfg{ww, hh, kk, sd};
          auto r = compute_cost(cfg, retained_from_prev, retain_these);
          if (r.feasible && r.latency < best.latency) {
            best = r;
          }
        }
      }
    }
  }

  return best;
}