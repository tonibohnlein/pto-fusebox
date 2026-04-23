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

    for (auto i : sg.ops_) {
      if (!sg.is_sink_op_vec_[i]) continue;
      { size_t t = prob.ops[i].output();
        tsrc[t] = {TS::FROM_NTW, TS::FROM_NTH, true}; }
    }

    auto merge_source = [](TS existing, TS incoming) -> TS {
      if (existing == TS::FROM_NK || incoming == TS::FROM_NK) return TS::FROM_NK;
      return existing;
    };

    bool has_tiling_conflict = false;

    auto assign_or_check = [&](size_t t, TS new_h, TS new_v) {
      if (!tsrc[t].assigned) {
        tsrc[t] = {new_h, new_v, true};
      } else {
        // Detect if this tensor gets conflicting roles from different consumers.
        // A conflict means the numerical propagation in is_valid_tiling could
        // fail for some (ntw, nth, nk) values — we can't skip it.
        if (tsrc[t].h != new_h || tsrc[t].v != new_v)
          has_tiling_conflict = true;
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

    sg.has_tiling_conflict_ = has_tiling_conflict;

    sg.tensor_tiling_.resize(num_tensors);
    for (size_t t = 0; t < num_tensors; t++) {
      if (tsrc[t].assigned) {
        sg.tensor_tiling_[t] = {tsrc[t].h, tsrc[t].v};
      }
    }

    // Compute minimum tensor dimensions per tile-count source.
    // This covers ALL tensors in the subgraph (boundary + ephemeral),
    // so is_valid_tiling can reject configs where derived tile counts
    // (ntw, nth, nk) would exceed any tensor's dimension.
    for (auto op_idx : sg.ops_) {
      auto check = [&](size_t t) {
        if (!tsrc[t].assigned) return;
        int64_t W = prob.tensors[t].width;
        int64_t H = prob.tensors[t].height;
        if (tsrc[t].h == TS::FROM_NTW) sg.min_ntw_dim_ = std::min(sg.min_ntw_dim_, W);
        if (tsrc[t].h == TS::FROM_NK)  sg.min_nk_h_dim_ = std::min(sg.min_nk_h_dim_, W);
        if (tsrc[t].v == TS::FROM_NTH) sg.min_nth_dim_ = std::min(sg.min_nth_dim_, H);
        if (tsrc[t].v == TS::FROM_NK)  sg.min_nk_v_dim_ = std::min(sg.min_nk_v_dim_, H);
      };
      for (auto t : prob.ops[op_idx].inputs) check(t);
      { size_t t = prob.ops[op_idx].output(); check(t); }
    }

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

    for (auto t : sg.boundary_inputs_) {
      size_t idx = ensure(t);
      if (is_produced[t]) sg.boundary_tensor_info_[idx].is_internally_produced = true;
    }

    for (auto t : sg.boundary_outputs_) {
      size_t idx = ensure(t);
      sg.boundary_tensor_info_[idx].is_boundary_out = true;
      sg.boundary_tensor_info_[idx].is_internally_produced = is_produced[t];
    }

    for (auto op_idx : sg.ops_) {
      const auto &op = prob.ops[op_idx];
      if (op.type == OpType::MatMul) {
        size_t out = op.output();
        if (sg.boundary_outputs_.count(out)) {
          size_t idx = ensure(out);
          sg.boundary_tensor_info_[idx].is_mm_out = true;
        }
      }
    }

    for (auto op_idx : sg.ops_) {
      {
        size_t t = prob.ops[op_idx].output();
        if (sg.boundary_outputs_.count(t) && is_produced[t]) {
          size_t idx = ensure(t);
          bool used_internally = false;
          for (auto cop : dag.tensor_consumers[t])
            if (is_in_sg[cop]) { used_internally = true; break; }
          if (used_internally)
            sg.boundary_tensor_info_[idx].is_internally_produced = true;
        }
      }
    }
  }

  sg.tensor_id_to_info_.assign(num_tensors, -1);
  for (size_t idx = 0; idx < sg.boundary_tensor_info_.size(); idx++)
    sg.tensor_id_to_info_[sg.boundary_tensor_info_[idx].id] = (int)idx;

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
    for (int64_t v : k_divides_)
      if (cfg.k < v && v % cfg.k != 0) return false;
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

  if (ntw > min_ntw_dim_ || nth > min_nth_dim_) return false;
  if (nk > min_nk_h_dim_ || nk > min_nk_v_dim_) return false;

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

  // Fast path: if no tensor has conflicting roles from multiple consumers,
  // divisibility checks are sufficient — numerical propagation always agrees.
  if (!has_tiling_conflict_)
    return true;

  // Slow path: symbolic propagation to catch multi-role conflicts.
  //
  // The old numerical approach compared tile counts (e.g. h_tiles=ntw vs
  // h_tiles=nk).  When ntw==nk the numbers coincide, but the tiling
  // dimensions are semantically different: ntw varies across spatial columns,
  // nk varies across temporal k-steps.  A tensor can't serve both roles.
  //
  // Fix: propagate TileSource labels.  Two different non-FIXED labels on the
  // same axis → reject.  Both evaluating to 1 (no actual tiling) is the only
  // exception — with one tile there's no positional ambiguity.

  using TS = BoundaryTensorInfo::TileSource;

  thread_local std::vector<TS> h_src, v_src;
  thread_local std::vector<bool> assigned;
  size_t nt = prob_->num_tensors();
  h_src.resize(nt);
  v_src.resize(nt);
  assigned.assign(nt, false);

  auto eval_ts = [&](TS s) -> int64_t {
    switch (s) {
      case TS::FIXED_1:  return 1;
      case TS::FROM_NTW: return ntw;
      case TS::FROM_NTH: return nth;
      case TS::FROM_NK:  return nk;
    }
    return 1;
  };

  // Symbolic compatibility: same label → OK.  Both evaluate to 1 → OK
  // (no tiling, no positional ambiguity).  Otherwise reject.
  auto compat = [&](TS existing, TS incoming) -> bool {
    if (existing == incoming) return true;
    return eval_ts(existing) == 1 && eval_ts(incoming) == 1;
  };

  // Merge: prefer a non-FIXED label (more informative for future checks).
  auto merge_ts = [](TS a, TS b) -> TS {
    return (a != TS::FIXED_1) ? a : b;
  };

  for (auto op_idx : ops_) {
    if (is_sink_op_vec_[op_idx]) {
      size_t out = prob_->ops[op_idx].output();
      h_src[out] = TS::FROM_NTW;
      v_src[out] = TS::FROM_NTH;
      assigned[out] = true;
    }
  }

  for (auto op_idx : reverse_topo_ops_) {
    const auto &op = prob_->ops[op_idx];
    size_t out = op.output();

    if (!assigned[out]) {
      h_src[out] = TS::FROM_NTW;
      v_src[out] = TS::FROM_NTH;
      assigned[out] = true;
    }

    TS out_h = h_src[out];
    TS out_v = v_src[out];

    auto assign_or_reject = [&](size_t t, TS new_h, TS new_v) -> bool {
      if (assigned[t]) {
        if (!compat(h_src[t], new_h) || !compat(v_src[t], new_v))
          return false;
        h_src[t] = merge_ts(h_src[t], new_h);
        v_src[t] = merge_ts(v_src[t], new_v);
      } else {
        h_src[t] = new_h;
        v_src[t] = new_v;
        assigned[t] = true;
      }
      return true;
    };

    if (op.type == OpType::Pointwise) {
      for (auto in : op.inputs)
        if (!assign_or_reject(in, out_h, out_v)) return false;
    } else {
      bool is_sink = is_sink_op_vec_[op_idx];

      TS lhs_h = is_sink ? TS::FROM_NK  : TS::FIXED_1;
      TS lhs_v = out_v;
      TS rhs_h = out_h;
      TS rhs_v = is_sink ? TS::FROM_NK  : TS::FIXED_1;

      if (!assign_or_reject(op.inputs[0], lhs_h, lhs_v)) return false;
      if (!assign_or_reject(op.inputs[1], rhs_h, rhs_v)) return false;
    }
  }

  // Verify propagated tile counts don't exceed tensor dimensions.
  for (size_t t = 0; t < nt; t++) {
    if (!assigned[t]) continue;
    int64_t ht = eval_ts(h_src[t]);
    int64_t vt = eval_ts(v_src[t]);
    int64_t W = prob_->tensors[t].width;
    int64_t H = prob_->tensors[t].height;
    if (ht > W || vt > H) return false;
    if (W % ht != 0 || H % vt != 0) return false;
  }

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
    if (t < tensor_id_to_info_.size() && tensor_id_to_info_[t] >= 0)
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
    const double nk_adjusted = is_sink_op_vec_[i] ? (double)nk : 1.0;
    comp_per_step += c / nk_adjusted * op_scale;
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