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

  // Classify ephemerals
  std::vector<bool> is_ephemeral(num_tensors, false);
  for (size_t t = 0; t < num_tensors; t++) {
    if (is_consumed[t] && !is_produced[t])
      sg.boundary_inputs_.insert(t);
    if (is_produced[t] && is_consumed[t]) {
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

  if (sg.boundary_outputs_.empty())
    return std::nullopt;

  enum class SliceW : uint8_t { W_param, K_param };
  enum class SliceH : uint8_t { H_param, K_param };
  std::vector<SliceW> eph_sw(num_tensors, SliceW::W_param);
  std::vector<SliceH> eph_sh(num_tensors, SliceH::H_param);

  {
    sg.is_sink_op_vec_.assign(num_ops, false);
    std::vector<size_t> sink_ops;
    for (auto i : sg.ops_) {
      bool has_internal_succ = false;
      for (auto t : prob.ops[i].outputs)
        for (auto cop : dag.tensor_consumers[t])
          if (is_in_sg[cop]) { has_internal_succ = true; break; }
      
      if (!has_internal_succ) {
        sink_ops.push_back(i);
        sg.is_sink_op_vec_[i] = true;
      }
    }

    if (sink_ops.empty()) return std::nullopt;
    size_t first_sink_out = prob.ops[sink_ops[0]].outputs[0];
    sg.out_W_ = prob.tensors[first_sink_out].width;
    sg.out_H_ = prob.tensors[first_sink_out].height;
    
    for (size_t si = 1; si < sink_ops.size(); si++) {
      size_t out = prob.ops[sink_ops[si]].outputs[0];
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

    if (!sg.has_pw_sink_) {
      for (auto s : sink_ops) {
        if (prob.ops[s].type == OpType::MatMul) {
          sg.output_K_ = sg.op_K(s);
          break;
        }
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
          for (auto t : prob.ops[op_idx].outputs)
            if (is_ephemeral[t])
              eph_order.push_back(t);

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
            size_t pw_out = op.outputs[0];
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
        size_t out = op.outputs[0];
        if (sg.boundary_outputs_.count(out)) {
          add_constraint(out, SliceW::W_param, SliceH::H_param);
        } else if (is_ephemeral[out]) {
          for (auto &r : eph_roles[out]) add_constraint(out, r.sw, r.sh);
        }
      } else {
        size_t out = op.outputs[0];
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
      for (auto t : prob.ops[i].outputs)
        tsrc[t] = {TS::FROM_NTW, TS::FROM_NTH, true};
    }

    auto merge_source = [](TS existing, TS incoming) -> TS {
      if (existing == TS::FROM_NK || incoming == TS::FROM_NK) return TS::FROM_NK;
      return existing;
    };

    auto assign_or_check = [&](size_t t, TS new_h, TS new_v) {
      if (!tsrc[t].assigned) {
        tsrc[t] = {new_h, new_v, true};
      } else {
        // Symbolic propagation ONLY for IO tracking. Numerical conflicts are caught dynamically later.
        tsrc[t].h = merge_source(tsrc[t].h, new_h);
        tsrc[t].v = merge_source(tsrc[t].v, new_v);
      }
    };

    for (auto op_idx : sg.reverse_topo_ops_) {
      const auto &op = prob.ops[op_idx];
      size_t out = op.outputs[0];

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
        size_t out = op.outputs[0];
        if (sg.boundary_outputs_.count(out)) {
          size_t idx = ensure(out);
          sg.boundary_tensor_info_[idx].is_mm_out = true;
        }
      }
    }

    for (auto op_idx : sg.ops_) {
      for (auto t : prob.ops[op_idx].outputs) {
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
// Tiling validity (Replicates `evaluator.cpp` SHAPES_MISALIGNED EXACTLY)
// ============================================================================

bool Subgraph::is_valid_tiling(const TileConfig &cfg) const {
  if (cfg.w <= 0 || cfg.h <= 0 || cfg.k <= 0)
    return false;

  for (int64_t v : w_divides_)
    if (cfg.w < v && v % cfg.w != 0) return false;
  for (int64_t v : h_divides_)
    if (cfg.h < v && v % cfg.h != 0) return false;
  if (!has_pw_sink_) {
    for (int64_t v : k_divides_)
      if (cfg.k < v && v % cfg.k != 0) return false;
  }

  int64_t ntw = std::max(out_W_ / cfg.w, (int64_t)1);
  int64_t nth = std::max(out_H_ / cfg.h, (int64_t)1);
  int64_t nk = has_matmul_ ? std::max(output_K_ / cfg.k, (int64_t)1) : 1;

  std::vector<int64_t> h_tiles(prob_->num_tensors(), -1);
  std::vector<int64_t> v_tiles(prob_->num_tensors(), -1);

  // Seed outputs of sink operators
  for (auto op_idx : ops_) {
    if (is_sink_op_vec_[op_idx]) {
      size_t out = prob_->ops[op_idx].outputs[0];
      h_tiles[out] = ntw;
      v_tiles[out] = nth;
    }
  }

  // Reverse propagation using actual numerical values
  for (auto op_idx : reverse_topo_ops_) {
    const auto &op = prob_->ops[op_idx];
    size_t out = op.outputs[0];
    int64_t out_h = h_tiles[out];
    int64_t out_v = v_tiles[out];

    if (out_h == -1) continue; // Unreachable tensor, skip

    if (op.type == OpType::Pointwise) {
      for (auto in : op.inputs) {
        if (h_tiles[in] != -1 && (h_tiles[in] != out_h || v_tiles[in] != out_v)) 
          return false;
        h_tiles[in] = out_h;
        v_tiles[in] = out_v;
      }
    } else { // MatMul
      bool is_sink = is_sink_op_vec_[op_idx];
      
      int64_t lhs_h = (is_sink && nk > 1) ? nk : 1;
      int64_t lhs_v = out_v;
      int64_t rhs_h = out_h;
      int64_t rhs_v = (is_sink && nk > 1) ? nk : 1;

      size_t lhs = op.inputs[0], rhs = op.inputs[1];

      if (h_tiles[lhs] != -1 && (h_tiles[lhs] != lhs_h || v_tiles[lhs] != lhs_v)) 
        return false;
      h_tiles[lhs] = lhs_h;
      v_tiles[lhs] = lhs_v;

      if (h_tiles[rhs] != -1 && (h_tiles[rhs] != rhs_h || v_tiles[rhs] != rhs_v)) 
        return false;
      h_tiles[rhs] = rhs_h;
      v_tiles[rhs] = rhs_v;
    }
  }

  return true;
}

// ============================================================================
// Working set
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
    int64_t slice = (ht > 0 && vt > 0) ? (W / ht) * (H / vt) : info.full_size;
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

  double comp_per_step = 0.0;
  for (auto i : ops_) {
    double c = (double)prob_->ops[i].base_cost;
    size_t out_t = prob_->ops[i].outputs[0];

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
      out_evict += (double)((W / std::max(ht, (int64_t)1)) *
                            (H / std::max(vt, (int64_t)1))) / B;
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
      double first_tile = tile_cost(true, true, true);
      double row_start = tile_cost(false, true, true);
      double within_row = tile_cost(false, false, true);
      result.latency = first_tile +
                       (double)(num_th - 1) * row_start +
                       (double)(num_tw - 1) * num_th * within_row;
    } else {
      result.latency = (double)num_tiles * tile_cost(true, true, true);
    }
  } else if (cfg.snake == SnakeDir::RowMajor) {
    double first = tile_cost(true, true, true);
    double row_trans = tile_cost(false, false, true);
    double within = tile_cost(false, true, false);
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

CostResult Subgraph::best_cost(const std::set<size_t> &retained_from_prev,
                               const std::set<size_t> &retain_these) const {
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
