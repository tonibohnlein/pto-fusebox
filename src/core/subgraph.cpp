#include "subgraph.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>

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

  std::set<size_t> op_set(sg.ops_.begin(), sg.ops_.end());
  std::set<size_t> produced, consumed;

  for (auto i : sg.ops_) {
    for (auto t : prob.ops[i].outputs)
      produced.insert(t);
    for (auto t : prob.ops[i].inputs)
      consumed.insert(t);
    if (prob.ops[i].type == OpType::MatMul) {
      sg.has_matmul_ = true;
      int64_t Ki = prob.tensors[prob.ops[i].inputs[0]].width;
      sg.max_K_ = std::max(sg.max_K_, Ki);
    }
  }

  // Boundary inputs: consumed but not produced inside
  for (auto t : consumed)
    if (!produced.count(t))
      sg.boundary_inputs_.insert(t);

  // Ephemeral: produced AND consumed inside
  for (auto t : produced)
    if (consumed.count(t))
      sg.ephemeral_.insert(t);

  // Boundary outputs: produced and not ephemeral
  for (auto t : produced)
    if (!sg.ephemeral_.count(t))
      sg.boundary_outputs_.insert(t);

  // Validate: ephemeral tensors must have exactly one consumer within
  // the subgraph. True ephemeral data is produced and immediately consumed
  // by the next op in the chain — it never sits in fast memory. If two ops
  // both need it, the tensor must be materialized (boundary, not ephemeral).
  {
    // Count consumers of each ephemeral tensor within the op set
    std::map<size_t, int> eph_consumer_count;
    for (auto t : sg.ephemeral_)
      eph_consumer_count[t] = 0;
    for (auto i : sg.ops_)
      for (auto t : prob.ops[i].inputs)
        if (sg.ephemeral_.count(t))
          eph_consumer_count[t]++;
    for (auto& [t, cnt] : eph_consumer_count)
      if (cnt != 1)
        return std::nullopt;
  }

  // Must have at least one boundary output
  if (sg.boundary_outputs_.empty())
    return std::nullopt;

  // Detect if any boundary output (sink) is produced by a PW op.
  // If so, k must be 1 (PW runs once per tile; with k>1 the interaction
  // between MM accumulation and PW execution is ambiguous).
  {
    // Map: tensor -> producing op index
    std::map<size_t, size_t> tensor_producer_in_sg;
    for (auto i : sg.ops_)
      for (auto t : prob.ops[i].outputs)
        tensor_producer_in_sg[t] = i;

    for (auto t : sg.boundary_outputs_) {
      auto it = tensor_producer_in_sg.find(t);
      if (it != tensor_producer_in_sg.end() &&
          prob.ops[it->second].type == OpType::Pointwise) {
        sg.has_pw_sink_ = true;
        break;
      }
    }
  }

  // All boundary outputs must have the same dimensions
  {
    auto it = sg.boundary_outputs_.begin();
    sg.out_W_ = prob.tensors[*it].width;
    sg.out_H_ = prob.tensors[*it].height;
    for (++it; it != sg.boundary_outputs_.end(); ++it) {
      if (prob.tensors[*it].width != sg.out_W_ ||
          prob.tensors[*it].height != sg.out_H_)
        return std::nullopt;
    }
  }

  // Validate: ops form a connected group. Two ops are connected if they:
  //   (a) have a producer-consumer edge (DAG pred/succ), OR
  //   (b) share a common input tensor (co-consumers)
  if (sg.ops_.size() > 1) {
    // Build co-consumer adjacency within the op set
    std::map<size_t, std::vector<size_t>> tensor_to_ops; // tensor -> ops in set that consume it
    for (auto i : sg.ops_)
      for (auto t : prob.ops[i].inputs)
        tensor_to_ops[t].push_back(i);

    std::set<size_t> visited;
    std::vector<size_t> bfs = {sg.ops_[0]};
    visited.insert(sg.ops_[0]);

    while (!bfs.empty()) {
      size_t u = bfs.back();
      bfs.pop_back();
      // DAG edges (producer-consumer)
      for (auto v : dag.op_preds[u]) {
        if (op_set.count(v) && !visited.count(v)) {
          visited.insert(v);
          bfs.push_back(v);
        }
      }
      for (auto v : dag.op_succs[u]) {
        if (op_set.count(v) && !visited.count(v)) {
          visited.insert(v);
          bfs.push_back(v);
        }
      }
      // Shared-input edges (co-consumers of same tensor)
      for (auto t : prob.ops[u].inputs) {
        for (auto v : tensor_to_ops[t]) {
          if (v != u && !visited.count(v)) {
            visited.insert(v);
            bfs.push_back(v);
          }
        }
      }
    }
    if (visited.size() != op_set.size())
      return std::nullopt;
  }

  // ---- Collect per-role tiling constraints ----
  // For each tensor, the slice shape depends on how it's used:
  //   MatMul output:  w × h  → w | W, h | H
  //   MatMul LHS:     k × h  → k | W, h | H
  //   MatMul RHS:     w × k  → w | W, k | H
  //   PW boundary:    w × h  → w | W, h | H (inputs and output)
  //   PW ephemeral output consumed as MatMul LHS: k × h → k | W, h | H
  //   PW ephemeral output consumed as MatMul RHS: w × k → w | W, k | H
  //   PW ephemeral consumed by PW: propagate from downstream consumer

  // Slice type: which tiling parameters constrain width and height
  enum class SliceW { W_param, K_param }; // width constrained by w or k
  enum class SliceH { H_param, K_param }; // height constrained by h or k

  // For each ephemeral tensor, determine slice type from its consumer
  std::map<size_t, std::pair<SliceW, SliceH>> eph_slice_type;

  // Build map: ephemeral tensor -> consumer op
  std::map<size_t, size_t> eph_consumer;
  for (auto i : sg.ops_)
    for (auto t : prob.ops[i].inputs)
      if (sg.ephemeral_.count(t))
        eph_consumer[t] = i;

  // Determine slice type for each ephemeral tensor.
  // Process in reverse topo order within the subgraph so that downstream
  // PW→PW chains propagate correctly.
  std::function<std::pair<SliceW, SliceH>(size_t)> get_eph_slice;
  get_eph_slice = [&](size_t tensor_id) -> std::pair<SliceW, SliceH> {
    auto it = eph_slice_type.find(tensor_id);
    if (it != eph_slice_type.end()) return it->second;

    size_t consumer_op = eph_consumer[tensor_id];
    const auto& cop = prob.ops[consumer_op];

    std::pair<SliceW, SliceH> result;
    if (cop.type == OpType::MatMul) {
      if (cop.inputs[0] == tensor_id)
        result = {SliceW::K_param, SliceH::H_param}; // LHS: k × h
      else
        result = {SliceW::W_param, SliceH::K_param}; // RHS: w × k
    } else {
      // PW: inputs have same slice type as output
      size_t pw_out = cop.outputs[0];
      if (sg.boundary_outputs_.count(pw_out))
        result = {SliceW::W_param, SliceH::H_param}; // boundary: w × h
      else
        result = get_eph_slice(pw_out); // propagate from PW output
    }

    eph_slice_type[tensor_id] = result;
    return result;
  };

  for (auto t : sg.ephemeral_)
    get_eph_slice(t);

  // Now collect constraints
  std::set<int64_t> w_set, h_set, k_set;

  // Helper: add constraints for a tensor given its slice type
  auto add_constraint = [&](size_t t, SliceW sw, SliceH sh) {
    int64_t W = prob.tensors[t].width;
    int64_t H = prob.tensors[t].height;
    if (sw == SliceW::W_param) w_set.insert(W); else k_set.insert(W);
    if (sh == SliceH::H_param) h_set.insert(H); else k_set.insert(H);
  };

  for (auto i : sg.ops_) {
    const auto& op = prob.ops[i];

    if (op.type == OpType::MatMul) {
      // LHS: always k × h
      add_constraint(op.inputs[0], SliceW::K_param, SliceH::H_param);
      // RHS: always w × k
      add_constraint(op.inputs[1], SliceW::W_param, SliceH::K_param);
      // Output: w × h (if boundary) or determined by consumer (if ephemeral)
      size_t out = op.outputs[0];
      if (sg.boundary_outputs_.count(out))
        add_constraint(out, SliceW::W_param, SliceH::H_param);
      else
        add_constraint(out, eph_slice_type[out].first, eph_slice_type[out].second);
    } else {
      // PW: determine output slice type, then apply to all inputs and output
      size_t out = op.outputs[0];
      SliceW sw; SliceH sh;
      if (sg.boundary_outputs_.count(out)) {
        sw = SliceW::W_param; sh = SliceH::H_param;
      } else {
        auto st = eph_slice_type[out];
        sw = st.first; sh = st.second;
      }
      add_constraint(out, sw, sh);
      for (auto t : op.inputs)
        add_constraint(t, sw, sh);
    }
  }

  sg.w_divides_.assign(w_set.begin(), w_set.end());
  sg.h_divides_.assign(h_set.begin(), h_set.end());
  sg.k_divides_.assign(k_set.begin(), k_set.end());

  // ---- Precompute per-boundary-tensor role info ----
  {
    std::map<size_t, BoundaryTensorInfo> info_map;
    auto ensure = [&](size_t t) -> BoundaryTensorInfo & {
      auto it = info_map.find(t);
      if (it == info_map.end()) {
        info_map[t] = {t};
        return info_map[t];
      }
      return it->second;
    };

    for (auto i : sg.ops_) {
      const auto &op = prob.ops[i];
      if (op.type == OpType::MatMul) {
        size_t lhs = op.inputs[0], rhs = op.inputs[1], out = op.outputs[0];
        if (sg.boundary_inputs_.count(lhs)) {
          auto &info = ensure(lhs);
          info.max_lhs_K = std::max(info.max_lhs_K, prob.tensors[lhs].width);
        }
        if (sg.boundary_inputs_.count(rhs))
          ensure(rhs).is_mm_rhs = true;
        if (sg.boundary_outputs_.count(out))
          ensure(out).is_mm_out = true;
      } else {
        for (auto t : op.inputs)
          if (sg.boundary_inputs_.count(t))
            ensure(t).is_pw_in = true;
      }
      for (auto t : op.outputs)
        if (sg.boundary_outputs_.count(t))
          ensure(t).is_boundary_out = true;
    }

    sg.boundary_tensor_info_.reserve(info_map.size());
    for (auto &[id, info] : info_map)
      sg.boundary_tensor_info_.push_back(info);
  }

  // ---- Build tiling candidates ----
  // Candidates for each parameter: divisors of gcd(all constraint values).
  // w must divide every value in w_divides_, so w divides gcd(w_divides_).
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
  // If any boundary output is produced by a PW op, k must be 1.
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

  // If any boundary output is a PW output, k must be 1
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
// Working set
//
// Peak fast memory usage during any single tile-step.
//
// For NON-retained boundary inputs: the hardware loads a tile-sized slice
//   (h×K for MatMul LHS, k×w for MatMul RHS, h×w for Pointwise).
//
// For retained_from_prev tensors: the FULL tensor is already resident.
//
// For boundary outputs: the current tile (h×w) is in fast memory.
//
// For retain_these tensors (outputs being retained): tiles accumulate
//   without eviction. Peak additional cost is (full_size - h×w).
// ============================================================================

int64_t Subgraph::working_set(const TileConfig &cfg,
                              const std::set<size_t> &retained_from_prev,
                              const std::set<size_t> &retain_these) const {
  int64_t ws = 0;

  for (auto &info : boundary_tensor_info_) {
    int64_t full =
        prob_->tensors[info.id].width * prob_->tensors[info.id].height;

    if (retained_from_prev.count(info.id)) {
      ws += full;
      continue;
    }

    // Take the maximum working set contribution across all roles
    int64_t max_size = 0;
    if (info.max_lhs_K > 0)
      max_size = std::max(max_size, cfg.h * info.max_lhs_K);
    if (info.is_mm_rhs)
      max_size = std::max(max_size, cfg.k * cfg.w);
    if (info.is_pw_in)
      max_size = std::max(max_size, cfg.h * cfg.w);
    // MatMul boundary outputs are accumulators that persist during reduction.
    // PW boundary outputs do NOT count — they reuse input buffer memory.
    if (info.is_mm_out)
      max_size = std::max(max_size, cfg.h * cfg.w);

    ws += max_size;
  }

  // Retained-from-prev tensors NOT in boundary_tensor_info_ still occupy memory
  for (auto t : retained_from_prev) {
    bool found = false;
    for (auto &info : boundary_tensor_info_)
      if (info.id == t) {
        found = true;
        break;
      }
    if (!found)
      ws += prob_->tensors[t].width * prob_->tensors[t].height;
  }

  // Retained output accumulates: at the last tile, the full tensor is resident.
  for (auto t : retain_these) {
    int64_t full = prob_->tensors[t].width * prob_->tensors[t].height;
    int64_t tile = cfg.h * cfg.w;
    if (full <= tile)
      continue; // single tile, no accumulation overhead

    // The current tile is already counted above (h×w for boundary_out).
    // Add the rest (previously-computed tiles sitting in memory).
    bool tile_counted = false;
    for (auto &info : boundary_tensor_info_) {
      if (info.id == t) {
        int64_t sz = 0;
        if (info.max_lhs_K > 0)
          sz = std::max(sz, cfg.h * info.max_lhs_K);
        if (info.is_mm_rhs)
          sz = std::max(sz, cfg.k * cfg.w);
        if (info.is_pw_in)
          sz = std::max(sz, cfg.h * cfg.w);
        if (info.is_mm_out)
          sz = std::max(sz, cfg.h * cfg.w);
        if (retained_from_prev.count(t))
          sz = full;
        tile_counted = (sz > 0);
        break;
      }
    }

    if (tile_counted)
      ws += full - tile;
    else
      ws += full;
  }

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
  result.num_k_passes = has_matmul_ ? (int)(max_K_ / cfg.k) : 1;
  int nk = result.num_k_passes;

  // Compute scale: native padding factor
  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  int64_t scale =
      ceil_div(cfg.w, prob_->native_w) * ceil_div(cfg.h, prob_->native_h);

  // Separate MatMul compute (per k-step) from PW compute (once per tile).
  double mm_comp = 0.0;
  double pw_comp = 0.0;
  for (auto i : ops_) {
    double c = (double)prob_->ops[i].base_cost;
    if (prob_->ops[i].type == OpType::MatMul) {
      int64_t Ki = op_K(i);
      mm_comp += c * ((double)cfg.k / Ki);
    } else {
      pw_comp += c;
    }
  }
  mm_comp *= (double)scale;
  pw_comp *= (double)scale;
  result.compute_per_step = mm_comp;

  // Memory transfer costs per tile-step.
  // Use a small set to avoid double-charging shared inputs.
  double lhs_load = 0, rhs_load = 0, pw_in_load = 0, out_evict = 0;
  std::set<size_t> xfer_counted;

  for (auto i : ops_) {
    const auto &op = prob_->ops[i];
    if (op.type == OpType::MatMul) {
      int64_t Ki = op_K(i);
      if (boundary_inputs_.count(op.inputs[0]) &&
          !retained_from_prev.count(op.inputs[0]) &&
          !xfer_counted.count(op.inputs[0])) {
        lhs_load += (double)(cfg.h * Ki) / B;
        xfer_counted.insert(op.inputs[0]);
      }
      if (boundary_inputs_.count(op.inputs[1]) &&
          !retained_from_prev.count(op.inputs[1]) &&
          !xfer_counted.count(op.inputs[1])) {
        rhs_load += (double)(cfg.k * cfg.w) / B;
        xfer_counted.insert(op.inputs[1]);
      }
    } else {
      for (auto t : op.inputs)
        if (boundary_inputs_.count(t) && !retained_from_prev.count(t) &&
            !xfer_counted.count(t)) {
          pw_in_load += (double)(cfg.h * cfg.w) / B;
          xfer_counted.insert(t);
        }
    }
    for (auto t : op.outputs)
      if (boundary_outputs_.count(t) && !retain_these.count(t) &&
          !xfer_counted.count(t)) {
        out_evict += (double)(cfg.h * cfg.w) / B;
        xfer_counted.insert(t);
      }
  }

  // Per-tile cost given reuse pattern
  auto tile_cost = [&](bool lhs_fresh, bool rhs_fresh) {
    double lat = 0;
    for (int ks = 0; ks < nk; ks++) {
      double mi = rhs_load; // RHS strips streamed each k-step
      if (ks == 0) {
        mi = 0;
        if (lhs_fresh)
          mi += lhs_load;
        if (rhs_fresh)
          mi += rhs_load;
        mi += pw_in_load;
      }
      double mo = (ks == nk - 1) ? out_evict : 0;
      double step_comp = (ks == nk - 1) ? mm_comp + pw_comp : mm_comp;
      lat += std::max(step_comp, mi + mo);
    }
    return lat;
  };

  if (cfg.snake == SnakeDir::None) {
    result.latency = (double)num_tiles * tile_cost(true, true);
  } else {
    int count_ff, count_rf, count_fr;
    if (cfg.snake == SnakeDir::RowMajor) {
      count_ff = 1;
      count_fr = num_th - 1;
      count_rf = (num_tw - 1) * num_th;
    } else { // ColMajor
      count_ff = 1;
      count_rf = num_tw - 1;
      count_fr = (num_th - 1) * num_tw;
    }
    result.latency = count_ff * tile_cost(true, true) +
                     count_rf * tile_cost(false, true) +
                     count_fr * tile_cost(true, false);
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

  std::vector<SnakeDir> snakes;
  if (has_matmul_) {
    snakes = {SnakeDir::RowMajor, SnakeDir::ColMajor};
  } else {
    snakes = {SnakeDir::None};
  }

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
            if (!is_feasible(cfg, retained_from_prev, retain_these))
              continue;
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