#include "core/subgraph.h"
#include <algorithm>
#include <cmath>
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
  // A tensor produced AND consumed within the subgraph is ephemeral ONLY IF
  // all its consumers are internal. If any consumer is external, the tensor
  // is a boundary output — it must be written to slow memory so that the
  // external consumer's subgraph can read it.
  std::vector<bool> is_ephemeral(num_tensors, false);

  for (size_t t = 0; t < num_tensors; t++) {
    if (is_consumed[t] && !is_produced[t])
      sg.boundary_inputs_.insert(t);
    if (is_produced[t] && is_consumed[t]) {
      // Check that ALL consumers are within this subgraph
      bool all_consumers_internal = true;
      for (auto cop : dag.tensor_consumers[t]) {
        if (!is_in_sg[cop]) {
          all_consumers_internal = false;
          break;
        }
      }
      if (all_consumers_internal)
        is_ephemeral[t] = true;
      // else: has external consumers → boundary output (written to slow mem)
    }
  }
  for (size_t t = 0; t < num_tensors; t++) {
    if (is_produced[t] && !is_ephemeral[t])
      sg.boundary_outputs_.insert(t);
    if (is_ephemeral[t])
      sg.ephemeral_.insert(t);
  }

  // Validate: ephemeral tensors must have exactly one INTERNAL consumer.
  // (The tile exists momentarily — can't fan out within a subgraph.)
  {
    std::vector<int> eph_count(num_tensors, 0);
    for (auto i : sg.ops_)
      for (auto t : prob.ops[i].inputs)
        if (is_ephemeral[t])
          eph_count[t]++;
    for (size_t t = 0; t < num_tensors; t++)
      if (is_ephemeral[t] && eph_count[t] != 1)
        return std::nullopt;
  }

  // Must have at least one boundary output
  if (sg.boundary_outputs_.empty())
    return std::nullopt;

  // Detect PW sinks (using vector for producer lookup, not map)
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

    // ---- Collect role-based tiling constraints ----
    // For ephemeral tensors, determine slice type by following the consumer chain
    // iteratively (no std::function/recursion needed).

    enum class SliceW : uint8_t { W_param, K_param };
    enum class SliceH : uint8_t { H_param, K_param };

    // For each ephemeral tensor: who produces it, who consumes it
    std::vector<int> eph_consumer(num_tensors, -1);
    for (auto i : sg.ops_)
      for (auto t : prob.ops[i].inputs)
        if (is_ephemeral[t])
          eph_consumer[t] = (int)i;

    // Determine slice type for each ephemeral tensor by following consumer chain
    std::vector<SliceW> eph_sw(num_tensors, SliceW::W_param);
    std::vector<SliceH> eph_sh(num_tensors, SliceH::H_param);
    std::vector<bool> eph_resolved(num_tensors, false);

    for (size_t t = 0; t < num_tensors; t++) {
      if (!is_ephemeral[t] || eph_resolved[t]) continue;

      // Follow chain: t → consumer_op → consumer_output → ...
      // until we hit a boundary output or a MatMul consumer
      size_t cur = t;
      // Collect chain for back-propagation
      std::vector<size_t> chain;
      SliceW sw = SliceW::W_param;
      SliceH sh = SliceH::H_param;

      while (true) {
        chain.push_back(cur);
        int cop = eph_consumer[cur];
        if (cop < 0) break; // shouldn't happen, but safety

        const auto &op = prob.ops[cop];
        if (op.type == OpType::MatMul) {
          // Determine if cur is LHS or RHS
          if (op.inputs[0] == cur)
            { sw = SliceW::K_param; sh = SliceH::H_param; }
          else
            { sw = SliceW::W_param; sh = SliceH::K_param; }
          break;
        } else {
          // PW: propagate from its output
          size_t pw_out = op.outputs[0];
          if (!is_ephemeral[pw_out]) {
            // PW output is boundary → w×h
            sw = SliceW::W_param; sh = SliceH::H_param;
            break;
          }
          if (eph_resolved[pw_out]) {
            sw = eph_sw[pw_out]; sh = eph_sh[pw_out];
            break;
          }
          cur = pw_out;
        }
      }

      // Apply to all tensors in chain
      for (auto ct : chain) {
        eph_sw[ct] = sw; eph_sh[ct] = sh;
        eph_resolved[ct] = true;
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
        if (sg.boundary_outputs_.count(out))
          add_constraint(out, SliceW::W_param, SliceH::H_param);
        else
          add_constraint(out, eph_sw[out], eph_sh[out]);
      } else {
        size_t out = op.outputs[0];
        SliceW sw; SliceH sh;
        if (sg.boundary_outputs_.count(out))
          { sw = SliceW::W_param; sh = SliceH::H_param; }
        else
          { sw = eph_sw[out]; sh = eph_sh[out]; }
        add_constraint(out, sw, sh);
        for (auto t : op.inputs)
          add_constraint(t, sw, sh);
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

  // ---- Precompute per-boundary-tensor role info ----
  {
    // Use vector indexed by tensor ID, then flatten to boundary_tensor_info_
    std::vector<int> tensor_in_info(num_tensors, -1); // -1 = not in info
    auto ensure = [&](size_t t) -> size_t {
      if (tensor_in_info[t] < 0) {
        tensor_in_info[t] = (int)sg.boundary_tensor_info_.size();
        sg.boundary_tensor_info_.push_back(
          {t, prob.tensors[t].width * prob.tensors[t].height});
      }
      return (size_t)tensor_in_info[t];
    };

    for (auto i : sg.ops_) {
      const auto &op = prob.ops[i];
      if (op.type == OpType::MatMul) {
        size_t lhs = op.inputs[0], rhs = op.inputs[1], out = op.outputs[0];
        if (sg.boundary_inputs_.count(lhs)) {
          size_t idx = ensure(lhs);
          auto &info = sg.boundary_tensor_info_[idx];
          info.max_lhs_K = std::max(info.max_lhs_K, prob.tensors[lhs].width);
        }
        if (sg.boundary_inputs_.count(rhs)) {
          sg.boundary_tensor_info_[ensure(rhs)].is_mm_rhs = true;
        }
        if (sg.boundary_outputs_.count(out)) {
          size_t idx = ensure(out);
          sg.boundary_tensor_info_[idx].is_mm_out = true;
          sg.boundary_tensor_info_[idx].is_boundary_out = true;
        }
      } else {
        for (auto t : op.inputs)
          if (sg.boundary_inputs_.count(t))
            sg.boundary_tensor_info_[ensure(t)].is_pw_in = true;
      }
      for (auto t : op.outputs)
        if (sg.boundary_outputs_.count(t))
          sg.boundary_tensor_info_[ensure(t)].is_boundary_out = true;
    }
  }

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
// Working set
// ============================================================================

int64_t Subgraph::working_set(const TileConfig &cfg,
                              const std::set<size_t> &retained_from_prev,
                              const std::set<size_t> &retain_these) const {
  int64_t ws = 0;

  for (auto &info : boundary_tensor_info_) {
    if (retained_from_prev.count(info.id)) {
      ws += info.full_size;
      continue;
    }

    int64_t max_size = 0;
    if (info.max_lhs_K > 0)
      max_size = std::max(max_size, cfg.h * info.max_lhs_K);
    if (info.is_mm_rhs)
      max_size = std::max(max_size, cfg.k * cfg.w);
    if (info.is_pw_in)
      max_size = std::max(max_size, cfg.h * cfg.w);
    if (info.is_mm_out)
      max_size = std::max(max_size, cfg.h * cfg.w);
    // PW boundary outputs need their tile in fast memory too.
    // (Redundant for MM outputs where is_mm_out already covers this.)
    if (info.is_boundary_out)
      max_size = std::max(max_size, cfg.h * cfg.w);

    ws += max_size;
  }

  // Retained-from-prev tensors NOT in boundary_tensor_info_
  for (auto t : retained_from_prev) {
    bool found = false;
    for (auto &info : boundary_tensor_info_)
      if (info.id == t) { found = true; break; }
    if (!found)
      ws += prob_->tensors[t].size();
  }

  // Retained output accumulation
  for (auto t : retain_these) {
    int64_t full = prob_->tensors[t].size();
    int64_t tile = cfg.h * cfg.w;
    if (full <= tile)
      continue;

    bool tile_counted = false;
    for (auto &info : boundary_tensor_info_) {
      if (info.id == t) {
        int64_t sz = 0;
        if (info.max_lhs_K > 0) sz = std::max(sz, cfg.h * info.max_lhs_K);
        if (info.is_mm_rhs) sz = std::max(sz, cfg.k * cfg.w);
        if (info.is_pw_in) sz = std::max(sz, cfg.h * cfg.w);
        if (info.is_mm_out) sz = std::max(sz, cfg.h * cfg.w);
        if (info.is_boundary_out) sz = std::max(sz, cfg.h * cfg.w);
        if (retained_from_prev.count(t)) sz = full;
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

  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  int64_t scale =
      ceil_div(cfg.w, prob_->native_w) * ceil_div(cfg.h, prob_->native_h);

  // Compute: separate MM (per k-step) from PW (once per tile)
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

  // Memory transfer costs — iterate over boundary_tensor_info_ (already
  // deduplicated) instead of iterating over ops with a std::set guard.
  double lhs_load = 0, rhs_load = 0, pw_in_load = 0, out_evict = 0;

  for (auto &info : boundary_tensor_info_) {
    bool retained_in = retained_from_prev.count(info.id);
    bool retained_out = retain_these.count(info.id);

    // Input transfer costs (skip if retained from prev — already in fast mem)
    if (!retained_in) {
      if (info.max_lhs_K > 0)
        lhs_load += (double)(cfg.h * info.max_lhs_K) / B;
      if (info.is_mm_rhs)
        rhs_load += (double)(cfg.k * cfg.w) / B;
      if (info.is_pw_in)
        pw_in_load += (double)(cfg.h * cfg.w) / B;
    }

    // Output eviction cost (skip if retained for next step)
    if (info.is_boundary_out && !retained_out)
      out_evict += (double)(cfg.h * cfg.w) / B;
  }

  // Per-tile cost given reuse pattern — O(1) analytical formula.
  // The nk k-steps have 3 distinct phases:
  //   Step 0:        load LHS (if fresh) + first RHS (if fresh) + PW inputs
  //   Steps 1..nk-2: load RHS strip (all identical)
  //   Step nk-1:     load RHS strip + evict output + PW compute
  // When nk=1, all three collapse into a single step.
  auto tile_cost = [&](bool lhs_fresh, bool rhs_fresh) -> double {
    // When k < K (nk > 1), the RHS strip loaded in the last k-step of the
    // previous tile covers a different k-range than step 0 of this tile.
    // LHS reuse IS valid: the full h×K strip stays resident across k-steps.
    if (nk > 1) rhs_fresh = true;

    if (nk == 1) {
      // Single step: everything happens at once
      double mi = pw_in_load;
      if (lhs_fresh) mi += lhs_load;
      if (rhs_fresh) mi += rhs_load;
      double mo = out_evict;
      return std::max(mm_comp + pw_comp, mi + mo);
    }

    // Step 0: LHS + first RHS + PW inputs, compute = mm only
    double mi0 = pw_in_load;
    if (lhs_fresh) mi0 += lhs_load;
    if (rhs_fresh) mi0 += rhs_load;
    double step0 = std::max(mm_comp, mi0);

    // Middle steps (1..nk-2): each loads one RHS strip, mm compute
    double mid_step = std::max(mm_comp, rhs_load);
    double mid_total = (nk >= 3) ? (double)(nk - 2) * mid_step : 0.0;

    // Last step (nk-1): RHS strip + evict + PW compute
    double last = std::max(mm_comp + pw_comp, rhs_load + out_evict);

    return step0 + mid_total + last;
  };

  if (cfg.snake == SnakeDir::None) {
    // Raster order (row-major): consecutive tiles in the same row share the
    // LHS row strip.  Moving to a new row reloads both LHS and RHS.
    //   Per row: 1 tile with both fresh + (num_tw-1) tiles with LHS reused.
    //   Rows: num_th rows.
    // For PW-only subgraphs lhs_load/rhs_load are 0 so the distinction is
    // immaterial — the formula collapses to num_tiles * tile_cost(true,true).
    if (has_matmul_ && num_tw > 1) {
      int count_ff = num_th;
      int count_rf = (num_tw - 1) * num_th;
      result.latency = count_ff * tile_cost(true, true) +
                       count_rf * tile_cost(false, true);
    } else {
      result.latency = (double)num_tiles * tile_cost(true, true);
    }
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