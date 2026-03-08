#include "core/subgraph.h"
#include <algorithm>
#include <cmath>

// ============================================================================
// Utility
// ============================================================================

static std::vector<int64_t> all_divisors(int64_t n) {
    std::vector<int64_t> result;
    for (int64_t i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            result.push_back(i);
            if (i != n / i) result.push_back(n / i);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

// ============================================================================
// Factory
// ============================================================================

std::optional<Subgraph> Subgraph::create(const Problem& prob,
                                          const DAG& dag,
                                          std::vector<size_t> op_indices) {
    if (op_indices.empty()) return std::nullopt;

    Subgraph sg;
    sg.prob_ = &prob;
    sg.dag_ = &dag;
    sg.ops_ = std::move(op_indices);

    std::set<size_t> op_set(sg.ops_.begin(), sg.ops_.end());
    std::set<size_t> produced, consumed;

    for (auto i : sg.ops_) {
        for (auto t : prob.ops[i].outputs) produced.insert(t);
        for (auto t : prob.ops[i].inputs) consumed.insert(t);
        if (prob.ops[i].type == OpType::MatMul) {
            sg.has_matmul_ = true;
            int64_t Ki = prob.tensors[prob.ops[i].inputs[0]].width;
            sg.max_K_ = std::max(sg.max_K_, Ki);
        }
    }

    // Collect distinct K values for k enumeration
    if (sg.has_matmul_) {
        std::set<int64_t> ks;
        for (auto i : sg.ops_)
            if (prob.ops[i].type == OpType::MatMul)
                ks.insert(prob.tensors[prob.ops[i].inputs[0]].width);
        sg.all_K_values_.assign(ks.begin(), ks.end());
    }

    // Boundary inputs: consumed but not produced inside
    for (auto t : consumed)
        if (!produced.count(t)) sg.boundary_inputs_.insert(t);

    // Ephemeral: produced AND consumed inside (local check — correct for
    // the execution model where recomputation creates independent instances)
    for (auto t : produced)
        if (consumed.count(t)) sg.ephemeral_.insert(t);

    // Boundary outputs: produced and not ephemeral
    for (auto t : produced)
        if (!sg.ephemeral_.count(t)) sg.boundary_outputs_.insert(t);

    // Validate: exactly one boundary output (single-sink)
    if (sg.boundary_outputs_.size() != 1) return std::nullopt;

    // Validate: ops form a connected sub-DAG (undirected reachability)
    if (sg.ops_.size() > 1) {
        std::set<size_t> visited;
        std::vector<size_t> bfs = {sg.ops_[0]};
        visited.insert(sg.ops_[0]);

        while (!bfs.empty()) {
            size_t u = bfs.back(); bfs.pop_back();
            // Check predecessors and successors within the op set
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
        }
        if (visited.size() != op_set.size()) return std::nullopt;
    }

    sg.sink_tensor_ = *sg.boundary_outputs_.begin();
    sg.out_W_ = prob.tensors[sg.sink_tensor_].width;
    sg.out_H_ = prob.tensors[sg.sink_tensor_].height;

    // Collect all output tensor dimensions (including ephemeral) for tiling validation.
    // Every op's output must be tileable at the same [w, h].
    {
        std::set<int64_t> ws, hs;
        for (auto i : sg.ops_) {
            for (auto t : prob.ops[i].outputs) {
                ws.insert(prob.tensors[t].width);
                hs.insert(prob.tensors[t].height);
            }
        }
        sg.all_out_widths_.assign(ws.begin(), ws.end());
        sg.all_out_heights_.assign(hs.begin(), hs.end());
    }

    return sg;
}

// ============================================================================
// Tiling validity
// ============================================================================

bool Subgraph::is_valid_tiling(const TileConfig& cfg) const {
    if (cfg.w <= 0 || cfg.h <= 0 || cfg.k <= 0) return false;

    // w must divide every op's output width
    for (int64_t ow : all_out_widths_)
        if (ow % cfg.w != 0) return false;

    // h must divide every op's output height
    for (int64_t oh : all_out_heights_)
        if (oh % cfg.h != 0) return false;

    // k must divide every MatMul's reduction dimension
    for (int64_t Ki : all_K_values_)
        if (Ki % cfg.k != 0) return false;

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
// For retained_from_prev tensors: the FULL tensor is already resident from
//   the previous step. It occupies width×height, not a tile-sized slice.
//   If the tensor is also a boundary input, the hardware reads slices from
//   the resident copy (zero transfer cost) but the full tensor still consumes
//   capacity.
//
// For retain_these tensors (this step's output being retained): tiles
//   accumulate without eviction. At the last tile, (num_tiles-1)×h×w of
//   previously-computed tiles sit in memory alongside the current tile's
//   working set. The peak additional cost is (full_output_size - h×w).
// ============================================================================

int64_t Subgraph::working_set(const TileConfig& cfg,
                               const std::set<size_t>& retained_from_prev,
                               const std::set<size_t>& retain_these) const {
    int64_t ws = 0;
    std::set<size_t> counted;

    for (auto i : ops_) {
        const auto& op = prob_->ops[i];
        if (op.type == OpType::MatMul) {
            int64_t Ki = op_K(i);
            size_t lhs = op.inputs[0], rhs = op.inputs[1], out = op.outputs[0];
            if (boundary_inputs_.count(lhs) && !counted.count(lhs)) {
                if (retained_from_prev.count(lhs))
                    ws += prob_->tensors[lhs].width * prob_->tensors[lhs].height;
                else
                    ws += cfg.h * Ki;
                counted.insert(lhs);
            }
            if (boundary_inputs_.count(rhs) && !counted.count(rhs)) {
                if (retained_from_prev.count(rhs))
                    ws += prob_->tensors[rhs].width * prob_->tensors[rhs].height;
                else
                    ws += cfg.k * cfg.w;
                counted.insert(rhs);
            }
            if (boundary_outputs_.count(out) && !counted.count(out)) {
                ws += cfg.h * cfg.w;
                counted.insert(out);
            }
        } else {
            for (auto t : op.inputs)
                if (boundary_inputs_.count(t) && !counted.count(t)) {
                    if (retained_from_prev.count(t))
                        ws += prob_->tensors[t].width * prob_->tensors[t].height;
                    else
                        ws += cfg.h * cfg.w;
                    counted.insert(t);
                }
        }
    }

    // Retained-from-prev tensors NOT used as boundary inputs still occupy memory
    for (auto t : retained_from_prev)
        if (!counted.count(t))
            ws += prob_->tensors[t].width * prob_->tensors[t].height;

    // Retained output accumulates: at the last tile, (full_size - h*w) of
    // previously-computed tiles are in memory alongside the current tile
    for (auto t : retain_these) {
        int64_t full = prob_->tensors[t].width * prob_->tensors[t].height;
        int64_t tile = cfg.h * cfg.w;
        if (full > tile)
            ws += full - tile;  // the h×w for current tile is already counted above
    }

    return ws;
}

bool Subgraph::is_feasible(const TileConfig& cfg,
                            const std::set<size_t>& retained_from_prev,
                            const std::set<size_t>& retain_these) const {
    return is_valid_tiling(cfg) &&
           working_set(cfg, retained_from_prev, retain_these) <= prob_->fast_memory_capacity;
}

// ============================================================================
// Cost computation (analytical, O(1) in tile count)
// ============================================================================

CostResult Subgraph::compute_cost(const TileConfig& cfg,
                                   const std::set<size_t>& retained_from_prev,
                                   const std::set<size_t>& retain_these) const {
    CostResult result;
    result.config = cfg;

    if (!is_valid_tiling(cfg)) return result;

    result.working_set = working_set(cfg, retained_from_prev, retain_these);

    if (result.working_set > prob_->fast_memory_capacity) return result;
    result.feasible = true;

    double B = (double)prob_->slow_memory_bandwidth;
    int num_tw = (int)(out_W_ / cfg.w);
    int num_th = (int)(out_H_ / cfg.h);
    int num_tiles = num_tw * num_th;
    result.num_spatial_tiles = num_tiles;
    result.num_k_passes = has_matmul_ ? (int)(max_K_ / cfg.k) : 1;
    int nk = result.num_k_passes;

    // Compute scale: number of native-sized sub-tiles per spatial tile.
    // Below native: ceil gives 1 (padded execution, full native cost).
    // Above native: ceil gives the number of native passes needed.
    auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    int64_t scale = ceil_div(cfg.w, prob_->native_w)
                  * ceil_div(cfg.h, prob_->native_h);

    // Separate MatMul compute (per k-step) from PW compute (once per tile).
    // Problem: "For Pointwise operations, k is ignored... executes once per spatial tile."
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
    result.compute_per_step = mm_comp;  // per k-step (PW added only at last step)

    // Memory transfer costs per tile-step
    double lhs_load = 0, rhs_load = 0, pw_in_load = 0, out_evict = 0;
    for (auto i : ops_) {
        const auto& op = prob_->ops[i];
        if (op.type == OpType::MatMul) {
            int64_t Ki = op_K(i);
            if (boundary_inputs_.count(op.inputs[0]) && !retained_from_prev.count(op.inputs[0]))
                lhs_load += (double)(cfg.h * Ki) / B;
            if (boundary_inputs_.count(op.inputs[1]) && !retained_from_prev.count(op.inputs[1]))
                rhs_load += (double)(cfg.k * cfg.w) / B;
        } else {
            for (auto t : op.inputs)
                if (boundary_inputs_.count(t) && !retained_from_prev.count(t))
                    pw_in_load += (double)(cfg.h * cfg.w) / B;
        }
        for (auto t : op.outputs)
            if (boundary_outputs_.count(t) && !retain_these.count(t))
                out_evict += (double)(cfg.h * cfg.w) / B;
    }

    // Per-tile cost given reuse pattern
    auto tile_cost = [&](bool lhs_fresh, bool rhs_fresh) {
        double lat = 0;
        for (int ks = 0; ks < nk; ks++) {
            double mi = rhs_load;
            if (ks == 0) {
                mi = 0;
                if (lhs_fresh) mi += lhs_load;
                if (rhs_fresh) mi += rhs_load;
                mi += pw_in_load;
            }
            double mo = (ks == nk - 1) ? out_evict : 0;
            // PW compute only at last k-step (PW runs once per tile)
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
        result.latency = count_ff * tile_cost(true, true)
                       + count_rf * tile_cost(false, true)
                       + count_fr * tile_cost(true, false);
    }

    return result;
}

// ============================================================================
// Granularity enumeration
// ============================================================================

CostResult Subgraph::best_cost(const std::set<size_t>& retained_from_prev,
                                const std::set<size_t>& retain_these) const {
    int64_t min_w = std::max<int64_t>(1, prob_->native_w / 4);
    int64_t min_h = std::max<int64_t>(1, prob_->native_h / 4);

    std::vector<SnakeDir> snakes = {SnakeDir::None};
    if (has_matmul_) {
        snakes.push_back(SnakeDir::RowMajor);
        snakes.push_back(SnakeDir::ColMajor);
    }

    // Filter raw candidates: w must divide all output widths, etc.
    auto filter_w = [&](const std::vector<int64_t>& raw) {
        std::vector<int64_t> result;
        for (int64_t ww : raw) {
            bool ok = true;
            for (int64_t ow : all_out_widths_)
                if (ow % ww != 0) { ok = false; break; }
            if (ok) result.push_back(ww);
        }
        return result;
    };
    auto filter_h = [&](const std::vector<int64_t>& raw) {
        std::vector<int64_t> result;
        for (int64_t hh : raw) {
            bool ok = true;
            for (int64_t oh : all_out_heights_)
                if (oh % hh != 0) { ok = false; break; }
            if (ok) result.push_back(hh);
        }
        return result;
    };
    auto filter_k = [&](const std::vector<int64_t>& raw) {
        std::vector<int64_t> result;
        for (int64_t kk : raw) {
            bool ok = true;
            for (int64_t Ki : all_K_values_)
                if (Ki % kk != 0) { ok = false; break; }
            if (ok) result.push_back(kk);
        }
        if (result.empty()) result = {1};
        return result;
    };

    // Search over given candidate lists
    auto search = [&](const std::vector<int64_t>& ws,
                      const std::vector<int64_t>& hs,
                      const std::vector<int64_t>& ks,
                      int64_t mw, int64_t mh) {
        CostResult best;
        for (int64_t ww : ws) {
            if (ww < mw) continue;
            for (int64_t hh : hs) {
                if (hh < mh) continue;
                for (int64_t kk : ks) {
                    for (auto sd : snakes) {
                        TileConfig cfg{ww, hh, kk, sd};
                        if (!is_feasible(cfg, retained_from_prev, retain_these)) continue;
                        auto r = compute_cost(cfg, retained_from_prev, retain_these);
                        if (r.feasible && r.latency < best.latency) best = r;
                    }
                }
            }
        }
        return best;
    };

    auto ws_cand = filter_w(all_divisors(out_W_));
    auto hs_cand = filter_h(all_divisors(out_H_));
    auto ks_cand = has_matmul_ ? filter_k(all_divisors(max_K_)) : std::vector<int64_t>{1};

    CostResult best = search(ws_cand, hs_cand, ks_cand, min_w, min_h);
    if (!best.feasible)
        best = search(ws_cand, hs_cand, ks_cand, 1, 1);  // relax min tile size

    return best;
}