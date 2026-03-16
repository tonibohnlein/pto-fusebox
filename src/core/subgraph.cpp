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
      // Ephemeral only if ALL consumers in the DAG are inside the subgraph.
      // If any consumer is external, the tensor must be a boundary output
      // so external consumers can access it from slow memory.
      bool all_internal = true;
      for (auto cop : dag.tensor_consumers[t])
        if (!is_in_sg[cop]) { all_internal = false; break; }
      if (all_internal)
        is_ephemeral[t] = true;
      // else: has external consumer → will be boundary output below
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

  // ---- Precompute per-boundary-tensor role info ----
  //
  // Each boundary tensor's role is determined by its position in the chain,
  // not by what op type directly touches it. A tensor feeding a MatMul LHS
  // via a PW chain has the same residency behavior as a direct MM LHS —
  // except that the ephemeral chain means the full h×K is never resident
  // (it's streamed at h×k per step instead).
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
        bool out_is_boundary = sg.boundary_outputs_.count(out) > 0;

        // Direct MM LHS: resident at h × K across all k-steps.
        // Check both boundary inputs AND internally-produced boundary outputs.
        // Special case: if this is the SINK MatMul (output is boundary) and
        // nk > 1, the LHS is actually streamed at h×k per step, not resident.
        // We mark is_sink_mm_lhs here; working_set/compute_cost switch at runtime.
        if (sg.boundary_inputs_.count(lhs)) {
          size_t idx = ensure(lhs);
          auto &info = sg.boundary_tensor_info_[idx];
          info.resident_K = std::max(info.resident_K, prob.tensors[lhs].width);
          if (out_is_boundary) info.is_sink_mm_lhs = true;
        } else if (sg.boundary_outputs_.count(lhs) && is_produced[lhs]) {
          // LHS is produced internally but has external consumers →
          // boundary output, NOT ephemeral. Still needs h×K in fast memory
          // for this MatMul to consume it, but no slow memory read.
          size_t idx = ensure(lhs);
          auto &info = sg.boundary_tensor_info_[idx];
          info.resident_K = std::max(info.resident_K, prob.tensors[lhs].width);
          info.is_internally_produced = true;
          if (out_is_boundary) info.is_sink_mm_lhs = true;
        }

        // Direct MM RHS:
        //   Output-producing MatMul (output is boundary): streamed at k × w.
        //   Upstream MatMul (output is ephemeral):
        //     If ephemeral feeds another MatMul (width maps to k): K_upstream × k
        //     If ephemeral feeds PW→boundary (width maps to w): K_upstream × w
        if (sg.boundary_inputs_.count(rhs)) {
          auto &info = sg.boundary_tensor_info_[ensure(rhs)];
          if (out_is_boundary) {
            info.is_streamed_k_by_w = true;
          } else {
            // Ephemeral output — check what its width maps to
            if (is_ephemeral[out] && eph_sw[out] == SliceW::W_param) {
              info.stream_fixed_by_w = std::max(
                  info.stream_fixed_by_w, prob.tensors[rhs].height);
            } else {
              info.stream_fixed_by_k = std::max(
                  info.stream_fixed_by_k, prob.tensors[rhs].height);
            }
          }
        } else if (sg.boundary_outputs_.count(rhs) && is_produced[rhs]) {
          // RHS is produced internally but has external consumers →
          // boundary output, needs streaming footprint but no slow memory read.
          auto &info = sg.boundary_tensor_info_[ensure(rhs)];
          info.is_internally_produced = true;
          if (out_is_boundary) {
            info.is_streamed_k_by_w = true;
          } else {
            if (is_ephemeral[out] && eph_sw[out] == SliceW::W_param) {
              info.stream_fixed_by_w = std::max(
                  info.stream_fixed_by_w, prob.tensors[rhs].height);
            } else {
              info.stream_fixed_by_k = std::max(
                  info.stream_fixed_by_k, prob.tensors[rhs].height);
            }
          }
        }

        if (out_is_boundary) {
          size_t idx = ensure(out);
          sg.boundary_tensor_info_[idx].is_mm_out = true;
          sg.boundary_tensor_info_[idx].is_boundary_out = true;
        }
      } else { // Pointwise
        // PW is element-wise: input shape = output shape. The output's shape
        // is determined by what it feeds downstream. Follow the ephemeral
        // chain to discover the role.
        //
        // TODO: This classification doesn't distinguish whether the downstream
        // MatMul is the output-producing one or an upstream one. For PW feeding
        // an upstream MatMul's RHS, the correct slice is fixed×k (not k×w),
        // and for PW feeding an upstream MatMul's LHS, the correct slice is a
        // once-per-tile h×K_upstream (not per-k-step h×k). This requires the
        // eph_sw/eph_sh propagation to carry the downstream MatMul identity.
        // In practice, the direct MatMul boundary RHS case (handled above) is
        // far more common than PW→ephemeral→upstream-MatMul chains.
        size_t pw_out = op.outputs[0];
        bool feeds_mm_lhs = is_ephemeral[pw_out] &&
                            eph_sw[pw_out] == SliceW::K_param &&
                            eph_sh[pw_out] == SliceH::H_param;
        bool feeds_mm_rhs = is_ephemeral[pw_out] &&
                            eph_sw[pw_out] == SliceW::W_param &&
                            eph_sh[pw_out] == SliceH::K_param;

        for (auto t : op.inputs) {
          if (sg.boundary_inputs_.count(t)) {
            auto &info = sg.boundary_tensor_info_[ensure(t)];
            if (feeds_mm_lhs)       info.is_streamed_h_by_k = true;
            else if (feeds_mm_rhs)  info.is_streamed_k_by_w = true;
            else                    info.is_tile_input = true;
          } else if (sg.boundary_outputs_.count(t) && is_produced[t]) {
            // PW input is produced internally but has external consumers.
            auto &info = sg.boundary_tensor_info_[ensure(t)];
            info.is_internally_produced = true;
            if (feeds_mm_lhs)       info.is_streamed_h_by_k = true;
            else if (feeds_mm_rhs)  info.is_streamed_k_by_w = true;
            else                    info.is_tile_input = true;
          }
        }
      }
      for (auto t : op.outputs)
        if (sg.boundary_outputs_.count(t))
          sg.boundary_tensor_info_[ensure(t)].is_boundary_out = true;
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
// Working set
// ============================================================================
int64_t Subgraph::working_set(const TileConfig &cfg,
                              const std::set<size_t> &retained_from_prev,
                              const std::set<size_t> &retain_these) const {
  // A tensor may appear in BOTH retained_from_prev and retain_these when
  // it passes through this step (entered from previous, forwarded to next).
  // Working set: charge it once as full_size (from retained_from_prev).
  // The retain_these post-pass skips tensors already in retained_from_prev
  // to avoid double-counting.

  int64_t ws = 0;
  int nk = has_matmul_ ? (int)(output_K_ / cfg.k) : 1;

  // ---- Main boundary tensor loop ----
  // For each boundary tensor, charge the maximum fast-memory footprint it
  // occupies during a single tile step:
  //   • retained_from_prev  → full tensor is already resident; charge full_size.
  //   • retain_these        → SKIP here; their full sizes are added in the
  //                           post-pass below.  Removing them from the per-tile
  //                           accounting avoids the role-based mismatch
  //                           (LHS strips are h×K, not h×w) that plagued the
  //                           old "full - tile" subtraction.
  //   • everything else     → charge the role-based tile slice.
  for (auto &info : boundary_tensor_info_) {
    if (retained_from_prev.count(info.id)) {
      ws += info.full_size;
      continue;
    }
    if (retain_these.count(info.id)) {
      // Handled in post-pass; skip per-tile slice accounting.
      continue;
    }

    int64_t max_size = 0;
    if (info.resident_K > 0) {
      if (info.is_sink_mm_lhs && nk > 1) {
        // Sink MatMul LHS with split-K: streamed h×k per step, not resident.
        max_size = std::max(max_size, cfg.h * cfg.k);
      } else {
        // Upstream MatMul LHS or nk=1: resident h×K across all k-steps.
        max_size = std::max(max_size, cfg.h * info.resident_K);
      }
    }
    if (info.is_streamed_k_by_w)
      max_size = std::max(max_size, cfg.k * cfg.w);
    if (info.is_streamed_h_by_k)
      max_size = std::max(max_size, cfg.h * cfg.k);
    if (info.stream_fixed_by_k > 0)
      max_size = std::max(max_size, info.stream_fixed_by_k * cfg.k);
    if (info.stream_fixed_by_w > 0)
      max_size = std::max(max_size, info.stream_fixed_by_w * cfg.w);
    if (info.is_tile_input)
      max_size = std::max(max_size, cfg.h * cfg.w);
    if (info.is_mm_out)
      max_size = std::max(max_size, cfg.h * cfg.w);
    if (info.is_boundary_out)
      max_size = std::max(max_size, cfg.h * cfg.w);
    ws += max_size;
  }

  // ---- retained_from_prev tensors NOT in boundary_tensor_info_ ----
  // A tensor carried from the previous step that this subgraph does not
  // directly use still occupies fast memory and must be counted.
  for (auto t : retained_from_prev) {
    if (t < tensor_id_to_info_.size() && tensor_id_to_info_[t] >= 0)
      continue;  // already counted in the main boundary tensor loop above
    ws += prob_->tensors[t].size();
  }

  // ---- Post-pass: retained output tensors ----
  // Each tensor in retain_these must fit in its entirety in fast memory so
  // that all completed tiles accumulate there while the next tile executes.
  // Charging full_size is the correct peak occupancy model:
  //   ws_peak = (all non-retained tile slices) + full_size(each retained tensor)
  // Skip tensors already in retained_from_prev — they're already counted
  // with full_size in the main loop above (pass-through case).
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

  // Memory transfer costs per role, iterated over boundary_tensor_info_
  // (already deduplicated).
  //
  // Four transfer categories by timing:
  //   resident_load   : loaded once per tile when fresh (h×K for upstream MM LHS)
  //   sink_lhs_load   : loaded per k-step when fresh (h×k for sink MM LHS with nk>1).
  //                     Row-level reuse same as resident_load, but distributed
  //                     across k-steps instead of concentrated at step 0.
  //   stream_load     : loaded per k-step (k×w for RHS slot, h×k for LHS-chain)
  //   tile_load       : loaded once per tile (h×w for tile-input PW)
  //   out_evict       : evicted per tile at final k-step
  double resident_load = 0, sink_lhs_load = 0;
  double stream_load = 0, tile_load = 0, out_evict = 0;

  for (auto &info : boundary_tensor_info_) {
    bool retained_in = retained_from_prev.count(info.id);
    bool retained_out = retain_these.count(info.id);

    // Input transfer costs (skip if retained from prev — already in fast mem,
    // or if internally produced — data is produced within this subgraph)
    if (!retained_in && !info.is_internally_produced) {
      if (info.resident_K > 0) {
        if (info.is_sink_mm_lhs && nk > 1) {
          // Sink MatMul LHS with split-K: h×k loaded per step.
          // Row reuse: same pattern as resident (reused within a row of tiles).
          sink_lhs_load += (double)(cfg.h * cfg.k) / B;
        } else {
          // Upstream MatMul LHS or nk=1: h×K resident, loaded once per tile.
          resident_load += (double)(cfg.h * info.resident_K) / B;
        }
      }
      if (info.is_streamed_k_by_w)
        stream_load += (double)(cfg.k * cfg.w) / B;
      if (info.is_streamed_h_by_k)
        stream_load += (double)(cfg.h * cfg.k) / B;
      if (info.stream_fixed_by_k > 0)
        stream_load += (double)(info.stream_fixed_by_k * cfg.k) / B;
      if (info.stream_fixed_by_w > 0)
        stream_load += (double)(info.stream_fixed_by_w * cfg.w) / B;
      if (info.is_tile_input)
        tile_load += (double)(cfg.h * cfg.w) / B;
    }

    // Output eviction cost (skip if retained for next step)
    if (info.is_boundary_out && !retained_out)
      out_evict += (double)(cfg.h * cfg.w) / B;
  }

  // Per-tile cost given reuse pattern — O(1) analytical formula.
  //
  // The nk k-steps have 3 distinct phases:
  //   Step 0:        load resident (if fresh) + first stream strip + tile inputs
  //   Steps 1..nk-2: load stream strip only
  //   Step nk-1:     load stream strip + evict output + PW compute
  // When nk=1, all three collapse into a single step.
  //
  // sink_lhs_load: loaded per k-step when resident_fresh (row-level reuse).
  // Unlike stream_load (always fresh when nk>1), sink_lhs is reused across
  // tiles within the same row — its position only depends on (d, row).
  auto tile_cost = [&](bool resident_fresh, bool stream_fresh) -> double {
    // When k < K (nk > 1), the stream strip loaded in the last k-step of the
    // previous tile covers a different k-range than step 0 of this tile.
    // Resident reuse IS valid: the full h×K stays resident across k-steps.
    if (nk > 1) stream_fresh = true;

    // sink_lhs follows resident freshness: reused within a row
    double slhs = resident_fresh ? sink_lhs_load : 0.0;

    if (nk == 1) {
      // Single step: everything happens at once
      // (sink_lhs_load is 0 when nk=1 by construction)
      double mi = tile_load;
      if (resident_fresh) mi += resident_load;
      if (stream_fresh) mi += stream_load;
      double mo = out_evict;
      return std::max(mm_comp + pw_comp, mi + mo);
    }

    // Step 0: resident + first stream strip + tile inputs + sink_lhs, compute = mm only
    double mi0 = tile_load + slhs;
    if (resident_fresh) mi0 += resident_load;
    if (stream_fresh) mi0 += stream_load;
    double step0 = std::max(mm_comp, mi0);

    // Middle steps (1..nk-2): each loads one stream strip + sink_lhs, mm compute
    double mid_step = std::max(mm_comp, stream_load + slhs);
    double mid_total = (nk >= 3) ? (double)(nk - 2) * mid_step : 0.0;

    // Last step (nk-1): stream strip + sink_lhs + evict + PW compute
    double last = std::max(mm_comp + pw_comp, stream_load + slhs + out_evict);

    return step0 + mid_total + last;
  };

  if (cfg.snake == SnakeDir::None) {
    // Raster order (row-major): consecutive tiles in the same row share the
    // resident data (row strip). Moving to a new row reloads resident.
    //   Per row: 1 tile with both fresh + (num_tw-1) tiles with resident reused.
    //   Rows: num_th rows.
    // For PW-only subgraphs resident_load/stream_load are 0 so the distinction
    // is immaterial — the formula collapses to num_tiles * tile_cost(true,true).
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