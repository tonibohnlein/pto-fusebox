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
  bool reduces_width = false, reduces_height = false;  // for reduced-axis homogeneity

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
    if (prob.ops[i].type == OpType::Reduction) {
      sg.has_reduction_ = true;
      // Reduced axis = the dim that collapses (input extent -> 1 in the output).
      size_t in0 = prob.ops[i].inputs[0], out = prob.ops[i].output();
      if (prob.tensors[out].width < prob.tensors[in0].width) {
        sg.reduced_axis_ = 1;  // width  (row reduction: [H,W] -> [H,1])
        sg.reduced_extent_ = std::max(sg.reduced_extent_, prob.tensors[in0].width);
        reduces_width = true;
      } else if (prob.tensors[out].height < prob.tensors[in0].height) {
        sg.reduced_axis_ = 2;  // height (col reduction: [H,W] -> [1,W])
        sg.reduced_extent_ = std::max(sg.reduced_extent_, prob.tensors[in0].height);
        reduces_height = true;
      }
    }
  }

  // 910B: subgraphs must be UNIT-HOMOGENEOUS. Cube (MatMul) and vector
  // (Pointwise/Reduction) ops run on different cores; their handoff routes
  // through DDR (no free ephemeral) and the unified single grid can't express
  // cube/vector overlap — so they are never fused into one group. Opaque ops
  // (gather/scatter/sort/transpose) are barriers: singleton groups only.
  // Gated on parallel-core mode; single-context (cores==1) keeps the
  // competition behavior (mixed fusion allowed).
  if (prob.num_cube_cores > 1 || prob.num_vector_cores > 1) {
    bool has_cube = false, has_vector = false, has_opaque = false;
    for (auto i : sg.ops_) {
      switch (prob.ops[i].type) {
        case OpType::MatMul: has_cube = true; break;
        case OpType::Pointwise:
        case OpType::Reduction: has_vector = true; break;
        case OpType::Opaque: has_opaque = true; break;
      }
    }
    if (has_cube && has_vector) return std::nullopt;            // no cube↔vector fusion
    if (has_opaque && sg.ops_.size() > 1) return std::nullopt;  // Opaque is a barrier
    // Reduced-axis homogeneity: a subgraph may not fuse reductions on DIFFERENT
    // axes (a width-reduction AND a height-reduction). A single unified tile is
    // coupled to the FULL extent on each reduced axis, so fusing both would force
    // the whole tensor into one tile on one core — no spatial parallelism, never
    // beneficial. The single reduced_axis_ also can't represent both (last wins),
    // so without this it would silently tile the un-forced reduced axis and break
    // that reduction. Force the partitioner to cut between them instead.
    if (reduces_width && reduces_height) return std::nullopt;
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
    // A reduction sink is the only place a reduced-axis (cross-core) split may
    // land — its partials reduce through DDR, which is fine for a boundary output
    // but would break ephemerality for an internal reduction (that case must be a
    // subgraph cut instead). Mirrors matmul's sink-only parallel split-K.
    for (auto s : sink_ops)
      if (prob.ops[s].type == OpType::Reduction) { sg.reduction_is_sink_ = true; break; }

    // Set output_K_ from the sink matmul (if any).
    //   MM-only sinks:  output_K_ = op_K(sink_mm) — standard temporal tiling.
    //   Mixed MM+PW sinks: output_K_ = op_K(sink_mm) — the MM sink determines
    //     K; has_pw_sink_ enforcement below ensures nk == 1.
    //   PW-only sinks:  output_K_ stays 1 — no temporal dimension.
    for (auto s : sink_ops) {
      if (prob.ops[s].type == OpType::MatMul) {
        sg.output_K_ = sg.op_K(s);
        sg.sink_mm_op_ = (int64_t)s;
        break;
      }
    }

    // Detect simple MM→PW epilogue pattern (issue #82):
    //   All sinks are PW, and walking backward from PW sinks through PW-only
    //   chains reaches exactly one MM. That MM is the "effective sink" —
    //   its k-loop runs, accumulates into its ephemeral output, then the PW
    //   chain fires once on the completed tile. Per #82 this is always valid.
    //
    //   When detected: output_K_ = effective sink MM's K, and we mark it in
    //   is_sink_op_vec_ so tiling propagation gives it FROM_NK inputs.
    if (sg.has_pw_sink_ && sg.has_matmul_ && sg.output_K_ == 1) {
      size_t found_mm = SIZE_MAX;
      bool valid = true;
      for (auto s : sink_ops) {
        if (prob.ops[s].type != OpType::Pointwise) {
          valid = false; break;  // mixed MM+PW sinks handled above
        }
        // BFS backward through PW-only chain from this PW sink
        std::vector<size_t> stack = {s};
        std::vector<bool> visited(num_ops, false);
        visited[s] = true;
        while (!stack.empty() && valid) {
          size_t op = stack.back(); stack.pop_back();
          for (auto t : prob.ops[op].inputs) {
            int prod = dag.tensor_producer[t];
            if (prod < 0 || !is_in_sg[(size_t)prod]) continue;
            if (visited[(size_t)prod]) continue;
            visited[(size_t)prod] = true;
            if (prob.ops[(size_t)prod].type == OpType::MatMul) {
              if (found_mm != SIZE_MAX && found_mm != (size_t)prod)
                valid = false;  // multiple MMs feed PW chain
              found_mm = (size_t)prod;
            } else {
              stack.push_back((size_t)prod);
            }
          }
        }
      }
      if (valid && found_mm != SIZE_MAX) {
        sg.has_simple_epilogue_ = true;
        sg.output_K_ = sg.op_K(found_mm);
        sg.is_sink_op_vec_[found_mm] = true;
        // Record the ephemeral accumulator tensor
        size_t accum_t = prob.ops[found_mm].output();
        if (is_ephemeral[accum_t])
          sg.ephemeral_accumulators_.push_back(accum_t);
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
        // Broadcast-aware: an input with extent 1 on an axis the output tiles
        // is REUSED across all tiles, not split — so it is FIXED_1 on that axis,
        // not FROM_NT*. Without this a [1,N] broadcast input looks like it has
        // ntw/nth tiles but only one element, and is_valid_tiling rejects it
        // (derived tile count > tensor dim).
        for (auto t : op.inputs) {
          TS th = (prob.tensors[t].width == 1 && prob.tensors[out].width > 1)
                      ? TS::FIXED_1 : out_h;
          TS tv = (prob.tensors[t].height == 1 && prob.tensors[out].height > 1)
                      ? TS::FIXED_1 : out_v;
          assign_or_check(t, th, tv);
        }
      } else if (op.type == OpType::Reduction) {
        // A reduction has ONE input and collapses one axis: the input is read
        // FULL along the reduced axis (FIXED_1) and follows the output tiling
        // along the other. Must NOT take the matmul LHS/RHS path below, which
        // dereferences op.inputs[1] (a single-input reduction has none).
        for (auto t : op.inputs) {
          if (sg.reduced_axis_ == 2)
            assign_or_check(t, out_h, TS::FIXED_1);  // height reduced
          else
            assign_or_check(t, TS::FIXED_1, out_v);  // width reduced (default)
        }
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
  // Candidate caps. No native cap (the competition super-native rule is retired):
  // cube tiles align to the 16 fractal; vector tiles have no cap and no alignment
  // (a large vector tile is a per-core kernel streamed in UB-chunks — the
  // streaming fits memory, not a small tile).
  const bool matmul_910b = sg.has_matmul_;
  const int64_t cand_cap = 0;
  const int64_t cand_align = matmul_910b ? 16 : 1;
  auto valid_candidates = [&](const std::vector<int64_t> &dims) -> std::vector<int64_t> {
    if (dims.empty()) return {1};
    int64_t mx = *std::max_element(dims.begin(), dims.end());
    if (mx <= 0) return {1};
    auto divs = all_divisors(mx);
    std::vector<int64_t> result;
    for (auto c : divs) {
      if (cand_cap > 0 && c > cand_cap) continue;          // super-native (competition)
      if (cand_align > 1 && c % cand_align != 0) continue;  // fractal-aligned (matmul)
      bool ok = true;
      for (auto v : dims) {
        if (c < v && v % c != 0) { ok = false; break; }
      }
      if (ok) result.push_back(c);
    }
    // Cube: pad a sub-16 dim UP to one 16-fractal (the cube is atomic at 16, so
    // a small-batch / GEMV output tiles to a single padded fractal — feasible,
    // not stuck at a sub-16 tile that fails the alignment check). Competition: 1.
    if (result.empty()) result.push_back(matmul_910b ? (int64_t)16 : (int64_t)1);
    return result;
  };

  sg.ws_cand_ = valid_candidates(sg.w_divides_);
  sg.hs_cand_ = valid_candidates(sg.h_divides_);
  // Reduction: the reduced axis cannot be spatially split — the whole row/col
  // is needed to reduce it — so the tile spans the FULL reduced extent (lifting
  // the native cap on that axis, like a cube tile). Only the non-reduced axis
  // is tiled, giving the spatial (per-core) parallelism. 910B-only.
  if (sg.has_reduction_ && sg.reduced_axis_ == 1)
    sg.ws_cand_ = std::vector<int64_t>{std::max(sg.out_W_, sg.reduced_extent_)};
  else if (sg.has_reduction_ && sg.reduced_axis_ == 2)
    sg.hs_cand_ = std::vector<int64_t>{std::max(sg.out_H_, sg.reduced_extent_)};
  // PW-sink subgraphs: force k = output_K_ so nk == 1 (no temporal tiling).
  //   PW-only sinks:  output_K_ == 1 → k == 1 in solution.
  //   Mixed MM+PW sinks: output_K_ == op_K(mm) → k == K in solution
  //     (full reduction in one pass).
  // Exception: simple MM→PW epilogue (#82) — enumerate k candidates
  //   since the MM's k-loop completes before PW fires.
  if (sg.has_pw_sink_ && !sg.has_simple_epilogue_)
    sg.ks_cand_ = std::vector<int64_t>{std::max(sg.output_K_, (int64_t)1)};
  else
    sg.ks_cand_ = valid_candidates(sg.k_divides_);

  // 910B cube: the single-core k-tile is DERIVED per-op (greedy L1-strip fit in
  // derive_exec / cube_peak_l1), NOT searched — so the k search axis collapses
  // to a single sentinel (cfg.k = output_K_). This is design B: per-op k costs
  // no search (vs ~Dk^(m-1) for a searched per-op k). compute_cost overwrites
  // result.config.k with the derived per-core k for display/emit.
  if (matmul_910b)
    sg.ks_cand_ = std::vector<int64_t>{std::max(sg.output_K_, (int64_t)1)};

  // Depth-first topological execution order (the fixed pebbling order). Post-
  // order DFS from each sink over in-subgraph producers: an op is emitted only
  // after all its producers, and a branch is finished before its sibling
  // starts — minimizing simultaneously-live intermediate bands. Deterministic:
  // sinks and producers are visited in topological-position order so the same
  // subgraph always yields the same order (required for the cost cache).
  {
    auto by_topo = [&](size_t a, size_t b) {
      return dag.topo_position(a) < dag.topo_position(b);
    };
    auto sg_producers = [&](size_t op) {
      std::vector<size_t> preds;
      for (auto t : prob.ops[op].inputs) {
        int p = dag.tensor_producer[t];
        if (p >= 0 && is_in_sg[(size_t)p]) preds.push_back((size_t)p);
      }
      std::sort(preds.begin(), preds.end(), by_topo);
      preds.erase(std::unique(preds.begin(), preds.end()), preds.end());
      return preds;
    };
    std::vector<size_t> sinks;
    for (auto i : sg.ops_)
      if (sg.is_sink_op_vec_[i]) sinks.push_back(i);
    std::sort(sinks.begin(), sinks.end(), by_topo);

    std::vector<bool> visited(num_ops, false);
    std::vector<std::pair<size_t, bool>> stack;  // (op, expanded?)
    for (auto root : sinks) {
      if (visited[root]) continue;
      stack.push_back({root, false});
      while (!stack.empty()) {
        auto [op, expanded] = stack.back();
        stack.pop_back();
        if (expanded) {  // all producers already emitted
          sg.dfs_order_.push_back(op);
          continue;
        }
        if (visited[op]) continue;
        visited[op] = true;
        stack.push_back({op, true});  // emit after producers
        auto preds = sg_producers(op);
        // Push in reverse so the smallest-topo producer is processed first.
        for (auto it = preds.rbegin(); it != preds.rend(); ++it)
          if (!visited[*it]) stack.push_back({*it, false});
      }
    }
  }

  return sg;
}

// ============================================================================
// Tiling validity (Replicates `evaluator.cpp` SHAPES_MISALIGNED EXACTLY)
// ============================================================================

bool Subgraph::is_valid_tiling(const TileConfig &cfg) const {
  if (cfg.w <= 0 || cfg.h <= 0 || cfg.k <= 0)
    return false;

  // The tile is a DDR<->L1/UB panel. Cube tiles are 16-fractal aligned; vector
  // tiles have no alignment or cap (a large vector tile is a per-core kernel
  // streamed in UB-chunks — fits_on_chip checks the chunk). The output is NOT
  // bounded by L0c here — that L1->L0 sub-tiling is AutoTileMatmulL0's job.
  const bool matmul_910b = has_matmul_;
  if (matmul_910b) {
    if (cfg.w % 16 != 0 || cfg.h % 16 != 0) return false;  // 16-fractal aligned
  }

  for (int64_t v : w_divides_)
    if (cfg.w < v && v % cfg.w != 0) return false;
  for (int64_t v : h_divides_)
    if (cfg.h < v && v % cfg.h != 0) return false;
  // 910B cube tile spans the full contraction (k = max_K_, accumulated in L0c);
  // the competition's per-op k-divisibility (temporal-tiling correctness) does
  // not apply, and for a chained group max_K_ may exceed a smaller op's K.
  if ((!has_pw_sink_ || has_simple_epilogue_) && !matmul_910b) {
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

  // PW-sink: no temporal tiling allowed, UNLESS simple MM→PW epilogue (#82).
  if (has_pw_sink_ && !has_simple_epilogue_ && nk > 1) return false;

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

  // Multi-role tensors are now modeled explicitly via multi-entry
  // boundary_tensor_info_ (per #70 + #73); divisibility checks above cover
  // shape constraints across all role orientations, and the 2-partial limit
  // is enforced at Subgraph::create. No symbolic-propagation conflict check
  // is needed — the former slow path rejected exactly the multi-role configs
  // #73 now accepts.
  return true;
}


// Greedy per-op k + red-blue pebble peak over the fixed execution order.
// See the header for the model. Intermediate bands are k-independent residents;
// each matmul's boundary operand strip is sized by a per-op k derived to fit the
// headroom the bands leave. Peak = max over steps of (live bands + this step's
// operand strip). Strips count BOUNDARY operands only — an intermediate operand
// is already a live band (the same boundary/ephemeral split the roofline uses).
int64_t Subgraph::derive_exec(const TileConfig &cfg, int64_t sink_K_eff,
                              const FlatSet<size_t> &retained_from_prev,
                              const FlatSet<size_t> &retain_these,
                              std::vector<int64_t> *perop_k_out) const {
  const double f = prob_->double_buffer ? 0.5 : 1.0;
  const double l1 = (double)prob_->l1_capacity * f;
  const auto &order = dfs_order_;

  // Position of each op in the execution order (-1 = not in this subgraph).
  std::vector<int> pos(prob_->num_ops(), -1);
  for (int i = 0; i < (int)order.size(); ++i) pos[order[i]] = i;

  // Live intermediate-band bytes per step. Each ephemeral tensor occupies a
  // [full_width, M-band h] band in L1/Mat from its producer step through its
  // last in-subgraph consumer (the pebble interval). k-INDEPENDENT — bands are
  // the only cross-step residents; output accumulators live in L0c (unbounded
  // at this DDR<->L1 level).
  std::vector<int64_t> band_at(order.size(), 0);
  for (size_t t : ephemeral_) {
    int pr = dag_->tensor_producer[t];
    int prod = (pr >= 0 && pos[pr] >= 0) ? pos[pr] : 0;
    int last = prod;
    for (auto c : dag_->tensor_consumers[t])
      if (pos[c] >= 0) last = std::max(last, pos[c]);
    int64_t h = std::min(cfg.h, prob_->tensors[t].height);
    int64_t bytes = prob_->tensors[t].width * h * dtype_bytes(prob_->tensors[t].dtype);
    for (int s = prod; s <= last; ++s) band_at[s] += bytes;
  }

  // Retained (coupling-layer) tensors are full-resident across the subgraph.
  // Disabled for 910B (cross-subgraph data routes DDR); handled defensively.
  int64_t base = 0;
  for (auto t : retained_from_prev) base += prob_->tensors[t].size_bytes();
  for (auto t : retain_these)
    if (!retained_from_prev.count(t)) base += prob_->tensors[t].size_bytes();

  if (perop_k_out) perop_k_out->assign(prob_->num_ops(), 0);

  int64_t peak = 0;
  for (int s = 0; s < (int)order.size(); ++s) {
    const size_t opi = order[s];
    const Op &op = prob_->ops[opi];
    const int64_t bands = base + band_at[s];
    if (op.type != OpType::MatMul) {
      // Non-matmul in a cube subgraph shouldn't occur (unit-homogeneity); count
      // its boundary-input [w,h] tiles defensively so the sweep stays sound.
      int64_t strip = 0;
      for (auto in : op.inputs)
        if (!ephemeral_.count(in))
          strip += std::min(cfg.w, prob_->tensors[in].width) *
                   std::min(cfg.h, prob_->tensors[in].height) *
                   dtype_bytes(prob_->tensors[in].dtype);
      peak = std::max(peak, bands + strip);
      continue;
    }
    const size_t lhs = op.inputs[0], rhs = op.inputs[1];
    const int64_t M_o = prob_->tensors[op.output()].height;
    const int64_t N_o = prob_->tensors[op.output()].width;
    const int64_t h = std::min(cfg.h, M_o);
    const int64_t w = std::min(cfg.w, N_o);
    const int64_t K_eff = is_sink_op_vec_[opi] ? sink_K_eff : op_K(opi);
    // Per unit of k, the boundary operand strip costs lhs_bytes*h (LHS [k,h]) +
    // rhs_bytes*w (RHS [w,k]); an intermediate operand contributes 0 here (it is
    // a band). per_unit==0 => both operands are bands, no DDR strip to size.
    const int64_t lhs_b = ephemeral_.count(lhs) ? 0 : dtype_bytes(prob_->tensors[lhs].dtype);
    const int64_t rhs_b = ephemeral_.count(rhs) ? 0 : dtype_bytes(prob_->tensors[rhs].dtype);
    const int64_t per_unit = lhs_b * h + rhs_b * w;
    int64_t kk;
    if (per_unit == 0) {
      kk = K_eff;  // no boundary strip; full K in one accumulation
    } else {
      const double headroom = l1 - (double)bands;
      if (headroom <= 0) return INT64_MAX;  // bands alone overflow L1
      int64_t max_kk = (int64_t)(headroom / (double)per_unit);
      max_kk = (max_kk / 16) * 16;          // 16-fractal aligned
      if (max_kk < 16) return INT64_MAX;     // not even one fractal strip fits
      if (max_kk > K_eff) max_kk = K_eff;
      kk = 0;  // largest 16-aligned divisor of K_eff not exceeding max_kk
      for (int64_t d = std::min(max_kk, K_eff); d >= 16; d -= 16)
        if (K_eff % d == 0) { kk = d; break; }
      if (kk == 0) return INT64_MAX;
    }
    if (perop_k_out) (*perop_k_out)[opi] = kk;
    peak = std::max(peak, bands + per_unit * kk);
  }
  return peak;
}

int64_t Subgraph::cube_peak_l1(const TileConfig &cfg,
                               std::vector<int64_t> *perop_k) const {
  return derive_exec(cfg, output_K_, {}, {}, perop_k);
}

// Vector (UB) pebble peak — see the header. Same interval-overlap sweep as the
// cube, over the UB pool: live ephemeral bands + the transient boundary tiles of
// the running op. The matmul band bug transposed to vector — softmax's e=[W,h]
// row band is ephemeral and was uncounted by the old static boundary-only sum.
int64_t Subgraph::vector_peak_ub(const TileConfig &cfg,
                                 const FlatSet<size_t> &retained_from_prev,
                                 const FlatSet<size_t> &retain_these,
                                 int64_t reduce_chunk, int stream_axis) const {
  const auto &order = dfs_order_;
  std::vector<int> pos(prob_->num_ops(), -1);
  for (int i = 0; i < (int)order.size(); ++i) pos[order[i]] = i;

  // Tile footprint: the reduced axis is coupled to its full extent (cfg forces
  // ws_/hs_cand there), so min(cfg,dim) gives full-extent on the reduced axis and
  // the spatial tile on the non-reduced one; a reduction output ([·,1]) collapses.
  // Single-core streaming caps the stream axis at reduce_chunk (the band held on
  // one core per chunk). stream_axis defaults to the reduced axis (reduction
  // online accumulation); a pure pointwise passes an explicit axis to stream.
  const int ax = stream_axis ? stream_axis : reduced_axis_;
  auto tile_bytes = [&](size_t t) -> int64_t {
    int64_t tw = std::min(cfg.w, prob_->tensors[t].width);
    int64_t th = std::min(cfg.h, prob_->tensors[t].height);
    if (ax == 1) tw = std::min(tw, reduce_chunk);
    else if (ax == 2) th = std::min(th, reduce_chunk);
    return tw * th * dtype_bytes(prob_->tensors[t].dtype);
  };

  // Ephemeral bands resident across [producer, last in-subgraph consumer].
  std::vector<int64_t> band_at(order.size(), 0);
  for (size_t t : ephemeral_) {
    int pr = dag_->tensor_producer[t];
    int prod = (pr >= 0 && pos[pr] >= 0) ? pos[pr] : 0;
    int last = prod;
    for (auto c : dag_->tensor_consumers[t])
      if (pos[c] >= 0) last = std::max(last, pos[c]);
    const int64_t b = tile_bytes(t);
    for (int s = prod; s <= last; ++s) band_at[s] += b;
  }

  // Retained (coupling) tensors resident across the subgraph (disabled on 910B).
  int64_t base = 0;
  for (auto t : retained_from_prev) base += prob_->tensors[t].size_bytes();
  for (auto t : retain_these)
    if (!retained_from_prev.count(t)) base += prob_->tensors[t].size_bytes();

  int64_t peak = 0;
  for (int s = 0; s < (int)order.size(); ++s) {
    const Op &op = prob_->ops[order[s]];
    int64_t transient = 0;
    for (auto in : op.inputs)  // boundary input tiles (ephemerals are bands)
      if (!ephemeral_.count(in) && !retained_from_prev.count(in) && !retain_these.count(in))
        transient += tile_bytes(in);
    const size_t o = op.output();  // boundary output tile (resident while written)
    if (!ephemeral_.count(o) && !retain_these.count(o)) transient += tile_bytes(o);
    peak = std::max(peak, base + band_at[s] + transient);
  }
  return peak;
}

// Derive the single-core k-stream of a vector subgraph — the analog of the
// matmul per-op seq-k. Materialize when the whole tile fits UB; else stream the
// largest UB-fitting chunk along the coupled reduced axis (reduction) or the
// larger tile axis (pointwise). Peak is monotone in the chunk, so binary-search.
Subgraph::VecStream Subgraph::vector_stream(const TileConfig &cfg,
                                            const FlatSet<size_t> &retained_from_prev,
                                            const FlatSet<size_t> &retain_these) const {
  const double f = prob_->double_buffer ? 0.5 : 1.0;
  const int64_t budget = (int64_t)((double)prob_->vec_capacity * f);
  if (vector_peak_ub(cfg, retained_from_prev, retain_these) <= budget)
    return {0, INT64_MAX};  // materialized — no sub-streaming

  int stream_axis;
  int64_t ext, min_chunk;
  if (has_reduction_) {  // reduction: stream the coupled reduced axis, min 16-granule
    stream_axis = reduced_axis_;
    ext = (reduced_axis_ == 1) ? std::max(out_W_, reduced_extent_)
                               : std::max(out_H_, reduced_extent_);
    min_chunk = 16;
  } else {  // pointwise: stream the larger tile axis (shrinks the footprint most)
    stream_axis = (cfg.w >= cfg.h) ? 1 : 2;
    ext = (stream_axis == 1) ? std::min(cfg.w, out_W_) : std::min(cfg.h, out_H_);
    min_chunk = 1;
  }
  int64_t lo = min_chunk, hi = std::max(ext, min_chunk), best = 0;
  while (lo <= hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (vector_peak_ub(cfg, retained_from_prev, retain_these, mid, stream_axis) <= budget) {
      best = mid; lo = mid + 1;
    } else hi = mid - 1;
  }
  return {best >= min_chunk ? stream_axis : 0, best};  // chunk 0 => infeasible
}

// 910B per-core two-pool feasibility (byte-based). Each core runs one tile, so
// the per-tile slice footprint IS the per-core footprint. Fork on cube/vector.
bool Subgraph::fits_on_chip(const TileConfig &cfg,
                            const FlatSet<size_t> &retained_from_prev,
                            const FlatSet<size_t> &retain_these) const {
  const bool cube = has_matmul_;
  // Cube: dynamic peak L1 over the execution order (live intermediate bands +
  // per-op-k-sized operand strips), NOT a static sum of all operand strips. The
  // sum both over-counted co-resident inputs and under-counted intermediate
  // bands that physically sit in L1/Mat between matmuls; derive_exec is the
  // red-blue pebble peak that the cost model's ephemeral-on-chip assumption
  // implies. Infeasible iff no per-op k assignment keeps the peak under L1.
  if (cube)
    return derive_exec(cfg, output_K_, retained_from_prev, retain_these, nullptr) !=
           INT64_MAX;

  // Vector: feasible iff the subgraph materializes OR streams to fit UB.
  return vector_stream(cfg, retained_from_prev, retain_these).chunk > 0;
}

bool Subgraph::is_feasible(const TileConfig &cfg,
                           const FlatSet<size_t> &retained_from_prev,
                           const FlatSet<size_t> &retain_these) const {
  return is_valid_tiling(cfg) &&
         fits_on_chip(cfg, retained_from_prev, retain_these);
}

// ============================================================================
// Cost computation
// ============================================================================

CostResult Subgraph::compute_cost(const TileConfig &cfg,
                                  const FlatSet<size_t> &retained_from_prev,
                                  const FlatSet<size_t> &retain_these) const {
  CostResult result;
  result.config = cfg;

  if (!is_valid_tiling(cfg) || !fits_on_chip(cfg, retained_from_prev, retain_these))
    return result;
  result.feasible = true;

  const double inv_B = 1.0 / (double)prob_->slow_memory_bandwidth;
  const int num_tw = std::max((int)(out_W_ / cfg.w), 1);
  const int num_th = std::max((int)(out_H_ / cfg.h), 1);
  const int num_tiles = num_tw * num_th;
  result.num_spatial_tiles = num_tiles;
  result.num_k_passes = has_matmul_ ? std::max((int)(output_K_ / cfg.k), 1) : 1;

  // 910B parallel-core roofline — the only cost model (the competition
  // single-context model was removed). Compute parallelizes across the unit's
  // cores (spatial tiles + split-K = independent work units); DDR is a SHARED
  // floor (one HBM) that does not divide.
  const int n_cores = has_matmul_ ? prob_->num_cube_cores : prob_->num_vector_cores;
  {
    if (has_matmul_) {
      // Matmul (cube). Compute is the full M*N*K work, INVARIANT to the tile:
      // 16-aligned cube tiles have no padding (the op_scale native-128 padding
      // is a single-level competition artifact that belongs at the L0 fractal,
      // not at our DDR->L1 level). Sum each matmul's own work scaled by ITS
      // output (#native tiles) — for a CHAINED group the intermediate has a
      // different shape than the sink, so a single sink-scaled basis would
      // mis-count it. Single matmul: output == sink, identical to before.
      // In-subgraph produced/consumed sets: an operand is ephemeral (on-chip) iff
      // produced within the subgraph; an op is a boundary-output op iff its output
      // is not consumed within the subgraph.
      FlatSet<size_t> produced, consumed;
      for (auto i : ops_) {
        produced.insert(prob_->ops[i].output());
        for (auto t : prob_->ops[i].inputs) consumed.insert(t);
      }
      double total_compute = 0.0;
      for (auto i : ops_) {
        if (prob_->ops[i].type != OpType::MatMul) continue;
        size_t o = prob_->ops[i].output();
        // RECOMPUTE under the M-partition: an INTERMEDIATE matmul's output is a
        // sink left-operand band, shared by all sink COLUMN-tiles in the same
        // M-band. When the sink fills the cores over M alone (num_tw==1) the band
        // is produced once — truly ephemeral. When M can't fill the cores and the
        // sink tiles over N (num_tw>1), each owning core recomputes the band, so
        // the intermediate compute is paid num_tw times. (The alternative — share
        // the band across cores via DDR — is just splitting the subgraph, costed
        // separately; the partitioner picks the cheaper.) Boundary/sink outputs
        // are produced once: factor 1.
        const double recompute = consumed.count(o) ? (double)num_tw : 1.0;
        if (prob_->cube_compute_cost > 0) {
          // 910B machine model: count the WHOLE 16x16x16 cube fractals the matmul
          // implies = ceil(out_W/16)*ceil(out_H/16)*ceil(K/16), times the per-
          // fractal cube cost. CEIL (not floor) so a sub-16 dim pads up to one
          // fractal — a small-batch GEMV (M<16) still costs a full fractal row,
          // matching the cube's atomic 16-granularity, instead of being dropped.
          const int64_t Km = prob_->tensors[prob_->ops[i].inputs[0]].width;  // contraction
          const double fractals = (double)((prob_->tensors[o].width + 15) / 16) *
                                  (double)((prob_->tensors[o].height + 15) / 16) *
                                  (double)((Km + 15) / 16);
          total_compute += fractals * (double)prob_->cube_compute_cost * recompute;
        } else {
          // Legacy: per-op base_cost interpreted as per-native-128-tile cost.
          total_compute += (double)prob_->ops[i].base_cost *
                           ((double)prob_->tensors[o].width / prob_->native_w) *
                           ((double)prob_->tensors[o].height / prob_->native_h);
        }
      }
      // --- BSP/superstep matmul roofline (910B; no shared L2) ----------------
      // Parallelize the (possibly fused-chain) subgraph over the SHARED output
      // dim M: each core owns a sink-output band and computes its own
      // intermediate slices ON-CHIP. Consequences:
      //   * DDR counts only (a) boundary/graph-input operands and (b) the sink
      //     output store. Intermediate tensors are EPHEMERAL (zero DDR) — a
      //     deeper matmul's partials are computed independently per output tile,
      //     never crossing cores, so they need no barrier.
      //   * The ONLY barrier is a sink split-K merge. With no shared L2, the S
      //     partials reduce through DDR strictly AFTER compute, so it is charged
      //     ADDITIVELY (a BSP superstep), never hidden under the streaming
      //     roofline. Intermediate k-split is disallowed: it would round-trip the
      //     intermediate through DDR, i.e. be no better than splitting the
      //     subgraph there (which also frees independent tiling) — so any k-split
      //     lands on the subgraph output only.
      const double M = (double)out_H_, N = (double)out_W_;
      auto sat = [&](double cores) { return std::max(1.0, (double)n_cores / cores); };
      // (produced/consumed sets built above for the recompute factor are reused.)
      // Sink output store = sum over boundary outputs, byte-based (intermediates
      // are is_internally_produced and excluded).
      double out_store = 0.0;
      for (const auto& info : boundary_tensor_info_)
        if (info.is_boundary_out)
          out_store += (double)info.full_size * dtype_bytes(prob_->tensors[info.id].dtype);
      // Operand reload: per matmul op, boundary operands only. The distribution-
      // aware MNK*(1/w + 1/h) splits into a left-operand (1/w) and right-operand
      // (1/h) term; a non-sink op tiles its intermediate FULL-width (w_i = N_i)
      // under the M-partition, so only its M-band height h is tiled.
      double reload = 0.0;
      for (auto i : ops_) {
        if (prob_->ops[i].type != OpType::MatMul) continue;
        const size_t lhs = prob_->ops[i].inputs[0];
        const size_t rhs = prob_->ops[i].inputs[1];
        const size_t o   = prob_->ops[i].output();
        const double N_i = (double)prob_->tensors[o].width;
        const double M_i = (double)prob_->tensors[o].height;
        const double K_i = (double)prob_->tensors[lhs].width;       // contraction
        const bool is_boundary_op = !consumed.count(o);
        const double w_i = is_boundary_op ? std::min((double)cfg.w, N_i) : N_i;
        const double h_i = std::min((double)cfg.h, M_i);            // shared M-band
        if (!produced.count(lhs))   // left operand [K_i, M_i] streamed from DDR
          reload += M_i * N_i * K_i / w_i * dtype_bytes(prob_->tensors[lhs].dtype);
        if (!produced.count(rhs))   // right operand [N_i, K_i] streamed from DDR
          reload += M_i * N_i * K_i / h_i * dtype_bytes(prob_->tensors[rhs].dtype);
      }
      // S=1 (spatial-only): no cross-core reduction, so the output store streams
      // out as tiles complete (overlaps compute) — classic roofline, no barrier.
      const double eff1 = (double)std::min<int64_t>(num_tiles, n_cores);
      const double ddr1 = (reload + out_store) * inv_B * sat(eff1);
      const double lat1 = std::max(total_compute / eff1, ddr1);
      // Sink split-K: gang up to output_K_/16 partials per sink tile to fill the
      // cores. Streaming phase (operand reload overlaps partial-compute) is a
      // roofline; the S partials then reduce through DDR AFTER compute — an
      // ADDITIVE barrier S*out_store*inv_B (the BSP superstep). output_K_ is the
      // sink matmul's contraction (NOT max_K_, which may be a deeper stage's K).
      const int64_t kfrac = std::max<int64_t>(1, output_K_ / 16);
      const int64_t cores_used = std::min<int64_t>(n_cores, (int64_t)num_tiles * kfrac);
      const int64_t S = std::max<int64_t>(1, (cores_used + num_tiles - 1) / num_tiles);
      const double effS = (double)cores_used;
      const double stream_ddrS = reload * inv_B * sat(effS);
      const double streamS = std::max(total_compute / effS, stream_ddrS);
      const double merge_tail = (double)S * out_store * inv_B * sat(effS);  // BSP barrier
      const double latS = streamS + merge_tail;
      if (S > 1 && latS < lat1) {  // sink split-K only if it helps
        result.latency = latS;
        result.parallel_split = (int)S;
        result.cores_used = (int)cores_used;
        result.compute_bound = (total_compute / effS) >= stream_ddrS;  // streaming phase
        result.ddr_traffic = stream_ddrS + merge_tail;
        // Displayed per-core k composes the two k-levers: the greedy single-core
        // L1-fit k (derive_exec, a divisor of output_K_), capped by the per-core
        // split-K fractal share ceil(output_K_/16 / S)*16. Largest divisor of
        // output_K_ not exceeding both -> k cleanly divides the contraction.
        std::vector<int64_t> pk;
        derive_exec(cfg, output_K_, retained_from_prev, retain_these, &pk);
        const int64_t l1_k = (sink_mm_op_ >= 0 && pk[sink_mm_op_] > 0)
                                 ? pk[sink_mm_op_] : output_K_;
        const int64_t share_k = ((kfrac + S - 1) / S) * 16;
        int64_t per_core_k = 16;
        for (int64_t d = std::min({l1_k, share_k, output_K_}); d >= 16; d -= 16)
          if (output_K_ % d == 0) { per_core_k = d; break; }
        result.config.k = per_core_k;
      } else {
        result.latency = lat1;
        result.parallel_split = 1;
        result.cores_used = (int)eff1;
        result.compute_bound = (total_compute / eff1) >= ddr1;
        result.ddr_traffic = ddr1;
        // Spatial-only (S=1): displayed k = the greedy single-core L1-fit k.
        std::vector<int64_t> pk;
        derive_exec(cfg, output_K_, retained_from_prev, retain_these, &pk);
        result.config.k = (sink_mm_op_ >= 0 && pk[sink_mm_op_] > 0)
                              ? pk[sink_mm_op_] : cfg.k;
      }
    } else {
      // Vector (PW/Reduction): compute parallelizes over spatial tiles; DDR is
      // the shared full-tensor floor (no cross-core reload to model here). The
      // element-wise work is INVARIANT to tiling (each element is touched once),
      // so use the summed base_cost — NOT num_tiles*op_scale, whose native-128
      // padding would inflate compute for the small sub-128 tiles we need to fill
      // the cores, wrongly favouring fewer tiles / fewer cores. (Same fix as the
      // matmul path; applies to pointwise and reduction alike.)
      double total_compute = 0.0;
      for (auto i : ops_) {
        if (prob_->vector_compute_cost > 0) {
          // 910B machine model: the vector unit processes vector_lanes elements
          // per SIMD step, so compute = ceil(#elements / lanes) * per-step cost.
          // #elements = the op's largest tensor (the [H,W] grid). lanes is a
          // machine parameter (SIMD width) to calibrate; default 1 (per-element).
          int64_t elems = (int64_t)prob_->tensors[prob_->ops[i].output()].width *
                          prob_->tensors[prob_->ops[i].output()].height;
          for (auto t : prob_->ops[i].inputs)
            elems = std::max(elems, (int64_t)prob_->tensors[t].width * prob_->tensors[t].height);
          const int64_t lanes = std::max<int64_t>(1, prob_->vector_lanes);
          const int64_t steps = (elems + lanes - 1) / lanes;
          total_compute += (double)steps * (double)prob_->vector_compute_cost;
        } else {
          total_compute += (double)prob_->ops[i].base_cost;  // legacy
        }
      }
      const double eff = (double)std::min<int64_t>(num_tiles, n_cores);
      double total_io = 0.0;  // byte-based: each boundary tensor in its own dtype
      for (const auto& info : boundary_tensor_info_) {
        const double bytes = (double)info.full_size * dtype_bytes(prob_->tensors[info.id].dtype);
        if (!retained_from_prev.count(info.id) && !info.is_internally_produced)
          total_io += bytes * inv_B;
        if (info.is_boundary_out && !retain_these.count(info.id))
          total_io += bytes * inv_B;
      }
      // Step 2b — single-core streaming recompute (flash 2-pass). When this
      // config cannot materialize the reduced band (UB overflow), the feasible
      // schedule STREAMS the reduced axis and recomputes reused ephemerals past
      // each reduction: the boundary inputs are re-read and the pointwise work
      // repeated, once per pass. N_passes = #reductions + 1 (each reduction is a
      // barrier that needs a fresh streamed pass — softmax: rowmax, then
      // exp+rowsum, then exp+div = 3). Pessimistic upper bound (every op counted
      // each pass); refine toward per-op recompute liveness later.
      if (has_reduction_) {
        const double budget = (double)prob_->vec_capacity * (prob_->double_buffer ? 0.5 : 1.0);
        if (budget > 0.0 && (double)vector_peak_ub(cfg) > budget) {
          int reductions = 0;
          for (auto i : ops_)
            if (prob_->ops[i].type == OpType::Reduction) reductions++;
          const double n_passes = (double)(reductions + 1);
          total_compute *= n_passes;
          total_io *= n_passes;
        }
      }
      // Same DDR bandwidth-saturation factor as the cube: HBM needs many cores'
      // DMA engines to saturate, so a DDR-bound tile on too few vector cores pays.
      const double sat_v = std::max(1.0, (double)n_cores / eff);
      double lat = std::max(total_compute / eff, total_io * sat_v);
      result.parallel_split = 1;
      result.cores_used = (int)eff;
      result.compute_bound = (total_compute / eff) >= total_io * sat_v;
      result.ddr_traffic = total_io;
      // Reduced-axis (cross-core) split — the vector analog of matmul split-K.
      // SINK-ONLY: the S per-core partials reduce across cores through DDR, which
      // is fine for a boundary reduction output but would break ephemerality for
      // an internal reduction (that case is expressed as a subgraph cut, not an
      // in-subgraph split). The split propagates to the pointwise ops feeding the
      // sink along the reduced axis; the single merge is an ADDITIVE BSP barrier.
      if (has_reduction_ && reduction_is_sink_ && num_tiles < n_cores) {
        const int64_t S = (n_cores + num_tiles - 1) / num_tiles;  // cores per spatial tile
        const double effS = (double)std::min<int64_t>(num_tiles * S, n_cores);
        const double sat_vS = std::max(1.0, (double)n_cores / effS);
        const double streamS = std::max(total_compute / effS, total_io * sat_vS);
        // Thin partial: the reduced output ([H,1] for a width-reduction, [1,W]
        // for a height-reduction). (S-1) cross-core merges through DDR.
        const double merge_tail =
            (double)(S - 1) * (double)(reduced_axis_ == 1 ? out_H_ : out_W_) * inv_B * sat_vS;
        const double latS = streamS + merge_tail;
        if (latS < lat) {
          lat = latS;
          result.parallel_split = (int)S;
          result.cores_used = (int)effS;
          result.compute_bound = (total_compute / effS) >= (total_io * sat_vS);
          result.ddr_traffic = total_io * sat_vS + merge_tail;
        }
      }
      result.latency = lat;
    }
    // Per-kernel pipeline fill — the DUAL of the eff core-fill incentive. A
    // tiling produces num_tiles "kernels"; each core runs ceil(num_tiles/n_cores)
    // of them in sequence, paying one fill per pass. eff penalizes too FEW tiles
    // (under-filled cores); this penalizes too MANY (over-tiling), so the optimum
    // sits at ~one kernel per core. Split-K fills a tile WITHIN a pass, so it
    // doesn't add rounds (it's spatial num_tiles that count). Gated; off => legacy.
    if (prob_->kernel_fill_cost > 0) {
      const int64_t rounds = (num_tiles + n_cores - 1) / n_cores;
      result.latency += (double)rounds * (double)prob_->kernel_fill_cost;
    }
  }

  return result;
}

// ============================================================================
// Granularity enumeration
// ============================================================================

CostResult Subgraph::best_cost(const FlatSet<size_t> &retained_from_prev,
                               const FlatSet<size_t> &retain_these) const {
  CostResult best;
  for (int64_t ww : ws_cand_) {
    for (int64_t hh : hs_cand_) {
      for (int64_t kk : ks_cand_) {
        TileConfig cfg{ww, hh, kk};
        if (!is_valid_tiling(cfg)) continue;
        if (!fits_on_chip(cfg, retained_from_prev, retain_these)) continue;
        auto r = compute_cost(cfg, retained_from_prev, retain_these);
        if (!r.feasible) continue;
        bool take;
        if (best.latency == std::numeric_limits<double>::infinity()) {
          take = true;
        } else {
          // Lexicographic tiebreak among equal-latency tiles:
          //   1. lower DDR traffic   — matmul reuse (less reload); flat for PW
          //   2. more cores used     — fill the unit's cores (parallelism)
          //   3. larger tile area    — best vectorization / least per-tile
          //      overhead (avoids the degenerate 1xN / 16x16 picks)
          //   4. larger k            — fewer L1 passes
          const double tol = 1e-6 * std::max(1.0, best.latency);
          const double dtol = 1e-9 * std::max(1.0, best.ddr_traffic);
          const long long ra = (long long)r.config.w * r.config.h;
          const long long ba = (long long)best.config.w * best.config.h;
          if (r.latency < best.latency - tol) {
            take = true;
          } else if (r.latency > best.latency + tol) {
            take = false;
          } else if (r.ddr_traffic < best.ddr_traffic - dtol) {
            take = true;
          } else if (r.ddr_traffic > best.ddr_traffic + dtol) {
            take = false;
          } else if (r.cores_used != best.cores_used) {
            take = r.cores_used > best.cores_used;
          } else if (ra != ba) {
            take = ra > ba;
          } else {
            take = r.config.k > best.config.k;
          }
        }
        if (take) best = r;
      }
    }
  }
  return best;
}