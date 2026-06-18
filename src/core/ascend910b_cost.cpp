#include "core/ascend910b_cost.h"
#include "core/subgraph_structure.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>
#include <tuple>

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

std::optional<Ascend910BCost> Ascend910BCost::create(const Problem &prob, const DAG &dag,
                                         std::vector<size_t> op_indices,
                                         bool allow_mixed) {
  if (op_indices.empty())
    return std::nullopt;

  Ascend910BCost sg;
  sg.prob_ = &prob;
  sg.dag_ = &dag;
  sg.ops_ = std::move(op_indices);

  const size_t num_tensors = prob.num_tensors();
  const size_t num_ops = prob.num_ops();

  std::vector<bool> is_in_sg(num_ops, false);
  std::vector<bool> is_produced(num_tensors, false);
  bool reduces_width = false, reduces_height = false;  // for reduced-axis homogeneity

  for (auto i : sg.ops_) {
    is_in_sg[i] = true;
    { size_t t = prob.ops[i].output();
      is_produced[t] = true; }
    if (prob.ops[i].type == OpType::Pointwise ||
        prob.ops[i].type == OpType::Reduction)
      sg.has_vector_ = true;
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
  // Gated on parallel-core mode (cores>1); the legacy single-core path
  // (cores<=1) does not enforce homogeneity (mixed fusion allowed).
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
    // Cube↔vector fusion is allowed only when the cost model opts in (allow_mixed
    // — set by Ascend910BMixed::create). The base Ascend910BCost is
    // unit-homogeneous: the handoff routes through DDR and the separated
    // two-kernel cost applies. NOTE: this admissibility gate is the ONLY
    // difference between the two models; the cost of a (valid) mixed group is the
    // shared compute_cost mixed branch, reached only when one is actually built.
    if (has_cube && has_vector && !allow_mixed)
      return std::nullopt;                                      // no cube↔vector fusion
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

  // Structural classification — boundary inputs/outputs, ephemerals, and sinks
  // — is computed ONCE by SubgraphStructure, the shared architecture-independent
  // layer. The cost model composes those facts and adds tiling/feasibility/cost
  // on top. (The execution-order DFS stays in the cost layer below: its roots
  // are refined by the #82 epilogue detection, so it is not purely structural.)
  //
  // Rule recap: a tensor produced AND consumed inside is ephemeral (zero memory
  // footprint, zero IO). Whether an external consumer can reach it is NOT the
  // subgraph's concern — that is the partition/solution ephemeral-gap check.
  SubgraphStructure structure(prob, dag, sg.ops_);
  if (!structure.valid())
    return std::nullopt;  // empty op set, or no boundary output
  sg.boundary_inputs_  = structure.boundary_inputs();
  sg.boundary_outputs_ = structure.boundary_outputs();
  sg.ephemeral_        = structure.ephemeral();
  // Local per-tensor ephemeral lookup the tiling code below indexes by tensor id.
  std::vector<bool> is_ephemeral(num_tensors, false);
  for (auto t : structure.ephemeral())
    is_ephemeral[t] = true;

  // Collect PW-produced ephemerals for the granule-fit check in is_valid_tiling
  // (PW has no k-loop, so its output slice must fit one (cfg.w, cfg.h) granule).
  for (auto i : sg.ops_) {
    size_t out_t = prob.ops[i].output();
    if (is_ephemeral[out_t] && prob.ops[i].type == OpType::Pointwise)
      sg.pw_produced_ephemerals_.push_back(out_t);
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
    // Structural sinks come from SubgraphStructure (an op whose output has no
    // in-subgraph consumer). The #82 epilogue detection below may additionally
    // mark an internal MM as an effective sink — a cost-model refinement on top
    // of this structural set.
    sg.is_sink_op_vec_.assign(num_ops, false);
    std::vector<size_t> sink_ops = structure.sinks();
    for (auto s : sink_ops)
      sg.is_sink_op_vec_[s] = true;
    // sink_ops is non-empty: structure.valid() ⇒ ≥1 boundary output ⇒ ≥1 sink.
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

  // Tile-size candidates: divisors of the role-required dims. There is no
  // super-native cap — cube tiles align to the 16-element fractal; vector tiles
  // have no alignment and no cap (a large vector tile is a per-core kernel
  // streamed in UB-chunks — the streaming fits memory, not a small tile).
  const bool matmul_910b = sg.has_matmul_;
  const int64_t cand_align = matmul_910b ? 16 : 1;
  auto valid_candidates = [&](const std::vector<int64_t> &dims) -> std::vector<int64_t> {
    if (dims.empty()) return {1};
    int64_t mx = *std::max_element(dims.begin(), dims.end());
    if (mx <= 0) return {1};
    auto divs = all_divisors(mx);
    std::vector<int64_t> result;
    for (auto c : divs) {
      if (cand_align > 1 && c % cand_align != 0) continue;  // fractal-aligned (cube)
      bool ok = true;
      for (auto v : dims) {
        if (c < v && v % c != 0) { ok = false; break; }
      }
      if (ok) result.push_back(c);
    }
    // Cube: pad a sub-16 dim UP to one 16-fractal (the cube is atomic at 16, so
    // a small-batch / GEMV output tiles to a single padded fractal — feasible,
    // not stuck at a sub-16 tile that fails the alignment check). Vector: 1.
    if (result.empty()) result.push_back(matmul_910b ? (int64_t)16 : (int64_t)1);
    return result;
  };

  sg.ws_cand_ = valid_candidates(sg.w_divides_);
  sg.hs_cand_ = valid_candidates(sg.h_divides_);
  // The unified grid tiles the SINK output, so a spatial tile never needs to
  // exceed the output dim (a wider/taller tile is just "one full tile"). For a
  // CHAINED cube subgraph an intermediate matmul's output width is the NEXT
  // matmul's contraction (a reduction axis, on-chip full-width), but the generic
  // MM-RHS rule adds it to w_set as a spatial W_param — so e.g. C=(A@B)@D with
  // N1 > N2 leaks w = N1 > out_W into ws_cand_. It is cost-neutral (it collapses
  // to num_tw=1 and clamps via min(cfg.w, N_i) in the roofline) but violates the
  // unified-grid contract in the emitted schedule. Bound the cube spatial
  // candidates to the output dim, allowing the 16-fractal pad for sub-16 outputs.
  // Reductions override ws_/hs_cand below to the full reduced extent and are
  // intentionally exempt (the reduced axis spans its whole extent).
  if (sg.has_matmul_) {
    auto clamp_to_output = [](std::vector<int64_t> &cand, int64_t out_dim) {
      const int64_t cap = std::max(out_dim, (int64_t)16);
      std::vector<int64_t> kept;
      for (auto c : cand)
        if (c <= cap) kept.push_back(c);
      if (!kept.empty()) cand.swap(kept);  // keep ≥1 candidate (sink dim always fits)
    };
    clamp_to_output(sg.ws_cand_, sg.out_W_);
    clamp_to_output(sg.hs_cand_, sg.out_H_);
  }
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

bool Ascend910BCost::is_valid_tiling(const TileConfig &cfg) const {
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
  // the per-op k-divisibility rule (temporal-tiling correctness) does not apply
  // here, and for a chained group max_K_ may exceed a smaller op's K.
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
  // is enforced at Ascend910BCost::create. No symbolic-propagation conflict check
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
int64_t Ascend910BCost::derive_exec(const TileConfig &cfg, int64_t sink_K_eff,
                              const FlatSet<size_t> &retained_from_prev,
                              const FlatSet<size_t> &retain_these,
                              std::vector<int64_t> *perop_k_out) const {
  const double f = prob_->double_buffer ? 0.5 : 1.0;
  const double l1 = (double)prob_->l1_capacity * f;
  const auto &order = dfs_order_;

  // Position of each op in the execution order (-1 = not in this subgraph).
  std::vector<int> pos(prob_->num_ops(), -1);
  for (int i = 0; i < (int)order.size(); ++i) pos[order[i]] = i;

  // Live intermediate-band bytes per step. The L1/Mat working set of a cube tile
  // has two charged contributors; the peak below sums only these:
  //
  //   (1) EPHEMERAL intermediates ARE charged. A fused intermediate (T in
  //       C=(A@B)@D) must become fully L1-resident for its consumer matmul to
  //       read it as an operand, so it occupies a [full_width, M-band h] band. We
  //       charge it across the whole interval [producer .. last consumer].
  //       Charging it AT the producer step too is deliberate and conservative:
  //         - T's band routinely EXCEEDS L0c (cube_capacity, 128KB) — e.g. a
  //           [256,256] FP32 band is 256KB > 128KB — so it cannot sit wholly in
  //           the L0c accumulator; it spills into L1 as it is produced.
  //         - Even when it fits L0c, at the producer->consumer transition T has
  //           materialised in L1 while the producer's operand strips may still be
  //           resident, so the two coexist.
  //       "Ephemeral" means no DDR round-trip — NOT zero memory.
  //
  //   (2) The boundary OUTPUT is NEVER charged to L1. On the 910B the L0c
  //       accumulator drains DIRECTLY back to DDR (write-back from L0c, no L1
  //       staging of the result), so a tile's own output never needs an L1 slot.
  //       (An ephemeral output is the consumer's input, charged via (1); a
  //       boundary output goes straight to DDR.) L0c sizing is the L0 sub-tiling
  //       level's job (AutoTileMatmulL0), not this DDR<->L1 model.
  //
  // Bands are k-INDEPENDENT; the per-step boundary-operand strip (sized by the
  // greedy k, below) is added on top of the live bands at each step.
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

int64_t Ascend910BCost::cube_peak_l1(const TileConfig &cfg,
                               std::vector<int64_t> *perop_k) const {
  return derive_exec(cfg, output_K_, {}, {}, perop_k);
}

// Vector (UB) pebble peak — see the header. Same interval-overlap sweep as the
// cube, over the UB pool: live ephemeral bands + the transient boundary tiles of
// the running op. The matmul band bug transposed to vector — softmax's e=[W,h]
// row band is ephemeral and was uncounted by the old static boundary-only sum.
int64_t Ascend910BCost::vector_peak_ub(const TileConfig &cfg,
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
Ascend910BCost::VecStream Ascend910BCost::vector_stream(const TileConfig &cfg,
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
bool Ascend910BCost::fits_on_chip(const TileConfig &cfg,
                            const FlatSet<size_t> &retained_from_prev,
                            const FlatSet<size_t> &retain_these) const {
  // Mixed cube+vector group (only Ascend910BMixed builds one): needs BOTH pools.
  // Shared here so is_feasible() and the mixed compute_cost agree. The base model
  // never admits a mixed group, so it never reaches this branch.
  if (has_matmul_ && has_vector_)
    return mixed_fits_on_chip(cfg, retained_from_prev, retain_these);

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

bool Ascend910BCost::mixed_fits_on_chip(const TileConfig &cfg,
                                  const FlatSet<size_t> & /*retained_from_prev*/,
                                  const FlatSet<size_t> & /*retain_these*/) const {
  // Unified two-pool pebble sweep over the execution order — the same interval-
  // overlap as the homogeneous derive_exec (L1) / vector_peak_ub (UB), but the
  // capacity BOUNDARY switches per op: a cube step is checked against L1, a vector
  // step against UB. Intermediates stay RED resident bands in their OWN unit's
  // pool:
  //   * an ephemeral read by an in-subgraph matmul is an L1 operand band
  //     ([full_width, h-band]) across its liveness — the binary-tree "keep the
  //     sibling's red pebble while computing the other branch" case;
  //   * a vector->vector ephemeral is a UB band ([w,h]).
  // A CROSSING cube<->vector ephemeral is NOT a resident band: its full extent
  // sits in the GM ring (DDR), so it only relieves the on-chip bound. Its per-tile
  // streaming footprint is still charged as a transient at the consuming step (the
  // popped [w,h] tile in UB; the producing matmul's output tile is charged via the
  // strip below). L0c output sizing is deferred to the L0 sub-tiling level
  // (AutoTileMatmulL0), exactly as in the homogeneous cube model — not bounded
  // here. The DDR roundtrip of every crossing is paid in compute_cost, not here.
  const double f = prob_->double_buffer ? 0.5 : 1.0;
  const auto &order = dfs_order_;
  const int n = (int)order.size();
  std::vector<int> pos(prob_->num_ops(), -1);
  for (int i = 0; i < n; ++i) pos[order[i]] = i;
  auto is_cube = [&](size_t op) { return prob_->ops[op].type == OpType::MatMul; };

  // Red bands by same-unit relationship, charged across their liveness interval.
  std::vector<int64_t> l1_band(n, 0), ub_band(n, 0);
  std::vector<bool> is_ub_band(prob_->num_tensors(), false);
  for (size_t t : ephemeral_) {
    const int pr = dag_->tensor_producer[t];
    const int prodpos = (pr >= 0 && pos[pr] >= 0) ? pos[pr] : 0;
    const bool prod_cube = pr >= 0 && is_cube((size_t)pr);
    int last_cube_use = -1, last_vec_use = -1;
    for (auto c : dag_->tensor_consumers[t]) {
      if (pos[c] < 0) continue;
      if (is_cube(c)) last_cube_use = std::max(last_cube_use, pos[c]);
      else            last_vec_use  = std::max(last_vec_use,  pos[c]);
    }
    if (last_cube_use >= 0) {  // L1 operand band (read by an in-subgraph matmul)
      const int64_t b = prob_->tensors[t].width *
                        std::min(cfg.h, prob_->tensors[t].height) *
                        dtype_bytes(prob_->tensors[t].dtype);
      for (int s = prodpos; s <= last_cube_use; ++s) l1_band[s] += b;
    }
    if (!prod_cube && last_vec_use >= 0) {  // vector->vector UB band
      const int64_t b = std::min(cfg.w, prob_->tensors[t].width) *
                        std::min(cfg.h, prob_->tensors[t].height) *
                        dtype_bytes(prob_->tensors[t].dtype);
      for (int s = prodpos; s <= last_vec_use; ++s) ub_band[s] += b;
      is_ub_band[t] = true;
    }
  }

  const double l1 = (double)prob_->l1_capacity  * f;
  const double ub = (double)prob_->vec_capacity * f;
  for (int s = 0; s < n; ++s) {
    const size_t opi = order[s];
    const Op &op = prob_->ops[opi];
    if (is_cube(opi)) {
      // L1: resident operand bands + this matmul's minimum 16-fractal operand
      // strip. An ephemeral operand is itself a band (counted in l1_band, so its
      // strip contribution is 0); larger k is a cost lever, not a feasibility one.
      const size_t lhs = op.inputs[0], rhs = op.inputs[1];
      const int64_t h = std::min(cfg.h, prob_->tensors[op.output()].height);
      const int64_t w = std::min(cfg.w, prob_->tensors[op.output()].width);
      const int64_t lhs_b = ephemeral_.count(lhs) ? 0 : dtype_bytes(prob_->tensors[lhs].dtype);
      const int64_t rhs_b = ephemeral_.count(rhs) ? 0 : dtype_bytes(prob_->tensors[rhs].dtype);
      const double strip16 = ((double)lhs_b * (double)h + (double)rhs_b * (double)w) * 16.0;
      if (prob_->l1_capacity > 0 && (double)l1_band[s] + strip16 > l1)
        return false;
    } else {  // vector step
      // UB: resident vector->vector bands + transient tiles. A boundary input, or
      // a crossing tile popped from the ring, is a transient [w,h] tile; a
      // vector->vector band is already in ub_band (skip to avoid double counting).
      int64_t transient = 0;
      for (auto in : op.inputs)
        if (!is_ub_band[in])
          transient += std::min(cfg.w, prob_->tensors[in].width) *
                       std::min(cfg.h, prob_->tensors[in].height) *
                       dtype_bytes(prob_->tensors[in].dtype);
      const size_t o = op.output();
      if (!is_ub_band[o])
        transient += std::min(cfg.w, prob_->tensors[o].width) *
                     std::min(cfg.h, prob_->tensors[o].height) *
                     dtype_bytes(prob_->tensors[o].dtype);
      if (prob_->vec_capacity > 0 && (double)ub_band[s] + (double)transient > ub)
        return false;
    }
  }
  return true;
}

bool Ascend910BCost::is_feasible(const TileConfig &cfg,
                           const FlatSet<size_t> &retained_from_prev,
                           const FlatSet<size_t> &retain_these) const {
  return is_valid_tiling(cfg) &&
         fits_on_chip(cfg, retained_from_prev, retain_these);
}

// ============================================================================
// Matmul operand reload (BYTES) — shared by the cube cost and the mixed cost
// ============================================================================

double Ascend910BCost::cube_operand_reload(const TileConfig &cfg,
                                           bool matmul_at_output_grid) const {
  // Build the in-subgraph produced/consumed sets (produced ⇒ on-chip ephemeral
  // operand, never reloaded from DDR; consumed ⇒ the op is an intermediate).
  FlatSet<size_t> produced, consumed;
  for (auto i : ops_) {
    produced.insert(prob_->ops[i].output());
    for (auto t : prob_->ops[i].inputs) consumed.insert(t);
  }
  // Distribution-aware reload: the left operand reloads with the N-tiling (1/w),
  // the right operand with the M-tiling (1/h); deduped per (tensor, role,
  // boundary?) so a shared operand in the same role is charged once. A consumed
  // (chained-intermediate) matmul tiles its output full-width (w_i = N_i) UNLESS
  // matmul_at_output_grid forces the output grid — the mixed feed-forward case,
  // where the matmul output is consumed elementwise by the vector stage and so is
  // tiled at cfg.w like a boundary output.
  double reload = 0.0;
  std::set<std::tuple<size_t, int, bool>> counted;  // (tensor, 0=LHS/1=RHS, is_boundary_op)
  for (auto i : ops_) {
    if (prob_->ops[i].type != OpType::MatMul) continue;
    const size_t lhs = prob_->ops[i].inputs[0];
    const size_t rhs = prob_->ops[i].inputs[1];
    const size_t o   = prob_->ops[i].output();
    const double N_i = (double)prob_->tensors[o].width;
    const double M_i = (double)prob_->tensors[o].height;
    const double K_i = (double)prob_->tensors[lhs].width;       // contraction
    const bool is_boundary_op = matmul_at_output_grid || !consumed.count(o);
    const double w_i = is_boundary_op ? std::min((double)cfg.w, N_i) : N_i;
    const double h_i = std::min((double)cfg.h, M_i);            // shared M-band
    if (!produced.count(lhs) && counted.emplace(lhs, 0, is_boundary_op).second)
      reload += M_i * N_i * K_i / w_i * dtype_bytes(prob_->tensors[lhs].dtype);
    if (!produced.count(rhs) && counted.emplace(rhs, 1, is_boundary_op).second)
      reload += M_i * N_i * K_i / h_i * dtype_bytes(prob_->tensors[rhs].dtype);
  }
  return reload;
}

// ============================================================================
// Cost computation
// ============================================================================

CostResult Ascend910BCost::compute_cost(const TileConfig &cfg,
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
      // In-subgraph consumed set: an op is a boundary-output op iff its output is
      // not consumed within the subgraph (drives the recompute factor below).
      FlatSet<size_t> consumed;
      for (auto i : ops_)
        for (auto t : prob_->ops[i].inputs) consumed.insert(t);
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
      // Matmul boundary-operand reload — distribution-aware MNK*(1/w + 1/h),
      // deduped per (tensor, role). See cube_operand_reload(); the cube path
      // treats a consumed (chained) intermediate matmul as a full-width band.
      const double reload = cube_operand_reload(cfg);
      // S=1 (spatial-only): no cross-core reduction, so the output store streams
      // out as tiles complete (overlaps compute) — classic roofline, no barrier.
      const double eff1 = (double)std::min<int64_t>(num_tiles, n_cores);
      const double ddr1 = (reload + out_store) * inv_B * sat(eff1);
      const double lat1 = std::max(total_compute / eff1, ddr1);
      // Sink split-K: split the sink contraction into S per-tile partials to
      // recruit idle cores. The streaming phase (operand reload overlaps partial
      // compute) is a roofline; the S partials then reduce through DDR AFTER
      // compute — an ADDITIVE barrier S*out_store*inv_B (a BSP superstep).
      //
      // S is a FIRST-CLASS design axis (like w,h): streamS DECREASES with S (more
      // cores) while the merge barrier may GROW with S (see below — the serial DDR
      // reduction when the target has no SetAtomicAdd), so latS(S) trades
      // parallelism against the merge cost. ENUMERATE S and take the min — the
      // optimum is interior when the merge is serial, and max-fill when
      // SetAtomicAdd folds the reduction into the write-back.
      // Useful range: S <= kfrac (>=1 fractal/partial) and S <= ceil(n_cores/
      // num_tiles) (beyond that effS caps at n_cores and a larger S only adds
      // merge). output_K_ is the sink matmul's contraction (NOT max_K_, a deeper
      // stage's K). Bounded by n_cores.
      const int64_t kfrac = std::max<int64_t>(1, output_K_ / 16);
      const int64_t S_max =
          std::min<int64_t>(kfrac, (n_cores + num_tiles - 1) / num_tiles);
      double best_latS = std::numeric_limits<double>::infinity();
      int64_t best_S = 1;
      double best_effS = eff1, best_stream_ddr = 0.0, best_merge = 0.0;
      for (int64_t S = 2; S <= S_max; ++S) {
        const double effS = (double)std::min<int64_t>((int64_t)num_tiles * S, n_cores);
        const double stream_ddrS = reload * inv_B * sat(effS);
        const double streamS = std::max(total_compute / effS, stream_ddrS);
        // Merge barrier. Two regimes, selected by the target's ddr_atomic_add
        // capability (see Problem::ddr_atomic_add):
        //   (1) ALWAYS — write the S partials to DDR (parallel across effS cores).
        //   (2) WITHOUT SetAtomicAdd — read the partials back and accumulate
        //       serially per output tile: the S adds target ONE DDR accumulator
        //       per tile, so they SERIALIZE per tile (parallelism is across the
        //       num_tiles accumulators, NOT across S). It gets sat(num_tiles) and
        //       grows ~linearly with S (no cancellation) — the real price of a
        //       DDR reduction, which makes the optimum S interior.
        //   WITH SetAtomicAdd — the partials accumulate in DDR during write-back,
        //       so (2) vanishes; the merge is just (1) (sat-discounted → cancels
        //       below full-fill), so split-K stays cheap and max-fill is optimal.
        double merge = (double)S * out_store * inv_B * sat(effS);                     // (1) write partials
        if (!prob_->ddr_atomic_add)
          merge += (double)(S + 1) * out_store * inv_B * sat((double)num_tiles);      // (2) serial read-back + sum
        const double latS = streamS + merge;
        if (latS < best_latS) {
          best_latS = latS; best_S = S; best_effS = effS;
          best_stream_ddr = stream_ddrS; best_merge = merge;
        }
      }
      if (best_S > 1 && best_latS < lat1) {  // sink split-K only if it helps
        result.latency = best_latS;
        result.parallel_split = (int)best_S;
        result.cores_used = (int)best_effS;
        result.compute_bound = (total_compute / best_effS) >= best_stream_ddr;  // streaming phase
        result.ddr_traffic = best_stream_ddr + best_merge;
        // Displayed per-core k composes the two k-levers: the greedy single-core
        // L1-fit k (derive_exec, a divisor of output_K_), capped by the per-core
        // split-K fractal share ceil(kfrac / S)*16. Largest divisor of output_K_
        // not exceeding both -> k cleanly divides the contraction.
        std::vector<int64_t> pk;
        derive_exec(cfg, output_K_, retained_from_prev, retain_these, &pk);
        const int64_t l1_k = (sink_mm_op_ >= 0 && pk[sink_mm_op_] > 0)
                                 ? pk[sink_mm_op_] : output_K_;
        const int64_t share_k = ((kfrac + best_S - 1) / best_S) * 16;
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
      // Reduced-axis (cross-core) split — the vector analog of the cube split-K.
      // SINK-ONLY: the S per-core partials reduce across cores through DDR (fine
      // for a boundary reduction output; an internal reduction split is a subgraph
      // cut, not an in-subgraph split). Mirrors the cube (§4.2): ENUMERATE S (the
      // optimum is interior when the merge is serial), and the merge has the SAME
      // two regimes via Problem::ddr_atomic_add. The reduced partial is thin
      // ([H,1] for a width-reduction, [1,W] for a height-reduction) -> red_dim.
      if (has_reduction_ && reduction_is_sink_ && num_tiles < n_cores) {
        const double red_dim = (double)(reduced_axis_ == 1 ? out_H_ : out_W_);
        const double sat_acc = std::max(1.0, (double)n_cores / (double)num_tiles);
        const int64_t S_max = (n_cores + num_tiles - 1) / num_tiles;
        for (int64_t S = 2; S <= S_max; ++S) {
          const double effS = (double)std::min<int64_t>(num_tiles * S, n_cores);
          const double sat_vS = std::max(1.0, (double)n_cores / effS);
          const double streamS = std::max(total_compute / effS, total_io * sat_vS);
          // (1) write/atomic-add the S thin partials (parallel; cancels below full-fill).
          double merge = (double)(S - 1) * red_dim * inv_B * sat_vS;
          // (2) WITHOUT SetAtomicAdd: read the partials back and sum serially per
          //     tile (one accumulator/tile -> sat(num_tiles), grows ∝ S).
          if (!prob_->ddr_atomic_add)
            merge += (double)(S + 1) * red_dim * inv_B * sat_acc;
          const double latS = streamS + merge;
          if (latS < lat) {
            lat = latS;
            result.parallel_split = (int)S;
            result.cores_used = (int)effS;
            result.compute_bound = (total_compute / effS) >= (total_io * sat_vS);
            result.ddr_traffic = total_io * sat_vS + merge;
          }
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
// Ascend910BMixed — the third subgraph type: a fused cube+vector mixed kernel.
// Cube-only and vector-only groups delegate to the (unchanged) base cost; only
// the mixed type is added here.
// ============================================================================

CostResult Ascend910BMixed::compute_cost(const TileConfig &cfg,
                                         const FlatSet<size_t> &retained_from_prev,
                                         const FlatSet<size_t> &retain_these) const {
  // The two homogeneous types are identical to the base model.
  if (!(has_matmul_ && has_vector_))
    return Ascend910BCost::compute_cost(cfg, retained_from_prev, retain_these);

  // ===== MIXED cube+vector kernel — 910B DDR-streamed, latency hidden =====
  // Cube ops run on the cube pool and vector ops on the vector pool CONCURRENTLY.
  // A cube↔vector intermediate cannot stay on chip (910B has no direct Acc→Vec
  // pipe), so it ROUND-TRIPS DDR — written by the producing unit, read by the
  // consuming unit (2x). That traffic is unavoidable, but the two stages OVERLAP
  // (the cube streams output tiles into DDR while the vector cores consume
  // already-written tiles), so the roundtrip LATENCY is hidden behind the slower
  // stage. The kernel tiles for UNITS (1 cube + 2 vector):
  //   lat = fill + max(cube_compute/eff_units, vector_compute/(2·eff_units), ddr·inv_B)
  // Compare the SEPARATED two-kernel cost (fusion saves the overlap + one fill;
  // the DDR total is the same on 910B). A future 950 makes the handoff direct
  // (the crossing intermediate avoids DDR) — same formula, cheaper ddr term.
  CostResult result;
  result.config = cfg;
  // Mixed kernels need BOTH on-chip pools (L1/L0c for the cube stage, UB for the
  // vector stage) — fits_on_chip dispatches to mixed_fits_on_chip here. A large
  // shared tile that overflows UB is infeasible to fuse even when the separate
  // kernels each fit their one pool.
  if (!is_valid_tiling(cfg) || !fits_on_chip(cfg, retained_from_prev, retain_these))
    return result;
  result.feasible = true;

  const double inv_B = 1.0 / (double)prob_->slow_memory_bandwidth;
  const int num_tw = std::max((int)(out_W_ / cfg.w), 1);
  const int num_th = std::max((int)(out_H_ / cfg.h), 1);
  const int num_tiles = num_tw * num_th;
  result.num_spatial_tiles = num_tiles;
  result.num_k_passes = std::max((int)(output_K_ / cfg.k), 1);

  double cube_compute = 0.0, vector_compute = 0.0;
  for (auto i : ops_) {
    const auto& op = prob_->ops[i];
    if (op.type == OpType::MatMul) {
      const size_t o = op.output();
      const int64_t Km = prob_->tensors[op.inputs[0]].width;  // contraction
      const double fractals = (double)((prob_->tensors[o].width + 15) / 16) *
                              (double)((prob_->tensors[o].height + 15) / 16) *
                              (double)((Km + 15) / 16);
      cube_compute += fractals * (double)std::max<int64_t>(prob_->cube_compute_cost, 1);
    } else {  // Pointwise / Reduction — element work, lane-stepped, tile-invariant
      int64_t elems = (int64_t)prob_->tensors[op.output()].width *
                      prob_->tensors[op.output()].height;
      for (auto t : op.inputs)
        elems = std::max(elems, (int64_t)prob_->tensors[t].width * prob_->tensors[t].height);
      const int64_t lanes = std::max<int64_t>(1, prob_->vector_lanes);
      vector_compute += (double)((elems + lanes - 1) / lanes) *
                        (double)std::max<int64_t>(prob_->vector_compute_cost, 1);
    }
  }

  // DDR traffic. CRITICAL: fusion on 910B does NOT reduce DDR — the matmul still
  // reloads its operands and the intermediate still round-trips DDR (only 950's
  // direct pipe removes the roundtrip). So the mixed DDR is, exactly like the
  // separated two kernels:
  //   (a) matmul operand reload — the SAME term as the cube model (shared helper;
  //       the matmul is the effective sink here, tiled at the output grid);
  //   (b) every cube↔vector-crossing intermediate, round-tripped (write+read = 2x
  //       through the off-chip GM ring);
  //   (c) vector-only boundary inputs (read) + boundary outputs (written).
  // A cube operand is already in (a), so it is excluded from (c) to avoid double
  // counting; an ephemeral that stays within one unit is free (not crossing).
  double ddr_bytes = cube_operand_reload(cfg, /*matmul_at_output_grid=*/true);  // (a)
  FlatSet<size_t> cube_operands;
  for (auto i : ops_)
    if (prob_->ops[i].type == OpType::MatMul)
      for (auto t : prob_->ops[i].inputs) cube_operands.insert(t);
  for (const auto& info : boundary_tensor_info_) {                              // (c)
    const double bytes = (double)info.full_size * dtype_bytes(prob_->tensors[info.id].dtype);
    if (!info.is_internally_produced && !cube_operands.count(info.id))
      ddr_bytes += bytes;  // vector-read boundary input (cube operands are in (a))
    if (info.is_boundary_out)
      ddr_bytes += bytes;  // boundary output store
  }
  FlatSet<size_t> in_sg(ops_.begin(), ops_.end());                             // (b)
  for (auto t : ephemeral_) {
    const int prod = dag_->tensor_producer[t];
    const bool prod_cube = prod >= 0 && prob_->ops[(size_t)prod].type == OpType::MatMul;
    bool crosses = false;
    for (auto c : dag_->tensor_consumers[t])
      if (in_sg.count(c) && (prob_->ops[c].type == OpType::MatMul) != prod_cube) {
        crosses = true; break;
      }
    if (crosses)
      ddr_bytes += 2.0 * (double)(prob_->tensors[t].width * prob_->tensors[t].height) *
                   dtype_bytes(prob_->tensors[t].dtype);
  }

  // Tiling for UNITS. With the fixed 1:2 cube:vector ratio, the atomic resource
  // is a UNIT = 1 cube + 2 vector cores (the 950 mix-cluster; on 910B the same
  // ratio + the GM ring). Regions tile over n_units = num_cube_cores; WITHIN a
  // unit the cube and its 2 vector cores are pipeline STAGES (not a finer output
  // grid), so the stage times divide by 1 and 2 cores per unit. DDR is the shared
  // floor (sat≈1: a mixed kernel issues DMA from both stages — refine later).
  const double n_units    = (double)prob_->num_cube_cores;  // 1:2 invariant
  const double eff_units  = std::min((double)num_tiles, n_units);
  const double cube_stage = cube_compute   / eff_units;
  const double vec_stage  = vector_compute / (2.0 * eff_units);
  const double ddr_lat    = ddr_bytes * inv_B;
  result.latency       = std::max({cube_stage, vec_stage, ddr_lat});
  result.parallel_split = 1;
  result.cores_used    = (int)(3.0 * eff_units);  // 1 cube + 2 vector per unit
  result.compute_bound = std::max(cube_stage, vec_stage) >= ddr_lat;
  result.ddr_traffic   = ddr_lat;

  // One kernel fill per unit-round (rounds = ceil(num_tiles / n_units)).
  if (prob_->kernel_fill_cost > 0) {
    const int64_t n_units_i = std::max<int64_t>(1, prob_->num_cube_cores);
    const int64_t rounds = (num_tiles + n_units_i - 1) / n_units_i;
    result.latency += (double)rounds * (double)prob_->kernel_fill_cost;
  }
  return result;
}

// ============================================================================
// Granularity enumeration
// ============================================================================

CostResult Ascend910BCost::best_cost(const FlatSet<size_t> &retained_from_prev,
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