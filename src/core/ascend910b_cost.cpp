#include "core/ascend910b_cost.h"
#include "core/subgraph_structure.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>
#include <set>
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
// Grounded pto-isa machine model (Ascend 910B / A2A3)
// ============================================================================
// All costs are in CORE CYCLES, matching pto-isa's EstimateLinearCycles (cube)
// and EstimateBandwidthCycles (transfers), using the grounded A2/A3 coefficients
// (per-direction bandwidths, core clock, L0/vector-register sizes).
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

// Vector REDUCTION tree coefficients (pto-isa perf-sim, vec_tile_study/vec_reduce, R^2~1.0,
// device-eval-pending). A reduction is NOT a single slope*repeat op: it lowers to a
// barrier-separated tree of count-mode passes, so the cost tracks the REDUCED-AXIS tree,
// not ROWS*COLS. The old slope_reduce*(ROWS*COLS/epr) overcounted a tall row-reduce up to 19x.
//   Row reduction (reduce W): 45*(K-1)+51 cycles, K=W/epr -- ROWS-INDEPENDENT, linear in W.
//   Col reduction (reduce H): 16*(H-1)+30*log2(H) cycles -- pairwise vadd tree across rows.
constexpr double kVecRowReducePass = 45.0;   // per barrier-isolated count-mode vadd pass
constexpr double kVecRowReduceFinal = 51.0;  // K=1 base (the final cross-lane vcadd block)
constexpr double kVecColReduceSlope = 16.0;  // streamed count-mode vadd per row-pair
constexpr double kVecColReduceLevel = 30.0;  // per-level startup (log2(H) barriers)

// Grounded per-op VECTOR compute cycles (pto-isa perf-sim, vec_tile_study). Shared by the
// vector-only and the mixed cube+vector paths so a reduction costs the same in both.
//   Pointwise: slope*repeat + (head+tail IF this op starts a vector stream). Fix 3: the
//     perf-sim pays head+tail only when the VEC queue is empty (a back-to-back chain overlaps
//     its startup), so the caller passes pw_stream_start=true only for the first pointwise op
//     of a stream (chain start, or after a reduction/matmul barrier) -- not per op.
//   Reduction (Fix 1): the REDUCED-AXIS tree -- ROWS-independent for reduce-W, log-depth for
//     reduce-H -- NOT slope_reduce*(ROWS*COLS/epr) (overcounted a tall row-reduce up to 19x).
//     Its barrier-isolated per-pass startups are already baked into the tree constants.
inline double VecOpCompute(const Problem *p, const Op &op, int reduced_axis, bool pw_stream_start) {
  const int64_t reg = p->vec_reg_bytes > 0 ? p->vec_reg_bytes : 256;
  const int64_t epr = std::max<int64_t>(1, reg / dtype_bytes(p->tensors[op.output()].dtype));
  if (op.type == OpType::Reduction) {
    const int64_t W = (int64_t)p->tensors[op.inputs[0]].width;
    const int64_t H = (int64_t)p->tensors[op.inputs[0]].height;
    if (reduced_axis == 2)  // reduce height: pairwise vadd tree across rows
      return kVecColReduceSlope * (double)std::max<int64_t>(0, H - 1) +
             kVecColReduceLevel * (H > 1 ? std::log2((double)H) : 0.0);
    const int64_t K = std::max<int64_t>(1, W / epr);  // reduce width: ROWS-independent
    return kVecRowReducePass * (double)(K - 1) + kVecRowReduceFinal;
  }
  int64_t elems = (int64_t)p->tensors[op.output()].width * p->tensors[op.output()].height;
  for (auto t : op.inputs)
    elems = std::max(elems, (int64_t)p->tensors[t].width * p->tensors[t].height);
  const int64_t repeat = (elems + epr - 1) / epr;
  return p->vec_slope_pw * (double)repeat + (pw_stream_start ? p->vec_op_head + p->vec_op_tail : 0.0);
}

// Per-direction "cycles per byte" for a transfer: a byte costs
// (1/2^30)/bw_GiBps * freq_hz cycles (pto-isa EstimateBandwidthCycles).
struct ByteCost {
  double reload = 0.0;  // GM->L1   (cube operand reload)
  double store = 0.0;   // L0C->GM  (cube output writeback)
  double l0a = 0.0;     // L1->L0A  (lhs extract)
  double l0b = 0.0;     // L1->L0B  (rhs extract)
  double ub_in = 0.0;   // GM->UB   (vector load)
  double ub_out = 0.0;  // UB->GM   (vector store)
};

ByteCost MakeByteCost(const Problem* p) {
  // Per-direction cycles/byte: a byte costs freq / (2^30 * bw_GiBps) cycles
  // (pto-isa EstimateBandwidthCycles); bandwidths are GiB/s, per direction.
  auto cpb = [&](double bw_gibps) { return p->cube_freq_hz / (kGiB * bw_gibps); };
  return {cpb(p->bw_gm_l1), cpb(p->bw_l0c_gm), cpb(p->bw_l1_l0a),
          cpb(p->bw_l1_l0b), cpb(p->bw_gm_ub), cpb(p->bw_ub_gm)};
}

// Cube MAC cost of one M x N x K matmul, in cycles. Grounded: the dtype-aware
// fractal count x cycles-per-repeat (pto-isa cce_costmodel_cube.hpp `mad`):
// kF = 32/dtype_bytes (fp32:8, fp16:16), cyc = 2 (fp32) else 1. cube_compute_cost
// (default 1) is a calibration multiplier.
double CubeMacCycles(const Problem* p, int64_t M, int64_t N, int64_t K, DType dt) {
  const int64_t kF = std::max<int64_t>(1, 32 / dtype_bytes(dt));
  const double repeats = (double)((M + 15) / 16) * (double)((N + 15) / 16) *
                         (double)((K + kF - 1) / kF);
  const double cyc = (dt == DType::FP32) ? 2.0 : 1.0;
  const double mult = (p->cube_compute_cost > 0) ? (double)p->cube_compute_cost : 1.0;
  return repeats * cyc * mult;
}

// L1->L0 operand extract for one M x N x K matmul, in cycles. The cube re-reads
// the lhs once per L0 N-block (l0_tile_n) and the rhs once per L0 M-block
// (l0_tile_m) — the same distribution-aware reuse as cube_operand_reload, one
// hierarchy level down. Double-buffering overlaps this with the MACs, so the
// caller takes max(MAC, extract). 0 when no L0 base tile.
double CubeExtractCycles(const Problem* p, const ByteCost& bc, int64_t M,
                         int64_t N, int64_t K, DType dt) {
  if (p->l0_tile_m <= 0 || p->l0_tile_n <= 0)
    return 0.0;
  const double db = (double)dtype_bytes(dt);
  const double MNK = (double)M * (double)N * (double)K;
  const double lhs_bytes = MNK / (double)p->l0_tile_n * db;
  const double rhs_bytes = MNK / (double)p->l0_tile_m * db;
  return lhs_bytes * bc.l0a + rhs_bytes * bc.l0b;
}

// Wave-aware compute makespan. The independent work units (spatial tiles x
// split-K partials) are EQUAL-cost under the current uniform-tile / equal-K-split
// representation, so they run in ceil(U/C) "waves" and the makespan is set by the
// fullest wave, NOT by an idealized total/C fractional division:
//
//   T_compute = ceil(U/C) * (W_total / U)
//
// vs the old W_total / min(U, C), which silently assumed work splits fractionally
// and so under-charged any U > C that is not a multiple of C (e.g. 32 units on 24
// cores: old = W/24, true = 2*(W/32) = W/16, a 33% miss). U = num_work_units; the
// effective parallelism is U/ceil(U/C), NOT min(U,C).
double WaveComputeCycles(double total_compute, int64_t num_work_units,
                         int64_t num_cores) {
  const int64_t units = std::max<int64_t>(1, num_work_units);
  const int64_t cores = std::max<int64_t>(1, num_cores);
  const int64_t waves = (units + cores - 1) / cores;
  return total_compute * (double)waves / (double)units;
}

// LPT makespan for a non-uniform parts_m x parts_n SpatialSchedule grid. The even
// split yields at most 4 distinct region shapes (m_ext, n_ext);
// `region_work(m_ext, n_ext)` gives one region's double-buffered cost. LPT-assign
// the P*Q regions across n_cores and return the busiest core's load. With
// parts == n_cores this is one wave -> the largest region's cost; the LPT also
// captures the +-1-fractal imbalance and multi-wave grids. The per-region work is
// supplied by the caller so a single-matmul sink and a chained group (sink region
// + backpropagated intermediate row-bands) share this distribution logic.
// `ksplit` > 1 applies split-K ON the grid: each region's K-contraction splits
// into ksplit equal partials, so the P*Q regions become P*Q*ksplit work units of
// work/ksplit each. This keeps split-K LPT-consistent with the grid (the equal-
// unit wave would optimistically ignore the +-1-fractal region imbalance).
template <typename RegionWork>
double LptMakespan(int64_t n_cores, const AxisPartition& pm, const AxisPartition& pn,
                   RegionWork region_work, int64_t ksplit = 1) {
  ksplit = std::max<int64_t>(1, ksplit);
  const int64_t m_sizes[2] = {pm.big, pm.small};
  const int64_t m_cnts[2] = {pm.num_big, pm.parts - pm.num_big};
  const int64_t n_sizes[2] = {pn.big, pn.small};
  const int64_t n_cnts[2] = {pn.num_big, pn.parts - pn.num_big};
  std::vector<double> regions;
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b) {
      const int64_t cnt = m_cnts[a] * n_cnts[b];
      if (cnt <= 0 || m_sizes[a] <= 0 || n_sizes[b] <= 0) continue;
      const double work = region_work(m_sizes[a], n_sizes[b]) / (double)ksplit;
      for (int64_t i = 0; i < cnt * ksplit; ++i) regions.push_back(work);
    }
  std::sort(regions.begin(), regions.end(), [](double x, double y) { return x > y; });
  std::vector<double> load(std::max<int64_t>(1, n_cores), 0.0);
  for (double w : regions) {  // longest-processing-time-first onto the least-loaded core
    size_t mn = 0;
    for (size_t c = 1; c < load.size(); ++c)
      if (load[c] < load[mn]) mn = c;
    load[mn] += w;
  }
  double mk = 0.0;
  for (double l : load) mk = std::max(mk, l);
  return mk;
}

}  // namespace

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
  // Unit-homogeneity admissibility (always enforced on the 910B): cube↔vector
  // fusion is allowed only when the cost model opts in (allow_mixed, set by
  // Ascend910BMixed); Opaque ops are singleton barriers; reductions on different
  // axes may not fuse. (A bare scope so the has_* locals stay local.)
  {
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
    // Skewability gate (mixed only): the mixed cost credits the cube∥vector OVERLAP
    // (`max`), which is the SKEWED cost. Only single-round-trip shapes skew — the 4
    // canonical c→v / v→c / v→c→v / c→v→c, i.e. ≤ 2 unit alternations along any
    // dependency path (#1900's depth-2 per-stage buffers cap the validated skew). A
    // deeper multi-round-trip (c→v→c→v, …) demotes to Sequential (up to 2.26× the max,
    // grounded by mixed_serial), so `max` would OVER-credit it — the solver could pick a
    // fusion far slower than the separated config it rejects. Reject here so the
    // partitioner cuts it into separate kernels rather than one over-credited mixed group.
    if (has_cube && has_vector) {  // allow_mixed is true here (else returned above)
      std::vector<int> alt_depth(num_ops, 0);
      int max_alt = 0;
      for (size_t i : dag.topological_order()) {
        if (!is_in_sg[i]) continue;
        const bool i_cube = prob.ops[i].type == OpType::MatMul;
        int d = 0;
        for (auto t : prob.ops[i].inputs) {
          const int prod = dag.tensor_producer[t];
          if (prod < 0 || !is_in_sg[(size_t)prod]) continue;  // boundary input — no crossing
          const bool p_cube = prob.ops[(size_t)prod].type == OpType::MatMul;
          d = std::max(d, alt_depth[(size_t)prod] + (p_cube != i_cube ? 1 : 0));
        }
        alt_depth[i] = d;
        max_alt = std::max(max_alt, d);
      }
      sg.mixed_round_trip_depth_ = max_alt;
      if (max_alt > 2) return std::nullopt;  // multi-round-trip → not skewable, force a cut
    }
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
  // are refined by the epilogue detection, so it is not purely structural.)
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

  // Rules 2/3 (prologue-PW geometric condition):
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
    // in-subgraph consumer). The epilogue detection below may additionally
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

    // Detect simple MM→PW epilogue pattern:
    //   All sinks are PW, and walking backward from PW sinks through PW-only
    //   chains reaches exactly one MM. That MM is the "effective sink" —
    //   its k-loop runs, accumulates into its ephemeral output, then the PW
    //   chain fires once on the completed tile. This is always valid.
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
    // boundary_tensor_info_ materialization:
    //   FULL signature (FIXED_1, FIXED_1) dominates — collapse to one full
    //     entry, drop partials.
    //   Multiple distinct partial signatures → one entry each, working set
    //     sums them. Capped at 2 partials per the row/col simplification.
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
        // boundary_tensor_info_.
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
    //   full-role collapse: if any signature is FULL (FIXED_1, FIXED_1), it
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

      // full-role collapse: full role subsumes all others.
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

  // The spatial w/h candidates feed the grid via w_divides_ / h_divides_ directly
  // (gen_grid partitions the SINK output over divisors of the core count); the grid
  // handles the reduction's full-reduced-extent axis itself. Only the k candidates
  // are still materialized here.
  // PW-sink subgraphs: force k = output_K_ so nk == 1 (no temporal tiling).
  //   PW-only sinks:  output_K_ == 1 → k == 1 in solution.
  //   Mixed MM+PW sinks: output_K_ == op_K(mm) → k == K in solution
  //     (full reduction in one pass).
  // Exception: simple MM→PW epilogue — enumerate k candidates
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

  // SpatialSchedule grid candidates: balanced ~C-region partitions the uniform
  // exact-divisor tiles cannot express (powers of two only yield power-of-two tile
  // counts, never a multiple of C). Used for BOTH the cube (C = cube cores) and the
  // vector (C = vector cores) paths. Each candidate is a (parts_m, parts_n,
  // split_k) TRIPLE: P*Q is a balanced 16-aligned spatial grid (a divisor of
  // {C, 2C}), bounded by each axis's 16-fractal cap; split_k is the parallel
  // contraction/reduction split. The WORK UNITS P*Q*S range freely -- filling all C
  // cores is a strong SOFT preference, but the cost (merge barrier vs streaming
  // gain) drives it (a small shape can be best at FEWER than C units when the split
  // merge outweighs recruiting idle cores). compute_cost evaluates each fixed
  // triple (no internal S sweep). PQ == 1 (the whole-output region) IS included
  // so the grid is self-sufficient on the 910B path: (1,1,S) is the pure
  // split-K / single-region fill the uniform whole-output tile used to provide.
  auto gen_grid = [&](int64_t C, int64_t maxP, int64_t maxQ,
                      const std::vector<int64_t> &s_vals) {
    std::set<int64_t> region_counts;  // balanced P*Q: divisors of {C, 2C} (incl. 1)
    for (int64_t R : {C, 2 * C})
      for (int64_t d = 1; d * d <= R; ++d)
        if (R % d == 0) { region_counts.insert(d); region_counts.insert(R / d); }
    for (int64_t PQ : region_counts) {
      for (int64_t P = 1; P <= PQ; ++P) {
        if (PQ % P != 0) continue;
        const int64_t Q = PQ / P;
        if (P > maxP || Q > maxQ) continue;
        for (int64_t S : s_vals) sg.grid_cand_.push_back({P, Q, S});
      }
    }
  };
  // Tile granularity: cube is 16x16-fractal aligned on both axes; vector has no
  // fractal constraint, so its free (row/height) axis tiles at 1 element and its
  // contiguous (width) axis at the 16-element DMA block. The finer row granule is
  // what lets a few-row reduction (softmax) tile enough regions to fill C.
  sg.grid_gran_h_ = matmul_910b ? 16 : 1;
  sg.grid_gran_w_ = 16;
  const int64_t Fm = (sg.out_H_ + sg.grid_gran_h_ - 1) / sg.grid_gran_h_;
  const int64_t Fn = (sg.out_W_ + sg.grid_gran_w_ - 1) / sg.grid_gran_w_;
  if (matmul_910b) {
    // Cube: the grid tiles the SINK output; the chain backpropagates from it, so
    // every matmul must share the sink's M (tensors[o].height == out_H_) -- the
    // (A@B)@D chain does; a DAG tiling M differently falls back to uniform. split_k
    // is the sink-K split S (S | kfrac, each partial a 16-aligned K-slice), so a
    // power-of-two shape that can't form C spatial regions still fills the cores
    // via split-K (64^2 = 4x4 fractals -> no P*Q=24, but (4,3) grid x S=2 = 24).
    bool shared_m = true;
    for (auto i : sg.ops_)
      if (prob.ops[i].type == OpType::MatMul &&
          prob.tensors[prob.ops[i].output()].height != sg.out_H_)
        shared_m = false;
    if (shared_m) {
      const int64_t kfrac = std::max<int64_t>(1, sg.output_K_ / 16);
      // A mixed subgraph with a REDUCTION (fused flash-attention style: matmul -> softmax)
      // must PIN the reduction's reduced axis spatially — the reduction consumes the full
      // axis online, and splitting it across cores would need a cross-core merge with no
      // mid-kernel sync. The matmul's OWN contraction rides seq-k / split-K, a separate axis
      // (the tiling chooses it). Pure cube (reduced_axis_ == 0) is unchanged: pm=Fm, pn=Fn.
      const int64_t pm = (sg.reduced_axis_ == 2) ? 1 : Fm;  // height reduced -> pin M (parts_m=1)
      const int64_t pn = (sg.reduced_axis_ == 1) ? 1 : Fn;  // width  reduced -> pin N (parts_n=1)
      gen_grid(std::max<int64_t>(1, prob.num_cube_cores), pm, pn, all_divisors(kfrac));
    }
  } else if (sg.has_vector_) {
    // Vector: tile the output across the AIV cores. A reduced axis cannot be
    // spatially tiled (the whole row/col must be present to reduce), so its parts
    // pin to 1: a width reduction ([H,W]->[H,1]) tiles only height (parts_n = 1), a
    // height reduction only width (parts_m = 1); pointwise tiles both.
    const int64_t C = std::max<int64_t>(1, prob.num_vector_cores);
    const int64_t maxP = (sg.reduced_axis_ == 2) ? 1 : Fm;  // height reduced -> no M-split
    const int64_t maxQ = (sg.reduced_axis_ == 1) ? 1 : Fn;  // width  reduced -> no N-split
    // The triple's split_k is the REDUCED-AXIS (cross-core accumulation) split --
    // the vector analog of cube split-K, meaningful ONLY for a reduction SINK. It
    // lets P_spatial * S fill the cores when the non-reduced axis alone can't (a
    // softmax whose query rows are few). S over the divisors of 2C, capped by the
    // reduced fractal count (S <= the splittable extent). Pure pointwise has no
    // axis to split -> S = 1.
    std::vector<int64_t> s_vals = {1};
    if (sg.reduced_axis_ != 0 && sg.reduction_is_sink_) {
      const int64_t rcap = std::max<int64_t>(1, sg.reduced_extent_ / 16);
      s_vals.clear();
      for (int64_t s : all_divisors(2 * C))
        if (s <= rcap) s_vals.push_back(s);
    }
    gen_grid(C, maxP, maxQ, s_vals);
  }

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

  // Grid (SpatialSchedule) mode: w,h carry the PHYSICAL (max) region extent of a
  // non-uniform parts_m x parts_n partition, which need NOT evenly divide the
  // output -- so skip the exact-divisor check (the 16-alignment above + the L1
  // fit in fits_on_chip still apply). parts are clamped to the fractal count in
  // partition_axis, so no region is empty.
  //
  // NOTE (uniform tile, not on the live solver path): best_cost is GRID-ONLY on
  // the 910B -- the SpatialSchedule grid, including the (1,1) whole-output region,
  // covers every fill, so the solver never emits a parts_m == 0 cube/vector config.
  // The cfg.parts_m == 0 exact-divisor branch below is reached ONLY by a
  // directly-constructed TileConfig{w,h,k} (unit tests / ad-hoc API calls). Kept
  // for that path; not exercised by the partition/search/solution pipeline.
  if (cfg.parts_m == 0) {
    for (int64_t v : w_divides_)
      if (cfg.w < v && v % cfg.w != 0) return false;
    for (int64_t v : h_divides_)
      if (cfg.h < v && v % cfg.h != 0) return false;
  }
  // 910B cube tile spans the full contraction (k = max_K_, accumulated in L0c);
  // the per-op k-divisibility rule (temporal-tiling correctness) does not apply
  // here, and for a chained group max_K_ may exceed a smaller op's K.
  if ((!has_pw_sink_ || has_simple_epilogue_) && !matmul_910b) {
    for (int64_t v : k_divides_) {
      // cfg.k > op_K is physically undefined — there's nothing
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

  // PW-sink: no temporal tiling allowed, UNLESS simple MM→PW epilogue.
  if (has_pw_sink_ && !has_simple_epilogue_ && nk > 1) return false;

  // Rules 2/3: prologue-PW geometric condition. A PW that feeds
  // an MM's LHS (via PW-only chain) requires cfg.w ≥ matmul.K so a single
  // PW tile spans the full LHS K-axis; feeding RHS requires cfg.h ≥ matmul.K.
  // Applies at all nk — the constraint is geometric (PW tile shape vs
  // K-axis), not conditional on split-K. No prologue PW → thresholds are 0
  // and these checks are no-ops.
  if (cfg.w < prologue_cfg_w_min_) return false;
  if (cfg.h < prologue_cfg_h_min_) return false;

  // Per-entry tensor-dim bounds (multi-role). For each
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
  // zero-cost and don't affect WS/IO accounting).
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
  // boundary_tensor_info_; divisibility checks above cover
  // shape constraints across all role orientations, and the 2-partial limit
  // is enforced at Ascend910BCost::create. No symbolic-propagation conflict check
  // is needed — the former slow path rejected exactly the multi-role configs
  // the multi-role accounting now accepts.
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
  // Full L1/Mat budget -- double-buffering does NOT reserve half. The two
  // ping-pong buffers are not "L1 + a spare half"; together they ARE the L1, so
  // the operand strip that fits is the full pool. Double-buffering is realized in
  // the emit by streaming each seq-K strip as >=2 sub-strips (load s+1 while
  // computing s) -- it HALVES the per-load k, not the resident operand. Reserving
  // half here would double-count the prefetch buffer and wrongly reject tiles
  // whose operand genuinely fits.
  const double l1 = (double)prob_->l1_capacity;
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
    // An ephemeral is an L1 OPERAND band only while an in-subgraph MATMUL reads it
    // (the consumer needs it L1-resident). One consumed only by a VECTOR op (a
    // cube->vector crossing) is NOT an L1 band — it drains from L0c to the GM ring.
    // Homogeneous cube: every ephemeral has a matmul consumer, so last_mm == last
    // consumer and this is unchanged; it only differs inside a mixed group.
    int last_mm = -1;
    for (auto c : dag_->tensor_consumers[t])
      if (pos[c] >= 0 && prob_->ops[c].type == OpType::MatMul)
        last_mm = std::max(last_mm, pos[c]);
    if (last_mm < 0) continue;  // no in-subgraph matmul consumer -> not an L1 band
    int64_t h = std::min(cfg.h, prob_->tensors[t].height);
    int64_t bytes = prob_->tensors[t].width * h * dtype_bytes(prob_->tensors[t].dtype);
    for (int s = prod; s <= last_mm; ++s) band_at[s] += bytes;
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
    // A VECTOR op does not pressure L1 — it runs on the vector unit (UB). Skip it
    // from the cube L1 sweep. (Homogeneous cube has no vector ops, so this is never
    // taken there; it only matters inside a mixed group.)
    if (op.type != OpType::MatMul)
      continue;
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

  // Ephemeral bands resident across [producer, last in-subgraph VECTOR consumer].
  // A vector->vector ephemeral is a UB band; a CUBE-produced ephemeral (a
  // cube->vector crossing) is NOT — it streams through the GM ring, so it is only
  // a transient at the consuming step (below). Homogeneous vector: every ephemeral
  // is vector-produced and vector-consumed, so is_ub_band == ephemeral and this is
  // unchanged; it only differs inside a mixed group.
  std::vector<int64_t> band_at(order.size(), 0);
  std::vector<bool> is_ub_band(prob_->num_tensors(), false);
  for (size_t t : ephemeral_) {
    int pr = dag_->tensor_producer[t];
    if (pr >= 0 && prob_->ops[(size_t)pr].type == OpType::MatMul) continue;  // crossing
    int prod = (pr >= 0 && pos[pr] >= 0) ? pos[pr] : 0;
    int last_vec = -1;
    for (auto c : dag_->tensor_consumers[t])
      if (pos[c] >= 0 && prob_->ops[c].type != OpType::MatMul)
        last_vec = std::max(last_vec, pos[c]);
    if (last_vec < 0) continue;  // no in-subgraph vector consumer -> not a UB band
    const int64_t b = tile_bytes(t);
    for (int s = prod; s <= last_vec; ++s) band_at[s] += b;
    is_ub_band[t] = true;
  }

  // Retained (coupling) tensors resident across the subgraph (disabled on 910B).
  int64_t base = 0;
  for (auto t : retained_from_prev) base += prob_->tensors[t].size_bytes();
  for (auto t : retain_these)
    if (!retained_from_prev.count(t)) base += prob_->tensors[t].size_bytes();

  int64_t peak = 0;
  for (int s = 0; s < (int)order.size(); ++s) {
    const Op &op = prob_->ops[order[s]];
    if (op.type == OpType::MatMul) continue;  // cube op: not in the UB (vector) sweep
    int64_t transient = 0;
    // UB-band inputs are already in band_at; a non-band input is a transient tile
    // (a boundary read, or a crossing tile popped from the GM ring).
    for (auto in : op.inputs)
      if (!is_ub_band[in] && !retained_from_prev.count(in) && !retain_these.count(in))
        transient += tile_bytes(in);
    const size_t o = op.output();  // boundary/crossing output tile (resident while written)
    if (!is_ub_band[o] && !retain_these.count(o)) transient += tile_bytes(o);
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
  // Full UB budget -- double-buffering streams the tile in sub-chunks (the emit
  // ping-pong), it does not reserve half the pool.
  const int64_t budget = (int64_t)prob_->vec_capacity;
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
                                  const FlatSet<size_t> &retained_from_prev,
                                  const FlatSet<size_t> &retain_these) const {
  // Two-pool feasibility for a mixed cube+vector kernel — REUSE the homogeneous
  // single-core streams, now that both are affinity-aware (each skips the other
  // unit's ops, and treats a cube↔vector crossing as ring-streamed, not a resident
  // band):
  //   * CUBE stage — derive_exec sweeps the L1 operand bands and derives each
  //     matmul's per-op SEQ-K (slicing the contraction to fit L1). A held
  //     cube→cube intermediate is an L1 band; a crossing matmul output drains
  //     L0c→ring (not an L1 band); vector ops are skipped. L0c output sizing is
  //     deferred to AutoTileMatmulL0, as in the homogeneous cube.
  //   * VECTOR stage — vector_stream streams the [w,h] tile through UB in chunks
  //     (down to a min-chunk; free for a pointwise, recompute-costed only for a
  //     reduction). A held vector→vector intermediate is a UB band; a crossing tile
  //     popped from the ring is a transient; cube ops are skipped.
  // The crossing's DDR roundtrip is paid in compute_cost, not in feasibility.
  return derive_exec(cfg, output_K_, retained_from_prev, retain_these, nullptr) != INT64_MAX &&
         vector_stream(cfg, retained_from_prev, retain_these).chunk > 0;
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
                                           bool matmul_at_output_grid,
                                           double *lhs_bytes_out,
                                           double *rhs_bytes_out) const {
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
  double reload = 0.0, lhs_bytes = 0.0, rhs_bytes = 0.0;  // lhs->L0A, rhs->L0B (MTE1 split)
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
    if (!produced.count(lhs) && counted.emplace(lhs, 0, is_boundary_op).second) {
      const double b = M_i * N_i * K_i / w_i * dtype_bytes(prob_->tensors[lhs].dtype);
      reload += b; lhs_bytes += b;
    }
    if (!produced.count(rhs) && counted.emplace(rhs, 1, is_boundary_op).second) {
      const double b = M_i * N_i * K_i / h_i * dtype_bytes(prob_->tensors[rhs].dtype);
      reload += b; rhs_bytes += b;
    }
  }
  if (lhs_bytes_out) *lhs_bytes_out = lhs_bytes;
  if (rhs_bytes_out) *rhs_bytes_out = rhs_bytes;
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

  const ByteCost bc = MakeByteCost(prob_);  // per-direction cycles/byte (grounded)
  // Grid (SpatialSchedule) mode: the spatial region count is parts_m x parts_n
  // exactly (a balanced non-uniform partition). Uniform mode: floor-divide the
  // output by the tile. (w,h carry the physical region extent in both.)
  const int num_tw = (cfg.parts_n > 0) ? (int)cfg.parts_n : std::max((int)(out_W_ / cfg.w), 1);
  const int num_th = (cfg.parts_m > 0) ? (int)cfg.parts_m : std::max((int)(out_H_ / cfg.h), 1);
  const int num_tiles = num_tw * num_th;
  result.num_spatial_tiles = num_tiles;
  result.num_k_passes = has_matmul_ ? std::max((int)(output_K_ / cfg.k), 1) : 1;

  // 910B parallel-core roofline — the only cost model (the competition
  // single-context model was removed). Compute parallelizes across the unit's
  // cores (spatial tiles + split-K = independent work units); DDR traffic divides
  // across each core's own GM pipe (MTE2/FixPipe) up to the aggregate HBM ceiling.
  const int n_cores = has_matmul_ ? prob_->num_cube_cores : prob_->num_vector_cores;
  {
    // Per-direction realized parallel GM-pipe count. Each core has its own MTE2
    // (GM->L1/UB) and FixPipe (L0C/UB->GM), so a direction's DDR traffic divides
    // across `active` cores' pipes up to the aggregate HBM ceiling:
    //   par(active, peak) = min(active, hbm_aggregate_gibps / per_core_peak)
    // and that direction's cycles = bytes * cyc_per_byte / par. Exactly pto-isa
    // BwEff (effective bw = min(active*peak, hbm)). hbm<=0 => uncapped (pure
    // per-core divide; the cores never saturate a finite HBM).
    const double hbm = prob_->hbm_aggregate_gibps;
    auto par = [&](double active, double peak_gibps) {
      const double cap = (hbm > 0.0 && peak_gibps > 0.0)
                             ? hbm / peak_gibps : std::numeric_limits<double>::infinity();
      return std::max(1.0, std::min(active, cap));
    };
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
      // Per-core cube work is HIERARCHICAL and double-buffered: producing a tile
      // needs both the cube MACs (Matrix pipe) AND the L1->L0A/L0B operand
      // extract (MTE1 pipe), which OVERLAP — so the work is max(MACs, extract),
      // not their sum. Accumulate the two pipes separately, then take the max.
      // (Validated against pto-isa's measured 7680^3 GEMM: with the L0 128x256
      // base tile the extract lands at ~0.6x the MACs — i.e. cube-bound, extract
      // hidden — matching its 63% vs 80.6% pipe-utilisation split.)
      double total_cube = 0.0;     // Matrix-pipe cycles (cube MACs)
      double total_extract = 0.0;  // MTE1-pipe cycles (L1->L0 operand extract)
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
        // Grounded machine model: dtype-aware fractal MACs + the hierarchical
        // L1->L0 extract (CubeMacCycles / CubeExtractCycles). CEIL granularity
        // so a sub-16 dim pads up to one fractal — a small-batch GEMV (M<16)
        // still costs a full fractal row, matching the cube's 16-granularity.
        const int64_t Mo = prob_->tensors[o].height;
        const int64_t No = prob_->tensors[o].width;
        const int64_t Ko = prob_->tensors[prob_->ops[i].inputs[0]].width;  // contraction
        const DType dt = prob_->tensors[o].dtype;
        total_cube += CubeMacCycles(prob_, Mo, No, Ko, dt) * recompute;
        total_extract += CubeExtractCycles(prob_, bc, Mo, No, Ko, dt) * recompute;
      }
      // Double-buffered overlap of the two per-core cube pipes (MACs || extract).
      const double total_compute = std::max(total_cube, total_extract);
      // --- BSP/superstep matmul roofline (910B; no shared L2) ----------------
      // Parallelize the (possibly fused-chain) subgraph over the SHARED output
      // dim M: each core owns a sink-output band and computes its own
      // intermediate slices ON-CHIP. Consequences:
      //   * DDR counts only (a) boundary/graph-input operands and (b) the sink
      //     output store. Intermediate tensors are EPHEMERAL (zero DDR) — a
      //     deeper matmul's partials are computed independently per output tile,
      //     never crossing cores, so they need no barrier.
      //   * A sink split-K writes the output S times (S atomic-add partials via
      //     SetAtomicAdd). That is extra DDR write traffic riding the roofline with
      //     compute — NOT an additive barrier (each core writes independently).
      //     Intermediate k-split is disallowed: it would round-trip the intermediate
      //     through DDR, i.e. be no better than splitting the subgraph there (which
      //     also frees independent tiling) — so any k-split lands on the output only.
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
      double reload_lhs_b = 0.0, reload_rhs_b = 0.0;
      const double reload = cube_operand_reload(cfg, /*matmul_at_output_grid=*/false,
                                                &reload_lhs_b, &reload_rhs_b);
      // L1->L0 extract cycles (MTE1), a TIEBREAKER only: lhs via L0A (fast 441),
      // rhs via L0B (slow 220.5). Same per-port byte split as the GM reload, charged
      // at the L0 ports — so among reload-equal transposed tiles the TALL one (large h,
      // less slow-L0B traffic) wins. Perf-sim-driven (gml1_decision); device eval pending.
      const double l1l0_extract = reload_lhs_b * bc.l0a + reload_rhs_b * bc.l0b;
      // DDR is two SEPARATE concurrent pipes: the operand reload (MTE2, GM->L1,
      // 135 GiB/s) and the output store (FixPipe, L0C->GM, 70 GiB/s). They are
      // distinct hardware (pto-isa scores GM reads/writes as independent groups),
      // so the DDR term is their MAX, not their sum. Each pipe's traffic DIVIDES
      // across active cores up to the aggregate HBM ceiling (see par()): cycles =
      // bytes*cyc_per_byte / par(active, peak). Compute is wave-aware (ceil(U/C)
      // waves); the DDR bytes are NOT wave-scaled (a K-split divides the operand
      // slices, it does not inflate the total reload bytes).
      // Grid mode: LPT makespan over the non-uniform (+-1-fractal) regions. The
      // per-region work is the SINK matmul at the region extent (m_ext, n_ext)
      // PLUS each consumed INTERMEDIATE as a full-width [m_ext, N_int] row-band:
      // the grid is the sink-output tiling, and the chain backpropagates from it
      // -- the sink M-partition slices every intermediate's rows, while the
      // intermediate is consumed full-width (it is the next matmul's contraction),
      // so each N-region in an M-band recomputes the band (counted per region,
      // reproducing the num_tw recompute factor of total_compute). Hoisted so the
      // split-K loop below reuses it (LPT-consistent split-K on the grid). Uniform
      // mode: the wave makespan over equal work units.
      const AxisPartition g_pm = partition_axis(out_H_, std::max<int64_t>(1, cfg.parts_m), grid_gran_h_);
      const AxisPartition g_pn = partition_axis(out_W_, std::max<int64_t>(1, cfg.parts_n), grid_gran_w_);
      // L0 base tile (pto-isa GEMM oracle); the K-step is baseK=64. Used for the
      // Phase-D pipeline depth. Fall back to {1,1} when ungrounded (extract==0
      // then, so the depth is immaterial).
      const int64_t l0m = std::max<int64_t>(1, prob_->l0_tile_m);
      const int64_t l0n = std::max<int64_t>(1, prob_->l0_tile_n);
      auto grid_region_work = [&](int64_t m_ext, int64_t n_ext) {
        double work = 0.0;
        for (auto i : ops_) {
          if (prob_->ops[i].type != OpType::MatMul) continue;
          const size_t o = prob_->ops[i].output();
          const int64_t Ko = prob_->tensors[prob_->ops[i].inputs[0]].width;
          const DType dt = prob_->tensors[o].dtype;
          const int64_t n = consumed.count(o) ? prob_->tensors[o].width  // full band
                                              : n_ext;                    // sink region
          const double mac = CubeMacCycles(prob_, m_ext, n, Ko, dt);
          const double ext = CubeExtractCycles(prob_, bc, m_ext, n, Ko, dt);
          // Phase D — double-buffer overlap over L L0-MAD steps:
          //   T = (mac + ext + (L-1)*max(mac,ext)) / L
          // L=1 (one L0 tile) => mac+ext (no steady state to overlap); L>>1 =>
          // max(mac,ext) (full ping-pong). L = #L0 sub-tiles of this matmul's
          // region: ceil(m/baseM)*ceil(n/baseN)*ceil(K/baseK), baseK=64.
          const int64_t L = std::max<int64_t>(
              1, ((m_ext + l0m - 1) / l0m) * ((n + l0n - 1) / l0n) *
                     ((Ko + 63) / 64));
          // Matmuls in a region run SEQUENTIALLY (the intermediate feeds the
          // sink), so their pipeline times SUM (not max-of-sums).
          work += (mac + ext + (double)(L - 1) * std::max(mac, ext)) / (double)L;
        }
        return work;
      };
      // parts_m > 0 (grid + LPT makespan) is the ONLY mode best_cost emits on the
      // 910B. The WaveComputeCycles arm is the uniform parts_m == 0 tile, reached
      // only by a directly-constructed TileConfig (tests / ad-hoc) -- see the
      // matching note in is_valid_tiling. Not produced by the live solver.
      // Double-buffer floor: the max(compute, ddr) overlap is only real when the
      // operand reload can ping-pong, i.e. the per-core contraction is halvable
      // into >=2 seq-K sub-strips (>= 32 = two K-fractals; the emit's implicit
      // halving needs that). A tiny contraction can't overlap -> reload and
      // compute SERIALIZE (compute + ddr).
      auto db_roofline = [&](double comp, double dram, int64_t per_core_K) {
        const bool overlap = per_core_K >= 32;
        return overlap ? std::max(comp, dram) : comp + dram;
      };
      // Sink split-K: split the sink contraction into S per-tile partials to
      // recruit idle cores. The output write-back grows to S partials (L0C->GM via
      // FixPipe, SetAtomicAdd — each core writes independently, no merge barrier),
      // but that store pipe OVERLAPS the operand feed, so split-K only pays for the
      // writes once S*store exceeds the feed (max, not sum).
      //
      // S is a FIRST-CLASS design axis (like w,h): more cores cut compute (and,
      // until HBM saturates, the per-core-divided operand feed) while the S output
      // writes grow the store pipe. ENUMERATE S and take the min. Useful range:
      // S <= kfrac (>=1 fractal/partial); the wave model + the growing store pipe
      // reject the excessive splits (the old ceil(n_cores/num_tiles) core-fill bound
      // is gone — a split that overfills a wave can still cut compute). output_K_ is
      // the sink matmul's contraction (NOT max_K_); only S | kfrac is emittable.
      const int64_t kfrac = std::max<int64_t>(1, output_K_ / 16);
      // Evaluate one split factor S (>=1). S=1 is the spatial-only roofline; S>=2
      // splits the sink contraction into S equal 16-aligned partials (each owns
      // output_K_/S), writing the output S times (S atomic-add partials).
      struct SplitEval { double lat, compute, ddr, active; };
      auto eval_S = [&](int64_t S) -> SplitEval {
        S = std::max<int64_t>(1, S);
        const int64_t unitsS = (int64_t)num_tiles * S;
        // Grid: LPT the P*Q regions with K split S ways (P*Q*S units) -- honest
        // about the +-1-fractal imbalance; uniform: the equal-unit wave.
        const double computeS =
            (cfg.parts_m > 0) ? LptMakespan(n_cores, g_pm, g_pn, grid_region_work, S)
                              : WaveComputeCycles(total_compute, unitsS, n_cores);
        const double activeS = (double)std::min<int64_t>(unitsS, n_cores);
        // DDR is two SEPARATE, concurrent pipes: the operand feed (MTE2, GM->L1)
        // and the output write-back (FixPipe, L0C->GM; S atomic-add partials for a
        // split). They are distinct hardware, and pto-isa scores GM reads and GM
        // writes as independent aggregate groups, so they OVERLAP -- the cube DDR
        // term is max(feed, writes), NOT their sum. Each is per-core-divided +
        // HBM-capped at its own peak (S=1 => a single output store).
        const double feed   = reload * bc.reload / par(activeS, prob_->bw_gm_l1);
        const double writes = (double)S * out_store * bc.store / par(activeS, prob_->bw_l0c_gm);
        const double ddrS = std::max(feed, writes);
        // Double-buffer floor: K/S < 32 (< 2 K-fractals) can't ping-pong -> serialize.
        const double latS = db_roofline(computeS, ddrS, output_K_ / S);
        return {latS, computeS, ddrS, activeS};
      };

      // S source: a SpatialSchedule TRIPLE fixes S (cfg.split_k from the (P,Q,S)
      // enumeration) and is evaluated as-is; an ad-hoc non-grid tile (split_k==0,
      // e.g. a directly-constructed TileConfig) sweeps S over the valid K-fractal
      // divisors and adopts a split only if it STRICTLY beats the spatial-only S=1.
      // Split-K is MODEL-AHEAD of the AutoFuse emit: gate on the buildable flag so a
      // buildable-mode harness never selects an unemittable split (default true = analytic).
      int64_t chosen_S = 1;
      SplitEval chosen = eval_S(1);
      if (prob_->allow_model_ahead_split_k) {
        if (cfg.split_k >= 1) {
          chosen_S = cfg.split_k;
          chosen = eval_S(chosen_S);
        } else {
          for (int64_t S : all_divisors(kfrac)) {
            if (S < 2) continue;
            const SplitEval e = eval_S(S);
            if (e.lat < chosen.lat) { chosen = e; chosen_S = S; }
          }
        }
      }
      result.uses_model_ahead_split_k = (chosen_S > 1);

      result.latency = chosen.lat;
      result.parallel_split = (int)chosen_S;
      result.cores_used = (int)chosen.active;
      result.compute_bound = chosen.compute >= chosen.ddr;
      result.ddr_traffic = chosen.ddr;
      result.l1l0_extract = l1l0_extract;  // MTE1 tiebreaker (tall tiles win the slow L0B port)
      // Displayed per-core k: the greedy single-core L1-fit k (derive_exec, a
      // divisor of output_K_), capped for a split by the per-core fractal share
      // ceil(kfrac/S)*16 -- the largest divisor of output_K_ not exceeding both.
      std::vector<int64_t> pk;
      derive_exec(cfg, output_K_, retained_from_prev, retain_these, &pk);
      const int64_t l1_k = (sink_mm_op_ >= 0 && pk[sink_mm_op_] > 0) ? pk[sink_mm_op_]
                                                                     : output_K_;
      if (chosen_S > 1) {
        const int64_t share_k = ((kfrac + chosen_S - 1) / chosen_S) * 16;
        int64_t per_core_k = 16;
        for (int64_t d = std::min({l1_k, share_k, output_K_}); d >= 16; d -= 16)
          if (output_K_ % d == 0) { per_core_k = d; break; }
        result.config.k = per_core_k;
      } else {
        result.config.k = (sink_mm_op_ >= 0 && pk[sink_mm_op_] > 0) ? pk[sink_mm_op_]
                                                                    : cfg.k;
      }
    } else {
      // Vector (PW/Reduction): compute parallelizes over spatial tiles; DDR is
      // the shared full-tensor floor (no cross-core reload to model here). The
      // element-wise work is INVARIANT to tiling (each element is touched once),
      // so SUM the per-op grounded compute over the FULL element count -- a
      // per-tile charge would inflate compute for the small sub-16 tiles we need
      // to fill the cores, wrongly favouring fewer tiles / fewer cores. (Same
      // tiling-invariance as the matmul fractal count.)
      // Per-op grounded compute, SUMMED over the full element count (tiling-invariant:
      // each element is touched once). Reductions use the axis-aware tree (Fix 1); see
      // VecOpCompute. Fix 3: head+tail is paid once per back-to-back pointwise stream, so a
      // pointwise op pays startup only when it STARTS a stream (first op, or after a
      // reduction barrier). A per-tile charge would inflate compute for small sub-16 tiles.
      double total_compute = 0.0;
      bool prev_pw = false;  // was the previous op a stream-continuing pointwise op?
      for (auto i : ops_) {
        const Op &op = prob_->ops[i];
        const bool pw = op.type != OpType::Reduction;
        total_compute += VecOpCompute(prob_, op, reduced_axis_, /*pw_stream_start=*/pw && !prev_pw);
        prev_pw = pw;
      }
      const double eff = (double)std::min<int64_t>(num_tiles, n_cores);
      // DDR per direction (byte-based cycles): GM->UB boundary loads and UB->GM
      // boundary stores, kept SEPARATE so each divides across active cores at ITS
      // OWN per-core peak (par), capped by the aggregate HBM ceiling.
      double io_in = 0.0, io_out = 0.0;
      for (const auto& info : boundary_tensor_info_) {
        const double bytes = (double)info.full_size * dtype_bytes(prob_->tensors[info.id].dtype);
        if (!retained_from_prev.count(info.id) && !info.is_internally_produced)
          io_in += bytes * bc.ub_in;   // GM->UB boundary load
        if (info.is_boundary_out && !retain_these.count(info.id))
          io_out += bytes * bc.ub_out; // UB->GM boundary store
      }
      // Step 2b — UB-overflow streaming (Fix 2, vec_stream). When the reduced band can't
      // materialize, the feasible schedule streams the reduced axis in chunks. The real
      // emit is ONLINE / flash (pto_macro_fa_softmax): each chunk's pointwise runs ONCE per
      // element and each band is read once, so compute and IO are NOT multiplied by
      // #reductions+1 (that was 3-4x pessimistic, masking every streamed softmax). The only
      // surcharge is a thin per-chunk correction -- the re-paid vector startup + an
      // O(ROWS*1) running max/sum rescale -- O(nchunks * #reductions) cheap work.
      if (has_reduction_) {
        const double budget = (double)prob_->vec_capacity;  // full UB (see vector_stream)
        const double peak = (double)vector_peak_ub(cfg);
        if (budget > 0.0 && peak > budget) {
          int reductions = 0;
          for (auto i : ops_)
            if (prob_->ops[i].type == OpType::Reduction) reductions++;
          const double nchunks = std::ceil(peak / budget);
          total_compute += nchunks * (double)reductions * (prob_->vec_op_head + prob_->vec_op_tail);
          // io_in / io_out unchanged: online streaming reads each band once (== materialized).
        }
      }
      const int64_t vreg = prob_->vec_reg_bytes > 0 ? prob_->vec_reg_bytes : 256;
      const int64_t dtb = dtype_bytes(prob_->tensors[*boundary_outputs_.begin()].dtype);
      // DMA-shape penalty (grounded): GM<->UB moves a tile as `h` strided segments
      // of `w` contiguous elements. The DMA reaches peak bandwidth only when that
      // contiguous run spans >= one transfer burst (vec_reg_bytes); below it the
      // per-descriptor setup dominates. Charge a max(1, burst/(w*dtype)) factor to
      // BOTH directions -- a wide / full-width tile is unpenalized (factor 1), only
      // sub-burst widths pay. (pto-isa BLOCK_BYTE_SIZE=32 is the DMA block.)
      const double dma_pen = std::max(1.0, (double)vreg / std::max(1.0, (double)cfg.w * (double)dtb));
      io_in  *= dma_pen;
      io_out *= dma_pen;
      // Per-direction DDR cycles: each direction divides across `active` cores' UB
      // DMA pipes up to the aggregate HBM ceiling (par), at its OWN per-core peak
      // (GM->UB load vs UB->GM store). io_out_eff lets the reduction split below
      // pass the S-times-written store without re-deriving the load term.
      auto ddr_io = [&](double active, double io_out_eff) {
        return io_in / par(active, prob_->bw_gm_ub) + io_out_eff / par(active, prob_->bw_ub_gm);
      };
      // Wave-aware compute makespan (ceil(num_tiles/cores) waves) -- a balanced
      // num_tiles==cores grid (one wave) beats an over-tiled count.
      const double compute_mk = WaveComputeCycles(total_compute, num_tiles, n_cores);
      // Double-buffer floor (grounded): the max(compute, DDR) roofline holds only
      // when the per-core tile streams in >= 2 SIMD-repeat chunks (>= 2*vec_reg_
      // bytes), so chunk s+1's load overlaps chunk s's compute. Too small to
      // ping-pong -> SERIALIZE (compute + DDR). BINARY: crossing grants max, no more.
      const double tile_bytes = (double)cfg.w * (double)cfg.h * (double)dtb;
      const bool db = tile_bytes >= 2.0 * (double)vreg;
      auto rfl = [&](double comp, double dram) { return db ? std::max(comp, dram) : comp + dram; };
      double lat = rfl(compute_mk, ddr_io(eff, io_out));
      result.parallel_split = 1;
      result.cores_used = (int)eff;
      result.compute_bound = compute_mk >= ddr_io(eff, io_out);
      result.ddr_traffic = ddr_io(eff, io_out);
      // Reduced-axis (cross-core) split — the vector analog of the cube split-K.
      // SINK-ONLY: the S per-core partials reduce across cores through DDR (fine
      // for a boundary reduction output; an internal reduction split is a subgraph
      // cut, not an in-subgraph split). Mirrors the cube (§4.2): ENUMERATE S and
      // take the min. The partials accumulate via SetAtomicAdd (910B always has it).
      // The reduced partial is thin ([H,1] for a width-reduction, [1,W] for a
      // height-reduction) -> red_dim.
      if (has_reduction_ && reduction_is_sink_) {
        const double red_dim = (double)(reduced_axis_ == 1 ? out_H_ : out_W_);
        struct RS { double lat, eff, ddr, compute; };
        auto eval_reduce_S = [&](int64_t S) -> RS {
          const double effS = (double)std::min<int64_t>(num_tiles * S, n_cores);
          const double compS = WaveComputeCycles(total_compute, num_tiles * S, n_cores);
          // The thin reduced output is written S times (S atomic-add partials):
          // extra UB->GM store traffic FOLDED into the roofline, NOT an additive
          // merge. red_dim is the thin partial ([H,1] / [1,W]); charge it the same
          // DMA-shape penalty as the base store.
          const double io_out_S = io_out + (double)(S - 1) * red_dim * bc.ub_out * dma_pen;
          const double ddrS = ddr_io(effS, io_out_S);
          const double streamS = rfl(compS, ddrS);
          return {streamS, effS, ddrS, compS};
        };
        auto take_S = [&](int64_t S) {
          const RS e = eval_reduce_S(S);
          if (e.lat < lat) {
            lat = e.lat;
            result.parallel_split = (int)S;
            result.cores_used = (int)e.eff;
            result.compute_bound = e.compute >= e.ddr;
            result.ddr_traffic = e.ddr;
          }
        };
        // The triple's split_k IS the reduced-axis split (lets P_spatial * S fill
        // the cores). An ad-hoc non-grid tile (split_k==0, e.g. a directly-
        // constructed TileConfig) sweeps S to fill instead. S=1 means no split.
        if (cfg.split_k > 1) {
          take_S(cfg.split_k);
        } else if (cfg.split_k == 0 && num_tiles < n_cores) {
          const int64_t S_max = (n_cores + num_tiles - 1) / num_tiles;
          for (int64_t S = 2; S <= S_max; ++S) take_S(S);
        }
      }
      result.latency = lat;
    }
    // Per-kernel pipeline fill — the DUAL of the eff core-fill incentive. A
    // tiling produces num_tiles "kernels"; each core runs ceil(num_tiles/n_cores)
    // of them in sequence, paying one fill per pass. eff penalizes too FEW tiles
    // (under-filled cores); this penalizes too MANY (over-tiling), so the optimum
    // sits at ~one kernel per core. Split-K fills a tile WITHIN a pass, so it
    // doesn't add rounds (it's spatial num_tiles that count). kernel_fill_cost==0
    // => no fill term.
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
  // The `max` overlap below is the SKEWED cost — valid only for a single round-trip.
  // create() rejects deeper multi-round-trips (mixed_round_trip_depth_ > 2); fail loud
  // if one slips through, rather than silently over-crediting a non-skewable group.
  assert(mixed_round_trip_depth_ <= 2 &&
         "mixed cost requires a skewable (<=2 round-trip) group");

  const ByteCost bc = MakeByteCost(prob_);  // per-direction cycles/byte (grounded)
  // Grid mode (parts_m/parts_n > 0) fixes the tile count directly: cfg.w/cfg.h are
  // the big-region EXTENTS, not exact divisors, so out_W_/cfg.w would mis-floor the
  // count. Mirrors the base cube path (Ascend910BCost::compute_cost); uniform/ad-hoc
  // tiles (parts_* == 0, a directly-built TileConfig) fall back to the divide.
  const int num_tw = (cfg.parts_n > 0) ? (int)cfg.parts_n : std::max((int)(out_W_ / cfg.w), 1);
  const int num_th = (cfg.parts_m > 0) ? (int)cfg.parts_m : std::max((int)(out_H_ / cfg.h), 1);
  const int num_tiles = num_tw * num_th;
  result.num_spatial_tiles = num_tiles;
  result.num_k_passes = std::max((int)(output_K_ / cfg.k), 1);

  // cube_mac / cube_extract: the cube stage is hierarchical and double-buffered
  // (same as the homogeneous cube path) — its time is max(MACs, L1->L0 extract).
  double cube_mac = 0.0, cube_extract = 0.0, vector_compute = 0.0;
  bool prev_pw = false;  // Fix 3: pointwise startup once per stream (matmul/reduction break it)
  for (auto i : ops_) {
    const auto& op = prob_->ops[i];
    if (op.type == OpType::MatMul) {
      prev_pw = false;  // a cube->vector crossing syncs -> next vector op restarts its stream
      const size_t o = op.output();
      const int64_t Mo = prob_->tensors[o].height;
      const int64_t No = prob_->tensors[o].width;
      const int64_t Ko = prob_->tensors[op.inputs[0]].width;  // contraction
      const DType dt = prob_->tensors[o].dtype;
      cube_mac += CubeMacCycles(prob_, Mo, No, Ko, dt);
      cube_extract += CubeExtractCycles(prob_, bc, Mo, No, Ko, dt);
    } else {  // Pointwise / Reduction — grounded per-op compute (reductions: axis-aware
      // tree, Fix 1; pointwise startup once per stream, Fix 3). Shared with the vector-only
      // path via VecOpCompute for one consistent cost.
      const bool pw = op.type != OpType::Reduction;
      vector_compute += VecOpCompute(prob_, op, reduced_axis_, /*pw_stream_start=*/pw && !prev_pw);
      prev_pw = pw;
    }
  }
  // Fix 2 (UB-overflow streaming surcharge — mirror the homogeneous vector path). A fused
  // flash-attention kernel (matmul -> softmax) keeps the reduced (keys) axis resident and
  // PINNED, but it usually overflows UB, so the schedule streams that axis in chunks ONLINE
  // (pto_macro_fa_softmax): each element is touched once, so compute is NOT multiplied by
  // #reductions+1 — the only surcharge is a thin per-chunk correction (re-paid vector startup +
  // an O(ROWS) running max/sum rescale). Same formula as the vector-only branch.
  if (has_reduction_) {
    const double budget = (double)prob_->vec_capacity;
    const double peak = (double)vector_peak_ub(cfg, retained_from_prev, retain_these);
    if (budget > 0.0 && peak > budget) {
      int reductions = 0;
      for (auto i : ops_)
        if (prob_->ops[i].type == OpType::Reduction) reductions++;
      const double nchunks = std::ceil(peak / budget);
      vector_compute += nchunks * (double)reductions * (prob_->vec_op_head + prob_->vec_op_tail);
    }
  }

  // Tiling for UNITS. With the fixed 1:2 cube:vector ratio, the atomic resource
  // is a UNIT = 1 cube + 2 vector cores (the 950 mix-cluster; on 910B the same
  // ratio + the GM ring). Regions tile over n_units = num_cube_cores; WITHIN a
  // unit the cube and its 2 vector cores are pipeline STAGES (not a finer output
  // grid), so the stage times divide by 1 and 2 cores per unit.
  const double n_units    = (double)prob_->num_cube_cores;  // 1:2 invariant
  const double eff_units  = std::min((double)num_tiles, n_units);

  // DDR traffic. CRITICAL: fusion on 910B does NOT reduce DDR — the matmul still
  // reloads its operands and the intermediate still round-trips DDR (only 950's
  // direct pipe removes the roundtrip). The GM ring is FOUR independent, per-unit
  // HBM ports that OVERLAP, so we accumulate bytes PER PORT and take the MAX over
  // ports (grounded by mixed_contention: each read port caps at par() = hbm/peak,
  // matching the sim to 0%; mixed_ddr_bound: the single-core GM is subsumed into
  // the per-unit critical path). The four ports:
  //   gm_l1  : GM->L1 cube reads    — matmul operand reload (a) + vec->cube crossing reads
  //   gm_ub  : GM->UB vector reads  — vector boundary inputs (c) + cube->vec crossing reads
  //   l0c_gm : L0C->GM cube writes  — cube->vec crossing writes + boundary out iff sink is MatMul
  //   ub_gm  : UB->GM vector writes — vec->cube crossing writes + boundary out iff sink is vector
  // A cube operand is already in (a)/gm_l1, so it is excluded from the vector
  // boundary-input reads to avoid double counting; a same-unit ephemeral is free
  // (not crossing). A crossing ephemeral splits into one WRITE (by the producer's
  // unit) + one READ (by the consuming unit) — the 2x roundtrip, per-port.
  double gm_l1_bytes  = cube_operand_reload(cfg, /*matmul_at_output_grid=*/true);  // (a)
  double gm_ub_bytes  = 0.0, l0c_gm_bytes = 0.0, ub_gm_bytes = 0.0;
  FlatSet<size_t> cube_operands;
  for (auto i : ops_)
    if (prob_->ops[i].type == OpType::MatMul)
      for (auto t : prob_->ops[i].inputs) cube_operands.insert(t);
  for (const auto& info : boundary_tensor_info_) {                              // (c)
    const double bytes = (double)info.full_size * dtype_bytes(prob_->tensors[info.id].dtype);
    if (!info.is_internally_produced && !cube_operands.count(info.id))
      gm_ub_bytes += bytes;  // vector-read boundary input (cube operands are in gm_l1)
    if (info.is_boundary_out)  // store direction is the sink unit: MatMul -> L0C->GM, vector -> UB->GM
      (info.is_mm_out ? l0c_gm_bytes : ub_gm_bytes) += bytes;
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
    if (crosses) {
      const double bytes = (double)(prob_->tensors[t].width * prob_->tensors[t].height) *
                           dtype_bytes(prob_->tensors[t].dtype);
      // Roundtrip = one WRITE by the producer's unit + one READ by the consumer's.
      if (prod_cube) { l0c_gm_bytes += bytes; gm_ub_bytes += bytes; }  // cube writes L0C->GM, vector reads GM->UB
      else           { ub_gm_bytes  += bytes; gm_l1_bytes += bytes; }  // vector writes UB->GM, cube reads GM->L1
    }
  }

  // Per-port HBM cap: each port's traffic DIVIDES across active units up to the
  // aggregate HBM ceiling (par = max(1, min(active, hbm/peak))). Duplicated from
  // the base model (see Ascend910BCost::compute_cost) — the mixed branch has no
  // access to the base's local lambda. hbm<=0 => uncapped per-unit divide.
  const double hbm = prob_->hbm_aggregate_gibps;
  auto par = [&](double active, double peak_gibps) {
    const double cap = (hbm > 0.0 && peak_gibps > 0.0)
                           ? hbm / peak_gibps : std::numeric_limits<double>::infinity();
    return std::max(1.0, std::min(active, cap));
  };
  // Each port -> cycles at its own cyc/byte (GM->L1 reload, GM->UB ub_in, L0C->GM store,
  // UB->GM ub_out) and its own per-unit peak, divided by par(). The four ports OVERLAP,
  // so ddr_lat = MAX over them (grounded: mixed_contention / mixed_ddr_bound). The VECTOR
  // ports + stage are split-K-INVARIANT (a sink split-K recruits CUBE cores only); the
  // cube ports, cube_stage, and ddr are recomputed per split factor S in eval_S below.
  const double gm_ub_lat = gm_ub_bytes * bc.ub_in  / par(eff_units, prob_->bw_gm_ub);
  const double ub_gm_lat = ub_gm_bytes * bc.ub_out / par(eff_units, prob_->bw_ub_gm);
  // Compute distribution: the LPT makespan over the non-uniform grid regions (the BUSIEST
  // unit), NOT the flat total/eff_units average, which under-predicts an imbalanced grid's
  // biggest region (up to ~2x at one region/unit, the few-tile decode corner). The CUBE region
  // work is recomputed PER REGION — max(Σ MAC, Σ extract) at the region extent — so it captures
  // the fractal/extract padding non-linearity (the per-region ceil) and lets extract vs MAC
  // dominate region-to-region, matching the mixed cube stage's own max(mac,extract) model. The
  // VECTOR region work is an output-area fraction (a documented vector-region approximation;
  // VecOpCompute is op-tensor-coupled, so a region-aware helper is a separate follow-up).
  // Uniform tiles (parts==0; ad-hoc/test) use the wave-aware total; eff_units/eff_cube (active
  // cores) still set the HBM par() + count.
  const double base_cube_work = std::max(cube_mac, cube_extract);  // MACs vs extract (total)
  const bool grid_mode = cfg.parts_m > 0;
  const AxisPartition g_pm = partition_axis(out_H_, std::max<int64_t>(1, cfg.parts_m), grid_gran_h_);
  const AxisPartition g_pn = partition_axis(out_W_, std::max<int64_t>(1, cfg.parts_n), grid_gran_w_);
  const double out_area = (double)std::max<int64_t>(1, out_H_ * out_W_);
  auto cube_region_work = [&](int64_t m_ext, int64_t n_ext) {
    double rmac = 0.0, rext = 0.0;  // per-region recompute: fractal padding + local extract/MAC max
    for (auto i : ops_) {
      const auto& op = prob_->ops[i];
      if (op.type != OpType::MatMul) continue;
      const int64_t Ko = prob_->tensors[op.inputs[0]].width;
      const DType dt = prob_->tensors[op.output()].dtype;
      rmac += CubeMacCycles(prob_, m_ext, n_ext, Ko, dt);
      rext += CubeExtractCycles(prob_, bc, m_ext, n_ext, Ko, dt);
    }
    return std::max(rmac, rext);
  };
  auto vec_region_work = [&](int64_t m_ext, int64_t n_ext) {
    return vector_compute * ((double)(m_ext * n_ext) / out_area);  // area-fraction (approximation)
  };
  // Vector stage runs on 2 cores per unit (split-K-invariant: a sink split-K recruits CUBE
  // cores only). Makespan over the grid, then halved across the unit's 2 AIV cores.
  const double vec_stage = (grid_mode
      ? LptMakespan((int64_t)n_units, g_pm, g_pn, vec_region_work)
      : WaveComputeCycles(vector_compute, num_tiles, (int64_t)n_units)) / 2.0;
  const double one_vec_tile = vector_compute / (2.0 * (double)num_tiles);

  // Pipeline wall (grounded EXACTLY by mixed_tile_study, the shape sweep). The cube and
  // vector units OVERLAP. In a 2-stage kernel (c->v or v->c) the output unit runs ONLY the
  // output op, so the wall is the SYMMETRIC cross-term: each unit's full stage plus ONE tile
  // of the OTHER (the un-overlapped fill/drain end) — max(cube_stage+one_vec_tile,
  // vec_stage+one_cube_tile, ddr_lat). This matches the sim to the decimal (c->v, v->c), folds
  // the fill INTO the max (so it is absorbed when DDR- or compute-bound by the other unit), and
  // charges an imbalanced fusion only one TINY non-bottleneck tile (a matmul + small epilogue is
  // NOT over-charged a full cube tile). A 3-stage kernel (v->c->v, c->v->c) has the output unit
  // busy from t=0 (it runs an earlier stage), so there is no idle end — plain max.
  // Detection: the sink unit is the producing unit of the boundary output (output_is_cube =
  // it is a MatMul). The fill is ABSORBED (3-stage, plain max) iff the sink unit can run at
  // t=0 — i.e. it has an "early stage" op whose whole input cone is same-unit + boundary,
  // independent of the opposite unit (the v-prologue of v->c->v, the first matmul of c->v->c).
  // Otherwise EVERY sink-unit op transitively waits on the opposite unit, so the sink idles
  // one opposite tile before starting and the fill ADDS (2-stage). Counting sink-unit ops is
  // NOT sufficient: a same-unit tail (c->v->v) has >1 sink op yet still idles at t=0. We flag
  // each op whose input cone touches the opposite unit (single forward-topo pass, producers
  // before consumers via reverse_topo_ops_), then look for a sink-unit op that does not.
  bool output_is_cube = false;
  for (const auto& info : boundary_tensor_info_)
    if (info.is_boundary_out) { output_is_cube = info.is_mm_out; break; }
  auto is_sink_unit = [&](size_t i) {
    return (prob_->ops[i].type == OpType::MatMul) == output_is_cube;
  };
  FlatSet<size_t> reaches_opp;  // ops whose input cone (incl. self) touches the opposite unit
  for (auto it = reverse_topo_ops_.rbegin(); it != reverse_topo_ops_.rend(); ++it) {
    const size_t i = *it;
    bool touches = !is_sink_unit(i);  // op i itself runs on the opposite unit
    if (!touches)
      for (auto t : prob_->ops[i].inputs) {
        const int prod = dag_->tensor_producer[t];
        if (prod >= 0 && reaches_opp.count((size_t)prod)) { touches = true; break; }
      }
    if (touches) reaches_opp.insert(i);
  }
  bool sink_runs_early_stage = false;
  for (auto i : ops_)
    if (is_sink_unit(i) && !reaches_opp.count(i)) { sink_runs_early_stage = true; break; }
  const bool two_stage = !sink_runs_early_stage;
  // Fill is absorbed only when the shape is 3-stage AND there is a successor tile to skew
  // against (num_tiles >= 2). A single-tile 3-stage kernel pays the cross-term (A2), so the
  // diagnostic flag must reflect the ACTUAL wall, not just the structural shape.
  result.pipeline_fill_absorbed = !two_stage && num_tiles >= 2;

  // Sink split-K (cube-sink, SINGLE matmul only). The sink matmul may split its
  // contraction across idle CUBE cores — atomic-add partials with NO merge barrier, so
  // the cores stay independent (exactly the base cube split-K; the vector prologue
  // overlaps orthogonally). Restricted to a single-matmul cube sink (v->c, v->v->c) so
  // ONLY the sink is ever split — never a mid-kernel matmul. split-K recruits cube cores
  // (eff_cube = min(num_tiles*S, n_units)) and grows the output write-back to S atomic
  // partials on L0C->GM; the vector stage/ports are untouched. S=1 == the spatial-only
  // cost, so a non-splittable kernel is unchanged and a split is taken only when it wins.
  int num_matmuls = 0;
  for (auto i : ops_) if (prob_->ops[i].type == OpType::MatMul) num_matmuls++;
  // Cube-sink split-K is MODEL-AHEAD of the emit (§12): gate on the buildable flag so a
  // buildable-mode harness never selects an unemittable mixed split (default true = analytic).
  const bool can_split = output_is_cube && num_matmuls == 1 && prob_->allow_model_ahead_split_k;
  const double sink_store_bytes = l0c_gm_bytes;  // single-matmul cube sink: L0C->GM == the sink store
  struct MixEval { double wall, ddr, max_stage, eff_cube; };
  auto eval_S = [&](int64_t S) -> MixEval {
    const double eff_cube = std::min((double)num_tiles * (double)S, n_units);
    // Cube compute makespan (busiest core), split-K aware (LptMakespan ksplit = S; the wave arm
    // divides num_tiles*S units). Uniform-grid LptMakespan reduces exactly to WaveComputeCycles.
    const double cube_stage = grid_mode
        ? LptMakespan((int64_t)n_units, g_pm, g_pn, cube_region_work, S)
        : WaveComputeCycles(base_cube_work, num_tiles * S, (int64_t)n_units);
    const double one_cube_tile = (base_cube_work / (double)num_tiles) / std::min((double)S, n_units);
    const double gm_l1_lat  = gm_l1_bytes * bc.reload / par(eff_cube, prob_->bw_gm_l1);
    const double l0c_gm_lat = (l0c_gm_bytes + (double)(S - 1) * sink_store_bytes) * bc.store
                              / par(eff_cube, prob_->bw_l0c_gm);
    const double ddr = std::max({gm_l1_lat, gm_ub_lat, l0c_gm_lat, ub_gm_lat});
    // Double-buffer floor (ported from the base cube path's db_roofline): the cube compute
    // hides its GM reload only with >=2 K-fractals per core (per_core_K = output_K_/S >= 32).
    // A thin-K matmul can't ping-pong, so compute SERIALIZES with the cube's GM ports
    // max(feed, write) -- matching the base and the emit's implicit halving. K>=32 => no-op.
    const double cube_dram = std::max(gm_l1_lat, l0c_gm_lat);
    const bool cube_db = output_K_ <= 1 || (output_K_ / std::max<int64_t>(1, S)) >= 32;
    const double cube_wall = cube_db ? cube_stage : cube_stage + cube_dram;
    // A single-tile kernel has nothing to skew against (mixed_tile_study NT=1: overlap_factor
    // 0.00), so a 3-stage fill is NOT absorbed there -- take the symmetric cross-term, which at
    // num_tiles==1 collapses to the sequential sum (cube+vec). Absorb (plain max) only NT>=2.
    const bool overlap_ok = !two_stage && num_tiles >= 2;
    const double wall = overlap_ok
        ? std::max({cube_wall, vec_stage, ddr})
        : std::max({cube_wall + one_vec_tile, vec_stage + one_cube_tile, ddr});
    return {wall, ddr, std::max(cube_stage, vec_stage), eff_cube};
  };
  MixEval best = eval_S(1);
  int64_t best_S = 1;
  if (can_split) {
    const int64_t kfrac = std::max<int64_t>(1, output_K_ / 16);
    for (int64_t S : all_divisors(kfrac)) {
      if (S < 2 || output_K_ / S < 32) continue;  // need >=2 K-fractals/partial to ping-pong
      const MixEval e = eval_S(S);
      if (e.wall < best.wall - 1e-9) { best = e; best_S = S; }
    }
  }
  result.latency        = best.wall;
  result.ddr_traffic    = best.ddr;
  result.compute_bound  = best.max_stage >= best.ddr;
  result.parallel_split = (int)best_S;
  result.uses_model_ahead_split_k = (best_S > 1);
  result.cores_used     = (int)(best.eff_cube + 2.0 * eff_units);  // cube (split) + 2 vector/tile

  // One kernel fill per unit-round (rounds = ceil(num_tiles / n_units)) — the per-LAUNCH
  // fill, a separate concern from the pipeline term above.
  if (prob_->kernel_fill_cost > 0) {
    const int64_t n_units_i = std::max<int64_t>(1, prob_->num_cube_cores);
    const int64_t rounds = (num_tiles + n_units_i - 1) / n_units_i;
    result.latency += (double)rounds * (double)prob_->kernel_fill_cost;
  }

  // Emitted per-core cube k: the single-core seq-k derived for the sink matmul
  // (same derivation as the homogeneous cube), so the lowered kernel and the
  // visualised tile carry the real contraction chunk, not the candidate cfg.k.
  std::vector<int64_t> pk;
  derive_exec(cfg, output_K_, retained_from_prev, retain_these, &pk);
  if (sink_mm_op_ >= 0 && (size_t)sink_mm_op_ < pk.size() && pk[sink_mm_op_] > 0)
    result.config.k = pk[sink_mm_op_];
  return result;
}

// ============================================================================
// Granularity enumeration
// ============================================================================

CostResult Ascend910BCost::best_cost(const FlatSet<size_t> &retained_from_prev,
                               const FlatSet<size_t> &retain_these) const {
  CostResult best;
  auto consider = [&](const TileConfig &cfg) {
        if (!is_valid_tiling(cfg)) return;
        if (!fits_on_chip(cfg, retained_from_prev, retain_these)) return;
        auto r = compute_cost(cfg, retained_from_prev, retain_these);
        if (!r.feasible) return;
        bool take;
        if (best.latency == std::numeric_limits<double>::infinity()) {
          take = true;
        } else {
          // Lexicographic tiebreak among equal-latency tiles:
          //   1. fewer split-K partials — a balanced spatial grid that fills the
          //      cores at the same latency beats a K-split (less merge traffic,
          //      no atomic-add serialization, simpler emit). split-K only wins
          //      when it is STRICTLY faster (caught by the latency test above).
          //   2. lower DDR traffic   — matmul reuse (less reload); flat for PW
          //   3. more cores used     — fill the unit's cores (parallelism)
          //   4. EVENLY-DIVIDING tile — a grid whose +-1-fractal region extents do
          //      NOT evenly divide the output (e.g. h=1376 on a 4096 axis) the tiling
          //      emit cannot realize cleanly; at equal latency prefer a tile whose
          //      extents evenly divide the output. An imbalanced grid is used only
          //      where it is strictly faster (power-of-two / few-row fills).
          //   5. larger tile area    — best vectorization / least per-tile
          //      overhead (avoids the degenerate 1xN / 16x16 picks)
          //   6. larger k            — fewer L1 passes
          const double tol = 1e-6 * std::max(1.0, best.latency);
          const double dtol = 1e-9 * std::max(1.0, best.ddr_traffic);
          const double etol = 1e-9 * std::max(1.0, best.l1l0_extract);
          const long long ra = (long long)r.config.w * r.config.h;
          const long long ba = (long long)best.config.w * best.config.h;
          // Emit-friendliness: a tile whose extents EVENLY DIVIDE the output (every
          // region identical) lowers cleanly; a grid's +-1-block extents (e.g.
          // h=1366 on a 4096 axis) the tiling emit can't realize. So at equal
          // latency prefer a dividing tile -- the imbalanced grid is used only when
          // it is strictly faster (the power-of-two / few-row fills that have no
          // dividing C-tiling). Uniform tiles always divide; a grid sometimes does.
          const bool r_div = (out_W_ % std::max<int64_t>(1, r.config.w) == 0) &&
                             (out_H_ % std::max<int64_t>(1, r.config.h) == 0);
          const bool b_div = (out_W_ % std::max<int64_t>(1, best.config.w) == 0) &&
                             (out_H_ % std::max<int64_t>(1, best.config.h) == 0);
          if (r.latency < best.latency - tol) {
            take = true;
          } else if (r.latency > best.latency + tol) {
            take = false;
          } else if (r.parallel_split != best.parallel_split) {
            take = r.parallel_split < best.parallel_split;
          } else if (r.ddr_traffic < best.ddr_traffic - dtol) {
            take = true;
          } else if (r.ddr_traffic > best.ddr_traffic + dtol) {
            take = false;
          } else if (std::abs(r.l1l0_extract - best.l1l0_extract) > etol) {
            // Lower L1->L0 extract (MTE1) wins: the GM reload is port-symmetric and
            // ties transposes, but the L0A/L0B ports are not, so the TALL tile (large
            // h, less slow-L0B traffic) is faster. Perf-sim-driven (pto-isa
            // gml1_decision: removes a ~4% aspect regret); device eval pending.
            take = r.l1l0_extract < best.l1l0_extract;
          } else if (r.cores_used != best.cores_used) {
            take = r.cores_used > best.cores_used;
          } else if (r_div != b_div) {
            take = r_div;  // emit-friendly evenly-dividing tile beats an imbalanced grid
          } else if (ra != ba) {
            take = ra > ba;
          } else {
            take = r.config.k > best.config.k;
          }
        }
        if (take) best = r;
  };

  // GRID-ONLY (910B, cube AND vector). Feasibility is via derive_exec /
  // vector_stream (the cube seq-k fits L1, the vector streams UB, regardless of the
  // spatial tile), so the SpatialSchedule grid -- including the (1,1) whole-output
  // region -- covers every fill; uniform exact-divisor tiles are redundant.
  // SpatialSchedule grids: each (parts_m, parts_n) lands a balanced region count
  // the uniform tiles can't (e.g. exactly n_cores). w,h carry the physical (max)
  // region extent so fits_on_chip / reload are unchanged.
  // One config per (parts_m, parts_n, split_k) triple. The single-core seq-k is
  // NOT enumerated: grid_k is just a structurally-valid placeholder (largest k
  // divisor). fits_on_chip -> derive_exec then DERIVES the greedy L1-fitting
  // per-op k and returns infeasible (INT64_MAX) iff NO such k exists -- so a
  // triple is only accepted when a memory-fitting k EXISTS, and that derived k is
  // what compute_cost writes back to result.config.k for the emit. The parallel
  // split is the triple's split_k, evaluated by compute_cost as a fixed S.
  const int64_t grid_k = ks_cand_.empty() ? std::max<int64_t>(output_K_, 1) : ks_cand_.back();
  for (const auto &g : grid_cand_) {
    const AxisPartition pm = partition_axis(out_H_, g.parts_m, grid_gran_h_);
    const AxisPartition pn = partition_axis(out_W_, g.parts_n, grid_gran_w_);
    consider(TileConfig{pn.big, pm.big, grid_k, pm.parts, pn.parts, g.split_k});
  }
  return best;
}