#include "io/io.h"
#include "core/types.h"
#include "solution/solution.h"
#include <fstream>
#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static const char* vector_stream_kind_name(VectorStreamKind kind) {
    switch (kind) {
        case VectorStreamKind::Materialized: return "materialized";
        case VectorStreamKind::Pointwise: return "pointwise";
        case VectorStreamKind::ReductionFolded: return "reduction_folded";
        case VectorStreamKind::ReductionSpanning: return "reduction_spanning";
        case VectorStreamKind::SoftmaxFlash: return "softmax_flash";
        case VectorStreamKind::LayerNormWelford: return "layernorm_welford";
        case VectorStreamKind::ModelAheadMultiReduction: return "model_ahead_multi_reduction";
    }
    return "unknown";
}

static json vector_loop_json(const VectorLoopPlan& loop) {
    return {{"first_chunk", loop.first_chunk},
            {"trip_count", loop.trip_count},
            {"pipeline_stages", loop.pipeline_stages}};
}

static const char* cube_axis_binding_name(CubeAxisBinding binding) {
    switch (binding) {
        case CubeAxisBinding::Full: return "full";
        case CubeAxisBinding::SpatialM: return "spatial_m";
        case CubeAxisBinding::SpatialN: return "spatial_n";
        case CubeAxisBinding::ParallelK: return "parallel_k";
        case CubeAxisBinding::SequentialK: return "sequential_k";
    }
    return "unknown";
}

static json cube_region_json(const CubeTensorRegionPlan& region) {
    return {{"tensor", region.tensor},
            {"height_binding", cube_axis_binding_name(region.height_binding)},
            {"width_binding", cube_axis_binding_name(region.width_binding)},
            {"height", region.height},
            {"width", region.width}};
}

static json cube_k_loop_json(const CubeKLoopPlan& loop) {
    return {{"l1_window_k", loop.l1_window_k},
            {"chunk", loop.chunk},
            {"full_chunks", loop.full_chunks},
            {"tail", loop.tail},
            {"pipeline_stages", loop.pipeline_stages}};
}

Problem read_problem(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open '" << filename << "'\n";
        std::exit(1);
    }

    json j;
    try {
        j = json::parse(f);
    } catch (const json::parse_error& e) {
        std::cerr << "Error: failed to parse '" << filename << "': " << e.what() << "\n";
        std::exit(1);
    }

    // Validate required top-level keys exist before accessing them.
    for (const char* key : {"widths", "heights", "inputs", "outputs",
                             "op_types",
                             "fast_memory_capacity", "cube_freq_hz"}) {
        if (!j.contains(key)) {
            std::cerr << "Error: missing required field '" << key
                      << "' in '" << filename << "'\n";
            std::exit(1);
        }
    }

    Problem p;

    // --- Tensors ---
    auto& widths  = j["widths"];
    auto& heights = j["heights"];
    if (widths.size() != heights.size()) {
        std::cerr << "Error: widths and heights arrays have different lengths\n";
        std::exit(1);
    }
    for (size_t i = 0; i < widths.size(); i++)
        p.tensors.push_back({widths[i].get<int64_t>(), heights[i].get<int64_t>()});

    const size_t num_tensors = p.num_tensors();

    // Optional per-tensor dtype (default FP32) — used by the 910B byte-based
    // two-pool working set. Accepts "FP32"/"FP16"/"BF16"/"INT32"/"INT16"/
    // "INT8"/"BOOL".
    if (j.contains("dtypes")) {
        auto& dts = j["dtypes"];
        for (size_t i = 0; i < dts.size() && i < num_tensors; i++) {
            const auto& s = dts[i].get_ref<const std::string&>();
            if (s == "FP16") p.tensors[i].dtype = DType::FP16;
            else if (s == "BF16") p.tensors[i].dtype = DType::BF16;
            else if (s == "INT32") p.tensors[i].dtype = DType::INT32;
            else if (s == "INT16") p.tensors[i].dtype = DType::INT16;
            else if (s == "INT8") p.tensors[i].dtype = DType::INT8;
            else if (s == "BOOL") p.tensors[i].dtype = DType::BOOL;
            else p.tensors[i].dtype = DType::FP32;
        }
    }

    // --- Ops ---
    auto& inputs     = j["inputs"];
    auto& outputs    = j["outputs"];
    auto& op_types   = j["op_types"];

    const size_t num_ops = op_types.size();
    if (inputs.size() != num_ops || outputs.size() != num_ops) {
        std::cerr << "Error: inputs/outputs/op_types arrays have "
                     "inconsistent lengths\n";
        std::exit(1);
    }

    for (size_t i = 0; i < num_ops; i++) {
        Op op;
        const auto& type_str = op_types[i].get_ref<const std::string&>();
        if (type_str == "MatMul") {
            op.type = OpType::MatMul;
        } else if (type_str == "Pointwise") {
            op.type = OpType::Pointwise;
        } else if (type_str == "Reduction") {
            op.type = OpType::Reduction;
        } else if (type_str == "Opaque") {
            op.type = OpType::Opaque;
        } else {
            std::cerr << "Error: unknown op type '" << type_str
                      << "' for op " << i << "\n";
            std::exit(1);
        }

        for (auto& t : inputs[i]) {
            size_t idx = t.get<size_t>();
            if (idx >= num_tensors) {
                std::cerr << "Error: op " << i << " input tensor index " << idx
                          << " out of range (num_tensors=" << num_tensors << ")\n";
                std::exit(1);
            }
            op.inputs.push_back(idx);
        }
        {
            size_t idx = outputs[i][0].get<size_t>();
            if (idx >= num_tensors) {
                std::cerr << "Error: op " << i << " output tensor index " << idx
                          << " out of range (num_tensors=" << num_tensors << ")\n";
                std::exit(1);
            }
            op.outputs.push_back(idx);
        }
        p.ops.push_back(std::move(op));
    }

    // --- Tensor integrity checks ---
    // 1. Each tensor must have at most one producing op.
    // 2. Warn about isolated tensors (no producer AND no consumer).
    {
        std::vector<int> producer_op(num_tensors, -1);
        std::vector<bool> is_consumed(num_tensors, false);

        for (size_t i = 0; i < num_ops; i++) {
            { size_t t = p.ops[i].output();
                if (producer_op[t] >= 0) {
                    std::cerr << "Error: tensor " << t << " produced by both op "
                              << producer_op[t] << " and op " << i << "\n";
                    std::exit(1);
                }
                producer_op[t] = (int)i;
            }
            for (auto t : p.ops[i].inputs)
                is_consumed[t] = true;
        }

        for (size_t t = 0; t < num_tensors; t++) {
            if (producer_op[t] < 0 && !is_consumed[t]) {
                std::cerr << "Warning: tensor " << t << " ("
                          << p.tensors[t].width << "x" << p.tensors[t].height
                          << ") is isolated — not produced or consumed by any op\n";
            }
        }
    }

    // --- Hardware parameters ---
    p.fast_memory_capacity  = j["fast_memory_capacity"].get<int64_t>();

    if (p.fast_memory_capacity <= 0) {
        std::cerr << "Error: hardware parameters must be positive\n";
        std::exit(1);
    }

    // Optional 910B parallel-core fields. Absent => defaults (1/1/0) keep the
    // single-context competition behavior. Present => the parallel roofline +
    // unit-homogeneity constraint activate (cube 24 / vector 48 cores, etc.).
    if (j.contains("num_cube_cores"))   p.num_cube_cores   = j["num_cube_cores"].get<int>();
    if (j.contains("num_vector_cores")) p.num_vector_cores = j["num_vector_cores"].get<int>();
    if (j.contains("cube_capacity"))    p.cube_capacity    = j["cube_capacity"].get<int64_t>();
    if (j.contains("vec_capacity"))     p.vec_capacity     = j["vec_capacity"].get<int64_t>();
    if (j.contains("l1_capacity"))      p.l1_capacity      = j["l1_capacity"].get<int64_t>();
    if (j.contains("cube_compute_cost"))   p.cube_compute_cost   = j["cube_compute_cost"].get<int64_t>();
    if (j.contains("kernel_fill_cost"))    p.kernel_fill_cost    = j["kernel_fill_cost"].get<int64_t>();
    // Grounded pto-isa machine model (optional; absent => legacy placeholders).
    if (j.contains("cube_freq_hz")) p.cube_freq_hz = j["cube_freq_hz"].get<double>();
    if (j.contains("bw_gm_l1"))     p.bw_gm_l1     = j["bw_gm_l1"].get<double>();
    if (j.contains("bw_l0c_gm"))    p.bw_l0c_gm    = j["bw_l0c_gm"].get<double>();
    if (j.contains("bw_l1_l0a"))    p.bw_l1_l0a    = j["bw_l1_l0a"].get<double>();
    if (j.contains("bw_l1_l0b"))    p.bw_l1_l0b    = j["bw_l1_l0b"].get<double>();
    if (j.contains("bw_gm_ub"))     p.bw_gm_ub     = j["bw_gm_ub"].get<double>();
    if (j.contains("bw_ub_gm"))     p.bw_ub_gm     = j["bw_ub_gm"].get<double>();
    if (j.contains("hbm_aggregate_gibps")) p.hbm_aggregate_gibps = j["hbm_aggregate_gibps"].get<double>();
    if (j.contains("l0_tile_m"))    p.l0_tile_m    = j["l0_tile_m"].get<int64_t>();
    if (j.contains("l0_tile_n"))    p.l0_tile_n    = j["l0_tile_n"].get<int64_t>();
    if (j.contains("vec_reg_bytes"))    p.vec_reg_bytes    = j["vec_reg_bytes"].get<int64_t>();
    if (j.contains("vec_op_head"))      p.vec_op_head      = j["vec_op_head"].get<double>();
    if (j.contains("vec_op_tail"))      p.vec_op_tail      = j["vec_op_tail"].get<double>();
    if (j.contains("vec_slope_pw"))     p.vec_slope_pw     = j["vec_slope_pw"].get<double>();
    if (j.contains("vec_slope_reduce")) p.vec_slope_reduce = j["vec_slope_reduce"].get<double>();

    // -------------------------------------------------------------------------
    // Precompute retainable_tensors.
    //
    // A tensor is retainable across subgraph boundaries if ALL of:
    //   1. Its full size fits in fast memory.
    //      (A tensor larger than capacity can never be legally pinned.)
    //   2. It has at least one consuming op.
    //      (Graph outputs have no consumers — they are evicted at the end and
    //      never read by a later subgraph, so retaining them is pointless.)
    //
    // Graph inputs with a single consumer ARE included. Although in a simple
    // linear schedule they are read once and discarded, recomputation-based
    // strategies (e.g. diamond graphs like Example 3B) may place the same
    // graph input in two separate subgraphs. Retaining it after the first
    // subgraph eliminates the redundant reload in the second.
    //
    // The ordering layer is responsible for deciding whether retention is
    // actually beneficial for a given schedule; this set is the permissive
    // upper bound of candidates.
    // -------------------------------------------------------------------------
    // 910B: NO cross-subgraph retention. Each subgraph runs across the cores; data
    // crossing a subgraph boundary (incl. the cube<->vector handoff) routes through
    // DDR/GM, and there is no shared fast memory that persists a tensor across
    // subgraph executions (L1/L0c and UB are per-core, transient). So
    // retainable_tensors stays empty — the per-core working set never pins a
    // cross-subgraph tensor.
    // -------------------------------------------------------------------------

    return p;
}

void write_solution(const std::string& filename, const Solution& sol) {
    json j;
    j["subgraphs"]         = json::array();
    j["granularities"]     = json::array();
    j["parts"]             = json::array();  // 910B spatial grid (parts_m, parts_n) per step
    j["splits"]            = json::array();  // 910B parallel split-K / reduction split per step
    j["cores"]             = json::array();  // 910B cube/vector cores used per step
    j["op_order"]          = json::array();  // DFS execution order per step (pebble order)
    j["seq_k"]             = json::array();  // 910B per-op single-core k-tile (in op_order)
    j["vector_stream"]     = json::array();  // solver-owned vector UB sub-stream plan per step
    j["cube_schedule"]     = json::array();  // solver-owned cube grid/band/K-loop plan per step
    j["tensors_to_retain"] = json::array();
    j["subgraph_latencies"] = json::array();

    for (size_t i = 0; i < sol.num_steps(); i++) {
        const auto& step = sol.step(i);
        const auto& cfg  = step.config;
        const auto& cost = sol.step_cost(i);
        // Emit descriptors are reconstructed only for final solution steps. The
        // hot local-search CostResult cache intentionally stores no stream plan.
        const VectorStreamPlan vector_plan =
            !step.subgraph.has_matmul()
                ? step.subgraph.vector_stream_plan(
                      cfg, sol.retained_entering(i), step.retain_these)
                : VectorStreamPlan{};
        const CubeSchedulePlan cube_plan =
            step.subgraph.has_matmul()
                ? step.subgraph.cube_schedule_plan(
                      cfg, sol.retained_entering(i), step.retain_these,
                      cost.parallel_split)
                : CubeSchedulePlan{};

        j["subgraphs"].push_back(step.subgraph.ops());
        j["granularities"].push_back({cfg.w, cfg.h, cfg.k});
        // Spatial grid shape: parts_m x parts_n regions (0,0 = a uniform tile,
        // region count = floor(out_W/w)*floor(out_H/h)). w,h above are the MAX
        // region extent (regions differ by <=1 block), NOT a uniform tile -- read
        // the region count from here, do not infer it from the tile size.
        j["parts"].push_back({cfg.parts_m, cfg.parts_n});
        // Parallel split + cores used for this tile (cube split-K / vector
        // reduction split; 1 = pure spatial). Computed for the chosen config.
        j["splits"].push_back(cost.parallel_split);
        j["cores"].push_back(cost.cores_used);
        // Execution order (the fixed pebbling order) — emitted because the peak
        // working set depends on it, so downstream must materialize this order.
        {
            const auto& order = step.subgraph.execution_order();
            j["op_order"].push_back(std::vector<size_t>(order.begin(), order.end()));
            // Per-op single-core k-tile, in execution order (cube-910B only). An
            // op's seq_k = its full K means it ran the contraction in one pass.
            const auto& prob = step.subgraph.problem();
            if (step.subgraph.has_matmul() && prob.num_cube_cores > 1 &&
                prob.l1_capacity > 0) {
                std::vector<int64_t> pk;
                step.subgraph.cube_peak_l1(cfg, &pk);  // L1-fit per-op (single core)
                const int64_t sink = step.subgraph.sink_matmul_op();
                std::vector<int64_t> ks;
                for (auto op : order)
                    // Sink: the composed per-core k (L1-fit capped by the split-K
                    // share, = granularities.k). Internals: the L1-fit single-core k.
                    ks.push_back((int64_t)op == sink ? cost.config.k
                                 : ((size_t)op < pk.size() ? pk[op] : 0));
                j["seq_k"].push_back(ks);
            } else if (!step.subgraph.has_matmul() && prob.num_vector_cores > 1 &&
                       prob.vec_capacity > 0) {
                // Vector single-core k-stream: one subgraph-wide chunk along the
                // streamed axis (the matmul-seq-k analog). Emit it per op (same value).
                const int64_t chunk = vector_plan.streamed() ? vector_plan.chunk : 0;
                j["seq_k"].push_back(std::vector<int64_t>(order.size(), chunk));
            } else {
                j["seq_k"].push_back(nullptr);
            }
        }
        if (vector_plan.feasible) {
            const VectorStreamPlan& plan = vector_plan;
            j["vector_stream"].push_back(
                {{"kind", vector_stream_kind_name(plan.kind)},
                 {"full_peak_ub_bytes", plan.full_peak_ub_bytes},
                 {"chunk_peak_ub_bytes", plan.chunk_peak_ub_bytes},
                 {"stream_band_count", plan.stream_band_count},
                 {"axis", plan.axis},
                 {"free_tile", plan.free_tile},
                 {"extent", plan.extent},
                 {"chunk", plan.chunk},
                 {"full_chunks", plan.full_chunks},
                 {"tail", plan.tail},
                 {"stream_passes", plan.stream_passes},
                 {"overlap_granted", plan.overlap_granted},
                 {"body", vector_loop_json(plan.body)},
                 {"stats", vector_loop_json(plan.stats)},
                 {"apply", vector_loop_json(plan.apply)}});
        } else {
            j["vector_stream"].push_back(nullptr);
        }
        if (cube_plan.feasible) {
            json matmuls = json::array();
            for (const auto& mm : cube_plan.matmuls) {
                matmuls.push_back(
                    {{"op", mm.op},
                     {"is_sink", mm.is_sink},
                     {"lhs_ephemeral", mm.lhs_ephemeral},
                     {"rhs_ephemeral", mm.rhs_ephemeral},
                     {"output_ephemeral", mm.output_ephemeral},
                     {"contraction", mm.contraction},
                     {"effective_contraction", mm.effective_contraction},
                     {"lhs", cube_region_json(mm.lhs)},
                     {"rhs", cube_region_json(mm.rhs)},
                     {"output", cube_region_json(mm.output)},
                     {"k_loop", cube_k_loop_json(mm.k_loop)}});
            }
            j["cube_schedule"].push_back(
                {{"emit_compatible", cube_plan.emit_compatible},
                 {"spatial_tiles", cube_plan.spatial_tiles},
                 {"split_k", cube_plan.split_k},
                 {"work_units", cube_plan.work_units},
                 {"peak_l1_bytes", cube_plan.peak_l1_bytes},
                 {"seed_required", cube_plan.seed_required},
                 {"model_overlap_granted", cube_plan.model_overlap_granted},
                 {"overlap_implementable", cube_plan.overlap_implementable},
                 {"matmuls", matmuls}});
        } else {
            j["cube_schedule"].push_back(nullptr);
        }
        j["tensors_to_retain"].push_back(
            std::vector<size_t>(step.retain_these.begin(), step.retain_these.end()));

        j["subgraph_latencies"].push_back(sol.step_latency(i));
    }

    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: cannot write '" << filename << "'\n";
        std::exit(1);
    }
    f << j.dump(2) << "\n";
    if (!f) {
        std::cerr << "Error: write failed for '" << filename << "'\n";
        std::exit(1);
    }
}
