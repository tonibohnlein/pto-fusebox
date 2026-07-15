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

static const char* vector_reduction_split_kind_name(VectorReductionSplitKind kind) {
    switch (kind) {
        case VectorReductionSplitKind::None: return "none";
        case VectorReductionSplitKind::ColSumAtomicAdd: return "col_sum_atomic_add";
    }
    return "unknown";
}

static bool parse_vector_primitive_family(const std::string& name, VectorPrimitiveFamily* family) {
    if (name == "generic") *family = VectorPrimitiveFamily::Generic;
    else if (name == "add") *family = VectorPrimitiveFamily::Add;
    else if (name == "mul") *family = VectorPrimitiveFamily::Mul;
    else if (name == "div") *family = VectorPrimitiveFamily::Div;
    else if (name == "exp") *family = VectorPrimitiveFamily::Exp;
    else if (name == "log") *family = VectorPrimitiveFamily::Log;
    else if (name == "abs") *family = VectorPrimitiveFamily::Abs;
    else if (name == "sqrt") *family = VectorPrimitiveFamily::Sqrt;
    else if (name == "rsqrt") *family = VectorPrimitiveFamily::Rsqrt;
    else if (name == "scalar_add") *family = VectorPrimitiveFamily::ScalarAdd;
    else if (name == "scalar_mul") *family = VectorPrimitiveFamily::ScalarMul;
    else if (name == "scalar_max") *family = VectorPrimitiveFamily::ScalarMax;
    else if (name == "scalar_min") *family = VectorPrimitiveFamily::ScalarMin;
    else if (name == "row_sum") *family = VectorPrimitiveFamily::RowSum;
    else if (name == "row_extrema") *family = VectorPrimitiveFamily::RowExtrema;
    else if (name == "col_sum") *family = VectorPrimitiveFamily::ColSum;
    else if (name == "col_extrema") *family = VectorPrimitiveFamily::ColExtrema;
    else if (name == "reduction") *family = VectorPrimitiveFamily::Reduction;
    else return false;
    return true;
}

static bool parse_vector_op_geometry(const std::string& name, VectorOpGeometry* geometry) {
    if (name == "generic") *geometry = VectorOpGeometry::Generic;
    else if (name == "flat") *geometry = VectorOpGeometry::Flat;
    else if (name == "row_expand") *geometry = VectorOpGeometry::RowExpand;
    else if (name == "col_expand") *geometry = VectorOpGeometry::ColExpand;
    else return false;
    return true;
}

static bool parse_vector_op_capability(const std::string& name,
                                       VectorOpCapability* capability) {
    if (name == "generic") *capability = VectorOpCapability::Generic;
    else if (name == "elementwise") *capability = VectorOpCapability::Elementwise;
    else if (name == "reduction_sum") *capability = VectorOpCapability::ReductionSum;
    else if (name == "reduction_max") *capability = VectorOpCapability::ReductionMax;
    else if (name == "unsupported") *capability = VectorOpCapability::Unsupported;
    else return false;
    return true;
}

static json vector_loop_json(const VectorLoopPlan& loop) {
    return {{"first_chunk", loop.first_chunk},
            {"trip_count", loop.trip_count},
            {"pipeline_stages", loop.pipeline_stages}};
}

static const char* vector_primitive_name(VectorPrimitiveKind kind) {
    switch (kind) {
        case VectorPrimitiveKind::Add: return "add";
        case VectorPrimitiveKind::Mul: return "mul";
        case VectorPrimitiveKind::Div: return "div";
        case VectorPrimitiveKind::Exp: return "exp";
        case VectorPrimitiveKind::RowExpandSub: return "row_expand_sub";
        case VectorPrimitiveKind::ScalarAdd: return "scalar_add";
        case VectorPrimitiveKind::ScalarMul: return "scalar_mul";
        case VectorPrimitiveKind::RowSum: return "row_sum";
        case VectorPrimitiveKind::RowMax: return "row_max";
        case VectorPrimitiveKind::Count: break;
    }
    return "unknown";
}

static json vector_phase_work_json(const VectorPhaseWorkPlan& phase) {
    json primitives = json::array();
    for (size_t i = 0; i < static_cast<size_t>(VectorPrimitiveKind::Count); ++i) {
        const auto kind = static_cast<VectorPrimitiveKind>(i);
        const auto& work = phase.primitives[i];
        if (work.wide == 0 && work.thin == 0 && work.stream_starts == 0) continue;
        primitives.push_back({{"kind", vector_primitive_name(kind)},
                              {"wide", static_cast<int64_t>(work.wide)},
                              {"thin", static_cast<int64_t>(work.thin)},
                              {"stream_starts", static_cast<int64_t>(work.stream_starts)}});
    }
    return {{"generated", phase.generated}, {"primitives", std::move(primitives)}};
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

static const char* cube_spatial_policy_name(CubeSpatialPolicy policy) {
    switch (policy) {
        case CubeSpatialPolicy::Uniform: return "uniform";
        case CubeSpatialPolicy::ClampedOverlap: return "clamped_overlap";
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

static const char* l0_stationarity_name(L0Stationarity stationarity) {
  switch (stationarity) {
    case L0Stationarity::Output:
      return "output";
    case L0Stationarity::A:
      return "a";
    case L0Stationarity::B:
      return "b";
  }
  return "unknown";
}

static const char* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::FP32:
      return "fp32";
    case DType::FP16:
      return "fp16";
    case DType::BF16:
      return "bf16";
    case DType::INT32:
      return "int32";
    case DType::INT16:
      return "int16";
    case DType::INT8:
      return "int8";
    case DType::BOOL:
      return "bool";
  }
  return "unknown";
}

static json l0_matmul_plan_json(const L0MatmulPlan& plan) {
  if (!plan.feasible) return nullptr;
  const char* target = "gm";
  if (plan.output_target == L0OutputTarget::Acc) target = "acc";
  if (plan.output_target == L0OutputTarget::L1) target = "l1";
  return {{"tile", {plan.m, plan.n, plan.k}},
          {"stationarity", l0_stationarity_name(plan.stationarity)},
          {"output_stationary_holds_a", plan.output_stationary_holds_a},
          {"buffer_depths", {plan.buffer_depth_a, plan.buffer_depth_b, plan.buffer_depth_c}},
          {"output_target", target},
          {"k_loop",
           {{"chunk", plan.k_loop.chunk},
            {"full_chunks", plan.k_loop.full_chunks},
            {"tail", plan.k_loop.tail},
            {"pipeline_stages", plan.k_loop.pipeline_stages}}},
          {"estimated_traffic_bytes", plan.estimated_traffic_bytes},
          {"estimated_cost_cycles", plan.estimated_cost_cycles},
          {"padded_compute_volume", plan.padded_compute_volume},
          {"phases",
           {{"load_cycles", plan.phases.load_cycles},
            {"mad_cycles", plan.phases.mad_cycles},
            {"init_cycles", plan.phases.init_cycles},
            {"rolled_cycles", plan.phases.rolled_cycles},
            {"tail_cycles", plan.phases.tail_cycles},
            {"drain_cycles", plan.phases.drain_cycles},
            {"wall_cycles", plan.phases.wall_cycles}}}};
}

static const char* mixed_engine_name(MixedEngine engine) {
    return engine == MixedEngine::Cube ? "cube" : "vector";
}

static const char* mixed_pipeline_mode_name(MixedPipelineMode mode) {
    switch (mode) {
        case MixedPipelineMode::Serial: return "serial";
        case MixedPipelineMode::OneWay: return "one_way";
        case MixedPipelineMode::SingleRoundTripSkew: return "single_round_trip_skew";
        case MixedPipelineMode::MultiRoundTripSequential: return "multi_round_trip_sequential";
    }
    return "unknown";
}

static const char* mixed_pipeline_axis_name(MixedPipelineAxis axis) {
    switch (axis) {
        case MixedPipelineAxis::SpatialRegion: return "spatial_region";
        case MixedPipelineAxis::VectorWidthChunk: return "vector_width_chunk";
        case MixedPipelineAxis::VectorHeightChunk: return "vector_height_chunk";
        case MixedPipelineAxis::AttentionKeyChunk: return "attention_key_chunk";
    }
    return "unknown";
}

static const char* mixed_vector_split_name(MixedVectorSplit split) {
    switch (split) {
        case MixedVectorSplit::None: return "none";
        case MixedVectorSplit::Rows: return "rows";
        case MixedVectorSplit::Columns: return "columns";
    }
    return "unknown";
}

static const char* mixed_transfer_direction_name(MixedTransferDirection direction) {
    return direction == MixedTransferDirection::CubeToVector ? "cube_to_vector"
                                                              : "vector_to_cube";
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
#ifdef PYPTO_FUSE_CUBE_VECTOR
    // Preserve the research executable's historical compile-time opt-in while
    // the production solver uses the same policy as a runtime Problem field.
    p.fuse_cube_vector = true;
#endif

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
        if (j.contains("vec_slopes") && i < j["vec_slopes"].size())
            op.vec_slope = j["vec_slopes"][i].get<double>();
        if (j.contains("vec_fixed_costs") && i < j["vec_fixed_costs"].size())
            op.vec_fixed = j["vec_fixed_costs"][i].get<double>();
        if (j.contains("vector_primitive_families") && i < j["vector_primitive_families"].size()) {
            const std::string name = j["vector_primitive_families"][i].get<std::string>();
            if (!parse_vector_primitive_family(name, &op.vector_primitive)) {
                std::cerr << "Error: unknown vector primitive family '" << name
                          << "' for op " << i << "\n";
                std::exit(1);
            }
        }
        if (j.contains("vector_op_geometries") && i < j["vector_op_geometries"].size()) {
            const std::string name = j["vector_op_geometries"][i].get<std::string>();
            if (!parse_vector_op_geometry(name, &op.vector_geometry)) {
                std::cerr << "Error: unknown vector op geometry '" << name
                          << "' for op " << i << "\n";
                std::exit(1);
            }
        }
        if (j.contains("vector_op_capabilities") &&
            i < j["vector_op_capabilities"].size()) {
            const std::string name = j["vector_op_capabilities"][i].get<std::string>();
            if (!parse_vector_op_capability(name, &op.vector_capability)) {
                std::cerr << "Error: unknown vector op capability '" << name
                          << "' for op " << i << "\n";
                std::exit(1);
            }
        }
        p.ops.push_back(std::move(op));
    }

    if (j.contains("required_outputs")) {
        for (const auto& value : j["required_outputs"]) {
            const size_t tensor = value.get<size_t>();
            if (tensor >= num_tensors) {
                std::cerr << "Error: required output tensor index " << tensor
                          << " out of range (num_tensors=" << num_tensors << ")\n";
                std::exit(1);
            }
            p.required_outputs.insert(tensor);
        }
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
    if (j.contains("per_task_overhead_cycles")) {
        p.per_task_overhead_cycles = j["per_task_overhead_cycles"].get<int64_t>();
    }
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
    if (j.contains("l0_matmul_config")) {
      const auto& l0 = j["l0_matmul_config"];
      auto& config = p.l0_matmul_config;
      if (l0.contains("l0a_bytes")) config.l0a_bytes = l0["l0a_bytes"].get<int64_t>();
      if (l0.contains("l0b_bytes")) config.l0b_bytes = l0["l0b_bytes"].get<int64_t>();
      if (l0.contains("l0c_bytes")) config.l0c_bytes = l0["l0c_bytes"].get<int64_t>();
      if (l0.contains("min_m")) config.min_m = l0["min_m"].get<int64_t>();
      if (l0.contains("min_n")) config.min_n = l0["min_n"].get<int64_t>();
      if (l0.contains("min_k")) config.min_k = l0["min_k"].get<int64_t>();
      if (l0.contains("align_m")) config.align_m = l0["align_m"].get<int64_t>();
      if (l0.contains("align_n")) config.align_n = l0["align_n"].get<int64_t>();
      if (l0.contains("align_k")) config.align_k = l0["align_k"].get<int64_t>();
      if (l0.contains("allow_a_stationary")) {
        config.allow_a_stationary = l0["allow_a_stationary"].get<bool>();
      }
      if (l0.contains("allow_b_stationary")) {
        config.allow_b_stationary = l0["allow_b_stationary"].get<bool>();
      }
      if (l0.contains("allow_double_buffer_c")) {
        config.allow_double_buffer_c = l0["allow_double_buffer_c"].get<bool>();
      }
      if (l0.contains("allow_padding")) {
        config.allow_padding = l0["allow_padding"].get<bool>();
      }
      if (l0.contains("allow_k_boundary")) {
        config.allow_k_boundary = l0["allow_k_boundary"].get<bool>();
      }
      if (l0.contains("bw_l0a")) config.bw_l0a = l0["bw_l0a"].get<double>();
      if (l0.contains("bw_l0b")) config.bw_l0b = l0["bw_l0b"].get<double>();
      if (l0.contains("bw_drain")) config.bw_drain = l0["bw_drain"].get<double>();
      if (l0.contains("bw_l0c_l1")) config.bw_l0c_l1 = l0["bw_l0c_l1"].get<double>();
      if (l0.contains("drain_fixed_cycles")) {
        config.drain_fixed_cycles = l0["drain_fixed_cycles"].get<double>();
      }
      if (l0.contains("drain_row_cycles")) {
        config.drain_row_cycles = l0["drain_row_cycles"].get<double>();
      }
      if (l0.contains("drain_penalty_cycles")) {
        config.drain_penalty_cycles = l0["drain_penalty_cycles"].get<double>();
      }
      if (l0.contains("drain_c0_bytes")) {
        config.drain_c0_bytes = l0["drain_c0_bytes"].get<int64_t>();
      }
      if (l0.contains("mad_head_cycles")) {
        config.mad_head_cycles = l0["mad_head_cycles"].get<int64_t>();
      }
      if (l0.contains("mad_k_fractal_bytes")) {
        config.mad_k_fractal_bytes = l0["mad_k_fractal_bytes"].get<int64_t>();
      }
    }
    if (j.contains("vec_reg_bytes"))    p.vec_reg_bytes    = j["vec_reg_bytes"].get<int64_t>();
    if (j.contains("vec_op_head"))      p.vec_op_head      = j["vec_op_head"].get<double>();
    if (j.contains("vec_op_tail"))      p.vec_op_tail      = j["vec_op_tail"].get<double>();
    if (j.contains("vec_slope_pw"))     p.vec_slope_pw     = j["vec_slope_pw"].get<double>();
    if (j.contains("vec_slope_reduce")) p.vec_slope_reduce = j["vec_slope_reduce"].get<double>();
    if (j.contains("require_uniform_cube_dag_grid")) {
      p.require_uniform_cube_dag_grid = j["require_uniform_cube_dag_grid"].get<bool>();
    }
    if (j.contains("use_hierarchical_cube_cost")) {
      p.use_hierarchical_cube_cost = j["use_hierarchical_cube_cost"].get<bool>();
    }
    if (j.contains("fuse_cube_vector")) {
      p.fuse_cube_vector = j["fuse_cube_vector"].get<bool>();
    }
    if (j.contains("require_buildable_mixed")) {
      p.require_buildable_mixed = j["require_buildable_mixed"].get<bool>();
    }
    if (j.contains("allow_model_ahead_mixed_multi_roundtrip")) {
      p.allow_model_ahead_mixed_multi_roundtrip =
          j["allow_model_ahead_mixed_multi_roundtrip"].get<bool>();
    }

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
    j["mixed_schedule"]    = json::array();  // solver-owned cross-engine stage/FIFO/loop plan
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
            step.subgraph.has_matmul() && !step.subgraph.is_mixed()
                ? step.subgraph.cube_schedule_plan(
                      cfg, sol.retained_entering(i), step.retain_these,
                      cost.parallel_split)
                : CubeSchedulePlan{};
        const MixedSchedulePlan mixed_plan =
            step.subgraph.is_mixed()
                ? step.subgraph.mixed_schedule_plan(
                      cfg, sol.retained_entering(i), step.retain_these,
                      cost.parallel_split)
                : MixedSchedulePlan{};

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
                 {"reduction_split",
                  {{"kind", vector_reduction_split_kind_name(plan.reduction_split_kind)},
                   {"factor", plan.reduction_split_factor},
                   {"partial_extent", plan.reduction_partial_extent}}},
                 {"body", vector_loop_json(plan.body)},
                 {"stats", vector_loop_json(plan.stats)},
                 {"apply", vector_loop_json(plan.apply)},
                 {"p4_work",
                  {{"generated", plan.p4_work.generated},
                   {"stats_init", vector_phase_work_json(plan.p4_work.stats_init)},
                   {"stats_update", vector_phase_work_json(plan.p4_work.stats_update)},
                   {"finalize", vector_phase_work_json(plan.p4_work.finalize)}}}});
        } else {
            j["vector_stream"].push_back(nullptr);
        }
        if (cube_plan.feasible) {
            json matmuls = json::array();
            for (const auto& mm : cube_plan.matmuls) {
              json variants = json::array();
              for (const auto& variant : mm.output_variants) {
                variants.push_back({{"shape", {variant.height, variant.width}},
                                    {"count", variant.count},
                                    {"l0_init", l0_matmul_plan_json(variant.l0_init)},
                                    {"l0_rolled", l0_matmul_plan_json(variant.l0_rolled)},
                                    {"l0_tail", l0_matmul_plan_json(variant.l0_tail)}});
              }
              matmuls.push_back({{"instance", mm.instance},
                                 {"op", mm.op},
                                 {"lhs_producer", mm.lhs_producer},
                                 {"rhs_producer", mm.rhs_producer},
                                 {"is_sink", mm.is_sink},
                                 {"lhs_ephemeral", mm.lhs_ephemeral},
                                 {"rhs_ephemeral", mm.rhs_ephemeral},
                                 {"output_ephemeral", mm.output_ephemeral},
                                 {"contraction", mm.contraction},
                                 {"effective_contraction", mm.effective_contraction},
                                 {"accumulator_dtype", dtype_name(mm.accumulator_dtype)},
                                 {"storage_dtype", dtype_name(mm.storage_dtype)},
                                 {"lhs", cube_region_json(mm.lhs)},
                                 {"rhs", cube_region_json(mm.rhs)},
                                 {"output", cube_region_json(mm.output)},
                                 {"k_loop", cube_k_loop_json(mm.k_loop)},
                                 {"output_tile", {mm.output_tile_m, mm.output_tile_n}},
                                 {"output_grid", {mm.output_tiles_m, mm.output_tiles_n}},
                                 {"output_variants", std::move(variants)},
                                 {"final_drain",
                                  {{"required", mm.final_drain.required},
                                   {"target_l1", mm.final_drain.target_l1},
                                   {"atomic", mm.final_drain.atomic},
                                   {"valid_rows", mm.final_drain.valid_rows},
                                   {"valid_cols", mm.final_drain.valid_cols},
                                   {"tile_count", mm.final_drain.tile_count},
                                   {"bytes", mm.final_drain.bytes},
                                   {"cycles", mm.final_drain.cycles}}}});
            }
            j["cube_schedule"].push_back({{"emit_compatible", cube_plan.emit_compatible},
                                          {"spatial_policy", cube_spatial_policy_name(cube_plan.spatial_policy)},
                                          {"spatial_tiles", cube_plan.spatial_tiles},
                                          {"split_k", cube_plan.split_k},
                                          {"work_units", cube_plan.work_units},
                                          {"peak_l1_bytes", cube_plan.peak_l1_bytes},
                                          {"seed_required", cube_plan.seed_required},
                                          {"seed",
                                           {{"present", cube_plan.seed.present},
                                            {"work_units", cube_plan.seed.work_units},
                                            {"valid_rows", cube_plan.seed.valid_rows},
                                            {"valid_cols", cube_plan.seed.valid_cols},
                                            {"bytes", cube_plan.seed.bytes}}},
                                          {"model_overlap_granted", cube_plan.model_overlap_granted},
                                          {"overlap_implementable", cube_plan.overlap_implementable},
                                          {"matmuls", matmuls}});
        } else {
            j["cube_schedule"].push_back(nullptr);
        }
        if (mixed_plan.feasible && mixed_plan.topology) {
            json stages = json::array();
            for (const auto& stage : mixed_plan.topology->stages) {
                stages.push_back({{"engine", mixed_engine_name(stage.engine)},
                                  {"ops", stage.ops}});
            }
            json transfers = json::array();
            for (const auto& transfer : mixed_plan.topology->transfers) {
                transfers.push_back(
                    {{"tensor", transfer.tensor},
                     {"producer_stage", transfer.producer_stage},
                     {"consumer_stage", transfer.consumer_stage},
                     {"producer_engine", mixed_engine_name(transfer.producer_engine)},
                     {"consumer_engine", mixed_engine_name(transfer.consumer_engine)}});
            }
            json fifos = json::array();
            for (const auto& fifo : mixed_plan.fifos) {
                fifos.push_back(
                    {{"tensor", fifo.tensor},
                     {"direction", mixed_transfer_direction_name(fifo.direction)},
                     {"valid_rows", fifo.valid_rows},
                     {"valid_cols", fifo.valid_cols},
                     {"slot_bytes", fifo.slot_bytes},
                     {"slot_count", fifo.slot_count},
                     {"reserved_bytes", fifo.reserved_bytes}});
            }
            j["mixed_schedule"].push_back(
                {{"emit_compatible", mixed_plan.emit_compatible},
                 {"mode", mixed_pipeline_mode_name(mixed_plan.mode)},
                 {"spatial_tiles", mixed_plan.spatial_tiles},
                 {"split_k", mixed_plan.split_k},
                 {"work_units", mixed_plan.work_units},
                 {"group_capacity", mixed_plan.group_capacity},
                 {"cube_window_k", mixed_plan.cube_window_k},
                 {"vector_stage_kind",
                  vector_stream_kind_name(mixed_plan.vector_stage_kind)},
                 {"vector_stage_peak_ub_bytes",
                  mixed_plan.vector_stage_peak_ub_bytes},
                 {"vector_split", mixed_vector_split_name(mixed_plan.vector_split)},
                 {"vector_lanes", mixed_plan.vector_lanes},
                 {"pipeline_axis", mixed_pipeline_axis_name(mixed_plan.loop.axis)},
                 {"pipeline_extent", mixed_plan.loop.extent},
                 {"pipeline_chunk", mixed_plan.loop.chunk},
                 {"items_per_spatial_tile", mixed_plan.loop.items_per_spatial_tile},
                 {"active_groups", mixed_plan.loop.active_groups},
                 {"min_trips_per_group", mixed_plan.loop.min_trips_per_group},
                 {"max_trips_per_group", mixed_plan.loop.max_trips_per_group},
                 {"pipeline_stages", mixed_plan.loop.pipeline_stages},
                 {"requested_skew_depth", mixed_plan.loop.requested_skew_depth},
                 {"model_overlap_granted", mixed_plan.model_overlap_granted},
                 {"overlap_implementable", mixed_plan.overlap_implementable},
                 {"pipeline_fill_absorbed", mixed_plan.pipeline_fill_absorbed},
                 {"max_alternations", mixed_plan.topology->max_alternations},
                 {"output_engines_uniform", mixed_plan.topology->output_engines_uniform},
                 {"stages", stages},
                 {"transfers", transfers},
                 {"fifos", fifos}});
        } else {
            j["mixed_schedule"].push_back(nullptr);
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
