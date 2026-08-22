#include "io/io.h"
#include "core/types.h"
#include "solution/solution.h"
#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

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
    else if (name == "cast") *family = VectorPrimitiveFamily::Cast;
    else if (name == "recip") *family = VectorPrimitiveFamily::Recip;
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

static bool parse_mixed_vector_semantic(const std::string& name, MixedVectorSemantic* semantic) {
  if (name == "none")
    *semantic = MixedVectorSemantic::None;
  else if (name == "neg")
    *semantic = MixedVectorSemantic::Neg;
  else if (name == "exp")
    *semantic = MixedVectorSemantic::Exp;
  else if (name == "scalar_add")
    *semantic = MixedVectorSemantic::ScalarAdd;
  else if (name == "recip")
    *semantic = MixedVectorSemantic::Recip;
  else if (name == "mul")
    *semantic = MixedVectorSemantic::Mul;
  else if (name == "cast")
    *semantic = MixedVectorSemantic::Cast;
  else
    return false;
  return true;
}

static bool parse_p4_pattern_kind(const std::string& name, P4PatternKind* kind) {
  if (name == "softmax_flash")
    *kind = P4PatternKind::SoftmaxFlash;
  else if (name == "layernorm_welford")
    *kind = P4PatternKind::LayerNormWelford;
  else
    return false;
  return true;
}

static bool parse_p4_substitution_value(const std::string& name,
                                        P4SubstitutionValue* value) {
  if (name == "running_max")
    *value = P4SubstitutionValue::RunningMax;
  else if (name == "running_sum")
    *value = P4SubstitutionValue::RunningSum;
  else if (name == "mean")
    *value = P4SubstitutionValue::Mean;
  else if (name == "variance")
    *value = P4SubstitutionValue::Variance;
  else
    return false;
  return true;
}

static const char* p4_substitution_value_name(P4SubstitutionValue value) {
  switch (value) {
    case P4SubstitutionValue::RunningMax: return "running_max";
    case P4SubstitutionValue::RunningSum: return "running_sum";
    case P4SubstitutionValue::Mean: return "mean";
    case P4SubstitutionValue::Variance: return "variance";
  }
  return "unknown";
}

static json vector_loop_json(const VectorLoopPlan& loop) {
    return {{"first_chunk", loop.first_chunk},
            {"trip_count", loop.trip_count},
            {"pipeline_stages", loop.pipeline_stages}};
}

static json vector_serial_phase_json(const VectorSerialPhasePlan& phase) {
    return {{"present", phase.present},
            {"chunk_index", phase.chunk_index},
            {"extent", phase.extent}};
}

static json axis_partition_json(const AxisPartition& partition) {
    return {{"big", partition.big},
            {"small", partition.small},
            {"num_big", partition.num_big},
            {"parts", partition.parts}};
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

static const char* vector_replay_phase_name(VectorReplayPhase phase) {
    switch (phase) {
        case VectorReplayPhase::Body: return "body";
        case VectorReplayPhase::Stats: return "stats";
        case VectorReplayPhase::Apply: return "apply";
        case VectorReplayPhase::Finalize: return "finalize";
    }
    return "unknown";
}

static json vector_replay_phases_json(const Problem& problem,
                                      const VectorStreamPlan& plan) {
    json result = json::array();
    const auto tensor_frames = BuildVectorTensorFrames(problem, plan);
    const auto workspace_frames = BuildVectorWorkspaceFrames(problem, plan);
    for (size_t index = 0; index < 4; ++index) {
        const auto phase = static_cast<VectorReplayPhase>(index);
        json lifetimes = json::array();
        if (plan.input_lifetimes) {
            for (const auto& lifetime : plan.input_lifetimes->phases[index]) {
                json uses = json::array();
                for (const auto& use : lifetime.uses)
                    uses.push_back({{"op", use.op}, {"arg", use.arg}});
                lifetimes.push_back({{"tensor", lifetime.tensor},
                                     {"first_use_step", lifetime.first_use_step},
                                     {"last_use_step", lifetime.last_use_step},
                                     {"use_count", lifetime.use_count},
                                     {"uses", std::move(uses)}});
            }
        }
        json frames = json::array();
        for (const VectorTensorFramePlan& frame : tensor_frames[index]) {
            frames.push_back(
                {{"tensor", frame.tensor},
                 {"logical", {frame.logical_rows, frame.logical_cols}},
                 {"physical", {frame.physical_rows, frame.physical_cols}}});
        }
        json workspaces = json::array();
        for (const VectorWorkspaceFramePlan& frame : workspace_frames[index]) {
            workspaces.push_back(
                {{"op", frame.op},
                 {"source_tensor", frame.source_tensor},
                 {"logical", {frame.logical_rows, frame.logical_cols}},
                 {"physical", {frame.physical_rows, frame.physical_cols}}});
        }
        json phase_json =
            {{"name", vector_replay_phase_name(phase)},
             {"ops", plan.input_lifetimes ? plan.input_lifetimes->ops[index]
                                           : std::vector<size_t>{}},
             {"input_lifetimes", std::move(lifetimes)},
             {"tensor_frames", std::move(frames)},
             {"workspaces", std::move(workspaces)}};
        if (phase == VectorReplayPhase::Body) {
            phase_json["loop"] = vector_loop_json(plan.body);
        } else if (phase == VectorReplayPhase::Stats) {
            phase_json["init"] = vector_serial_phase_json(plan.stats_init);
            phase_json["loop"] = vector_loop_json(plan.stats);
            phase_json["tail"] = vector_serial_phase_json(plan.stats_tail);
        } else if (phase == VectorReplayPhase::Apply) {
            phase_json["loop"] = vector_loop_json(plan.apply);
            phase_json["tail"] = vector_serial_phase_json(plan.apply_tail);
        } else {
            phase_json["serial"] = vector_serial_phase_json(plan.finalize);
        }
        result.push_back(std::move(phase_json));
    }
    return result;
}

static json vector_stream_plan_json(const Problem& problem,
                                    const VectorStreamPlan& plan) {
    const char* coordinate_transform =
        plan.coordinate_transform == VectorCoordinateTransform::SingletonColumnToRow
            ? "singleton_column_to_row"
            : "none";
    const char* spatial_policy =
        plan.spatial_policy == VectorSpatialPolicy::ClampedOverlap
            ? "clamped_overlap"
            : "exact_balanced";
    json p4_recipe = nullptr;
    if (plan.p4_recipe && plan.p4_recipe->kind != P4PatternKind::None) {
        const bool softmax =
            plan.p4_recipe->kind == P4PatternKind::SoftmaxFlash;
        json bindings = json::array();
        for (const P4ApplyBinding& binding : plan.p4_recipe->apply_bindings) {
            bindings.push_back(
                {{"op", binding.op},
                 {"value", p4_substitution_value_name(binding.value)}});
        }
        p4_recipe =
            {{"version", softmax ? "softmax_flash.v1" : "welford.v1"},
             {"input_tensor", plan.p4_recipe->input_tensor},
             {"state",
              softmax ? json::array({"running_max", "running_sum"})
                      : json::array({"running_mean", "running_m2",
                                     "running_count"})},
             {"apply_substitutions", std::move(bindings)}};
    }
    return {{"kind", vector_stream_kind_name(plan.kind)},
            {"coordinate_transform", coordinate_transform},
            {"spatial_policy", spatial_policy},
            {"work_units", plan.work_units},
            {"m_partition", axis_partition_json(plan.m_partition)},
            {"n_partition", axis_partition_json(plan.n_partition)},
            {"full_peak_ub_bytes", plan.full_peak_ub_bytes},
            {"chunk_peak_ub_bytes", plan.chunk_peak_ub_bytes},
            {"stream_band_count", plan.stream_band_count},
            {"physical_frame",
             {{"element_granule", plan.physical_element_granule},
              {"iteration_rows", plan.iteration_rows},
              {"iteration_cols", plan.iteration_cols},
              {"reduced_axis", plan.reduced_axis},
              {"align_rows", plan.align_rows}}},
            {"axis", plan.axis},
            {"free_tile", plan.free_tile},
            {"free_tile_alloc", plan.free_tile_alloc},
            {"extent", plan.extent},
            {"chunk", plan.chunk},
            {"full_chunks", plan.full_chunks},
            {"tail", plan.tail},
            {"stream_passes", plan.stream_passes},
            {"phases", vector_replay_phases_json(problem, plan)},
            {"tile", {plan.tile_h, plan.tile_w}},
            {"strip", {plan.strip_h, plan.strip_w}},
            {"strip_grid", {plan.row_strips, plan.width_strips}},
            {"overlap_granted", plan.overlap_granted},
            {"reduction_split",
             {{"kind", vector_reduction_split_kind_name(plan.reduction_split_kind)},
              {"factor", plan.reduction_split_factor},
              {"partial_extent", plan.reduction_partial_extent},
              {"seed",
               {{"present", plan.reduction_seed.present},
                {"work_units", plan.reduction_seed.work_units},
                {"valid_rows", plan.reduction_seed.valid_rows},
                {"valid_cols", plan.reduction_seed.valid_cols}}}}},
            {"p4_work",
             {{"generated", plan.p4_work.generated},
              {"stats_init", vector_phase_work_json(plan.p4_work.stats_init)},
              {"stats_update", vector_phase_work_json(plan.p4_work.stats_update)},
              {"finalize", vector_phase_work_json(plan.p4_work.finalize)}}},
            {"p4_recipe", std::move(p4_recipe)}};
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

static const char* cube_split_merge_policy_name(CubeSplitMergePolicy policy) {
    switch (policy) {
        case CubeSplitMergePolicy::None: return "none";
        case CubeSplitMergePolicy::FirstPartialThenAtomic: return "first_partial_then_atomic";
        case CubeSplitMergePolicy::AivZeroSeedThenAtomic: return "aiv_zero_seed_then_atomic";
    }
    return "unknown";
}

static const char* cube_operand_role_name(CubeOperandRole role) {
    switch (role) {
        case CubeOperandRole::Lhs: return "lhs";
        case CubeOperandRole::Rhs: return "rhs";
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

static const char* mixed_cross_core_protocol_name(MixedCrossCoreProtocol protocol) {
  switch (protocol) {
    case MixedCrossCoreProtocol::Unsupported:
      return "unsupported";
    case MixedCrossCoreProtocol::OneWay:
      return "one_way";
    case MixedCrossCoreProtocol::SingleRoundTripBundle:
      return "single_round_trip_bundle";
  }
  return "unknown";
}

static const char* mixed_pipeline_axis_name(MixedPipelineAxis axis) {
  switch (axis) {
    case MixedPipelineAxis::SpatialRegion:
      return "spatial_region";
    case MixedPipelineAxis::VectorWidthChunk:
      return "vector_width_chunk";
    case MixedPipelineAxis::VectorHeightChunk:
      return "vector_height_chunk";
    case MixedPipelineAxis::AttentionKeyChunk:
      return "attention_key_chunk";
    case MixedPipelineAxis::IntermediateFeatureChunk:
      return "intermediate_feature_chunk";
  }
  return "unknown";
}

static const char* mixed_algorithm_name(MixedAlgorithmKind algorithm) {
  switch (algorithm) {
    case MixedAlgorithmKind::Generic:
      return "generic";
    case MixedAlgorithmKind::DenseSwiGluMlp:
      return "dense_swiglu_mlp";
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

    if (j.contains("schema_version")) {
        const std::string schema = j["schema_version"].get<std::string>();
        if (schema != "pto_fusebox.problem.v1") {
            std::cerr << "Error: unsupported problem schema '" << schema
                      << "' (expected 'pto_fusebox.problem.v1')\n";
            std::exit(1);
        }
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
        if (dts.size() != num_tensors) {
            std::cerr << "Error: explicit dtypes array must match tensor count\n";
            std::exit(1);
        }
        for (size_t i = 0; i < dts.size(); i++) {
            const auto& s = dts[i].get_ref<const std::string&>();
            if (s == "FP32") p.tensors[i].dtype = DType::FP32;
            else if (s == "FP16") p.tensors[i].dtype = DType::FP16;
            else if (s == "BF16") p.tensors[i].dtype = DType::BF16;
            else if (s == "INT32") p.tensors[i].dtype = DType::INT32;
            else if (s == "INT16") p.tensors[i].dtype = DType::INT16;
            else if (s == "INT8") p.tensors[i].dtype = DType::INT8;
            else if (s == "BOOL") p.tensors[i].dtype = DType::BOOL;
            else {
                std::cerr << "Error: unknown dtype '" << s << "' for tensor " << i << "\n";
                std::exit(1);
            }
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
        if (j.contains("mixed_vector_semantics") &&
            i < j["mixed_vector_semantics"].size()) {
            const std::string name =
                j["mixed_vector_semantics"][i].get<std::string>();
            if (!parse_mixed_vector_semantic(name, &op.mixed_vector_semantic)) {
                std::cerr << "Error: unknown mixed vector semantic '" << name
                          << "' for op " << i << "\n";
                std::exit(1);
            }
        }
        if (j.contains("mixed_emit_compatible") &&
            i < j["mixed_emit_compatible"].size()) {
            op.mixed_emit_compatible = j["mixed_emit_compatible"][i].get<bool>();
        }
        p.ops.push_back(std::move(op));
    }

    if (j.contains("p4_patterns")) {
        for (size_t pattern_index = 0; pattern_index < j["p4_patterns"].size();
             ++pattern_index) {
            const auto& encoded = j["p4_patterns"][pattern_index];
            if (!encoded.contains("kind") || !encoded.contains("ops")) {
                std::cerr << "Error: P4 pattern " << pattern_index
                          << " requires kind and ops\n";
                std::exit(1);
            }
            P4Pattern pattern;
            const std::string kind = encoded["kind"].get<std::string>();
            if (!parse_p4_pattern_kind(kind, &pattern.kind)) {
                std::cerr << "Error: unknown P4 pattern kind '" << kind << "'\n";
                std::exit(1);
            }
            for (const auto& item : encoded["ops"]) {
                const size_t op = item.get<size_t>();
                if (op >= num_ops) {
                    std::cerr << "Error: P4 pattern " << pattern_index
                              << " references out-of-range op " << op << "\n";
                    std::exit(1);
                }
                pattern.ops.insert(op);
            }
            if (encoded.contains("apply_substitutions")) {
                for (const auto& item : encoded["apply_substitutions"]) {
                    const bool named = item.is_object();
                    if (named && !item.contains("op")) {
                        std::cerr << "Error: P4 pattern " << pattern_index
                                  << " named substitution requires an op\n";
                        std::exit(1);
                    }
                    const size_t op = named ? item.at("op").get<size_t>()
                                            : item.get<size_t>();
                    if (op >= num_ops || pattern.ops.count(op) == 0) {
                        std::cerr << "Error: P4 pattern " << pattern_index
                                  << " substitution op " << op
                                  << " is not a member of the pattern\n";
                        std::exit(1);
                    }
                    pattern.apply_substitutions.insert(op);
                    if (named) {
                        if (!item.contains("value") ||
                            !item.at("value").is_string()) {
                            std::cerr << "Error: P4 pattern " << pattern_index
                                      << " named substitution requires a value\n";
                            std::exit(1);
                        }
                        P4SubstitutionValue value;
                        const std::string name =
                            item.at("value").get<std::string>();
                        if (!parse_p4_substitution_value(name, &value)) {
                            std::cerr << "Error: P4 pattern " << pattern_index
                                      << " has unknown substitution value '"
                                      << name << "'\n";
                            std::exit(1);
                        }
                        pattern.apply_bindings.push_back({op, value});
                    }
                }
            }
            if (!pattern.apply_bindings.empty()) {
                const std::array<P4SubstitutionValue, 2> expected =
                    pattern.kind == P4PatternKind::SoftmaxFlash
                        ? std::array<P4SubstitutionValue, 2>{
                              P4SubstitutionValue::RunningMax,
                              P4SubstitutionValue::RunningSum}
                        : std::array<P4SubstitutionValue, 2>{
                              P4SubstitutionValue::Mean,
                              P4SubstitutionValue::Variance};
                std::array<bool, 2> found{false, false};
                FlatSet<size_t> binding_ops;
                for (const P4ApplyBinding& binding : pattern.apply_bindings) {
                    binding_ops.insert(binding.op);
                    for (size_t index = 0; index < expected.size(); ++index) {
                        if (binding.value == expected[index]) found[index] = true;
                    }
                }
                if (pattern.apply_bindings.size() != expected.size() ||
                    binding_ops.size() != expected.size() || !found[0] ||
                    !found[1]) {
                    std::cerr << "Error: P4 pattern " << pattern_index
                              << " named substitutions do not match its "
                                 "semantic contract\n";
                    std::exit(1);
                }
            }
            p.p4_patterns.push_back(std::move(pattern));
        }
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

    if (j.contains("vector_coordinate_transform")) {
        const std::string transform = j["vector_coordinate_transform"].get<std::string>();
        if (transform == "none") {
            p.vector_coordinate_transform = VectorCoordinateTransform::None;
        } else if (transform == "singleton_column_to_row") {
            p.vector_coordinate_transform = VectorCoordinateTransform::SingletonColumnToRow;
        } else {
            std::cerr << "Error: unknown vector coordinate transform '" << transform << "'\n";
            std::exit(1);
        }
    }

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
    if (j.contains("cube_split_sync_cycles")) {
        p.cube_split_sync_cycles = j["cube_split_sync_cycles"].get<int64_t>();
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
      if (l0.contains("l0c_align_m")) config.l0c_align_m = l0["l0c_align_m"].get<int64_t>();
      if (l0.contains("box_align_m")) config.box_align_m = l0["box_align_m"].get<int64_t>();
      if (l0.contains("box_align_n")) config.box_align_n = l0["box_align_n"].get<int64_t>();
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
      if (l0.contains("mad_fp32_passes")) {
        config.mad_fp32_passes = l0["mad_fp32_passes"].get<int64_t>();
      }
      if (l0.contains("mad_head_cycles")) {
        config.mad_head_cycles = l0["mad_head_cycles"].get<int64_t>();
      }
      if (l0.contains("mad_k_fractal_bytes")) {
        config.mad_k_fractal_bytes = l0["mad_k_fractal_bytes"].get<int64_t>();
      }
    }
    if (j.contains("vec_reg_bytes"))    p.vec_reg_bytes    = j["vec_reg_bytes"].get<int64_t>();
    if (j.contains("vec_dma_align_bytes")) {
      p.vec_dma_align_bytes = j["vec_dma_align_bytes"].get<int64_t>();
    }
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
    if (j.contains("allow_model_ahead_split_k")) {
      p.allow_model_ahead_split_k = j["allow_model_ahead_split_k"].get<bool>();
    }
    if (j.contains("allow_model_ahead_multi_reduction_stream")) {
      p.allow_model_ahead_multi_reduction_stream =
          j["allow_model_ahead_multi_reduction_stream"].get<bool>();
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

std::string solution_json(const Solution& sol) {
    json j;
    j["schema_version"] = "pto_fusebox.solution.v5";
    j["steps"] = json::array();

    for (size_t i = 0; i < sol.num_steps(); i++) {
        const auto& step = sol.step(i);
        if (!sol.retained_entering(i).empty() || !step.retain_these.empty()) {
            throw std::logic_error(
                "solution.v5 cannot serialize cross-kernel fast-memory retention");
        }
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
                      cost.parallel_split, cost.cube_split_merge_policy)
                : CubeSchedulePlan{};
        const MixedSchedulePlan mixed_plan =
            step.subgraph.is_mixed()
                ? step.subgraph.mixed_schedule_plan(
                      cfg, sol.retained_entering(i), step.retain_these,
                      cost.parallel_split)
                : MixedSchedulePlan{};

        json serialized_step;
        serialized_step["ops"] = step.subgraph.ops();
        const int64_t launch_k =
            cube_plan.feasible ? cube_plan.config.k : cfg.k;
        serialized_step["launch"] =
            {{"tile", {cfg.w, cfg.h, launch_k}},
             {"parts", {cfg.parts_m, cfg.parts_n}},
             {"split", cost.parallel_split},
             {"cores", cost.cores_used}};
        // Spatial grid shape: parts_m x parts_n regions (0,0 = a uniform tile,
        // region count = floor(out_W/w)*floor(out_H/h)). w,h above are the MAX
        // region extent (regions differ by <=1 block), NOT a uniform tile -- read
        // the region count from here, do not infer it from the tile size.
        // Execution order (the fixed pebbling order) — emitted because the peak
        // working set depends on it, so downstream must materialize this order.
        {
            const auto& order = step.subgraph.execution_order();
            serialized_step["op_order"] =
                std::vector<size_t>(order.begin(), order.end());
            // Per-op single-core k-tile, in execution order (cube-910B only). An
            // op's seq_k = its full K means it ran the contraction in one pass.
            const auto& prob = step.subgraph.problem();
            if (cube_plan.feasible) {
                std::vector<int64_t> ks;
                for (auto op : order) {
                    const auto request = std::find_if(
                        cube_plan.matmuls.begin(), cube_plan.matmuls.end(),
                        [op](const CubeMatmulSchedule& matmul) {
                            return matmul.op == op;
                        });
                    if (request == cube_plan.matmuls.end()) {
                        throw std::logic_error(
                            "cube plan omits an operation from its execution order");
                    }
                    ks.push_back(request->k_loop.l1_window_k);
                }
                serialized_step["sequential_tiles"] = ks;
            } else if (step.subgraph.has_matmul() && prob.num_cube_cores > 1 &&
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
                serialized_step["sequential_tiles"] = ks;
            } else if (!step.subgraph.has_matmul() && prob.num_vector_cores > 1 &&
                       prob.vec_capacity > 0) {
                // Only reduced-axis streaming maps to sequential tiles. A
                // pointwise strip loop is already fully described by the
                // vector plan's strip/grid/body fields; serializing its strip
                // size here would give two conflicting owners for one loop.
                const bool reduction_streamed =
                    vector_plan.kind != VectorStreamKind::Materialized &&
                    vector_plan.kind != VectorStreamKind::Pointwise;
                const int64_t chunk = reduction_streamed ? vector_plan.chunk : 0;
                serialized_step["sequential_tiles"] =
                    std::vector<int64_t>(order.size(), chunk);
            } else {
                serialized_step["sequential_tiles"] = nullptr;
            }
        }
        if (vector_plan.feasible) {
            serialized_step["kind"] = "vector";
            serialized_step["plan"] =
                vector_stream_plan_json(sol.problem(), vector_plan);
        }
        if (cube_plan.feasible) {
            json matmuls = json::array();
            json resident_boundaries = json::array();
            for (const auto& resident : cube_plan.resident_boundaries) {
                resident_boundaries.push_back(
                    {{"id", resident.id},
                     {"region", cube_region_json(resident.region)},
                     {"role", cube_operand_role_name(resident.role)},
                     {"first_use", resident.first_use},
                     {"last_use", resident.last_use},
                     {"use_count", resident.use_count},
                     {"bytes", resident.bytes}});
            }
            for (const auto& mm : cube_plan.matmuls) {
                json variants = json::array();
                for (const auto& variant : mm.output_variants) {
                    variants.push_back(
                        {{"shape", {variant.height, variant.width}},
                         {"count", variant.count},
                         {"l0_init",
                          l0_matmul_plan_json(variant.l0_init)},
                         {"l0_rolled",
                          l0_matmul_plan_json(variant.l0_rolled)},
                         {"l0_tail",
                          l0_matmul_plan_json(variant.l0_tail)}});
                }
        matmuls.push_back({{"instance", mm.instance},
                           {"op", mm.op},
                           {"lhs_producer", mm.lhs_producer},
                           {"rhs_producer", mm.rhs_producer},
                           {"lhs_resident_boundary", mm.lhs_resident_boundary},
                           {"rhs_resident_boundary", mm.rhs_resident_boundary},
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
                           {"retained_panels",
                            {{"lhs", mm.retained_panels.lhs},
                             {"rhs", mm.retained_panels.rhs},
                             {"lhs_bytes", mm.retained_panels.lhs_bytes},
                             {"rhs_bytes", mm.retained_panels.rhs_bytes}}},
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
      serialized_step["kind"] = "cube";
      serialized_step["plan"] =
          {{"emit_compatible", cube_plan.emit_compatible},
           {"spatial_policy", cube_spatial_policy_name(cube_plan.spatial_policy)},
           {"m_partition", axis_partition_json(cube_plan.m_partition)},
           {"n_partition", axis_partition_json(cube_plan.n_partition)},
           {"spatial_tiles", cube_plan.spatial_tiles},
           {"split_k", cube_plan.split_k},
           {"work_units", cube_plan.work_units},
           {"peak_l1_bytes", cube_plan.peak_l1_bytes},
           {"split_merge_policy", cube_split_merge_policy_name(cube_plan.split_merge_policy)},
           {"first_partial_then_atomic",
            {{"present", cube_plan.first_partial_then_atomic.present},
             {"first_work_units", cube_plan.first_partial_then_atomic.first_work_units},
             {"atomic_work_units", cube_plan.first_partial_then_atomic.atomic_work_units},
             {"synchronization_cycles", cube_plan.first_partial_then_atomic.synchronization_cycles}}},
           {"aiv_zero_seed_then_atomic",
            {{"present", cube_plan.aiv_zero_seed_then_atomic.present},
             {"seed_work_units", cube_plan.aiv_zero_seed_then_atomic.seed_work_units},
             {"atomic_work_units", cube_plan.aiv_zero_seed_then_atomic.atomic_work_units},
             {"seed_bytes", cube_plan.aiv_zero_seed_then_atomic.seed_bytes},
             {"synchronization_cycles", cube_plan.aiv_zero_seed_then_atomic.synchronization_cycles}}},
           {"model_overlap_granted", cube_plan.model_overlap_granted},
           {"overlap_implementable", cube_plan.overlap_implementable},
           {"execution_order", cube_plan.execution_order},
           {"resident_boundaries", resident_boundaries},
           {"matmuls", matmuls}};
    }
    if (mixed_plan.feasible && mixed_plan.topology) {
      json topology_stages = json::array();
      for (const auto& stage : mixed_plan.topology->stages) {
        topology_stages.push_back(
            {{"engine", mixed_engine_name(stage.engine)}, {"ops", stage.ops}});
      }
      json stages = json::array();
      for (const auto& stage : mixed_plan.stages) {
        stages.push_back(
            {{"topology_stage", stage.topology_stage},
             {"engine", mixed_engine_name(stage.engine)},
             {"ops", stage.ops},
             {"valid_rows", stage.valid_rows},
             {"valid_cols", stage.valid_cols},
             {"cube_window_k", stage.cube_window_k},
             {"vector_stream",
              stage.vector_stream.feasible
                  ? vector_stream_plan_json(sol.problem(), stage.vector_stream)
                  : json(nullptr)}});
      }
      json transfers = json::array();
      for (const auto& transfer : mixed_plan.topology->transfers) {
        transfers.push_back({{"tensor", transfer.tensor},
                             {"producer_stage", transfer.producer_stage},
                             {"consumer_stage", transfer.consumer_stage},
                             {"producer_engine", mixed_engine_name(transfer.producer_engine)},
                             {"consumer_engine", mixed_engine_name(transfer.consumer_engine)}});
      }
      json fifos = json::array();
      for (const auto& fifo : mixed_plan.fifos) {
        fifos.push_back({{"tensor", fifo.tensor},
                         {"direction", mixed_transfer_direction_name(fifo.direction)},
                         {"valid_rows", fifo.valid_rows},
                         {"valid_cols", fifo.valid_cols},
                         {"slot_bytes", fifo.slot_bytes},
                         {"slot_count", fifo.slot_count},
                         {"reserved_bytes", fifo.reserved_bytes},
                         {"pipe_id", fifo.pipe_id},
                         {"bundle", fifo.bundle}});
      }
      json dense_mlp = nullptr;
      if (mixed_plan.dense_mlp.present) {
        dense_mlp = {{"input_extent", mixed_plan.dense_mlp.input_extent},
                     {"intermediate_extent", mixed_plan.dense_mlp.intermediate_extent},
                     {"intermediate_chunk", mixed_plan.dense_mlp.intermediate_chunk},
                     {"intermediate_chunks", mixed_plan.dense_mlp.intermediate_chunks},
                     {"output_extent", mixed_plan.dense_mlp.output_extent},
                     {"gate_window_k", mixed_plan.dense_mlp.gate_window_k},
                     {"up_window_k", mixed_plan.dense_mlp.up_window_k},
                     {"persistent_accumulator_bytes", mixed_plan.dense_mlp.persistent_accumulator_bytes},
                     {"first_chunk_initializes", mixed_plan.dense_mlp.first_chunk_initializes},
                     {"later_chunks_accumulate", mixed_plan.dense_mlp.later_chunks_accumulate}};
      }
      auto protocol_stage = [](size_t stage) {
        return stage == std::numeric_limits<size_t>::max() ? json(nullptr) : json(stage);
      };
      serialized_step["kind"] = "mixed";
      serialized_step["plan"] =
          {{"emit_compatible", mixed_plan.emit_compatible},
           {"source_codegen_ready", mixed_plan.source_codegen_ready},
           {"algorithm", mixed_algorithm_name(mixed_plan.algorithm)},
           {"protocol", mixed_cross_core_protocol_name(mixed_plan.protocol)},
           {"mode", mixed_pipeline_mode_name(mixed_plan.mode)},
           {"m_partition", axis_partition_json(mixed_plan.m_partition)},
           {"n_partition", axis_partition_json(mixed_plan.n_partition)},
           {"spatial_tiles", mixed_plan.spatial_tiles},
           {"split_k", mixed_plan.split_k},
           {"work_units", mixed_plan.work_units},
           {"group_capacity", mixed_plan.group_capacity},
           {"cube_window_k", mixed_plan.cube_window_k},
           {"cube_stage_peak_l1_bytes", mixed_plan.cube_stage_peak_l1_bytes},
           {"vector_stage_kind", vector_stream_kind_name(mixed_plan.vector_stage_kind)},
           {"vector_stage_peak_ub_bytes", mixed_plan.vector_stage_peak_ub_bytes},
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
           {"protocol_producer_stages", mixed_plan.topology->protocol.producer_stages},
           {"protocol_peer_stage", protocol_stage(mixed_plan.topology->protocol.peer_stage)},
           {"protocol_sink_stage", protocol_stage(mixed_plan.topology->protocol.sink_stage)},
           {"protocol_producer_bundle", mixed_plan.topology->protocol.producer_bundle_transfers},
           {"protocol_reply_bundle", mixed_plan.topology->protocol.reply_bundle_transfers},
           {"protocol_skew_compatible", mixed_plan.topology->protocol.skew_pass_compatible},
           {"topology_stages", topology_stages},
           {"stages", stages},
           {"transfers", transfers},
           {"fifos", fifos},
           {"dense_mlp", dense_mlp}};
        }
        serialized_step["latency_cycles"] = sol.step_latency(i);
        j["steps"].push_back(std::move(serialized_step));
    }

    return j.dump(2) + "\n";
}

void write_solution(const std::string& filename, const Solution& sol) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: cannot write '" << filename << "'\n";
        std::exit(1);
    }
    f << solution_json(sol);
    if (!f) {
        std::cerr << "Error: write failed for '" << filename << "'\n";
        std::exit(1);
    }
}
