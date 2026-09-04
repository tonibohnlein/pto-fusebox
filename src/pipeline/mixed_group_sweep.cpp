#include "pipeline/mixed_group_sweep.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/ascend910b_cost.h"

namespace {

using json = nlohmann::json;

const char* direction_name(MixedTransferDirection direction) {
  switch (direction) {
    case MixedTransferDirection::CubeToVector:
      return "cube_to_vector";
    case MixedTransferDirection::VectorToCube:
      return "vector_to_cube";
  }
  throw std::logic_error("unknown mixed transfer direction");
}

const char* engine_name(MixedEngine engine) {
  switch (engine) {
    case MixedEngine::Cube:
      return "cube";
    case MixedEngine::Vector:
      return "vector";
  }
  throw std::logic_error("unknown mixed engine");
}

const char* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::FP32: return "fp32";
    case DType::FP16: return "fp16";
    case DType::BF16: return "bf16";
    case DType::INT32: return "int32";
    case DType::INT16: return "int16";
    case DType::INT8: return "int8";
    case DType::BOOL: return "bool";
  }
  throw std::logic_error("unknown mixed FIFO wire dtype");
}

}  // namespace

std::string mixed_group_sweep_json(const Problem& problem, const DAG& dag) {
  if (problem.num_ops() == 0) {
    throw std::invalid_argument("mixed group sweep requires a non-empty DAG");
  }
  std::vector<size_t> operations(problem.num_ops());
  std::iota(operations.begin(), operations.end(), 0);
  auto subgraph = Ascend910BMixed::create(problem, dag, operations);
  if (!subgraph || !subgraph->has_matmul() || !subgraph->has_vector()) {
    throw std::invalid_argument(
        "mixed group sweep requires one connected cube+vector region");
  }
  const CostResult selected = subgraph->best_cost();
  if (!selected.feasible || selected.mixed_active_groups <= 0) {
    throw std::invalid_argument("mixed group sweep found no selected candidate");
  }
  const auto candidates = subgraph->enumerate_mixed_group_costs(selected.config);
  if (candidates.empty()) {
    throw std::invalid_argument(
        "mixed group sweep supports generic one-way or round-trip plans only");
  }

  json output = {
      {"schema_version", "pto_fusebox.mixed_group_sweep.v2"},
      {"selection_resolution_cycles",
       Ascend910BCost::kMixedGroupSelectionResolutionCycles},
      {"selected_candidate_id",
       "g" + std::to_string(selected.mixed_active_groups)},
      {"tile",
       {{"h", selected.config.h},
        {"w", selected.config.w},
        {"k", selected.config.k},
        {"parts_m", selected.config.parts_m},
        {"parts_n", selected.config.parts_n}}},
      {"candidates", json::array()},
  };
  bool found_selected = false;
  for (const MixedGroupCostCandidate& candidate : candidates) {
    const CostResult& cost = candidate.cost;
    const MixedCostBreakdown& breakdown = candidate.breakdown;
    const MixedSchedulePlan plan = subgraph->mixed_schedule_plan(
        cost.config, {}, {}, cost.parallel_split, breakdown.active_groups);
    if (!plan.feasible || plan.loop.active_groups != breakdown.active_groups ||
        plan.loop.max_trips_per_group != breakdown.trips_per_group ||
        std::abs(cost.latency - breakdown.total_cycles) >
            1e-9 * std::max(1.0, std::abs(cost.latency))) {
      throw std::logic_error(
          "mixed candidate changed between costing and plan reconstruction");
    }
    const bool is_selected =
        breakdown.active_groups == selected.mixed_active_groups;
    found_selected = found_selected || is_selected;
    json fifos = json::array();
    for (const MixedFifoPlan& fifo : plan.fifos) {
      fifos.push_back({{"tensor", fifo.tensor},
                       {"pipe_id", fifo.pipe_id},
                       {"direction", direction_name(fifo.direction)},
                       {"wire_dtype", dtype_name(fifo.wire_dtype)},
                       {"bundle", fifo.bundle},
                       {"spatial_m", fifo.spatial_m},
                       {"spatial_n", fifo.spatial_n},
                       {"valid_rows", fifo.valid_rows},
                       {"valid_cols", fifo.valid_cols},
                       {"slot_bytes", fifo.slot_bytes},
                       {"slot_count", fifo.slot_count},
                       {"reserved_bytes", fifo.reserved_bytes}});
    }
    json stages = json::array();
    for (const MixedStagePlan& stage : plan.stages) {
      stages.push_back({{"engine", engine_name(stage.engine)},
                        {"topology_stage", stage.topology_stage},
                        {"ops", stage.ops},
                        {"valid_rows", stage.valid_rows},
                        {"valid_cols", stage.valid_cols},
                        {"cube_window_k", stage.cube_window_k}});
    }
    output["candidates"].push_back(
        {{"id", "g" + std::to_string(breakdown.active_groups)},
         {"selected", is_selected},
         {"groups", breakdown.active_groups},
         {"trips_per_group", breakdown.trips_per_group},
         {"pipeline_stages", breakdown.pipeline_stages},
         {"overlap_implementable", breakdown.overlap_implementable},
         {"cube_stage_peak_l1_bytes", plan.cube_stage_peak_l1_bytes},
         {"vector_stage_peak_ub_bytes", plan.vector_stage_peak_ub_bytes},
         {"model",
          {{"cube_phase_cycles", breakdown.cube_phase_cycles},
           {"vector_phase_cycles", breakdown.vector_phase_cycles},
           {"gm_l1_bytes", breakdown.gm_l1_bytes},
           {"gm_ub_bytes", breakdown.gm_ub_bytes},
           {"l0c_gm_bytes", breakdown.l0c_gm_bytes},
           {"ub_gm_bytes", breakdown.ub_gm_bytes},
           {"gm_l1_effective_parallelism",
            breakdown.gm_l1_effective_parallelism},
           {"gm_ub_effective_parallelism",
            breakdown.gm_ub_effective_parallelism},
           {"l0c_gm_effective_parallelism",
            breakdown.l0c_gm_effective_parallelism},
           {"ub_gm_effective_parallelism",
            breakdown.ub_gm_effective_parallelism},
           {"gm_l1_cycles", breakdown.gm_l1_cycles},
           {"gm_ub_cycles", breakdown.gm_ub_cycles},
           {"l0c_gm_cycles", breakdown.l0c_gm_cycles},
           {"ub_gm_cycles", breakdown.ub_gm_cycles},
           {"ddr_wall_cycles", breakdown.ddr_wall_cycles},
           {"pipeline_wall_cycles", breakdown.pipeline_wall_cycles},
           {"kernel_fill_cycles", breakdown.kernel_fill_cycles},
           {"group_overhead_cycles", breakdown.group_overhead_cycles},
           {"total_cycles", breakdown.total_cycles}}},
         {"fifos", std::move(fifos)},
         {"stages", std::move(stages)}});
  }
  if (!found_selected) {
    throw std::logic_error(
        "mixed group sweep omitted the model-selected group count");
  }
  return output.dump(2) + "\n";
}
