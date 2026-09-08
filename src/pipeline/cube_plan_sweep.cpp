#include "pipeline/cube_plan_sweep.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/subgraph.h"
#include "io/io.h"
#include "solution/solution.h"

namespace {

using json = nlohmann::json;

std::string candidate_id(const TileConfig& config) {
  return "p" + std::to_string(config.parts_m) + "_q" +
         std::to_string(config.parts_n) + "_s" +
         std::to_string(config.split_k) + "_k" + std::to_string(config.k);
}

bool same_grid(const TileConfig& left, const TileConfig& right,
               int left_split, int right_split) {
  return left.parts_m == right.parts_m &&
         left.parts_n == right.parts_n && left_split == right_split &&
         left.k == right.k;
}

}  // namespace

std::string cube_plan_sweep_json(const Problem& problem, const DAG& dag) {
  if (problem.num_ops() == 0 ||
      std::any_of(problem.ops.begin(), problem.ops.end(),
                  [](const Op& op) { return op.type != OpType::MatMul; })) {
    throw std::invalid_argument(
        "cube plan sweep requires a non-empty homogeneous MatMul DAG");
  }

  std::vector<size_t> operations(problem.num_ops());
  std::iota(operations.begin(), operations.end(), 0);
  auto subgraph = Subgraph::create(problem, dag, operations);
  if (!subgraph || !subgraph->has_matmul() || subgraph->has_vector()) {
    throw std::invalid_argument(
        "cube plan sweep could not construct a homogeneous MatMul DAG");
  }

  auto candidates = subgraph->enumerate_plans();
  if (candidates.empty()) {
    throw std::invalid_argument(
        "cube plan sweep found no feasible candidates");
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& left,
                                                     const auto& right) {
    return std::tie(left.first.parts_m, left.first.parts_n,
                    left.first.split_k, left.first.w, left.first.h,
                    left.first.k) <
           std::tie(right.first.parts_m, right.first.parts_n,
                    right.first.split_k, right.first.w, right.first.h,
                    right.first.k);
  });

  const CostResult selected = subgraph->best_cost();
  if (!selected.feasible) {
    throw std::invalid_argument(
        "cube plan sweep found no selected feasible candidate");
  }

  json output = {
      {"schema_version", "pto_fusebox.cube_plan_sweep.v2"},
      {"selected_candidate_id", nullptr},
      {"candidates", json::array()},
      {"rejections", json::array()},
  };
  for (const CubePlanCandidateDiagnostic &diagnostic :
       subgraph->diagnose_cube_plans()) {
    if (diagnostic.rejection_code.empty()) continue;
    output["rejections"].push_back(
        {{"grid",
          {{"parts_m", diagnostic.config.parts_m},
           {"parts_n", diagnostic.config.parts_n},
           {"split_k", diagnostic.config.split_k},
           {"sequential_k_limit", diagnostic.config.k}}},
         {"reason", diagnostic.rejection_code}});
  }
  bool found_selected = false;
  std::set<std::string> emitted_ids;
  for (const auto& [enumerated_config, enumerated_cost] : candidates) {
    Solution forced(problem, dag,
                    {{Subgraph(*subgraph), enumerated_config, {}}});
    const auto validation = forced.validate();
    if (!validation.valid || forced.num_steps() != 1 ||
        !forced.step_cost(0).feasible) {
      throw std::logic_error(
          "enumerated cube candidate did not reconstruct a valid solution");
    }
    const CostResult& cost = forced.step_cost(0);
    if (!std::isfinite(enumerated_cost.latency) ||
        !std::isfinite(cost.latency) ||
        enumerated_config.parts_m != cost.config.parts_m ||
        enumerated_config.parts_n != cost.config.parts_n ||
        static_cast<int>(enumerated_cost.parallel_split) != cost.parallel_split ||
        cost.config.k > enumerated_config.k ||
        std::abs(enumerated_cost.latency - cost.latency) >
            1e-9 * std::max(1.0, std::abs(cost.latency))) {
      throw std::logic_error(
          "enumerated cube candidate " + candidate_id(enumerated_config) +
          " changed during forced reconstruction: enumerated latency=" +
          std::to_string(enumerated_cost.latency) +
          ", forced latency=" + std::to_string(cost.latency) +
          ", enumerated grid=" + std::to_string(enumerated_config.parts_m) +
          "x" + std::to_string(enumerated_config.parts_n) + " k<=" +
          std::to_string(enumerated_config.k) + ", forced grid=" +
          std::to_string(cost.config.parts_m) + "x" +
          std::to_string(cost.config.parts_n) + " k=" +
          std::to_string(cost.config.k) + ", split=" +
          std::to_string(enumerated_cost.parallel_split) + "/" +
          std::to_string(cost.parallel_split));
    }

    const std::string id = candidate_id(cost.config);
    if (!emitted_ids.insert(id).second) continue;
    const bool is_selected =
        same_grid(cost.config, selected.config,
                  cost.parallel_split,
                  selected.parallel_split);
    if (is_selected) {
      if (found_selected) {
        throw std::logic_error(
            "cube plan sweep selected more than one candidate");
      }
      output["selected_candidate_id"] = id;
      found_selected = true;
    }

    output["candidates"].push_back(
        {{"id", id},
         {"selected", is_selected},
         {"enumerated_grid",
          {{"parts_m", enumerated_config.parts_m},
           {"parts_n", enumerated_config.parts_n},
           {"split_k", enumerated_config.split_k},
           {"sequential_k_limit", enumerated_config.k},
           {"realized_sequential_k", cost.config.k}}},
         {"model",
          {{"latency_cycles", cost.latency},
           {"cores_used", cost.cores_used},
           {"compute_bound", cost.compute_bound},
           {"ddr_traffic_cycles", cost.ddr_traffic},
           {"l1_l0_extract_cycles", cost.l1l0_extract},
           {"uses_model_ahead_split_k", cost.uses_model_ahead_split_k}}},
         {"solution", nlohmann::json::parse(solution_json(forced))}});
  }
  if (!found_selected) {
    throw std::logic_error(
        "cube plan sweep could not identify the model-selected candidate");
  }
  return output.dump(2) + "\n";
}
