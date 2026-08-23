#include "core/dag.h"
#include "pipeline/cube_plan_sweep.h"

#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <string>

namespace {

int passed = 0;
int failed = 0;

void check(const char* label, bool condition) {
  if (condition) {
    ++passed;
  } else {
    ++failed;
    std::cout << "FAIL: " << label << "\n";
  }
}

Problem deep_k_problem() {
  constexpr int64_t m = 128;
  constexpr int64_t k = 8192;
  constexpr int64_t n = 128;
  Problem problem;
  problem.tensors = {{k, m, DType::FP32},
                     {n, k, DType::FP32},
                     {n, m, DType::FP32}};
  problem.ops = {{OpType::MatMul, {0, 1}, {2}}};
  problem.required_outputs.insert(2);
  problem.fast_memory_capacity = 1LL << 30;
  problem.num_cube_cores = 24;
  problem.num_vector_cores = 48;
  problem.cube_capacity = 128 * 1024;
  problem.l1_capacity = 512 * 1024;
  problem.vec_capacity = 192 * 1024;
  problem.kernel_fill_cost = 10000;
  problem.cube_freq_hz = 1.85e9;
  problem.bw_gm_l1 = 135.0;
  problem.bw_l0c_gm = 70.0;
  problem.bw_l1_l0a = 441.0;
  problem.bw_l1_l0b = 220.5;
  problem.bw_ub_gm = 188.46;
  problem.hbm_aggregate_gibps = 900.0;
  problem.l0_tile_m = 128;
  problem.l0_tile_n = 256;
  problem.allow_model_ahead_split_k = true;
  problem.use_hierarchical_cube_cost = true;
  return problem;
}

Problem split_dag_problem() {
  Problem problem = deep_k_problem();
  problem.tensors = {{64, 16, DType::BF16},
                     {2048, 64, DType::BF16},
                     {2048, 16, DType::BF16},
                     {16, 2048, DType::BF16},
                     {16, 16, DType::FP32}};
  problem.ops = {{OpType::MatMul, {0, 1}, {2}},
                 {OpType::MatMul, {2, 3}, {4}}};
  problem.required_outputs = {4};
  return problem;
}

}  // namespace

int main() {
  const Problem problem = deep_k_problem();
  const DAG dag = DAG::build(problem);
  const auto sweep =
      nlohmann::json::parse(cube_plan_sweep_json(problem, dag));

  check("sweep schema is versioned",
        sweep.at("schema_version") == "pto_fusebox.cube_plan_sweep.v1");
  check("sweep has candidates", !sweep.at("candidates").empty());
  check("sweep identifies its selected candidate",
        sweep.at("selected_candidate_id").is_string());

  std::set<std::string> ids;
  int selected_count = 0;
  bool has_spatial = false;
  bool has_split = false;
  int selected_split = 0;
  bool all_forced_solutions = true;
  bool all_costs_are_finite = true;
  bool split_contract_is_explicit = true;
  for (const auto& candidate : sweep.at("candidates")) {
    ids.insert(candidate.at("id").get<std::string>());
    selected_count += candidate.at("selected").get<bool>() ? 1 : 0;
    const int split =
        candidate.at("enumerated_grid").at("split_k").get<int>();
    has_spatial = has_spatial || split == 1;
    has_split = has_split || split > 1;
    if (candidate.at("selected").get<bool>()) selected_split = split;
    const auto& solution = candidate.at("solution");
    all_costs_are_finite =
        all_costs_are_finite &&
        std::isfinite(candidate.at("model").at("latency_cycles").get<double>()) &&
        std::isfinite(solution.at("steps").front().at("latency_cycles").get<double>());
    all_forced_solutions =
        all_forced_solutions &&
        solution.at("schema_version") == "pto_fusebox.solution.v6" &&
        solution.at("steps").size() == 1 &&
        solution.at("steps").front().at("launch").at("split") == split;
    if (split > 1) {
      const auto& plan = solution.at("steps").front().at("plan");
      const std::string policy = plan.at("split_merge_policy").get<std::string>();
      const bool policy_contract =
          (policy == "first_partial_then_atomic" &&
           plan.at("first_partial_then_atomic").at("present") == true &&
           plan.at("aiv_zero_seed_then_atomic").at("present") == false) ||
          (policy == "aiv_zero_seed_then_atomic" &&
           plan.at("first_partial_then_atomic").at("present") == false &&
           plan.at("aiv_zero_seed_then_atomic").at("present") == true);
      split_contract_is_explicit =
          split_contract_is_explicit &&
          candidate.at("model").at("uses_model_ahead_split_k") == true &&
          policy_contract;
    }
  }
  check("candidate ids are unique", ids.size() == sweep.at("candidates").size());
  check("exactly one candidate is selected", selected_count == 1);
  check("sweep contains a no-split control", has_spatial);
  check("deep-K sweep contains split-K candidates", has_split);
  check("deep-K sweep selects a split-K candidate", selected_split == 16);
  check("every candidate is an ordinary forced solution", all_forced_solutions);
  check("every candidate has a finite reconstructed cost", all_costs_are_finite);
  check("split-K candidates retain their merge contract",
        split_contract_is_explicit);

  const auto selected_policy = [](const Problem& policy_problem) {
    const DAG policy_dag = DAG::build(policy_problem);
    const auto policy_sweep =
        nlohmann::json::parse(cube_plan_sweep_json(policy_problem, policy_dag));
    for (const auto& candidate : policy_sweep.at("candidates")) {
      if (candidate.at("selected").get<bool>()) {
        return candidate.at("solution")
            .at("steps")
            .front()
            .at("plan")
            .at("split_merge_policy")
            .get<std::string>();
      }
    }
    return std::string{};
  };
  Problem expensive_vector_seed = problem;
  expensive_vector_seed.per_task_overhead_cycles = 100000;
  Problem no_vector_cores = problem;
  no_vector_cores.num_vector_cores = 0;
  check("policy selection includes AIV seed-task overhead",
        selected_policy(problem) == "aiv_zero_seed_then_atomic" &&
            selected_policy(expensive_vector_seed) ==
                "first_partial_then_atomic" &&
            selected_policy(no_vector_cores) ==
                "first_partial_then_atomic");

  const Problem dag_problem = split_dag_problem();
  const auto dag_sweep = nlohmann::json::parse(
      cube_plan_sweep_json(dag_problem, DAG::build(dag_problem)));
  bool has_replayable_split_dag = false;
  for (const auto& candidate : dag_sweep.at("candidates")) {
    if (candidate.at("enumerated_grid").at("split_k").get<int>() <= 1) {
      continue;
    }
    const auto& plan = candidate.at("solution").at("steps").front().at("plan");
    if (!plan.at("emit_compatible").get<bool>()) continue;
    const auto& upstream = plan.at("matmuls").at(0);
    const auto& sink = plan.at("matmuls").at(1);
    has_replayable_split_dag =
        plan.at("matmuls").size() == 2 && !upstream.at("is_sink").get<bool>() &&
        sink.at("is_sink").get<bool>() &&
        upstream.at("effective_contraction") == upstream.at("contraction") &&
        sink.at("effective_contraction").get<int64_t>() *
                plan.at("split_k").get<int64_t>() ==
            sink.at("contraction").get<int64_t>() &&
        upstream.at("final_drain").at("target_l1").get<bool>() &&
        !upstream.at("final_drain").at("atomic").get<bool>() &&
        sink.at("final_drain").at("atomic").get<bool>();
    if (has_replayable_split_dag) break;
  }
  check("homogeneous cube DAG sweep carries a unique split sink",
        has_replayable_split_dag);

  std::cout << passed << " passed, " << failed << " failed\n";
  return failed == 0 ? 0 : 1;
}
