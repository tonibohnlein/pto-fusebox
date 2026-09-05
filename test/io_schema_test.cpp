#include "core/dag.h"
#include "io/io.h"
#include "solution/solution.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unistd.h>

static int g_pass = 0;
static int g_fail = 0;

static void check(const char* label, bool condition) {
  if (condition) {
    ++g_pass;
  } else {
    ++g_fail;
    std::cout << "FAIL: " << label << "\n";
  }
}

int main() {
  const std::string path = "/tmp/pto_fusebox_io_schema_test_" +
                           std::to_string(static_cast<long long>(getpid())) + ".json";
  std::ofstream file(path);
  file << R"JSON({
    "schema_version": "pto_fusebox.problem.v1",
    "widths": [64, 1, 64],
    "heights": [8, 8, 8],
    "dtypes": ["FP32", "FP32", "FP32"],
    "inputs": [[0], [0, 1]],
    "outputs": [[1], [2]],
    "op_types": ["Reduction", "Pointwise"],
    "vector_primitive_families": ["row_sum", "add"],
    "vector_op_geometries": ["flat", "row_expand"],
    "vector_op_capabilities": ["reduction_sum", "elementwise"],
    "mixed_vector_semantics": ["none", "none"],
    "mixed_emit_compatible": [true, false],
    "required_outputs": [2],
    "p4_patterns": [{"kind": "softmax_flash", "ops": [0, 1],
                     "apply_substitutions": [{"op": 0,
                                               "value": "running_max"},
                                              {"op": 1,
                                               "value": "running_sum"}]}],
    "fast_memory_capacity": 1073741824,
    "cube_freq_hz": 1850000000.0,
    "vec_dma_align_bytes": 64,
    "tcvt_safe_fragment_widths": [
      {"source_dtype": "INT32", "target_dtype": "FP16", "width": 128}
    ],
    "require_source_codegen": true,
    "allow_model_ahead_split_k": false,
    "allow_model_ahead_multi_reduction_stream": false
  })JSON";
  file.close();

  Problem problem = read_problem(path);
  check("problem schema parses", problem.num_ops() == 2);
  check("P4 pattern parses", problem.p4_patterns.size() == 1);
  check("P4 substitution parses", problem.p4_patterns[0].apply_substitutions.count(0) == 1);
  check("P4 named binding parses",
        problem.p4_patterns[0].apply_bindings.size() == 2 &&
            problem.p4_patterns[0].apply_bindings[0].op == 0 &&
            problem.p4_patterns[0].apply_bindings[0].value ==
                P4SubstitutionValue::RunningMax);
  check("mixed compatibility parses", !problem.ops[1].mixed_emit_compatible);
  check("DMA alignment parses", problem.vec_dma_align_bytes == 64);
  check("tcvt safe fragment parses",
        problem.tcvt_safe_fragment_widths.size() == 1 &&
            problem.tcvt_safe_fragment_widths[0].source_dtype == DType::INT32 &&
            problem.tcvt_safe_fragment_widths[0].target_dtype == DType::FP16 &&
            problem.tcvt_safe_fragment_widths[0].width == 128);
  check("external source policy parses", problem.require_source_codegen);
  check("model-ahead split-K parses", !problem.allow_model_ahead_split_k);
  check("model-ahead P4 parses", !problem.allow_model_ahead_multi_reduction_stream);

  DAG dag = DAG::build(problem);
  Solution empty(problem, dag, {});
  const auto solution = nlohmann::json::parse(solution_json(empty));
  check("solution schema is published",
        solution.at("schema_version") == "pto_fusebox.solution.v7");
  check("solution steps are nested", solution.at("steps").is_array());
  check("empty solution has no steps", solution.at("steps").empty());

  auto producer = Subgraph::create(problem, dag, {0});
  check("retention serializer control subgraph creates", producer.has_value());
  if (producer) {
    const auto cost = producer->best_cost();
    Solution retained(problem, dag,
                      {{*producer, cost.config, FlatSet<size_t>{1}}});
    bool rejected = false;
    try {
      static_cast<void>(solution_json(retained));
    } catch (const std::logic_error&) {
      rejected = true;
    }
    check("solution schema rejects cross-kernel retention", rejected);
  }

  std::remove(path.c_str());
  std::cout << g_pass << " passed, " << g_fail << " failed\n";
  return g_fail == 0 ? 0 : 1;
}
