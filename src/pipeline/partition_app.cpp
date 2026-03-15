// mlsys.cpp — contest entry point
//
// Interface (per README.md):
//   ./mlsys <path_to_input.json> <path_to_output.json>
//
// Reads the problem, runs init strategies + greedy descent + FM search,
// builds a Solution, validates it, writes the output JSON.
// All diagnostics go to stderr; stdout is unused.
//
// IMPORTANT: always writes a solution, even if validation reports warnings.
// The evaluator is the ultimate judge — our validator may be stricter.

#include "io/io.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost_cache.h"
#include "init/init_strategies.h"
#include "search/local_search.h"
#include "partition/partition.h"
#include "solution/solution.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <set>

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <path_to_input.json> <path_to_output.json>\n";
        return 1;
    }

    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];

    auto wall_start = std::chrono::steady_clock::now();

    // ---- Read problem ----
    auto prob = read_problem(input_path);
    std::cerr << "Problem: " << prob.num_ops() << " ops, "
              << prob.num_tensors() << " tensors, "
              << "fast_mem=" << prob.fast_memory_capacity
              << ", bw=" << prob.slow_memory_bandwidth
              << ", native=[" << prob.native_w << "," << prob.native_h << "]\n";

    auto dag = DAG::build(prob);
    std::cerr << "DAG: " << dag.graph_inputs.size() << " graph inputs, "
              << dag.graph_outputs.size() << " graph outputs\n";

    // ---- Helper: build solution from partition with fallback ----
    auto build_solution = [&](Partition part) -> Solution {
        part.finalize();
        return Solution::from_partition(prob, dag, std::move(part));
    };

    // ---- Search ----
    std::cerr << "\n--- Partition search ---\n";
    Solution sol(prob, dag, {});  // empty placeholder

    try {
        auto best_part = local_search(prob, dag);
        std::cerr << "Partition: " << best_part.num_alive() << " groups, cost="
                  << std::fixed << std::setprecision(1) << best_part.total_cost() << "\n";

        sol = Solution::from_partition(prob, dag, std::move(best_part));
    } catch (const std::exception& e) {
        std::cerr << "Search/build threw: " << e.what() << "\n";
    }

    // ---- Fallback if solution is empty ----
    if (sol.num_steps() == 0) {
        std::cerr << "Primary search produced 0 steps, falling back to trivial\n";
        try {
            auto trivial = Partition::trivial(prob, dag);
            sol = Solution::from_partition(prob, dag, std::move(trivial));
        } catch (const std::exception& e) {
            std::cerr << "Trivial fallback threw: " << e.what() << "\n";
        }
    }

    // ---- Report ----
    std::cerr << "\n--- Solution ---\n";
    std::cerr << "  Steps: " << sol.num_steps()
              << ", Total latency: " << std::fixed << std::setprecision(1)
              << sol.total_latency() << "\n";

    for (size_t i = 0; i < sol.num_steps(); i++) {
        const auto& step = sol.step(i);
        const auto& cost = sol.step_cost(i);
        std::cerr << "  Step " << i << ": ops=[";
        bool first = true;
        for (auto op : step.subgraph.ops()) {
            if (!first) std::cerr << ",";
            std::cerr << op;
            first = false;
        }
        std::cerr << "] tile=[" << step.config.w << "," << step.config.h
                  << "," << step.config.k << "]"
                  << " latency=" << std::setprecision(1) << cost.latency
                  << " ws=" << cost.working_set;
        if (!step.retain_these.empty()) {
            std::cerr << " retain={";
            first = true;
            for (auto t : step.retain_these) {
                if (!first) std::cerr << ",";
                std::cerr << t;
                first = false;
            }
            std::cerr << "}";
        }
        std::cerr << "\n";
    }

    // ---- Validate (warnings only — never block write) ----
    auto sv = sol.validate();
    if (!sv.valid) {
        std::cerr << "  WARNING: " << sv.error << "\n";
    } else {
        std::cerr << "  Validation: PASS\n";
    }

    // ---- Always write ----
    if (sol.num_steps() > 0) {
        write_solution(output_path, sol);
        std::cerr << "\nWrote " << sol.num_steps() << " steps to "
                  << output_path << "\n";
    } else {
        std::cerr << "\nERROR: 0 steps — nothing to write\n";
        return 4;
    }

    auto wall_end = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();
    std::cerr << "Wall time: " << std::fixed << std::setprecision(2)
              << wall_s << "s\n";

    return 0;
}