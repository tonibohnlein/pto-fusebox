#include "io/io.h"
#include "pipeline/solver.h"
#include "io/verify.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

// Determine time budget (seconds) from the benchmark filename.
// Competition budgets: B1=2s, B5=5s, B9=15s, B13=30s, B17=60s.
// Unknown benchmarks get 60s.
static double get_time_budget(const std::string& filename) {
    // Extract the number from "mlsys-2026-N" pattern
    auto pos = filename.rfind("mlsys-2026-");
    if (pos != std::string::npos) {
        std::string num_str = filename.substr(pos + 11);
        // Strip .json
        auto dot = num_str.find('.');
        if (dot != std::string::npos) num_str = num_str.substr(0, dot);
        
        int num = 0;
        try { num = std::stoi(num_str); } catch (...) {}
        
        if (num >= 1 && num <= 4)   return 2.0;
        if (num >= 5 && num <= 8)   return 5.0;
        if (num >= 9 && num <= 12)  return 15.0;
        if (num >= 13 && num <= 16) return 30.0;
        if (num >= 17 && num <= 20) return 60.0;
        if (num >= 21 && num <= 25) return 120.0;
    }
    return 60.0;  // default for unknown benchmarks
}

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string(argv[1]) == "--verify") {
        return verify_examples() ? 0 : 1;
    }

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.json> <output.json>\n";
        std::cerr << "       " << argv[0] << " --verify\n";
        return 1;
    }

    auto prob = read_problem(argv[1]);
    std::cerr << "Problem: " << prob.num_tensors() << " tensors, "
              << prob.num_ops() << " ops, fast_mem=" << prob.fast_memory_capacity
              << " bw=" << prob.slow_memory_bandwidth
              << " retainable=" << prob.retainable_tensors.size()
              << "/" << prob.num_tensors() << "\n";

    // Set wall-clock deadline: 75% of budget for FM search, 25% for greedy/tabu/postopt
    double budget = get_time_budget(argv[1]);
    auto start = SteadyClock::now();
    auto deadline = start + std::chrono::milliseconds((int)(budget * 750));  // 75%
    std::cerr << "Time budget: " << budget << "s (FM deadline: " 
              << std::fixed << std::setprecision(1) << (budget * 0.75) << "s)\n";

    auto sol = solve(prob, deadline);

    auto elapsed = std::chrono::duration<double>(SteadyClock::now() - start).count();

    std::cout << "Schedule: " << sol.num_steps() << " steps\n";
    for (size_t i = 0; i < sol.num_steps(); i++) {
        const auto& s = sol.step(i);
        const auto& cfg = s.config;
        std::cout << "  Step " << i << ": ops={";
        for (size_t j = 0; j < s.subgraph.ops().size(); j++) {
            if (j) std::cout << ",";
            std::cout << s.subgraph.ops()[j];
        }
        std::cout << "} [" << cfg.w << "," << cfg.h << "," << cfg.k << "]";
        if (!s.retain_these.empty()) {
            std::cout << " retain={";
            bool first = true;
            for (auto t : s.retain_these) {
                if (!first) std::cout << ",";
                std::cout << "T" << t;
                first = false;
            }
            std::cout << "}";
        }
        std::cout << " lat=" << sol.step_latency(i) << "\n";
    }
    std::cout << "Total: " << sol.total_latency() << "\n";
    std::cerr << "Wall time: " << std::fixed << std::setprecision(1) 
              << elapsed << "s / " << budget << "s\n";

    write_solution(argv[2], sol);
    return 0;
}