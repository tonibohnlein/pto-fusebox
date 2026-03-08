#include "io/io.h"
#include "pipeline/solver.h"
#include "io/verify.h"
#include <iostream>
#include <string>

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

    auto sol = solve(prob);

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

    write_solution(argv[2], sol);
    return 0;
}
