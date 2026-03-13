#include "io/io.h"
#include "pipeline/solver.h"
#include "search/verbose.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>

static double get_time_budget(const std::string& filename) {
    auto pos = filename.rfind("mlsys-2026-");
    if (pos != std::string::npos) {
        std::string num_str = filename.substr(pos + 11);
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
    return 60.0;
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-v") verbose = true;
        else args.push_back(argv[i]);
    }
    g_verbose = verbose;

    if (args.size() != 2) {
        std::cerr << "Usage: " << argv[0] << " [-v] <input.json> <output.json>\n";
        return 1;
    }

    auto prob = read_problem(args[0]);

    int n_mm = 0, n_pw = 0;
    for (size_t i = 0; i < prob.num_ops(); i++) {
        if (prob.ops[i].type == OpType::MatMul) n_mm++; else n_pw++;
    }

    int64_t total_tensor_size = 0, max_tensor = 0;
    for (size_t i = 0; i < prob.num_tensors(); i++) {
        int64_t s = prob.tensors[i].size();
        total_tensor_size += s;
        max_tensor = std::max(max_tensor, s);
    }

    std::cerr << "Problem: " << prob.num_tensors() << " tensors, "
              << prob.num_ops() << " ops (" << n_mm << " MM + " << n_pw << " PW)"
              << ", fast_mem=" << prob.fast_memory_capacity
              << " bw=" << prob.slow_memory_bandwidth
              << " native=[" << prob.native_w << "," << prob.native_h << "]"
              << "\n         retainable=" << prob.retainable_tensors.size()
              << "/" << prob.num_tensors()
              << " max_tensor=" << max_tensor
              << " total_data=" << std::fixed << std::setprecision(1)
              << total_tensor_size / 1e6 << "M\n";

    // Use 95% of budget (5% safety margin for I/O)
    double budget = get_time_budget(args[0]);
    auto start = SteadyClock::now();
    auto deadline = start + std::chrono::milliseconds((int)(budget * 950));
    std::cerr << "Time budget: " << std::fixed << std::setprecision(0)
              << budget << "s (deadline=" << (budget * 0.95) << "s)\n\n";

    auto sol = solve(prob, deadline);

    auto elapsed = std::chrono::duration<double>(SteadyClock::now() - start).count();

    // Detailed schedule
    double total_lat = sol.total_latency();
    std::cerr << "\nSchedule: " << sol.num_steps() << " steps\n";

    for (size_t i = 0; i < sol.num_steps(); i++) {
        const auto& s = sol.step(i);
        const auto& cfg = s.config;

        int step_mm = 0, step_pw = 0;
        for (auto op : s.subgraph.ops()) {
            if (prob.ops[op].type == OpType::MatMul) step_mm++; else step_pw++;
        }
        std::string type_tag = (step_mm && step_pw) ? "MM+PW" :
                               (step_mm ? "MM" : "PW");
        std::string snake_str = cfg.snake == SnakeDir::RowMajor ? "RM" :
                                cfg.snake == SnakeDir::ColMajor ? "CM" : "--";

        double lat = sol.step_latency(i);
        double pct = 100.0 * lat / total_lat;

        std::cerr << "  Step " << std::setw(2) << i << ": "
                  << std::setw(3) << s.subgraph.ops().size() << " ops"
                  << " (" << std::setw(5) << type_tag << ")"
                  << " [" << std::setw(4) << cfg.w
                  << "," << std::setw(4) << cfg.h
                  << "," << std::setw(4) << cfg.k << "]"
                  << " " << snake_str
                  << " lat=" << std::setw(12) << std::fixed << std::setprecision(1) << lat
                  << " (" << std::setw(4) << std::setprecision(1) << pct << "%)";
        if (!s.retain_these.empty()) {
            std::cerr << " retain={";
            bool first = true;
            for (auto t : s.retain_these) {
                if (!first) std::cerr << ",";
                std::cerr << "T" << t;
                first = false;
            }
            std::cerr << "}";
        }
        std::cerr << "\n";
    }

    std::cerr << "\nTotal: " << std::fixed << std::setprecision(1) << total_lat
              << "  Wall: " << std::setprecision(1) << elapsed << "s / " << budget << "s"
              << " (" << std::setprecision(0) << (100.0 * elapsed / budget) << "%)\n";

    write_solution(args[1], sol);
    return 0;
}