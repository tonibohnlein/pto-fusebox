// ============================================================================
// evaluate.cpp — Standalone solution evaluator using our cost model.
//
// Usage: ./evaluate <problem.json> <solution.json>
//
// Reads a problem instance and solution, creates Subgraphs for each step,
// evaluates working_set + compute_cost, and reports per-step and total results.
// Also validates feasibility, op coverage, and retention rules.
//
// Build: link with io.cpp, cost.cpp, dag.cpp, subgraph.cpp (+ nlohmann/json)
// ============================================================================

#include "io/io.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <set>
#include <map>
#include <cmath>

using json = nlohmann::json;

// Detect SnakeDir from a traversal order by comparing against known patterns
static SnakeDir detect_snake(const std::vector<int>& order, int ntw, int nth) {
    if (order.empty()) return SnakeDir::None;
    auto rm = make_traversal(ntw, nth, SnakeDir::RowMajor);
    if (order == rm) return SnakeDir::RowMajor;
    auto cm = make_traversal(ntw, nth, SnakeDir::ColMajor);
    if (order == cm) return SnakeDir::ColMajor;
    auto raster = make_traversal(ntw, nth, SnakeDir::None);
    if (order == raster) return SnakeDir::None;
    // Unknown pattern — try all and report mismatch
    return SnakeDir::None;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <problem.json> <solution.json>\n";
        return 1;
    }

    // ---- Read problem ----
    Problem prob = read_problem(argv[1]);
    DAG dag = DAG::build(prob);

    std::cerr << "Problem: " << prob.num_tensors() << " tensors, "
              << prob.num_ops() << " ops, fast_mem=" << prob.fast_memory_capacity
              << " bw=" << prob.slow_memory_bandwidth
              << " native=[" << prob.native_w << "," << prob.native_h << "]\n";

    // ---- Read solution ----
    std::ifstream sf(argv[2]);
    if (!sf.is_open()) {
        std::cerr << "Error: cannot open solution '" << argv[2] << "'\n";
        return 1;
    }
    json sol;
    try { sol = json::parse(sf); }
    catch (const json::parse_error& e) {
        std::cerr << "Error: failed to parse solution: " << e.what() << "\n";
        return 1;
    }

    auto& sg_json   = sol["subgraphs"];
    auto& gran_json = sol["granularities"];
    auto& ret_json  = sol["tensors_to_retain"];
    auto& trav_json = sol["traversal_orders"];

    size_t num_steps = sg_json.size();
    if (gran_json.size() != num_steps || ret_json.size() != num_steps) {
        std::cerr << "Error: subgraphs/granularities/tensors_to_retain size mismatch\n";
        return 1;
    }

    bool has_latencies = sol.contains("subgraph_latencies") &&
                         sol["subgraph_latencies"].size() == num_steps;

    // ---- Evaluate each step ----
    std::set<size_t> covered_ops;
    std::set<size_t> entering;  // retained from previous step
    double total_our = 0;
    double total_ref = 0;
    bool all_feasible = true;

    std::cerr << "\n" << std::string(110, '=') << "\n";
    std::cerr << std::setw(4) << "Step"
              << std::setw(6) << "Ops"
              << std::setw(18) << "Granularity"
              << std::setw(8) << "Snake"
              << std::setw(10) << "WS"
              << std::setw(10) << "Capacity"
              << std::setw(8) << "Feas"
              << std::setw(14) << "Our Cost"
              << std::setw(14) << "Ref Cost"
              << std::setw(10) << "Delta%"
              << std::setw(8) << "Retain"
              << "\n";
    std::cerr << std::string(110, '-') << "\n";

    for (size_t i = 0; i < num_steps; i++) {
        // Parse ops
        std::vector<size_t> ops;
        for (auto& o : sg_json[i]) ops.push_back(o.get<size_t>());
        for (auto o : ops) covered_ops.insert(o);

        // Parse granularity
        auto& g = gran_json[i];
        int64_t w = g[0].get<int64_t>();
        int64_t h = g[1].get<int64_t>();
        int64_t k = g[2].get<int64_t>();

        // Parse retain
        std::set<size_t> retain;
        for (auto& t : ret_json[i]) retain.insert(t.get<size_t>());

        // Parse traversal → detect snake
        // Default to "hsnake" (RowMajor) for all subgraphs.
        // When traversal_order is empty or null, use RowMajor as default.
        SnakeDir snake = SnakeDir::RowMajor;  // match reference default
        if (i < trav_json.size() && !trav_json[i].is_null() && !trav_json[i].empty()) {
            auto sg_opt = Subgraph::create(prob, dag, ops);
            if (sg_opt) {
                int ntw = std::max((int)(sg_opt->output_width() / w), 1);
                int nth = std::max((int)(sg_opt->output_height() / h), 1);
                std::vector<int> trav;
                for (auto& v : trav_json[i]) trav.push_back(v.get<int>());
                snake = detect_snake(trav, ntw, nth);
            }
        }

        TileConfig cfg{w, h, k, snake};

        // Create subgraph
        auto sg_opt = Subgraph::create(prob, dag, ops);
        if (!sg_opt) {
            std::cerr << std::setw(4) << i
                      << "  *** SUBGRAPH CREATION FAILED (ops:";
            for (auto o : ops) std::cerr << " " << o;
            std::cerr << ") ***\n";
            all_feasible = false;
            entering = retain;
            continue;
        }

        // Evaluate
        auto& sg = *sg_opt;
        int64_t ws = sg.working_set(cfg, entering, retain);
        auto cost = sg.compute_cost(cfg, entering, retain);

        double ref_lat = has_latencies ? sol["subgraph_latencies"][i].get<double>() : 0;
        double our_lat = cost.latency;
        double delta_pct = (ref_lat > 0.01) ? 100.0 * (our_lat - ref_lat) / ref_lat : 0;

        bool feasible = cost.feasible;
        if (!feasible) all_feasible = false;

        total_our += our_lat;
        total_ref += ref_lat;

        // Check valid tiling
        bool valid_tiling = sg.is_valid_tiling(cfg);

        // Retain validation: only boundary outputs
        bool retain_ok = true;
        for (auto t : retain) {
            if (!sg.boundary_outputs().count(t)) {
                retain_ok = false;
                break;
            }
        }

        std::string snake_str = (snake == SnakeDir::RowMajor) ? "RM" :
                                (snake == SnakeDir::ColMajor) ? "CM" : "--";

        std::cerr << std::setw(4) << i
                  << std::setw(6) << ops.size()
                  << "  [" << std::setw(4) << w << "," << std::setw(4) << h
                  << "," << std::setw(4) << k << "]"
                  << std::setw(5) << snake_str
                  << std::setw(10) << ws
                  << std::setw(10) << prob.fast_memory_capacity
                  << std::setw(6) << (feasible ? "  OK" : "FAIL")
                  << std::setw(2) << (valid_tiling ? "" : "T!")
                  << std::fixed << std::setprecision(1)
                  << std::setw(14) << our_lat
                  << std::setw(14) << ref_lat
                  << std::setw(9) << std::showpos << delta_pct << std::noshowpos << "%"
                  << "  ";

        if (!retain.empty()) {
            std::cerr << "{";
            bool first = true;
            for (auto t : retain) {
                if (!first) std::cerr << ",";
                std::cerr << "T" << t;
                first = false;
            }
            std::cerr << "}";
            if (!retain_ok) std::cerr << " !!NOT_BOUNDARY_OUT";
        }

        std::cerr << "\n";

        // Detail on infeasibility
        if (!feasible) {
            std::cerr << "      WS breakdown: ";
            // Show entering and retain overhead
            int64_t enter_size = 0;
            for (auto t : entering) enter_size += prob.tensors[t].size();
            int64_t retain_size = 0;
            for (auto t : retain) retain_size += prob.tensors[t].size();
            std::cerr << "entering=" << enter_size << " retain=" << retain_size
                      << " boundary_in=" << sg.boundary_inputs().size()
                      << " boundary_out=" << sg.boundary_outputs().size()
                      << " eph=" << sg.ephemeral().size();
            if (!valid_tiling) std::cerr << " INVALID_TILING";
            std::cerr << "\n";
        }

        // Also try best_cost to see optimal config
        if (!feasible || our_lat > ref_lat * 1.5) {
            auto best = sg.best_cost(entering, retain);
            if (best.feasible) {
                std::cerr << "      best_cost: [" << best.config.w << ","
                          << best.config.h << "," << best.config.k << "] lat="
                          << best.latency << " ws=" << best.working_set << "\n";
            } else {
                auto best_bare = sg.best_cost({}, {});
                if (best_bare.feasible) {
                    std::cerr << "      best_cost(bare): [" << best_bare.config.w << ","
                              << best_bare.config.h << "," << best_bare.config.k
                              << "] lat=" << best_bare.latency
                              << " ws=" << best_bare.working_set << "\n";
                } else {
                    std::cerr << "      NO feasible config found!\n";
                }
            }
        }

        entering = retain;
    }

    std::cerr << std::string(110, '-') << "\n";

    // ---- Op coverage ----
    bool full_coverage = true;
    for (size_t i = 0; i < prob.num_ops(); i++) {
        if (!covered_ops.count(i)) {
            std::cerr << "  MISSING: Op " << i << " not covered by any step\n";
            full_coverage = false;
        }
    }

    // ---- Summary ----
    std::cerr << "\n=== Summary ===\n";
    std::cerr << "  Steps:      " << num_steps << "\n";
    std::cerr << "  Coverage:   " << covered_ops.size() << "/" << prob.num_ops()
              << (full_coverage ? " (complete)" : " (INCOMPLETE!)") << "\n";
    std::cerr << "  Feasible:   " << (all_feasible ? "YES" : "NO") << "\n";
    std::cerr << std::fixed << std::setprecision(1);
    std::cerr << "  Our total:  " << total_our << "\n";
    if (has_latencies) {
        std::cerr << "  Ref total:  " << total_ref << "\n";
        double total_delta = 100.0 * (total_our - total_ref) / total_ref;
        std::cerr << "  Delta:      " << std::showpos << total_delta
                  << std::noshowpos << "%\n";
    }

    // ---- Also try our best_cost for each step (what we'd choose) ----
    std::cerr << "\n=== Our optimal per-step (no retention context) ===\n";
    double total_bare = 0;
    entering.clear();
    for (size_t i = 0; i < num_steps; i++) {
        std::vector<size_t> ops;
        for (auto& o : sg_json[i]) ops.push_back(o.get<size_t>());
        auto sg_opt = Subgraph::create(prob, dag, ops);
        if (!sg_opt) continue;
        auto best = sg_opt->best_cost({}, {});
        if (best.feasible) {
            total_bare += best.latency;
            std::string snake_str = (best.config.snake == SnakeDir::RowMajor) ? "RM" :
                                    (best.config.snake == SnakeDir::ColMajor) ? "CM" : "--";
            std::cerr << "  Step " << std::setw(2) << i << ": ["
                      << std::setw(4) << best.config.w << ","
                      << std::setw(4) << best.config.h << ","
                      << std::setw(4) << best.config.k << "] "
                      << snake_str << "  lat=" << std::setw(12) << best.latency
                      << "  ws=" << best.working_set << "\n";
        } else {
            std::cerr << "  Step " << std::setw(2) << i << ": INFEASIBLE (bare)\n";
            total_bare += 1e18;
        }
    }
    std::cerr << "  Bare total: " << total_bare << "\n";

    // Output JSON-style result to stdout for scripting
    std::cout << "{\n"
              << "  \"feasible\": " << (all_feasible ? "true" : "false") << ",\n"
              << "  \"coverage\": " << (full_coverage ? "true" : "false") << ",\n"
              << "  \"our_total\": " << std::fixed << std::setprecision(2) << total_our << ",\n"
              << "  \"ref_total\": " << total_ref << ",\n"
              << "  \"bare_total\": " << total_bare << ",\n"
              << "  \"num_steps\": " << num_steps << "\n"
              << "}\n";

    return (all_feasible && full_coverage) ? 0 : 1;
}