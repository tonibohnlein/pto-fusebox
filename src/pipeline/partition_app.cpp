// test_partition.cpp — standalone test for init strategies + local search
//
// Build: add to your CMakeLists or compile with all .cpp files:
//   g++ -O2 -std=c++20 -o test_partition test_partition.cpp \
//       io/io.cpp core/dag.cpp core/subgraph.cpp core/cost.cpp \
//       partition/partition.cpp init/init_strategies.cpp \
//       search/local_search.cpp search/fm_outer.cpp \
//       -I. -lnlohmann_json  (or however your build links json)
//
// Usage:
//   ./test_partition <benchmark.json>
//   ./test_partition benchmarks/mlsys-2026-1.json

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
#include <cassert>

// ============================================================================
// Detailed partition validator (beyond verify_partition_feasibility)
// ============================================================================

struct DetailedValidation {
    bool valid = true;
    std::string error;
    size_t num_groups = 0;
    double total_cost = 0;
    size_t max_group_size = 0;
    size_t total_ops_covered = 0;
    bool has_recomputation = false;
};

static DetailedValidation validate_partition_detailed(
        const Partition& part, const Problem& prob, const DAG& dag) {
    DetailedValidation v;

    // --- Coverage ---
    std::vector<int> op_coverage(prob.num_ops(), 0);
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        v.num_groups++;
        v.total_cost += part.groups[gi].cost;
        v.max_group_size = std::max(v.max_group_size, part.groups[gi].ops.size());
        for (auto op : part.groups[gi].ops)
            op_coverage[op]++;
    }
    for (size_t i = 0; i < prob.num_ops(); i++) {
        if (op_coverage[i] == 0) {
            v.valid = false;
            v.error = "Op " + std::to_string(i) + " not covered by any group";
            return v;
        }
        if (op_coverage[i] > 1) v.has_recomputation = true;
        v.total_ops_covered += op_coverage[i];
    }

    // --- Memory feasibility (re-evaluate from scratch) ---
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        const auto& g = part.groups[gi];

        auto sg = Subgraph::create(prob, dag,
                      std::vector<size_t>(g.ops.begin(), g.ops.end()));
        if (!sg) {
            v.valid = false;
            v.error = "G" + std::to_string(gi) + ": Subgraph::create failed (ops:";
            for (auto op : g.ops) v.error += " " + std::to_string(op);
            v.error += ")";
            return v;
        }
        auto c = sg->best_cost();
        if (!c.feasible) {
            v.valid = false;
            v.error = "G" + std::to_string(gi) + ": no feasible tiling";
            return v;
        }

        // Check stored cost is consistent (within 1% tolerance)
        double rel_err = std::abs(c.latency - g.cost) / std::max(1.0, g.cost);
        if (rel_err > 0.01) {
            std::cerr << "  WARNING: G" << gi << " stored cost=" << g.cost
                      << " but re-evaluated=" << c.latency
                      << " (rel_err=" << rel_err << ")\n";
        }
    }

    // --- Condensed DAG acyclicity ---
    // Build a fresh condensed DAG and check for cycles via topological sort.
    {
        // Map each op to its alive group(s)
        std::vector<std::vector<size_t>> op_to_gs(prob.num_ops());
        for (size_t gi = 0; gi < part.groups.size(); gi++)
            if (part.groups[gi].alive)
                for (auto op : part.groups[gi].ops)
                    op_to_gs[op].push_back(gi);

        // Build group-level edges
        std::vector<std::set<size_t>> g_succs(part.groups.size());
        std::vector<int> g_indeg(part.groups.size(), 0);

        for (size_t gi = 0; gi < part.groups.size(); gi++) {
            if (!part.groups[gi].alive) continue;
            for (auto op : part.groups[gi].ops) {
                for (auto t : prob.ops[op].inputs) {
                    int prod = dag.tensor_producer[t];
                    if (prod < 0) continue;
                    for (auto gj : op_to_gs[(size_t)prod]) {
                        if (gj != gi && g_succs[gj].insert(gi).second)
                            g_indeg[gi]++;
                    }
                }
            }
        }

        // Kahn's algorithm
        std::vector<size_t> queue;
        for (size_t gi = 0; gi < part.groups.size(); gi++)
            if (part.groups[gi].alive && g_indeg[gi] == 0)
                queue.push_back(gi);

        size_t processed = 0;
        size_t qi = 0;
        while (qi < queue.size()) {
            size_t u = queue[qi++];
            processed++;
            for (auto w : g_succs[u])
                if (--g_indeg[w] == 0)
                    queue.push_back(w);
        }

        if (processed != v.num_groups) {
            v.valid = false;
            v.error = "Condensed group DAG has a cycle! (processed "
                      + std::to_string(processed) + " of "
                      + std::to_string(v.num_groups) + " groups)";
            return v;
        }
    }

    // --- Ephemeral gap: every boundary input must be available ---
    // For each alive group, every boundary input tensor (not a graph input)
    // must be produced as a boundary output by some alive group.
    {
        // Collect all boundary outputs across all alive groups
        std::set<size_t> all_boundary_outputs;
        for (size_t gi = 0; gi < part.groups.size(); gi++) {
            if (!part.groups[gi].alive) continue;
            auto sg = Subgraph::create(prob, dag,
                          std::vector<size_t>(part.groups[gi].ops.begin(),
                                              part.groups[gi].ops.end()));
            if (!sg) continue;
            for (auto t : sg->boundary_outputs())
                all_boundary_outputs.insert(t);
        }

        // Check every group's boundary inputs
        for (size_t gi = 0; gi < part.groups.size(); gi++) {
            if (!part.groups[gi].alive) continue;
            auto sg = Subgraph::create(prob, dag,
                          std::vector<size_t>(part.groups[gi].ops.begin(),
                                              part.groups[gi].ops.end()));
            if (!sg) continue;
            for (auto t : sg->boundary_inputs()) {
                if (dag.tensor_producer[t] < 0) continue;  // graph input — OK
                if (!all_boundary_outputs.count(t)) {
                    v.valid = false;
                    v.error = "Ephemeral gap: G" + std::to_string(gi)
                              + " needs T" + std::to_string(t)
                              + " but no group writes it as boundary output";
                    return v;
                }
            }
        }
    }

    return v;
}

// ============================================================================
// Print group details
// ============================================================================

static void print_partition(const Partition& part, const Problem& prob) {
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        const auto& g = part.groups[gi];
        std::cerr << "  G" << gi << " [";
        bool first = true;
        for (auto op : g.ops) {
            if (!first) std::cerr << ",";
            std::cerr << op;
            first = false;
        }
        std::cerr << "] cost=" << std::fixed << std::setprecision(1) << g.cost;

        // Show op types
        std::cerr << " (";
        first = true;
        for (auto op : g.ops) {
            if (!first) std::cerr << "+";
            std::cerr << (prob.ops[op].type == OpType::MatMul ? "MM" : "PW");
            first = false;
        }
        std::cerr << ")";

        if (g.best_cfg.w > 0) {
            std::cerr << " tile=[" << g.best_cfg.w << "," << g.best_cfg.h
                      << "," << g.best_cfg.k << "]";
        }
        std::cerr << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <benchmark.json> [--verbose]\n";
        return 1;
    }

    bool verbose = false;
    for (int i = 2; i < argc; i++)
        if (std::string(argv[i]) == "--verbose") verbose = true;

    auto prob = read_problem(argv[1]);
    std::cerr << "Problem: " << prob.num_ops() << " ops, "
              << prob.num_tensors() << " tensors, "
              << "fast_mem=" << prob.fast_memory_capacity
              << ", bandwidth=" << prob.slow_memory_bandwidth
              << ", native=[" << prob.native_w << "," << prob.native_h << "]\n";

    auto dag = DAG::build(prob);
    std::cerr << "DAG built: " << dag.graph_inputs.size() << " inputs, "
              << dag.graph_outputs.size() << " outputs\n\n";

    CostCache cache;

    // --- Test each init strategy ---
    auto strategies = all_init_strategies();
    for (auto& s : strategies) {
        std::cerr << "=== Strategy: " << s.name << " ===\n";
        auto t0 = std::chrono::steady_clock::now();

        auto part = s.init(prob, dag, &cache);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto v = validate_partition_detailed(part, prob, dag);

        std::cerr << "  Groups: " << v.num_groups
                  << ", Cost: " << std::fixed << std::setprecision(1) << v.total_cost
                  << ", Time: " << std::setprecision(1) << ms << "ms"
                  << ", MaxGroupSize: " << v.max_group_size
                  << (v.has_recomputation ? ", HAS RECOMPUTATION" : "")
                  << "\n";

        if (!v.valid) {
            std::cerr << "  *** VALIDATION FAILED: " << v.error << " ***\n";
        } else {
            std::cerr << "  Validation: PASS\n";
        }

        if (verbose) print_partition(part, prob);

        // --- Greedy descent ---
        auto t2 = std::chrono::steady_clock::now();
        auto descended = greedy_descent(std::move(part));
        auto t3 = std::chrono::steady_clock::now();
        double ms2 = std::chrono::duration<double, std::milli>(t3 - t2).count();

        auto v2 = validate_partition_detailed(descended, prob, dag);

        std::cerr << "  After greedy: Groups=" << v2.num_groups
                  << ", Cost=" << std::setprecision(1) << v2.total_cost
                  << ", Time=" << std::setprecision(1) << ms2 << "ms"
                  << (v2.has_recomputation ? ", HAS RECOMPUTATION" : "")
                  << "\n";

        if (!v2.valid) {
            std::cerr << "  *** POST-GREEDY VALIDATION FAILED: " << v2.error << " ***\n";
        } else {
            std::cerr << "  Post-greedy validation: PASS\n";
        }

        if (verbose) print_partition(descended, prob);
        std::cerr << "\n";
    }

    // --- Full local_search pipeline ---
    std::cerr << "=== Full local_search pipeline ===\n";
    auto t0 = std::chrono::steady_clock::now();
    auto best_part = local_search(prob, dag);
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto vf = validate_partition_detailed(best_part, prob, dag);

    std::cerr << "  Final: Groups=" << vf.num_groups
              << ", Cost=" << std::fixed << std::setprecision(1) << vf.total_cost
              << ", Time=" << std::setprecision(0) << total_ms << "ms"
              << (vf.has_recomputation ? ", HAS RECOMPUTATION" : "")
              << "\n";

    if (!vf.valid) {
        std::cerr << "  *** FINAL VALIDATION FAILED: " << vf.error << " ***\n";
        return 2;
    }
    std::cerr << "  Final validation: PASS\n";

    if (verbose) print_partition(best_part, prob);

    // --- Build Solution and validate ---
    std::cerr << "\n=== Building Solution from partition ===\n";
    auto sol = Solution::from_partition(prob, dag, std::move(best_part));

    std::cerr << "  Steps: " << sol.num_steps()
              << ", Total latency: " << std::fixed << std::setprecision(1)
              << sol.total_latency() << "\n";

    auto sv = sol.validate();
    if (!sv.valid) {
        std::cerr << "  *** SOLUTION VALIDATION FAILED: " << sv.error << " ***\n";
        return 3;
    }
    std::cerr << "  Solution validation: PASS\n";

    // --- Cache stats ---
    std::cerr << "\nCache: " << cache.size() << " entries, "
              << cache.hits() << " hits, " << cache.misses() << " misses";
    if (cache.overcapacity() > 0)
        std::cerr << ", " << cache.overcapacity() << " overcapacity";
    std::cerr << "\n";

    return 0;
}