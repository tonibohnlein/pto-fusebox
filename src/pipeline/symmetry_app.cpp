// ============================================================================
// symmetry_app: read a problem JSON, discover symmetric patterns, print them.
//
// Usage:  symmetry_app <problem.json> [--verbose]
//
// Output (one line per pattern, parseable):
//   PATTERN sym=<N> ops_per_replica=<M> total_ops=<N*M>
//     replica 0: op1 op2 op3 ...
//     replica 1: op4 op5 op6 ...
//     ...
// ============================================================================

#include "io/io.h"
#include "core/dag.h"
#include "symmetry/merkle_hash.h"
#include "core/symmetry.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.json> [--verbose]\n";
        return 1;
    }

    std::string filename = argv[1];
    bool verbose = false;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v")
            verbose = true;
    }

    // ---- Read problem ----
    auto t0 = std::chrono::steady_clock::now();
    auto prob = read_problem(filename);
    auto t1 = std::chrono::steady_clock::now();

    std::cout << "Problem: " << filename << "\n";
    std::cout << "  ops:     " << prob.num_ops() << "\n";
    std::cout << "  tensors: " << prob.num_tensors() << "\n";
    std::cout << "  fast_mem:" << prob.fast_memory_capacity << "\n";
    std::cout << "  read:    "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n\n";

    // ---- Build DAG ----
    auto t2 = std::chrono::steady_clock::now();
    auto dag = DAG::build(prob);
    auto t3 = std::chrono::steady_clock::now();
    std::cout << "DAG built: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms\n";

    // ---- Merkle hashes ----
    auto t4 = std::chrono::steady_clock::now();
    auto merkle = MerkleHashes::compute(prob, dag);
    auto t5 = std::chrono::steady_clock::now();

    std::cout << "Merkle hashes: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
              << " ms\n";
    std::cout << "  equivalence classes: " << merkle.num_classes() << "\n";
    std::cout << "  symmetric ops:       " << merkle.symmetric_ops()
              << " / " << prob.num_ops() << "\n";

    // Print orbits
    if (verbose) {
        std::cout << "\nMerkle orbits:\n";
        size_t orbit_id = 0;
        for (auto& [hash, ops] : merkle.equiv_classes) {
            std::cout << "  orbit " << orbit_id++ << " (size "
                      << std::setw(3) << ops.size() << "): ";
            // Print type
            if (!ops.empty()) {
                auto& op = prob.ops[ops[0]];
                std::cout << (op.type == OpType::MatMul ? "MM " : "PW ");
                // Print output tensor shape
                if (!op.outputs.empty()) {
                    auto& t = prob.tensors[op.outputs[0]];
                    std::cout << t.width << "x" << t.height << " ";
                }
            }
            std::cout << "[";
            for (size_t i = 0; i < ops.size(); i++) {
                if (i) std::cout << ",";
                if (i >= 20) { std::cout << "...+" << (ops.size() - 20); break; }
                std::cout << ops[i];
            }
            std::cout << "]\n";
        }
    }

    // ---- Symmetry detection ----
    std::cout << "\n";
    auto t6 = std::chrono::steady_clock::now();
    auto patterns = SymmetryDetector::discover(prob, dag, merkle, verbose);
    auto t7 = std::chrono::steady_clock::now();

    std::cout << "Symmetry detection: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count()
              << " ms\n";
    std::cout << "  patterns found: " << patterns.size() << "\n\n";

    // ---- Print patterns ----
    if (patterns.empty()) {
        std::cout << "No symmetric patterns detected.\n";
    } else {
        for (size_t pi = 0; pi < patterns.size(); pi++) {
            auto& pat = patterns[pi];
            size_t total = pat.symmetry * pat.component_size();
            double coverage = 100.0 * (double)total / (double)prob.num_ops();

            std::cout << "PATTERN " << pi
                      << "  sym=" << pat.symmetry
                      << "  ops_per_replica=" << pat.component_size()
                      << "  total_ops=" << total
                      << "  coverage=" << std::fixed << std::setprecision(1)
                      << coverage << "%\n";

            for (size_t ci = 0; ci < pat.components.size(); ci++) {
                std::cout << "  replica " << ci << ":";
                size_t printed = 0;
                for (auto op : pat.components[ci]) {
                    std::cout << " " << op;
                    if (++printed >= 30 && pat.components[ci].size() > 30) {
                        std::cout << " ...+" << (pat.components[ci].size() - 30);
                        break;
                    }
                }
                std::cout << "\n";
            }

            // Print op types within representative component
            if (verbose && !pat.components.empty()) {
                std::map<OpType, int> type_count;
                for (auto op : pat.components[0])
                    type_count[prob.ops[op].type]++;
                std::cout << "  composition:";
                for (auto [type, cnt] : type_count)
                    std::cout << " " << (type == OpType::MatMul ? "MM" : "PW")
                              << "=" << cnt;
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        // ---- Summary ----
        if (patterns.size() > 1) {
            std::cout << "Summary:\n";
            size_t max_sym = patterns[0].symmetry;
            size_t max_region = 0;
            for (auto& p : patterns)
                max_region = std::max(max_region, p.component_size());
            std::cout << "  max symmetry factor: " << max_sym << "\n";
            std::cout << "  largest symmetric region: " << max_region << " ops\n";
            std::cout << "  search reduction: solve 1 replica instead of "
                      << max_sym << " → ~"
                      << (int)(100.0 * (1.0 - 1.0 / max_sym))
                      << "% less work\n";
        }
    }

    return 0;
}