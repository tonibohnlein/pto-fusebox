// ============================================================================
// symmetry_app: read a problem JSON, discover symmetric patterns, print them.
//
// Detects two kinds of symmetry:
//   Parallel: disconnected isomorphic components (attention heads, MLP blocks)
//   Series:   repeating motifs along a chain (stacked transformer layers)
//
// Usage:  symmetry_app <problem.json> [--verbose]
// ============================================================================

#include "io/io.h"
#include "core/dag.h"
#include "symmetry/merkle_hash.h"
#include "symmetry/symmetry.h"
#include "symmetry/series.h"
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

    auto t2 = std::chrono::steady_clock::now();
    auto dag = DAG::build(prob);
    auto t3 = std::chrono::steady_clock::now();
    std::cout << "DAG built: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms\n";
    std::cout << "  longest chain: " << dag.longest_chain() << " edges\n";

    auto t4 = std::chrono::steady_clock::now();
    auto merkle = MerkleHashes::compute(prob, dag);
    auto t5 = std::chrono::steady_clock::now();

    std::cout << "Merkle hashes: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
              << " ms (" << dag.longest_chain() << " iterations)\n";
    std::cout << "  equivalence classes: " << merkle.num_classes() << "\n";
    std::cout << "  symmetric ops:       " << merkle.symmetric_ops()
              << " / " << prob.num_ops() << "\n";

    // ---- Parallel symmetry detection ----
    std::cout << "\n=== Parallel Patterns ===\n";
    auto t6 = std::chrono::steady_clock::now();
    auto parallel = SymmetryDetector::discover(prob, dag, merkle, verbose);
    auto t7 = std::chrono::steady_clock::now();

    std::cout << "Detection: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count()
              << " ms, found " << parallel.size() << " patterns\n\n";

    if (parallel.empty()) {
        std::cout << "  (none)\n\n";
    } else {
        for (size_t pi = 0; pi < parallel.size(); pi++) {
            auto& pat = parallel[pi];
            std::cout << "PARALLEL " << pi
                      << "  sym=" << pat.symmetry
                      << "  ops_per_replica=" << pat.component_size()
                      << "  total=" << pat.symmetry * pat.component_size()
                      << "  coverage=" << std::fixed << std::setprecision(1)
                      << (100.0 * pat.symmetry * pat.component_size() / prob.num_ops())
                      << "%\n";
            for (size_t ci = 0; ci < pat.components.size(); ci++) {
                std::cout << "  replica " << ci << ":";
                size_t printed = 0;
                for (auto op : pat.components[ci]) {
                    std::cout << " " << op;
                    if (++printed >= 20) {
                        std::cout << " ...+" << (pat.components[ci].size() - 20);
                        break;
                    }
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    // ---- Series pattern detection ----
    std::cout << "=== Series Patterns ===\n";
    auto t8 = std::chrono::steady_clock::now();
    auto series_pats = SeriesDetector::discover(prob, dag, merkle, verbose);
    auto t9 = std::chrono::steady_clock::now();

    std::cout << "Detection: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t9 - t8).count()
              << " ms, found " << series_pats.size() << " patterns\n\n";

    if (series_pats.empty()) {
        std::cout << "  (none)\n\n";
    } else {
        for (size_t pi = 0; pi < series_pats.size(); pi++) {
            auto& pat = series_pats[pi];
            std::cout << "SERIES " << pi
                      << "  blocks=" << pat.num_blocks
                      << "  block_size=" << pat.block_size
                      << "  total=" << pat.total_ops()
                      << "  coverage=" << std::fixed << std::setprecision(1)
                      << pat.coverage(prob.num_ops()) << "%\n";
            for (size_t bi = 0; bi < pat.blocks.size() && bi < 6; bi++) {
                std::cout << "  block " << bi << ":";
                for (auto op : pat.blocks[bi])
                    std::cout << " " << op;
                std::cout << "\n";
            }
            if (pat.blocks.size() > 6)
                std::cout << "  ...+" << (pat.blocks.size() - 6) << " more\n";
            std::cout << "\n";
        }
    }

    // ---- Summary ----
    size_t parallel_ops = 0, series_ops = 0;
    for (auto& p : parallel) parallel_ops += p.symmetry * p.component_size();
    for (auto& s : series_pats) series_ops += s.total_ops();
    std::cout << "=== Summary ===\n";
    std::cout << "  parallel: " << parallel_ops << "/" << prob.num_ops() << " ops covered\n";
    std::cout << "  series:   " << series_ops << "/" << prob.num_ops() << " ops covered\n";
    if (!parallel.empty())
        std::cout << "  best parallel: sym=" << parallel[0].symmetry
                  << " -> solve 1/" << parallel[0].symmetry << "\n";
    if (!series_pats.empty())
        std::cout << "  best series: " << series_pats[0].num_blocks
                  << " blocks of " << series_pats[0].block_size
                  << " -> solve 1/" << series_pats[0].num_blocks << "\n";

    return 0;
}