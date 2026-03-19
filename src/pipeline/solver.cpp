#include "pipeline/solver.h"
#include "partition/partition.h"
#include "search/parallel_search.h"
#include "search/solution_search.h"
#include "solution/ordering.h"
#include "symmetry/symmetry.h"
#include "symmetry/series.h"
#include <algorithm>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <thread>
#include <vector>

// ============================================================================
// Full pipeline
// ============================================================================

Solution solve(const Problem& prob, const DAG& dag, TimePoint deadline) {

    auto now              = SteadyClock::now();
    auto effective_dl     = deadline;
    if (deadline == TimePoint::max() ||
        (deadline - now) > std::chrono::seconds(300))
        effective_dl = now + std::chrono::seconds(5);
    auto total_budget = effective_dl - now;

    auto feasibly_ret = compute_feasibly_retainable(prob, dag);
    bool has_retain   = !feasibly_ret.empty();

    std::cerr << "  Retainable tensors: " << feasibly_ret.size()
              << " / " << prob.retainable_tensors.size() << "\n";

    TimePoint phase1_dl, phase2_dl, phase3_dl;
    if (has_retain) {
        phase1_dl = now + std::chrono::duration_cast<SteadyClock::duration>(total_budget * 35 / 100);
        phase2_dl = phase1_dl + std::chrono::duration_cast<SteadyClock::duration>(total_budget * 5 / 100);
        phase3_dl = now + total_budget;
    } else {
        phase1_dl = now + std::chrono::duration_cast<SteadyClock::duration>(total_budget * 95 / 100);
        phase2_dl = now + total_budget;
        phase3_dl = now;
    }

    // ================================================================
    // Phase 1: Partition pool via parallel search
    // ================================================================
    std::cerr << "Phase 1: Parallel search...\n";
    ParallelConfig pcfg;
    pcfg.fm.deadline = phase1_dl;
    // When there are no retainable tensors Phase 3 is skipped entirely, so
    // Phase 1 owns ~95% of the budget. Disable early-stop so the evo loop
    // keeps running until the deadline rather than quitting after 25 stagnant
    // generations (~1s) and leaving 12s unused.
    pcfg.early_stop = has_retain;

    // Collect discovered symmetry patterns for later use (mutations, etc.)
    std::vector<SymmetricPattern> parallel_patterns;
    std::vector<SeriesPattern> series_patterns;
    pcfg.out_parallel_patterns = &parallel_patterns;
    pcfg.out_series_patterns = &series_patterns;

    auto partition_pool = parallel_search(prob, dag, pcfg);

    if (!parallel_patterns.empty() || !series_patterns.empty())
        std::cerr << "  Discovered: " << parallel_patterns.size()
                  << " parallel + " << series_patterns.size()
                  << " series patterns\n";

    std::cerr << "  Partition pool: " << partition_pool.size()
              << " entries, best=" << partition_pool[0].total_cost() << "\n";

    // parallel_search() stack-allocates its CostCache and stores raw pointers
    // to it in every partition.  That cache is destroyed when parallel_search
    // returns.  Null every cache pointer now so finalize() falls back to
    // Subgraph::create directly (safe, cheap: only pool_size partitions × N groups).
    for (auto& p : partition_pool)
        p.cache = nullptr;

    // ================================================================
    // Phase 2: Build solution pool from each partition
    // ================================================================
    std::cerr << "Phase 2: Build solutions from " << partition_pool.size()
              << " partitions" << (has_retain ? "" : " (no-retain fast path)")
              << "...\n";

    std::vector<Solution> solution_pool;
    std::mutex            sol_mutex;

    {
        int hw_threads = (int)std::thread::hardware_concurrency();
        if (hw_threads <= 0) hw_threads = 4;
        int            n_tasks = (int)partition_pool.size();
        std::atomic<int> next_task{0};

        auto build_worker = [&]() {
            while (true) {
                int tid = next_task.fetch_add(1);
                if (tid >= n_tasks) break;
                if (SteadyClock::now() >= phase2_dl) break;

                // Pre-finalize acyclicity check: reject partitions that Phase 1
                // left in a broken state (e.g., TENSOR_EXTRACT introduced a cycle).
                // rebuild_index is needed for is_acyclic's Kahn's algorithm.
                partition_pool[tid].rebuild_index();
                if (!partition_pool[tid].is_acyclic()) {
                    std::lock_guard<std::mutex> lock(sol_mutex);
                    std::cerr << "    Partition " << tid
                              << " SKIPPED: is_acyclic() failed before finalize\n";
                    continue;
                }

                // Finalize once: re-populates Group::sg, best_cfg, and the
                // group-level DAG (all stale after Phase 1 search mutations).
                // Each thread owns its own tid slot, so this mutation is safe.
                partition_pool[tid].finalize();
                const Partition& part = partition_pool[tid];

                // Diagnostic: check for finalize issues
                {
                    int null_sg = 0, infeasible_sg = 0;
                    for (size_t gi = 0; gi < part.groups.size(); gi++) {
                        if (!part.groups[gi].alive) continue;
                        if (!part.groups[gi].sg) null_sg++;
                        else if (part.groups[gi].cost >= 1e17) infeasible_sg++;
                    }
                    if (null_sg > 0 || infeasible_sg > 0) {
                        std::lock_guard<std::mutex> lock(sol_mutex);
                        std::cerr << "    Partition " << tid
                                  << ": finalize issues: " << null_sg << " null sg, "
                                  << infeasible_sg << " infeasible\n";
                    }
                }

                // Post-finalize: verify group DAG has correct edges.
                // Count groups with in_deg=0 that have boundary inputs from
                // other groups — these would cause ordering failures.
                {
                    int bad_roots = 0;
                    for (size_t gi = 0; gi < part.groups.size(); gi++) {
                        if (!part.groups[gi].alive || !part.groups[gi].sg) continue;
                        if (part.group_in_deg[gi] != 0) continue;
                        for (auto t : part.groups[gi].sg->boundary_inputs()) {
                            if (dag.tensor_producer[t] < 0) continue;
                            // T needs to come from somewhere — is it produced by any other step?
                            bool has_source = false;
                            for (size_t gj = 0; gj < part.groups.size(); gj++) {
                                if (gj == gi || !part.groups[gj].alive || !part.groups[gj].sg) continue;
                                if (part.groups[gj].sg->boundary_outputs().count(t)) {
                                    has_source = true; break;
                                }
                            }
                            if (has_source) { bad_roots++; break; }
                        }
                    }
                    if (bad_roots > 0) {
                        std::lock_guard<std::mutex> lock(sol_mutex);
                        std::cerr << "    Partition " << tid
                                  << " WARNING: " << bad_roots
                                  << " root group(s) with unsatisfied boundary inputs"
                                  << " (group DAG edges missing)\n";
                    }
                }

                // --- DFS + Beam via Solution::from_partition ---
                // Partition is already finalized above; from_partition takes
                // a const reference (no deep copy, no redundant finalize).
                {
                    auto sol = Solution::from_partition(prob, dag, part);
                    auto vr = sol.validate();
                    std::lock_guard<std::mutex> lock(sol_mutex);
                    if (vr.valid) {
                        solution_pool.push_back(std::move(sol));
                    } else {
                        std::cerr << "    Partition " << tid
                                  << " (cost=" << part.total_cost()
                                  << ", " << part.num_alive() << " groups)"
                                  << " → solution REJECTED: " << vr.error
                                  << " (latency=" << sol.total_latency() << ")\n";
                    }
                }

                if (!has_retain) continue;
                if (SteadyClock::now() >= phase2_dl) continue;

                // --- Random-retain variant for pool diversity ---
                // Generates a randomised ordering biased toward pre-chosen
                // retention pairs; steps_from_ordering handles feasibility.
                std::mt19937 rng(42 + tid * 1337);
                auto rand_res  = random_ordering(part, feasibly_ret, rng);
                auto rand_steps = Solution::steps_from_ordering(prob, dag, part, rand_res);

                Solution rand_sol(prob, dag, std::move(rand_steps));
                {
                    auto rvr = rand_sol.validate();
                    std::lock_guard<std::mutex> lock(sol_mutex);
                    if (rvr.valid)
                        solution_pool.push_back(std::move(rand_sol));
                }
            }
        };

        int nt = std::min(hw_threads, n_tasks);
        std::vector<std::thread> threads;
        threads.reserve(nt);
        for (int i = 0; i < nt; i++) threads.emplace_back(build_worker);
        for (auto& t : threads) t.join();
    }

    std::sort(solution_pool.begin(), solution_pool.end(),
              [](const Solution& a, const Solution& b) {
                  return a.total_latency() < b.total_latency();
              });

    if (solution_pool.empty()) {
        partition_pool[0].finalize();  // may not have been finalized if all workers timed out
        solution_pool.push_back(Solution::from_partition(prob, dag, partition_pool[0]));
    }

    double after_build = solution_pool[0].total_latency();
    std::cerr << "  Solution pool: " << solution_pool.size()
              << " solutions, best=" << after_build << "\n";

    // ================================================================
    // Phase 3: Solution-level evolutionary search + FM polish
    // ================================================================
    Solution final_sol(prob, dag, {});
    double   after_sol_evo;

    if (has_retain) {
        std::cerr << "Phase 3: Solution evolution + FM from "
                  << solution_pool.size() << " starting points...\n";
        SolutionFMConfig sfm_cfg;
        sfm_cfg.deadline = phase3_dl;
        final_sol      = solution_evo_search(prob, dag, std::move(solution_pool), sfm_cfg);
        after_sol_evo  = final_sol.total_latency();
    } else {
        std::cerr << "Phase 3: Skipped (no retainable tensors)\n";
        final_sol     = std::move(solution_pool[0]);
        after_sol_evo = final_sol.total_latency();
    }

    auto vr = final_sol.validate();
    if (!vr.valid)
        std::cerr << "  WARNING: " << vr.error << "\n";

    // Summary
    double partition_cost = partition_pool[0].total_cost();
    double final_cost     = final_sol.total_latency();
    std::cerr << "  === Summary ===\n"
              << "  Partition:  " << partition_cost << "\n"
              << "  Build:      " << after_build;
    if (after_build < partition_cost - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << 100.0 * (partition_cost - after_build) / partition_cost << "%)";
    std::cerr << "\n  Sol-Evo:    " << after_sol_evo;
    if (after_sol_evo < after_build - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << 100.0 * (after_build - after_sol_evo) / after_build << "%)";
    std::cerr << "\n  Final:      " << final_cost;
    if (final_cost < partition_cost - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << 100.0 * (partition_cost - final_cost) / partition_cost << "% total)";
    std::cerr << "\n";

    return final_sol;
}