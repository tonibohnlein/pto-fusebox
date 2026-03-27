#include "pipeline/solver.h"
#include "partition/partition.h"
#include "core/cost_cache.h"
#include "search/parallel_search.h"
#include "search/local_search.h"   // partition_has_gap
#include "search/coupling_parallel_search.h"
#include "solution/solution.h"     // compute_feasibly_retainable
#include "solution/ordering.h"
#include "symmetry/symmetry.h"
#include "symmetry/series.h"
#include "symmetry/merkle_hash.h"
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
    if (deadline == TimePoint::max())
        effective_dl = now + std::chrono::seconds(5);
    auto total_budget = effective_dl - now;

    auto feasibly_ret = compute_feasibly_retainable(prob, dag);
    bool has_retain   = !feasibly_ret.empty();

    std::cerr << "  Retainable tensors: " << feasibly_ret.size()
              << " / " << prob.retainable_tensors.size() << "\n";

    TimePoint phase1_dl, phase2_dl;
    if (has_retain) {
        phase1_dl = now + std::chrono::duration_cast<SteadyClock::duration>(total_budget * 35 / 100);
        phase2_dl = phase1_dl + std::chrono::duration_cast<SteadyClock::duration>(total_budget * 5 / 100);
    } else {
        phase1_dl = now + std::chrono::duration_cast<SteadyClock::duration>(total_budget * 95 / 100);
        phase2_dl = now + total_budget;
    }

    // ================================================================
    // Phase 1: Partition pool via parallel search
    // ================================================================
    std::cerr << "Phase 1: Parallel search...\n";

    // Shared cost cache: lives for the entire solver pipeline.
    // Phase 1 populates the base map (ops → CostResult).
    // Phase 2/3 reuse base map entries and add retention variants.
    CostCache shared_cache;

    ParallelConfig pcfg;
    pcfg.fm.deadline = phase1_dl;
    pcfg.cache = &shared_cache;
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

    // Null partition cache pointers — partitions don't need the cache after
    // Phase 1 (finalize() uses Subgraph::create directly). The shared_cache
    // itself stays alive for Phase 2/3.
    for (auto& p : partition_pool)
        p.cache = nullptr;

    // ================================================================
    // Phase 2: Build solution pool (no-retain) OR finalize for coupling
    // ================================================================
    auto phase2_start = SteadyClock::now();

    std::vector<Solution> solution_pool;
    std::mutex            sol_mutex;
    double after_build    = partition_pool[0].total_cost();
    double after_sol_evo  = after_build;
    Solution final_sol(prob, dag, {});

    if (!has_retain) {
        // No retainable tensors: ordering has no effect on cost (all groups
        // have empty retain_these). Just finalize the best partition and emit
        // a solution from a single topological ordering.
        std::cerr << "Phase 2: Build solution (no-retain)...\n";

        partition_pool[0].rebuild_index();
        partition_pool[0].finalize(&shared_cache);
        auto dfs_res  = dfs_ordering(partition_pool[0]);
        auto steps    = Solution::steps_from_ordering(
            prob, dag, partition_pool[0], dfs_res, &shared_cache);
        final_sol = Solution(prob, dag, std::move(steps));
        if (!final_sol.validate().valid)
            final_sol = Solution::from_partition(prob, dag, partition_pool[0]);

        after_build = final_sol.total_latency();
        auto phase2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            SteadyClock::now() - phase2_start).count();
        std::cerr << "  Solution pool: 1 solutions, best=" << after_build
                  << " (Phase 2: " << phase2_ms << "ms)\n";
        std::cerr << "Phase 3: Skipped (no retainable tensors)\n";
    } else {
        // Phase 2 (has_retain): build coupled pool from partition pool.
        // For each partition: init_from (finalizes internally) → seed coupling
        // edges from DFS/beam ordering → coupling_greedy_descent to refine.
        std::cerr << "Phase 2: Building coupled partitions from "
                  << partition_pool.size() << " partitions...\n";

        // Helper: read retain_per_step from an ordering and set the coupling
        // edges directly on cp (no acyclicity check needed — DFS/beam
        // orderings respect topological order by construction).
        auto seed_coupling_from_ordering = [&](CoupledPartition& cp,
                                                const OrderingResult& res) {
            const auto& part = cp.part;
            while (cp.next_group.size() < part.groups.size())
                cp.next_group.push_back(SIZE_MAX);
            while (cp.prev_group.size() < part.groups.size())
                cp.prev_group.push_back(SIZE_MAX);

            for (size_t i = 0; i + 1 < res.order.size(); i++) {
                if (i >= res.retain_per_step.size()) break;
                const auto& ret = res.retain_per_step[i];
                if (ret.empty()) continue;
                size_t ga = res.order[i], gb = res.order[i + 1];
                if (!part.groups[ga].sg || !part.groups[gb].sg) continue;
                const auto& bouts = part.groups[ga].sg->boundary_outputs();
                const auto& bins  = part.groups[gb].sg->boundary_inputs();
                for (auto t : ret) {
                    if (!feasibly_ret.count(t)) continue;
                    if (!bouts.count(t) || !bins.count(t)) continue;
                    // Only couple if ga is still a free tail and gb a free head
                    if (cp.next_group[ga] != SIZE_MAX && cp.next_group[ga] != gb) continue;
                    if (cp.prev_group[gb] != SIZE_MAX && cp.prev_group[gb] != ga) continue;
                    cp.next_group[ga] = gb;
                    cp.prev_group[gb] = ga;
                    cp.retained[{ga, gb}].insert(t);
                }
            }
        };

        int hw_threads = (int)std::thread::hardware_concurrency();
        if (hw_threads <= 0) hw_threads = 4;
        int n_tasks = (int)partition_pool.size();
        std::atomic<int> next_task{0};
        std::mutex cp_mutex;
        std::vector<CoupledPartition> coupled_pool;

        // Helper: seed coupling from ordering, push pre-greedy copy, run
        // greedy descent, push post-greedy result. Both go into the pool for
        // diversity.
        auto add_variant = [&](CoupledPartition cp, const OrderingResult& res) {
            seed_coupling_from_ordering(cp, res);
            { std::lock_guard<std::mutex> lk(cp_mutex); coupled_pool.push_back(cp); }
            coupling_greedy_descent(cp, feasibly_ret, phase2_dl);
            { std::lock_guard<std::mutex> lk(cp_mutex); coupled_pool.push_back(std::move(cp)); }
        };

        auto gen0_worker = [&]() {
            std::mt19937 rng(std::random_device{}());
            while (true) {
                int tid = next_task.fetch_add(1);
                if (tid >= n_tasks) break;
                if (SteadyClock::now() >= phase2_dl) break;

                Partition part = partition_pool[tid];
                part.rebuild_index();
                if (partition_has_gap(part)) continue;

                // Finalize once: builds .sg for all groups (needed for coupling
                // move evaluation). All variants share the same finalized partition.
                CoupledPartition cp_base;
                cp_base.init_from(std::move(part), &shared_cache);
                // finalize() builds .sg for ordering functions (dfs/beam/random
                // ordering use sg->boundary_outputs/inputs for retain scoring).
                cp_base.part.finalize(&shared_cache);
                int n_alive = (int)cp_base.part.num_alive();

                // DFS-seeded variant
                if (SteadyClock::now() < phase2_dl)
                    add_variant(cp_base, dfs_ordering(cp_base.part));

                // Top 2 partitions: beam-search-seeded variant
                if (tid < 2 && SteadyClock::now() < phase2_dl) {
                    int bw = (n_alive > 25) ? 3 : (n_alive > 15) ? 5 : 8;
                    add_variant(cp_base, beam_search_ordering(cp_base.part, bw));
                }

                // Random-ordering variant for diversity
                if (SteadyClock::now() < phase2_dl)
                    add_variant(cp_base, random_ordering(cp_base.part, feasibly_ret, rng));
            }
        };

        int nt = std::min(hw_threads, n_tasks);
        std::vector<std::thread> threads;
        threads.reserve(nt);
        for (int i = 0; i < nt; i++) threads.emplace_back(gen0_worker);
        for (auto& t : threads) t.join();

        if (coupled_pool.empty()) {
            Partition part = partition_pool[0];
            part.rebuild_index();
            CoupledPartition cp;
            cp.init_from(std::move(part), &shared_cache);
            cp.part.finalize(&shared_cache);
            auto res = dfs_ordering(cp.part);
            seed_coupling_from_ordering(cp, res);
            coupling_greedy_descent(cp, feasibly_ret);
            coupled_pool.push_back(std::move(cp));
        }

        double best_coupled = std::min_element(coupled_pool.begin(), coupled_pool.end(),
            [](const CoupledPartition& a, const CoupledPartition& b) {
                return a.total_cost() < b.total_cost();
            })->total_cost();
        auto phase2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            SteadyClock::now() - phase2_start).count();
        std::cerr << "  Coupled pool: " << coupled_pool.size()
                  << " entries, best=" << best_coupled
                  << " (Phase 2: " << phase2_ms << "ms)\n";
        after_build = best_coupled;

        // ================================================================
        // Phase 3: Coupling evo on the pool built in Phase 2
        // ================================================================
        // Sort best-first so Phase 3's DiversityPool claims the cheapest
        // entries in its initial (non-eviction) slots.
        std::sort(coupled_pool.begin(), coupled_pool.end(),
            [](const CoupledPartition& a, const CoupledPartition& b) {
                return a.total_cost() < b.total_cost();
            });

        std::cerr << "Phase 3: Coupling evo search...\n";
        CouplingParallelConfig ccfg;
        ccfg.pool_size   = std::min((int)coupled_pool.size(), 16);
        ccfg.fm.deadline = effective_dl;
        ccfg.cache       = &shared_cache;
        ccfg.early_stop  = false;  // deadline governs termination; don't bail early
        auto coupling_sol = coupling_parallel_search(std::move(coupled_pool),
                                                     feasibly_ret,
                                                     effective_dl,
                                                     ccfg);
        auto vr2 = coupling_sol.validate();
        if (!vr2.valid)
            std::cerr << "  WARNING: coupling solution invalid: " << vr2.error << "\n";
        final_sol = std::move(coupling_sol);
    }
    after_sol_evo = final_sol.total_latency();

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
    std::cerr << "\n  Cache: base=" << shared_cache.size()
              << " (" << shared_cache.base_hits() << "h/" << shared_cache.base_misses() << "m)"
              << " ret=" << shared_cache.ret_size()
              << " (" << shared_cache.ret_hits() << "h/" << shared_cache.ret_misses() << "m)";
    std::cerr << "\n";

    return final_sol;
}