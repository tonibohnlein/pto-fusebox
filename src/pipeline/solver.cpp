#include "pipeline/solver.h"
#include "partition/partition.h"
#include "core/cost_cache.h"
#include "search/parallel_search.h"
#include "search/evolution.h"
#include "search/local_search.h"   // partition_has_gap, greedy_descent
#include "search/coupling_parallel_search.h"
#include "search/symm_mutations.h"
#include "solution/solution.h"     // compute_feasibly_retainable
#include "solution/ordering.h"
#include "init/init_strategies.h"
#include "init/symm_init.h"
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
// Shared helpers (used by both solve() and solve_v2())
// ============================================================================

namespace {

// Seed coupling edges from an ordering result: for each consecutive pair
// (ga, gb) in the ordering, if ga produces retained tensors consumed by gb,
// create a coupling edge ga→gb with those tensors.
void seed_coupling_from_ordering(CoupledPartition& cp,
                                 const OrderingResult& res,
                                 const FlatSet<size_t>& feasibly_ret) {
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
        const auto& bins  = part.groups[gb].sg->boundary_inputs();
        for (auto t : ret) {
            if (!feasibly_ret.count(t)) continue;
            // T must be produced in ga (boundary output or ephemeral) and
            // consumed in gb (boundary input).
            bool produced_in_ga = (part.groups[ga].sg->boundary_outputs().count(t) ||
                                   part.groups[ga].sg->ephemeral().count(t));
            if (!produced_in_ga || !bins.count(t)) continue;
            // Only couple if ga is still a free tail and gb a free head
            if (cp.next_group[ga] != SIZE_MAX && cp.next_group[ga] != gb) continue;
            if (cp.prev_group[gb] != SIZE_MAX && cp.prev_group[gb] != ga) continue;
            cp.next_group[ga] = gb;
            cp.prev_group[gb] = ga;
            cp.retained[{ga, gb}].insert(t);
        }
    }
}

// Seed coupling from an ordering and push into the pool.
// No greedy descent here — Phase 3's coupled evolution handles refinement.
void add_coupled_variant(CoupledPartition cp,
                         const OrderingResult& res,
                         const FlatSet<size_t>& feasibly_ret,
                         std::mutex& cp_mutex,
                         std::vector<CoupledPartition>& coupled_pool,
                         TimePoint /*deadline*/) {
    seed_coupling_from_ordering(cp, res, feasibly_ret);
    { std::lock_guard<std::mutex> lk(cp_mutex); coupled_pool.push_back(std::move(cp)); }
}

// Pick the best valid solution between a coupled and uncoupled candidate.
// Prefers coupled if both valid and coupled is cheaper (with 0.01 tolerance).
// Falls back to coupled if both invalid (it at least has retention).
Solution pick_best_valid(Solution coupled, Solution uncoupled) {
    auto vr_c = coupled.validate();
    auto vr_u = uncoupled.validate();

    if (vr_c.valid && vr_u.valid)
        return (coupled.total_latency() < uncoupled.total_latency() - 0.01)
              ? std::move(coupled) : std::move(uncoupled);
    if (vr_c.valid)
        return std::move(coupled);
    if (vr_u.valid)
        return std::move(uncoupled);
    return std::move(coupled);
}

} // anonymous namespace

// ============================================================================
// Full pipeline
// ============================================================================

Solution solve(const Problem& prob, const DAG& dag, TimePoint deadline) {

    auto now              = SteadyClock::now();
    auto effective_dl     = deadline;
    if (deadline == TimePoint::max())
        effective_dl = now + std::chrono::seconds(5);
    auto total_budget = effective_dl - now;

    // 910B never retains tensors across kernels: cross-subgraph data routes
    // through DDR (no shared L2), so the retain/coupling pipeline (Phase 3) is
    // permanently disabled — the no-retention path always runs.
    auto feasibly_ret = compute_feasibly_retainable(prob, dag);
    const bool has_retain = false;

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

        int hw_threads = (int)std::thread::hardware_concurrency();
        if (hw_threads <= 0) hw_threads = 4;
        int n_tasks = (int)partition_pool.size();
        std::atomic<int> next_task{0};
        std::mutex cp_mutex;
        std::vector<CoupledPartition> coupled_pool;

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

                // Bare variant (no coupling) — ensures the uncoupled partition
                // is always in the pool as a fallback.
                { std::lock_guard<std::mutex> lk(cp_mutex); coupled_pool.push_back(cp_base); }

                // DFS-seeded variant
                if (SteadyClock::now() < phase2_dl)
                    add_coupled_variant(cp_base, dfs_ordering(cp_base.part),
                                        feasibly_ret, cp_mutex, coupled_pool, phase2_dl);

                // Top 2 partitions: beam-search-seeded variant
                if (tid < 2 && SteadyClock::now() < phase2_dl) {
                    int bw = (n_alive > 25) ? 3 : (n_alive > 15) ? 5 : 8;
                    add_coupled_variant(cp_base, beam_search_ordering(cp_base.part, bw),
                                        feasibly_ret, cp_mutex, coupled_pool, phase2_dl);
                }

                // Random-ordering variant for diversity
                if (SteadyClock::now() < phase2_dl)
                    add_coupled_variant(cp_base, random_ordering(cp_base.part, feasibly_ret, rng),
                                        feasibly_ret, cp_mutex, coupled_pool, phase2_dl);
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
            seed_coupling_from_ordering(cp, res, feasibly_ret);
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

        // Build symmetry context for Phase 3 crossover + symmetry mutations
        std::optional<MerkleHashes> merkle;
        std::optional<symm_mutations::PatternContext> symm_ctx;
        if (!parallel_patterns.empty() || !series_patterns.empty()) {
            merkle = MerkleHashes::compute(prob, dag);
            symm_ctx = symm_mutations::build_context(
                prob, dag, parallel_patterns, series_patterns, *merkle);
        }

        CouplingParallelConfig ccfg;
        ccfg.pool_size   = std::min((int)coupled_pool.size(), 16);
        ccfg.fm.deadline = effective_dl;
        ccfg.cache       = &shared_cache;
        ccfg.early_stop  = false;  // deadline governs termination; don't bail early
        ccfg.merkle      = merkle  ? &*merkle  : nullptr;
        ccfg.symm_ctx    = symm_ctx ? &*symm_ctx : nullptr;
        auto coupling_sol = coupling_parallel_search(std::move(coupled_pool),
                                                     feasibly_ret,
                                                     effective_dl,
                                                     ccfg);
        auto vr2 = coupling_sol.validate();
        if (!vr2.valid)
            std::cerr << "  WARNING: coupling solution invalid: " << vr2.error << "\n";
        // Build uncoupled fallback from best partition (no retention).
        Solution uncoupled_fb(prob, dag, {});
        {
            Partition fb = partition_pool[0];
            fb.rebuild_index();
            fb.finalize(&shared_cache);
            uncoupled_fb = Solution::from_partition(prob, dag, fb);
        }

        final_sol = pick_best_valid(std::move(coupling_sol), std::move(uncoupled_fb));
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

// ============================================================================
// V2 pipeline: init-only seeding → evo loop (partition or coupled)
//
// Differences from solve():
//   - Phase 1 runs ONLY init heuristics (no greedy descent, no FM)
//   - No Phase 2 (no ordering+greedy coupling seeding)
//   - The evo loop handles everything: mutation → greedy → FM
//   - Fork on has_retain: partition evo loop vs coupled evo loop
// ============================================================================

Solution solve_v2(const Problem& prob, const DAG& dag, TimePoint deadline) {
    auto now = SteadyClock::now();
    auto effective_dl = deadline;
    if (deadline == TimePoint::max())
        effective_dl = now + std::chrono::seconds(5);

    // 910B never retains tensors across kernels (cross-subgraph data routes
    // through DDR) — the retain/coupling pipeline is permanently disabled.
    auto feasibly_ret = compute_feasibly_retainable(prob, dag);
    const bool has_retain = false;

    CostCache shared_cache;

    // ================================================================
    // Seed pool: run init heuristics only (no greedy, no FM)
    // ================================================================
    std::cerr << "Seeding pool (init heuristics only)...\n";

    // Run all init strategies to produce raw partitions.
    // Each strategy: init → greedy descent (cheap) → insert into pool.
    // NO FM refinement.
    auto init_start = SteadyClock::now();

    // Use all_init_strategies() to get the registered strategies.
    auto strategies = all_init_strategies();

    std::vector<Partition> seed_pool;
    std::mutex pool_mutex;

    // Also run symm_init
    auto symm_parts = init_from_patterns(prob, dag, &shared_cache);
    for (auto& sp : symm_parts) {
        sp.rebuild_index();
        if (!partition_has_gap(sp))
            seed_pool.push_back(std::move(sp));
    }

    int hw_threads = (int)std::thread::hardware_concurrency();
    if (hw_threads <= 0) hw_threads = 4;

    // Run each init strategy + greedy descent (no FM).
    // Add 3 extra random inits for diversity.
    int n_strats = (int)strategies.size();
    int n_random_extra = 3;
    int n_total = n_strats + n_random_extra;

    {
        std::atomic<int> next{0};

        auto worker = [&]() {
            while (true) {
                int idx = next.fetch_add(1);
                if (idx >= n_total) break;

                Partition part = (idx < n_strats)
                    ? strategies[idx].init(prob, dag, &shared_cache)
                    : init_random(prob, dag, &shared_cache);

                std::string name = (idx < n_strats) ? strategies[idx].name : "random";

                part.rebuild_index();

                // Add pre-greedy partition for diversity (if valid)
                if (!partition_has_gap(part)) {
                    Partition pre_greedy = part;
                    pre_greedy.rebuild_index();
                    std::lock_guard<std::mutex> lk(pool_mutex);
                    seed_pool.push_back(std::move(pre_greedy));
                }

                // Quick greedy descent (no FM)
                part = greedy_descent(std::move(part));
                part.rebuild_index();

                if (partition_has_gap(part)) return;

                double cost = part.total_cost();
                std::lock_guard<std::mutex> lk(pool_mutex);
                seed_pool.push_back(std::move(part));

                std::cerr << "    init [" << name << "]: cost=" << cost << "\n";
            }
        };

        int nt = std::min(hw_threads, n_total);
        std::vector<std::thread> threads;
        for (int i = 0; i < nt; i++) threads.emplace_back(worker);
        for (auto& t : threads) t.join();
    }

    if (seed_pool.empty()) {
        // Fallback: trivial partition
        auto part = Partition::trivial(prob, dag);
        part.cache = &shared_cache;
        part.rebuild_index();
        seed_pool.push_back(std::move(part));
    }

    // Sort by cost
    std::sort(seed_pool.begin(), seed_pool.end(),
        [](const Partition& a, const Partition& b) {
            return a.total_cost() < b.total_cost();
        });

    auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        SteadyClock::now() - init_start).count();
    std::cerr << "  Seed pool: " << seed_pool.size()
              << " entries, best=" << seed_pool[0].total_cost()
              << " (" << init_ms << "ms)\n";

    double seed_best = seed_pool[0].total_cost();

    // ================================================================
    // Evo loop: fork on has_retain
    // ================================================================

    Solution final_sol(prob, dag, {});

    if (!has_retain) {
        // No retention: run partition evo loop directly.
        // Use parallel_search with the seed pool already built.
        std::cerr << "Partition evo loop (no retention)...\n";

        ParallelConfig pcfg;
        pcfg.fm.deadline = effective_dl;
        pcfg.cache = &shared_cache;
        pcfg.early_stop = false;

        auto result_pool = parallel_search(prob, dag, pcfg);

        // Build solution from best partition
        result_pool[0].rebuild_index();
        result_pool[0].finalize(&shared_cache);
        final_sol = Solution::from_partition(prob, dag, result_pool[0], 8, &shared_cache);

    } else {
        // Has retention: build coupled pool from seeds via Phase 2 logic,
        // then run coupled evo.
        std::cerr << "Building coupled pool from seeds...\n";
        auto phase2_start = SteadyClock::now();
        // Use ~5% of remaining budget for Phase 2 seeding
        auto phase2_dl = SteadyClock::now() + std::chrono::duration_cast<SteadyClock::duration>(
            (effective_dl - SteadyClock::now()) * 5 / 100);

        std::vector<CoupledPartition> coupled_pool;
        std::mutex cp_mutex;

        {
            std::atomic<int> next_task{0};
            int n_tasks = (int)seed_pool.size();

            auto gen0_worker = [&]() {
                std::mt19937 rng(std::random_device{}());
                while (true) {
                    int tid = next_task.fetch_add(1);
                    if (tid >= n_tasks) break;
                    if (SteadyClock::now() >= phase2_dl) break;

                    Partition part = seed_pool[tid];
                    part.rebuild_index();
                    if (partition_has_gap(part)) continue;

                    CoupledPartition cp_base;
                    cp_base.init_from(std::move(part), &shared_cache);
                    cp_base.part.finalize(&shared_cache);

                    // Bare variant (no coupling)
                    { std::lock_guard<std::mutex> lk(cp_mutex); coupled_pool.push_back(cp_base); }

                    // DFS-seeded variant
                    if (SteadyClock::now() < phase2_dl)
                        add_coupled_variant(cp_base, dfs_ordering(cp_base.part),
                                            feasibly_ret, cp_mutex, coupled_pool, phase2_dl);

                    // Random coupling variant: pick random retainable tensors
                    // and couple their producer/consumer groups directly.
                    if (SteadyClock::now() < phase2_dl) {
                        CoupledPartition cp_rand = cp_base;
                        std::vector<size_t> ret_vec(feasibly_ret.begin(), feasibly_ret.end());
                        std::shuffle(ret_vec.begin(), ret_vec.end(), rng);
                        const DAG& dag_ref = *cp_rand.part.dag;
                        for (auto t : ret_vec) {
                            int prod = dag_ref.tensor_producer[t];
                            if (prod < 0) continue;
                            size_t ga = SIZE_MAX;
                            for (auto g : cp_rand.part.groups_of((size_t)prod))
                                if (cp_rand.part.groups[g].alive) { ga = g; break; }
                            if (ga == SIZE_MAX) continue;
                            if (ga >= cp_rand.next_group.size() || cp_rand.next_group[ga] != SIZE_MAX) continue;
                            bool produced = cp_rand.part.groups[ga].ops.count((size_t)prod);
                            if (!produced) continue;
                            for (auto cop : dag_ref.tensor_consumers[t]) {
                                for (auto gb : cp_rand.part.groups_of(cop)) {
                                    if (gb == ga || !cp_rand.part.groups[gb].alive) continue;
                                    if (gb >= cp_rand.prev_group.size() || cp_rand.prev_group[gb] != SIZE_MAX) continue;
                                    if (!is_boundary_input_of(cp_rand.part.groups[gb].ops, t, dag_ref)) continue;
                                    auto ev = eval_couple(cp_rand, ga, gb, t);
                                    if (ev.feasible) {
                                        apply_couple(cp_rand, ga, gb, t);
                                        break;
                                    }
                                }
                                if (ga < cp_rand.next_group.size() && cp_rand.next_group[ga] != SIZE_MAX) break;
                            }
                        }
                        { std::lock_guard<std::mutex> lk(cp_mutex); coupled_pool.push_back(std::move(cp_rand)); }
                    }
                }
            };

            int nt = std::min(hw_threads, n_tasks);
            std::vector<std::thread> threads;
            for (int i = 0; i < nt; i++) threads.emplace_back(gen0_worker);
            for (auto& t : threads) t.join();
        }

        if (coupled_pool.empty()) {
            Partition part = seed_pool[0];
            part.rebuild_index();
            CoupledPartition cp;
            cp.init_from(std::move(part), &shared_cache);
            cp.part.finalize(&shared_cache);
            coupled_pool.push_back(std::move(cp));
        }

        auto phase2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            SteadyClock::now() - phase2_start).count();
        double best_coupled = std::min_element(coupled_pool.begin(), coupled_pool.end(),
            [](const CoupledPartition& a, const CoupledPartition& b) {
                return a.total_cost() < b.total_cost();
            })->total_cost();
        std::cerr << "  Coupled pool: " << coupled_pool.size()
                  << " entries, best=" << best_coupled
                  << " (" << phase2_ms << "ms)\n";

        // Sort best-first
        std::sort(coupled_pool.begin(), coupled_pool.end(),
            [](const CoupledPartition& a, const CoupledPartition& b) {
                return a.total_cost() < b.total_cost();
            });

        // Build symmetry context for crossover + symmetry mutations
        auto v2_merkle = MerkleHashes::compute(prob, dag);
        auto v2_parallel = SymmetryDetector::discover(prob, dag, v2_merkle);
        auto v2_series = SeriesDetector::discover(prob, dag, v2_merkle);
        std::optional<symm_mutations::PatternContext> v2_symm;
        if (!v2_parallel.empty() || !v2_series.empty())
            v2_symm = symm_mutations::build_context(
                prob, dag, v2_parallel, v2_series, v2_merkle);

        CouplingParallelConfig ccfg;
        ccfg.pool_size = std::min((int)coupled_pool.size(), 16);
        ccfg.fm.deadline = effective_dl;
        ccfg.cache = &shared_cache;
        ccfg.early_stop = false;
        ccfg.merkle   = &v2_merkle;
        ccfg.symm_ctx = v2_symm ? &*v2_symm : nullptr;

        auto coupling_sol = coupling_parallel_search(
            std::move(coupled_pool), feasibly_ret, effective_dl, ccfg);

        // Build uncoupled fallback
        Solution uncoupled_fb(prob, dag, {});
        {
            Partition fb = seed_pool[0];
            fb.rebuild_index();
            fb.finalize(&shared_cache);
            uncoupled_fb = Solution::from_partition(prob, dag, fb, 8, &shared_cache);
        }

        final_sol = pick_best_valid(std::move(coupling_sol), std::move(uncoupled_fb));
    }

    double final_cost = final_sol.total_latency();
    auto vr = final_sol.validate();
    if (!vr.valid)
        std::cerr << "  WARNING: " << vr.error << "\n";

    std::cerr << "  === Summary ===\n"
              << "  Seed:   " << seed_best << "\n"
              << "  Final:  " << final_cost;
    if (final_cost < seed_best - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << 100.0 * (seed_best - final_cost) / seed_best << "%)";
    std::cerr << "\n  Cache: base=" << shared_cache.size()
              << " (" << shared_cache.base_hits() << "h/" << shared_cache.base_misses() << "m)"
              << " ret=" << shared_cache.ret_size()
              << " (" << shared_cache.ret_hits() << "h/" << shared_cache.ret_misses() << "m)";
    std::cerr << "\n";

    return final_sol;
}