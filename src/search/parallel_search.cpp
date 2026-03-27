#include "search/parallel_search.h"
#include "search/verbose.h"
#include "search/pool.h"
#include "search/symm_mutations.h"
#include "init/init_strategies.h"
#include "init/symm_init.h"
#include "search/local_search.h"
#include "search/fm_outer.h"
#include "search/evolution.h"
#include "core/cost_cache.h"
#include "symmetry/merkle_hash.h"
#include "symmetry/symmetry.h"
#include "symmetry/series.h"
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <atomic>

using Clock = std::chrono::steady_clock;

// ============================================================================
// Pool entry
// ============================================================================

struct PoolEntry {
    Partition partition;
    double cost = 1e18;
    std::string origin;

    PoolEntry() = default;
    PoolEntry(Partition p, double c, std::string o)
        : partition(std::move(p)), cost(c), origin(std::move(o)) {}
};

// ============================================================================
// Partition distance
// ============================================================================

static double partition_distance(const Partition& a, const Partition& b,
                                  const MerkleHashes* mh = nullptr) {
    size_t n = a.prob->num_ops();

    // Build op → group maps
    std::vector<int> ga_map(n, -1), gb_map(n, -1);
    int num_ga = 0, num_gb = 0;
    for (size_t gi = 0; gi < a.groups.size(); gi++)
        if (a.groups[gi].alive) {
            for (auto op : a.groups[gi].ops) ga_map[op] = num_ga;
            num_ga++;
        }
    for (size_t gi = 0; gi < b.groups.size(); gi++)
        if (b.groups[gi].alive) {
            for (auto op : b.groups[gi].ops) gb_map[op] = num_gb;
            num_gb++;
        }
    if (num_ga == 0 || num_gb == 0) return 0.0;

    // Apply Merkle canonicalisation: symmetric variants → distance 0
    if (mh) merkle_canonicalise(*mh, ga_map, gb_map);

    // Build contingency table: n_ij = |group_i_in_a ∩ group_j_in_b|
    // Flat 1D vector: single allocation instead of num_ga+1 heap allocations.
    std::vector<int> table(num_ga * num_gb, 0);
    std::vector<int> row_sum(num_ga, 0), col_sum(num_gb, 0);
    int total = 0;

    for (size_t op = 0; op < n; op++) {
        if (ga_map[op] < 0 || gb_map[op] < 0) continue;
        table[ga_map[op] * num_gb + gb_map[op]]++;
        row_sum[ga_map[op]]++;
        col_sum[gb_map[op]]++;
        total++;
    }
    if (total <= 1) return 0.0;

    // Adjusted Rand Index (ARI) via contingency table.
    // ARI corrects for chance: random partitions → ARI ≈ 0, identical → ARI = 1.
    // We return 1 - ARI as a distance (0 = identical, 1 = maximally different).
    auto choose2 = [](int64_t x) -> int64_t { return x * (x - 1) / 2; };

    int64_t same_a = 0, same_b = 0, agree = 0;
    for (int i = 0; i < num_ga; i++) same_a += choose2(row_sum[i]);
    for (int j = 0; j < num_gb; j++) same_b += choose2(col_sum[j]);
    for (int i = 0; i < num_ga; i++)
        for (int j = 0; j < num_gb; j++)
            agree += choose2(table[i * num_gb + j]);

    int64_t total_pairs = choose2(total);
    if (total_pairs == 0) return 0.0;

    // ARI = (agree - expected) / (max_possible - expected)
    // expected = same_a * same_b / total_pairs
    // max_possible = (same_a + same_b) / 2
    double expected = (double)same_a * same_b / total_pairs;
    double max_agree = (double)(same_a + same_b) / 2.0;
    double denom = max_agree - expected;
    if (std::abs(denom) < 1e-12) return 0.0;  // both partitions are identical

    double ari = ((double)agree - expected) / denom;
    ari = std::clamp(ari, 0.0, 1.0);
    return 1.0 - ari;  // distance: 0 = identical, 1 = maximally different
}

// ============================================================================
// Pool management
// ============================================================================

// Pool management is handled by DiversityPool<PoolEntry> (see pool.h).
// partition_distance() is passed as the distance function at construction.

// ============================================================================
// Parallel generational search
// ============================================================================

std::vector<Partition> parallel_search(const Problem& prob, const DAG& dag,
                          const ParallelConfig& cfg) {
    auto strategies = all_init_strategies();
    int num_strategies = (int)strategies.size();

    int hw_threads = (int)std::thread::hardware_concurrency();
    if (hw_threads <= 0) hw_threads = 4;
    int num_threads = cfg.num_threads > 0 ? cfg.num_threads : hw_threads;

    CostCache local_cache;
    CostCache& shared_cache = cfg.cache ? *cfg.cache : local_cache;
    std::mutex log_mutex;

    // Compute Merkle hashes once — used for symmetry-aware pool deduplication
    MerkleHashes mh = MerkleHashes::compute(prob, dag);
    const MerkleHashes* mhp = mh.symmetric_ops() > 0 ? &mh : nullptr;
    if (mhp)
        std::cerr << "  Merkle: " << mh.num_classes()
                  << " structural classes, " << mh.symmetric_ops()
                  << " symmetric ops\n";

    auto deadline = cfg.fm.deadline;
    bool has_deadline = (deadline != TimePoint::max());

    // ================================================================
    // Task list: init strategies for gen0, then evo work
    // ================================================================

    int tasks_per_strategy = std::max(1, num_threads / num_strategies);

    int random_idx = -1;
    for (int s = 0; s < num_strategies; s++)
        if (strategies[s].name == "random") { random_idx = s; break; }

    struct Gen0Task { int strategy_idx; unsigned seed; bool is_symm = false; };
    std::vector<Gen0Task> gen0_task_list;
    for (int s = 0; s < num_strategies; s++)
        for (int t = 0; t < tasks_per_strategy; t++)
            gen0_task_list.push_back({s, (unsigned)(42 + s * 100 + t * 7)});

    // Pad up to num_threads with random inits, reserving the last slot for symm
    if (random_idx >= 0) {
        int extra = num_threads - (int)gen0_task_list.size();
        for (int i = 0; i < extra - 1; i++)
            gen0_task_list.push_back({random_idx, (unsigned)(9999 + i * 31)});
    }
    // Last task: symmetry-aware initialization
    gen0_task_list.push_back({random_idx >= 0 ? random_idx : 0, 10061, true});
    int gen0_tasks = (int)gen0_task_list.size();

    FMOuterConfig gen0_fm = cfg.fm;
    gen0_fm.max_passes = std::min(gen0_fm.max_passes, 50);
    gen0_fm.deadline = deadline;

    // ================================================================
    // Pool + shared state (created before workers launch)
    // ================================================================

    PoolConfig pool_cfg;
    pool_cfg.hard_cap = cfg.pool_size;
    DiversityPool<PoolEntry> pool(pool_cfg,
        [&mhp](const PoolEntry& a, const PoolEntry& b) {
            return partition_distance(a.partition, b.partition, mhp);
        },
        [](const PoolEntry& e) { return e.cost; }
    );

    // Symmetry context — built by the symm worker, published via atomic flag.
    // Workers read symm_ctx only after symm_ready is true (acquire/release).
    symm_mutations::PatternContext symm_ctx;
    std::atomic<bool> symm_ready{false};

    // Discovered patterns — written by symm worker, read after join.
    std::vector<SymmetricPattern> discovered_parallel;
    std::vector<SeriesPattern> discovered_series;

    std::atomic<int> next_gen0{0};
    std::atomic<double> best_ever{1e18};
    std::atomic<int> tasks_since_improve{0};
    std::atomic<int> total_tasks{0};
    std::atomic<int> init_tasks_done{0};
    std::atomic<int> evo_tasks_done{0};
    std::atomic<int> evo_improving{0};
    std::atomic<int64_t> evo_fm_ms_total{0};      // total FM time across all evo tasks
    std::atomic<int> evo_fm_passes_total{0};       // total FM inner passes across evo
    int early_stop_threshold = num_threads * 25;

    std::cerr << "  Init: " << gen0_tasks << " tasks on " << num_threads << " threads\n";

    // ================================================================
    // Unified async worker: gen0 task → evo loop
    // ================================================================

    auto worker = [&]() {
        // VERBOSE=2 enables per-pass FM detail in worker threads
        const char* venv = std::getenv("VERBOSE");
        g_verbose = (venv && venv[0] == '2');
        std::mt19937 rng(std::random_device{}());

        // --- Phase A: pick a gen0 init task (if any remain) ---
        int tid = next_gen0.fetch_add(1);
        if (tid < gen0_tasks) {
            auto start = Clock::now();
            auto& task = gen0_task_list[tid];

            // Symmetry-aware initialization
            if (task.is_symm) {
                auto symm_parts = init_from_patterns(prob, dag, &shared_cache);

                if (!symm_parts.empty()) {
                    // Discover patterns + build symm_ctx
                    auto par_pats = SymmetryDetector::discover(prob, dag, mh, false);
                    auto ser_pats = SeriesDetector::discover(prob, dag, mh, false);

                    symm_ctx = symm_mutations::build_context(prob, dag,
                        par_pats, ser_pats, mh);

                    // Extract representative solutions from symm partitions
                    for (size_t pi = 0; pi < par_pats.size(); pi++) {
                        auto& pat = par_pats[pi];
                        if (pat.symmetry < 2) continue;
                        const auto& rep = pat.components[0];
                        for (auto& sp : symm_parts) {
                            auto config = symm_mutations::extract_config_from_partition(sp, rep);
                            if (!config.empty()) {
                                symm_mutations::RepSolution sol;
                                sol.groups = std::move(config);
                                if (pi < symm_ctx.parallel_solutions.size())
                                    symm_ctx.parallel_solutions[pi].push_back(std::move(sol));
                                break;
                            }
                        }
                    }

                    discovered_parallel = std::move(par_pats);
                    discovered_series = std::move(ser_pats);

                    // Insert symm partitions into pool
                    {
                        std::unique_lock lock(pool.mutex());
                        for (size_t i = 0; i < symm_parts.size(); i++) {
                            double cost = symm_parts[i].total_cost();
                            pool.insert(PoolEntry(std::move(symm_parts[i]), cost,
                                "symm_" + std::to_string(i)));
                        }
                    }

                    // Publish symm_ctx — all writes above are visible after release
                    symm_ready.store(true, std::memory_order_release);

                    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        Clock::now() - start).count();
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cerr << "    init [symm]: " << symm_parts.size()
                              << " partitions in " << ms << "ms\n";
                } else {
                    // No patterns — publish empty symm_ctx so workers don't wait
                    symm_ready.store(true, std::memory_order_release);

                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cerr << "    init [symm]: no patterns, falling back to random\n";
                    // Fall through to standard init
                    goto standard_init;
                }
                goto evo_phase;
            }

            standard_init: {
                auto part = strategies[task.strategy_idx].init(prob, dag, &shared_cache);
                double init_cost = part.total_cost();

                part = greedy_descent(std::move(part));
                double greedy_cost = part.total_cost();

                // Save greedy result as fallback
                {
                    Partition greedy_copy(part);
                    greedy_copy.rebuild_index();
                    if (!partition_has_gap(greedy_copy)) {
                        std::unique_lock lock(pool.mutex());
                        pool.insert(PoolEntry(std::move(greedy_copy), greedy_cost,
                            "greedy+" + strategies[task.strategy_idx].name));
                    }
                }

                FMOuterConfig fc = gen0_fm;
                fc.pass_config.seed = task.seed;
                auto fm = fm_outer_loop(std::move(part), fc);

                // Insert FM best
                PoolEntry fm_result(std::move(fm.best_partition), fm.best_cost,
                    strategies[task.strategy_idx].name);
                fm_result.partition.rebuild_index();
                if (partition_has_gap(fm_result.partition))
                    fm_result.cost = 1e18;

                // Greedy-kick end state for diversity
                PoolEntry kick_result;
                if (fm.end_cost < 1e17 && fm.end_cost > fm.best_cost + 0.01) {
                    auto kicked = greedy_descent(std::move(fm.end_partition));
                    double kick_cost = kicked.total_cost();
                    kicked.rebuild_index();
                    if (!partition_has_gap(kicked))
                        kick_result = {std::move(kicked), kick_cost,
                            "kick+" + strategies[task.strategy_idx].name};
                }

                {
                    std::unique_lock lock(pool.mutex());
                    if (fm_result.cost < 1e17) pool.insert(std::move(fm_result));
                    if (kick_result.cost < 1e17) pool.insert(std::move(kick_result));
                }

                if (init_tasks_done.fetch_add(1) + 1 >= gen0_tasks)
                    shared_cache.freeze_base();

                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::now() - start).count();
                std::lock_guard<std::mutex> lock(log_mutex);
                std::cerr << "    init [" << strategies[task.strategy_idx].name
                          << " s=" << task.seed << "]: init=" << init_cost
                          << " → greedy=" << greedy_cost;
                if (fm.best_cost < greedy_cost - 0.01)
                    std::cerr << " → FM=" << fm.best_cost
                              << " (-" << std::fixed << std::setprecision(1)
                              << 100.0 * (greedy_cost - fm.best_cost) / greedy_cost << "%)";
                else
                    std::cerr << " → FM=same";
                std::cerr << " (" << fm.total_passes << "p "
                          << fm.total_moves << "m " << fm.elapsed_ms << "ms)\n";
            }
        }

        // --- Phase B: evo loop (no barrier — start immediately) ---
        evo_phase:
        if (!has_deadline) return;

        // Update best_ever from pool after our init task
        {
            std::shared_lock lock(pool.mutex());
            if (!pool.empty()) {
                double pb = pool.best_cost();
                double prev = best_ever.load();
                while (pb < prev && !best_ever.compare_exchange_weak(prev, pb))
                    ;
            }
        }

        while (Clock::now() < deadline) {
            if (cfg.early_stop &&
                tasks_since_improve.load() >= early_stop_threshold)
                break;

            // Derive heat from stagnation
            int stale = tasks_since_improve.load();
            double heat = std::clamp(1.0 + stale * 0.05, 0.3, 4.0);

            // --- Select parent(s) under shared lock ---
            Partition child;
            std::string origin;

            {
                std::shared_lock lock(pool.mutex());
                if (pool.empty()) {
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;  // wait for first init result
                }

                size_t pool_sz = pool.size();
                int task_id = total_tasks.load();
                bool do_crossover = (pool_sz >= 2) && (task_id % 3 == 0);
                bool do_symm = !do_crossover &&
                    symm_ready.load(std::memory_order_acquire) &&
                    !symm_ctx.empty() && (rng() % 5 == 0);

                if (do_crossover) {
                    auto [p1, p2] = pool.select_for_crossover(rng);
                    child = crossover(pool[p1].partition, pool[p2].partition, rng, mhp);
                    child.cache = &shared_cache;
                    origin = "xover";
                } else if (do_symm) {
                    size_t pi = pool.select_for_mutation(rng);
                    Partition parent(pool[pi].partition);
                    lock.unlock();

                    auto result = (rng() % 2 == 0)
                        ? symm_mutations::inject_representative_solution(
                            std::move(parent), symm_ctx, prob, dag, rng)
                        : symm_mutations::align_symmetric_reps(
                            std::move(parent), symm_ctx, prob, dag, rng);
                    if (result) {
                        child = std::move(*result);
                        child.cache = &shared_cache;
                        origin = "symm";
                    } else {
                        int num_muts = std::max(2, (int)((4 + (int)(rng() % 5)) * heat));
                        child = mutate_compound(std::move(parent), num_muts, rng);
                        child.cache = &shared_cache;
                        origin = "mutate(" + std::to_string(num_muts) + ")";
                    }
                } else {
                    size_t pi = pool.select_for_mutation(rng);
                    int num_muts = std::max(2, (int)((4 + (int)(rng() % 5)) * heat));
                    child = mutate_compound(Partition(pool[pi].partition), num_muts, rng);
                    child.cache = &shared_cache;
                    origin = "mutate(" + std::to_string(num_muts) + ")";
                }
            }

            // Skip if mutation/crossover created an ephemeral gap
            if (partition_has_gap(child)) {
                total_tasks.fetch_add(1);
                tasks_since_improve.fetch_add(1);
                continue;
            }

            // One greedy pass on the mutated child: if it improves, use the
            // greedy result as the FM starting point; otherwise start from the
            // raw mutation result.
            {
                double pre_greedy = child.total_cost();
                Partition greedy_child = greedy_descent(child);
                if (!partition_has_gap(greedy_child) &&
                    greedy_child.total_cost() < pre_greedy - 0.01)
                    child = std::move(greedy_child);
            }

            // FM outer loop (no lock held — this takes seconds)
            FMOuterConfig fc = cfg.fm;
            fc.max_passes = std::min(fc.max_passes, 50);
            fc.max_no_improve = std::min(fc.max_no_improve, 15);
            fc.deadline = deadline;
            fc.pass_config.max_drift_fraction = std::clamp(
                fc.pass_config.max_drift_fraction * heat, 0.10, 2.0);
            fc.pass_config.seed = (unsigned)(rng());
            auto fm = fm_outer_loop(std::move(child), fc);

            // Track evo FM stats
            evo_tasks_done.fetch_add(1);
            evo_fm_ms_total.fetch_add(fm.elapsed_ms);
            evo_fm_passes_total.fetch_add(fm.total_passes);

            // Validate and insert FM best under unique lock
            PoolEntry fm_result(std::move(fm.best_partition), fm.best_cost, origin);
            fm_result.partition.rebuild_index();
            if (partition_has_gap(fm_result.partition))
                fm_result.cost = 1e18;

            // Greedy-kick end state for diversity
            PoolEntry kick_result;
            if (fm.end_cost < 1e17 && fm.end_cost > fm.best_cost + 0.01 &&
                Clock::now() < deadline) {
                auto kicked = greedy_descent(std::move(fm.end_partition));
                double kick_cost = kicked.total_cost();
                kicked.rebuild_index();
                if (!partition_has_gap(kicked))
                    kick_result = {std::move(kicked), kick_cost, "kick+" + origin};
            }

            double current_best;
            {
                std::unique_lock lock(pool.mutex());
                if (fm_result.cost < 1e17) pool.insert(std::move(fm_result));
                if (kick_result.cost < 1e17) pool.insert(std::move(kick_result));
                current_best = pool.best_cost();
            }

            // Track improvement
            int task_num = total_tasks.fetch_add(1) + 1;
            double prev_best = best_ever.load();
            if (current_best < prev_best - 0.01) {
                while (current_best < prev_best - 0.01 &&
                       !best_ever.compare_exchange_weak(prev_best, current_best))
                    ;
                tasks_since_improve.store(0);
                evo_improving.fetch_add(1);
                std::lock_guard<std::mutex> lock(log_mutex);
                std::cerr << "  Task " << task_num << ": best=" << current_best
                          << " [" << origin << "] ***\n";
            } else {
                tasks_since_improve.fetch_add(1);
            }
        }
    };

    {
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; i++) threads.emplace_back(worker);
        for (auto& t : threads) t.join();
    }

    // Ensure pool has at least one entry (fallback for trivial problems)
    if (pool.empty()) {
        auto fallback = Partition::trivial(prob, dag);
        fallback.cache = &shared_cache;
        pool.insert(PoolEntry(std::move(fallback), fallback.total_cost(), "trivial"));
    }

    // Export discovered patterns to caller
    if (cfg.out_parallel_patterns)
        *cfg.out_parallel_patterns = std::move(discovered_parallel);
    if (cfg.out_series_patterns)
        *cfg.out_series_patterns = std::move(discovered_series);

    pool.sort_by_cost();

    int n_init = init_tasks_done.load();
    int n_evo = evo_tasks_done.load();
    int n_evo_imp = evo_improving.load();
    int64_t evo_ms = evo_fm_ms_total.load();
    int evo_passes = evo_fm_passes_total.load();

    std::cerr << "  Final pool: " << pool.size() << " entries, best=" << pool.best_cost() << "\n";
    std::cerr << "  Init: " << n_init << " tasks";
    if (n_evo > 0) {
        std::cerr << "  Evo: " << n_evo << " tasks (" << n_evo_imp << " improving)";
        std::cerr << "  FM: " << evo_passes << " passes, "
                  << (n_evo > 0 ? evo_ms / n_evo : 0) << "ms/task avg";
    }
    std::cerr << "\n";
    std::cerr << "  Cache: " << shared_cache.size() << " entries, "
              << shared_cache.hits() << " hits, " << shared_cache.misses() << " misses\n";

    std::vector<Partition> result;
    result.reserve(pool.size());
    for (size_t i = 0; i < pool.size(); i++)
        result.push_back(std::move(pool[i].partition));
    return result;
}