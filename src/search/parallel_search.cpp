#include "search/parallel_search.h"
#include "search/verbose.h"
#include "init/init_strategies.h"
#include "search/local_search.h"
#include "search/fm_outer.h"
#include "search/evolution.h"
#include "core/cost_cache.h"
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <atomic>

using Clock = std::chrono::steady_clock;

// ============================================================================
// Pool entry
// ============================================================================

struct PoolEntry {
    Partition partition;
    double cost = 1e18;
    std::string origin;
};

// ============================================================================
// Partition distance
// ============================================================================

static double partition_distance(const Partition& a, const Partition& b) {
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

    // Build contingency table: n_ij = |group_i_in_a ∩ group_j_in_b|
    // Also track row/col totals.
    std::vector<std::vector<int>> table(num_ga, std::vector<int>(num_gb, 0));
    std::vector<int> row_sum(num_ga, 0), col_sum(num_gb, 0);
    int total = 0;

    for (size_t op = 0; op < n; op++) {
        if (ga_map[op] < 0 || gb_map[op] < 0) continue;
        table[ga_map[op]][gb_map[op]]++;
        row_sum[ga_map[op]]++;
        col_sum[gb_map[op]]++;
        total++;
    }
    if (total <= 1) return 0.0;

    // Rand index via contingency table:
    //   same_in_a = Σ_i C(row_i, 2)
    //   same_in_b = Σ_j C(col_j, 2)
    //   agreements = Σ_ij C(n_ij, 2)
    //   disagree = same_in_a + same_in_b - 2 * agreements
    //   total_pairs = C(total, 2)
    auto choose2 = [](int64_t x) -> int64_t { return x * (x - 1) / 2; };

    int64_t same_a = 0, same_b = 0, agree = 0;
    for (int i = 0; i < num_ga; i++) same_a += choose2(row_sum[i]);
    for (int j = 0; j < num_gb; j++) same_b += choose2(col_sum[j]);
    for (int i = 0; i < num_ga; i++)
        for (int j = 0; j < num_gb; j++)
            agree += choose2(table[i][j]);

    int64_t disagree = same_a + same_b - 2 * agree;
    int64_t total_pairs = choose2(total);
    return total_pairs > 0 ? (double)disagree / total_pairs : 0.0;
}

// ============================================================================
// Pool management
// ============================================================================

static bool pool_insert(std::vector<PoolEntry>& pool, PoolEntry entry,
                         size_t max_pool) {
    // Compute min distance to existing entries
    double min_dist = 1.0;
    size_t closest_idx = 0;
    for (size_t i = 0; i < pool.size(); i++) {
        double dist = partition_distance(pool[i].partition, entry.partition);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }

    // Near-duplicate: only replace if strictly better
    if (min_dist < 0.01) {
        if (entry.cost < pool[closest_idx].cost - 0.01) {
            pool[closest_idx] = std::move(entry);
            return true;
        }
        return false;
    }

    // Pool not full: always add
    if (pool.size() < max_pool) {
        pool.push_back(std::move(entry));
        return true;
    }

    // Pool full: diversity-aware eviction.
    // Never evict the best-cost entry.
    size_t best_cost_idx = 0;
    for (size_t i = 1; i < pool.size(); i++)
        if (pool[i].cost < pool[best_cost_idx].cost) best_cost_idx = i;

    // Find the least-unique entry (smallest nearest-neighbor distance), excluding best
    size_t least_unique = SIZE_MAX;
    double least_unique_dist = 2.0;
    for (size_t i = 0; i < pool.size(); i++) {
        if (i == best_cost_idx) continue;
        double nn_dist = 1.0;
        for (size_t j = 0; j < pool.size(); j++) {
            if (i == j) continue;
            double d = partition_distance(pool[i].partition, pool[j].partition);
            nn_dist = std::min(nn_dist, d);
        }
        if (nn_dist < least_unique_dist) {
            least_unique_dist = nn_dist;
            least_unique = i;
        }
    }

    if (least_unique == SIZE_MAX) return false;

    // Replace least-unique if candidate brings more diversity or better cost with diversity
    bool more_diverse = (min_dist > least_unique_dist + 0.01);
    bool better_cost = (entry.cost < pool[least_unique].cost - 0.01);
    bool decent_diversity = (min_dist > 0.02);

    if (more_diverse || (better_cost && decent_diversity)) {
        pool[least_unique] = std::move(entry);
        return true;
    }
    return false;
}

static void pool_sort(std::vector<PoolEntry>& pool) {
    std::sort(pool.begin(), pool.end(),
              [](const PoolEntry& a, const PoolEntry& b) { return a.cost < b.cost; });
}

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

    CostCache shared_cache;
    std::mutex log_mutex;

    auto deadline = cfg.fm.deadline;
    bool has_deadline = (deadline != TimePoint::max());

    // ================================================================
    // Generation 0: init strategies → greedy+tabu+FM
    // Fill all threads: multiple seeds per strategy if we have spare threads.
    // ================================================================

    auto gen0_deadline = deadline;
    if (has_deadline) {
        auto total = deadline - Clock::now();
        gen0_deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(total * 5 / 10);
    }

    FMOuterConfig gen0_fm = cfg.fm;
    gen0_fm.deadline = gen0_deadline;

    int tasks_per_strategy = std::max(1, num_threads / num_strategies);
    int gen0_tasks = num_strategies * tasks_per_strategy;

    // If we still have spare threads, fill with extra random inits
    // (random strategy produces different results each call)
    int random_idx = -1;
    for (int s = 0; s < num_strategies; s++)
        if (strategies[s].name == "random") { random_idx = s; break; }

    struct Gen0Task { int strategy_idx; unsigned seed; };
    std::vector<Gen0Task> gen0_task_list;
    for (int s = 0; s < num_strategies; s++)
        for (int t = 0; t < tasks_per_strategy; t++)
            gen0_task_list.push_back({s, (unsigned)(42 + s * 100 + t * 7)});

    // Pad up to num_threads with random inits
    if (random_idx >= 0) {
        int extra = num_threads - (int)gen0_task_list.size();
        for (int i = 0; i < extra; i++)
            gen0_task_list.push_back({random_idx,
                (unsigned)(9999 + i * 31)});
    }
    gen0_tasks = (int)gen0_task_list.size();

    std::vector<PoolEntry> gen0_results(gen0_tasks);
    std::atomic<int> next_task{0};

    auto gen0_worker = [&]() {
        g_verbose = false;
        while (true) {
            int tid = next_task.fetch_add(1);
            if (tid >= gen0_tasks) break;
            auto start = Clock::now();
            auto& task = gen0_task_list[tid];

            auto part = strategies[task.strategy_idx].init(prob, dag);
            part.cache = &shared_cache;
            double init_cost = part.total_cost();

            part = greedy_descent(std::move(part));
            double greedy_cost = part.total_cost();

            FMOuterConfig fc = gen0_fm;
            fc.pass_config.seed = task.seed;
            auto fm = fm_outer_loop(std::move(part), fc);
            double fm_cost = fm.best_cost;

            gen0_results[tid] = {std::move(fm.best_partition), fm_cost,
                                  strategies[task.strategy_idx].name};
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - start).count();
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cerr << "    gen0 [" << strategies[task.strategy_idx].name
                      << " s=" << task.seed << "]: init=" << init_cost
                      << " → greedy=" << greedy_cost;
            if (fm_cost < greedy_cost - 0.01)
                std::cerr << " → FM=" << fm_cost
                          << " (-" << std::fixed << std::setprecision(1)
                          << 100.0 * (greedy_cost - fm_cost) / greedy_cost << "%)";
            else
                std::cerr << " → FM=same";
            std::cerr << " (" << ms << "ms)\n";
        }
    };

    std::cerr << "  Gen 0: " << gen0_tasks << " tasks on " 
              << std::min(num_threads, gen0_tasks) << " threads\n";
    {
        std::vector<std::thread> threads;
        int nt = std::min(num_threads, gen0_tasks);
        for (int i = 0; i < nt; i++) threads.emplace_back(gen0_worker);
        for (auto& t : threads) t.join();
    }

    std::vector<PoolEntry> pool;
    for (auto& r : gen0_results)
        pool_insert(pool, std::move(r), cfg.pool_size);
    pool_sort(pool);

    std::cerr << "  Pool after gen0: " << pool.size() << " entries, best=" << pool[0].cost << "\n";
    double gen0_best = pool[0].cost;

    // ================================================================
    // Generation 1+: mutation + crossover → greedy → FM
    // ================================================================

    int gen = 1;
    int gens_no_improve = 0;
    double best_ever = pool[0].cost;

    while (has_deadline && Clock::now() < deadline) {
        auto remaining = deadline - Clock::now();
        auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count();
        if (gen_ms < 200) break;

        FMOuterConfig gen_fm = cfg.fm;
        gen_fm.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(remaining * 8 / 10);
        gen_fm.max_passes = std::min(gen_fm.max_passes, 50);
        gen_fm.max_no_improve = std::min(gen_fm.max_no_improve, 15);

        int mut_tasks = std::min(num_threads, std::max(2, (int)pool.size()));
        std::vector<PoolEntry> mut_results(mut_tasks);
        next_task.store(0);

        auto evo_worker = [&]() {
            g_verbose = false;
            while (true) {
                int tid = next_task.fetch_add(1);
                if (tid >= mut_tasks) break;
                if (Clock::now() >= deadline) break;

                std::mt19937 rng(std::random_device{}());
                auto start = Clock::now();

                Partition child;
                child.prob = &prob;
                child.dag = &dag;
                child.cache = &shared_cache;
                std::string origin;

                // Mix of operators:
                // - Half mutation, half crossover (when pool has ≥2)
                bool do_crossover = (pool.size() >= 2) && (tid % 3 == 0);

                if (do_crossover) {
                    // Pick two different parents (uniform random)
                    size_t p1 = rng() % pool.size();
                    size_t p2 = rng() % pool.size();
                    while (p2 == p1 && pool.size() > 1) p2 = rng() % pool.size();

                    child = crossover(pool[p1].partition, pool[p2].partition, rng);
                    child.cache = &shared_cache;
                    origin = "xover";
                } else {
                    // Pick parent (uniform random — pool diversity maintained by insertion)
                    size_t pi = rng() % pool.size();

                    int num_muts = 2 + (int)(rng() % 5);  // 2..6 mutations
                    child = mutate_compound(Partition(pool[pi].partition), num_muts, rng);
                    child.cache = &shared_cache;
                    origin = "mutate(" + std::to_string(num_muts) + ")";
                }

                // Refine: greedy + FM
                double after_mutate = child.total_cost();
                child = greedy_descent(std::move(child));
                double after_greedy = child.total_cost();

                FMOuterConfig fc = gen_fm;
                fc.pass_config.seed = (unsigned)(rng());
                auto fm = fm_outer_loop(std::move(child), fc);
                double after_fm = fm.best_cost;

                mut_results[tid] = {std::move(fm.best_partition), after_fm, origin};

                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::now() - start).count();
                if (after_fm < best_ever - 0.01) {
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cerr << "    gen" << gen << " [" << origin
                              << "]: mutated=" << after_mutate
                              << " → greedy=" << after_greedy
                              << " → FM=" << after_fm
                              << " (" << ms << "ms) ***NEW BEST***\n";
                }
            }
        };

        {
            std::vector<std::thread> threads;
            int nt = std::min(num_threads, mut_tasks);
            for (int i = 0; i < nt; i++) threads.emplace_back(evo_worker);
            for (auto& t : threads) t.join();
        }

        int accepted = 0;
        for (auto& r : mut_results)
            if (r.cost < 1e17 && pool_insert(pool, std::move(r), cfg.pool_size))
                accepted++;
        pool_sort(pool);

        if (pool[0].cost < best_ever - 0.01) {
            best_ever = pool[0].cost;
            gens_no_improve = 0;
            std::cerr << "  Gen " << gen << ": best=" << best_ever
                      << " (+" << accepted << ") ***\n";
        } else {
            gens_no_improve++;
        }

        if (gens_no_improve >= 25) {
            std::cerr << "  Stopping after " << gen << " generations\n";
            break;
        }
        gen++;
    }

    std::cerr << "  Final pool: " << pool.size() << " entries, best=" << pool[0].cost
              << " (" << gen - 1 << " generations)";
    if (pool[0].cost < gen0_best - 0.01)
        std::cerr << "  evo improved by " << std::fixed << std::setprecision(1)
                  << 100.0 * (gen0_best - pool[0].cost) / gen0_best << "%";
    std::cerr << "\n";
    std::cerr << "  Cache: " << shared_cache.size() << " entries, "
              << shared_cache.hits() << " hits, " << shared_cache.misses() << " misses\n";

    std::vector<Partition> result;
    result.reserve(pool.size());
    for (auto& pe : pool)
        result.push_back(std::move(pe.partition));
    return result;
}