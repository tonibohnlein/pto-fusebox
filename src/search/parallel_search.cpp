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
    // Build op→group maps once (O(ops×groups) but only done once per call)
    std::vector<int> ga_map(n, -1), gb_map(n, -1);
    for (size_t gi = 0; gi < a.groups.size(); gi++)
        if (a.groups[gi].alive)
            for (auto op : a.groups[gi].ops)
                ga_map[op] = (int)gi;
    for (size_t gi = 0; gi < b.groups.size(); gi++)
        if (b.groups[gi].alive)
            for (auto op : b.groups[gi].ops)
                gb_map[op] = (int)gi;

    int disagree = 0, total = 0;
    for (size_t i = 0; i < n; i++) {
        if (ga_map[i] < 0 || gb_map[i] < 0) continue;
        for (size_t j = i + 1; j < n; j++) {
            if (ga_map[j] < 0 || gb_map[j] < 0) continue;
            if ((ga_map[i] == ga_map[j]) != (gb_map[i] == gb_map[j])) disagree++;
            total++;
        }
    }
    return total > 0 ? (double)disagree / total : 0.0;
}

// ============================================================================
// Pool management
// ============================================================================

static bool pool_insert(std::vector<PoolEntry>& pool, PoolEntry entry,
                         size_t max_pool) {
    // If similar to a better member, reject. If similar but better, replace.
    for (auto& pe : pool) {
        double dist = partition_distance(pe.partition, entry.partition);
        if (dist < 0.03) {
            if (entry.cost < pe.cost - 0.01) {
                pe = std::move(entry);
                return true;
            }
            return false;
        }
    }
    if (pool.size() < max_pool) {
        pool.push_back(std::move(entry));
        return true;
    }
    size_t worst = 0;
    for (size_t i = 1; i < pool.size(); i++)
        if (pool[i].cost > pool[worst].cost) worst = i;
    if (entry.cost < pool[worst].cost - 0.01) {
        pool[worst] = std::move(entry);
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

Partition parallel_search(const Problem& prob, const DAG& dag,
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

    struct Gen0Task { int strategy_idx; unsigned seed; };
    std::vector<Gen0Task> gen0_task_list;
    for (int s = 0; s < num_strategies; s++)
        for (int t = 0; t < tasks_per_strategy; t++)
            gen0_task_list.push_back({s, (unsigned)(42 + s * 100 + t * 7)});
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
        pool_insert(pool, std::move(r), 8);
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
                    // Pick two different parents via tournament
                    size_t p1 = rng() % pool.size();
                    size_t p2 = rng() % pool.size();
                    while (p2 == p1 && pool.size() > 1) p2 = rng() % pool.size();

                    child = crossover(pool[p1].partition, pool[p2].partition, rng);
                    child.cache = &shared_cache;
                    origin = "xover";
                } else {
                    // Pick parent via tournament
                    size_t p1 = rng() % pool.size();
                    size_t p2 = rng() % pool.size();
                    size_t pi = (pool[p1].cost <= pool[p2].cost) ? p1 : p2;

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
            if (r.cost < 1e17 && pool_insert(pool, std::move(r), 8))
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

    return std::move(pool[0].partition);
}