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
    auto group_of = [&](const Partition& p, size_t op) -> int {
        for (size_t gi = 0; gi < p.groups.size(); gi++)
            if (p.groups[gi].alive && p.groups[gi].ops.count(op))
                return (int)gi;
        return -1;
    };
    int disagree = 0, total = 0;
    for (size_t i = 0; i < n; i++) {
        int ga = group_of(a, i), gb = group_of(b, i);
        if (ga < 0 || gb < 0) continue;
        for (size_t j = i + 1; j < n; j++) {
            int ga2 = group_of(a, j), gb2 = group_of(b, j);
            if (ga2 < 0 || gb2 < 0) continue;
            if ((ga == ga2) != (gb == gb2)) disagree++;
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
    // ================================================================

    auto gen0_deadline = deadline;
    if (has_deadline) {
        auto total = deadline - Clock::now();
        gen0_deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(total * 6 / 10);
    }

    FMOuterConfig gen0_fm = cfg.fm;
    gen0_fm.deadline = gen0_deadline;

    int gen0_tasks = num_strategies;
    std::vector<PoolEntry> gen0_results(gen0_tasks);
    std::atomic<int> next_task{0};

    auto gen0_worker = [&]() {
        g_verbose = false;
        while (true) {
            int tid = next_task.fetch_add(1);
            if (tid >= gen0_tasks) break;
            auto start = Clock::now();

            auto part = strategies[tid].init(prob, dag);
            part.cache = &shared_cache;
            part = local_search_from(std::move(part));

            FMOuterConfig fc = gen0_fm;
            fc.pass_config.seed = (unsigned)(42 + tid * 100);
            auto fm = fm_outer_loop(std::move(part), fc);

            gen0_results[tid] = {std::move(fm.best_partition), fm.best_cost,
                                  strategies[tid].name};
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - start).count();
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cerr << "    gen0 [" << gen0_results[tid].origin
                      << "]: " << gen0_results[tid].cost << " (" << ms << "ms)\n";
        }
    };

    std::cerr << "  Gen 0: " << gen0_tasks << " init tasks\n";
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

    std::cerr << "  Pool: " << pool.size() << " entries, best=" << pool[0].cost << "\n";

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
        gen_fm.max_passes = std::min(gen_fm.max_passes, 30);
        gen_fm.max_no_improve = std::min(gen_fm.max_no_improve, 10);

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
                child = greedy_descent(std::move(child));

                FMOuterConfig fc = gen_fm;
                fc.pass_config.seed = (unsigned)(rng());
                auto fm = fm_outer_loop(std::move(child), fc);

                mut_results[tid] = {std::move(fm.best_partition), fm.best_cost, origin};

                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::now() - start).count();
                if (mut_results[tid].cost < best_ever - 0.01) {
                    std::lock_guard<std::mutex> lock(log_mutex);
                    std::cerr << "    gen" << gen << " [" << origin
                              << "]: " << mut_results[tid].cost
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

        if (gens_no_improve >= 15) {
            std::cerr << "  Stopping after " << gen << " generations\n";
            break;
        }
        gen++;
    }

    std::cerr << "  Final pool: " << pool.size() << " entries, best=" << pool[0].cost
              << " (" << gen - 1 << " generations)\n";
    std::cerr << "  Cache: " << shared_cache.size() << " entries, "
              << shared_cache.hits() << " hits, " << shared_cache.misses() << " misses\n";

    return std::move(pool[0].partition);
}