#include "search/coupling_parallel_search.h"
#include <cassert>
#include "search/pool.h"
#include "search/evolution.h"
#include "search/local_search.h"   // partition_has_gap
#include "search/coupled_fm_outer.h"
#include "core/cost_cache.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

// ============================================================================
// Pool entry
// ============================================================================

struct CouplingPoolEntry {
    CoupledPartition cp;
    double cost = 1e18;
    std::string origin;

    CouplingPoolEntry() = default;
    CouplingPoolEntry(CoupledPartition c, double cost_, std::string o)
        : cp(std::move(c)), cost(cost_), origin(std::move(o)) {}
};

// ============================================================================
// ARI-based partition distance (same algorithm as parallel_search.cpp)
// ============================================================================

static double coupling_partition_distance(const CouplingPoolEntry& a,
                                           const CouplingPoolEntry& b) {
    const Partition& pa = a.cp.part;
    const Partition& pb = b.cp.part;
    size_t n = pa.prob->num_ops();

    std::vector<int> ga_map(n, -1), gb_map(n, -1);
    int num_ga = 0, num_gb = 0;
    for (size_t gi = 0; gi < pa.groups.size(); gi++)
        if (pa.groups[gi].alive) {
            for (auto op : pa.groups[gi].ops) ga_map[op] = num_ga;
            num_ga++;
        }
    for (size_t gi = 0; gi < pb.groups.size(); gi++)
        if (pb.groups[gi].alive) {
            for (auto op : pb.groups[gi].ops) gb_map[op] = num_gb;
            num_gb++;
        }
    if (num_ga == 0 || num_gb == 0) return 0.0;

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

    auto choose2 = [](int64_t x) -> int64_t { return x * (x - 1) / 2; };
    int64_t same_a = 0, same_b = 0, agree = 0;
    for (int i = 0; i < num_ga; i++) same_a += choose2(row_sum[i]);
    for (int j = 0; j < num_gb; j++) same_b += choose2(col_sum[j]);
    for (int i = 0; i < num_ga; i++)
        for (int j = 0; j < num_gb; j++)
            agree += choose2(table[i * num_gb + j]);

    int64_t total_pairs = choose2(total);
    if (total_pairs == 0) return 0.0;

    double expected  = (double)same_a * same_b / total_pairs;
    double max_agree = (double)(same_a + same_b) / 2.0;
    double denom     = max_agree - expected;
    if (std::abs(denom) < 1e-12) return 0.0;

    double ari = ((double)agree - expected) / denom;
    ari = std::clamp(ari, 0.0, 1.0);
    return 1.0 - ari;
}

// ============================================================================
// coupling_parallel_search — evo loop only (Phase 2 built the pool)
// ============================================================================

Solution coupling_parallel_search(
    std::vector<CoupledPartition> coupled_pool,
    const std::set<size_t>&       feasibly_ret,
    CouplingTimePoint             deadline,
    const CouplingParallelConfig& cfg)
{
    // Caller (solver.cpp Phase 2) guarantees at least one entry.
    assert(!coupled_pool.empty() && "coupling_parallel_search: empty pool");

    int hw_threads = (int)std::thread::hardware_concurrency();
    if (hw_threads <= 0) hw_threads = 4;
    int num_threads = cfg.num_threads > 0 ? cfg.num_threads : hw_threads;

    bool has_deadline = (deadline != CouplingTimePoint::max());
    int  early_stop_threshold = num_threads * 25;

    CostCache local_cache;
    CostCache& cache = cfg.cache ? *cfg.cache : local_cache;

    std::mutex log_mutex;

    // ----------------------------------------------------------------
    // Build pool from the Phase-2-initialized CoupledPartitions
    // ----------------------------------------------------------------
    PoolConfig pool_cfg;
    pool_cfg.hard_cap = (size_t)cfg.pool_size;
    DiversityPool<CouplingPoolEntry> pool(pool_cfg,
        coupling_partition_distance,
        [](const CouplingPoolEntry& e) { return e.cost; });

    {
        std::unique_lock lock(pool.mutex());
        for (size_t i = 0; i < coupled_pool.size(); i++) {
            double cost = coupled_pool[i].total_cost();
            pool.insert(CouplingPoolEntry(std::move(coupled_pool[i]), cost,
                "init_" + std::to_string(i)));
        }
    }

    std::cerr << "  Coupling evo: pool=" << pool.size()
              << " best=" << pool.best_cost()
              << " on " << num_threads << " threads\n";

    if (!has_deadline) {
        pool.sort_by_cost();
        return pool[0].cp.to_solution();
    }

    // ----------------------------------------------------------------
    // Shared counters
    // ----------------------------------------------------------------
    std::atomic<double> best_ever{pool.best_cost()};
    std::atomic<int>    tasks_since_improve{0};
    std::atomic<int>    total_tasks{0};
    std::atomic<int>    evo_tasks_done{0};

    // ================================================================
    // Evo worker: select full CoupledPartition → mutate_compound_coupled →
    //             coupled_fm_outer_loop (partition + coupling moves) → insert
    // ================================================================
    auto worker = [&]() {
        std::mt19937 rng(std::random_device{}());

        while (Clock::now() < deadline) {
            if (cfg.early_stop &&
                tasks_since_improve.load() >= early_stop_threshold)
                break;

            // Select parent and copy the full CoupledPartition (under shared lock)
            CoupledPartition cp_copy;
            int num_muts;
            {
                std::shared_lock lock(pool.mutex());
                int stale = tasks_since_improve.load();
                double heat = std::clamp(1.0 + stale * 0.05, 0.3, 4.0);
                num_muts = std::max(2, (int)((4 + (int)(rng() % 5)) * heat));
                size_t pi = pool.select_for_mutation(rng);
                cp_copy = pool[pi].cp;  // copy full CoupledPartition
            }

            // Mutate at CoupledPartition level — invalidates stale coupling edges
            cp_copy = mutate_compound_coupled(std::move(cp_copy), num_muts, rng);
            if (partition_has_gap(cp_copy.part) || !cp_copy.part.is_acyclic()) {
                total_tasks.fetch_add(1);
                tasks_since_improve.fetch_add(1);
                continue;
            }
            cp_copy.part.cache = &cache;

            // Combined FM: partition moves + coupling moves on the full
            // CoupledPartition. coupled_fm_outer_loop finalizes .sg internally.
            FMOuterConfig fc = cfg.fm;
            fc.max_passes       = std::min(fc.max_passes,     50);
            fc.max_no_improve   = std::min(fc.max_no_improve, 15);
            fc.deadline         = deadline;
            fc.pass_config.seed = (unsigned)rng();
            auto fm = coupled_fm_outer_loop(std::move(cp_copy), feasibly_ret, fc, &cache);

            CoupledPartition child_cp = std::move(fm.best_cp);
            std::string origin = "mutate(" + std::to_string(num_muts) + ")";
            double cost = child_cp.total_cost();

            double current_best;
            {
                std::unique_lock lock(pool.mutex());
                pool.insert(CouplingPoolEntry(std::move(child_cp), cost, origin));
                current_best = pool.best_cost();
            }

            evo_tasks_done.fetch_add(1);
            int task_num = total_tasks.fetch_add(1) + 1;
            double prev_best = best_ever.load();
            if (current_best < prev_best - 0.01) {
                while (current_best < prev_best - 0.01 &&
                       !best_ever.compare_exchange_weak(prev_best, current_best))
                    ;
                tasks_since_improve.store(0);
                {
                    std::lock_guard<std::mutex> lk(log_mutex);
                    std::cerr << "  Coupling task " << task_num << ": best="
                              << current_best << " [" << origin << "] ***\n";
                }
            } else {
                tasks_since_improve.fetch_add(1);
            }
        }
    };

    {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int i = 0; i < num_threads; i++) threads.emplace_back(worker);
        for (auto& t : threads) t.join();
    }

    pool.sort_by_cost();
    std::cerr << "  Coupling final: pool=" << pool.size()
              << " best=" << pool.best_cost()
              << " evo=" << evo_tasks_done.load() << " tasks\n";

    return pool[0].cp.to_solution();
}
