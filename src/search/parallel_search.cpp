#include "search/parallel_search.h"
#include "search/verbose.h"
#include "init/init_strategies.h"
#include "search/local_search.h"
#include "search/fm_outer.h"
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <chrono>

// ============================================================================
// A search task: (init strategy, seed offset) → Partition
// ============================================================================

struct SearchTask {
    int strategy_idx;
    unsigned seed_offset;
};

struct SearchResult {
    Partition partition;
    double cost = 1e18;
    int task_id = -1;
    std::string desc;
};

// Run one search task: init → greedy → FM
static SearchResult run_task(const Problem& prob, const DAG& dag,
                              const InitStrategy& strategy,
                              unsigned seed_offset, int task_id,
                              const FMOuterConfig& fm_template) {
    SearchResult result;
    result.task_id = task_id;
    result.desc = strategy.name + " (seed=" + std::to_string(seed_offset) + ")";

    // Phase 1: Initialize
    auto part = strategy.init(prob, dag);

    // Phase 2: Greedy local search
    part = local_search_from(std::move(part));

    // Phase 3: FM refinement — use propagated config with per-task seed
    FMOuterConfig fm_cfg = fm_template;
    fm_cfg.pass_config.seed = seed_offset;

    auto fm_result = fm_outer_loop(std::move(part), fm_cfg);

    result.partition = std::move(fm_result.best_partition);
    result.cost = fm_result.best_cost;
    return result;
}

// ============================================================================
// Parallel search
// ============================================================================

Partition parallel_search(const Problem& prob, const DAG& dag,
                          const ParallelConfig& cfg) {
    auto strategies = all_init_strategies();
    int num_strategies = (int)strategies.size();

    // Generate tasks: scale to available parallelism
    int hw_threads = (int)std::thread::hardware_concurrency();
    if (hw_threads <= 0) hw_threads = 4;
    int num_threads = cfg.num_threads > 0 ? cfg.num_threads : hw_threads;

    // Target: ~1 task per thread, at least 1 per strategy
    int tasks_per_init = cfg.tasks_per_init;
    if (tasks_per_init <= 0)
        tasks_per_init = std::max(1, num_threads / num_strategies);

    std::vector<SearchTask> tasks;
    for (int s = 0; s < num_strategies; s++) {
        for (int t = 0; t < tasks_per_init; t++) {
            tasks.push_back({s, (unsigned)(42 + s * 100 + t * 7)});
        }
    }

    int num_tasks = (int)tasks.size();
    num_threads = std::min(num_threads, num_tasks);

    std::cerr << "  Parallel search: " << num_tasks << " tasks on "
              << num_threads << " threads\n";

    // Results storage
    std::vector<SearchResult> results(num_tasks);
    std::mutex log_mutex;

    // Task queue
    std::atomic<int> next_task{0};

    auto worker = [&]() {
        g_verbose = false;  // suppress internal logging in worker threads
        while (true) {
            int task_id = next_task.fetch_add(1);
            if (task_id >= num_tasks) break;

            auto& task = tasks[task_id];
            auto start = std::chrono::steady_clock::now();

            results[task_id] = run_task(prob, dag,
                                         strategies[task.strategy_idx],
                                         task.seed_offset, task_id,
                                         cfg.fm);

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();

            {
                std::lock_guard<std::mutex> lock(log_mutex);
                std::cerr << "    task " << task_id << " ["
                          << results[task_id].desc << "]: cost="
                          << results[task_id].cost << " (" << elapsed << "ms)\n";
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++)
        threads.emplace_back(worker);
    for (auto& t : threads)
        t.join();

    // Find best
    int best_idx = 0;
    for (int i = 1; i < num_tasks; i++)
        if (results[i].cost < results[best_idx].cost)
            best_idx = i;

    std::cerr << "  Best: task " << best_idx << " ["
              << results[best_idx].desc << "] cost="
              << results[best_idx].cost << "\n";

    return std::move(results[best_idx].partition);
}