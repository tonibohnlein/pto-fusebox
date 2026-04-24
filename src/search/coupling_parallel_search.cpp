#include "search/coupling_parallel_search.h"
#include <cassert>
#include "search/pool.h"
#include "search/evolution.h"
#include "search/symm_mutations.h"
#include "search/local_search.h"   // partition_has_gap
#include "search/coupled_fm_outer.h"
#include "search/coupling_search.h"
#include "core/cost_cache.h"
#include "symmetry/merkle_hash.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
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
    std::vector<int> group_map;  // cached op→group index for distance computation

    CouplingPoolEntry() = default;
    CouplingPoolEntry(CoupledPartition c, double cost_, std::string o)
        : cp(std::move(c)), cost(cost_), origin(std::move(o))
    {
        // Precompute group map
        size_t n = cp.part.prob->num_ops();
        group_map.assign(n, -1);
        int gidx = 0;
        for (size_t gi = 0; gi < cp.part.groups.size(); gi++)
            if (cp.part.groups[gi].alive) {
                for (auto op : cp.part.groups[gi].ops) group_map[op] = gidx;
                gidx++;
            }
    }
};

// ============================================================================
// ARI-based partition distance (same algorithm as parallel_search.cpp)
// ============================================================================

static double coupling_partition_distance(const CouplingPoolEntry& a,
                                           const CouplingPoolEntry& b) {
    size_t n = a.group_map.size();

    // --- Component 1: Adjusted Rand Index on op-to-group assignment ---
    // Uses cached group_map from pool entry (precomputed on insertion).

    const auto& ga_map = a.group_map;
    const auto& gb_map = b.group_map;
    int num_ga = 0, num_gb = 0;
    for (auto v : ga_map) if (v >= num_ga) num_ga = v + 1;
    for (auto v : gb_map) if (v >= num_gb) num_gb = v + 1;

    double ari_dist = 0.0;
    if (num_ga > 0 && num_gb > 0) {
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

        if (total > 1) {
            auto choose2 = [](int64_t x) -> int64_t { return x * (x - 1) / 2; };
            int64_t same_a = 0, same_b = 0, agree = 0;
            for (int i = 0; i < num_ga; i++) same_a += choose2(row_sum[i]);
            for (int j = 0; j < num_gb; j++) same_b += choose2(col_sum[j]);
            for (int i = 0; i < num_ga; i++)
                for (int j = 0; j < num_gb; j++)
                    agree += choose2(table[i * num_gb + j]);

            int64_t total_pairs = choose2(total);
            if (total_pairs > 0) {
                double expected  = (double)same_a * same_b / total_pairs;
                double max_agree = (double)(same_a + same_b) / 2.0;
                double denom     = max_agree - expected;
                if (std::abs(denom) > 1e-12) {
                    double ari = ((double)agree - expected) / denom;
                    ari_dist = 1.0 - std::clamp(ari, 0.0, 1.0);
                }
            }
        }
    }

    // --- Component 2: Jaccard distance on retained tensor sets ---

    // Collect the set of all retained tensor IDs from each coupled partition.
    FlatSet<size_t> ra, rb;
    for (auto& [edge, tensors] : a.cp.retained)
        ra.insert(tensors.begin(), tensors.end());
    for (auto& [edge, tensors] : b.cp.retained)
        rb.insert(tensors.begin(), tensors.end());

    double jaccard_dist = 0.0;
    if (!ra.empty() || !rb.empty()) {
        size_t intersection = 0;
        for (auto t : ra)
            if (rb.count(t)) intersection++;
        size_t union_size = ra.size() + rb.size() - intersection;
        jaccard_dist = 1.0 - (double)intersection / (double)union_size;
    }

    // --- Blend: partition structure (dominant) + coupling topology ---
    constexpr double alpha = 0.6;
    return alpha * ari_dist + (1.0 - alpha) * jaccard_dist;
}

// ============================================================================
// coupling_parallel_search — evo loop only (Phase 2 built the pool)
// ============================================================================

Solution coupling_parallel_search(
    std::vector<CoupledPartition> coupled_pool,
    const FlatSet<size_t>&       feasibly_ret,
    CouplingTimePoint             deadline,
    const CouplingParallelConfig& cfg)
{
    // Caller (solver.cpp Phase 2) guarantees at least one entry.
    assert(!coupled_pool.empty() && "coupling_parallel_search: empty pool");

    // Save prob/dag pointers before the pool moves entries.
    const Problem& prob = *coupled_pool[0].part.prob;
    const DAG& dag = *coupled_pool[0].part.dag;

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
        for (size_t i = 0; i < pool.size(); i++) {
            auto sol = pool[i].cp.to_solution();
            auto vr = sol.validate();
            if (vr.valid) return sol;
            std::cerr << "WARNING: pool[" << i << "] (cost="
                      << pool[i].cost << ") invalid: " << vr.error << "\n";
        }
        return pool[0].cp.to_solution();  // all invalid, return best anyway
    }

    // ----------------------------------------------------------------
    // Shared counters
    // ----------------------------------------------------------------
    std::atomic<double> best_ever{pool.best_cost()};
    std::atomic<int>    tasks_since_improve{0};
    std::atomic<int>    total_tasks{0};
    std::atomic<int>    evo_tasks_done{0};

    // ================================================================
    // Evo worker: three operators — crossover, symmetry mutation, compound mutation
    // ================================================================
    const bool has_symm = cfg.symm_ctx && !cfg.symm_ctx->empty();

    auto worker = [&]() {
        std::mt19937 rng(std::random_device{}());

        while (Clock::now() < deadline) {
            if (cfg.early_stop &&
                tasks_since_improve.load() >= early_stop_threshold)
                break;

            int stale = tasks_since_improve.load();
            double heat = std::clamp(1.0 + stale * 0.05, 0.3, 4.0);
            int num_muts = std::max(2, (int)((4 + (int)(rng() % 5)) * heat));
            int task_id = total_tasks.load();

            // --- Operator selection ---
            size_t pool_sz;
            {
                std::shared_lock lock(pool.mutex());
                pool_sz = pool.size();
            }

            bool do_crossover = (pool_sz >= 2) && (task_id % 3 == 0);
            // Scale symmetry frequency with number of patterns:
            // more patterns → more symmetric structure to exploit → higher frequency.
            // 0 patterns: never. 1-2 patterns: ~25%. 3+: ~40%.
            bool do_symm = false;
            if (!do_crossover && has_symm) {
                size_t n_patterns = cfg.symm_ctx->parallel.size()
                                  + cfg.symm_ctx->series.size();
                int symm_pct = (n_patterns >= 3) ? 40 : 25;
                do_symm = ((int)(rng() % 100) < symm_pct);
            }

            CoupledPartition cp_child;
            std::string origin;

            if (do_crossover) {
                // --- Crossover: combine partition structure from two parents,
                //     then inherit coupling edges where possible ---
                Partition child_part;
                std::vector<std::pair<size_t, FlatSet<size_t>>> parent_retained_list;
                bool used_dag_cut = false;
                {
                    std::shared_lock lock(pool.mutex());

                    bool use_dag_cut = (rng() % 2 == 0) && pool.size() >= 3;
                    if (use_dag_cut) {
                        // DAG-cut crossover: collect top candidates from pool
                        size_t k = std::min<size_t>(5, pool.size());
                        std::vector<const Partition*> candidates;
                        for (size_t ci = 0; ci < k; ci++)
                            candidates.push_back(&pool[ci].cp.part);
                        child_part = crossover_dag_cut(candidates, dag, prob,
                                                        &cache, rng);
                        used_dag_cut = true;
                        // Collect retained tensors from all candidates for seeding
                        for (size_t ci = 0; ci < k; ci++)
                            for (auto& [edge, tensors] : pool[ci].cp.retained)
                                parent_retained_list.push_back({edge.first, tensors});
                    } else {
                        auto [p1, p2] = pool.select_for_crossover(rng);
                        child_part = crossover(pool[p1].cp.part, pool[p2].cp.part,
                                               rng, cfg.merkle);
                        // Collect retained tensors from both parents for seeding
                        for (auto pidx : {p1, p2})
                            for (auto& [edge, tensors] : pool[pidx].cp.retained)
                                parent_retained_list.push_back({edge.first, tensors});
                    }
                }
                child_part.cache = &cache;
                if (partition_has_gap(child_part)) {  // partition_has_gap checks acyclicity internally
                    total_tasks.fetch_add(1);
                    tasks_since_improve.fetch_add(1);
                    continue;
                }
                // Build coupled partition from crossover child
                cp_child.part = std::move(child_part);
                cp_child.part.cache = &cache;
                size_t ng = cp_child.part.groups.size();
                cp_child.next_group.assign(ng, SIZE_MAX);
                cp_child.prev_group.assign(ng, SIZE_MAX);
                cp_child.retained.clear();

                // Seed coupling from parents' retained tensors: for each
                // tensor a parent retained, try to couple the child's
                // producer→consumer groups.
                cp_child.part.rebuild_index();
                cp_child.part.finalize(&cache);
                for (auto& [parent_ga_unused, tensors] : parent_retained_list) {
                    (void)parent_ga_unused;
                    for (auto t : tensors) {
                        if (!feasibly_ret.count(t)) continue;
                        int prod = dag.tensor_producer[t];
                        if (prod < 0) continue;
                        size_t ga = SIZE_MAX;
                        for (auto g : cp_child.part.groups_of((size_t)prod))
                            if (cp_child.part.groups[g].alive) { ga = g; break; }
                        if (ga == SIZE_MAX) continue;
                        if (ga >= cp_child.next_group.size() ||
                            cp_child.next_group[ga] != SIZE_MAX) continue;
                        for (auto cop : dag.tensor_consumers[t]) {
                            for (auto gb : cp_child.part.groups_of(cop)) {
                                if (gb == ga || !cp_child.part.groups[gb].alive) continue;
                                if (gb >= cp_child.prev_group.size() ||
                                    cp_child.prev_group[gb] != SIZE_MAX) continue;
                                if (!is_boundary_input_of(
                                        cp_child.part.groups[gb].ops, t, dag)) continue;
                                auto ev = eval_couple(cp_child, ga, gb, t);
                                if (ev.feasible) {
                                    apply_couple(cp_child, ga, gb, t);
                                    break;
                                }
                            }
                            if (ga < cp_child.next_group.size() &&
                                cp_child.next_group[ga] != SIZE_MAX) break;
                        }
                    }
                }
                origin = used_dag_cut ? "dagcut" : "xover";

            } else if (do_symm) {
                // --- Symmetry mutation: inject or align representatives ---
                Partition parent_part;
                {
                    std::shared_lock lock(pool.mutex());
                    size_t pi = pool.select_for_mutation(rng);
                    parent_part = pool[pi].cp.part;
                }
                auto result = symm_mutations::align_symmetric_reps(
                          std::move(parent_part), *cfg.symm_ctx, prob, dag, rng);
                if (!result) {
                    // Fallback to compound mutation
                    CoupledPartition cp_copy;
                    {
                        std::shared_lock lock(pool.mutex());
                        size_t pi = pool.select_for_mutation(rng);
                        cp_copy = pool[pi].cp;
                    }
                    cp_copy = mutate_compound_coupled(std::move(cp_copy), num_muts, rng);
                    auto is_ret = [&](size_t t) { return cp_copy.is_retained(t); };
                    if (partition_has_gap(cp_copy.part, is_ret)) {
                        total_tasks.fetch_add(1);
                        tasks_since_improve.fetch_add(1);
                        continue;
                    }
                    cp_child = std::move(cp_copy);
                    origin = "mutate(" + std::to_string(num_muts) + ")";
                } else {
                    if (partition_has_gap(*result)) {
                        total_tasks.fetch_add(1);
                        tasks_since_improve.fetch_add(1);
                        continue;
                    }
                    cp_child.part = std::move(*result);
                    cp_child.part.cache = &cache;
                    size_t ng = cp_child.part.groups.size();
                cp_child.next_group.assign(ng, SIZE_MAX);
                cp_child.prev_group.assign(ng, SIZE_MAX);
                cp_child.retained.clear();
                    origin = "symm";
                }

            } else {
                // --- Compound mutation: partition + coupling moves ---
                CoupledPartition cp_copy;
                {
                    std::shared_lock lock(pool.mutex());
                    size_t pi = pool.select_for_mutation(rng);
                    cp_copy = pool[pi].cp;
                }
                cp_copy = mutate_compound_coupled(std::move(cp_copy), num_muts, rng);
                auto is_ret2 = [&](size_t t) { return cp_copy.is_retained(t); };
                if (partition_has_gap(cp_copy.part, is_ret2)) {
                    total_tasks.fetch_add(1);
                    tasks_since_improve.fetch_add(1);
                    continue;
                }
                cp_child = std::move(cp_copy);
                origin = "mutate(" + std::to_string(num_muts) + ")";
            }

            cp_child.part.cache = &cache;

            // Combined FM: partition moves + coupling moves
            FMOuterConfig fc = cfg.fm;
            fc.max_passes       = std::min(fc.max_passes,     20);
            fc.max_no_improve   = std::min(fc.max_no_improve, 8);
            fc.deadline         = deadline;
            fc.pass_config.seed = (unsigned)rng();
            auto fm = coupled_fm_outer_loop(std::move(cp_child), feasibly_ret, fc, &cache);

            CoupledPartition child_cp = std::move(fm.best_cp);

            // Reject if FM pass introduced an ephemeral gap.
            {
                child_cp.part.rebuild_index();
                auto is_ret = [&](size_t t) { return child_cp.is_retained(t); };
                if (partition_has_gap(child_cp.part, is_ret)) {
                    total_tasks.fetch_add(1);
                    tasks_since_improve.fetch_add(1);
                    continue;
                }
            }

            double cost = child_cp.total_cost();

            // Greedy refinement kick on FM's best — cheap polish that catches
            // any local coupling/partition moves FM left behind.  Inserted as
            // a separate pool entry so both the FM state and the refined state
            // are present and can seed future evo tasks.
            CouplingPoolEntry refined_entry;
            if (cost < 1e17 && Clock::now() < deadline) {
                CoupledPartition refined_cp = child_cp;
                double refined_cost = coupling_greedy_descent(refined_cp, feasibly_ret, deadline);
                refined_cp.part.rebuild_index();
                auto is_ret_r = [&](size_t t) { return refined_cp.is_retained(t); };
                if (!partition_has_gap(refined_cp.part, is_ret_r) &&
                    refined_cost < cost - 0.01) {
                    refined_entry = CouplingPoolEntry(std::move(refined_cp), refined_cost,
                                                      "kick+" + origin);
                }
            }

            double current_best;
            {
                std::unique_lock lock(pool.mutex());
                pool.insert(CouplingPoolEntry(std::move(child_cp), cost, origin));
                if (refined_entry.cost < 1e17)
                    pool.insert(std::move(refined_entry));
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

    // Try pool entries best-to-worst until one converts to a valid solution.
    for (size_t i = 0; i < pool.size(); i++) {
        auto sol = pool[i].cp.to_solution();
        auto vr = sol.validate();
        if (vr.valid) return sol;
        std::cerr << "WARNING: pool[" << i << "] (cost="
                  << pool[i].cost << ") invalid: " << vr.error << "\n";
    }
    // All invalid — return best anyway (solver.cpp will fall back to uncoupled)
    return pool[0].cp.to_solution();
}
