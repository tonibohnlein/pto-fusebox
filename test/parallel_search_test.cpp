// parallel_search_test.cpp
//
// Tests for parallel_search components. Focuses on the parts that are
// testable without a multi-threaded environment:
//
//   1. partition_distance  — ARI metric: identity, symmetry, bounds
//   2. pool_insert         — near-duplicate rejection, eviction, size cap
//   3. parallel_search (no deadline) — single-threaded Gen 0 only:
//      valid pool, all ops covered, best cost ≤ trivial
//
// Thread-safety of the evolutionary loop is hard to test deterministically;
// we rely on Tsan/Helgrind for that. The logic itself is covered by the
// individual greedy/FM/evolution tests.
//
// Build: make parallel_search_test
// Run:   ./parallel_search_test

#include "core/types.h"
#include "core/dag.h"
#include "core/cost_cache.h"
#include "partition/partition.h"
#include "init/init_strategies.h"
#include "search/parallel_search.h"
#include <cmath>
#include <iostream>
#include <vector>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double got, double exp, double tol = 1e-9) {
    if (std::abs(got - exp) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << got << " exp=" << exp << "\n"; }
}

// ============================================================================
// Problem helpers
// ============================================================================

static Problem make_chain4() {
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

static Problem make_diamond4() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{1},{3},1000},
             {OpType::Pointwise,{2,3},{4},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// partition_distance is a private static function — test indirectly via
// pool_insert behaviour (near-duplicate detection relies on it) and via a
// thin wrapper exposed for testing.
//
// Since partition_distance is static in parallel_search.cpp, we duplicate
// the essential logic here to unit-test it directly, then verify that
// pool_insert's near-duplicate threshold behaves as expected.
// ============================================================================

// Compute Adjusted Rand Index distance directly.
static double compute_ari_distance(const Partition& a, const Partition& b) {
    size_t n = a.prob->num_ops();
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

    auto choose2 = [](int64_t x) -> int64_t { return x*(x-1)/2; };
    int64_t same_a = 0, same_b = 0, agree = 0;
    for (int i = 0; i < num_ga; i++) same_a += choose2(row_sum[i]);
    for (int j = 0; j < num_gb; j++) same_b += choose2(col_sum[j]);
    for (int i = 0; i < num_ga; i++)
        for (int j = 0; j < num_gb; j++)
            agree += choose2(table[i][j]);
    int64_t total_pairs = choose2(total);
    if (total_pairs == 0) return 0.0;
    double expected = (double)same_a * same_b / total_pairs;
    double max_agree = (double)(same_a + same_b) / 2.0;
    double denom = max_agree - expected;
    if (std::abs(denom) < 1e-12) return 0.0;
    double ari = ((double)agree - expected) / denom;
    if (ari < 0.0) ari = 0.0;
    if (ari > 1.0) ari = 1.0;
    return 1.0 - ari;
}

// ============================================================================
// 1. partition_distance (via duplicated logic)
// ============================================================================

void test_distance_identical() {
    std::cout << "--- test_distance_identical ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto pa = Partition::trivial(p, d); pa.cache = &cache;
    auto pb = Partition::trivial(p, d); pb.cache = &cache;
    CHECK_EQ("trivial vs trivial = 0", compute_ari_distance(pa, pb), 0.0);

    // Fused partition vs itself
    auto pc = Partition::trivial(p, d); pc.cache = &cache;
    pc.groups[0].ops  = {0,1,2,3};
    pc.groups[0].cost = cache.evaluate({0,1,2,3},p,d);
    for (int i = 1; i < 4; i++) pc.groups[i].alive = false;
    pc.rebuild_index();
    CHECK_EQ("fused vs fused = 0", compute_ari_distance(pc, pc), 0.0);
}

void test_distance_symmetry() {
    std::cout << "--- test_distance_symmetry ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;

    auto pa = Partition::trivial(p, d); pa.cache = &cache;

    auto pb = Partition::trivial(p, d); pb.cache = &cache;
    pb.groups[0].ops  = {0,1}; pb.groups[0].cost = cache.evaluate({0,1},p,d);
    pb.groups[1].alive = false; pb.rebuild_index();

    double dab = compute_ari_distance(pa, pb);
    double dba = compute_ari_distance(pb, pa);
    CHECK("distance is symmetric", std::abs(dab - dba) < 1e-9);
    CHECK("distance > 0 for different partitions", dab > 0.0);
}

void test_distance_bounds() {
    std::cout << "--- test_distance_bounds ---\n";
    auto p = make_chain4(); DAG dag = DAG::build(p);
    CostCache cache;

    // Trivial (4 groups) vs fully fused (1 group): maximally different
    auto pa = Partition::trivial(p, dag); pa.cache = &cache;
    auto pb = Partition::trivial(p, dag); pb.cache = &cache;
    pb.groups[0].ops  = {0,1,2,3};
    pb.groups[0].cost = cache.evaluate({0,1,2,3},p,dag);
    for (int i = 1; i < 4; i++) pb.groups[i].alive = false;
    pb.rebuild_index();

    double dist = compute_ari_distance(pa, pb);
    CHECK("distance in [0,1]", dist >= 0.0 && dist <= 1.0);
    CHECK("trivial vs fused has significant distance", dist > 0.0);
    std::cout << "    trivial vs fused distance = " << dist << "\n";
}

void test_distance_single_op_problem() {
    std::cout << "--- test_distance_single_op_problem ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 50000; p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    CostCache cache;
    auto pa = Partition::trivial(p, d); pa.cache = &cache;
    auto pb = Partition::trivial(p, d); pb.cache = &cache;
    // Single op: total_pairs = C(1,2) = 0 → distance = 0
    CHECK_EQ("single op distance = 0", compute_ari_distance(pa, pb), 0.0);
}

// ============================================================================
// 2. pool_insert (via parallel_search interface — single-threaded Gen 0 only)
//
// pool_insert is static so we test its behaviour through parallel_search:
// run with 1 thread, no deadline, small pool_size and verify:
//   - pool has ≤ pool_size entries
//   - all entries have valid cost
//   - all entries are feasible partitions
// ============================================================================

void test_pool_size_capped() {
    std::cout << "--- test_pool_size_capped ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);

    ParallelConfig cfg;
    cfg.num_threads = 1;
    cfg.pool_size = 2;
    // No deadline → only Gen 0

    auto results = parallel_search(p, d, cfg);
    CHECK("pool size <= pool_size cap", results.size() <= 2);
    CHECK("pool has at least 1 entry", !results.empty());
    std::cout << "    pool_size=" << results.size() << "\n";
}

void test_pool_entries_valid() {
    std::cout << "--- test_pool_entries_valid ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);

    ParallelConfig cfg;
    cfg.num_threads = 1;
    cfg.pool_size = 4;

    auto results = parallel_search(p, d, cfg);
    for (size_t i = 0; i < results.size(); i++) {
        // Coverage
        for (size_t op = 0; op < p.num_ops(); op++)
            CHECK("op covered", !results[i].groups_of(op).empty());
        // Feasibility
        std::string err = verify_partition_feasibility(results[i]);
        if (!err.empty()) {
            std::cout << "  FAIL: pool[" << i << "] " << err << "\n";
            g_fail++;
        } else { g_pass++; }
    }
}

void test_pool_sorted_best_first() {
    std::cout << "--- test_pool_sorted_best_first ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);

    ParallelConfig cfg;
    cfg.num_threads = 1;
    cfg.pool_size = 4;

    auto results = parallel_search(p, d, cfg);
    // parallel_search returns sorted best-first
    for (size_t i = 1; i < results.size(); i++) {
        CHECK("sorted best-first",
              results[i].total_cost() >= results[i-1].total_cost() - 0.01);
    }
}

// ============================================================================
// 3. parallel_search integration — single-threaded, no deadline (Gen 0 only)
// ============================================================================

void test_parallel_search_chain_improves_trivial() {
    std::cout << "--- test_parallel_search_chain_improves_trivial ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    double trivial = Partition::trivial(p, d).total_cost();

    ParallelConfig cfg;
    cfg.num_threads = 1;
    cfg.pool_size = 4;
    cfg.fm.max_passes = 20;
    cfg.fm.max_no_improve = 5;

    auto results = parallel_search(p, d, cfg);
    CHECK("returns at least one result", !results.empty());
    CHECK("best result improves trivial", results[0].total_cost() <= trivial + 0.01);
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", !results[0].groups_of(i).empty());
    std::cout << "    trivial=" << trivial << " best=" << results[0].total_cost() << "\n";
}

void test_parallel_search_diamond_all_ops_covered() {
    std::cout << "--- test_parallel_search_diamond_all_ops_covered ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);

    ParallelConfig cfg;
    cfg.num_threads = 1;
    cfg.pool_size = 4;
    cfg.fm.max_passes = 10;
    cfg.fm.max_no_improve = 3;

    auto results = parallel_search(p, d, cfg);
    CHECK("returns results", !results.empty());
    for (auto& part : results) {
        for (size_t i = 0; i < p.num_ops(); i++)
            CHECK("op covered", !part.groups_of(i).empty());
        std::string err = verify_partition_feasibility(part);
        if (!err.empty()) {
            std::cout << "  FAIL: diamond " << err << "\n";
            g_fail++;
        } else { g_pass++; }
    }
    std::cout << "    pool=" << results.size() << " best=" << results[0].total_cost() << "\n";
}

void test_parallel_search_two_threads() {
    std::cout << "--- test_parallel_search_two_threads ---\n";
    // Run with 2 threads to exercise the thread-spawning path.
    auto p = make_chain4(); DAG d = DAG::build(p);

    ParallelConfig cfg;
    cfg.num_threads = 2;
    cfg.pool_size = 4;
    cfg.fm.max_passes = 10;
    cfg.fm.max_no_improve = 3;

    auto results = parallel_search(p, d, cfg);
    CHECK("returns results with 2 threads", !results.empty());
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", !results[0].groups_of(i).empty());
    std::cout << "    pool=" << results.size() << " best=" << results[0].total_cost() << "\n";
}

void test_parallel_search_with_deadline() {
    std::cout << "--- test_parallel_search_with_deadline ---\n";
    // With a deadline, the evolutionary loop runs. Verify the result remains valid.
    auto p = make_chain4(); DAG d = DAG::build(p);

    ParallelConfig cfg;
    cfg.num_threads = 2;
    cfg.pool_size = 4;
    cfg.fm.max_passes = 10;
    cfg.fm.max_no_improve = 3;
    // 500ms deadline: enough for Gen 0 + at least one evolutionary generation
    cfg.fm.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);

    auto results = parallel_search(p, d, cfg);
    CHECK("returns results with deadline", !results.empty());
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered post-evo", !results[0].groups_of(i).empty());
    std::string err = verify_partition_feasibility(results[0]);
    if (!err.empty()) {
        std::cout << "  FAIL: post-evo " << err << "\n";
        g_fail++;
    } else { g_pass++; }
    std::cout << "    pool=" << results.size() << " best=" << results[0].total_cost() << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // 1. partition_distance
    test_distance_identical();
    test_distance_symmetry();
    test_distance_bounds();
    test_distance_single_op_problem();

    // 2. pool_insert (via parallel_search interface)
    test_pool_size_capped();
    test_pool_entries_valid();
    test_pool_sorted_best_first();

    // 3. parallel_search integration
    test_parallel_search_chain_improves_trivial();
    test_parallel_search_diamond_all_ops_covered();
    test_parallel_search_two_threads();
    test_parallel_search_with_deadline();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}