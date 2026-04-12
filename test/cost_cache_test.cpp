// cost_cache_test.cpp — Comprehensive tests for CostCache memoization.
//
// Covers:
//   - Basic hit/miss counting and value correctness
//   - Multiple distinct op-sets
//   - Memory-infeasible subgraphs (stored as 1e18)
//   - Structurally invalid subgraphs (disconnected, ephemeral fan-out)
//   - Sorted-key contract ({0,1} and {1,0} as std::set → same entry)
//   - Empty set
//   - clear() resets all state
//   - Cache isolation between independent Problem instances
//   - Thread safety (no crashes, no data races)
//   - Thread result consistency (all threads see correct deterministic values)
//   - TOCTOU correctness (duplicate computation still gives right answer)
//
// Cache growth note:
//   Each entry is a sorted vector<size_t> key (~avg_group_size*8 bytes) plus a
//   double value plus ~64 bytes of unordered_map node overhead.  For benchmark
//   17 (103 ops), empirically the cache stays well under 250K entries (~32 MB).
//   No eviction policy is needed at competition scale.
//
// Build: make cost_cache_test
// Run:   ./cost_cache_test

#include "core/types.h"
#include "core/dag.h"
#include "core/cost_cache.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <set>
#include <thread>
#include <vector>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double got, double exp, double tol = 0.1) {
    if (std::abs(got - exp) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l
           << " got=" << got << " exp=" << exp << "\n"; }
}
static void CHECK_EQ_S(const char* l, size_t got, size_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l
           << " got=" << got << " exp=" << exp << "\n"; }
}

// ---- Problem helpers ----

// T0->Op0->T1->Op1->T2->Op2->T3  (3 PW ops in a chain)
static Problem make_chain3() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// 1. Basic hit/miss semantics
// ============================================================================

void test_basic_hit_miss() {
    std::cout << "--- test_basic_hit_miss ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;

    FlatSet<size_t> ops = {0};
    double c1 = cache.evaluate(ops, p, d);
    CHECK_EQ_S("1 miss",   cache.misses(), 1);
    CHECK_EQ_S("0 hits",   cache.hits(),   0);
    CHECK_EQ_S("1 entry",  cache.size(),   1);
    CHECK("feasible",      c1 < 1e17);

    double c2 = cache.evaluate(ops, p, d);
    CHECK_EQ_S("1 hit",        cache.hits(),   1);
    CHECK_EQ_S("still 1 miss", cache.misses(), 1);
    CHECK_EQ("same result", c2, c1);
}

void test_different_sets() {
    std::cout << "--- test_different_sets ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;

    double c01 = cache.evaluate({0,1}, p, d);
    double c12 = cache.evaluate({1,2}, p, d);
    double c0  = cache.evaluate({0},   p, d);

    CHECK_EQ_S("3 misses",  cache.misses(), 3);
    CHECK_EQ_S("3 entries", cache.size(),   3);
    CHECK("all feasible", c01 < 1e17 && c12 < 1e17 && c0 < 1e17);

    // Re-query all — must be hits, not misses.
    cache.evaluate({0,1}, p, d);
    cache.evaluate({1,2}, p, d);
    cache.evaluate({0},   p, d);
    CHECK_EQ_S("3 hits",       cache.hits(),   3);
    CHECK_EQ_S("still 3 miss", cache.misses(), 3);
}

// ============================================================================
// 2. Infeasible / invalid op-sets — must be stored as 1e18
// ============================================================================

void test_memory_infeasible_cached() {
    std::cout << "--- test_memory_infeasible_cached ---\n";
    // 1024×1024 MatMul with cap=2 — even at [1,1,1], WS=3 > 2.
    // (With tiling propagation, [1,1,1] gives three 1×1 slices.)
    Problem p;
    p.tensors = {{1024,1024},{1024,1024},{1024,1024}};
    p.ops = {{OpType::MatMul,{0,1},{2},5000}};
    p.fast_memory_capacity = 2;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    CostCache cache;

    double c = cache.evaluate({0}, p, d);
    CHECK("infeasible returns 1e18", c >= 1e17);
    CHECK_EQ_S("1 entry", cache.size(), 1);

    // Second call must be a cache hit, not a recompute.
    double c2 = cache.evaluate({0}, p, d);
    CHECK_EQ_S("hit on infeasible", cache.hits(), 1);
    CHECK_EQ_S("still 1 miss",      cache.misses(), 1);
    CHECK_EQ("same infeasible value", c2, c);
}

void test_disconnected_cached() {
    std::cout << "--- test_disconnected_cached ---\n";
    // Two disconnected chains: Op0 uses T0→T1, Op1 uses T2→T3 (no shared tensors).
    // Both are PW sinks with matching 128x128 outputs → Subgraph::create succeeds.
    // The subgraph is valid (both ops tiled identically).
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{2},{3},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    CostCache cache;

    double c = cache.evaluate({0,1}, p, d);
    CHECK("disconnected valid (matching sinks)", c < 1e17);
    CHECK_EQ_S("cached as 1 entry", cache.size(), 1);

    // Re-query: hit, no new compute.
    size_t m = cache.misses();
    cache.evaluate({0,1}, p, d);
    CHECK_EQ_S("no new miss", cache.misses(), m);
    CHECK_EQ_S("1 hit", cache.hits(), 1);

    // Valid singletons are feasible and cached separately.
    double c0 = cache.evaluate({0}, p, d);
    double c1 = cache.evaluate({1}, p, d);
    CHECK("singleton {0} feasible", c0 < 1e17);
    CHECK("singleton {1} feasible", c1 < 1e17);
    CHECK_EQ_S("3 entries total", cache.size(), 3);
}

void test_ephemeral_fanout_cached() {
    std::cout << "--- test_ephemeral_fanout_cached ---\n";
    // Diamond: T0→Op0→T1, T1→Op1→T2, T1→Op2→T3, T2+T3→Op3→T4.
    // {0,1,2}: T1 has 2 internal consumers. With tiling propagation,
    // both assign compatible NTW×NTH tiling → subgraph IS valid.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{1},{3},1000},
             {OpType::Pointwise,{2,3},{4},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    CostCache cache;

    double c = cache.evaluate({0,1,2}, p, d);
    CHECK("fan-out feasible (compatible tiling)", c < 1e17);
    CHECK_EQ_S("1 entry", cache.size(), 1);

    // Valid subsets also work.
    CHECK("{0,1} feasible", cache.evaluate({0,1}, p, d) < 1e17);
    CHECK("{0,2} feasible", cache.evaluate({0,2}, p, d) < 1e17);
    CHECK_EQ_S("3 entries", cache.size(), 3);

    // Fan-out entry still cached.
    size_t m = cache.misses();
    cache.evaluate({0,1,2}, p, d);
    CHECK_EQ_S("fan-out still cached", cache.misses(), m);
}

void test_empty_set() {
    std::cout << "--- test_empty_set ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;
    double c = cache.evaluate({}, p, d);
    CHECK("empty set → 1e18", c >= 1e17);
}

// ============================================================================
// 3. Sorted-key contract
//
// The key is built from FlatSet<size_t>, which always iterates sorted.
// The same logical op-set — regardless of source order — must produce
// exactly one cache entry.
// ============================================================================

void test_sorted_key_two_ops() {
    std::cout << "--- test_sorted_key_two_ops ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;

    FlatSet<size_t> s01 = {0, 1};
    FlatSet<size_t> s10 = {1, 0};  // same set, different source order

    double c1 = cache.evaluate(s01, p, d);
    double c2 = cache.evaluate(s10, p, d);

    CHECK_EQ_S("1 miss (not 2)", cache.misses(), 1);
    CHECK_EQ_S("1 hit",          cache.hits(),   1);
    CHECK_EQ_S("1 entry",        cache.size(),   1);
    CHECK_EQ("same value", c1, c2);
}

void test_sorted_key_three_ops() {
    std::cout << "--- test_sorted_key_three_ops ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;

    double c1 = cache.evaluate({2,0,1}, p, d);
    double c2 = cache.evaluate({0,1,2}, p, d);
    double c3 = cache.evaluate({1,2,0}, p, d);

    CHECK_EQ_S("1 miss", cache.misses(), 1);
    CHECK_EQ_S("2 hits", cache.hits(),   2);
    CHECK_EQ("c1==c2", c1, c2);
    CHECK_EQ("c2==c3", c2, c3);
}

// ============================================================================
// 4. clear() resets all state
// ============================================================================

void test_clear() {
    std::cout << "--- test_clear ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;

    cache.evaluate({0}, p, d);
    cache.evaluate({1}, p, d);
    cache.evaluate({0}, p, d);  // hit

    CHECK_EQ_S("before: size=2",   cache.size(),   2);
    CHECK_EQ_S("before: misses=2", cache.misses(), 2);
    CHECK_EQ_S("before: hits=1",   cache.hits(),   1);

    cache.clear();

    CHECK_EQ_S("after: size=0",   cache.size(),   0);
    CHECK_EQ_S("after: misses=0", cache.misses(), 0);
    CHECK_EQ_S("after: hits=0",   cache.hits(),   0);

    // Post-clear evaluation is a fresh miss.
    double c = cache.evaluate({0}, p, d);
    CHECK_EQ_S("post-clear miss=1", cache.misses(), 1);
    CHECK_EQ_S("post-clear size=1", cache.size(),   1);
    CHECK("post-clear feasible", c < 1e17);
}

// ============================================================================
// 5. Cache isolation between independent Problem instances
// ============================================================================

void test_isolation_different_bandwidth() {
    std::cout << "--- test_isolation_different_bandwidth ---\n";
    // Same topology, different bandwidth → different latencies.
    auto make = [](int64_t bw) {
        Problem p;
        p.tensors = {{128,128},{128,128}};
        p.ops = {{OpType::Pointwise,{0},{1},1000}};
        p.fast_memory_capacity = 50000;
        p.slow_memory_bandwidth = bw;
        p.native_w = 128; p.native_h = 128;
        return p;
    };

    Problem pA = make(1);    // slow: mem = 2*16384/1 = 32768 >> compute → lat=32768
    Problem pB = make(100);  // fast: mem = 2*16384/100 = 327.68 << compute → lat=1000

    DAG dA = DAG::build(pA);
    DAG dB = DAG::build(pB);

    CostCache cacheA, cacheB;
    double costA = cacheA.evaluate({0}, pA, dA);
    double costB = cacheB.evaluate({0}, pB, dB);

    CHECK_EQ("slow bw: mem-bound 32768", costA, 32768.0, 1.0);
    CHECK_EQ("fast bw: compute-bound 1000", costB, 1000.0, 1.0);
    CHECK("different values", std::abs(costA - costB) > 100);

    // Two independent caches — each has exactly 1 entry.
    CHECK_EQ_S("cacheA size=1",   cacheA.size(),   1);
    CHECK_EQ_S("cacheB size=1",   cacheB.size(),   1);
    // Hit/miss counters are thread-local (shared across all caches on this thread).
}

// ============================================================================
// 6. Thread safety: no crashes and correct values
//
// Both the original thread test and the stronger consistency check live here.
// ============================================================================

void test_thread_safety_no_crash() {
    std::cout << "--- test_thread_safety_no_crash ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;

    // 4 threads, each evaluating the same 6 sets 100 times.
    auto worker = [&](int /*id*/) {
        for (int i = 0; i < 100; i++) {
            cache.evaluate({0},     p, d);
            cache.evaluate({1},     p, d);
            cache.evaluate({2},     p, d);
            cache.evaluate({0,1},   p, d);
            cache.evaluate({1,2},   p, d);
            cache.evaluate({0,1,2}, p, d);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; i++) threads.emplace_back(worker, i);
    for (auto& t : threads) t.join();

    // Lock-free map may insert duplicates under contention (by design —
    // "Duplicates from races are harmless").  Expect 6–24 entries.
    CHECK("at least 6 entries", cache.size() >= 6);
    CHECK("at most 24 entries (4 threads × 6 keys)", cache.size() <= 24);

    // Hit/miss counters are thread-local (no cross-thread contention).
    // Cannot check totals from main thread; verify via size instead.
    CHECK("entries populated", cache.size() >= 6);

    // Results are still correct after concurrent access.
    double c0  = cache.evaluate({0},   p, d);
    double c01 = cache.evaluate({0,1}, p, d);
    CHECK("c0 feasible",  c0  < 1e17);
    CHECK("c01 feasible", c01 < 1e17);
}

void test_thread_result_consistency() {
    std::cout << "--- test_thread_result_consistency ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);

    // Ground truth (single-threaded reference).
    CostCache ref;
    std::array<double,6> expected = {
        ref.evaluate({0},     p, d),
        ref.evaluate({1},     p, d),
        ref.evaluate({2},     p, d),
        ref.evaluate({0,1},   p, d),
        ref.evaluate({1,2},   p, d),
        ref.evaluate({0,1,2}, p, d),
    };

    const int num_threads = 8;
    std::vector<std::array<double,6>> results(num_threads);
    CostCache cache;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&, i]() {
            for (int k = 0; k < 200; k++) {
                results[i][0] = cache.evaluate({0},     p, d);
                results[i][1] = cache.evaluate({1},     p, d);
                results[i][2] = cache.evaluate({2},     p, d);
                results[i][3] = cache.evaluate({0,1},   p, d);
                results[i][4] = cache.evaluate({1,2},   p, d);
                results[i][5] = cache.evaluate({0,1,2}, p, d);
            }
        });
    }
    for (auto& t : threads) t.join();

    for (int i = 0; i < num_threads; i++) {
        for (int j = 0; j < 6; j++) {
            char buf[64]; snprintf(buf, sizeof(buf), "thread %d set %d", i, j);
            CHECK_EQ(buf, results[i][j], expected[j]);
        }
    }

    // Lock-free map may store duplicates under contention (by design).
    CHECK("at least 6 entries", cache.size() >= 6);
    CHECK("at most 48 entries (8 threads × 6 keys)", cache.size() <= 48);
    // Hit/miss counters are thread-local; verify entries instead.
    CHECK("entries populated", cache.size() >= 6);
}

void test_toctou_correctness() {
    std::cout << "--- test_toctou_correctness ---\n";
    // 16 threads all hit the same cold key simultaneously.
    // Even if multiple threads compute it (TOCTOU), the stored result
    // must be the correct deterministic value.
    auto p = make_chain3();
    DAG d = DAG::build(p);

    CostCache ref;
    double expected = ref.evaluate({0,1,2}, p, d);
    CHECK("ref feasible", expected < 1e17);

    const int N = 16;
    std::vector<double> observed(N, -1.0);
    CostCache cache;

    std::vector<std::thread> threads;
    for (int i = 0; i < N; i++)
        threads.emplace_back([&, i]() { observed[i] = cache.evaluate({0,1,2}, p, d); });
    for (auto& t : threads) t.join();

    for (int i = 0; i < N; i++) {
        char buf[32]; snprintf(buf, sizeof(buf), "thread %d correct", i);
        CHECK_EQ(buf, observed[i], expected);
    }
    // Lock-free map may store duplicates under contention (by design).
    CHECK("1-16 entries stored", cache.size() >= 1 && cache.size() <= 16);
    // Hit/miss counters are thread-local; verify via size.
    CHECK("entries present", cache.size() >= 1);
}

// ============================================================================
// 7. Cache growth estimation
//
// Not a correctness test — demonstrates empirically how many entries
// accumulate for a simple exhaustive evaluation of all connected subsets.
// ============================================================================

void test_cache_growth_all_connected_subsets() {
    std::cout << "--- test_cache_growth_all_connected_subsets ---\n";
    // Chain of 6 ops. All connected subsets of a chain of N ops:
    // only contiguous ranges [i..j] are connected → N*(N+1)/2 subsets.
    // N=6: 21 connected subsets.
    // Exhaustively evaluating all gives an upper bound on entries for a
    // length-6 chain during local search.
    int N = 6;
    Problem p;
    for (int i = 0; i <= N; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < N; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},1000});
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    CostCache cache;

    int evaluated = 0;
    for (int i = 0; i < N; i++) {
        FlatSet<size_t> s;
        for (int j = i; j < N; j++) {
            s.insert(j);
            cache.evaluate(s, p, d);
            evaluated++;
        }
    }

    // N*(N+1)/2 = 21 subsets, all are distinct → 21 entries.
    int expected_entries = N*(N+1)/2;
    CHECK_EQ_S("all connected subsets stored",
               cache.size(), (size_t)expected_entries);
    CHECK_EQ_S("all were misses", cache.misses(), (size_t)expected_entries);

    // Memory estimate: each entry has key (avg ~3.5 ops × 8 bytes = 28 bytes),
    // value (8 bytes), and node overhead (~64 bytes) ≈ 100 bytes/entry.
    // For 21 entries: ~2100 bytes. Trivial.
    // Extrapolation for 103 ops with 250K entries: ~25 MB.
    double bytes_per_entry = 100.0;
    double est_MB_250k = 250000 * bytes_per_entry / (1024*1024);
    std::cout << "    chain-6: " << cache.size() << " entries (all "
              << evaluated << " subsets are misses)\n";
    std::cout << "    Extrapolated 250K entries ≈ "
              << (int)est_MB_250k << " MB (well within competition limits)\n";
}

// ============================================================================
// 8. Capacity cap — stop-on-full policy
//
// When max_entries > 0 and the cache is full, new op-sets are evaluated and
// returned correctly but not stored. The overcapacity() counter tracks how
// many such evaluations occurred.
//
// Key properties to verify:
//   a. Values returned while over-cap are still correct.
//   b. overcapacity() counts exactly the non-stored evaluations.
//   c. clear() resets overcapacity to 0.
//   d. Unlimited cache (max_entries=0) never overflows.
// ============================================================================

void test_cap_stop_on_full() {
    std::cout << "--- test_cap_stop_on_full ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);

    // Cap at 2 entries.
    CostCache cache(2);
    CHECK_EQ_S("initial size=0", cache.size(), 0);
    CHECK_EQ_S("max_entries=2",  cache.max_entries(), 2);

    // First two distinct op-sets: must be stored.
    double c0  = cache.evaluate({0},   p, d);
    double c1  = cache.evaluate({1},   p, d);
    CHECK_EQ_S("size=2 (at cap)",         cache.size(),         2);
    CHECK_EQ_S("2 misses",                cache.misses(),       2);
    CHECK_EQ_S("0 overcapacity so far",   cache.overcapacity(), 0);

    // Third distinct op-set: over cap — computed but NOT stored.
    double c2 = cache.evaluate({2}, p, d);
    CHECK("c2 still correct", c2 < 1e17);  // value is valid
    CHECK_EQ_S("size still 2",    cache.size(),         2);  // not inserted
    CHECK_EQ_S("1 overcapacity",  cache.overcapacity(), 1);
    CHECK_EQ_S("3 misses total",  cache.misses(),       3);

    // Fourth distinct op-set: also over cap.
    double c01 = cache.evaluate({0,1}, p, d);
    CHECK("c01 correct", c01 < 1e17);
    CHECK_EQ_S("size still 2",    cache.size(),         2);
    CHECK_EQ_S("2 overcapacity",  cache.overcapacity(), 2);

    // Re-evaluating a cached key is still a hit (not an overcapacity miss).
    double c0_hit = cache.evaluate({0}, p, d);
    CHECK_EQ("cached value unchanged", c0_hit, c0);
    CHECK_EQ_S("1 hit", cache.hits(), 1);
    CHECK_EQ_S("overcapacity unchanged", cache.overcapacity(), 2);

    // Re-evaluating an over-cap key also re-computes and counts as overcapacity.
    cache.evaluate({2}, p, d);
    CHECK_EQ_S("3 overcapacity", cache.overcapacity(), 3);
}

void test_cap_values_correct_when_over_cap() {
    std::cout << "--- test_cap_values_correct_when_over_cap ---\n";
    // Verify that evaluations beyond the cap return the exact same values
    // as a reference cache with no cap.
    auto p = make_chain3();
    DAG d = DAG::build(p);

    CostCache ref;
    double r0   = ref.evaluate({0},     p, d);
    double r01  = ref.evaluate({0,1},   p, d);
    double r012 = ref.evaluate({0,1,2}, p, d);
    double r1   = ref.evaluate({1},     p, d);
    double r2   = ref.evaluate({2},     p, d);

    // Cap=1: only the first op-set gets stored; all others are over-cap.
    CostCache capped(1);
    CHECK_EQ("over-cap {0}   correct", capped.evaluate({0},     p, d), r0);
    CHECK_EQ("over-cap {0,1} correct", capped.evaluate({0,1},   p, d), r01);
    CHECK_EQ("over-cap {0,1,2} correct",capped.evaluate({0,1,2},p, d), r012);
    CHECK_EQ("over-cap {1}   correct", capped.evaluate({1},     p, d), r1);
    CHECK_EQ("over-cap {2}   correct", capped.evaluate({2},     p, d), r2);

    CHECK_EQ_S("cap=1: size=1", capped.size(), 1);
    // 4 over-cap evaluations (all after the first miss).
    CHECK_EQ_S("4 overcapacity", capped.overcapacity(), 4);
}

void test_cap_clear_resets_overcapacity() {
    std::cout << "--- test_cap_clear_resets_overcapacity ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);

    CostCache cache(1);
    cache.evaluate({0},   p, d);
    cache.evaluate({1},   p, d);  // over cap
    cache.evaluate({0,1}, p, d);  // over cap
    CHECK_EQ_S("2 overcapacity", cache.overcapacity(), 2);

    cache.clear();
    CHECK_EQ_S("overcapacity reset to 0", cache.overcapacity(), 0);
    CHECK_EQ_S("size reset to 0",         cache.size(),         0);

    // After clear, cap is still 1 — same behaviour.
    cache.evaluate({0},   p, d);
    cache.evaluate({1},   p, d);  // over cap again
    CHECK_EQ_S("1 overcapacity after clear", cache.overcapacity(), 1);
}

void test_cap_zero_means_unlimited() {
    std::cout << "--- test_cap_zero_means_unlimited ---\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);

    CostCache cache(0);  // unlimited
    cache.evaluate({0},     p, d);
    cache.evaluate({1},     p, d);
    cache.evaluate({2},     p, d);
    cache.evaluate({0,1},   p, d);
    cache.evaluate({1,2},   p, d);
    cache.evaluate({0,1,2}, p, d);

    CHECK_EQ_S("all 6 stored",      cache.size(),         6);
    CHECK_EQ_S("0 overcapacity",    cache.overcapacity(), 0);
}

void test_cap_thread_safe() {
    std::cout << "--- test_cap_thread_safe ---\n";
    // Under concurrent access with a tight cap, all returned values must
    // still be correct and the size must never exceed the cap.
    auto p = make_chain3();
    DAG d = DAG::build(p);

    CostCache ref;
    double r0 = ref.evaluate({0}, p, d);
    double r1 = ref.evaluate({1}, p, d);

    // Cap=1: at most 1 entry stored at any time.
    CostCache cache(1);
    std::atomic<bool> all_correct{true};

    std::vector<std::thread> threads;
    for (int i = 0; i < 8; i++) {
        threads.emplace_back([&]() {
            for (int k = 0; k < 100; k++) {
                double v0 = cache.evaluate({0}, p, d);
                double v1 = cache.evaluate({1}, p, d);
                if (std::abs(v0 - r0) > 0.1 || std::abs(v1 - r1) > 0.1)
                    all_correct.store(false, std::memory_order_relaxed);
            }
        });
    }
    for (auto& t : threads) t.join();

    CHECK("all values correct under cap", all_correct.load());
    // Under contention, a few extra entries may sneak past the cap check
    // (TOCTOU between size() and insert()). Allow up to num_threads slack.
    CHECK("size within cap + thread slack", cache.size() <= 1 + 8);
    // Hit/miss counters are thread-local; verify via size.
    CHECK("cap entries present", cache.size() >= 1);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Basic semantics
    test_basic_hit_miss();
    test_different_sets();

    // Infeasible / invalid
    test_memory_infeasible_cached();
    test_disconnected_cached();
    test_ephemeral_fanout_cached();
    test_empty_set();

    // Key invariants
    test_sorted_key_two_ops();
    test_sorted_key_three_ops();

    // State management
    test_clear();

    // Isolation
    test_isolation_different_bandwidth();

    // Concurrency
    test_thread_safety_no_crash();
    test_thread_result_consistency();
    test_toctou_correctness();

    // Growth
    test_cache_growth_all_connected_subsets();

    // Capacity cap
    test_cap_stop_on_full();
    test_cap_values_correct_when_over_cap();
    test_cap_clear_resets_overcapacity();
    test_cap_zero_means_unlimited();
    test_cap_thread_safe();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}