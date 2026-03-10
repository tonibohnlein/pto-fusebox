// cost_cache_test.cpp — Tests for CostCache memoization

#include "core/types.h"
#include "core/dag.h"
#include "core/cost_cache.h"
#include <iostream>
#include <cmath>
#include <set>
#include <thread>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++; else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double g, double e, double t = 0.1) {
    if (std::abs(g - e) < t) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}

// T0->Op0->T1->Op1->T2->Op2->T3
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

void test_cache_basic() {
    std::cout << "=== Cache: basic hit/miss ===\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;
    
    std::set<size_t> ops = {0};
    double c1 = cache.evaluate(ops, p, d);
    CHECK("miss count 1", cache.misses() == 1);
    CHECK("hit count 0", cache.hits() == 0);
    CHECK("size 1", cache.size() == 1);
    CHECK("feasible", c1 < 1e17);
    
    double c2 = cache.evaluate(ops, p, d);
    CHECK("hit count 1", cache.hits() == 1);
    CHECK("miss count still 1", cache.misses() == 1);
    CHECK_EQ("same result", c2, c1);
}

void test_cache_different_sets() {
    std::cout << "=== Cache: different op sets ===\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;
    
    double c_01 = cache.evaluate({0, 1}, p, d);
    double c_12 = cache.evaluate({1, 2}, p, d);
    double c_0 = cache.evaluate({0}, p, d);
    
    CHECK("3 misses", cache.misses() == 3);
    CHECK("3 entries", cache.size() == 3);
    CHECK("all feasible", c_01 < 1e17 && c_12 < 1e17 && c_0 < 1e17);
    
    // Re-query all — should be hits
    cache.evaluate({0, 1}, p, d);
    cache.evaluate({1, 2}, p, d);
    cache.evaluate({0}, p, d);
    CHECK("3 hits", cache.hits() == 3);
    CHECK("still 3 misses", cache.misses() == 3);
}

void test_cache_infeasible() {
    std::cout << "=== Cache: infeasible sets ===\n";
    Problem p;
    p.tensors = {{1024,1024},{1024,1024},{1024,1024}};
    p.ops = {{OpType::MatMul,{0,1},{2},5000}};
    p.fast_memory_capacity = 100;  // way too small
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    CostCache cache;
    
    double c = cache.evaluate({0}, p, d);
    CHECK("infeasible returns 1e18", c >= 1e17);
    
    // Should still cache the infeasible result
    double c2 = cache.evaluate({0}, p, d);
    CHECK("cached infeasible", cache.hits() == 1);
    CHECK_EQ("same infeasible result", c2, c);
}

void test_cache_thread_safety() {
    std::cout << "=== Cache: thread safety ===\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;
    
    // 4 threads all evaluate the same sets concurrently
    auto worker = [&](int id) {
        for (int i = 0; i < 100; i++) {
            cache.evaluate({0}, p, d);
            cache.evaluate({1}, p, d);
            cache.evaluate({2}, p, d);
            cache.evaluate({0, 1}, p, d);
            cache.evaluate({1, 2}, p, d);
            cache.evaluate({0, 1, 2}, p, d);
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; i++) threads.emplace_back(worker, i);
    for (auto& t : threads) t.join();
    
    CHECK("cache entries ≤ 6", cache.size() <= 6);
    CHECK("total queries = 2400", cache.hits() + cache.misses() == 2400);
    CHECK("many hits", cache.hits() > 2000);  // most should be hits
    
    // All results should be deterministic
    double c0 = cache.evaluate({0}, p, d);
    double c01 = cache.evaluate({0, 1}, p, d);
    CHECK("consistent after threads", c0 < 1e17 && c01 < 1e17);
}

void test_cache_empty_set() {
    std::cout << "=== Cache: empty set ===\n";
    auto p = make_chain3();
    DAG d = DAG::build(p);
    CostCache cache;
    
    double c = cache.evaluate({}, p, d);
    CHECK("empty set infeasible", c >= 1e17);
}

int main() {
    test_cache_basic();
    test_cache_different_sets();
    test_cache_infeasible();
    test_cache_thread_safety();
    test_cache_empty_set();
    
    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << g_pass + g_fail << " tests\n";
    return g_fail > 0 ? 1 : 0;
}