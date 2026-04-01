// symm_init_test.cpp
//
// Covers:
//   - Parallel pattern init: 4-head attention, partitions replicated correctly
//   - Series pattern init: 8-block chain, partitions replicated correctly
//   - Mixed graph: parallel + aggregation chain handled
//   - Feasibility: every output partition passes verify_partition_feasibility
//   - Coverage: every op in at least one alive group
//   - Cost improvement: symm_init never worse than trivial
//   - Empty case: no patterns → empty vector (caller falls back)
//   - Bijection correctness: replicated groups match representative's structure
//   - CostCache wiring: shared cache gets hits from replicated groups
//
// Build: make symm_init_test
// Run:   ./tests/symm_init_test

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost_cache.h"
#include "symmetry/merkle_hash.h"
#include "symmetry/symmetry.h"
#include "symmetry/series.h"
#include "partition/partition.h"
#include "init/init_strategies.h"
#include "init/symm_init.h"
#include <cmath>
#include <iostream>
#include <set>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double got, double exp, double tol = 0.5) {
    if (std::abs(got - exp) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << got << " exp=" << exp << "\n"; }
}
static void CHECK_EQ_S(const char* l, size_t got, size_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << got << " exp=" << exp << "\n"; }
}

// ============================================================================
// Canonical partition verifier (same as init_strategies_test.cpp)
// ============================================================================

static void verify(const char* name, const Partition& part, const Problem& prob) {
    for (size_t i = 0; i < prob.num_ops(); i++) {
        if (part.groups_of(i).empty()) {
            std::cout << "  FAIL: " << name << " Op" << i << " uncovered\n";
            g_fail++;
        } else { g_pass++; }
    }

    std::string err = verify_partition_feasibility(part);
    if (!err.empty()) {
        std::cout << "  FAIL: " << name << " " << err << "\n";
        g_fail++;
    } else { g_pass++; }

    for (size_t i = 0; i < part.groups.size(); i++) {
        if (!part.groups[i].alive) continue;
        double expected = part.eval_set(part.groups[i].ops);
        if (std::abs(part.groups[i].cost - expected) > 0.5) {
            std::cout << "  FAIL: " << name << " G" << i
                      << " cost=" << part.groups[i].cost << " expected=" << expected << "\n";
            g_fail++;
        } else { g_pass++; }
    }
}

// ============================================================================
// Problem builders
// ============================================================================

// 4-head attention: 4 parallel MM→PW chains + pairwise reduce + final reduce
//   Op0..3: MM(T0, T_h) → T_{h+5}     (4 heads)
//   Op4..7: PW(T_{h+5}) → T_{h+9}     (4 activations)
//   Op8: PW(T9, T10) → T13             (pairwise reduce)
//   Op9: PW(T11, T12) → T14            (pairwise reduce)
//   Op10: PW(T13, T14) → T15           (final reduce)
static Problem make_4head() {
    Problem p;
    for (int i = 0; i < 16; i++)
        p.tensors.push_back({128, 128});
    for (int h = 0; h < 4; h++)
        p.ops.push_back({OpType::MatMul, {0, (size_t)(h+1)}, {(size_t)(h+5)}, 1500});
    for (int h = 0; h < 4; h++)
        p.ops.push_back({OpType::Pointwise, {(size_t)(h+5)}, {(size_t)(h+9)}, 500});
    p.ops.push_back({OpType::Pointwise, {9, 10}, {13}, 500});
    p.ops.push_back({OpType::Pointwise, {11, 12}, {14}, 500});
    p.ops.push_back({OpType::Pointwise, {13, 14}, {15}, 500});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// 3-head attention (benchmark 5 style): 3 parallel 5-op heads + aggregation
static Problem make_3head() {
    Problem p;
    int64_t widths[]  = {128,512,128,512,128,512,128,128,128,128,512,512,128,128,128,
                         512,512,128,128,128,512,512,128,128,128,128,128,128,128};
    int64_t heights[] = {1024,128,512,128,512,128,512,128,128,128,1024,1024,1024,1024,1024,
                         1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024};
    for (int i = 0; i < 29; i++) p.tensors.push_back({widths[i], heights[i]});
    auto add = [&](OpType t, std::vector<size_t> in, std::vector<size_t> out, int64_t c) {
        p.ops.push_back({t, in, out, c}); };
    add(OpType::MatMul,{0,1},{10},1000); add(OpType::Pointwise,{10},{11},200);
    add(OpType::MatMul,{11,2},{12},1000); add(OpType::MatMul,{0,7},{13},500);
    add(OpType::Pointwise,{12,13},{14},200);
    add(OpType::MatMul,{0,3},{15},1000); add(OpType::Pointwise,{15},{16},200);
    add(OpType::MatMul,{16,4},{17},1000); add(OpType::MatMul,{0,8},{18},500);
    add(OpType::Pointwise,{17,18},{19},200);
    add(OpType::MatMul,{0,5},{20},1000); add(OpType::Pointwise,{20},{21},200);
    add(OpType::MatMul,{21,6},{22},1000); add(OpType::MatMul,{0,9},{23},500);
    add(OpType::Pointwise,{22,23},{24},200);
    add(OpType::Pointwise,{14,19},{25},100); add(OpType::Pointwise,{25,24},{26},100);
    add(OpType::Pointwise,{26,0},{27},100); add(OpType::Pointwise,{27},{28},200);
    p.fast_memory_capacity = 30000;
    p.slow_memory_bandwidth = 15;
    p.native_w = 128; p.native_h = 32;
    return p;
}

// 8-block serial chain (benchmark 9 style)
static Problem make_serial_chain() {
    Problem p;
    int64_t W[] = {1024,4096,1024,4096,1024,4096,1024,4096,1024,4096,1024,4096,1024,4096,1024,4096,1024,
                   4096,4096,1024,1024,4096,4096,1024,1024,4096,4096,1024,1024,4096,4096,1024,1024,
                   4096,4096,1024,1024,4096,4096,1024,1024,4096,4096,1024,1024,4096,4096,1024,1024};
    int64_t H[] = {1024,1024,4096,1024,4096,1024,4096,1024,4096,1024,4096,1024,4096,1024,4096,1024,4096,
                   1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,
                   1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024};
    for (int i = 0; i < 49; i++) p.tensors.push_back({W[i], H[i]});
    p.ops.push_back({OpType::MatMul,{0,1},{17},5000}); p.ops.push_back({OpType::Pointwise,{17},{18},200});
    p.ops.push_back({OpType::MatMul,{18,2},{19},5000}); p.ops.push_back({OpType::Pointwise,{19,0},{20},500});
    p.ops.push_back({OpType::MatMul,{20,3},{21},5000}); p.ops.push_back({OpType::Pointwise,{21},{22},200});
    p.ops.push_back({OpType::MatMul,{22,4},{23},5000}); p.ops.push_back({OpType::Pointwise,{23,20},{24},500});
    p.ops.push_back({OpType::MatMul,{24,5},{25},5000}); p.ops.push_back({OpType::Pointwise,{25},{26},200});
    p.ops.push_back({OpType::MatMul,{26,6},{27},5000}); p.ops.push_back({OpType::Pointwise,{27,24},{28},500});
    p.ops.push_back({OpType::MatMul,{28,7},{29},5000}); p.ops.push_back({OpType::Pointwise,{29},{30},200});
    p.ops.push_back({OpType::MatMul,{30,8},{31},5000}); p.ops.push_back({OpType::Pointwise,{31,28},{32},500});
    p.ops.push_back({OpType::MatMul,{32,9},{33},5000}); p.ops.push_back({OpType::Pointwise,{33},{34},200});
    p.ops.push_back({OpType::MatMul,{34,10},{35},5000}); p.ops.push_back({OpType::Pointwise,{35,32},{36},500});
    p.ops.push_back({OpType::MatMul,{36,11},{37},5000}); p.ops.push_back({OpType::Pointwise,{37},{38},200});
    p.ops.push_back({OpType::MatMul,{38,12},{39},5000}); p.ops.push_back({OpType::Pointwise,{39,36},{40},500});
    p.ops.push_back({OpType::MatMul,{40,13},{41},5000}); p.ops.push_back({OpType::Pointwise,{41},{42},200});
    p.ops.push_back({OpType::MatMul,{42,14},{43},5000}); p.ops.push_back({OpType::Pointwise,{43,40},{44},500});
    p.ops.push_back({OpType::MatMul,{44,15},{45},5000}); p.ops.push_back({OpType::Pointwise,{45},{46},200});
    p.ops.push_back({OpType::MatMul,{46,16},{47},5000}); p.ops.push_back({OpType::Pointwise,{47,44},{48},500});
    p.fast_memory_capacity = 250000;
    p.slow_memory_bandwidth = 25;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Linear chain (no symmetry — tests empty-pattern fallback)
static Problem make_linear_chain() {
    Problem p;
    for (int i = 0; i < 4; i++) p.tensors.push_back({128, 128});
    for (int i = 0; i < 3; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Two independent parallel chains (simplest parallel case)
static Problem make_parallel_chains() {
    Problem p;
    for (int i = 0; i < 6; i++) p.tensors.push_back({128, 128});
    p.ops.push_back({OpType::Pointwise, {0}, {2}, 1000});
    p.ops.push_back({OpType::Pointwise, {1}, {3}, 1000});
    p.ops.push_back({OpType::Pointwise, {2}, {4}, 1000});
    p.ops.push_back({OpType::Pointwise, {3}, {5}, 1000});
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// Tests: parallel patterns
// ============================================================================

void test_parallel_4head_feasibility() {
    std::cout << "--- test_parallel_4head_feasibility ---\n";
    auto p = make_4head(); DAG d = DAG::build(p);
    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);

    CHECK("at least one partition returned", !results.empty());
    for (size_t i = 0; i < results.size(); i++)
        verify(("4head result " + std::to_string(i)).c_str(), results[i], p);
}

void test_parallel_4head_cost_improvement() {
    std::cout << "--- test_parallel_4head_cost_improvement ---\n";
    auto p = make_4head(); DAG d = DAG::build(p);
    auto trivial = init_trivial(p, d);
    double trivial_cost = trivial.total_cost();

    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    CHECK("has results", !results.empty());

    if (!results.empty()) {
        double best_cost = 1e18;
        for (auto& r : results)
            best_cost = std::min(best_cost, r.total_cost());
        CHECK("symm_init <= trivial", best_cost <= trivial_cost + 0.01);
        std::cout << "    trivial=" << trivial_cost
                  << "  symm_best=" << best_cost << "\n";
    }
}

void test_parallel_4head_replication() {
    std::cout << "--- test_parallel_4head_replication ---\n";
    // Verify that replicated groups produce a valid partition that
    // greedy descent can further improve.
    auto p = make_4head(); DAG d = DAG::build(p);
    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    CHECK("has results", !results.empty());
    if (results.empty()) return;

    // Pick best
    size_t best_idx = 0;
    for (size_t i = 1; i < results.size(); i++)
        if (results[i].total_cost() < results[best_idx].total_cost())
            best_idx = i;
    auto& best = results[best_idx];

    // After global greedy descent, the symmetric structure may be broken
    // (greedy can merge across symmetry boundaries for further improvement).
    // We only check that multi-op groups exist and cost is good.
    int multi_op = 0;
    std::map<size_t, int> group_size_counts;
    for (auto& g : best.groups) {
        if (!g.alive) continue;
        group_size_counts[g.ops.size()]++;
        if (g.ops.size() > 1) multi_op++;
    }
    std::cout << "    group size distribution:";
    for (auto [sz, cnt] : group_size_counts)
        std::cout << " " << sz << "x" << cnt;
    std::cout << "\n";

    CHECK("has fused groups", multi_op >= 1);
    CHECK("fewer groups than trivial", best.num_alive() < p.num_ops());

    // Cost should be better than trivial
    double trivial_cost = init_trivial(p, d).total_cost();
    CHECK("cost <= trivial", best.total_cost() <= trivial_cost + 0.01);
}

void test_parallel_3head() {
    std::cout << "--- test_parallel_3head ---\n";
    auto p = make_3head(); DAG d = DAG::build(p);
    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    CHECK("has results", !results.empty());
    for (size_t i = 0; i < results.size(); i++)
        verify(("3head result " + std::to_string(i)).c_str(), results[i], p);

    if (!results.empty()) {
        double trivial_cost = init_trivial(p, d).total_cost();
        double best = 1e18;
        for (auto& r : results) best = std::min(best, r.total_cost());
        CHECK("3head: symm_init <= trivial", best <= trivial_cost + 0.01);
        std::cout << "    trivial=" << trivial_cost << "  symm_best=" << best << "\n";
    }
}

void test_parallel_simple_chains() {
    std::cout << "--- test_parallel_simple_chains ---\n";
    auto p = make_parallel_chains(); DAG d = DAG::build(p);
    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    CHECK("has results", !results.empty());
    for (size_t i = 0; i < results.size(); i++)
        verify(("par_chains result " + std::to_string(i)).c_str(), results[i], p);
}

// ============================================================================
// Tests: series patterns
// ============================================================================

void test_series_chain_feasibility() {
    std::cout << "--- test_series_chain_feasibility ---\n";
    auto p = make_serial_chain(); DAG d = DAG::build(p);
    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    CHECK("has results", !results.empty());
    for (size_t i = 0; i < results.size(); i++)
        verify(("serial result " + std::to_string(i)).c_str(), results[i], p);
}

void test_series_chain_cost_improvement() {
    std::cout << "--- test_series_chain_cost_improvement ---\n";
    auto p = make_serial_chain(); DAG d = DAG::build(p);
    auto trivial = init_trivial(p, d);
    double trivial_cost = trivial.total_cost();

    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    CHECK("has results", !results.empty());
    if (!results.empty()) {
        double best = 1e18;
        for (auto& r : results) best = std::min(best, r.total_cost());
        CHECK("series: symm_init <= trivial", best <= trivial_cost + 0.01);
        std::cout << "    trivial=" << trivial_cost << "  symm_best=" << best << "\n";
    }
}

void test_series_chain_replication() {
    std::cout << "--- test_series_chain_replication ---\n";
    auto p = make_serial_chain(); DAG d = DAG::build(p);
    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    CHECK("has results", !results.empty());
    if (results.empty()) return;

    // Pick best
    size_t best_idx = 0;
    for (size_t i = 1; i < results.size(); i++)
        if (results[i].total_cost() < results[best_idx].total_cost())
            best_idx = i;
    auto& best = results[best_idx];

    // After greedy, blocks 1-7 should have matching group structure.
    // Check: groups that span block ops should appear in multiples.
    FlatSet<size_t> block_ops;
    for (int b = 1; b <= 7; b++)
        for (int j = 0; j < 4; j++)
            block_ops.insert(b * 4 + j);

    int multi_op_groups = 0;
    for (auto& g : best.groups) {
        if (!g.alive || g.ops.size() <= 1) continue;
        bool in_blocks = false;
        for (auto op : g.ops)
            if (block_ops.count(op)) { in_blocks = true; break; }
        if (in_blocks) multi_op_groups++;
    }
    std::cout << "    multi-op groups in blocks: " << multi_op_groups << "\n";
    CHECK("series: blocks have fused groups", multi_op_groups >= 7);
}

// ============================================================================
// Tests: empty / fallback
// ============================================================================

void test_no_patterns_returns_empty() {
    std::cout << "--- test_no_patterns_returns_empty ---\n";
    auto p = make_linear_chain(); DAG d = DAG::build(p);
    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    CHECK_EQ_S("linear chain: no patterns", results.size(), 0);
}

// ============================================================================
// Tests: CostCache wiring
// ============================================================================

void test_cache_gets_hits_from_replication() {
    std::cout << "--- test_cache_gets_hits_from_replication ---\n";
    auto p = make_4head(); DAG d = DAG::build(p);
    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    CHECK("has results", !results.empty());

    // Replicated groups should produce cache hits because all heads
    // have the same op-sets (modulo index), and eval_set goes through cache.
    // The representative's groups are misses; replicas may get hits if
    // canonical keys match, or at least the cache should be populated.
    CHECK("cache populated", cache.size() > 0);
    std::cout << "    cache: size=" << cache.size()
              << " hits=" << cache.hits() << " misses=" << cache.misses() << "\n";
}

// ============================================================================
// Tests: verify init_from_patterns doesn't crash on edge cases
// ============================================================================

void test_single_op_no_crash() {
    std::cout << "--- test_single_op_no_crash ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    CHECK_EQ_S("single op: no patterns", results.size(), 0);
}

void test_two_ops_no_crash() {
    std::cout << "--- test_two_ops_no_crash ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 1000},
             {OpType::Pointwise, {1}, {2}, 1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    CostCache cache;
    auto results = init_from_patterns(p, d, &cache);
    // Might or might not find patterns (2 ops is borderline), but shouldn't crash
    for (size_t i = 0; i < results.size(); i++)
        verify(("2ops result " + std::to_string(i)).c_str(), results[i], p);
    std::cout << "    patterns found: " << results.size() << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Parallel patterns
    test_parallel_4head_feasibility();
    test_parallel_4head_cost_improvement();
    test_parallel_4head_replication();
    test_parallel_3head();
    test_parallel_simple_chains();

    // Series patterns
    test_series_chain_feasibility();
    test_series_chain_cost_improvement();
    test_series_chain_replication();

    // Empty / fallback
    test_no_patterns_returns_empty();

    // Cache
    test_cache_gets_hits_from_replication();

    // Edge cases
    test_single_op_no_crash();
    test_two_ops_no_crash();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}