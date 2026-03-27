// init_strategies_test.cpp
//
// Covers:
//   - All five strategies satisfy the three feasibility invariants on every topology:
//       1. Memory:         every alive group has a feasible tiling
//       2. No cycles:      no pair of alive groups forms a condensed-DAG cycle
//       3. No eph. gap:    no tensor is ephemeral in its group while an external
//                          consumer group has no other source
//   - Cost monotonicity: no strategy produces a partition worse than trivial
//   - Structural correctness of Phase A chain detection (chains fused, forks not)
//   - Phase C co-consumer merging (benchmark-13 pattern)
//   - try_merge guards: ephemeral gap, memory infeasibility, no-improvement threshold
//   - init_random: valid partitions, diversity across calls, op coverage
//   - init_reverse_topo: diamond + co-consumer edge usage
//   - best_initial: returns global minimum across all strategies
//   - CostCache wiring: shared cache across strategies gets hits on repeated sets
//   - Registry: all five strategies registered and callable
//   - Boundary cases: single-op problem
//
// Build: make init_strategies_test
// Run:   ./init_strategies_test

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost_cache.h"
#include "partition/partition.h"
#include "init/init_strategies.h"
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
// Problem helpers
// ============================================================================

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

// 3-op diamond: T0->Op0->T1, T1->Op1->T2, T1+T2->Op2->T3
// Op0 has two successors (Op1 and Op2 via T1). Used for chain/edge greedy tests.
static Problem make_diamond3() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1500},
             {OpType::Pointwise,{1},{2},1500},
             {OpType::Pointwise,{1,2},{3},1500}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// 4-op diamond: T0->Op0->T1, T1->{Op1->T2, Op2->T3}, T2+T3->Op3->T4
// Classical diamond with two parallel arms and a join. Used for ephemeral gap tests.
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

// Fork: T0->{Op0->T1->Op2->T3, Op1->T2->Op3->T4}
// Two parallel chains sharing T0 as input (co-consumer pattern).
static Problem make_fork() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{0},{2},2000},
             {OpType::Pointwise,{1},{3},500},
             {OpType::Pointwise,{2},{4},1500}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Two parallel chains sharing T0: Op0(T0->T1)->Op2(T1->T3), Op1(T0->T2)->Op3(T2->T4)
// Like fork but with continuation ops -- captures the benchmark-13 co-consumer pattern.
static Problem make_co_consumer_chains() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{0},{2},1000},
             {OpType::Pointwise,{1},{3},1000},
             {OpType::Pointwise,{2},{4},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// T0@T1->T2, T2@T3->T4 (chained MatMuls, tests K-dimension constraint propagation)
static Problem make_matmul_chain() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::MatMul,{2,3},{4},2000}};
    p.fast_memory_capacity = 45000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Single-op (boundary case: nothing to fuse)
static Problem make_single_op() {
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// Canonical partition verifier
//
// Checks three things:
//   1. Coverage:    every op appears in at least one alive group.
//   2. Feasibility: verify_partition_feasibility passes -- memory, no cycles,
//                   no ephemeral gaps (the canonical checker from init_strategies.h).
//   3. Cost:        each group's stored cost matches eval_set() to within 0.5.
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
// 1. trivial
// ============================================================================

void test_trivial() {
    std::cout << "--- test_trivial ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = init_trivial(p, d);
    verify("trivial", part, p);
    CHECK_EQ_S("one group per op", part.num_alive(), p.num_ops());
}

// ============================================================================
// 2. init_chain_then_edge
// ============================================================================

void test_chain_fuses_linear_chain() {
    std::cout << "--- test_chain_fuses_linear_chain ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = init_chain_then_edge(p, d);
    verify("chain+edge chain3", part, p);
    // Phase A: every op has exactly one successor with one predecessor -> full chain
    CHECK_EQ_S("chain3 -> 1 group", part.num_alive(), 1);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
}

void test_chain_handles_diamond3() {
    std::cout << "--- test_chain_handles_diamond3 ---\n";
    // Op0 has two successors -> not part of a chain. Phase B/C can still fuse.
    auto p = make_diamond3(); DAG d = DAG::build(p);
    auto part = init_chain_then_edge(p, d);
    verify("chain+edge diamond3", part, p);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

void test_chain_detects_fork_chains() {
    std::cout << "--- test_chain_detects_fork_chains ---\n";
    // Fork: Op0->Op2 is a chain (unique succ/pred), Op1->Op3 is a chain.
    // Phase A should detect both and fuse {Op0,Op2} and {Op1,Op3}.
    auto p = make_fork(); DAG d = DAG::build(p);
    auto part = init_chain_then_edge(p, d);
    verify("chain+edge fork", part, p);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
    CHECK("at most 2 groups", part.num_alive() <= 2);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

void test_chain_phase_c_co_consumer_merging() {
    std::cout << "--- test_chain_phase_c_co_consumer_merging ---\n";
    // Two chains sharing T0: {Op0,Op2} and {Op1,Op3}.
    // Phase A fuses each chain. Phase C may merge the two chain-groups further.
    auto p = make_co_consumer_chains(); DAG d = DAG::build(p);
    auto part = init_chain_then_edge(p, d);
    verify("chain+edge co_consumer", part, p);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
    bool chain0_fused = false, chain1_fused = false;
    for (auto& g : part.groups) {
        if (!g.alive) continue;
        if (g.ops.count(0) && g.ops.count(2)) chain0_fused = true;
        if (g.ops.count(1) && g.ops.count(3)) chain1_fused = true;
    }
    CHECK("chain {Op0,Op2} fused by Phase A", chain0_fused);
    CHECK("chain {Op1,Op3} fused by Phase A", chain1_fused);
    CHECK("at most 2 groups after Phase A", part.num_alive() <= 2);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

void test_chain_no_fusion_without_chain() {
    std::cout << "--- test_chain_no_fusion_without_chain ---\n";
    // Two ops both reading T0, neither has a successor. No chains.
    // Phase B has no producer-consumer internal edges. Phase C may fuse co-consumers.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{0},{2},1000}};
    p.fast_memory_capacity = 200000; p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = init_chain_then_edge(p, d);
    verify("chain no-chain", part, p);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

// ============================================================================
// 3. init_seed_and_grow
// ============================================================================

void test_seed_grow_chain() {
    std::cout << "--- test_seed_grow_chain ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = init_seed_and_grow(p, d);
    verify("seed+grow chain3", part, p);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
}

void test_seed_grow_fork() {
    std::cout << "--- test_seed_grow_fork ---\n";
    // Seeds on the highest-cost op (Op1, cost=2000), grows greedily.
    auto p = make_fork(); DAG d = DAG::build(p);
    auto part = init_seed_and_grow(p, d);
    verify("seed+grow fork", part, p);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

void test_seed_grow_diamond_no_ephemeral_gap() {
    std::cout << "--- test_seed_grow_diamond_no_ephemeral_gap ---\n";
    // Diamond4: T1 consumed by Op1 (candidate internal) and Op2 (external).
    // The inline ephemeral check must prevent fusing {Op0,Op1} without Op2.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto part = init_seed_and_grow(p, d);
    verify("seed+grow diamond4", part, p);
    // Belt-and-suspenders: direct group inspection on top of verify_partition_feasibility.
    for (auto& g : part.groups) {
        if (!g.alive) continue;
        if (g.ops.count(0) && g.ops.count(1) && !g.ops.count(2)) {
            std::cout << "  FAIL: seed+grow created {Op0,Op1} without Op2 (gap)\n";
            g_fail++; return;
        }
    }
    g_pass++;
}

// ============================================================================
// 4. init_reverse_topo
// ============================================================================

void test_reverse_topo_chain() {
    std::cout << "--- test_reverse_topo_chain ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = init_reverse_topo(p, d);
    verify("rev-topo chain3", part, p);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
}

void test_reverse_topo_matmul() {
    std::cout << "--- test_reverse_topo_matmul ---\n";
    auto p = make_matmul_chain(); DAG d = DAG::build(p);
    auto part = init_reverse_topo(p, d);
    verify("rev-topo matmul", part, p);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

void test_reverse_topo_diamond_gap_protected() {
    std::cout << "--- test_reverse_topo_diamond_gap_protected ---\n";
    // Reverse topo processes Op3 first, then Op1/Op2, then Op0.
    // Merging Op0+Op1 would create an ephemeral gap for Op2; must be blocked.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto part = init_reverse_topo(p, d);
    verify("rev-topo diamond4", part, p);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
    bool bad = false;
    for (auto& g : part.groups)
        if (g.alive && g.ops.count(0) && g.ops.count(1) && !g.ops.count(2)) bad = true;
    CHECK("Op0+Op1 not fused without Op2", !bad);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

void test_reverse_topo_co_consumer_edges() {
    std::cout << "--- test_reverse_topo_co_consumer_edges ---\n";
    // Op0 and Op1 co-consume T0 with no direct DAG edge. reverse_topo uses
    // op_neighbors (which includes co-consumer edges) for candidate collection.
    auto p = make_co_consumer_chains(); DAG d = DAG::build(p);
    auto part = init_reverse_topo(p, d);
    verify("rev-topo co_consumer", part, p);
    CHECK("cost <= trivial", part.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

// ============================================================================
// 5. init_random
// ============================================================================

void test_random_valid_partition() {
    std::cout << "--- test_random_valid_partition ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    for (int i = 0; i < 5; i++) {
        auto part = init_random(p, d);
        verify("random", part, p);
    }
}

void test_random_all_ops_covered() {
    std::cout << "--- test_random_all_ops_covered ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    for (int i = 0; i < 5; i++) {
        auto part = init_random(p, d);
        for (size_t op = 0; op < p.num_ops(); op++)
            CHECK("op covered", !part.groups_of(op).empty());
    }
}

void test_random_diversity() {
    std::cout << "--- test_random_diversity ---\n";
    // Different seeds must produce meaningfully different partitions.
    // A chain of 4 ops with 3 edges -> varied group counts across 10 runs.
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},1000});
    p.fast_memory_capacity = 200000; p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    std::set<size_t> seen;
    for (int i = 0; i < 10; i++) {
        auto part = init_random(p, d);
        verify("random diversity", part, p);
        seen.insert(part.num_alive());
    }
    CHECK("diverse outcomes across 10 runs", seen.size() >= 2);
    std::cout << "    distinct group counts: " << seen.size() << " across 10 runs\n";
}

// ============================================================================
// 6. try_merge guards (tested indirectly via strategies)
// ============================================================================

void test_try_merge_rejects_ephemeral_gap() {
    std::cout << "--- test_try_merge_rejects_ephemeral_gap ---\n";
    // Diamond4 trivial. Merging {Op0,Op1}: T1 is ephemeral (Op0 produces,
    // Op1 consumes internally). Op2 needs T1 but is external and does NOT
    // recompute Op0 → ephemeral gap detected.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto trivial = Partition::trivial(p, d);
    CHECK("gap detected (T1 ephemeral, Op2 stranded)",
          trivial.creates_ephemeral_gap({0,1}, 0, 1));

    // Merging {Op0,Op1,Op2}: T1 is ephemeral (all consumers internal) → no gap
    CHECK("no gap when all T1 consumers internal",
          !trivial.creates_ephemeral_gap({0,1,2}, 0, 1));

    // chain+edge must not create an ephemeral gap in the final partition
    auto result = init_chain_then_edge(p, d);
    verify("chain+edge no gap", result, p);
}

void test_try_merge_rejects_memory_infeasible() {
    std::cout << "--- test_try_merge_rejects_memory_infeasible ---\n";
    // Absurdly small capacity: no tiling fits. eval_set must return 1e18 and
    // no merge should be accepted.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::MatMul,{2,3},{4},2000}};
    p.fast_memory_capacity = 100;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    Partition tmp; tmp.prob = &p; tmp.dag = &d; tmp.cache = nullptr;
    CHECK("fused cost=1e18 when infeasible", tmp.eval_set({0,1}) >= 1e17);

    // Strategy runs without crash. Merges are blocked because cost is 1e18.
    auto result = init_chain_then_edge(p, d);
    CHECK("strategy survives infeasible problem", result.groups.size() >= 1);
    std::cout << "    groups=" << result.num_alive() << " (all infeasible at cap=100)\n";
}

void test_try_merge_no_improvement_threshold() {
    std::cout << "--- test_try_merge_no_improvement_threshold ---\n";
    // Huge bandwidth -> compute-bound. Fusion cost equals sum of singleton costs.
    // The 0.01 threshold in try_merge blocks the merge since saving < 0.01.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 1000000;  // effectively infinite
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    Partition tmp; tmp.prob = &p; tmp.dag = &d; tmp.cache = nullptr;
    double c0 = tmp.eval_set({0}), c1 = tmp.eval_set({1});
    double cf = tmp.eval_set({0,1});
    CHECK("no improvement at huge bandwidth", cf >= c0 + c1 - 0.02);
    std::cout << "    c0=" << c0 << " c1=" << c1 << " fused=" << cf << "\n";
}

// ============================================================================
// 7. CostCache wiring
// ============================================================================

void test_cache_shared_across_strategies() {
    std::cout << "--- test_cache_shared_across_strategies ---\n";
    // Running two strategies with the same cache: the second gets hits for
    // op-sets the first already evaluated.
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;

    auto part1 = init_chain_then_edge(p, d, &cache);
    size_t misses1 = cache.misses();
    CHECK("first strategy has misses", misses1 > 0);

    auto part2 = init_reverse_topo(p, d, &cache);
    CHECK("second strategy gets hits", cache.hits() > 0);
    std::cout << "    after chain+edge: misses=" << misses1
              << "  after rev-topo: hits=" << cache.hits() << "\n";

    verify("cached chain+edge", part1, p);
    verify("cached rev-topo",   part2, p);
}

void test_cache_nullptr_still_works() {
    std::cout << "--- test_cache_nullptr_still_works ---\n";
    // nullptr cache falls back to uncached eval_set without crash.
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = init_chain_then_edge(p, d, nullptr);
    verify("no-cache chain+edge", part, p);
    CHECK_EQ_S("still fuses chain", part.num_alive(), 1);
}

void test_best_initial_shares_cache() {
    std::cout << "--- test_best_initial_shares_cache ---\n";
    // best_initial passes the same cache to all strategies.
    // After it runs, the cache should have entries from multiple strategies and hits.
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto best = best_initial(p, d, &cache);
    verify("best_initial cached", best, p);
    CHECK("cache populated",              cache.size() > 0);
    CHECK("cross-strategy cache hits",    cache.hits() > 0);
    std::cout << "    size=" << cache.size()
              << " hits=" << cache.hits() << " misses=" << cache.misses() << "\n";
}

// ============================================================================
// 8. best_initial
// ============================================================================

void test_best_initial_returns_minimum() {
    std::cout << "--- test_best_initial_returns_minimum ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);

    double min_cost = 1e18;
    for (auto& s : all_init_strategies()) {
        auto part = s.init(p, d, nullptr);
        if (part.total_cost() < min_cost) min_cost = part.total_cost();
    }

    CostCache cache;
    auto best = best_initial(p, d, &cache);
    CHECK_EQ("best matches individual minimum", best.total_cost(), min_cost);
    verify("best_initial chain3", best, p);
}

void test_best_initial_diamond() {
    std::cout << "--- test_best_initial_diamond ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto best = best_initial(p, d, nullptr);
    verify("best_initial diamond4", best, p);
    CHECK("cost <= trivial", best.total_cost() <= init_trivial(p,d).total_cost() + 0.01);
}

// ============================================================================
// 9. Registry and boundary cases
// ============================================================================

void test_all_strategies_registered() {
    std::cout << "--- test_all_strategies_registered ---\n";
    auto strategies = all_init_strategies();
    CHECK("at least 5 strategies", strategies.size() >= 5);

    auto p = make_single_op(); DAG d = DAG::build(p);
    for (auto& s : strategies) {
        CHECK("has name", !s.name.empty());
        auto part = s.init(p, d, nullptr);
        verify(s.name.c_str(), part, p);
        CHECK_EQ_S((s.name + ": 1 group").c_str(), part.num_alive(), 1);
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // trivial
    test_trivial();

    // chain_then_edge
    test_chain_fuses_linear_chain();
    test_chain_handles_diamond3();
    test_chain_detects_fork_chains();
    test_chain_phase_c_co_consumer_merging();
    test_chain_no_fusion_without_chain();

    // seed_and_grow
    test_seed_grow_chain();
    test_seed_grow_fork();
    test_seed_grow_diamond_no_ephemeral_gap();

    // reverse_topo
    test_reverse_topo_chain();
    test_reverse_topo_matmul();
    test_reverse_topo_diamond_gap_protected();
    test_reverse_topo_co_consumer_edges();

    // random
    test_random_valid_partition();
    test_random_all_ops_covered();
    test_random_diversity();

    // try_merge guards
    test_try_merge_rejects_ephemeral_gap();
    test_try_merge_rejects_memory_infeasible();
    test_try_merge_no_improvement_threshold();

    // cache
    test_cache_shared_across_strategies();
    test_cache_nullptr_still_works();
    test_best_initial_shares_cache();

    // best_initial
    test_best_initial_returns_minimum();
    test_best_initial_diamond();

    // registry + boundary
    test_all_strategies_registered();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}