// init_strategies_test.cpp
// Tests for partition initialization strategies.
//
// Each strategy must satisfy invariants:
//   1. Every op is covered by at least one group
//   2. Each group forms a valid Subgraph (connected, valid boundary outputs)
//   3. Group costs match eval_set()
//   4. Total cost <= trivial partition cost (no strategy should be worse)

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "partition/partition.h"
#include "init/init_strategies.h"
#include <cmath>
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++; else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double g, double e, double t = 0.1) {
    if (std::abs(g - e) < t) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}

// ==================== Test problems ====================

// Chain: T0 -> Op0 -> T1 -> Op1 -> T2 -> Op2 -> T3
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

// Diamond: T0->Op0->T1->{Op1->T2, Op2(T1,T2)->T3}
static Problem make_diamond() {
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

// Fork: T0->{Op0->T1, Op1->T2}, T1->Op2->T3, T2->Op3->T4
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

// MatMul chain: T0@T1->T2->T2@T3->T4 (tests K dimension handling)
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

// ==================== Generic invariant checks ====================

static void verify_partition(const char* strategy_name, const Partition& part,
                              const Problem& prob, const DAG& dag) {
    // Invariant 1: every op covered
    for (size_t i = 0; i < prob.num_ops(); i++) {
        auto gs = part.groups_of(i);
        if (gs.empty()) {
            std::cout << "  FAIL: " << strategy_name << " Op" << i << " uncovered\n";
            g_fail++;
        } else {
            g_pass++;
        }
    }

    // Invariant 2: each alive group is a valid Subgraph
    for (size_t i = 0; i < part.groups.size(); i++) {
        if (!part.groups[i].alive) continue;
        auto sg = Subgraph::create(prob, dag, {part.groups[i].ops.begin(),
                                               part.groups[i].ops.end()});
        if (!sg) {
            std::cout << "  FAIL: " << strategy_name << " G" << i << " invalid subgraph\n";
            g_fail++;
        } else {
            g_pass++;
        }
    }

    // Invariant 3: group costs are correct
    for (size_t i = 0; i < part.groups.size(); i++) {
        if (!part.groups[i].alive) continue;
        double expected = part.eval_set(part.groups[i].ops);
        if (std::abs(part.groups[i].cost - expected) > 0.01) {
            std::cout << "  FAIL: " << strategy_name << " G" << i
                      << " cost=" << part.groups[i].cost << " expected=" << expected << "\n";
            g_fail++;
        } else {
            g_pass++;
        }
    }
}

// ==================== Per-strategy tests ====================

void test_trivial_strategy() {
    std::cout << "--- test_trivial_strategy ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = init_trivial(p, d);

    verify_partition("trivial", part, p, d);
    CHECK("one group per op", part.num_alive() == p.num_ops());
}

void test_chain_then_edge_chain() {
    std::cout << "--- test_chain_then_edge_chain ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto trivial = init_trivial(p, d);
    auto part = init_chain_then_edge(p, d);

    verify_partition("chain+edge", part, p, d);
    // Chain of 3 should fuse into 1 group
    CHECK("single group", part.num_alive() == 1);
    CHECK("cost <= trivial", part.total_cost() <= trivial.total_cost() + 0.01);
}

void test_chain_then_edge_diamond() {
    std::cout << "--- test_chain_then_edge_diamond ---\n";
    // Diamond has no pure chain (Op0 has 2 successors), but edge greedy can fuse
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto trivial = init_trivial(p, d);
    auto part = init_chain_then_edge(p, d);

    verify_partition("chain+edge diamond", part, p, d);
    CHECK("cost <= trivial", part.total_cost() <= trivial.total_cost() + 0.01);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

void test_chain_then_edge_fork() {
    std::cout << "--- test_chain_then_edge_fork ---\n";
    // Fork: Op0 and Op1 both read T0 but produce different outputs
    // Chains: Op0->Op2 (unique succ/pred), Op1->Op3 (unique succ/pred)
    auto p = make_fork(); DAG d = DAG::build(p);
    auto trivial = init_trivial(p, d);
    auto part = init_chain_then_edge(p, d);

    verify_partition("chain+edge fork", part, p, d);
    CHECK("cost <= trivial", part.total_cost() <= trivial.total_cost() + 0.01);
    // Should detect the two chains: {Op0,Op2} and {Op1,Op3}
    CHECK("at most 2 groups", part.num_alive() <= 2);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

void test_seed_and_grow_chain() {
    std::cout << "--- test_seed_and_grow_chain ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto trivial = init_trivial(p, d);
    auto part = init_seed_and_grow(p, d);

    verify_partition("seed+grow", part, p, d);
    CHECK("cost <= trivial", part.total_cost() <= trivial.total_cost() + 0.01);
}

void test_seed_and_grow_fork() {
    std::cout << "--- test_seed_and_grow_fork ---\n";
    // Seed should start from Op1 (cost=2000), then Op3 (cost=1500)
    auto p = make_fork(); DAG d = DAG::build(p);
    auto trivial = init_trivial(p, d);
    auto part = init_seed_and_grow(p, d);

    verify_partition("seed+grow fork", part, p, d);
    CHECK("cost <= trivial", part.total_cost() <= trivial.total_cost() + 0.01);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

void test_reverse_topo_chain() {
    std::cout << "--- test_reverse_topo_chain ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto trivial = init_trivial(p, d);
    auto part = init_reverse_topo(p, d);

    verify_partition("rev-topo", part, p, d);
    CHECK("cost <= trivial", part.total_cost() <= trivial.total_cost() + 0.01);
}

void test_reverse_topo_matmul() {
    std::cout << "--- test_reverse_topo_matmul ---\n";
    auto p = make_matmul_chain(); DAG d = DAG::build(p);
    auto trivial = init_trivial(p, d);
    auto part = init_reverse_topo(p, d);

    verify_partition("rev-topo matmul", part, p, d);
    CHECK("cost <= trivial", part.total_cost() <= trivial.total_cost() + 0.01);
    std::cout << "    groups=" << part.num_alive() << " cost=" << part.total_cost() << "\n";
}

// ==================== best_initial ====================

void test_best_initial_picks_best() {
    std::cout << "--- test_best_initial_picks_best ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);

    // Run all strategies individually
    auto strategies = all_init_strategies();
    double min_cost = 1e18;
    for (auto& s : strategies) {
        auto part = s.init(p, d);
        if (part.total_cost() < min_cost)
            min_cost = part.total_cost();
    }

    // best_initial should match
    auto best = best_initial(p, d);
    CHECK_EQ("best matches minimum", best.total_cost(), min_cost);
    verify_partition("best_initial", best, p, d);
}

void test_best_initial_diamond() {
    std::cout << "--- test_best_initial_diamond ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto trivial = init_trivial(p, d);
    auto best = best_initial(p, d);

    verify_partition("best_initial diamond", best, p, d);
    CHECK("best <= trivial", best.total_cost() <= trivial.total_cost() + 0.01);
}

// ==================== Strategy registry ====================

void test_all_strategies_registered() {
    std::cout << "--- test_all_strategies_registered ---\n";
    auto strategies = all_init_strategies();
    CHECK("at least 4 strategies", strategies.size() >= 4);

    // Each strategy must have a non-empty name and callable init
    for (auto& s : strategies) {
        CHECK("has name", !s.name.empty());
        // Verify callable by running on tiny problem
        Problem p;
        p.tensors = {{128,128},{128,128}};
        p.ops = {{OpType::Pointwise,{0},{1},1000}};
        p.fast_memory_capacity = 50000;
        p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128;
        DAG d = DAG::build(p);
        auto part = s.init(p, d);
        CHECK("produces valid partition", part.num_alive() >= 1);
    }
}

// ==================== Main ====================

int main() {
    test_trivial_strategy();
    test_chain_then_edge_chain();
    test_chain_then_edge_diamond();
    test_chain_then_edge_fork();
    test_seed_and_grow_chain();
    test_seed_and_grow_fork();
    test_reverse_topo_chain();
    test_reverse_topo_matmul();
    test_best_initial_picks_best();
    test_best_initial_diamond();
    test_all_strategies_registered();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}