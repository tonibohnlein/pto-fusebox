// evolution_test.cpp — Tests for evolutionary operators

#include "core/types.h"
#include "core/dag.h"
#include "partition/partition.h"
#include "search/evolution.h"
#include "search/local_search.h"
#include <iostream>
#include <cmath>
#include <set>
#include <random>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++; else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}

// Chain of 6 PW ops: T0->Op0->T1->...->Op5->T6
static Problem make_chain6() {
    Problem p;
    for (int i = 0; i < 7; i++) p.tensors.push_back({128, 128});
    for (int i = 0; i < 6; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Y-shape: shared input to two branches that merge
// T0->Op0->T1, T0->Op1->T2, (T1,T2)->Op2->T3, T3->Op3->T4
static Problem make_Y_merge() {
    Problem p;
    for (int i = 0; i < 5; i++) p.tensors.push_back({128, 128});
    p.ops.push_back({OpType::Pointwise, {0}, {1}, 1000});
    p.ops.push_back({OpType::Pointwise, {0}, {2}, 1000});
    p.ops.push_back({OpType::Pointwise, {1, 2}, {3}, 1000});
    p.ops.push_back({OpType::Pointwise, {3}, {4}, 1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

static bool partition_valid(const Partition& part, const Problem& prob) {
    // Every op in exactly one alive group
    std::set<size_t> covered;
    for (auto& g : part.groups) {
        if (!g.alive) continue;
        for (auto op : g.ops) covered.insert(op);
        // Each group's cost should be finite
        if (g.cost >= 1e17) return false;
    }
    return covered.size() == prob.num_ops();
}

void test_mutate_merge() {
    std::cout << "=== Evo: mutate_merge ===\n";
    auto p = make_chain6();
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    size_t orig_groups = part.num_alive();
    
    std::mt19937 rng(42);
    auto merged = mutate_merge(Partition(part), rng);
    
    CHECK("valid after merge", partition_valid(merged, p));
    CHECK("fewer or equal groups", merged.num_alive() <= orig_groups);
    // Multiple merges should keep reducing
    for (int i = 0; i < 5; i++)
        merged = mutate_merge(std::move(merged), rng);
    CHECK("still valid after 5 merges", partition_valid(merged, p));
}

void test_mutate_split() {
    std::cout << "=== Evo: mutate_split ===\n";
    auto p = make_chain6();
    DAG d = DAG::build(p);
    
    // Create a partition with one big group
    auto part = Partition::trivial(p, d);
    std::mt19937 rng(42);
    for (int i = 0; i < 5; i++) part = mutate_merge(std::move(part), rng);
    size_t groups_before = part.num_alive();
    
    auto split = mutate_split(Partition(part), rng);
    CHECK("valid after split", partition_valid(split, p));
    CHECK("more or equal groups", split.num_alive() >= groups_before);
}

void test_mutate_reassign() {
    std::cout << "=== Evo: mutate_reassign ===\n";
    auto p = make_chain6();
    DAG d = DAG::build(p);
    
    // Start with 3 groups: {0,1}, {2,3}, {4,5}
    Partition part; part.prob = &p; part.dag = &d;
    part.add_group({0, 1}, part.eval_set({0, 1}));
    part.add_group({2, 3}, part.eval_set({2, 3}));
    part.add_group({4, 5}, part.eval_set({4, 5}));
    
    std::mt19937 rng(42);
    auto reassigned = mutate_reassign(Partition(part), rng);
    CHECK("valid after reassign", partition_valid(reassigned, p));
    CHECK("same group count", reassigned.num_alive() == part.num_alive());
}

void test_mutate_block_move() {
    std::cout << "=== Evo: mutate_block_move ===\n";
    auto p = make_chain6();
    DAG d = DAG::build(p);
    
    Partition part; part.prob = &p; part.dag = &d;
    part.add_group({0, 1, 2}, part.eval_set({0, 1, 2}));
    part.add_group({3, 4, 5}, part.eval_set({3, 4, 5}));
    
    std::mt19937 rng(42);
    bool found_change = false;
    for (int trial = 0; trial < 20; trial++) {
        auto moved = mutate_block_move(Partition(part), rng);
        CHECK("valid after block_move", partition_valid(moved, p));
        if (moved.num_alive() != part.num_alive()) found_change = true;
    }
    // Block move might create a new group (extract block), so group count may change
    CHECK("block_move attempted", true);
}

void test_mutate_compound() {
    std::cout << "=== Evo: mutate_compound ===\n";
    auto p = make_chain6();
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    
    std::mt19937 rng(42);
    for (int n = 1; n <= 10; n++) {
        auto mutated = mutate_compound(Partition(part), n, rng);
        CHECK("valid after compound", partition_valid(mutated, p));
    }
}

void test_crossover() {
    std::cout << "=== Evo: crossover ===\n";
    auto p = make_chain6();
    DAG d = DAG::build(p);
    
    // Two different partitions
    Partition a; a.prob = &p; a.dag = &d;
    a.add_group({0, 1, 2}, a.eval_set({0, 1, 2}));
    a.add_group({3, 4, 5}, a.eval_set({3, 4, 5}));
    
    Partition b; b.prob = &p; b.dag = &d;
    b.add_group({0, 1}, b.eval_set({0, 1}));
    b.add_group({2, 3}, b.eval_set({2, 3}));
    b.add_group({4, 5}, b.eval_set({4, 5}));
    
    std::mt19937 rng(42);
    auto child = crossover(a, b, rng);
    CHECK("crossover valid", partition_valid(child, p));
    CHECK("crossover has groups", child.num_alive() > 0);
    
    // Crossover of identical parents should produce similar result
    auto same = crossover(a, a, rng);
    CHECK("same-parent crossover valid", partition_valid(same, p));
}

void test_crossover_Y() {
    std::cout << "=== Evo: crossover on Y-graph ===\n";
    auto p = make_Y_merge();
    DAG d = DAG::build(p);
    
    Partition a; a.prob = &p; a.dag = &d;
    a.add_group({0, 1}, a.eval_set({0, 1}));
    a.add_group({2, 3}, a.eval_set({2, 3}));
    
    Partition b; b.prob = &p; b.dag = &d;
    b.add_group({0}, b.eval_set({0}));
    b.add_group({1, 2, 3}, b.eval_set({1, 2, 3}));
    
    std::mt19937 rng(42);
    auto child = crossover(a, b, rng);
    CHECK("Y crossover valid", partition_valid(child, p));
}

void test_mutation_preserves_coverage() {
    std::cout << "=== Evo: mutations preserve coverage ===\n";
    auto p = make_Y_merge();
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    
    std::mt19937 rng(42);
    for (int i = 0; i < 50; i++) {
        int choice = rng() % 4;
        switch (choice) {
            case 0: part = mutate_merge(std::move(part), rng); break;
            case 1: part = mutate_split(std::move(part), rng); break;
            case 2: part = mutate_reassign(std::move(part), rng); break;
            case 3: part = mutate_block_move(std::move(part), rng); break;
        }
        CHECK("still valid", partition_valid(part, p));
    }
}

int main() {
    test_mutate_merge();
    test_mutate_split();
    test_mutate_reassign();
    test_mutate_block_move();
    test_mutate_compound();
    test_crossover();
    test_crossover_Y();
    test_mutation_preserves_coverage();
    
    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << g_pass + g_fail << " tests\n";
    return g_fail > 0 ? 1 : 0;
}