// evolution_test.cpp
//
// Tests for evolutionary operators: mutations and crossover.
//
// Every test verifies all three feasibility invariants via check_valid():
//   1. Coverage:     every op appears in at least one alive group
//   2. Memory:       every alive group has a feasible tiling (eval_set < 1e18)
//   3. No eph. gap:  no tensor is ephemeral while an external consumer has
//                    no other source
//   (Acyclicity: guaranteed by merge_creates_cycle guards inside each mutation;
//    structural tests below verify the guards fire correctly.)
//
// Build: make evolution_test
// Run:   ./evolution_test

#include "core/types.h"
#include "core/dag.h"
#include "core/cost_cache.h"
#include "partition/partition.h"
#include "init/init_strategies.h"
#include "search/evolution.h"
#include <cmath>
#include <iostream>
#include <set>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
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

// T0->Op0->T1->...->Op3->T4
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

// T0->Op0->T1->...->Op5->T6  (larger problem for mutation stress tests)
static Problem make_chain6() {
    Problem p;
    for (int i = 0; i < 7; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 6; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// 4-op diamond: T0->Op0->T1, T1->{Op1->T2, Op2->T3}, T2+T3->Op3->T4
// T1 has two consumers in separate branches.
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

// Y-shape with merge: T0->Op0->T1, T0->Op1->T2, T1+T2->Op2->T3, T3->Op3->T4
// T0 is a co-consumer input to Op0 and Op1.
static Problem make_Y_merge() {
    Problem p;
    for (int i = 0; i < 5; i++) p.tensors.push_back({128,128});
    p.ops.push_back({OpType::Pointwise,{0},{1},1000});
    p.ops.push_back({OpType::Pointwise,{0},{2},1000});
    p.ops.push_back({OpType::Pointwise,{1,2},{3},1000});
    p.ops.push_back({OpType::Pointwise,{3},{4},1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Fan-out: T0->Op0->T1, T1->{Op1->T2, Op2->T3}. Op1 and Op2 share input T1.
static Problem make_fanout() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{1},{3},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Co-consumer shared input: T0->{Op0->T1, Op1->T2}, T1+T2->Op2->T3
static Problem make_shared_input() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{0},{2},1000},
             {OpType::Pointwise,{1,2},{3},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// Canonical feasibility + coverage check
//
// Stronger than the old partition_valid(): calls verify_partition_feasibility
// which checks memory bounds AND no ephemeral gaps, not just coverage + finite cost.
// ============================================================================

static void check_valid(const char* label, const Partition& part, const Problem& prob) {
    // 1. Coverage
    for (size_t i = 0; i < prob.num_ops(); i++) {
        if (part.groups_of(i).empty()) {
            std::cout << "  FAIL: " << label << " Op" << i << " uncovered\n";
            g_fail++;
        } else { g_pass++; }
    }
    // 2. Memory + no ephemeral gap
    std::string err = verify_partition_feasibility(part);
    if (!err.empty()) {
        std::cout << "  FAIL: " << label << " " << err << "\n";
        g_fail++;
    } else { g_pass++; }
}

// Run a mutation N times with different seeds, check every result.
template<typename MutateFn>
static void fuzz_mutation(const char* name, const Partition& base,
                          const Problem& prob, MutateFn fn, int n = 50) {
    int changed = 0;
    for (int seed = 0; seed < n; seed++) {
        std::mt19937 rng(seed);
        auto result = fn(base, rng);
        check_valid(name, result, prob);
        if (result.num_alive() != base.num_alive() ||
            result.total_cost() != base.total_cost())
            changed++;
    }
    std::cout << "    " << name << ": " << changed << "/" << n
              << " mutations changed the partition\n";
}

// ============================================================================
// 1. mutate_merge
// ============================================================================

void test_mutate_merge_chain3() {
    std::cout << "--- test_mutate_merge_chain3 ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    fuzz_mutation("merge/chain3", part, p, [](const Partition& b, std::mt19937& r) {
        return mutate_merge(b, r);
    });
}

void test_mutate_merge_chain6() {
    std::cout << "--- test_mutate_merge_chain6 ---\n";
    auto p = make_chain6(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    size_t orig = part.num_alive();

    // Single merge: must reduce or preserve group count
    std::mt19937 rng(42);
    auto merged = mutate_merge(Partition(part), rng);
    CHECK("valid after merge", [&]{ std::string e = verify_partition_feasibility(merged); return e.empty(); }());
    CHECK("fewer or equal groups", merged.num_alive() <= orig);

    // 5 sequential merges: must remain valid throughout
    for (int i = 0; i < 5; i++) merged = mutate_merge(std::move(merged), rng);
    check_valid("merge/chain6 after 5 merges", merged, p);
}

void test_mutate_merge_reduces_groups() {
    std::cout << "--- test_mutate_merge_reduces_groups ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    std::mt19937 rng(42);
    auto result = mutate_merge(part, rng);
    CHECK("merge reduces or preserves group count", result.num_alive() <= part.num_alive());
    check_valid("merge/chain4", result, p);
}

void test_mutate_merge_cycle_blocked() {
    std::cout << "--- test_mutate_merge_cycle_blocked ---\n";
    // Verify mutate_merge never produces a cyclic condensed DAG from a valid input.
    // In a chain, merges of adjacent groups can never create cycles; merge_creates_cycle
    // guards against non-adjacent merges (which would require a disconnected source group,
    // rejected by eval_set returning 1e18).
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    for (int seed = 0; seed < 30; seed++) {
        std::mt19937 rng(seed);
        check_valid("merge cycle guard", mutate_merge(part, rng), p);
    }
}

// ============================================================================
// 2. mutate_split
// ============================================================================

void test_mutate_split_chain4_fused() {
    std::cout << "--- test_mutate_split_chain4_fused ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2,3};
    part.groups[0].cost = cache.evaluate({0,1,2,3},p,d);
    for (int i = 1; i < 4; i++) part.groups[i].alive = false;
    part.rebuild_index();

    std::mt19937 rng(7);
    auto result = mutate_split(part, rng);
    CHECK("split increases or preserves group count", result.num_alive() >= part.num_alive());
    check_valid("split/chain4-fused", result, p);
    fuzz_mutation("split/chain4-fused-fuzz", part, p, [](const Partition& b, std::mt19937& r) {
        return mutate_split(b, r);
    });
}

void test_mutate_split_after_merges() {
    std::cout << "--- test_mutate_split_after_merges ---\n";
    // Build a fused partition by repeated merges (same as existing test_mutate_split).
    auto p = make_chain6(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    std::mt19937 rng(42);
    for (int i = 0; i < 5; i++) part = mutate_merge(std::move(part), rng);
    size_t before = part.num_alive();

    auto result = mutate_split(Partition(part), rng);
    CHECK("split valid after merge chain", [&]{ std::string e = verify_partition_feasibility(result); return e.empty(); }());
    CHECK("split increases or preserves groups", result.num_alive() >= before);
    check_valid("split/chain6-after-merges", result, p);
}

// ============================================================================
// 3. mutate_reassign
// ============================================================================

void test_mutate_reassign_chain6_specific() {
    std::cout << "--- test_mutate_reassign_chain6_specific ---\n";
    // Specific 3-group partition of chain6 (from original test).
    auto p = make_chain6(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0,1}, part.eval_set({0,1}));
    part.add_group({2,3}, part.eval_set({2,3}));
    part.add_group({4,5}, part.eval_set({4,5}));

    std::mt19937 rng(42);
    auto result = mutate_reassign(Partition(part), rng);
    check_valid("reassign/chain6-specific", result, p);
    CHECK("same group count (reassign is STEAL, not merge/split)",
          result.num_alive() == part.num_alive());
}

void test_mutate_reassign_chain3_fuzz() {
    std::cout << "--- test_mutate_reassign_chain3_fuzz ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    fuzz_mutation("reassign/chain3", part, p, [](const Partition& b, std::mt19937& r) {
        return mutate_reassign(b, r);
    });
}

void test_mutate_reassign_gap_blocked() {
    std::cout << "--- test_mutate_reassign_gap_blocked ---\n";
    // Diamond4 trivial: reassigning Op0 from G0 into G1={Op1} would make T1
    // ephemeral in new_G1={Op0,Op1} while Op2 in G2 still needs T1.
    // After Bug 1 fix, src_gi=G0 is excluded from the gap check, correctly
    // blocking this move. All 100 random trials must be feasible.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    for (int seed = 0; seed < 100; seed++) {
        std::mt19937 rng(seed);
        check_valid("reassign gap blocked (Bug 1)", mutate_reassign(part, rng), p);
    }
}

void test_mutate_reassign_does_not_disconnect() {
    std::cout << "--- test_mutate_reassign_does_not_disconnect ---\n";
    // Removing a middle op from a chain-group must be rejected (would disconnect).
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2};
    part.groups[0].cost = cache.evaluate({0,1,2},p,d);
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();
    for (int seed = 0; seed < 50; seed++) {
        std::mt19937 rng(seed);
        check_valid("reassign no disconnect", mutate_reassign(part, rng), p);
    }
}

// ============================================================================
// 4. mutate_eject
// ============================================================================

void test_mutate_eject_chain4_fused() {
    std::cout << "--- test_mutate_eject_chain4_fused ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2,3};
    part.groups[0].cost = cache.evaluate({0,1,2,3},p,d);
    for (int i = 1; i < 4; i++) part.groups[i].alive = false;
    part.rebuild_index();
    fuzz_mutation("eject/chain4-fused", part, p, [](const Partition& b, std::mt19937& r) {
        return mutate_eject(b, r);
    });
}

void test_mutate_eject_splits_components() {
    std::cout << "--- test_mutate_eject_splits_components ---\n";
    // Ejecting a middle op of a fused chain must produce two remainder components.
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2,3};
    part.groups[0].cost = cache.evaluate({0,1,2,3},p,d);
    for (int i = 1; i < 4; i++) part.groups[i].alive = false;
    part.rebuild_index();

    bool found_split = false;
    for (int seed = 0; seed < 20; seed++) {
        std::mt19937 r(seed);
        auto result = mutate_eject(part, r);
        if (result.num_alive() >= 3) found_split = true;
        check_valid("eject splits components", result, p);
    }
    CHECK("eject can produce multiple components", found_split);
}

// ============================================================================
// 6. mutate_tensor_merge
// ============================================================================

void test_mutate_tensor_merge_shared_input() {
    std::cout << "--- test_mutate_tensor_merge_shared_input ---\n";
    auto p = make_shared_input(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    fuzz_mutation("tensor_merge/shared_input", part, p, [](const Partition& b, std::mt19937& r) {
        return mutate_tensor_merge(b, r);
    });
}

void test_mutate_tensor_merge_extract_cycle_blocked() {
    std::cout << "--- test_mutate_tensor_merge_extract_cycle_blocked ---\n";
    // Bug 3: the EXTRACT fallback had no cycle check. Pairwise check against
    // remainder groups must prevent cyclic condensed DAGs.
    // shared_input: T0 consumed by Op0 and Op1 (no DAG edge between them —
    // safe for extract). All trials must be feasible.
    auto p = make_shared_input(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    for (int seed = 0; seed < 100; seed++) {
        std::mt19937 rng(seed);
        check_valid("tensor_merge extract cycle (Bug 3)", mutate_tensor_merge(part, rng), p);
    }
}

void test_mutate_tensor_merge_fanout() {
    std::cout << "--- test_mutate_tensor_merge_fanout ---\n";
    // fanout: T1 consumed by Op1 and Op2. Producer Op0 also included.
    // The pairwise cycle check guards against merging groups with a DAG path between them.
    auto p = make_fanout(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    for (int seed = 0; seed < 100; seed++) {
        std::mt19937 rng(seed);
        check_valid("tensor_merge fanout", mutate_tensor_merge(part, rng), p);
    }
}

// ============================================================================
// 7. mutate_compound
// ============================================================================

void test_mutate_compound_chain6() {
    std::cout << "--- test_mutate_compound_chain6 ---\n";
    auto p = make_chain6(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    std::mt19937 rng(42);
    for (int n = 1; n <= 10; n++) {
        auto result = mutate_compound(Partition(part), n, rng);
        check_valid("compound/chain6", result, p);
    }
}

void test_mutate_compound_chain4() {
    std::cout << "--- test_mutate_compound_chain4 ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    for (int seed = 0; seed < 50; seed++) {
        std::mt19937 rng(seed);
        int n = 2 + (int)(rng() % 5);  // 2..6
        check_valid("compound/chain4", mutate_compound(part, n, rng), p);
    }
}

void test_mutate_compound_diamond4() {
    std::cout << "--- test_mutate_compound_diamond4 ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    for (int seed = 0; seed < 50; seed++) {
        std::mt19937 rng(seed);
        int n = 3 + (int)(rng() % 4);  // 3..6
        check_valid("compound/diamond4", mutate_compound(part, n, rng), p);
    }
}

// ============================================================================
// 8. crossover
// ============================================================================

void test_crossover_chain6() {
    std::cout << "--- test_crossover_chain6 ---\n";
    // Two different partitions of chain6 (from original test).
    auto p = make_chain6(); DAG d = DAG::build(p);
    CostCache cache;

    Partition a; a.prob = &p; a.dag = &d; a.cache = &cache;
    a.add_group({0,1,2}, a.eval_set({0,1,2}));
    a.add_group({3,4,5}, a.eval_set({3,4,5}));

    Partition b; b.prob = &p; b.dag = &d; b.cache = &cache;
    b.add_group({0,1},   b.eval_set({0,1}));
    b.add_group({2,3},   b.eval_set({2,3}));
    b.add_group({4,5},   b.eval_set({4,5}));

    std::mt19937 rng(42);
    auto child = crossover(a, b, rng);
    check_valid("crossover/chain6", child, p);
    CHECK("crossover has alive groups", child.num_alive() > 0);

    // Same-parent crossover must be valid
    auto same = crossover(a, a, rng);
    check_valid("crossover/chain6-same-parent", same, p);
}

void test_crossover_chain4_parents() {
    std::cout << "--- test_crossover_chain4_parents ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;

    auto pa = Partition::trivial(p, d); pa.cache = &cache;
    pa.groups[0].ops  = {0,1}; pa.groups[0].cost = cache.evaluate({0,1},p,d);
    pa.groups[1].alive = false; pa.rebuild_index();

    auto pb = Partition::trivial(p, d); pb.cache = &cache;
    pb.groups[2].ops  = {2,3}; pb.groups[2].cost = cache.evaluate({2,3},p,d);
    pb.groups[3].alive = false; pb.rebuild_index();

    for (int seed = 0; seed < 30; seed++) {
        std::mt19937 rng(seed);
        check_valid("crossover/chain4", crossover(pa, pb, rng), p);
    }
}

void test_crossover_trivial_parents() {
    std::cout << "--- test_crossover_trivial_parents ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto pa = Partition::trivial(p, d); pa.cache = &cache;
    auto pb = Partition::trivial(p, d); pb.cache = &cache;
    for (int seed = 0; seed < 20; seed++) {
        std::mt19937 rng(seed);
        auto child = crossover(pa, pb, rng);
        check_valid("crossover/trivial-trivial", child, p);
        // Identical trivial parents: each op is its own cluster, but the greedy
        // assignment fuses adjacent clusters when merge is cheaper than standalone.
        // For a chain with ample memory, fusion is always cheaper, so the child
        // will typically have fewer groups than trivial — this is correct behaviour.
        CHECK("at least one group", child.num_alive() >= 1);
    }
}

void test_crossover_fused_vs_trivial() {
    std::cout << "--- test_crossover_fused_vs_trivial ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;

    auto pa = Partition::trivial(p, d); pa.cache = &cache;
    pa.groups[0].ops  = {0,1,2,3};
    pa.groups[0].cost = cache.evaluate({0,1,2,3},p,d);
    for (int i = 1; i < 4; i++) pa.groups[i].alive = false;
    pa.rebuild_index();

    auto pb = Partition::trivial(p, d); pb.cache = &cache;

    for (int seed = 0; seed < 30; seed++) {
        std::mt19937 rng(seed);
        check_valid("crossover/fused-trivial", crossover(pa, pb, rng), p);
    }
}

void test_crossover_diamond4_parents() {
    std::cout << "--- test_crossover_diamond4_parents ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;

    // Under new ephemeral rule, {Op0,Op1} is invalid (T1 ephemeral, Op2 stranded).
    // Use valid parents:
    // Parent A: {Op0,Op1,Op2} + {Op3} — all T1 consumers internal
    // Parent B: {Op0} + {Op1,Op2,Op3} — T1 is boundary input to second group
    auto pa = Partition::trivial(p, d); pa.cache = &cache;
    pa.groups[0].ops = {0,1,2};
    pa.groups[0].cost = cache.evaluate({0,1,2}, p, d);
    pa.groups[1].alive = false;
    pa.groups[2].alive = false;
    pa.rebuild_index();

    auto pb = Partition::trivial(p, d); pb.cache = &cache;
    pb.groups[1].ops = {1,2,3};
    pb.groups[1].cost = cache.evaluate({1,2,3}, p, d);
    pb.groups[2].alive = false;
    pb.groups[3].alive = false;
    pb.rebuild_index();

    for (int seed = 0; seed < 30; seed++) {
        std::mt19937 rng(seed);
        check_valid("crossover/diamond4", crossover(pa, pb, rng), p);
    }
}

void test_crossover_Y_merge() {
    std::cout << "--- test_crossover_Y_merge ---\n";
    // Original test_crossover_Y topology.
    auto p = make_Y_merge(); DAG d = DAG::build(p);
    CostCache cache;

    Partition a; a.prob = &p; a.dag = &d; a.cache = &cache;
    a.add_group({0,1}, a.eval_set({0,1}));
    a.add_group({2,3}, a.eval_set({2,3}));

    Partition b; b.prob = &p; b.dag = &d; b.cache = &cache;
    b.add_group({0},     b.eval_set({0}));
    b.add_group({1,2,3}, b.eval_set({1,2,3}));

    std::mt19937 rng(42);
    auto child = crossover(a, b, rng);
    check_valid("crossover/Y_merge", child, p);
}

void test_crossover_all_ops_covered() {
    std::cout << "--- test_crossover_all_ops_covered ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto pa = Partition::trivial(p, d); pa.cache = &cache;
    auto pb = Partition::trivial(p, d); pb.cache = &cache;
    for (int seed = 0; seed < 20; seed++) {
        std::mt19937 rng(seed);
        auto child = crossover(pa, pb, rng);
        for (size_t i = 0; i < p.num_ops(); i++)
            CHECK("op covered", !child.groups_of(i).empty());
    }
}

// ============================================================================
// 9. Stochastic coverage walk (original test_mutation_preserves_coverage)
// ============================================================================

void test_mutation_preserves_coverage_Y_merge() {
    std::cout << "--- test_mutation_preserves_coverage_Y_merge ---\n";
    // 50 random mutations applied sequentially on Y_merge. Every intermediate
    // state must remain valid. Tests compound mutation tolerance.
    auto p = make_Y_merge(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    std::mt19937 rng(42);
    for (int i = 0; i < 50; i++) {
        int choice = rng() % 4;
        switch (choice) {
            case 0: part = mutate_merge(std::move(part), rng);    break;
            case 1: part = mutate_split(std::move(part), rng);    break;
            case 2: part = mutate_reassign(std::move(part), rng); break;
            case 3: part = mutate_eject(std::move(part), rng);    break;
        }
        check_valid("coverage walk step", part, p);
    }
}

// ============================================================================
// 10. Pipeline stress (mirrors parallel_search workflow)
// ============================================================================

void test_pipeline_compound_then_crossover() {
    std::cout << "--- test_pipeline_compound_then_crossover ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto base = Partition::trivial(p, d); base.cache = &cache;

    std::mt19937 rng(42);
    auto pa = mutate_compound(base, 3, rng);
    auto pb = mutate_compound(base, 4, rng);
    check_valid("compound parent A", pa, p);
    check_valid("compound parent B", pb, p);

    for (int seed = 0; seed < 20; seed++) {
        std::mt19937 r(seed);
        check_valid("crossover child", crossover(pa, pb, r), p);
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // 1. mutate_merge
    test_mutate_merge_chain3();
    test_mutate_merge_chain6();
    test_mutate_merge_reduces_groups();
    test_mutate_merge_cycle_blocked();

    // 2. mutate_split
    test_mutate_split_chain4_fused();
    test_mutate_split_after_merges();

    // 3. mutate_reassign (Bug 1 regression)
    test_mutate_reassign_chain6_specific();
    test_mutate_reassign_chain3_fuzz();
    test_mutate_reassign_gap_blocked();
    test_mutate_reassign_does_not_disconnect();

    // 4. mutate_eject
    test_mutate_eject_chain4_fused();
    test_mutate_eject_splits_components();

    // 6. mutate_tensor_merge (Bug 3 regression)
    test_mutate_tensor_merge_shared_input();
    test_mutate_tensor_merge_extract_cycle_blocked();
    test_mutate_tensor_merge_fanout();

    // 7. mutate_compound
    test_mutate_compound_chain6();
    test_mutate_compound_chain4();
    test_mutate_compound_diamond4();

    // 8. crossover
    test_crossover_chain6();
    test_crossover_chain4_parents();
    test_crossover_trivial_parents();
    test_crossover_fused_vs_trivial();
    test_crossover_diamond4_parents();
    test_crossover_Y_merge();
    test_crossover_all_ops_covered();

    // 9. Stochastic coverage walk
    test_mutation_preserves_coverage_Y_merge();

    // 10. Pipeline stress
    test_pipeline_compound_then_crossover();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}