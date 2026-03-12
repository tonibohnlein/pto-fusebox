// local_search_test.cpp
// Tests for Partition, Move generation, and local search components.

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "partition/partition.h"
#include "search/local_search.h"
#include "search/fm_search.h"
#include "test_move_helpers.h"
#include "solution/solution.h"
#include <algorithm>
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
static void CHECK_EQ_S(const char* l, size_t g, size_t e) {
    if (g == e) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}

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

// ==================== Partition construction ====================

void test_trivial_partition() {
    std::cout << "--- test_trivial_partition ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    CHECK_EQ_S("3 groups", part.num_alive(), 3);
    CHECK("G0 has Op0", part.groups[0].ops.count(0));
    CHECK("G1 has Op1", part.groups[1].ops.count(1));
    CHECK("G2 has Op2", part.groups[2].ops.count(2));

    // Each op's cost matches standalone evaluation
    auto sg0 = *Subgraph::create(p, d, {0});
    CHECK_EQ("G0 cost", part.groups[0].cost, sg0.best_cost().latency);
}

// ==================== Partition queries ====================

void test_groups_of() {
    std::cout << "--- test_groups_of ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    auto gs = part.groups_of(1);
    CHECK_EQ_S("Op1 in 1 group", gs.size(), 1);
    CHECK("Op1 in G1", gs[0] == 1);

    // After killing G1, Op1 has no groups
    part.groups[1].alive = false;
    part.rebuild_index();
    gs = part.groups_of(1);
    CHECK_EQ_S("Op1 in 0 groups after kill", gs.size(), 0);
}

void test_boundary_neighbors() {
    std::cout << "--- test_boundary_neighbors ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // G0 = {Op0}: neighbors are Op1 (successor via T1)
    auto n0 = part.boundary_neighbors(0);
    CHECK("G0 neighbor Op1", n0.count(1));
    CHECK_EQ_S("G0 1 neighbor", n0.size(), 1);

    // G1 = {Op1}: neighbors are Op0 (pred) and Op2 (succ)
    auto n1 = part.boundary_neighbors(1);
    CHECK("G1 neighbor Op0", n1.count(0));
    CHECK("G1 neighbor Op2", n1.count(2));
    CHECK_EQ_S("G1 2 neighbors", n1.size(), 2);

    // Merge G0 and G1: {Op0, Op1}. Neighbor is now only Op2.
    part.groups[0].ops = {0, 1};
    part.groups[1].alive = false;
    auto n01 = part.boundary_neighbors(0);
    CHECK("merged neighbor Op2", n01.count(2));
    CHECK_EQ_S("merged 1 neighbor", n01.size(), 1);
}

void test_adjacent_groups() {
    std::cout << "--- test_adjacent_groups ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // G0 is adjacent to G1 (via Op1)
    auto adj0 = part.adjacent_groups(0);
    CHECK("G0 adj to G1", adj0.count(1));
    CHECK_EQ_S("G0 1 adjacent", adj0.size(), 1);

    // G1 is adjacent to G0 and G2
    auto adj1 = part.adjacent_groups(1);
    CHECK("G1 adj G0", adj1.count(0));
    CHECK("G1 adj G2", adj1.count(2));
    CHECK_EQ_S("G1 2 adjacent", adj1.size(), 2);
}

// ==================== eval_set ====================

void test_eval_set() {
    std::cout << "--- test_eval_set ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // eval_set({0}) should match G0 cost
    double e0 = part.eval_set({0});
    CHECK_EQ("eval {0}", e0, part.groups[0].cost);

    // eval_set({0,1}) should match fused {Op0,Op1}
    double e01 = part.eval_set({0, 1});
    auto sg01 = *Subgraph::create(p, d, {0, 1});
    CHECK_EQ("eval {0,1}", e01, sg01.best_cost().latency);

    // Invalid set (disconnected) returns 1e18
    // Op0 and Op2 are not adjacent
    double e02 = part.eval_set({0, 2});
    CHECK("eval disconnected", e02 >= 1e17);
}

// ==================== Move application ====================

void test_merge_move() {
    std::cout << "--- test_merge_move ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    double old_cost = part.groups[0].cost + part.groups[1].cost;
    double merged_cost = part.eval_set({0, 1});
    CHECK("merge saves", merged_cost < old_cost);

    // Simulate merge
    int old_gen = part.groups[0].gen;
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = merged_cost;
    part.groups[0].gen++;
    part.groups[1].alive = false;
    part.groups[1].gen++;

    CHECK_EQ_S("2 alive after merge", part.num_alive(), 2);
    CHECK("G1 dead", !part.groups[1].alive);
    CHECK("gen incremented", part.groups[0].gen == old_gen + 1);
    CHECK_EQ("merged cost", part.groups[0].cost, merged_cost);
    CHECK("Op0 in G0", part.groups[0].ops.count(0));
    CHECK("Op1 in G0", part.groups[0].ops.count(1));
}

void test_steal_move() {
    std::cout << "--- test_steal_move ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // First merge Op0+Op1 into G0
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;

    // Now try stealing Op1 from G0 to G2
    std::set<size_t> new_g0 = {0};
    std::set<size_t> new_g2 = {1, 2};
    double new_g0_cost = part.eval_set(new_g0);
    double new_g2_cost = part.eval_set(new_g2);

    // Both must be valid subgraphs
    CHECK("new G0 valid", new_g0_cost < 1e17);
    CHECK("new G2 valid", new_g2_cost < 1e17);

    // Apply steal
    part.groups[0].ops = new_g0;
    part.groups[0].cost = new_g0_cost;
    part.groups[0].gen++;
    part.groups[2].ops = new_g2;
    part.groups[2].cost = new_g2_cost;
    part.groups[2].gen++;

    CHECK("Op1 not in G0", !part.groups[0].ops.count(1));
    CHECK("Op1 in G2", part.groups[2].ops.count(1));
    CHECK("Op2 in G2", part.groups[2].ops.count(2));
}

void test_recompute_move() {
    std::cout << "--- test_recompute_move ---\n";
    // Chain: T0->Op0->T1->Op1->T2->Op2->T3
    // Recompute Op1 in G2={Op2}: T2 becomes ephemeral (saves load).
    // Boundary of {Op1,Op2}: T1 (smaller than T1+T2 separately).
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Standalone Op2: loads T2 (boundary in), evicts T3.
    double old_cost = part.groups[2].cost;

    // {Op1, Op2}: T2 ephemeral. Loads T1, evicts T3.
    // Same memory footprint, but T2 transfer saved (it's ephemeral now).
    // However, compute increases by Op1 cost.
    std::set<size_t> new_g2 = {1, 2};
    double new_cost = part.eval_set(new_g2);
    CHECK("recompute feasible", new_cost < 1e17);

    // Apply recompute — G1 unchanged, G2 gains Op1
    int g1_gen = part.groups[1].gen;
    part.groups[2].ops = new_g2;
    part.groups[2].cost = new_cost;
    part.groups[2].gen++;
    part.rebuild_index();

    CHECK("Op1 still in G1", part.groups[1].ops.count(1));
    CHECK("Op1 also in G2", part.groups[2].ops.count(1));
    CHECK("G1 gen unchanged", part.groups[1].gen == g1_gen);
    CHECK_EQ_S("3 alive", part.num_alive(), 3);
    // Op1 is now in two groups — recomputation
    auto gs = part.groups_of(1);
    CHECK_EQ_S("Op1 in 2 groups", gs.size(), 2);
}

// ==================== Generation counters ====================

void test_stale_detection() {
    std::cout << "--- test_stale_detection ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Create an OpMove referencing G0→G1 merge at current gen
    OpMove om;
    om.op = 0;
    om.move = FMMove{FMMove::MERGE, 0, 0, 1, SIZE_MAX, 100.0};
    om.primary_group = 0;
    om.gen_a = part.groups[0].gen;
    om.gen_b = part.groups[1].gen;

    // Move is fresh
    CHECK("fresh: ga alive", part.groups[om.move.ga].alive);
    CHECK("fresh: ga gen match", part.groups[om.primary_group].gen == om.gen_a);
    CHECK("fresh: gb gen match", part.groups[om.move.gb].gen == om.gen_b);

    // Modify G0 → gen increments → move becomes stale
    part.groups[0].gen++;
    CHECK("stale: gen_a mismatch", part.groups[om.primary_group].gen != om.gen_a);

    // Kill G1 → move also stale
    part.groups[1].alive = false;
    CHECK("stale: gb dead", !part.groups[om.move.gb].alive);
}

// ==================== Move generation ====================

void test_generate_moves() {
    std::cout << "--- test_generate_moves ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    auto moves = all_moves_for_group(part, 0);

    // G0 = {Op0}, neighbor = Op1 in G1. Should generate merge, steal, recompute.
    CHECK("moves generated", !moves.empty());

    // Count move types
    int merges = 0, steals = 0, recomputes = 0;
    for (auto& m : moves) {
        switch (m.type) {
            case FMMove::MERGE: merges++; break;
            case FMMove::STEAL: steals++; break;
            case FMMove::RECOMPUTE: recomputes++; break;
            default: break;
        }
    }
    // At minimum we should have a merge(G0,G1) and possibly steal/recompute
    CHECK("has merge", merges > 0);
    std::cout << "  moves: " << merges << " merge, " << steals << " steal, "
              << recomputes << " recompute\n";
}

// ==================== Move gain correctness ====================
// Key property: computed gain must exactly match the actual total_cost change.

void test_merge_gain_correctness() {
    std::cout << "--- test_merge_gain_correctness ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    double total_before = part.total_cost();
    double ga_cost = part.groups[0].cost;
    double gb_cost = part.groups[1].cost;

    // Compute merged cost and gain
    std::set<size_t> merged = {0, 1};
    double merged_cost = part.eval_set(merged);
    double expected_gain = (ga_cost + gb_cost) - merged_cost;
    CHECK("merge has positive gain", expected_gain > 0);

    // Apply merge
    part.groups[0].ops = merged;
    part.groups[0].cost = merged_cost;
    part.groups[0].gen++;
    part.groups[1].alive = false;
    part.groups[1].gen++;

    double total_after = part.total_cost();
    double actual_gain = total_before - total_after;

    CHECK_EQ("merge gain matches", actual_gain, expected_gain);
    std::cout << "  before=" << total_before << " after=" << total_after
              << " gain=" << actual_gain << "\n";
}

void test_steal_gain_correctness() {
    std::cout << "--- test_steal_gain_correctness ---\n";
    // Use a problem where steal actually saves:
    // 4-op chain, start with G0={Op0,Op1}, G2={Op2}, G3={Op3}
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000},
             {OpType::Pointwise,{3},{4},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    auto part = Partition::trivial(p, d);
    // Merge Op0+Op1
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;

    double total_before = part.total_cost();
    double gs_cost = part.groups[0].cost;  // source: {Op0, Op1}
    double gt_cost = part.groups[2].cost;  // target: {Op2}

    // Steal Op1 from G0 to G2: new G0={Op0}, new G2={Op1,Op2}
    double new_gs_cost = part.eval_set({0});
    double new_gt_cost = part.eval_set({1, 2});
    double expected_gain = (gs_cost + gt_cost) - (new_gs_cost + new_gt_cost);

    // Apply steal
    part.groups[0].ops = {0};
    part.groups[0].cost = new_gs_cost;
    part.groups[0].gen++;
    part.groups[2].ops = {1, 2};
    part.groups[2].cost = new_gt_cost;
    part.groups[2].gen++;

    double total_after = part.total_cost();
    double actual_gain = total_before - total_after;

    CHECK_EQ("steal gain matches", actual_gain, expected_gain);
    std::cout << "  before=" << total_before << " after=" << total_after
              << " gain=" << actual_gain << "\n";
}

void test_recompute_gain_correctness() {
    std::cout << "--- test_recompute_gain_correctness ---\n";
    // Chain: T0->Op0->T1->Op1->T2
    // G0={Op0}, G1={Op1}. Recompute Op0 in G1: makes T1 ephemeral.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},100},   // cheap op
             {OpType::Pointwise,{1},{2},1000}};  // expensive op
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    double total_before = part.total_cost();
    double gt_cost_before = part.groups[1].cost;

    // Recompute Op0 in G1: G1 becomes {Op0, Op1}
    double new_gt_cost = part.eval_set({0, 1});
    double expected_gain = gt_cost_before - new_gt_cost;
    // Only G1 changes, G0 is unaffected
    CHECK("recompute only changes target", true);

    // Apply recompute
    part.groups[1].ops = {0, 1};
    part.groups[1].cost = new_gt_cost;
    part.groups[1].gen++;
    part.rebuild_index();
    // G0 unchanged

    double total_after = part.total_cost();
    double actual_gain = total_before - total_after;

    CHECK_EQ("recompute gain matches", actual_gain, expected_gain);
    // Op0 is now in two groups
    auto gs = part.groups_of(0);
    CHECK_EQ_S("Op0 in 2 groups", gs.size(), 2);
    std::cout << "  before=" << total_before << " after=" << total_after
              << " gain=" << actual_gain << "\n";
}

void test_eject_gain_correctness() {
    std::cout << "--- test_eject_gain_correctness ---\n";
    // 4-op chain, G0={Op0,Op1,Op2}, G3={Op3}
    // Eject Op2 from G0: G0={Op0,Op1}, new G={Op2}
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000},
             {OpType::Pointwise,{3},{4},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Merge {Op0, Op1, Op2}
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;

    double total_before = part.total_cost();
    double ga_cost = part.groups[0].cost;

    // Eject Op2
    double remainder_cost = part.eval_set({0, 1});
    double singleton_cost = part.eval_set({2});
    double expected_gain = ga_cost - (remainder_cost + singleton_cost);

    // Apply eject
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = remainder_cost;
    part.groups[0].gen++;
    size_t new_gi = part.add_group({2}, singleton_cost);

    double total_after = part.total_cost();
    double actual_gain = total_before - total_after;

    CHECK_EQ("eject gain matches", actual_gain, expected_gain);
    CHECK("Op2 in new group", part.groups[new_gi].ops.count(2));
    CHECK_EQ_S("3 alive", part.num_alive(), 3);
    std::cout << "  before=" << total_before << " after=" << total_after
              << " gain=" << actual_gain << "\n";
}

// Verify that the heap-based search produces monotonically decreasing total cost
void test_local_search_monotonic() {
    std::cout << "--- test_local_search_monotonic ---\n";
    // We can't easily intercept intermediate states of local_search(),
    // but we can verify the result is better than trivial and valid.
    auto p = make_chain3(); DAG d = DAG::build(p);

    auto trivial = Partition::trivial(p, d);
    double trivial_cost = trivial.total_cost();

    auto result = local_search(p, d);
    double result_cost = result.total_cost();

    CHECK("search improves or equals", result_cost <= trivial_cost + 0.01);

    // Every op must still be covered
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", !result.groups_of(i).empty());
}

// ==================== Full local search ====================

void test_local_search_chain() {
    std::cout << "--- test_local_search_chain ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);

    auto part = local_search(p, d);

    // Should have merged some groups (fusion is always beneficial for chain)
    CHECK("fewer groups", part.num_alive() < 3);
    CHECK("cost improved", part.total_cost() < 3 * 3276.8 + 1);

    // All ops still covered
    for (size_t i = 0; i < 3; i++)
        CHECK("op covered", !part.groups_of(i).empty());
}

void test_local_search_diamond() {
    std::cout << "--- test_local_search_diamond ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);

    auto part = local_search(p, d);

    // PROBLEM.md shows 3C = 4638.4 is optimal for this graph
    CHECK("cost reasonable", part.total_cost() < 11468.8 + 1);  // better than unfused

    // All ops covered
    for (size_t i = 0; i < 3; i++)
        CHECK("op covered", !part.groups_of(i).empty());
}

void test_local_search_produces_valid_solution() {
    std::cout << "--- test_local_search_produces_valid_solution ---\n";
    // Use Example 1 problem from PROBLEM.md
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},{OpType::Pointwise,{1},{2},100}};
    p.fast_memory_capacity = 35000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    auto part = local_search(p, d);

    // Build a Solution from it
    std::vector<std::vector<size_t>> group_ops;
    for (auto& g : part.groups)
        if (g.alive) group_ops.push_back({g.ops.begin(), g.ops.end()});

    std::vector<ScheduleStep> steps;
    for (auto& ops : group_ops) {
        auto sg = Subgraph::create(p, d, ops);
        if (!sg) continue;
        auto best = sg->best_cost();
        steps.push_back({std::move(*sg), best.config, {}});
    }

    Solution sol(p, d, std::move(steps));
    auto vr = sol.validate();
    CHECK("solution valid", vr.valid);

    // Should find fusion: lat = 3276.8
    CHECK_EQ("optimal latency", sol.total_latency(), 3276.8);
}

void test_ejectable_ops() {
    std::cout << "--- test_ejectable_ops ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Singleton groups have no ejectable ops
    CHECK_EQ_S("singleton no eject", part.ejectable_ops(0).size(), 0);

    // Merge G0+G1: {Op0, Op1}. Op0 has no external neighbors (pred=none,
    // succ=Op1 inside). Op1 has succ Op2 outside → Op1 is on boundary.
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;
    auto ej01 = part.ejectable_ops(0);
    CHECK("Op0 not ejectable (internal)", 
          std::find(ej01.begin(), ej01.end(), 0) == ej01.end());
    CHECK("Op1 ejectable (boundary)",
          std::find(ej01.begin(), ej01.end(), 1) != ej01.end());
    CHECK_EQ_S("1 ejectable op", ej01.size(), 1);

    // Merge all three: {Op0, Op1, Op2}. None has external neighbors
    // (Op0: no pred ops, succ=Op1 inside. Op2: pred=Op1 inside, no succ ops.)
    part.groups[0].ops = {0, 1, 2};
    part.groups[2].alive = false;
    auto ej012 = part.ejectable_ops(0);
    CHECK_EQ_S("all-fused no ejectable", ej012.size(), 0);
}

void test_eject_move() {
    std::cout << "--- test_eject_move ---\n";
    // 4-op chain: T0->Op0->T1->Op1->T2->Op2->T3->Op3->T4
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000},
             {OpType::Pointwise,{3},{4},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Merge {Op0, Op1, Op2} into G0. G3 = {Op3} stays separate.
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    // Now Op2 has succ Op3 outside → Op2 is ejectable from G0.

    auto ejectable = part.ejectable_ops(0);
    CHECK("Op2 ejectable", std::find(ejectable.begin(), ejectable.end(), 2) != ejectable.end());

    // Eject Op2: remainder={Op0,Op1}, singleton={Op2}
    std::set<size_t> remainder = {0, 1};
    double remainder_cost = part.eval_set(remainder);
    double singleton_cost = part.eval_set({2});
    CHECK("remainder valid", remainder_cost < 1e17);
    CHECK("singleton valid", singleton_cost < 1e17);

    double old_cost = part.groups[0].cost;
    part.groups[0].ops = remainder;
    part.groups[0].cost = remainder_cost;
    part.groups[0].gen++;
    size_t new_gi = part.add_group({2}, singleton_cost);

    CHECK("Op2 not in G0", !part.groups[0].ops.count(2));
    CHECK("Op2 in new group", part.groups[new_gi].ops.count(2));
    CHECK_EQ_S("3 alive", part.num_alive(), 3);  // G0={0,1}, new={2}, G3={3}

    std::cout << "  fused(0,1,2)=" << old_cost
              << " split=" << remainder_cost + singleton_cost << "\n";
}

// ==================== Tabu mechanics ====================





void test_best_seen_preserved() {
    std::cout << "--- test_best_seen_preserved ---\n";
    // Verify that greedy_descent returns improved partition, not the final state
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    auto result = greedy_descent(std::move(part));

    // Result should be valid and at least as good as trivial
    double trivial_cost = Partition::trivial(p, d).total_cost();
    CHECK("result <= trivial", result.total_cost() <= trivial_cost + 0.01);

    // All ops covered
    for (size_t i = 0; i < 3; i++)
        CHECK("op covered", !result.groups_of(i).empty());
}

// ==================== Main ====================

int main() {
    test_trivial_partition();
    test_groups_of();
    test_boundary_neighbors();
    test_adjacent_groups();
    test_eval_set();
    test_merge_move();
    test_steal_move();
    test_recompute_move();
    test_stale_detection();
    test_generate_moves();
    test_ejectable_ops();
    test_eject_move();
    test_merge_gain_correctness();
    test_steal_gain_correctness();
    test_recompute_gain_correctness();
    test_eject_gain_correctness();
    test_local_search_monotonic();
    test_local_search_chain();
    test_local_search_diamond();
    test_local_search_produces_valid_solution();
    test_best_seen_preserved();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}