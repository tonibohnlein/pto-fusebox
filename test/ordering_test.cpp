// ordering_test.cpp
//
// Tests for dfs_ordering, beam_search_ordering, and random_ordering.
//
// What we check for every ordering:
//   1. Topological validity: every group appears exactly once, all groups
//      appear, and each group's DAG predecessors appear before it.
//   2. retain_per_step populated: DFS must set retain_per_step (Fix 1).
//   3. Retention quality: beam ≤ DFS total latency (beam explores retentions,
//      DFS just greedily picks order).
//   4. steps_from_ordering feasibility: the ScheduleSteps built from each
//      ordering all have finite latency.
//
// Build: make ordering_test
// Run:   ./ordering_test

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost_cache.h"
#include "partition/partition.h"
#include "solution/ordering.h"
#include "solution/solution.h"
#include "init/init_strategies.h"
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
    // T1 and T2 are intermediate — mark them retainable
    p.retainable_tensors = {1, 2};
    return p;
}

// T0->Op0->T1->Op1->T2->Op2->T3->Op3->T4
static Problem make_chain4() {
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    p.retainable_tensors = {1, 2, 3};
    return p;
}

// 4-op diamond: T0->Op0->T1, T1->{Op1->T2, Op2->T3}, T2+T3->Op3->T4
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
    p.retainable_tensors = {1, 2, 3};
    return p;
}

// ============================================================================
// Helpers
// ============================================================================

// Build a trivial partition (one op per group) and finalize it.
static Partition make_finalized(const Problem& p, const DAG& d) {
    CostCache cache;
    auto part = Partition::trivial(p, d);
    part.cache = &cache;
    part.finalize();
    return part;
}

// Build a partition via best_initial (fused) and finalize it.
static Partition make_fused_finalized(const Problem& p, const DAG& d) {
    CostCache cache;
    auto part = best_initial(p, d, &cache);
    part.finalize();
    return part;
}

// Check that an ordering is topologically valid for the given partition.
static bool topo_valid(const OrderingResult& r, const Partition& part) {
    size_t n_alive = part.num_alive();
    if (r.order.size() != n_alive) return false;

    // All groups appear exactly once
    std::set<size_t> seen;
    for (auto g : r.order) {
        if (!part.groups[g].alive) return false;
        if (!seen.insert(g).second) return false;  // duplicate
    }

    // Each group's predecessors appear before it
    std::set<size_t> scheduled;
    for (auto g : r.order) {
        // Check all in-edges of g in the group DAG
        for (size_t pred = 0; pred < part.groups.size(); pred++) {
            if (!part.groups[pred].alive) continue;
            // Is pred a predecessor of g?
            bool is_pred = false;
            for (auto s : part.group_succs[pred])
                if (s == g) { is_pred = true; break; }
            if (is_pred && !scheduled.count(pred)) return false;
        }
        scheduled.insert(g);
    }
    return true;
}

// ============================================================================
// 1. dfs_ordering
// ============================================================================

void test_dfs_topo_chain3() {
    std::cout << "--- test_dfs_topo_chain3 ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    auto r = dfs_ordering(part);
    CHECK("dfs: topologically valid", topo_valid(r, part));
    CHECK_EQ_S("dfs: all groups covered", r.order.size(), part.num_alive());
    std::cout << "    order:";
    for (auto g : r.order) std::cout << " G" << g;
    std::cout << "\n";
}

void test_dfs_topo_diamond4() {
    std::cout << "--- test_dfs_topo_diamond4 ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    auto r = dfs_ordering(part);
    CHECK("dfs diamond: topologically valid", topo_valid(r, part));
    CHECK_EQ_S("dfs diamond: all groups", r.order.size(), part.num_alive());
}

void test_dfs_retain_per_step_populated() {
    std::cout << "--- test_dfs_retain_per_step_populated ---\n";
    // Fix 1: dfs_ordering must populate retain_per_step.
    // For a chain of 3 groups where T1 and T2 are retainable, steps 0 and 1
    // should have non-empty retain sets.
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    auto r = dfs_ordering(part);

    CHECK("retain_per_step has correct size", r.retain_per_step.size() == r.order.size());

    // At least one intermediate step should have a retention hint
    bool any_nonempty = false;
    for (size_t i = 0; i + 1 < r.order.size(); i++)
        if (!r.retain_per_step[i].empty()) any_nonempty = true;
    CHECK("at least one step has retention hint (Fix 1)", any_nonempty);

    std::cout << "    retain_per_step:";
    for (auto& rs : r.retain_per_step) std::cout << " {" << rs.size() << "}";
    std::cout << "\n";
}

void test_dfs_fused_partition() {
    std::cout << "--- test_dfs_fused_partition ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = make_fused_finalized(p, d);
    auto r = dfs_ordering(part);
    CHECK("dfs fused: topologically valid", topo_valid(r, part));
    CHECK("dfs fused: retain_per_step sized", r.retain_per_step.size() == r.order.size());
}

// ============================================================================
// 2. beam_search_ordering
// ============================================================================

void test_beam_topo_chain3() {
    std::cout << "--- test_beam_topo_chain3 ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    auto r = beam_search_ordering(part, 5);
    CHECK("beam: topologically valid", topo_valid(r, part));
    CHECK_EQ_S("beam: all groups", r.order.size(), part.num_alive());
    CHECK("beam: retain_per_step sized", r.retain_per_step.size() == r.order.size());
}

void test_beam_topo_diamond4() {
    std::cout << "--- test_beam_topo_diamond4 ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    auto r = beam_search_ordering(part, 5);
    CHECK("beam diamond: topologically valid", topo_valid(r, part));
    CHECK_EQ_S("beam diamond: all groups", r.order.size(), part.num_alive());
}

void test_beam_total_latency_set() {
    std::cout << "--- test_beam_total_latency_set ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    auto r = beam_search_ordering(part, 5);
    CHECK("beam total_latency > 0", r.total_latency > 0.0);
    CHECK("beam total_latency finite", r.total_latency < 1e17);
    std::cout << "    total_latency=" << r.total_latency << "\n";
}

void test_beam_width_one_vs_five() {
    std::cout << "--- test_beam_width_one_vs_five ---\n";
    // Beam width=1 degenerates toward greedy; width=5 can only do better.
    // Both must be topologically valid.
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    auto r1 = beam_search_ordering(part, 1);
    auto r5 = beam_search_ordering(part, 5);
    CHECK("width=1 topo valid", topo_valid(r1, part));
    CHECK("width=5 topo valid", topo_valid(r5, part));
    CHECK("width=5 latency <= width=1", r5.total_latency <= r1.total_latency + 0.01);
    std::cout << "    w1=" << r1.total_latency << " w5=" << r5.total_latency << "\n";
}

// ============================================================================
// 3. random_ordering
// ============================================================================

void test_random_topo_chain3() {
    std::cout << "--- test_random_topo_chain3 ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    // Collect retainable tensors
    std::set<size_t> feasibly_ret = p.retainable_tensors;

    for (int seed = 0; seed < 20; seed++) {
        std::mt19937 rng(seed);
        auto r = random_ordering(part, feasibly_ret, rng);
        CHECK("random: topo valid", topo_valid(r, part));
        CHECK_EQ_S("random: all groups", r.order.size(), part.num_alive());
    }
}

void test_random_topo_diamond4() {
    std::cout << "--- test_random_topo_diamond4 ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    std::set<size_t> feasibly_ret = p.retainable_tensors;

    for (int seed = 0; seed < 20; seed++) {
        std::mt19937 rng(seed);
        auto r = random_ordering(part, feasibly_ret, rng);
        CHECK("random diamond: topo valid", topo_valid(r, part));
    }
}

void test_random_diversity() {
    std::cout << "--- test_random_diversity ---\n";
    // Different seeds should occasionally produce different orderings.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    std::set<size_t> feasibly_ret = p.retainable_tensors;

    std::set<std::vector<size_t>> seen_orders;
    for (int seed = 0; seed < 30; seed++) {
        std::mt19937 rng(seed);
        auto r = random_ordering(part, feasibly_ret, rng);
        seen_orders.insert(r.order);
    }
    CHECK("random: produces diverse orderings", seen_orders.size() > 1);
    std::cout << "    distinct orderings: " << seen_orders.size() << "/30\n";
}

// ============================================================================
// 4. steps_from_ordering / Solution::from_partition
// ============================================================================

void test_steps_from_dfs_ordering_feasible() {
    std::cout << "--- test_steps_from_dfs_ordering_feasible ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    auto r = dfs_ordering(part);
    auto steps = Solution::steps_from_ordering(p, d, part, r);
    CHECK("steps non-empty", !steps.empty());
    for (auto& s : steps)
        CHECK("step has finite latency", s.subgraph.best_cost().latency < 1e17);
}

void test_steps_from_beam_ordering_feasible() {
    std::cout << "--- test_steps_from_beam_ordering_feasible ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);
    auto r = beam_search_ordering(part, 5);
    auto steps = Solution::steps_from_ordering(p, d, part, r);
    CHECK("steps non-empty", !steps.empty());
    for (auto& s : steps)
        CHECK("step has finite latency", s.subgraph.best_cost().latency < 1e17);
}

void test_beam_le_dfs_latency() {
    std::cout << "--- test_beam_le_dfs_latency ---\n";
    // Beam explores retention; DFS now also populates retain_per_step (Fix 1)
    // so both benefit. Beam should still be ≤ DFS since it searches more broadly.
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = make_finalized(p, d);

    auto dfs_r  = dfs_ordering(part);
    auto beam_r = beam_search_ordering(part, 10);

    auto dfs_steps  = Solution::steps_from_ordering(p, d, part, dfs_r);
    auto beam_steps = Solution::steps_from_ordering(p, d, part, beam_r);

    Solution dfs_sol (p, d, std::move(dfs_steps));
    Solution beam_sol(p, d, std::move(beam_steps));

    CHECK("beam total_latency <= dfs total_latency",
          beam_sol.total_latency() <= dfs_sol.total_latency() + 0.01);
    std::cout << "    dfs=" << dfs_sol.total_latency()
              << " beam=" << beam_sol.total_latency() << "\n";
}

void test_from_partition_valid_solution() {
    std::cout << "--- test_from_partition_valid_solution ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.finalize();

    auto sol = Solution::from_partition(p, d, part);
    auto vr = sol.validate();
    CHECK("solution valid", vr.valid);
    if (!vr.valid) std::cout << "    error: " << vr.error << "\n";
    CHECK("total latency finite", sol.total_latency() < 1e17);
    CHECK("all ops scheduled", sol.num_steps() >= 1);
    std::cout << "    steps=" << sol.num_steps() << " latency=" << sol.total_latency() << "\n";
}

void test_from_partition_diamond_valid() {
    std::cout << "--- test_from_partition_diamond_valid ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = best_initial(p, d, &cache);
    part.finalize();

    auto sol = Solution::from_partition(p, d, part);
    auto vr = sol.validate();
    CHECK("diamond solution valid", vr.valid);
    CHECK("diamond latency finite", sol.total_latency() < 1e17);
    std::cout << "    steps=" << sol.num_steps() << " latency=" << sol.total_latency() << "\n";
}

// ============================================================================
// 5. Edge cases
// ============================================================================

void test_single_group_partition() {
    std::cout << "--- test_single_group_partition ---\n";
    // Fully fused: one group covers all ops.
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2};
    part.groups[0].cost = cache.evaluate({0,1,2},p,d);
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();
    part.finalize();

    auto dfs_r  = dfs_ordering(part);
    auto beam_r = beam_search_ordering(part, 5);

    CHECK("dfs single group: 1 step",  dfs_r.order.size()  == 1);
    CHECK("beam single group: 1 step", beam_r.order.size() == 1);
    CHECK("dfs retain_per_step sized", dfs_r.retain_per_step.size() == 1);

    auto sol = Solution::from_partition(p, d, std::move(part));
    CHECK("single group solution valid", sol.validate().valid);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // 1. dfs_ordering
    test_dfs_topo_chain3();
    test_dfs_topo_diamond4();
    test_dfs_retain_per_step_populated();
    test_dfs_fused_partition();

    // 2. beam_search_ordering
    test_beam_topo_chain3();
    test_beam_topo_diamond4();
    test_beam_total_latency_set();
    test_beam_width_one_vs_five();

    // 3. random_ordering
    test_random_topo_chain3();
    test_random_topo_diamond4();
    test_random_diversity();

    // 4. steps_from_ordering / from_partition
    test_steps_from_dfs_ordering_feasible();
    test_steps_from_beam_ordering_feasible();
    test_beam_le_dfs_latency();
    test_from_partition_valid_solution();
    test_from_partition_diamond_valid();

    // 5. Edge cases
    test_single_group_partition();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}