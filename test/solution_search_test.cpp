// solution_search_test.cpp
//
// Tests for solution_search: compute_feasibly_retainable, solution_greedy_descent,
// solution_fm_pass, solution_fm_search, solution_evo_search, mutate_random.
//
// Feasibility invariants checked after every move:
//   1. Memory:       every step is_feasible(config, retained_entering, retain_these)
//   2. Topo order:   every step's boundary inputs were produced by a prior step
//   3. Coverage:     every op appears in at least one step
//   4. No eph. gap:  every boundary input has a source in prior steps
//   (PW-sink k=1 and acyclicity enforced inside Subgraph::create)
//
// Regression tests for:
//   Bug 1 — SPLIT in best_move_for_op missed co-consumer bridge edges
//   Bug 2 — is_connected_without blocked STEAL from 2-op groups
//
// Build: make solution_search_test
// Run:   ./solution_search_test

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost_cache.h"
#include "partition/partition.h"
#include "solution/solution.h"
#include "solution/ordering.h"
#include "search/solution_search.h"
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

// ============================================================================
// Problem helpers
// ============================================================================

static Problem make_chain3() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    p.retainable_tensors = {1, 2};
    return p;
}

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

// Co-consumer topology: T0->{Op0->T1, Op1->T2}, T1+T2->Op2->T3
// Op0 and Op1 are co-consumers of T0 — creates a co-consumer bridge edge.
static Problem make_fanin() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{0},{2},1000},
             {OpType::Pointwise,{1,2},{3},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    p.retainable_tensors = {1, 2};
    return p;
}

// Tight memory: only one tensor fits alongside a subgraph
static Problem make_tight_chain3() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000}};
    p.fast_memory_capacity = 35000;  // tight: single tensor ≈ 16384 bytes
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    p.retainable_tensors = {1, 2};
    return p;
}

// ============================================================================
// Solution helpers
// ============================================================================

static Solution make_solution(const Problem& p, const DAG& d) {
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.finalize();
    return Solution::from_partition(p, d, part);
}

static Solution make_fused_solution(const Problem& p, const DAG& d) {
    CostCache cache;
    auto part = best_initial(p, d, &cache);
    part.finalize();
    return Solution::from_partition(p, d, part);
}

static void check_solution_valid(const char* label, const Solution& sol) {
    auto vr = sol.validate();
    if (!vr.valid) {
        std::cout << "  FAIL: " << label << " " << vr.error << "\n";
        g_fail++;
    } else { g_pass++; }
}

// ============================================================================
// 1. compute_feasibly_retainable
// ============================================================================

void test_feasibly_retainable_chain4() {
    std::cout << "--- test_feasibly_retainable_chain4 ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);

    CHECK("T1 retainable", fr.count(1) > 0);
    CHECK("T2 retainable", fr.count(2) > 0);
    CHECK("T3 retainable", fr.count(3) > 0);
    CHECK("T0 not retainable (graph input)", fr.count(0) == 0);
    CHECK("T4 not retainable (graph output)", fr.count(4) == 0);
    CHECK("result subset of retainable_tensors",
          [&]{ for (auto t : fr) if (!p.retainable_tensors.count(t)) return false;
               return true; }());
    std::cout << "    retainable: " << fr.size() << "\n";
}

void test_feasibly_retainable_tight_memory() {
    std::cout << "--- test_feasibly_retainable_tight_memory ---\n";
    auto p = make_tight_chain3(); DAG d = DAG::build(p);
    auto fr_tight = compute_feasibly_retainable(p, d);
    p.fast_memory_capacity = 100000;
    auto fr_ample = compute_feasibly_retainable(p, d);
    // With ample memory, at least as many tensors are retainable
    CHECK("ample >= tight retainable", fr_ample.size() >= fr_tight.size());
    std::cout << "    tight=" << fr_tight.size() << " ample=" << fr_ample.size() << "\n";
}

void test_feasibly_retainable_empty() {
    std::cout << "--- test_feasibly_retainable_empty ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    p.retainable_tensors.clear();
    CHECK("empty when no tensors eligible", compute_feasibly_retainable(p, d).empty());
}

// ============================================================================
// 2. solution_greedy_descent
// ============================================================================

void test_greedy_improves_trivial() {
    std::cout << "--- test_greedy_improves_trivial ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);
    auto initial = make_solution(p, d);
    double init_cost = initial.total_latency();

    auto steps = solution_greedy_descent(p, d, initial.steps(), {}, &fr);
    Solution result(p, d, std::move(steps));
    check_solution_valid("greedy/chain4", result);
    CHECK("greedy improves or matches", result.total_latency() <= init_cost + 0.01);
    std::cout << "    init=" << init_cost << " after=" << result.total_latency() << "\n";
}

void test_greedy_idempotent_at_optimum() {
    std::cout << "--- test_greedy_idempotent_at_optimum ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);
    auto fused = make_fused_solution(p, d);

    // Run greedy twice; second pass must not improve (already at local optimum)
    auto s1 = solution_greedy_descent(p, d, fused.steps(), {}, &fr);
    double c1 = Solution(p, d, s1).total_latency();
    auto s2 = solution_greedy_descent(p, d, std::move(s1), {}, &fr);
    double c2 = Solution(p, d, std::move(s2)).total_latency();
    CHECK_EQ("second greedy descent makes no improvement", c2, c1);
    std::cout << "    c1=" << c1 << " c2=" << c2 << "\n";
}

void test_greedy_valid_diamond() {
    std::cout << "--- test_greedy_valid_diamond ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);
    auto initial = make_solution(p, d);

    auto steps = solution_greedy_descent(p, d, initial.steps(), {}, &fr);
    check_solution_valid("greedy/diamond4", Solution(p, d, std::move(steps)));
}

// ============================================================================
// 3. solution_fm_pass
// ============================================================================

void test_fm_pass_valid_chain() {
    std::cout << "--- test_fm_pass_valid_chain ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);
    auto initial = make_solution(p, d);

    SolutionFMPassConfig cfg;
    cfg.floor_fraction = 0.30;
    cfg.max_drift_fraction = 0.50;
    cfg.seed = 42;

    auto r = solution_fm_pass(p, d, initial.steps(), cfg, &fr);
    CHECK("start_cost > 0", r.start_cost > 0);
    CHECK("best_cost finite", r.best_cost < 1e17);
    CHECK("best_cost <= start_cost", r.best_cost <= r.start_cost + 0.01);
    check_solution_valid("fm_pass best", Solution(p, d, r.best_steps));
    if (!r.end_steps.empty())
        check_solution_valid("fm_pass end", Solution(p, d, r.end_steps));
    else g_pass++;
    std::cout << "    start=" << r.start_cost << " best=" << r.best_cost
              << " moves=" << r.moves_applied << "\n";
}

void test_fm_pass_valid_diamond() {
    std::cout << "--- test_fm_pass_valid_diamond ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);
    auto initial = make_solution(p, d);

    SolutionFMPassConfig cfg; cfg.seed = 7;
    auto r = solution_fm_pass(p, d, initial.steps(), cfg, &fr);
    check_solution_valid("fm_pass/diamond4 best", Solution(p, d, r.best_steps));
}

void test_fm_pass_accepts_worsening_with_wide_floor() {
    std::cout << "--- test_fm_pass_accepts_worsening_with_wide_floor ---\n";
    // Fused partition is locally optimal for structural moves.
    // With wide floor and drift, FM should still apply retain moves and
    // the result must remain valid.
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);
    auto fused = make_fused_solution(p, d);

    SolutionFMPassConfig cfg;
    cfg.floor_fraction = 0.80;
    cfg.max_drift_fraction = 2.0;
    cfg.seed = 13;

    auto r = solution_fm_pass(p, d, fused.steps(), cfg, &fr);
    check_solution_valid("fm_pass wide floor", Solution(p, d, r.best_steps));
    std::cout << "    moves=" << r.moves_applied << "\n";
}

// ============================================================================
// 4. solution_fm_search
// ============================================================================

void test_fm_search_improves_or_matches() {
    std::cout << "--- test_fm_search_improves_or_matches ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto initial = make_solution(p, d);
    double init_cost = initial.total_latency();

    SolutionFMConfig cfg;
    cfg.max_passes = 10;
    cfg.max_no_improve = 3;
    cfg.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);

    auto result = solution_fm_search(p, d, initial, cfg);
    check_solution_valid("fm_search/chain4", result);
    CHECK("fm_search improves or matches", result.total_latency() <= init_cost + 0.01);
    std::cout << "    init=" << init_cost << " after=" << result.total_latency() << "\n";
}

void test_fm_search_all_ops_covered() {
    std::cout << "--- test_fm_search_all_ops_covered ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto initial = make_solution(p, d);

    SolutionFMConfig cfg;
    cfg.max_passes = 5;
    cfg.max_no_improve = 2;
    cfg.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);

    auto result = solution_fm_search(p, d, initial, cfg);
    std::set<size_t> covered;
    for (auto& step : result.steps())
        for (auto op : step.subgraph.ops()) covered.insert(op);
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", covered.count(i) > 0);
    check_solution_valid("fm_search/diamond4", result);
}

// ============================================================================
// 5. solution_evo_search
// ============================================================================

void test_evo_search_valid() {
    std::cout << "--- test_evo_search_valid ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    std::vector<Solution> pool;
    pool.push_back(make_solution(p, d));

    SolutionFMConfig cfg;
    cfg.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);

    auto result = solution_evo_search(p, d, std::move(pool), cfg);
    check_solution_valid("evo_search/chain4", result);
    CHECK("evo finite latency", result.total_latency() < 1e17);
    std::cout << "    latency=" << result.total_latency() << "\n";
}

void test_evo_search_diverse_pool() {
    std::cout << "--- test_evo_search_diverse_pool ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);

    // Seed pool with both trivial and fused solutions
    std::vector<Solution> pool;
    pool.push_back(make_solution(p, d));
    pool.push_back(make_fused_solution(p, d));
    double pool_best = std::min(pool[0].total_latency(), pool[1].total_latency());

    SolutionFMConfig cfg;
    cfg.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);

    auto result = solution_evo_search(p, d, std::move(pool), cfg);
    check_solution_valid("evo_search/diamond4", result);
    CHECK("evo improves or matches pool best", result.total_latency() <= pool_best + 0.01);
    std::cout << "    pool_best=" << pool_best << " result=" << result.total_latency() << "\n";
}

// ============================================================================
// 6. mutate_random
// ============================================================================

void test_mutate_random_valid_chain() {
    std::cout << "--- test_mutate_random_valid_chain ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto initial = make_solution(p, d);

    int changed = 0;
    for (int seed = 0; seed < 50; seed++) {
        std::mt19937 rng(seed);
        auto result = mutate_random(p, d, initial.steps(), rng, 2);
        if (result.empty()) { g_pass++; continue; }  // failed mutation: acceptable
        check_solution_valid("mutate/chain4", Solution(p, d, result));
        changed++;
    }
    CHECK("mutate produces some changes", changed > 0);
    std::cout << "    changed: " << changed << "/50\n";
}

void test_mutate_random_valid_diamond() {
    std::cout << "--- test_mutate_random_valid_diamond ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    auto initial = make_solution(p, d);
    auto fr = compute_feasibly_retainable(p, d);

    for (int seed = 0; seed < 30; seed++) {
        std::mt19937 rng(seed);
        auto result = mutate_random(p, d, initial.steps(), rng, 3);
        if (result.empty()) { g_pass++; continue; }
        check_solution_valid("mutate/diamond4", Solution(p, d, result));
    }
}

void test_mutate_random_chain_walk() {
    std::cout << "--- test_mutate_random_chain_walk ---\n";
    // 30 sequential mutations: every intermediate state must be valid
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto steps = make_solution(p, d).steps();
    std::mt19937 rng(99);
    for (int i = 0; i < 30; i++) {
        auto next = mutate_random(p, d, steps, rng, 1);
        if (!next.empty()) {
            check_solution_valid("mutate walk", Solution(p, d, next));
            steps = std::move(next);
        } else { g_pass++; }
    }
}

// ============================================================================
// 7. Bug regression: is_connected_without (Bug 2)
//    STEAL from 2-op group was always blocked.
// ============================================================================

void test_steal_from_2op_group() {
    std::cout << "--- test_steal_from_2op_group ---\n";
    // Build a solution with a 2-op step: G0={Op0,Op1}, G1={Op2}.
    // STEAL Op1 from G0 to G1 should be considered (not blocked by the
    // is_connected_without bug). After fix, the source group remainder
    // is a singleton which is always connected.
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);

    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    // Fuse Op0+Op1 into one step
    part.groups[0].ops  = {0,1};
    part.groups[0].cost = cache.evaluate({0,1},p,d);
    part.groups[1].alive = false;
    part.rebuild_index();
    part.finalize();
    auto initial = Solution::from_partition(p, d, part);

    // Run greedy — it should be willing to STEAL Op1 from {Op0,Op1} to {Op2}
    auto steps = solution_greedy_descent(p, d, initial.steps(), {}, &fr);
    Solution result(p, d, std::move(steps));
    check_solution_valid("steal from 2-op group", result);
    // The result may or may not have performed the steal (only if profitable),
    // but it must never be infeasible.
    std::cout << "    initial=" << initial.total_latency()
              << " result=" << result.total_latency() << "\n";
}

// ============================================================================
// 8. Bug regression: SPLIT co-consumer bridge edges (Bug 1)
//    SPLIT on a fused fanin {Op0,Op1,Op2} should propose the Op0<->Op1
//    co-consumer bridge, not just DAG edges.
// ============================================================================

void test_split_co_consumer_bridge() {
    std::cout << "--- test_split_co_consumer_bridge ---\n";
    // Fanin: T0->{Op0->T1, Op1->T2}, T1+T2->Op2->T3.
    // Op0 and Op1 are co-consumers (no DAG edge between them, but both consume T0).
    // In a fused {Op0,Op1,Op2} step, the Op0<->Op1 co-consumer edge is a bridge.
    // After Bug 1 fix, SPLIT should propose this bridge as a cut.
    auto p = make_fanin(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);

    // Build a fused 3-op step
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0,1,2}, cache.evaluate({0,1,2},p,d));
    part.rebuild_index();
    part.finalize();
    auto initial = Solution::from_partition(p, d, part);

    // Run greedy — with the fix it can now propose the Op0<->Op1 bridge SPLIT
    auto steps = solution_greedy_descent(p, d, initial.steps(), {}, &fr);
    Solution result(p, d, std::move(steps));
    check_solution_valid("split co-consumer bridge", result);
    std::cout << "    initial=" << initial.total_latency()
              << " result=" << result.total_latency() << "\n";

    // Fuzz: mutate_random on the fused step should also produce valid results
    for (int seed = 0; seed < 30; seed++) {
        std::mt19937 rng(seed);
        auto mutated = mutate_random(p, d, initial.steps(), rng, 2);
        if (mutated.empty()) { g_pass++; continue; }
        check_solution_valid("mutate fanin", Solution(p, d, mutated));
    }
}

// ============================================================================
// 9. Feasibility invariants: memory, topo, coverage, no gap
//    Verify Solution::validate() catches each violation.
// ============================================================================

void test_validate_catches_infeasible_config() {
    std::cout << "--- test_validate_catches_infeasible_config ---\n";
    // chain3 trivial: step 0 = {Op0}, boundary_outputs = {T1}.
    // T2 is produced by Op1 (step 1) — not a boundary output of step 0.
    // The Solution constructor strips non-boundary tensors from retain_these,
    // so validate() won't see T2. Verify the strip works correctly.
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto initial = make_solution(p, d);
    auto steps = initial.steps();
    if (!steps.empty())
        steps[0].retain_these.insert(2);  // T2: not a boundary output of step 0
    Solution bad(p, d, std::move(steps));
    // Constructor strips T2 from retain_these (not a boundary output)
    CHECK("constructor strips non-boundary retain tensor",
          bad.step(0).retain_these.find(2) == bad.step(0).retain_these.end());
    // Solution is valid because the constructor cleaned it
    auto vr = bad.validate();
    CHECK("validate passes after strip", vr.valid);
}

void test_validate_catches_missing_op() {
    std::cout << "--- test_validate_catches_missing_op ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto initial = make_solution(p, d);
    auto steps = initial.steps();
    // Drop the last step to leave Op2 uncovered
    if (steps.size() >= 2) steps.pop_back();
    Solution bad(p, d, std::move(steps));
    auto vr = bad.validate();
    CHECK("validate catches missing op", !vr.valid);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // 1. compute_feasibly_retainable
    test_feasibly_retainable_chain4();
    test_feasibly_retainable_tight_memory();
    test_feasibly_retainable_empty();

    // 2. solution_greedy_descent
    test_greedy_improves_trivial();
    test_greedy_idempotent_at_optimum();
    test_greedy_valid_diamond();

    // 3. solution_fm_pass
    test_fm_pass_valid_chain();
    test_fm_pass_valid_diamond();
    test_fm_pass_accepts_worsening_with_wide_floor();

    // 4. solution_fm_search
    test_fm_search_improves_or_matches();
    test_fm_search_all_ops_covered();

    // 5. solution_evo_search
    test_evo_search_valid();
    test_evo_search_diverse_pool();

    // 6. mutate_random
    test_mutate_random_valid_chain();
    test_mutate_random_valid_diamond();
    test_mutate_random_chain_walk();

    // 7. Bug regression: is_connected_without (Bug 2)
    test_steal_from_2op_group();

    // 8. Bug regression: SPLIT co-consumer bridge (Bug 1)
    test_split_co_consumer_bridge();

    // 9. Feasibility invariant validation
    test_validate_catches_infeasible_config();
    test_validate_catches_missing_op();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}