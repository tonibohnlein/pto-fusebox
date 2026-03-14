// solver_test.cpp
//
// Integration tests for the full three-phase solver pipeline.
//
// Each test exercises a specific problem topology through solve() or through
// individual pipeline phases (parallel_search → solution building → solution
// search), verifying:
//   1. The returned Solution passes Solution::validate() — coverage, memory
//      feasibility, topological order, no ephemeral gap
//   2. Total latency is finite and ≤ the trivial (no-fusion, no-retain) baseline
//   3. All ops appear in the final schedule
//
// These tests use short deadlines (200–500ms) so they run quickly in CI.
// Full-quality results require the normal time budgets from main.cpp.
//
// Build: make solver_test
// Run:   ./solver_test

#include "core/types.h"
#include "core/dag.h"
#include "core/cost_cache.h"
#include "partition/partition.h"
#include "init/init_strategies.h"
#include "search/parallel_search.h"
#include "search/solution_search.h"
#include "search/local_search.h"
#include "solution/solution.h"
#include "solution/ordering.h"
#include "core/dag.h"
#include "pipeline/solver.h"
#include "search/verbose.h"
#include <chrono>
#include <iostream>
#include <set>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_LE(const char* l, double got, double exp) {
    if (got <= exp + 0.5) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << got << " exp≤" << exp << "\n"; }
}
static void CHECK_EQ_S(const char* l, size_t got, size_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << got << " exp=" << exp << "\n"; }
}

using Clock = std::chrono::steady_clock;

// ============================================================================
// Problem helpers
// ============================================================================

// 6-op PW chain with tight memory (forces fusion decisions)
static Problem make_chain6_tight() {
    Problem p;
    for (int i = 0; i <= 6; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 6; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},500});
    p.fast_memory_capacity = 40000;   // ~2.4 tensors fit
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    p.retainable_tensors = {1,2,3,4,5};
    return p;
}

// MatMul + PW chain (tests k-splitting, PW-sink rule)
static Problem make_mm_pw_chain() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{2},{3},500},
             {OpType::Pointwise,{3},{4},500}};
    p.fast_memory_capacity = 60000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    p.retainable_tensors = {2, 3};
    return p;
}

// Diamond with fan-out from a shared tensor (tests ephemeral gap prevention)
static Problem make_diamond() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{1},{3},1000},
             {OpType::Pointwise,{2,3},{4},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    p.retainable_tensors = {1,2,3};
    return p;
}

// Parallel MatMuls sharing RHS (co-consumer retention opportunity)
static Problem make_parallel_mm() {
    Problem p;
    for (int i = 0; i < 9; i++) p.tensors.push_back({128,128});
    // T0 = shared RHS; T1..T4 = LHS; T5..T8 = outputs
    p.ops = {{OpType::MatMul,{1,0},{5},1000},
             {OpType::MatMul,{2,0},{6},1000},
             {OpType::MatMul,{3,0},{7},1000},
             {OpType::MatMul,{4,0},{8},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    p.retainable_tensors = {0};   // shared RHS is the retention opportunity
    return p;
}

// No retainable tensors — Phase 3 must be skipped
static Problem make_no_retain() {
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    // retainable_tensors left empty
    return p;
}

// ============================================================================
// Trivial baseline: one step per op, no fusion, no retention
// ============================================================================
static double trivial_latency(const Problem& p, const DAG& d) {
    CostCache cache;
    Partition part = Partition::trivial(p, d);
    part.cache = &cache;
    return part.total_cost();
}

// ============================================================================
// Full validation helper
// ============================================================================
static void check_solution(const char* label, const Solution& sol,
                           const Problem& p, double baseline) {
    auto vr = sol.validate();
    if (!vr.valid)
        std::cout << "  FAIL detail: " << vr.error << "\n";
    CHECK((std::string(label) + ": valid").c_str(), vr.valid);
    CHECK((std::string(label) + ": finite latency").c_str(),
          sol.total_latency() < 1e17);
    CHECK_LE((std::string(label) + ": ≤ trivial").c_str(),
             sol.total_latency(), baseline);

    // Coverage
    std::set<size_t> covered;
    for (size_t i = 0; i < sol.num_steps(); i++)
        for (auto op : sol.step(i).subgraph.ops()) covered.insert(op);
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK((std::string(label) + ": op covered").c_str(), covered.count(i) > 0);
}

// ============================================================================
// 1. solve() on each problem topology — end-to-end
// ============================================================================

void test_solve_chain6() {
    std::cout << "--- test_solve_chain6 ---\n";
    auto p = make_chain6_tight(); DAG d = DAG::build(p);
    double base = trivial_latency(p, d);
    auto deadline = Clock::now() + std::chrono::milliseconds(500);
    auto sol = solve(p, d, deadline);
    check_solution("solve/chain6", sol, p, base);
    std::cout << "    trivial=" << base << " solved=" << sol.total_latency()
              << " steps=" << sol.num_steps() << "\n";
}

void test_solve_mm_pw_chain() {
    std::cout << "--- test_solve_mm_pw_chain ---\n";
    auto p = make_mm_pw_chain(); DAG d = DAG::build(p);
    double base = trivial_latency(p, d);
    auto deadline = Clock::now() + std::chrono::milliseconds(500);
    auto sol = solve(p, d, deadline);
    check_solution("solve/mm_pw", sol, p, base);
    // Verify PW-sink k=1 is respected in all steps with MM+PW
    for (size_t i = 0; i < sol.num_steps(); i++) {
        bool has_mm = false, has_pw = false;
        for (auto op : sol.step(i).subgraph.ops()) {
            if (p.ops[op].type == OpType::MatMul) has_mm = true;
            else has_pw = true;
        }
        if (has_mm && has_pw)
            CHECK("MM+PW step: k=1", sol.step(i).config.k == 1);
    }
    std::cout << "    trivial=" << base << " solved=" << sol.total_latency()
              << " steps=" << sol.num_steps() << "\n";
}

void test_solve_diamond() {
    std::cout << "--- test_solve_diamond ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    double base = trivial_latency(p, d);
    auto deadline = Clock::now() + std::chrono::milliseconds(500);
    auto sol = solve(p, d, deadline);
    check_solution("solve/diamond", sol, p, base);
    std::cout << "    trivial=" << base << " solved=" << sol.total_latency()
              << " steps=" << sol.num_steps() << "\n";
}

void test_solve_parallel_mm() {
    std::cout << "--- test_solve_parallel_mm ---\n";
    auto p = make_parallel_mm(); DAG d = DAG::build(p);
    double base = trivial_latency(p, d);
    auto deadline = Clock::now() + std::chrono::milliseconds(500);
    auto sol = solve(p, d, deadline);
    check_solution("solve/parallel_mm", sol, p, base);
    std::cout << "    trivial=" << base << " solved=" << sol.total_latency()
              << " steps=" << sol.num_steps() << "\n";
}

void test_solve_no_retain() {
    std::cout << "--- test_solve_no_retain ---\n";
    // When retainable_tensors is empty, Phase 3 must be skipped and the
    // solver still returns a valid solution.
    auto p = make_no_retain(); DAG d = DAG::build(p);
    double base = trivial_latency(p, d);
    auto deadline = Clock::now() + std::chrono::milliseconds(300);
    auto sol = solve(p, d, deadline);
    check_solution("solve/no_retain", sol, p, base);
    // All retain sets must be empty since no tensor is retainable
    for (size_t i = 0; i < sol.num_steps(); i++)
        CHECK("no_retain: empty retain", sol.step(i).retain_these.empty());
    std::cout << "    trivial=" << base << " solved=" << sol.total_latency()
              << " steps=" << sol.num_steps() << "\n";
}

// ============================================================================
// 2. Phase 1 only: parallel_search + solution building
// ============================================================================

void test_phase1_partition_pool() {
    std::cout << "--- test_phase1_partition_pool ---\n";
    auto p = make_chain6_tight(); DAG d = DAG::build(p);
    double base = trivial_latency(p, d);

    ParallelConfig cfg;
    cfg.num_threads = 2;
    cfg.pool_size = 4;
    cfg.fm.deadline = Clock::now() + std::chrono::milliseconds(300);
    auto pool = parallel_search(p, d, cfg);

    CHECK("pool non-empty", !pool.empty());
    CHECK("pool best < trivial", pool[0].total_cost() < base + 0.01);
    // Null caches (stale after parallel_search returns)
    for (auto& pt : pool) pt.cache = nullptr;

    // Build solutions from the pool
    for (auto& pt : pool) {
        auto sol = Solution::from_partition(p, d, pt);
        auto vr = sol.validate();
        CHECK("pool solution valid", vr.valid);
        CHECK_LE("pool solution ≤ trivial", sol.total_latency(), base);
    }
    std::cout << "    pool=" << pool.size() << " best_partition=" << pool[0].total_cost() << "\n";
}

// ============================================================================
// 3. Phase 2: ordering algorithms
// ============================================================================

void test_phase2_dfs_vs_beam() {
    std::cout << "--- test_phase2_dfs_vs_beam ---\n";
    auto p = make_chain6_tight(); DAG d = DAG::build(p);

    CostCache cache;
    auto part = best_initial(p, d, &cache);
    part = greedy_descent(std::move(part));
    part.finalize();

    auto dfs_r  = dfs_ordering(part);
    auto beam_r = beam_search_ordering(part, 10);

    auto dfs_steps  = Solution::steps_from_ordering(p, d, part, dfs_r);
    auto beam_steps = Solution::steps_from_ordering(p, d, part, beam_r);

    Solution dfs_sol (p, d, std::move(dfs_steps));
    Solution beam_sol(p, d, std::move(beam_steps));

    CHECK("dfs ordering valid",  dfs_sol.validate().valid);
    CHECK("beam ordering valid", beam_sol.validate().valid);
    CHECK_LE("beam ≤ dfs", beam_sol.total_latency(), dfs_sol.total_latency());
    std::cout << "    dfs=" << dfs_sol.total_latency()
              << " beam=" << beam_sol.total_latency() << "\n";
}

void test_phase2_random_ordering_valid() {
    std::cout << "--- test_phase2_random_ordering_valid ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);

    CostCache cache;
    auto part = best_initial(p, d, &cache);
    part.finalize();

    for (int seed = 0; seed < 10; seed++) {
        std::mt19937 rng(seed);
        auto res   = random_ordering(part, fr, rng);
        auto steps = Solution::steps_from_ordering(p, d, part, res);
        Solution sol(p, d, std::move(steps));
        CHECK("random ordering valid", sol.validate().valid);
    }
}

// ============================================================================
// 4. Phase 3: solution search
// ============================================================================

void test_phase3_greedy_improves() {
    std::cout << "--- test_phase3_greedy_improves ---\n";
    auto p = make_chain6_tight(); DAG d = DAG::build(p);
    auto fr = compute_feasibly_retainable(p, d);

    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    auto initial = Solution::from_partition(p, d, std::move(part));
    double init_cost = initial.total_latency();

    auto result_steps = solution_greedy_descent(p, d, initial.steps(), {}, &fr);
    Solution result(p, d, std::move(result_steps));
    CHECK("greedy valid", result.validate().valid);
    CHECK_LE("greedy ≤ initial", result.total_latency(), init_cost);
    std::cout << "    initial=" << init_cost << " after=" << result.total_latency() << "\n";
}

void test_phase3_fm_search() {
    std::cout << "--- test_phase3_fm_search ---\n";
    auto p = make_chain6_tight(); DAG d = DAG::build(p);

    CostCache cache;
    auto part = best_initial(p, d, &cache);
    part = greedy_descent(std::move(part));
    auto initial = Solution::from_partition(p, d, std::move(part));
    double init_cost = initial.total_latency();

    SolutionFMConfig cfg;
    cfg.max_passes = 5;
    cfg.max_no_improve = 2;
    cfg.deadline = Clock::now() + std::chrono::milliseconds(300);

    auto result = solution_fm_search(p, d, initial, cfg);
    CHECK("fm_search valid", result.validate().valid);
    CHECK_LE("fm_search ≤ initial", result.total_latency(), init_cost);
    for (size_t i = 0; i < p.num_ops(); i++) {
        std::set<size_t> cov;
        for (size_t j = 0; j < result.num_steps(); j++)
            for (auto op : result.step(j).subgraph.ops()) cov.insert(op);
        CHECK("all ops covered", cov.count(i) > 0);
    }
    std::cout << "    initial=" << init_cost << " after=" << result.total_latency() << "\n";
}

void test_phase3_evo_search() {
    std::cout << "--- test_phase3_evo_search ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);

    // Seed pool with two different solutions
    std::vector<Solution> pool;
    {
        CostCache cache;
        auto part = Partition::trivial(p, d); part.cache = &cache;
        pool.push_back(Solution::from_partition(p, d, std::move(part)));
    }
    {
        CostCache cache;
        auto part = best_initial(p, d, &cache);
        pool.push_back(Solution::from_partition(p, d, std::move(part)));
    }
    double pool_best = std::min(pool[0].total_latency(), pool[1].total_latency());

    SolutionFMConfig cfg;
    cfg.deadline = Clock::now() + std::chrono::milliseconds(400);

    auto result = solution_evo_search(p, d, std::move(pool), cfg);
    CHECK("evo_search valid", result.validate().valid);
    CHECK_LE("evo_search ≤ pool best", result.total_latency(), pool_best);
    std::cout << "    pool_best=" << pool_best << " after=" << result.total_latency() << "\n";
}

// ============================================================================
// 5. Feasibility invariants: PW-sink k=1, memory, no ephemeral gap
// ============================================================================

void test_invariant_pw_sink_k1() {
    std::cout << "--- test_invariant_pw_sink_k1 ---\n";
    // Every step in the final solution that has a MM feeding a PW sink
    // must have k=1. Verify this on the MM+PW chain problem.
    auto p = make_mm_pw_chain(); DAG d = DAG::build(p);
    auto deadline = Clock::now() + std::chrono::milliseconds(400);
    auto sol = solve(p, d, deadline);

    bool found_mm_pw = false;
    bool k1_ok = true;
    for (size_t i = 0; i < sol.num_steps(); i++) {
        bool has_mm = false, has_pw_sink = false;
        for (auto op : sol.step(i).subgraph.ops()) {
            if (p.ops[op].type == OpType::MatMul) has_mm = true;
            else has_pw_sink = true;
        }
        if (has_mm && has_pw_sink) {
            found_mm_pw = true;
            if (sol.step(i).config.k != 1) k1_ok = false;
        }
    }
    // If MM and PW were fused into the same step, k must be 1
    if (found_mm_pw)
        CHECK("PW-sink k=1 in fused MM+PW step", k1_ok);
    else {
        g_pass++;  // not fused — constraint trivially satisfied
        std::cout << "    MM and PW not fused (k=1 constraint trivially met)\n";
    }
}

void test_invariant_memory_feasibility() {
    std::cout << "--- test_invariant_memory_feasibility ---\n";
    auto p = make_chain6_tight(); DAG d = DAG::build(p);
    auto deadline = Clock::now() + std::chrono::milliseconds(400);
    auto sol = solve(p, d, deadline);

    // Solution::validate() already checks is_feasible per step.
    // Here we additionally verify each step's cost is finite.
    CHECK("solution valid (memory check)", sol.validate().valid);
    for (size_t i = 0; i < sol.num_steps(); i++)
        CHECK("step cost finite", sol.step_latency(i) < 1e17);
}

void test_invariant_no_ephemeral_gap() {
    std::cout << "--- test_invariant_no_ephemeral_gap ---\n";
    // Diamond topology has T1 with two consumers. If Op0 and Op1 are fused,
    // T1 is ephemeral in that step — the gap check must ensure Op2 (external)
    // can still obtain T1 from another step's boundary output.
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto deadline = Clock::now() + std::chrono::milliseconds(400);
    auto sol = solve(p, d, deadline);

    // validate() checks the ephemeral gap at solution level
    auto vr = sol.validate();
    CHECK("no ephemeral gap (diamond)", vr.valid);
    if (!vr.valid) std::cout << "    error: " << vr.error << "\n";

    // Direct check: every step's boundary inputs must be produced by a prior step
    std::set<size_t> available(d.graph_inputs.begin(), d.graph_inputs.end());
    bool topo_ok = true;
    for (size_t i = 0; i < sol.num_steps(); i++) {
        for (auto t : sol.step(i).subgraph.boundary_inputs()) {
            if (!available.count(t)) { topo_ok = false; break; }
        }
        for (auto op : sol.step(i).subgraph.ops())
            for (auto t : p.ops[op].outputs) available.insert(t);
    }
    CHECK("topological order maintained", topo_ok);
}

// ============================================================================
// 6. Edge cases
// ============================================================================

void test_single_op_problem() {
    std::cout << "--- test_single_op_problem ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    DAG d = DAG::build(p);
    auto deadline = Clock::now() + std::chrono::milliseconds(200);
    auto sol = solve(p, d, deadline);
    CHECK("single op: valid", sol.validate().valid);
    CHECK_EQ_S("single op: 1 step", sol.num_steps(), 1);
    std::cout << "    latency=" << sol.total_latency() << "\n";
}

void test_two_independent_ops() {
    std::cout << "--- test_two_independent_ops ---\n";
    // T0->Op0->T1, T0->Op1->T2 — co-consumers, no data dependency between them
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{0},{2},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    DAG d = DAG::build(p);
    auto deadline = Clock::now() + std::chrono::milliseconds(200);
    auto sol = solve(p, d, deadline);
    CHECK("two independent: valid", sol.validate().valid);

    std::set<size_t> covered;
    for (size_t i = 0; i < sol.num_steps(); i++)
        for (auto op : sol.step(i).subgraph.ops()) covered.insert(op);
    CHECK("Op0 covered", covered.count(0) > 0);
    CHECK("Op1 covered", covered.count(1) > 0);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    g_verbose = false;

    // 1. End-to-end solve()
    test_solve_chain6();
    test_solve_mm_pw_chain();
    test_solve_diamond();
    test_solve_parallel_mm();
    test_solve_no_retain();

    // 2. Phase 1: partition pool
    test_phase1_partition_pool();

    // 3. Phase 2: ordering
    test_phase2_dfs_vs_beam();
    test_phase2_random_ordering_valid();

    // 4. Phase 3: solution search
    test_phase3_greedy_improves();
    test_phase3_fm_search();
    test_phase3_evo_search();

    // 5. Feasibility invariants
    test_invariant_pw_sink_k1();
    test_invariant_memory_feasibility();
    test_invariant_no_ephemeral_gap();

    // 6. Edge cases
    test_single_op_problem();
    test_two_independent_ops();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}