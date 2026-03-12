// solution_search_test.cpp — Tests for solution-level FM search

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "solution/solution.h"
#include "search/solution_search.h"
#include "postopt/post_opt.h"
#include <iostream>
#include <cmath>
#include <set>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++; else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double g, double e, double t = 0.1) {
    if (std::abs(g - e) < t) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}

// Chain of 4 PW: T0->Op0->T1->Op1->T2->Op2->T3->Op3->T4
static Problem make_chain4_pw() {
    Problem p;
    for (int i = 0; i < 5; i++) p.tensors.push_back({128, 128});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    for (size_t i = 0; i < p.tensors.size(); i++)
        if (p.tensors[i].size() <= p.fast_memory_capacity)
            p.retainable_tensors.insert(i);
    return p;
}

// Chain with retainable shared tensor
// T0->Op0->T1, T1->Op1->T2, T1->Op2->T3 (T1 shared between Op1 and Op2)
static Problem make_shared_input() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{1},{3},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    for (size_t i = 0; i < p.tensors.size(); i++)
        if (p.tensors[i].size() <= p.fast_memory_capacity)
            p.retainable_tensors.insert(i);
    return p;
}

// Build a solution from individual op groups
static Solution make_singleton_solution(const Problem& p, const DAG& d) {
    std::vector<ScheduleStep> steps;
    for (size_t i = 0; i < p.num_ops(); i++) {
        auto sg = Subgraph::create(p, d, {i});
        if (!sg) continue;
        auto c = sg->best_cost();
        if (!c.feasible) continue;
        ScheduleStep ss;
        ss.subgraph = std::move(*sg);
        ss.config = c.config;
        steps.push_back(std::move(ss));
    }
    return Solution(p, d, std::move(steps));
}

// Build a solution with specific groupings
static Solution make_grouped_solution(const Problem& p, const DAG& d,
                                       std::vector<std::vector<size_t>> groups) {
    std::vector<ScheduleStep> steps;
    for (auto& ops : groups) {
        auto sg = Subgraph::create(p, d, ops);
        if (!sg) continue;
        auto c = sg->best_cost();
        if (!c.feasible) continue;
        ScheduleStep ss;
        ss.subgraph = std::move(*sg);
        ss.config = c.config;
        steps.push_back(std::move(ss));
    }
    return Solution(p, d, std::move(steps));
}

void test_greedy_never_worsens() {
    std::cout << "=== SolFM: greedy_descent never worsens ===\n";
    auto p = make_chain4_pw();
    DAG d = DAG::build(p);
    auto sol = make_singleton_solution(p, d);
    double before = sol.total_latency();
    
    auto improved = solution_greedy_descent(p, d, sol.steps());
    Solution after(p, d, std::move(improved));
    
    CHECK("greedy doesn't worsen", after.total_latency() <= before + 0.01);
    auto vr = after.validate();
    CHECK("greedy produces valid solution", vr.valid);
}

void test_retain_add_improves() {
    std::cout << "=== SolFM: retain_add finds improvement ===\n";
    auto p = make_shared_input();
    DAG d = DAG::build(p);
    
    // Op0 produces T1, then Op1 and Op2 both consume T1.
    // If Op1 is in step 1 and Op2 in step 2, retaining T1 should help.
    auto sol = make_grouped_solution(p, d, {{0, 1}, {2}});
    double before = sol.total_latency();
    
    // Run retain optimization
    auto opt = optimize_retain(p, d, std::move(sol));
    double after = opt.total_latency();
    CHECK("retain found improvement or same", after <= before + 0.01);
    auto vr = opt.validate();
    CHECK("valid after retain", vr.valid);
}

void test_fm_pass_basic() {
    std::cout << "=== SolFM: fm_pass basic ===\n";
    auto p = make_chain4_pw();
    DAG d = DAG::build(p);
    auto sol = make_singleton_solution(p, d);
    
    SolutionFMPassConfig cfg;
    cfg.floor_fraction = 0.3;
    cfg.max_drift_fraction = 0.5;
    
    auto result = solution_fm_pass(p, d, sol.steps(), cfg);
    CHECK("pass completes", true);
    CHECK("best ≤ start", result.best_cost <= result.start_cost + 0.01);
    
    Solution best_sol(p, d, std::move(result.best_steps));
    auto vr = best_sol.validate();
    CHECK("pass result valid", vr.valid);
}

void test_fm_pass_respects_floor() {
    std::cout << "=== SolFM: fm_pass floor control ===\n";
    auto p = make_chain4_pw();
    DAG d = DAG::build(p);
    auto sol = make_singleton_solution(p, d);
    
    // Floor = 0: only accept improving moves
    SolutionFMPassConfig cfg_strict;
    cfg_strict.floor_fraction = 0.0;
    auto strict = solution_fm_pass(p, d, sol.steps(), cfg_strict);
    
    // Floor = 0.5: accept worsening moves
    SolutionFMPassConfig cfg_loose;
    cfg_loose.floor_fraction = 0.5;
    cfg_loose.max_drift_fraction = 1.0;
    auto loose = solution_fm_pass(p, d, sol.steps(), cfg_loose);
    
    CHECK("loose applied ≥ strict moves", loose.moves_applied >= strict.moves_applied);
}

void test_fm_search_monotone() {
    std::cout << "=== SolFM: fm_search returns best seen ===\n";
    auto p = make_chain4_pw();
    DAG d = DAG::build(p);
    auto sol = make_singleton_solution(p, d);
    double before = sol.total_latency();
    
    SolutionFMConfig cfg;
    cfg.max_passes = 5;
    cfg.max_no_improve = 3;
    cfg.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    
    auto result = solution_fm_search(p, d, std::move(sol), cfg);
    CHECK("fm_search ≤ initial", result.total_latency() <= before + 0.01);
    auto vr = result.validate();
    CHECK("fm_search result valid", vr.valid);
}

void test_filter_retain_correctness() {
    std::cout << "=== SolFM: filter_retain correctness ===\n";
    auto p = make_chain4_pw();
    DAG d = DAG::build(p);
    
    // Create subgraph with ops {1, 2}: boundary_inputs={T1}, boundary_outputs={}, sink=T3
    auto sg = Subgraph::create(p, d, {1, 2});
    CHECK("subgraph created", sg.has_value());
    
    // T1 is a boundary input, T0 is not related, T3 is the sink
    CHECK("T1 is boundary input", sg->boundary_inputs().count(1));
    CHECK("T0 not boundary", !sg->boundary_inputs().count(0));
    
    // Retaining T0 should be invalid (not a boundary)
    // Retaining T1 should be valid
    auto retain = sg->best_cost({1}, {});  // T1 entering, nothing retained
    CHECK("can receive T1", retain.feasible);
}

void test_split_produces_valid_retain() {
    std::cout << "=== SolFM: split doesn't create invalid retains ===\n";
    auto p = make_chain4_pw();
    DAG d = DAG::build(p);
    
    // One big group {0,1,2,3}, split it, check retains are valid
    auto sol = make_grouped_solution(p, d, {{0, 1, 2, 3}});
    
    SolutionFMConfig cfg;
    cfg.max_passes = 3;
    cfg.max_no_improve = 2;
    cfg.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    
    auto result = solution_fm_search(p, d, std::move(sol), cfg);
    auto vr = result.validate();
    CHECK("no invalid retains after split", vr.valid);
    if (!vr.valid) std::cout << "    " << vr.error << "\n";
}

void test_merge_preserves_coverage() {
    std::cout << "=== SolFM: merge preserves op coverage ===\n";
    auto p = make_chain4_pw();
    DAG d = DAG::build(p);
    auto sol = make_singleton_solution(p, d);
    
    // Run with high floor to encourage merges
    SolutionFMConfig cfg;
    cfg.max_passes = 5;
    cfg.max_no_improve = 3;
    cfg.pass_config.floor_fraction = 0.5;
    cfg.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    
    auto result = solution_fm_search(p, d, std::move(sol), cfg);
    auto vr = result.validate();
    CHECK("coverage preserved after merges", vr.valid);
    if (!vr.valid) std::cout << "    " << vr.error << "\n";
}

void test_greedy_kick_escape() {
    std::cout << "=== SolFM: outer loop with greedy-kick ===\n";
    auto p = make_chain4_pw();
    DAG d = DAG::build(p);
    
    // Start from a suboptimal grouping
    auto sol = make_singleton_solution(p, d);
    double before = sol.total_latency();
    
    SolutionFMConfig cfg;
    cfg.max_passes = 10;
    cfg.max_no_improve = 5;
    cfg.pass_config.floor_fraction = 0.3;
    cfg.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    
    auto result = solution_fm_search(p, d, std::move(sol), cfg);
    CHECK("outer loop ≤ initial", result.total_latency() <= before + 0.01);
    auto vr = result.validate();
    CHECK("outer loop valid", vr.valid);
}

void test_deadline_respected() {
    std::cout << "=== SolFM: deadline respected ===\n";
    auto p = make_chain4_pw();
    DAG d = DAG::build(p);
    auto sol = make_singleton_solution(p, d);
    
    // Set deadline to 100ms from now
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    SolutionFMConfig cfg;
    cfg.max_passes = 1000;  // would take forever
    cfg.max_no_improve = 500;
    cfg.deadline = deadline;
    
    auto start = std::chrono::steady_clock::now();
    auto result = solution_fm_search(p, d, std::move(sol), cfg);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    
    // Should finish within ~200ms (100ms deadline + some overhead)
    CHECK("finished within 500ms", elapsed < 500);
    auto vr = result.validate();
    CHECK("deadline result valid", vr.valid);
}

int main() {
    test_greedy_never_worsens();
    test_retain_add_improves();
    test_fm_pass_basic();
    test_fm_pass_respects_floor();
    test_fm_search_monotone();
    test_filter_retain_correctness();
    test_split_produces_valid_retain();
    test_merge_preserves_coverage();
    test_greedy_kick_escape();
    test_deadline_respected();
    
    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << g_pass + g_fail << " tests\n";
    return g_fail > 0 ? 1 : 0;
}