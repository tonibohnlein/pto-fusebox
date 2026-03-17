// solution_gain_test.cpp
// Tests that solution FM move predictions match actual cost changes.
// Each test: build a multi-step solution, compute predicted saving for a
// specific move, apply it via rebuild, verify predicted ≈ actual.

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "solution/solution.h"
#include <iostream>
#include <cmath>
#include <set>
#include <vector>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++; else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double g, double e, double tol = 1.0) {
    if (std::abs(g - e) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << " Δ=" << (g-e) << "\n"; }
}

// ============================================================================
// Helper: simulate_rebuild_cost (same as in solution_search.cpp)
// ============================================================================
static double simulate_rebuild_cost(std::vector<ScheduleStep> &steps,
                                    size_t start) {
    double total = 0;
    std::set<size_t> cur;
    if (start > 0)
        cur = steps[start - 1].retain_these;
    for (size_t i = start; i < steps.size(); i++) {
        std::set<size_t> useful;
        for (auto t : steps[i].retain_these) {
            bool is_out = steps[i].subgraph.boundary_outputs().count(t);
            bool needed = (i + 1 < steps.size()) &&
                          steps[i + 1].subgraph.boundary_inputs().count(t);
            if (is_out && needed) useful.insert(t);
        }
        steps[i].retain_these = useful;
        auto bc = steps[i].subgraph.best_cost(cur, steps[i].retain_these);
        if (bc.feasible) {
            steps[i].config = bc.config;
            total += bc.latency;
        } else {
            auto bc2 = steps[i].subgraph.best_cost(cur, {});
            if (bc2.feasible) {
                steps[i].config = bc2.config;
                steps[i].retain_these.clear();
                total += bc2.latency;
            } else {
                total += 1e18;
            }
        }
        cur = steps[i].retain_these;
    }
    return total;
}

// ============================================================================
// Helper: build steps and compute costs via rebuild
// ============================================================================
struct StepSpec {
    std::vector<size_t> ops;
    std::set<size_t> retain;
};

static std::vector<ScheduleStep> build_steps(const Problem& p, const DAG& d,
                                              const std::vector<StepSpec>& specs) {
    std::vector<ScheduleStep> steps;
    for (auto& spec : specs) {
        auto sg = Subgraph::create(p, d, spec.ops);
        if (!sg) { std::cerr << "Subgraph::create failed\n"; continue; }
        ScheduleStep s;
        s.subgraph = std::move(*sg);
        s.config = s.subgraph.best_cost({}, {}).config;
        s.retain_these = spec.retain;
        steps.push_back(std::move(s));
    }
    // Run rebuild to get consistent costs and configs
    simulate_rebuild_cost(steps, 0);
    return steps;
}

static double total_cost(const std::vector<ScheduleStep>& steps) {
    auto tmp = steps;
    return simulate_rebuild_cost(tmp, 0);
}

// ============================================================================
// PW chain: Op0:T0→T1, Op1:T1→T2, Op2:T2→T3, Op3:T3→T4
// Each 128x128, cost=100, bw=50, cap=50000.
// Singleton PW cost = max(100, 32768/50) = 655.36
// With entering: cost = max(100, 16384/50) = 327.68
// ============================================================================
static Problem make_pw_chain4() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},100},
             {OpType::Pointwise,{1},{2},100},
             {OpType::Pointwise,{2},{3},100},
             {OpType::Pointwise,{3},{4},100}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 50;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// Test 1: RETAIN_REMOVE on clean state — prediction matches actual
// ============================================================================
void test_retain_remove_clean() {
    std::cout << "--- test_retain_remove_clean ---\n";
    auto p = make_pw_chain4(); DAG d = DAG::build(p);

    auto steps_with = build_steps(p, d, {
        {{0}, {1}}, {{1}, {}}, {{2}, {}}, {{3}, {}}
    });
    double cost_with = total_cost(steps_with);

    // Predict removal
    auto tmp = steps_with;
    tmp[0].retain_these.erase(1);
    double predicted_cost = simulate_rebuild_cost(tmp, 0);
    double predicted_saving = cost_with - predicted_cost;

    // Actual: build without retention
    auto steps_without = build_steps(p, d, {
        {{0}, {}}, {{1}, {}}, {{2}, {}}, {{3}, {}}
    });
    double actual_saving = cost_with - total_cost(steps_without);

    std::cout << "  predicted=" << predicted_saving << " actual=" << actual_saving << "\n";
    CHECK_EQ("retain_remove clean", predicted_saving, actual_saving);
}

// ============================================================================
// Test 2: RETAIN_ADD on clean state — prediction matches actual
// ============================================================================
void test_retain_add_clean() {
    std::cout << "--- test_retain_add_clean ---\n";
    auto p = make_pw_chain4(); DAG d = DAG::build(p);

    auto steps_without = build_steps(p, d, {
        {{0}, {}}, {{1}, {}}, {{2}, {}}, {{3}, {}}
    });
    double cost_without = total_cost(steps_without);

    // Predict addition
    auto tmp = steps_without;
    tmp[0].retain_these.insert(1);
    double predicted_cost = simulate_rebuild_cost(tmp, 0);
    double predicted_saving = cost_without - predicted_cost;

    // Actual
    auto steps_with = build_steps(p, d, {
        {{0}, {1}}, {{1}, {}}, {{2}, {}}, {{3}, {}}
    });
    double actual_saving = cost_without - total_cost(steps_with);

    std::cout << "  predicted=" << predicted_saving << " actual=" << actual_saving << "\n";
    CHECK_EQ("retain_add clean", predicted_saving, actual_saving);
}

// ============================================================================
// Test 3: Stale RETAIN_REMOVE — tensor already pruned → skip
// ============================================================================
void test_retain_remove_stale() {
    std::cout << "--- test_retain_remove_stale ---\n";
    auto p = make_pw_chain4(); DAG d = DAG::build(p);

    auto steps = build_steps(p, d, {
        {{0}, {1}}, {{1}, {}}, {{2}, {}}, {{3}, {}}
    });

    // Simulate what happens after rebuild prunes T1 from step 0
    steps[0].retain_these.clear();

    // A RETAIN_REMOVE move for T1 should see in_retain=false
    CHECK("stale: T1 not in retain", !steps[0].retain_these.count(1));

    // Even if we try, saving = 0
    auto tmp = steps;
    tmp[0].retain_these.erase(1); // no-op
    double cost_before = simulate_rebuild_cost(steps, 0);
    double cost_after = simulate_rebuild_cost(tmp, 0);
    CHECK_EQ("stale saving=0", cost_before - cost_after, 0.0, 0.01);
}

// ============================================================================
// Test 4: RETAIN chain — remove middle link
// [Op0 ret T1] [Op1 ret T2] [Op2] [Op3]
// Remove T2 from step 1: should not break T1 at step 0
// ============================================================================
void test_retain_chain_remove_middle() {
    std::cout << "--- test_retain_chain_remove_middle ---\n";
    auto p = make_pw_chain4(); DAG d = DAG::build(p);

    auto steps_chain = build_steps(p, d, {
        {{0}, {1}}, {{1}, {2}}, {{2}, {}}, {{3}, {}}
    });
    double cost_chain = total_cost(steps_chain);

    // Predict removing T2
    auto tmp = steps_chain;
    tmp[1].retain_these.erase(2);
    double predicted_cost = simulate_rebuild_cost(tmp, 0);
    double predicted_saving = cost_chain - predicted_cost;

    // Actual
    auto steps_no_t2 = build_steps(p, d, {
        {{0}, {1}}, {{1}, {}}, {{2}, {}}, {{3}, {}}
    });
    double actual_saving = cost_chain - total_cost(steps_no_t2);

    std::cout << "  predicted=" << predicted_saving << " actual=" << actual_saving << "\n";
    CHECK_EQ("chain remove middle", predicted_saving, actual_saving);

    // T1 should still be retained at step 0 after rebuild
    CHECK("T1 still retained", tmp[0].retain_these.count(1));
}

// ============================================================================
// Test 5: RETAIN_ADD cascade — adding at step 0 doesn't change step 2+
// ============================================================================
void test_retain_add_no_cascade() {
    std::cout << "--- test_retain_add_no_cascade ---\n";
    auto p = make_pw_chain4(); DAG d = DAG::build(p);

    auto steps = build_steps(p, d, {
        {{0}, {}}, {{1}, {}}, {{2}, {}}, {{3}, {}}
    });

    // Record step 2,3 costs
    auto s2 = steps;
    simulate_rebuild_cost(s2, 0);
    double cost2_before = s2[2].subgraph.compute_cost(s2[2].config, {}, {}).latency;
    double cost3_before = s2[3].subgraph.compute_cost(s2[3].config, {}, {}).latency;

    // Add T1 retention
    auto s2_with = steps;
    s2_with[0].retain_these.insert(1);
    simulate_rebuild_cost(s2_with, 0);
    double cost2_after = s2_with[2].subgraph.compute_cost(s2_with[2].config, {}, {}).latency;
    double cost3_after = s2_with[3].subgraph.compute_cost(s2_with[3].config, {}, {}).latency;

    std::cout << "  step2: " << cost2_before << " → " << cost2_after << "\n";
    std::cout << "  step3: " << cost3_before << " → " << cost3_after << "\n";
    CHECK_EQ("step2 unchanged", cost2_before, cost2_after, 0.01);
    CHECK_EQ("step3 unchanged", cost3_before, cost3_after, 0.01);
}

// ============================================================================
// Test 6: EJECT prediction — split step into remainder + singleton
// Fused [Op0,Op1]: T0→T1→T2. Eject Op0 → [Op0] + [Op1].
// Step 2 [Op2] was entering with T2 retained from step 0. After eject,
// the singleton Op0 has no retain → step 1(Op1) enters with {} → step 2
// also enters with {}.
// ============================================================================
void test_eject_breaks_retain_chain() {
    std::cout << "--- test_eject_breaks_retain_chain ---\n";
    auto p = make_pw_chain4(); DAG d = DAG::build(p);

    // Fused: [Op0,Op1 ret T2] [Op2] [Op3]
    auto steps_fused = build_steps(p, d, {
        {{0,1}, {2}}, {{2}, {}}, {{3}, {}}
    });
    double cost_fused = total_cost(steps_fused);
    std::cout << "  fused cost=" << cost_fused << "\n";

    // Eject Op0: [Op0] + [Op1 ret T2] [Op2] [Op3]
    auto sg0 = Subgraph::create(p, d, {0});
    auto sg1 = Subgraph::create(p, d, {1});
    if (!sg0 || !sg1) { std::cout << "  SKIP: subgraph creation failed\n"; return; }

    auto c0 = sg0->best_cost({}, {});
    auto c1 = sg1->best_cost({}, {2});

    // Build ejected steps
    std::vector<ScheduleStep> steps_ejected;
    { ScheduleStep s; s.subgraph = std::move(*sg0); s.config = c0.config; steps_ejected.push_back(std::move(s)); }
    { ScheduleStep s; s.subgraph = std::move(*sg1); s.config = c1.config; s.retain_these = {2}; steps_ejected.push_back(std::move(s)); }
    // Copy remaining steps
    for (size_t i = 1; i < steps_fused.size(); i++)
        steps_ejected.push_back(steps_fused[i]);

    double cost_ejected = simulate_rebuild_cost(steps_ejected, 0);
    double saving = cost_fused - cost_ejected;
    std::cout << "  ejected cost=" << cost_ejected << " saving=" << saving << "\n";

    // Verify simulate matches fresh build
    auto steps_fresh = build_steps(p, d, {
        {{0}, {}}, {{1}, {2}}, {{2}, {}}, {{3}, {}}
    });
    double cost_fresh = total_cost(steps_fresh);
    std::cout << "  fresh cost=" << cost_fresh << "\n";
    CHECK_EQ("eject simulate matches fresh", cost_ejected, cost_fresh);
}

// ============================================================================
// Test 7: Retention staleness guard at apply time
// Verify that a RETAIN_REMOVE move for a tensor not in retain_these
// is correctly detected and skipped.
// ============================================================================
void test_apply_staleness_guard() {
    std::cout << "--- test_apply_staleness_guard ---\n";
    auto p = make_pw_chain4(); DAG d = DAG::build(p);

    auto steps = build_steps(p, d, {
        {{0}, {}}, {{1}, {}}, {{2}, {}}, {{3}, {}}
    });

    // Step 0 has no retain. A RETAIN_REMOVE for T1 is invalid.
    CHECK("T1 not in step 0 retain", !steps[0].retain_these.count(1));

    // The apply_move function should check and return SIZE_MAX.
    // We can't call apply_move directly (it's static), but we verify
    // the invariant: erasing a non-existent element is a no-op.
    auto tmp = steps;
    tmp[0].retain_these.erase(1); // no-op
    double cost_before = total_cost(steps);
    double cost_after = total_cost(tmp);
    CHECK_EQ("erase no-op → cost unchanged", cost_before, cost_after, 0.01);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    test_retain_remove_clean();
    test_retain_add_clean();
    test_retain_remove_stale();
    test_retain_chain_remove_middle();
    test_retain_add_no_cascade();
    test_eject_breaks_retain_chain();
    test_apply_staleness_guard();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}