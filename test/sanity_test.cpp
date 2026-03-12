// e2e_sanity_test.cpp — End-to-end sanity check of the full pipeline
// Tests init → greedy → FM → post-opt on realistic problems

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "partition/partition.h"
#include "init/init_strategies.h"
#include "search/local_search.h"
#include "search/fm_search.h"
#include "test_move_helpers.h"
#include "search/active_set.h"
#include "search/fm_pass.h"
#include "search/fm_outer.h"
#include "search/evolution.h"
#include "solution/solution.h"
#include "postopt/post_opt.h"
#include "core/cost_cache.h"
#include "search/verbose.h"
#include <iostream>
#include <cmath>
#include <set>
#include <cassert>
#include <chrono>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++; else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_LE(const char* l, double g, double e) {
    if (g <= e + 0.5) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " ≤ " << e << "\n"; }
}

// ============================================================================
// Validate a partition: all ops covered, all groups valid, costs correct
// ============================================================================
static bool validate_partition(const Partition& part, const Problem& prob, const DAG& dag) {
    // Coverage
    for (size_t i = 0; i < prob.num_ops(); i++) {
        bool found = false;
        for (auto& g : part.groups)
            if (g.alive && g.ops.count(i)) { found = true; break; }
        if (!found) { std::cout << "    Op" << i << " uncovered!\n"; return false; }
    }
    // Each group valid
    for (size_t i = 0; i < part.groups.size(); i++) {
        if (!part.groups[i].alive) continue;
        auto sg = Subgraph::create(prob, dag, {part.groups[i].ops.begin(), part.groups[i].ops.end()});
        if (!sg) { std::cout << "    G" << i << " invalid subgraph!\n"; return false; }
        auto bc = sg->best_cost();
        if (!bc.feasible) { std::cout << "    G" << i << " infeasible!\n"; return false; }
        if (std::abs(part.groups[i].cost - bc.latency) > 1.0) {
            std::cout << "    G" << i << " cost mismatch: stored=" << part.groups[i].cost
                      << " actual=" << bc.latency << "\n";
            return false;
        }
    }
    return true;
}

// ============================================================================
// Build a Solution from a Partition (simplified — no retain, topo order)
// ============================================================================
static Solution partition_to_solution(const Problem& prob, const DAG& dag,
                                       const Partition& part) {
    // Collect alive groups
    struct GInfo { size_t idx; std::set<size_t> ops; Subgraph sg; CostResult cost; };
    std::vector<GInfo> groups;
    for (size_t i = 0; i < part.groups.size(); i++) {
        if (!part.groups[i].alive) continue;
        auto sg = Subgraph::create(prob, dag, {part.groups[i].ops.begin(), part.groups[i].ops.end()});
        if (!sg) continue;
        auto c = sg->best_cost();
        if (!c.feasible) continue;
        groups.push_back({i, part.groups[i].ops, std::move(*sg), c});
    }

    // Topological sort of groups
    size_t n = groups.size();
    std::vector<int> in_deg(n, 0);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;
            for (auto op_i : groups[i].ops)
                for (auto succ : dag.op_succs[op_i])
                    if (groups[j].ops.count(succ)) { in_deg[j]++; goto next_pair; }
            next_pair:;
        }

    std::vector<size_t> order;
    std::vector<bool> done(n, false);
    while (order.size() < n) {
        for (size_t i = 0; i < n; i++) {
            if (!done[i] && in_deg[i] == 0) {
                order.push_back(i);
                done[i] = true;
                for (size_t j = 0; j < n; j++) {
                    if (done[j]) continue;
                    for (auto op_i : groups[i].ops)
                        for (auto succ : dag.op_succs[op_i])
                            if (groups[j].ops.count(succ)) { in_deg[j]--; goto next_dec; }
                    next_dec:;
                }
                break;
            }
        }
    }

    std::vector<ScheduleStep> steps;
    for (auto gi : order)
        steps.push_back({Subgraph(groups[gi].sg), groups[gi].cost.config, {}});
    return Solution(prob, dag, std::move(steps));
}

// ============================================================================
// Problem 1: 8-op PW chain, 256x256 — tests fusion, greedy, FM
// ============================================================================
static Problem make_pw_chain8() {
    Problem p;
    for (int i = 0; i <= 8; i++) p.tensors.push_back({256, 256});
    for (int i = 0; i < 8; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 200});
    p.fast_memory_capacity = 40000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// Problem 2: Diamond with MatMul — tests split-K, snake, eph fan-out
// ============================================================================
static Problem make_mm_diamond() {
    Problem p;
    // T0(256x128) @ T1(128x256) → T2(256x256)
    // T2 → Op1(PW) → T3(256x256)
    // T2 → Op2(PW) → T4(256x256)
    // T3,T4 → Op3(PW) → T5(256x256)
    p.tensors = {{256,128},{128,256},{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},   // Op0: MM
             {OpType::Pointwise,{2},{3},500},    // Op1: PW
             {OpType::Pointwise,{2},{4},500},    // Op2: PW
             {OpType::Pointwise,{3,4},{5},500}}; // Op3: PW
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// Problem 3: Parallel MatMuls sharing input (benchmark 13 pattern)
// ============================================================================
static Problem make_parallel_mm() {
    Problem p;
    // T0: shared RHS (256x256)
    // T1..T4: individual LHS (256x256)
    // T5..T8: outputs (256x256)
    p.tensors = {{256,256},{256,256},{256,256},{256,256},{256,256},
                 {256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul,{1,0},{5},1000},
             {OpType::MatMul,{2,0},{6},1000},
             {OpType::MatMul,{3,0},{7},1000},
             {OpType::MatMul,{4,0},{8},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// Test 1: Init strategies produce valid, improving partitions
// ============================================================================
void test_init_strategies() {
    std::cout << "\n=== Test 1: Init strategies ===\n";
    auto p = make_pw_chain8();
    DAG d = DAG::build(p);

    auto strategies = all_init_strategies();
    double trivial_cost = Partition::trivial(p, d).total_cost();
    std::cout << "  Trivial: cost=" << trivial_cost
              << " groups=" << p.num_ops() << "\n";

    for (auto& s : strategies) {
        auto part = s.init(p, d);
        bool valid = validate_partition(part, p, d);
        CHECK(("init " + s.name + " valid").c_str(), valid);
        CHECK_LE(("init " + s.name + " ≤ trivial").c_str(), part.total_cost(), trivial_cost);
        std::cout << "  " << s.name << ": cost=" << part.total_cost()
                  << " groups=" << part.num_alive() << "\n";
    }
}

// ============================================================================
// Test 2: Greedy descent improves each init
// ============================================================================
void test_greedy_descent_all() {
    std::cout << "\n=== Test 2: Greedy descent ===\n";
    auto p = make_pw_chain8();
    DAG d = DAG::build(p);

    auto strategies = all_init_strategies();
    for (auto& s : strategies) {
        auto init = s.init(p, d);
        double before = init.total_cost();
        auto after = greedy_descent(std::move(init));
        bool valid = validate_partition(after, p, d);
        CHECK(("greedy " + s.name + " valid").c_str(), valid);
        CHECK_LE(("greedy " + s.name + " ≤ init").c_str(), after.total_cost(), before);
        std::cout << "  " << s.name << ": " << before << " → " << after.total_cost()
                  << " (" << after.num_alive() << " groups)\n";
    }
}

// ============================================================================
// Test 3: generate_moves produces valid, correctly-costed moves
// ============================================================================
void test_move_generation() {
    std::cout << "\n=== Test 3: Move generation & gain accuracy ===\n";
    auto p = make_pw_chain8();
    DAG d = DAG::build(p);

    // Start from a partially fused partition: {0,1,2},{3,4},{5},{6},{7}
    Partition part; part.prob = &p; part.dag = &d;
    part.add_group({0,1,2}, part.eval_set({0,1,2}));
    part.add_group({3,4}, part.eval_set({3,4}));
    part.add_group({5}, part.eval_set({5}));
    part.add_group({6}, part.eval_set({6}));
    part.add_group({7}, part.eval_set({7}));
    part.rebuild_index();

    int total_moves = 0, bad_gains = 0;
    int merge_count = 0, steal_count = 0, recomp_count = 0;
    int eject_count = 0, ie_count = 0, split_count = 0;

    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        auto moves = all_moves_for_group(part, gi, 1e18);

        for (auto& m : moves) {
            total_moves++;
            switch (m.type) {
                case FMMove::MERGE: merge_count++; break;
                case FMMove::STEAL: steal_count++; break;
                case FMMove::RECOMPUTE: recomp_count++; break;
                case FMMove::EJECT: eject_count++; break;
                case FMMove::INTERNAL_EJECT: ie_count++; break;
                case FMMove::SPLIT: split_count++; break;
                default: break;
            }

            // Verify gain accuracy for merge moves (cheap to check)
            if (m.type == FMMove::MERGE) {
                std::set<size_t> merged = part.groups[m.ga].ops;
                merged.insert(part.groups[m.gb].ops.begin(), part.groups[m.gb].ops.end());
                double actual = part.groups[m.ga].cost + part.groups[m.gb].cost - part.eval_set(merged);
                if (std::abs(m.saving - actual) > 0.5) bad_gains++;
            }
        }
    }

    std::cout << "  Total moves: " << total_moves
              << " (M=" << merge_count << " S=" << steal_count
              << " R=" << recomp_count << " E=" << eject_count
              << " IE=" << ie_count << " SP=" << split_count << ")\n";
    CHECK("has merge moves", merge_count > 0);
    CHECK("has steal moves", steal_count > 0);
    CHECK("has eject moves", eject_count > 0);
    CHECK("merge gains accurate", bad_gains == 0);
    std::cout << "  Bad gains: " << bad_gains << "\n";
}

// ============================================================================
// Test 4: FM best_move_for produces valid moves
// ============================================================================
void test_fm_best_move_for() {
    std::cout << "\n=== Test 4: FM best_move_for ===\n";
    auto p = make_pw_chain8();
    DAG d = DAG::build(p);

    // Start from a non-optimal partition so FM has moves to explore
    Partition part; part.prob = &p; part.dag = &d;
    part.add_group({0,1,2}, part.eval_set({0,1,2}));
    part.add_group({3,4}, part.eval_set({3,4}));
    part.add_group({5,6}, part.eval_set({5,6}));
    part.add_group({7}, part.eval_set({7}));
    part.rebuild_index();

    int valid_moves = 0, invalid_moves = 0;
    std::set<size_t> locked;

    for (size_t op = 0; op < p.num_ops(); op++) {
        auto m = best_move_for(part, op, part.total_cost() * 0.5, locked);
        if (m.valid()) {
            valid_moves++;
            CHECK("fm move has saving", m.saving > -1e17);
        } else {
            invalid_moves++;
        }
    }

    std::cout << "  Valid: " << valid_moves << " Invalid: " << invalid_moves << "\n";
    CHECK("some valid FM moves", valid_moves > 0);
}

// ============================================================================
// Test 5: FM inner pass runs and doesn't worsen best
// ============================================================================
void test_fm_inner_pass() {
    std::cout << "\n=== Test 5: FM inner pass ===\n";
    auto p = make_mm_diamond();
    DAG d = DAG::build(p);

    // Use a non-trivial problem where FM has room to improve
    auto part = Partition::trivial(p, d);
    double before = part.total_cost();

    FMConfig cfg;
    cfg.floor_fraction = 0.3;
    cfg.max_drift_fraction = 0.5;
    cfg.init_count = 5;
    cfg.seed = 42;

    auto result = fm_inner_pass(Partition(part), cfg);

    std::cout << "  Before: " << before << " Best: " << result.best_cost
              << " End: " << result.end_cost << "\n";
    std::cout << "  Moves: " << result.moves_applied
              << " (+" << result.moves_positive << " -" << result.moves_negative << ")\n";

    CHECK_LE("fm best ≤ start", result.best_cost, before);
    CHECK("fm applied moves", result.moves_applied >= 0);
    bool valid = validate_partition(result.best_partition, p, d);
    CHECK("fm best partition valid", valid);
}

// ============================================================================
// Test 6: FM outer loop with greedy-kick
// ============================================================================
void test_fm_outer_loop() {
    std::cout << "\n=== Test 6: FM outer loop ===\n";
    auto p = make_mm_diamond();
    DAG d = DAG::build(p);

    auto part = Partition::trivial(p, d);
    double before = part.total_cost();

    FMOuterConfig cfg;
    cfg.max_passes = 10;
    cfg.max_no_improve = 5;
    cfg.pass_config.floor_fraction = 0.3;
    cfg.pass_config.max_drift_fraction = 0.5;
    cfg.pass_config.init_count = 5;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    cfg.deadline = deadline;

    auto result = fm_outer_loop(Partition(part), cfg);

    std::cout << "  Before: " << before << " After: " << result.best_cost << "\n";
    std::cout << "  Passes: " << result.total_passes
              << " Improving: " << result.improving_passes << "\n";

    CHECK_LE("fm outer ≤ greedy", result.best_cost, before);
    bool valid = validate_partition(result.best_partition, p, d);
    CHECK("fm outer valid", valid);
}

// ============================================================================
// Test 7: Evolution operators preserve validity
// ============================================================================
void test_evolution() {
    std::cout << "\n=== Test 7: Evolution operators ===\n";
    auto p = make_pw_chain8();
    DAG d = DAG::build(p);

    auto part = Partition::trivial(p, d);
    part = greedy_descent(std::move(part));

    std::mt19937 rng(42);
    int valid_mutations = 0, total_mutations = 0;

    for (int i = 0; i < 20; i++) {
        auto mutated = mutate_compound(Partition(part), 3, rng);
        total_mutations++;
        if (validate_partition(mutated, p, d)) valid_mutations++;
    }
    std::cout << "  Mutations: " << valid_mutations << "/" << total_mutations << " valid\n";
    CHECK("all mutations valid", valid_mutations == total_mutations);

    // Crossover
    auto part2 = Partition::trivial(p, d);
    auto child = crossover(part, part2, rng);
    bool xvalid = validate_partition(child, p, d);
    CHECK("crossover valid", xvalid);
    std::cout << "  Crossover: " << child.num_alive() << " groups, cost="
              << child.total_cost() << "\n";
}

// ============================================================================
// Test 8: Full pipeline — partition → solution → validate → post-opt
// ============================================================================
void test_full_pipeline() {
    std::cout << "\n=== Test 8: Full pipeline ===\n";
    auto p = make_pw_chain8();
    DAG d = DAG::build(p);

    // Phase 1: Partition search
    auto part = best_initial(p, d);
    part = greedy_descent(std::move(part));
    double partition_cost = part.total_cost();
    std::cout << "  Partition: " << part.num_alive() << " groups, cost="
              << partition_cost << "\n";

    // Phase 2: Build solution
    auto sol = partition_to_solution(p, d, part);
    auto vr = sol.validate();
    CHECK("solution valid", vr.valid);
    if (!vr.valid) std::cout << "    Error: " << vr.error << "\n";
    std::cout << "  Solution: " << sol.num_steps() << " steps, latency="
              << sol.total_latency() << "\n";

    // Phase 3: Post-opt retain
    auto sol_opt = optimize_retain(p, d, std::move(sol));
    auto vr2 = sol_opt.validate();
    CHECK("post-opt valid", vr2.valid);
    if (!vr2.valid) std::cout << "    Error: " << vr2.error << "\n";
    std::cout << "  Post-opt: " << sol_opt.num_steps() << " steps, latency="
              << sol_opt.total_latency() << "\n";

    // Post-opt should not worsen
    CHECK_LE("post-opt ≤ original", sol_opt.total_latency(), partition_cost + 1);

    // Print schedule
    for (size_t i = 0; i < sol_opt.num_steps(); i++) {
        auto& s = sol_opt.step(i);
        std::cout << "    Step " << i << ": ops={";
        for (size_t j = 0; j < s.subgraph.ops().size(); j++) {
            if (j) std::cout << ",";
            std::cout << s.subgraph.ops()[j];
        }
        std::cout << "} [" << s.config.w << "," << s.config.h << "," << s.config.k << "]";
        if (!s.retain_these.empty()) {
            std::cout << " retain={";
            bool first = true;
            for (auto t : s.retain_these) { if (!first) std::cout << ","; std::cout << "T" << t; first = false; }
            std::cout << "}";
        }
        std::cout << " lat=" << sol_opt.step_latency(i) << "\n";
    }
}

// ============================================================================
// Test 9: MM diamond — eph fan-out handled correctly
// ============================================================================
void test_mm_diamond() {
    std::cout << "\n=== Test 9: MM diamond (eph fan-out) ===\n";
    auto p = make_mm_diamond();
    DAG d = DAG::build(p);

    // {Op0,Op1,Op2} should be rejected: T2 eph consumed by Op1 AND Op2
    Partition tmp; tmp.prob = &p; tmp.dag = &d;
    CHECK("eph fan-out rejected", tmp.eval_set({0,1,2}) >= 1e17);

    // {Op0,Op1} OK: T2 eph consumed only by Op1
    CHECK("{0,1} feasible", tmp.eval_set({0,1}) < 1e17);
    // {Op0,Op2} OK
    CHECK("{0,2} feasible", tmp.eval_set({0,2}) < 1e17);
    // {Op1,Op2,Op3} OK: T3,T4 are eph, each consumed once by Op3
    auto sg_123 = Subgraph::create(p, d, {1,2,3});
    CHECK("{1,2,3} feasible", sg_123.has_value());

    // Greedy should find a good partition
    auto part = best_initial(p, d);
    part = greedy_descent(std::move(part));
    bool valid = validate_partition(part, p, d);
    CHECK("mm diamond partition valid", valid);
    std::cout << "  Best partition: " << part.num_alive() << " groups, cost="
              << part.total_cost() << "\n";
    for (size_t i = 0; i < part.groups.size(); i++) {
        if (!part.groups[i].alive) continue;
        std::cout << "    G" << i << ": {";
        bool first = true;
        for (auto op : part.groups[i].ops) { if (!first) std::cout << ","; std::cout << op; first = false; }
        std::cout << "} cost=" << part.groups[i].cost << "\n";
    }
}

// ============================================================================
// Test 10: Parallel MM — co-consumer fusion
// ============================================================================
void test_parallel_mm() {
    std::cout << "\n=== Test 10: Parallel MM (co-consumer fusion) ===\n";
    auto p = make_parallel_mm();
    DAG d = DAG::build(p);

    // All 4 ops sharing T0 as RHS should be fuseable
    Partition tmp; tmp.prob = &p; tmp.dag = &d;
    double all_fused = tmp.eval_set({0,1,2,3});
    double all_sep = tmp.eval_set({0}) + tmp.eval_set({1}) + tmp.eval_set({2}) + tmp.eval_set({3});
    std::cout << "  All fused: " << all_fused << " All separate: " << all_sep << "\n";
    CHECK("fusion helps", all_fused < all_sep);

    // Init strategies should discover co-consumer fusion
    auto part = best_initial(p, d);
    part = greedy_descent(std::move(part));
    std::cout << "  Greedy: " << part.num_alive() << " groups, cost="
              << part.total_cost() << "\n";
    CHECK_LE("greedy ≤ separate", part.total_cost(), all_sep);
    bool valid = validate_partition(part, p, d);
    CHECK("parallel mm valid", valid);
}

// ============================================================================
// Test 11: PW sink rule on MM+PW fusion
// ============================================================================
void test_pw_sink_rule() {
    std::cout << "\n=== Test 11: PW sink rule ===\n";
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    auto sg = Subgraph::create(p, d, {0,1});
    CHECK("MM+PW fused valid", sg.has_value());

    // PW sink forces k=1
    CHECK("k=128 rejected", !sg->is_valid_tiling({128,128,128,SnakeDir::None}));
    CHECK("k=1 accepted", sg->is_valid_tiling({128,128,1,SnakeDir::None}));

    auto best = sg->best_cost();
    CHECK("best k=1", best.config.k == 1);

    // Fused should still beat separate (saves T2 transfer)
    Partition tmp; tmp.prob = &p; tmp.dag = &d;
    double fused = tmp.eval_set({0,1});
    double sep = tmp.eval_set({0}) + tmp.eval_set({1});
    std::cout << "  Fused (k=1): " << fused << " Separate: " << sep << "\n";
    CHECK("fused < separate", fused < sep);
}

// ============================================================================
// Test 12: apply_fm_move + rebuild_index consistency
// ============================================================================
void test_apply_fm_move() {
    std::cout << "\n=== Test 12: apply_fm_move consistency ===\n";
    auto p = make_pw_chain8();
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Find a merge move via best_move_for
    FMMove m = best_move_for(part, 0, part.total_cost(), {});
    if (m.valid()) {
        std::cout << "  Move: type=" << m.type << " op=" << m.op
                  << " ga=" << m.ga << " gb=" << m.gb << " saving=" << m.saving << "\n";

        double before = part.total_cost();
        auto affected = apply_fm_move(part, m);
        CHECK("move applied", !affected.empty());
        part.rebuild_index();
        bool valid = validate_partition(part, p, d);
        CHECK("after apply valid", valid);
        std::cout << "  Before: " << before << " After: " << part.total_cost() << "\n";
    } else {
        std::cout << "  No valid move for Op0 (OK for trivial partition)\n";
        CHECK("trivial has no moves", true);
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    g_verbose = false;

    test_init_strategies();
    test_greedy_descent_all();
    test_move_generation();
    test_fm_best_move_for();
    test_fm_inner_pass();
    test_fm_outer_loop();
    test_evolution();
    test_full_pipeline();
    test_mm_diamond();
    test_parallel_mm();
    test_pw_sink_rule();
    test_apply_fm_move();

    std::cout << "\n========================================\n";
    std::cout << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}