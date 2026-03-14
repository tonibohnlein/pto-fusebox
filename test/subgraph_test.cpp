// subgraph_gaps_test.cpp
// Targeted tests for gaps identified in the subgraph review:
//
//   1. working_set() returns INT64_MAX for invalid tilings
//   2. is_valid_tiling() edge cases (zero/negative params, exact boundaries)
//   3. Tensor serving multiple roles simultaneously (LHS + PW input)
//   4. best_cost() when all candidates are infeasible, including at 1×1
//
// Build: make subgraph_gaps_test
// Run:   ./subgraph_gaps_test

#include "core/dag.h"
#include "core/subgraph.h"
#include "core/types.h"
#include <climits>
#include <cmath>
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double got, double exp, double tol = 0.1) {
    if (std::abs(got - exp) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l
           << " got=" << got << " exp=" << exp << "\n"; }
}
static void CHECK_EQ_I(const char* l, int64_t got, int64_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l
           << " got=" << got << " exp=" << exp << "\n"; }
}

static TileConfig TC(int64_t w, int64_t h, int64_t k, SnakeDir s = SnakeDir::None) {
    return {w, h, k, s};
}
static Subgraph make_sg(const Problem& p, const DAG& d, std::vector<size_t> ops) {
    auto sg = Subgraph::create(p, d, std::move(ops));
    if (!sg) { std::cerr << "FATAL: Subgraph::create failed\n"; std::exit(1); }
    return std::move(*sg);
}

// ============================================================================
// 1. working_set() with invalid tiling
//
// An invalid tiling is one that violates divisibility constraints (w doesn't
// divide the output width, k doesn't divide K, etc.) or the PW-sink k=1 rule.
// working_set() now guards at the top: returns INT64_MAX for invalid tilings
// so callers that bypass is_valid_tiling() get an obviously wrong value
// rather than silent garbage.
// ============================================================================

void test_ws_invalid_tiling_nondivisible_w() {
    std::cout << "--- test_ws_invalid_tiling_nondivisible_w ---\n";
    // PW: T0(128×128) → T1(128×128). Valid w values divide 128.
    // w=3 does NOT divide 128 → invalid tiling.
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // w=3 does not divide 128
    CHECK("w=3 invalid", !sg.is_valid_tiling(TC(3, 128, 1)));
    CHECK_EQ_I("ws w=3 returns INT64_MAX",
               sg.working_set(TC(3, 128, 1)), INT64_MAX);

    // w=64 is valid
    CHECK("w=64 valid", sg.is_valid_tiling(TC(64, 128, 1)));
    CHECK("ws w=64 not INT64_MAX",
          sg.working_set(TC(64, 128, 1)) != INT64_MAX);
}

void test_ws_invalid_tiling_nondivisible_k() {
    std::cout << "--- test_ws_invalid_tiling_nondivisible_k ---\n";
    // MatMul T0(128×128) @ T1(128×128) → T2(128×128). K=128.
    // k=3 does NOT divide 128 → invalid.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK("k=3 invalid", !sg.is_valid_tiling(TC(128, 128, 3)));
    CHECK_EQ_I("ws k=3 returns INT64_MAX",
               sg.working_set(TC(128, 128, 3)), INT64_MAX);

    // k=32 is valid (32 divides 128)
    CHECK("k=32 valid", sg.is_valid_tiling(TC(128, 128, 32)));
    CHECK("ws k=32 not INT64_MAX",
          sg.working_set(TC(128, 128, 32)) != INT64_MAX);
}

void test_ws_invalid_tiling_pw_sink_k_gt_1() {
    std::cout << "--- test_ws_invalid_tiling_pw_sink_k_gt_1 ---\n";
    // Fused MM+PW: PW sink rule forces k=1.
    // k=32 is invalid even though it divides K.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK("k=32 invalid (PW sink)", !sg.is_valid_tiling(TC(128, 128, 32)));
    CHECK_EQ_I("ws k=32 returns INT64_MAX",
               sg.working_set(TC(128, 128, 32)), INT64_MAX);

    // k=1 is valid
    CHECK("k=1 valid", sg.is_valid_tiling(TC(128, 128, 1)));
    CHECK("ws k=1 not INT64_MAX",
          sg.working_set(TC(128, 128, 1)) != INT64_MAX);
}

// ============================================================================
// 2. is_valid_tiling() edge cases
// ============================================================================

void test_is_valid_tiling_zero_params() {
    std::cout << "--- test_is_valid_tiling_zero_params ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // All zero/negative params must be rejected
    CHECK("w=0 invalid",  !sg.is_valid_tiling(TC(0,  128, 1)));
    CHECK("h=0 invalid",  !sg.is_valid_tiling(TC(128, 0,  1)));
    CHECK("k=0 invalid",  !sg.is_valid_tiling(TC(128, 128, 0)));
    CHECK("all zero",     !sg.is_valid_tiling(TC(0,  0,   0)));
    // Negative values (represented as large unsigned or negative signed int64)
    CHECK("w=-1 invalid", !sg.is_valid_tiling(TC(-1, 128, 1)));
    CHECK("k=-1 invalid", !sg.is_valid_tiling(TC(128, 128, -1)));
}

void test_is_valid_tiling_exact_divisors() {
    std::cout << "--- test_is_valid_tiling_exact_divisors ---\n";
    // 768×768 MatMul (768 = 256×3). Valid divisors include non-pow2 values.
    Problem p;
    p.tensors = {{768,768},{768,768},{768,768}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // Exact boundary: the largest valid value equals the full dimension
    CHECK("w=768 valid (= output width)", sg.is_valid_tiling(TC(768, 768, 768)));
    CHECK("w=1 valid",  sg.is_valid_tiling(TC(1,   1,   1)));

    // Divisors of 768 that are not power of 2 must also be valid
    CHECK("w=96 valid",  sg.is_valid_tiling(TC(96,  96,  96)));
    CHECK("w=192 valid", sg.is_valid_tiling(TC(192, 192, 192)));
    CHECK("w=384 valid", sg.is_valid_tiling(TC(384, 384, 384)));

    // Non-divisors must be invalid
    CHECK("w=100 invalid (100∤768)", !sg.is_valid_tiling(TC(100, 128, 128)));
    CHECK("w=500 invalid",           !sg.is_valid_tiling(TC(500, 128, 128)));
    CHECK("k=100 invalid (100∤768)", !sg.is_valid_tiling(TC(128, 128, 100)));
}

void test_is_valid_tiling_matmul_k_must_divide_K() {
    std::cout << "--- test_is_valid_tiling_matmul_k_must_divide_K ---\n";
    // K=96 (= 32×3). Verify k must divide 96.
    Problem p;
    p.tensors = {{96,128},{128,96},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK("k=32 valid",  sg.is_valid_tiling(TC(128, 128, 32)));
    CHECK("k=48 valid",  sg.is_valid_tiling(TC(128, 128, 48)));
    CHECK("k=96 valid",  sg.is_valid_tiling(TC(128, 128, 96)));
    CHECK("k=64 invalid (64∤96)", !sg.is_valid_tiling(TC(128, 128, 64)));
    CHECK("k=128 invalid (128>96, ∤96)", !sg.is_valid_tiling(TC(128, 128, 128)));
}

// ============================================================================
// 3. Tensor serving multiple roles simultaneously
//
// A boundary tensor can appear in multiple ops with different roles. The
// boundary_tensor_info_ `ensure()` + max logic deduplicates it and takes the
// max slice size across all roles. This must be reflected in working_set().
//
// Test case: T0 is used as:
//   (a) LHS by Op0 (MatMul) — slice = h × K0
//   (b) PW input by Op1 (Pointwise) — slice = h × w
// Both ops share T0 as a boundary input. The working set must charge
// max(h × K0, h × w) for T0, not double-count it.
// ============================================================================

void test_dual_role_lhs_and_pw_input() {
    std::cout << "--- test_dual_role_lhs_and_pw_input ---\n";
    // T0(128×128): used as LHS by Op0 (K=128, so h×K = 128×128 = 16384)
    //              and as PW input by Op1 (h×w = 128×128 = 16384)
    // T1(128×128): RHS of Op0
    // T2(128×128): Op0 output (ephemeral — consumed by Op2)
    // T3(128×128): Op1 output (ephemeral — consumed by Op2)
    // T4(128×128): Op2 (PW) output — boundary output
    //
    // Op0: T0(LHS) @ T1(RHS) → T2
    // Op1: T0(PW input) → T3
    // Op2: T2 + T3 → T4  (PW with 2 inputs)
    //
    // T0, T1 are boundary inputs. T4 is boundary output.
    // T2, T3 are ephemeral (each has exactly one internal consumer: Op2).
    // has_pw_sink_ = true (Op2 is PW producing boundary output T4) → k must be 1.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    // T0, T1, T2,  T3,  T4
    p.ops = {{OpType::MatMul,{0,1},{2},2000},    // Op0: T0@T1→T2
             {OpType::Pointwise,{0},{3},500},    // Op1: T0→T3
             {OpType::Pointwise,{2,3},{4},300}}; // Op2: T2+T3→T4 (PW sink)
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0, 1, 2});

    CHECK("dual-role subgraph valid", sg.has_value());
    if (!sg) return;

    CHECK("T0 boundary in", sg->boundary_inputs().count(0));
    CHECK("T1 boundary in", sg->boundary_inputs().count(1));
    CHECK("T2 ephemeral",   sg->ephemeral().count(2));
    CHECK("T3 ephemeral",   sg->ephemeral().count(3));
    CHECK("T4 boundary out",sg->boundary_outputs().count(4));
    CHECK("PW sink: k=1 forced", !sg->is_valid_tiling(TC(128,128,32)));
    CHECK("k=1 valid",           sg->is_valid_tiling(TC(128,128,1)));

    // Working set at [128,128,1]:
    // T0 is both LHS (h×K=128×128=16384) and PW in (h×w=128×128=16384).
    // max of the two roles = 16384. T0 is counted ONCE.
    // T1 is RHS (k×w = 1×128 = 128).
    // T4 is boundary out (h×w = 16384).
    // Total = 16384 + 128 + 16384 = 32896
    // (T2, T3 are ephemeral → 0)
    int64_t ws = sg->working_set(TC(128,128,1));
    CHECK_EQ_I("ws dual-role T0 counted once", ws, 16384 + 128 + 16384);

    // Verify cost is feasible and T0 is not double-counted
    auto c = sg->compute_cost(TC(128,128,1));
    CHECK("feasible", c.feasible);
    // With k=1, nk = 128/1 = 128 k-passes (MM only per step, PW at last).
    // Transfer: lhs_load = h×K/B = 128×128/10 = 1638.4
    //           rhs_load = k×w/B = 1×128/10 = 12.8 (per k-step)
    //           pw_in_load for T0 = h×w/B = but T0 is retained as LHS across
    //           k-steps. Actually T0 appears in boundary_tensor_info_ once
    //           with max_lhs_K=128 and is_pw_in=true; lhs_load dominates
    //           and pw_in_load for T0 is deduped (same info entry, max_lhs_K wins).
    // The key check is that the working set is 32896 (not 49280 = 16384×2 + 128 + 16384
    // which would happen from double-counting T0).
    CHECK("no double-count: ws < 50000", ws < 50000);
    std::cout << "    ws=" << ws << " (expect 32896)\n";
}

void test_dual_role_rhs_and_pw_input() {
    std::cout << "--- test_dual_role_rhs_and_pw_input ---\n";
    // T1(128×128): used as RHS by Op0 (MatMul: k×w = k×128)
    //              and as PW input by Op1 (h×w = 128×128)
    // At [128,128,128]: RHS slice = k×w = 128×128 = 16384 = h×w. Same size.
    // At [128,128,32]:  RHS slice = k×w = 32×128  = 4096 < h×w = 16384.
    //                   max role = 16384 (PW input dominates).
    //
    // Op0: T0(LHS) @ T1(RHS) → T2 (ephemeral)
    // Op1: T1(PW in) → T3 (ephemeral)
    // Op2: T2 + T3 → T4 (boundary out, PW sink)
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{1},{3},500},
             {OpType::Pointwise,{2,3},{4},300}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0, 1, 2});
    CHECK("valid", sg.has_value());
    if (!sg) return;

    // At k=1: RHS slice = 1×128=128, PW in slice = 128×128=16384. max = 16384.
    // T0 LHS: h×K = 128×128 = 16384.
    // T4 out: h×w = 16384. Total ws = 16384 + 16384 + 16384 = 49152.
    int64_t ws_k1 = sg->working_set(TC(128,128,1));
    CHECK_EQ_I("ws k=1 T1 max(RHS,PW)=h*w", ws_k1, 16384 + 16384 + 16384);

    // T1 counted once at max role (h×w = 16384), not twice (4096 + 16384)
    CHECK("no double-count k=1", ws_k1 == 49152);
    std::cout << "    ws k=1 = " << ws_k1 << " (expect 49152)\n";
}

// ============================================================================
// 4. best_cost() when all candidates are infeasible
//
// If fast_memory_capacity is so small that even the smallest possible tile
// (w=1, h=1) exceeds capacity, best_cost() must return an infeasible result
// rather than a garbage CostResult.
// ============================================================================

void test_best_cost_all_infeasible() {
    std::cout << "--- test_best_cost_all_infeasible ---\n";
    // 128×128 MatMul. Even at [1,1,1] the working set is:
    // LHS strip h×K = 1×128 = 128, RHS strip k×w = 1×1 = 1,
    // output tile h×w = 1×1 = 1. Total = 130.
    // Set capacity = 10 → infeasible even at 1×1.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 10;  // deliberately tiny
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto best = sg.best_cost();
    CHECK("best_cost all-infeasible: not feasible", !best.feasible);
    // latency should be infinity (default)
    CHECK("best_cost all-infeasible: lat=inf", best.latency == std::numeric_limits<double>::infinity());
    // working_set should be 0 (default CostResult), not a garbage value
    CHECK("best_cost all-infeasible: ws=0", best.working_set == 0);
}

void test_best_cost_all_infeasible_pw() {
    std::cout << "--- test_best_cost_all_infeasible_pw ---\n";
    // PW 128×128. At [1,1,1]: ws = pw_in(1×1=1) + out(1×1=1) = 2.
    // Set capacity = 1 → even [1,1,1] is infeasible.
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 1;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto best = sg.best_cost();
    CHECK("PW all-infeasible: not feasible", !best.feasible);
    CHECK("PW all-infeasible: lat=inf",
          best.latency == std::numeric_limits<double>::infinity());
}

void test_best_cost_tight_fits_with_fallback() {
    std::cout << "--- test_best_cost_tight_fits_with_fallback ---\n";
    // 128×128 MatMul. At [128,128,128]: ws = 128*128 + 128*128 + 128*128 = 49152.
    // Set capacity = 30000 → [128,128,128] fails. [64,64,128]: ws =
    // h*K + k*w + h*w = 64*128 + 128*64 + 64*64 = 8192+8192+4096 = 20480 ≤ 30000.
    // So best_cost first pass (with native/4=32 floor) should succeed.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 30000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // Largest tile infeasible
    CHECK("[128,128,128] infeasible", !sg.is_feasible(TC(128,128,128)));

    // best_cost should still find something via fallback enumeration
    auto best = sg.best_cost();
    CHECK("tight: feasible result found", best.feasible);
    CHECK("tight: ws fits", best.working_set <= 30000);
    // The result should be better than naive [64,64,128] raster
    // (which best_cost should also consider)
    CHECK("tight: positive latency", best.latency > 0);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] ws=" << best.working_set
              << " lat=" << best.latency << "\n";
}

void test_best_cost_native_floor_skips_tiny_tiles() {
    std::cout << "--- test_best_cost_native_floor_skips_tiny_tiles ---\n";
    // 128x128 MatMul, cap=100000, native=[128,128].
    //
    // Key insight: k=64 and k=128 produce EXACTLY the same latency (4915.2):
    //   k=128 (nk=1): tile_cost = max(2000, 1638.4+1638.4+1638.4) = 4915.2
    //   k=64  (nk=2): step0=max(1000,2457.6)=2457.6, last=max(1000,2457.6)=2457.6
    //                 total = 4915.2
    // Since candidates are sorted ascending and the search uses strict <,
    // k=64 is found first. Asserting k=128 would be wrong.
    // The correct assertions: w=h=128 (largest spatial tile), latency=4915.2.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto best = sg.best_cost();
    CHECK("feasible", best.feasible);
    // Spatial tile must be full 128x128 (only 1 tile, no padding).
    CHECK("best w=128", best.config.w == 128);
    CHECK("best h=128", best.config.h == 128);
    // Optimal latency regardless of k (k=64 and k=128 tie at 4915.2).
    CHECK_EQ("best latency=4915.2", best.latency, 4915.2);
    // Must NOT return a tiny tile: native/4=32 floor on first pass prevents it.
    CHECK("w >= native/4=32", best.config.w >= 32);
    CHECK("h >= native/4=32", best.config.h >= 32);
    // Confirm [1,1,1] is vastly worse.
    auto worst = sg.compute_cost(TC(1,1,1));
    CHECK("[1,1,1] feasible (for comparison)", worst.feasible);
    CHECK("best << [1,1,1]", best.latency < worst.latency * 0.01);
    std::cout << "    best=[" << best.config.w << "," << best.config.h << ","
              << best.config.k << "] lat=" << best.latency
              << "  [1,1,1] lat=" << worst.latency << "\n";
}

// ============================================================================
// 5. working_set() invalid-tiling guard with retain sets
//    Verify the guard fires even when retain sets are provided.
// ============================================================================

void test_ws_invalid_with_retain_sets() {
    std::cout << "--- test_ws_invalid_with_retain_sets ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // Invalid tiling with retain sets should still return INT64_MAX
    CHECK_EQ_I("invalid tiling + retained_from_prev → INT64_MAX",
               sg.working_set(TC(3, 128, 128), {0}), INT64_MAX);
    CHECK_EQ_I("invalid tiling + retain_these → INT64_MAX",
               sg.working_set(TC(128, 128, 3), {}, {2}), INT64_MAX);
    CHECK_EQ_I("invalid tiling + both sets → INT64_MAX",
               sg.working_set(TC(3, 128, 3), {0}, {2}), INT64_MAX);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Gap 1: working_set invalid tiling guard
    test_ws_invalid_tiling_nondivisible_w();
    test_ws_invalid_tiling_nondivisible_k();
    test_ws_invalid_tiling_pw_sink_k_gt_1();
    test_ws_invalid_with_retain_sets();

    // Gap 2: is_valid_tiling edge cases
    test_is_valid_tiling_zero_params();
    test_is_valid_tiling_exact_divisors();
    test_is_valid_tiling_matmul_k_must_divide_K();

    // Gap 3: dual-role tensors
    test_dual_role_lhs_and_pw_input();
    test_dual_role_rhs_and_pw_input();

    // Gap 4: best_cost all-infeasible
    test_best_cost_all_infeasible();
    test_best_cost_all_infeasible_pw();
    test_best_cost_tight_fits_with_fallback();
    test_best_cost_native_floor_skips_tiny_tiles();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}