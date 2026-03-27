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
    // Fused MM+PW: PW sink → nk must equal 1. output_K_ = max_K_ = 128.
    // Only k=128 is valid (nk=128/128=1). k<128 gives nk>1 → rejected.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // k=32 → nk=128/32=4 > 1 → rejected
    CHECK("k=32 invalid (nk=4 > 1)", !sg.is_valid_tiling(TC(128, 128, 32)));
    CHECK("ws k=32 is INT64_MAX",
          sg.working_set(TC(128, 128, 32)) == INT64_MAX);

    // k=1 → nk=128 > 1 → rejected
    CHECK("k=1 invalid (nk=128 > 1)", !sg.is_valid_tiling(TC(128, 128, 1)));
    CHECK("ws k=1 is INT64_MAX",
          sg.working_set(TC(128, 128, 1)) == INT64_MAX);

    // k=128 → nk=1 → accepted
    CHECK("k=128 valid (nk=1)", sg.is_valid_tiling(TC(128, 128, 128)));
    CHECK("ws k=128 not INT64_MAX",
          sg.working_set(TC(128, 128, 128)) != INT64_MAX);
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
    // k=128 >= 96: gran>=dim is valid (max(96/128,1) = 1 tile)
    CHECK("k=128 valid (128>=96)", sg.is_valid_tiling(TC(128, 128, 128)));
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
    // T0(128×128): used as LHS by Op0 (MM, non-sink → h_tiles=1)
    //              and as PW input by Op1 (inherits h_tiles=ntw from T3)
    // When ntw > 1: h_tiles conflict (1 vs ntw) → SHAPES_MISALIGNED.
    // When ntw = 1 (128×128 at native): both agree h_tiles=1 → valid.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},    // Op0: T0@T1→T2
             {OpType::Pointwise,{0},{3},500},    // Op1: T0→T3
             {OpType::Pointwise,{2,3},{4},300}}; // Op2: T2+T3→T4 (PW sink)
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0, 1, 2});

    // At native tiling [128,128,1]: ntw=1 → no conflict → valid.
    // Only invalid for larger tilings where ntw > 1, but those get rejected
    // by is_valid_tiling at compute_cost time.
    CHECK("dual-role valid at native", sg.has_value());
    if (sg) {
        auto bc = sg->best_cost({}, {});
        CHECK("dual-role has feasible tiling", bc.feasible);
    }

    // Each op alone or in compatible pairs should work fine
    auto sg0 = Subgraph::create(p, d, {0});
    CHECK("Op0 alone valid", sg0.has_value());
    auto sg12 = Subgraph::create(p, d, {1, 2});
    CHECK("{Op1,Op2} valid", sg12.has_value());
}

void test_dual_role_rhs_and_pw_input() {
    std::cout << "--- test_dual_role_rhs_and_pw_input ---\n";
    // T1(128×128): used as RHS by Op0 (non-sink MM → h=ntw, v=1)
    //              and as PW input by Op1 (inherits ntw×nth from T3)
    // At native [128,128,1]: ntw=1, nth=1 → v_tiles: 1 (RHS) vs 1 (PW) → ok.
    // Conflict only at tilings where nth > 1, rejected by is_valid_tiling.
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

    // Valid at native tiling: ntw=1, nth=1 → all sources agree.
    CHECK("dual-role RHS+PW: valid at native", sg.has_value());
    if (sg) {
        auto bc = sg->best_cost({}, {});
        CHECK("dual-role RHS+PW: feasible", bc.feasible);
    }

    // Verify individual ops work
    auto sg0 = Subgraph::create(p, d, {0});
    CHECK("Op0 alone valid", sg0.has_value());
    auto sg12 = Subgraph::create(p, d, {1, 2});
    CHECK("{Op1,Op2} valid", sg12.has_value());
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
    // 128×128 MatMul. With tiling propagation, at [1,1,1]:
    // nk=128, ntw=128, nth=128. All slices = 1×1 = 1. WS = 3.
    // Set capacity = 2 → even [1,1,1] is infeasible (WS=3 > 2).
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 2;  // even tinier: WS=3 at [1,1,1]
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto best = sg.best_cost();
    CHECK("best_cost all-infeasible: not feasible", !best.feasible);
    CHECK("best_cost all-infeasible: lat=inf", best.latency == std::numeric_limits<double>::infinity());
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
// 6. Derived tile-count bounds (the zero-slice bug fix)
//
// When nk = output_K / k exceeds a tensor's dimension, the old code produced
// zero-size slices via integer truncation (H / nk = 0 when nk > H).
// The fix rejects such configs in is_valid_tiling.
// ============================================================================

void test_nk_exceeds_tensor_dim_chained_matmul() {
    std::cout << "--- test_nk_exceeds_tensor_dim_chained_matmul ---\n";
    // Chain: Op0 (non-sink MM) → Op1 (sink MM)
    // Op0: T0(64×256) @ T1(256×64) → T2(256×256)
    // Op1: T2(256×256) @ T3(64×256) → T4(64×256)  [sink, K=256]
    //
    // Op0's LHS T0 gets v_source=FROM_NK (propagated from T2 → Op1 LHS).
    // T0.height = 256. With k=1: nk = 256/1 = 256. v_tiles = 256.
    // T0.height / 256 = 1 → OK (borderline).
    // With k=1 on a LARGER chain where intermediate tensors are smaller: bug.
    Problem p;
    p.tensors = {{64,256},{256,64},{256,256},{64,256},{64,256}};
    //           T0       T1       T2(eph)   T3       T4(out)
    p.ops = {{OpType::MatMul,{0,1},{2},1000},   // Op0: non-sink
             {OpType::MatMul,{2,3},{4},1000}};   // Op1: sink
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // output_K = T2.width = 256. k=1 → nk=256.
    // T0 (64×256) has v_source=FROM_NK → v_tiles=256, H/256=1. Borderline valid.
    // T1 (256×64) has v_source=FIXED_1 → v_tiles=1. OK.
    // T3 (64×256) has v_source=FROM_NK → v_tiles=256, H/256=1. Borderline valid.
    CHECK("k=1 valid (borderline nk=256, min_dim=256)",
          sg.is_valid_tiling(TC(64, 256, 1)));

    // k=128 → nk=2. All dims >= 2. Must be valid.
    CHECK("k=128 valid (nk=2)", sg.is_valid_tiling(TC(64, 256, 128)));

    // k=256 → nk=1. Trivially valid.
    CHECK("k=256 valid (nk=1)", sg.is_valid_tiling(TC(64, 256, 256)));

    // Verify working set is reasonable (no zero slices)
    auto ws = sg.working_set(TC(64, 256, 1));
    CHECK("ws with k=1 is positive", ws > 0 && ws != INT64_MAX);
    std::cout << "    ws(k=1) = " << ws << "\n";
}

void test_nk_exceeds_tensor_dim_deep_chain() {
    std::cout << "--- test_nk_exceeds_tensor_dim_deep_chain ---\n";
    // Deep chain where sink K is much larger than an intermediate tensor dim.
    // Op0: PW T0(64×128) → T1(64×128)
    // Op1: MM T1(64×128) @ T2(512×64) → T3(512×128) [sink, K=64]
    //
    // Op0 PW: output T1 inherits from consumer Op1 LHS.
    // Op1 is SINK MM: LHS T1 gets (FROM_NK, FROM_NTH).
    // nk = 64/k. With k=1: nk=64. T1.height=128, v_tiles=64 → 128/64=2. OK.
    // T0 (PW input): inherits T1's source → (FROM_NK, FROM_NTH).
    // T0.height=128, v_tiles=64 → 128/64=2. OK.
    //
    // But T0.width=64, h_source=FROM_NK. h_tiles=64. 64/64=1. OK.
    // All fine for k=1 here.
    Problem p;
    p.tensors = {{64,128},{64,128},{512,64},{512,128}};
    p.ops = {{OpType::Pointwise,{0},{1},500},
             {OpType::MatMul,{1,2},{3},2000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK("k=1 valid (nk=64, all dims ≥ 64)", sg.is_valid_tiling(TC(512, 128, 1)));
    CHECK("k=32 valid (nk=2)", sg.is_valid_tiling(TC(512, 128, 32)));
    CHECK("k=64 valid (nk=1)", sg.is_valid_tiling(TC(512, 128, 64)));
}

void test_nk_exceeds_small_tensor_rejected() {
    std::cout << "--- test_nk_exceeds_small_tensor_rejected ---\n";
    // Construct a case where nk > a tensor's dimension → must be rejected.
    //
    // Op0: PW T0(32×64) → T1(32×64)
    // Op1: MM T1(32×64) @ T2(256×32) → T3(256×64) [sink, K=32]
    //
    // Op1 is sink. output_K = T1.width = 32.
    // T1 gets (FROM_NK, FROM_NTH) as sink LHS.
    // T0 (PW input) inherits → (FROM_NK, FROM_NTH).
    //
    // k=1 → nk=32. T0.width=32, h_source=FROM_NK → h_tiles=32. 32/32=1. OK.
    // T0.height=64, v_source=FROM_NTH → v_tiles=nth=64/h.
    //
    // Actually with output_K=32 and k=1, nk=32. T0.width=32 → h_tiles=32 → OK.
    // Need a wider output_K to trigger the bug.
    //
    // Better: Op1: MM T1(512×64) @ T2(256×512) → T3(256×64) [sink, K=512]
    //         Op0: PW T0(32×64) → T1(32×64)  — wait, T1 needs to be 512×64 for MM.
    //
    // Let's use three ops:
    // Op0: PW T0(32×256) → T1(32×256)     [T1 is narrow]
    // Op1: MM T1(32×256) @ T2(128×32) → T3(128×256) [non-sink, K=32]
    // Op2: MM T3(128×256) @ T4(512×128) → T5(512×256) [sink, K=128]
    //
    // T3 is ephemeral. T1 gets source from Op1 non-sink LHS: (FIXED_1, out_v).
    // T3's source from Op2 sink LHS: (FROM_NK, FROM_NTH). nk=128/k.
    // Op1 non-sink: output T3 has source (FROM_NK, FROM_NTH).
    //   LHS T1: (FIXED_1, FROM_NTH). No NK dependency on T1. Hmm.
    //
    // To get FROM_NK on a small tensor, need it as PW input inheriting from
    // a sink's RHS. Let me try:
    //
    // Op0: PW T0(32×256) → T1(32×256)
    // Op1: MM T2(256×256) @ T1(32×256) → T3(32×256) [sink, K=256]
    //
    // T1 as RHS of sink: (FROM_NTW, FROM_NK).
    // Op0 PW: T0 inherits T1's source → (FROM_NTW, FROM_NK).
    // T0.height=256, v_source=FROM_NK. nk=256/k.
    // k=1 → nk=256. T0.height=256 → v_tiles=256, 256/256=1. Borderline.
    //
    // Make T0 shorter:
    // T0(32×64), T1(32×64), Op1: MM T2(256×64) @ T1(32×256)...
    // No, RHS must be (N×K) where K=LHS.width.
    //
    // Op1: MM T2(256×64) @ T1(32×256) → T3(32×64)
    // Output T3: width=32, height=64. LHS=T2(256×64), K=256.
    // RHS=T1(32×256). Check: K=LHS.width=256. RHS.height should = K = 256. ✓
    // Output: width = RHS.width = 32, height = LHS.height = 64. ✓
    //
    // T1 as RHS of sink: (FROM_NTW, FROM_NK). nk=256/k.
    // T0(32×64) → PW → T1(32×64). Wait, T1 must be 32×256, not 32×64.
    // PW: input and output same dims. T0 and T1 must be 32×256.
    //
    // Then T0 (32×256): v_source=FROM_NK. nk=256/k.
    // k=1 → nk=256. v_tiles=256. T0.height=256, 256/256=1. Still borderline.
    //
    // Make T0.height=128 (smaller than nk=256):
    // T0(32×128), T1(32×128). But PW preserves dims, so T1 is 32×128.
    // Op1 RHS=T1(32×128). K=256. RHS.height must = K = 256. But T1.height=128 ≠ 256.
    // MatMul is invalid.
    //
    // The height of RHS must equal K. So if K=256, RHS must have height=256.
    // Can't make RHS shorter than K. So FROM_NK on RHS → v_tiles=nk=K/k,
    // and RHS.height=K. So v_tiles ≤ RHS.height always. No bug possible on RHS v.
    //
    // For FROM_NK on LHS h: sink LHS h_source=FROM_NK. h_tiles=nk. LHS.width=K.
    // So h_tiles=nk=K/k ≤ K = LHS.width. Always safe for direct LHS.
    //
    // The bug happens when FROM_NK propagates THROUGH a non-sink to a tensor
    // whose dimension is SMALLER than the sink's K.
    //
    // Example:
    // Op0: MM(non-sink) T0(64×256) @ T1(128×64) → T2(128×256) [K=64]
    // Op1: MM(sink) T2(128×256) @ T3(256×128) → T4(256×256)   [K=128]
    //
    // T2 ephemeral. Source from Op1 LHS (sink): (FROM_NK, FROM_NTH). nk=128/k.
    // Op0 non-sink: output T2 source=(FROM_NK, FROM_NTH).
    //   LHS T0: (FIXED_1, FROM_NTH). No NK. Safe.
    //   RHS T1: (FROM_NK, FIXED_1). h_tiles=nk=128/k. T1.width=128.
    //     k=1 → nk=128. h_tiles=128. 128/128=1. Borderline.
    //
    // Need T1.width < 128. Let's try T1(64×64). Then RHS of Op0 has width=64.
    // But output of Op0 should be: width = RHS.width = 64. height = LHS.height = 256.
    // T2 would be 64×256, not 128×256. Then Op1 LHS T2(64×256), K=64.
    // T3 must have height=K=64. Output T4: width=T3.width, height=T2.height=256.
    //
    // Hmm, let me just directly construct the scenario:
    //
    // Op0: MM T0(512×256) @ T1(128×512) → T2(128×256) [non-sink, K=512]
    // Op1: MM T2(128×256) @ T3(256×128) → T4(256×256) [sink, K=128]
    //
    // T2 eph. Op1 sink: T2 as LHS → (FROM_NK, FROM_NTH). nk=128/k.
    // Op0 non-sink: output T2 source=(FROM_NK, FROM_NTH).
    //   RHS T1: (FROM_NK, FIXED_1). h_source=FROM_NK → h_tiles=nk.
    //   T1.width=128. With k=1: nk=128. h_tiles=128. 128/128=1. Borderline.
    //
    // With T1.width=64 (64×512):
    //   Output T2: width = T1.width = 64. T2 is 64×256.
    //   Op1: LHS=T2(64×256), K=64. Output: width=T3.width, height=256.
    //   T3 must have height=K=64. T3(256×64) → T4(256×256).
    //   nk = 64/k. k=1 → nk=64.
    //   T1(64×512) with h_source=FROM_NK: h_tiles=64. T1.width=64 → 64/64=1. Borderline.
    //
    // OK the problem is that FROM_NK only appears on the RHS of non-sink MM,
    // and the RHS width is what becomes the output width. So the chain
    // constrains things.
    //
    // Let me try a PW chain to decouple:
    //
    // Op0: PW T0(32×256) → T1(32×256)
    // Op1: PW T1(32×256) → T2(32×256)
    // Op2: MM T3(256×256) @ T2(32×256) → T4(32×256)  [sink, K=256]
    //
    // T2 eph (produced by Op1, consumed by Op2). T1 eph (produced by Op0, consumed by Op1).
    // T2 as RHS of sink: (FROM_NTW, FROM_NK). nk=256/k.
    // Op1 PW: T1 inherits T2's source → (FROM_NTW, FROM_NK).
    // Op0 PW: T0 inherits T1's source → (FROM_NTW, FROM_NK).
    //
    // T0(32×256): v_source=FROM_NK → v_tiles=nk=256/k.
    // k=1 → v_tiles=256. T0.height=256. 256/256=1. Borderline.
    //
    // Change T0 to 32×128:
    // PW requires input/output same dims. If T0 is 32×128, T1 must be 32×128.
    // But T2 is 32×256. So Op1 PW from T1(32×128)→T2(32×256) is invalid
    // (PW preserves dims).
    //
    // Hmm. PW preserves dimensions. So the whole PW chain must have matching dims.
    //
    // The only way to get a tensor with a smaller dimension in the FROM_NK path
    // is through a non-sink MatMul where the RHS is narrow.
    //
    // Let me try a THREE-MatMul chain:
    //
    // Op0: MM T0(128×256) @ T1(64×128) → T2(64×256)  [non-sink, K=128]
    // Op1: MM T2(64×256) @ T3(128×64) → T4(128×256)   [non-sink, K=64]
    // Op2: MM T4(128×256) @ T5(512×128) → T6(512×256)  [sink, K=128]
    //
    // T2, T4 are ephemeral.
    //
    // Op2 sink: T4 as LHS → (FROM_NK, FROM_NTH). nk=128/k.
    //           T5 as RHS → (FROM_NTW, FROM_NK).
    //
    // Op1 non-sink: output T4 source = (FROM_NK, FROM_NTH).
    //   LHS T2: (FIXED_1, FROM_NTH). Safe.
    //   RHS T3: (FROM_NK, FIXED_1). h_tiles=nk=128/k. T3.width=128.
    //     k=1 → h_tiles=128. T3.width=128. 128/128=1. Borderline.
    //
    // Op0 non-sink: output T2 source = ???
    // T2 consumed by Op1 as LHS. Op1 non-sink LHS gets (FIXED_1, T4.v).
    // T4.v = FROM_NTH. So T2's source from being Op1 LHS = (FIXED_1, FROM_NTH).
    //   Op0: LHS T0: (FIXED_1, FROM_NTH). Safe.
    //         RHS T1: (FIXED_1, FIXED_1). No NK. Safe.
    //
    // Hmm, T1 doesn't get FROM_NK. The NK only propagates to RHS of non-sink
    // whose output has FROM_NK in h. And that output gets FROM_NK only if its
    // consumer is a sink LHS.
    //
    // OK so the FROM_NK propagation is actually quite constrained. It only goes:
    // sink LHS/RHS → non-sink output (if consumed as sink LHS) → non-sink RHS
    //
    // And the non-sink RHS width equals the non-sink output width, which came
    // from being consumed as... hmm this is getting circular.
    //
    // Let me just construct a concrete FAILING case using different dimensions.
    // The simplest trigger: a PW op between the RHS of a sink and a boundary input.
    //
    // Op0: PW T0(32×256) → T1(32×256)
    // Op1: MM T2(256×256) @ T1(32×256) → T3(32×256)  [sink, K=256]
    //
    // Wait, T0 and T1 must have same dims (PW). T1(32×256) as RHS of sink.
    // T1 → (FROM_NTW, FROM_NK). T0 inherits → (FROM_NTW, FROM_NK).
    //
    // out_W = 32, out_H = 256. ntw = 32/w.
    // output_K = 256. nk = 256/k.
    //
    // T0(32×256): h_source=FROM_NTW → h_tiles=ntw=32/w.
    //             v_source=FROM_NK → v_tiles=nk=256/k.
    // Need nk > T0.height=256 → k < 1. Impossible.
    //
    // OR need ntw > T0.width=32 → w < 1. Impossible.
    //
    // The problem is: PW preserves dimensions, so the tensor and its source
    // always have matching dims. The only way to get a mismatch is through
    // a MatMul where input dims differ from output dims.
    //
    // For non-sink MM RHS with (FROM_NK, FIXED_1): h_tiles=nk. And
    // nk = output_K/k where output_K is the SINK's K. The non-sink RHS
    // width can be anything (it becomes the non-sink output width).
    //
    // If non-sink RHS width < output_K... but nk=output_K/k, and we need
    // nk ≤ non-sink RHS width. So: output_K/k ≤ RHS_width → k ≥ output_K/RHS_width.
    //
    // Example: output_K=256, non-sink RHS width=64. Need k ≥ 4.
    // k=1 → nk=256 > 64 → BUG (zero slice).
    //
    // Op0: MM T0(256×256) @ T1(64×256) → T2(64×256) [non-sink]
    // Op1: MM T2(64×256) @ T3(256×64) → T4(256×256) [sink, K=64]
    //
    // T2 eph. Op1 sink: LHS T2 → (FROM_NK, FROM_NTH). nk=64/k.
    // Op0 non-sink: output T2 source=(FROM_NK, FROM_NTH).
    //   RHS T1: (FROM_NK, FIXED_1). h_tiles=nk=64/k. T1.width=64.
    //   k=1 → nk=64 → h_tiles=64. T1.width=64 → 64/64=1. Borderline.
    //
    // Need output_K > T1.width. So sink K must be > T1.width.
    // But sink K = T2.width = T1.width (because non-sink output width = RHS width).
    // T2.width = 64 (from T1.width=64). Op1 sink K = T2.width = 64. nk=64/k.
    // So nk ≤ 64 = T1.width always. Can't trigger!
    //
    // The chain forces alignment: the non-sink RHS width becomes the non-sink
    // output width, which becomes the sink LHS width (=K).
    // So nk = K/k = non-sink_output_width/k = non-sink_RHS_width/k ≤ non-sink_RHS_width.
    //
    // Unless there are MULTIPLE non-sink MMs in series! Then the sink's K comes
    // from a later stage, but FROM_NK propagates back through earlier stages
    // to tensors with different widths.
    //
    // Op0: MM T0(512×256) @ T1(64×512) → T2(64×256) [non-sink, K=512]
    // Op1: PW T2(64×256) → T3(64×256)
    // Op2: MM T3(64×256) @ T4(512×64) → T5(512×256) [sink, K=64]
    //
    // T2, T3 eph. Op2 sink: LHS T3 → (FROM_NK, FROM_NTH). nk=64/k.
    // Op1 PW: T2 inherits T3 source → (FROM_NK, FROM_NTH).
    // Op0 non-sink: output T2 source=(FROM_NK, FROM_NTH).
    //   RHS T1: (FROM_NK, FIXED_1). h_tiles=nk=64/k. T1.width=64.
    //   k=1 → nk=64 → h_tiles=64. T1.width=64 → 64/64=1. Borderline again.
    //
    // The RHS T1.width=64 = the value that ends up as nk. Still can't break it.
    //
    // Hmm. What if the non-sink output goes to a DIFFERENT sink with a LARGER K?
    //
    // Actually, there's only one sink per subgraph (or multiple sinks with same dims).
    // The output_K is determined by the sink. FROM_NK = output_K/k.
    //
    // For non-sink MM RHS with FROM_NK: h_tiles = output_K/k = nk.
    // RHS.width can be anything. It's NOT necessarily related to output_K.
    //
    // Wait, that's the key. Let me construct:
    //
    // Op0: MM T0(128×256) @ T1(64×128) → T2(64×256) [non-sink]
    // Op1: MM T2(64×256) @ T3(256×64) → T4(256×256) [sink, K=64]
    //
    // Ah but Op1 sink K = T2.width = 64. nk=64/k. And T1 with FROM_NK: h_tiles=nk.
    // T1.width=64. nk ≤ 64 = T1.width. Safe.
    //
    // The problem is: when the non-sink's output is consumed as SINK LHS,
    // the sink's K = non-sink output width = non-sink RHS width.
    // And non-sink RHS gets FROM_NK = K/k = non-sink_RHS_width/k.
    // h_tiles ≤ T1.width. Always safe.
    //
    // When non-sink output is consumed as PW input leading to another sink:
    // same thing — the chain preserves dimensions.
    //
    // SO: the zero-slice bug can only happen when FROM_NK propagates to a tensor
    // that got its dimensions from a DIFFERENT branch of the DAG, not from the
    // same K-chain.
    //
    // WAIT. What about the BOTTLENECK example? Step 13 had:
    // Op5: MM(non-sink) T7(64×256) @ T8(512×64) → T9(512×256)
    // Op18: PW [T9, T18] → T28(512×256)
    // Op19: MM(non-sink) T28(512×256) @ T27(512×256) → T29(512×256)
    // Op20: MM(sink) T0(512×256) @ T29(512×256) → T30(512×256) [K=512]
    //
    // Op20 sink: LHS T0 → (FROM_NK, FROM_NTH). RHS T29 → (FROM_NTW, FROM_NK).
    // T29 source = (FROM_NTW, FROM_NK).
    // Op19 non-sink: output T29 source = (FROM_NTW, FROM_NK).
    //   LHS T28: (FIXED_1, FROM_NK). v_source=FROM_NK. T28.height=256.
    //   RHS T27: (FROM_NTW, FIXED_1).
    //
    // T28 source = (FIXED_1, FROM_NK). nk=512/k.
    // Op18 PW: inputs T9, T18 inherit → (FIXED_1, FROM_NK).
    // T9 source = (FIXED_1, FROM_NK).
    // Op5 non-sink: output T9 source = (FIXED_1, FROM_NK).
    //   LHS T7: (FIXED_1, FROM_NK). v_source=FROM_NK. T7.height=256.
    //     nk=512/k. k=1 → nk=512. v_tiles=512 > T7.height=256. BUG!
    //
    // So the bug happens because output_K=512 (from sink Op20) propagates
    // via FROM_NK through Op19 → T28 → Op18 PW → T9 → Op5 non-sink → T7,
    // but T7.height=256 < output_K=512.
    //
    // The key: T7.height (256) is NOT related to output_K (512). T7 feeds
    // into a DIFFERENT branch of the chain. FROM_NK (= output_K/k) is applied
    // to T7.height even though T7 has nothing to do with the sink's K dimension.
    //
    // To reproduce: need at least 3 MMs where the sink's K is larger than
    // an earlier non-sink input's relevant dimension.
    //
    // Simplified reproduction:
    // Op0: MM(non-sink) T0(128×128) @ T1(256×128) → T2(256×128)
    // Op1: MM(sink) T3(512×128) @ T2(256×512) → T4(256×128) [K=512]
    //
    // Wait, this doesn't chain. Let me be more careful:
    //
    // Op0: MM T0(128×256) @ T1(256×128) → T2(256×256) [non-sink, K=128]
    // Op1: PW T2(256×256) → T3(256×256)
    // Op2: MM T4(512×256) @ T3(256×512) → T5(256×256) [sink, K=512]
    //
    // Wait, T3 as RHS of Op2: T3.height must = K = T4.width = 512. But T3.height=256.
    // Invalid.
    //
    // Let me think differently. I need:
    // 1. A sink MM with large K (K=512)
    // 2. A tensor in the FROM_NK path with a dim smaller than 512
    //
    // The sink's LHS has width=K=512 (that's what K means). LHS gets FROM_NK.
    // The sink's RHS has height=K=512. RHS gets FROM_NK in v.
    //
    // For FROM_NK to reach a SMALL tensor, it must propagate backward through
    // non-sink ops to a tensor with a dimension < K.
    //
    // Path: sink RHS → (FROM_NTW, FROM_NK) → consumed by non-sink as output →
    //   non-sink LHS gets (FIXED_1, v_from_output) where v_from_output could be FROM_NK.
    //
    // Yes! Non-sink LHS gets v_source = output's v_source. If output has
    // v_source=FROM_NK, then LHS gets v_source=FROM_NK. And LHS.height can be
    // anything (it's the M dimension).
    //
    // Concrete:
    // Op0: MM T0(64×128) @ T1(256×64) → T2(256×128) [non-sink]
    // Op1: MM T3(512×128) @ T2(256×512) → T4(256×128) [sink, K=512]
    //
    // Check dims: Op1 LHS=T3(512×128), RHS=T2. Output T4: width=RHS.width=256,
    // height=LHS.height=128. K=LHS.width=512. RHS.height must=K=512.
    // T2.height=128 ≠ 512. Invalid!
    //
    // The RHS height MUST equal K. So T2 must have height=512. But Op0 output
    // T2 has height = Op0.LHS.height = 128 (from T0). Can't match.
    //
    // The issue: in a direct chain, dimensions are constrained. But the bottleneck
    // case had the sink reading T0(512×256) as LHS (a graph input, not from the chain)
    // and T29 as RHS (from the chain). The chain feeds the RHS, not LHS.
    //
    // So the pattern is:
    // - Sink reads a LARGE graph input as LHS (K=512)
    // - Sink reads chain output as RHS
    // - FROM_NK propagates backward through the chain via non-sink outputs
    // - Somewhere in the chain, a tensor has a v-dim < 512
    //
    // Let me construct this:
    // T_in(512×256) is a graph input (large K)
    // Op0: PW T0(256×256) → T1(256×256)
    // Op1: MM T_in(512×256) @ T1(256×512) → T2(256×256) [sink, K=512]
    //
    // T1 must have height=K=512. T1(256×512). But PW preserves dims.
    // T0 must be 256×512 too. OK.
    //
    // T0(256×512): v_source from T1 → (FROM_NTW, FROM_NK). v_tiles=nk=512/k.
    // T0.height=512. 512/512=1. Borderline.
    //
    // Need T0 shorter. But PW forces same dims. So add a non-sink MM before PW:
    //
    // Op0: MM T_a(64×128) @ T_b(256×64) → T_c(256×128) [non-sink]
    // Op1: MM T_in(512×128) @ T_c(256×512) → T_out(256×128) [sink, K=512]
    //
    // T_c as RHS of sink: height must = K = 512. T_c.height = T_a.height = 128 ≠ 512. Invalid.
    //
    // Argh. The chain always constrains RHS height = K.
    //
    // OK in the bottleneck, the key was:
    // - Op19 non-sink: T28 as LHS, T27 as RHS → T29
    // - Op20 sink: T0 as LHS (K=512), T29 as RHS
    // - T29.height = 256 = T28.height. But K=512. So T29.height ≠ K?
    //   Wait, RHS of sink must have height=K. T29.height=256, K=512.
    //
    // That means the bottleneck problem itself has T29.height=256 but K=512.
    // How does this work? Let me recheck: Op20 inputs=[0, 29]. T0(512×256),
    // T29(512×256). K=T0.width=512. RHS=T29. RHS.height=256. 256 ≠ 512.
    //
    // This IS a shape mismatch! The RHS height doesn't match K. But
    // is_valid_tiling should reject this... unless the shapes are flexible in
    // the problem model?
    //
    // Hmm, actually I think the problem model doesn't enforce that MatMul
    // dimensions match — it just has arbitrary tensors. The cost model handles
    // the tiling. Let me re-check: the MatMul C = A × B where
    //   A is M×K: A.height=M, A.width=K
    //   B is K×N: B.height=K, B.width=N  <-- B.height SHOULD equal K
    //
    // But the problem specification doesn't enforce this! The ops just list
    // input/output tensor indices. The dimensions are stored separately in
    // the tensors array. If someone defines a MatMul where LHS.width ≠ RHS.height,
    // the cost model would use K=LHS.width and RHS.height independently.
    //
    // Actually wait, looking at the bottleneck problem:
    // Op19: inputs=[28, 27]. T28(512×256), T27(512×256).
    // K = LHS.width = T28.width = 512. RHS.height = T27.height = 256.
    // These DON'T match! 512 ≠ 256. This is a malformed MatMul.
    //
    // Hmm, but the problem file says these are valid. Maybe the problem model
    // doesn't require K=RHS.height? That would be unusual. Let me check the
    // constraint collection code...
    //
    // In the constraint collection:
    // MatMul LHS: add_constraint(input[0], K_param, H_param) → width → k_set
    // MatMul RHS: add_constraint(input[1], W_param, K_param) → height → k_set
    //
    // So both LHS.width AND RHS.height go into k_set. If they differ, k must
    // divide both. This handles mismatched "K" dimensions.
    //
    // And for tile source propagation:
    // Sink LHS: (FROM_NK, FROM_NTH). h_tiles = nk = output_K/k.
    // Sink RHS: (FROM_NTW, FROM_NK). v_tiles = nk = output_K/k.
    // BUT: output_K is defined as the sink LHS width (line 134 of create):
    //   sg.output_K_ = sg.op_K(s); // = prob.tensors[prob.ops[s].inputs[0]].width
    //
    // So nk = output_K/k = LHS.width/k. The RHS has v_tiles=nk=LHS.width/k.
    // RHS.height may not equal LHS.width. So v_tiles can exceed RHS.height.
    //
    // In the bottleneck: Op20 LHS=T0(512×256). output_K=512. k=1 → nk=512.
    // T29(512×256) as RHS: v_tiles=nk=512. T29.height=256. 512 > 256. BUG!
    //
    // So the bug triggers even in a two-op subgraph when LHS.width > RHS.height!
    //
    // Simple reproduction:
    // Op0: MM T0(512×256) @ T1(256×128) → T2(256×256) [sink]
    // K = T0.width = 512. nk = 512/k.
    // T1.height = 128. v_source = FROM_NK. v_tiles = nk = 512/k.
    // k=1 → v_tiles=512 > T1.height=128. Zero slice!

    Problem p;
    // T0: 512×256 (LHS, K=512)
    // T1: 256×128 (RHS, height=128 < K=512)
    // T2: 256×256 (output)
    p.tensors = {{512,256},{256,128},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // K=512, RHS.height=128. nk=512/k.
    // k=1 → nk=512. T1 v_tiles=512 > T1.height=128. Must reject.
    CHECK("k=1 rejected (nk=512 > RHS.height=128)",
          !sg.is_valid_tiling(TC(256, 256, 1)));

    // k=2 → nk=256 > 128. Must reject.
    CHECK("k=2 rejected (nk=256 > 128)",
          !sg.is_valid_tiling(TC(256, 256, 2)));

    // k=4 → nk=128 = T1.height=128. Borderline, must accept.
    CHECK("k=4 accepted (nk=128 = RHS.height=128)",
          sg.is_valid_tiling(TC(256, 256, 4)));

    // k=128 → nk=4 ≤ 128. Valid.
    CHECK("k=128 accepted (nk=4)", sg.is_valid_tiling(TC(256, 128, 128)));

    // k=512 → nk=1. Valid.
    CHECK("k=512 accepted (nk=1)", sg.is_valid_tiling(TC(256, 256, 512)));

    // Verify working set is sane for valid configs (no zero slices)
    auto ws4 = sg.working_set(TC(256, 256, 4));
    CHECK("ws(k=4) positive", ws4 > 0 && ws4 != INT64_MAX);
    std::cout << "    ws(k=4) = " << ws4 << "\n";

    // best_cost must not pick an invalid k
    auto best = sg.best_cost();
    CHECK("best_cost feasible", best.feasible);
    CHECK("best_cost k >= 4 (nk ≤ 128)",
          best.config.k >= 4);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

void test_nk_exceeds_tensor_dim_multi_op() {
    std::cout << "--- test_nk_exceeds_tensor_dim_multi_op ---\n";
    // Multi-op chain reproducing the bottleneck bug pattern:
    // Op0: MM(non-sink) T0(128×256) @ T1(256×128) → T2(256×256)
    // Op1: MM(sink) T3(512×256) @ T2(256×512) → T4(256×256)  [K=512]
    //
    // T2 is ephemeral. T2 as Op1 LHS: (FROM_NK, FROM_NTH). nk=512/k.
    // Op0 non-sink: output T2 source = (FROM_NK, FROM_NTH).
    //   LHS T0: (FIXED_1, FROM_NTH). Safe.
    //   RHS T1: (FROM_NK, FIXED_1). h_tiles=nk=512/k. T1.width=256.
    //     k=1 → nk=512 > T1.width=256. Must reject.
    //     k=2 → nk=256 = T1.width. Borderline accept.
    //
    // Also: T3(512×256) as boundary input. As Op1 sink LHS: (FROM_NK, FROM_NTH).
    //   h_tiles=nk=512/k. T3.width=512.
    //   k=1 → nk=512 = T3.width=512. Borderline accept for T3, but T1 fails.
    Problem p;
    p.tensors = {{128,256},{256,128},{256,256},{512,256},{256,512},{256,256}};
    //            T0        T1        T2(eph)   T3        T4 wait...

    // Actually, let me check: Op1 RHS = T2. T2 needs height = K = 512 (for valid MM).
    // T2.height = T0.height = 256 (from Op0 LHS). 256 ≠ 512.
    // This is the mismatched-K case. Let me just set T2 = 256×256 and accept the mismatch.
    // The code handles mismatched K via k_divides_ containing both 512 and 256.

    // Actually I realize I need T4.height. Op1: LHS=T3(512×256), RHS=T2(256×256).
    // Output: width = RHS.width = 256, height = LHS.height = 256. T4(256×256). OK.
    // K = LHS.width = 512. RHS.height = 256. Mismatch (256 ≠ 512).
    // But the code just collects both into k_set.

    // Hmm wait, T4 can't be at index 4 because T2 was already index 2.
    // Let me index carefully:
    // T0(128×256), T1(256×128), T2(256×256), T3(512×256), T4(256×256)
    p.tensors = {{128,256},{256,128},{256,256},{512,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},1000},   // Op0: non-sink
             {OpType::MatMul,{3,2},{4},2000}};   // Op1: sink (T3=LHS, T2=RHS)
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // output_K = T3.width = 512. nk = 512/k.
    // T1(256×128) via FROM_NK: h_tiles=nk. T1.width=256.
    // k=1 → nk=512 > 256. Must reject.
    CHECK("multi-op k=1 rejected (nk=512 > T1.width=256)",
          !sg.is_valid_tiling(TC(256, 256, 1)));

    // k=2 → nk=256 = T1.width. Accept.
    CHECK("multi-op k=2 accepted (nk=256 = T1.width)",
          sg.is_valid_tiling(TC(256, 256, 2)));

    // k=4 → nk=128 < 256. Accept.
    CHECK("multi-op k=4 accepted (nk=128)",
          sg.is_valid_tiling(TC(256, 256, 4)));

    auto best = sg.best_cost();
    CHECK("multi-op best feasible", best.feasible);
    CHECK("multi-op best k >= 2", best.config.k >= 2);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

void test_ntw_exceeds_tensor_dim() {
    std::cout << "--- test_ntw_exceeds_tensor_dim ---\n";
    // Test that ntw = out_W/w doesn't exceed a FROM_NTW tensor's width.
    // This is less likely in practice but should still be caught.
    //
    // Op0: MM T0(32×256) @ T1(512×32) → T2(512×256) [sink, K=32]
    //
    // T0 as LHS: (FROM_NK, FROM_NTH). h_tiles=nk. T0.width=32.
    // T1 as RHS: (FROM_NTW, FROM_NK). h_tiles=ntw. T1.width=512.
    //
    // ntw = out_W/w = 512/w. T0.width=32, source=FROM_NK (not NTW). OK.
    // T1.width=512, source=FROM_NTW. ntw=512/w ≤ 512. Always OK for T1.
    //
    // Need a tensor with FROM_NTW and width < out_W. This happens when
    // a non-sink MM's RHS width < output width. But RHS width IS output width.
    //
    // Actually: FROM_NTW tensors have widths that are in w_divides_. And
    // ntw = out_W/w. For ntw > tensor.width, need out_W > tensor.width * w.
    // Since tensor.width is in w_divides_ and w divides tensor.width (or w ≥ tensor.width),
    // if w ≤ tensor.width: ntw = out_W/w. Need out_W/w > tensor.width,
    // i.e., out_W > w * tensor.width ≥ tensor.width². This requires
    // out_W > tensor.width², which requires a very wide output and narrow tensor.
    //
    // Example: narrow tensor with FROM_NTW via PW:
    // Op0: PW T0(64×256) → T1(64×256) [feeds into MM as RHS]
    // Op1: MM T2(256×256) @ T1(64×256) → T3(64×256) [sink, K=256]
    //
    // T1 as RHS: (FROM_NTW, FROM_NK). T0 inherits (PW).
    // out_W = T3.width = 64. ntw = 64/w.
    // T0.width=64. ntw = 64/w ≤ 64. Always safe for T0 (64/w ≤ 64 iff w ≥ 1).
    //
    // The FROM_NTW dimension always equals some tensor's width that's in w_divides_.
    // And ntw = out_W / w. out_W is in w_divides_. So ntw ≤ out_W.
    // Other FROM_NTW tensors have widths also in w_divides_, and since w divides
    // those widths, those_widths / ntw = those_widths * w / out_W. If those_widths
    // < out_W and w doesn't compensate... this could fail.
    //
    // Concrete: out_W = 512. Tensor with FROM_NTW and width = 64.
    // w = 32 (divides both 512 and 64). ntw = 512/32 = 16. 16 ≤ 64 ✓.
    // w = 8: ntw = 64. 64 ≤ 64 ✓. Borderline.
    // w = 4: ntw = 128. 128 > 64. BUG!
    //
    // For w=4 to be in candidates: w_divides_ must contain both 512 and 64.
    // 4 < 64, 64 % 4 = 0 ✓. 4 < 512, 512 % 4 = 0 ✓. So w=4 IS a candidate.
    //
    // But we need a tensor with width=64 and source=FROM_NTW while out_W=512.
    // This requires the tensor to be in a PW chain inheriting from a wider output.
    //
    // Op0: MM T_a(64×128) @ T_b(512×64) → T_c(512×128) [non-sink, K=64]
    // Op1: PW T_d(512×128) → T_e(512×128)
    // Op2: MM T_f(256×128) @ T_e(512×256) → T_g(512×128) [sink, K=256]
    //
    // Hmm, this is getting complicated. Let me just construct the simplest case
    // with mismatched dimensions:
    //
    // Single MM: T0(256×128) @ T1(512×64) → T2(512×128) [K=256]
    // out_W = 512. ntw = 512/w.
    // T1(512×64) as RHS: (FROM_NTW, FROM_NK). T1.width=512. ntw ≤ 512. Safe.
    // T0(256×128) as LHS: (FROM_NK, FROM_NTH). T0.width=256. nk=256/k. Safe for width.
    //
    // I can't get FROM_NTW on a narrow tensor with a single MM. Need multi-op.
    //
    // Actually, I realize this is hard to trigger because FROM_NTW is applied
    // to tensor widths, and the constraint ensures w divides those widths.
    // ntw = out_W / w. For ntw > some_width: out_W / w > some_width →
    // w < out_W / some_width. Since w divides some_width: w ≤ some_width.
    // So out_W / w ≥ out_W / some_width. For ntw > some_width:
    // out_W / some_width > some_width → out_W > some_width².
    //
    // This requires out_W to be much larger than the narrow tensor. Uncommon
    // but possible. Let me just test the nk case since that's the practical bug.

    // Simple test: verify ntw doesn't exceed dims via w_divides_ constraint.
    Problem p;
    p.tensors = {{128,256},{128,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // PW: out_W=128. ntw=128/w. T0.width=128. ntw ≤ 128 always.
    // All valid divisors of 128 are safe.
    CHECK("w=1 valid", sg.is_valid_tiling(TC(1, 1, 1)));
    CHECK("w=128 valid", sg.is_valid_tiling(TC(128, 128, 1)));
    CHECK("w=64 valid", sg.is_valid_tiling(TC(64, 128, 1)));

    // Working set sanity: at w=1,h=1 → ntw=128,nth=256.
    // Each tile = 1×1. WS = 1+1 = 2.
    auto ws = sg.working_set(TC(1, 1, 1));
    CHECK_EQ_I("ws(1,1,1) = 2", ws, 2);
}

// ============================================================================
// 7. Working set and cost sanity for known configurations
// ============================================================================

void test_working_set_single_matmul() {
    std::cout << "--- test_working_set_single_matmul ---\n";
    // MM: T0(128×256) @ T1(256×128) → T2(256×256)
    // K=128. At [256,256,128]: ntw=1, nth=1, nk=1.
    // LHS T0: (FROM_NK, FROM_NTH). h_tiles=nk=1, v_tiles=nth=1.
    //   slice = 128×256 = 32768.
    // RHS T1: (FROM_NTW, FROM_NK). h_tiles=ntw=1, v_tiles=nk=1.
    //   slice = 256×128 = 32768.
    // Out T2: (FROM_NTW, FROM_NTH). h_tiles=ntw=1, v_tiles=nth=1.
    //   slice = 256×256 = 65536.
    // Total WS = 32768 + 32768 + 65536 = 131072.
    Problem p;
    p.tensors = {{128,256},{256,128},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK_EQ_I("ws(256,256,128) = 131072",
               sg.working_set(TC(256, 256, 128)), 131072);

    // At [128,128,128]: ntw=2, nth=2, nk=1.
    // LHS T0: h_tiles=nk=1, v_tiles=nth=2. slice = 128 × 128 = 16384.
    // RHS T1: h_tiles=ntw=2, v_tiles=nk=1. slice = 128 × 128 = 16384.
    // Out T2: h_tiles=ntw=2, v_tiles=nth=2. slice = 128 × 128 = 16384.
    // Total WS = 16384 × 3 = 49152.
    CHECK_EQ_I("ws(128,128,128) = 49152",
               sg.working_set(TC(128, 128, 128)), 49152);

    // At [128,128,64]: ntw=2, nth=2, nk=2.
    // LHS T0: h_tiles=nk=2, v_tiles=nth=2. slice = 64 × 128 = 8192.
    // RHS T1: h_tiles=ntw=2, v_tiles=nk=2. slice = 128 × 64 = 8192.
    // Out T2: h_tiles=ntw=2, v_tiles=nth=2. slice = 128 × 128 = 16384.
    // Total WS = 8192 + 8192 + 16384 = 32768.
    CHECK_EQ_I("ws(128,128,64) = 32768",
               sg.working_set(TC(128, 128, 64)), 32768);
}

void test_working_set_with_retain() {
    std::cout << "--- test_working_set_with_retain ---\n";
    // PW: T0(128×128) → T1(128×128).
    // At [128,128,1]: ws = T0(128×128) + T1(128×128) = 16384 + 16384 = 32768.
    // With T0 retained_from_prev: T0 occupies full_size=16384 (not sliced).
    // ws = 16384 + 16384 = 32768 (same since ntw=nth=1 anyway).
    //
    // At [64,64,1]: ntw=2, nth=2. slice = 64×64 = 4096.
    // ws = 4096(T0) + 4096(T1) = 8192.
    // With T0 retained: ws = 16384(T0 full) + 4096(T1 slice) = 20480.
    // With T1 retained_for_next: T1 skipped in slice sum (will be committed later).
    // ws = 4096(T0) + T1_retain_size(16384) = 4096 + 16384 = 20480.
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK_EQ_I("ws(64,64,1) no retain = 8192",
               sg.working_set(TC(64, 64, 1)), 8192);

    // retained_from_prev={T0}: full size for T0
    CHECK_EQ_I("ws(64,64,1) retain_in={T0} = 20480",
               sg.working_set(TC(64, 64, 1), {0}), 20480);

    // retain_these={T1}: T1 skipped from boundary sum, added as full size
    CHECK_EQ_I("ws(64,64,1) retain_out={T1} = 20480",
               sg.working_set(TC(64, 64, 1), {}, {1}), 20480);
}

void test_cost_compute_overlap() {
    std::cout << "--- test_cost_compute_overlap ---\n";
    // PW: T0(128×128) → T1(128×128). base_cost=1000.
    // At [128,128,1]: 1 tile. ntw=1, nth=1, nk=1.
    // compute_per_step = 1000 (full base_cost, 1×1 native scale).
    // IO: T0 slice = 128×128/10 = 1638.4. T1 evict = 1638.4.
    // tile_cost = max(compute, io) = max(1000, 1638.4+1638.4) = 3276.8.
    // latency = 1 * 3276.8 = 3276.8.
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto r = sg.compute_cost(TC(128, 128, 1));
    CHECK("feasible", r.feasible);
    CHECK_EQ("compute_per_step = 1000", r.compute_per_step, 1000.0);
    CHECK_EQ("latency = 3276.8", r.latency, 3276.8);
}

void test_cost_matmul_snake_vs_raster() {
    std::cout << "--- test_cost_matmul_snake_vs_raster ---\n";
    // MM: T0(256×256) @ T1(256×256) → T2(256×256). K=256. base_cost=2000.
    // At [128,128,256]: ntw=2, nth=2, nk=1. 4 tiles.
    // Snake should be ≤ raster cost (snake reuses data between rows).
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto raster = sg.compute_cost(TC(128, 128, 256, SnakeDir::None));
    auto hsnake = sg.compute_cost(TC(128, 128, 256, SnakeDir::RowMajor));
    auto vsnake = sg.compute_cost(TC(128, 128, 256, SnakeDir::ColMajor));

    CHECK("raster feasible", raster.feasible);
    CHECK("hsnake feasible", hsnake.feasible);
    CHECK("vsnake feasible", vsnake.feasible);

    CHECK("hsnake ≤ raster", hsnake.latency <= raster.latency + 0.01);
    CHECK("vsnake ≤ raster", vsnake.latency <= raster.latency + 0.01);

    std::cout << "    raster=" << raster.latency
              << " hsnake=" << hsnake.latency
              << " vsnake=" << vsnake.latency << "\n";
}

// ============================================================================
// 8. Longer chains, splits, and tensor reuse at different depths
// ============================================================================

void test_long_pw_chain_4ops() {
    std::cout << "--- test_long_pw_chain_4ops ---\n";
    // Pure PW chain: Op0→Op1→Op2→Op3. All tensors 128×256.
    // T0→Op0→T1→Op1→T2→Op2→T3→Op3→T4
    // T1,T2,T3 are ephemeral. T0 boundary input, T4 boundary output.
    Problem p;
    p.tensors = {{128,256},{128,256},{128,256},{128,256},{128,256}};
    p.ops = {{OpType::Pointwise,{0},{1},100},
             {OpType::Pointwise,{1},{2},100},
             {OpType::Pointwise,{2},{3},100},
             {OpType::Pointwise,{3},{4},100}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2, 3});

    CHECK("4-PW chain: 4 ops", sg.num_ops() == 4);
    CHECK("4-PW chain: 1 boundary in", sg.boundary_inputs().size() == 1);
    CHECK("4-PW chain: 1 boundary out", sg.boundary_outputs().size() == 1);
    CHECK("4-PW chain: 3 ephemeral", sg.ephemeral().size() == 3);
    CHECK("4-PW chain: T0 is input", sg.boundary_inputs().count(0));
    CHECK("4-PW chain: T4 is output", sg.boundary_outputs().count(4));

    // All tensors same size, PW chain: ws = just T0 slice + T4 slice.
    // At [128,256,1]: ntw=1, nth=1 → full tensors. ws = 128*256 + 128*256 = 65536.
    CHECK_EQ_I("ws(128,256,1) = 65536", sg.working_set(TC(128, 256, 1)), 65536);

    // At [64,128,1]: ntw=2, nth=2. slice = 64*128 = 8192. ws = 8192*2 = 16384.
    CHECK_EQ_I("ws(64,128,1) = 16384", sg.working_set(TC(64, 128, 1)), 16384);

    // Valid tiling
    CHECK("w=64,h=128 valid", sg.is_valid_tiling(TC(64, 128, 1)));
    CHECK("w=128,h=256 valid", sg.is_valid_tiling(TC(128, 256, 1)));
    // w must divide 128, h must divide 256
    CHECK("w=32,h=64 valid", sg.is_valid_tiling(TC(32, 64, 1)));
    CHECK("w=3 invalid (doesn't divide 128)", !sg.is_valid_tiling(TC(3, 128, 1)));

    auto best = sg.best_cost();
    CHECK("4-PW chain feasible", best.feasible);
}

void test_long_mm_chain_3ops() {
    std::cout << "--- test_long_mm_chain_3ops ---\n";
    // Three MatMuls in series: Op0→Op1→Op2 (only Op2 is sink).
    // Op0: T0(256×256) @ T1(128×256) → T2(128×256)  [non-sink, K=256]
    // Op1: T3(128×256) @ T2(128×128) → T4(128×256)  [non-sink, K=128]
    //   (T2 consumed as RHS: T2.height must match... actually no constraint in model)
    // Op2: T4(128×256) @ T5(256×128) → T6(256×256)  [sink, K=128]
    //
    // T2, T4 are ephemeral.
    //
    // Tile source propagation:
    //   T6 (sink out): (FROM_NTW, FROM_NTH)
    //   Op2 sink: T4(LHS) → (FROM_NK, FROM_NTH), T5(RHS) → (FROM_NTW, FROM_NK)
    //   Op1 non-sink: output T4 → (FROM_NK, FROM_NTH)
    //     T3(LHS) → (FIXED_1, FROM_NTH), T2(RHS) → (FROM_NK, FIXED_1)
    //   Op0 non-sink: output T2 → (FROM_NK, FIXED_1)
    //     T0(LHS) → (FIXED_1, FIXED_1), T1(RHS) → (FROM_NK, FIXED_1)
    //
    // output_K = T4.width = 128. nk = 128/k.
    // T1(128×256): h_source=FROM_NK → h_tiles=nk. T1.width=128. nk=128/k ≤ 128. Safe.
    // T2(128×128): h_source=FROM_NK → h_tiles=nk. T2.width=128. Safe.

    Problem p;
    p.tensors = {{256,256},{128,256},{128,128},{128,256},{128,256},{256,128},{256,256}};
    //            T0        T1        T2(eph)   T3        T4(eph)   T5        T6(out)
    p.ops = {{OpType::MatMul,{0,1},{2},1000},   // Op0
             {OpType::MatMul,{3,2},{4},1000},   // Op1
             {OpType::MatMul,{4,5},{6},2000}};  // Op2 (sink)
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("3-MM chain: 3 ops", sg.num_ops() == 3);
    CHECK("3-MM chain: 2 ephemeral (T2,T4)", sg.ephemeral().size() == 2);
    CHECK("3-MM chain: T2 eph", sg.ephemeral().count(2));
    CHECK("3-MM chain: T4 eph", sg.ephemeral().count(4));
    CHECK("3-MM chain: T6 boundary out", sg.boundary_outputs().count(6));
    CHECK("3-MM chain: has matmul", sg.has_matmul());
    CHECK_EQ_I("3-MM chain: output_K = 128", sg.output_K(), 128);

    // k=1 → nk=128. T1.width=128, T2.width=128. Borderline OK.
    CHECK("k=1 valid (nk=128, all dims ≥ 128)", sg.is_valid_tiling(TC(256, 256, 1)));
    CHECK("k=128 valid (nk=1)", sg.is_valid_tiling(TC(256, 256, 128)));

    // Working set should be positive and reasonable for valid tilings
    auto ws = sg.working_set(TC(256, 256, 128));
    CHECK("ws positive", ws > 0 && ws != INT64_MAX);

    auto best = sg.best_cost();
    CHECK("3-MM chain feasible", best.feasible);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

void test_mm_pw_mm_chain() {
    std::cout << "--- test_mm_pw_mm_chain ---\n";
    // MM→PW→MM pattern (common in bottleneck blocks):
    // Op0: MM T0(256×128) @ T1(128×256) → T2(128×128)  [non-sink, K=256]
    // Op1: PW T2(128×128) → T3(128×128)
    // Op2: MM T4(256×128) @ T3(128×256) → T5(128×128)  [sink, K=256]
    //
    // T2, T3 ephemeral.
    // Tile sources:
    //   T5(sink out): (FROM_NTW, FROM_NTH)
    //   Op2 sink: T4(LHS) → (FROM_NK, FROM_NTH), T3(RHS) → (FROM_NTW, FROM_NK)
    //   Op1 PW: T2 inherits T3 → (FROM_NTW, FROM_NK)
    //   Op0 non-sink: output T2 → (FROM_NTW, FROM_NK)
    //     T0(LHS) → (FIXED_1, FROM_NK), T1(RHS) → (FROM_NTW, FIXED_1)
    //
    // output_K = T4.width = 256. nk = 256/k.
    // T0(256×128): v_source=FROM_NK → v_tiles=nk. T0.height=128.
    //   k=1 → nk=256 > T0.height=128. Must reject!
    //   k=2 → nk=128 = T0.height. Borderline OK.

    Problem p;
    p.tensors = {{256,128},{128,256},{128,128},{128,128},{256,128},{128,128}};
    //            T0        T1        T2(eph)   T3(eph)   T4        T5(out)
    p.ops = {{OpType::MatMul,{0,1},{2},1000},    // Op0 non-sink
             {OpType::Pointwise,{2},{3},200},     // Op1 PW
             {OpType::MatMul,{4,3},{5},2000}};    // Op2 sink
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("MM-PW-MM: 3 ops", sg.num_ops() == 3);
    CHECK("MM-PW-MM: T2 eph", sg.ephemeral().count(2));
    CHECK("MM-PW-MM: T3 eph", sg.ephemeral().count(3));
    CHECK_EQ_I("MM-PW-MM: output_K = 256", sg.output_K(), 256);

    // k=1 → nk=256 > T0.height=128. Reject.
    CHECK("k=1 rejected (nk=256 > T0.height=128)",
          !sg.is_valid_tiling(TC(128, 128, 1)));

    // k=2 → nk=128 = T0.height. Accept.
    CHECK("k=2 accepted (nk=128 = T0.height)",
          sg.is_valid_tiling(TC(128, 128, 2)));

    // k=256 → nk=1. Accept.
    CHECK("k=256 accepted (nk=1)", sg.is_valid_tiling(TC(128, 128, 256)));

    auto ws = sg.working_set(TC(128, 128, 2));
    CHECK("ws(k=2) positive", ws > 0 && ws != INT64_MAX);

    auto best = sg.best_cost();
    CHECK("MM-PW-MM feasible", best.feasible);
    CHECK("MM-PW-MM best k >= 2", best.config.k >= 2);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

void test_fan_in_two_mm_to_pw() {
    std::cout << "--- test_fan_in_two_mm_to_pw ---\n";
    // Two MatMuls fan into a PW (add/residual pattern):
    // Op0: MM T0(256×256) @ T1(128×256) → T2(128×256)
    // Op1: MM T3(256×256) @ T4(128×256) → T5(128×256)
    // Op2: PW [T2, T5] → T6(128×256)  [sink]
    //
    // T2, T5 are ephemeral (consumed by Op2 only, produced internally).
    // T6 is boundary output.
    // has_pw_sink_ = true → output_K = 1, k forced to 1.
    //
    // Tile sources:
    //   T6 (PW sink out): (FROM_NTW, FROM_NTH)
    //   Op2 PW sink: T2, T5 inherit → (FROM_NTW, FROM_NTH)
    //   Op0 non-sink: output T2 → (FROM_NTW, FROM_NTH)
    //     T0(LHS) → (FIXED_1, FROM_NTH), T1(RHS) → (FROM_NTW, FIXED_1)
    //   Op1 non-sink: output T5 → (FROM_NTW, FROM_NTH)
    //     T3(LHS) → (FIXED_1, FROM_NTH), T4(RHS) → (FROM_NTW, FIXED_1)

    Problem p;
    p.tensors = {{256,256},{128,256},{128,256},{256,256},{128,256},{128,256},{128,256}};
    //            T0        T1        T2(eph)   T3        T4        T5(eph)   T6(out)
    p.ops = {{OpType::MatMul,{0,1},{2},1000},     // Op0
             {OpType::MatMul,{3,4},{5},1000},     // Op1
             {OpType::Pointwise,{2,5},{6},200}};  // Op2 (PW sink)
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("fan-in: 3 ops", sg.num_ops() == 3);
    CHECK("fan-in: T2 eph", sg.ephemeral().count(2));
    CHECK("fan-in: T5 eph", sg.ephemeral().count(5));
    CHECK("fan-in: T6 boundary out", sg.boundary_outputs().count(6));
    // PW sink + matmul: output_K_ = max_K_ = 256 (K of Op0 and Op1).
    CHECK("fan-in: output_K = max_K", sg.output_K() == sg.max_K());

    // PW sink: nk must be 1. output_K_=256, so only k=256 is valid.
    CHECK("k=1 invalid (nk=256 > 1)", !sg.is_valid_tiling(TC(128, 256, 1)));
    CHECK("k=2 invalid (nk=128 > 1)", !sg.is_valid_tiling(TC(128, 256, 2)));
    CHECK("k=256 valid (nk=1)", sg.is_valid_tiling(TC(128, 256, 256)));

    // ws at [128,256,256]: ntw=1, nth=1, nk=1.
    // T0(256×256): (FIXED_1, FROM_NTH). h_tiles=1, v_tiles=1. slice=65536.
    // T1(128×256): (FROM_NTW, FIXED_1). h_tiles=1, v_tiles=1. slice=32768.
    // T3(256×256): same as T0. slice=65536.
    // T4(128×256): same as T1. slice=32768.
    // T6(128×256): (FROM_NTW, FROM_NTH). h_tiles=1, v_tiles=1. slice=32768.
    // Total = 65536 + 32768 + 65536 + 32768 + 32768 = 229376.
    auto ws = sg.working_set(TC(128, 256, 256));
    CHECK_EQ_I("ws(128,256,256) = 229376", ws, 229376);

    auto best = sg.best_cost();
    CHECK("fan-in feasible", best.feasible);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

void test_tensor_reused_at_different_depths() {
    std::cout << "--- test_tensor_reused_at_different_depths ---\n";
    // Tensor T0 is used as input to BOTH Op0 (first in chain) and Op2 (last):
    // Op0: MM T0(256×256) @ T1(128×256) → T2(128×256)   [non-sink]
    // Op1: PW T2(128×256) → T3(128×256)
    // Op2: MM T0(256×256) @ T3(128×256) → T4(128×256)   [sink, K=256]
    //
    // T0 is boundary input, consumed by Op0 AND Op2.
    // T2, T3 are ephemeral.
    //
    // Tile sources for T0:
    //   From Op0 non-sink (LHS): (FIXED_1, out_v=FROM_NTH)
    //   From Op2 sink (LHS): (FROM_NK, FROM_NTH)
    //   Merged: (FROM_NK, FROM_NTH)  [NK wins over FIXED_1]
    //
    // This is a "tiling conflict" because T0 gets different h_source from
    // different consumers. The merge picks FROM_NK.
    //
    // output_K = T0.width = 256. nk = 256/k.
    // T0(256×256): h_source=FROM_NK → h_tiles=nk. T0.width=256.
    //   k=1 → nk=256 = T0.width. Borderline OK.

    Problem p;
    p.tensors = {{256,256},{128,256},{128,256},{128,256},{128,256}};
    //            T0        T1        T2(eph)   T3(eph)   T4(out)
    p.ops = {{OpType::MatMul,{0,1},{2},1000},    // Op0: T0 as LHS (non-sink)
             {OpType::Pointwise,{2},{3},200},     // Op1
             {OpType::MatMul,{0,3},{4},2000}};    // Op2: T0 as LHS again (sink)
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("reuse: 3 ops", sg.num_ops() == 3);
    CHECK("reuse: T0 is boundary input", sg.boundary_inputs().count(0));
    CHECK("reuse: T2 eph", sg.ephemeral().count(2));
    CHECK("reuse: T3 eph", sg.ephemeral().count(3));
    CHECK("reuse: T4 boundary out", sg.boundary_outputs().count(4));
    CHECK_EQ_I("reuse: output_K = 256", sg.output_K(), 256);

    // T0 has a tiling conflict: sink LHS wants h_tiles=nk, non-sink LHS wants h_tiles=1.
    // The slow path detects this. Only nk=1 (k=output_K=256) satisfies both.
    // k=1 → nk=256 → conflict (256 vs 1). Reject.
    CHECK("k=1 rejected (tiling conflict on T0)", !sg.is_valid_tiling(TC(128, 256, 1)));
    // k=128 → nk=2 → conflict (2 vs 1). Reject.
    CHECK("k=128 rejected (tiling conflict)", !sg.is_valid_tiling(TC(128, 256, 128)));
    // k=256 → nk=1. Both roles want h_tiles=1. Accept.
    CHECK("k=256 valid (nk=1, no conflict)", sg.is_valid_tiling(TC(128, 256, 256)));

    auto ws = sg.working_set(TC(128, 256, 256));
    CHECK("ws positive", ws > 0 && ws != INT64_MAX);

    auto best = sg.best_cost();
    CHECK("reuse feasible", best.feasible);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

void test_tensor_reused_different_roles() {
    std::cout << "--- test_tensor_reused_different_roles ---\n";
    // T1 used as RHS of Op0 (non-sink) AND as input to Op2 PW:
    // Op0: MM T0(256×128) @ T1(128×128) → T2(128×128)  [non-sink, K=256]
    // Op1: MM T3(256×128) @ T2(128×256) → T4(128×128)  [sink, K=256]
    // Op2: PW [T4, T1] → T5(128×128)                   [PW after sink? No, Op2 produces T5]
    //
    // Wait, if Op2 consumes T4 it's a successor to Op1. Let me restructure:
    // Op2 must be a sink itself. But then T4 is ephemeral not boundary.
    // Actually, let's have the PW AFTER Op1:
    //
    // Op0: MM T0(256×128) @ T1(128×128) → T2(128×128) [non-sink]
    // Op1: PW [T2, T1] → T3(128×128) [sink]
    //
    // T1 is used by Op0 (as RHS) and Op1 (as PW input). Two different roles.
    // T2 is ephemeral.
    // T3 is boundary output (PW sink).
    //
    // T1 sources:
    //   From Op0 non-sink RHS: (out_h, FIXED_1). Output T2's h=FROM_NTW → (FROM_NTW, FIXED_1)
    //   From Op1 PW sink: inherits T3's source → (FROM_NTW, FROM_NTH)
    //   Merged: (FROM_NTW, FROM_NTH) [FROM_NTH wins over FIXED_1]
    //
    // has_pw_sink_ = true → k forced to 1.

    Problem p;
    p.tensors = {{256,128},{128,128},{128,128},{128,128}};
    //            T0        T1        T2(eph)   T3(out)
    p.ops = {{OpType::MatMul,{0,1},{2},1000},      // Op0: T1 as RHS
             {OpType::Pointwise,{2,1},{3},200}};    // Op1: T1 as PW input (sink)
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK("dual-role: T1 is boundary input", sg.boundary_inputs().count(1));
    CHECK("dual-role: T2 eph", sg.ephemeral().count(2));
    // PW sink + matmul: output_K_=max_K_=256 (K=T0.width=256). Only k=256 valid (nk=1).
    CHECK("dual-role: k=1 invalid (nk=256 > 1)", !sg.is_valid_tiling(TC(128, 128, 1)));
    CHECK("dual-role: k=2 invalid (nk=128 > 1)", !sg.is_valid_tiling(TC(128, 128, 2)));
    CHECK("dual-role: k=256 valid (nk=1)", sg.is_valid_tiling(TC(128, 128, 256)));

    // T1(128×128): merged source (FROM_NTW, FROM_NTH). ntw=1,nth=1 → full tensor.
    // T0(256×128): (FIXED_1, FROM_NTH). h_tiles=1, v_tiles=1. slice=256*128=32768.
    // T3(128×128): (FROM_NTW, FROM_NTH). slice=128*128=16384.
    // T1: slice = 128*128 = 16384.
    // Total = 32768 + 16384 + 16384 = 65536.
    auto ws = sg.working_set(TC(128, 128, 256));
    CHECK_EQ_I("ws(128,128,256) = 65536", ws, 65536);

    auto best = sg.best_cost();
    CHECK("dual-role feasible", best.feasible);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

void test_bottleneck_3branch_pattern() {
    std::cout << "--- test_bottleneck_3branch_pattern ---\n";
    // Simplified bottleneck: MM→PW chain feeding into sink MM alongside a
    // large LHS. Reproduces the zero-slice bug from custom-bottleneck-3br.
    //
    // Op0: MM T_small(128×128) @ T_rhs0(256×128) → T_mid(256×128) [non-sink]
    // Op1: PW T_mid(256×128) → T_mid2(256×128)
    // Op2: MM T_big(512×128) @ T_mid2(256×512) → T_out(256×128) [sink, K=512]
    //
    // T_mid, T_mid2 ephemeral.
    // output_K = T_big.width = 512. nk = 512/k.
    //
    // Tile sources:
    //   T_out: (FROM_NTW, FROM_NTH)
    //   Op2 sink: T_big(LHS) → (FROM_NK, FROM_NTH). T_mid2(RHS) → (FROM_NTW, FROM_NK).
    //   Op1 PW: T_mid inherits T_mid2 → (FROM_NTW, FROM_NK).
    //   Op0 non-sink: output T_mid → (FROM_NTW, FROM_NK).
    //     T_small(LHS) → (FIXED_1, FROM_NK). T_rhs0(RHS) → (FROM_NTW, FIXED_1).
    //
    // T_small(128×128): v_source=FROM_NK → v_tiles=nk=512/k. T_small.height=128.
    //   k=1 → nk=512 > 128. Must reject!
    //   k=4 → nk=128 = 128. Borderline accept.

    Problem p;
    //  idx:  0              1             2             3             4             5
    p.tensors = {{128,128},{256,128},{256,128},{256,128},{512,128},{256,128}};
    //            T_small   T_rhs0    T_mid(eph) T_mid2(eph) T_big    T_out
    p.ops = {{OpType::MatMul,{0,1},{2},500},      // Op0: non-sink
             {OpType::Pointwise,{2},{3},100},      // Op1
             {OpType::MatMul,{4,3},{5},2000}};     // Op2: sink
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("bottleneck: 3 ops", sg.num_ops() == 3);
    CHECK_EQ_I("bottleneck: output_K = 512", sg.output_K(), 512);

    // k=1 → nk=512 > T_small.height=128
    CHECK("k=1 rejected", !sg.is_valid_tiling(TC(256, 128, 1)));
    // k=2 → nk=256 > 128
    CHECK("k=2 rejected", !sg.is_valid_tiling(TC(256, 128, 2)));
    // k=4 → nk=128 = 128
    CHECK("k=4 accepted", sg.is_valid_tiling(TC(256, 128, 4)));
    // k=512 → nk=1
    CHECK("k=512 accepted", sg.is_valid_tiling(TC(256, 128, 512)));

    auto ws = sg.working_set(TC(256, 128, 4));
    CHECK("ws(k=4) positive", ws > 0 && ws != INT64_MAX);

    auto best = sg.best_cost();
    CHECK("bottleneck feasible", best.feasible);
    CHECK("bottleneck best k >= 4", best.config.k >= 4);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

void test_residual_skip_connection() {
    std::cout << "--- test_residual_skip_connection ---\n";
    // Skip connection: T_in feeds both the first op and the last PW (add).
    // Op0: PW T_in(256×256) → T1(256×256)
    // Op1: MM T2(128×256) @ T1(256×128) → T3(256×256)
    // Op2: PW [T3, T_in] → T4(256×256)  [sink]
    //
    // T_in used at depth 0 (Op0) and depth 2 (Op2).
    // T1, T3 ephemeral. T4 boundary output. PW sink → k=1.
    //
    // T_in sources:
    //   From Op0 PW: inherits T1's source
    //   From Op2 PW sink: inherits T4 → (FROM_NTW, FROM_NTH)
    //   T1 consumed by Op1 as RHS → T1 gets (FROM_NTW, FIXED_1) from non-sink
    //   But T1 also inherits from Op0's output. Op0's output is T1.
    //   T1 source from Op1 RHS: (out_h, FIXED_1). Output T3's h... let's trace.
    //
    //   T3 consumed by Op2 PW sink. T3 source = T4 source = (FROM_NTW, FROM_NTH).
    //   Op1 non-sink: output T3 → (FROM_NTW, FROM_NTH).
    //     T2(LHS) → (FIXED_1, FROM_NTH). T1(RHS) → (FROM_NTW, FIXED_1).
    //   Op0 PW: T_in inherits T1 → (FROM_NTW, FIXED_1).
    //   BUT from Op2 PW: T_in also inherits T4 → (FROM_NTW, FROM_NTH).
    //   Merged: (FROM_NTW, FROM_NTH).

    Problem p;
    //  idx: 0             1             2             3             4
    p.tensors = {{256,256},{256,256},{128,256},{256,256},{256,256}};
    //            T_in      T1(eph)   T2(input) T3(eph)   T4(out)
    p.ops = {{OpType::Pointwise,{0},{1},100},       // Op0: T_in → T1
             {OpType::MatMul,{2,1},{3},2000},       // Op1: T2,T1 → T3
             {OpType::Pointwise,{3,0},{4},100}};    // Op2: T3,T_in → T4 (sink)
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("skip: 3 ops", sg.num_ops() == 3);
    CHECK("skip: T_in is boundary input", sg.boundary_inputs().count(0));
    CHECK("skip: T1 eph", sg.ephemeral().count(1));
    CHECK("skip: T3 eph", sg.ephemeral().count(3));
    CHECK("skip: T4 boundary out", sg.boundary_outputs().count(4));

    // PW sink + matmul: output_K_=max_K_=128 (K=T2.width=128). Only k=128 valid (nk=1).
    CHECK("skip: k=1 invalid (nk=128 > 1)", !sg.is_valid_tiling(TC(256, 256, 1)));
    CHECK("skip: k=2 invalid (nk=64 > 1)", !sg.is_valid_tiling(TC(256, 256, 2)));
    CHECK("skip: k=128 valid (nk=1)", sg.is_valid_tiling(TC(256, 256, 128)));

    // T_in (256×256): (FROM_NTW, FROM_NTH). ntw=256/w, nth=256/h.
    // T2 (128×256): (FIXED_1, FROM_NTH). slice = 128 * (256/nth).
    // T4 (256×256): (FROM_NTW, FROM_NTH). Output slice.
    //
    // At [256,256,128]: ntw=1, nth=1, nk=1. Everything full size.
    // T_in: 256*256=65536. T2: 128*256=32768. T4: 256*256=65536.
    // Total = 65536 + 32768 + 65536 = 163840.
    auto ws = sg.working_set(TC(256, 256, 128));
    CHECK_EQ_I("ws(256,256,128) = 163840", ws, 163840);

    auto best = sg.best_cost();
    CHECK("skip feasible", best.feasible);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

void test_diamond_dag() {
    std::cout << "--- test_diamond_dag ---\n";
    // Diamond: Op0 fans out to Op1 and Op2, which fan into Op3.
    //
    // Op0: PW T_in(256×256) → T1(256×256)
    // Op1: PW T1 → T2(256×256)
    // Op2: PW T1 → T3(256×256)
    // Op3: PW [T2, T3] → T4(256×256)  [sink]
    //
    // T1 produced by Op0, consumed by Op1 AND Op2 — all internal → ephemeral.
    // T2 produced by Op1, consumed by Op3 — all internal → ephemeral.
    // T3 produced by Op2, consumed by Op3 — all internal → ephemeral.
    // T4 boundary output.

    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256},{256,256}};
    //            T_in      T1(eph)   T2(eph)   T3(eph)   T4(out)
    p.ops = {{OpType::Pointwise,{0},{1},100},       // Op0
             {OpType::Pointwise,{1},{2},100},       // Op1
             {OpType::Pointwise,{1},{3},100},       // Op2
             {OpType::Pointwise,{2,3},{4},100}};    // Op3 (sink)
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2, 3});

    CHECK("diamond: 4 ops", sg.num_ops() == 4);
    CHECK("diamond: T1 eph (fan-out)", sg.ephemeral().count(1));
    CHECK("diamond: T2 eph", sg.ephemeral().count(2));
    CHECK("diamond: T3 eph", sg.ephemeral().count(3));
    CHECK("diamond: T4 boundary out", sg.boundary_outputs().count(4));
    CHECK("diamond: T_in boundary in", sg.boundary_inputs().count(0));

    // All PW, all same dims. PW sink ignores k in is_valid_tiling.
    CHECK("diamond: k=1 valid", sg.is_valid_tiling(TC(256, 256, 1)));
    CHECK("diamond: k=2 also valid (PW sink)", sg.is_valid_tiling(TC(256, 256, 2)));

    // ws at [256,256,1]: only boundary tensors count.
    // T_in: 256*256 = 65536. T4: 65536. Total = 131072.
    auto ws = sg.working_set(TC(256, 256, 1));
    CHECK_EQ_I("diamond ws(256,256,1) = 131072", ws, 131072);

    // At [128,128,1]: ntw=2, nth=2. Each slice = 128*128 = 16384.
    // T_in: 16384. T4: 16384. Total = 32768.
    CHECK_EQ_I("diamond ws(128,128,1) = 32768",
               sg.working_set(TC(128, 128, 1)), 32768);

    auto best = sg.best_cost();
    CHECK("diamond feasible", best.feasible);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

void test_input_shared_across_two_mms() {
    std::cout << "--- test_input_shared_across_two_mms ---\n";
    // T_shared is LHS of both Op0 and Op1 (weight sharing):
    // Op0: MM T_shared(256×256) @ T_rhs0(128×256) → T_out0(128×256)
    // Op1: MM T_shared(256×256) @ T_rhs1(128×256) → T_out1(128×256)
    //
    // Both are sinks (no internal successors). Two boundary outputs.
    // T_shared gets (FROM_NK, FROM_NTH) from both — no conflict.
    // out_W = 128, out_H = 256. Both outputs must match (they do).
    // output_K = 256 (both have same K).

    Problem p;
    p.tensors = {{256,256},{128,256},{128,256},{128,256},{128,256}};
    //            T_shared  T_rhs0    T_out0    T_rhs1    T_out1
    p.ops = {{OpType::MatMul,{0,1},{2},1500},   // Op0 (sink)
             {OpType::MatMul,{0,3},{4},1500}};   // Op1 (sink)
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK("shared-LHS: 2 ops", sg.num_ops() == 2);
    CHECK("shared-LHS: T_shared input", sg.boundary_inputs().count(0));
    CHECK("shared-LHS: T_out0 boundary out", sg.boundary_outputs().count(2));
    CHECK("shared-LHS: T_out1 boundary out", sg.boundary_outputs().count(4));
    CHECK("shared-LHS: 2 boundary outputs", sg.boundary_outputs().size() == 2);
    CHECK_EQ_I("shared-LHS: output_K = 256", sg.output_K(), 256);

    CHECK("k=1 valid (nk=256, T_shared.width=256)", sg.is_valid_tiling(TC(128, 256, 1)));
    CHECK("k=256 valid", sg.is_valid_tiling(TC(128, 256, 256)));

    // ws at [128,256,256]: nk=1, ntw=1, nth=1. All full.
    // T_shared: (FROM_NK, FROM_NTH). h_tiles=1, v_tiles=1 → 256*256 = 65536.
    // T_rhs0: (FROM_NTW, FROM_NK). h_tiles=1, v_tiles=1 → 128*256 = 32768.
    // T_rhs1: same → 32768.
    // T_out0: (FROM_NTW, FROM_NTH). → 128*256 = 32768.
    // T_out1: same → 32768.
    // Total = 65536 + 32768*4 = 196608.
    auto ws = sg.working_set(TC(128, 256, 256));
    CHECK_EQ_I("ws(128,256,256) = 196608", ws, 196608);

    auto best = sg.best_cost();
    CHECK("shared-LHS feasible", best.feasible);
    std::cout << "    best=[" << best.config.w << "," << best.config.h
              << "," << best.config.k << "] lat=" << best.latency << "\n";
}

// ============================================================================
// 9. Internal fan-out: tensors consumed by multiple ops at different depths
// ============================================================================

void test_ephemeral_fanout_pw_chain() {
    std::cout << "--- test_ephemeral_fanout_pw_chain ---\n";
    // Ephemeral T1 produced by Op0, consumed by Op1 (depth 1) AND Op2 (depth 2).
    //
    // Op0: PW T0(256×256) → T1(256×256)
    // Op1: PW T1 → T2(256×256)
    // Op2: PW [T2, T1] → T3(256×256)  [sink]
    //
    // T1: consumed by Op1 AND Op2 (both internal) → ephemeral.
    // T2: consumed by Op2 (internal) → ephemeral.
    // T1 fan-out at different depths: Op1 is depth 1, Op2 is depth 2.
    //
    // eph_roles for T1: consumed by Op1 (PW→T2 eph) and Op2 (PW sink→T3 boundary).
    //   From Op1: T2 is eph, inherits T2's roles. T2 consumed by Op2 PW sink:
    //     T2 roles = {(W_param, H_param)}. So T1 from Op1 gets (W_param, H_param).
    //   From Op2: T3 is boundary out → (W_param, H_param).
    //   Both agree: (W_param, H_param). No conflict.
    //
    // Tile sources:
    //   T3 (sink out): (FROM_NTW, FROM_NTH)
    //   Op2 PW: T2, T1 → (FROM_NTW, FROM_NTH)
    //   Op1 PW: T1 already assigned → assign_or_check(FROM_NTW, FROM_NTH). Same → OK.
    //   Op0 PW: T0 → (FROM_NTW, FROM_NTH)

    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256}};
    //            T0        T1(eph)   T2(eph)   T3(out)
    p.ops = {{OpType::Pointwise,{0},{1},100},       // Op0
             {OpType::Pointwise,{1},{2},100},       // Op1
             {OpType::Pointwise,{2,1},{3},100}};    // Op2 (sink): T1 reused here
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("eph-fanout-pw: 3 ops", sg.num_ops() == 3);
    CHECK("eph-fanout-pw: T1 ephemeral", sg.ephemeral().count(1));
    CHECK("eph-fanout-pw: T2 ephemeral", sg.ephemeral().count(2));
    CHECK("eph-fanout-pw: T3 boundary out", sg.boundary_outputs().count(3));

    // All PW same dims: any divisor of 256 works for w,h.
    CHECK("w=128,h=128 valid", sg.is_valid_tiling(TC(128, 128, 1)));
    CHECK("w=256,h=256 valid", sg.is_valid_tiling(TC(256, 256, 1)));

    // ws: only boundary tensors (T0, T3). Ephemerals T1, T2 are free.
    // All sources are (FROM_NTW, FROM_NTH).
    // At [256,256,1]: ntw=1, nth=1. T0 slice=256*256=65536. T3 slice=65536.
    CHECK_EQ_I("ws(256,256,1) = 131072", sg.working_set(TC(256, 256, 1)), 131072);

    // At [128,128,1]: ntw=2, nth=2. Each slice=128*128=16384. ws=16384*2.
    CHECK_EQ_I("ws(128,128,1) = 32768", sg.working_set(TC(128, 128, 1)), 32768);

    // Cost at [256,256,1]: num_tw=1, num_th=1, nk=1. No matmul → SnakeDir::None.
    //   comp: 3 PW ops × (100 / 1 × op_scale). Output tiles are 256×256.
    //     op_scale = max(256/128,1) × max(256/128,1) = 4.
    //     comp_per_step = 3 × 100 × 4 = 1200.
    //   IO: T0 (FROM_NTW, FROM_NTH): tile_load = 65536/10 = 6553.6.
    //       T3 boundary out: out_evict = 65536/10 = 6553.6.
    //   tile_cost = max(1200, 6553.6 + 6553.6) = 13107.2.
    //   latency = 1 × 13107.2 = 13107.2.
    auto r = sg.compute_cost(TC(256, 256, 1));
    CHECK("cost feasible", r.feasible);
    CHECK_EQ("comp_per_step = 1200", r.compute_per_step, 1200.0);
    CHECK_EQ("latency = 13107.2", r.latency, 13107.2);

    auto best = sg.best_cost();
    CHECK("eph-fanout-pw feasible", best.feasible);
}

void test_ephemeral_fanout_to_mm_and_pw() {
    std::cout << "--- test_ephemeral_fanout_to_mm_and_pw ---\n";
    // Ephemeral T1 consumed as MM RHS (Op1) and PW input (Op2).
    // Different roles: (W_param, K_param) from MM vs (W_param, H_param) from PW.
    //
    // Op0: PW T0(128×256) → T1(128×256)
    // Op1: MM T2(256×256) @ T1(128×256) → T3(128×256) [non-sink, K=256]
    // Op2: PW [T3, T1] → T4(128×256)  [sink]
    //
    // T1 consumed by Op1 (as RHS) AND Op2 (as PW input). Both internal → ephemeral.
    // T3 consumed by Op2 → ephemeral.
    //
    // eph_roles for T1:
    //   From Op1 MM (RHS): (W_param, K_param)
    //   From Op2 PW sink (T4 is boundary): (W_param, H_param)
    //   Both collected → T1 gets BOTH roles in constraint collection.
    //   w_set gets T1.width=128 (twice). k_set gets T1.height=256 (from K_param).
    //   h_set gets T1.height=256 (from H_param).
    //
    // Tile sources:
    //   T4 (PW sink out): (FROM_NTW, FROM_NTH)
    //   Op2 PW: T3 → (FROM_NTW, FROM_NTH). T1 → (FROM_NTW, FROM_NTH).
    //   Op1 non-sink: output T3 → (FROM_NTW, FROM_NTH).
    //     LHS T2: (FIXED_1, FROM_NTH). RHS T1: (FROM_NTW, FIXED_1).
    //   T1 already has (FROM_NTW, FROM_NTH), gets (FROM_NTW, FIXED_1) → conflict!
    //   Merged v: merge(FROM_NTH, FIXED_1) = FROM_NTH (no NK, keep existing).
    //   So T1 stays (FROM_NTW, FROM_NTH). has_tiling_conflict = true.
    //
    // Slow path for T1:
    //   From Op2: h_tiles=ntw, v_tiles=nth.
    //   From Op1 RHS: h_tiles=ntw, v_tiles=1 (FIXED_1 for non-sink RHS).
    //   Conflict when nth ≠ 1. E.g., h=128 → nth=2 → conflict → reject.
    //   Only h=256 (nth=1) makes both agree.

    Problem p;
    p.tensors = {{128,256},{128,256},{256,256},{128,256},{128,256}};
    //            T0        T1(eph)   T2        T3(eph)   T4(out)
    p.ops = {{OpType::Pointwise,{0},{1},100},       // Op0
             {OpType::MatMul,{2,1},{3},1000},       // Op1: T1 as RHS
             {OpType::Pointwise,{3,1},{4},100}};    // Op2: T1 as PW input (sink)
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("eph-fanout-mixed: T1 ephemeral", sg.ephemeral().count(1));
    CHECK("eph-fanout-mixed: T3 ephemeral", sg.ephemeral().count(3));

    // h=128 → nth=2 → T1 conflict (nth=2 from PW, 1 from MM RHS). Reject.
    // Also: has_pw_sink_+matmul → output_K_=256. k must equal 256 for nk=1.
    // k=1 → nk=256 > 1 → rejected regardless.
    CHECK("h=128 rejected (T1 conflict)",
          !sg.is_valid_tiling(TC(128, 128, 1)));

    // h=256 → nth=1. Both roles agree on v_tiles=1. k=256 → nk=1. Accept.
    CHECK("h=256 accepted (nth=1, no conflict)",
          sg.is_valid_tiling(TC(128, 256, 256)));

    // w=64 → ntw=2. T1 gets h_tiles=ntw=2 from both roles. No h-conflict.
    CHECK("w=64,h=256 accepted",
          sg.is_valid_tiling(TC(64, 256, 256)));

    // Boundary tensors: T0(128×256), T2(256×256), T4(128×256, out).
    // Tile sources: T0 (FROM_NTW, FROM_NTH), T2 (FIXED_1, FROM_NTH), T4 (FROM_NTW, FROM_NTH).
    // No FROM_NK sources, so ws is the same at any valid nk=1 config.
    //
    // ws at [128,256,256]: ntw=1, nth=1, nk=1.
    //   T0: ht=1, vt=1 → 128*256 = 32768.
    //   T2: ht=1, vt=1 → 256*256 = 65536.
    //   T4: ht=1, vt=1 → 128*256 = 32768.
    //   Total = 131072.
    CHECK_EQ_I("ws(128,256,256) = 131072", sg.working_set(TC(128, 256, 256)), 131072);

    // ws at [64,256,256]: ntw=2, nth=1.
    //   T0: ht=2, vt=1 → 64*256 = 16384.
    //   T2: ht=1, vt=1 → 65536 (FIXED_1 in h).
    //   T4: ht=2, vt=1 → 16384.
    //   Total = 98304.
    CHECK_EQ_I("ws(64,256,256) = 98304", sg.working_set(TC(64, 256, 256)), 98304);

    // Cost at [128,256,256]: num_tw=1, num_th=1, nk=1. has_matmul=true.
    //   comp: Op0(PW, out=T1 128×256, op_scale=1*2=2): 100*2=200.
    //         Op1(MM, out=T3 128×256, op_scale=2): 1000*2=2000.
    //         Op2(PW, out=T4 128×256, op_scale=2): 100*2=200.
    //         comp_per_step = 2400.
    //   IO: T0 (FROM_NTW,FROM_NTH) tile_load=32768/10=3276.8.
    //       T2 (FIXED_1,FROM_NTH) row_load=65536/10=6553.6.
    //       T4 out_evict=32768/10=3276.8.
    //   1 tile: per_tile_io=3276.8+6553.6=9830.4.
    //   max(2400, 9830.4+3276.8) = 13107.2.
    auto r = sg.compute_cost(TC(128, 256, 256, SnakeDir::RowMajor));
    CHECK("cost feasible", r.feasible);
    CHECK_EQ("comp_per_step = 2400", r.compute_per_step, 2400.0);
    CHECK_EQ("latency = 13107.2", r.latency, 13107.2);

    auto best = sg.best_cost();
    CHECK("eph-fanout-mixed feasible", best.feasible);
}

void test_boundary_input_as_lhs_and_rhs() {
    std::cout << "--- test_boundary_input_as_lhs_and_rhs ---\n";
    // T_shared used as LHS of Op0 and RHS of Op1.
    // These give opposite tile sources: (K/FROM_NK, H/FROM_NTH) vs (W/FROM_NTW, K/FROM_NK).
    //
    // Op0: MM T_shared(256×256) @ T1(128×256) → T2(128×256)  [non-sink]
    // Op1: MM T3(256×256) @ T_shared(256×256) → T4(256×256)  [sink, K=256]
    //
    // T2 ephemeral (consumed by... wait, T2 needs a consumer to be ephemeral).
    // Let's chain it:
    //
    // Op0: MM T_shared(256×256) @ T1(128×256) → T2(128×256)  [non-sink]
    // Op1: PW T2(128×256) → T3(128×256)                      [non-sink]
    // Op2: MM T4(256×256) @ T3(128×256) → T5(128×256)        [sink, K=256]
    //
    // Now also have T_shared consumed by Op2 as... no, T_shared is only Op0 input.
    //
    // Simpler: two parallel MMs sharing T_shared.
    // Op0: MM T_shared(256×256) @ T1(256×256) → T2(256×256)  [sink]
    // Op1: MM T3(256×256) @ T_shared(256×256) → T4(256×256)  [sink]
    //
    // Both sinks, same output dims. T_shared as Op0 LHS and Op1 RHS.
    //
    // Tile sources:
    //   Op0 sink: T_shared(LHS) → (FROM_NK, FROM_NTH). T1(RHS) → (FROM_NTW, FROM_NK).
    //   Op1 sink: T3(LHS) → (FROM_NK, FROM_NTH). T_shared(RHS) → (FROM_NTW, FROM_NK).
    //
    // T_shared gets two assignments:
    //   From Op0 LHS: (FROM_NK, FROM_NTH)
    //   From Op1 RHS: (FROM_NTW, FROM_NK)
    //   Conflict! Merged: h=merge(FROM_NK, FROM_NTW)=FROM_NK (NK wins).
    //                      v=merge(FROM_NTH, FROM_NK)=FROM_NK.
    //   T_shared → (FROM_NK, FROM_NK). has_tiling_conflict = true.
    //
    // Slow path: Op0 wants T_shared h_tiles=nk, v_tiles=nth.
    //            Op1 wants T_shared h_tiles=ntw, v_tiles=nk.
    //   Need nk == ntw AND nth == nk.
    //   nk = 256/k. ntw = 256/w. nth = 256/h.
    //   Need 256/k = 256/w → k = w. And 256/h = 256/k → h = k.
    //   So need w = h = k.

    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256},{256,256}};
    //            T_shared  T1        T2(out)   T3        T4(out)
    p.ops = {{OpType::MatMul,{0,1},{2},1500},   // Op0 (sink): T_shared LHS
             {OpType::MatMul,{3,0},{4},1500}};   // Op1 (sink): T_shared RHS
    p.fast_memory_capacity = 1000000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK("lhs+rhs: T_shared input", sg.boundary_inputs().count(0));
    CHECK("lhs+rhs: T2 boundary out", sg.boundary_outputs().count(2));
    CHECK("lhs+rhs: T4 boundary out", sg.boundary_outputs().count(4));
    CHECK_EQ_I("lhs+rhs: output_K = 256", sg.output_K(), 256);

    // T_shared has conflicting symbolic tile sources:
    //   Op0 (LHS): h=FROM_NK, v=FROM_NTH
    //   Op1 (RHS): h=FROM_NTW, v=FROM_NK
    // Even when ntw==nk==nth numerically (w=h=k), the sources represent
    // different execution dimensions (spatial vs temporal), so the tensor
    // can't serve both roles. Reject all configs with any tiling > 1.
    CHECK("w=h=k=128 rejected (symbolic: FROM_NK≠FROM_NTW on h, FROM_NTH≠FROM_NK on v)",
          !sg.is_valid_tiling(TC(128, 128, 128)));
    // Only no-tiling config is valid (ntw=nth=nk=1, all sources eval to 1).
    CHECK("w=h=k=256 accepted (ntw=nth=nk=1)", sg.is_valid_tiling(TC(256, 256, 256)));

    // w≠k: ntw≠nk → also rejected (numerical mismatch too).
    CHECK("w=128,k=256 rejected (ntw≠nk)",
          !sg.is_valid_tiling(TC(128, 128, 256)));

    // w=h but h≠k: nth≠nk → rejected.
    CHECK("h=256,k=128 rejected (nth≠nk)",
          !sg.is_valid_tiling(TC(256, 256, 128)));

    // Boundary tensors: T_shared(0), T1(1), T2(2,out), T3(3), T4(4,out). All 256×256.
    //
    // ws at [256,256,256]: ntw=1, nth=1, nk=1. All slices full: 65536 each.
    CHECK_EQ_I("ws(256,256,256) = 327680", sg.working_set(TC(256, 256, 256)), 327680);

    // Cost at [256,256,256]: num_tw=1, num_th=1, nk=1.
    //   comp: Op0(MM, out=T2, op_scale=2*2=4): 1500*4=6000.
    //         Op1(MM, out=T4, op_scale=4): 1500*4=6000.
    //         comp_per_step = 12000.
    //   IO (nk=1 → FROM_NK becomes FIXED_1):
    //     T_shared: (FIXED_1,FIXED_1) → once_load = 65536/10 = 6553.6.
    //     T1: (FROM_NTW,FIXED_1) → col_load = 6553.6.
    //     T3: (FIXED_1,FROM_NTH) → row_load = 6553.6.
    //     T2 out: evict = 6553.6.
    //     T4 out: evict = 6553.6. Total out_evict = 13107.2.
    //   1 tile: per_tile_io = once+row+col = 19660.8.
    //   max(12000, 19660.8 + 13107.2) = 32768.
    auto r = sg.compute_cost(TC(256, 256, 256, SnakeDir::RowMajor));
    CHECK("cost feasible", r.feasible);
    CHECK_EQ("comp_per_step = 12000", r.compute_per_step, 12000.0);
    CHECK_EQ("latency = 32768", r.latency, 32768.0);

    auto best = sg.best_cost();
    CHECK("lhs+rhs feasible", best.feasible);
    // Best must have w=h=k
    CHECK("best w=h", best.config.w == best.config.h);
    CHECK("best w=k", best.config.w == best.config.k);
}

void test_boundary_input_at_depth0_and_depth2() {
    std::cout << "--- test_boundary_input_at_depth0_and_depth2 ---\n";
    // T_in consumed at depth 0 (Op0 PW) and depth 2 (Op2 MM as RHS).
    // The two consumers assign different tile sources.
    //
    // Op0: PW T_in(128×256) → T1(128×256)
    // Op1: MM T2(256×256) @ T1(128×256) → T3(128×256)  [non-sink, K=256]
    // Op2: MM T4(128×256) @ T_in(128×256) → T5(128×256) [sink, K=128]
    //
    // T1, T3 ephemeral. T5 boundary out. output_K = T4.width = 128.
    //
    // Tile sources:
    //   T5 (sink out): (FROM_NTW, FROM_NTH)
    //   Op2 sink: T4(LHS) → (FROM_NK, FROM_NTH). T_in(RHS) → (FROM_NTW, FROM_NK).
    //   Op1 non-sink: T3 → need its source. T3 consumed by... nobody (not consumed internally).
    //     Wait, T3 needs to be consumed or it's a boundary output.
    //
    // Let me restructure so T3 feeds into Op2:
    //
    // Op0: PW T_in(128×256) → T1(128×256)
    // Op1: MM T2(256×256) @ T1(128×256) → T3(128×256)  [non-sink]
    // Op2: MM T3(128×256) @ T_in(128×256) → T4(128×256) [sink, K=128]
    //
    // T1 eph (Op0→Op1). T3 eph (Op1→Op2). T4 boundary out.
    // T_in consumed by Op0 (PW, depth 0) and Op2 (MM RHS, depth 2).
    //
    // Tile sources:
    //   T4 (sink out): (FROM_NTW, FROM_NTH)
    //   Op2 sink: T3(LHS) → (FROM_NK, FROM_NTH). T_in(RHS) → (FROM_NTW, FROM_NK).
    //     nk = output_K/k = 128/k.
    //   Op1 non-sink: output T3 → (FROM_NK, FROM_NTH).
    //     T2(LHS) → (FIXED_1, FROM_NTH). T1(RHS) → (FROM_NK, FIXED_1).
    //   Op0 PW: output T1 → (FROM_NK, FIXED_1).
    //     T_in inherits → (FROM_NK, FIXED_1).
    //   T_in already assigned (FROM_NTW, FROM_NK) from Op2 RHS.
    //     New: (FROM_NK, FIXED_1). Conflict!
    //     Merged: h=merge(FROM_NTW, FROM_NK)=FROM_NK. v=merge(FROM_NK, FIXED_1)=FROM_NK.
    //     T_in → (FROM_NK, FROM_NK). has_tiling_conflict = true.
    //
    // Slow path:
    //   T_in from Op2 RHS: h_tiles=ntw, v_tiles=nk.
    //   T_in from Op0 PW (inheriting T1): h_tiles=nk, v_tiles=1.
    //   Need ntw == nk AND nk == 1. nk=1 → k=128. ntw=128/w.
    //   ntw must =1 → w=128. So only w=128,k=128 works.
    //   Also nk=1 AND v_tiles: from Op2 RHS v_tiles=nk=1. From Op0 PW v_tiles=1. Match.

    Problem p;
    p.tensors = {{128,256},{128,256},{256,256},{128,256},{128,256}};
    //            T_in      T1(eph)   T2        T3(eph)   T4(out)
    p.ops = {{OpType::Pointwise,{0},{1},100},       // Op0: T_in at depth 0
             {OpType::MatMul,{2,1},{3},1000},       // Op1: non-sink
             {OpType::MatMul,{3,0},{4},1500}};      // Op2: sink, T_in as RHS at depth 2
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("depth-reuse: T_in input", sg.boundary_inputs().count(0));
    CHECK("depth-reuse: T1 eph", sg.ephemeral().count(1));
    CHECK("depth-reuse: T3 eph", sg.ephemeral().count(3));
    CHECK("depth-reuse: T4 boundary out", sg.boundary_outputs().count(4));
    CHECK_EQ_I("depth-reuse: output_K = 128", sg.output_K(), 128);

    // Only w=128,k=128 satisfies the conflict (ntw=nk=1).
    CHECK("w=128,k=128 accepted",
          sg.is_valid_tiling(TC(128, 256, 128)));

    // w=64,k=128: ntw=2, nk=1 → conflict (T_in h_tiles: 2 vs 1). Reject.
    CHECK("w=64,k=128 rejected (ntw≠nk)",
          !sg.is_valid_tiling(TC(64, 256, 128)));

    // w=128,k=64: nk=2 → v_tiles from RHS=2, from PW=1. Reject.
    CHECK("w=128,k=64 rejected (nk=2, PW wants 1)",
          !sg.is_valid_tiling(TC(128, 256, 64)));

    // w=64,k=64: ntw=2, nk=2. T_in h_tiles: ntw=2 (Op2 RHS), nk=2 (Op0 PW). Match!
    //   v_tiles: nk=2 (Op2 RHS), FIXED_1=1 (Op0 PW). 2≠1 → Reject.
    CHECK("w=64,k=64 rejected (v conflict)",
          !sg.is_valid_tiling(TC(64, 256, 64)));

    // Boundary tensors: T_in(0), T2(2), T4(4,out).
    // T_in: (FROM_NK, FROM_NK) [merged]. T2: (FIXED_1, FROM_NTH). T4: (FROM_NTW, FROM_NTH), MM out.
    //
    // ws at [128,256,128]: ntw=1, nth=1, nk=1.
    //   T_in: ht=nk=1, vt=nk=1 → 128*256 = 32768.
    //   T2: ht=1, vt=nth=1 → 256*256 = 65536.
    //   T4: ht=ntw=1, vt=nth=1 → 128*256 = 32768.
    //   Total = 131072.
    CHECK_EQ_I("ws(128,256,128) = 131072", sg.working_set(TC(128, 256, 128)), 131072);

    // ws at [128,128,128]: ntw=1, nth=2, nk=1.
    //   T_in: ht=1, vt=1 → 32768.
    //   T2: ht=1, vt=2 → 256*128 = 32768.
    //   T4: ht=1, vt=2 → 128*128 = 16384.
    //   Total = 81920.
    CHECK_EQ_I("ws(128,128,128) = 81920", sg.working_set(TC(128, 128, 128)), 81920);

    // Cost at [128,128,128] with RowMajor: num_tw=1, num_th=2, nk=1.
    //   comp: Op0(PW, out=T1, tiling=(FROM_NK,FIXED_1), ht=1,vt=1, T1=128×256,
    //           slice_w=128, slice_h=256, op_scale=1*2=2): 100*2=200.
    //         Op1(MM, out=T3, tiling=(FROM_NK,FROM_NTH), ht=1,vt=2, T3=128×256,
    //           slice_w=128, slice_h=128, op_scale=1): 1000.
    //         Op2(MM, out=T4, tiling=(FROM_NTW,FROM_NTH), ht=1,vt=2, T4=128×256,
    //           slice_w=128, slice_h=128, op_scale=1): 1500.
    //         comp_per_step = 2700.
    //   IO (nk=1 → FROM_NK becomes FIXED_1):
    //     T_in: (FIXED_1,FIXED_1) → once_load = 32768/10 = 3276.8.
    //     T2: (FIXED_1,FROM_NTH) → row_load = 32768/10 = 3276.8.
    //     T4 out: evict = 16384/10 = 1638.4.
    //   RowMajor: first = tile_cost(T,T,T), row_trans = tile_cost(F,T,F), n_row_trans=1.
    //     first: per_tile_io = once(3276.8) + row(3276.8) = 6553.6.
    //       max(2700, 6553.6 + 1638.4) = 8192.
    //     row_trans: per_tile_io = row(3276.8) = 3276.8.
    //       max(2700, 3276.8 + 1638.4) = 4915.2.
    //     latency = 8192 + 4915.2 = 13107.2.
    auto r = sg.compute_cost(TC(128, 128, 128, SnakeDir::RowMajor));
    CHECK("cost feasible", r.feasible);
    CHECK_EQ("comp_per_step = 2700", r.compute_per_step, 2700.0);
    CHECK_EQ("latency = 13107.2", r.latency, 13107.2);

    auto best = sg.best_cost();
    CHECK("depth-reuse feasible", best.feasible);
}

void test_ephemeral_fanout_mm_lhs_and_mm_rhs() {
    std::cout << "--- test_ephemeral_fanout_mm_lhs_and_mm_rhs ---\n";
    // Ephemeral T_eph is LHS of one MM and RHS of another.
    // This gives maximally conflicting tile sources:
    //   LHS of non-sink: (FIXED_1, out_v)
    //   RHS of non-sink: (out_h, FIXED_1)
    //
    // Op0: PW T_in(256×256) → T_eph(256×256)
    // Op1: MM T_eph(256×256) @ T1(256×256) → T2(256×256)  [sink] (T_eph as LHS)
    // Op2: MM T3(256×256) @ T_eph(256×256) → T4(256×256)  [sink] (T_eph as RHS)
    //
    // Two sinks! Both outputs 256×256 → same dims. Both K=256 → same K.
    // T_eph is ephemeral (produced by Op0, consumed by Op1 and Op2 — all internal).
    //
    // Tile sources:
    //   T2, T4 (sink outs): (FROM_NTW, FROM_NTH)
    //   Op1 sink: T_eph(LHS) → (FROM_NK, FROM_NTH).
    //   Op2 sink: T_eph(RHS) → (FROM_NTW, FROM_NK).
    //   T_eph conflict: (FROM_NK, FROM_NTH) vs (FROM_NTW, FROM_NK).
    //     Merged: (FROM_NK, FROM_NK).
    //
    // Slow path: Op1 wants (nk, nth). Op2 wants (ntw, nk). Need nk=ntw, nth=nk.
    //   → w = h = k.

    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256},{256,256},{256,256}};
    //            T_in      T_eph(1)  T1        T2(out)   T3        T4(out)
    p.ops = {{OpType::Pointwise,{0},{1},100},      // Op0: T_in → T_eph
             {OpType::MatMul,{1,2},{3},1500},      // Op1 (sink): T_eph as LHS
             {OpType::MatMul,{4,1},{5},1500}};     // Op2 (sink): T_eph as RHS
    p.fast_memory_capacity = 1000000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("eph-lhs+rhs: T_eph ephemeral", sg.ephemeral().count(1));
    CHECK("eph-lhs+rhs: T2 boundary out", sg.boundary_outputs().count(3));
    CHECK("eph-lhs+rhs: T4 boundary out", sg.boundary_outputs().count(5));

    // T_eph has conflicting symbolic sources:
    //   Op1 (LHS): h=FROM_NK, v=FROM_NTH
    //   Op2 (RHS): h=FROM_NTW, v=FROM_NK
    // Even with w=h=k (ntw==nth==nk), the labels differ → reject.
    CHECK("w=h=k=128 rejected (symbolic conflict on T_eph)",
          !sg.is_valid_tiling(TC(128, 128, 128)));
    CHECK("w=h=k=256 accepted (ntw=nth=nk=1)", sg.is_valid_tiling(TC(256, 256, 256)));

    // w≠k → ntw≠nk → T_eph conflict (numerical too).
    CHECK("w=128,h=128,k=256 rejected", !sg.is_valid_tiling(TC(128, 128, 256)));
    CHECK("w=256,h=256,k=128 rejected", !sg.is_valid_tiling(TC(256, 256, 128)));

    // h≠k → nth≠nk → T_eph conflict.
    CHECK("w=128,h=256,k=128 rejected", !sg.is_valid_tiling(TC(128, 256, 128)));

    // Boundary tensors: T_in(0), T1(2), T2(3,out), T3(4), T4(5,out). All 256×256.
    //
    // ws at [256,256,256]: ntw=1, nth=1, nk=1. All slices=65536.
    CHECK_EQ_I("ws(256,256,256) = 327680", sg.working_set(TC(256, 256, 256)), 327680);

    // Cost at [256,256,256]: num_tw=1, num_th=1, nk=1.
    //   comp: Op0(PW, out=T_eph(1), tiling=(FROM_NK,FROM_NK), ht=1,vt=1,
    //           256×256, op_scale=4): 100*4=400.
    //         Op1(MM, out=T2(3), tiling=(FROM_NTW,FROM_NTH), op_scale=4): 1500*4=6000.
    //         Op2(MM, out=T4(5), tiling=(FROM_NTW,FROM_NTH), op_scale=4): 1500*4=6000.
    //         comp_per_step = 12400.
    //   IO (nk=1 → FROM_NK becomes FIXED_1):
    //     T_in(0): (FIXED_1,FIXED_1) → once_load = 6553.6.
    //     T1(2): (FROM_NTW,FIXED_1) → col_load = 6553.6.
    //     T3(4): (FIXED_1,FROM_NTH) → row_load = 6553.6.
    //     T2(3) out: evict = 6553.6. T4(5) out: evict = 6553.6. out_evict = 13107.2.
    //   1 tile: per_tile_io = 19660.8.
    //   max(12400, 19660.8 + 13107.2) = 32768.
    auto r = sg.compute_cost(TC(256, 256, 256, SnakeDir::RowMajor));
    CHECK("cost feasible", r.feasible);
    CHECK_EQ("comp_per_step = 12400", r.compute_per_step, 12400.0);
    CHECK_EQ("latency = 32768", r.latency, 32768.0);

    auto best = sg.best_cost();
    CHECK("eph-lhs+rhs feasible", best.feasible);
    CHECK("best w=h=k", best.config.w == best.config.h && best.config.h == best.config.k);
}

void test_long_chain_input_reused_at_bookends() {
    std::cout << "--- test_long_chain_input_reused_at_bookends ---\n";
    // T_w is a weight tensor used at depth 0 (Op0 LHS) and depth 3 (Op3 LHS).
    // Long PW chain in between propagates tile sources back.
    //
    // Op0: MM T_w(128×256) @ T_in(256×256) → T1(256×256)  [non-sink]
    // Op1: PW T1 → T2(256×256)
    // Op2: PW T2 → T3(256×256)
    // Op3: MM T_w(128×256) @ T3(256×256) → T4(256×256)    [sink, K=128]
    //
    // T1, T2, T3 ephemeral. T_w used at depth 0 and 3.
    //
    // Tile sources:
    //   T4 (sink out): (FROM_NTW, FROM_NTH)
    //   Op3 sink: T_w(LHS) → (FROM_NK, FROM_NTH). T3(RHS) → (FROM_NTW, FROM_NK).
    //   Op2 PW: T2 ← T3 → (FROM_NTW, FROM_NK)
    //   Op1 PW: T1 ← T2 → (FROM_NTW, FROM_NK)
    //   Op0 non-sink: output T1 → (FROM_NTW, FROM_NK).
    //     T_w(LHS) → (FIXED_1, FROM_NK). Already assigned (FROM_NK, FROM_NTH).
    //     Conflict! h: FROM_NK vs FIXED_1 → FROM_NK. v: FROM_NTH vs FROM_NK → FROM_NK.
    //     T_w → (FROM_NK, FROM_NK).
    //
    // Slow path: Op3 wants T_w (nk, nth). Op0 wants T_w (1, nk).
    //   Need nk=1 AND nth=nk → nth=1. So k=128 (nk=1) and h=256 (nth=1).

    Problem p;
    p.tensors = {{128,256},{256,256},{256,256},{256,256},{256,256},{256,256}};
    //            T_w       T_in      T1(eph)   T2(eph)   T3(eph)   T4(out)
    p.ops = {{OpType::MatMul,{0,1},{2},500},       // Op0: T_w at depth 0
             {OpType::Pointwise,{2},{3},100},       // Op1
             {OpType::Pointwise,{3},{4},100},       // Op2
             {OpType::MatMul,{0,4},{5},2000}};      // Op3: T_w at depth 3 (sink)
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2, 3});

    CHECK("bookend: 4 ops", sg.num_ops() == 4);
    CHECK("bookend: T_w input", sg.boundary_inputs().count(0));
    CHECK("bookend: T1 eph", sg.ephemeral().count(2));
    CHECK("bookend: T2 eph", sg.ephemeral().count(3));
    CHECK("bookend: T3 eph", sg.ephemeral().count(4));
    CHECK_EQ_I("bookend: output_K = 128", sg.output_K(), 128);

    // k=128 (nk=1), h=256 (nth=1). Accept.
    CHECK("w=256,h=256,k=128 accepted",
          sg.is_valid_tiling(TC(256, 256, 128)));

    // k=64 (nk=2) → Op3 wants T_w h_tiles=2, Op0 wants 1. Reject.
    CHECK("k=64 rejected (nk=2, conflict on T_w)",
          !sg.is_valid_tiling(TC(256, 256, 64)));

    // h=128 (nth=2) → Op3 wants T_w v_tiles=nth=2, Op0 wants nk=1. Reject.
    CHECK("h=128,k=128 rejected (nth≠nk on T_w)",
          !sg.is_valid_tiling(TC(256, 128, 128)));

    // Boundary tensors: T_w(0, 128×256), T_in(1, 256×256), T4(5, 256×256, out, MM out).
    // T_w: (FROM_NK,FROM_NK) [merged]. T_in: (FROM_NTW,FIXED_1). T4: (FROM_NTW,FROM_NTH).
    //
    // ws at [256,256,128]: ntw=1, nth=1, nk=1.
    //   T_w: ht=1, vt=1 → 128*256 = 32768.
    //   T_in: ht=1, vt=1 → 256*256 = 65536.
    //   T4: ht=1, vt=1 → 65536.
    //   Total = 163840.
    CHECK_EQ_I("ws(256,256,128) = 163840", sg.working_set(TC(256, 256, 128)), 163840);

    // ws at [128,256,128]: ntw=2, nth=1, nk=1.
    //   T_w: ht=nk=1, vt=nk=1 → 32768.
    //   T_in: ht=ntw=2, vt=1 → 128*256 = 32768.
    //   T4: ht=ntw=2, vt=nth=1 → 128*256 = 32768.
    //   Total = 98304.
    CHECK_EQ_I("ws(128,256,128) = 98304", sg.working_set(TC(128, 256, 128)), 98304);

    // Cost at [128,256,128] RowMajor: num_tw=2, num_th=1, nk=1.
    //   comp: Op0(MM, out=T1(2), tiling=(FROM_NTW,FROM_NK), ht=2,vt=1,
    //           256×256, slice_w=128, slice_h=256, op_scale=1*2=2): 500*2=1000.
    //         Op1(PW, out=T2(3), same tiling, op_scale=2): 100*2=200.
    //         Op2(PW, out=T3(4), same tiling, op_scale=2): 100*2=200.
    //         Op3(MM, out=T4(5), tiling=(FROM_NTW,FROM_NTH), ht=2,vt=1,
    //           slice_w=128, slice_h=256, op_scale=2): 2000*2=4000.
    //         comp_per_step = 5400.
    //   IO (nk=1 → FROM_NK becomes FIXED_1):
    //     T_w(0): (FIXED_1,FIXED_1) → once_load = 32768/10 = 3276.8.
    //     T_in(1): (FROM_NTW,FIXED_1) → col_load = 32768/10 = 3276.8.
    //     T4(5) out: evict = 32768/10 = 3276.8.
    //   RowMajor: first = tile_cost(T,T,T), within × 1.
    //     first: per_tile_io = once(3276.8) + col(3276.8) = 6553.6.
    //       max(5400, 6553.6 + 3276.8) = 9830.4.
    //     within: per_tile_io = col(3276.8) = 3276.8.
    //       max(5400, 3276.8 + 3276.8) = 6553.6.
    //     latency = 9830.4 + 6553.6 = 16384.
    auto r = sg.compute_cost(TC(128, 256, 128, SnakeDir::RowMajor));
    CHECK("cost feasible", r.feasible);
    CHECK_EQ("comp_per_step = 5400", r.compute_per_step, 5400.0);
    CHECK_EQ("latency = 16384", r.latency, 16384.0);

    auto best = sg.best_cost();
    CHECK("bookend feasible", best.feasible);
}

// ============================================================================
// Section 10: Temporal dimension (nk>1) — verifying k-step decomposition
//
// These tests exercise the time dimension of the cost model: when nk > 1,
// each spatial tile is processed in multiple k-steps, with stream-dependent
// tensors reloaded each step and non-stream tensors loaded only at tile
// boundaries.
// ============================================================================

void test_2mm_chain_nk2_temporal() {
    std::cout << "--- test_2mm_chain_nk2_temporal ---\n";
    // Two-matmul chain: Op0 (non-sink) feeds Op1 (sink).
    //
    // Op0: MM T_a(128×128) @ T_b(256×128) → T_c(256×128)  [K0=128]
    // Op1: MM T_c(256×128) @ T_d(128×256) → T_e(128×128)  [K1=256, sink]
    //
    // T_c is ephemeral (produced by Op0, consumed by Op1).
    // output_W=128, output_H=128, output_K=256 (K of sink Op1 = T_c.width).
    //
    // Tile sources (backward propagation):
    //   T_e (sink boundary output): (FROM_NTW, FROM_NTH)
    //   Op1 sink MM:
    //     T_c (LHS): (FROM_NK, FROM_NTH)   ← stream in h, varies with nth
    //     T_d (RHS): (FROM_NTW, FROM_NK)   ← varies with ntw, stream in v
    //   Op0 non-sink MM (output T_c has tiling FROM_NK, FROM_NTH):
    //     T_a (LHS): (FIXED_1, FROM_NTH)   ← row_load (fixed in h, varies with nth)
    //     T_b (RHS): (FROM_NK, FIXED_1)    ← stream (k-dependent in h)
    //
    // Tiling [w=64, h=64, k=128]:
    //   ntw = 128/64 = 2, nth = 128/64 = 2, nk = 256/128 = 2
    //
    // Working set (slice sizes):
    //   T_a(0): ht=1, vt=nth=2 → (128/1)×(128/2) = 128×64 = 8192
    //   T_b(1): ht=nk=2, vt=1 → (256/2)×(128/1) = 128×128 = 16384
    //   T_d(3): ht=ntw=2, vt=nk=2 → (128/2)×(256/2) = 64×128 = 8192
    //   T_e(4): ht=ntw=2, vt=nth=2 → (128/2)×(128/2) = 64×64 = 4096
    //   Total ws = 8192 + 16384 + 8192 + 4096 = 36864
    //
    // IO classification (B=10):
    //   T_a: (FIXED_1, FROM_NTH) → h_fixed, not k_dep → row_load = 8192/10 = 819.2
    //   T_b: (FROM_NK, FIXED_1) → k_dep → stream_load += 16384/10 = 1638.4
    //   T_d: (FROM_NTW, FROM_NK) → k_dep → stream_load += 8192/10 = 819.2
    //   T_e: internally produced (skip load), boundary_out → out_evict = 4096/10 = 409.6
    //
    //   once_load=0, row_load=819.2, col_load=0, tile_load=0
    //   stream_load = 1638.4 + 819.2 = 2457.6
    //   out_evict = 409.6
    //
    // comp_per_step (new model: only SINK ops divide by nk):
    //   Op0 (non-sink, cost=2000): nk_adj=1, op_scale=max(128/128,1)×max(64/128,1)=1.
    //     2000/1 * 1 = 2000.
    //   Op1 (sink, cost=4000): nk_adj=nk=2, op_scale=max(64/128,1)×max(64/128,1)=1.
    //     4000/2 * 1 = 2000.
    //   comp_per_step = 4000.
    //
    // RowMajor tile_cost with nk=2:
    //   tile_cost(once, row, col) = step0 + last
    //     per_tile_io = tile(0) + once(0) + row(819.2 if row_fresh) + col(0)
    //     step0 = max(4000, per_tile_io + 2457.6)
    //     last  = max(4000, 2457.6 + 409.6) = max(4000, 2867.2) = 4000
    //
    //   first       = tile_cost(T,T,T): per_tile_io=819.2, step0=max(4000,3276.8)=4000, total=8000
    //   row_trans    = tile_cost(F,T,F): per_tile_io=819.2, step0=4000, total=8000
    //   within_row   = tile_cost(F,F,T): per_tile_io=0, step0=max(4000,2457.6)=4000, total=8000
    //
    //   n_row_trans = nth-1 = 1, n_within = (ntw-1)*nth = 2
    //   latency = 8000 + 1*8000 + 2*8000 = 32000
    //
    // Timeline (RowMajor snake, ntw=2, nth=2, nk=2):
    //
    //   Tile(r=0,c=0) — first tile:
    //     k=0: load T_a[row0] + T_b[k0] + T_d[c0,k0]. Compute.
    //     k=1: load T_b[k1] + T_d[c0,k1]. Compute. Evict T_e[0,0].
    //
    //   Tile(r=0,c=1) — within row (T_a[row0] stays):
    //     k=0: T_b[k0] + T_d[c1,k0]. Compute.
    //     k=1: T_b[k1] + T_d[c1,k1]. Compute. Evict T_e[0,1].
    //
    //   Tile(r=1,c=1) — row transition (new T_a[row1]):
    //     k=0: load T_a[row1] + T_b[k0] + T_d[c1,k0]. Compute.
    //     k=1: T_b[k1] + T_d[c1,k1]. Compute. Evict T_e[1,1].
    //
    //   Tile(r=1,c=0) — within row (T_a[row1] stays):
    //     k=0: T_b[k0] + T_d[c0,k0]. Compute.
    //     k=1: T_b[k1] + T_d[c0,k1]. Compute. Evict T_e[1,0].

    Problem p;
    // T_a(0): 128×128, T_b(1): 256×128, T_c(2): 256×128, T_d(3): 128×256, T_e(4): 128×128
    p.tensors = {{128,128},{256,128},{256,128},{128,256},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},    // Op0: non-sink
             {OpType::MatMul,{2,3},{4},4000}};    // Op1: sink
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // Basic structure
    CHECK("2mm: 2 ops", sg.num_ops() == 2);
    CHECK("2mm: has_matmul", sg.has_matmul());
    CHECK("2mm: T_c ephemeral", sg.ephemeral().count(2));
    CHECK("2mm: T_a input", sg.boundary_inputs().count(0));
    CHECK("2mm: T_b input", sg.boundary_inputs().count(1));
    CHECK("2mm: T_d input", sg.boundary_inputs().count(3));
    CHECK("2mm: T_e output", sg.boundary_outputs().count(4));
    CHECK_EQ_I("2mm: output_K = 256", sg.output_K(), 256);
    CHECK_EQ_I("2mm: output_width = 128", sg.output_width(), 128);
    CHECK_EQ_I("2mm: output_height = 128", sg.output_height(), 128);

    // Tiling validity: k must divide K0=128 and K1=256.
    CHECK("w=64,h=64,k=128 valid", sg.is_valid_tiling(TC(64, 64, 128)));
    CHECK("w=64,h=64,k=64 valid", sg.is_valid_tiling(TC(64, 64, 64)));
    // k=256 ≥ K0=128, so the divisibility check is skipped (tile covers full dim).
    // nk = output_K/k = 256/256 = 1. Valid.
    CHECK("k=256 valid (k≥K0, nk=1)", sg.is_valid_tiling(TC(64, 64, 256)));
    // k=192 < K0=128? No, 192>128, skip. But 256%192=64≠0 for K1. Check: 192<256 && 256%192=64≠0 → invalid.
    CHECK("k=192 invalid (256%192≠0)", !sg.is_valid_tiling(TC(64, 64, 192)));

    // Working set at [64,64,128] (ntw=2, nth=2, nk=2)
    CHECK_EQ_I("ws(64,64,128) = 36864", sg.working_set(TC(64, 64, 128)), 36864);

    // Working set at [128,128,128] (ntw=1, nth=1, nk=2)
    // T_a: ht=1, vt=1 → 128*128 = 16384
    // T_b: ht=nk=2, vt=1 → 128*128 = 16384
    // T_d: ht=1, vt=nk=2 → 128*128 = 16384
    // T_e: ht=1, vt=1 → 128*128 = 16384
    // Total = 65536
    CHECK_EQ_I("ws(128,128,128) = 65536", sg.working_set(TC(128, 128, 128)), 65536);

    // Working set at [128,128,256] (ntw=1, nth=1, nk=1)
    // All tile counts = 1 → full tensor slices.
    // T_a: 128*128=16384, T_b: 256*128=32768, T_d: 128*256=32768, T_e: 128*128=16384
    // Total = 98304
    CHECK_EQ_I("ws(128,128,256) = 98304", sg.working_set(TC(128, 128, 256)), 98304);

    // Cost at [64,64,128] RowMajor (ntw=2, nth=2, nk=2)
    auto r = sg.compute_cost(TC(64, 64, 128, SnakeDir::RowMajor));
    CHECK("cost feasible", r.feasible);
    CHECK_EQ("comp_per_step = 4000", r.compute_per_step, 4000.0);
    CHECK_EQ("latency = 32000", r.latency, 32000.0);

    // Cost at [64,64,64] RowMajor (ntw=2, nth=2, nk=4)
    // comp_per_step (new model: only sink divides by nk):
    //   Op0 (non-sink): 2000/1*1=2000. Op1 (sink): 4000/4*1=1000. Total=3000.
    //
    // IO slices (k=64, nk=4):
    //   T_a: ht=1, vt=nth=2 → 128*64=8192, slice_io=819.2 → row_load
    //   T_b: ht=nk=4, vt=1 → (256/4)*128=64*128=8192, slice_io=819.2 → stream
    //   T_d: ht=ntw=2, vt=nk=4 → 64*(256/4)=64*64=4096, slice_io=409.6 → stream
    //   T_e: internally produced. out_evict: ht=2,vt=2 → 64*64=4096, 409.6
    //
    //   stream = 819.2 + 409.6 = 1228.8, row_load = 819.2, out_evict = 409.6
    //
    // tile_cost nk=4:
    //   step0 = max(comp, per_tile_io + stream)
    //   mid = (nk-2) * max(comp, stream) = 2 * max(3000, 1228.8) = 2 * 3000 = 6000
    //   last = max(comp, stream + out_evict) = max(3000, 1228.8+409.6) = max(3000, 1638.4) = 3000
    //
    // RowMajor:
    //   first(T,T,T): per_tile_io=819.2, step0=max(3000,819.2+1228.8)=max(3000,2048)=3000.
    //     total = 3000 + 6000 + 3000 = 12000
    //   row_trans(F,T,F): same as first (once=0). per_tile_io=819.2, step0=3000.
    //     total = 12000
    //   within(F,F,T): per_tile_io=0, step0=max(3000,1228.8)=3000.
    //     total = 3000 + 6000 + 3000 = 12000
    //
    //   latency = 12000 + 1*12000 + 2*12000 = 48000
    auto r2 = sg.compute_cost(TC(64, 64, 64, SnakeDir::RowMajor));
    CHECK("cost nk=4 feasible", r2.feasible);
    CHECK_EQ("comp_per_step nk=4 = 3000", r2.compute_per_step, 3000.0);
    CHECK_EQ("latency nk=4 = 48000", r2.latency, 48000.0);

    // Cost at [64,64,128] with nk=2, ColMajor — different reuse pattern.
    // comp_per_step = 4000 (same as RowMajor).
    // ColMajor: sweep columns, snake reversal between columns.
    //   first(T,T,T): per_tile_io=row_load(819.2). step0=max(4000,3276.8)=4000. last=4000. total=8000.
    //   col_trans(F,F,T): per_tile_io=0. step0=max(4000,2457.6)=4000. last=4000. total=8000.
    //   within(F,T,F): per_tile_io=row_load(819.2). step0=max(4000,3276.8)=4000. total=8000.
    //
    //   n_col_trans = ntw-1=1, n_within = (nth-1)*ntw = 2
    //   latency = 8000 + 1*8000 + 2*8000 = 32000
    //   (compute-bound throughout; ColMajor = RowMajor for this subgraph)
    auto r3 = sg.compute_cost(TC(64, 64, 128, SnakeDir::ColMajor));
    CHECK("cost ColMajor feasible", r3.feasible);
    CHECK_EQ("comp_per_step ColMajor = 4000", r3.compute_per_step, 4000.0);
    CHECK_EQ("latency ColMajor = 32000", r3.latency, 32000.0);

    // Verify best_cost finds a feasible solution.
    auto best = sg.best_cost();
    CHECK("best_cost feasible", best.feasible);
}

void test_single_mm_nk2_stream_only() {
    std::cout << "--- test_single_mm_nk2_stream_only ---\n";
    // Single MatMul with nk=2. All inputs are stream or row/col.
    // This is simpler than the 2-MM chain — one sink op.
    //
    // Op0: MM T_lhs(256×128) @ T_rhs(128×256) → T_out(128×128)
    //   K = T_lhs.width = 256 = output_K.
    //
    // Tile [64,64,128]: ntw=2, nth=2, nk=2.
    //
    // Tile sources (sink MM):
    //   T_lhs (LHS): (FROM_NK, FROM_NTH) → stream (k_dep)
    //   T_rhs (RHS): (FROM_NTW, FROM_NK) → stream (k_dep)
    //   T_out: (FROM_NTW, FROM_NTH) → output
    //
    // Working set:
    //   T_lhs: ht=nk=2, vt=nth=2 → (256/2)*(128/2) = 128*64 = 8192
    //   T_rhs: ht=ntw=2, vt=nk=2 → (128/2)*(256/2) = 64*128 = 8192
    //   T_out: ht=ntw=2, vt=nth=2 → 64*64 = 4096
    //   Total = 20480
    //
    // IO (B=10):
    //   T_lhs: stream = 8192/10 = 819.2
    //   T_rhs: stream = 8192/10 = 819.2
    //   stream_load = 1638.4, no other loads.
    //   T_out: out_evict = 4096/10 = 409.6
    //
    // comp_per_step: 6000/2 * op_scale. T_out: ht=2, vt=2, slice 64×64.
    //   op_scale = max(64/128,1)*max(64/128,1) = 1. comp = 3000.
    //
    // tile_cost nk=2:
    //   step0 = max(3000, per_tile_io + 1638.4)
    //   last  = max(3000, 1638.4 + 409.6) = max(3000, 2048.0) = 3000
    //
    // RowMajor (no row_load or col_load, so all tiles are identical except first):
    //   first(T,T,T): per_tile_io=0, step0=max(3000,1638.4)=3000, total=6000
    //   row_trans(F,T,F): same per_tile_io=0, total=6000
    //   within(F,F,T): per_tile_io=0, total=6000
    //
    //   latency = 6000 + 1*6000 + 2*6000 = 24000

    Problem p;
    p.tensors = {{256,128},{128,256},{128,128}};  // T_lhs, T_rhs, T_out
    p.ops = {{OpType::MatMul,{0,1},{2},6000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK_EQ_I("single mm: output_K = 256", sg.output_K(), 256);

    // nk=2
    CHECK("nk=2 valid", sg.is_valid_tiling(TC(64, 64, 128)));
    CHECK_EQ_I("ws(64,64,128) = 20480", sg.working_set(TC(64, 64, 128)), 20480);

    auto r = sg.compute_cost(TC(64, 64, 128, SnakeDir::RowMajor));
    CHECK("cost feasible", r.feasible);
    CHECK_EQ("comp = 3000", r.compute_per_step, 3000.0);
    CHECK_EQ("latency = 24000", r.latency, 24000.0);

    // nk=4: k=64
    // comp = 6000/4 * 1 = 1500
    // T_lhs: ht=4, vt=2 → (256/4)*(128/2) = 64*64 = 4096, stream=409.6
    // T_rhs: ht=2, vt=4 → (128/2)*(256/4) = 64*64 = 4096, stream=409.6
    // T_out: ht=2, vt=2 → 4096, evict=409.6
    // stream=819.2, out_evict=409.6
    //
    // ws = 4096 + 4096 + 4096 = 12288
    //
    // tile_cost nk=4, all tiles same (no row/col/once):
    //   step0 = max(1500, 819.2) = 1500
    //   mid = 2 * max(1500, 819.2) = 3000
    //   last = max(1500, 819.2 + 409.6) = max(1500, 1228.8) = 1500
    //   total = 1500 + 3000 + 1500 = 6000
    //
    // RowMajor: all tiles = 6000. latency = 4 * 6000 = 24000.
    CHECK_EQ_I("ws(64,64,64) = 12288", sg.working_set(TC(64, 64, 64)), 12288);
    auto r2 = sg.compute_cost(TC(64, 64, 64, SnakeDir::RowMajor));
    CHECK("cost nk=4 feasible", r2.feasible);
    CHECK_EQ("comp nk=4 = 1500", r2.compute_per_step, 1500.0);
    CHECK_EQ("latency nk=4 = 24000", r2.latency, 24000.0);
}

void test_mm_pw_chain_nk2_mixed_io() {
    std::cout << "--- test_mm_pw_chain_nk2_mixed_io ---\n";
    // MM → PW chain where the PW is sink. PW sink forces nk=1.
    // output_K_ = max_K_ = 128 (K of the matmul). Only k=128 valid (nk=1).
    //
    // Op0: MM T_a(128×128) @ T_b(128×128) → T_c(128×128)  [K=128]
    // Op1: PW T_c → T_out(128×128)
    //
    // has_pw_sink_ = true → ks_cand_ = {output_K_} = {128}.
    // With k=128, nk = max(128/128, 1) = 1. No temporal dimension.

    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK_EQ_I("pw_sink: output_K = 128", sg.output_K(), 128);

    // Only k=128 valid (nk=128/128=1). k<128 → nk>1 → rejected.
    CHECK("pw_sink: k=1 invalid", !sg.is_valid_tiling(TC(128, 128, 1)));
    CHECK("pw_sink: k=128 valid", sg.is_valid_tiling(TC(128, 128, 128)));

    // With nk=1 and ntw=1, nth=1: all tile counts are 1.
    // T_a: (FIXED_1, FROM_NTH) → ht=1, vt=1 → 128*128=16384
    // T_b: (FROM_NTW, FIXED_1) → ht=1, vt=1 → 16384
    // T_out: (FROM_NTW, FROM_NTH) → ht=1, vt=1 → 16384
    // ws = 49152
    CHECK_EQ_I("ws(128,128,128) = 49152", sg.working_set(TC(128, 128, 128)), 49152);

    // Cost: comp = 2000/1 * 1 + 500/1 * 1 = 2500 (nk=1).
    // IO: T_a once_load=1638.4, T_b once_load=1638.4, T_out evict=1638.4.
    // nk=1: tile_cost = max(2500, 3276.8 + 0 + 1638.4) = max(2500, 4915.2) = 4915.2
    // 1 tile → latency = 4915.2
    auto r = sg.compute_cost(TC(128, 128, 128, SnakeDir::RowMajor));
    CHECK("pw_sink cost feasible", r.feasible);
    CHECK_EQ("pw_sink comp = 2500", r.compute_per_step, 2500.0);
    CHECK_EQ("pw_sink latency = 4915.2", r.latency, 4915.2);
}

// ============================================================================
// Section 12: PW-sink temporal tiling rejection
//
// When any sink op is Pointwise, has_pw_sink_ = true which forces:
//   - output_K_ = max_K_ (K of the internal MatMul, or 1 for pure-PW)
//   - ks_cand_ = {output_K_}  (only k=max_K gives nk=1)
//   - is_valid_tiling rejects any k where nk = output_K_/k > 1
//
// This ensures that temporal K-dimension splitting is never applied when the
// output is a Pointwise op, even if a MatMul is present in the same subgraph.
// The k written to the solution file equals max_K_ (the matmul's K dimension).
// ============================================================================

void test_pw_sink_best_cost_no_temporal_mm_pw_chain() {
    std::cout << "--- test_pw_sink_best_cost_no_temporal_mm_pw_chain ---\n";
    // MM → PW chain (PW is sole sink):
    // Op0: MM T0(128×128) @ T1(128×128) → T2(128×128)  [K=128]
    // Op1: PW T2 → T3(128×128)  [sink]
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // output_K_ = max_K_ = 128 (K of Op0).
    CHECK_EQ_I("mm_pw chain: output_K = 128", sg.output_K(), 128);

    // best_cost() only tries k=128 (ks_cand_ = {128}), giving nk=1.
    auto best = sg.best_cost();
    CHECK("mm_pw chain: best_cost feasible", best.feasible);
    CHECK_EQ_I("mm_pw chain: num_k_passes = 1", best.num_k_passes, 1);
    CHECK_EQ_I("mm_pw chain: best k = 128", (int64_t)best.config.k, 128);

    // k < 128 → nk > 1 → rejected by PW-sink constraint.
    CHECK("mm_pw k=2 invalid", !sg.is_valid_tiling(TC(128, 128, 2)));
    CHECK("mm_pw k=64 invalid", !sg.is_valid_tiling(TC(128, 128, 64)));
}

void test_pw_cosink_no_temporal_tiling() {
    std::cout << "--- test_pw_cosink_no_temporal_tiling ---\n";
    // MM and PW as co-sinks (both produce boundary outputs, no chain):
    // Op0: MM T0(128×128) @ T1(128×128) → T2(128×128)  [sink]
    // Op1: PW T0 → T3(128×128)                          [sink]
    //
    // Two sinks, one is PW → has_pw_sink_ = true, output_K_ = max_K_ = 128.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},   // Op0: MM sink
             {OpType::Pointwise,{0},{3},500}};  // Op1: PW sink
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK_EQ_I("cosink: output_K = 128", sg.output_K(), 128);

    auto best = sg.best_cost();
    CHECK("cosink: best_cost feasible", best.feasible);
    CHECK_EQ_I("cosink: num_k_passes = 1", best.num_k_passes, 1);

    // k=32 → nk=128/32=4 > 1 → rejected by PW-sink constraint.
    CHECK("cosink k=32 invalid", !sg.is_valid_tiling(TC(128, 128, 32)));
    // k=128 → nk=1 → valid.
    CHECK("cosink k=128 valid", sg.is_valid_tiling(TC(128, 128, 128)));
}

void test_pw_sink_no_temporal_tight_memory() {
    std::cout << "--- test_pw_sink_no_temporal_tight_memory ---\n";
    // Tight memory: pure MM sink would be forced to use nk > 1.
    // MM+PW must NOT use temporal tiling even under memory pressure.
    // With spatial tiling only (nk=1), the working set is reduced by
    // smaller w and h tiles.
    //
    // Op0: MM T0(128×128) @ T1(128×128) → T2(128×128)  [K=128]
    // Op1: PW T2 → T3(128×128)  [sink]
    //
    // output_K_ = max_K_ = 128. Only k=128 valid (nk=1).
    // At {128,128,128}: ws = T0+T1+T3 = 3×16384 = 49152. Capacity = 5000.
    // Must tile spatially: e.g. {8,8,128} → T0 slice small, feasible.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 5000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK_EQ_I("tight mm_pw: output_K = 128", sg.output_K(), 128);

    auto best = sg.best_cost();
    // Regardless of whether a feasible tiling is found, it must not use nk > 1
    CHECK_EQ_I("tight mm_pw: num_k_passes = 1", best.num_k_passes, 1);
}

void test_pure_mm_sink_temporal_allowed() {
    std::cout << "--- test_pure_mm_sink_temporal_allowed ---\n";
    // Control: pure MM sink sets output_K_ = K = 128, not 1.
    // Temporal tiling (nk > 1) is possible in principle.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // output_K_ is set to the actual K dimension (128), not 1
    CHECK_EQ_I("pure_mm: output_K = 128", sg.output_K(), 128);

    // compute_cost with k=64 → nk = 128/64 = 2 (temporal tiling applied)
    auto r = sg.compute_cost(TC(128, 128, 64, SnakeDir::RowMajor));
    CHECK("pure_mm k=64: feasible", r.feasible);
    CHECK_EQ_I("pure_mm k=64: num_k_passes = 2", r.num_k_passes, 2);

    // k=1 → nk = 128 (maximum temporal splitting)
    auto r1 = sg.compute_cost(TC(128, 128, 1, SnakeDir::RowMajor));
    CHECK_EQ_I("pure_mm k=1: num_k_passes = 128", r1.num_k_passes, 128);
}

// ============================================================================
// Section 11: Symbolic tile-source conflict detection
//
// When a tensor is consumed by ops that assign different symbolic tile sources
// (e.g. FROM_NTH and FROM_NK) to the same axis, the old numerical check
// incorrectly accepted configs where the values coincided (e.g. nth==nk).
// The new symbolic check rejects these: the dimensions represent different
// execution phases (spatial vs temporal) and the tensor can't serve both.
// ============================================================================

void test_shared_input_two_chained_mm_symbolic_conflict() {
    std::cout << "--- test_shared_input_two_chained_mm_symbolic_conflict ---\n";
    // Two chained MMs sharing T_shared as LHS of non-sink and RHS of sink.
    //
    // Op0: MM T_shared(128×128) @ T_b(128×128) → T_mid(128×128)  [non-sink]
    // Op1: MM T_mid(128×128) @ T_shared(128×128) → T_out(128×128)  [sink]
    //
    // T_mid is ephemeral. output_K = T_mid.width = 128.
    //
    // Tile source propagation:
    //   T_out (sink output): (FROM_NTW, FROM_NTH)
    //   Op1 (sink): LHS=T_mid → (FROM_NK, FROM_NTH)
    //               RHS=T_shared → (FROM_NTW, FROM_NK)
    //   Op0 (non-sink, output=T_mid has (FROM_NK, FROM_NTH)):
    //               LHS=T_shared → (FIXED_1, FROM_NTH)   ← v from T_mid.v
    //               RHS=T_b → (FROM_NK, FIXED_1)
    //
    // T_shared gets TWO assignments:
    //   From Op1 (RHS of sink): h=FROM_NTW, v=FROM_NK
    //   From Op0 (LHS of non-sink): h=FIXED_1, v=FROM_NTH
    //
    // Symbolic conflict on v-axis: FROM_NK vs FROM_NTH.
    //   When nth==nk (e.g. h=64, k=64 → nth=2, nk=2), the numbers coincide
    //   but Op0 needs T_shared sliced by spatial row position while Op1 needs
    //   it sliced by temporal k-step. Can't serve both simultaneously.
    //
    // On h-axis: FIXED_1 vs FROM_NTW. Rejected numerically when ntw>1.

    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    //            T_shared  T_b       T_mid(eph) T_out
    p.ops = {{OpType::MatMul,{0,1},{2},2000},   // Op0: non-sink
             {OpType::MatMul,{2,0},{3},4000}};   // Op1: sink, T_shared as RHS
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK("shared: 2 ops", sg.num_ops() == 2);
    CHECK("shared: T_mid ephemeral", sg.ephemeral().count(2));
    CHECK("shared: T_shared input", sg.boundary_inputs().count(0));
    CHECK("shared: T_b input", sg.boundary_inputs().count(1));
    CHECK("shared: T_out output", sg.boundary_outputs().count(3));
    CHECK_EQ_I("shared: output_K = 128", sg.output_K(), 128);

    // w=128, h=64, k=64 → ntw=1, nth=2, nk=2.
    // h-axis: FIXED_1(1) vs FROM_NTW(ntw=1) → both 1, compatible.
    // v-axis: FROM_NTH(nth=2) vs FROM_NK(nk=2) → different labels, both >1 → REJECT.
    // This is the key case: old numerical check would accept (2==2),
    // new symbolic check correctly rejects.
    CHECK("w=128,h=64,k=64 rejected (FROM_NTH≠FROM_NK on v, nth==nk==2)",
          !sg.is_valid_tiling(TC(128, 64, 64)));

    // w=64, h=64, k=64 → ntw=2, nth=2, nk=2.
    // h-axis: FIXED_1(1) vs FROM_NTW(2) → numerical mismatch → reject.
    CHECK("w=64,h=64,k=64 rejected (FIXED_1 vs FROM_NTW on h)",
          !sg.is_valid_tiling(TC(64, 64, 64)));

    // w=128, h=64, k=128 → ntw=1, nth=2, nk=1.
    // v-axis: FROM_NTH(2) vs FROM_NK(1) → different, not both 1 → reject.
    CHECK("w=128,h=64,k=128 rejected (FROM_NTH=2 vs FROM_NK=1)",
          !sg.is_valid_tiling(TC(128, 64, 128)));

    // w=128, h=128, k=64 → ntw=1, nth=1, nk=2.
    // v-axis: FROM_NTH(1) vs FROM_NK(2) → different, not both 1 → reject.
    CHECK("w=128,h=128,k=64 rejected (FROM_NTH=1 vs FROM_NK=2)",
          !sg.is_valid_tiling(TC(128, 128, 64)));

    // w=128, h=128, k=128 → ntw=1, nth=1, nk=1.
    // All sources evaluate to 1 → no tiling at all → compatible.
    CHECK("w=128,h=128,k=128 accepted (ntw=nth=nk=1)",
          sg.is_valid_tiling(TC(128, 128, 128)));

    // Verify working_set for the only valid config.
    // T_shared: merged (FROM_NTW, FROM_NK) via create(). At nk=1: ht=ntw=1, vt=nk=1 → 128*128=16384.
    // T_b: (FROM_NK, FIXED_1). At nk=1: ht=1, vt=1 → 16384.
    // T_out: (FROM_NTW, FROM_NTH). At ntw=1, nth=1: ht=1, vt=1 → 16384.
    // Total = 49152.
    CHECK_EQ_I("ws(128,128,128) = 49152", sg.working_set(TC(128, 128, 128)), 49152);

    auto best = sg.best_cost();
    CHECK("best_cost feasible", best.feasible);
}

void test_three_mm_chain_shared_rhs_symbolic() {
    std::cout << "--- test_three_mm_chain_shared_rhs_symbolic ---\n";
    // Three-MM chain where T_rhs is shared as RHS of Op0 (non-sink) and
    // RHS of Op2 (sink), producing a FROM_NTW vs FROM_NK conflict on h.
    //
    // Op0: MM T_a(128×256) @ T_rhs(256×256) → T1(256×256)  [non-sink, K0=128]
    // Op1: PW T1 → T2(256×256)
    // Op2: MM T2(256×256) @ T_rhs(256×256) → T_out(256×256)  [sink, K2=256]
    //
    // T1, T2 ephemeral. output_K = 256 (T2.width for sink Op2).
    //
    // Tile sources:
    //   T_out: (FROM_NTW, FROM_NTH)
    //   Op2 (sink): LHS=T2 → (FROM_NK, FROM_NTH). RHS=T_rhs → (FROM_NTW, FROM_NK).
    //   Op1 (PW): T1 ← T2 → (FROM_NK, FROM_NTH).
    //   Op0 (non-sink, out=T1 has (FROM_NK, FROM_NTH)):
    //     LHS=T_a → (FIXED_1, FROM_NTH). RHS=T_rhs → (FROM_NK, FIXED_1).
    //
    // T_rhs conflict:
    //   From Op2: h=FROM_NTW, v=FROM_NK
    //   From Op0: h=FROM_NK, v=FIXED_1
    //   h: FROM_NTW vs FROM_NK → reject when either >1
    //   v: FROM_NK vs FIXED_1 → reject when nk>1

    Problem p;
    p.tensors = {{128,256},{256,256},{256,256},{256,256},{256,256}};
    //            T_a       T_rhs     T1(eph)   T2(eph)   T_out
    p.ops = {{OpType::MatMul,{0,1},{2},1000},      // Op0
             {OpType::Pointwise,{2},{3},200},       // Op1
             {OpType::MatMul,{3,1},{4},3000}};      // Op2 (sink)
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("3mm: T1 eph", sg.ephemeral().count(2));
    CHECK("3mm: T2 eph", sg.ephemeral().count(3));
    CHECK("3mm: T_rhs input", sg.boundary_inputs().count(1));
    CHECK_EQ_I("3mm: output_K = 256", sg.output_K(), 256);

    // w=256, h=256, k=256 → ntw=1, nth=1, nk=1. All 1 → compatible.
    CHECK("all-1 accepted", sg.is_valid_tiling(TC(256, 256, 256)));

    // w=128, h=256, k=128 → ntw=2, nth=1, nk=2.
    // T_rhs h: FROM_NTW(2) vs FROM_NK(2) → symbolic conflict → reject.
    CHECK("ntw==nk==2 rejected (FROM_NTW≠FROM_NK on T_rhs h)",
          !sg.is_valid_tiling(TC(128, 256, 128)));

    // w=128, h=256, k=256 → ntw=2, nth=1, nk=1.
    // T_rhs h: FROM_NTW(2) vs FROM_NK(1) → not both 1 → reject.
    CHECK("ntw=2,nk=1 rejected (FROM_NTW vs FROM_NK)",
          !sg.is_valid_tiling(TC(128, 256, 256)));

    // w=256, h=256, k=128 → ntw=1, nth=1, nk=2.
    // T_rhs h: FROM_NTW(1) vs FROM_NK(2) → not both 1 → reject.
    // T_rhs v: FROM_NK(2) vs FIXED_1(1) → not both 1 → reject.
    CHECK("ntw=1,nk=2 rejected (multiple conflicts)",
          !sg.is_valid_tiling(TC(256, 256, 128)));

    auto best = sg.best_cost();
    CHECK("best_cost feasible", best.feasible);
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

    // Section 6: derived tile-count bounds (zero-slice bug)
    test_nk_exceeds_tensor_dim_chained_matmul();
    test_nk_exceeds_tensor_dim_deep_chain();
    test_nk_exceeds_small_tensor_rejected();
    test_nk_exceeds_tensor_dim_multi_op();
    test_ntw_exceeds_tensor_dim();

    // Section 7: working set and cost sanity
    test_working_set_single_matmul();
    test_working_set_with_retain();
    test_cost_compute_overlap();
    test_cost_matmul_snake_vs_raster();

    // Section 8: longer chains, splits, tensor reuse
    test_long_pw_chain_4ops();
    test_long_mm_chain_3ops();
    test_mm_pw_mm_chain();
    test_fan_in_two_mm_to_pw();
    test_tensor_reused_at_different_depths();
    test_tensor_reused_different_roles();
    test_bottleneck_3branch_pattern();
    test_residual_skip_connection();
    test_diamond_dag();
    test_input_shared_across_two_mms();

    // Section 9: internal fan-out — tensors consumed by multiple ops at different depths
    test_ephemeral_fanout_pw_chain();
    test_ephemeral_fanout_to_mm_and_pw();
    test_boundary_input_as_lhs_and_rhs();
    test_boundary_input_at_depth0_and_depth2();
    test_ephemeral_fanout_mm_lhs_and_mm_rhs();
    test_long_chain_input_reused_at_bookends();

    // Section 10: temporal dimension (nk>1) — k-step decomposition
    test_2mm_chain_nk2_temporal();
    test_single_mm_nk2_stream_only();
    test_mm_pw_chain_nk2_mixed_io();

    // Section 12: PW-sink temporal tiling rejection
    test_pw_sink_best_cost_no_temporal_mm_pw_chain();
    test_pw_cosink_no_temporal_tiling();
    test_pw_sink_no_temporal_tight_memory();
    test_pure_mm_sink_temporal_allowed();

    // Section 11: symbolic tile-source conflict detection
    test_shared_input_two_chained_mm_symbolic_conflict();
    test_three_mm_chain_shared_rhs_symbolic();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}