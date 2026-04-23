// granule_bound_test.cpp
//
// Coverage for:
//   * Super-native granule rejection (#74 Q1, #78 Q3, #80 Q3, #81 Q1)
//     cfg.{w,h,k} must be ≤ native, where native_w == native_h == native_k
//     per #80 Q1.
//   * Granule-fit check on ephemerals:
//       PW-produced ephemeral: slice ≤ (cfg.w, cfg.h)
//       MM-produced ephemeral: slice ≤ native

#include "core/dag.h"
#include "core/subgraph.h"
#include "core/types.h"
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}

static TileConfig TC(int64_t w, int64_t h, int64_t k,
                     SnakeDir s = SnakeDir::None) { return {w, h, k, s}; }

static Subgraph make_sg(const Problem& p, const DAG& d,
                        std::vector<size_t> ops) {
    auto sg = Subgraph::create(p, d, std::move(ops));
    if (!sg) { std::cerr << "FATAL: Subgraph::create failed\n"; std::exit(1); }
    return std::move(*sg);
}

// ============================================================================
// Super-native bound: cfg.{w,h,k} must be ≤ native.
// Tensors 256×256, native 128 — so cfg=256 on any axis is super-native.
// ============================================================================

static Problem mm_256_native_128() {
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

void test_super_native_w_rejected() {
    std::cout << "--- test_super_native_w_rejected ---\n";
    auto p = mm_256_native_128();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK("w=128 (native) OK",        sg.is_valid_tiling(TC(128,128,128)));
    CHECK("w=64  (sub-native) OK",    sg.is_valid_tiling(TC(64, 128,128)));
    CHECK("w=256 (super-native) REJECTED", !sg.is_valid_tiling(TC(256,128,128)));
}

void test_super_native_h_rejected() {
    std::cout << "--- test_super_native_h_rejected ---\n";
    auto p = mm_256_native_128();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK("h=128 (native) OK",        sg.is_valid_tiling(TC(128,128,128)));
    CHECK("h=64  (sub-native) OK",    sg.is_valid_tiling(TC(128,64, 128)));
    CHECK("h=256 (super-native) REJECTED", !sg.is_valid_tiling(TC(128,256,128)));
}

void test_super_native_k_rejected() {
    std::cout << "--- test_super_native_k_rejected ---\n";
    auto p = mm_256_native_128();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK("k=128 (native) OK",        sg.is_valid_tiling(TC(128,128,128)));
    CHECK("k=64  (sub-native) OK",    sg.is_valid_tiling(TC(128,128,64)));
    CHECK("k=256 (super-native) REJECTED", !sg.is_valid_tiling(TC(128,128,256)));
}

void test_best_cost_stays_within_native() {
    std::cout << "--- test_best_cost_stays_within_native ---\n";
    // best_cost iterates candidate lists; super-native should never show up.
    auto p = mm_256_native_128();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto best = sg.best_cost();
    CHECK("best.w ≤ native", best.config.w <= 128);
    CHECK("best.h ≤ native", best.config.h <= 128);
    CHECK("best.k ≤ native", best.config.k <= 128);
}

// Sanity: small native (64) shouldn't falsely cap away valid small tiles.
void test_small_native_all_subdivisors_ok() {
    std::cout << "--- test_small_native_all_subdivisors_ok ---\n";
    Problem p;
    p.tensors = {{64,64},{64,64},{64,64}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 64; p.native_h = 64;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK("cfg=(64,64,64) = native OK",  sg.is_valid_tiling(TC(64,64,64)));
    CHECK("cfg=(32,32,32) sub-native OK", sg.is_valid_tiling(TC(32,32,32)));
    CHECK("cfg=(128,64,64) super-w REJECTED", !sg.is_valid_tiling(TC(128,64,64)));
    CHECK("cfg=(64,64,128) super-k REJECTED", !sg.is_valid_tiling(TC(64,64,128)));
}

// ============================================================================
// Granule-fit: PW-produced ephemeral. Slice must fit in (cfg.w, cfg.h).
//
// Subgraph: PW(T0) → T1 (ephemeral) → MM(T1, T2) → T3 (boundary output).
// T1's role from MM LHS: (FROM_NK, FROM_NTH) → slice = (T1.width/nk, T1.height/nth)
// T1.width = K (MM's reduction dim). At nk > 1, slice_w = K/nk = cfg.k.
// Granule-fit requires cfg.w ≥ cfg.k (i.e. PW granule covers a single k-slice).
// ============================================================================

void test_granule_fit_pw_ephemeral_rejects_when_slice_exceeds_cfg() {
    std::cout << "--- test_granule_fit_pw_ephemeral_rejects_when_slice_exceeds_cfg ---\n";
    // T0 and T1 are PW input/output, shape (K=256 wide × out_H=256 tall).
    // T2 is MM RHS (256×256). T3 is MM output (256×256).
    // native=256 lets us vary cfg up to 256.
    Problem p;
    p.tensors = {{256,256}, {256,256}, {256,256}, {256,256}};
    p.ops = {{OpType::Pointwise, {0},   {1}, 500},
             {OpType::MatMul,    {1,2}, {3}, 2000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 256; p.native_h = 256;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // At nk=1 (cfg.k=256): T1 slice_w = 256. PW granule (cfg.w, cfg.h) must
    // hold it, so cfg.w ≥ 256 required.
    CHECK("cfg=(256,256,256) nk=1, fits", sg.is_valid_tiling(TC(256,256,256)));
    CHECK("cfg=(128,256,256) nk=1, slice_w=256>cfg.w=128 REJECTED",
          !sg.is_valid_tiling(TC(128,256,256)));

    // At nk=2 (cfg.k=128): T1 slice_w = 128. PW granule cfg.w must ≥ 128.
    CHECK("cfg=(128,256,128) nk=2, slice_w=128=cfg.w OK",
          sg.is_valid_tiling(TC(128,256,128)));
    CHECK("cfg=(64,256,128) nk=2, slice_w=128>cfg.w=64 REJECTED",
          !sg.is_valid_tiling(TC(64,256,128)));

    // At nk=4 (cfg.k=64): T1 slice_w = 64. PW granule cfg.w must ≥ 64.
    CHECK("cfg=(64,256,64) nk=4, slice_w=64=cfg.w OK",
          sg.is_valid_tiling(TC(64,256,64)));
    CHECK("cfg=(32,256,64) nk=4, slice_w=64>cfg.w=32 REJECTED",
          !sg.is_valid_tiling(TC(32,256,64)));
}

// ============================================================================
// Granule-fit: MM-produced ephemeral. Slice must fit in native.
// For chained MM → MM, the inner MM output is ephemeral; its slice derives
// from cfg, so it's always ≤ cfg ≤ native. This is a defensive check.
// ============================================================================

void test_granule_fit_mm_ephemeral_trivially_holds() {
    std::cout << "--- test_granule_fit_mm_ephemeral_trivially_holds ---\n";
    // Chain MM → MM with sink output 128×128, K=128.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {3}, 2000},
             {OpType::MatMul, {3,2}, {4}, 2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // All cfgs within native should be accepted — MM-produced ephemeral
    // slice is derived from cfg and never exceeds native.
    CHECK("cfg=(128,128,128) OK", sg.is_valid_tiling(TC(128,128,128)));
    CHECK("cfg=(64, 64, 64)  OK", sg.is_valid_tiling(TC(64, 64, 64)));
    CHECK("cfg=(32, 128,32)  OK", sg.is_valid_tiling(TC(32,128,32)));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    test_super_native_w_rejected();
    test_super_native_h_rejected();
    test_super_native_k_rejected();
    test_best_cost_stays_within_native();
    test_small_native_all_subdivisors_ok();
    test_granule_fit_pw_ephemeral_rejects_when_slice_exceeds_cfg();
    test_granule_fit_mm_ephemeral_trivially_holds();

    std::cout << "\n=== " << g_pass << " passed, "
              << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
