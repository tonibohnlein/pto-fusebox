// fanin_matmul_test.cpp
// Tests the fan-in pattern where two MatMul outputs feed a third MatMul.
//
//   T0, T1 -> [Op0] -> T2 (ephemeral)
//   T3, T4 -> [Op1] -> T5 (ephemeral)
//   T2, T5 -> [Op2] -> T6 (sink)
//
// The [w,h,k] tiling is applied uniformly per-op: every MatMul uses its
// own LHS as resident row strip (h x K_i) and its own RHS as streamed
// k-slice (k x w). The topology (which downstream op consumes the output
// and in which role) does NOT affect how an op is tiled.
// Ephemeral intermediates contribute 0 to working set and memory transfers.

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "solution/solution.h"
#include <cmath>
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) { if(c) g_pass++; else { g_fail++; std::cout<<"  FAIL: "<<l<<"\n"; } }
static void CHECK_EQ(const char* l, double g, double e, double t=0.1) {
    if(std::abs(g-e)<t) g_pass++; else { g_fail++; std::cout<<"  FAIL: "<<l<<" got="<<g<<" exp="<<e<<"\n"; } }
static void CHECK_EQ_I(const char* l, int64_t g, int64_t e) {
    if(g==e) g_pass++; else { g_fail++; std::cout<<"  FAIL: "<<l<<" got="<<g<<" exp="<<e<<"\n"; } }
static TileConfig TC(int64_t w, int64_t h, int64_t k, SnakeDir s=SnakeDir::None) { return {w,h,k,s}; }
static Subgraph make_sg(const Problem& p, const DAG& d, std::vector<size_t> ops) {
    auto sg = Subgraph::create(p, d, std::move(ops));
    if(!sg) { std::cerr<<"FATAL: Subgraph::create failed\n"; std::exit(1); }
    return std::move(*sg);
}

// ============================================================================
// Square fan-in: all 128x128, B=10, cap=70000
// ============================================================================

static Problem fanin_square() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},   // Op0: T0 @ T1 -> T2
             {OpType::MatMul,{3,4},{5},2000},    // Op1: T3 @ T4 -> T5
             {OpType::MatMul,{2,5},{6},2000}};   // Op2: T2 @ T5 -> T6
    p.fast_memory_capacity = 70000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

void test_fanin_boundary() {
    std::cout << "=== Fan-in: boundary analysis ===\n";
    auto p = fanin_square(); DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("T2 ephemeral", sg.ephemeral().count(2));
    CHECK("T5 ephemeral", sg.ephemeral().count(5));
    CHECK("T0 boundary in", sg.boundary_inputs().count(0));
    CHECK("T1 boundary in", sg.boundary_inputs().count(1));
    CHECK("T3 boundary in", sg.boundary_inputs().count(3));
    CHECK("T4 boundary in", sg.boundary_inputs().count(4));
    CHECK("4 boundary inputs", sg.boundary_inputs().size() == 4);
    CHECK("T6 is boundary output", sg.boundary_outputs().count(6));
}

void test_fanin_working_set() {
    std::cout << "\n=== Fan-in: working set ===\n";
    auto p = fanin_square(); DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // At [128,128,32] nk=4:
    // T0: FIXED_1×NTH → 128×128 = 16384 (col_load)
    // T1: NK×FIXED_1 → 32×128  = 4096  (stream)
    // T3: FIXED_1×NK → 128×32  = 4096  (stream)
    // T4: NTW×FIXED_1 → 128×128 = 16384 (row_load)
    // T6: NTW×NTH → 128×128 = 16384 (evict)
    // Total = 3*16384 + 2*4096 = 57344
    CHECK_EQ_I("ws k=32", sg.working_set(TC(128,128,32)), 57344);

    // Two non-sink LHS row strips vs one in a chain
    std::cout << "  Fan-in ws=57344 vs chain-3 ws=57344 (both have 2 col_load + 2 stream)\n";

    // At k=128: 3*16384 + 2*16384 = 81920 > 70000
    CHECK("infeasible k=128", !sg.is_feasible(TC(128,128,128)));
}

void test_fanin_cost_k32() {
    std::cout << "\n=== Fan-in: cost at [128,128,32] ===\n";
    auto p = fanin_square(); DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});
    auto c = sg.compute_cost(TC(128,128,32));

    CHECK("feasible", c.feasible);
    CHECK_EQ_I("tiles", c.num_spatial_tiles, 1);
    CHECK_EQ_I("k_passes", c.num_k_passes, 4);
    // New model: only sink Op2 divides by nk=4.
    // Op0 (non-sink): 2000/1*1=2000. Op1 (non-sink): 2000/1*1=2000. Op2 (sink): 2000/4*1=500.
    // Total = 4500
    CHECK_EQ("comp/step", c.compute_per_step, 4500.0);

    double B = 10.0;
    // IO (ntw=1, nth=1):
    //   T0: (FIXED_1, FROM_NTH) → once_load = 128×128/10 = 1638.4
    //   T1: (FROM_NK, FIXED_1)  → stream = 32×128/10 = 409.6
    //   T3: (FIXED_1, FROM_NK)  → stream = 128×32/10 = 409.6  (LHS of non-sink Op1)
    //   T4: (FROM_NTW, FIXED_1) → once_load = 128×128/10 = 1638.4  (ntw=1)
    //   stream_total = 819.2;  once_total = 3276.8
    // k=0: max(4500, once(3276.8)+stream(819.2)) = max(4500, 4096) = 4500
    double k0 = std::max(4500.0, 2*128.0*128.0/B + 2*32.0*128.0/B);
    CHECK_EQ("k=0", k0, 4500.0);
    // k=1,2: max(4500, stream(819.2)) = 4500
    double k_mid = std::max(4500.0, 2*32.0*128.0/B);
    CHECK_EQ("k_mid", k_mid, 4500.0);
    // k=3: max(4500, stream(819.2)+evict(1638.4)) = max(4500, 2457.6) = 4500
    double k_last = std::max(4500.0, 2*32.0*128.0/B + 128.0*128.0/B);
    CHECK_EQ("k_last", k_last, 4500.0);

    double total = k0 + 2*k_mid + k_last;  // 4 × 4500 = 18000
    CHECK_EQ("total hand", total, 18000.0);
    CHECK_EQ("total code", c.latency, total);
}

void test_fanin_cost_k64() {
    std::cout << "\n=== Fan-in: cost at [128,128,64] ===\n";
    auto p = fanin_square(); DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});
    auto c = sg.compute_cost(TC(128,128,64));

    CHECK_EQ_I("k_passes", c.num_k_passes, 2);
    // New model: only sink Op2 divides by nk=2.
    // Op0 (non-sink): 2000/1*1=2000. Op1 (non-sink): 2000/1*1=2000. Op2 (sink): 2000/2*1=1000.
    // Total = 5000
    CHECK_EQ("comp/step", c.compute_per_step, 5000.0);

    double B = 10.0;
    // IO (ntw=1, nth=1):
    //   T0: once_load = 1638.4; T4: once_load = 1638.4; once_total = 3276.8
    //   T1: stream = 64×128/10 = 819.2; T3: stream = 128×64/10 = 819.2; stream_total = 1638.4
    //   T6: evict = 1638.4
    // k=0: max(5000, once(3276.8)+stream(1638.4)) = max(5000, 4915.2) = 5000
    double k0 = std::max(5000.0, 2*128.0*128.0/B + 2*64.0*128.0/B);
    CHECK_EQ("k=0", k0, 5000.0);
    // k=1 (last): max(5000, stream(1638.4)+evict(1638.4)) = max(5000, 3276.8) = 5000
    double k1 = std::max(5000.0, 2*64.0*128.0/B + 128.0*128.0/B);
    CHECK_EQ("k=1", k1, 5000.0);
    CHECK_EQ("total", c.latency, k0+k1);  // 10000
}

// ============================================================================
// Non-square fan-in: Op1 has K1=256 (T3 is 256-wide)
// Verifies that per-op K is used correctly even in fan-in topology.
// ============================================================================

void test_fanin_nonsquare() {
    std::cout << "\n=== Fan-in: non-square K ===\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},
                 {256,128},  // T3: LHS of Op1, K1=256
                 {128,256},  // T4: RHS of Op1
                 {128,128},  // T5 (ephemeral)
                 {128,128}}; // T6 (sink)
    p.ops = {{OpType::MatMul,{0,1},{2},2000},    // K0=128
             {OpType::MatMul,{3,4},{5},2000},     // K1=256
             {OpType::MatMul,{2,5},{6},2000}};    // K2=128
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK_EQ_I("max_K=256", sg.max_K(), 256);

    // ws at [128,128,32]: output_K = 128 (sink Op2), nk = 128/32 = 4
    // Tiling propagation:
    //   T0: FIXED_1×NTH → 128×128 = 16384 (col_load)
    //   T1: NK×FIXED_1 → 32×128 = 4096 (stream)
    //   T3: FIXED_1×NK → 256×32 = 8192 (stream, non-square!)
    //   T4: NTW×FIXED_1 → 128×256 = 32768 (row_load, full non-square tensor)
    //   T6: NTW×NTH → 128×128 = 16384 (evict)
    // Total = 16384 + 4096 + 8192 + 32768 + 16384 = 77824
    CHECK_EQ_I("ws", sg.working_set(TC(128,128,32)), 77824);

    // k_passes = output_K/k = 128/32 = 4
    auto c = sg.compute_cost(TC(128,128,32));
    CHECK_EQ_I("k_passes", c.num_k_passes, 4);

    // New model: only sink Op2 divides by nk=4.
    // Op0 (non-sink): 2000/1*1=2000. Op1 (non-sink): 2000/1*1=2000. Op2 (sink): 2000/4*1=500.
    // Total = 4500
    CHECK_EQ("comp/step", c.compute_per_step, 4500.0);

    double B = 10.0;
    // IO (ntw=1, nth=1):
    //   T0: (FIXED_1, FROM_NTH) → once_load = 128×128/10 = 1638.4
    //   T1: (FROM_NK, FIXED_1)  → stream = 32×128/10 = 409.6
    //   T3: (FIXED_1, FROM_NK)  → stream = 256×32/10 = 819.2  (T3 is 256×128; LHS of Op1)
    //   T4: (FROM_NTW, FIXED_1) → once_load = 128×256/10 = 3276.8  (T4 is 128×256; ntw=1)
    //   T6: evict = 128×128/10 = 1638.4
    // k=0: max(4500, once(1638.4+3276.8)+stream(409.6+819.2)) = max(4500, 6144) = 6144  (IO-bound)
    double k0 = std::max(4500.0, 128.0*128.0/B + 128.0*256.0/B + 32.0*128.0/B + 256.0*32.0/B);
    CHECK_EQ("k=0", k0, 6144.0);
    // k=1,2: max(4500, stream(1228.8)) = 4500
    double k_mid = std::max(4500.0, 32.0*128.0/B + 256.0*32.0/B);
    CHECK_EQ("k_mid", k_mid, 4500.0);
    // k=3: max(4500, stream(1228.8)+evict(1638.4)) = max(4500, 2867.2) = 4500
    double k_last = std::max(4500.0, 32.0*128.0/B + 256.0*32.0/B + 128.0*128.0/B);
    CHECK_EQ("k_last", k_last, 4500.0);

    double total = k0 + 2*k_mid + k_last;  // 6144 + 2×4500 + 4500 = 19644
    CHECK_EQ("total", c.latency, total);
}

// ============================================================================
// Unfused vs fused comparison
// ============================================================================

void test_fanin_unfused() {
    std::cout << "\n=== Fan-in: unfused comparison ===\n";
    auto p = fanin_square(); DAG d = DAG::build(p);
    auto c0 = make_sg(p,d,{0}).compute_cost(TC(128,128,32));
    auto c1 = make_sg(p,d,{1}).compute_cost(TC(128,128,32));
    auto c2 = make_sg(p,d,{2}).compute_cost(TC(128,128,32));
    // Single MM at nk=4: stream=(4096+4096)/10=819.2, evict=1638.4, comp=500.
    // d=0,1,2: 819.2 each. d=3: 2457.6. total = 3×819.2+2457.6 = 4915.2
    CHECK_EQ("each op", c0.latency, 4915.2);
    double unfused = c0.latency + c1.latency + c2.latency;  // 14745.6
    CHECK_EQ("unfused", unfused, 14745.6);
    // Fused = 18000
    std::cout << "  Unfused=" << unfused << " Fused=18000 (saved "
              << (int)(100*(unfused-18000.0)/unfused) << "%)\n";
}

// ============================================================================
// Partial fusion: {Op0, Op2} chain + Op1 separate
// ============================================================================

void test_fanin_partial() {
    std::cout << "\n=== Fan-in: partial {Op0,Op2} chain ===\n";
    auto p = fanin_square(); DAG d = DAG::build(p);

    auto sg02 = make_sg(p, d, {0, 2});
    CHECK("T2 ephemeral", sg02.ephemeral().count(2));
    CHECK("T5 boundary in", sg02.boundary_inputs().count(5));
    // ws: T0(16384)+T1(4096)+T5(4096)+T6(16384) = 40960
    CHECK_EQ_I("ws", sg02.working_set(TC(128,128,32)), 40960);

    auto c02 = sg02.compute_cost(TC(128,128,32));
    // Same as chain of 2: Op0 non-sink(2000) + Op2 sink(500) = comp 2500, total = 10000
    CHECK_EQ("latency", c02.latency, 10000.0);

    auto sg1 = make_sg(p, d, {1});
    auto c1 = sg1.compute_cost(TC(128,128,32));

    Solution sol(p, d, {
        {std::move(sg1), TC(128,128,32), {}},
        {std::move(sg02), TC(128,128,32), {}},
    });
    CHECK("valid", sol.validate().valid);
    std::cout << "  Partial=" << sol.total_latency() << " vs Full=18000 vs Unfused=14745.6\n";
}

// ============================================================================
// Full fusion as Solution
// ============================================================================

void test_fanin_solution() {
    std::cout << "\n=== Fan-in: full fusion as Solution ===\n";
    auto p = fanin_square(); DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});
    Solution sol(p, d, {{std::move(sg), TC(128,128,32), {}}});
    CHECK("valid", sol.validate().valid);
    // Full fan-in fused: comp=4500, total=18000
    CHECK_EQ("total", sol.total_latency(), 18000.0);
}

// ============================================================================
// Tiling constraint: intermediate dimensions restrict valid [w, h, k]
//
// With tiling propagation, granularity >= dimension is always valid
// (max(dim/gran, 1) = 1 tile). Divisibility is only required when
// gran < dim. This means w=128 with intermediate width=100 is VALID
// (gives 1 tile), but w=64 with width=100 is INVALID (100%64≠0).
// ============================================================================

void test_fanin_tiling_constraint() {
    std::cout << "\n=== Fan-in: tiling constraint (non-pow2 intermediate) ===\n";
    Problem p;
    // Redo with T1.width=100 → T2.width=100 (non-pow2 intermediate)
    p.tensors = {{128,128},  // T0: K0 = 128
                 {100,128},  // T1: width=100 → T2.width = 100
                 {100,128},  // T2: Op0 output (100 wide!)
                 {128,128},  // T3
                 {128,128},  // T4
                 {128,128},  // T5
                 {128,128}}; // T6: sink (128 wide)
    p.ops = {{OpType::MatMul,{0,1},{2},2000},    // T0(128×128) @ T1(100×128) → T2(100×128)
             {OpType::MatMul,{3,4},{5},2000},
             {OpType::MatMul,{2,5},{6},2000}};   // T2(100×128) @ T5(128×128) → T6(128×128)
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    auto sg = make_sg(p, d, {0, 1, 2});

    // w=128: 128 >= 100 → valid (max(100/128,1) = 1 tile for T1/T2 dimension)
    CHECK("w=128 valid (gran>=dim)", sg.is_valid_tiling(TC(128,128,4)));
    // w=64: 64 < 100, 100%64=36≠0 → invalid
    CHECK("w=64 invalid", !sg.is_valid_tiling(TC(64,128,4)));
    // w=4: divides both 100 and 128 → valid
    CHECK("w=4 valid", sg.is_valid_tiling(TC(4,128,4)));
    // w=2: divides both → valid
    CHECK("w=2 valid", sg.is_valid_tiling(TC(2,128,4)));

    // w=128 is feasible and produces a valid cost
    auto c = sg.compute_cost(TC(128,128,4));
    CHECK("w=128 cost feasible", c.feasible);

    // w=64 is rejected
    auto c_bad = sg.compute_cost(TC(64,128,4));
    CHECK("w=64 cost infeasible", !c_bad.feasible);

    // best_cost should find something valid
    auto best = sg.best_cost();
    CHECK("best feasible", best.feasible);
    CHECK("best latency positive", best.latency > 0);
    std::cout << "  Best config: w=" << best.config.w
              << " h=" << best.config.h
              << " k=" << best.config.k << "\n";
}

// ============================================================================
// Tiling constraint: mismatched intermediate heights
//
// With gran >= dim allowed, only gran < dim needs divisibility.
// h=128 with height=96: 128 >= 96 → valid (1 tile).
// h=64 with height=96: 64 < 96, 96%64≠0 → invalid.
// ============================================================================

void test_fanin_tiling_height_constraint() {
    std::cout << "\n=== Fan-in: height/width/k constraint ===\n";
    Problem p;
    p.tensors = {{128,128},  // T0: LHS Op0 (K0=128)
                 {96,128},   // T1: RHS Op0
                 {96,128},   // T2: Op0 output (w=96, h=128)
                 {128,96},   // T3: LHS Op1 (K1=128, h=96)
                 {128,128},  // T4: RHS Op1
                 {128,96},   // T5: Op1 output (w=128, h=96)
                 {128,128}}; // T6: sink (w=128, h=128)
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::MatMul,{3,4},{5},2000},
             {OpType::MatMul,{2,5},{6},2000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // h=128: 128 >= 96 → valid (gran >= dim)
    CHECK("h=128 valid (gran>=dim)", sg.is_valid_tiling(TC(32,128,32)));
    // h=64: 64 < 96, 96%64=32≠0 → invalid
    CHECK("h=64 invalid", !sg.is_valid_tiling(TC(32,64,32)));
    // h=32: divides 128 and 96 → valid
    CHECK("h=32 valid", sg.is_valid_tiling(TC(32,32,32)));

    // w=128: 128 >= 96 → valid
    CHECK("w=128 valid (gran>=dim)", sg.is_valid_tiling(TC(128,32,32)));
    // w=32: divides 96 and 128 → valid
    CHECK("w=32 valid", sg.is_valid_tiling(TC(32,32,32)));

    // k=64: 64 < 96, 96%64≠0 → invalid (T2.width=96 is in k_divides)
    CHECK("k=64 invalid", !sg.is_valid_tiling(TC(32,32,64)));
    // k=32: divides all → valid
    CHECK("k=32 valid", sg.is_valid_tiling(TC(32,32,32)));

    auto best = sg.best_cost();
    CHECK("best feasible", best.feasible);
    CHECK("best latency positive", best.latency > 0);
    std::cout << "  Best: w=" << best.config.w << " h=" << best.config.h
              << " k=" << best.config.k << " lat=" << best.latency << "\n";
}

int main() {
    test_fanin_boundary();
    test_fanin_working_set();
    test_fanin_cost_k32();
    test_fanin_cost_k64();
    test_fanin_nonsquare();
    test_fanin_unfused();
    test_fanin_partial();
    test_fanin_solution();
    test_fanin_tiling_constraint();
    test_fanin_tiling_height_constraint();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass+g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}