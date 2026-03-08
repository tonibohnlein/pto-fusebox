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
    CHECK("T6 sink", sg.sink_tensor() == 6);
}

void test_fanin_working_set() {
    std::cout << "\n=== Fan-in: working set ===\n";
    auto p = fanin_square(); DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // At [128,128,32]:
    // T0: LHS of Op0, h*K0 = 128*128 = 16384 (resident)
    // T1: RHS of Op0, k*w  = 32*128  = 4096  (streamed)
    // T3: LHS of Op1, h*K1 = 128*128 = 16384 (resident)
    // T4: RHS of Op1, k*w  = 32*128  = 4096  (streamed)
    // T6: accumulator, h*w = 128*128 = 16384
    // T2, T5: ephemeral = 0
    // Total = 3*16384 + 2*4096 = 57344
    CHECK_EQ_I("ws k=32", sg.working_set(TC(128,128,32)), 57344);

    // Two LHS row strips vs one in a chain — the fan-in is more memory-hungry
    // Chain of 3 at same params: 1*16384 + 3*4096 + 16384 = 45056
    std::cout << "  Fan-in ws=57344 vs chain-3 ws=45056 (two LHS strips)\n";

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
    // 3 ops * 2000 * 32/128 = 1500
    CHECK_EQ("comp/step", c.compute_per_step, 1500.0);

    double B = 10.0;
    // k=0: load T0(1638.4) + T1(409.6) + T3(1638.4) + T4(409.6) = 4096
    //      lat = max(1500, 4096) = 4096
    double k0 = std::max(1500.0, 2*128.0*128.0/B + 2*32.0*128.0/B);
    CHECK_EQ("k=0", k0, 4096.0);
    // k=1,2: T1(409.6) + T4(409.6) = 819.2. lat = max(1500, 819.2) = 1500
    double k_mid = std::max(1500.0, 2*32.0*128.0/B);
    CHECK_EQ("k_mid", k_mid, 1500.0);
    // k=3: 819.2 + T6 evict(1638.4) = 2457.6. lat = max(1500, 2457.6) = 2457.6
    double k_last = std::max(1500.0, 2*32.0*128.0/B + 128.0*128.0/B);
    CHECK_EQ("k_last", k_last, 2457.6);

    double total = k0 + 2*k_mid + k_last;  // 4096+3000+2457.6 = 9553.6
    CHECK_EQ("total hand", total, 9553.6);
    CHECK_EQ("total code", c.latency, total);
}

void test_fanin_cost_k64() {
    std::cout << "\n=== Fan-in: cost at [128,128,64] ===\n";
    auto p = fanin_square(); DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});
    auto c = sg.compute_cost(TC(128,128,64));

    CHECK_EQ_I("k_passes", c.num_k_passes, 2);
    CHECK_EQ("comp/step", c.compute_per_step, 3000.0);

    double B = 10.0;
    // k=0: T0(1638.4)+T1(819.2)+T3(1638.4)+T4(819.2) = 4915.2
    double k0 = std::max(3000.0, 2*128.0*128.0/B + 2*64.0*128.0/B);
    CHECK_EQ("k=0", k0, 4915.2);
    // k=1: T1(819.2)+T4(819.2)+T6(1638.4) = 3276.8
    double k1 = std::max(3000.0, 2*64.0*128.0/B + 128.0*128.0/B);
    CHECK_EQ("k=1", k1, 3276.8);
    CHECK_EQ("total", c.latency, k0+k1);  // 8192
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

    // ws at [128,128,32]: k must divide both 128 and 256
    // T0: h*K0 = 128*128 = 16384
    // T1: k*w  = 32*128  = 4096
    // T3: h*K1 = 128*256 = 32768  (Op1's LHS has K1=256, bigger row strip)
    // T4: k*w  = 32*128  = 4096
    // T6: h*w  = 128*128 = 16384
    // Total = 73728
    CHECK_EQ_I("ws", sg.working_set(TC(128,128,32)), 73728);

    // k_passes = max_K/k = 256/32 = 8
    auto c = sg.compute_cost(TC(128,128,32));
    CHECK_EQ_I("k_passes", c.num_k_passes, 8);

    // comp/step: Op0 = 2000*32/128 = 500, Op1 = 2000*32/256 = 250, Op2 = 2000*32/128 = 500
    CHECK_EQ("comp/step", c.compute_per_step, 1250.0);

    double B = 10.0;
    // k=0: T0(1638.4) + T1(409.6) + T3(128*256/10=3276.8) + T4(409.6) = 5734.4
    double k0 = std::max(1250.0, 128.0*128.0/B + 32.0*128.0/B + 128.0*256.0/B + 32.0*128.0/B);
    CHECK_EQ("k=0", k0, 5734.4);
    // k=1..6: T1(409.6) + T4(409.6) = 819.2
    double k_mid = std::max(1250.0, 2*32.0*128.0/B);
    CHECK_EQ("k_mid", k_mid, 1250.0);
    // k=7: 819.2 + T6(1638.4) = 2457.6
    double k_last = std::max(1250.0, 2*32.0*128.0/B + 128.0*128.0/B);
    CHECK_EQ("k_last", k_last, 2457.6);

    double total = k0 + 6*k_mid + k_last;
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
    CHECK_EQ("each op", c0.latency, 5096.0);
    double unfused = c0.latency + c1.latency + c2.latency;  // 15288
    CHECK_EQ("unfused", unfused, 15288.0);
    // Fused = 9553.6, saving = 37%
    std::cout << "  Unfused=" << unfused << " Fused=9553.6 (saved "
              << (int)(100*(unfused-9553.6)/unfused) << "%)\n";
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
    // Same as chain of 2 = 6915.2
    CHECK_EQ("latency", c02.latency, 6915.2);

    auto sg1 = make_sg(p, d, {1});
    auto c1 = sg1.compute_cost(TC(128,128,32));

    Solution sol(p, d, {
        {std::move(sg1), TC(128,128,32), {}},
        {std::move(sg02), TC(128,128,32), {}},
    });
    CHECK("valid", sol.validate().valid);
    std::cout << "  Partial=" << sol.total_latency() << " vs Full=9553.6 vs Unfused=15288\n";
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
    CHECK_EQ("total", sol.total_latency(), 9553.6);
}

// ============================================================================
// Tiling constraint: intermediate dimensions restrict valid [w, h, k]
//
// Op0: T0(100×128) @ T1(128×100) → T2(100×128)  ← width 100, NOT power of 2
// Op1: T3(128×128) @ T4(128×128) → T5(128×128)
// Op2: T2 @ T5 → T6(128×128)                     ← sink width 128
//
// w must divide both 128 (sink T6) and 100 (ephemeral T2).
// gcd(128, 100) = 4. So w ∈ {1, 2, 4} only — no w=64 or w=128!
// ============================================================================

void test_fanin_tiling_constraint() {
    std::cout << "\n=== Fan-in: tiling constraint (non-pow2 intermediate) ===\n";
    Problem p;
    p.tensors = {{100,128},  // T0: LHS of Op0, K0=100
                 {128,100},  // T1: RHS of Op0
                 {128,128},  // T2: Op0 output (width=T1.width=128? No!)
                 {128,128},  // T3: LHS of Op1
                 {128,128},  // T4: RHS of Op1
                 {128,128},  // T5: Op1 output
                 {128,128}}; // T6: sink

    // Wait: T2 = T0 @ T1. T0 is (w=100, h=128) = (K=100, H=128).
    // T1 is (w=128, h=100) = (W=128, K=100). So T2.width = T1.width = 128,
    // T2.height = T0.height = 128. Both 128. That doesn't restrict anything.
    //
    // Let me make a case where the intermediate has a weird width.
    // For T2 to have width 100: T2 = T0 @ T1, T2.width = T1.width.
    // So T1.width = 100.

    // Redo:
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

    // T2 is ephemeral, width=100. T6 is sink, width=128.
    // w must divide both 100 and 128. gcd(100,128) = 4.
    // w=128: 100 % 128 ≠ 0 → invalid
    CHECK("w=128 invalid", !sg.is_valid_tiling(TC(128,128,4)));
    // w=64: 100 % 64 ≠ 0 → invalid
    CHECK("w=64 invalid", !sg.is_valid_tiling(TC(64,128,4)));
    // w=4: 100 % 4 = 0, 128 % 4 = 0 → valid
    CHECK("w=4 valid", sg.is_valid_tiling(TC(4,128,4)));
    // w=2: valid
    CHECK("w=2 valid", sg.is_valid_tiling(TC(2,128,4)));

    // compute_cost should return infeasible for invalid tiling
    auto c_bad = sg.compute_cost(TC(128,128,4));
    CHECK("w=128 cost infeasible", !c_bad.feasible);

    auto c_good = sg.compute_cost(TC(4,128,4));
    CHECK("w=4 cost feasible", c_good.feasible);

    // best_cost should find something (restricted to small w)
    auto best = sg.best_cost();
    CHECK("best feasible", best.feasible);
    CHECK("best w divides 100", 100 % best.config.w == 0);
    CHECK("best w divides 128", 128 % best.config.w == 0);
    std::cout << "  Best config: w=" << best.config.w
              << " h=" << best.config.h
              << " k=" << best.config.k << "\n";
}

// ============================================================================
// Tiling constraint: mismatched intermediate heights
//
// Dimensionally consistent fan-in where intermediate tensors have
// different heights, restricting valid h (and w and k).
//
//   T2: (96, 128)  — Op0 output. T5: (128, 96) — Op1 output. T6: (128, 128).
//   h must divide gcd(128, 96) = 32.
//   w must divide gcd(96, 128) = 32.
//   k must divide gcd(128, 128, 96) = 32.
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

    // h=128: T5.height=96, 96%128!=0
    CHECK("h=128 invalid", !sg.is_valid_tiling(TC(32,128,32)));
    // h=64: 96%64!=0
    CHECK("h=64 invalid", !sg.is_valid_tiling(TC(32,64,32)));
    // h=32: divides 128 and 96
    CHECK("h=32 valid", sg.is_valid_tiling(TC(32,32,32)));

    // w=128: T2.width=96, 96%128!=0
    CHECK("w=128 invalid", !sg.is_valid_tiling(TC(128,32,32)));
    // w=32: divides 96 and 128
    CHECK("w=32 valid", sg.is_valid_tiling(TC(32,32,32)));

    // k=64: K2=96 (T2.width), 96%64!=0
    CHECK("k=64 invalid", !sg.is_valid_tiling(TC(32,32,64)));
    // k=32: divides 128, 128, 96
    CHECK("k=32 valid", sg.is_valid_tiling(TC(32,32,32)));

    auto best = sg.best_cost();
    CHECK("best feasible", best.feasible);
    CHECK("best w divides 96", 96 % best.config.w == 0);
    CHECK("best h divides 96", 96 % best.config.h == 0);
    CHECK("best k divides 96", 96 % best.config.k == 0);
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
