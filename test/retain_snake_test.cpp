// retain_snake_test.cpp
// Hand-calculated cost model tests for retained tensors and snake traversals.
// Every expected value is computed by hand and documented inline.

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost.h"
#include <cmath>
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK_EQ(const char* l, double g, double e, double t = 0.1) {
    if (std::abs(g - e) < t) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}
static void CHECK_EQ_I(const char* l, int64_t g, int64_t e) {
    if (g == e) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}
static void CHECK(const char* l, bool c) {
    if (c) g_pass++; else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}

// Tile-by-tile simulation for verification
static double simulate(int ntw, int nth, int nk,
                double mm_comp, double pw_comp,
                double lhs_load, double rhs_load, double out_evict,
                SnakeDir snake, bool lhs_retained) {
    auto tiles = make_traversal(ntw, nth, snake);
    double total = 0;
    int pr = -1, pc = -1;
    for (int tp = 0; tp < (int)tiles.size(); tp++) {
        int r = tiles[tp] / ntw, c = tiles[tp] % ntw;
        bool lf, rf;
        if (snake == SnakeDir::None) { lf = true; rf = true; }
        else if (tp == 0) { lf = true; rf = true; }
        else { lf = (r != pr); rf = (c != pc); }
        for (int ks = 0; ks < nk; ks++) {
            double mi = 0;
            if (ks == 0) { if (lf && !lhs_retained) mi += lhs_load; if (rf) mi += rhs_load; }
            else { mi += rhs_load; }
            double mo = (ks == nk - 1) ? out_evict : 0;
            double sc = (ks == nk - 1) ? mm_comp + pw_comp : mm_comp;
            total += std::max(sc, mi + mo);
        }
        pr = r; pc = c;
    }
    return total;
}

// ==================== Retain working set tests ====================

void test_retain_ws_matmul_lhs() {
    std::cout << "--- test_retain_ws_matmul_lhs ---\n";
    // T0(256x256) @ T1(256x256) -> T2(256x256). K=256. Cap=100000.
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // No retain: ws = h*K + k*w + h*w = 128*256 + 128*128 + 128*128 = 65536
    CHECK_EQ_I("no ret ws", sg->compute_cost({128,128,128,SnakeDir::None}).working_set, 65536);
    // T0 retained: ws = full(65536) + k*w(16384) + h*w(16384) = 98304
    CHECK_EQ_I("ret T0 ws", sg->compute_cost({128,128,128,SnakeDir::None},{0}).working_set, 98304);
    // T1 retained: ws = h*K(32768) + full(65536) + h*w(16384) = 114688 > 100000
    CHECK("ret T1 infeasible", !sg->compute_cost({128,128,128,SnakeDir::None},{1}).feasible);
}

void test_retain_ws_pw() {
    std::cout << "--- test_retain_ws_pw ---\n";
    // PW T0(256x256) -> T1(256x256). Cap=70000.
    Problem p;
    p.tensors = {{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 70000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // No retain: ws = h*w = 16384
    CHECK_EQ_I("PW no ret", sg->compute_cost({128,128,1,SnakeDir::None}).working_set, 16384);
    // T0 retained: ws = full(65536)
    CHECK_EQ_I("PW ret T0", sg->compute_cost({128,128,1,SnakeDir::None},{0}).working_set, 65536);
}

void test_retain_ws_output_accumulation() {
    std::cout << "--- test_retain_ws_output_accumulation ---\n";
    // PW T0(256x256) -> T1(256x256). Retain output T1. Cap=80000.
    Problem p;
    p.tensors = {{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // Retain output: ws = input(16384) + full_out - tile_out = 16384 + 65536-16384 = 65536
    auto c = sg->compute_cost({128,128,1,SnakeDir::None}, {}, {1});
    CHECK_EQ_I("ret out ws", c.working_set, 65536);
    CHECK("ret out feasible", c.feasible);
}

void test_retain_ws_unused_tensor() {
    std::cout << "--- test_retain_ws_unused_tensor ---\n";
    // PW T0(128x128) -> T1(128x128). T2(256x256) retained but unused.
    Problem p;
    p.tensors = {{128,128},{128,128},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // ws = T0(16384) + T2_full(65536) = 81920 > 80000
    auto c = sg->compute_cost({128,128,1,SnakeDir::None}, {2});
    CHECK_EQ_I("unused ret ws", c.working_set, 81920);
    CHECK("unused ret infeasible", !c.feasible);
}

// ==================== Retain transfer cost tests ====================

void test_retain_transfer_lhs() {
    std::cout << "--- test_retain_transfer_lhs ---\n";
    // T0(256x256) @ T1(256x256) -> T2(256x256). K=256.
    // [128,128,128]: 4 tiles, 2 k-passes.
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // No retain: 4 tiles × (max(1000,4915.2) + max(1000,3276.8)) = 4 × 8192 = 32768
    CHECK_EQ("no ret lat", sg->compute_cost({128,128,128,SnakeDir::None}).latency, 32768.0);
    // T0 retained: 4 tiles × (max(1000,1638.4) + max(1000,3276.8)) = 4 × 4915.2 = 19660.8
    CHECK_EQ("ret T0 lat", sg->compute_cost({128,128,128,SnakeDir::None},{0}).latency, 19660.8);
}

void test_retain_transfer_pw() {
    std::cout << "--- test_retain_transfer_pw ---\n";
    Problem p;
    p.tensors = {{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 70000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // No retain: 4 tiles × max(1000, 3276.8) = 4 × 3276.8 = 13107.2
    CHECK_EQ("PW no ret", sg->compute_cost({128,128,1,SnakeDir::None}).latency, 13107.2);
    // T0 retained: 4 tiles × max(1000, 1638.4) = 4 × 1638.4 = 6553.6
    CHECK_EQ("PW ret in", sg->compute_cost({128,128,1,SnakeDir::None},{0}).latency, 6553.6);
}

void test_retain_transfer_output() {
    std::cout << "--- test_retain_transfer_output ---\n";
    Problem p;
    p.tensors = {{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // Retain output: 4 tiles × max(1000, 1638.4+0) = 4 × 1638.4 = 6553.6
    CHECK_EQ("ret out lat", sg->compute_cost({128,128,1,SnakeDir::None},{},{1}).latency, 6553.6);
}

void test_retain_two_step_chain() {
    std::cout << "--- test_retain_two_step_chain ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg0 = Subgraph::create(p, d, {0});
    auto sg1 = Subgraph::create(p, d, {1});

    // Step 0 retain T1: max(1000, 1638.4) = 1638.4
    CHECK_EQ("chain step0", sg0->compute_cost({128,128,1,SnakeDir::None},{},{1}).latency, 1638.4);
    // Step 1 T1 retained: max(1000, 1638.4) = 1638.4
    CHECK_EQ("chain step1", sg1->compute_cost({128,128,1,SnakeDir::None},{1}).latency, 1638.4);
}

void test_retain_fused_mm_pw_splitk() {
    std::cout << "--- test_retain_fused_mm_pw_splitk ---\n";
    // Op0: MM T0@T1→T2. Op1: PW T2→T3. Fused, T2 eph. K=128.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},{OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0,1});

    // [128,128,32]: 4 k-passes. mm=500, pw=500.
    // No ret: k0=max(500,2048)=2048, k1,2=max(500,409.6)=500, k3=max(1000,2048)=2048 → 5096
    CHECK_EQ("fused no ret", sg->compute_cost({128,128,32,SnakeDir::None}).latency, 5096.0);
    // T0 ret: k0=max(500,409.6)=500, k1,2=500, k3=max(1000,2048)=2048 → 3548
    CHECK_EQ("fused ret", sg->compute_cost({128,128,32,SnakeDir::None},{0}).latency, 3548.0);
}

// ==================== Snake × retain cross-product tests ====================
// 3×2 tile grid. T0(256x256) @ T1(384x256) → T2(384x256). K=256.
// [128,128,128]: ntw=3, nth=2, nk=2.

void test_snake_3x2_no_retain() {
    std::cout << "--- test_snake_3x2_no_retain ---\n";
    Problem p;
    p.tensors = {{256,256},{384,256},{384,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},5000}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // ff: k0=max(2500,4915.2)=4915.2, k1=max(2500,3276.8)=3276.8 → 8192
    // rf: k0=max(2500,1638.4)=2500, k1=3276.8 → 5776.8
    // fr: k0=max(2500,3276.8)=3276.8, k1=3276.8 → 6553.6
    double ff=8192, rf=5776.8, fr=6553.6;
    
    CHECK_EQ("None", sg->compute_cost({128,128,128,SnakeDir::None}).latency, 6*ff);
    // RM: ff=1, rf=4, fr=1
    CHECK_EQ("RM", sg->compute_cost({128,128,128,SnakeDir::RowMajor}).latency, ff+4*rf+fr);
    // CM: ff=1, rf=2, fr=3
    CHECK_EQ("CM", sg->compute_cost({128,128,128,SnakeDir::ColMajor}).latency, ff+2*rf+3*fr);
    // RM < CM (3 cols → 4 LHS reuses vs 1)
    CHECK("RM < CM", sg->compute_cost({128,128,128,SnakeDir::RowMajor}).latency <
                      sg->compute_cost({128,128,128,SnakeDir::ColMajor}).latency);
}

void test_snake_3x2_lhs_retained() {
    std::cout << "--- test_snake_3x2_lhs_retained ---\n";
    Problem p;
    p.tensors = {{256,256},{384,256},{384,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},5000}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // LHS=0. All tile types: k0=max(2500,rhs)=2500 (comp dominates), k1=3276.8
    // ff=rf=fr=5776.8. All modes = 6*5776.8 = 34660.8
    CHECK_EQ("None ret", sg->compute_cost({128,128,128,SnakeDir::None},{0}).latency, 34660.8);
    CHECK_EQ("RM ret", sg->compute_cost({128,128,128,SnakeDir::RowMajor},{0}).latency, 34660.8);
    CHECK_EQ("CM ret", sg->compute_cost({128,128,128,SnakeDir::ColMajor},{0}).latency, 34660.8);
}

void test_snake_3x2_membound_retained() {
    std::cout << "--- test_snake_3x2_membound_retained ---\n";
    // Low compute (250) so memory transfers dominate.
    Problem p;
    p.tensors = {{256,256},{384,256},{384,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},500}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // LHS=0. ff/rf: k0=max(250,1638.4)=1638.4, k1=3276.8 → 4915.2
    // fr: k0=max(250,0)=250, k1=3276.8 → 3526.8
    double fast=4915.2, slow=3526.8;

    // RM: ff=1, rf=4, fr=1. 5*fast + slow = 28102.8
    CHECK_EQ("RM ret", sg->compute_cost({128,128,128,SnakeDir::RowMajor},{0}).latency,
             5*fast + slow);
    // CM: ff=1, rf=2, fr=3. 3*fast + 3*slow = 25326
    CHECK_EQ("CM ret", sg->compute_cost({128,128,128,SnakeDir::ColMajor},{0}).latency,
             3*fast + 3*slow);
    // CM wins: more fr tiles → more RHS reuse
    CHECK("CM < RM", sg->compute_cost({128,128,128,SnakeDir::ColMajor},{0}).latency <
                     sg->compute_cost({128,128,128,SnakeDir::RowMajor},{0}).latency);
}

void test_snake_3x2_membound_no_retain() {
    std::cout << "--- test_snake_3x2_membound_no_retain ---\n";
    Problem p;
    p.tensors = {{256,256},{384,256},{384,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},500}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // ff: k0=max(250,4915.2)=4915.2, k1=3276.8 → 8192
    // rf: k0=max(250,1638.4)=1638.4, k1=3276.8 → 4915.2
    // fr: k0=max(250,3276.8)=3276.8, k1=3276.8 → 6553.6

    // RM: ff=1, rf=4, fr=1. 8192+4*4915.2+6553.6 = 34406.4
    CHECK_EQ("RM nret", sg->compute_cost({128,128,128,SnakeDir::RowMajor}).latency,
             8192+4*4915.2+6553.6);
    // CM: ff=1, rf=2, fr=3. 8192+2*4915.2+3*6553.6 = 37683.2
    CHECK_EQ("CM nret", sg->compute_cost({128,128,128,SnakeDir::ColMajor}).latency,
             8192+2*4915.2+3*6553.6);
    // RM wins: more rf tiles → more LHS reuse
    CHECK("RM < CM", sg->compute_cost({128,128,128,SnakeDir::RowMajor}).latency <
                     sg->compute_cost({128,128,128,SnakeDir::ColMajor}).latency);
}

void test_snake_retain_preference_flip() {
    std::cout << "--- test_snake_retain_preference_flip ---\n";
    // Without retain: RowMajor wins (LHS reuse from wide grid)
    // With LHS retained: ColMajor wins (RHS reuse from tall-ish grid, LHS free)
    Problem p;
    p.tensors = {{256,256},{384,256},{384,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},500}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    auto rm_nret = sg->compute_cost({128,128,128,SnakeDir::RowMajor}).latency;
    auto cm_nret = sg->compute_cost({128,128,128,SnakeDir::ColMajor}).latency;
    auto rm_ret = sg->compute_cost({128,128,128,SnakeDir::RowMajor},{0}).latency;
    auto cm_ret = sg->compute_cost({128,128,128,SnakeDir::ColMajor},{0}).latency;

    CHECK("no ret: RM wins", rm_nret < cm_nret);
    CHECK("ret: CM wins", cm_ret < rm_ret);
    std::cout << "  no_ret: RM=" << rm_nret << " CM=" << cm_nret
              << " | ret: RM=" << rm_ret << " CM=" << cm_ret << "\n";
}

// ==================== Simulation cross-validation ====================

void test_simulation_matches_code() {
    std::cout << "--- test_simulation_matches_code ---\n";
    // Run simulation for all 8 combinations and verify code matches
    Problem p;
    p.tensors = {{256,256},{384,256},{384,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},500}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    int ntw=3, nth=2, nk=2;
    double B=10, mm=500.0*128/256, pw=0;
    double lhs=128.0*256/B, rhs=128.0*128/B, out=128.0*128/B;

    for (auto snake : {SnakeDir::None, SnakeDir::RowMajor, SnakeDir::ColMajor}) {
        for (bool ret : {false, true}) {
            double sim = simulate(ntw,nth,nk,mm,pw,lhs,rhs,out,snake,ret);
            auto code = ret ? sg->compute_cost({128,128,128,snake},{0})
                            : sg->compute_cost({128,128,128,snake});
            const char* sn = snake==SnakeDir::None?"None":snake==SnakeDir::RowMajor?"RM":"CM";
            char buf[64]; snprintf(buf, sizeof(buf), "%s %s", sn, ret?"ret":"nret");
            CHECK_EQ(buf, code.latency, sim);
        }
    }
}

// ==================== Main ====================

int main() {
    test_retain_ws_matmul_lhs();
    test_retain_ws_pw();
    test_retain_ws_output_accumulation();
    test_retain_ws_unused_tensor();
    test_retain_transfer_lhs();
    test_retain_transfer_pw();
    test_retain_transfer_output();
    test_retain_two_step_chain();
    test_retain_fused_mm_pw_splitk();
    test_snake_3x2_no_retain();
    test_snake_3x2_lhs_retained();
    test_snake_3x2_membound_retained();
    test_snake_3x2_membound_no_retain();
    test_snake_retain_preference_flip();
    test_simulation_matches_code();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}