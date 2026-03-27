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

// Tile-by-tile simulation for verification.
// With tiling propagation: sink MM at nk>1 has both LHS and RHS as stream_load.
// They reload every d-step, no position-based reuse across tiles.
static double simulate(int ntw, int nth, int nk,
                double mm_comp, double pw_comp,
                double stream_load, double out_evict,
                SnakeDir snake, bool lhs_retained, double lhs_stream) {
    auto tiles = make_traversal(ntw, nth, snake);
    double total = 0;
    for (int tp = 0; tp < (int)tiles.size(); tp++) {
        for (int ks = 0; ks < nk; ks++) {
            double mi = lhs_retained ? (stream_load - lhs_stream) : stream_load;
            double mo = (ks == nk - 1) ? out_evict : 0;
            double sc = mm_comp + pw_comp;  // all ops every d-step
            total += std::max(sc, mi + mo);
        }
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

    // No retain: nk=2, ntw=2, nth=2. All slices 128×128=16384. WS=49152.
    CHECK_EQ_I("no ret ws", sg->compute_cost({128,128,128,SnakeDir::None}).working_set, 49152);
    // T0 retained: full(65536) + T1_slice(16384) + T2_slice(16384) = 98304
    CHECK_EQ_I("ret T0 ws", sg->compute_cost({128,128,128,SnakeDir::None},{0}).working_set, 98304);
    // T1 retained: T0_slice(16384) + full(65536) + T2_slice(16384) = 98304
    // 98304 < 100000 → feasible
    CHECK("ret T1 feasible", sg->compute_cost({128,128,128,SnakeDir::None},{1}).feasible);
}

void test_retain_ws_pw() {
    std::cout << "--- test_retain_ws_pw ---\n";
    // PW T0(256x256) -> T1(256x256). Cap=90000 (PW output counts in WS).
    Problem p;
    p.tensors = {{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 90000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // No retain: ws = pw_in(h*w=16384) + boundary_out(h*w=16384) = 32768
    CHECK_EQ_I("PW no ret", sg->compute_cost({128,128,1,SnakeDir::None}).working_set, 32768);
    // T0 retained: ws = full(65536) + boundary_out(16384) = 81920
    CHECK_EQ_I("PW ret T0", sg->compute_cost({128,128,1,SnakeDir::None},{0}).working_set, 81920);
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

    // Retain output: ws = input_tile(16384) + full_retained_out(65536) = 81920
    // PW output not counted in base ws, so retain adds full tensor size.
    auto c = sg->compute_cost({128,128,1,SnakeDir::None}, {}, {1});
    CHECK_EQ_I("ret out ws", c.working_set, 81920);
    // 81920 > 80000 = infeasible at cap=80000
    CHECK("ret out infeasible at 80000", !c.feasible);
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

    // ws = T0_in(16384) + T1_out(16384) + T2_full(65536) = 98304 > 80000
    auto c = sg->compute_cost({128,128,1,SnakeDir::None}, {2});
    CHECK_EQ_I("unused ret ws", c.working_set, 98304);
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

    // No retain, raster: nk=2. Both LHS and RHS are stream_load (NK×NTH, NTW×NK).
    // Every tile: d=0: max(1000, stream(3276.8))=3276.8. d=1: max(1000, stream(3276.8)+evict(1638.4))=4915.2.
    // Per tile = 8192. Total = 4 × 8192 = 32768.
    CHECK_EQ("no ret lat", sg->compute_cost({128,128,128,SnakeDir::None}).latency, 32768.0);
    // T0 retained: T0 not loaded. stream = T1 only = 1638.4.
    // d=0: max(1000, 1638.4) = 1638.4. d=1: max(1000, 1638.4+1638.4) = 3276.8.
    // Per tile = 4915.2. Total = 4 × 4915.2 = 19660.8.
    CHECK_EQ("ret T0 lat", sg->compute_cost({128,128,128,SnakeDir::None},{0}).latency, 19660.8);
}

void test_retain_transfer_pw() {
    std::cout << "--- test_retain_transfer_pw ---\n";
    Problem p;
    p.tensors = {{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 90000;  // PW output counts in WS: retained needs 81920
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // No retain: 4 tiles × max(1000, 1638.4+1638.4) = 4 × 3276.8 = 13107.2
    CHECK_EQ("PW no ret", sg->compute_cost({128,128,1,SnakeDir::None}).latency, 13107.2);
    // T0 retained: 4 tiles × max(1000, 0+1638.4) = 4 × 1638.4 = 6553.6
    CHECK_EQ("PW ret in", sg->compute_cost({128,128,1,SnakeDir::None},{0}).latency, 6553.6);
}

void test_retain_transfer_output() {
    std::cout << "--- test_retain_transfer_output ---\n";
    Problem p;
    p.tensors = {{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 100000;  // must hold input_tile(16384) + full_retained_out(65536) = 81920
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // Retain output: 4 tiles × max(1000, 1638.4+0) = 4 × 1638.4 = 6553.6
    // (no eviction since output is retained, only input transfer)
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

void test_retain_fused_mm_pw_k1() {
    std::cout << "--- test_retain_fused_mm_pw_k1 ---\n";
    // Op0: MM T0@T1→T2. Op1: PW T2→T3. Fused, T2 eph. K=128.
    // PW-only sink (T3 produced by PW, T2 ephemeral): output_K_=1, all k valid.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},{OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0,1});
    CHECK("has_pw_sink", sg.has_value());

    // PW-only sink (T3 produced by PW, T2 ephemeral so MM is not a sink):
    // output_K_=1, nk=1 for any k. All k values valid.
    CHECK("k=128 accepted (nk=1)", sg->is_valid_tiling({128,128,128,SnakeDir::None}));
    CHECK("k=32 accepted (nk=1)", sg->is_valid_tiling({128,128,32,SnakeDir::None}));
    CHECK("k=1 accepted (nk=1)", sg->is_valid_tiling({128,128,1,SnakeDir::None}));

    // best_cost picks k=1 (best for PW-only sink: output_K_=1)
    auto best = sg->best_cost();
    CHECK("best feasible", best.feasible);
    CHECK("best k=1", best.config.k == 1);
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

    // With tiling propagation: sink MM at nk=2.
    // T0(LHS)→NK(2)×NTH(2)=stream, T1(RHS)→NTW(3)×NK(2)=stream.
    // Both stream → reload every tile. No traversal reuse at all.
    // Per tile: d=0: max(2500, stream(3276.8+1638.4))=4915.2.
    //           d=1: max(2500, stream(4915.2)+evict(1638.4))=6553.6.
    //           = 11468.8? No wait...
    // Actually: stream per d-step = (128*256/nk + 384*128/nk) / B
    //   T0 slice: 256/2 × 256/2 = 128×128 = 16384 → 1638.4
    //   T1 slice: 384/3 × 256/2 = 128×128 = 16384 → 1638.4
    //   evict T2: 384/3 × 256/2 = 128×128 = 16384 → 1638.4
    // d=0: max(2500, 1638.4+1638.4) = max(2500, 3276.8) = 3276.8
    // d=1: max(2500, 3276.8+1638.4) = max(2500, 4915.2) = 4915.2
    // per tile = 8192. 6 tiles × 8192 = 49152.
    double all_same = 49152.0;

    CHECK_EQ("None", sg->compute_cost({128,128,128,SnakeDir::None}).latency, all_same);
    CHECK_EQ("RM", sg->compute_cost({128,128,128,SnakeDir::RowMajor}).latency, all_same);
    CHECK_EQ("CM", sg->compute_cost({128,128,128,SnakeDir::ColMajor}).latency, all_same);
    // All equal (stream_load → no traversal reuse)
    CHECK("RM == CM", std::abs(sg->compute_cost({128,128,128,SnakeDir::RowMajor}).latency -
                                sg->compute_cost({128,128,128,SnakeDir::ColMajor}).latency) < 0.1);
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

    // LHS=T0 retained. nk=2 → FR degrades to FF.
    // With LHS free, FF=RF (LHS load is 0 in both cases).
    // All tiles: k0=max(250,RHS(1638.4))=1638.4, k1=max(250,RHS+evict(3276.8))=3276.8 → 4915.2
    double all_same = 6 * 4915.2;  // 29491.2

    // RM: all tiles identical = 29491.2
    CHECK_EQ("RM ret", sg->compute_cost({128,128,128,SnakeDir::RowMajor},{0}).latency, all_same);
    // CM: all tiles identical = 29491.2
    CHECK_EQ("CM ret", sg->compute_cost({128,128,128,SnakeDir::ColMajor},{0}).latency, all_same);
    // CM == RM (nk>1 kills RHS reuse that would have made CM better)
    CHECK("CM == RM", std::abs(sg->compute_cost({128,128,128,SnakeDir::ColMajor},{0}).latency -
                               sg->compute_cost({128,128,128,SnakeDir::RowMajor},{0}).latency) < 0.1);
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

    // Stream_load for both LHS and RHS (nk=2). No traversal reuse.
    // Per tile: d=0: max(250, 3276.8)=3276.8. d=1: max(250, 3276.8+1638.4)=4915.2.
    // = 8192 per tile. 6 tiles = 49152.
    double all_same = 49152.0;

    CHECK_EQ("RM nret", sg->compute_cost({128,128,128,SnakeDir::RowMajor}).latency, all_same);
    CHECK_EQ("CM nret", sg->compute_cost({128,128,128,SnakeDir::ColMajor}).latency, all_same);
    // All equal
    CHECK("RM == CM", std::abs(sg->compute_cost({128,128,128,SnakeDir::RowMajor}).latency -
                               sg->compute_cost({128,128,128,SnakeDir::ColMajor}).latency) < 0.1);
}

void test_snake_retain_preference_flip() {
    std::cout << "--- test_snake_retain_preference_flip ---\n";
    // With tiling propagation at nk>1: both LHS and RHS are stream_load.
    // No traversal reuse → all modes equal, with or without retain.
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

    CHECK("no ret: RM == CM", std::abs(rm_nret - cm_nret) < 0.1);
    CHECK("ret: CM == RM", std::abs(cm_ret - rm_ret) < 0.1);
    std::cout << "  no_ret: RM=" << rm_nret << " CM=" << cm_nret
              << " | ret: RM=" << rm_ret << " CM=" << cm_ret << "\n";
}

// ==================== Simulation cross-validation ====================

void test_simulation_matches_code() {
    std::cout << "--- test_simulation_matches_code ---\n";
    Problem p;
    p.tensors = {{256,256},{384,256},{384,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},500}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    int ntw=3, nth=2, nk=2;
    double B=10;
    // comp_per_step = base_cost / nk = 500/2 = 250
    double mm=250.0, pw=0;
    // stream = T0_slice + T1_slice = (128*128 + 128*128) / B = 3276.8
    double stream = (128.0*128 + 128.0*128) / B;
    double lhs_stream = 128.0*128/B;  // T0 slice alone
    double out = 128.0*128/B;

    for (auto snake : {SnakeDir::None, SnakeDir::RowMajor, SnakeDir::ColMajor}) {
        for (bool ret : {false, true}) {
            double sim = simulate(ntw,nth,nk,mm,pw,stream,out,snake,ret,lhs_stream);
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
    test_retain_fused_mm_pw_k1();
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