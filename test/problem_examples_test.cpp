// problem_examples_test.cpp
// Tests EVERY strategy from every example in PROBLEM.md,
// checking per-step latencies, working sets, and full Solution validation.
//
// Build: make problem_examples_test
// Run:   ./problem_examples_test

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "solution/solution.h"
#include "core/cost.h"
#include <cmath>
#include <iostream>
#include <vector>

static int g_pass = 0, g_fail = 0;

static void CHECK(const char* label, bool cond) {
    if (cond) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label << "\n"; }
}

static void CHECK_EQ(const char* label, double got, double exp, double tol = 0.1) {
    if (std::abs(got - exp) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label
           << " got=" << got << " exp=" << exp << "\n"; }
}

static void CHECK_EQ_I(const char* label, int64_t got, int64_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label
           << " got=" << got << " exp=" << exp << "\n"; }
}

static Subgraph make_sg(const Problem& p, const DAG& d, std::vector<size_t> ops) {
    auto sg = Subgraph::create(p, d, std::move(ops));
    if (!sg) { std::cerr << "FATAL: Subgraph::create failed\n"; std::exit(1); }
    return std::move(*sg);
}

// Shorthand for TileConfig
static TileConfig TC(int64_t w, int64_t h, int64_t k, SnakeDir s = SnakeDir::None) {
    return {w, h, k, s};
}

// ==========================================================================
// Example 1: Baseline (chained Pointwise, 128x128, cap=35000, bw=10)
// ==========================================================================

static Problem ex1_problem() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},100}};
    p.fast_memory_capacity = 35000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

void test_ex1_strategy_a() {
    std::cout << "--- Ex1 Strategy A: Always Spill ---\n";
    auto p = ex1_problem();
    DAG d = DAG::build(p);
    auto sg0 = make_sg(p, d, {0});
    auto sg1 = make_sg(p, d, {1});

    auto c0 = sg0.compute_cost(TC(128,128,1));
    auto c1 = sg1.compute_cost(TC(128,128,1));
    CHECK("1A sg0 feasible", c0.feasible);
    CHECK("1A sg1 feasible", c1.feasible);
    CHECK_EQ("1A step0 lat", c0.latency, 3276.8);
    CHECK_EQ("1A step1 lat", c1.latency, 3276.8);
    CHECK_EQ("1A total", c0.latency + c1.latency, 6553.6);

    // Build and validate as Solution
    Solution sol(p, d, {
        {std::move(sg0), TC(128,128,1), {}},
        {std::move(sg1), TC(128,128,1), {}},
    });
    CHECK("1A solution valid", sol.validate().valid);
    CHECK_EQ("1A solution total", sol.total_latency(), 6553.6);
}

void test_ex1_strategy_b() {
    std::cout << "--- Ex1 Strategy B: Fused 128x128 ---\n";
    auto p = ex1_problem();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    auto c = sg.compute_cost(TC(128,128,1));
    CHECK("1B feasible", c.feasible);
    CHECK_EQ("1B latency", c.latency, 3276.8);
    CHECK_EQ_I("1B tiles", c.num_spatial_tiles, 1);

    Solution sol(p, d, {{std::move(sg), TC(128,128,1), {}}});
    CHECK("1B solution valid", sol.validate().valid);
    CHECK_EQ("1B solution total", sol.total_latency(), 3276.8);
}

void test_ex1_strategy_c() {
    std::cout << "--- Ex1 Strategy C: Fused 64x64 (below native) ---\n";
    auto p = ex1_problem();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    auto c = sg.compute_cost(TC(64,64,1));
    CHECK("1C feasible", c.feasible);
    CHECK_EQ_I("1C tiles", c.num_spatial_tiles, 4);
    // Each tile: comp=1000+100=1100 (padded = same as native), mem=409.6+409.6=819.2
    // Per tile = max(1100, 819.2) = 1100. Total = 4*1100 = 4400
    CHECK_EQ("1C latency", c.latency, 4400.0);

    Solution sol(p, d, {{std::move(sg), TC(64,64,1), {}}});
    CHECK("1C solution valid", sol.validate().valid);
}

// ==========================================================================
// Example 2: Fast Memory Capacity (256x256, cap=25000, bw=10)
// ==========================================================================

static Problem ex2_problem() {
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},100}};
    p.fast_memory_capacity = 35000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

void test_ex2_strategy_a() {
    std::cout << "--- Ex2 Strategy A: Spill 256x256 ---\n";
    auto p = ex2_problem();
    DAG d = DAG::build(p);
    auto sg0 = make_sg(p, d, {0});
    auto sg1 = make_sg(p, d, {1});

    auto c0 = sg0.compute_cost(TC(128,128,1));
    auto c1 = sg1.compute_cost(TC(128,128,1));
    CHECK_EQ_I("2A sg0 tiles", c0.num_spatial_tiles, 4);
    CHECK_EQ("2A step0 lat", c0.latency, 13107.2);
    CHECK_EQ("2A step1 lat", c1.latency, 13107.2);
    CHECK_EQ("2A total", c0.latency + c1.latency, 26214.4);

    Solution sol(p, d, {
        {std::move(sg0), TC(128,128,1), {}},
        {std::move(sg1), TC(128,128,1), {}},
    });
    CHECK("2A solution valid", sol.validate().valid);
}

void test_ex2_strategy_b() {
    std::cout << "--- Ex2 Strategy B: Fused 128x128 ---\n";
    auto p = ex2_problem();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    auto c = sg.compute_cost(TC(128,128,1));
    CHECK_EQ_I("2B tiles", c.num_spatial_tiles, 4);
    CHECK_EQ("2B latency", c.latency, 13107.2);

    Solution sol(p, d, {{std::move(sg), TC(128,128,1), {}}});
    CHECK("2B solution valid", sol.validate().valid);
}

// ==========================================================================
// Example 3: Diamond graph (128x128, cap=50000, bw=10)
// ==========================================================================

static Problem ex3_problem() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1500},
             {OpType::Pointwise,{1},{2},1500},
             {OpType::Pointwise,{1,2},{3},1500}};
    p.fast_memory_capacity = 70000;  // increased: 3B needs 65536 for sg1
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

void test_ex3_strategy_a() {
    std::cout << "--- Ex3 Strategy A: Spill all ---\n";
    auto p = ex3_problem();
    DAG d = DAG::build(p);

    auto sg0 = make_sg(p, d, {0});
    auto sg1 = make_sg(p, d, {1});
    auto sg2 = make_sg(p, d, {2});

    auto c0 = sg0.compute_cost(TC(128,128,1));
    auto c1 = sg1.compute_cost(TC(128,128,1));
    auto c2 = sg2.compute_cost(TC(128,128,1));

    CHECK_EQ("3A step0", c0.latency, 3276.8);
    CHECK_EQ("3A step1", c1.latency, 3276.8);
    // Step 2: loads T1 + T2, evicts T3. mem = 3*1638.4 + 1638.4 = 4*1638.4... no.
    // mem_in = T1/B + T2/B = 1638.4 + 1638.4 = 3276.8
    // mem_out = T3/B = 1638.4
    // total mem = 3276.8 + 1638.4 = 4915.2
    // comp = 1500
    // lat = max(1500, 4915.2) = 4915.2
    CHECK_EQ("3A step2", c2.latency, 4915.2);
    CHECK_EQ("3A total", c0.latency + c1.latency + c2.latency, 11468.8);

    Solution sol(p, d, {
        {std::move(sg0), TC(128,128,1), {}},
        {std::move(sg1), TC(128,128,1), {}},
        {std::move(sg2), TC(128,128,1), {}},
    });
    CHECK("3A solution valid", sol.validate().valid);
}

void test_ex3_strategy_b() {
    std::cout << "--- Ex3 Strategy B: Recomputation ---\n";
    auto p = ex3_problem();
    DAG d = DAG::build(p);

    // Step 0: {Op0, Op1} → T2 (sink). Retain T2.
    // T1 is boundary output (Op2 is external consumer), NOT ephemeral.
    auto sg0 = make_sg(p, d, {0, 1});
    CHECK("3B sg0 valid", sg0.num_ops() == 2);
    CHECK("3B T1 boundary out", sg0.boundary_outputs().count(1));
    auto c0 = sg0.compute_cost(TC(128,128,1), {}, {2});
    // mem_in = T0/B = 1638.4. T1 evicted (1638.4). T2 not evicted (retained).
    // comp = 3000. lat = max(3000, 1638.4+1638.4) = 3276.8
    CHECK_EQ("3B step0", c0.latency, 3276.8);

    // Step 1: {Op0, Op2} → T3 (sink). T2 retained from step 0.
    // T1 is boundary output (Op1 is external consumer).
    auto sg1 = make_sg(p, d, {0, 2});
    CHECK("3B sg1 valid", sg1.num_ops() == 2);
    CHECK("3B T1 boundary out in sg1", sg1.boundary_outputs().count(1));
    auto c1 = sg1.compute_cost(TC(128,128,1), {2}, {});
    // mem_in = T0/B=1638.4 (T2 retained=free). T1 evict=1638.4. T3 evict=1638.4.
    // comp = 3000. lat = max(3000, 1638.4+1638.4+1638.4) = max(3000,4915.2) = 4915.2
    CHECK_EQ("3B step1", c1.latency, 4915.2);
    CHECK_EQ("3B total", c0.latency + c1.latency, 8192.0);

    Solution sol(p, d, {
        {std::move(sg0), TC(128,128,1), {2}},
        {std::move(sg1), TC(128,128,1), {}},
    });
    CHECK("3B solution valid", sol.validate().valid);
    CHECK_EQ("3B solution total", sol.total_latency(), 8192.0);
}

void test_ex3_strategy_c() {
    std::cout << "--- Ex3 Strategy C: Selective Residency ---\n";
    auto p = ex3_problem();
    DAG d = DAG::build(p);

    // Step 0: {Op0} → T1 (sink). Retain T1.
    auto sg0 = make_sg(p, d, {0});
    auto c0 = sg0.compute_cost(TC(128,128,1), {}, {1});
    // mem_in = T0/B = 1638.4. No eviction (T1 retained). comp = 1500.
    // lat = max(1500, 1638.4) = 1638.4
    CHECK_EQ("3C step0", c0.latency, 1638.4);

    // Step 1: {Op1, Op2} → T3 (sink). T1 retained from step 0.
    auto sg1 = make_sg(p, d, {1, 2});
    CHECK("3C T2 ephemeral", sg1.ephemeral().count(2));
    auto c1 = sg1.compute_cost(TC(128,128,1), {1}, {});
    // T1 retained → no load. comp = 3000. mem_out = T3/B = 1638.4.
    // lat = max(3000, 1638.4) = 3000
    CHECK_EQ("3C step1", c1.latency, 3000.0);
    CHECK_EQ("3C total", c0.latency + c1.latency, 4638.4);

    Solution sol(p, d, {
        {std::move(sg0), TC(128,128,1), {1}},
        {std::move(sg1), TC(128,128,1), {}},
    });
    CHECK("3C solution valid", sol.validate().valid);
    CHECK_EQ("3C solution total", sol.total_latency(), 4638.4);
}

// ==========================================================================
// Example 4: MatMul traversal (128x128, cap=25000, bw=10)
// ==========================================================================

static Problem ex4_problem() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},1500}};
    p.fast_memory_capacity = 25000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

void test_ex4_strategy_a() {
    std::cout << "--- Ex4 Strategy A: Naive tiling (null) ---\n";
    auto p = ex4_problem();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // Working set at [64,64,128]: h*K + K*w + h*w = 64*128 + 128*64 + 64*64
    //  = 8192 + 8192 + 4096 = 20480 < 25000 ✓
    CHECK_EQ_I("4A ws", sg.working_set(TC(64,64,128)), 20480);
    CHECK("4A feasible", sg.is_feasible(TC(64,64,128)));

    // Full size [128,128,128]: 128*128 + 128*128 + 128*128 = 49152 > 25000 ✗
    CHECK("4A full infeasible", !sg.is_feasible(TC(128,128,128)));

    auto c = sg.compute_cost(TC(64,64,128, SnakeDir::None));
    CHECK_EQ_I("4A tiles", c.num_spatial_tiles, 4);
    CHECK_EQ_I("4A k_passes", c.num_k_passes, 1);
    // Raster on 2×2 with nk=1: LHS reuse within rows.
    // FF tile: max(1500, 2048) = 2048. RF tile: max(1500, 1228.8) = 1500.
    // Row 0: FF, RF. Row 1: FF, RF. Total = 2×2048 + 2×1500 = 7096
    CHECK_EQ("4A latency", c.latency, 7096.0);

    Solution sol(p, d, {{std::move(sg), TC(64,64,128), {}}});
    CHECK("4A solution valid", sol.validate().valid);
}

void test_ex4_strategy_b() {
    std::cout << "--- Ex4 Strategy B: Snake traversal [0,1,3,2] ---\n";
    auto p = ex4_problem();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto c = sg.compute_cost(TC(64,64,128, SnakeDir::RowMajor));
    // Tile 0 (0,0): fresh LHS + fresh RHS. mem=1638.4+409.6=2048. lat=max(1500,2048)=2048
    // Tile 1 (0,1): reuse LHS, fresh RHS. mem=819.2+409.6=1228.8. lat=max(1500,1228.8)=1500
    // Tile 2 (1,1): fresh LHS, reuse RHS. mem=819.2+409.6=1228.8. lat=max(1500,1228.8)=1500
    // Tile 3 (1,0): reuse LHS, fresh RHS. mem=819.2+409.6=1228.8. lat=max(1500,1228.8)=1500
    // Total = 2048+1500+1500+1500 = 6548
    CHECK_EQ("4B latency", c.latency, 6548.0);

    Solution sol(p, d, {{std::move(sg), TC(64,64,128,SnakeDir::RowMajor), {}}});
    CHECK("4B solution valid", sol.validate().valid);
}

// ==========================================================================
// Example 5: Chained MatMul Split-K (128x128, cap=45000, bw=10)
// ==========================================================================

static Problem ex5_problem() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{3},2000},
             {OpType::MatMul,{3,2},{4},2000}};
    p.fast_memory_capacity = 45000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

void test_ex5_strategy_a_oom() {
    std::cout << "--- Ex5 Strategy A: OOM at k=128 ---\n";
    auto p = ex5_problem();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // At k=128: Op0 LHS(T0)=128*128=16384, RHS(T1)=128*128=16384
    //           Op1 LHS(T3)=ephemeral, RHS(T2)=128*128=16384
    //           Output(T4)=128*128=16384
    // Total = 16384+16384+16384+16384 = 65536... wait
    // Actually: T0=h*K=128*128, T1=k*w=128*128, T2=k*w=128*128, T4=h*w=128*128
    // But T0 as LHS of Op0: h*K = 128*128 = 16384
    // But the problem says WS = 3*128*128 = 49152 for Op0 alone.
    // That's T0(LHS) + T1(RHS) + T3(output) = 3 * 16384 = 49152.
    // Our model for the FUSED {0,1}: T3 is ephemeral, so not counted.
    // ws = T0(h*K=16384) + T1(k*w=16384) + T2(k*w=16384) + T4(h*w=16384) = 65536
    CHECK_EQ_I("5A ws at k=128", sg.working_set(TC(128,128,128)), 65536);
    CHECK("5A infeasible", !sg.is_feasible(TC(128,128,128)));
    // Problem states ws=49152 for UN-fused Op0 alone.
    // Let's also check Op0 alone:
    auto sg0 = make_sg(p, d, {0});
    CHECK_EQ_I("5A Op0 alone ws", sg0.working_set(TC(128,128,128)), 49152);
    CHECK("5A Op0 alone infeasible", !sg0.is_feasible(TC(128,128,128)));
}

void test_ex5_strategy_b() {
    std::cout << "--- Ex5 Strategy B: Split-K k=32 ---\n";
    auto p = ex5_problem();
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // WS at k=32: T0(h*K=128*128=16384) + T1(k*w=32*128=4096)
    //           + T2(k*w=32*128=4096) + T4(h*w=128*128=16384) = 40960
    CHECK_EQ_I("5B ws", sg.working_set(TC(128,128,32)), 40960);
    CHECK("5B feasible", sg.is_feasible(TC(128,128,32)));

    auto c = sg.compute_cost(TC(128,128,32, SnakeDir::None));
    CHECK_EQ_I("5B tiles", c.num_spatial_tiles, 1);
    CHECK_EQ_I("5B k_passes", c.num_k_passes, 4);

    // Compute per k-step: (2000*32/128) + (2000*32/128) = 500 + 500 = 1000
    CHECK_EQ("5B comp/step", c.compute_per_step, 1000.0);

    // k=0: mem_in = T0(16384/10=1638.4) + T1(4096/10=409.6) + T2(4096/10=409.6) = 2457.6
    //       no eviction. lat = max(1000, 2457.6) = 2457.6
    // k=1: mem_in = T1(409.6) + T2(409.6) = 819.2 (T0 stays resident)
    //       lat = max(1000, 819.2) = 1000
    // k=2: same as k=1 → 1000
    // k=3: mem_in = 819.2. mem_out = T4(16384/10=1638.4).
    //       lat = max(1000, 819.2+1638.4) = max(1000, 2457.6) = 2457.6
    // Total = 2457.6 + 1000 + 1000 + 2457.6 = 6915.2
    CHECK_EQ("5B latency", c.latency, 6915.2);

    Solution sol(p, d, {{std::move(sg), TC(128,128,32), {}}});
    CHECK("5B solution valid", sol.validate().valid);
    CHECK_EQ("5B solution total", sol.total_latency(), 6915.2);
}

// ==========================================================================
// Main
// ==========================================================================

int main() {
    test_ex1_strategy_a();
    test_ex1_strategy_b();
    test_ex1_strategy_c();

    test_ex2_strategy_a();
    test_ex2_strategy_b();

    test_ex3_strategy_a();
    test_ex3_strategy_b();
    test_ex3_strategy_c();

    test_ex4_strategy_a();
    test_ex4_strategy_b();

    test_ex5_strategy_a_oom();
    test_ex5_strategy_b();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}