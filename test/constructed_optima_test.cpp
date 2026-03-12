// constructed_optima_test.cpp
// Hand-constructed benchmark instances where we KNOW the optimal partition,
// tiling, and cost. Tests that our solver finds the optimum or gets close.
// Each instance tests a different aspect of the cost model and search.

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "partition/partition.h"
#include "search/local_search.h"
#include "search/parallel_search.h"
#include "pipeline/solver.h"
#include "io/io.h"
#include <iostream>
#include <cmath>
#include <cassert>

static int g_pass = 0, g_fail = 0;
static void CHECK_EQ(const char* l, double g, double e, double t = 0.5) {
    if (std::abs(g - e) < t) { g_pass++; }
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e
                               << " diff=" << (g-e) << "\n"; }
}
static void CHECK_LE(const char* l, double g, double e) {
    if (g <= e + 0.5) { g_pass++; }
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " should be <=" << e << "\n"; }
}
static void CHECK(const char* l, bool c) {
    if (c) g_pass++; else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}

// ========================================================================
// Instance 1: "Splitting is optimal"
//
// 4 PW ops in chain, 512x512 tensors, cap=270000, B=10.
// native = 128x128.
//
// Option A: Fuse all 4 → {Op0,Op1,Op2,Op3}
//   ws = 128*128 = 16384 (PW in-place). 16384 << 270000. Feasible at [128,128].
//   Compute per tile: 4 * 500 = 2000. Tiles: (512/128)^2 = 16.
//   Memory per tile: load T0_strip(128*128/10=1638.4) + evict T4_strip(1638.4) = 3276.8
//   Per tile: max(2000, 3276.8) = 3276.8. Total: 16 * 3276.8 = 52428.8
//
// Option B: Two groups {Op0,Op1} + {Op2,Op3}
//   Each group: compute=1000, tiles=16, memory=3276.8
//   Per tile: max(1000, 3276.8) = 3276.8. Each group: 52428.8. Total: 104857.6
//   WORSE — no benefit from splitting equal PW.
//
// Option C: All singletons
//   Each: compute=500, tiles=16, memory=3276.8. Each=52428.8. Total: 4*52428.8 = 209715.2
//   WORST.
//
// So for equal-size PW: fusing all is ALWAYS best (memory cost dominates,
// same memory cost regardless of fusion, fewer intermediate transfers).
// Optimal = 52428.8.
// ========================================================================

void test_fuse_all_optimal() {
    std::cout << "=== Instance 1: Fuse-all optimal (equal PW) ===\n";
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({512, 512});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 500});
    p.fast_memory_capacity = 270000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    // Verify hand calculation
    DAG d = DAG::build(p);
    Partition tmp; tmp.prob = &p; tmp.dag = &d;
    double fused4 = tmp.eval_set({0,1,2,3});
    double split22 = tmp.eval_set({0,1}) + tmp.eval_set({2,3});
    double all_sep = tmp.eval_set({0}) + tmp.eval_set({1}) + tmp.eval_set({2}) + tmp.eval_set({3});

    std::cout << "  fused(4)=" << fused4 << " split(2+2)=" << split22
              << " all_sep=" << all_sep << "\n";
    CHECK_EQ("fused4", fused4, 52428.8);
    CHECK("fused < split", fused4 < split22);
    CHECK("fused < sep", fused4 < all_sep);

    // Run solver
    auto sol = solve(p);
    std::cout << "  solver: " << sol.total_latency() << "\n";
    CHECK_LE("solver finds optimal", sol.total_latency(), fused4);
}

// ========================================================================
// Instance 2: "Splitting is forced by memory"
//
// 2 MatMul ops chained: Op0(T0@T1→T2), Op1(T2@T3→T4).
// T0..T4 all 256x256. K=256. Cap=50000, B=10. Native=128x128.
//
// Singleton Op0 at [128,128,128]:
//   ws = LHS(128*256=32768) + RHS(128*128=16384) + out(128*128=16384) = 65536 > 50000. OOM!
//   At [64,64,128]: ws = 64*256 + 128*64 + 64*64 = 16384+8192+4096 = 28672 ≤ 50000. OK.
//   comp = 3000 * 128/256 = 1500. scale = 1. tiles = (256/64)^2 = 16. nk = 2.
//   Per tile (no snake): k0: lhs(64*256/10=1638.4)+rhs(128*64/10=819.2)=2457.6 → max(1500,2457.6)=2457.6
//                         k1: rhs(819.2)+evict(64*64/10=409.6)=1228.8 → max(1500,1228.8)=1500
//   Per tile=3957.6. Total = 16*3957.6 = 63321.6
//
// Fused {Op0,Op1}: T2 ephemeral. Sink=T4.
//   ws at [64,64,128]: T0_lhs(64*256=16384)+T1_rhs(128*64=8192)+T3_rhs(128*64=8192)+T4_out(64*64=4096)=36864 ≤50000
//   mm_comp = 2*3000*128/256 = 3000. pw_comp=0. nk=2. tiles=16.
//   Per tile (no snake): k0: lhs(1638.4)+rhs_T1(819.2)+rhs_T3(0, not loaded at k0?)
//   Hmm, wait. How does fused MM+MM work? Both T1 and T3 are RHS inputs.
//   T1 is RHS of Op0. T3 is RHS of Op1. Both boundary inputs.
//   At k=0: load LHS(T0,1638.4) + RHS_Op0(T1,819.2) + RHS_Op1(T3,819.2) = 3276.8
//   At k=1: RHS_Op0(819.2) + RHS_Op1(819.2) + evict T4(409.6) = 2048
//   Per tile: max(3000,3276.8) + max(3000,2048) = 3276.8 + 3000 = 6276.8
//   Total = 16 * 6276.8 = 100428.8
//
// Wait, that's WORSE than 2 singletons (2*63321.6=126643.2)... actually 100k < 126k. Better!
// But check with snake:
// Singleton Op0 at [64,64,128] with RowMajor snake: 4 cols, 4 rows.
//   ff=1, rf=3*4=12, fr=3.
//   ff: 3957.6. rf: k0: rhs(819.2) → max(1500,819.2)=1500. k1: 1500. =3000.
//   fr: k0: lhs(1638.4) → max(1500,1638.4)=1638.4. k1: 1500. =3138.4.
//   Total: 3957.6 + 12*3000 + 3*3138.4 = 3957.6 + 36000 + 9415.2 = 49372.8
//
// So singleton with snake: 2*49372.8 = 98745.6 < fused 100428.8.
// Actually these are close. Let me just verify the solver gets a good answer.
// ========================================================================

void test_mm_chain_tight_memory() {
    std::cout << "=== Instance 2: MM chain tight memory ===\n";
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},3000},
             {OpType::MatMul,{2,3},{4},3000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    DAG d = DAG::build(p);
    Partition tmp; tmp.prob = &p; tmp.dag = &d;

    // Evaluate various strategies
    double sep = tmp.eval_set({0}) + tmp.eval_set({1});
    double fused = tmp.eval_set({0,1});

    std::cout << "  separate=" << sep << " fused=" << fused << "\n";

    auto sol = solve(p);
    std::cout << "  solver: " << sol.total_latency() << "\n";
    CHECK_LE("solver ≤ separate", sol.total_latency(), sep + 1);
    CHECK_LE("solver ≤ fused", sol.total_latency(), fused + 1);
}

// ========================================================================
// Instance 3: "Recompute beats spill" (Example 3B from PROBLEM.md)
//
// Diamond: T0→Op0→T1→{Op1→T2, Op2(T1,T2)→T3}
// 128x128, cap=50000, B=10, base=1500.
//
// Strategy C (retain T1): {Op0}+retain T1, then {Op1,Op2} with T1 resident.
//   Step 0: max(1500, 1638.4) = 1638.4 (load T0, no evict T1)
//   Step 1: max(3000, 1638.4) = 3000 (T1 resident, evict T3)
//   Total: 4638.4
//
// We construct this and verify our solver gets ≤ 4638.4.
// ========================================================================

void test_diamond_retain_optimal() {
    std::cout << "=== Instance 3: Diamond partition optimal ===\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1500},
             {OpType::Pointwise,{1},{2},1500},
             {OpType::Pointwise,{1,2},{3},1500}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    // {0,1,2} is rejected: T1 is ephemeral but consumed by both Op1 and Op2 (fan-out).
    // Best partition without retain: {0}+{1,2} = 3276.8+3276.8 = 6553.6
    // With retain: {0} retain T1, then {1,2} with T1 resident = 4638.4
    DAG d = DAG::build(p);
    Partition tmp; tmp.prob = &p; tmp.dag = &d;
    CHECK("fuse-all rejected (eph fan-out)", tmp.eval_set({0,1,2}) >= 1e17);

    double best_no_retain = tmp.eval_set({0}) + tmp.eval_set({1,2});
    double retain_strat = 4638.4;
    std::cout << "  best_no_retain=" << best_no_retain
              << " retain_strat=" << retain_strat << "\n";
    CHECK_EQ("best_no_retain", best_no_retain, 6553.6);

    auto sol = solve(p);
    std::cout << "  solver: " << sol.total_latency() << "\n";
    // Solver should find at least the no-retain partition; with retain it could do 4638.4
    CHECK_LE("solver ≤ no_retain", sol.total_latency(), best_no_retain + 1);
}

// ========================================================================
// Instance 4: "Compute-bound fusion"
//
// Chain of 3 PW ops. 128x128 tensors. Very high compute (50000).
// Cap=50000, B=10.
//
// Singleton: max(50000, 3276.8) = 50000. Total = 3 * 50000 = 150000.
// Fused all: max(150000, 3276.8) = 150000. Total = 150000.
//   Same compute! But fewer transfers.
//   Actually, fused: load T0(1638.4) + evict T3(1638.4) = 3276.8.
//   max(150000, 3276.8) = 150000.
//   Separate: 3 × max(50000, 3276.8) = 3 × 50000 = 150000.
//   Same! Because compute dominates in BOTH cases.
//
// But with B=1 (very slow memory):
//   Singleton: max(50000, 2*16384) = max(50000, 32768) = 50000. Total = 150000.
//   Fused: max(150000, 32768) = 150000.
//   Still same.
//
// When does fusion actually help? When memory_time > compute_time for singletons
// but compute_time > memory_time for the fused group.
// Use low compute, expensive transfers:
//   base=100, B=10, 256x256 tensors.
//   Singleton: tiles=4, per tile: max(100, 3276.8)=3276.8. Each=13107.2. Total=3*13107.2=39321.6
//   Fused(3): tiles=4, compute=300, per tile: max(300, 3276.8)=3276.8. Total=13107.2.
//   Fused saves 26214.4 (66%)!
// ========================================================================

void test_memory_bound_fusion() {
    std::cout << "=== Instance 4: Memory-bound fusion ===\n";
    Problem p;
    for (int i = 0; i <= 3; i++) p.tensors.push_back({256, 256});
    for (int i = 0; i < 3; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 100});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    DAG d = DAG::build(p);
    Partition tmp; tmp.prob = &p; tmp.dag = &d;

    double sep = tmp.eval_set({0}) + tmp.eval_set({1}) + tmp.eval_set({2});
    double fused3 = tmp.eval_set({0,1,2});

    std::cout << "  separate=" << sep << " fused(3)=" << fused3 << "\n";
    CHECK_EQ("sep", sep, 39321.6);
    CHECK_EQ("fused", fused3, 13107.2);

    auto sol = solve(p);
    std::cout << "  solver: " << sol.total_latency() << "\n";
    CHECK_LE("solver finds fused", sol.total_latency(), fused3 + 1);
}

// ========================================================================
// Instance 5: "Snake direction matters"
//
// Single MatMul: T0(256x128) @ T1(128x256) → T2(256x256). K=128.
// Cap=50000, B=10, base=100 (memory-bound).
//
// At [128,128,128]: output 2x2 grid, nk=1.
//   LHS strip: h*K = 128*128 = 16384. load=1638.4
//   RHS strip: k*w = 128*128 = 16384. load=1638.4
//   out: 128*128 = 16384. evict=1638.4
//   ws: 16384+16384+16384 = 49152 ≤ 50000. OK.
//
// No snake: 4 tiles, each: max(100, 1638.4+1638.4+1638.4)=4915.2. Total=19660.8
//
// RowMajor snake: 2 cols, 2 rows. ff=1, rf=1, fr=1.
//   Actually: snake order (0,0)→(0,1)→(1,1)→(1,0)
//   ff=1, rf=2, fr=1.
//   Wait, 2x2 grid: ntw=2, nth=2.
//   RM: ff=1, rf=(ntw-1)*nth=1*2=2, fr=nth-1=1
//   ff: max(100,4915.2)=4915.2
//   rf: max(100,1638.4+1638.4)=3276.8  (skip LHS, load RHS+evict)
//   fr: max(100,1638.4+1638.4)=3276.8  (load LHS+evict, skip RHS)
//   Total: 4915.2 + 2*3276.8 + 3276.8 = 14745.6
//
// ColMajor: ff=1, rf=ntw-1=1, fr=(nth-1)*ntw=1*2=2
//   Total: 4915.2 + 1*3276.8 + 2*3276.8 = 14745.6
//   Same for 2x2 (symmetric).
//
// Now make it asymmetric: T2(384x256) → 3x2 output grid.
// T0(256x128), T1(128x384), T2(384x256). K=128 (=T0.width... wait)
// Actually K = inner dimension. For T0(256x128) @ T1(128x384):
// T0 is LHS: 256 rows, 128 cols → height=256, K=128.
// T1 is RHS: 128 rows, 384 cols → K=128, width=384.
// Output: 256x384. At [128,128,128]: ntw=3, nth=2.
//
// RowMajor: ff=1, rf=(3-1)*2=4, fr=2-1=1. 4 LHS-reuse tiles.
// ColMajor: ff=1, rf=3-1=2, fr=(2-1)*3=3. 3 RHS-reuse tiles.
// For memory-bound (base=100):
//   ff: max(100, lhs+rhs+evict) = lhs+rhs+evict
//   rf: max(100, rhs+evict) = rhs+evict
//   fr: max(100, lhs+evict) = lhs+evict
// LHS=128*128/10=1638.4, RHS=128*128/10=1638.4, evict=128*128/10=1638.4
// ff=4915.2, rf=3276.8, fr=3276.8
// RM: 4915.2 + 4*3276.8 + 1*3276.8 = 4915.2+16384 = 21299.2? Wait:
// 4915.2 + 4*3276.8 + 1*3276.8 = 4915.2 + 13107.2 + 3276.8 = 21299.2
// CM: 4915.2 + 2*3276.8 + 3*3276.8 = 4915.2 + 6553.6 + 9830.4 = 21299.2
// Still same! Because LHS and RHS have equal cost. Need asymmetric.
//
// Use T0(256x512), T1(512x256), T2(256x256). K=512.
// At [128,128,128]: nk=4, ntw=2, nth=2.
// LHS=128*512/10=6553.6, RHS=128*128/10=1638.4, evict=128*128/10=1638.4.
// comp per k-step = 100*128/512 = 25.
// ff: k0=max(25,6553.6+1638.4)=8192, k1-2=max(25,1638.4)=1638.4, k3=max(25,1638.4+1638.4)=3276.8
//     = 8192 + 2*1638.4 + 3276.8 = 14745.6
// rf: k0=max(25,0+1638.4)=1638.4, k1-2=1638.4, k3=3276.8 = 1638.4+2*1638.4+3276.8=8192
// fr: k0=max(25,6553.6+0)=6553.6, k1-2=1638.4, k3=3276.8 = 6553.6+2*1638.4+3276.8=13107.2
//
// RM: ntw=2,nth=2: ff=1, rf=(2-1)*2=2, fr=1
//   1*14745.6 + 2*8192 + 1*13107.2 = 14745.6+16384+13107.2 = 44236.8
// CM: ff=1, rf=2-1=1, fr=(2-1)*2=2
//   1*14745.6 + 1*8192 + 2*13107.2 = 14745.6+8192+26214.4 = 49152
// RM wins! Because LHS is expensive (6553.6), reusing it more (rf=2 vs 1) saves more.
// ========================================================================

void test_snake_direction_matters() {
    std::cout << "=== Instance 5: Snake direction matters ===\n";
    Problem p;
    p.tensors = {{256,512},{512,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},100}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    auto c_none = sg->compute_cost({128,128,128,SnakeDir::None});
    auto c_rm = sg->compute_cost({128,128,128,SnakeDir::RowMajor});
    auto c_cm = sg->compute_cost({128,128,128,SnakeDir::ColMajor});

    std::cout << "  none=" << c_none.latency << " RM=" << c_rm.latency
              << " CM=" << c_cm.latency << "\n";
    // With K=512, nk=4 at k=128. The hand calculation is complex.
    // Just verify RM < CM (LHS is expensive: 128*512/10 = 6553.6 vs RHS 1638.4)
    CHECK("RM < CM", c_rm.latency < c_cm.latency);
    // With nk>1: raster gets LHS reuse within rows, FR degrades to FF.
    // RM and raster give the same tile pattern. So RM ≤ none.
    CHECK("RM <= none", c_rm.latency <= c_none.latency + 0.1);

    auto sol = solve(p);
    std::cout << "  solver: " << sol.total_latency() << "\n";
    CHECK_LE("solver picks RM", sol.total_latency(), c_rm.latency + 1);
}

// ========================================================================
// Instance 6: "Recompute is strictly optimal"
//
// Y-graph: T0 → Op0 → T1 → Op1 → T2
//                               → Op2 → T3
// where Op1 and Op2 BOTH consume T1.
// T0=128x128, T1=256x256, T2=128x128, T3=128x128.
// base_costs: Op0=100, Op1=100, Op2=100. B=10, cap=100000.
//
// Strategy A: All singletons {Op0},{Op1},{Op2}
//   Op0: load T0(1638.4) + evict T1(6553.6) = 8192 → max(100,8192) = 8192
//   Op1: load T1(6553.6) + evict T2(1638.4) = 8192 → max(100,8192) = 8192
//   Op2: load T1(6553.6) + evict T3(1638.4) = 8192 → max(100,8192) = 8192
//   Total: 24576
//
// Strategy B: Fuse {Op0,Op1}, separate {Op2}
//   {Op0,Op1}: load T0(1638.4) + evict T2(1638.4) = 3276.8 → max(200,3276.8) = 3276.8
//     At [128,128]: tiles=(256/128)^2=4. Per tile: max(200,3276.8)=3276.8.
//     Wait — output of {Op0,Op1} is T2(128x128). But T1 is 256x256 (ephemeral inside).
//     The output tiling is driven by sink tensor T2(128x128). At [128,128]: 1 tile.
//     load T0(128*128/10=1638.4) + evict T2(128*128/10=1638.4) = 3276.8.
//     max(200, 3276.8) = 3276.8.
//   {Op2}: load T1(tile strip 128*128/10=1638.4) + evict T3(1638.4) = 3276.8.
//     But wait — T1 was never evicted! It was ephemeral in {Op0,Op1}.
//     So Op2 needs T1 from slow memory, but it was never written there.
//     This means T1 must be evicted by {Op0,Op1} OR Op0 must be recomputed.
//
// Strategy C: Recompute Op0 in {Op0,Op2}
//   {Op0,Op1}: T1 ephemeral. load T0(1638.4) + evict T2(1638.4). Cost = 3276.8.
//   {Op0,Op2}: T1 ephemeral. load T0(1638.4) + evict T3(1638.4). Cost = 3276.8.
//   Wait, but T2 is the sink of {Op0,Op1} (128x128), and T3 is the sink of {Op0,Op2}.
//   Both sinks are 128x128 → 1 tile at [128,128]. 
//   Total: 3276.8 + 3276.8 = 6553.6
//
// Strategy D: Fuse {Op0,Op1}, retain T1, {Op2} uses T1 from fast memory.
//   {Op0,Op1} retain T1: ws = T0(16384) + T1_full(65536) = 81920 ≤ 100000.
//   load T0(1638.4), no evict T2? Wait T2 is the sink. If not retained, evict T2(1638.4).
//   Cost: max(200, 1638.4+1638.4) = 3276.8.
//   {Op2}: T1 retained from prev. ws = T1_full(65536). At [128,128]: 1 tile.
//   load T1: 0 (retained). evict T3(1638.4). max(100, 1638.4) = 1638.4.
//   Total: 3276.8 + 1638.4 = 4915.2
//
// Strategy C (recompute) = 6553.6. Strategy D (retain) = 4915.2. D wins!
// But Strategy C doesn't need retain. Let me verify:
// Actually, Strategy D requires T1 to fit in fast memory (65536 ≤ 100000, yes).
//
// Let me now make T1 too large to retain: cap=60000.
// Then D is infeasible (65536 > 60000).
// Strategy C (recompute): 6553.6. Strategy A: 24576.
// C is optimal. Let's verify.
// ========================================================================

void test_recompute_optimal() {
    std::cout << "=== Instance 6: Recompute is optimal ===\n";
    Problem p;
    p.tensors = {{128,128},{256,256},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},100},  // Op0: T0→T1
             {OpType::Pointwise,{1},{2},100},  // Op1: T1→T2
             {OpType::Pointwise,{1},{3},100}}; // Op2: T1→T3
    p.fast_memory_capacity = 60000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    DAG d = DAG::build(p);
    Partition tmp; tmp.prob = &p; tmp.dag = &d;

    double all_sep = tmp.eval_set({0}) + tmp.eval_set({1}) + tmp.eval_set({2});
    double fuse01 = tmp.eval_set({0,1});
    double fuse02 = tmp.eval_set({0,2});
    double recomp = fuse01 + fuse02;  // Op0 recomputed in both

    std::cout << "  all_sep=" << all_sep << " fuse01=" << fuse01
              << " fuse02=" << fuse02 << " recompute=" << recomp << "\n";

    CHECK("recompute < sep", recomp < all_sep);

    auto sol = solve(p);
    std::cout << "  solver: " << sol.total_latency() << " optimal=" << recomp << "\n";
    CHECK_LE("solver ≤ recompute", sol.total_latency(), recomp + 1);
}

// ========================================================================
// Instance 7: "Split-K fusion beats separation"
//
// Op0: MM T0(128x128)@T1(128x128)→T2(128x128). K=128.
// Op1: MM T2(128x128)@T3(128x128)→T4(128x128). K=128.
// cap=45000, B=10, base=2000.
//
// This is Example 5 from PROBLEM.md! Optimal = 6915.2 (fused with k=32).
// ========================================================================

void test_splitk_fusion() {
    std::cout << "=== Instance 7: Split-K fusion (Example 5) ===\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{3},2000},
             {OpType::MatMul,{3,2},{4},2000}};
    p.fast_memory_capacity = 45000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    double optimal = 6915.2;

    auto sol = solve(p);
    std::cout << "  solver: " << sol.total_latency() << " optimal=" << optimal << "\n";
    CHECK_LE("solver ≤ splitk optimal", sol.total_latency(), optimal + 1);
}

// ========================================================================
// Instance 8: "Fused MM+PW beats separate"
//
// Op0: MM T0(256x256)@T1(256x256)→T2(256x256). K=256.
// Op1: PW T2(256x256)→T3(256x256).
// base=3000 each. cap=80000, B=10.
//
// Fused {Op0,Op1}: T2 ephemeral. PW sink rule forces k=1.
//   At [128,128,1]: ws = T0_lhs(128*256=32768) + T1_rhs(1*128=128) + T3_out(0, PW)
//     Actually with k=1: LHS=128*256=32768, RHS=1*128=128, out=0 (PW out not counted)
//     ws = 32768+128 = 32896 ≤ 80000. Feasible.
//   With k=1, nk=256. Many k-steps but each is cheap.
//   The cost is lower than separate because T2 is ephemeral (no transfer).
//
// Separate {Op0},{Op1}: Op0 evicts T2, Op1 loads T2. Extra bandwidth cost.
//
// We verify fused < separate and solver finds it.
// ========================================================================

void test_retain_helps() {
    std::cout << "=== Instance 8: Fused MM+PW beats separate ===\n";
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},3000},
             {OpType::Pointwise,{2},{3},3000}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    DAG d = DAG::build(p);
    Partition tmp; tmp.prob = &p; tmp.dag = &d;
    double fused = tmp.eval_set({0,1});
    double sep = tmp.eval_set({0}) + tmp.eval_set({1});

    std::cout << "  fused=" << fused << " sep=" << sep << "\n";
    CHECK("fused < sep", fused < sep);

    auto sol = solve(p);
    std::cout << "  solver: " << sol.total_latency() << "\n";
    CHECK_LE("solver ≤ fused", sol.total_latency(), fused + 1);
}

// ========================================================================
// Instance 9: "Non-trivial optimal partition"
//
// 6 PW ops in a chain. 512x512 tensors. base=5000, cap=270000, B=10.
//
// At [128,128]: PW ws = 128*128 = 16384.
// Fuse all 6: compute = 30000. tiles = 16.
//   max(30000, 3276.8) = 30000. Total = 16*30000 = 480000.
// Fuse 3+3: compute = 15000 each.
//   max(15000, 3276.8) = 15000. Each = 240000. Total = 480000. Same!
// All sep: compute = 5000 each.
//   max(5000, 3276.8) = 5000. Each = 80000. Total = 6*80000 = 480000. Same!
//
// Compute always dominates for PW. All partitions give 480000.
// This tests that the solver doesn't crash on ties.
// ========================================================================

void test_all_partitions_equal() {
    std::cout << "=== Instance 9: All partitions equal ===\n";
    Problem p;
    for (int i = 0; i <= 6; i++) p.tensors.push_back({512, 512});
    for (int i = 0; i < 6; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 5000});
    p.fast_memory_capacity = 270000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    auto sol = solve(p);
    std::cout << "  solver: " << sol.total_latency() << " (expected ~480000)\n";
    CHECK_EQ("compute-bound equal", sol.total_latency(), 480000.0, 100);
}

// ========================================================================
// Instance 10: "Wide fan-out"
//
// Op0: T0→T1. Then Op1: T1→T2, Op2: T1→T3, Op3: T1→T4, Op4: T1→T5.
// 128x128, base=100, cap=50000, B=10.
//
// Best: recompute Op0 in each downstream group.
// {Op0,Op1}: load T0(1638.4)+evict T2(1638.4) = 3276.8.
// {Op0,Op2}: same = 3276.8.
// {Op0,Op3}: same = 3276.8.
// {Op0,Op4}: same = 3276.8.
// Total: 4 * 3276.8 = 13107.2
//
// vs all separate: Op0(3276.8) + 4*(load T1(1638.4)+evict(1638.4))=4*3276.8
//   = 3276.8 + 13107.2 = 16384.
//
// Recompute is better by 3276.8 (saves the eviction of T1 from Op0).
// ========================================================================

void test_wide_fanout_recompute() {
    std::cout << "=== Instance 10: Wide fan-out recompute ===\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},100},   // Op0: T0→T1
             {OpType::Pointwise,{1},{2},100},   // Op1: T1→T2
             {OpType::Pointwise,{1},{3},100},   // Op2: T1→T3
             {OpType::Pointwise,{1},{4},100},   // Op3: T1→T4
             {OpType::Pointwise,{1},{5},100}};  // Op4: T1→T5
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    double recompute_all = 4 * 3276.8;  // 13107.2
    double all_sep = 3276.8 + 4 * 3276.8;  // 16384

    auto sol = solve(p);
    std::cout << "  solver: " << sol.total_latency()
              << " recompute=" << recompute_all << " sep=" << all_sep << "\n";
    CHECK_LE("solver ≤ recompute", sol.total_latency(), recompute_all + 1);
}

int main() {
    test_fuse_all_optimal();
    test_mm_chain_tight_memory();
    test_diamond_retain_optimal();
    test_memory_bound_fusion();
    test_snake_direction_matters();
    test_recompute_optimal();
    test_splitk_fusion();
    test_retain_helps();
    test_all_partitions_equal();
    test_wide_fanout_recompute();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}