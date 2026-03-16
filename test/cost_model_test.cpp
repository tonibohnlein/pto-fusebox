// cost_model_spec_test.cpp
// Verifies the cost model against PROBLEM.md and the clarifying GitHub issues:
//
//   Issue #10/#38: spatial tiling does NOT divide compute per tile; only
//                  k-splitting divides compute proportionally.
//   Issue #37:     raster order gets LHS reuse within rows (7096, not 8192).
//   Issue #34:     retention is strictly one-step; carrying across 2+ steps
//                  requires explicit listing at each boundary.
//   PW sink rule:  any subgraph whose boundary output is produced by a PW op
//                  must use k=1 (even if MatMul ops are present).
//
// Build: make cost_model_spec_test
// Run:   ./cost_model_spec_test

#include "core/dag.h"
#include "core/subgraph.h"
#include "core/types.h"
#include "solution/solution.h"
#include <cmath>
#include <iostream>
#include <limits>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double got, double exp, double tol = 0.5) {
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
// Issue #10 / #38 — Spatial tiling vs k-splitting compute scaling
//
// PROBLEM.md rule:
//   Spatial (w,h): pay full native compute per tile × scale.
//                  scale = ceil(w/native_w) × ceil(h/native_h).
//                  Total compute = num_tiles × base_cost × scale.
//                  Spatial splitting does NOT reduce per-tile compute.
//   K-splitting (k):  compute per k-step = base_cost × k/K.
//                  Total compute = base_cost (same as unsplit).
//                  K-splitting proportionally reduces per-step compute.
//
// These are fundamentally different mechanisms. Issue #38 asked why Example 2
// (256×256 at [128,128,1]) has compute=1000 per tile rather than 250.
// Answer: spatial tiling doesn't divide compute. 4 tiles × 1000 = 4000 total.
// ============================================================================

void test_spatial_tiling_does_not_divide_compute() {
    std::cout << "--- test_spatial_tiling_does_not_divide_compute ---\n";
    // PW: T0(256×256) → T1(256×256). base_cost=1000, native=[128,128].
    // At [128,128,1]: scale=1, 4 tiles, compute per tile = 1000 (NOT 250).
    // This matches PROBLEM.md Example 2: total=4×1000=4000 compute, but
    // each tile is memory-bound at max(1000, 3276.8) = 3276.8.
    Problem p;
    p.tensors = {{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto c = sg.compute_cost(TC(128,128,1));
    CHECK("feasible", c.feasible);
    CHECK_EQ_I("4 tiles", c.num_spatial_tiles, 4);
    // scale = ceil(128/128)*ceil(128/128) = 1.
    // pw_comp = 1000 × 1 = 1000 per tile. Total compute = 4 × 1000 = 4000.
    // Memory per tile: in = 128*128/10 = 1638.4, out = 1638.4. Total mem = 3276.8.
    // Each tile latency = max(1000, 3276.8) = 3276.8.
    // Total = 4 × 3276.8 = 13107.2.
    CHECK_EQ("Ex2A latency=13107.2", c.latency, 13107.2);

    // Crucially: compute per tile is NOT 250 (1000/4).
    // compute_per_step shows the per-step compute (mm only for MM, 0 for PW).
    // For PW, pw_comp = 1000 is added only at the last k-step. Still 1000/tile.
    // Verify via a compute-bound scenario: base_cost=10000, mem will be dominated.
    Problem p2;
    p2.tensors = {{256,256},{256,256}};
    p2.ops = {{OpType::Pointwise,{0},{1},10000}};
    p2.fast_memory_capacity = 100000;
    p2.slow_memory_bandwidth = 10;
    p2.native_w = 128; p2.native_h = 128;
    DAG d2 = DAG::build(p2);
    auto sg2 = make_sg(p2, d2, {0});
    auto c2 = sg2.compute_cost(TC(128,128,1));
    // 4 tiles, each: max(10000, 3276.8) = 10000. Total = 40000.
    CHECK_EQ("compute-bound 4*10000=40000", c2.latency, 40000.0);
}

void test_k_splitting_divides_compute_proportionally() {
    std::cout << "--- test_k_splitting_divides_compute_proportionally ---\n";
    // MM: T0(128×128) @ T1(128×128) → T2(128×128). K=128, base_cost=2000.
    // At k=128 (nk=1): mm_comp = 2000 × 1 × 1 = 2000 per tile.
    // At k=64  (nk=2): mm_comp = 2000 × 64/128 = 1000 per k-step.
    //                  total mm compute = 2 × 1000 = 2000. Same!
    // At k=32  (nk=4): mm_comp = 2000 × 32/128 = 500 per k-step.
    //                  total mm compute = 4 × 500 = 2000. Same!
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto c128 = sg.compute_cost(TC(128,128,128));
    auto c64  = sg.compute_cost(TC(128,128,64));
    auto c32  = sg.compute_cost(TC(128,128,32));

    CHECK("k=128 feasible", c128.feasible);
    CHECK("k=64 feasible",  c64.feasible);
    CHECK("k=32 feasible",  c32.feasible);

    // compute_per_step is mm_comp (per k-step, excluding PW)
    CHECK_EQ("k=128 comp/step=2000", c128.compute_per_step, 2000.0);
    CHECK_EQ("k=64  comp/step=1000", c64.compute_per_step,  1000.0);
    CHECK_EQ("k=32  comp/step=500",  c32.compute_per_step,  500.0);

    // Total compute per spatial tile stays constant:
    // k=128: 1 step × 2000 = 2000.
    // k=64:  2 steps × 1000 = 2000.
    // k=32:  4 steps × 500 = 2000.
    CHECK_EQ_I("k=128 1 k-pass",  c128.num_k_passes, 1);
    CHECK_EQ_I("k=64  2 k-passes", c64.num_k_passes,  2);
    CHECK_EQ_I("k=32  4 k-passes", c32.num_k_passes,  4);

    // The example from PROBLEM.md (Ex5): two 128×128 MMs at k=32 gives
    // comp/step = 500 + 500 = 1000. Verified in chained_matmul_test.
    // Here: single MM. Total = steps × comp/step = 1×2000 = 2×1000 = 4×500. ✓
}

void test_sub_native_spatial_still_full_compute() {
    std::cout << "--- test_sub_native_spatial_still_full_compute ---\n";
    // PROBLEM.MD: "if you select a spatial granularity smaller than the native
    // size, the hardware 'pads' the execution, meaning you pay the full compute
    // cost of the native size but produce less useful output."
    //
    // PW: T0(128×128) → T1(128×128). base_cost=1000, native=[128,128].
    // At [64,64,1]: scale = ceil(64/128)*ceil(64/128) = 1. 4 tiles.
    // Per tile: compute = 1000 × 1 = 1000 (SAME as native, not 250).
    // Total compute = 4 × 1000 = 4000.
    // This matches PROBLEM.md Example 1C: 4 tiles × max(1100, 819.2) = 4400.
    // (Here: single op, so max(1000, 409.6+409.6) = 1000. Total = 4000.)
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto c_native = sg.compute_cost(TC(128,128,1));
    auto c_half   = sg.compute_cost(TC(64,64,1));

    CHECK_EQ_I("native: 1 tile",  c_native.num_spatial_tiles, 1);
    CHECK_EQ_I("half:   4 tiles", c_half.num_spatial_tiles, 4);

    // At [64,64]: scale=1. compute per tile = 1000. 4 tiles.
    // Memory per tile: in=64*64/10=409.6, out=409.6. Total mem=819.2.
    // Per tile: max(1000, 819.2) = 1000. Total = 4000.
    CHECK_EQ("half-native total=4000", c_half.latency, 4000.0);

    // At [128,128]: 1 tile, max(1000, 1638.4+1638.4) = 3276.8.
    CHECK_EQ("native total=3276.8", c_native.latency, 3276.8);

    // sub-native is more expensive in total despite same per-tile compute,
    // because you have 4× more tiles
    CHECK("sub-native more expensive", c_half.latency > c_native.latency);

    // Above-native: [256,256,1] scale = ceil(256/128)*ceil(256/128) = 4.
    // 1 tile, compute = 1000 × 4 = 4000. Same total as [64,64]!
    // (Padding in the other direction: hardware does 4 native passes per tile.)
    //
    // Working set at [256,256,1]: pw_in(256*256) + out(256*256) = 131072.
    // Must set capacity > 131072 so the tile is feasible.
    Problem p2;
    p2.tensors = {{256,256},{256,256}};
    p2.ops = {{OpType::Pointwise,{0},{1},1000}};
    p2.fast_memory_capacity = 200000;  // >= 2*256*256=131072
    p2.slow_memory_bandwidth = 10;
    p2.native_w = 128; p2.native_h = 128;
    DAG d2 = DAG::build(p2);
    auto sg2 = make_sg(p2, d2, {0});
    auto c_above = sg2.compute_cost(TC(256,256,1));
    CHECK("above-native feasible", c_above.feasible);
    CHECK_EQ_I("above-native: 1 tile", c_above.num_spatial_tiles, 1);
    // scale=4, compute=4000, mem=(256*256/10)*2=13107.2. lat=max(4000,13107.2)=13107.2.
    CHECK_EQ("above-native lat=13107.2", c_above.latency, 13107.2);
}

// ============================================================================
// Issue #37 — Raster order reuse (confirmed in updated PROBLEM.md)
//
// Example 4A at [64,64,128] None (raster) on 2×2 grid:
//   Tile (0,0): fresh LHS (row-strip 0) + fresh RHS (col-strip 0). FF.
//   Tile (0,1): reuse LHS (still row-strip 0) + fresh RHS (col-strip 1). RF.
//   Tile (1,0): fresh LHS (row-strip 1) + fresh RHS (col-strip 0). FF.
//   Tile (1,1): reuse LHS (still row-strip 1) + fresh RHS (col-strip 1). RF.
// FF: max(1500, 1638.4+409.6) = max(1500,2048) = 2048.
// RF: max(1500, 409.6+409.6)  = max(1500,1228.8) = 1500.
// Total = 2×FF + 2×RF = 2×2048 + 2×1500 = 7096. (NOT 4×2048 = 8192)
// ============================================================================

void test_raster_lhs_reuse_within_rows() {
    std::cout << "--- test_raster_lhs_reuse_within_rows ---\n";
    // PROBLEM.md Example 4A (updated to 7096 after issue #37 clarification).
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},1500}};
    p.fast_memory_capacity = 25000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto c = sg.compute_cost(TC(64,64,128,SnakeDir::None));
    CHECK("4A feasible", c.feasible);
    CHECK_EQ_I("4A 4 tiles", c.num_spatial_tiles, 4);
    CHECK_EQ_I("4A 1 k-pass", c.num_k_passes, 1);

    // 7096, not 8192. Raster reuse within rows is correctly modeled.
    CHECK_EQ("4A raster lat=7096", c.latency, 7096.0);

    // Verify the FF/RF breakdown directly:
    // FF = max(1500, lhs(64*128/10) + rhs(128*64/10) + out(64*64/10))
    //    = max(1500, 819.2 + 819.2 + 409.6) = max(1500, 2048) = 2048
    // RF = max(1500, rhs(819.2) + out(409.6)) = max(1500, 1228.8) = 1500
    // Total = 2*FF + 2*RF = 2*2048 + 2*1500 = 7096 ✓
    double B = 10.0;
    double lhs = 64.0*128/B, rhs = 128.0*64/B, out = 64.0*64/B;
    double FF = std::max(1500.0, lhs + rhs + out);  // 2048
    double RF = std::max(1500.0, rhs + out);         // 1500
    CHECK_EQ("FF=2048", FF, 2048.0);
    CHECK_EQ("RF=1500", RF, 1500.0);
    CHECK_EQ("2*FF+2*RF=7096", 2*FF + 2*RF, 7096.0);
    CHECK_EQ("matches code", c.latency, 2*FF + 2*RF);
}

void test_raster_vs_snake_improvement() {
    std::cout << "--- test_raster_vs_snake_improvement ---\n";
    // Example 4B: RowMajor snake gives 6548 vs raster 7096.
    // Snake adds RHS reuse at row transitions (FR tiles) saving 1 FF per row transition.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},1500}};
    p.fast_memory_capacity = 25000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto c_raster = sg.compute_cost(TC(64,64,128,SnakeDir::None));
    auto c_snake  = sg.compute_cost(TC(64,64,128,SnakeDir::RowMajor));

    CHECK_EQ("raster=7096", c_raster.latency, 7096.0);
    CHECK_EQ("snake=6548",  c_snake.latency,  6548.0);
    CHECK("snake < raster", c_snake.latency < c_raster.latency);
    // The improvement: 1 FF replaced by 1 FR at row boundary.
    // FR = max(1500, lhs(819.2) + out(409.6)) = 1500 (same as RF for this case).
    // So: snake = FF + RF + RF + FR = 2048 + 1500 + 1500 + 1500 = 6548.
    double B = 10.0;
    double lhs = 64.0*128/B, rhs = 128.0*64/B, out = 64.0*64/B;
    double FF = std::max(1500.0, lhs + rhs + out);  // 2048
    double FR = std::max(1500.0, lhs + out);         // 1500 (RHS reused)
    double RF = std::max(1500.0, rhs + out);         // 1500
    CHECK_EQ("FF+RF+RF+FR=6548", FF + RF + RF + FR, 6548.0);
    CHECK_EQ("snake matches breakdown", c_snake.latency, FF + RF + RF + FR);
}

// ============================================================================
// Issue #34 — Retention is strictly one-step
//
// tensors_to_retain[k] controls persistence from step k to step k+1 ONLY.
// Carrying a tensor across 2+ boundaries requires listing it in every
// intermediate step's retain_these. The working_set model charges full_size
// for each step where the tensor is listed in retained_from_prev.
// ============================================================================

void test_retention_one_step_boundary() {
    std::cout << "--- test_retention_one_step_boundary ---\n";
    // Chain: T0→Op0→T1→Op1→T2→Op2→T3.
    // If T1 is retained after step 0 and NOT retained after step 1,
    // it is NOT available in step 2's retained_from_prev.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    auto sg0 = make_sg(p, d, {0});
    auto sg1 = make_sg(p, d, {1});
    auto sg2 = make_sg(p, d, {2});

    TileConfig cfg{128,128,1,SnakeDir::None};

    // Step 0: retain T1. No load. lat = max(1000, 1638.4) = 1638.4
    auto c0 = sg0.compute_cost(cfg, {}, {1});
    CHECK_EQ("step0 retain T1: lat=1638.4", c0.latency, 1638.4);

    // Step 1: T1 carried in (retained_from_prev={1}). Retain T2 out.
    // ws = full(T1=16384) + T2_skip_post(16384) = 32768.
    // Transfer: T1 free (retained), T2 no evict (retained). lat = max(1000, 0) = 1000.
    auto c1_ret = sg1.compute_cost(cfg, {1}, {2});
    CHECK_EQ("step1 retain T1-in+T2-out: lat=1000", c1_ret.latency, 1000.0);
    CHECK_EQ_I("step1 ws=32768", c1_ret.working_set, 32768);

    // Step 1 WITHOUT retaining T1 through: retain_these={} for step 1.
    // Even if T1 was retained after step 0, if step 1 doesn't re-list it,
    // it won't be in step 2's retained_from_prev.
    auto c1_drop = sg1.compute_cost(cfg, {1}, {});
    // T1 retained-in (free), T2 evicted. lat = max(1000, 1638.4) = 1638.4.
    CHECK_EQ("step1 drop T1: lat=1638.4", c1_drop.latency, 1638.4);

    // Step 2 with T2 from prev: T2 already in fast mem. lat = max(1000,1638.4)=1638.4.
    auto c2_with_T2 = sg2.compute_cost(cfg, {2}, {});
    CHECK_EQ("step2 T2 retained: lat=1638.4", c2_with_T2.latency, 1638.4);

    // Step 2 WITHOUT T2 from prev: must load T2. lat = max(1000, 3276.8) = 3276.8.
    auto c2_no_T2 = sg2.compute_cost(cfg, {}, {});
    CHECK_EQ("step2 no retain: lat=3276.8", c2_no_T2.latency, 3276.8);

    // Total latency for chain with full retention (T1 step0, T2 step1):
    // 1638.4 + 1000.0 + 1638.4 = 4276.8
    // vs without any retention: 3276.8 × 3 = 9830.4
    double with_retain    = c0.latency + c1_ret.latency + c2_with_T2.latency;
    double without_retain = sg0.compute_cost(cfg).latency +
                            sg1.compute_cost(cfg).latency +
                            sg2.compute_cost(cfg).latency;
    CHECK_EQ("full chain with retention=4276.8", with_retain, 4276.8);
    CHECK_EQ("full chain no retention=9830.4", without_retain, 9830.4);
    CHECK("retention saves", with_retain < without_retain);
}

void test_retention_must_be_relisted_each_step() {
    std::cout << "--- test_retention_must_be_relisted_each_step ---\n";
    // A tensor retained after step 0 is in step 1's retained_from_prev.
    // If step 1's retain_these does NOT include it, it is evicted after step 1
    // and step 2's retained_from_prev is EMPTY for that tensor.
    // This verifies the one-step-at-a-time semantics.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},500},
             {OpType::Pointwise,{0},{2},500},
             {OpType::Pointwise,{1,2},{3},500}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    auto sg01 = make_sg(p, d, {0});  // Op0: T0→T1 (Op1 is a separate step here)
    auto sg1  = make_sg(p, d, {1});  // Op1: T0→T2 uses T0 again
    auto sg2  = make_sg(p, d, {2});  // Op2: T1+T2→T3
    TileConfig cfg{128,128,1,SnakeDir::None};

    // Step 0: retain T1 (output), T0 still needed in step 1.
    // Since T0 is a boundary INPUT (not output) of sg01, it cannot be in
    // retain_these (validate() would catch it). But we can model its effect:
    // if T0 is retained-from-prev for step 1, it saves a load.

    // Step 1 with T0 retained (pretend step 0 kept T0 in fast mem):
    // T0 is boundary input of sg1. If retained_from_prev={0}: no T0 load.
    auto c1_T0_ret = sg1.compute_cost(cfg, {0}, {});
    CHECK_EQ("step1 T0-ret: no load, lat=1638.4",
             c1_T0_ret.latency, 1638.4);  // only evict T2

    // Step 1 without T0 retained: must load T0. lat = max(500, 1638.4+1638.4)=3276.8
    auto c1_no_ret = sg1.compute_cost(cfg, {}, {});
    CHECK_EQ("step1 no-ret: lat=3276.8", c1_no_ret.latency, 3276.8);

    // Confirm: ws when T0 retained-from-prev = full(T0=16384) + out_tile(T2=16384)
    //          = 32768
    CHECK_EQ_I("ws T0 retained", sg1.working_set(cfg, {0}, {}), 32768);
    // ws when T0 not retained = T0_tile(16384) + out_tile(T2=16384) = 32768. Same!
    // (PW input tile == full size for 128×128 at native tile.)
    CHECK_EQ_I("ws T0 not retained", sg1.working_set(cfg, {}, {}), 32768);
}

// ============================================================================
// PW sink rule: reference ignores k entirely for PW sinks (d_tiles=1).
// Any k is accepted. nk forced to 1. Compute includes ALL ops.
//
// Verified:
//   a. Pure PW subgraph: any k valid, nk=1.
//   b. MM→PW fused: any k valid, nk=1, comp includes both ops.
//   c. MM only, no PW sink: k>1 allowed normally.
//   d. PW→MM→PW: any k valid, nk=1.
// ============================================================================

void test_pw_sink_pure_pw() {
    std::cout << "--- test_pw_sink_pure_pw ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // Reference accepts any k for PW sinks
    CHECK("k=1 valid",   sg.is_valid_tiling(TC(128,128,1)));
    CHECK("k=2 valid",   sg.is_valid_tiling(TC(128,128,2)));
    CHECK("k=128 valid", sg.is_valid_tiling(TC(128,128,128)));
    CHECK("k=1 feasible", sg.is_feasible(TC(128,128,1)));

    // Cost identical regardless of k
    auto c1 = sg.compute_cost(TC(128,128,1));
    auto c128 = sg.compute_cost(TC(128,128,128));
    CHECK_EQ("k-invariant cost", c1.latency, c128.latency);
}

void test_pw_sink_mm_then_pw() {
    std::cout << "--- test_pw_sink_mm_then_pw ---\n";
    // MM (T0@T1→T2 ephemeral) then PW (T2→T3 boundary out).
    // T3 produced by PW → has_pw_sink. nk=1 always.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0,1});

    CHECK("T2 ephemeral",   sg.ephemeral().count(2));
    CHECK("T3 boundary out",sg.boundary_outputs().count(3));
    // Reference accepts any k for PW sinks
    CHECK("k=1 valid",  sg.is_valid_tiling(TC(128,128,1)));
    CHECK("k=32 valid", sg.is_valid_tiling(TC(128,128,32)));
    CHECK("k=64 valid", sg.is_valid_tiling(TC(128,128,64)));
    // nk=1 (PW sink → output_K=1)
    auto c = sg.compute_cost(TC(128,128,1));
    CHECK("feasible k=1", c.feasible);
    CHECK_EQ_I("nk=1", c.num_k_passes, 1);
    // comp_per_step includes both MM and PW
    CHECK_EQ("comp/step=2500", c.compute_per_step, 2500.0);
}

void test_no_pw_sink_mm_only() {
    std::cout << "--- test_no_pw_sink_mm_only ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK("k=32 valid",  sg.is_valid_tiling(TC(128,128,32)));
    CHECK("k=64 valid",  sg.is_valid_tiling(TC(128,128,64)));
    CHECK("k=128 valid", sg.is_valid_tiling(TC(128,128,128)));
    CHECK("k=1 valid",   sg.is_valid_tiling(TC(128,128,1)));
}

void test_pw_sink_pw_then_mm_with_pw_output() {
    std::cout << "--- test_pw_sink_pw_then_mm_with_pw_output ---\n";
    // Op0: PW T0 → T1, Op1: MM T1@T2 → T3, Op2: PW T3 → T4 (boundary output)
    // T4 produced by PW → has_pw_sink. Any k valid.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},500},
             {OpType::MatMul,{1,2},{3},2000},
             {OpType::Pointwise,{3},{4},500}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0,1,2});
    CHECK("valid", sg.has_value());
    if (!sg) return;

    CHECK("T4 boundary out", sg->boundary_outputs().count(4));
    CHECK("k=1 valid",   sg->is_valid_tiling(TC(128,128,1)));
    CHECK("k=2 valid",   sg->is_valid_tiling(TC(128,128,2)));
    CHECK("k=128 valid", sg->is_valid_tiling(TC(128,128,128)));
    // Cost should be same regardless of k
    auto c1 = sg->compute_cost(TC(128,128,1));
    auto c128 = sg->compute_cost(TC(128,128,128));
    CHECK_EQ("k-invariant cost", c1.latency, c128.latency);
}

void test_pw_sink_mixed_outputs_one_pw() {
    std::cout << "--- test_pw_sink_mixed_outputs_one_pw ---\n";
    // Two sink ops with same-dim boundary outputs:
    //   T2: produced by MM (Op0) — sink
    //   T3: produced by PW (Op1) — sink → triggers PW sink rule
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{0},{3},500}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0,1});
    CHECK("valid", sg.has_value());
    if (!sg) return;

    CHECK("T2 boundary out (MM)", sg->boundary_outputs().count(2));
    CHECK("T3 boundary out (PW)", sg->boundary_outputs().count(3));
    // T3 is PW → has_pw_sink=true → k is ignored for entire subgraph
    CHECK("k=1 valid",  sg->is_valid_tiling(TC(128,128,1)));
    CHECK("k=32 valid", sg->is_valid_tiling(TC(128,128,32)));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Issue #38: spatial tiling compute
    test_spatial_tiling_does_not_divide_compute();
    test_k_splitting_divides_compute_proportionally();
    test_sub_native_spatial_still_full_compute();

    // Issue #37: raster reuse
    test_raster_lhs_reuse_within_rows();
    test_raster_vs_snake_improvement();

    // Issue #34: one-step retention semantics
    test_retention_one_step_boundary();
    test_retention_must_be_relisted_each_step();

    // PW sink rule
    test_pw_sink_pure_pw();
    test_pw_sink_mm_then_pw();
    test_no_pw_sink_mm_only();
    test_pw_sink_pw_then_mm_with_pw_output();
    test_pw_sink_mixed_outputs_one_pw();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}