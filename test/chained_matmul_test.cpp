// chained_matmul_test.cpp
// Tests cost model on chained MatMul subgraphs with hand-computed values.
// 
// Chain of 2: (T0 @ T1) @ T2 → T4    (= PROBLEM.md Example 5)
// Chain of 3: ((T0 @ T1) @ T3) @ T5 → T6
//
// Build: make chained_matmul_test
// Run:   ./chained_matmul_test

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "solution/solution.h"
#include <cmath>
#include <iostream>

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

static TileConfig TC(int64_t w, int64_t h, int64_t k, SnakeDir s = SnakeDir::None) {
    return {w, h, k, s};
}

static Subgraph make_sg(const Problem& p, const DAG& d, std::vector<size_t> ops) {
    auto sg = Subgraph::create(p, d, std::move(ops));
    if (!sg) { std::cerr << "FATAL: Subgraph::create failed\n"; std::exit(1); }
    return std::move(*sg);
}

// ============================================================================
// Chain of 2: (T0 @ T1) @ T2 = T4
// This is Example 5 from PROBLEM.md. We re-derive all values.
//
//   Op0: MatMul(T0, T1) → T3    K_0 = T0.width = 128
//   Op1: MatMul(T3, T2) → T4    K_1 = T3.width = 128
//
//   All tensors 128×128, B=10, capacity=45000
// ============================================================================

void test_chain2() {
    std::cout << "=== Chain of 2 MatMuls ===\n";

    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    //              T0        T1        T2        T3        T4
    p.ops = {{OpType::MatMul, {0,1}, {3}, 2000},    // Op0: T0 @ T1 → T3
             {OpType::MatMul, {3,2}, {4}, 2000}};   // Op1: T3 @ T2 → T4
    p.fast_memory_capacity = 45000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    // --- Boundary analysis ---
    auto sg = make_sg(p, d, {0, 1});
    CHECK("T3 ephemeral", sg.ephemeral().count(3));
    CHECK("T0 boundary in", sg.boundary_inputs().count(0));
    CHECK("T1 boundary in", sg.boundary_inputs().count(1));
    CHECK("T2 boundary in", sg.boundary_inputs().count(2));
    CHECK("T4 sink", sg.sink_tensor() == 4);
    CHECK_EQ_I("max_K=128", sg.max_K(), 128);

    // --- Working set at [128,128,32] ---
    // T0: LHS of Op0, h×K₀ = 128×128 = 16384
    // T1: RHS of Op0, k×w  = 32×128  = 4096
    // T2: RHS of Op1, k×w  = 32×128  = 4096
    // T4: accumulator, h×w = 128×128 = 16384
    // Total = 40960
    CHECK_EQ_I("ws [128,128,32]", sg.working_set(TC(128,128,32)), 40960);

    // --- Working set at [128,128,128] → OOM ---
    // T0: 16384, T1: 16384, T2: 16384, T4: 16384 = 65536 > 45000
    CHECK_EQ_I("ws [128,128,128]", sg.working_set(TC(128,128,128)), 65536);
    CHECK("OOM at k=128", !sg.is_feasible(TC(128,128,128)));

    // --- Cost at [128,128,32] ---
    auto c = sg.compute_cost(TC(128,128,32));
    CHECK("feasible", c.feasible);
    CHECK_EQ_I("tiles", c.num_spatial_tiles, 1);
    CHECK_EQ_I("k_passes", c.num_k_passes, 4);

    // Compute per step: Op0 = 2000×32/128 = 500, Op1 = 500. Total = 1000
    CHECK_EQ("comp/step", c.compute_per_step, 1000.0);

    // k=0: mem_in = T0(1638.4) + T1_slice(409.6) + T2_slice(409.6) = 2457.6
    //      mem_out = 0.  lat = max(1000, 2457.6) = 2457.6
    // k=1: mem_in = T1(409.6) + T2(409.6) = 819.2 (T0 resident)
    //      lat = max(1000, 819.2) = 1000
    // k=2: same → 1000
    // k=3: mem_in = 819.2, mem_out = T4(1638.4). lat = max(1000, 2457.6) = 2457.6
    // Total = 2457.6 + 1000 + 1000 + 2457.6 = 6915.2
    CHECK_EQ("latency", c.latency, 6915.2);

    // --- Also test at [128,128,64]: 2 k-passes ---
    auto c64 = sg.compute_cost(TC(128,128,64));
    CHECK_EQ_I("k64 ws", sg.working_set(TC(128,128,64)), 
               128*128 + 64*128 + 64*128 + 128*128);  // 16384+8192+8192+16384 = 49152
    // 49152 > 45000 → infeasible
    CHECK("k64 infeasible", !c64.feasible);
}

// ============================================================================
// Chain of 3: ((T0 @ T1) @ T3) @ T5 = T6
//
//   Op0: MatMul(T0, T1) → T2    K_0 = 128
//   Op1: MatMul(T2, T3) → T4    K_1 = 128
//   Op2: MatMul(T4, T5) → T6    K_2 = 128
//
//   All tensors 128×128, B=10, capacity=50000
// ============================================================================

void test_chain3() {
    std::cout << "\n=== Chain of 3 MatMuls ===\n";

    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
    //              T0        T1        T2        T3        T4        T5        T6
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},    // Op0: T0 @ T1 → T2
             {OpType::MatMul, {2,3}, {4}, 2000},    // Op1: T2 @ T3 → T4
             {OpType::MatMul, {4,5}, {6}, 2000}};   // Op2: T4 @ T5 → T6
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    // --- Boundary analysis ---
    std::cout << "  Boundary analysis:\n";
    auto sg = make_sg(p, d, {0, 1, 2});
    CHECK("T2 ephemeral", sg.ephemeral().count(2));
    CHECK("T4 ephemeral", sg.ephemeral().count(4));
    CHECK("T0 boundary in", sg.boundary_inputs().count(0));
    CHECK("T1 boundary in", sg.boundary_inputs().count(1));
    CHECK("T3 boundary in", sg.boundary_inputs().count(3));
    CHECK("T5 boundary in", sg.boundary_inputs().count(5));
    CHECK("T6 sink", sg.sink_tensor() == 6);
    CHECK("4 boundary inputs", sg.boundary_inputs().size() == 4);
    CHECK_EQ_I("max_K=128", sg.max_K(), 128);

    // --- Working set at [128,128,32] ---
    // T0: LHS of Op0, h×K₀ = 128×128 = 16384
    // T1: RHS of Op0, k×w  = 32×128  = 4096
    // T3: RHS of Op1, k×w  = 32×128  = 4096
    // T5: RHS of Op2, k×w  = 32×128  = 4096
    // T6: accumulator, h×w = 128×128 = 16384
    // Total = 16384 + 3×4096 + 16384 = 45056
    std::cout << "  Working set:\n";
    CHECK_EQ_I("ws [128,128,32]", sg.working_set(TC(128,128,32)), 45056);
    CHECK("feasible at k=32", sg.is_feasible(TC(128,128,32)));

    // At k=64: T1,T3,T5 each 64×128=8192. Total = 16384+3×8192+16384 = 57344
    CHECK_EQ_I("ws [128,128,64]", sg.working_set(TC(128,128,64)), 57344);
    CHECK("infeasible at k=64", !sg.is_feasible(TC(128,128,64)));

    // --- Cost at [128,128,32], null traversal ---
    std::cout << "  Cost at [128,128,32]:\n";
    auto c = sg.compute_cost(TC(128,128,32));
    CHECK("feasible", c.feasible);
    CHECK_EQ_I("tiles", c.num_spatial_tiles, 1);
    CHECK_EQ_I("k_passes", c.num_k_passes, 4);

    // Compute per step: 3 × (2000×32/128) = 3×500 = 1500
    CHECK_EQ("comp/step", c.compute_per_step, 1500.0);

    // k=0: mem_in = T0(1638.4) + T1(409.6) + T3(409.6) + T5(409.6) = 2867.2
    //      lat = max(1500, 2867.2) = 2867.2
    // k=1: mem_in = T1(409.6) + T3(409.6) + T5(409.6) = 1228.8
    //      lat = max(1500, 1228.8) = 1500
    // k=2: same → 1500
    // k=3: mem_in = 1228.8, mem_out = T6(1638.4)
    //      lat = max(1500, 1228.8 + 1638.4) = max(1500, 2867.2) = 2867.2
    // Total = 2867.2 + 1500 + 1500 + 2867.2 = 8734.4
    CHECK_EQ("latency", c.latency, 8734.4);

    // --- Verify via breakdown ---
    // First k-step is memory bound (loading T0 + 3 RHS slices)
    double B = 10.0;
    double T0_load = 128.0 * 128.0 / B;       // 1638.4
    double rhs_slice = 32.0 * 128.0 / B;      // 409.6
    double T6_evict = 128.0 * 128.0 / B;      // 1638.4
    double comp = 1500.0;

    double k0_mem = T0_load + 3 * rhs_slice;  // 2867.2
    double k0 = std::max(comp, k0_mem);        // 2867.2
    CHECK_EQ("k=0 lat", k0, 2867.2);

    double k_mid_mem = 3 * rhs_slice;          // 1228.8
    double k_mid = std::max(comp, k_mid_mem);  // 1500
    CHECK_EQ("k=1 lat", k_mid, 1500.0);

    double k_last_mem = 3 * rhs_slice + T6_evict;  // 2867.2
    double k_last = std::max(comp, k_last_mem);     // 2867.2
    CHECK_EQ("k=3 lat", k_last, 2867.2);

    double total_hand = k0 + 2 * k_mid + k_last;
    CHECK_EQ("hand total", total_hand, 8734.4);
    CHECK_EQ("matches compute_cost", c.latency, total_hand);
}

// ============================================================================
// Chain of 3, separate subgraphs (no fusion) — to verify fusion savings
// ============================================================================

void test_chain3_unfused() {
    std::cout << "\n=== Chain of 3 MatMuls — unfused ===\n";

    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::MatMul, {2,3}, {4}, 2000},
             {OpType::MatMul, {4,5}, {6}, 2000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    // Each op alone. Working set: h×K + k×w + h×w = 16384+k×128+16384
    // At k=32: 16384 + 4096 + 16384 = 36864 < 50000 ✓

    auto sg0 = make_sg(p, d, {0});
    auto sg1 = make_sg(p, d, {1});
    auto sg2 = make_sg(p, d, {2});

    auto c0 = sg0.compute_cost(TC(128,128,32));
    auto c1 = sg1.compute_cost(TC(128,128,32));
    auto c2 = sg2.compute_cost(TC(128,128,32));

    // Each op alone at [128,128,32], 4 k-passes:
    // comp per step = 2000 × 32/128 = 500
    // k=0: mem_in = LHS(1638.4) + RHS(409.6) = 2048. lat = max(500, 2048) = 2048
    // k=1,2: mem_in = RHS(409.6). lat = max(500, 409.6) = 500
    // k=3: mem_in = 409.6, mem_out = out(1638.4). lat = max(500, 2048) = 2048
    // Total per op = 2048 + 500 + 500 + 2048 = 5096
    CHECK_EQ("Op0 lat", c0.latency, 5096.0);
    CHECK_EQ("Op1 lat", c1.latency, 5096.0);
    CHECK_EQ("Op2 lat", c2.latency, 5096.0);
    CHECK_EQ("unfused total", c0.latency + c1.latency + c2.latency, 15288.0);

    // Build as Solution
    Solution sol(p, d, {
        {std::move(sg0), TC(128,128,32), {}},
        {std::move(sg1), TC(128,128,32), {}},
        {std::move(sg2), TC(128,128,32), {}},
    });
    CHECK("unfused valid", sol.validate().valid);
    CHECK_EQ("unfused sol total", sol.total_latency(), 15288.0);

    // Fused chain = 8734.4 (from test_chain3), saving = 43%
    std::cout << "  Fusion saves: " << 15288.0 << " → 8734.4 ("
              << (int)(100.0 * (15288.0 - 8734.4) / 15288.0) << "%)\n";
}

// ============================================================================
// Chain of 3 with spatial tiling (256×256 tensors, smaller tiles)
// ============================================================================

void test_chain3_tiled() {
    std::cout << "\n=== Chain of 3 MatMuls — 256×256 tiled ===\n";

    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::MatMul, {2,3}, {4}, 2000},
             {OpType::MatMul, {4,5}, {6}, 2000}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 20;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    auto sg = make_sg(p, d, {0, 1, 2});

    // All K values are 256 (LHS tensors are 256 wide)
    CHECK_EQ_I("K=256", sg.max_K(), 256);

    // At [128, 128, 32]:
    // T0: h×K = 128×256 = 32768
    // T1: k×w = 32×128 = 4096
    // T3: k×w = 4096
    // T5: k×w = 4096
    // T6: h×w = 128×128 = 16384
    // Total = 32768 + 3×4096 + 16384 = 61440 < 80000 ✓
    CHECK_EQ_I("ws [128,128,32]", sg.working_set(TC(128,128,32)), 61440);
    CHECK("feasible", sg.is_feasible(TC(128,128,32)));

    auto c = sg.compute_cost(TC(128,128,32));
    // Spatial tiles: (256/128)² = 4
    // k-passes: 256/32 = 8
    CHECK_EQ_I("tiles", c.num_spatial_tiles, 4);
    CHECK_EQ_I("k_passes", c.num_k_passes, 8);

    // Compute per step: 3 × (2000 × 32/256) = 3 × 250 = 750
    CHECK_EQ("comp/step", c.compute_per_step, 750.0);

    // Null traversal (no reuse): each of the 4 tiles loads T0 row strip fresh
    //
    // Per tile, 8 k-steps:
    //   k=0: T0(128×256/20=1638.4) + T1(32×128/20=204.8) × 3 = 1638.4+614.4 = 2252.8
    //         lat = max(750, 2252.8) = 2252.8
    //   k=1..6: 3×204.8 = 614.4. lat = max(750, 614.4) = 750
    //   k=7: 614.4 + T6_evict(128×128/20=819.2) = 1433.6. lat = max(750, 1433.6) = 1433.6
    // Per tile total = 2252.8 + 6×750 + 1433.6 = 8186.4
    // 4 tiles × 8186.4 = 32745.6

    double B = 20.0;
    double t0_load = 128.0 * 256.0 / B;   // 1638.4
    double rhs_slice = 32.0 * 128.0 / B;  // 204.8
    double t6_evict = 128.0 * 128.0 / B;  // 819.2
    double comp = 750.0;

    double k0 = std::max(comp, t0_load + 3 * rhs_slice);          // max(750, 2252.8)
    double k_mid = std::max(comp, 3 * rhs_slice);                  // max(750, 614.4)
    double k_last = std::max(comp, 3 * rhs_slice + t6_evict);     // max(750, 1433.6)
    double per_tile = k0 + 6 * k_mid + k_last;
    double total_null = 4 * per_tile;

    CHECK_EQ("k0 lat", k0, 2252.8);
    CHECK_EQ("k_mid lat", k_mid, 750.0);
    CHECK_EQ("k_last lat", k_last, 1433.6);
    CHECK_EQ("per tile", per_tile, 8186.4);
    CHECK_EQ("null total", c.latency, total_null);

    // Snake traversal: 2×2 grid
    // Row snake: [0,1,3,2]
    //   Tile 0 (r=0,c=0): FF → k0 + 6×k_mid + k_last = 8186.4
    //   Tile 1 (r=0,c=1): RF (reuse LHS) → no T0 load
    //     k=0: 3×204.8 = 614.4. lat = max(750, 614.4) = 750
    //     k=1..6: 750
    //     k=7: 614.4 + 819.2 = 1433.6. lat = 1433.6
    //     Total: 750 + 6×750 + 1433.6 = 6683.6
    //   Tile 2 (r=1,c=1): FR (reuse RHS) → T0 loaded, no T1/T3/T5 change at k=0
    //     Wait, FR means fresh LHS, reuse RHS.
    //     k=0: T0 load(1638.4) + 0 (RHS reused) + PW=0 = 1638.4. lat=max(750,1638.4)=1638.4
    //     k=1..6: 3×204.8=614.4. lat = 750
    //     k=7: 614.4+819.2=1433.6
    //     Total: 1638.4 + 6×750 + 1433.6 = 7572
    //   Tile 3 (r=1,c=0): RF (reuse LHS)
    //     Same as tile 1: 6683.6
    //
    // Snake total = 8186.4 + 6683.6 + 7572.0 + 6683.6 = 29125.6

    auto c_snake = sg.compute_cost(TC(128,128,32, SnakeDir::RowMajor));

    double rf_k0 = std::max(comp, 3 * rhs_slice);  // 750 (no T0 load)
    double rf_per_tile = rf_k0 + 6 * k_mid + k_last;  // 750+4500+1433.6 = 6683.6

    double fr_k0 = std::max(comp, t0_load);  // 1638.4 (no RHS load at k=0)
    double fr_per_tile = fr_k0 + 6 * k_mid + k_last;  // 1638.4+4500+1433.6 = 7572

    double total_snake = per_tile + rf_per_tile + fr_per_tile + rf_per_tile;
    // = 8186.4 + 6683.6 + 7572 + 6683.6 = 29125.6

    CHECK_EQ("RF tile", rf_per_tile, 6683.6);
    CHECK_EQ("FR tile", fr_per_tile, 7572.0);
    CHECK_EQ("snake total hand", total_snake, 29125.6);
    CHECK_EQ("snake total code", c_snake.latency, total_snake);

    // Snake is cheaper than null
    CHECK("snake < null", c_snake.latency < c.latency);
    std::cout << "  Null=" << c.latency << " Snake=" << c_snake.latency
              << " (saved " << (int)(100*(c.latency - c_snake.latency)/c.latency) << "%)\n";
}

// ============================================================================
// Chain of 2 as full Solution (with validation)
// ============================================================================

void test_chain2_solution() {
    std::cout << "\n=== Chain of 2 as Solution ===\n";

    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {3}, 2000},
             {OpType::MatMul, {3,2}, {4}, 2000}};
    p.fast_memory_capacity = 45000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    // Single fused subgraph
    auto sg = make_sg(p, d, {0, 1});
    Solution sol(p, d, {{std::move(sg), TC(128,128,32), {}}});
    CHECK("valid", sol.validate().valid);
    CHECK_EQ("total", sol.total_latency(), 6915.2);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    test_chain2();
    test_chain3();
    test_chain3_unfused();
    test_chain3_tiled();
    test_chain2_solution();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}
