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
    CHECK("T4 is boundary output", sg.boundary_outputs().count(4));
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
    p.fast_memory_capacity = 60000;   // increased: WS=57344 needs >50000
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
    CHECK("T6 is boundary output", sg.boundary_outputs().count(6));
    CHECK("4 boundary inputs", sg.boundary_inputs().size() == 4);
    CHECK_EQ_I("max_K=128", sg.max_K(), 128);

    // --- Working set at [128,128,32] (nk=4, ntw=1, nth=1) ---
    // Tiling propagation (rev topo: Op2 sink, Op1, Op0):
    //   T6: NTW×NTH → 16384
    //   T4 (sink LHS): NK×NTH → 32×128 = 4096
    //   T5 (sink RHS): NTW×NK → 128×32 = 4096
    //   T2 (non-sink LHS of Op1): FIXED_1×NTH → 128×128 = 16384
    //   T3 (non-sink RHS of Op1): NK×FIXED_1 → 32×128 = 4096
    //     (inherits h=NK from T4's h_source)
    //   T0 (non-sink LHS of Op0): FIXED_1×NTH → 128×128 = 16384
    //   T1 (non-sink RHS of Op0): FIXED_1×FIXED_1 → 128×128 = 16384 (!)
    //     (inherits h=FIXED_1 from T2's h_source which is FIXED_1)
    // WS = T0(16384) + T1(16384) + T3(4096) + T5(4096) + T6(16384) = 57344
    std::cout << "  Working set:\n";
    CHECK_EQ_I("ws [128,128,32]", sg.working_set(TC(128,128,32)), 57344);
    CHECK("feasible at k=32", sg.is_feasible(TC(128,128,32)));

    // At k=64 (nk=2): T3=64×128=8192, T5=128×64=8192. T0,T1,T6 still 16384 each.
    // Total = 16384+16384+8192+8192+16384 = 65536 > 60000
    CHECK_EQ_I("ws [128,128,64]", sg.working_set(TC(128,128,64)), 65536);
    CHECK("infeasible at k=64", !sg.is_feasible(TC(128,128,64)));

    // --- Cost at [128,128,32] ---
    std::cout << "  Cost at [128,128,32]:\n";
    auto c = sg.compute_cost(TC(128,128,32));
    CHECK("feasible", c.feasible);
    CHECK_EQ_I("tiles", c.num_spatial_tiles, 1);
    CHECK_EQ_I("k_passes", c.num_k_passes, 4);

    // Compute per step: 3 × (2000/4) = 1500
    CHECK_EQ("comp/step", c.compute_per_step, 1500.0);

    // IO (ntw=1, nth=1, 1 spatial tile):
    //   once_load = T1(1638.4)
    //   col_load  = T0(1638.4)  (FIXED_1 in h → col_load)
    //   stream    = T3(409.6) + T5(409.6) = 819.2
    //   evict     = T6(1638.4)
    // d=0: max(1500, once+col+stream) = max(1500, 4096.0) = 4096.0
    // d=1,2: max(1500, stream) = 1500
    // d=3: max(1500, stream+evict) = max(1500, 2457.6) = 2457.6
    // Total = 4096 + 2×1500 + 2457.6 = 9553.6
    CHECK_EQ("latency", c.latency, 9553.6);

    // --- Verify via breakdown ---
    double B = 10.0;
    double once = 128.0 * 128.0 / B;         // T1: 1638.4
    double col = 128.0 * 128.0 / B;          // T0: 1638.4
    double stream = 2.0 * 32.0 * 128.0 / B;  // T3+T5: 819.2
    double evict = 128.0 * 128.0 / B;        // T6: 1638.4
    double comp = 1500.0;

    double k0 = std::max(comp, once + col + stream);      // 4096
    double k_mid = std::max(comp, stream);                  // 1500
    double k_last = std::max(comp, stream + evict);        // 2457.6
    CHECK_EQ("k=0 lat", k0, 4096.0);
    CHECK_EQ("k=mid lat", k_mid, 1500.0);
    CHECK_EQ("k=3 lat", k_last, 2457.6);

    double total_hand = k0 + 2 * k_mid + k_last;
    CHECK_EQ("hand total", total_hand, 9553.6);
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
    p.fast_memory_capacity = 60000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    // Each op alone. At [128,128,32] nk=4, ntw=1, nth=1:
    // Both LHS and RHS are stream_load (FROM_NK). Slices = 32×128 = 4096 each.
    // stream = 819.2. evict = 1638.4. comp = 500.
    // d=0: max(500, 819.2)=819.2. d=1,2: 819.2. d=3: max(500, 2457.6)=2457.6.
    // total = 819.2 + 2×819.2 + 2457.6 = 4915.2

    auto sg0 = make_sg(p, d, {0});
    auto sg1 = make_sg(p, d, {1});
    auto sg2 = make_sg(p, d, {2});

    auto c0 = sg0.compute_cost(TC(128,128,32));
    auto c1 = sg1.compute_cost(TC(128,128,32));
    auto c2 = sg2.compute_cost(TC(128,128,32));

    CHECK_EQ("Op0 lat", c0.latency, 4915.2);
    CHECK_EQ("Op1 lat", c1.latency, 4915.2);
    CHECK_EQ("Op2 lat", c2.latency, 4915.2);
    CHECK_EQ("unfused total", c0.latency + c1.latency + c2.latency, 14745.6);

    Solution sol(p, d, {
        {std::move(sg0), TC(128,128,32), {}},
        {std::move(sg1), TC(128,128,32), {}},
        {std::move(sg2), TC(128,128,32), {}},
    });
    CHECK("unfused valid", sol.validate().valid);
    CHECK_EQ("unfused sol total", sol.total_latency(), 14745.6);

    // Fused chain = 9553.6 (from test_chain3), saving ≈ 35%
    std::cout << "  Fusion saves: " << 14745.6 << " → 9553.6 ("
              << (int)(100.0 * (14745.6 - 9553.6) / 14745.6) << "%)\n";
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
    CHECK_EQ("total=6915.2", sol.total_latency(), 6915.2);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    test_chain2();
    test_chain3();
    test_chain3_unfused();
    test_chain2_solution();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}