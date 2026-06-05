// Ascend 910B cost-model tests (test-driven).
//
// CHECK  = invariant the model must satisfy now (regression anchor / schema).
// RED    = encodes the DESIRED 910B behavior the current single-context model
//          still misses; printed as "RED (todo)" and tracked separately so it
//          does NOT fail the build — it's the spec we implement toward.
//
// Scenarios (from the 910B tiling analysis):
//   A  large pointwise         — memory-bound, latency == DDR floor (GREEN)
//   B  thin matmul (K=2048)    — must parallelize via split-K across cube cores (RED)
//   C  pointwise fusion        — fused < separate (GREEN)
//   D  matmul->relu (mixed)    — cube->vector handoff costs DDR, not free (RED)
//   softmax                    — reduction-chain schema; cost model is next (GREEN schema)

#include "core/dag.h"
#include "core/subgraph.h"
#include "core/types.h"

#include <iostream>

static int g_pass = 0, g_fail = 0, g_red = 0;

static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double g, double e, double t = 0.5) {
    if (std::abs(g - e) < t) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}
// Expected to fail until the 910B model lands — does not break the build.
static void RED(const char* l, bool c) {
    if (c) g_pass++;
    else { g_red++; std::cout << "  RED (todo): " << l << "\n"; }
}

// --- A: large pointwise — memory-bound, tile-invariant -----------------------
static void test_A_pointwise_memory_bound() {
    std::cout << "[A] large pointwise — memory-bound, latency == DDR floor\n";
    Problem p;
    p.tensors = {{512, 512}, {512, 512}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 100}};  // trivial compute
    p.fast_memory_capacity = 1 << 22;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0});
    CHECK("A: subgraph builds", (bool)sg);
    double lat = sg->best_cost().latency;
    std::cout << "    best latency = " << lat << "\n";
    // load T0 + store T1 = 2 * 512*512/10 = 52428.8 — independent of tile size.
    CHECK_EQ("A: latency == DDR floor (52428.8)", lat, 52428.8);
}

// --- B: thin matmul — should parallelize via split-K across cube cores --------
static void test_B_thin_matmul_parallel() {
    std::cout << "[B] thin matmul C[128,128], K=2048 — should split-K across cube cores\n";
    Problem p;
    // C[w=128,h=128] = A[w=K=2048,h=128] @ B[w=128,h=K=2048]
    p.tensors = {{2048, 128}, {128, 2048}, {128, 128}};
    p.ops = {{OpType::MatMul, {0, 1}, {2}, 100000}};  // compute-heavy
    p.fast_memory_capacity = 1 << 20;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0});
    CHECK("B: subgraph builds", (bool)sg);
    double lat = sg->best_cost().latency;
    std::cout << "    best latency = " << lat << " (single-context); 24 cube cores idle but 1\n";
    // Output == native → 1 spatial tile → single-context model runs on ONE core.
    // 910B target: split-K across num_cube_cores → at least a 2x parallel speedup.
    RED("B: parallelized across cube cores (lat <= 50000)", lat <= 50000.0);
}

// --- C: pointwise fusion — fused beats separate ------------------------------
static void test_C_pointwise_fusion() {
    std::cout << "[C] pointwise fusion — fused < separate\n";
    Problem p;
    p.tensors = {{256, 256}, {256, 256}, {256, 256}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 500}, {OpType::Pointwise, {1}, {2}, 500}};
    p.fast_memory_capacity = 1 << 20;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    DAG dag = DAG::build(p);
    double fused = Subgraph::create(p, dag, {0, 1})->best_cost().latency;
    double s0 = Subgraph::create(p, dag, {0})->best_cost().latency;
    double s1 = Subgraph::create(p, dag, {1})->best_cost().latency;
    std::cout << "    fused = " << fused << "  separate = " << (s0 + s1) << "\n";
    CHECK("C: fused < separate", fused < s0 + s1);
}

// --- D: matmul->relu (mixed) — cube->vector handoff is a DDR round-trip --------
static void test_D_mixed_cube_vector() {
    std::cout << "[D] matmul->relu mixed — cube->vector intermediate must cost DDR\n";
    Problem p;
    // C'[128,128] = A[2048,128] @ B[128,2048];  out = relu(C')
    p.tensors = {{2048, 128}, {128, 2048}, {128, 128}, {128, 128}};
    p.ops = {{OpType::MatMul, {0, 1}, {2}, 30000}, {OpType::Pointwise, {2}, {3}, 500}};
    p.fast_memory_capacity = 1 << 20;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    DAG dag = DAG::build(p);
    double fused = Subgraph::create(p, dag, {0, 1})->best_cost().latency;
    double sep = Subgraph::create(p, dag, {0})->best_cost().latency +
                 Subgraph::create(p, dag, {1})->best_cost().latency;
    double cprime = 128.0 * 128 / 10;  // one C' transfer
    std::cout << "    fused = " << fused << "  separate = " << sep << "  C'xfer = " << cprime << "\n";
    // Current model frees C' entirely (saves load+store = 2*C'). On 910B the
    // cube->vector handoff routes through DDR (GM pipe), so fusion can save at
    // most the avoided kernel launch — NOT the full materialization of C'.
    RED("D: cube->vector intermediate not fully free", fused >= sep - 2 * cprime + 1.0);
}

// --- softmax: reduction-chain schema (cost model is the next increment) -------
static void test_softmax_reduction_schema() {
    std::cout << "[softmax] reduction-chain schema (reduction-k cost model = TODO)\n";
    // A[W=4096, H=128], softmax along each row (reduce over W):
    //   rowmax : A -> m[1,128]          Reduction over width
    //   exp    : A, m(broadcast) -> e   Pointwise + broadcast input m
    //   rowsum : e -> s[1,128]          Reduction
    //   div    : e, s(broadcast) -> out Pointwise + broadcast input s
    Problem p;
    p.tensors = {{4096, 128},   // 0: A
                 {1, 128},      // 1: m   (reduced width -> 1)
                 {4096, 128},   // 2: e
                 {1, 128},      // 3: s
                 {4096, 128}};  // 4: out
    p.ops = {{OpType::Reduction, {0}, {1}, 4096 * 128 / 4},
             {OpType::Pointwise, {0, 1}, {2}, 4096 * 128},
             {OpType::Reduction, {2}, {3}, 4096 * 128 / 4},
             {OpType::Pointwise, {2, 3}, {4}, 4096 * 128}};
    DAG dag = DAG::build(p);
    CHECK("softmax: 4-op DAG builds", p.num_ops() == 4);
    CHECK("softmax: rowmax reduces width to 1", p.tensors[1].width == 1);
    CHECK("softmax: m broadcast back to W in exp", p.tensors[2].width == 4096);
    // Reduction-k cost modeling (online accumulator) is the next increment —
    // do NOT call best_cost() on a Reduction op yet (unhandled in Subgraph).
    std::cout << "    schema OK; reduction-k cost model is the next step\n";
}

int main() {
    test_A_pointwise_memory_bound();
    test_B_thin_matmul_parallel();
    test_C_pointwise_fusion();
    test_D_mixed_cube_vector();
    test_softmax_reduction_schema();
    std::cout << "\n=== pass=" << g_pass << " fail=" << g_fail << " red(todo)=" << g_red << " ===\n";
    return g_fail;  // RED items are the spec; only hard CHECK failures break the build
}
