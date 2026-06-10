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

#include <cmath>
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
// Activate the 910B parallel model (one die): 24 cube + 48 vector cores,
// L0c accumulator = 128 KB/core (the cube tile bound, replacing the native cap).
static void set_910b(Problem& p) {
    p.num_cube_cores = 24;
    p.num_vector_cores = 48;
    p.cube_capacity = 128 * 1024;
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
    set_910b(p);
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
    p.ops = {{OpType::MatMul, {0, 1}, {2}, 2400000}};  // compute-heavy (2.4M = 24*100k)
    p.fast_memory_capacity = 1 << 20;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0});
    CHECK("B: subgraph builds", (bool)sg);
    double lat = sg->best_cost().latency;
    std::cout << "    best latency = " << lat << " (910B); single-context (1 core) would be ~2.4M\n";
    // Output == native → 1 spatial tile. Split-K across the 24 cube cores
    // (config enumeration picks a small k => nk independent work units) divides
    // the 2.4M compute by ~24 -> ~100000, vs 2.4M on a single core.
    CHECK("B: split-K parallelizes across cube cores (lat <= 200000)", lat <= 200000.0);
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
    set_910b(p);
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
    set_910b(p);
    DAG dag = DAG::build(p);
    // 910B: matmul (cube) + relu (vector) can't share a subgraph — the unit-
    // homogeneity constraint rejects the mixed group, so the cube->vector edge
    // is never an intra-group "free" ephemeral. The per-unit singletons stay
    // valid (each runs on its own core pool, pipelined at the kernel level).
    auto mixed = Subgraph::create(p, dag, {0, 1});
    std::cout << "    mixed group accepted? " << (mixed.has_value() ? "yes" : "no (rejected)") << "\n";
    CHECK("D: mixed cube+vector group is rejected", !mixed.has_value());
    CHECK("D: matmul-only subgraph valid", Subgraph::create(p, dag, {0}).has_value());
    CHECK("D: relu-only subgraph valid", Subgraph::create(p, dag, {1}).has_value());
}

// --- softmax: reduction-chain schema (cost model is the next increment) -------
static void test_softmax_reduction_schema() {
    std::cout << "[softmax] reduction-chain — fused cost + fusion benefit\n";
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
    p.fast_memory_capacity = 1 << 24;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    CHECK("softmax: rowmax reduces width to 1", p.tensors[1].width == 1);
    CHECK("softmax: m broadcast back to W in exp", p.tensors[2].width == 4096);
    DAG dag = DAG::build(p);
    // Fused: one vector subgraph — A in, out out; m/e/s are ephemeral (the whole
    // softmax stays on-chip, only the [H,W] input/output touch DDR).
    auto fused_sg = Subgraph::create(p, dag, {0, 1, 2, 3});
    CHECK("softmax: fused vector subgraph builds", (bool)fused_sg);
    double fused = fused_sg->best_cost().latency;
    CHECK("softmax: fused cost is finite (tiling feasible)", std::isfinite(fused));
    // Split into the two natural halves {rowmax,exp} + {rowsum,div}: now the
    // intermediate e[H,W] is a boundary tensor — written by the first group and
    // re-read by the second (an extra [H,W] DDR round-trip). Fusing all four
    // keeps e ephemeral, eliminating it. (Per-op split is degenerate here: a
    // broadcast [1,H] input alone is infeasible, so we compare feasible halves.)
    double half1 = Subgraph::create(p, dag, {0, 1})->best_cost().latency;
    double half2 = Subgraph::create(p, dag, {2, 3})->best_cost().latency;
    std::cout << "    fused=" << fused << "  split halves=" << (half1 + half2)
              << " (" << half1 << "+" << half2 << ")\n";
    CHECK("softmax: fused < split halves (ephemeral e avoids a DDR round-trip)",
          fused < half1 + half2);
}

// --- R: few-row reduction — split the reduced axis across vector cores --------
// Reduction analog of split-K (test F): only the non-reduced dim (rows) gives
// spatial parallelism, so a few-row reduction can't fill the 48 vector cores
// spatially. Splitting the reduced axis across cores fills them, paying only a
// thin per-partial merge (out_H_ here) — the flash-style parallel reduction.
static void test_R_reduction_split() {
    std::cout << "[R] few-row reduction — split reduced axis across vector cores\n";
    Problem p;
    p.tensors = {{4096, 4}, {1, 4}};                       // A[4 rows x 4096], rowmax -> m[1,4]
    p.ops = {{OpType::Reduction, {0}, {1}, 4096LL * 4 * 1000}};  // compute-heavy
    p.fast_memory_capacity = 1 << 24;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    Problem p1 = p;
    p1.num_cube_cores = 1;
    p1.num_vector_cores = 1;  // single-context baseline (no parallel override)
    DAG dag = DAG::build(p);
    double par = Subgraph::create(p, dag, {0})->best_cost().latency;
    DAG dag1 = DAG::build(p1);
    double single = Subgraph::create(p1, dag1, {0})->best_cost().latency;
    std::cout << "    48 cores (split reduced)=" << par << "  1 core=" << single << "\n";
    CHECK("R: 4 rows but split-reduced fills cores (large speedup vs 1 core)",
          par < single / 5.0);
}

// --- E: matmul tile shape — balanced (2D) reloads less than skewed (1D) ------
static void test_E_matmul_2d_vs_1d() {
    std::cout << "[E] matmul tile shape — balanced (2D) reloads less than skewed (1D)\n";
    Problem p;
    p.tensors = {{512, 512}, {512, 512}, {512, 512}};  // A, B, C : M=N=K=512
    p.ops = {{OpType::MatMul, {0, 1}, {2}, 100}};       // tiny compute -> memory-bound
    p.fast_memory_capacity = 1 << 24;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0});
    CHECK("E: subgraph builds", (bool)sg);
    double balanced = sg->compute_cost(TileConfig{256, 128, 512}).latency;  // w·h=32K, ratio 2:1
    double skewed = sg->compute_cost(TileConfig{512, 64, 512}).latency;     // w·h=32K, ratio 8:1
    std::cout << "    balanced[w256,h128]=" << balanced << "  skewed[w512,h64]=" << skewed << "\n";
    CHECK("E: balanced (2D) tile reloads less than skewed (1D)", balanced < skewed);
}

// --- F: split-K parallelizes a thin matmul (beats single-core) ---------------
static void test_F_split_k() {
    std::cout << "[F] thin matmul — split-K across cube cores beats single-core\n";
    Problem p;
    p.tensors = {{2048, 128}, {128, 2048}, {128, 128}};  // C[128,128], K=2048 (1 native tile)
    p.ops = {{OpType::MatMul, {0, 1}, {2}, 2400000}};     // compute-heavy
    p.fast_memory_capacity = 1 << 24;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    Problem p1 = p;
    p1.num_cube_cores = 1;
    p1.num_vector_cores = 1;  // single-context baseline
    DAG dag = DAG::build(p);
    double par = Subgraph::create(p, dag, {0})->best_cost().latency;
    DAG dag1 = DAG::build(p1);
    double single = Subgraph::create(p1, dag1, {0})->best_cost().latency;
    std::cout << "    24 cores (split-K)=" << par << "  1 core=" << single << "\n";
    CHECK("F: split-K gives a large speedup vs single core", par < single / 5.0);
}

// --- G: cube tile can exceed native 128 (L0c bound, not the native cap) -------
static void test_G_super_native_tile() {
    std::cout << "[G] matmul tile can be >128 (L0c bound, not the native cap)\n";
    Problem p;
    p.tensors = {{512, 512}, {512, 512}, {512, 512}};  // M=N=K=512
    p.ops = {{OpType::MatMul, {0, 1}, {2}, 100}};       // memory-bound -> prefers big balanced tile
    p.fast_memory_capacity = 1 << 24;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    DAG dag = DAG::build(p);
    auto best = Subgraph::create(p, dag, {0})->best_cost();
    std::cout << "    best tile = " << best.config.w << "x" << best.config.h << "\n";
    CHECK("G: best tile exceeds native 128 (super-native, L0c-bounded)",
          best.config.w > 128 || best.config.h > 128);
}

// --- H: split-K (big tile) beats spatial-fill (small tiles) on RELOAD --------
// F shows split-K beats a single core; H shows *why it is chosen over the
// spatial alternative*: filling the cores spatially needs tiny tiles, and small
// tiles reload far more input. With the op_scale fix the matmul compute is
// invariant to the tile (16-aligned => no fractal padding), so the gap below is
// pure reload — split-K wins for the right reason, not a native-128 artifact.
static void test_H_split_k_beats_spatial() {
    std::cout << "[H] thin matmul — split-K on a big tile beats small-tile spatial-fill\n";
    Problem p;
    p.tensors = {{2048, 128}, {128, 2048}, {128, 128}};  // C[128,128], K=2048
    p.ops = {{OpType::MatMul, {0, 1}, {2}, 2400000}};     // compute-heavy
    p.fast_memory_capacity = 1 << 24;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0});
    CHECK("H: subgraph builds", (bool)sg);
    // Big [128,128] = 1 spatial tile -> model splits K across the 24 cube cores.
    // Small [16,16] = 64 spatial tiles -> fills cores spatially, but each tile
    // reloads its A row-strip + B col-strip => 8x the DDR traffic of the big tile.
    double big_splitk = sg->compute_cost(TileConfig{128, 128, 2048}).latency;
    double small_spatial = sg->compute_cost(TileConfig{16, 16, 2048}).latency;
    std::cout << "    big[128x128]+split-K=" << big_splitk
              << "  small[16x16] spatial=" << small_spatial << "\n";
    CHECK("H: split-K (big tile) beats spatial-fill (small tiles) on reload",
          big_splitk < small_spatial);
}

int main() {
    test_A_pointwise_memory_bound();
    test_B_thin_matmul_parallel();
    test_C_pointwise_fusion();
    test_D_mixed_cube_vector();
    test_softmax_reduction_schema();
    test_R_reduction_split();
    test_E_matmul_2d_vs_1d();
    test_F_split_k();
    test_G_super_native_tile();
    test_H_split_k_beats_spatial();
    std::cout << "\n=== pass=" << g_pass << " fail=" << g_fail << " red(todo)=" << g_red << " ===\n";
    return g_fail;  // RED items are the spec; only hard CHECK failures break the build
}
