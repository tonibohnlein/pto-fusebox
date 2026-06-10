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
#include <cstdio>
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
    // Each op as its own subgraph: the intermediates m/e/s all become boundary
    // tensors that round-trip DDR. Fusing all four keeps them ephemeral. (The
    // broadcast [1,H] inputs m/s now cost finitely as boundary inputs too — the
    // role inference treats a broadcast axis as FIXED_1, not tiled.)
    double separate = 0.0;
    for (size_t i = 0; i < 4; i++)
        separate += Subgraph::create(p, dag, {i})->best_cost().latency;
    std::cout << "    fused=" << fused << "  separate(sum of 4 ops)=" << separate << "\n";
    CHECK("softmax: fused << separate (ephemeral m/e/s avoid DDR round-trips)",
          fused < separate);
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

// Single matmul C[N,M] = A[K,M] @ B[N,K]. base_cost = per-native-tile FLOPs.
static Problem mk_mm(int64_t M, int64_t N, int64_t K, int64_t cost) {
    Problem p;
    p.tensors = {{K, M}, {N, K}, {N, M}};
    p.ops = {{OpType::MatMul, {0, 1}, {2}, cost}};
    p.fast_memory_capacity = 1 << 26;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    return p;
}

// Print the solver's chosen solution: tile [w x h], #spatial tiles, the
// parallel split (matmul split-K S / reduction split; 1 = pure spatial),
// effective cores, and whether compute- or DDR-bound.
static void viz_row(const char* name, const Problem& p, std::vector<size_t> ops) {
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, ops);
    printf("  %-22s ", name);
    if (!sg) { printf("NULLOPT\n"); return; }
    auto r = sg->best_cost();
    printf("out[%4lldx%-4lld] K=%-5lld tile[%4lldx%-4lld] sp=%-3d split=%-2d cores=%-2d %-9s lat=%.0f\n",
           (long long)sg->output_width(), (long long)sg->output_height(), (long long)sg->max_K(),
           (long long)r.config.w, (long long)r.config.h, r.num_spatial_tiles,
           r.parallel_split, r.cores_used, r.compute_bound ? "[compute]" : "[DDR]", r.latency);
}

// Build a chained two-matmul C=(A@B)@D and an independent pair Q=X@Wq,K=X@Wk.
static Problem mk_chained(int64_t M, int64_t K1, int64_t N1, int64_t N2, int64_t c1, int64_t c2) {
    Problem p;  // A[K1,M] B[N1,K1] -> T[N1,M]; T[N1,M] D[N2,N1] -> C[N2,M]
    p.tensors = {{K1, M}, {N1, K1}, {N1, M}, {N2, N1}, {N2, M}};
    p.ops = {{OpType::MatMul, {0, 1}, {2}, c1}, {OpType::MatMul, {2, 3}, {4}, c2}};
    p.fast_memory_capacity = 1 << 26;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    return p;
}

// --- two matmuls: chained fusion compute is sane; split-K is per-matmul -------
static void test_two_matmul() {
    std::cout << "[2MM] two-matmul subgraph: chained fusion compute + per-matmul split-K\n";
    // Chained, compute-heavy. base_cost = native^2 * K (per-native-tile FLOPs).
    auto p = mk_chained(128, 256, 128, 64, 16384 * 256, 16384 * 128);
    DAG dag = DAG::build(p);
    double m1 = Subgraph::create(p, dag, {0})->best_cost().latency;
    double m2 = Subgraph::create(p, dag, {1})->best_cost().latency;
    double fused = Subgraph::create(p, dag, {0, 1})->best_cost().latency;
    std::cout << "    M1=" << m1 << " M2=" << m2 << " fused=" << fused << "\n";
    // Fused does BOTH matmuls' work — must not undercut either part (the old bug
    // scaled all compute by the narrower sink output, so fused < M1 alone).
    CHECK("2MM: chained fused >= each part (compute counted per-matmul)",
          fused >= m1 - 0.5 && fused >= m2 - 0.5);
    // Compute-bound: fused latency == sum of the two compute loads on shared cores
    // (fusing two matmuls only helps LATENCY when DDR-bound — by keeping the
    // intermediate T on-chip; under compute saturation the cores still do M1+M2).
    CHECK("2MM: chained fused == M1+M2 when compute-bound", std::abs(fused - (m1 + m2)) < 1.0);

    // (2) INDEPENDENT shared-input: Q=X@Wq, K=X@Wk (same X). Both cube, no
    // intermediate between them — fusing only saves re-reading X (negligible
    // vs compute here), so fused == sum.
    {
        Problem q;  // 0X[Kc,M] 1Wq[N,Kc] 2Q[N,M] 3Wk[N,Kc] 4K[N,M]
        int64_t M = 128, Kc = 256, N = 128;
        q.tensors = {{Kc, M}, {N, Kc}, {N, M}, {N, Kc}, {N, M}};
        q.ops = {{OpType::MatMul, {0, 1}, {2}, 16384 * Kc}, {OpType::MatMul, {0, 3}, {4}, 16384 * Kc}};
        q.fast_memory_capacity = 1 << 26; q.slow_memory_bandwidth = 10;
        q.native_w = 128; q.native_h = 128; set_910b(q);
        DAG dq = DAG::build(q);
        double a = Subgraph::create(q, dq, {0})->best_cost().latency;
        double b = Subgraph::create(q, dq, {1})->best_cost().latency;
        double f = Subgraph::create(q, dq, {0, 1})->best_cost().latency;
        CHECK("2MM: independent shared-input fused == sum (compute-bound)", std::abs(f - (a + b)) < 1.0);
    }

    // (3) ACCUMULATE C=(A@B)+(E@F): graph-level split-K — two matmuls summed by
    // a vector add. The add is a DIFFERENT unit (vector), so it CANNOT fuse with
    // the cube matmuls: {mm,mm,add} is rejected; the two cube matmuls alone are
    // a valid group. This is the manual parallel-K-split, with the add as merge.
    {
        Problem a;  // 0A 1B 2P=A@B  3E 4F 5Q=E@F  6C=P+Q
        int64_t M = 128, K = 256, N = 128;
        a.tensors = {{K, M}, {N, K}, {N, M}, {K, M}, {N, K}, {N, M}, {N, M}};
        a.ops = {{OpType::MatMul, {0, 1}, {2}, 16384 * K},
                 {OpType::MatMul, {3, 4}, {5}, 16384 * K},
                 {OpType::Pointwise, {2, 5}, {6}, N * M}};
        a.fast_memory_capacity = 1 << 26; a.slow_memory_bandwidth = 10;
        a.native_w = 128; a.native_h = 128; set_910b(a);
        DAG da = DAG::build(a);
        CHECK("2MM: accumulate {mm,mm,add} rejected (cube+vector not homogeneous)",
              !Subgraph::create(a, da, {0, 1, 2}).has_value());
        CHECK("2MM: accumulate {mm,add} rejected (cube+vector)",
              !Subgraph::create(a, da, {0, 2}).has_value());
        CHECK("2MM: accumulate two cube matmuls {mm,mm} is a valid group",
              Subgraph::create(a, da, {0, 1}).has_value());
        CHECK("2MM: accumulate add alone is a valid vector group",
              Subgraph::create(a, da, {2}).has_value());
    }
}

// --- compare vs a hand-crafted pypto tiling (test_auto_tile_matmul_l0.py) -----
// That pypto test fixes the L0-LEVEL tiling for a thin matmul: output tile
// [16,64] in the Acc/L0c, the K=2048 contraction tiled by the 16-fractal, run
// SINGLE-core sequentially (tile.matmul -> tile.matmul_acc K-loop). Our solver
// works one level UP (DDR->L1, multi-core): for the same thin matmul it keeps
// the whole [16,64] output as ONE tile and SPLIT-Ks the contraction across the
// 24 cube cores. The two compose: our cross-core split-K partial, then pypto's
// per-core fractal K-loop inside each core. So our solver ADDS the parallelism
// the single-core L0 pass cannot express; they don't conflict.
static void test_pypto_l0_comparison() {
    std::cout << "[PYPTO] test_auto_tile_matmul_l0 matmul (M16 N64 K2048) vs our solver\n";
    // out[N=64, M=16] = A[K=2048, M=16] @ B[N=64, K=2048]; base = per-native FLOPs.
    Problem p;
    p.tensors = {{2048, 16}, {64, 2048}, {64, 16}};
    p.ops = {{OpType::MatMul, {0, 1}, {2}, 16384 * 2048}};
    p.fast_memory_capacity = 1 << 24;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    auto r = Subgraph::create(p, DAG::build(p), {0})->best_cost();
    std::cout << "    pypto L0 (hand-crafted): out tile[16,64] Acc, K tiled by 16, 1 core sequential\n";
    std::cout << "    our solver (DDR->L1)   : out tile[" << r.config.w << "," << r.config.h
              << "] k=" << r.config.k << " spatial=" << r.num_spatial_tiles
              << " split-K=" << r.parallel_split << " cores=" << r.cores_used << "\n";
    // Thin output (16x64) << 24 cores => our level parallelizes via split-K.
    CHECK("PYPTO: our solver split-Ks the thin L0 matmul across cube cores", r.parallel_split > 1);
    // The output is too small to tile spatially — one DDR->L1 tile == the whole output.
    CHECK("PYPTO: our DDR->L1 tile keeps the full output (1 spatial tile)", r.num_spatial_tiles == 1);
    // Our k spans the FULL contraction; the 16-fractal K-loop is pypto's L0 level.
    CHECK("PYPTO: our k-tile is the full contraction (16-fractal is pypto's L0)", r.config.k == 2048);
}

// --- sweep output shape: spatial parallelism vs split-K transition -----------
static void test_sweep_matmul_dims() {
    std::cout << "[SWEEP] single matmul, compute-heavy K=512 — strategy vs output shape\n";
    // With 16-fractal alignment, a [w,h] output sub-tiles into up to (w/16)(h/16)
    // spatial tiles, so the 24 cube cores fill SPATIALLY for any output >= a few
    // hundred wide. Split-K is the fallback only when even 16-aligned sub-tiling
    // yields fewer than #cores tiles — i.e. a TINY output. As L grows the solver
    // shifts split-K -> pure spatial; the split factor is non-increasing.
    int prev_split = 1 << 30;
    for (int L : {16, 32, 64, 256, 1024}) {
        auto p = mk_mm(L, L, 512, 16384 * 512);
        char nm[32];
        snprintf(nm, sizeof(nm), "square %dx%d", L, L);
        viz_row(nm, p, {0});
        int s = Subgraph::create(p, DAG::build(p), {0})->best_cost().parallel_split;
        CHECK("SWEEP: split-K factor is non-increasing as output grows", s <= prev_split);
        prev_split = s;
    }
    // Tiny 16x16 output = 1 fractal tile << 24 cores -> must split-K; a large
    // 1024x1024 fills the cores spatially -> no split-K.
    auto ps = mk_mm(16, 16, 512, 16384 * 512);
    auto pl = mk_mm(1024, 1024, 512, 16384 * 512);
    CHECK("SWEEP: 16x16 output split-Ks across cores (tiny output)",
          Subgraph::create(ps, DAG::build(ps), {0})->best_cost().parallel_split > 1);
    CHECK("SWEEP: 1024x1024 output is pure spatial (no split-K)",
          Subgraph::create(pl, DAG::build(pl), {0})->best_cost().parallel_split == 1);
    // Same tiny output but LARGE K: split-K stays preferred because small spatial
    // tiles would be reload-bound (test H's mechanism), not just under-filled.
    auto pk = mk_mm(128, 128, 8192, 2400000);
    int sk = Subgraph::create(pk, DAG::build(pk), {0})->best_cost().parallel_split;
    std::cout << "    128x128 K=8192 -> split=" << sk << "\n";
    CHECK("SWEEP: 128x128 large-K reload-bound -> split-K", sk > 1);
    // A LARGE matmul fills the L0c accumulator: the tile area approaches
    // cube_capacity/4 (=32768 elems). This is WHY big matmuls get big tiles and
    // only small outputs get small ones — the cap is L0c, not a fixed granule.
    auto rbig = Subgraph::create(mk_mm(1024, 1024, 512, 16384 * 512),
                                 DAG::build(mk_mm(1024, 1024, 512, 1)), {0});
    auto pbig = mk_mm(1024, 1024, 512, 16384 * 512);
    auto cbig = Subgraph::create(pbig, DAG::build(pbig), {0})->best_cost();
    std::cout << "    1024x1024 -> tile[" << cbig.config.w << "x" << cbig.config.h
              << "] area=" << (cbig.config.w * cbig.config.h) << " (L0c/4=32768)\n";
    CHECK("SWEEP: large output gets a big L0c-filling tile (area >= 128x128)",
          cbig.config.w * cbig.config.h >= 128 * 128);
    (void)rbig;
}

// --- visualization: print every scenario's solver solution -------------------
static void test_visualize() {
    std::cout << "\n==================== SOLVER SOLUTION VISUALIZATION ====================\n";
    std::cout << "-- single matmul: large outputs get the L0c-max tile (128x256) --\n";
    viz_row("square 2048x2048", mk_mm(2048, 2048, 2048, 16384 * 2048), {0});
    viz_row("square 1024x1024", mk_mm(1024, 1024, 512, 16384 * 512), {0});
    viz_row("square 256x256", mk_mm(256, 256, 512, 16384 * 512), {0});
    viz_row("square 768x768", mk_mm(768, 768, 512, 16384 * 512), {0});
    std::cout << "-- single matmul: tiny / large-K outputs -> split-K --\n";
    viz_row("tiny 16x16", mk_mm(16, 16, 512, 16384 * 512), {0});
    viz_row("tiny 64x64", mk_mm(64, 64, 512, 16384 * 512), {0});
    viz_row("128x128 K=8192", mk_mm(128, 128, 8192, 2400000), {0});
    std::cout << "-- single matmul: memory-bound (cheap compute) -> big balanced tile --\n";
    viz_row("square 512x512 cheap", mk_mm(512, 512, 512, 512), {0});
    std::cout << "-- two matmuls --\n";
    {
        auto p = mk_chained(128, 256, 128, 64, 16384 * 256, 16384 * 128);
        DAG dag = DAG::build(p);
        viz_row("chained M1 (A@B)", p, {0});
        viz_row("chained M2 (T@D)", p, {1});
        viz_row("chained FUSED", p, {0, 1});
    }
    std::cout << "-- vector: pointwise / softmax / reduction --\n";
    {
        Problem p;
        p.tensors = {{4096, 128}, {1, 128}, {4096, 128}, {1, 128}, {4096, 128}};
        p.ops = {{OpType::Reduction, {0}, {1}, 4096 * 128 / 4},
                 {OpType::Pointwise, {0, 1}, {2}, 4096 * 128},
                 {OpType::Reduction, {2}, {3}, 4096 * 128 / 4},
                 {OpType::Pointwise, {2, 3}, {4}, 4096 * 128}};
        p.fast_memory_capacity = 1 << 24;
        p.slow_memory_bandwidth = 10;
        p.native_w = 128;
        p.native_h = 128;
        set_910b(p);
        viz_row("softmax fused", p, {0, 1, 2, 3});
        viz_row("rowmax alone", p, {0});
    }
    {
        Problem p;
        p.tensors = {{4096, 4}, {1, 4}};
        p.ops = {{OpType::Reduction, {0}, {1}, 4096LL * 4 * 1000}};
        p.fast_memory_capacity = 1 << 24;
        p.slow_memory_bandwidth = 10;
        p.native_w = 128;
        p.native_h = 128;
        set_910b(p);
        viz_row("few-row reduction", p, {0});
    }
    std::cout << "======================================================================\n\n";
}

int main() {
    test_visualize();
    test_two_matmul();
    test_pypto_l0_comparison();
    test_sweep_matmul_dims();
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
