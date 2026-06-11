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
    p.cube_capacity = 128 * 1024;   // L0c accumulator (output)
    p.l1_capacity = 512 * 1024;     // L1/Mat operand pool (cube)
    p.vec_capacity = 192 * 1024;    // UB (vector)
    // Machine-cost compute model (replaces per-op base_cost): cube = #16x16x16
    // fractals * cube_compute_cost; vector = ceil(#elements/vector_lanes) *
    // vector_compute_cost. These are PLACEHOLDER values (relative units) until
    // grounded from real 910B specs; the per-op base_cost is now ignored.
    p.cube_compute_cost = 4096;     // per 16x16x16 cube fractal
    p.vector_compute_cost = 1;      // per vector SIMD step
    p.vector_lanes = 256;           // elements per vector SIMD step (to calibrate)
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
    // byte-based: (load T0 + store T1) = 2 * 512*512 * 4B / 10 = 209715.2 (tile-invariant).
    CHECK_EQ("A: latency == DDR floor (byte-based, 209715.2)", lat, 209715.2);
}

// --- B: thin matmul — should parallelize via split-K across cube cores --------
static void test_B_thin_matmul_parallel() {
    std::cout << "[B] thin matmul C[64,64], K=2048 — should split-K across cube cores\n";
    Problem p;
    // C[w=64,h=64] = A[w=K=2048,h=64] @ B[w=64,h=K=2048]. Output is only 4x4 = 16
    // native-16 tiles < 24 cores, so even 16-aligned spatial sub-tiling cannot
    // fill the cores — split-K is genuinely required. (A 128x128 output, by
    // contrast, is 8x8 = 64 tiles and fills the cores SPATIALLY, no split-K.)
    p.tensors = {{2048, 64}, {64, 2048}, {64, 64}};
    p.ops = {{OpType::MatMul, {0, 1}, {2}, 2400000}};  // compute-heavy (2.4M = 24*100k)
    p.fast_memory_capacity = 1 << 20;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0});
    CHECK("B: subgraph builds", (bool)sg);
    auto r = sg->best_cost();
    // Geometry compute = #16x16x16 fractals * cube_compute_cost; one core alone
    // would take that whole serial time. Split-K across the 24 cube cores divides
    // it ~24x. (Absolute value depends on the machine cost; assert the parallelism.)
    const double serial = (64.0 / 16) * (64.0 / 16) * (2048.0 / 16) * p.cube_compute_cost;  // #fractals * cube_cost
    std::cout << "    best latency = " << r.latency << " cores=" << r.cores_used
              << " split=" << r.parallel_split << " (1-core serial=" << serial << ")\n";
    CHECK("B: split-K fills the 24 cube cores", r.cores_used == 24 && r.parallel_split > 1);
    CHECK("B: split-K parallelizes (lat << 1-core serial)", r.latency < serial / 5.0);
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
    // k chosen to fit the L1 operand pool (reload is k-independent — it streams
    // the full K either way — so this still isolates the 2D-vs-1D tile shape).
    // Compare the DDR traffic (reload) directly: a matmul is compute-bound under
    // the geometry model, so the reload is hidden in the latency — but the tiebreak
    // still uses it. Balanced (2D) tiles reload less than skewed (1D).
    double balanced = sg->compute_cost(TileConfig{256, 128, 128}).ddr_traffic;  // ratio 2:1
    double skewed = sg->compute_cost(TileConfig{512, 64, 64}).ddr_traffic;      // ratio 8:1
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
    DAG dag = DAG::build(p);
    double par = Subgraph::create(p, dag, {0})->best_cost().latency;
    // Geometry serial = the whole fractal compute on one cube core. Split-K across
    // the 24 cube cores must divide it ~24x. Compare against the geometry serial,
    // not a cores=1 competition baseline (different model: base_cost padded-native
    // tile vs fractal geometry — not apples-to-apples).
    const double serial = (128.0 / 16) * (128.0 / 16) * (2048.0 / 16) * p.cube_compute_cost;
    std::cout << "    24 cores (split-K)=" << par << "  1-core serial=" << serial << "\n";
    CHECK("F: split-K gives a large speedup vs single core", par < serial / 5.0);
}

// --- G: cube tile can exceed native 128 (L0c bound, not the native cap) -------
static void test_G_super_native_tile() {
    std::cout << "[G] matmul tile can be >128 (L0c bound, not the native cap)\n";
    Problem p;
    // Large 1024x1024 output: among the equal-latency split=1 configs (>=24
    // spatial tiles, compute-bound), the reload tiebreak prefers the BIGGEST
    // balanced tile — here ~256x128 (= L0c-bounded: 256*128*4B = 131072 = L0c).
    // A 512x512 output is too small: a >128 balanced tile would yield <24 tiles
    // and need split-K (a merge tail), so it falls back to <=128 tiles. The point
    // stands — tiles are bounded by L0c (128KB), not the native 128.
    p.tensors = {{512, 1024}, {1024, 512}, {1024, 1024}};  // M=N=1024, K=512
    p.ops = {{OpType::MatMul, {0, 1}, {2}, 100}};
    p.fast_memory_capacity = 1 << 26;
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
    // k fits L1 (big tile's operand strips are larger, so it needs a smaller k);
    // reload streams the full K=2048 regardless, so the comparison still holds.
    double big_splitk = sg->compute_cost(TileConfig{128, 128, 512}).latency;
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
    // Chained, FAT sink E[128,256] (fills the cores so the fused case needs no
    // heavy split-K), large intermediate C[256,256]. base_cost ignored (geometry).
    auto p = mk_chained(256, 512, 256, 128, 16384 * 512, 16384 * 256);
    p.cube_compute_cost = 1000;  // DDR-relevant regime (see test_fusion_decision_matrix)
    DAG dag = DAG::build(p);
    double m1 = Subgraph::create(p, dag, {0})->best_cost().latency;
    double m2 = Subgraph::create(p, dag, {1})->best_cost().latency;
    double fused = Subgraph::create(p, dag, {0, 1})->best_cost().latency;
    std::cout << "    M1=" << m1 << " M2=" << m2 << " fused=" << fused << "\n";
    // Fused does BOTH matmuls' work — must not undercut either part (the old bug
    // scaled all compute by the narrower sink output, so fused < M1 alone).
    CHECK("2MM: chained fused >= each part (compute counted per-matmul)",
          fused >= m1 - 0.5 && fused >= m2 - 0.5);
    // BSP model: the intermediate T is EPHEMERAL when fused (kept on-chip via the
    // M-partition, no DDR round-trip), so separately mm2 must reload T from DDR.
    // With a fat sink the fused case parallelizes mostly over M (little recompute),
    // so fusion strictly helps — fused < M1 + M2. (For a THIN sink the fused case
    // is forced into split-K or N-tile recompute and can LOSE — see test_fusion_
    // decision_matrix; that is the correct, faithful outcome, not a regression.)
    CHECK("2MM: chained (fat sink) fused < M1+M2 (keeps intermediate on-chip)",
          fused < m1 + m2 - 0.5);

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

// --- machine-cost compute model: geometry x per-step cost (910B) --------------
// Cube compute = (#16x16x16 fractals) * cube_compute_cost. Vector compute =
// ceil(#elements / vector_lanes) * vector_compute_cost. Both are MACHINE params
// (not per-op base_cost). Opt-in: when the params are 0 the model falls back to
// base_cost (so the rest of the suite is unchanged).
static void test_geometry_compute() {
    std::cout << "[GEOM] machine-cost compute: cube=#fractals*cost, vector=#steps*cost\n";
    // Cube: out[128,128], K=2048 -> 128/16 * 128/16 * 2048/16 = 8192 fractals.
    Problem c;
    c.tensors = {{2048, 128}, {128, 2048}, {128, 128}};
    c.ops = {{OpType::MatMul, {0, 1}, {2}, 999}};  // base_cost IGNORED when cube_compute_cost set
    c.fast_memory_capacity = 1 << 26; c.slow_memory_bandwidth = 10; c.native_w = 128; c.native_h = 128;
    set_910b(c);
    c.cube_compute_cost = 10000;  // per fractal (machine param)
    auto rc = Subgraph::create(c, DAG::build(c), {0})->best_cost();
    const double fractals = 128.0 / 16 * 128.0 / 16 * 2048.0 / 16;  // 8192
    const double exp_c = fractals * 10000.0 / rc.cores_used;
    std::cout << "    cube: fractals=" << (long long)fractals << " cores=" << rc.cores_used
              << " -> lat=" << rc.latency << " (expect " << exp_c << ")\n";
    CHECK("GEOM: cube compute = #fractals * cube_cost (compute-bound)",
          rc.compute_bound && std::abs(rc.latency - exp_c) < 1.0);

    // Vector: pointwise [512,512] -> 262144 elements; lanes=256 -> 1024 SIMD steps.
    auto run_vec = [](int64_t lanes) {
        Problem v;
        v.tensors = {{512, 512}, {512, 512}};
        v.ops = {{OpType::Pointwise, {0}, {1}, 999}};
        v.fast_memory_capacity = 1 << 26; v.slow_memory_bandwidth = 10; v.native_w = 128; v.native_h = 128;
        set_910b(v);
        v.vector_compute_cost = 40000;  // per SIMD step
        v.vector_lanes = lanes;
        return Subgraph::create(v, DAG::build(v), {0})->best_cost();
    };
    auto rv = run_vec(256);
    const double steps = std::ceil(512.0 * 512 / 256);  // 1024
    const double exp_v = steps * 40000.0 / rv.cores_used;
    std::cout << "    vector: 262144 elems / 256 lanes = " << (long long)steps << " steps cores="
              << rv.cores_used << " -> lat=" << rv.latency << " (expect " << exp_v << ")\n";
    CHECK("GEOM: vector compute = ceil(#elements/lanes) * step_cost (compute-bound)",
          rv.compute_bound && std::abs(rv.latency - exp_v) < 1.0);
    // Doubling the lanes halves the SIMD steps -> halves the vector compute.
    CHECK("GEOM: 2x vector_lanes halves the vector compute", run_vec(128).latency > rv.latency * 1.9);
}

// --- compare our DDR->L1 tile vs pypto ChooseL0Tile (mind the two levels!) -----
// Memory chain: DDR -> L1/Mat(512KB) -> L0a/L0b(64KB ea) -> cube -> L0c(128KB acc).
// pypto ChooseL0Tile (tests/ut/ir/transforms/test_{auto_tile_matmul_l0,
// l0_tile_chooser}.py) is the L1->L0 level: output [m,n] in L0c, k-FRAGMENT in
// L0a/L0b (so its k=64 is the L1->L0 k). OUR solver is the DDR->L1 level: output
// [w,h] in L0c, k-STRIP in L1 (so our k is the DDR->L1 k). Two axes:
//   OUTPUT tile: SAME level (both bound by L0c) -> directly comparable. Ours uses
//                clean divisors ([128,256]); pypto allows aligned-padded ([176,176]).
//   k-tile:      DIFFERENT, ADJACENT levels. Our DDR->L1 k-strip CONTAINS pypto's
//                L1->L0 k-fragments (512 = 8 x 64) -- a composition, NOT a same-
//                level k comparison. The true like-for-like pypto DDR->L1 k is the
//                author's pl.range chunking in examples/, not ChooseL0Tile.
static void test_pypto_tiling_comparison() {
    std::cout << "[PYPTO2] our DDR->L1 tiling vs pypto ChooseL0Tile hand-crafted L0 tiles\n";
    struct Case { const char* name; int64_t M, N, K; int p_m, p_n, p_k; };
    const Case cases[] = {
        {"skinny 16x64@2048", 16, 64, 2048, 16, 64, 256},      // skinny_gemm_pipelined: k=256
        {"large sq 4096@4096", 4096, 4096, 4096, 176, 176, 64},  // large_square_gemm: m~n~176, k=64
        {"short-N 512x128@2048", 512, 128, 2048, 256, 128, 64},  // short_n: m>=256, n=128, k=64
    };
    for (const auto& c : cases) {
        Problem p;
        p.tensors = {{c.K, c.M}, {c.N, c.K}, {c.N, c.M}};  // A[K,M] B[N,K] -> C[N,M]
        p.tensors[0].dtype = DType::BF16;  // A operand
        p.tensors[1].dtype = DType::BF16;  // B operand
        p.tensors[2].dtype = DType::FP32;  // C accumulator
        p.ops = {{OpType::MatMul, {0, 1}, {2}, 16384LL * c.K}};
        p.fast_memory_capacity = 1LL << 30;
        p.slow_memory_bandwidth = 10;
        p.native_w = 128;
        p.native_h = 128;
        set_910b(p);
        auto r = Subgraph::create(p, DAG::build(p), {0})->best_cost();
        printf("  %-22s ours[N=%lld M=%lld k=%lld] split=%d | pypto L0[m=%d n=%d k=%d]\n",
               c.name, (long long)r.config.w, (long long)r.config.h, (long long)r.config.k,
               r.parallel_split, c.p_m, c.p_n, c.p_k);
        if (r.parallel_split == 1) {
            // Spatial matmul: one core does the full K, streamed in our DDR->L1
            // k-strip, which CONTAINS pypto's L1->L0 k-fragments (composition
            // across adjacent levels — NOT a same-level k comparison).
            CHECK("PYPTO2: spatial -> our DDR->L1 k-strip >= pypto's L1->L0 k-fragment",
                  r.config.k >= c.p_k);
        } else {
            // Thin matmul: our solver SPLIT-Ks across cores (each does K/S), a
            // multi-core strategy pypto's single-core L0 chooser doesn't express;
            // the per-core k (K/S) is then a different quantity than pypto's L0 k.
            CHECK("PYPTO2: thin matmul -> our solver split-Ks across cores", r.parallel_split > 1);
        }
        // Our level bounds OPERANDS by L1 (not the output by L0c — that L0c
        // accumulator sub-tiling is pypto/AutoTileMatmulL0's L1->L0 job).
        long long opnd = (long long)r.config.k * (r.config.w + r.config.h) * 2;  // BF16 strips
        CHECK("PYPTO2: our operand strips fit L1 (512KB)", opnd <= 512 * 1024);
    }
}

// --- per-core two-pool working set: cube L1 k-tiling / vector UB / dbl-buffer --
static void test_two_pool_working_set() {
    std::cout << "[POOL] per-core two-pool: cube operands->L1 (k-tile), vector->UB, double-buffer\n";
    // Cube: a LARGE spatial output (split=1, so split-K doesn't cap k) with big K.
    // The DDR->L1 k-tile is set purely by the L1 operand budget; full-K strips
    // blow 512 KB L1, so k shrinks to fit. (A small output would split-K and the
    // per-core K-share would cap k instead, masking the L1 effect.)
    Problem c;
    c.tensors = {{4096, 2048}, {2048, 4096}, {2048, 2048}};  // A[K,M] B[N,K] -> C[N,M], out 2048^2
    c.ops = {{OpType::MatMul, {0, 1}, {2}, 16384 * 4096}};
    c.fast_memory_capacity = 1 << 28;
    c.slow_memory_bandwidth = 10;
    c.native_w = 128;
    c.native_h = 128;
    set_910b(c);
    auto rc = Subgraph::create(c, DAG::build(c), {0})->best_cost();
    long long opnd = (long long)rc.config.k * (rc.config.w + rc.config.h) * 4;  // FP32 operand bytes
    std::cout << "    cube out[2048,2048] K=4096: tile[" << rc.config.w << "x" << rc.config.h
              << "] k=" << rc.config.k << " split=" << rc.parallel_split
              << " operand=" << opnd << "B (L1=" << (512 * 1024) << ")\n";
    CHECK("POOL: cube k-tiled to fit L1 (k < full K=4096)", rc.config.k < 4096);
    CHECK("POOL: cube operand strips fit L1 (512KB)", opnd <= 512 * 1024);
    // Double-buffering halves L1 -> a strictly smaller k-tile.
    Problem cd = c;
    cd.double_buffer = true;
    auto rcd = Subgraph::create(cd, DAG::build(cd), {0})->best_cost();
    std::cout << "    double-buffer on: k=" << rcd.config.k << " (single-buffer k=" << rc.config.k << ")\n";
    CHECK("POOL: double-buffering shrinks the cube k-tile (half L1)", rcd.config.k < rc.config.k);
    // Vector: a pointwise tile (in + out resident) must fit the 192 KB UB.
    Problem v;
    v.tensors = {{512, 512}, {512, 512}};
    v.ops = {{OpType::Pointwise, {0}, {1}, 100}};
    v.fast_memory_capacity = 1 << 28;
    v.slow_memory_bandwidth = 10;
    v.native_w = 128;
    v.native_h = 128;
    set_910b(v);
    auto rv = Subgraph::create(v, DAG::build(v), {0})->best_cost();
    long long ub = (long long)rv.config.w * rv.config.h * 2 * 4;  // in + out, FP32
    std::cout << "    vector tile[" << rv.config.w << "x" << rv.config.h << "] footprint=" << ub
              << "B (UB=" << (192 * 1024) << ")\n";
    CHECK("POOL: vector tile fits UB (192KB)", ub <= 192 * 1024);
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
    std::cout << "    pypto L0 (hand-crafted): out tile[16,64] Acc, k=256 (ChooseL0Tile), 1 core sequential\n";
    std::cout << "    our solver (DDR->L1)   : out tile[" << r.config.w << "," << r.config.h
              << "] k=" << r.config.k << " spatial=" << r.num_spatial_tiles
              << " split-K=" << r.parallel_split << " cores=" << r.cores_used << "\n";
    // Thin output (16x64) << 24 cores => our level parallelizes via split-K.
    CHECK("PYPTO: our solver split-Ks the thin L0 matmul across cube cores", r.parallel_split > 1);
    // BSP model: the split-K merge is an additive tail S*out_store, so the solver
    // spreads a LITTLE spatially (here 2 tiles) to halve the partial count S and
    // shrink the tail, rather than putting the whole output in one tile and
    // split-K'ing it 24 ways. The invariant is that it FILLS the cores (spatial x
    // split), not that it keeps a single spatial tile.
    CHECK("PYPTO: solver fills the cube cores (spatial x split-K)", r.cores_used == 24);
    // Three k-levels now compose: OUR DDR->L1 k-tile (largest k whose operand
    // strips fit L1, < full K=2048), then AutoTileMatmulL0's 16-fractal K-loop
    // inside each core, then the hardware. So our k is L1-bounded, not full K.
    CHECK("PYPTO: our k is L1-tiled at the DDR->L1 level (16 <= k < K)",
          r.config.k >= 16 && r.config.k < 2048);
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
    // Tiny 16x16 output = 1 fractal tile << 24 cores -> heavy split-K. As the
    // output grows the solver fills the cores SPATIALLY (with big L1-bounded
    // tiles), so the split factor falls sharply. A huge 4096x4096 reaches pure
    // spatial (>=24 big tiles, split=1).
    auto ps = mk_mm(16, 16, 512, 16384 * 512);
    auto pl = mk_mm(4096, 4096, 512, 16384 * 512);
    int split_tiny = Subgraph::create(ps, DAG::build(ps), {0})->best_cost().parallel_split;
    int split_huge = Subgraph::create(pl, DAG::build(pl), {0})->best_cost().parallel_split;
    CHECK("SWEEP: tiny 16x16 output split-Ks heavily across cores", split_tiny > 1);
    CHECK("SWEEP: split-K falls sharply as output grows (huge << tiny)", split_huge < split_tiny);
    // Thin output (64x64 = 16 native-16 tiles < 24 cores) with LARGE K: spatial
    // sub-tiling can't fill the cores, so split-K is required. (128x128 = 64 tiles
    // would fill spatially under the BSP model and take split=1 — large K raises
    // COMPUTE, keeping it compute-bound and spatial, not split-K.)
    auto pk = mk_mm(64, 64, 8192, 2400000);
    int sk = Subgraph::create(pk, DAG::build(pk), {0})->best_cost().parallel_split;
    std::cout << "    64x64 K=8192 -> split=" << sk << "\n";
    CHECK("SWEEP: 64x64 thin output, large K -> split-K fills cores", sk > 1);
    // A LARGE matmul gets a big DDR<->L1 tile (operand strips fill L1, NOT the
    // L0c accumulator — that sub-tiling is the later L1->L0 pass). So the output
    // tile can exceed L0c; only small outputs are forced small.
    auto pbig = mk_mm(1024, 1024, 512, 16384 * 512);
    auto cbig = Subgraph::create(pbig, DAG::build(pbig), {0})->best_cost();
    std::cout << "    1024x1024 -> tile[" << cbig.config.w << "x" << cbig.config.h
              << "] area=" << (cbig.config.w * cbig.config.h) << " (output may exceed L0c)\n";
    CHECK("SWEEP: large output gets a non-degenerate tile (area >= 128x128)",
          cbig.config.w * cbig.config.h >= 128 * 128);
}

// --- visualization: print every scenario's solver solution -------------------
static void test_visualize() {
    std::cout << "\n==================== SOLVER SOLUTION VISUALIZATION ====================\n";
    std::cout << "-- single matmul: large outputs get a big L1-bounded tile (output may exceed L0c) --\n";
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

// --- fusion decision matrix: fuse only when it strictly helps -----------------
// Encodes the BSP + recompute model's verdicts. A fused chain is parallelized
// over the SHARED dim M; an intermediate is ephemeral ONLY when M fills the
// cores (num_tw==1). When the sink must tile over N (M too small, or a thin
// sink), each sink column-tile recomputes the intermediate band (compute x
// num_tw) — so fusion can LOSE to splitting, which round-trips the intermediate
// via DDR instead. The partitioner picks the cheaper; these assert which wins.
static void test_fusion_decision_matrix() {
    std::cout << "[FDM] fusion decision matrix (fuse iff strictly cheaper than split)\n";
    auto cmp = [](const char* tag, int64_t M, int64_t K1, int64_t N1, int64_t N2) {
        auto p = mk_chained(M, K1, N1, N2, 1000, 1000);
        // Lower the cube cost into the regime where DDR matters (matches the
        // benchmarks). At set_910b's 4096 the chain is so compute-bound that the
        // intermediate's round-trip is fully hidden and fusion is latency-neutral
        // for ALL shapes — the fuse-vs-split decision only varies once DDR binds.
        p.cube_compute_cost = 1000;
        DAG dag = DAG::build(p);
        auto fz = Subgraph::create(p, dag, {0, 1});
        double f = fz ? fz->best_cost().latency : 1e18;
        double s = Subgraph::create(p, dag, {0})->best_cost().latency +
                   Subgraph::create(p, dag, {1})->best_cost().latency;
        std::cout << "    " << tag << ": fused=" << f << " split=" << s
                  << (f < s ? "  -> FUSE" : "  -> SPLIT") << "\n";
        return std::pair<double, double>(f, s);
    };
    // FAT sink, big intermediate, M fills cores -> fusion saves the C round-trip.
    auto win = cmp("fuse-wins (fat sink, big C)", 256, 512, 256, 128);
    CHECK("FDM: fat-sink chain FUSES (fused < split)", win.first < win.second - 0.5);
    // Small M -> sink tiles over N2 -> intermediate recomputed per column-tile ->
    // fusion costs more than splitting (which round-trips C via DDR once).
    auto sm = cmp("small-M (N2-tiled, recompute)", 64, 256, 256, 512);
    CHECK("FDM: small-M chain SPLITS (recompute makes fusion lose)", sm.first > sm.second + 0.5);
    // Wide intermediate (huge C compute): recomputing it per column-tile dwarfs
    // its DDR round-trip -> split.
    auto wide = cmp("wide-C (huge intermediate)", 256, 512, 2048, 256);
    CHECK("FDM: wide-intermediate chain SPLITS (recompute >> round-trip)",
          wide.first > wide.second + 0.5);
}

// --- DFS execution order: post-order from sinks, producer before consumer -----
// The fixed pebbling order the peak-working-set sweep walks (and that gets
// emitted with the solution). For a chain C=(A@B)@D the producer matmul must
// precede the sink; a singleton subgraph is just its own op.
static void test_dfs_order() {
    std::cout << "[ORDER] DFS execution order (producer before consumer)\n";
    auto p = mk_chained(256, 512, 256, 128, 1000, 1000);  // op0=T=A@B, op1=C=T@D (sink)
    DAG dag = DAG::build(p);
    auto fused = Subgraph::create(p, dag, {0, 1});
    CHECK("ORDER: fused chain creates", (bool)fused);
    const auto& ord = fused->execution_order();
    CHECK("ORDER: order covers both chain ops", ord.size() == 2);
    CHECK("ORDER: producer op0 emitted before sink op1",
          ord.size() == 2 && ord[0] == 0 && ord[1] == 1);
    auto single = Subgraph::create(p, dag, {1});
    CHECK("ORDER: singleton order is the op itself",
          single && single->execution_order().size() == 1 &&
          single->execution_order()[0] == 1);
}

// --- derive_exec: intermediate bands + greedy per-op k + pebble peak ----------
// The dynamic peak that replaces the old static operand-strip sum. For the chain
// C=(A@B)@D the intermediate band T must be counted (the static input-sum
// ignored it), and each matmul's single-core k is derived to fit the headroom.
static void test_exec_enumeration() {
    std::cout << "[EXEC] derive_exec: bands + greedy per-op k + peak\n";
    auto p = mk_chained(256, 512, 256, 128, 1000, 1000);  // FP32; T=[256,256]
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0, 1});
    CHECK("EXEC: chain creates", (bool)sg);
    TileConfig cfg{128, 256, 0};                          // full M-band, k derived
    std::vector<int64_t> kk;
    int64_t peak = sg->cube_peak_l1(cfg, &kk);
    CHECK("EXEC: chain feasible at [128,256]", peak != INT64_MAX);
    // T band = 256*256*4 = 262144; op0 strip at k=128 = 1536*128 = 196608.
    CHECK_EQ("EXEC: peak L1 = T band + op0 strip", (double)peak, 458752.0, 0.5);
    CHECK("EXEC: op0 internal seq-k sliced to 128", kk.size() > 1 && kk[0] == 128);
    CHECK("EXEC: op1 sink seq-k = full output_K 256", kk.size() > 1 && kk[1] == 256);
    CHECK("EXEC: peak counts the intermediate band (old sum ignored it)",
          (double)peak > 262144.0);
}

// --- thread 1: single-core k-split is allowed on INTERNAL matmuls -------------
// A chain whose internal matmul has a huge K: its full-K operand strip blows L1,
// so the chain is feasible ONLY because the internal mm derives a SLICED k
// (single-core accumulation, no cross-core merge). The sink keeps its small full K.
static void test_seq_k_intermediate() {
    std::cout << "[SEQK] internal matmul slices its own k to fit L1 (no merge)\n";
    auto p = mk_chained(128, 8192, 64, 64, 1000, 1000);  // op0: K1=8192 internal
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0, 1});
    std::vector<int64_t> kk;
    int64_t peak = sg->cube_peak_l1({64, 128, 0}, &kk);
    CHECK("SEQK: chain feasible (internal mm sliced to fit)", peak != INT64_MAX);
    CHECK("SEQK: internal mm0 k sliced below full K=8192", kk.size() > 1 && kk[0] < 8192);
    CHECK("SEQK: internal mm0 k is a valid 16-fractal strip", kk.size() > 1 && kk[0] >= 16);
    CHECK("SEQK: sink mm1 keeps its small full K=64", kk.size() > 1 && kk[1] == 64);
    // Without slicing the internal full-K strip (768 bytes/k * 8192) would be ~6 MB
    // >> 512 KB L1 — the static model had no way to express this.
}

// --- thread 2: the intermediate BAND gates feasibility (static sum ignored it) -
// A chain with a very WIDE intermediate T: its [N1, h] band alone overflows L1 at
// a full M-band, so the dynamic peak REJECTS h=128 (correctly) while feasible at
// h=16. The old static sum counted only boundary INPUTS and never saw the band,
// so it would have wrongly accepted the overflowing config.
static void test_peak_band_feasibility() {
    std::cout << "[BAND] intermediate band gates feasibility (peak vs static sum)\n";
    auto p = mk_chained(128, 64, 4096, 64, 1000, 1000);  // T = [4096, 128], FP32
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0, 1});
    // h=128: band = 4096*128*4 = 2 MB > 512 KB L1 -> infeasible.
    CHECK("BAND: full M-band infeasible (T band overflows L1)",
          sg->cube_peak_l1({64, 128, 0}) == INT64_MAX);
    // h=16: band = 4096*16*4 = 256 KB < 512 KB -> feasible.
    CHECK("BAND: small M-band feasible (band fits, peak under L1)",
          sg->cube_peak_l1({64, 16, 0}) != INT64_MAX);
}

// --- chained 2mm: the three k-split regimes, each a distinct optimal tiling ----
// The fused chain C=(A@B)@D selects one of three contraction strategies for its
// optimal tiling, isolated here by shape. Two orthogonal axes decide it:
//
//   AXIS 1 — sink output area [N2 x M].  Spatial tiles ~ (N2/16)(M/16). When this
//     is >= the cube cores (~24) the sink fills the cores SPATIALLY -> split=1.
//     When it is far below, the sink's contraction is split ACROSS cores to fill
//     them -> PARALLEL split-K (BSP merge tail). Boundary ~ N2*M/256 ~ 24 cores,
//     i.e. sink area ~ a few thousand 16-fractals (modulated by DDR-boundness:
//     a memory-bound sink fills fewer cores).
//
//   AXIS 2 — a matmul's full-K operand strip vs L1.  Strip = K * (lhs_b*h +
//     rhs_b*w_eff) bytes; when it exceeds the L1 headroom (L1 - live bands) the
//     matmul SEQ-slices its own contraction on ONE core (no merge). Boundary
//     K ~ headroom / (dtype_bytes * (h + w_eff)).  For the chains below
//     (tile ~[*,32], FP32) onset is ~K1 = 2048.
//
// The regimes (all on the FUSED subgraph's best tiling):
//   NO-K        sink fills spatially (split=1), every K fits L1  -> seq_k = K
//   INTERNAL-K  sink fills spatially (split=1), internal K1 > L1 -> internal
//               seq-slices on-core; sink unaffected
//   PARALLEL-K  sink area too small to fill cores -> split>1; all K fit L1, so
//               no on-core seq-slice (the split is core-fill, not an L1 limit)
static void test_chain_ksplit_variants() {
    std::cout << "[KVAR] chained 2mm: no-k / internal-k / parallel-k regimes\n";
    struct R { int split; int64_t k0, k1, K1, N1; };
    auto run = [](int64_t M, int64_t K1, int64_t N1, int64_t N2, int64_t cc) {
        auto p = mk_chained(M, K1, N1, N2, 1000, 1000);
        p.cube_compute_cost = cc;
        DAG dag = DAG::build(p);
        auto sg = Subgraph::create(p, dag, {0, 1});
        auto r = sg->best_cost();
        std::vector<int64_t> pk; sg->cube_peak_l1(r.config, &pk);  // op0=internal, op1=sink
        return R{r.parallel_split, pk[0], pk[1], K1, N1};
    };
    // (1) NO k-split: small contractions, sink large enough to fill cores.
    auto nok = run(256, 256, 64, 512, 1000);
    CHECK("KVAR/no-k: sink not parallel-split (spatial fill)", nok.split == 1);
    CHECK("KVAR/no-k: internal runs full K1 (no seq-slice)", nok.k0 == nok.K1);
    CHECK("KVAR/no-k: sink runs full K (no seq-slice)", nok.k1 == nok.N1);
    // (2) INTERNAL single-core k-split: huge K1 forces the internal matmul to
    // slice its contraction on one core; the sink still fills cores spatially.
    auto ink = run(1024, 8192, 64, 1024, 4096);
    CHECK("KVAR/internal-k: sink not parallel-split", ink.split == 1);
    CHECK("KVAR/internal-k: internal seq-slices below K1 (no merge)",
          ink.k0 < ink.K1 && ink.k0 >= 16);
    CHECK("KVAR/internal-k: sink still runs full K", ink.k1 == ink.N1);
    // (3) PARALLEL k-split: sink output too small to fill cores -> split across
    // cores. Operands fit L1, so neither matmul seq-slices.
    auto par = run(64, 128, 512, 64, 1000);
    CHECK("KVAR/parallel-k: sink parallel-split across cores", par.split > 1);
    CHECK("KVAR/parallel-k: internal full K1 (L1-fit, no slice)", par.k0 == par.K1);
    CHECK("KVAR/parallel-k: sink L1-fit full K (split is core-fill, not L1)",
          par.k1 == par.N1);
}

int main() {
    test_dfs_order();
    test_exec_enumeration();
    test_seq_k_intermediate();
    test_peak_band_feasibility();
    test_chain_ksplit_variants();
    test_visualize();
    test_two_matmul();
    test_fusion_decision_matrix();
    test_two_pool_working_set();
    test_geometry_compute();
    test_pypto_l0_comparison();
    test_pypto_tiling_comparison();
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
