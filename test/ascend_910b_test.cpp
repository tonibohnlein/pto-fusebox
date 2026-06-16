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
#include "pipeline/solver.h"
#include "solution/solution.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

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
    p.kernel_fill_cost = 10000;     // per-kernel pipeline fill (placeholder; the dual
                                    // of eff — drives the tiling to ~one kernel/core)
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
    p.kernel_fill_cost = 0;  // isolate the DDR floor (fill is a separate layer)
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
    p.ops = {{OpType::Reduction, {0}, {1}, 4096LL * 4}};
    p.fast_memory_capacity = 1 << 24;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    set_910b(p);
    DAG dag = DAG::build(p);
    auto r = Subgraph::create(p, dag, {0})->best_cost();
    std::cout << "    cores=" << r.cores_used << " split=" << r.parallel_split << "\n";
    // Only 4 rows -> spatial (non-reduced) tiling can't fill the 48 vector cores,
    // so the sink reduction splits its reduced axis ACROSS cores to fill them.
    CHECK("R: few-row reduction split-fills the vector cores",
          r.cores_used == 48 && r.parallel_split > 1);
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
        q.native_w = 128; q.native_h = 128; set_910b(q); q.kernel_fill_cost = 0;
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
    c.cube_compute_cost = 10000;
    c.kernel_fill_cost = 0;  // isolate the compute roofline  // per fractal (machine param)
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
        v.vector_compute_cost = 40000;
        v.kernel_fill_cost = 0;  // isolate the compute roofline  // per SIMD step
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
    struct R { int split; int64_t k0, k1, K1, N1; int64_t w, h, outW, outH; };
    auto run = [](int64_t M, int64_t K1, int64_t N1, int64_t N2, int64_t cc) {
        auto p = mk_chained(M, K1, N1, N2, 1000, 1000);
        p.cube_compute_cost = cc;
        DAG dag = DAG::build(p);
        auto sg = Subgraph::create(p, dag, {0, 1});
        auto r = sg->best_cost();
        std::vector<int64_t> pk; sg->cube_peak_l1(r.config, &pk);  // op0=internal, op1=sink
        // sink C = [N2, M] -> the unified grid tiles this output (out_W=N2, out_H=M).
        return R{r.parallel_split, pk[0], pk[1], K1, N1, r.config.w, r.config.h, N2, M};
    };
    // Unified-grid invariant (all regimes): the emitted spatial tile tiles the
    // SINK output, so it never exceeds the output dims. Regression for the bug
    // where a chained intermediate matmul WIDER than the sink (N1 > N2) leaked
    // its width N1 into the spatial-w candidates, emitting w = N1 > out_W.
    auto grid_within_output = [](const R& r) { return r.w <= r.outW && r.h <= r.outH; };
    // (1) NO k-split: small contractions, sink large enough to fill cores.
    auto nok = run(256, 256, 64, 512, 1000);
    CHECK("KVAR/no-k: spatial tile within sink output (unified grid)", grid_within_output(nok));
    CHECK("KVAR/no-k: sink not parallel-split (spatial fill)", nok.split == 1);
    CHECK("KVAR/no-k: internal runs full K1 (no seq-slice)", nok.k0 == nok.K1);
    CHECK("KVAR/no-k: sink runs full K (no seq-slice)", nok.k1 == nok.N1);
    // (2) INTERNAL single-core k-split: huge K1 forces the internal matmul to
    // slice its contraction on one core; the sink still fills cores spatially.
    auto ink = run(1024, 8192, 64, 1024, 4096);
    CHECK("KVAR/internal-k: spatial tile within sink output (unified grid)", grid_within_output(ink));
    CHECK("KVAR/internal-k: sink not parallel-split", ink.split == 1);
    CHECK("KVAR/internal-k: internal seq-slices below K1 (no merge)",
          ink.k0 < ink.K1 && ink.k0 >= 16);
    CHECK("KVAR/internal-k: sink still runs full K", ink.k1 == ink.N1);
    // (3) PARALLEL k-split: sink output too small to fill cores -> split across
    // cores. Operands fit L1, so neither matmul seq-slices.
    auto par = run(64, 128, 512, 64, 1000);
    // THE bug trigger: intermediate T is [N1=512, M] — 8x wider than the sink
    // C [N2=64, M]. The emitted w must tile the 64-wide sink, NOT the 512 intermediate.
    CHECK("KVAR/parallel-k: spatial tile within sink output (NOT the wider intermediate)",
          grid_within_output(par));
    CHECK("KVAR/parallel-k: sink parallel-split across cores", par.split > 1);
    CHECK("KVAR/parallel-k: internal full K1 (L1-fit, no slice)", par.k0 == par.K1);
    CHECK("KVAR/parallel-k: sink L1-fit full K (split is core-fill, not L1)",
          par.k1 == par.N1);
}

// --- vector pebble: the ephemeral row-band is counted in UB ------------------
// Vector analog of test_peak_band_feasibility. A reduction subgraph x->exp->rowsum
// has an ephemeral e=[W,h] row band (W coupled to full extent). The old static
// sum counted only the boundary x/s; the pebble peak counts e, so a tile whose
// boundary input alone fits UB can still overflow once e is on-chip.
static void test_vector_band_ub() {
    std::cout << "[VBAND] vector UB pebble counts the ephemeral row band\n";
    Problem p;  // x[4096,H] -> e=exp(x)[4096,H] (ephemeral) -> s=rowsum(e)[1,H] (sink)
    p.tensors = {{4096, 128}, {4096, 128}, {1, 128}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 4096 * 128},
             {OpType::Reduction, {1}, {2}, 4096 * 128}};
    p.fast_memory_capacity = 1 << 24;
    p.slow_memory_bandwidth = 10; p.native_w = 128; p.native_h = 128;
    set_910b(p);
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0, 1});
    const int64_t UB = 192 * 1024;
    const int64_t xtile8 = 4096LL * 8 * 4;                 // boundary x tile at h=8 = 131072
    int64_t peak8 = sg->vector_peak_ub({4096, 8, 0});
    CHECK("VBAND: ephemeral e adds on top of the boundary input", peak8 > xtile8);
    CHECK("VBAND: e band overflows UB where boundary-only would fit",
          peak8 > UB && xtile8 < UB);
    CHECK("VBAND: a small M-band fits UB once e is counted",
          sg->vector_peak_ub({4096, 2, 0}) <= UB);
}

// --- reduction parallel split is SINK-ONLY -----------------------------------
// A bare rowmax (reduction IS the sink) may split its reduced axis across cores.
// A chain whose sink is a POINTWISE with an internal reduction may NOT — the
// internal reduction's partials would round-trip DDR (breaking ephemerality), so
// the model leaves it spatial-only (a cut would be the partitioner's job).
static void test_reduction_sink_gating() {
    std::cout << "[RGATE] reduced-axis split only when the reduction is the sink\n";
    auto mk = []() { Problem p; p.slow_memory_bandwidth = 10;
                     p.native_w = 128; p.native_h = 128; return p; };
    // (a) bare rowmax, few rows -> sink reduction -> splits across cores.
    Problem a = mk();
    a.tensors = {{4096, 4}, {1, 4}};
    a.ops = {{OpType::Reduction, {0}, {1}, 4096LL * 4 * 1000}};
    a.fast_memory_capacity = 1 << 24; set_910b(a);
    DAG da = DAG::build(a);
    CHECK("RGATE: bare reduction sink splits the reduced axis",
          Subgraph::create(a, da, {0})->best_cost().parallel_split > 1);
    // (b) x -> m=rowmax(x) -> y=sub(x,m): pointwise SINK, internal reduction, few
    // rows. Cannot fill cores (no split allowed) -> parallel_split stays 1.
    Problem b = mk();
    b.tensors = {{4096, 4}, {1, 4}, {4096, 4}};
    b.ops = {{OpType::Reduction, {0}, {1}, 4096LL * 4 * 1000},
             {OpType::Pointwise, {0, 1}, {2}, 4096LL * 4}};
    b.fast_memory_capacity = 1 << 24; set_910b(b);
    DAG db = DAG::build(b);
    auto sg = Subgraph::create(b, db, {0, 1});
    CHECK("RGATE: internal-reduction chain (pw sink) does NOT split",
          sg && sg->best_cost().parallel_split == 1);
}

// --- single-core W-streaming makes a large-W reduction feasible --------------
// A softmax whose row [W,1] of A+e overflows UB even at h=1 cannot materialize;
// the single-core streaming chunk (the reduction accumulated chunk-by-chunk, e
// recomputed past it) shrinks the band so it fits. The model must REPRESENT this
// optimum as feasible (regime 2 of the softmax taxonomy).
static Problem mk_softmax(int64_t W, int64_t H) {
    Problem p;
    p.tensors = {{W, H}, {1, H}, {W, H}, {1, H}, {W, H}};
    p.ops = {{OpType::Reduction, {0}, {1}, W * H / 4}, {OpType::Pointwise, {0, 1}, {2}, W * H},
             {OpType::Reduction, {2}, {3}, W * H / 4}, {OpType::Pointwise, {2, 3}, {4}, W * H}};
    p.fast_memory_capacity = 1 << 24; p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128; set_910b(p);
    return p;
}
static void test_vector_streaming_reduction() {
    std::cout << "[STREAM] large-W reduction streams to fit UB (materialize can't)\n";
    Problem p = mk_softmax(32768, 128);
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0, 1, 2, 3});
    const int64_t UB = 192 * 1024;
    TileConfig cfg{32768, 1, 0};  // full reduced extent, minimal M-band
    CHECK("STREAM: materialize overflows UB even at h=1", sg->vector_peak_ub(cfg) > UB);
    CHECK("STREAM: a small W-chunk fits UB (single-core streaming)",
          sg->vector_peak_ub(cfg, {}, {}, 256) <= UB);
    CHECK("STREAM: fused large-W softmax feasible (escalates to streaming)",
          std::isfinite(sg->best_cost().latency));
    // Step 2b — the streamed case pays the flash recompute. W=8192 materializes
    // (1 pass); W=24576 (3x the data) streams and recomputes over N_passes=3 (the
    // two reductions + final), so its cost jumps ~9x (3x data x 3x passes) — well
    // beyond the 3x a pure data scaling would give.
    Problem pm = mk_softmax(8192, 128);   // materializes
    Problem ps = mk_softmax(24576, 128);  // streams (3x W)
    DAG dm = DAG::build(pm), ds = DAG::build(ps);
    double mat = Subgraph::create(pm, dm, {0, 1, 2, 3})->best_cost().latency;
    double str = Subgraph::create(ps, ds, {0, 1, 2, 3})->best_cost().latency;
    CHECK("STREAM: streamed cost includes the recompute (>4x the 3x-data scaling)",
          str > 4.0 * mat);
}

// --- matmul sensibility invariants across a shape grid -----------------------
// Locks the "sensible solution" properties we eyeballed: every single-matmul
// tiling fills the cube cores, keeps k a clean divisor of K, keeps the L1
// operand strip within budget, and only split-Ks when the output is too small to
// fill the cores spatially.
static void test_matmul_sensibility() {
    std::cout << "[MMSENS] single-matmul tilings are sensible across shapes\n";
    struct Case { const char* tag; int64_t M, N, K; };
    const Case cases[] = {
        {"sq256", 256, 256, 256}, {"sq1024", 1024, 1024, 1024}, {"sq4096", 4096, 4096, 4096},
        {"tall", 4096, 128, 2048}, {"fat", 128, 4096, 2048}, {"thinK", 64, 64, 8192},
        {"tiny", 16, 16, 512}, {"deepK", 512, 512, 8192}, {"wideN", 256, 8192, 1024},
        {"flatK", 2048, 2048, 256}};
    for (const auto& c : cases) {
        Problem p;
        p.tensors = {{c.K, c.M}, {c.N, c.K}, {c.N, c.M}};  // A[K,M] B[N,K] -> C[N,M]
        p.ops = {{OpType::MatMul, {0, 1}, {2}, 16384 * c.K}};
        p.fast_memory_capacity = 1LL << 30; p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128; set_910b(p);
        DAG d = DAG::build(p);
        auto r = Subgraph::create(p, d, {0})->best_cost();
        char b[80];
        auto C = [&](const char* what, bool ok) {
            snprintf(b, sizeof b, "MMSENS/%s: %s", c.tag, what); CHECK(b, ok);
        };
        // total atomic 16^3 fractals; <24 would justify fewer cores (none here).
        const int64_t fractals = (c.M / 16) * (c.N / 16) * (c.K / 16);
        C("feasible", std::isfinite(r.latency));
        C("k divides K", r.config.k > 0 && c.K % r.config.k == 0);
        C("L1 strip within 512KB",
          (long long)r.config.k * (r.config.w + r.config.h) * 4 <= 512 * 1024);
        C("fills 24 cube cores", fractals < 24 ? r.cores_used == fractals : r.cores_used == 24);
        C("split-K only when spatial tiles < cores",
          r.parallel_split == 1 || r.num_spatial_tiles < 24);
    }
}

// --- vector/reduction sensibility invariants ---------------------------------
// Pointwise and reduction subgraphs fill the vector cores (spatially, or via the
// sink-reduced split for few-row reductions), stay UB-feasible, and stream only
// when materialize overflows.
static void test_vector_sensibility() {
    std::cout << "[VECSENS] vector/reduction tilings fill cores and fit UB\n";
    auto run = [](const char* tag, const Problem& p, std::vector<size_t> ops,
                  bool expect_split) {
        DAG d = DAG::build(p);
        auto sg = Subgraph::create(p, d, ops);
        char b[80];
        auto C = [&](const char* what, bool ok) {
            snprintf(b, sizeof b, "VECSENS/%s: %s", tag, what); CHECK(b, ok);
        };
        if (!sg) { C("builds", false); return; }
        auto r = sg->best_cost();
        C("feasible", std::isfinite(r.latency));
        C("fills 48 vector cores", r.cores_used == 48);
        if (expect_split) C("few-row reduction split-fills cores", r.parallel_split > 1);
    };
    auto base = [](Problem& p) { p.fast_memory_capacity = 1 << 24; p.slow_memory_bandwidth = 10;
                                 p.native_w = 128; p.native_h = 128; set_910b(p); };
    Problem pw; pw.tensors = {{4096, 4096}, {4096, 4096}};
    pw.ops = {{OpType::Pointwise, {0}, {1}, 4096 * 4096}}; base(pw);
    run("pointwise", pw, {0}, false);
    Problem rm; rm.tensors = {{1024, 512}, {1, 512}};       // many rows -> spatial fill
    rm.ops = {{OpType::Reduction, {0}, {1}, 1024 * 512}}; base(rm);
    run("rowmax-manyrows", rm, {0}, false);
    Problem rf; rf.tensors = {{4096, 4}, {1, 4}};           // few rows -> reduced split
    rf.ops = {{OpType::Reduction, {0}, {1}, 4096LL * 4 * 1000}}; base(rf);
    run("rowmax-fewrows", rf, {0}, true);
    Problem ss = mk_softmax(2048, 128);   run("softmax-smallW", ss, {0, 1, 2, 3}, false);
    Problem sl = mk_softmax(32768, 128);  run("softmax-largeW-stream", sl, {0, 1, 2, 3}, false);
}

// --- realistic model stages: FFN + attention across batch sizes --------------
// FFN (X@W1 -> relu -> @W2) and attention (Q@K^T -> softmax -> @V) must each
// partition into THREE homogeneous subgraphs (cube / vector / cube — the
// cube<->vector handoff is DDR, never fused). Verifies core-filling and that a
// sub-fractal (decode) batch pads to a 16-fractal row: same cube cost as B=16.
static CostResult eval(const Problem& p, std::vector<size_t> ops) {
    DAG d = DAG::build(p);  // kept alive across best_cost() (uses dag_)
    auto sg = Subgraph::create(p, d, ops);
    return sg ? sg->best_cost() : CostResult{};  // default latency = +inf
}
static bool creatable(const Problem& p, std::vector<size_t> ops) {
    DAG d = DAG::build(p);
    return (bool)Subgraph::create(p, d, ops);
}
static void test_model_stages() {
    std::cout << "[MODEL] FFN / attention stages across batch sizes\n";
    auto ffn = [](int64_t B) {
        Problem p; int64_t dm = 512, df = 2048;
        p.tensors = {{dm, B}, {df, dm}, {df, B}, {df, B}, {dm, df}, {dm, B}};
        p.ops = {{OpType::MatMul, {0, 1}, {2}, 16384 * dm}, {OpType::Pointwise, {2}, {3}, df * B},
                 {OpType::MatMul, {3, 4}, {5}, 16384 * df}};
        p.fast_memory_capacity = 1LL << 30; p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128; set_910b(p); return p;
    };
    char b[80];
    double mm1_decode = -1, mm1_b16 = -1;
    for (int64_t B : {1, 8, 16, 128, 512}) {
        Problem p = ffn(B);
        auto C = [&](const char* w, bool ok) { snprintf(b, sizeof b, "FFN B%lld: %s",
                                                         (long long)B, w); CHECK(b, ok); };
        C("cube+vector not fusable (homogeneity)", !creatable(p, {0, 1}));
        auto m1 = eval(p, {0}), rl = eval(p, {1}), m2 = eval(p, {2});
        C("mm1 feasible + fills 24 cube cores", std::isfinite(m1.latency) && m1.cores_used == 24);
        // relu fills 48 once there are enough rows; a tiny decode batch (B<16,
        // few thousand elements) legitimately uses fewer cores (the kernel-fill
        // dual makes over-parallelizing a tiny op cost more than it saves).
        C("relu feasible", std::isfinite(rl.latency));
        if (B >= 16) C("relu fills 48 vector cores", rl.cores_used == 48);
        C("mm2 feasible + fills 24 cube cores", std::isfinite(m2.latency) && m2.cores_used == 24);
        if (B == 1) mm1_decode = m1.latency;
        if (B == 16) mm1_b16 = m1.latency;
    }
    // Decode (B=1) and B=16 have the SAME cube compute (both pad to one fractal
    // row); the small residual difference is the byte-based DDR (B=1 reads fewer
    // real rows) + the per-kernel fill. So they're equal up to a few percent, not
    // bit-identical.
    CHECK("MODEL/FFN: decode (B=1) matmul cost ~= B=16 (fractal-quantized)",
          mm1_decode > 0 && std::abs(mm1_decode - mm1_b16) < 0.05 * mm1_b16);

    // Attention: Q@K^T -> softmax(over keys) -> @V  (d=64, S_kv=2048).
    auto attn = [](int64_t B) {
        Problem p; int64_t d = 64, S = 2048;
        p.tensors = {{d, B}, {S, d}, {S, B}, {1, B}, {S, B}, {1, B}, {S, B}, {d, S}, {d, B}};
        p.ops = {{OpType::MatMul, {0, 1}, {2}, 16384 * d}, {OpType::Reduction, {2}, {3}, S * B},
                 {OpType::Pointwise, {2, 3}, {4}, S * B}, {OpType::Reduction, {4}, {5}, S * B},
                 {OpType::Pointwise, {4, 5}, {6}, S * B}, {OpType::MatMul, {6, 7}, {8}, 16384 * S}};
        p.fast_memory_capacity = 1LL << 30; p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128; set_910b(p); return p;
    };
    for (int64_t B : {16, 128, 512}) {
        Problem p = attn(B);
        auto C = [&](const char* w, bool ok) { snprintf(b, sizeof b, "ATTN B%lld: %s",
                                                         (long long)B, w); CHECK(b, ok); };
        C("QK^T(cube)+softmax(vec) not fusable", !creatable(p, {0, 1}));
        auto sc = eval(p, {0}), sm = eval(p, {1, 2, 3, 4}), pv = eval(p, {5});
        C("scores feasible + fills 24 cube cores", std::isfinite(sc.latency) && sc.cores_used == 24);
        C("softmax(over keys) feasible", std::isfinite(sm.latency));
        C("PV feasible + fills 24 cube cores", std::isfinite(pv.latency) && pv.cores_used == 24);
        if (B >= 128) C("softmax fills 48 vector cores (enough queries)", sm.cores_used == 48);
    }
}

// --- larger instances: the full solver on 10+ op DAGs ------------------------
// Drives the partitioner (solve) on realistic multi-op graphs and checks the
// solution is structurally sound: valid, a true partition (every op covered
// exactly once — no recompute on 910B), and every step UNIT-HOMOGENEOUS (a step
// never mixes cube (MatMul) with vector (Pointwise/Reduction); Opaque is a
// singleton). These hold for ANY valid solution, so the test is robust to the
// evo search's non-determinism.
static bool step_homogeneous(const Problem& p, const ScheduleStep& s) {
    bool cube = false, vec = false, opaque = false;
    for (auto i : s.subgraph.ops()) switch (p.ops[i].type) {
        case OpType::MatMul: cube = true; break;
        case OpType::Pointwise: case OpType::Reduction: vec = true; break;
        case OpType::Opaque: opaque = true; break;
    }
    if (cube && vec) return false;
    if (opaque && s.subgraph.num_ops() > 1) return false;
    return true;
}
static void check_solution(const char* tag, const Problem& p) {
    DAG dag = DAG::build(p);
    auto sol = solve(p, dag);
    char b[80];
    auto C = [&](const char* w, bool ok) { snprintf(b, sizeof b, "BIG/%s: %s", tag, w); CHECK(b, ok); };
    C("solution valid", sol.validate().valid);
    C("total latency finite > 0", std::isfinite(sol.total_latency()) && sol.total_latency() > 0);
    std::vector<int> cover(p.num_ops(), 0);
    bool homo = true;
    for (size_t i = 0; i < sol.num_steps(); i++) {
        for (auto op : sol.step(i).subgraph.ops()) cover[op]++;
        if (!step_homogeneous(p, sol.step(i))) homo = false;
    }
    bool once = true;
    for (int c : cover) if (c != 1) once = false;
    C("partition covers every op exactly once", once);
    C("every step is unit-homogeneous (cube/vector never mixed)", homo);
}
static void test_large_instances() {
    std::cout << "[BIG] full solver on 10+ op DAGs (partition + tiling)\n";
    // (1) Transformer block: scores=Q@K^T -> softmax -> O=P@V -> O@Wo -> @W1 ->
    // relu -> @W2 (10 ops; cube/vector interleaved). d=512, B=128, S=256, df=2048.
    {
        Problem p;
        p.tensors = {{512, 128}, {256, 512}, {256, 128}, {1, 128}, {256, 128}, {1, 128},
                     {256, 128}, {512, 256}, {512, 128}, {512, 512}, {512, 128}, {2048, 512},
                     {2048, 128}, {2048, 128}, {512, 2048}, {512, 128}};
        p.ops = {{OpType::MatMul, {0, 1}, {2}, 16384 * 512},     // scores
                 {OpType::Reduction, {2}, {3}, 256 * 128},        // colmax
                 {OpType::Pointwise, {2, 3}, {4}, 256 * 128},     // exp
                 {OpType::Reduction, {4}, {5}, 256 * 128},        // colsum
                 {OpType::Pointwise, {4, 5}, {6}, 256 * 128},     // div -> P
                 {OpType::MatMul, {6, 7}, {8}, 16384 * 256},      // O = P@V
                 {OpType::MatMul, {8, 9}, {10}, 16384 * 512},     // O@Wo
                 {OpType::MatMul, {10, 11}, {12}, 16384 * 512},   // @W1
                 {OpType::Pointwise, {12}, {13}, 2048 * 128},     // relu
                 {OpType::MatMul, {13, 14}, {15}, 16384 * 2048}}; // @W2
        p.fast_memory_capacity = 1LL << 30; p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128; set_910b(p);
        check_solution("transformer-block", p);
    }
    // (2) Deep matmul chain (8 cube ops): C0=A@B0, C1=C0@B1, ... all cube; tests
    // partition + fusion on a long homogeneous chain.
    {
        Problem p;
        int64_t M = 256, D = 256;
        p.tensors.push_back({D, M});                 // 0: A [D,M]
        size_t prev = 0;
        for (int i = 0; i < 8; i++) {
            p.tensors.push_back({D, D});             // weight Bi [D,D]
            size_t w = p.tensors.size() - 1;
            p.tensors.push_back({D, M});             // output Ci [D,M]
            size_t out = p.tensors.size() - 1;
            p.ops.push_back({OpType::MatMul, {prev, w}, {out}, 16384 * D});
            prev = out;
        }
        p.fast_memory_capacity = 1LL << 30; p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128; set_910b(p);
        check_solution("deep-chain-8mm", p);
    }
}

// --- reduced-axis homogeneity: reductions on different axes cannot fuse -------
// A unified tile is coupled to the full extent on each reduced axis; fusing a
// width-reduction with a height-reduction would force the whole tensor into one
// tile (no parallelism) and the single reduced_axis_ can't represent both. The
// partitioner must cut between them. (Softmax's reductions share an axis -> fine.)
static void test_mixed_axis_reduction_rejected() {
    std::cout << "[RAXIS] reductions on different axes cannot fuse\n";
    Problem p;  // x[256,128]; m=rowmax(x)[1,128] (reduce width); c=colsum(x)[256,1] (reduce height)
    p.tensors = {{256, 128}, {1, 128}, {256, 1}};
    p.ops = {{OpType::Reduction, {0}, {1}, 256 * 128}, {OpType::Reduction, {0}, {2}, 256 * 128}};
    p.fast_memory_capacity = 1 << 24; p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128; set_910b(p);
    DAG d = DAG::build(p);
    CHECK("RAXIS: width-reduction + height-reduction NOT fusable", !Subgraph::create(p, d, {0, 1}));
    CHECK("RAXIS: width-reduction alone creates", (bool)Subgraph::create(p, d, {0}));
    CHECK("RAXIS: height-reduction alone creates", (bool)Subgraph::create(p, d, {1}));
    Problem sm = mk_softmax(2048, 128);  // both reductions reduce the SAME axis (width)
    DAG ds = DAG::build(sm);
    CHECK("RAXIS: same-axis reductions (softmax) still fuse",
          (bool)Subgraph::create(sm, ds, {0, 1, 2, 3}));
}

// --- kernel-fill: the cost prefers ~one kernel per core ----------------------
// With the native-128 vector cap removed and the per-kernel fill (the dual of
// eff), a huge pointwise that used to shatter into 16384 tiles now collapses to
// ~cores large kernels, each streamed in UB-chunks.
static void test_kernel_fill_one_per_core() {
    std::cout << "[KFILL] cost prefers ~one kernel per core (cap removed + fill)\n";
    Problem p;
    p.tensors = {{16384, 16384}, {16384, 16384}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 16384LL * 16384}};
    p.fast_memory_capacity = 1LL << 30; p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128; set_910b(p);
    DAG d = DAG::build(p);
    auto r = Subgraph::create(p, d, {0})->best_cost();
    CHECK("KFILL: huge pointwise uses ~one kernel per core (<= 2x cores)",
          r.num_spatial_tiles <= 2 * 48 && r.cores_used == 48);
    CHECK("KFILL: the per-core tile exceeds UB (streamed, not materialized)",
          (long long)r.config.w * r.config.h * 4 * 2 > 192 * 1024);
    // The fill genuinely enters the cost (same tiling search, fill on vs off).
    Problem p0 = p; p0.kernel_fill_cost = 0;
    double with_fill = r.latency;
    double no_fill = Subgraph::create(p0, d, {0})->best_cost().latency;
    CHECK("KFILL: kernel-fill adds to the cost", with_fill > no_fill);
}

// --- explicit single-core k-stream for pointwise (the matmul-seq-k analog) ----
// A vector tile that overflows UB is not "always feasible" by fiat — it derives
// an EXPLICIT chunk along the larger axis (like matmul seq-k / the reduction
// chunk). A tile that fits is materialized (no sub-streaming).
static void test_pointwise_stream_explicit() {
    std::cout << "[PWSTREAM] pointwise derives an explicit single-core k-stream\n";
    Problem p;
    p.tensors = {{8192, 256}, {8192, 256}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 8192 * 256}};
    p.fast_memory_capacity = 1LL << 30; p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128; set_910b(p);
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});
    const int64_t UB = 192 * 1024;
    TileConfig big{8192, 256, 0};   // whole tensor as one kernel -> overflows UB
    CHECK("PWSTREAM: large tile overflows UB", sg->vector_peak_ub(big) > UB);
    auto s = sg->vector_stream(big);
    CHECK("PWSTREAM: an explicit chunk is derived (streams the wider axis)",
          s.axis == 1 && s.chunk > 0 && s.chunk < 8192);
    CHECK("PWSTREAM: the chunk actually fits UB",
          sg->vector_peak_ub(big, {}, {}, s.chunk, s.axis) <= UB);
    TileConfig small{128, 128, 0};  // fits UB -> materialized, no sub-stream
    CHECK("PWSTREAM: a UB-fitting tile materializes (no stream)",
          sg->vector_stream(small).axis == 0);
}

int main() {
    test_dfs_order();
    test_exec_enumeration();
    test_seq_k_intermediate();
    test_peak_band_feasibility();
    test_chain_ksplit_variants();
    test_matmul_sensibility();
    test_model_stages();
    test_large_instances();
    test_mixed_axis_reduction_rejected();
    test_kernel_fill_one_per_core();
    test_pointwise_stream_explicit();
    test_vector_band_ub();
    test_reduction_sink_gating();
    test_vector_streaming_reduction();
    test_vector_sensibility();
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
