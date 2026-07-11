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
#include "core/subgraph_structure.h"
#include "core/types.h"
#include "pipeline/solver.h"
#include "solution/solution.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

static int g_pass = 0, g_fail = 0;

static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double g, double e, double t = 0.5) {
    if (std::abs(g - e) < t) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}
// Activate the grounded 910B parallel model (one die): 24 cube + 48 vector cores,
// L0c accumulator = 128 KB/core (the cube tile bound, replacing the native cap),
// and the pto-isa A2/A3 cost coefficients (per-direction bandwidths, core clock,
// L0 GEMM base tile, vector register + per-op compute slopes). `cube_cost` is the
// grounded cube-compute CALIBRATION MULTIPLIER on the pto-isa fractal-cycle model
// (repeats * cyc); the realistic default is 1 (the model is already in real
// cycles — the production adapter leaves it unset). Tests that specifically
// exercise a COMPUTE-bound property (fractal quantization, split-K compute
// speedup) dial it up with an explicit multiplier, e.g. set_910b(p, 4096).
static void set_910b(Problem& p, int64_t cube_cost = 1) {
    p.num_cube_cores = 24;
    p.num_vector_cores = 48;
    p.cube_capacity = 128 * 1024;   // L0c accumulator (output)
    p.l1_capacity = 512 * 1024;     // L1/Mat operand pool (cube)
    p.vec_capacity = 192 * 1024;    // UB (vector)
    p.cube_compute_cost = cube_cost;  // grounded cube-compute calibration multiplier
    p.kernel_fill_cost = 10000;     // per-kernel pipeline fill (the dual of eff —
                                    // drives the tiling to ~one kernel/core)
    // Grounded pto-isa A2/A3 coefficients: per-direction bandwidths in GiB/s, core
    // clock, L0 GEMM base tile, vector register size + per-op compute slopes.
    p.cube_freq_hz = 1.85e9;   // core clock (A2A3)
    p.bw_gm_l1   = 135.0;      // GM->L1 operand reload
    p.bw_l0c_gm  = 70.0;       // L0C->GM output store
    p.bw_l1_l0a  = 441.0;      // L1->L0A lhs extract
    p.bw_l1_l0b  = 220.5;      // L1->L0B rhs extract
    p.bw_gm_ub   = 100.9;      // GM->UB vector load
    p.bw_ub_gm   = 188.46;     // UB->GM vector store
    // Realistic A3 aggregate HBM read bandwidth (~900 GiB/s). par() then binds at
    // ~900/135 = 6.7 cores: beyond that, reload-bound matmuls saturate HBM rather than
    // scale linearly. Perf-sim VALIDATED in the saturation regime (pto-isa
    // gml1_multicore: per-core bw = min(135, 900/B) to <=0.4%, aggregate caps at 900).
    // The exact aggregate is a DEVICE-EVAL-PENDING number; 900 is the perf-sim estimate.
    p.hbm_aggregate_gibps = 900.0;
    p.l0_tile_m  = 128;        // L0 GEMM base M (pto-isa oracle)
    p.l0_tile_n  = 256;        // L0 GEMM base N (pto-isa oracle)
    p.vec_reg_bytes = 256;     // vector register size (bytes)
    p.vec_op_head = 14.0;      // per-op pipeline startup cycles
    p.vec_op_tail = 18.0;      // per-op drain cycles
    p.vec_slope_pw = 2.0;      // elementwise cycles/repeat (vmul-ish)
    p.vec_slope_reduce = 14.0; // reduction cycles/repeat (vreducev2)
}

// HBM aggregate read floor (cycles) for one matmul C[N,M]=A[K,M]·B[N,K]: NO tiling can
// read the operands from HBM faster than the shared aggregate bandwidth, so any feasible
// latency is >= (M·K·bytes_a + K·N·bytes_b)/hbm_aggregate. This is the physical invariant
// the realistic HBM cap enforces (validated by pto-isa gml1_multicore) -- it REPLACES the
// pre-cap "fills 24 cube cores" assertions, which assumed unphysical linear core scaling
// (24·135 = 3240 GiB/s, reading 3.6× faster than HBM allows). Beyond the saturation knee
// (~hbm/peak ≈ 6.7 cores) more cores can't lower the feed, so the solver fills cores only
// up to its balanced-tile/saturation point (8–24, not always 24) — but always clears this
// floor. Per-direction cost/byte = freq / (2^30 · bw); see MakeByteCost.
static double hbm_read_floor(const Problem& p, int64_t M, int64_t N, int64_t K,
                             DType a_dtype, DType b_dtype) {
    const double cpb = p.cube_freq_hz / (1024.0 * 1024.0 * 1024.0 * p.hbm_aggregate_gibps);
    return ((double)M * K * dtype_bytes(a_dtype) + (double)K * N * dtype_bytes(b_dtype)) * cpb;
}

// --- A: large pointwise — memory-bound, tile-invariant -----------------------
static void test_A_pointwise_memory_bound() {
    std::cout << "[A] large pointwise — memory-bound, latency == DDR floor\n";
    Problem p;
    p.tensors = {{512, 512}, {512, 512}};
    p.ops = {{OpType::Pointwise, {0}, {1}}};  // trivial compute
    p.fast_memory_capacity = 1 << 22;
    set_910b(p);
    p.kernel_fill_cost = 0;  // isolate the DDR floor (fill is a separate layer)
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0});
    CHECK("A: subgraph builds", (bool)sg);
    double lat = sg->best_cost().latency;
    std::cout << "    best latency = " << lat << "\n";
    // Grounded DDR floor: load T0 (GM->UB) + store T1 (UB->GM), byte-based and
    // tile-invariant. cost-per-byte = freq / (2^30 * bw_GiBps) (see MakeByteCost).
    auto cpb = [&](double bw) { return p.cube_freq_hz / (1024.0 * 1024.0 * 1024.0 * bw); };
    const double bytes = 512.0 * 512.0 * 4.0;
    const double floor = bytes * cpb(p.bw_gm_ub) + bytes * cpb(p.bw_ub_gm);  // ~27491.6
    CHECK_EQ("A: latency == grounded DDR floor (GM->UB load + UB->GM store)", lat, floor);
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
    p.ops = {{OpType::MatMul, {0, 1}, {2}}};  // compute-heavy (2.4M = 24*100k)
    p.fast_memory_capacity = 1 << 20;
    set_910b(p, 4096);  // compute-bound: split-K's win is its COMPUTE parallelization
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
    p.ops = {{OpType::Pointwise, {0}, {1}}, {OpType::Pointwise, {1}, {2}}};
    p.fast_memory_capacity = 1 << 20;
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
    p.ops = {{OpType::MatMul, {0, 1}, {2}}, {OpType::Pointwise, {2}, {3}}};
    p.fast_memory_capacity = 1 << 20;
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
    p.ops = {{OpType::Reduction, {0}, {1}},
             {OpType::Pointwise, {0, 1}, {2}},
             {OpType::Reduction, {2}, {3}},
             {OpType::Pointwise, {2, 3}, {4}}};
    p.fast_memory_capacity = 1 << 24;
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
    p.ops = {{OpType::Reduction, {0}, {1}}};
    p.fast_memory_capacity = 1 << 24;
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
    p.ops = {{OpType::MatMul, {0, 1}, {2}}};       // tiny compute -> memory-bound
    p.fast_memory_capacity = 1 << 24;
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
    p.ops = {{OpType::MatMul, {0, 1}, {2}}};     // compute-heavy
    p.fast_memory_capacity = 1 << 24;
    set_910b(p, 4096);  // compute-bound: split-K's win is its COMPUTE parallelization
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
    p.ops = {{OpType::MatMul, {0, 1}, {2}}};
    p.fast_memory_capacity = 1 << 26;
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
    p.ops = {{OpType::MatMul, {0, 1}, {2}}};     // compute-heavy
    p.fast_memory_capacity = 1 << 24;
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
    p.ops = {{OpType::MatMul, {0, 1}, {2}}};
    p.fast_memory_capacity = 1 << 26;
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
    p.ops = {{OpType::MatMul, {0, 1}, {2}}, {OpType::MatMul, {2, 3}, {4}}};
    p.fast_memory_capacity = 1 << 26;
    set_910b(p);
    return p;
}

// --- two matmuls: chained fusion compute is sane; split-K is per-matmul -------
static void test_two_matmul() {
    std::cout << "[2MM] two-matmul subgraph: chained fusion compute + per-matmul split-K\n";
    // Chained, FAT sink E[128,256] (fills the cores so the fused case needs no
    // heavy split-K), large intermediate C[256,256]. base_cost ignored (geometry).
    auto p = mk_chained(256, 512, 256, 128, 16384 * 512, 16384 * 256);
    // Grounded realistic multiplier (cc=1 from set_910b): the pto-isa fractal model
    // already sits at the DDR-relevant knee for these shapes, so the intermediate
    // round-trip saving is visible (no artificial compute inflation).
    DAG dag = DAG::build(p);
    double m1 = Subgraph::create(p, dag, {0})->best_cost().latency;
    double m2 = Subgraph::create(p, dag, {1})->best_cost().latency;
    double fused = Subgraph::create(p, dag, {0, 1})->best_cost().latency;
    std::cout << "    M1=" << m1 << " M2=" << m2 << " fused=" << fused << "\n";
    // Fat-sink chain in the DDR-relevant regime: fusing keeps the big intermediate
    // T EPHEMERAL on-chip (no DDR round-trip), so fusion strictly helps -- fused <
    // M1 + M2. (For a THIN sink the fused case is forced into split-K or N-tile
    // recompute and can LOSE -- see test_fusion_decision_matrix; the correct outcome.)
    CHECK("2MM: chained (fat sink) fused < M1+M2 (keeps intermediate on-chip)",
          fused < m1 + m2 - 0.5);
    // Per-matmul COMPUTE accounting: in a compute-bound regime (latency ~= compute)
    // the fused kernel does BOTH matmuls' fractals, so it cannot undercut either part
    // (the old bug scaled all compute by the narrower sink output -> fused < M1). The
    // DDR-bound regime above does NOT test this: there fused can legitimately be < M1
    // alone, because M1 by itself pays to write the fat intermediate T to DDR that the
    // fused kernel avoids.
    {
        auto pc = mk_chained(256, 512, 256, 128, 16384 * 512, 16384 * 256);
        set_910b(pc, 4096);  // compute-bound multiplier -- isolate the compute roofline
        pc.kernel_fill_cost = 0;
        DAG dc = DAG::build(pc);
        double c1 = Subgraph::create(pc, dc, {0})->best_cost().latency;
        double c2 = Subgraph::create(pc, dc, {1})->best_cost().latency;
        double cf = Subgraph::create(pc, dc, {0, 1})->best_cost().latency;
        CHECK("2MM: chained fused >= each part (compute counted per-matmul)",
              cf >= c1 - 0.5 && cf >= c2 - 0.5);
    }

    // (2) INDEPENDENT shared-input: Q=X@Wq, K=X@Wk (same X). Both cube, no
    // intermediate between them — fusing only saves re-reading X (negligible
    // vs compute here), so fused == sum.
    {
        Problem q;  // 0X[Kc,M] 1Wq[N,Kc] 2Q[N,M] 3Wk[N,Kc] 4K[N,M]
        int64_t M = 128, Kc = 256, N = 128;
        q.tensors = {{Kc, M}, {N, Kc}, {N, M}, {N, Kc}, {N, M}};
        q.ops = {{OpType::MatMul, {0, 1}, {2}}, {OpType::MatMul, {0, 3}, {4}}};
        q.fast_memory_capacity = 1 << 26;
 set_910b(q, 4096); q.kernel_fill_cost = 0;  // compute-bound regime
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
        a.ops = {{OpType::MatMul, {0, 1}, {2}},
                 {OpType::MatMul, {3, 4}, {5}},
                 {OpType::Pointwise, {2, 5}, {6}}};
        a.fast_memory_capacity = 1 << 26;
 set_910b(a);
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

// --- shared-input reuse: a boundary operand feeding several fused ops is loaded
// ONCE, not once per op (the reload dedup). Run MEMORY-BOUND (low cube_compute_
// cost) so the reload — and the saving — actually drives the latency and flips
// fuse-vs-separate. (The placeholder cube_compute_cost=4096 is compute-bound and
// hides this; real 910B matmul sits below the roofline knee.)
static void test_shared_input_reuse() {
    std::cout << "[REUSE] shared-LHS reload counted once when fused (memory-bound)\n";
    // Two independent matmuls sharing the SAME LHS t0: Q=t0@t1, K=t0@t3. Both
    // tile the same M-band, so both want t0[K, m-band] — one L1 load serves both.
    Problem p;  // t0[K,M] shared LHS; t1,t3 distinct RHS; t2,t4 outputs (same shape)
    int64_t M = 1536, K = 512, N = 256;
    p.tensors = {{K, M}, {N, K}, {N, M}, {N, K}, {N, M}};  // t0, t1, t2=out0, t3, t4=out1
    p.ops = {{OpType::MatMul, {0, 1}, {2}},
             {OpType::MatMul, {0, 3}, {4}}};
    p.fast_memory_capacity = 1 << 26;

    set_910b(p);
    p.cube_compute_cost = 16;   // memory-bound: HBM dominates (vs the 4096 placeholder)
    p.kernel_fill_cost = 0;     // isolate the roofline
    DAG dag = DAG::build(p);
    double q = Subgraph::create(p, dag, {0})->best_cost().latency;       // Q alone
    double k = Subgraph::create(p, dag, {1})->best_cost().latency;       // K alone
    double fused = Subgraph::create(p, dag, {0, 1})->best_cost().latency;  // fused, t0 shared
    std::cout << "    Q=" << q << " K=" << k << " fused=" << fused << " (sum=" << q + k << ")\n";
    // THE FLIP: with the shared-t0 dedup, fused loads t0 once instead of twice, so
    // fusing strictly beats running them separately. Without the dedup, fused == sum
    // (t0 double-counted) and there is no IO benefit to fusing.
    CHECK("REUSE: shared-LHS fused < separate sum (t0 loaded once)", fused < q + k - 1.0);
    CHECK("REUSE: identical matmuls cost the same alone", std::abs(q - k) < 1.0);
    // The whole saving is ~one t0 reload through DDR (the other operands t1,t3 and
    // both outputs are still counted in the fused case).
    CHECK("REUSE: saving is a meaningful fraction of a single matmul", (q + k) - fused > 0.1 * q);
}

// --- grounded machine-cost compute model: pto-isa fractal MACs + vector roofline -
// Cube compute = grounded fractal MACs: repeats = ceil(M/16)*ceil(N/16)*ceil(K/kF),
// kF = 32/dtype_bytes (FP32:8, FP16:16), cyc = 2 (FP32) else 1, x cube_compute_cost
// (the calibration multiplier, default 1). Vector compute = head + slope*repeat +
// tail per op. The cube CAN be compute-bound (dial the multiplier up); the vector is
// DDR-bound on 910B (compute parallelizes across cores while DDR is the shared
// aggregate) -- the realistic regime, so we assert the vector DDR floor instead.
static void test_geometry_compute() {
    std::cout << "[GEOM] grounded compute: cube fractal MACs (compute-bound); vector DDR-bound\n";
    // --- CUBE: C[N=96, M=64], K=2048, FP32. out 96x64 is a 4x6 = 24-region grid that
    // fills the 24 cube cores EXACTLY (one wave); a big multiplier puts it on the
    // compute roofline, so latency = total_MAC_cycles / cores.
    Problem c;
    c.tensors = {{2048, 64}, {96, 2048}, {96, 64}};  // A[K,M] B[N,K] -> C[N,M], FP32
    c.ops = {{OpType::MatMul, {0, 1}, {2}}};  // base_cost ignored (grounded geometry)
    c.fast_memory_capacity = 1 << 26;
    set_910b(c, 10000);  // big cube multiplier -> compute roofline (DDR hidden)
    c.kernel_fill_cost = 0;
    auto rc = Subgraph::create(c, DAG::build(c), {0})->best_cost();
    const double kF = 32.0 / 4.0;  // FP32: 32 / dtype_bytes
    const double repeats = std::ceil(64.0 / 16) * std::ceil(96.0 / 16) * std::ceil(2048.0 / kF);
    const double exp_c = repeats * 2.0 * 10000.0 / rc.cores_used;  // cyc=2 (FP32) * multiplier
    std::cout << "    cube: repeats=" << (long long)repeats << " cores=" << rc.cores_used
              << " -> lat=" << rc.latency << " (expect " << exp_c << ")\n";
    // Compute-bound latency is the MAC roofline PLUS the small L1->L0 extract Phase-D
    // fill (~ext/L per region, a separate hierarchical term << the MACs). Verify the
    // MAC formula to a tight relative tolerance the extract sits comfortably inside;
    // a wrong kF / cyc would miss by >= 2x, far outside this band.
    CHECK("GEOM: cube compute = grounded fractal MACs / cores (compute-bound)",
          rc.compute_bound && std::abs(rc.latency - exp_c) < exp_c * 1e-4);

    // --- VECTOR: pointwise [512,512], FP32. Vector compute parallelizes across cores
    // while DDR is the shared aggregate, so a pointwise is DDR-BOUND -- its latency is
    // the GM<->UB byte floor (load + store), tile-invariant. (The head+slope*repeat+
    // tail compute formula is exercised where it bites -- UB-overflow streaming
    // N_passes -- in the vector sensibility tests; it never drives a simple pointwise.)
    Problem v;
    v.tensors = {{512, 512}, {512, 512}};
    v.ops = {{OpType::Pointwise, {0}, {1}}};
    v.fast_memory_capacity = 1 << 26;
    set_910b(v); v.kernel_fill_cost = 0;
    auto rv = Subgraph::create(v, DAG::build(v), {0})->best_cost();
    auto cpb = [&](double bw) { return v.cube_freq_hz / (1024.0 * 1024.0 * 1024.0 * bw); };
    const double vfloor = 512.0 * 512 * 4 * cpb(v.bw_gm_ub) + 512.0 * 512 * 4 * cpb(v.bw_ub_gm);
    std::cout << "    vector: lat=" << rv.latency << " (DDR floor " << vfloor
              << ", compute_bound=" << rv.compute_bound << ")\n";
    CHECK("GEOM: vector pointwise is DDR-bound (compute parallelizes, DDR aggregate)",
          !rv.compute_bound);
    CHECK("GEOM: vector DDR-bound latency = GM<->UB byte floor (load + store)",
          std::abs(rv.latency - vfloor) < 1.0);
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
        p.ops = {{OpType::MatMul, {0, 1}, {2}}};
        p.fast_memory_capacity = 1LL << 30;
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
    c.ops = {{OpType::MatMul, {0, 1}, {2}}};
    c.fast_memory_capacity = 1 << 28;
    set_910b(c);
    auto rc = Subgraph::create(c, DAG::build(c), {0})->best_cost();
    long long opnd = (long long)rc.config.k * (rc.config.w + rc.config.h) * 4;  // FP32 operand bytes
    std::cout << "    cube out[2048,2048] K=4096: tile[" << rc.config.w << "x" << rc.config.h
              << "] k=" << rc.config.k << " split=" << rc.parallel_split
              << " operand=" << opnd << "B (L1=" << (512 * 1024) << ")\n";
    CHECK("POOL: cube k-tiled to fit L1 (k < full K=4096)", rc.config.k < 4096);
    // Full L1/Mat pool: the operand strip uses all 512 KB (double-buffering
    // streams it as ping-pong sub-strips in the emit -- it does not reserve half).
    CHECK("POOL: cube operand strips fit L1 (512KB)", opnd <= 512 * 1024);
    // Vector: a pointwise tile (in + out resident) must fit the 192 KB UB.
    Problem v;
    v.tensors = {{512, 512}, {512, 512}};
    v.ops = {{OpType::Pointwise, {0}, {1}}};
    v.fast_memory_capacity = 1 << 28;
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
    p.ops = {{OpType::MatMul, {0, 1}, {2}}};
    p.fast_memory_capacity = 1 << 24;
    set_910b(p);
    auto r = Subgraph::create(p, DAG::build(p), {0})->best_cost();
    std::cout << "    pypto L0 (hand-crafted): out tile[16,64] Acc, k=256 (ChooseL0Tile), 1 core sequential\n";
    std::cout << "    our solver (DDR->L1)   : out tile[" << r.config.w << "," << r.config.h
              << "] k=" << r.config.k << " spatial=" << r.num_spatial_tiles
              << " split-K=" << r.parallel_split << " cores=" << r.cores_used << "\n";
    // Thin output (16x64) << 24 cores => our level parallelizes via split-K.
    CHECK("PYPTO: our solver split-Ks the thin L0 matmul across cube cores", r.parallel_split > 1);
    // BSP model: the split-K merge is an additive tail S*out_store, so the solver
    // spreads a LITTLE spatially (here a few tiles) to halve the partial count S and
    // shrink the tail, rather than putting the whole output in one tile and
    // split-K'ing it 24 ways. Under the realistic HBM cap (gml1_multicore) extra split-K
    // can't lower the capped feed, so it saturates HBM below 24 cores (here 12). The
    // invariant is that it PARALLELIZES (spatial x split) and clears the HBM read floor,
    // not that it rigidly fills 24 cores.
    CHECK("PYPTO: solver parallelizes within the HBM-bound core budget",
          r.cores_used > 1 && r.cores_used <= 24
          && r.latency + 1e-6 >= hbm_read_floor(p, 16, 64, 2048,
                                                p.tensors[0].dtype, p.tensors[1].dtype));
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
        p.ops = {{OpType::Reduction, {0}, {1}},
                 {OpType::Pointwise, {0, 1}, {2}},
                 {OpType::Reduction, {2}, {3}},
                 {OpType::Pointwise, {2, 3}, {4}}};
        p.fast_memory_capacity = 1 << 24;
        set_910b(p);
        viz_row("softmax fused", p, {0, 1, 2, 3});
        viz_row("rowmax alone", p, {0});
    }
    {
        Problem p;
        p.tensors = {{4096, 4}, {1, 4}};
        p.ops = {{OpType::Reduction, {0}, {1}}};
        p.fast_memory_capacity = 1 << 24;
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
        // Grounded realistic multiplier (cc=1 from set_910b): the pto-isa fractal
        // model sits at the DDR-relevant knee, so the fuse-vs-split decision is
        // driven by the real intermediate round-trip vs recompute trade-off (no
        // artificial compute inflation that would hide the round-trip for all shapes).
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
    // Small M, but a SMALL intermediate T[64,256] that fits on-chip -> the sink's
    // N2-tiling reuses the held T (no recompute), so fusion still saves the round-trip
    // and WINS. (Under an inflated compute multiplier the recompute looked expensive
    // and this split; at the realistic grounded cc=1 the held-T reuse dominates.)
    auto sm = cmp("small-M (N2-tiled, T held on-chip)", 64, 256, 256, 512);
    CHECK("FDM: small-M chain FUSES (small T held on-chip, no recompute)",
          sm.first < sm.second - 0.5);
    // Wide intermediate T[2048,256]: too big to hold while tiling the sink, so it is
    // recomputed per column-tile and that recompute dwarfs its DDR round-trip -> split.
    auto wide = cmp("wide-C (huge intermediate)", 256, 512, 2048, 256);
    CHECK("FDM: wide-intermediate chain SPLITS (recompute >> round-trip)",
          wide.first > wide.second + 0.5);
}

// --- DFS execution order: post-order from sinks, producer before consumer -----
// The fixed pebbling order the peak-working-set sweep walks (and that gets
// emitted with the solution). For a chain C=(A@B)@D the producer matmul must
// precede the sink; a singleton subgraph is just its own op.
// --- SubgraphStructure: the shared, arch-independent structural layer --------
// SubgraphStructure classifies boundary/ephemeral tensors, sinks, and the DFS
// execution order from (Problem, DAG, ops) ALONE — no machine params. The arch
// cost model (today: Subgraph) composes it. This pins that the extracted
// structure agrees with the live Subgraph on every accepted op set, and that
// structural validity is INDEPENDENT of arch admissibility.
static void test_subgraph_structure() {
    std::cout << "[STRUCT] SubgraphStructure matches Subgraph; arch-independent\n";

    // Chained C=(A@B)@D: op0 produces ephemeral T, op1 is the sink.
    auto pc = mk_chained(256, 512, 256, 128, 1000, 1000);
    DAG dc = DAG::build(pc);
    for (const std::vector<size_t>& ops : {std::vector<size_t>{0, 1},
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{1}}) {
        auto sg = Subgraph::create(pc, dc, ops);
        SubgraphStructure s(pc, dc, ops);
        CHECK("STRUCT: chain subgraph accepted by both", (bool)sg && s.valid());
        if (!sg) continue;
        CHECK("STRUCT: boundary_inputs match", s.boundary_inputs() == sg->boundary_inputs());
        CHECK("STRUCT: boundary_outputs match", s.boundary_outputs() == sg->boundary_outputs());
        CHECK("STRUCT: ephemeral match", s.ephemeral() == sg->ephemeral());
        CHECK("STRUCT: execution_order match", s.execution_order() == sg->execution_order());
    }
    // Hand-checked classification for the fused chain {0,1}:
    //   inputs {0=A,1=B,3=D}, ephemeral {2=T}, output {4=C}, sink op1, order [0,1].
    {
        SubgraphStructure s(pc, dc, {0, 1});
        CHECK("STRUCT: chain inputs = {0,1,3}", s.boundary_inputs() == FlatSet<size_t>{0, 1, 3});
        CHECK("STRUCT: chain ephemeral = {2}", s.ephemeral() == FlatSet<size_t>{2});
        CHECK("STRUCT: chain output = {4}", s.boundary_outputs() == FlatSet<size_t>{4});
        CHECK("STRUCT: chain sinks = {1}", s.sinks().size() == 1 && s.sinks()[0] == 1);
        CHECK("STRUCT: chain order = [0,1]",
              s.execution_order().size() == 2 && s.execution_order()[0] == 0 &&
              s.execution_order()[1] == 1);
    }

    // Empty op set → structurally invalid.
    CHECK("STRUCT: empty ops invalid", !SubgraphStructure(pc, dc, {}).valid());

    // Structural validity is INDEPENDENT of arch admissibility: a mixed
    // cube+vector group is REJECTED by the 910B cost model (Subgraph), yet it is
    // a perfectly well-formed structure (it has a boundary output).
    {
        Problem pm;
        pm.tensors = {{64, 64}, {64, 64}, {64, 64}, {64, 64}};  // 0=A 1=B 2=T 3=out
        pm.ops = {{OpType::MatMul, {0, 1}, {2}},          // op0: cube
                  {OpType::Pointwise, {2}, {3}}};          // op1: vector
        pm.fast_memory_capacity = 1 << 20;
        set_910b(pm);
        DAG dm = DAG::build(pm);
        auto sg = Subgraph::create(pm, dm, {0, 1});
        SubgraphStructure s(pm, dm, {0, 1});
        CHECK("STRUCT: mixed group rejected by arch cost model", !sg.has_value());
        CHECK("STRUCT: mixed group is structurally valid", s.valid());
        CHECK("STRUCT: mixed group ephemeral = {2}", s.ephemeral() == FlatSet<size_t>{2});
        CHECK("STRUCT: mixed group output = {3}", s.boundary_outputs() == FlatSet<size_t>{3});
    }
}

// --- cube+vector fusion: the Ascend910BMixed model ---------------------------
// MM -> PW (C=A@B, D=relu(C)). The homogeneous Ascend910BCost rejects the mixed
// group (unit-homogeneity); Ascend910BMixed permits it and the fused cost beats
// running the two kernels separately — the cube and vector pools overlap (the
// DDR roundtrip latency is hidden) and only one kernel fill is paid. The two
// models differ ONLY in admissibility; a pure cube or pure vector group costs
// identically under both.
static void test_cube_vector_fusion() {
    std::cout << "[FUSE] cube+vector fusion — Ascend910BMixed model\n";
    Problem p;
    // C[N=256,M=256] = A[K=512,M=256] @ B[N=256,K=512];  D = relu(C).
    p.tensors = {{512, 256}, {256, 512}, {256, 256}, {256, 256}};  // A, B, C, D
    p.ops = {{OpType::MatMul, {0, 1}, {2}},
             {OpType::Pointwise, {2}, {3}}};
    p.fast_memory_capacity = 1 << 26;
    set_910b(p, 4096);  // compute-bound matmul so the overlap is the lever
    DAG dag = DAG::build(p);

    // Homogeneous model rejects the mixed group; the mixed model builds it.
    CHECK("FUSE: homogeneous model rejects mixed group",
          !Subgraph::create(p, dag, {0, 1}).has_value());
    auto fused = Ascend910BMixed::create(p, dag, {0, 1});
    CHECK("FUSE: Ascend910BMixed builds the mixed group", (bool)fused);
    if (!fused) return;
    auto fr = fused->best_cost();
    double f = fr.latency;
    // Pure groups: identical cost under either model (the mixed branch is unreached).
    double mm = Subgraph::create(p, dag, {0})->best_cost().latency;
    double pw = Subgraph::create(p, dag, {1})->best_cost().latency;
    std::cout << "    fused=" << f << "  separated=" << (mm + pw)
              << "  (mm=" << mm << " pw=" << pw << ")  cores=" << fr.cores_used << "\n";
    CHECK("FUSE: fused < separated (overlap + one fill)", f < mm + pw);
    // Tiling for units: cores_used = 3 per active unit (1 cube + 2 vector).
    const int eff_units = (fr.num_spatial_tiles < p.num_cube_cores)
                              ? fr.num_spatial_tiles : p.num_cube_cores;
    CHECK("FUSE: cores_used = 3 per active unit", fr.cores_used == 3 * eff_units);
}

// --- mixed kernel tiles for UNITS (1 cube + 2 vector) ------------------------
// Toni's 4-cube/8-vector example: 4 units of {1 cube + 2 vector}. The output is
// tiled into regions over units; within a unit the cube streams its region to
// the 2 vector cores (which split it). An active unit consumes 3 cores, and the
// vector stage runs on 2 cores per unit (vector_compute / (2·eff_units)).
static void test_mixed_unit_tiling() {
    std::cout << "[UNIT] mixed kernel tiles for units (1 cube + 2 vector)\n";
    Problem p;
    // C[64,64] = A[K=128,M=64] @ B[N=64,K=128];  D = relu(C).
    p.tensors = {{128, 64}, {64, 128}, {64, 64}, {64, 64}};  // A, B, C, D
    p.ops = {{OpType::MatMul, {0, 1}, {2}},
             {OpType::Pointwise, {2}, {3}}};
    p.fast_memory_capacity = 1 << 24;
    set_910b(p, 4096);
    p.num_cube_cores = 4;
    p.num_vector_cores = 8;  // 1:2 ratio -> 4 units of {1 cube + 2 vector}
    DAG dag = DAG::build(p);

    auto fused = Ascend910BMixed::create(p, dag, {0, 1});
    CHECK("UNIT: mixed group builds (4 cube + 8 vector)", (bool)fused);
    if (!fused) return;
    auto fr = fused->best_cost();
    const int eff_units = (fr.num_spatial_tiles < p.num_cube_cores)
                              ? fr.num_spatial_tiles : p.num_cube_cores;
    std::cout << "    tiles=" << fr.num_spatial_tiles << " eff_units=" << eff_units
              << " cores_used=" << fr.cores_used << " lat=" << fr.latency << "\n";
    CHECK("UNIT: active units <= n_units (= num_cube_cores)", eff_units <= 4);
    CHECK("UNIT: cores_used = 3 per active unit (1 cube + 2 vector)",
          fr.cores_used == 3 * eff_units);
}

// --- MM->PW mixed kernel variants: the DDR-bound regime ----------------------
// Most MM->PW kernels are memory-bound. DDR-domination shows up as
// compute_bound=false and lat = rounds*fill + ddr_lat (the cube/vector stages
// hidden under DDR). The DDR bytes = boundary inputs + boundary outputs +
// 2x(cube->vector crossing intermediate) — the intermediate round-trips the GM
// ring on 910B. We also print the separated MM+PW cost for comparison.
static void test_mixed_mm_pw_variants() {
    std::cout << "[MIXVAR] MM->PW mixed kernel — DDR-bound regime + variants\n";
    auto mk = [](int64_t M, int64_t N, int64_t K, int64_t cc) {
        Problem p;
        p.tensors = {{K, M}, {N, K}, {N, M}, {N, M}};  // A[K,M] B[N,K] C[N,M] D[N,M]
        p.ops = {{OpType::MatMul, {0, 1}, {2}},
                 {OpType::Pointwise, {2}, {3}}};
        p.fast_memory_capacity = 1 << 26;
        set_910b(p, cc);
        return p;
    };

    // (1) memory-bound matmul (cc=1, K=128 below the compute knee): MM->PW should be
    // DDR-dominated. K sits well below the compute-bound flip (this shape stays DDR-bound
    // up to ~cc=3, a comfortable 3x margin at the realistic cc=1) so the regime is robust,
    // not a knee case. Under the grid-accurate tile count (num_tiles = parts_m*parts_n) a
    // square 256^3 landed right on the knee and tipped compute-bound; K=128 clears it.
    {
        Problem p = mk(256, 256, 128, 1);
        DAG dag = DAG::build(p);
        auto fr = Ascend910BMixed::create(p, dag, {0, 1})->best_cost();
        auto mm = Subgraph::create(p, dag, {0})->best_cost();
        auto pw = Subgraph::create(p, dag, {1})->best_cost();
        const int rounds = (fr.num_spatial_tiles + p.num_cube_cores - 1) / p.num_cube_cores;
        // Grounded DDR is charged at the GM->L1 feed bandwidth (the mixed model lumps
        // all ddr bytes onto bc.reload); cost-per-byte = freq / (2^30 * bw_gm_l1).
        auto cpb = [&](double bw) { return p.cube_freq_hz / (1024.0 * 1024.0 * 1024.0 * bw); };
        const double Cb = 256.0 * 256.0 * 4;             // intermediate C/D bytes (N*M*4; A,B are half at K=128)
        const double twoC = 2.0 * Cb * cpb(p.bw_gm_l1);  // the cube<->vector roundtrip
        std::cout << "  mem-bound : fused lat=" << fr.latency << " compute_bound=" << fr.compute_bound
                  << " ddr=" << fr.ddr_traffic << " tiles=" << fr.num_spatial_tiles
                  << "  | separated mm.ddr=" << mm.ddr_traffic << " pw.ddr=" << pw.ddr_traffic << "\n";
        CHECK("MIXVAR: memory-bound MM->PW is DDR-bound", !fr.compute_bound);
        CHECK("MIXVAR: DDR-bound latency = ddr + rounds*fill (compute hidden)",
              std::abs(fr.latency - (fr.ddr_traffic + rounds * (double)p.kernel_fill_cost)) < 1.0);
        // Under the grounded port-split the crossing roundtrip does NOT sum onto one ddr
        // port: the C write (L0C->GM) and C read (GM->UB) ride SEPARATE overlapping pipes,
        // and ddr_traffic is the MAX over the four ports — so it stays BELOW both the old
        // flat 2C roundtrip and the (2C + D) sum. Fusion still charges the reload + roundtrip
        // + output (not free on 910B), just on independent overlapping pipes.
        // (grounded: mixed_ddr_bound — ports overlap so max not sum; mixed_contention par cap.)
        const double roundtrip_plus_out = (2.0 * Cb + Cb) * cpb(p.bw_gm_l1);  // 2C + D (old flat sum)
        CHECK("MIXVAR: crossing roundtrip splits across overlapping ports (ddr = max, not sum)",
              fr.ddr_traffic > 0.0 && fr.ddr_traffic < twoC - 1.0 &&
                  fr.ddr_traffic < roundtrip_plus_out - 1.0);
        // Fusion still wins (saves a kernel fill + overlaps compute) but does NOT
        // halve the DDR — the win is modest in the DDR-bound regime.
        CHECK("MIXVAR: fused < separated (saves fill + overlap, not DDR)",
              fr.latency < mm.latency + pw.latency);
    }
    // (2) compute-bound matmul (cc=4096): the cube stage dominates.
    {
        Problem p = mk(256, 256, 256, 4096);
        DAG dag = DAG::build(p);
        auto fr = Ascend910BMixed::create(p, dag, {0, 1})->best_cost();
        std::cout << "  compute   : fused lat=" << fr.latency
                  << " compute_bound=" << fr.compute_bound << " ddr=" << fr.ddr_traffic << "\n";
        CHECK("MIXVAR: heavy matmul MM->PW is compute-bound", fr.compute_bound);
    }
    // (3) wide intermediate (big C, cheap matmul): the 2x roundtrip SPLITS onto separate
    // overlapping pipes (L0C->GM write + GM->UB read), so it does NOT dominate ddr — each half
    // caps at its own port (~502 cyc) while the cheap matmul's cube stage (~12.5k) dominates.
    // So under the grounded port-split a wide-C MM->PW is COMPUTE-bound (it was DDR-bound only
    // under the old flat single-port ddr that piled the whole 2C roundtrip onto GM->L1).
    // (grounded: mixed_ddr_bound — the crossing write and read overlap on distinct ports.)
    {
        Problem p = mk(64, 1024, 64, 64);
        DAG dag = DAG::build(p);
        auto fr = Ascend910BMixed::create(p, dag, {0, 1})->best_cost();
        std::cout << "  wide-C    : fused lat=" << fr.latency
                  << " compute_bound=" << fr.compute_bound << " ddr=" << fr.ddr_traffic << "\n";
        CHECK("MIXVAR: wide-intermediate MM->PW is compute-bound (roundtrip splits across ports)",
              fr.compute_bound);
    }
}

// The mixed cube||vector pipeline pays a fill/drain tile ONLY when the sink unit is
// idle at t=0 (every sink-unit op transitively waits on the opposite unit). If the
// sink unit has an "early stage" op whose cone is same-unit + boundary (runs from
// t=0), the fill is ABSORBED into the max. This is a STRUCTURAL property of the op
// graph — independent of tile dims — so we probe the 4 canonical shapes with uniform
// square tiles. Grounded by mixed_tile_study's shape sweep (v->c->v / c->v->c absorb;
// c->v / v->c add). Guards against the old output_unit_op_count==1 heuristic, which
// mis-classified same-unit tails (c->v->v has 2 vector ops yet still idles at t=0).
static void test_mixed_pipeline_stages() {
    std::cout << "[MIXSTAGE] fill absorption across the 4 canonical mixed shapes\n";
    const Tensor sq{128, 128};  // uniform square tile — dims don't affect the (structural)
                                // stage classification; small + equal so every op is feasible.
    using OT = OpType;
    auto absorbed = [&](std::vector<Tensor> tensors, std::vector<Op> ops,
                        std::vector<size_t> group) {
        Problem p;
        p.tensors = std::move(tensors);
        p.ops = std::move(ops);
        p.fast_memory_capacity = 1 << 26;
        set_910b(p, /*cc=*/1);
        DAG dag = DAG::build(p);
        auto mixed = Ascend910BMixed::create(p, dag, std::move(group));
        return mixed ? mixed->best_cost() : CostResult{};  // nullopt -> feasible stays false
    };

    // c->v->v : MatMul -> PW -> PW. Sink = vector; BOTH vector ops transitively depend
    // on the cube, so the vector unit is idle at t=0 -> fill ADDS (2-stage).
    {
        auto fr = absorbed({sq, sq, sq, sq, sq},
                           {{OT::MatMul, {0, 1}, {2}}, {OT::Pointwise, {2}, {3}}, {OT::Pointwise, {3}, {4}}},
                           {0, 1, 2});
        CHECK("MIXSTAGE: c->v->v feasible", fr.feasible);
        CHECK("MIXSTAGE: c->v->v pays 2-stage fill (same-unit tail still idles at t=0)",
              !fr.pipeline_fill_absorbed);
    }
    // v->v->c : PW -> PW -> MatMul. Sink = cube; the matmul waits on the vector prologue
    // chain -> cube idle at t=0 -> fill ADDS (2-stage). t3 = boundary matmul operand.
    {
        auto fr = absorbed({sq, sq, sq, sq, sq},
                           {{OT::Pointwise, {0}, {1}}, {OT::Pointwise, {1}, {2}}, {OT::MatMul, {2, 3}, {4}}},
                           {0, 1, 2});
        CHECK("MIXSTAGE: v->v->c feasible", fr.feasible);
        CHECK("MIXSTAGE: v->v->c pays 2-stage fill (not absorbed)", !fr.pipeline_fill_absorbed);
    }
    // v->c->v : PW -> MatMul -> PW. Sink = vector; the PROLOGUE pw reads boundary inputs
    // and runs at t=0 (an earlier vector stage) -> fill ABSORBED (3-stage). t2 = boundary
    // matmul operand.
    {
        auto fr = absorbed({sq, sq, sq, sq, sq},
                           {{OT::Pointwise, {0}, {1}}, {OT::MatMul, {1, 2}, {3}}, {OT::Pointwise, {3}, {4}}},
                           {0, 1, 2});
        CHECK("MIXSTAGE: v->c->v feasible", fr.feasible);
        CHECK("MIXSTAGE: v->c->v absorbs fill (prologue vector runs at t=0)",
              fr.pipeline_fill_absorbed);
    }
    // c->v->c : MatMul -> PW -> MatMul. Sink = cube; the FIRST matmul reads boundary
    // inputs and runs at t=0 (an earlier cube stage) -> fill ABSORBED (3-stage). t4 =
    // boundary second-matmul operand.
    {
        auto fr = absorbed({sq, sq, sq, sq, sq, sq},
                           {{OT::MatMul, {0, 1}, {2}}, {OT::Pointwise, {2}, {3}}, {OT::MatMul, {3, 4}, {5}}},
                           {0, 1, 2});
        CHECK("MIXSTAGE: c->v->c feasible", fr.feasible);
        CHECK("MIXSTAGE: c->v->c absorbs fill (first matmul runs at t=0)",
              fr.pipeline_fill_absorbed);
    }
    // c->v->c->v : a depth-3 multi-round-trip. The `max` overlap is the SKEWED cost,
    // valid only for a single round-trip (depth <= 2); a depth-3 chain demotes to
    // sequential, so the admission gate REJECTS it (create -> nullopt -> infeasible)
    // and the partitioner cuts it into separate kernels. Guards the skewability gate.
    {
        auto fr = absorbed({sq, sq, sq, sq, sq, sq, sq},
                           {{OT::MatMul, {0, 1}, {2}}, {OT::Pointwise, {2}, {3}},
                            {OT::MatMul, {3, 4}, {5}}, {OT::Pointwise, {5}, {6}}},
                           {0, 1, 2, 3});
        CHECK("MIXSTAGE: c->v->c->v (depth 3) rejected as non-skewable", !fr.feasible);
    }
}

// Sink split-K in a mixed kernel. The sink matmul splits its contraction across idle
// CUBE cores — atomic-add partials with no merge barrier, so the cores stay independent
// (exactly the base cube split-K; the vector overlaps orthogonally). ONLY the sink is
// ever split, and only when it is the SOLE matmul: a vector-sink (c->v) does not split,
// nor does a multi-matmul cube sink (c->v->c). A small output (few tiles) leaves idle
// cube cores for a large-K sink to recruit.
static void test_mixed_sink_split_k() {
    std::cout << "[MIXSPLIT] cube-sink mixed kernel split-Ks the sole sink matmul\n";
    auto run = [&](std::vector<Tensor> ts, std::vector<Op> ops, std::vector<size_t> grp, int64_t cc) {
        Problem p; p.tensors = std::move(ts); p.ops = std::move(ops);
        p.fast_memory_capacity = 1 << 26; set_910b(p, cc);
        DAG dag = DAG::build(p);
        auto m = Ascend910BMixed::create(p, dag, std::move(grp));
        return m ? m->best_cost() : CostResult{};
    };
    using OT = OpType;
    // v->c: PW prologue on the matmul LHS -> matmul SINK. Output [N=256, M=64] is few
    // tiles, so idle cube cores; the sole sink matmul split-Ks across them.
    {
        auto fr = run({{128, 64}, {128, 64}, {256, 128}, {256, 64}},  // A0 A1 B[N,K] C[N,M], K=128
                      {{OT::Pointwise, {0}, {1}}, {OT::MatMul, {1, 2}, {3}}}, {0, 1}, /*cc=*/256);
        CHECK("MIXSPLIT: v->c feasible", fr.feasible);
        CHECK("MIXSPLIT: v->c cube-sink split-Ks the idle cores", fr.parallel_split > 1);
        CHECK("MIXSPLIT: v->c split recruits cube cores beyond spatial tiling",
              fr.cores_used > 3 * fr.num_spatial_tiles);
    }
    // c->v: same matmul but a vector EPILOGUE (vector sink). The sink is not the matmul,
    // so split-K is sink-only -> no split.
    {
        auto fr = run({{128, 64}, {256, 128}, {256, 64}, {256, 64}},  // A[K,M] B[N,K] C[N,M] D
                      {{OT::MatMul, {0, 1}, {2}}, {OT::Pointwise, {2}, {3}}}, {0, 1}, /*cc=*/256);
        CHECK("MIXSPLIT: c->v feasible", fr.feasible);
        CHECK("MIXSPLIT: c->v vector-sink does NOT split (sink-only)", fr.parallel_split == 1);
    }
    // c->v->c: cube sink but TWO matmuls. Only the sink may split, and the rule is
    // single-matmul-only (splitting a mid-kernel matmul is never allowed) -> no split.
    {
        const Tensor sq{128, 128};
        auto fr = run({sq, sq, sq, sq, sq, sq},
                      {{OT::MatMul, {0, 1}, {2}}, {OT::Pointwise, {2}, {3}}, {OT::MatMul, {3, 4}, {5}}},
                      {0, 1, 2}, /*cc=*/256);
        CHECK("MIXSPLIT: c->v->c feasible", fr.feasible);
        CHECK("MIXSPLIT: c->v->c multi-matmul cube sink does NOT split", fr.parallel_split == 1);
    }
}

// A single-tile kernel has no second tile to skew against, so even a 3-stage shape
// (whose fill is normally ABSORBED) pays the un-overlapped SUM at one tile -- the
// mixed_tile_study NT=1 row measured overlap_factor 0.00. This guards the num_tiles>=2
// fill gate: without it, a single-tile 3-stage kernel is credited full overlap (the plain
// max), which the sim contradicts and which biases the solver toward the low-batch
// flash-decode corner. We compare a balanced compute-bound c->v->c (MM -> PW*16 -> MM) at
// one tile vs two tiles: the WALL (latency minus the per-launch fill) ratio is ~3.3 (the
// single tile is the sum), whereas the old plain-max gave exactly 2.0 (pure tiling).
static void test_mixed_single_tile_no_skew() {
    std::cout << "[MIXNT1] single-tile 3-stage kernel pays the un-overlapped sum\n";
    const Tensor sq{128, 128};
    const int P = 16;  // pointwise chain length: balances vec_stage against the two matmuls
    Problem p;
    p.tensors.assign(P + 5, sq);  // MM1{0,1}->2 ; PW chain 2->..->(2+P) ; sink MM2{(2+P),(3+P)}->(4+P)
    p.ops.push_back({OpType::MatMul, {0, 1}, {2}});
    for (int i = 0; i < P; ++i) p.ops.push_back({OpType::Pointwise, {(size_t)(2 + i)}, {(size_t)(3 + i)}});
    p.ops.push_back({OpType::MatMul, {(size_t)(2 + P), (size_t)(3 + P)}, {(size_t)(4 + P)}});
    p.fast_memory_capacity = 1 << 26;
    set_910b(p, /*cc=*/1);  // low compute multiplier so the matmuls don't dwarf the vector
    DAG dag = DAG::build(p);
    std::vector<size_t> grp;
    for (size_t i = 0; i < p.ops.size(); ++i) grp.push_back(i);
    auto m = Ascend910BMixed::create(p, dag, grp);
    auto wall = [&](TileConfig cfg) {
        auto fr = m->compute_cost(cfg, {}, {});
        const int r = (fr.num_spatial_tiles + p.num_cube_cores - 1) / p.num_cube_cores;
        return std::make_pair(fr, fr.latency - (double)r * (double)p.kernel_fill_cost);  // pipeline wall
    };
    auto [f1, w1] = wall(TileConfig{128, 128, 128});  // one output tile
    auto [f2, w2] = wall(TileConfig{128, 64, 128});   // two output tiles
    CHECK("MIXNT1: single tile does NOT absorb fill (no successor to skew against), compute-bound",
          f1.feasible && !f1.pipeline_fill_absorbed && f1.compute_bound && f1.num_spatial_tiles == 1);
    CHECK("MIXNT1: the SAME shape at 2 tiles DOES absorb (confirms it is a 3-stage shape)",
          f2.feasible && f2.pipeline_fill_absorbed && f2.num_spatial_tiles == 2);
    // The single tile is charged the un-overlapped sum (cube+vec), so wall1/wall2 > 2.0 (the
    // plain-max pure-tiling ratio the old code gave); ~3.3 here. Threshold 2.3 leaves margin.
    CHECK("MIXNT1: single-tile wall is the un-overlapped sum (ratio > 2.3, not the 2.0 plain-max)",
          w1 > 2.3 * w2);
}

// A non-uniform grid's cube_stage is the LPT makespan (busiest core = biggest region), not the
// flat total/eff_units average -- which under-predicts an imbalanced grid (up to ~2x at one
// region/core). We tile a W=48 output (3 fractals) two ways: whole (1 tile = total cube work T)
// and a parts_n=2 grid, which partition_axis splits [32,16] (big=2 fractals, small=1). The two
// regions land one-per-core, so the makespan is the BIG [32]-region = 2/3 T -- ABOVE the flat
// average T/2 the old code charged, and below T. Compute-bound so cube_stage drives the wall.
static void test_mixed_grid_makespan() {
    std::cout << "[MIXGRID] non-uniform grid cube_stage is the busiest-region makespan, not the average\n";
    Problem p;
    p.tensors = {{32, 32}, {48, 32}, {48, 32}, {48, 32}};  // A[K,M] B[N,K] C[N,M] D[N,M], out W=48
    p.ops = {{OpType::MatMul, {0, 1}, {2}}, {OpType::Pointwise, {2}, {3}}};
    p.fast_memory_capacity = 1 << 26;
    set_910b(p, /*cc=*/256);  // compute-bound: cube_stage drives the wall
    DAG dag = DAG::build(p);
    auto m = Ascend910BMixed::create(p, dag, {0, 1});
    auto wall = [&](TileConfig cfg) {
        auto fr = m->compute_cost(cfg, {}, {});
        const int r = (fr.num_spatial_tiles + p.num_cube_cores - 1) / p.num_cube_cores;
        return std::make_pair(fr, fr.latency - (double)r * (double)p.kernel_fill_cost);
    };
    auto [f1, total] = wall(TileConfig{48, 32, 32});          // whole output: 1 tile -> total work T
    auto [fg, grid] = wall(TileConfig{32, 32, 32, 1, 2, 0});  // grid parts_n=2 -> [32,16] regions
    CHECK("MIXGRID: single-tile and grid configs feasible, grid is 2 tiles compute-bound",
          f1.feasible && fg.feasible && fg.compute_bound && fg.num_spatial_tiles == 2);
    // Makespan = the big [32]-region (2/3 T), above the flat average T/2, below the total T.
    CHECK("MIXGRID: non-uniform grid pays the busiest-region makespan, not the flat average",
          grid > 1.2 * (total / 2.0) && grid < total);
}

// The MULTI-MATMUL case (reviewer concern (b)): a c->v->c on a non-uniform grid exercises the
// PER-REGION recompute of the cube stage. The region work is max(Σ MAC, Σ extract) recomputed at
// each region extent, NOT base_cube_work * area_fraction, so the makespan is not just proportional
// to output area: with two matmuls + fractal padding, the big [32]-region of a W=48->[32,16] grid
// costs ~0.80 T (above the 0.67 T an area fraction would give), while a flat average charges T/2.
static void test_mixed_grid_makespan_multimatmul() {
    std::cout << "[MIXGRID2] non-uniform grid on a c->v->c uses the per-region cube recompute\n";
    Problem p;
    p.tensors = {{32, 32}, {32, 32}, {32, 32}, {32, 32}, {48, 32}, {48, 32}};
    // MM1{0,1}->2 (C1[32,32]) ; PW{2}->3 ; MM2{3,4}->5 (C2[48,32], contraction N1=32) — out W=48
    p.ops = {{OpType::MatMul, {0, 1}, {2}}, {OpType::Pointwise, {2}, {3}}, {OpType::MatMul, {3, 4}, {5}}};
    p.fast_memory_capacity = 1 << 26;
    set_910b(p, /*cc=*/256);  // compute-bound: the cube makespan drives the wall
    DAG dag = DAG::build(p);
    auto m = Ascend910BMixed::create(p, dag, {0, 1, 2});
    auto wall = [&](TileConfig cfg) {
        auto fr = m->compute_cost(cfg, {}, {});
        const int r = (fr.num_spatial_tiles + p.num_cube_cores - 1) / p.num_cube_cores;
        return std::make_pair(fr, fr.latency - (double)r * (double)p.kernel_fill_cost);
    };
    auto [f1, total] = wall(TileConfig{48, 32, 32});          // whole output -> total cube work T
    auto [fg, grid] = wall(TileConfig{32, 32, 32, 1, 2, 0});  // grid parts_n=2 -> [32,16] regions
    CHECK("MIXGRID2: c->v->c single-tile + grid feasible, grid 2 tiles compute-bound",
          f1.feasible && fg.feasible && fg.compute_bound && fg.num_spatial_tiles == 2);
    // The per-region makespan exceeds BOTH the flat average (T/2) and the area fraction (2/3 T),
    // and stays below the sequential total T.
    CHECK("MIXGRID2: per-region makespan > area-fraction (imbalance not just proportional to area)",
          grid > 0.72 * total && grid < total);
}

// Fused FLASH ATTENTION: a mixed QK->softmax kernel keeps the reduced (keys) axis RESIDENT and
// PINNED (parts_n=1), tiling only the query axis. The [query_tile x keys] scores overflow UB, so
// feasibility comes from ONLINE streaming of the keys axis (flash), not materialization -- a
// reduction can be mid-kernel iff its reduced axis is not split across cores. The matmul's own
// contraction rides seq-k, a separate axis. Guards the grid pinning (step 1) + the streaming
// surcharge (step 3) that make this feasible AND correctly costed.
static void test_mixed_flash_attention() {
    std::cout << "[MIXFA] fused QK->softmax pins the reduced axis + streams online (flash attention)\n";
    Problem p; int64_t d = 64, S = 2048, B = 128;   // Q@K^T -> softmax(over keys) ; d=64, S_kv=2048
    p.tensors = {{d, B}, {S, d}, {S, B}, {1, B}, {S, B}, {1, B}, {S, B}, {d, S}, {d, B}};
    p.ops = {{OpType::MatMul, {0, 1}, {2}}, {OpType::Reduction, {2}, {3}},
             {OpType::Pointwise, {2, 3}, {4}}, {OpType::Reduction, {4}, {5}},
             {OpType::Pointwise, {4, 5}, {6}}, {OpType::MatMul, {6, 7}, {8}}};
    p.fast_memory_capacity = 1LL << 30;
    set_910b(p);
    DAG dag = DAG::build(p);
    auto sg = Ascend910BMixed::create(p, dag, {0, 1, 2, 3, 4});  // QK -> softmax (fused mixed)
    CHECK("MIXFA: fused QK->softmax is admissible (mixed cube+vector)", (bool)sg);
    auto fr = sg->best_cost();
    CHECK("MIXFA: fused QK->softmax is feasible", fr.feasible && std::isfinite(fr.latency));
    // keys = the reduced axis = width (S->1); the grid must PIN it (parts_n == 1) and tile only
    // the query axis (parts_m > 1). Without step 1's pinning it split keys -> infeasible.
    CHECK("MIXFA: reduced (keys) axis pinned (parts_n==1), queries tiled (parts_m>1)",
          fr.config.parts_n == 1 && fr.config.parts_m > 1);
    // Feasibility is via STREAMING: the materialized scores overflow UB, so keys is streamed
    // online (flash), not resident whole.
    CHECK("MIXFA: feasibility comes from online streaming (scores overflow UB)",
          sg->vector_peak_ub(fr.config, {}, {}) > p.vec_capacity
              && sg->vector_stream(fr.config, {}, {}).chunk > 0);
}

// Split-K (base lone matmul and mixed cube-sink) is MODEL-AHEAD of the AutoFuse emit. The
// buildable flag Problem::allow_model_ahead_split_k gates it: default true (analytic) credits
// parallel_split > 1; false (buildable) forces S=1 so best_cost never selects an unemittable
// split. CostResult::uses_model_ahead_split_k flags a chosen split. This proves the flag
// disables the split path for BOTH base and mixed while leaving the analytic result intact.
static void test_model_ahead_split_k_flag() {
    std::cout << "[MIXBUILD] buildable flag gates model-ahead split-K (base + mixed)\n";
    // (1) base lone matmul, a FIXED split_k=2 config (the (P,Q,S) schedule path) through
    // compute_cost directly -- best_cost prefers spatial tiling for these shapes, so we
    // exercise the split path explicitly. C[16,16] K=2048: 1 tile, split the contraction 2 ways.
    {
        Problem p;
        p.tensors = {{2048, 16}, {16, 2048}, {16, 16}};  // A[K,M] B[N,K] C[N,M], K=2048
        p.ops = {{OpType::MatMul, {0, 1}, {2}}};
        p.fast_memory_capacity = 1 << 24; set_910b(p, 4096);
        DAG dag = DAG::build(p);
        const TileConfig split_cfg{16, 16, 2048, 1, 1, 2};  // grid 1x1, fixed split_k=2
        auto analytic = Subgraph::create(p, dag, {0})->compute_cost(split_cfg, {}, {});
        p.allow_model_ahead_split_k = false;  // buildable mode
        auto buildable = Subgraph::create(p, dag, {0})->compute_cost(split_cfg, {}, {});
        CHECK("MIXBUILD: base analytic honors the fixed split_k=2 (flagged model-ahead)",
              analytic.feasible && analytic.parallel_split == 2 && analytic.uses_model_ahead_split_k);
        CHECK("MIXBUILD: base buildable forces S=1 despite cfg.split_k=2 (flag respected)",
              buildable.feasible && buildable.parallel_split == 1 && !buildable.uses_model_ahead_split_k);
        CHECK("MIXBUILD: base analytic <= buildable (the gated split only helps)",
              analytic.latency <= buildable.latency);
    }
    // (2) mixed cube-sink v->c (the MIXSPLIT shape): split-Ks the sole sink matmul.
    {
        Problem p;
        p.tensors = {{128, 64}, {128, 64}, {256, 128}, {256, 64}};  // A0 A1 B[N,K] C[N,M]
        p.ops = {{OpType::Pointwise, {0}, {1}}, {OpType::MatMul, {1, 2}, {3}}};
        p.fast_memory_capacity = 1 << 26; set_910b(p, 256);
        DAG dag = DAG::build(p);
        auto analytic = Ascend910BMixed::create(p, dag, {0, 1})->best_cost();
        p.allow_model_ahead_split_k = false;  // buildable mode
        auto buildable = Ascend910BMixed::create(p, dag, {0, 1})->best_cost();
        CHECK("MIXBUILD: mixed analytic cube-sink split-Ks (S>1, flagged model-ahead)",
              analytic.parallel_split > 1 && analytic.uses_model_ahead_split_k);
        CHECK("MIXBUILD: mixed buildable forces S=1 (no model-ahead split)",
              buildable.parallel_split == 1 && !buildable.uses_model_ahead_split_k);
    }
}

// --- mixed feasibility: both pools STREAM to fit (cube seq-k, vector chunks) --
// A fused MM->PW needs both pools, but each streams to fit like its homogeneous
// counterpart: the cube seq-k-slices the contraction to fit L1, and the vector
// streams its tile through UB. So a small UB does NOT wall fusion — the PW streams
// (free, elementwise) — and MM->PW fuses across UB sizes. (The genuine hard
// constraint is a held cube L1 operand band; see test_mixed_multistage_pebble.)
static void test_mixed_two_pool_feasibility() {
    std::cout << "[2POOL] mixed feasibility: vector streams UB, cube seq-k fits L1\n";
    auto mk = [](int64_t ub_bytes) {
        Problem p;
        p.tensors = {{256, 256}, {256, 256}, {256, 256}, {256, 256}};  // A B C D, FP32
        p.ops = {{OpType::MatMul, {0, 1}, {2}}, {OpType::Pointwise, {2}, {3}}};
        p.fast_memory_capacity = 1 << 26;
        set_910b(p, 64);
        p.l1_capacity = 512 * 1024;
        p.cube_capacity = 128 * 1024;
        p.vec_capacity = ub_bytes;
        return p;
    };

    // Even a 16KB UB (a 128x128 PW tile is 8x too big to materialize) does NOT make
    // fusion infeasible — the vector stage streams its tile through UB.
    {
        Problem p = mk(16 * 1024);
        DAG dag = DAG::build(p);
        auto sg = Ascend910BMixed::create(p, dag, {0, 1});
        CHECK("2POOL: mixed group builds", (bool)sg);
        if (!sg) return;
        CHECK("2POOL: large shared tile feasible at tiny UB (PW streams)",
              sg->is_feasible(TileConfig{128, 128, 256}));
        CHECK("2POOL: small shared tile also feasible",
              sg->is_feasible(TileConfig{64, 64, 256}));
    }

    // MM->PW fuses across UB sizes — the PW streams either way, so the tight UB
    // costs nothing extra (free for a pointwise) and fuses identically to roomy.
    {
        Problem roomy = mk(256 * 1024), tight = mk(16 * 1024);
        DAG dr = DAG::build(roomy), dt = DAG::build(tight);
        auto fr = Ascend910BMixed::create(roomy, dr, {0, 1})->best_cost();
        auto ft = Ascend910BMixed::create(tight, dt, {0, 1})->best_cost();
        double mm = Subgraph::create(roomy, dr, {0})->best_cost().latency;  // pure cube: UB-independent
        double pw = Subgraph::create(roomy, dr, {1})->best_cost().latency;
        std::cout << "  roomy-UB fused=" << fr.latency << " tile=" << fr.config.w << "x" << fr.config.h
                  << "  tight-UB fused=" << ft.latency << " tile=" << ft.config.w << "x" << ft.config.h
                  << "  separated=" << (mm + pw) << "\n";
        CHECK("2POOL: roomy UB -> fusion wins", fr.latency < mm + pw);
        CHECK("2POOL: tight UB still fuses (vector streams, no UB wall)", ft.latency < mm + pw);
        CHECK("2POOL: tight UB fuses identically to roomy (PW stream is free)",
              std::abs(ft.latency - fr.latency) < 1.0);
    }
}

// --- mixed pebble SWEEP charges held same-unit bands (multi-stage) -----------
// The per-op check saw each op alone; the interval-overlap sweep charges an
// intermediate that stays resident across later ops. (1) MM1->MM2->PW: T1=A@B is
// MM2's operand, so it is a held L1 band during MM2 (large h overflows L1).
// (2) MM->PW1->PW2: T=relu(C) is a vector->vector UB band held into PW2.
static void test_mixed_multistage_pebble() {
    std::cout << "[MSWEEP] mixed pebble sweep charges held same-unit bands\n";

    // (1) Cube L1 band: T1=A@B feeds MM2 as an operand -> L1-resident band.
    {
        Problem p;
        // A[K1,M] B[N1,K1] -> T1[N1,M];  T1[N1,M] D[N2,N1] -> C[N2,M];  E=relu(C).
        p.tensors = {{256, 256}, {1024, 256}, {1024, 256}, {256, 1024}, {256, 256}, {256, 256}};
        p.ops = {{OpType::MatMul, {0, 1}, {2}},   // T1 = A @ B
                 {OpType::MatMul, {2, 3}, {4}},   // C  = T1 @ D
                 {OpType::Pointwise, {4}, {5}}};   // E  = relu(C)
        p.fast_memory_capacity = 1 << 26;
        set_910b(p, 64);
        p.l1_capacity = 512 * 1024;
        p.cube_capacity = 128 * 1024;
        p.vec_capacity = 192 * 1024;
        DAG dag = DAG::build(p);
        auto sg = Ascend910BMixed::create(p, dag, {0, 1, 2});
        CHECK("MSWEEP: MM1->MM2->PW mixed group builds", (bool)sg);
        if (sg) {
            // T1 L1 band = T1.width(1024) * min(cfg.h,256) * 4:
            //   h=16  -> 1024*16*4  = 64KB     (+ strips) <= 512KB L1 -> feasible
            //   h=128 -> 1024*128*4 = 512KB    (+ strips) >  512KB L1 -> infeasible
            CHECK("MSWEEP: small h fits (held T1 L1 band small)",
                  sg->is_feasible(TileConfig{128, 16, 1024}));
            CHECK("MSWEEP: large h infeasible (held T1 L1 band overflows L1)",
                  !sg->is_feasible(TileConfig{128, 128, 1024}));
        }
    }

    // (2) Vector->vector UB band: T=relu(C) feeds PW2 -> UB-resident band.
    {
        Problem p;
        // A[K,M] B[N,K] -> C[N,M];  T=relu(C);  D=gelu(T).  N=M=K=256.
        p.tensors = {{256, 256}, {256, 256}, {256, 256}, {256, 256}, {256, 256}};
        p.ops = {{OpType::MatMul, {0, 1}, {2}},   // C = A @ B
                 {OpType::Pointwise, {2}, {3}},    // T = relu(C)
                 {OpType::Pointwise, {3}, {4}}};   // D = gelu(T)
        p.fast_memory_capacity = 1 << 26;
        set_910b(p, 64);
        p.l1_capacity = 512 * 1024;
        p.cube_capacity = 128 * 1024;
        p.vec_capacity = 48 * 1024;  // tight UB
        DAG dag = DAG::build(p);
        auto sg = Ascend910BMixed::create(p, dag, {0, 1, 2});
        CHECK("MSWEEP: MM->PW1->PW2 mixed group builds", (bool)sg);
        if (sg) {
            // The held vector->vector T band + transient overflow UB at 128x64
            // (2*128*64*4 = 64KB > 48KB) when materialized — but the vector stage
            // STREAMS the tile through UB, so both tiles are feasible. (A held
            // vector band is relieved by streaming; only a held CUBE L1 operand
            // band — case (1) — is a hard, un-streamable constraint.)
            CHECK("MSWEEP: 64x64 fits UB (held T band materializes)",
                  sg->is_feasible(TileConfig{64, 64, 256}));
            CHECK("MSWEEP: 128x64 feasible (held T band streams through UB)",
                  sg->is_feasible(TileConfig{128, 64, 256}));
        }
    }
}

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
    // (1) NO k-split: small contractions, sink large enough to fill cores. Realistic
    // grounded multiplier (cc=1) -- an inflated multiplier would make a marginal
    // split-K look attractive even though the sink already fills the cores spatially.
    auto nok = run(256, 256, 64, 512, 1);
    CHECK("KVAR/no-k: spatial tile within sink output (unified grid)", grid_within_output(nok));
    CHECK("KVAR/no-k: sink not parallel-split (spatial fill)", nok.split == 1);
    CHECK("KVAR/no-k: internal runs full K1 (no seq-slice)", nok.k0 == nok.K1);
    CHECK("KVAR/no-k: sink runs full K (no seq-slice)", nok.k1 == nok.N1);
    // (2) INTERNAL single-core k-split: huge K1 forces the internal matmul to
    // slice its contraction on one core; the sink still fills cores spatially.
    // M=384 = 24 M-fractals -> the (24,1) grid fills the cores with eff_par 24
    // EXACTLY, so the sink needs no parallel split. (At M=1024 = 64 M-fractals
    // the (24,1) grid is coarse, eff_par 21.3, and the model correctly prefers a
    // split-K core-fill instead -- see doc section 7 / the GEOM 24-divisibility note.)
    auto ink = run(384, 8192, 64, 1024, 4096);
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
    p.ops = {{OpType::Pointwise, {0}, {1}},
             {OpType::Reduction, {1}, {2}}};
    p.fast_memory_capacity = 1 << 24;

    set_910b(p);
    DAG dag = DAG::build(p);
    auto sg = Subgraph::create(p, dag, {0, 1});
    const int64_t UB = 192 * 1024;
    const int64_t xtile8 = 4096LL * 8 * 4;                 // boundary x tile at h=8 = 131072
    int64_t peak8 = sg->vector_peak_ub({4096, 8, 0});
    CHECK("VBAND: ephemeral e adds on top of the boundary input", peak8 > xtile8);
    CHECK("VBAND: e band overflows UB where boundary-only would fit",
          peak8 > UB && xtile8 < UB);
    // A sub-granule free tile does NOT escape the overflow: the emit allocates DMA-block-padded
    // tiles, so h=2 pads to the same granule floor as h=8 (which assertion 2 shows overflows UB).
    // A large reduced axis fits UB only by STREAMING it (chunked), not by shrinking the free tile.
    CHECK("VBAND: streaming the reduced axis fits UB",
          sg->vector_peak_ub({4096, 8, 0}, {}, {}, /*reduce_chunk=*/64, /*stream_axis=*/1) <= UB);
}

// --- reduction parallel split is SINK-ONLY -----------------------------------
// A bare rowmax (reduction IS the sink) may split its reduced axis across cores.
// A chain whose sink is a POINTWISE with an internal reduction may NOT — the
// internal reduction's partials would round-trip DDR (breaking ephemerality), so
// the model leaves it spatial-only (a cut would be the partitioner's job).
static void test_reduction_sink_gating() {
    std::cout << "[RGATE] reduced-axis split only when the reduction is the sink\n";
    auto mk = []() { Problem p;
 return p; };
    // (a) bare rowmax, few rows -> sink reduction -> splits across cores.
    Problem a = mk();
    a.tensors = {{4096, 4}, {1, 4}};
    a.ops = {{OpType::Reduction, {0}, {1}}};
    a.fast_memory_capacity = 1 << 24; set_910b(a);
    DAG da = DAG::build(a);
    CHECK("RGATE: bare reduction sink splits the reduced axis",
          Subgraph::create(a, da, {0})->best_cost().parallel_split > 1);
    // (b) x -> m=rowmax(x) -> y=sub(x,m): pointwise SINK, internal reduction, few
    // rows. Cannot fill cores (no split allowed) -> parallel_split stays 1.
    Problem b = mk();
    b.tensors = {{4096, 4}, {1, 4}, {4096, 4}};
    b.ops = {{OpType::Reduction, {0}, {1}},
             {OpType::Pointwise, {0, 1}, {2}}};
    b.fast_memory_capacity = 1 << 24; set_910b(b);
    DAG db = DAG::build(b);
    auto sg = Subgraph::create(b, db, {0, 1});
    CHECK("RGATE: internal-reduction chain (pw sink) does NOT split",
          sg && sg->best_cost().parallel_split == 1);
}

// --- C2: a STREAMED reduction sink fills via the free axis, not a split -------
// A reduction whose reduced axis overflows UB lowers (in the AutoFuse emit) to a
// single-core chunk-accumulation loop parallelized over the FREE axis ALONE — the
// emit's stream path returns before the S2 atomic-add split, so no cross-core
// reduced-axis split is ever realized. The model must price it at
// parallel_split == 1 and fill cores via the free-axis grid (parts_n), NOT via a
// fictional reduced-axis split. Device probe C2: costing the split for a streamed
// reduction inverted the argmin (split-heavy/occ=1 costed cheapest but ran the
// SLOWEST; device-best fills cores over parts_n). Discriminating shape: a col_sum
// with a FEW-tile free axis (W=128 -> 8 tiles < 48 cores) over a huge streamed
// reduced axis (H=16384). Without the gate the model split the reduced axis to
// "fill" 48 cores (fictional); with it, it fills the 8 free tiles honestly.
static void test_streamed_reduction_sink_no_split() {
    std::cout << "[STREAMSPLIT] streamed reduction sink fills via free axis, not a fictional split\n";
    Problem p;
    p.tensors = {{128, 16384}, {128, 1}};   // A[H=16384, W=128] -> col_sum m[1,128]
    p.ops = {{OpType::Reduction, {0}, {1}}};
    p.fast_memory_capacity = 1 << 24; set_910b(p);
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});
    const int64_t UB = 192 * 1024;
    // Even the NARROWEST free tile (w=16) over the full reduced axis overflows UB,
    // so EVERY cfg for this shape streams (pure streamed reduction).
    CHECK("STREAMSPLIT: reduced band streams (materialize overflows UB even at w=16)",
          sg->vector_peak_ub(TileConfig{16, 16384, 0}) > UB);
    auto r = sg->best_cost();
    std::cout << "    split=" << r.parallel_split << " cores=" << r.cores_used
              << " lat=" << r.latency << "\n";
    // C2: no cross-core reduced-axis split for a streamed reduction (the emit can't build one).
    CHECK("STREAMSPLIT: streamed reduction sink does NOT split the reduced axis",
          r.parallel_split == 1);
    // Cores filled by the free axis (W=128 -> 8 tiles), never a fictional 48-way reduced split.
    CHECK("STREAMSPLIT: fills via the free-axis grid (parts_n), capped by the 8 free tiles",
          r.cores_used <= 8);
}

// --- R0: vector_peak_ub couples the reduced axis to its FULL extent ----------
// On the live grid path a bare reduction sink's cfg collapses the reduced axis to the thin
// output extent (col_sum output [1,W] -> cfg.h derived from out_H_ = 1). vector_peak_ub must
// STILL size the reduced-axis read from the TENSOR's own extent (the FIXED_1 role) — else a
// streamed reduction looks "materialized, fits UB" at the FEASIBILITY and COMPUTE sites and the
// streaming cost/decision logic never fires. Before R0 the coupling lived only inside the split
// gate; feasibility (vector_stream) and the compute surcharge used the raw collapsed cfg.
static void test_streaming_detected_via_coupled_peak() {
    std::cout << "[R0] vector_peak_ub couples the reduced axis full even when cfg collapses it\n";
    const int64_t UB = 192 * 1024;
    auto peak_at_grid_cfg = [&](int64_t H) -> int64_t {
        Problem p; p.tensors = {{128, H}, {128, 1}};   // col_sum m[1,128] over height H
        p.ops = {{OpType::Reduction, {0}, {1}}};
        p.fast_memory_capacity = 1 << 24; set_910b(p);
        DAG d = DAG::build(p);
        // Grid cfg for a col reduction: height collapses to the output (h=1); width tiled.
        return Subgraph::create(p, d, {0})->vector_peak_ub(TileConfig{16, 1, 0});
    };
    // Large reduced extent -> the full-axis read (H=16384) overflows UB -> DETECTED as streaming,
    // even though cfg.h = 1. (Before R0 the reduced axis was sized from cfg.h=1 -> looked tiny.)
    CHECK("R0: large bare reduction detected as streaming (reduced axis coupled full at h=1 cfg)",
          peak_at_grid_cfg(16384) > UB);
    // Small reduced extent -> genuinely materializes; confirms the peak scales with the TRUE
    // reduced extent, not a constant (guards against dropping the coupling AND over-coupling).
    CHECK("R0: small bare reduction materializes (peak scales with the true reduced extent)",
          peak_at_grid_cfg(64) <= UB);
}

// --- single-core W-streaming makes a large-W reduction feasible --------------
// A softmax whose row [W,1] of A+e overflows UB even at h=1 cannot materialize;
// the single-core streaming chunk (the reduction accumulated chunk-by-chunk, e
// recomputed past it) shrinks the band so it fits. The model must REPRESENT this
// optimum as feasible (regime 2 of the softmax taxonomy).
static Problem mk_softmax(int64_t W, int64_t H) {
    Problem p;
    p.tensors = {{W, H}, {1, H}, {W, H}, {1, H}, {W, H}};
    p.ops = {{OpType::Reduction, {0}, {1}}, {OpType::Pointwise, {0, 1}, {2}},
             {OpType::Reduction, {2}, {3}}, {OpType::Pointwise, {2, 3}, {4}}};
    p.fast_memory_capacity = 1 << 24;
 set_910b(p);
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
    // Step 2b (Fix 2, vec_stream) — the streamed emit is ONLINE (flash): the wide body runs
    // once per element, so the cost scales ~LINEARLY with the data, NOT super-linearly.
    // W=8192 materializes (1 pass); W=24576 (3x the data) streams but does NOT recompute over
    // N_passes=3. The old model multiplied by #reductions+1 (a ~9x blowup, >4x mat); the
    // perf-sim refuted that -- there is no recompute multiplier, only a thin per-chunk
    // surcharge. Measured ratio ~2.4x (sub-linear: the materialized case carries more fixed
    // per-element overhead). So assert it scales (>2x) WITHOUT a recompute blowup (<3.5x).
    Problem pm = mk_softmax(8192, 128);   // materializes
    Problem ps = mk_softmax(24576, 128);  // streams (3x W)
    DAG dm = DAG::build(pm), ds = DAG::build(ps);
    double mat = Subgraph::create(pm, dm, {0, 1, 2, 3})->best_cost().latency;
    double str = Subgraph::create(ps, ds, {0, 1, 2, 3})->best_cost().latency;
    CHECK("STREAM: streamed cost scales ~linearly (online, no #reductions+1 recompute blowup)",
          str > 2.0 * mat && str < 3.5 * mat);
}

// --- G1/P4: streamed MULTI-reduction buildability is candidate-local ----------
// In buildable mode a streamed >1-reduction group is infeasible unless its COMPLETE op set is an
// exact P4 pattern supplied by the IR adapter. The analytic override retains the historical
// model-ahead behaviour for standalone research instances.
static void test_g1_multi_reduction_stream_decline() {
    std::cout << "[G1/P4] streamed multi-reduction requires an exact candidate pattern\n";
    Problem p = mk_softmax(32768, 128);   // 2 reductions (max, sum), streams (huge reduced W)
    DAG dag = DAG::build(p);
    CHECK("G1: analytic mode prices streamed softmax feasible (model-ahead, assumes P4)",
          std::isfinite(Subgraph::create(p, dag, {0, 1, 2, 3})->best_cost().latency));
    p.allow_model_ahead_multi_reduction_stream = false;  // buildable: no P4 emit yet
    CHECK("G1: buildable mode marks streamed softmax INfeasible (partitioner must cut it)",
          !std::isfinite(Subgraph::create(p, dag, {0, 1, 2, 3})->best_cost().latency));
    p.p4_patterns.push_back({P4PatternKind::SoftmaxFlash, FlatSet<size_t>{0, 1, 2, 3}});
    auto exact_softmax = Subgraph::create(p, dag, {0, 1, 2, 3})->best_cost();
    CHECK("P4: exact complete softmax pattern is buildable",
          std::isfinite(exact_softmax.latency));
    const auto& stream = exact_softmax.vector_stream;
    CHECK("P4PLAN: selected CostResult carries a feasible softmax stream plan",
          stream.feasible && stream.kind == VectorStreamKind::SoftmaxFlash &&
              stream.axis == 1 && stream.extent == 32768 && stream.stream_passes == 2);
    CHECK("P4PLAN: chunk geometry exactly covers full chunks plus a serial tail",
          stream.chunk > 0 && stream.full_chunks >= 1 &&
              stream.full_chunks * stream.chunk + stream.tail == stream.extent);
    CHECK("P4PLAN: stats peel chunk zero while apply rolls every full chunk",
          stream.stats.first_chunk == 1 &&
              stream.stats.trip_count == std::max<int64_t>(0, stream.full_chunks - 1) &&
              stream.apply.first_chunk == 0 && stream.apply.trip_count == stream.full_chunks);
    CHECK("P4PLAN: long softmax stats and apply phases are stage-2 pipelines",
          stream.stats.pipeline_stages == 2 && stream.apply.pipeline_stages == 2);
    CHECK("P4: a different subgroup does not inherit function-level permission",
          !std::isfinite(Subgraph::create(p, dag, {0, 1, 2})->best_cost().latency));

    Problem ln = p;
    ln.p4_patterns.clear();
    ln.p4_patterns.push_back({P4PatternKind::LayerNormWelford, FlatSet<size_t>{0, 1, 2, 3}});
    DAG ln_dag = DAG::build(ln);
    auto exact_layernorm = Subgraph::create(ln, ln_dag, {0, 1, 2, 3})->best_cost();
    CHECK("P4PLAN: adapter-supplied Welford kind survives into the selected stream plan",
          exact_layernorm.vector_stream.kind == VectorStreamKind::LayerNormWelford);
}

// --- G3: a spanning-output streamed reduction reads its input TWICE (A7) -------
// A streamed reduction whose live-out SPANS the reduced axis (rowmax then sub(x,m) -> [W,H]) needs
// the FINALIZED whole-axis stat per element, so the emit re-streams the input in an APPLY pass —
// the input is read twice. Streamed reductions are DDR-bound, so the model must price io_in x2 for
// the spanning case (else it under-prices the streamed group and over-fuses). Control: a pointwise
// twin relu(x) with the SAME [W,H] input and store, which reads the input ONCE. Both stream the
// wide W across 48 cores. With the io_in x2 the spanning/twin DDR ratio is ~2.5; WITHOUT it (the
// spanning input read once) it is ~1.5 (a pure config difference — the reduction pins the reduced
// axis so it tiles differently). So `spanning > 2x twin` holds iff the second input read is priced.
static void test_g3_spanning_reduction_rereads_input() {
    std::cout << "[G3] spanning-output streamed reduction prices a second input read (io_in x2)\n";
    const int64_t W = 32768, H = 128;  // huge reduced W -> both stream
    auto ddr = [&](bool spanning) -> double {
        Problem p;
        if (spanning) {
            p.tensors = {{W, H}, {1, H}, {W, H}};  // x -> m=rowmax[1,H] -> s=sub(x,m)[W,H] (spans W, 2 reads)
            p.ops = {{OpType::Reduction, {0}, {1}}, {OpType::Pointwise, {0, 1}, {2}}};
        } else {
            p.tensors = {{W, H}, {W, H}};  // x -> relu(x)[W,H] (pointwise twin: same store, 1 read)
            p.ops = {{OpType::Pointwise, {0}, {1}}};
        }
        p.fast_memory_capacity = 1 << 24;
        set_910b(p);
        DAG d = DAG::build(p);
        const std::vector<size_t> ops = spanning ? std::vector<size_t>{0, 1} : std::vector<size_t>{0};
        return Subgraph::create(p, d, ops)->best_cost().ddr_traffic;  // both stream over the wide W
    };
    const double twin = ddr(false), spanning = ddr(true);
    CHECK("G3: spanning streamed reduction reads its input twice (DDR > 2x the 1-read twin)",
          spanning > 2.0 * twin);
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
        p.ops = {{OpType::MatMul, {0, 1}, {2}}};
        p.fast_memory_capacity = 1LL << 30;
 set_910b(p);
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
        // Realistic HBM cap (gml1_multicore): the solver fills cores up to its balanced-
        // tile / saturation point (here 8–24, NOT always 24 — sq1024 is compute-bound yet
        // uses 16), but never reads operands faster than the aggregate bandwidth. Assert
        // the HBM read floor + that it still parallelizes when there is enough work.
        C("respects HBM aggregate read floor",
          r.latency + 1e-6 >= hbm_read_floor(p, c.M, c.N, c.K,
                                             p.tensors[0].dtype, p.tensors[1].dtype));
        C("parallelizes within the 24 cube cores",
          r.cores_used >= 1 && r.cores_used <= 24 && (fractals < 24 || r.cores_used > 1));
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
    auto base = [](Problem& p) { p.fast_memory_capacity = 1 << 24;
 set_910b(p); };
    Problem pw; pw.tensors = {{4096, 4096}, {4096, 4096}};
    pw.ops = {{OpType::Pointwise, {0}, {1}}}; base(pw);
    run("pointwise", pw, {0}, false);
    Problem rm; rm.tensors = {{1024, 512}, {1, 512}};       // many rows -> spatial fill
    rm.ops = {{OpType::Reduction, {0}, {1}}}; base(rm);
    run("rowmax-manyrows", rm, {0}, false);
    Problem rf; rf.tensors = {{4096, 4}, {1, 4}};           // few rows -> reduced split
    rf.ops = {{OpType::Reduction, {0}, {1}}}; base(rf);
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
        p.ops = {{OpType::MatMul, {0, 1}, {2}}, {OpType::Pointwise, {2}, {3}},
                 {OpType::MatMul, {3, 4}, {5}}};
        p.fast_memory_capacity = 1LL << 30;
 set_910b(p, 4096); return p;  // compute-bound: decode fractal-quant
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
        p.ops = {{OpType::MatMul, {0, 1}, {2}}, {OpType::Reduction, {2}, {3}},
                 {OpType::Pointwise, {2, 3}, {4}}, {OpType::Reduction, {4}, {5}},
                 {OpType::Pointwise, {4, 5}, {6}}, {OpType::MatMul, {6, 7}, {8}}};
        p.fast_memory_capacity = 1LL << 30;
 set_910b(p); return p;
    };
    for (int64_t B : {16, 128, 512}) {
        Problem p = attn(B);
        auto C = [&](const char* w, bool ok) { snprintf(b, sizeof b, "ATTN B%lld: %s",
                                                         (long long)B, w); CHECK(b, ok); };
        C("QK^T(cube)+softmax(vec) not fusable", !creatable(p, {0, 1}));
        auto sc = eval(p, {0}), sm = eval(p, {1, 2, 3, 4}), pv = eval(p, {5});
        const int64_t d = 64, S = 2048;  // attn dims (mirror the attn() lambda)
        // Realistic HBM cap (gml1_multicore): these thin-M (M=B) matmuls saturate HBM
        // below 24 cores (sc 8–16, pv 12), so assert the HBM read floor + parallelism
        // rather than a rigid 24-core fill. scores = Q[d,B]·K[S,d] (M=B,N=S,K=d);
        // PV = P[S,B]·V[d,S] (M=B,N=d,K=S).
        C("scores feasible + respects HBM read floor",
          std::isfinite(sc.latency)
              && sc.latency + 1e-6 >= hbm_read_floor(p, B, S, d, p.tensors[0].dtype, p.tensors[1].dtype)
              && sc.cores_used > 1 && sc.cores_used <= 24);
        C("softmax(over keys) feasible", std::isfinite(sm.latency));
        C("PV feasible + respects HBM read floor",
          std::isfinite(pv.latency)
              && pv.latency + 1e-6 >= hbm_read_floor(p, B, d, S, p.tensors[6].dtype, p.tensors[7].dtype)
              && pv.cores_used > 1 && pv.cores_used <= 24);
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
        p.ops = {{OpType::MatMul, {0, 1}, {2}},     // scores
                 {OpType::Reduction, {2}, {3}},        // colmax
                 {OpType::Pointwise, {2, 3}, {4}},     // exp
                 {OpType::Reduction, {4}, {5}},        // colsum
                 {OpType::Pointwise, {4, 5}, {6}},     // div -> P
                 {OpType::MatMul, {6, 7}, {8}},      // O = P@V
                 {OpType::MatMul, {8, 9}, {10}},     // O@Wo
                 {OpType::MatMul, {10, 11}, {12}},   // @W1
                 {OpType::Pointwise, {12}, {13}},     // relu
                 {OpType::MatMul, {13, 14}, {15}}}; // @W2
        p.fast_memory_capacity = 1LL << 30;
 set_910b(p);
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
            p.ops.push_back({OpType::MatMul, {prev, w}, {out}});
            prev = out;
        }
        p.fast_memory_capacity = 1LL << 30;
 set_910b(p);
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
    p.ops = {{OpType::Reduction, {0}, {1}}, {OpType::Reduction, {0}, {2}}};
    p.fast_memory_capacity = 1 << 24;
 set_910b(p);
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
    p.ops = {{OpType::Pointwise, {0}, {1}}};
    p.fast_memory_capacity = 1LL << 30;
 set_910b(p);
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
    p.ops = {{OpType::Pointwise, {0}, {1}}};
    p.fast_memory_capacity = 1LL << 30;
 set_910b(p);
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});
    const int64_t UB = 192 * 1024;
    TileConfig big{8192, 256, 0};   // whole tensor as one kernel -> overflows UB
    CHECK("PWSTREAM: large tile overflows UB", sg->vector_peak_ub(big) > UB);
    auto s = sg->vector_stream(big);
    auto plan = sg->vector_stream_plan(big);
    CHECK("PWSTREAM: an explicit chunk is derived (streams the wider axis)",
          s.axis == 1 && s.chunk > 0 && s.chunk < 8192);
    CHECK("PWPLAN: public stream plan preserves the compatibility axis/chunk view",
          plan.feasible && plan.kind == VectorStreamKind::Pointwise &&
              plan.axis == s.axis && plan.chunk == s.chunk && plan.extent == 8192);
    CHECK("PWPLAN: geometry and body loop describe every full chunk plus the tail",
          plan.full_chunks * plan.chunk + plan.tail == plan.extent &&
              plan.body.first_chunk == 0 && plan.body.trip_count == plan.full_chunks);
    CHECK("PWSTREAM: the chunk actually fits UB",
          sg->vector_peak_ub(big, {}, {}, s.chunk, s.axis) <= UB);
    TileConfig valid_big{8192, 256, 1};
    auto cost = sg->compute_cost(valid_big);
    CHECK("PWPLAN: compute_cost carries the derived plan without changing feasibility",
          cost.feasible && cost.vector_stream.feasible &&
              cost.vector_stream.chunk == plan.chunk);
    TileConfig small{128, 128, 0};  // fits UB -> materialized, no sub-stream
    CHECK("PWSTREAM: a UB-fitting tile materializes (no stream)",
          sg->vector_stream(small).axis == 0);
    auto materialized = sg->vector_stream_plan(small);
    CHECK("PWPLAN: materialized plan is explicit and has no loop geometry",
          materialized.feasible && !materialized.streamed() &&
              materialized.kind == VectorStreamKind::Materialized &&
              materialized.chunk == 0 && materialized.body.trip_count == 0);
}

int main() {
    test_subgraph_structure();
    test_cube_vector_fusion();
    test_mixed_unit_tiling();
    test_mixed_mm_pw_variants();
    test_mixed_pipeline_stages();
    test_mixed_sink_split_k();
    test_mixed_single_tile_no_skew();
    test_mixed_grid_makespan();
    test_mixed_grid_makespan_multimatmul();
    test_mixed_flash_attention();
    test_model_ahead_split_k_flag();
    test_mixed_two_pool_feasibility();
    test_mixed_multistage_pebble();
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
    test_streamed_reduction_sink_no_split();
    test_streaming_detected_via_coupled_peak();
    test_vector_streaming_reduction();
    test_g1_multi_reduction_stream_decline();
    test_g3_spanning_reduction_rereads_input();
    test_vector_sensibility();
    test_visualize();
    test_two_matmul();
    test_shared_input_reuse();
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
    std::cout << "\n=== pass=" << g_pass << " fail=" << g_fail << " ===\n";
    return g_fail;  // RED items are the spec; only hard CHECK failures break the build
}
