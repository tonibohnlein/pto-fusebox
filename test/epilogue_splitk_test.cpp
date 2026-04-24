// epilogue_splitk_test.cpp
//
// Tests for the MM→PW epilogue split-K feature (issue #82).
//
// Background:
//   For a subgraph whose only sinks are Pointwise ops, and where the PW
//   chain(s) are fed by exactly ONE MatMul (the "effective sink MM"), the
//   MM's k-loop can legally tile below K without breaking the PW epilogue:
//     • Per spatial tile the MM iterates its K in chunks of cfg.k,
//       accumulating into its ephemeral output tile.
//     • After the k-loop completes the tile is a fully-formed value; the
//       PW chain then fires once on that tile.
//   Valid iff the accumulator + streamed inputs + output all fit in fast
//   memory (per #82).
//
// Coverage:
//   - Detection: simple MM→PW, MM→PW→PW chain, PW→MM→PW (prologue+epilogue).
//   - Rejection: MM→MM→PW (two MMs in chain), fan-in 2×MM→PW, mixed MM+PW
//     sinks, PW-only subgraph (no MM).
//   - Working set: ephemeral MM accumulator counted when nk>1; NOT counted
//     at nk=1 (classic ephemeral).
//   - Feasibility: concrete example from the design discussion —
//     MM→PW 128×128 @ cap=45000 is OOM at nk=1 but fits at nk=4.
//   - Cost: total matches (MM_base + PW_base) × num_spatial_tiles.
//   - Integration: k-divides still enforced, super-native still rejected.

#include "core/dag.h"
#include "core/subgraph.h"
#include "core/types.h"
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ_I(const char* l, int64_t got, int64_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l
           << " got=" << got << " exp=" << exp << "\n"; }
}
static void CHECK_EQ(const char* l, double got, double exp, double tol = 0.1) {
    if (std::abs(got - exp) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l
           << " got=" << got << " exp=" << exp << "\n"; }
}

static TileConfig TC(int64_t w, int64_t h, int64_t k,
                     SnakeDir s = SnakeDir::None) { return {w, h, k, s}; }

static Subgraph make_sg(const Problem& p, const DAG& d,
                        std::vector<size_t> ops) {
    auto sg = Subgraph::create(p, d, std::move(ops));
    if (!sg) { std::cerr << "FATAL: Subgraph::create failed\n"; std::exit(1); }
    return std::move(*sg);
}

// ============================================================================
// (1) Simple MM→PW — the core case
// ============================================================================

void test_simple_mm_pw_detected() {
    std::cout << "--- test_simple_mm_pw_detected ---\n";
    // MM: T0 @ T1 → T2 (eph)
    // PW: T2 → T3 (boundary out)
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::Pointwise, {2}, {3}, 100}};
    p.fast_memory_capacity = 1'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // output_K now reflects the effective sink MM's K, not the default 1.
    CHECK_EQ_I("output_K = 128 (from effective sink MM)", sg.output_K(), 128);
    CHECK("T2 ephemeral", sg.ephemeral().count(2));

    // Split-K tilings are now valid.
    CHECK("cfg=[128,128,1]   valid (nk=128)", sg.is_valid_tiling(TC(128,128,1)));
    CHECK("cfg=[128,128,32]  valid (nk=4)",   sg.is_valid_tiling(TC(128,128,32)));
    CHECK("cfg=[128,128,64]  valid (nk=2)",   sg.is_valid_tiling(TC(128,128,64)));
    CHECK("cfg=[128,128,128] valid (nk=1)",   sg.is_valid_tiling(TC(128,128,128)));

    // Invalid: cfg.k > K (per #75).
    CHECK("cfg=[128,128,256] rejected (>K)",  !sg.is_valid_tiling(TC(128,128,256)));
    // Invalid: cfg.k doesn't divide K.
    CHECK("cfg=[128,128,96]  rejected (96∤128)", !sg.is_valid_tiling(TC(128,128,96)));
}

// ============================================================================
// (2) Concrete example from design discussion:
//     128×128 tensors, capacity=45000, MM cost=2000, PW cost=100.
//     At nk=1  → WS 49152, INFEASIBLE (>45000).
//     At nk=4  → WS 40960, feasible.  <-- the headline case
// ============================================================================

void test_design_example_nk4_fits_nk1_ooms() {
    std::cout << "--- test_design_example_nk4_fits_nk1_ooms ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::Pointwise, {2}, {3}, 100}};
    p.fast_memory_capacity = 45000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // nk=1: all three boundary tensors full-size + no accumulator =
    //   T0(16384) + T1(16384) + T3(16384) = 49152 > 45000 → OOM.
    CHECK_EQ_I("ws(128,128,128) = 49152 (nk=1)",
               sg.working_set(TC(128,128,128)), 49152);
    CHECK("not feasible at nk=1",            !sg.is_feasible(TC(128,128,128)));

    // nk=4: strip inputs + accumulator + output:
    //   T0 stream 128×32=4096 + T1 stream 32×128=4096 +
    //   T3 out 128×128=16384 + T2 accumulator w·h=16384 = 40960 ≤ 45000.
    CHECK_EQ_I("ws(128,128,32) = 40960 (nk=4)",
               sg.working_set(TC(128,128,32)), 40960);
    CHECK("feasible at nk=4",                 sg.is_feasible(TC(128,128,32)));

    // Cost/latency sanity — both should match the design doc's 4915.2 at nk=4.
    auto c = sg.compute_cost(TC(128,128,32));
    CHECK_EQ_I("num_k_passes = 4",   c.num_k_passes, 4);
    CHECK_EQ_I("num_spatial_tiles=1", c.num_spatial_tiles, 1);
    CHECK_EQ("comp/step = 525",      c.compute_per_step, 525.0);
    CHECK_EQ("latency = 4915.2",     c.latency, 4915.2);
}

// ============================================================================
// (3) MM→PW→PW chain — detection walks through PW chain to the MM.
// ============================================================================

void test_mm_pw_pw_chain() {
    std::cout << "--- test_mm_pw_pw_chain ---\n";
    // MM: T0 @ T1 → T2 (eph)
    // PW1: T2 → T3 (eph)
    // PW2: T3 → T4 (boundary out)
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::Pointwise, {2}, {3}, 100},
             {OpType::Pointwise, {3}, {4}, 50}};
    p.fast_memory_capacity = 45000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // Detection should succeed — the BFS from PW2 walks T3 → PW1 → T2 → MM.
    CHECK_EQ_I("output_K = 128 (epilogue detected through PW chain)",
               sg.output_K(), 128);
    // Accumulator WS still counts only the MM output (T2).  T3 is a pipeline
    // ephemeral between PW1 and PW2, with zero cost.
    CHECK_EQ_I("ws(128,128,32) = 40960 (only T2 is accumulator)",
               sg.working_set(TC(128,128,32)), 40960);
    CHECK("split-K feasible",      sg.is_feasible(TC(128,128,32)));
    CHECK("full-K infeasible",    !sg.is_feasible(TC(128,128,128)));
}

// ============================================================================
// (4) PW→MM→PW — BFS stops at MM (valid); detection still succeeds.
// ============================================================================

void test_pw_mm_pw_chain() {
    std::cout << "--- test_pw_mm_pw_chain ---\n";
    // PW0: T_in → T0 (eph, MM LHS)
    // MM:  T0 @ T1 → T2 (eph)
    // PW1: T2 → T3 (boundary out)
    //
    // With Rule 2 (prologue-PW): PW0 feeds MM's LHS → requires cfg.w ≥ K.
    // With K=128 and cfg.w=128, the rule is satisfied.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 50},
             {OpType::MatMul, {1,2}, {3}, 2000},
             {OpType::Pointwise, {3}, {4}, 100}};
    p.fast_memory_capacity = 60000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // BFS from PW1 backward: T3 → MM.  Stops at MM (doesn't descend into
    // MM's inputs).  One MM found → epilogue detected.
    CHECK_EQ_I("output_K = 128 (epilogue detected)", sg.output_K(), 128);
    // cfg.w must be ≥ K due to Rule 2/3 (prologue PW).
    CHECK("cfg.w=128, cfg.k=32 valid",  sg.is_valid_tiling(TC(128,128,32)));
    CHECK("cfg.w=64,  cfg.k=32 rejected (Rule 2 violated)",
          !sg.is_valid_tiling(TC(64,128,32)));
}

// ============================================================================
// (5) MM→MM→PW — the BFS STOPS at each MM (by design), so only MM1 (the
// MM directly feeding the PW chain) is found.  MM0 is not descended into.
// This is correct per the design discussion's "Case 4":
//   • MM0 is non-sink, produces fresh T2 strips per k-step (FROM_NK), free.
//   • MM1 is the effective sink, accumulates into T4 — that's the ephemeral
//     accumulator that costs w×h when nk>1.
//   • PW fires once on T4 at the end of the k-loop.
// ============================================================================

void test_mm_mm_pw_is_detected() {
    std::cout << "--- test_mm_mm_pw_is_detected ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::MatMul, {2,3}, {4}, 2000},
             {OpType::Pointwise, {4}, {5}, 100}};
    p.fast_memory_capacity = 1'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // Only MM1 (the MM directly producing the PW's input) is found.
    CHECK_EQ_I("output_K = 128 (from MM1)", sg.output_K(), 128);
    CHECK("cfg=[128,128,32] valid (split-K enabled)",
          sg.is_valid_tiling(TC(128,128,32)));
    CHECK("cfg=[128,128,32] feasible",
          sg.is_feasible(TC(128,128,32)));
}

// ============================================================================
// (6) Two parallel MMs → single PW (fan-in) — detection MUST REJECT.
// ============================================================================

void test_fan_in_not_detected() {
    std::cout << "--- test_fan_in_not_detected ---\n";
    // MM0: T0 @ T1 → T2 (eph)
    // MM1: T3 @ T4 → T5 (eph)
    // PW:  T2, T5 → T6 (boundary out)
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::MatMul, {3,4}, {5}, 2000},
             {OpType::Pointwise, {2,5}, {6}, 100}};
    p.fast_memory_capacity = 1'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // Two distinct MMs feed the PW → epilogue rejected.
    CHECK_EQ_I("output_K = 1 (fan-in rejected)", sg.output_K(), 1);
    // With output_K=1 and has_pw_sink without epilogue, nk is always clamped
    // to 1 so is_valid_tiling doesn't reject cfg.k>1 by itself — but the
    // enumerator (best_cost → ks_cand_) only considers cfg.k=1.
    auto c = sg.compute_cost(TC(128,128,1));
    CHECK_EQ_I("num_k_passes = 1 (no split-K)", c.num_k_passes, 1);
    auto best = sg.best_cost();
    CHECK_EQ_I("best_cost picks cfg.k = 1", best.config.k, 1);
}

// ============================================================================
// (7) Mixed MM + PW sinks — the epilogue detection is skipped entirely.
//
// Mechanism: the detection block at subgraph.cpp:200 is guarded by
//    if (has_pw_sink_ && has_matmul_ && output_K_ == 1) { ... }
// The earlier sink loop (subgraph.cpp:187) already set output_K_ from
// whichever MM sink exists, so output_K_ != 1 and the epilogue block
// is never entered.  This is correct behavior: with a real MM sink
// already driving the subgraph's k-loop, there is no "effective sink
// MM" to retroactively promote — the MM sink IS the sink.
//
// The test below confirms:
//   - output_K is set from the MM sink (not 1, not the inner MM's K).
//   - Classic mixed-MM+PW rule applies: cfg.k must = output_K (nk=1).
// ============================================================================

void test_mixed_sinks_skip_epilogue_detection() {
    std::cout << "--- test_mixed_sinks_skip_epilogue_detection ---\n";
    // MM0: T0 @ T1 → T2 (eph)      ← non-sink, feeds PW
    // PW:  T2 → T3 (boundary out)  ← PW sink (candidate for epilogue)
    // MM1: T0 @ T4 → T5 (boundary out)  ← MM sink (disables epilogue path)
    // Same output (W,H)=(128,128) required for boundary outs.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::Pointwise, {2}, {3}, 100},
             {OpType::MatMul, {0,4}, {5}, 2000}};
    p.fast_memory_capacity = 10'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // The MM sink (MM1) sets output_K = 128 BEFORE the epilogue-detection
    // guard is checked, so the guard `output_K_ == 1` fails and the
    // detection block is never entered. Classic mixed-MM+PW behavior
    // applies: cfg.k must equal output_K (nk=1), which means cfg.k=128.
    CHECK_EQ_I("output_K = 128 (from MM sink, detection skipped)",
               sg.output_K(), 128);
    CHECK("cfg=[128,128,128] valid (cfg.k == output_K)",
          sg.is_valid_tiling(TC(128,128,128)));
    CHECK("cfg=[128,128,32]  rejected (nk>1 not allowed with MM sink + PW sink)",
          !sg.is_valid_tiling(TC(128,128,32)));
}

// ============================================================================
// (8) PW-only subgraph — no MM, epilogue detection inert.
// ============================================================================

void test_pw_only_unchanged() {
    std::cout << "--- test_pw_only_unchanged ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 100},
             {OpType::Pointwise, {1}, {2}, 50}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK_EQ_I("output_K = 1 (no MM)", sg.output_K(), 1);
    CHECK("cfg=[128,128,1] valid",    sg.is_valid_tiling(TC(128,128,1)));
    // For PW-only (no MM), nk is always 1 regardless of cfg.k; the enumerator
    // only picks cfg.k=1 via ks_cand_ = {output_K_=1}.
    auto best = sg.best_cost();
    CHECK_EQ_I("best_cost picks cfg.k = 1 (PW-only)", best.config.k, 1);
    auto c = sg.compute_cost(TC(128,128,1));
    CHECK_EQ_I("num_k_passes = 1", c.num_k_passes, 1);
}

// ============================================================================
// (9) Working set breakdown — exact accounting of each contributing term
//     at nk=1 vs nk=4 for simple MM→PW.
// ============================================================================

void test_ws_breakdown_accumulator_cost() {
    std::cout << "--- test_ws_breakdown_accumulator_cost ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::Pointwise, {2}, {3}, 100}};
    p.fast_memory_capacity = 1'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // nk=1 (cfg.k=128): no accumulator cost — MM output is a true
    // pipeline ephemeral in this case since there's only one k-step.
    //   T0 full 128×128 = 16384  (LHS, FROM_NK × FROM_NTH at nk=1 → 16384)
    //   T1 full 128×128 = 16384
    //   T3 full 128×128 = 16384
    //   T2 ephemeral but NOT counted (nk=1)
    //   Total = 49152
    CHECK_EQ_I("ws at nk=1 = 49152 (no accumulator cost)",
               sg.working_set(TC(128,128,128)), 49152);

    // nk=2 (cfg.k=64):
    //   T0 stream 64×128 = 8192
    //   T1 stream 128×64 = 8192
    //   T3 full  128×128 = 16384
    //   T2 accumulator w·h = 16384
    //   Total = 49152
    CHECK_EQ_I("ws at nk=2 = 49152 (accumulator matches the strip savings)",
               sg.working_set(TC(128,128,64)), 49152);

    // nk=4 (cfg.k=32):
    //   T0 stream 32×128 = 4096
    //   T1 stream 128×32 = 4096
    //   T3 full  128×128 = 16384
    //   T2 accumulator    = 16384
    //   Total = 40960
    CHECK_EQ_I("ws at nk=4 = 40960 (split-K shrinks strips)",
               sg.working_set(TC(128,128,32)), 40960);

    // nk=32 (cfg.k=4):
    //   T0 stream 4×128 = 512
    //   T1 stream 128×4 = 512
    //   T3 full 16384
    //   T2 accumulator 16384
    //   Total = 33792
    CHECK_EQ_I("ws at nk=32 = 33792 (very fine split)",
               sg.working_set(TC(128,128,4)), 33792);
}

// ============================================================================
// (10) Cost invariance: total compute across all spatial × k iterations
// should equal (MM_base + PW_base) × num_spatial_tiles regardless of cfg.k.
// ============================================================================

void test_cost_total_is_invariant() {
    std::cout << "--- test_cost_total_is_invariant ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::Pointwise, {2}, {3}, 100}};
    p.fast_memory_capacity = 1'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // Uniform /nk rule: per-step = (2000+100)/nk.
    // Total compute = ntiles × nk × per-step = ntiles × (2000+100).
    // 1 spatial tile at cfg.w=cfg.h=128 → total = 2100 regardless of cfg.k.
    for (int64_t k : {1, 2, 4, 8, 16, 32, 64, 128}) {
        auto c = sg.compute_cost(TC(128, 128, k));
        if (!c.feasible) continue;  // not applicable
        double total_compute = c.compute_per_step *
                               c.num_spatial_tiles * c.num_k_passes;
        CHECK_EQ(("total compute invariant at k=" + std::to_string(k)).c_str(),
                 total_compute, 2100.0, 1.0);
    }
}

// ============================================================================
// What counts as a "simple epilogue"?
//
// The detection logic (Subgraph::create) defines it precisely:
//
//   1. ALL sinks of the subgraph are Pointwise ops.  Any MatMul sink
//      disables the epilogue path (output_K is already set from that MM,
//      and the `output_K_ == 1` guard skips the detection block).
//
//   2. At least one MatMul exists in the subgraph.
//
//   3. Backward BFS from each PW sink traversing PW-only ops reaches
//      exactly ONE unique MatMul.  The BFS:
//        - iterates ALL inputs of each visited PW (not just one);
//        - descends into PW-type producers inside the subgraph;
//        - STOPS at MatMul producers (records the MM, does not descend);
//        - skips producers outside the subgraph (boundary inputs) and
//          skips model-input tensors (no producer).
//
// This means the PW "chain" can actually be a PW DAG: branches, fan-in
// from other PWs, multi-input PW ops, boundary inputs mixed in with
// MM outputs — all fine, provided the MatMul count constraint holds.
// The tests below lock in that behavior.
// ============================================================================

// PW consumes the MM output AND a boundary-input tensor (multi-input PW).
// Still a simple epilogue: the only MM is the one feeding T_mm_out.
void test_multi_input_pw_with_boundary_still_simple() {
    std::cout << "--- test_multi_input_pw_with_boundary_still_simple ---\n";
    // MM: T0 @ T1 → T_mm (eph)
    // PW: T_mm, T_aux → T_out (boundary out, PW sink)
    // T_aux is a boundary input (no producer inside subgraph).
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    //            T0        T1        T_mm      T_aux     T_out
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::Pointwise, {2, 3}, {4}, 100}};
    p.fast_memory_capacity = 1'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK_EQ_I("output_K = 128 (epilogue detected with multi-input PW)",
               sg.output_K(), 128);
    CHECK("cfg.k=32 (nk=4) valid", sg.is_valid_tiling(TC(128,128,32)));
}

// PW fan-in from TWO PW ancestors, both of which only descend into ONE
// MM.  Detection should succeed.
void test_pw_fanin_from_multiple_pws_same_mm() {
    std::cout << "--- test_pw_fanin_from_multiple_pws_same_mm ---\n";
    // MM:  T0 @ T1 → T_mm (eph)
    // PW0: T_mm   → T_a (eph)
    // PW1: T_mm   → T_b (eph)        (fan-out of MM output)
    // PW2: T_a, T_b → T_out (sink)
    //
    // BFS from PW2: T_a → PW0 → T_mm → MM (found);
    //               T_b → PW1 → T_mm → MM (already visited, or same MM).
    // One MM → epilogue detected.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},
                 {128,128},{128,128},{128,128}};
    //            T0        T1        T_mm      T_a       T_b       T_out
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::Pointwise, {2}, {3}, 50},
             {OpType::Pointwise, {2}, {4}, 50},
             {OpType::Pointwise, {3,4}, {5}, 100}};
    p.fast_memory_capacity = 1'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2, 3});

    CHECK_EQ_I("output_K = 128 (PW DAG, one MM)", sg.output_K(), 128);
    CHECK("cfg.k=32 valid (split-K enabled)",
          sg.is_valid_tiling(TC(128,128,32)));
}

// Multiple PW sinks, but both fed by PW chains that all terminate at the
// SAME single MM.  Detection should succeed.
void test_multiple_pw_sinks_share_same_mm() {
    std::cout << "--- test_multiple_pw_sinks_share_same_mm ---\n";
    // MM:  T0 @ T1 → T_mm (eph)
    // PW0: T_mm → T_o1 (sink1)
    // PW1: T_mm → T_o2 (sink2)
    // Both sinks trace back to the same MM via their BFS.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    //            T0        T1        T_mm      T_o1      T_o2
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::Pointwise, {2}, {3}, 50},
             {OpType::Pointwise, {2}, {4}, 50}};
    p.fast_memory_capacity = 1'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK_EQ_I("output_K = 128 (multi-sink, same MM)", sg.output_K(), 128);
    CHECK("cfg.k=32 valid", sg.is_valid_tiling(TC(128,128,32)));
}

// Multiple PW sinks, each reaching a DIFFERENT MM via its BFS.  Not simple.
void test_multiple_pw_sinks_different_mms_rejected() {
    std::cout << "--- test_multiple_pw_sinks_different_mms_rejected ---\n";
    // MM0: T0 @ T1 → T_a (eph)
    // MM1: T2 @ T3 → T_b (eph)
    // PW0: T_a → T_o1 (sink1)
    // PW1: T_b → T_o2 (sink2)
    // Each sink walks back to a different MM → detection rejects.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},
                 {128,128},{128,128},{128,128},{128,128}};
    //            T0        T1        T2        T3        T_a       T_b       T_o1      T_o2
    p.ops = {{OpType::MatMul, {0,1}, {4}, 2000},
             {OpType::MatMul, {2,3}, {5}, 2000},
             {OpType::Pointwise, {4}, {6}, 50},
             {OpType::Pointwise, {5}, {7}, 50}};
    p.fast_memory_capacity = 1'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2, 3});

    CHECK_EQ_I("output_K = 1 (two different MMs → rejected)",
               sg.output_K(), 1);
}

// ============================================================================
// (11) Tiling propagation check: LHS/RHS of the effective sink MM get
// FROM_NK on one axis (streamable), not FIXED_1.
// ============================================================================

void test_effective_sink_mm_tiling_propagation() {
    std::cout << "--- test_effective_sink_mm_tiling_propagation ---\n";
    // A slightly larger output so ntw > 1 is achievable.  4096-wide tensors
    // at cfg.w=128 → ntw=32; MM output (4096,128) so nth=1.
    Problem p;
    p.tensors = {{4096,128},{4096,4096},{4096,128},{4096,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 5000},
             {OpType::Pointwise, {2}, {3}, 200}};
    p.fast_memory_capacity = 600000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK_EQ_I("output_K = 4096", sg.output_K(), 4096);
    // Feasibility with nk>1 should work.  At cfg=[128,128,128] nk=32.
    CHECK("cfg=[128,128,128] valid",  sg.is_valid_tiling(TC(128,128,128)));
    CHECK("cfg=[128,128,128] feasible", sg.is_feasible(TC(128,128,128)));
    // cfg.k cannot exceed native (super-native bound); best_cost picks
    // the highest valid power-of-2 k that still fits & divides K.
    auto best = sg.best_cost();
    CHECK("best_cost is feasible",  best.feasible);
    CHECK("best_cost cfg.k ≤ 128",  best.config.k <= 128);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    test_simple_mm_pw_detected();
    test_design_example_nk4_fits_nk1_ooms();
    test_mm_pw_pw_chain();
    test_pw_mm_pw_chain();
    test_mm_mm_pw_is_detected();
    test_fan_in_not_detected();
    test_mixed_sinks_skip_epilogue_detection();
    test_pw_only_unchanged();
    test_ws_breakdown_accumulator_cost();
    test_cost_total_is_invariant();
    test_multi_input_pw_with_boundary_still_simple();
    test_pw_fanin_from_multiple_pws_same_mm();
    test_multiple_pw_sinks_share_same_mm();
    test_multiple_pw_sinks_different_mms_rejected();
    test_effective_sink_mm_tiling_propagation();

    std::cout << "\n=== " << g_pass << " passed, " << g_fail
              << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
