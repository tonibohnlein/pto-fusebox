// prologue_pw_test.cpp
//
// Tests for issue #71 Rules 2/3 — the "prologue pointwise" geometric
// condition:
//
//   Rule 2: Pointwise feeding a MatMul's LHS  →  require cfg.w ≥ matmul.K
//   Rule 3: Pointwise feeding a MatMul's RHS  →  require cfg.h ≥ matmul.K
//
// Intuition: a MatMul consumes K-wide slices of its inputs. If the PW
// producer only emits a cfg.w × cfg.h granule per invocation, the MM can
// only start iterating its k-loop once the PW has produced enough data
// along the reduction axis. The condition cfg.w ≥ K (LHS) or cfg.h ≥ K
// (RHS) guarantees the full reduction strip exists before the MM needs
// it.
//
// Covers:
//   * Direct PW→MM prologue on LHS (Rule 2)
//   * Direct PW→MM prologue on RHS (Rule 3)
//   * Organizer's concrete #71 Q2 example (256×256, cfg.w = K = 256)
//   * Multi-PW chain before MM (BFS through PW-only chain)
//   * Pure-MM chains are unaffected (no prologue PW)
//   * MM→PW (epilogue, not prologue) unaffected
//   * Boundary threshold: cfg exactly equal to K is accepted; one below
//     is rejected
//   * Max across multiple prologue PW targets

#include "core/dag.h"
#include "core/subgraph.h"
#include "core/types.h"
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
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
// Rule 2 (LHS prologue): direct PW → MM(LHS).
//   PW: T0 → T1          (T1 is PW output, shape same as T0)
//   MM: T1 × T2 → T3     (T1 is LHS → MM.K = T1.width)
// With T1 (128×128) and MM.K = 128, Rule 2 demands cfg.w ≥ 128.
// ============================================================================

void test_rule2_lhs_prologue_basic() {
    std::cout << "--- test_rule2_lhs_prologue_basic ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    //            T0        T1        T2        T3
    p.ops = {{OpType::Pointwise, {0}, {1}, 1000},     // PW: T0 → T1
             {OpType::MatMul,    {1,2}, {3}, 2000}};  // MM: T1 × T2 → T3
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // MM.K = T1.width = 128. Rule 2 requires cfg.w ≥ 128.
    CHECK("cfg=(128,128,128) accepted (w = K)",
          sg.is_valid_tiling(TC(128, 128, 128)));
    CHECK("cfg=(64,128,64) rejected (w < K, Rule 2)",
          !sg.is_valid_tiling(TC(64, 128, 64)));
    CHECK("cfg=(32,128,32) rejected (w < K, Rule 2)",
          !sg.is_valid_tiling(TC(32, 128, 32)));
}

// ============================================================================
// Rule 3 (RHS prologue): direct PW → MM(RHS).
//   PW: T0 → T1
//   MM: T2 × T1 → T3     (T1 is RHS → MM.K = T2.width)
// With T2 (128×128), MM.K = 128. Rule 3 demands cfg.h ≥ 128.
// ============================================================================

void test_rule3_rhs_prologue_basic() {
    std::cout << "--- test_rule3_rhs_prologue_basic ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    //            T0        T1        T2        T3
    p.ops = {{OpType::Pointwise, {0}, {1}, 1000},     // PW: T0 → T1
             {OpType::MatMul,    {2,1}, {3}, 2000}};  // MM: T2 × T1 → T3
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // MM.K = T2.width = 128. Rule 3 requires cfg.h ≥ 128.
    CHECK("cfg=(128,128,128) accepted (h = K)",
          sg.is_valid_tiling(TC(128, 128, 128)));
    CHECK("cfg=(128,64,64) rejected (h < K, Rule 3)",
          !sg.is_valid_tiling(TC(128, 64, 64)));
    CHECK("cfg=(128,32,32) rejected (h < K, Rule 3)",
          !sg.is_valid_tiling(TC(128, 32, 32)));
}

// ============================================================================
// Organizer's concrete example from #71 Q2:
//   256×256 tensors, cfg = (256, 256, 128), native = (256, 256)
//   PW → MM where PW feeds LHS. With w = 256 ≥ matmul.K = 256, the PW
//   produces the full 256×256 LHS in a single tile — Rule 2 holds.
// ============================================================================

void test_rule2_organizer_q2_example() {
    std::cout << "--- test_rule2_organizer_q2_example ---\n";
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256}};
    //            T0        T1        T2        T3
    p.ops = {{OpType::Pointwise, {0}, {1}, 1000},     // PW: T0 → T1
             {OpType::MatMul,    {1,2}, {3}, 2000}};  // MM: T1 × T2 → T3
    // Generous FMC so we isolate the rule 2/3 effect.
    p.fast_memory_capacity = 10'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 256; p.native_h = 256;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // cfg.w = 256 = K = 256 → accepted (organizer's answer).
    CHECK("organizer Q2: cfg=(256,256,128) accepted",
          sg.is_valid_tiling(TC(256, 256, 128)));
    // cfg.w = 128 < K = 256 → rejected.
    CHECK("cfg=(128,256,128) rejected (w < K=256)",
          !sg.is_valid_tiling(TC(128, 256, 128)));
    // cfg.w = 256, cfg.k = 256: cfg.k = K on both axes still fine.
    CHECK("cfg=(256,256,256) accepted (w = K, k = K)",
          sg.is_valid_tiling(TC(256, 256, 256)));
}

// ============================================================================
// Multi-PW chain before MM: PW1 → PW2 → MM. The BFS must walk through
// the PW chain and still impose Rule 2 on every PW in the chain.
// ============================================================================

void test_rule2_pw_chain_before_mm() {
    std::cout << "--- test_rule2_pw_chain_before_mm ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    //            T0        T1        T2        T3        T4
    p.ops = {{OpType::Pointwise, {0}, {1}, 1000},     // PW1: T0 → T1
             {OpType::Pointwise, {1}, {2}, 1000},     // PW2: T1 → T2
             {OpType::MatMul,    {2,3}, {4}, 2000}};  // MM:  T2 × T3 → T4
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("cfg=(128,128,128) accepted (w = K)",
          sg.is_valid_tiling(TC(128, 128, 128)));
    CHECK("cfg=(64,128,64) rejected (Rule 2 propagates through PW chain)",
          !sg.is_valid_tiling(TC(64, 128, 64)));
}

// ============================================================================
// Negative control: pure-MM chain (no prologue PW). Rule 2/3 should not
// reject any cfg on account of missing PW producers.
// ============================================================================

void test_pure_mm_chain_no_rule2_constraint() {
    std::cout << "--- test_pure_mm_chain_no_rule2_constraint ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    //            T0        T1        T2        T3        T4
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::MatMul, {2,3}, {4}, 2000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // Small cfg.w and cfg.h are fine — there is no prologue PW.
    CHECK("cfg=(64,64,64) accepted (no prologue PW)",
          sg.is_valid_tiling(TC(64, 64, 64)));
    CHECK("cfg=(32,32,32) accepted (no prologue PW)",
          sg.is_valid_tiling(TC(32, 32, 32)));
}

// ============================================================================
// Negative control: MM → PW (epilogue, not prologue). The PW is fed by
// the MM, not feeding it — Rule 2/3 does not apply.
// ============================================================================

void test_mm_pw_epilogue_no_rule2() {
    std::cout << "--- test_mm_pw_epilogue_no_rule2 ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    //            T0        T1        T2        T3
    p.ops = {{OpType::MatMul,    {0,1}, {2}, 2000},  // MM: T0 × T1 → T2
             {OpType::Pointwise, {2},   {3}, 1000}}; // PW: T2 → T3 (epilogue)
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // The epilogue PW is downstream of MM; MM has no PW prologue.
    // PW-sink subgraph forces k=1 elsewhere, but cfg.w/cfg.h are free.
    CHECK("cfg=(64,64,1) accepted (epilogue PW only)",
          sg.is_valid_tiling(TC(64, 64, 1)));
    CHECK("cfg=(32,32,1) accepted (epilogue PW only)",
          sg.is_valid_tiling(TC(32, 32, 1)));
}

// ============================================================================
// Threshold boundary: cfg exactly equal to K passes, K-1 fails.
// (Our divisibility constraints mean we can't simply use K-1, but we can
// contrast K with values strictly below that still divide dims.)
// ============================================================================

void test_rule2_threshold_exactly_K() {
    std::cout << "--- test_rule2_threshold_exactly_K ---\n";
    // Shape 256×256 so K=256; divisors 128, 64, 32 all fall below K.
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 1000},
             {OpType::MatMul,    {1,2}, {3}, 2000}};
    p.fast_memory_capacity = 10'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 256; p.native_h = 256;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK("cfg.w=256 accepted (= K)",  sg.is_valid_tiling(TC(256,256,128)));
    CHECK("cfg.w=128 rejected (< K)",  !sg.is_valid_tiling(TC(128,256,128)));
    CHECK("cfg.w=64 rejected (< K)",   !sg.is_valid_tiling(TC( 64,256, 64)));
    CHECK("cfg.w=32 rejected (< K)",   !sg.is_valid_tiling(TC( 32,256, 32)));
}

// ============================================================================
// Rules 2 + 3 active simultaneously on different axes.
// A single PW produces a shared tensor fed to two MMs: one uses it as
// LHS (→ Rule 2 on cfg.w), another uses it as RHS (→ Rule 3 on cfg.h).
// With a non-square intermediate, the two K values differ, so both
// axes carry independent thresholds.
//
//   PW:   T0 (128×128) → T1 (128×128)
//   MM_A: T1 × T2 → T3     (T1 as LHS; K_A = T1.width = 128 → cfg.w ≥ 128)
//   MM_B: T4 × T1 → T5     (T1 as RHS; K_B = T1.height = 128 — matches
//                           T4.width; but to make K_B distinct we'd need
//                           non-square T1, which makes boundary-output
//                           (W,H) mismatch. Keep square T1 here and just
//                           verify both rules fire.)
// ============================================================================

void test_rule2_and_rule3_simultaneous() {
    std::cout << "--- test_rule2_and_rule3_simultaneous ---\n";
    // T1 square → K_A = K_B = 128. Both Rule 2 and Rule 3 active, both
    // demanding cfg.w ≥ 128 and cfg.h ≥ 128.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
    //            T0        T1        T2        T3 (out) T4        T5 (out)
    p.ops = {{OpType::Pointwise, {0},   {1}, 1000},
             {OpType::MatMul,    {1,2}, {3}, 2000},    // MM_A: LHS = T1
             {OpType::MatMul,    {4,1}, {5}, 2000}};   // MM_B: RHS = T1
    p.fast_memory_capacity = 10'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("cfg=(128,128,128) accepted (both rules satisfied)",
          sg.is_valid_tiling(TC(128, 128, 128)));
    CHECK("cfg=(64,128,64) rejected (w < 128, Rule 2)",
          !sg.is_valid_tiling(TC(64, 128, 64)));
    CHECK("cfg=(128,64,64) rejected (h < 128, Rule 3)",
          !sg.is_valid_tiling(TC(128, 64, 64)));
    CHECK("cfg=(64,64,64) rejected (both rules violated)",
          !sg.is_valid_tiling(TC(64, 64, 64)));
}

// ============================================================================
// Max of K across multiple PW→MM pairs on the same axis.
// Sequential chain: PW_A → MM_A → PW_B → MM_B. Both MM_A and MM_B have
// PW prologues feeding their LHS, with different K values. Threshold on
// cfg.w = max(K_A, K_B).
//
//   PW_A: T0 (128×256) → T1 (128×256)
//   MM_A: T1 × T2 → T3       T2 = (256, 128), T3 = (256, 256).  K_A = 128
//   PW_B: T3 (256×256) → T4 (256×256)
//   MM_B: T4 × T5 → T6       T5 = (128, 256), T6 = (128, 256).  K_B = 256
// ============================================================================

void test_rule2_max_across_same_axis() {
    std::cout << "--- test_rule2_max_across_same_axis ---\n";
    Problem p;
    p.tensors = {{128,256},{128,256},{256,128},{256,256},
                 {256,256},{128,256},{128,256}};
    //            T0         T1         T2         T3 (eph)
    //            T4 (eph)   T5         T6 (out)
    p.ops = {{OpType::Pointwise, {0},   {1}, 1000},     // PW_A
             {OpType::MatMul,    {1,2}, {3}, 2000},     // MM_A: K_A=128
             {OpType::Pointwise, {3},   {4}, 1000},     // PW_B
             {OpType::MatMul,    {4,5}, {6}, 2000}};    // MM_B: K_B=256
    p.fast_memory_capacity = 20'000'000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 256; p.native_h = 256;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2, 3});

    // cfg.k ≤ min(K) = 128 per #75. Use cfg.k = 128.
    CHECK("cfg.w=256 accepted (= max K)",
          sg.is_valid_tiling(TC(256, 256, 128)));
    CHECK("cfg.w=128 rejected (≥ K_A=128 but < K_B=256)",
          !sg.is_valid_tiling(TC(128, 256, 128)));
    CHECK("cfg.w=64 rejected (< both Ks)",
          !sg.is_valid_tiling(TC(64, 256, 64)));
}

// ============================================================================
// Subgraph containing only the PW (no downstream MM in the subgraph) —
// Rule 2/3 must not fire, since the MM is outside the subgraph boundary.
// ============================================================================

void test_rule2_mm_outside_subgraph_no_constraint() {
    std::cout << "--- test_rule2_mm_outside_subgraph_no_constraint ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise, {0},   {1}, 1000},   // PW: T0 → T1
             {OpType::MatMul,    {1,2}, {3}, 2000}};  // MM: T1 × T2 → T3
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    // Subgraph containing only the PW op.
    auto sg = make_sg(p, d, {0});

    // With MM outside, no prologue constraint — cfg.w can be small.
    CHECK("pure-PW subgraph: cfg=(64,64,1) accepted",
          sg.is_valid_tiling(TC(64, 64, 1)));
    CHECK("pure-PW subgraph: cfg=(32,32,1) accepted",
          sg.is_valid_tiling(TC(32, 32, 1)));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    test_rule2_lhs_prologue_basic();
    test_rule3_rhs_prologue_basic();
    test_rule2_organizer_q2_example();
    test_rule2_pw_chain_before_mm();
    test_pure_mm_chain_no_rule2_constraint();
    test_mm_pw_epilogue_no_rule2();
    test_rule2_threshold_exactly_K();
    test_rule2_and_rule3_simultaneous();
    test_rule2_max_across_same_axis();
    test_rule2_mm_outside_subgraph_no_constraint();

    std::cout << "\n=== " << g_pass << " passed, " << g_fail
              << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
