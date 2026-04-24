// multi_role_test.cpp
//
// Coverage for multi-role tensor accounting (issues #70 + #73):
//
//   * A boundary tensor accessed under multiple distinct role signatures
//     gets separate entries in boundary_tensor_info_, with WS summing each
//     entry's slice size (#73).
//   * If any signature is FULL (FIXED_1, FIXED_1), it subsumes all partials
//     — a single full entry is kept, partials dropped (#70 collapse).
//   * User's 2-partial simplification: subgraphs where any tensor has >2
//     distinct partial signatures are rejected (Subgraph::create returns
//     nullopt).
//   * A @ A (same-tensor two inputs) is handled naturally: the two
//     assign_or_check calls push distinct signatures, both materialize.

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

static TileConfig TC(int64_t w, int64_t h, int64_t k,
                     SnakeDir s = SnakeDir::None) { return {w, h, k, s}; }

static Subgraph make_sg(const Problem& p, const DAG& d,
                        std::vector<size_t> ops) {
    auto sg = Subgraph::create(p, d, std::move(ops));
    if (!sg) { std::cerr << "FATAL: Subgraph::create failed\n"; std::exit(1); }
    return std::move(*sg);
}

static size_t entries_for(const Subgraph& sg, size_t tensor_id) {
    return sg.boundary_entries_for(tensor_id);
}

// ============================================================================
// Single-role baseline: sanity check that the refactor didn't break the
// common case. A simple MM has each input with exactly one role.
// ============================================================================

void test_single_role_one_entry() {
    std::cout << "--- test_single_role_one_entry ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    CHECK_EQ_I("T0 entries (LHS only)", entries_for(sg, 0), 1);
    CHECK_EQ_I("T1 entries (RHS only)", entries_for(sg, 1), 1);
    CHECK_EQ_I("T2 entries (boundary out)", entries_for(sg, 2), 1);
}

// ============================================================================
// Two distinct partials: T1 used as LHS of one MM and RHS of another.
// Under #73 the two slices coexist, each with its own WS contribution.
// ============================================================================

void test_two_distinct_partials() {
    std::cout << "--- test_two_distinct_partials ---\n";
    //  MM1: T1 × T0 → T3  (T1 is LHS of MM1)
    //  MM2: T2 × T1 → T4  (T1 is RHS of MM2)
    //  Both are sinks (parallel fan-out). Matching 128×128 output shape.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {1,0}, {3}, 2000},
             {OpType::MatMul, {2,1}, {4}, 2000}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // T1 has two distinct partial signatures → 2 entries.
    CHECK_EQ_I("T1 entries (LHS + RHS roles)", entries_for(sg, 1), 2);

    // Other tensors have a single role → 1 entry each.
    CHECK_EQ_I("T0 entries", entries_for(sg, 0), 1);
    CHECK_EQ_I("T2 entries", entries_for(sg, 2), 1);
    CHECK_EQ_I("T3 entries (output)", entries_for(sg, 3), 1);
    CHECK_EQ_I("T4 entries (output)", entries_for(sg, 4), 1);
}

// ============================================================================
// Identical role signatures dedup: T1 consumed by two MMs as RHS (same
// signature both times) → only 1 entry.
// ============================================================================

void test_identical_roles_dedup() {
    std::cout << "--- test_identical_roles_dedup ---\n";
    //  MM1: T0 × T1 → T3  (T1 is RHS)
    //  MM2: T2 × T1 → T4  (T1 is RHS)
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {3}, 2000},
             {OpType::MatMul, {2,1}, {4}, 2000}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // T1 used as RHS twice — same signature → dedup to 1 entry.
    CHECK_EQ_I("T1 entries (deduped)", entries_for(sg, 1), 1);
}

// ============================================================================
// A @ A: a single MatMul with both inputs being the same tensor.
// Handled naturally by the refactor — the two assign_or_check calls
// (LHS and RHS) push distinct signatures into roles_per_tensor.
// ============================================================================

void test_a_at_a_two_entries() {
    std::cout << "--- test_a_at_a_two_entries ---\n";
    //  MM: T0 × T0 → T1  (A @ A)
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,0}, {1}, 2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // T0 is used twice with different role signatures (LHS + RHS) → 2 entries.
    // Per #73 "self-matmul": both slices resident simultaneously.
    CHECK_EQ_I("T0 entries (A@A: LHS + RHS)", entries_for(sg, 0), 2);
    CHECK_EQ_I("T1 entries (output)", entries_for(sg, 1), 1);
}

// ============================================================================
// #70 FULL collapse: when a tensor has a (FIXED_1, FIXED_1) role signature
// from any consumer, it subsumes all partials. In a 3-MM chain A → B → C
// (all non-sink except C), the innermost MM's input ends up with role
// (FIXED_1, FIXED_1) due to propagation.
//
// Chain: A (non-sink) → B (non-sink) → C (sink)
//   C sink:    LHS=(FROM_NK, FROM_NTH), RHS=(FROM_NTW, FROM_NK)
//   B output = C.LHS = (FROM_NK, FROM_NTH). B non-sink:
//     B.LHS = (FIXED_1, FROM_NTH)
//     B.RHS = (FROM_NK, FIXED_1)
//   A output = B.LHS = (FIXED_1, FROM_NTH). A non-sink:
//     A.LHS = (FIXED_1, FROM_NTH)
//     A.RHS = (FIXED_1, FIXED_1)  ← FULL
// So T1 (A's RHS) gets FULL role and collapses to a single entry.
// ============================================================================

void test_full_role_collapse() {
    std::cout << "--- test_full_role_collapse ---\n";
    //  MM_A: T0 × T1 → T2    (non-sink)
    //  MM_B: T2 × T3 → T4    (non-sink)
    //  MM_C: T4 × T5 → T6    (sink)
    //  All 128×128, native 128.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},
                 {128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::MatMul, {2,3}, {4}, 2000},
             {OpType::MatMul, {4,5}, {6}, 2000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // T1 ends up with FULL role — single entry, not split.
    CHECK_EQ_I("T1 entries (FULL role, no split)", entries_for(sg, 1), 1);
    // T0, T3, T5 are also boundary inputs, single-role.
    CHECK_EQ_I("T0 entries", entries_for(sg, 0), 1);
    CHECK_EQ_I("T3 entries", entries_for(sg, 3), 1);
    CHECK_EQ_I("T5 entries", entries_for(sg, 5), 1);
    CHECK_EQ_I("T6 entries (output)", entries_for(sg, 6), 1);
}

// ============================================================================
// 2-partial limit: a tensor with ≥3 distinct partial signatures causes
// Subgraph::create to return nullopt. Contrived via 3 parallel MMs all
// using T0 under different roles:
//   MM1: T1 × T0 → T4  (T0 is RHS)
//   MM2: T0 × T2 → T5  (T0 is LHS)
//   MM3: T3 × T0 → T6  (T0 is RHS)  — same as MM1, deduped
// Hmm, to get 3 distinct partials I need varied role orientations. Use:
//   MM1: T0 × T1 → T_out_a       T0 is LHS     (FROM_NK, FROM_NTH)
//   MM2: T2 × T0 → T_out_b       T0 is RHS     (FROM_NTW, FROM_NK)
//   PW:  T0 → T_out_c            T0 is PW-in   (FROM_NTW, FROM_NTH)
// Three distinct partial signatures → reject.
// ============================================================================

void test_three_partials_rejected() {
    std::cout << "--- test_three_partials_rejected ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},  // T0, T1, T2
                 {128,128},{128,128},{128,128}}; // T3, T4, T5 (outputs)
    p.ops = {{OpType::MatMul,    {0,1}, {3}, 2000},  // T0 LHS
             {OpType::MatMul,    {2,0}, {4}, 2000},  // T0 RHS
             {OpType::Pointwise, {0},   {5}, 500}};  // T0 as PW input
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    auto sg_opt = Subgraph::create(p, d, {0, 1, 2});
    CHECK("subgraph REJECTED (3 distinct partials for T0)",
          !sg_opt.has_value());
}

// ============================================================================
// Working-set sum verification: with 2 partial entries for T1, the WS
// includes contributions from both entries (per #73 "sum of distinct
// slices").
// ============================================================================

void test_working_set_sums_multi_role() {
    std::cout << "--- test_working_set_sums_multi_role ---\n";
    //  Same LHS+RHS setup as test_two_distinct_partials.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {1,0}, {3}, 2000},
             {OpType::MatMul, {2,1}, {4}, 2000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // At cfg=(128,128,128): ntw=nth=nk=1. Every slice = full tensor = 128×128.
    // T1 has 2 entries, each slice = 16384. T0, T2, T3, T4 one entry each.
    // Total = (1 + 2 + 1 + 1 + 1) × 16384 = 6 × 16384 = 98304.
    auto ws = sg.working_set(TC(128, 128, 128), {}, {});
    CHECK_EQ_I("WS sums T1's two role entries", ws, 6L * 128 * 128);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    test_single_role_one_entry();
    test_two_distinct_partials();
    test_identical_roles_dedup();
    test_a_at_a_two_entries();
    test_full_role_collapse();
    test_three_partials_rejected();
    test_working_set_sums_multi_role();

    std::cout << "\n=== " << g_pass << " passed, " << g_fail
              << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
