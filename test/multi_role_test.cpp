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
// Extended A @ A coverage
// ============================================================================

// A @ A where the op is the only thing in the subgraph — verify exact WS
// contribution of the two T0 entries.
void test_a_at_a_ws_exact() {
    std::cout << "--- test_a_at_a_ws_exact ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,0}, {1}, 2000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // cfg=(128,128,128): ntw=nth=nk=1. Every slice = full tensor = 16384.
    // T0 has 2 entries (LHS + RHS), each contributes 16384 → T0 total 32768.
    // T1 (output) has 1 entry → 16384.
    // Working set = 32768 + 16384 = 49152.
    auto ws = sg.working_set(TC(128, 128, 128), {}, {});
    CHECK_EQ_I("A@A ws(128,128,128) = 49152 (T0×2 + T1)", ws, 49152);
}

// A @ A vs A @ B should yield different WS: A@A has T0 with 2 entries
// (same tensor, 2 roles), A@B has T0+T1 each with 1 entry.
// Same total memory footprint by coincidence here (both tensors are
// identical size), but the entry counts differ.
void test_a_at_a_vs_a_at_b_entries() {
    std::cout << "--- test_a_at_a_vs_a_at_b_entries ---\n";
    // A @ A case.
    Problem p_aa;
    p_aa.tensors = {{128,128},{128,128}};
    p_aa.ops = {{OpType::MatMul, {0,0}, {1}, 2000}};
    p_aa.fast_memory_capacity = 200000;
    p_aa.slow_memory_bandwidth = 10;
    p_aa.native_w = 128; p_aa.native_h = 128;
    DAG d_aa = DAG::build(p_aa);
    auto sg_aa = make_sg(p_aa, d_aa, {0});

    CHECK_EQ_I("A@A: T0 has 2 entries", entries_for(sg_aa, 0), 2);

    // A @ B baseline.
    Problem p_ab;
    p_ab.tensors = {{128,128},{128,128},{128,128}};
    p_ab.ops = {{OpType::MatMul, {0,1}, {2}, 2000}};
    p_ab.fast_memory_capacity = 200000;
    p_ab.slow_memory_bandwidth = 10;
    p_ab.native_w = 128; p_ab.native_h = 128;
    DAG d_ab = DAG::build(p_ab);
    auto sg_ab = make_sg(p_ab, d_ab, {0});

    CHECK_EQ_I("A@B: T0 has 1 entry", entries_for(sg_ab, 0), 1);
    CHECK_EQ_I("A@B: T1 has 1 entry", entries_for(sg_ab, 1), 1);
}

// A @ A inside a larger subgraph (followed by a PW). T0 still has 2 entries,
// the PW op consumes the MM output (single-role).
void test_a_at_a_followed_by_pw() {
    std::cout << "--- test_a_at_a_followed_by_pw ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,    {0,0}, {1}, 2000},
             {OpType::Pointwise, {1},   {2}, 500}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    CHECK_EQ_I("T0 entries (A@A still 2)", entries_for(sg, 0), 2);
    // T1 is ephemeral (produced by MM, consumed by PW) → not in boundary.
    CHECK_EQ_I("T1 entries (ephemeral, none)", entries_for(sg, 1), 0);
    CHECK_EQ_I("T2 entries (output)", entries_for(sg, 2), 1);
}

// ============================================================================
// Extended multi-role coverage
// ============================================================================

// Three ops produce three different roles but one collapses via #70 full:
// impossible to construct FULL from 3 partials, so this actually stays a
// 2-partial case after any legitimate combination. Instead: test a 4-MM
// chain where propagation yields FULL + 1 partial for some tensor.
//
// Easier: build a subgraph where two partials + one PW consumer → 3 partials
// but the PW role happens to match one of the MM roles → deduped to 2.
void test_multi_role_pw_dedups_with_mm() {
    std::cout << "--- test_multi_role_pw_dedups_with_mm ---\n";
    // T0 consumed by:
    //   PW (role inherited from its output's consumer)
    //   MM (role FROM_NK, FROM_NTH for LHS)
    // If the PW role happens to = the MM role, they dedup.
    // Construction: PW output is T1 (boundary_out), role (FROM_NTW, FROM_NTH).
    //   PW input T0 inherits → (FROM_NTW, FROM_NTH).
    // If T0 is ALSO MM RHS via another op with role (FROM_NTW, FROM_NK) —
    // different v_source → distinct. So 2 distinct partials, not dedup.
    // Distinct but both valid, no 3rd role → subgraph accepted.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise, {0},   {1}, 500},     // T0 as PW input
             {OpType::MatMul,    {2,0}, {3}, 2000}};   // T0 as MM RHS
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // T0 has 2 distinct partials (PW input ≠ MM RHS roles).
    CHECK_EQ_I("T0 entries (PW + MM RHS)", entries_for(sg, 0), 2);
}

// Multi-role within a longer chain: shared weight tensor consumed at
// multiple depths in a chain of MMs.
void test_shared_weight_at_multiple_depths() {
    std::cout << "--- test_shared_weight_at_multiple_depths ---\n";
    // T_w(0) used as LHS of MM0 and MM2 (both sinks).
    // MM0: T_w × T1 → T3 (sink)
    // MM1: T_w × T2 → T4 (sink)  — same role as MM0 → dedup to 1 entry.
    // MM2: T5 × T_w → T6 (sink)  — different role (RHS) → 2nd entry.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},
                 {128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {3}, 2000},  // T_w as LHS of MM0
             {OpType::MatMul, {0,2}, {4}, 2000},  // T_w as LHS of MM1
             {OpType::MatMul, {5,0}, {6}, 2000}}; // T_w as RHS of MM2
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // T_w has 2 distinct signatures (LHS-of-sink, RHS-of-sink). Two LHS
    // uses dedup. Total = 2 entries.
    CHECK_EQ_I("T_w entries (LHS×2 + RHS, deduped)", entries_for(sg, 0), 2);
}

// Multi-role ephemeral: a PW output consumed by two MMs with different
// roles. Note: ephemerals aren't in boundary_tensor_info_, but we can
// verify the subgraph is accepted (no rejection for multi-role ephemeral).
void test_multi_role_ephemeral_accepted() {
    std::cout << "--- test_multi_role_ephemeral_accepted ---\n";
    // PW: T0 → T1 (ephemeral)
    // MM0: T1 × T2 → T3 (sink)  — T1 as LHS
    // MM1: T4 × T1 → T5 (sink)  — T1 as RHS
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},
                 {128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise, {0},   {1}, 500},
             {OpType::MatMul,    {1,2}, {3}, 2000},
             {OpType::MatMul,    {4,1}, {5}, 2000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg_opt = Subgraph::create(p, d, {0, 1, 2});

    // Subgraph should be accepted (no 3rd partial for any tensor).
    CHECK("subgraph accepted (ephemeral multi-role OK)",
          sg_opt.has_value());
    if (sg_opt) {
        // T1 is ephemeral → 0 entries in boundary_tensor_info_.
        CHECK_EQ_I("T1 ephemeral (0 boundary entries)",
                   entries_for(*sg_opt, 1), 0);
    }
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

    test_a_at_a_ws_exact();
    test_a_at_a_vs_a_at_b_entries();
    test_a_at_a_followed_by_pw();
    test_multi_role_pw_dedups_with_mm();
    test_shared_weight_at_multiple_depths();
    test_multi_role_ephemeral_accepted();

    std::cout << "\n=== " << g_pass << " passed, " << g_fail
              << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
