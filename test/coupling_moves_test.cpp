// coupling_moves_test.cpp
//
// Unit tests for coupling-aware partition move fixups:
//   - MERGE (two-group): fixup_coupling_merge — chain link transfer
//   - TENSOR_MERGE (N-group): same fixup loop per dead group
//   - MERGE: chain-level cycle rejection via acyclic_chain_merge_groups
//   - SPLIT: fixup_coupling_split — incoming/outgoing link redirect
//
// Three topologies:
//   chain6_tight: 6 PW ops in a linear chain (same as coupling_test.cpp)
//   cycle5:       5-op topology where merging two chain endpoints creates a
//                 chain-level cycle that acyclic_merge_local would miss
//   skip4:        5-op topology with a 2-op merged group splittable at a
//                 bridge, with incoming/outgoing couplings from both sides

#include "search/coupled_fm_search.h"
#include "search/coupling_search.h"
#include "search/partition_moves.h"
#include "partition/partition.h"
#include "core/cost_cache.h"
#include "solution/solution.h"
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, size_t got, size_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << got << " exp=" << exp << "\n"; }
}

// ============================================================================
// Consistency check helper
//
// Verifies the coupling data structure invariants:
//   1. next_group/prev_group arrays are large enough
//   2. All coupling edges point to alive groups with symmetric next↔prev
//   3. Every retained map key matches a next_group edge
//   4. Every next_group edge has a retained map entry
//   5. All retained tensors are boundary outputs of their source group
// ============================================================================

static bool cp_is_consistent(const CoupledPartition& cp) {
    const size_t n = cp.part.groups.size();
    const auto& dag = *cp.part.dag;

    if (cp.next_group.size() < n || cp.prev_group.size() < n) return false;

    // Symmetric next↔prev
    for (size_t g = 0; g < n; g++) {
        if (!cp.part.groups[g].alive) continue;
        size_t h = cp.next_group[g];
        if (h == SIZE_MAX) continue;
        if (h >= n || !cp.part.groups[h].alive) return false;
        if (cp.prev_group[h] != g) return false;
    }
    for (size_t g = 0; g < n; g++) {
        if (!cp.part.groups[g].alive) continue;
        size_t p = cp.prev_group[g];
        if (p == SIZE_MAX) continue;
        if (p >= n || !cp.part.groups[p].alive) return false;
        if (cp.next_group[p] != g) return false;
    }

    // retained map keys match next_group edges; tensors are valid boundary outputs
    for (auto& [edge, tensors] : cp.retained) {
        auto [src, dst] = edge;
        if (src >= n || dst >= n) return false;
        if (!cp.part.groups[src].alive || !cp.part.groups[dst].alive) return false;
        if (cp.next_group[src] != dst) return false;
        if (tensors.empty()) return false;
        for (auto t : tensors)
            if (!is_boundary_output_of(cp.part.groups[src].ops, t, dag))
                return false;
    }

    // Every next_group edge has a retained entry with at least one tensor
    for (size_t g = 0; g < n; g++) {
        if (!cp.part.groups[g].alive) continue;
        size_t h = cp.next_group[g];
        if (h == SIZE_MAX) continue;
        if (!cp.retained.count({g, h})) return false;
        if (cp.retained.at({g, h}).empty()) return false;
    }

    return true;
}

// Find the (first alive) group containing op.
static size_t gid_of(const CoupledPartition& cp, size_t op) {
    for (size_t g = 0; g < cp.part.groups.size(); g++)
        if (cp.part.groups[g].alive && cp.part.groups[g].ops.count(op))
            return g;
    return SIZE_MAX;
}

// ============================================================================
// Problem helpers
// ============================================================================

// chain6_tight: 6 PW ops Op0..Op5, tensors T0..T6.
// fast_memory_capacity = 40000; each 128×128 tensor = 16384 bytes.
// Trivial partition: G0={Op0}..G5={Op5}.
static Problem make_chain6_tight() {
    Problem p;
    for (int i = 0; i <= 6; i++) p.tensors.push_back({128, 128});
    for (int i = 0; i < 6; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i + 1)}, 500});
    p.fast_memory_capacity = 40000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128;
    p.native_h = 128;
    p.retainable_tensors = {1, 2, 3, 4, 5};
    return p;
}

// cycle5: 5 PW ops, 6 tensors.
//   Op0(Pa): T0 → T1     — T1 consumed by BOTH Op1(X) and Op2(ga)
//   Op1(X):  T1 → T2     — T2 consumed by Op4(Nb)
//   Op2(ga): T1 → T3     — T3 consumed by Op3(gb)
//   Op3(gb): T3 → T4     — T4 consumed by Op4(Nb)
//   Op4(Nb): {T2,T4} → T5
//
// Group DAG (trivial partition G0..G4):
//   G0 → G1 (T1), G0 → G2 (T1), G1 → G4 (T2), G2 → G3 (T3), G3 → G4 (T4)
//
// Target chains: C_A={G0,G2}, C_X={G1}, C_B={G3,G4}.
// Chain-level DAG: C_A → C_X → C_B, C_A → C_B (G2→G3).
// MERGE(G2,G3) would merge C_A∪C_B={G0,G2,G3,G4} — C_X feeds G4 → cycle.
static Problem make_cycle5() {
    Problem p;
    for (int i = 0; i <= 5; i++) p.tensors.push_back({64, 64});
    // Op0(Pa)
    p.ops.push_back({OpType::Pointwise, {0}, {1}, 200});
    // Op1(X)
    p.ops.push_back({OpType::Pointwise, {1}, {2}, 200});
    // Op2(ga): also consumes T1
    p.ops.push_back({OpType::Pointwise, {1}, {3}, 200});
    // Op3(gb)
    p.ops.push_back({OpType::Pointwise, {3}, {4}, 200});
    // Op4(Nb): two inputs
    p.ops.push_back({OpType::Pointwise, {2, 4}, {5}, 200});
    p.fast_memory_capacity = 20000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 64;
    p.native_h = 64;
    p.retainable_tensors = {1, 2, 3, 4};
    return p;
}

// skip4: 5 PW ops, 7 tensors.
//   Op0(A): T0 → T1
//   Op1(B): T2 → T3
//   Op2:    T3 → T4
//   Op3:    {T1,T4} → T5   ← two boundary inputs after merge(G2,G3)
//   Op4:    T5 → T6
//
// After trivial partition and merge(G2,G3): G_12={Op2,Op3} at index 2.
//   G_12 boundary inputs: T1 (for Op3=side_b), T3 (for Op2=side_a)
//   G_12 boundary output: T5 (from Op3=side_b)
//   G_12 internal: T4 (Op2→Op3)
//
// Bridge edge in G_12: (Op2, Op3) via T4.
// Split at (op_a=2, op_b=3): side_a={Op2} stays at G_12, side_b={Op3} → gb_new.
static Problem make_skip4() {
    Problem p;
    for (int i = 0; i <= 6; i++) p.tensors.push_back({32, 32});
    p.ops.push_back({OpType::Pointwise, {0},    {1},    300});  // Op0
    p.ops.push_back({OpType::Pointwise, {2},    {3},    300});  // Op1
    p.ops.push_back({OpType::Pointwise, {3},    {4},    300});  // Op2
    p.ops.push_back({OpType::Pointwise, {1, 4}, {5},    300});  // Op3
    p.ops.push_back({OpType::Pointwise, {5},    {6},    300});  // Op4
    p.fast_memory_capacity = 5000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 32;
    p.native_h = 32;
    p.retainable_tensors = {1, 3, 5};
    return p;
}

// Trivial coupled partition for chain6_tight.
static CoupledPartition make_chain_cp() {
    static Problem p = make_chain6_tight();
    static DAG d = DAG::build(p);
    Partition part = Partition::trivial(p, d);
    part.finalize();
    CoupledPartition cp;
    cp.init_from(std::move(part));
    return cp;
}

// Trivial coupled partition for cycle5.
static CoupledPartition make_cycle5_cp() {
    static Problem p = make_cycle5();
    static DAG d = DAG::build(p);
    Partition part = Partition::trivial(p, d);
    part.finalize();
    CoupledPartition cp;
    cp.init_from(std::move(part));
    return cp;
}

// Coupled partition for skip4 with G2={Op2,Op3} (merged from trivial).
// out_g12 is set to the index of the merged group (2).
// out_g4  is set to the index of G4={Op4} (4).
static CoupledPartition make_skip4_merged_cp(size_t& out_g12, size_t& out_g4) {
    static Problem p = make_skip4();
    static DAG d = DAG::build(p);
    Partition part = Partition::trivial(p, d);
    part.finalize();
    // Merge G2 (absorbs G3): G2={Op2,Op3}.
    partition_moves::apply_merge(part, 2, 3);
    part.rebuild_index();
    part.rebuild_group_dag();
    CoupledPartition cp;
    cp.init_from(std::move(part));
    out_g12 = 2;
    out_g4  = 4;
    return cp;
}

// ============================================================================
// 1. Smoke test: trivial cp is consistent
// ============================================================================

void test_consistency_trivial() {
    std::cout << "--- test_consistency_trivial ---\n";
    auto cp = make_chain_cp();
    CHECK("trivial cp consistent", cp_is_consistent(cp));
}

// ============================================================================
// 2. MERGE with no coupling — consistency preserved
// ============================================================================

void test_merge_no_coupling_consistent() {
    std::cout << "--- test_merge_no_coupling_consistent ---\n";
    auto cp = make_chain_cp();

    CoupledFMMove m;
    m.type   = CoupledFMMove::MERGE;
    m.op     = 0;
    m.ga     = 0;
    m.gb     = 1;
    m.saving = 0.0;

    auto aff = apply_coupled_fm_move(cp, m);
    CHECK("merge applied", !aff.empty());
    cp.part.rebuild_group_dag();
    CHECK("consistent after merge (no coupling)", cp_is_consistent(cp));
}

// ============================================================================
// 3. TENSOR_MERGE — chain link transferred to survivor
//
// Setup: G1→G2 coupled via T2. Apply TENSOR_MERGE({G0,G1}) (G0 survives).
// Expected: G0 inherits G1's outgoing link → G0→G2 via T2.
// ============================================================================

void test_tensor_merge_chain_link_transferred() {
    std::cout << "--- test_tensor_merge_chain_link_transferred ---\n";
    auto cp = make_chain_cp();

    // T2 is boundary output of G1={Op1} and boundary input of G2={Op2}.
    auto rc = apply_couple(cp, 1, 2, 2);
    CHECK("G1→G2 couple succeeded", !rc.empty());
    CHECK("G1→G2 set", cp.next_group[1] == 2);

    // Apply TENSOR_MERGE({G0, G1}): G0 survives, G1 dies.
    CoupledFMMove m;
    m.type         = CoupledFMMove::TENSOR_MERGE;
    m.op           = 0;
    m.tensor_groups = {0, 1};
    m.saving       = 0.0;

    auto aff = apply_coupled_fm_move(cp, m);
    CHECK("tensor_merge applied", !aff.empty());

    // G0 should now have G2 as its next (inherited from G1).
    CHECK("G0 → G2", cp.next_group[0] == 2);
    CHECK("G2 ← G0", cp.prev_group[2] == 0);
    CHECK("retained has {0,2}", cp.retained.count({0, 2}) > 0);
    if (cp.retained.count({0, 2}))
        CHECK("T2 in retained", cp.retained.at({0, 2}).count(2) > 0);

    // G1's slots cleared.
    CHECK("G1 next cleared", cp.next_group[1] == SIZE_MAX);
    CHECK("G1 prev cleared", cp.prev_group[1] == SIZE_MAX);

    cp.part.rebuild_group_dag();
    CHECK("consistent after tensor_merge", cp_is_consistent(cp));
}

// ============================================================================
// 4. MERGE — two-group: chain link transferred to survivor
//
// Setup: G1→G2 coupled via T2. Apply MERGE(G0, G1) (G0 survives, G1 dies).
// Expected: G0 inherits G1's outgoing link → G0→G2 via T2.
// ============================================================================

void test_merge_chain_link_transferred() {
    std::cout << "--- test_merge_chain_link_transferred ---\n";
    auto cp = make_chain_cp();

    auto rc = apply_couple(cp, 1, 2, 2);
    CHECK("G1→G2 couple succeeded", !rc.empty());

    CoupledFMMove m;
    m.type   = CoupledFMMove::MERGE;
    m.op     = 0;
    m.ga     = 0;
    m.gb     = 1;
    m.saving = 0.0;

    auto aff = apply_coupled_fm_move(cp, m);
    CHECK("merge applied", !aff.empty());

    CHECK("G0 → G2", cp.next_group[0] == 2);
    CHECK("G2 ← G0", cp.prev_group[2] == 0);
    CHECK("retained has {0,2}", cp.retained.count({0, 2}) > 0);
    if (cp.retained.count({0, 2}))
        CHECK("T2 in retained", cp.retained.at({0, 2}).count(2) > 0);

    cp.part.rebuild_group_dag();
    CHECK("consistent after merge", cp_is_consistent(cp));
}

// ============================================================================
// 5. Chain-level cycle rejection
//
// Setup: cycle5 with chains C_A={G0,G2} and C_B={G3,G4}.
// MERGE(G2,G3) would merge C_A∪C_B={G0,G2,G3,G4}, but C_X={G1} has edges
// G0→G1 (T1) and G1→G4 (T2), creating cycle C_M→C_X→C_M.
// Expected: acyclic_chain_merge_groups({G2,G3}) = false.
// Expected: best_coupled_move_for_op(Op3) does NOT return MERGE(G2,G3).
// ============================================================================

void test_merge_chain_cycle_rejected() {
    std::cout << "--- test_merge_chain_cycle_rejected ---\n";
    auto cp = make_cycle5_cp();

    size_t g0 = gid_of(cp, 0);  // Pa
    size_t g2 = gid_of(cp, 2);  // ga
    size_t g3 = gid_of(cp, 3);  // gb
    size_t g4 = gid_of(cp, 4);  // Nb

    // Chain C_A: G0 → G2 (T1: produced by Op0=G0, consumed by Op2=G2)
    auto r1 = apply_couple(cp, g0, g2, 1);
    CHECK("C_A coupled", !r1.empty());

    // Chain C_B: G3 → G4 (T4: produced by Op3=G3, consumed by Op4=G4)
    auto r2 = apply_couple(cp, g3, g4, 4);
    CHECK("C_B coupled", !r2.empty());

    // acyclic_chain_merge_groups must detect the cycle.
    CHECK("chain merge G2,G3 creates cycle",
          !acyclic_chain_merge_groups(cp, {g2, g3}));

    // best_coupled_move_for_op for Op3 must NOT propose MERGE(G2,G3).
    FlatSet<size_t> fr;
    auto best = best_coupled_move_for_op(cp, 3, fr);
    bool is_bad_merge = (best.type == CoupledFMMove::MERGE &&
                         ((best.ga == g2 && best.gb == g3) ||
                          (best.ga == g3 && best.gb == g2)));
    CHECK("best move is not MERGE(G2,G3)", !is_bad_merge);
}

// Also verify that without chains the same pair IS chain-acyclic
// (confirming the test is sensitive to chain state).
void test_merge_no_chains_chain_acyclic() {
    std::cout << "--- test_merge_no_chains_chain_acyclic ---\n";
    auto cp = make_cycle5_cp();

    size_t g2 = gid_of(cp, 2);
    size_t g3 = gid_of(cp, 3);

    // Without coupling chains, merging G2+G3 does not create a chain cycle.
    CHECK("no-chain merge G2,G3 is chain-acyclic",
          acyclic_chain_merge_groups(cp, {g2, g3}));
}

// ============================================================================
// 6. SPLIT with no coupling — consistency preserved
// ============================================================================

void test_split_no_coupling_consistent() {
    std::cout << "--- test_split_no_coupling_consistent ---\n";
    size_t g12, g4;
    auto cp = make_skip4_merged_cp(g12, g4);
    CHECK("initial cp consistent", cp_is_consistent(cp));

    CoupledFMMove m;
    m.type   = CoupledFMMove::SPLIT;
    m.ga     = g12;
    m.op     = 2;    // Op2 → side_a (stays at ga=g12)
    m.op2    = 3;    // Op3 → side_b (goes to gb_new)
    m.saving = 0.0;

    auto aff = apply_coupled_fm_move(cp, m);
    CHECK("split applied", !aff.empty());

    // Two groups in affected: the original g12 (side_a) and a new group (side_b).
    CHECK_EQ("two groups affected", aff.size(), 2);

    cp.part.rebuild_group_dag();
    CHECK("consistent after split (no coupling)", cp_is_consistent(cp));
}

// ============================================================================
// 7. SPLIT — incoming coupling stays at side_a
//
// Setup: G1→G_12 via T3 (T3 consumed by Op2=side_a).
// Expected: coupling stays G1→ga after split.
// ============================================================================

void test_split_incoming_stays_at_side_a() {
    std::cout << "--- test_split_incoming_stays_at_side_a ---\n";
    size_t g12, g4;
    auto cp = make_skip4_merged_cp(g12, g4);

    // G1 (={Op1}) produces T3; T3 consumed by Op2 (side_a after split).
    auto rc = apply_couple(cp, 1, g12, 3);
    CHECK("G1→G12 couple succeeded", !rc.empty());

    CoupledFMMove m;
    m.type   = CoupledFMMove::SPLIT;
    m.ga     = g12;
    m.op     = 2;
    m.op2    = 3;
    m.saving = 0.0;

    auto aff = apply_coupled_fm_move(cp, m);
    CHECK("split applied", !aff.empty());

    // Incoming edge should remain on ga (T3 is consumed by Op2 = side_a).
    CHECK("G1 → ga (coupling stays)", cp.next_group[1] == g12);
    CHECK("ga ← G1", cp.prev_group[g12] == 1);
    CHECK("retained T3 on {1,g12}", cp.retained.count({1, g12}) > 0);
    if (cp.retained.count({1, g12}))
        CHECK("T3 in retained", cp.retained.at({1, g12}).count(3) > 0);

    cp.part.rebuild_group_dag();
    CHECK("consistent after split", cp_is_consistent(cp));
}

// ============================================================================
// 8. SPLIT — incoming coupling redirected to side_b
//
// Setup: G0→G_12 via T1 (T1 consumed by Op3=side_b).
// Expected: coupling redirects to G0→gb_new after split.
// ============================================================================

void test_split_incoming_redirected_to_side_b() {
    std::cout << "--- test_split_incoming_redirected_to_side_b ---\n";
    size_t g12, g4;
    auto cp = make_skip4_merged_cp(g12, g4);

    // G0 (={Op0}) produces T1; T1 consumed by Op3 (side_b after split).
    auto rc = apply_couple(cp, 0, g12, 1);
    CHECK("G0→G12 couple succeeded", !rc.empty());

    CoupledFMMove m;
    m.type   = CoupledFMMove::SPLIT;
    m.ga     = g12;
    m.op     = 2;
    m.op2    = 3;
    m.saving = 0.0;

    auto aff = apply_coupled_fm_move(cp, m);
    CHECK("split applied", !aff.empty());

    size_t gb_new = SIZE_MAX;
    for (auto g : aff) if (g != g12) { gb_new = g; break; }
    CHECK("gb_new found", gb_new != SIZE_MAX);

    // Coupling should redirect to gb_new (T1 consumed by Op3 = side_b).
    CHECK("G0 → gb_new", cp.next_group[0] == gb_new);
    CHECK("gb_new ← G0", cp.prev_group[gb_new] == 0);
    CHECK("retained T1 on {0,gb_new}", cp.retained.count({0, gb_new}) > 0);
    if (cp.retained.count({0, gb_new}))
        CHECK("T1 in retained", cp.retained.at({0, gb_new}).count(1) > 0);

    // ga should have no incoming anymore.
    CHECK("ga no incoming", cp.prev_group[g12] == SIZE_MAX);

    cp.part.rebuild_group_dag();
    CHECK("consistent after split", cp_is_consistent(cp));
}

// ============================================================================
// 9. SPLIT — outgoing coupling redirected to side_b
//
// Setup: G_12→G4 via T5 (T5 produced by Op3=side_b).
// Expected: coupling redirects to gb_new→G4 after split.
// ============================================================================

void test_split_outgoing_redirected_to_side_b() {
    std::cout << "--- test_split_outgoing_redirected_to_side_b ---\n";
    size_t g12, g4;
    auto cp = make_skip4_merged_cp(g12, g4);

    // G_12 produces T5 (via Op3); T5 consumed by G4={Op4}.
    auto rc = apply_couple(cp, g12, g4, 5);
    CHECK("G12→G4 couple succeeded", !rc.empty());

    CoupledFMMove m;
    m.type   = CoupledFMMove::SPLIT;
    m.ga     = g12;
    m.op     = 2;
    m.op2    = 3;
    m.saving = 0.0;

    auto aff = apply_coupled_fm_move(cp, m);
    CHECK("split applied", !aff.empty());

    size_t gb_new = SIZE_MAX;
    for (auto g : aff) if (g != g12) { gb_new = g; break; }
    CHECK("gb_new found", gb_new != SIZE_MAX);

    // Coupling redirects to gb_new (T5 produced by Op3 = side_b).
    CHECK("gb_new → G4", cp.next_group[gb_new] == g4);
    CHECK("G4 ← gb_new", cp.prev_group[g4] == gb_new);
    CHECK("retained T5 on {gb_new,g4}", cp.retained.count({gb_new, g4}) > 0);
    if (cp.retained.count({gb_new, g4}))
        CHECK("T5 in retained", cp.retained.at({gb_new, g4}).count(5) > 0);

    // ga should have no outgoing anymore.
    CHECK("ga no outgoing", cp.next_group[g12] == SIZE_MAX);

    cp.part.rebuild_group_dag();
    CHECK("consistent after split", cp_is_consistent(cp));
}

// ============================================================================
// 10. SPLIT — full chain G1→G_12→G4 splits correctly
//
// Setup: G1→G_12 (T3, stays at ga) and G_12→G4 (T5, redirects to gb_new).
// Expected: G1→ga and gb_new→G4; ga has no outgoing; gb_new has no incoming.
// ============================================================================

void test_split_chain_transferred() {
    std::cout << "--- test_split_chain_transferred ---\n";
    size_t g12, g4;
    auto cp = make_skip4_merged_cp(g12, g4);

    // Chain: G1 → G_12 → G4
    auto r1 = apply_couple(cp, 1, g12, 3);  // stays at ga after split
    CHECK("G1→G12 couple succeeded", !r1.empty());
    auto r2 = apply_couple(cp, g12, g4, 5);  // redirects to gb_new after split
    CHECK("G12→G4 couple succeeded", !r2.empty());

    CoupledFMMove m;
    m.type   = CoupledFMMove::SPLIT;
    m.ga     = g12;
    m.op     = 2;
    m.op2    = 3;
    m.saving = 0.0;

    auto aff = apply_coupled_fm_move(cp, m);
    CHECK("split applied", !aff.empty());

    size_t gb_new = SIZE_MAX;
    for (auto g : aff) if (g != g12) { gb_new = g; break; }
    CHECK("gb_new found", gb_new != SIZE_MAX);

    // Incoming T3 stays at ga (T3 consumed by Op2=side_a).
    CHECK("G1 → ga", cp.next_group[1] == g12);
    CHECK("ga ← G1", cp.prev_group[g12] == 1);
    CHECK("retained T3 on {1,ga}", cp.retained.count({1, g12}) > 0);

    // Outgoing T5 redirects to gb_new (T5 produced by Op3=side_b).
    CHECK("gb_new → G4", cp.next_group[gb_new] == g4);
    CHECK("G4 ← gb_new", cp.prev_group[g4] == gb_new);
    CHECK("retained T5 on {gb_new,g4}", cp.retained.count({gb_new, g4}) > 0);

    // ga has no outgoing; gb_new has no incoming.
    CHECK("ga no outgoing", cp.next_group[g12] == SIZE_MAX);
    CHECK("gb_new no incoming", cp.prev_group[gb_new] == SIZE_MAX);

    cp.part.rebuild_group_dag();
    CHECK("consistent after chain split", cp_is_consistent(cp));
}

// ============================================================================
// 11. COUPLE add-to-existing-edge
//
// Setup: chain6_tight, G0→G1 coupled via T1.
// G0 also has T1 as its only boundary output — no second retainable tensor
// flows G0→G1 in this problem, so we use the skip4 fixture instead.
//
// skip4: G1→G_12 coupled via T3. G1 ONLY produces T3, so no second tensor.
// Use the cycle5 fixture: G0→G2 via T1. G0 also produces T2... wait, no.
//
// Use a simpler test: apply_couple(G0,G1,T1) then apply_couple(G0,G1,T1) again
// — must fail (duplicate). Then manually verify that adding a distinct second
// tensor to the existing edge succeeds by directly calling apply_couple and
// eval_couple with the existing-edge path.
//
// For a genuine second-tensor test we need a problem where two separate
// retainable tensors flow from ga to gb.  Build one inline.
// ============================================================================

// 2-output producer topology:
//   Op0: {T0} → {T1, T2}   (produces two tensors)
//   Op1: {T1, T2} → {T3}   (consumes both)
// G0={Op0}, G1={Op1}. T1 and T2 are both boundary outputs of G0 and
// boundary inputs of G1.
static Problem make_two_output() {
    Problem p;
    // T0 input, T1=op0 output, T2=op1 output, T3=op2 output
    for (int i = 0; i <= 3; i++) p.tensors.push_back({32, 32});
    p.ops.push_back({OpType::Pointwise, {0}, {1}, 300});     // Op0: T0 -> T1
    p.ops.push_back({OpType::Pointwise, {0}, {2}, 300});     // Op1: T0 -> T2
    p.ops.push_back({OpType::Pointwise, {1, 2}, {3}, 300});  // Op2: T1,T2 -> T3
    p.fast_memory_capacity = 5000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 32;
    p.native_h = 32;
    p.retainable_tensors = {1, 2};
    return p;
}

static CoupledPartition make_two_output_cp() {
    static Problem p = make_two_output();
    static DAG d = DAG::build(p);
    // G0={op0, op1} produces T1 and T2; G1={op2} consumes both.
    Partition part = Partition::trivial(p, d);
    // Merge op0 and op1 into group 0
    FlatSet<size_t> merged = {0, 1};
    part.groups[0].ops = merged;
    part.groups[0].cost = part.eval_set(merged);
    part.groups[1].alive = false;
    part.rebuild_index();
    part.finalize();
    CoupledPartition cp;
    cp.init_from(std::move(part));
    return cp;
}

void test_couple_add_to_existing_edge() {
    std::cout << "--- test_couple_add_to_existing_edge ---\n";
    auto cp = make_two_output_cp();

    // G0={op0,op1}, G1 dead, G2={op2}.
    // First COUPLE: G0→G2 via T1 — creates new edge.
    auto r1 = apply_couple(cp, 0, 2, 1);
    CHECK("first couple succeeded", !r1.empty());
    CHECK("edge G0→G2 exists", cp.next_group[0] == 2);
    CHECK_EQ("retained set size = 1", cp.retained[{0,2}].size(), 1);

    // Duplicate: coupling T1 again must fail (already retained).
    auto r_dup = apply_couple(cp, 0, 2, 1);
    CHECK("duplicate couple fails", r_dup.empty());
    CHECK_EQ("retained set still 1 after duplicate", cp.retained[{0,2}].size(), 1);

    // eval_couple for T1 on existing edge must be infeasible (duplicate guard).
    auto ev_dup = eval_couple(cp, 0, 2, 1);
    CHECK("eval_couple duplicate infeasible", !ev_dup.feasible);

    // Second COUPLE: G0→G2 via T2 — adds to existing edge.
    auto ev2 = eval_couple(cp, 0, 2, 2);
    CHECK("eval_couple T2 on existing edge feasible", ev2.feasible);

    auto r2 = apply_couple(cp, 0, 2, 2);
    CHECK("second couple succeeded", !r2.empty());
    CHECK_EQ("retained set size = 2", cp.retained[{0,2}].size(), 2);
    CHECK("T1 still retained", cp.retained[{0,2}].count(1) > 0);
    CHECK("T2 now retained",   cp.retained[{0,2}].count(2) > 0);

    // Edge pointers unchanged.
    CHECK("next_group[0] still 2", cp.next_group[0] == 2);
    CHECK("prev_group[2] still 0", cp.prev_group[2] == 0);

    CHECK("consistent", cp_is_consistent(cp));
}

void test_couple_add_blocked_when_edge_missing() {
    std::cout << "--- test_couple_add_blocked_when_edge_missing ---\n";
    auto cp = make_two_output_cp();

    // G0={op0,op1}, G2={op2}. Couple G0→G2 via T1 first.
    apply_couple(cp, 0, 2, 1);  // G0→G2 via T1
    // G0 is now non-free (next_group[0]=2). Out-of-range target must fail.
    auto ev = eval_couple(cp, 0, 99, 2);
    CHECK("out-of-range gb infeasible", !ev.feasible);

    // eval_couple(G0, G2, T2) with G0→G2 already set = existing edge,
    // which IS feasible (adding T2 to the existing edge).
    auto ev2 = eval_couple(cp, 0, 2, 2);
    CHECK("existing edge eval feasible", ev2.feasible);
}

// ============================================================================
// 14. FORCE_RETAIN with existing outgoing chain link on g_dst
//
// Topology (linear chain, 4 ops, 5 tensors):
//   Op0(A): T0 → T1
//   Op1(B): T1 → T2
//   Op2(C): T2 → T3
//   Op3(D): T3 → T4
//
// Setup: trivial partition → merge G1+G2 → G_BC = {Op1,Op2} at index 1.
//   Bridge in G_BC: (Op1, Op2) via T2 (T2 produced by Op1, consumed by Op2).
//   T1 = boundary input of G_BC (consumed by Op1=side_a).
//   T3 = boundary output of G_BC (produced by Op2=side_b).
//   G_BC has existing outgoing coupling G_BC → G_D via T3.
//
// FORCE_RETAIN(ga=G_A, g_dst=G_BC, op_a_dst=Op1, op_b_dst=Op2, t=T1):
//   Splits G_BC at (Op1,Op2): slot_a={Op1} inherits G_BC's index,
//                              slot_b={Op2} gets a new index.
//   Creates:  G_A → slot_a via T1  (new coupling)
//   Transfers: slot_b → G_D via T3 (old outgoing link moved from G_BC to slot_b)
//
// Expected chain structure after:
//   Chain 1: G_A → slot_a={Op1}   (T1 retained)
//   Chain 2: slot_b={Op2} → G_D   (T3 retained, transferred from G_BC)
//   slot_a has no outgoing coupling; slot_b has no incoming coupling.
// ============================================================================

static Problem make_force_retain_chain4() {
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({32, 32});
    p.ops.push_back({OpType::Pointwise, {0}, {1}, 300});  // Op0(A): T0→T1
    p.ops.push_back({OpType::Pointwise, {1}, {2}, 300});  // Op1(B): T1→T2
    p.ops.push_back({OpType::Pointwise, {2}, {3}, 300});  // Op2(C): T2→T3
    p.ops.push_back({OpType::Pointwise, {3}, {4}, 300});  // Op3(D): T3→T4
    p.fast_memory_capacity = 5000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 32;
    p.native_h = 32;
    p.retainable_tensors = {1, 2, 3};
    return p;
}

void test_force_retain_with_existing_outgoing_link() {
    std::cout << "--- test_force_retain_with_existing_outgoing_link ---\n";

    static Problem p = make_force_retain_chain4();
    static DAG d = DAG::build(p);
    Partition part = Partition::trivial(p, d);
    part.finalize();

    // Merge G1(Op1) into G1, absorbing G2(Op2): G_BC = {Op1,Op2} at index 1.
    partition_moves::apply_merge(part, 1, 2);
    part.rebuild_index();
    part.rebuild_group_dag();

    CoupledPartition cp;
    cp.init_from(std::move(part));

    // Indices after merge: G_A=0, G_BC=1, G2=dead, G_D=3.
    size_t g_a  = 0;
    size_t g_bc = 1;
    size_t g_d  = 3;

    // Verify setup: T3 is boundary output of G_BC.
    CHECK("T3 boundary output of G_BC",
          is_boundary_output_of(cp.part.groups[g_bc].ops, 3, *cp.part.dag));
    // T3 is boundary input of G_D.
    CHECK("T3 boundary input of G_D",
          is_boundary_input_of(cp.part.groups[g_d].ops, 3, *cp.part.dag));

    // Set up existing outgoing chain link: G_BC → G_D via T3.
    auto rc = apply_couple(cp, g_bc, g_d, 3);
    CHECK("G_BC → G_D couple succeeded", !rc.empty());
    CHECK("G_BC has outgoing link", cp.next_group[g_bc] == g_d);
    CHECK("G_D has incoming link",  cp.prev_group[g_d]  == g_bc);

    // Evaluate FORCE_RETAIN: ga=G_A, g_dst=G_BC, op_a_dst=Op1(1), op_b_dst=Op2(2), t=T1(1).
    auto ev = eval_force_retain(cp, g_a, g_bc, 1 /*Op1*/, 2 /*Op2*/, 1 /*T1*/);
    CHECK("eval_force_retain feasible", ev.feasible);
    std::cout << "    saving=" << ev.saving << "\n";

    // Apply.
    auto aff = apply_force_retain(cp, g_a, g_bc, 1 /*Op1*/, 2 /*Op2*/, 1 /*T1*/);
    CHECK("apply_force_retain succeeded", !aff.empty());

    // Find slot_b (the new group that got {Op2}).
    size_t slot_a = g_bc;  // reuses g_dst's index
    size_t slot_b = SIZE_MAX;
    for (auto g : aff)
        if (g != slot_a && g != g_a) { slot_b = g; break; }
    CHECK("slot_b found", slot_b != SIZE_MAX);

    // Chain 1: G_A → slot_a via T1.
    CHECK("G_A → slot_a", cp.next_group[g_a]   == slot_a);
    CHECK("slot_a ← G_A", cp.prev_group[slot_a] == g_a);
    CHECK("retained T1 on G_A→slot_a",
          cp.retained.count({g_a, slot_a}) && cp.retained.at({g_a, slot_a}).count(1));

    // slot_a has no outgoing coupling (it's the tail of chain 1).
    CHECK("slot_a no outgoing", cp.next_group[slot_a] == SIZE_MAX);

    // Chain 2: slot_b → G_D via T3 (transferred from G_BC→G_D).
    CHECK("slot_b → G_D", cp.next_group[slot_b] == g_d);
    CHECK("G_D ← slot_b",  cp.prev_group[g_d]   == slot_b);
    CHECK("retained T3 on slot_b→G_D",
          cp.retained.count({slot_b, g_d}) && cp.retained.at({slot_b, g_d}).count(3));

    // slot_b has no incoming coupling (it's the head of chain 2).
    CHECK("slot_b no incoming", cp.prev_group[slot_b] == SIZE_MAX);

    // The old G_BC→G_D link is gone.
    CHECK("old G_BC→G_D retained gone", !cp.retained.count({g_bc, g_d}));

    CHECK("consistent", cp_is_consistent(cp));
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << "=== coupling_moves_test ===\n";

    test_consistency_trivial();
    test_merge_no_coupling_consistent();
    test_tensor_merge_chain_link_transferred();
    test_merge_chain_link_transferred();
    test_merge_chain_cycle_rejected();
    test_merge_no_chains_chain_acyclic();
    test_split_no_coupling_consistent();
    test_split_incoming_stays_at_side_a();
    test_split_incoming_redirected_to_side_b();
    test_split_outgoing_redirected_to_side_b();
    test_split_chain_transferred();
    test_couple_add_to_existing_edge();
    test_couple_add_blocked_when_edge_missing();
    test_force_retain_with_existing_outgoing_link();

    std::cout << "\n=== " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail ? 1 : 0;
}
