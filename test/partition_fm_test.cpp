// partition_search_test.cpp
//
// Tests for three things we changed/added this session:
//
//   1. DAG::merge_creates_cycle  — new bitmask implementation, extended cases
//   2. Partition::creates_ephemeral_gap — new function, zero prior coverage
//   3. Partition::finalize() + Solution::from_partition — correctness for
//      recompute partitions (root cause of the constructed_optima Instance 6 failure)
//
// Build: add to CMakeLists similarly to other *_test targets.

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "partition/partition.h"
#include "solution/solution.h"
#include "solution/ordering.h"
#include <cmath>
#include <iostream>
#include <set>

static int g_pass = 0, g_fail = 0;

static void CHECK(const char* label, bool cond) {
    if (cond) { g_pass++; }
    else { g_fail++; std::cout << "  FAIL: " << label << "\n"; }
}

static void CHECK_EQ(const char* label, double got, double exp, double tol = 0.5) {
    if (std::abs(got - exp) < tol) { g_pass++; }
    else { g_fail++; std::cout << "  FAIL: " << label
                               << " got=" << got << " exp=" << exp << "\n"; }
}

// ============================================================================
// Test problems
// ============================================================================

// Chain: T0→Op0→T1→Op1→T2→Op2→T3→Op3→T4
static Problem make_chain4() {
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({128, 128});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Diamond: T0→Op0→T1→{Op1→T2, Op2→T3}, (T2,T3)→Op3→T4
static Problem make_diamond4() {
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({128, 128});
    p.ops = {
        {OpType::Pointwise, {0}, {1}, 1000},      // Op0: T0→T1
        {OpType::Pointwise, {1}, {2}, 1000},      // Op1: T1→T2
        {OpType::Pointwise, {1}, {3}, 1000},      // Op2: T1→T3
        {OpType::Pointwise, {2,3}, {4}, 1000}     // Op3: T2,T3→T4
    };
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Y-graph: T0→Op0→T1, T1→Op1→T2, T1→Op2→T3
// Uses the exact same tensors as constructed_optima Instance 6 so we can
// verify cost numbers directly.
static Problem make_Y_instance6() {
    Problem p;
    p.tensors = {{128,128},{256,256},{128,128},{128,128}};
    p.ops = {
        {OpType::Pointwise, {0}, {1}, 100},   // Op0: T0(128²)→T1(256²)
        {OpType::Pointwise, {1}, {2}, 100},   // Op1: T1(256²)→T2(128²)
        {OpType::Pointwise, {1}, {3}, 100}    // Op2: T1(256²)→T3(128²)
    };
    p.fast_memory_capacity = 60000;           // T1(65536) > 60000 → can't retain
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// Section 1: merge_creates_cycle
// ============================================================================

// --- 1a: chain of 4 ---
void test_cycle_chain4() {
    std::cout << "=== cycle: chain of 4 ===\n";
    auto p = make_chain4();
    DAG d = DAG::build(p);

    // Adjacent pairs: no cycle
    CHECK("chain {0},{1} ok",   !d.merge_creates_cycle({0}, {1}));
    CHECK("chain {1},{2} ok",   !d.merge_creates_cycle({1}, {2}));
    CHECK("chain {2},{3} ok",   !d.merge_creates_cycle({2}, {3}));

    // One hop gap: cycle (1 lies between 0 and 2)
    CHECK("chain {0},{2} cycle", d.merge_creates_cycle({0}, {2}));
    // Two hop gap: also cycle
    CHECK("chain {0},{3} cycle", d.merge_creates_cycle({0}, {3}));
    CHECK("chain {1},{3} cycle", d.merge_creates_cycle({1}, {3}));

    // Multi-op groups — still adjacent, no cycle
    CHECK("chain {0,1},{2} ok", !d.merge_creates_cycle({0,1}, {2}));
    CHECK("chain {0},{1,2} ok", !d.merge_creates_cycle({0}, {1,2}));
    CHECK("chain {0,1},{2,3} ok", !d.merge_creates_cycle({0,1}, {2,3}));

    // Merging {0,2} with {1} yields S={0,1,2}.  The condensed DAG becomes S→{3},
    // which is acyclic.  merge_creates_cycle asks "does the RESULT have a cycle?"
    // — not "was the starting partition valid?".  Correct answer is false.
    CHECK("chain {0,2},{1} no cycle in result", !d.merge_creates_cycle({0,2}, {1}));
}

// --- 1b: diamond ---
void test_cycle_diamond() {
    std::cout << "=== cycle: diamond ===\n";
    auto p = make_diamond4();
    DAG d = DAG::build(p);

    // Parallel branches (Op1 and Op2) share parent but not successor edge
    // Merging them is safe: no external op between them that could loop back
    CHECK("diamond parallel {1},{2} ok", !d.merge_creates_cycle({1}, {2}));

    // Root→leaf skip creates cycle
    CHECK("diamond {0},{3} cycle",  d.merge_creates_cycle({0}, {3}));

    // Root with one branch: no cycle (Op2 is the only thing "between" but it
    // doesn't have a path back into {Op0,Op1})
    CHECK("diamond {0},{1} ok",  !d.merge_creates_cycle({0}, {1}));
    CHECK("diamond {0},{2} ok",  !d.merge_creates_cycle({0}, {2}));

    // Leaf with one branch: ok
    CHECK("diamond {1},{3} ok",  !d.merge_creates_cycle({1}, {3}));
    CHECK("diamond {2},{3} ok",  !d.merge_creates_cycle({2}, {3}));

    // Merging root with both branches: still ok (no node left "outside" that
    // can form a back-edge, just the sink Op3 which isn't in either set)
    CHECK("diamond {0},{1,2} ok", !d.merge_creates_cycle({0}, {1,2}));

    // Merging root+sink skips the branches: cycle
    CHECK("diamond {0,3},{1} cycle", d.merge_creates_cycle({0,3}, {1}));
    CHECK("diamond {0,3},{2} cycle", d.merge_creates_cycle({0,3}, {2}));
}

// --- 1c: symmetry and single-op identity ---
void test_cycle_symmetry() {
    std::cout << "=== cycle: symmetry & single ops ===\n";
    auto p = make_chain4();
    DAG d = DAG::build(p);

    // merge_creates_cycle must be symmetric
    for (size_t a = 0; a < 4; a++)
        for (size_t b = 0; b < 4; b++) {
            bool ab = d.merge_creates_cycle({a}, {b});
            bool ba = d.merge_creates_cycle({b}, {a});
            CHECK("symmetric", ab == ba);
        }

    // Merging a set with itself is always cycle-free (degenerate; S = A∪B = A)
    // Our implementation: external_out from S \ S is empty → false
    CHECK("self-merge {0}",   !d.merge_creates_cycle({0}, {0}));
    CHECK("self-merge {0,1}", !d.merge_creates_cycle({0,1}, {0,1}));
}

// --- 1d: can_reach helper ---
void test_can_reach() {
    std::cout << "=== can_reach: chain and diamond ===\n";
    auto p = make_chain4();
    DAG d = DAG::build(p);

    CHECK("0 can reach 3",  d.can_reach(0, 3));
    CHECK("0 can reach 1",  d.can_reach(0, 1));
    CHECK("3 cannot reach 0", !d.can_reach(3, 0));
    CHECK("2 cannot reach 0", !d.can_reach(2, 0));
    CHECK("self reach 0",    d.can_reach(0, 0));
    CHECK("self reach 3",    d.can_reach(3, 3));

    auto pd = make_diamond4();
    DAG dd = DAG::build(pd);
    CHECK("d 0→3", dd.can_reach(0, 3));
    CHECK("d 1→3", dd.can_reach(1, 3));
    CHECK("d 2→3", dd.can_reach(2, 3));
    CHECK("d 1 not reach 2", !dd.can_reach(1, 2)); // parallel branches
    CHECK("d 2 not reach 1", !dd.can_reach(2, 1));
    CHECK("d 3 not reach 0", !dd.can_reach(3, 0));
}

// ============================================================================
// Section 2: creates_ephemeral_gap
// ============================================================================

// --- 2a: Y-graph baseline ---
// Trivial partition {Op0}, {Op1}, {Op2}.
// Proposing to merge {Op0+Op1} creates a gap because T1 becomes ephemeral
// and Op2 (in a separate group) needs T1 from slow memory.
void test_gap_Y_merge_creates_gap() {
    std::cout << "=== gap: Y merge creates gap ===\n";
    auto p = make_Y_instance6();
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    // Group 0={Op0}, group 1={Op1}, group 2={Op2}

    // Proposing {Op0+Op1}: T1 ephemeral, Op2 needs T1 but no group exports it
    CHECK("Y {Op0+Op1} creates gap",
          part.creates_ephemeral_gap({0, 1}, 0, 1));

    // Same for {Op0+Op2}: T1 ephemeral, Op1 needs T1
    CHECK("Y {Op0+Op2} creates gap",
          part.creates_ephemeral_gap({0, 2}, 0, 2));

    // Fusing all three {Op0+Op1+Op2}: T1 ephemeral but both consumers are
    // internal → no external consumer → no gap.
    // However {Op0+Op1+Op2} is actually rejected by Subgraph::create because
    // T1 has two internal consumers (fan-out). creates_ephemeral_gap only
    // checks partition-level gaps, not subgraph validity.
    // T1 consumers: Op1 (in proposed) and Op2 (in proposed) → consumed_internally=true
    // No external consumer of T1 → no gap from partition perspective.
    CHECK("Y {Op0+Op1+Op2} no gap (all internal)",
          !part.creates_ephemeral_gap({0, 1, 2}, 0, 2));
}

// --- 2b: chain has NO ephemeral gaps ---
// In a chain each internal tensor has exactly one consumer, so no external
// consumer can be left stranded.
void test_gap_chain_no_gap() {
    std::cout << "=== gap: chain has no gaps ===\n";
    auto p = make_chain4();
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    // groups: 0={Op0}, 1={Op1}, 2={Op2}, 3={Op3}

    CHECK("chain {Op0+Op1} no gap", !part.creates_ephemeral_gap({0,1}, 0, 1));
    CHECK("chain {Op1+Op2} no gap", !part.creates_ephemeral_gap({1,2}, 1, 2));
    CHECK("chain {Op0+Op1+Op2} no gap", !part.creates_ephemeral_gap({0,1,2}, 0, 2));
    CHECK("chain all-fused no gap", !part.creates_ephemeral_gap({0,1,2,3}, 0, 3));
}

// --- 2c: recompute EXEMPTS the gap ---
// After merging {Op0+Op1} into group ga, the partition has:
//   ga = {Op0+Op1},  gb = {Op2}
// Proposing {Op0+Op2} (which RECOMPUTEs Op0 alongside Op2) should NOT
// create a gap: T1 is ephemeral in the proposed {Op0+Op2}, but Op1 is in ga
// which ALSO contains Op0 (the producer of T1) — so Op1 can recompute T1.
void test_gap_recompute_exempts() {
    std::cout << "=== gap: recompute exempts ephemeral gap ===\n";
    auto p = make_Y_instance6();
    DAG d = DAG::build(p);

    // Build partition: ga={Op0+Op1}, gb={Op2}
    Partition part;
    part.prob = &p;
    part.dag  = &d;
    size_t ga = part.add_group({0, 1}, part.eval_set({0, 1}));  // {Op0,Op1}
    size_t gb = part.add_group({2},    part.eval_set({2}));      // {Op2}
    // add_group incrementally maintains op_to_groups_

    // Verify setup: op 0 (Op0) is in group ga
    CHECK("Op0 in ga", part.groups_of(0).size() == 1 &&
                       part.groups_of(0)[0] == ga);
    CHECK("Op2 in gb", part.groups_of(2).size() == 1 &&
                       part.groups_of(2)[0] == gb);

    // Propose {Op0+Op2} (RECOMPUTE-style), replacing gb, keeping ga.
    // T1 is ephemeral in {Op0+Op2}. External consumer Op1 is in ga, which
    // also contains Op0 → ga can recompute T1 for Op1.  No gap.
    CHECK("recompute {Op0+Op2} no gap",
          !part.creates_ephemeral_gap({0, 2}, SIZE_MAX, gb));

    // Sanity: now swap — propose {Op0+Op1} replacing ga, keeping gb.
    // T1 ephemeral in {Op0+Op1}. External consumer Op2 is in gb = {Op2}.
    // gb does NOT contain Op0.  Gap.
    CHECK("recompute {Op0+Op1} with gap",
          part.creates_ephemeral_gap({0, 1}, ga, SIZE_MAX));
}

// --- 2d: gap with a shared boundary output ---
// If another alive group already exports T1 to slow memory (boundary_output),
// then proposing a group that makes T1 ephemeral does NOT create a gap —
// other consumers can still load T1 from slow memory.
void test_gap_other_group_exports() {
    std::cout << "=== gap: other group already exports tensor ===\n";
    auto p = make_Y_instance6();
    DAG d = DAG::build(p);

    // Partition: g0={Op0} (exports T1), g1={Op1} (consumes T1), g2={Op2} (consumes T1)
    // Start trivial
    Partition part = Partition::trivial(p, d);
    // groups[0]={Op0}, groups[1]={Op1}, groups[2]={Op2}

    // Proposing {Op0+Op1}: T1 ephemeral in proposal, Op2 in group 2 needs T1.
    // Group 0 ({Op0}) is being excluded (it's the source for this merge).
    // No other alive group exports T1 (only group 0 did, but it's excluded).
    CHECK("no other exporter → gap",
          part.creates_ephemeral_gap({0, 1}, 0, 1));

    // Now add an extra group that ALSO produces T1 via Op0 (simulates recompute
    // of Op0 elsewhere).  Use a fresh partition.
    Partition part2;
    part2.prob = &p;
    part2.dag  = &d;
    size_t g_singleton_Op0 = part2.add_group({0}, part2.eval_set({0})); // exports T1
    size_t g_Op1 = part2.add_group({1}, part2.eval_set({1}));
    size_t g_Op2 = part2.add_group({2}, part2.eval_set({2}));

    // Proposing {Op0+Op1} replacing g_singleton_Op0 and g_Op1.
    // T1 ephemeral. External consumer Op2 in g_Op2.
    // Are there other exporters of T1 besides g_singleton_Op0?
    // No — g_singleton_Op0 is the only group with Op0, and it's excluded.
    CHECK("only one exporter excluded → gap",
          part2.creates_ephemeral_gap({0, 1},
                                      g_singleton_Op0, g_Op1));

    // If we DON'T exclude g_singleton_Op0 (i.e. it stays alive), then the
    // proposal {Op0+Op1} makes T1 ephemeral BUT g_singleton_Op0 still exports
    // T1 to slow memory.  Op2 can load it.  No gap.
    // exclude_ga = SIZE_MAX (don't exclude the singleton), exclude_gb = g_Op1
    CHECK("exporter stays alive → no gap",
          !part2.creates_ephemeral_gap({0, 1},
                                       SIZE_MAX, g_Op1));
}

// ============================================================================
// Section 3: Partition::finalize() correctness
// ============================================================================

// --- 3a: trivial partition already has sg populated ---
void test_finalize_trivial() {
    std::cout << "=== finalize: trivial partition ===\n";
    auto p = make_chain4();
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.finalize();  // idempotent

    for (size_t i = 0; i < part.groups.size(); i++) {
        if (!part.groups[i].alive) continue;
        CHECK("trivial group has sg", part.groups[i].sg.has_value());
        CHECK("trivial group cost finite", part.groups[i].cost < 1e17);
    }
    // group DAG: chain topology — each group has exactly one predecessor
    // except the first
    CHECK("chain DAG in_deg[0]==0", part.group_in_deg[0] == 0);
    CHECK("chain DAG in_deg[1]==1", part.group_in_deg[1] == 1);
    CHECK("chain DAG in_deg[2]==1", part.group_in_deg[2] == 1);
    CHECK("chain DAG in_deg[3]==1", part.group_in_deg[3] == 1);
}

// --- 3b: finalize populates sg for a merged partition ---
void test_finalize_merged() {
    std::cout << "=== finalize: merged partition ===\n";
    auto p = make_chain4();
    DAG d = DAG::build(p);

    Partition part;
    part.prob = &p;
    part.dag  = &d;
    part.add_group({0, 1, 2}, part.eval_set({0, 1, 2}));
    part.add_group({3},       part.eval_set({3}));

    // Before finalize: sg is not set (add_group doesn't populate it)
    CHECK("before finalize: g0.sg absent", !part.groups[0].sg.has_value());

    part.finalize();

    CHECK("after finalize: g0.sg present", part.groups[0].sg.has_value());
    CHECK("after finalize: g1.sg present", part.groups[1].sg.has_value());
    CHECK("g0 boundary inputs contain T0",
          part.groups[0].sg->boundary_inputs().count(0));
    CHECK("g0 boundary outputs contain T3",
          part.groups[0].sg->boundary_outputs().count(3));

    // Group DAG: g0→g1
    CHECK("merged DAG in_deg[0]==0", part.group_in_deg[0] == 0);
    CHECK("merged DAG in_deg[1]==1", part.group_in_deg[1] == 1);
}

// --- 3c: finalize on a RECOMPUTE partition ---
// The classic Instance 6 setup: {Op0+Op1} and {Op0+Op2}.
// Op0 appears in both groups.  finalize() must:
//  - Create valid Subgraph for each group
//  - Build group DAG with both groups having in_deg=0 (both depend only on T0
//    which is a graph input, not produced by any group)
void test_finalize_recompute_partition() {
    std::cout << "=== finalize: recompute partition ===\n";
    auto p = make_Y_instance6();
    DAG d = DAG::build(p);

    Partition part;
    part.prob = &p;
    part.dag  = &d;
    size_t ga = part.add_group({0, 1}, part.eval_set({0, 1}));  // {Op0,Op1}
    size_t gb = part.add_group({0, 2}, part.eval_set({0, 2}));  // {Op0,Op2}

    part.finalize();

    CHECK("ga sg present", part.groups[ga].sg.has_value());
    CHECK("gb sg present", part.groups[gb].sg.has_value());

    // ga: T0→Op0→T1(ephemeral)→Op1→T2. boundary_inputs={T0}, boundary_outputs={T2}
    CHECK("ga bi={T0}", part.groups[ga].sg->boundary_inputs()  == std::set<size_t>{0});
    CHECK("ga bo={T2}", part.groups[ga].sg->boundary_outputs() == std::set<size_t>{2});

    // gb: T0→Op0→T1(ephemeral)→Op2→T3. boundary_inputs={T0}, boundary_outputs={T3}
    CHECK("gb bi={T0}", part.groups[gb].sg->boundary_inputs()  == std::set<size_t>{0});
    CHECK("gb bo={T3}", part.groups[gb].sg->boundary_outputs() == std::set<size_t>{3});

    // Both groups depend only on T0 (graph input) → both in_deg = 0
    CHECK("recompute DAG in_deg[ga]==0", part.group_in_deg[ga] == 0);
    CHECK("recompute DAG in_deg[gb]==0", part.group_in_deg[gb] == 0);

    // Op0 appears in both groups
    CHECK("Op0 in two groups", part.groups_of(0).size() == 2);
}

// ============================================================================
// Section 4: Solution::from_partition on a recompute partition
//
// This is the direct regression test for constructed_optima Instance 6.
// The old solver.cpp used GroupDAGInfo (which failed for recompute partitions);
// the new solver.cpp uses Solution::from_partition which handles them.
// ============================================================================

void test_from_partition_recompute() {
    std::cout << "=== from_partition: recompute partition (Instance 6 regression) ===\n";
    auto p = make_Y_instance6();
    DAG d = DAG::build(p);

    // Hand-compute expected costs (same numbers as constructed_optima_test):
    //   {Op0+Op1}: T1 ephemeral. load T0(128*128/10=1638.4) + evict T2(1638.4)
    //              = max(200, 3276.8) = 3276.8
    //   {Op0+Op2}: same structure, same cost = 3276.8
    //   Total recompute cost = 6553.6
    Partition tmp; tmp.prob = &p; tmp.dag = &d;
    double cost_01 = tmp.eval_set({0, 1});
    double cost_02 = tmp.eval_set({0, 2});
    CHECK_EQ("cost {Op0+Op1}", cost_01, 3276.8);
    CHECK_EQ("cost {Op0+Op2}", cost_02, 3276.8);

    // Build the recompute partition
    Partition part;
    part.prob = &p;
    part.dag  = &d;
    part.add_group({0, 1}, cost_01);
    part.add_group({0, 2}, cost_02);

    // from_partition calls finalize(), then dfs+beam ordering
    Solution sol = Solution::from_partition(p, d, part);

    CHECK("sol not empty", !sol.steps().empty());
    CHECK("sol has 2 steps", sol.steps().size() == 2);

    auto vr = sol.validate();
    CHECK("sol is valid", vr.valid);
    if (!vr.valid) std::cout << "    error: " << vr.error << "\n";

    CHECK_EQ("sol total latency == recompute cost",
             sol.total_latency(), cost_01 + cost_02);

    // All three ops must be covered (Op0 recomputed in both steps)
    std::set<size_t> covered;
    for (auto& step : sol.steps())
        for (auto op : step.subgraph.ops())
            covered.insert(op);
    CHECK("Op0 covered", covered.count(0));
    CHECK("Op1 covered", covered.count(1));
    CHECK("Op2 covered", covered.count(2));
}

// --- Also verify that a fused partition (with ephemeral gap) produces an
//     INVALID solution, confirming that validate() actually catches the bug. ---
void test_from_partition_gap_is_invalid() {
    std::cout << "=== from_partition: fused-with-gap partition is invalid ===\n";
    auto p = make_Y_instance6();
    DAG d = DAG::build(p);

    // Partition: {Op0+Op1} and {Op2}.
    // T1 is ephemeral in {Op0+Op1}, but {Op2} needs T1 from slow memory.
    Partition tmp; tmp.prob = &p; tmp.dag = &d;
    double cost_01 = tmp.eval_set({0, 1});
    double cost_2  = tmp.eval_set({2});

    Partition part;
    part.prob = &p;
    part.dag  = &d;
    part.add_group({0, 1}, cost_01);
    part.add_group({2},    cost_2);

    Solution sol = Solution::from_partition(p, d, part);
    auto vr = sol.validate();

    // The solution is INVALID: Op2 needs T1 from slow memory but T1 was
    // never written (it was ephemeral in {Op0+Op1}).
    CHECK("fused-gap solution is invalid", !vr.valid);
    if (vr.valid) std::cout << "  (expected invalid but got valid)\n";
}

// ============================================================================
// Section 5: DFS and beam orderings on recompute partitions
// ============================================================================

void test_ordering_recompute() {
    std::cout << "=== ordering: recompute partition ===\n";
    auto p = make_Y_instance6();
    DAG d = DAG::build(p);

    Partition part;
    part.prob = &p;
    part.dag  = &d;
    part.add_group({0, 1}, part.eval_set({0, 1}));
    part.add_group({0, 2}, part.eval_set({0, 2}));
    part.finalize();

    // Both orderings must include all alive groups
    auto dfs  = dfs_ordering(part);
    CHECK("dfs order covers 2 groups", dfs.order.size() == 2);

    auto beam = beam_search_ordering(part, 5);
    CHECK("beam order covers 2 groups", beam.order.size() == 2);

    // In both orderings, neither group appears twice
    std::set<size_t> dfs_seen(dfs.order.begin(), dfs.order.end());
    CHECK("dfs no duplicates", dfs_seen.size() == 2);

    std::set<size_t> beam_seen(beam.order.begin(), beam.order.end());
    CHECK("beam no duplicates", beam_seen.size() == 2);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // 1. Cycle detection
    test_cycle_chain4();
    test_cycle_diamond();
    test_cycle_symmetry();
    test_can_reach();

    // 2. Ephemeral gap detection
    test_gap_Y_merge_creates_gap();
    test_gap_chain_no_gap();
    test_gap_recompute_exempts();
    test_gap_other_group_exports();

    // 3. finalize()
    test_finalize_trivial();
    test_finalize_merged();
    test_finalize_recompute_partition();

    // 4. Solution::from_partition (Instance 6 regression)
    test_from_partition_recompute();
    test_from_partition_gap_is_invalid();

    // 5. Ordering
    test_ordering_recompute();

    std::cout << "\n" << g_pass << " passed, " << g_fail
              << " failed out of " << g_pass + g_fail << " tests\n";
    return g_fail > 0 ? 1 : 0;
}