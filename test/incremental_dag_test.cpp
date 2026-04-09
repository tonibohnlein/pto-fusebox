// incremental_dag_test.cpp
// Tests that GroupDAG::update (incremental) produces the same adjacency
// as GroupDAG::build (full rebuild) after partition moves.

#include "core/dag.h"
#include "core/group_dag.h"
#include "core/types.h"
#include "partition/partition.h"
#include "search/fm_search.h"
#include "search/partition_moves.h"
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}

// Compare incremental GroupDAG against a fresh full build.
static bool verify_matches_full(GroupDAG& gdag, const Partition& part,
                                 const char* label) {
    GroupDAG fresh;
    fresh.build(part);

    bool match = true;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        if (gdag.succs(gi) != fresh.succs(gi)) {
            std::cout << "  MISMATCH succs at " << label << " G" << gi << ":\n";
            std::cout << "    incr: {";
            for (auto g : gdag.succs(gi)) std::cout << g << ",";
            std::cout << "}\n    full: {";
            for (auto g : fresh.succs(gi)) std::cout << g << ",";
            std::cout << "}\n";
            match = false;
        }
        if (gdag.preds(gi) != fresh.preds(gi)) {
            std::cout << "  MISMATCH preds at " << label << " G" << gi << ":\n";
            std::cout << "    incr: {";
            for (auto g : gdag.preds(gi)) std::cout << g << ",";
            std::cout << "}\n    full: {";
            for (auto g : fresh.preds(gi)) std::cout << g << ",";
            std::cout << "}\n";
            match = false;
        }
    }
    return match;
}

// Simple chain: Op0→T1→Op1→T2→Op2→T3→Op3→T4
static Problem make_chain4() {
    Problem p;
    p.tensors = {{64,64},{64,64},{64,64},{64,64},{64,64}};
    p.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
        {OpType::Pointwise, {2}, {3}, 300},
        {OpType::Pointwise, {3}, {4}, 300},
    };
    p.fast_memory_capacity = 20000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 64; p.native_h = 64;
    return p;
}

// Diamond: Op0→T1, Op1→T2, Op2(T1)→T3, Op3(T2,T3)→T4
static Problem make_diamond() {
    Problem p;
    p.tensors = {{64,64},{64,64},{64,64},{64,64},{64,64}};
    p.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {0}, {2}, 300},
        {OpType::Pointwise, {1}, {3}, 300},
        {OpType::Pointwise, {2, 3}, {4}, 300},
    };
    p.fast_memory_capacity = 20000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 64; p.native_h = 64;
    return p;
}

void test_full_build() {
    std::cout << "=== test_full_build ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // G0→G1→G2→G3
    CHECK("G0→G1", gdag.succs(0).count(1));
    CHECK("G1→G2", gdag.succs(1).count(2));
    CHECK("G2→G3", gdag.succs(2).count(3));
    CHECK("G3 no succ", gdag.succs(3).empty());
    CHECK("G0 no pred", gdag.preds(0).empty());
    CHECK("G1 pred G0", gdag.preds(1).count(0));
}

void test_can_reach() {
    std::cout << "=== test_can_reach ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    CHECK("G0 reaches G3", gdag.can_reach(0, 3));
    CHECK("G0 reaches G1", gdag.can_reach(0, 1));
    CHECK("G3 does not reach G0", !gdag.can_reach(3, 0));
    CHECK("G2 does not reach G0", !gdag.can_reach(2, 0));
    CHECK("G0 reaches itself", gdag.can_reach(0, 0));
}

void test_merge_creates_cycle() {
    std::cout << "=== test_merge_creates_cycle ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Merging G0+G3: merged→G2 via T1, G2→merged via T3 → CYCLE
    CHECK("merge G0,G3 creates cycle", gdag.merge_creates_cycle({0, 3}));
    // Merging G1+G2: both feed G3, neither depends on the other → safe
    CHECK("merge G1,G2 no cycle", !gdag.merge_creates_cycle({1, 2}));
}

void test_steal_incremental() {
    std::cout << "=== test_steal_incremental ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Steal Op1 from G1 to G0
    auto affected = partition_moves::apply_steal(part, 1, 1, 0);
    CHECK("steal applied", !affected.empty());
    part.rebuild_index();
    gdag.update(part, affected);

    CHECK("steal matches full", verify_matches_full(gdag, part, "steal"));

    // Verify reachability after steal
    // G0={Op0,Op1}→G2→G3, G1 dead
    CHECK("G0 reaches G2", gdag.can_reach(0, 2));
    CHECK("G0 reaches G3", gdag.can_reach(0, 3));
}

void test_merge_incremental() {
    std::cout << "=== test_merge_incremental ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    double mc = part.eval_set({0, 1});
    auto affected = partition_moves::apply_merge(part, 0, 1, mc);
    CHECK("merge applied", !affected.empty());
    part.rebuild_index();
    gdag.update(part, affected);

    CHECK("merge matches full", verify_matches_full(gdag, part, "merge"));
}

void test_eject_incremental() {
    std::cout << "=== test_eject_incremental ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    // Start with G0={Op0,Op1,Op2}
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    auto er = part.eval_eject(1, 0);
    if (!er.feasible) {
        std::cout << "  skip: eject not feasible\n";
        return;
    }
    auto affected = partition_moves::apply_eject(part, 1, 0);
    CHECK("eject applied", !affected.empty());
    part.rebuild_index();
    gdag.update(part, affected);

    CHECK("eject matches full", verify_matches_full(gdag, part, "eject"));
}

void test_diamond_merge_incremental() {
    std::cout << "=== test_diamond_merge_incremental ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Merge G0 and G2 (Op0 + Op2, T1 becomes ephemeral)
    double mc = part.eval_set({0, 2});
    auto affected = partition_moves::apply_merge(part, 0, 2, mc);
    CHECK("diamond merge applied", !affected.empty());
    part.rebuild_index();
    gdag.update(part, affected);

    CHECK("diamond merge matches full",
          verify_matches_full(gdag, part, "diamond_merge"));
}

void test_multiple_moves() {
    std::cout << "=== test_multiple_moves ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Move 1: Merge G0+G1
    double mc1 = part.eval_set({0, 1});
    auto a1 = partition_moves::apply_merge(part, 0, 1, mc1);
    part.rebuild_index();
    gdag.update(part, a1);
    CHECK("move1 matches", verify_matches_full(gdag, part, "move1"));

    // Move 2: Merge G0+G2
    double mc2 = part.eval_set({0, 1, 2});
    auto a2 = partition_moves::apply_merge(part, 0, 2, mc2);
    part.rebuild_index();
    gdag.update(part, a2);
    CHECK("move2 matches", verify_matches_full(gdag, part, "move2"));
}

void test_topo_order() {
    std::cout << "=== test_topo_order ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Chain: G0→G1→G2→G3. Topo order must be monotonically increasing.
    CHECK("topo G0 < G1", gdag.topo_pos(0) < gdag.topo_pos(1));
    CHECK("topo G1 < G2", gdag.topo_pos(1) < gdag.topo_pos(2));
    CHECK("topo G2 < G3", gdag.topo_pos(2) < gdag.topo_pos(3));
}

void test_topo_fast_reject() {
    std::cout << "=== test_topo_fast_reject ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // can_reach should use topo order for O(1) rejection
    CHECK("G3 cannot reach G0 (topo)", !gdag.can_reach(3, 0));
    CHECK("G2 cannot reach G0 (topo)", !gdag.can_reach(2, 0));
    CHECK("G0 can reach G3 (forward)", gdag.can_reach(0, 3));
}

void test_topo_after_merge() {
    std::cout << "=== test_topo_after_merge ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Merge G1+G2
    double mc = part.eval_set({1, 2});
    auto affected = partition_moves::apply_merge(part, 1, 2, mc);
    part.rebuild_index();
    gdag.update(part, affected);

    // G0 → G1(={Op1,Op2}) → G3. Topo should respect this.
    CHECK("topo G0 < G1 after merge", gdag.topo_pos(0) < gdag.topo_pos(1));
    CHECK("topo G1 < G3 after merge", gdag.topo_pos(1) < gdag.topo_pos(3));
    // G2 is dead
    CHECK("G2 dead topo -1", gdag.topo_pos(2) == -1);
}

void test_topo_diamond() {
    std::cout << "=== test_topo_diamond ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // G0 must be before G2 and G1. G3 must be after G2 and G1.
    CHECK("topo G0 < G2", gdag.topo_pos(0) < gdag.topo_pos(2));
    CHECK("topo G0 < G1", gdag.topo_pos(0) < gdag.topo_pos(1));
    CHECK("topo G2 < G3", gdag.topo_pos(2) < gdag.topo_pos(3));
    CHECK("topo G1 < G3", gdag.topo_pos(1) < gdag.topo_pos(3));
}

void test_acyclicity_after_build() {
    std::cout << "=== test_acyclicity_after_build ===\n";
    // Chain: should be acyclic
    auto p1 = make_chain4();
    auto d1 = DAG::build(p1);
    auto part1 = Partition::trivial(p1, d1);
    part1.rebuild_index();

    GroupDAG gdag1;
    gdag1.build(part1);
    CHECK("chain is acyclic (no self-cycle)", !gdag1.can_reach(1, 0));
    CHECK("chain is acyclic (no reverse)", !gdag1.can_reach(3, 0));
    CHECK("chain forward works", gdag1.can_reach(0, 3));

    // Diamond: check various reachability
    auto p2 = make_diamond();
    auto d2 = DAG::build(p2);
    auto part2 = Partition::trivial(p2, d2);
    part2.rebuild_index();

    GroupDAG gdag2;
    gdag2.build(part2);
    CHECK("diamond G0→G3", gdag2.can_reach(0, 3));
    CHECK("diamond G1→G3", gdag2.can_reach(1, 3));
    CHECK("diamond G0→G2", gdag2.can_reach(0, 2));
    CHECK("diamond !G3→G0", !gdag2.can_reach(3, 0));
    CHECK("diamond !G2→G1", !gdag2.can_reach(2, 1));
}

void test_acyclicity_after_moves() {
    std::cout << "=== test_acyclicity_after_moves ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // After merging G1+G2: G0→G_merged→G3
    double mc = part.eval_set({1, 2});
    auto affected = partition_moves::apply_merge(part, 1, 2, mc);
    part.rebuild_index();
    gdag.update(part, affected);

    CHECK("post-merge G0 reaches G3", gdag.can_reach(0, 3));
    CHECK("post-merge !G3→G0", !gdag.can_reach(3, 0));
    // G0→G1(merged {Op1,Op2})→G3: merging G0+G3 creates cycle:
    // merged produces T1 for G1, G1 produces T3 for G3(=merged) → cycle.
    CHECK("post-merge merge G0,G3 cycle", gdag.merge_creates_cycle({0, 3}));
}

void test_eval_merge() {
    std::cout << "=== test_eval_merge ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // G0+G3 creates cycle (G0→G2→G3 and G0→G1→G3, merging endpoints)
    CHECK("eval_merge G0,G3 cycle", gdag.eval_merge(0, 3));
    // G1+G2 safe (parallel branches, no back-edge)
    CHECK("eval_merge G1,G2 no cycle", !gdag.eval_merge(1, 2));
    // G0+G1 safe (G0 feeds G1, no reverse)
    CHECK("eval_merge G0,G2 no cycle", !gdag.eval_merge(0, 2));
}

void test_apply_merge_method() {
    std::cout << "=== test_apply_merge_method ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    double mc = part.eval_set({0, 1});
    auto affected = partition_moves::apply_merge(part, 0, 1, mc);
    part.rebuild_index();
    gdag.apply_generic(part, affected);

    CHECK("apply_merge matches full", verify_matches_full(gdag, part, "apply_merge"));
    CHECK("topo still valid G0<G3", gdag.topo_pos(0) < gdag.topo_pos(3));
}

void test_apply_steal_method() {
    std::cout << "=== test_apply_steal_method ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    auto affected = partition_moves::apply_steal(part, 1, 1, 0);
    part.rebuild_index();
    gdag.apply_generic(part, affected);

    CHECK("apply_steal matches full", verify_matches_full(gdag, part, "apply_steal"));
}

void test_apply_eject_method() {
    std::cout << "=== test_apply_eject_method ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    auto er = part.eval_eject(1, 0);
    if (!er.feasible) { std::cout << "  skip\n"; return; }
    auto affected = partition_moves::apply_eject(part, 1, 0);
    part.rebuild_index();
    gdag.apply_generic(part, affected);

    CHECK("apply_eject matches full", verify_matches_full(gdag, part, "apply_eject"));
}

// ============================================================================
// Cycle detection tests — verify GroupDAG catches cycles correctly
// ============================================================================

// Wider diamond with more paths:
//   Op0: T0 → T1
//   Op1: T0 → T2
//   Op2: T1 → T3
//   Op3: T2 → T4
//   Op4: T3, T4 → T5
//   Op5: T5 → T6
static Problem make_wide_diamond() {
    Problem p;
    p.tensors = {{64,64},{64,64},{64,64},{64,64},{64,64},{64,64},{64,64}};
    p.ops = {
        {OpType::Pointwise, {0}, {1}, 300},       // Op0
        {OpType::Pointwise, {0}, {2}, 300},       // Op1
        {OpType::Pointwise, {1}, {3}, 300},       // Op2
        {OpType::Pointwise, {2}, {4}, 300},       // Op3
        {OpType::Pointwise, {3, 4}, {5}, 300},    // Op4
        {OpType::Pointwise, {5}, {6}, 300},       // Op5
    };
    p.fast_memory_capacity = 20000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 64; p.native_h = 64;
    return p;
}

void test_cycle_merge_endpoints() {
    std::cout << "=== test_cycle_merge_endpoints ===\n";
    // Wide diamond trivial partition: G0→G2→G4→G5, G1→G3→G4→G5
    // Merging first and last group (G0+G5) creates a cycle:
    //   merged(G0,G5) → G2 → G4 → merged(G0,G5) → cycle
    auto p = make_wide_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    CHECK("merge G0,G5 creates cycle", gdag.eval_merge(0, 5));
    CHECK("merge G0,G4 creates cycle", gdag.eval_merge(0, 4));
    CHECK("merge G1,G4 creates cycle", gdag.eval_merge(1, 4));

    // These should NOT create cycles:
    CHECK("merge G0,G1 no cycle", !gdag.eval_merge(0, 1));
    CHECK("merge G2,G3 no cycle", !gdag.eval_merge(2, 3));
    CHECK("merge G4,G5 no cycle", !gdag.eval_merge(4, 5));
    CHECK("merge G0,G2 no cycle", !gdag.eval_merge(0, 2));
    CHECK("merge G1,G3 no cycle", !gdag.eval_merge(1, 3));
}

void test_cycle_merge_after_steal() {
    std::cout << "=== test_cycle_merge_after_steal ===\n";
    // Chain: G0→G1→G2→G3.
    // Steal Op1 from G1 to G0: G0={Op0,Op1}, G1 dead.
    // After: G0→G2→G3 (safe steal, no cycle).
    // Then check: merging G0 with G3 would create cycle via G2.
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Steal Op1 from G1 to G0 (safe: G0={Op0,Op1}, G1 dies)
    auto affected = partition_moves::apply_steal(part, 1, 1, 0);
    CHECK("steal applied", !affected.empty());
    part.rebuild_index();
    gdag.apply_generic(part, affected);

    CHECK("post-steal adjacency correct", verify_matches_full(gdag, part, "steal"));
    CHECK("post-steal G0 reaches G2", gdag.can_reach(0, 2));
    CHECK("post-steal G0 reaches G3", gdag.can_reach(0, 3));

    // Merging G0+G3: G0={Op0,Op1}→G2→G3. merged→G2→merged = cycle.
    CHECK("post-steal merge G0,G3 cycle", gdag.eval_merge(0, 3));
    // Merging G0+G2: G0→G2 is forward, no reverse → safe.
    CHECK("post-steal merge G0,G2 no cycle", !gdag.eval_merge(0, 2));
}

void test_cycle_multi_merge() {
    std::cout << "=== test_cycle_multi_merge ===\n";
    // Wide diamond: merging G2+G3 (parallel branches) is safe.
    // Then merging result with G0 should still be safe (G0 feeds both).
    auto p = make_wide_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Merge G2+G3 (parallel, safe)
    CHECK("merge G2,G3 no cycle", !gdag.eval_merge(2, 3));
    double mc = part.eval_set({2, 3});
    auto a1 = partition_moves::apply_merge(part, 2, 3, mc);
    part.rebuild_index();
    gdag.apply_generic(part, a1);
    CHECK("post-merge matches", verify_matches_full(gdag, part, "merge23"));

    // Now G2={Op2,Op3}. Merge G2+G4 should be safe (G2 feeds G4).
    CHECK("merge G2,G4 no cycle", !gdag.eval_merge(2, 4));

    // But merge G0+G4 should create cycle: G0→G2→G4 and G0→G1→G4
    CHECK("merge G0,G4 cycle", gdag.eval_merge(0, 4));
}

void test_cycle_reachability_transitivity() {
    std::cout << "=== test_cycle_reachability_transitivity ===\n";
    // Chain: G0→G1→G2→G3.
    // Verify transitive reachability in both directions.
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Forward reachability
    CHECK("G0→G1", gdag.can_reach(0, 1));
    CHECK("G0→G2", gdag.can_reach(0, 2));
    CHECK("G0→G3", gdag.can_reach(0, 3));
    CHECK("G1→G2", gdag.can_reach(1, 2));
    CHECK("G1→G3", gdag.can_reach(1, 3));
    CHECK("G2→G3", gdag.can_reach(2, 3));

    // No reverse reachability
    CHECK("!G1→G0", !gdag.can_reach(1, 0));
    CHECK("!G2→G0", !gdag.can_reach(2, 0));
    CHECK("!G3→G0", !gdag.can_reach(3, 0));
    CHECK("!G2→G1", !gdag.can_reach(2, 1));
    CHECK("!G3→G1", !gdag.can_reach(3, 1));
    CHECK("!G3→G2", !gdag.can_reach(3, 2));

    // Self-reachability
    CHECK("G0→G0", gdag.can_reach(0, 0));
    CHECK("G3→G3", gdag.can_reach(3, 3));
}

void test_cycle_topo_consistency_after_moves() {
    std::cout << "=== test_cycle_topo_consistency_after_moves ===\n";
    // After every move, verify topo order is consistent with edges:
    // for every edge gi→gj, topo_pos[gi] < topo_pos[gj].
    auto p = make_wide_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    auto check_topo_consistent = [&](const char* label) {
        for (size_t gi = 0; gi < gdag.size(); gi++) {
            if (gdag.topo_pos(gi) < 0) continue;
            for (auto gj : gdag.succs(gi)) {
                if (gdag.topo_pos(gj) < 0) continue;
                if (gdag.topo_pos(gi) >= gdag.topo_pos(gj)) {
                    std::cout << "  TOPO VIOLATION at " << label
                              << ": G" << gi << "(pos=" << gdag.topo_pos(gi)
                              << ") → G" << gj << "(pos=" << gdag.topo_pos(gj) << ")\n";
                    return false;
                }
            }
        }
        return true;
    };

    CHECK("initial topo consistent", check_topo_consistent("initial"));

    // Merge G2+G3
    double mc = part.eval_set({2, 3});
    auto a1 = partition_moves::apply_merge(part, 2, 3, mc);
    part.rebuild_index();
    gdag.apply_generic(part, a1);
    CHECK("post-merge23 topo consistent", check_topo_consistent("merge23"));

    // Merge G0+G1
    double mc2 = part.eval_set({0, 1});
    auto a2 = partition_moves::apply_merge(part, 0, 1, mc2);
    part.rebuild_index();
    gdag.apply_generic(part, a2);
    CHECK("post-merge01 topo consistent", check_topo_consistent("merge01"));

    // Merge G0+G2 (now G0={Op0,Op1}, G2={Op2,Op3})
    double mc3 = part.eval_set({0, 1, 2, 3});
    auto a3 = partition_moves::apply_merge(part, 0, 2, mc3);
    part.rebuild_index();
    gdag.apply_generic(part, a3);
    CHECK("post-merge02 topo consistent", check_topo_consistent("merge02"));
}

void test_cycle_compare_with_partition_acyclic_checks() {
    std::cout << "=== test_cycle_compare_with_partition_acyclic_checks ===\n";
    // Verify GroupDAG.eval_merge agrees with Partition.acyclic_merge_local
    // on various merge candidates.
    auto p = make_wide_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Test all pairs
    size_t ng = part.groups.size();
    int mismatches = 0;
    for (size_t ga = 0; ga < ng; ga++) {
        if (!part.groups[ga].alive) continue;
        for (size_t gb = ga + 1; gb < ng; gb++) {
            if (!part.groups[gb].alive) continue;
            bool part_ok = part.acyclic_merge_local(ga, gb);
            bool gdag_ok = !gdag.eval_merge(ga, gb);
            if (part_ok != gdag_ok) {
                std::cout << "  MISMATCH: merge G" << ga << ",G" << gb
                          << " partition=" << part_ok << " gdag=" << gdag_ok << "\n";
                mismatches++;
            }
        }
    }
    CHECK("all merge pairs agree with partition", mismatches == 0);
}

void test_eval_steal_vs_partition() {
    std::cout << "=== test_eval_steal_vs_partition ===\n";
    // Compare GroupDAG::eval_steal with Partition::acyclic_steal_local
    // on all valid steal candidates in a chain.
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    int mismatches = 0;
    for (size_t op = 0; op < p.num_ops(); op++) {
        for (size_t from = 0; from < part.groups.size(); from++) {
            if (!part.groups[from].alive || !part.groups[from].ops.count(op)) continue;
            for (size_t to = 0; to < part.groups.size(); to++) {
                if (to == from || !part.groups[to].alive) continue;
                bool part_ok = part.acyclic_steal_local(op, from, to);
                bool gdag_ok = !gdag.eval_steal(part, op, from, to);
                if (part_ok != gdag_ok) {
                    std::cout << "  MISMATCH: steal op" << op << " from G" << from
                              << " to G" << to << " partition=" << part_ok
                              << " gdag=" << gdag_ok << "\n";
                    mismatches++;
                }
            }
        }
    }
    CHECK("all steal checks agree", mismatches == 0);
}

void test_eval_recompute_vs_partition() {
    std::cout << "=== test_eval_recompute_vs_partition ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    int mismatches = 0;
    for (size_t op = 0; op < p.num_ops(); op++) {
        for (size_t gb = 0; gb < part.groups.size(); gb++) {
            if (!part.groups[gb].alive) continue;
            if (part.groups[gb].ops.count(op)) continue;  // already in gb
            bool part_ok = part.acyclic_recompute_local(op, gb);
            bool gdag_ok = !gdag.eval_recompute(part, op, gb);
            if (part_ok != gdag_ok) {
                std::cout << "  MISMATCH: recompute op" << op << " into G" << gb
                          << " partition=" << part_ok << " gdag=" << gdag_ok << "\n";
                mismatches++;
            }
        }
    }
    CHECK("all recompute checks agree", mismatches == 0);
}

void test_eval_de_recompute_vs_partition() {
    std::cout << "=== test_eval_de_recompute_vs_partition ===\n";
    // Need a partition with recomputation to test de_recompute.
    // Chain: G0={Op0,Op1}, G1={Op0,Op2}, G2={Op3} — Op0 recomputed.
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    // Set up recomputation: Op0 in both G0 and G1
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].ops = {0, 2};
    part.groups[1].cost = part.eval_set({0, 2});
    part.groups[2].alive = false;  // Op2 moved to G1
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Op0 is in G0 and G1
    CHECK("op0 in 2 groups", part.groups_of(0).size() == 2);

    // Compare de_recompute checks
    int mismatches = 0;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        if (!part.groups[gi].ops.count(0)) continue;
        bool part_ok = part.acyclic_de_recompute_local(0, gi);
        bool gdag_ok = !gdag.eval_de_recompute(part, 0, gi);
        if (part_ok != gdag_ok) {
            std::cout << "  MISMATCH: de_recompute op0 from G" << gi
                      << " partition=" << part_ok << " gdag=" << gdag_ok << "\n";
            mismatches++;
        }
    }
    CHECK("all de_recompute checks agree", mismatches == 0);
}

void test_eval_steal_diamond() {
    std::cout << "=== test_eval_steal_diamond ===\n";
    // Diamond: more complex steal scenarios
    auto p = make_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    int mismatches = 0;
    for (size_t op = 0; op < p.num_ops(); op++) {
        for (size_t from = 0; from < part.groups.size(); from++) {
            if (!part.groups[from].alive || !part.groups[from].ops.count(op)) continue;
            for (size_t to = 0; to < part.groups.size(); to++) {
                if (to == from || !part.groups[to].alive) continue;
                bool part_ok = part.acyclic_steal_local(op, from, to);
                bool gdag_ok = !gdag.eval_steal(part, op, from, to);
                if (part_ok != gdag_ok) {
                    std::cout << "  MISMATCH: steal op" << op << " from G" << from
                              << " to G" << to << " partition=" << part_ok
                              << " gdag=" << gdag_ok << "\n";
                    mismatches++;
                }
            }
        }
    }
    CHECK("diamond steal checks agree", mismatches == 0);
}

void test_eval_recompute_diamond() {
    std::cout << "=== test_eval_recompute_diamond ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    int mismatches = 0;
    for (size_t op = 0; op < p.num_ops(); op++) {
        for (size_t gb = 0; gb < part.groups.size(); gb++) {
            if (!part.groups[gb].alive || part.groups[gb].ops.count(op)) continue;
            bool part_ok = part.acyclic_recompute_local(op, gb);
            bool gdag_ok = !gdag.eval_recompute(part, op, gb);
            if (part_ok != gdag_ok) {
                std::cout << "  MISMATCH: recompute op" << op << " into G" << gb
                          << " partition=" << part_ok << " gdag=" << gdag_ok << "\n";
                mismatches++;
            }
        }
    }
    CHECK("diamond recompute checks agree", mismatches == 0);
}

void test_eval_split_vs_partition() {
    std::cout << "=== test_eval_split_vs_partition ===\n";
    // Chain with merged group: G0={Op0,Op1,Op2}, G3={Op3}
    // Split G0 at each bridge and compare with partition.
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    // Split {Op0,Op1,Op2} at bridge (Op0,Op1): side_a={Op0}, side_b={Op1,Op2}
    FlatSet<size_t> side_a1 = {0}, side_b1 = {1, 2};
    bool part_ok1 = part.acyclic_split_local(side_a1, side_b1, 0);
    bool gdag_ok1 = !gdag.eval_split(part, side_a1, side_b1, 0);
    CHECK("split (Op0)|(Op1,Op2) agree", part_ok1 == gdag_ok1);

    // Split at bridge (Op1,Op2): side_a={Op0,Op1}, side_b={Op2}
    FlatSet<size_t> side_a2 = {0, 1}, side_b2 = {2};
    bool part_ok2 = part.acyclic_split_local(side_a2, side_b2, 0);
    bool gdag_ok2 = !gdag.eval_split(part, side_a2, side_b2, 0);
    CHECK("split (Op0,Op1)|(Op2) agree", part_ok2 == gdag_ok2);

    // Diamond with merged group: test split there too
    auto p2 = make_diamond();
    auto d2 = DAG::build(p2);
    auto part2 = Partition::trivial(p2, d2);
    // Merge G0+G2: {Op0, Op2}
    part2.groups[0].ops = {0, 2};
    part2.groups[0].cost = part2.eval_set({0, 2});
    part2.groups[2].alive = false;
    part2.rebuild_index();

    GroupDAG gdag2;
    gdag2.build(part2);

    FlatSet<size_t> sa = {0}, sb = {2};
    bool p2_ok = part2.acyclic_split_local(sa, sb, 0);
    bool g2_ok = !gdag2.eval_split(part2, sa, sb, 0);
    CHECK("diamond split (Op0)|(Op2) agree", p2_ok == g2_ok);
}

void test_eval_extract_vs_partition() {
    std::cout << "=== test_eval_extract_vs_partition ===\n";
    // Wide diamond: extract various op subsets and compare.
    auto p = make_wide_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    int mismatches = 0;
    // Test extracting pairs of ops
    for (size_t a = 0; a < p.num_ops(); a++) {
        for (size_t b = a + 1; b < p.num_ops(); b++) {
            FlatSet<size_t> extract = {a, b};
            bool part_ok = part.acyclic_extract_local(extract);
            bool gdag_ok = !gdag.eval_extract(part, extract);
            if (part_ok != gdag_ok) {
                std::cout << "  MISMATCH: extract {" << a << "," << b << "}"
                          << " partition=" << part_ok << " gdag=" << gdag_ok << "\n";
                mismatches++;
            }
        }
    }
    CHECK("all extract pairs agree", mismatches == 0);
}

void test_eval_add_ops_into_vs_partition() {
    std::cout << "=== test_eval_add_ops_into_vs_partition ===\n";
    // Wide diamond: try adding each op into each other group.
    auto p = make_wide_diamond();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();

    GroupDAG gdag;
    gdag.build(part);

    int mismatches = 0;
    for (size_t op = 0; op < p.num_ops(); op++) {
        for (size_t gi = 0; gi < part.groups.size(); gi++) {
            if (!part.groups[gi].alive) continue;
            if (part.groups[gi].ops.count(op)) continue;
            FlatSet<size_t> new_ops = {op};
            bool part_ok = part.acyclic_add_ops_into(new_ops, gi);
            bool gdag_ok = !gdag.eval_add_ops_into(part, new_ops, gi);
            if (part_ok != gdag_ok) {
                std::cout << "  MISMATCH: add op" << op << " into G" << gi
                          << " partition=" << part_ok << " gdag=" << gdag_ok << "\n";
                mismatches++;
            }
        }
    }
    CHECK("all add_ops_into agree", mismatches == 0);
}

// ============================================================================
// Verify incremental index (op_to_groups_) after apply_fm_move
// ============================================================================

static void verify_index(const Partition& part, const char* label) {
    // Build expected index from scratch
    std::vector<std::vector<size_t>> expected(part.prob->num_ops());
    for (size_t gi = 0; gi < part.groups.size(); gi++)
        if (part.groups[gi].alive)
            for (auto op : part.groups[gi].ops)
                expected[op].push_back(gi);

    bool ok = true;
    for (size_t op = 0; op < part.prob->num_ops(); op++) {
        auto got = part.groups_of(op);
        auto& exp = expected[op];
        std::sort(exp.begin(), exp.end());
        auto got_sorted = got;
        std::sort(got_sorted.begin(), got_sorted.end());
        if (got_sorted != exp) {
            std::cout << "  INDEX MISMATCH at " << label << ": op" << op
                      << " got={";
            for (auto g : got) std::cout << g << ",";
            std::cout << "} expected={";
            for (auto g : exp) std::cout << g << ",";
            std::cout << "}\n";
            ok = false;
        }
    }
    CHECK(label, ok);
}

void test_incremental_index_steal() {
    std::cout << "=== test_incremental_index_steal ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();
    part.group_dag();  // trigger lazy build

    FMMove m;
    m.type = FMMove::STEAL;
    m.op = 1; m.ga = 1; m.gb = 0; m.saving = 1.0;
    auto affected = apply_fm_move(part, m);
    CHECK("steal applied", !affected.empty());
    verify_index(part, "steal index correct");
}

void test_incremental_index_merge() {
    std::cout << "=== test_incremental_index_merge ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();
    part.group_dag();

    FMMove m;
    m.type = FMMove::MERGE;
    m.op = 0; m.ga = 0; m.gb = 1; m.saving = 1.0;
    auto affected = apply_fm_move(part, m);
    CHECK("merge applied", !affected.empty());
    verify_index(part, "merge index correct");
}

void test_incremental_index_eject() {
    std::cout << "=== test_incremental_index_eject ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    // Merge first to have a multi-op group
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();
    part.group_dag();

    FMMove m;
    m.type = FMMove::EJECT;
    m.op = 1; m.ga = 0; m.saving = 1.0;
    auto affected = apply_fm_move(part, m);
    CHECK("eject applied", !affected.empty());
    verify_index(part, "eject index correct");
}

void test_incremental_index_split() {
    std::cout << "=== test_incremental_index_split ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();
    part.group_dag();

    FMMove m;
    m.type = FMMove::SPLIT;
    m.op = 0; m.op2 = 1; m.ga = 0; m.saving = 1.0;
    auto affected = apply_fm_move(part, m);
    CHECK("split applied", !affected.empty());
    verify_index(part, "split index correct");
}

void test_incremental_index_recompute() {
    std::cout << "=== test_incremental_index_recompute ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();
    part.group_dag();

    // Recompute op0 into G1
    FMMove m;
    m.type = FMMove::RECOMPUTE;
    m.op = 0; m.ga = 0; m.gb = 1; m.saving = 1.0;
    auto affected = apply_fm_move(part, m);
    CHECK("recompute applied", !affected.empty());
    verify_index(part, "recompute index correct");
    CHECK("op0 in 2 groups", part.groups_of(0).size() == 2);
}

void test_incremental_index_de_recompute() {
    std::cout << "=== test_incremental_index_de_recompute ===\n";
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    // Set up recomputation: op0 in G0 and G1
    part.groups[1].ops.insert(0);
    part.groups[1].cost = part.eval_set(part.groups[1].ops);
    part.rebuild_index();
    part.group_dag();

    CHECK("op0 in 2 groups before", part.groups_of(0).size() == 2);

    FMMove m;
    m.type = FMMove::DE_RECOMPUTE;
    m.op = 0; m.ga = 1; m.saving = 1.0;
    auto affected = apply_fm_move(part, m);
    CHECK("de_recompute applied", !affected.empty());
    verify_index(part, "de_recompute index correct");
    CHECK("op0 in 1 group after", part.groups_of(0).size() == 1);
}

void test_incremental_index_sequence() {
    std::cout << "=== test_incremental_index_sequence ===\n";
    // Apply a sequence of moves and verify index after each
    auto p = make_chain4();
    auto d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.rebuild_index();
    part.group_dag();

    // Move 1: Steal op1 from G1 to G0
    {
        FMMove m;
        m.type = FMMove::STEAL; m.op = 1; m.ga = 1; m.gb = 0; m.saving = 1.0;
        auto aff = apply_fm_move(part, m);
        CHECK("seq steal applied", !aff.empty());
        verify_index(part, "seq after steal");
    }

    // Move 2: Steal op2 from G2 to G0
    {
        FMMove m;
        m.type = FMMove::STEAL; m.op = 2; m.ga = 2; m.gb = 0; m.saving = 1.0;
        auto aff = apply_fm_move(part, m);
        CHECK("seq steal2 applied", !aff.empty());
        verify_index(part, "seq after steal2");
    }

    // Move 3: Split G0 at bridge (0,1)
    {
        FMMove m;
        m.type = FMMove::SPLIT; m.op = 0; m.op2 = 1; m.ga = 0; m.saving = 1.0;
        auto aff = apply_fm_move(part, m);
        if (!aff.empty()) {
            verify_index(part, "seq after split");
        } else {
            std::cout << "  (split not feasible, skip)\n";
        }
    }

    // Verify GroupDAG also matches
    GroupDAG fresh;
    fresh.build(part);
    bool dag_ok = true;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        if (part.group_dag().succs(gi) != fresh.succs(gi)) dag_ok = false;
    }
    CHECK("seq GroupDAG matches fresh build", dag_ok);
}

int main() {
    test_full_build();
    test_can_reach();
    test_merge_creates_cycle();
    test_steal_incremental();
    test_merge_incremental();
    test_eject_incremental();
    test_diamond_merge_incremental();
    test_multiple_moves();
    test_topo_order();
    test_topo_fast_reject();
    test_topo_after_merge();
    test_topo_diamond();
    test_acyclicity_after_build();
    test_acyclicity_after_moves();
    test_eval_merge();
    test_apply_merge_method();
    test_apply_steal_method();
    test_apply_eject_method();
    test_cycle_merge_endpoints();
    test_cycle_merge_after_steal();
    test_cycle_multi_merge();
    test_cycle_reachability_transitivity();
    test_cycle_topo_consistency_after_moves();
    test_cycle_compare_with_partition_acyclic_checks();
    test_eval_steal_vs_partition();
    test_eval_recompute_vs_partition();
    test_eval_de_recompute_vs_partition();
    test_eval_steal_diamond();
    test_eval_recompute_diamond();

    test_eval_split_vs_partition();
    test_eval_extract_vs_partition();
    test_eval_add_ops_into_vs_partition();
    test_incremental_index_steal();
    test_incremental_index_merge();
    test_incremental_index_eject();
    test_incremental_index_split();
    test_incremental_index_recompute();
    test_incremental_index_de_recompute();
    test_incremental_index_sequence();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail ? 1 : 0;
}
