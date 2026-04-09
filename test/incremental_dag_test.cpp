// incremental_dag_test.cpp
// Tests that GroupDAG::update (incremental) produces the same adjacency
// as GroupDAG::build (full rebuild) after partition moves.

#include "core/dag.h"
#include "core/group_dag.h"
#include "core/types.h"
#include "partition/partition.h"
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
    gdag.apply_merge(part, 0, 1, affected);

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
    gdag.apply_steal(part, affected);

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
    gdag.apply_eject(part, affected);

    CHECK("apply_eject matches full", verify_matches_full(gdag, part, "apply_eject"));
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

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail ? 1 : 0;
}
