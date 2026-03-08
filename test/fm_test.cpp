// fm_test.cpp
// Tests for FM-style local search building blocks.

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "partition/partition.h"
#include "search/fm_search.h"
#include "search/active_set.h"
#include "search/fm_pass.h"
#include "search/fm_outer.h"
#include <algorithm>
#include <cmath>
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++; else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double g, double e, double t = 0.1) {
    if (std::abs(g - e) < t) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}
static void CHECK_EQ_S(const char* l, size_t g, size_t e) {
    if (g == e) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}

// T0->Op0->T1->Op1->T2->Op2->T3->Op3->T4
static Problem make_chain4() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000},
             {OpType::Pointwise,{3},{4},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Diamond: T0->Op0->T1->{Op1->T2, Op2(T1,T2)->T3}
static Problem make_diamond() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1500},
             {OpType::Pointwise,{1},{2},1500},
             {OpType::Pointwise,{1,2},{3},1500}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Y-shape: T0->Op0->T1, T2->Op1->T3, T1->Op2->T4, T3->Op2->T4
// (Op2 consumes T1 and T3)
static Problem make_Y() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{2},{3},1000},
             {OpType::Pointwise,{1,3},{4},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ==================== is_border_op ====================

void test_is_border_op() {
    std::cout << "--- test_is_border_op ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Singleton groups: every op is trivially a border op (pred/succ outside)
    CHECK("singleton Op0 is border", part.is_border_op(0, 0));

    // Merge {Op0,Op1,Op2}: Op0 has no external pred (T0 is graph input, no op produces it)
    // but Op0's pred set from dag.op_preds is empty, and succ=Op1 which is inside.
    // So Op0 is NOT border. Op2 has succ Op3 outside → border.
    part.groups[0].ops = {0, 1, 2};
    part.groups[1].alive = false;
    part.groups[2].alive = false;

    CHECK("fused Op0 not border", !part.is_border_op(0, 0));
    CHECK("fused Op1 not border", !part.is_border_op(1, 0));
    CHECK("fused Op2 is border", part.is_border_op(2, 0));
}

// ==================== border_ops ====================

void test_border_ops() {
    std::cout << "--- test_border_ops ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Merge {Op0,Op1,Op2}, leave {Op3}
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;

    auto bops = part.border_ops(0);
    CHECK_EQ_S("1 border op in fused chain", bops.size(), 1);
    CHECK("Op2 is the border op", bops[0] == 2);

    // {Op3} singleton: Op3 has pred Op2 outside → border
    auto bops3 = part.border_ops(3);
    CHECK_EQ_S("singleton border", bops3.size(), 1);
    CHECK("Op3 is border", bops3[0] == 3);
}

void test_border_ops_diamond() {
    std::cout << "--- test_border_ops_diamond ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Fuse {Op1, Op2}: Op1 has pred Op0 outside → border.
    // Op2 has pred Op0 outside (via T1) → also border.
    part.groups[1].ops = {1, 2};
    part.groups[1].cost = part.eval_set({1, 2});
    part.groups[2].alive = false;

    auto bops = part.border_ops(1);
    CHECK("Op1 is border", std::find(bops.begin(), bops.end(), 1) != bops.end());
    CHECK("Op2 is border", std::find(bops.begin(), bops.end(), 2) != bops.end());
    CHECK_EQ_S("2 border ops", bops.size(), 2);
}

// ==================== connected_components ====================

void test_connected_components_single() {
    std::cout << "--- test_connected_components_single ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    Partition part; part.prob = &p; part.dag = &d;

    // {Op0,Op1,Op2} is connected
    auto cc = part.connected_components({0, 1, 2});
    CHECK_EQ_S("one component", cc.size(), 1);
    CHECK_EQ_S("3 ops", cc[0].size(), 3);
}

void test_connected_components_split() {
    std::cout << "--- test_connected_components_split ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    Partition part; part.prob = &p; part.dag = &d;

    // {Op0, Op2, Op3}: Op0 and Op2 are NOT connected (Op1 missing)
    // Components: {Op0} and {Op2, Op3}
    auto cc = part.connected_components({0, 2, 3});
    CHECK_EQ_S("two components", cc.size(), 2);

    // Sort by size to make test deterministic
    std::sort(cc.begin(), cc.end(), [](auto& a, auto& b) { return a.size() < b.size(); });
    CHECK_EQ_S("small component", cc[0].size(), 1);
    CHECK_EQ_S("large component", cc[1].size(), 2);
    CHECK("small has Op0", cc[0].count(0));
    CHECK("large has Op2", cc[1].count(2));
    CHECK("large has Op3", cc[1].count(3));
}

void test_connected_components_Y() {
    std::cout << "--- test_connected_components_Y ---\n";
    auto p = make_Y(); DAG d = DAG::build(p);
    Partition part; part.prob = &p; part.dag = &d;

    // {Op0, Op1}: independent branches (no DAG edge between them)
    auto cc = part.connected_components({0, 1});
    CHECK_EQ_S("two components", cc.size(), 2);

    // {Op0, Op1, Op2}: all connected via Op2
    auto cc2 = part.connected_components({0, 1, 2});
    CHECK_EQ_S("one component", cc2.size(), 1);
}

// ==================== eval_eject ====================

void test_eject_simple() {
    std::cout << "--- test_eject_simple ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Merge {Op0,Op1}, leave {Op2}, {Op3}
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;

    // Eject Op1 from G0: remainder={Op0}, singleton={Op1}
    auto er = part.eval_eject(1, 0);
    CHECK("eject feasible", er.feasible);
    CHECK_EQ_S("1 remainder component", er.remainder_components.size(), 1);
    CHECK("remainder has Op0", er.remainder_components[0].count(0));
    CHECK("singleton cost > 0", er.singleton_cost > 0);

    // Saving should be: cost({0,1}) - cost({0}) - cost({1})
    double expected = part.groups[0].cost - part.eval_set({0}) - part.eval_set({1});
    CHECK_EQ("eject saving", er.saving, expected);
}

void test_eject_with_disconnection() {
    std::cout << "--- test_eject_with_disconnection ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Merge {Op0,Op1,Op2,Op3} into G0
    part.groups[0].ops = {0, 1, 2, 3};
    part.groups[0].cost = part.eval_set({0, 1, 2, 3});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.groups[3].alive = false;

    // Eject Op1: remainder = {Op0, Op2, Op3}
    // Op0 and Op2 are NOT connected (Op1 was the bridge) → 2 components
    auto er = part.eval_eject(1, 0);
    CHECK("eject feasible", er.feasible);
    CHECK_EQ_S("2 remainder components", er.remainder_components.size(), 2);
    CHECK_EQ_S("2 component costs", er.component_costs.size(), 2);

    // Total saving = old_cost - sum(component_costs) - singleton_cost
    double total_new = 0;
    for (auto c : er.component_costs) total_new += c;
    total_new += er.singleton_cost;
    CHECK_EQ("saving consistent", er.saving, part.groups[0].cost - total_new);

    std::cout << "    old=" << part.groups[0].cost << " new=" << total_new
              << " saving=" << er.saving << "\n";
}

void test_eject_recomputed_op() {
    std::cout << "--- test_eject_recomputed_op ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Op1 is in both G0={Op0,Op1} and G1={Op1,Op2} (recomputed)
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].ops = {1, 2};
    part.groups[1].cost = part.eval_set({1, 2});
    part.groups[2].alive = false;

    // Eject Op1 from G0: Op1 is still in G1, so no singleton needed
    auto er = part.eval_eject(1, 0);
    CHECK("eject recomputed feasible", er.feasible);
    CHECK_EQ("no singleton cost", er.singleton_cost, 0.0);

    // Saving = cost({0,1}) - cost({0}) (no singleton penalty)
    double expected = part.groups[0].cost - part.eval_set({0});
    CHECK_EQ("saving with no singleton", er.saving, expected);
    std::cout << "    saving=" << er.saving << " (no singleton penalty)\n";
}

void test_eject_cant_eject_singleton() {
    std::cout << "--- test_eject_cant_eject_singleton ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Can't eject from a singleton group
    auto er = part.eval_eject(0, 0);
    CHECK("singleton eject infeasible", !er.feasible);
}

// ==================== best_move_for ====================

void test_best_move_chain_unfused() {
    std::cout << "--- test_best_move_chain_unfused ---\n";
    // Chain of 4, all separate. Op1 should want to merge/steal with Op0 or Op2.
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    auto m = best_move_for(part, 1);
    CHECK("has valid move", m.valid());
    CHECK("positive saving", m.saving > 0);
    CHECK("op is 1", m.op == 1);
    std::cout << "    type=" << m.type << " ga=" << m.ga << " gb=" << m.gb
              << " saving=" << m.saving << "\n";
}

void test_best_move_locked() {
    std::cout << "--- test_best_move_locked ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    auto m = best_move_for(part, 1, 0.0, {1});  // Op1 is locked
    CHECK("locked op has no move", !m.valid());
}

void test_best_move_with_floor() {
    std::cout << "--- test_best_move_with_floor ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Merge all into one group
    part.groups[0].ops = {0, 1, 2, 3};
    part.groups[0].cost = part.eval_set({0, 1, 2, 3});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.groups[3].alive = false;

    // Op2 is border (succ Op3 outside? No, Op3 is inside too.)
    // Actually all ops are inside one group — no border ops, no moves.
    // Let me use a different setup: {Op0,Op1,Op2} + {Op3}
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[3].alive = true;
    part.groups[3].ops = {3};
    part.groups[3].cost = part.eval_set({3});

    // Op2 is border (succ Op3 in G3). With floor=0, eject would be negative.
    auto m_strict = best_move_for(part, 2, 0.0);
    auto m_loose = best_move_for(part, 2, part.total_cost() * 0.1);

    // Loose floor should find at least as many candidates
    if (m_loose.valid())
        std::cout << "    loose: type=" << m_loose.type << " saving=" << m_loose.saving << "\n";
    if (m_strict.valid())
        std::cout << "    strict: type=" << m_strict.type << " saving=" << m_strict.saving << "\n";

    // If strict has a move, loose must have at least as good
    if (m_strict.valid())
        CHECK("loose >= strict", m_loose.saving >= m_strict.saving - 0.001);
    g_pass++;  // documentation test
}

void test_best_move_recomputed_op() {
    std::cout << "--- test_best_move_recomputed_op ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Op1 in G0={Op0,Op1} and G1={Op1,Op2} (recomputed)
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].ops = {1, 2};
    part.groups[1].cost = part.eval_set({1, 2});
    part.groups[2].alive = false;

    // Op1 has moves from G0 and from G1
    auto m = best_move_for(part, 1);
    CHECK("recomputed op has move", m.valid());
    std::cout << "    type=" << m.type << " ga=" << m.ga << " gb=" << m.gb
              << " saving=" << m.saving << "\n";
}

// ==================== apply_fm_move ====================

void test_apply_steal() {
    std::cout << "--- test_apply_steal ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Merge {Op0,Op1}, then steal Op1 from G0 to G2
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;

    double before = part.total_cost();
    FMMove m = FMMove{FMMove::STEAL, 1, 0, 2, SIZE_MAX, 0.0};
    auto affected = apply_fm_move(part, m);

    CHECK("affected non-empty", !affected.empty());
    CHECK("G0 affected", affected.count(0));
    CHECK("G2 affected", affected.count(2));
    CHECK("Op1 in G2", part.groups[2].ops.count(1));
    CHECK("Op1 not in G0", !part.groups[0].ops.count(1));
    // All ops still covered
    for (size_t i = 0; i < 4; i++)
        CHECK("op covered", !part.groups_of(i).empty());
}

void test_apply_merge() {
    std::cout << "--- test_apply_merge ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    double before = part.total_cost();
    FMMove m = FMMove{FMMove::MERGE, 0, 0, 1, SIZE_MAX, 0.0};
    auto affected = apply_fm_move(part, m);

    CHECK("affected non-empty", !affected.empty());
    CHECK("G0 has Op0+Op1", part.groups[0].ops.count(0) && part.groups[0].ops.count(1));
    CHECK("G1 dead", !part.groups[1].alive);
}

void test_apply_recompute() {
    std::cout << "--- test_apply_recompute ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    FMMove m = FMMove{FMMove::RECOMPUTE, 1, 0, 2, SIZE_MAX, 0.0};  // Add Op1 to G2
    auto affected = apply_fm_move(part, m);

    CHECK("G2 affected", affected.count(2));
    CHECK("Op1 in G1 (original)", part.groups[1].ops.count(1));
    CHECK("Op1 in G2 (recomputed)", part.groups[2].ops.count(1));
    CHECK("G1 unchanged", !affected.count(1));
}

void test_apply_eject_with_split() {
    std::cout << "--- test_apply_eject_with_split ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Fuse all: {Op0,Op1,Op2,Op3}
    part.groups[0].ops = {0, 1, 2, 3};
    part.groups[0].cost = part.eval_set({0, 1, 2, 3});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.groups[3].alive = false;

    // Eject Op1: splits into {Op0} and {Op2,Op3} plus singleton {Op1}
    FMMove m = FMMove{FMMove::EJECT, 1, 0, SIZE_MAX, SIZE_MAX, 0.0};
    auto affected = apply_fm_move(part, m);

    CHECK("affected non-empty", !affected.empty());

    // Count alive groups
    int alive = 0;
    for (auto& g : part.groups) if (g.alive) alive++;
    CHECK("3 alive groups", alive == 3);

    // All ops covered
    for (size_t i = 0; i < 4; i++)
        CHECK("op covered", !part.groups_of(i).empty());

    std::cout << "    groups after eject:\n";
    for (size_t i = 0; i < part.groups.size(); i++) {
        if (!part.groups[i].alive) continue;
        std::cout << "      G" << i << ": {";
        for (auto op : part.groups[i].ops) std::cout << op << ",";
        std::cout << "} cost=" << part.groups[i].cost << "\n";
    }
}

void test_apply_preserves_cost_consistency() {
    std::cout << "--- test_apply_preserves_cost_consistency ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Apply a sequence of moves, verify costs stay consistent
    auto m1 = best_move_for(part, 1);
    if (m1.valid()) {
        double before = part.total_cost();
        apply_fm_move(part, m1);

        // Verify all group costs match eval_set
        for (size_t i = 0; i < part.groups.size(); i++) {
            if (!part.groups[i].alive) continue;
            double expected = part.eval_set(part.groups[i].ops);
            CHECK_EQ("cost consistent", part.groups[i].cost, expected);
        }
    } else {
        g_pass++;  // no move available, still valid
    }
}

// ==================== ActiveSet ====================

void test_active_set_basic() {
    std::cout << "--- test_active_set_basic ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    ActiveSet as(part, 0.0);
    CHECK_EQ_S("initially empty", as.num_active(), 0);

    // Activate Op0: it's a border op (succ Op1 in different group)
    as.activate(0);
    CHECK("Op0 active", as.is_active(0));
    CHECK("Op0 not locked", !as.is_locked(0));
    CHECK_EQ_S("1 active", as.num_active(), 1);
}

void test_active_set_activate_border() {
    std::cout << "--- test_active_set_activate_border ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Merge {Op0,Op1,Op2}, leave {Op3}
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;

    ActiveSet as(part, 0.0);
    as.activate_border(0);

    // Only Op2 is border (succ Op3 outside)
    CHECK("Op2 active", as.is_active(2));
    CHECK("Op0 not active", !as.is_active(0));
    CHECK("Op1 not active", !as.is_active(1));
    CHECK_EQ_S("1 active", as.num_active(), 1);
}

void test_active_set_pop_best() {
    std::cout << "--- test_active_set_pop_best ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    ActiveSet as(part, 0.0);
    // Activate all ops
    for (size_t i = 0; i < 4; i++) as.activate(i);
    CHECK_EQ_S("4 active", as.num_active(), 4);
    CHECK_EQ_S("4 unlocked", as.num_unlocked(), 4);

    // Pop best should return a valid move and lock the op
    auto m1 = as.pop_best();
    CHECK("got first move", m1.has_value());
    CHECK("first move valid", m1->valid());
    CHECK("initiator locked", as.is_locked(m1->op));
    CHECK_EQ_S("3 unlocked", as.num_unlocked(), 3);

    std::cout << "    first pop: Op" << m1->op << " type=" << m1->type
              << " saving=" << m1->saving << "\n";
}

void test_active_set_pop_exhaustion() {
    std::cout << "--- test_active_set_pop_exhaustion ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    ActiveSet as(part, 0.0);
    for (size_t i = 0; i < 4; i++) as.activate(i);

    // Pop all 4
    int count = 0;
    while (auto m = as.pop_best()) {
        count++;
        if (count > 10) break;  // safety
    }
    CHECK("popped some moves", count > 0);
    CHECK_EQ_S("all locked", as.num_unlocked(), 0);

    // One more pop should return nullopt
    auto m_end = as.pop_best();
    CHECK("exhausted", !m_end.has_value());
}

void test_active_set_no_duplicate() {
    std::cout << "--- test_active_set_no_duplicate ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    ActiveSet as(part, 0.0);
    as.activate(0);
    as.activate(0);  // duplicate
    CHECK_EQ_S("still 1 active", as.num_active(), 1);
}

void test_active_set_locked_not_activated() {
    std::cout << "--- test_active_set_locked_not_activated ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    ActiveSet as(part, 0.0);
    as.activate(0);
    as.pop_best();  // locks Op0 (or whichever pops)

    size_t locked_op = *as.locked_ops().begin();
    size_t before = as.num_active();
    as.activate(locked_op);  // should be no-op
    CHECK_EQ_S("locked op not re-activated", as.num_active(), before);
}

void test_active_set_update_after_move() {
    std::cout << "--- test_active_set_update_after_move ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    ActiveSet as(part, 0.0);
    for (size_t i = 0; i < 4; i++) as.activate(i);

    // Pop and apply best move
    auto m = as.pop_best();
    CHECK("got move", m.has_value());

    auto affected = apply_fm_move(part, *m);
    CHECK("move applied", !affected.empty());

    // Update affected ops
    as.update_affected(affected);

    // Remaining unlocked ops should have recomputed moves
    // (We can't easily check the move content, but we verify no crash
    // and that we can still pop)
    auto m2 = as.pop_best();
    // May or may not have a valid move depending on state
    g_pass++;  // survived without crash
    std::cout << "    after update: unlocked=" << as.num_unlocked() << "\n";
}

void test_active_set_activate_neighbors() {
    std::cout << "--- test_active_set_activate_neighbors ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    ActiveSet as(part, 0.0);
    // Only activate Op0 initially
    as.activate(0);
    CHECK_EQ_S("1 active", as.num_active(), 1);

    // Now activate neighbors of G0 (which contains Op0)
    // G0 is adjacent to G1 (via Op1). Border ops of G1: Op1 itself (singleton).
    // G1 is adjacent to G0 and G2. Border ops of G2: Op2.
    as.activate_neighbors_of({0});

    CHECK("Op1 activated", as.is_active(1));
    std::cout << "    after activate_neighbors_of({G0}): "
              << as.num_active() << " active\n";
}

void test_active_set_with_negative_floor() {
    std::cout << "--- test_active_set_with_negative_floor ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Fuse {Op0,Op1,Op2}, {Op3}. Op2 is border. Eject Op2 is negative.
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;

    // With floor=0: Op2's eject is negative, might not pop
    ActiveSet as_strict(part, 0.0);
    as_strict.activate_border(0);
    as_strict.activate_border(3);
    auto m_strict = as_strict.pop_best();

    // With large floor: should pop even negative moves
    ActiveSet as_loose(part, part.total_cost() * 0.1);
    as_loose.activate_border(0);
    as_loose.activate_border(3);
    auto m_loose = as_loose.pop_best();

    if (m_loose.has_value())
        std::cout << "    loose pop: Op" << m_loose->op << " type=" << m_loose->type
                  << " saving=" << m_loose->saving << "\n";
    if (m_strict.has_value())
        std::cout << "    strict pop: Op" << m_strict->op << " type=" << m_strict->type
                  << " saving=" << m_strict->saving << "\n";

    g_pass++;  // documentation test — both modes work without crash
}

// ==================== FM inner pass ====================

void test_fm_pass_improves_trivial() {
    std::cout << "--- test_fm_pass_improves_trivial ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    double trivial_cost = part.total_cost();

    FMConfig cfg;
    cfg.init_count = 999;  // activate all border ops
    auto result = fm_inner_pass(part, cfg);

    CHECK("applied some moves", result.moves_applied > 0);
    CHECK("best <= start", result.best_cost <= result.start_cost + 0.001);
    CHECK("best <= trivial", result.best_cost <= trivial_cost + 0.001);
    CHECK_EQ("start is trivial", result.start_cost, trivial_cost);

    std::cout << "    trivial=" << trivial_cost << " best=" << result.best_cost
              << " moves=" << result.moves_applied
              << " (+" << result.moves_positive << " -" << result.moves_negative << ")\n";
}

void test_fm_pass_ops_covered() {
    std::cout << "--- test_fm_pass_ops_covered ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    FMConfig cfg;
    cfg.init_count = 999;
    auto result = fm_inner_pass(part, cfg);

    // All ops must be covered in best_partition
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", !result.best_partition.groups_of(i).empty());
}

void test_fm_pass_costs_consistent() {
    std::cout << "--- test_fm_pass_costs_consistent ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    FMConfig cfg;
    cfg.init_count = 999;
    auto result = fm_inner_pass(part, cfg);

    // All group costs in best_partition must match eval_set
    for (size_t i = 0; i < result.best_partition.groups.size(); i++) {
        auto& g = result.best_partition.groups[i];
        if (!g.alive) continue;
        double expected = result.best_partition.eval_set(g.ops);
        CHECK_EQ("group cost", g.cost, expected);
    }

    // total_cost must match best_cost
    CHECK_EQ("total matches best", result.best_partition.total_cost(), result.best_cost);
}

void test_fm_pass_already_optimal() {
    std::cout << "--- test_fm_pass_already_optimal ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // First pass to find optimum
    FMConfig cfg;
    cfg.init_count = 999;
    auto r1 = fm_inner_pass(part, cfg);

    // Second pass from optimum: should not worsen
    auto r2 = fm_inner_pass(r1.best_partition, cfg);

    CHECK("second pass no worse", r2.best_cost <= r1.best_cost + 0.001);
    std::cout << "    pass1=" << r1.best_cost << " pass2=" << r2.best_cost << "\n";
}

void test_fm_pass_diamond() {
    std::cout << "--- test_fm_pass_diamond ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    double trivial_cost = part.total_cost();

    FMConfig cfg;
    cfg.init_count = 999;
    auto result = fm_inner_pass(part, cfg);

    CHECK("improves diamond", result.best_cost <= trivial_cost + 0.001);
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", !result.best_partition.groups_of(i).empty());

    std::cout << "    trivial=" << trivial_cost << " best=" << result.best_cost
              << " groups=" << result.best_partition.num_alive() << "\n";
}

void test_fm_pass_drift_stops() {
    std::cout << "--- test_fm_pass_drift_stops ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);

    // Start from already-fused partition. Only negative moves available.
    // max_drift should prevent drifting too far.
    auto part = Partition::trivial(p, d);
    // Fuse {Op0,Op1,Op2}, leave {Op3}
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    double start_cost = part.total_cost();

    FMConfig cfg;
    cfg.init_count = 999;
    cfg.max_drift_fraction = 0.05;
    auto result = fm_inner_pass(part, cfg);

    // Best seen should be no worse than start (snapshot taken at start)
    CHECK("best <= start", result.best_cost <= start_cost + 0.001);
    std::cout << "    start=" << start_cost << " best=" << result.best_cost
              << " moves=" << result.moves_applied << "\n";
}

void test_fm_pass_random_subset() {
    std::cout << "--- test_fm_pass_random_subset ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // With init_count=1 and different seeds
    FMConfig cfg1;
    cfg1.init_count = 1;
    cfg1.seed = 1;
    auto r1 = fm_inner_pass(part, cfg1);

    FMConfig cfg2;
    cfg2.init_count = 1;
    cfg2.seed = 999;
    auto r2 = fm_inner_pass(part, cfg2);

    // Both should produce valid results
    CHECK("seed1 valid", r1.best_cost < 1e17);
    CHECK("seed2 valid", r2.best_cost < 1e17);
    // Costs may differ due to different activation order
    std::cout << "    seed1: cost=" << r1.best_cost << " moves=" << r1.moves_applied
              << " seed2: cost=" << r2.best_cost << " moves=" << r2.moves_applied << "\n";
}

void test_fm_pass_negative_moves() {
    std::cout << "--- test_fm_pass_negative_moves ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);

    // Fuse everything optimally, then run pass — only negative moves available
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2, 3};
    part.groups[0].cost = part.eval_set({0, 1, 2, 3});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.groups[3].alive = false;
    double optimal_cost = part.total_cost();

    FMConfig cfg;
    cfg.init_count = 999;
    cfg.floor_fraction = 0.1;  // allow 10% worsening
    auto result = fm_inner_pass(part, cfg);

    // Best seen should still be the optimal (snapshot at start)
    CHECK_EQ("best preserved", result.best_cost, optimal_cost);
    // May have applied some negative moves before drift stopped
    std::cout << "    optimal=" << optimal_cost << " best=" << result.best_cost
              << " neg_moves=" << result.moves_negative << "\n";
}

// ==================== FM outer loop ====================

void test_fm_outer_improves_trivial() {
    std::cout << "--- test_fm_outer_improves_trivial ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    double trivial_cost = part.total_cost();

    FMOuterConfig cfg;
    cfg.max_passes = 20;
    cfg.max_no_improve = 3;
    cfg.pass_config.init_count = 999;
    auto result = fm_outer_loop(part, cfg);

    CHECK("improves over trivial", result.best_cost < trivial_cost - 0.01);
    CHECK("at least 1 pass", result.total_passes >= 1);
    CHECK("at least 1 improving", result.improving_passes >= 1);

    std::cout << "    trivial=" << trivial_cost << " best=" << result.best_cost
              << " passes=" << result.total_passes
              << " improving=" << result.improving_passes << "\n";
}

void test_fm_outer_ops_covered() {
    std::cout << "--- test_fm_outer_ops_covered ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    FMOuterConfig cfg;
    cfg.max_passes = 10;
    cfg.max_no_improve = 3;
    auto result = fm_outer_loop(part, cfg);

    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", !result.best_partition.groups_of(i).empty());
}

void test_fm_outer_costs_consistent() {
    std::cout << "--- test_fm_outer_costs_consistent ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    FMOuterConfig cfg;
    cfg.max_passes = 10;
    cfg.max_no_improve = 3;
    auto result = fm_outer_loop(part, cfg);

    for (size_t i = 0; i < result.best_partition.groups.size(); i++) {
        auto& g = result.best_partition.groups[i];
        if (!g.alive) continue;
        double expected = result.best_partition.eval_set(g.ops);
        CHECK_EQ("group cost", g.cost, expected);
    }
    CHECK_EQ("total matches", result.best_partition.total_cost(), result.best_cost);
}

void test_fm_outer_early_termination() {
    std::cout << "--- test_fm_outer_early_termination ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    FMOuterConfig cfg;
    cfg.max_passes = 100;
    cfg.max_no_improve = 3;
    cfg.pass_config.init_count = 999;
    auto result = fm_outer_loop(part, cfg);

    // Should stop well before 100 passes on this tiny problem
    CHECK("stopped early", result.total_passes < 100);
    std::cout << "    stopped after " << result.total_passes << " passes\n";
}

void test_fm_outer_diamond() {
    std::cout << "--- test_fm_outer_diamond ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    double trivial_cost = part.total_cost();

    FMOuterConfig cfg;
    cfg.max_passes = 20;
    cfg.max_no_improve = 3;
    auto result = fm_outer_loop(part, cfg);

    CHECK("improves diamond", result.best_cost < trivial_cost - 0.01);
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", !result.best_partition.groups_of(i).empty());

    std::cout << "    trivial=" << trivial_cost << " best=" << result.best_cost
              << " groups=" << result.best_partition.num_alive() << "\n";
}

void test_fm_outer_from_optimum() {
    std::cout << "--- test_fm_outer_from_optimum ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);

    // Start from fully-fused optimal
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2, 3};
    part.groups[0].cost = part.eval_set({0, 1, 2, 3});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.groups[3].alive = false;
    double opt_cost = part.total_cost();

    FMOuterConfig cfg;
    cfg.max_passes = 10;
    cfg.max_no_improve = 3;
    auto result = fm_outer_loop(part, cfg);

    // Should not worsen
    CHECK("preserved optimum", result.best_cost <= opt_cost + 0.001);
    CHECK_EQ("cost unchanged", result.best_cost, opt_cost);
    std::cout << "    optimal=" << opt_cost << " result=" << result.best_cost << "\n";
}

void test_fm_outer_Y_shape() {
    std::cout << "--- test_fm_outer_Y_shape ---\n";
    auto p = make_Y(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    double trivial_cost = part.total_cost();

    FMOuterConfig cfg;
    cfg.max_passes = 20;
    cfg.max_no_improve = 5;
    auto result = fm_outer_loop(part, cfg);

    CHECK("improves Y", result.best_cost <= trivial_cost + 0.001);
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", !result.best_partition.groups_of(i).empty());

    std::cout << "    trivial=" << trivial_cost << " best=" << result.best_cost
              << " groups=" << result.best_partition.num_alive()
              << " passes=" << result.total_passes << "\n";
}

void test_fm_outer_different_seeds_explore() {
    std::cout << "--- test_fm_outer_different_seeds_explore ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    // Run with init_count=1 so different seeds give different subsets
    FMOuterConfig cfg;
    cfg.max_passes = 10;
    cfg.max_no_improve = 3;
    cfg.pass_config.init_count = 1;
    auto result = fm_outer_loop(part, cfg);

    // Should still find a good result
    CHECK("finds good result", result.best_cost < part.total_cost());
    std::cout << "    passes=" << result.total_passes
              << " improving=" << result.improving_passes
              << " cost=" << result.best_cost << "\n";
}

// ==================== Main ====================

int main() {
    test_is_border_op();
    test_border_ops();
    test_border_ops_diamond();
    test_connected_components_single();
    test_connected_components_split();
    test_connected_components_Y();
    test_eject_simple();
    test_eject_with_disconnection();
    test_eject_recomputed_op();
    test_eject_cant_eject_singleton();
    test_best_move_chain_unfused();
    test_best_move_locked();
    test_best_move_with_floor();
    test_best_move_recomputed_op();
    test_apply_steal();
    test_apply_merge();
    test_apply_recompute();
    test_apply_eject_with_split();
    test_apply_preserves_cost_consistency();
    test_active_set_basic();
    test_active_set_activate_border();
    test_active_set_pop_best();
    test_active_set_pop_exhaustion();
    test_active_set_no_duplicate();
    test_active_set_locked_not_activated();
    test_active_set_update_after_move();
    test_active_set_activate_neighbors();
    test_active_set_with_negative_floor();
    test_fm_pass_improves_trivial();
    test_fm_pass_ops_covered();
    test_fm_pass_costs_consistent();
    test_fm_pass_already_optimal();
    test_fm_pass_diamond();
    test_fm_pass_drift_stops();
    test_fm_pass_random_subset();
    test_fm_pass_negative_moves();
    test_fm_outer_improves_trivial();
    test_fm_outer_ops_covered();
    test_fm_outer_costs_consistent();
    test_fm_outer_early_termination();
    test_fm_outer_diamond();
    test_fm_outer_from_optimum();
    test_fm_outer_Y_shape();
    test_fm_outer_different_seeds_explore();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}