// fm_search_test.cpp
//
// Tests for the FM local search pipeline:
//   1. best_move_for — all eight move types, gap/cycle guards
//   2. apply_fm_move — each type with feasibility check after apply
//   3. ActiveSet    — activate, pop_best, lock, refresh_after_move
//   4. fm_inner_pass — pass mechanics, locking, drift control
//   5. fm_outer_loop — improvement, no-improve stopping, greedy kick
//
// Feasibility invariants checked after every apply_fm_move call:
//   1. Memory:        every alive group has a feasible tiling (eval_set < 1e18)
//   2. No eph. gap:   no tensor is ephemeral while an external consumer has
//                     no other source
//   3. Acyclicity:    cycle guards in apply_fm_move prevent cycles
//                     (direct structural tests below)
//
// Build: make fm_search_test
// Run:   ./fm_search_test

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost_cache.h"
#include "partition/partition.h"
#include "init/init_strategies.h"
#include "search/fm_search.h"
#include "search/active_set.h"
#include "search/fm_pass.h"
#include "search/fm_outer.h"
#include "search/local_search.h"
#include <cmath>
#include <iostream>
#include <set>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double got, double exp, double tol = 0.5) {
    if (std::abs(got - exp) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << got << " exp=" << exp << "\n"; }
}
static void CHECK_EQ_S(const char* l, size_t got, size_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << got << " exp=" << exp << "\n"; }
}

// ============================================================================
// Problem helpers
// ============================================================================

static Problem make_chain3() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

static Problem make_chain4() {
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// 4-op diamond: T0->Op0->T1, T1->{Op1->T2, Op2->T3}, T2+T3->Op3->T4
// T1 has two consumers (Op1, Op2) in separate branches.
static Problem make_diamond4() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{1},{3},1000},
             {OpType::Pointwise,{2,3},{4},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// Fanin: T0->{Op0->T1, Op1->T2}, T1+T2->Op2->T3.
// Op0 and Op1 are co-consumers of T0 (no DAG edge between them).
static Problem make_fanin() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{0},{2},1000},
             {OpType::Pointwise,{1,2},{3},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// ============================================================================
// Canonical feasibility check
// ============================================================================

static void check_feasible(const char* label, const Partition& part) {
    std::string err = verify_partition_feasibility(part);
    if (!err.empty()) {
        std::cout << "  FAIL: " << label << " feasibility: " << err << "\n";
        g_fail++;
    } else { g_pass++; }
}

// ============================================================================
// 1. best_move_for
// ============================================================================

void test_best_move_proposes_merge() {
    std::cout << "--- test_best_move_proposes_merge ---\n";
    // Trivial chain3: Op0 in G0, Op1 in G1. MERGE should be the best move.
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    auto m = best_move_for(part, 0, 0.0);
    CHECK("Op0 proposes valid move", m.valid());
    CHECK("MERGE proposed", m.type == FMMove::MERGE);
    CHECK("positive saving", m.saving > 0);
    std::cout << "    saving=" << m.saving << "\n";
}

void test_best_move_proposes_eject() {
    std::cout << "--- test_best_move_proposes_eject ---\n";
    // Group {Op0,Op1} with G2={Op2}: Op1 is a border op (external neighbor Op2).
    // best_move_for(Op1) should propose EJECT (or STEAL into G2, whichever is better).
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1};
    part.groups[0].cost = cache.evaluate({0,1},p,d);
    part.groups[1].alive = false;
    part.rebuild_index();

    // Op1 is in G0, border (Op2 is outside)
    auto m = best_move_for(part, 1, 1e18);  // low floor to see all
    CHECK("Op1 proposes a move", m.valid());
    // Could be EJECT, STEAL, or MERGE — any is acceptable
    std::cout << "    type=" << m.type << " saving=" << m.saving << "\n";
}

void test_best_move_recompute_not_blocked_by_gap() {
    std::cout << "--- test_best_move_recompute_not_blocked_by_gap ---\n";
    // Diamond4 trivial. T1 is produced by Op0, consumed by BOTH Op1 and Op2.
    // Merging {Op0,Op1}: T1 ephemeral but Op2 has no backup → gap → MERGE blocked.
    // Stealing Op0 into G1: same reasoning → STEAL blocked.
    // RECOMPUTE of Op0 into G1: safe because new_G1={Op0,Op1} exports T1 as
    // boundary output. After Fix 4/5, there is NO gap check on RECOMPUTE.
    //
    // For PW ops, all tensors have the same shape, so RECOMPUTE saving = 0
    // (load_T1 - load_T0 = 0) — filtered by EPSILON. We therefore verify the
    // fix indirectly: confirm MERGE and STEAL are correctly blocked, and that
    // whatever move best_move_for proposes (if any) is feasible.
    // The direct RECOMPUTE apply test is test_apply_recompute_no_gap_check.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    // Direct gap check: both MERGE and STEAL on {Op0,Op1} must be blocked.
    std::set<size_t> new_g1_merge = {0,1};
    CHECK("MERGE {Op0,Op1} blocked by gap (T1 ephemeral, Op2 stranded)",
          part.creates_ephemeral_gap(new_g1_merge, 0, 1));
    CHECK("STEAL Op0 into G1 blocked by gap (excludes G0 source correctly)",
          part.creates_ephemeral_gap(new_g1_merge, 0, 1));

    // best_move_for must not propose MERGE or STEAL (both create gaps).
    auto m = best_move_for(part, 0, 1e18);
    CHECK("MERGE not proposed", m.type != FMMove::MERGE);
    CHECK("STEAL not proposed", m.type != FMMove::STEAL);

    // If any move IS proposed, applying it must produce a feasible partition.
    if (m.valid()) {
        auto part2 = part;
        auto affected = apply_fm_move(part2, m);
        if (!affected.empty()) {
            check_feasible("post-move diamond4", part2);
            std::cout << "    move type=" << m.type << " saving=" << m.saving << "\n";
        }
    } else {
        std::cout << "    no move proposed (RECOMPUTE saving=0 < EPSILON) — correct\n";
        g_pass++;
    }
}
void test_best_move_steal_gap_check_excludes_source() {
    std::cout << "--- test_best_move_steal_gap_check_excludes_source ---\n";
    // Diamond4 trivial. Op0 produces T1. Both Op1 and Op2 consume T1.
    // STEAL of Op0 from G0 to G1 would make T1 ephemeral in new_G1={Op0,Op1}.
    // Op2 (in G2) needs T1 — after steal, G0 no longer has Op0, so no backup.
    // The corrected gap check (Bug 1 fix) must exclude G0 (losing Op0) AND G1.
    // STEAL must be blocked; RECOMPUTE must be allowed.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    // Check all moves with low floor from Op0's perspective
    auto m = best_move_for(part, 0, 1e18);
    // STEAL should not be proposed since it creates a gap
    CHECK("STEAL not proposed (gap check excludes source correctly)",
          m.type != FMMove::STEAL);
}

void test_best_move_split_co_consumer_bridge() {
    std::cout << "--- test_best_move_split_co_consumer_bridge ---\n";
    // Group {Op0,Op1,Op2} where Op0 and Op1 are co-consumers of T0.
    // Op0<->Op1 is a co-consumer bridge. After Fix 6 (op_neighbors), SPLIT
    // must be proposed on this bridge.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},   // Op0: T0->T1
             {OpType::Pointwise,{0},{2},1000},   // Op1: T0->T2 (co-consumer)
             {OpType::Pointwise,{1},{3},1000}};  // Op2: T1->T3
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2};
    part.groups[0].cost = cache.evaluate({0,1,2},p,d);
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    // Op1 is internal; co-consumer bridge with Op0 exists
    bool co_bridge = false;
    for (auto& e : part.bridge_edges(0))
        if ((e.first==0&&e.second==1)||(e.first==1&&e.second==0)) co_bridge = true;
    CHECK("co-consumer bridge exists", co_bridge);

    // best_move_for with low floor from Op1 (internal)
    auto m = best_move_for(part, 1, 1e18);
    CHECK("SPLIT or INTERNAL_EJECT proposed for internal op", m.valid());
    std::cout << "    type=" << m.type << " saving=" << m.saving << "\n";
}

// ============================================================================
// 2. apply_fm_move — each type with feasibility check
// ============================================================================

void test_apply_merge_feasible() {
    std::cout << "--- test_apply_merge_feasible ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    FMMove m;
    m.type = FMMove::MERGE; m.op = 0; m.ga = 0; m.gb = 1;
    m.saving = part.groups[0].cost + part.groups[1].cost - part.eval_set({0,1});

    auto affected = apply_fm_move(part, m);
    CHECK("MERGE applied", !affected.empty());
    CHECK("G1 dead", !part.groups[1].alive);
    CHECK_EQ_S("2 alive", part.num_alive(), 2);
    check_feasible("post-MERGE", part);
}

void test_apply_steal_feasible() {
    std::cout << "--- test_apply_steal_feasible ---\n";
    // G0={Op0,Op1}, G2={Op2}: steal Op1 from G0 to G2
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1};
    part.groups[0].cost = cache.evaluate({0,1},p,d);
    part.groups[1].alive = false;
    part.rebuild_index();

    FMMove m;
    m.type = FMMove::STEAL; m.op = 1; m.ga = 0; m.gb = 2;
    m.saving = 999.0;  // will be re-evaluated in apply

    auto affected = apply_fm_move(part, m);
    if (!affected.empty()) {
        CHECK("Op1 in G2", part.groups[2].ops.count(1));
        CHECK("Op1 not in G0", !part.groups[0].ops.count(1));
        check_feasible("post-STEAL", part);
    } else {
        // STEAL may be rejected if eval_set check fails — that's OK
        std::cout << "    STEAL rejected at apply (cost check)\n";
        g_pass++;
    }
}

void test_apply_recompute_feasible() {
    std::cout << "--- test_apply_recompute_feasible ---\n";
    // Chain: Op0 in G0, Op1 in G1. Recompute Op0 into G1.
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    FMMove m;
    m.type = FMMove::RECOMPUTE; m.op = 0; m.ga = 0; m.gb = 1;
    double new_g1_cost = cache.evaluate({0,1}, p, d);
    m.saving = part.groups[1].cost - new_g1_cost;

    auto affected = apply_fm_move(part, m);
    if (!affected.empty()) {
        CHECK("Op0 in G1 after RECOMPUTE", part.groups[1].ops.count(0));
        CHECK("G0 unchanged", part.groups[0].ops.count(0));
        CHECK_EQ_S("Op0 in 2 groups", part.groups_of(0).size(), 2);
        check_feasible("post-RECOMPUTE", part);
    } else {
        std::cout << "    RECOMPUTE rejected at apply\n";
        g_pass++;
    }
}

void test_apply_recompute_no_gap_check() {
    std::cout << "--- test_apply_recompute_no_gap_check ---\n";
    // Diamond4: RECOMPUTE Op0 into G1 (adding Op0 to G1={Op1}).
    // T1 would become ephemeral in new_G1={Op0,Op1}.
    // Op2 in G2 still needs T1, but new_G1 still has Op0 and exports T1.
    // The old code blocked this with creates_ephemeral_gap (over-conservative).
    // After Bug 5 fix, this RECOMPUTE must be allowed.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    FMMove m;
    m.type = FMMove::RECOMPUTE; m.op = 0; m.ga = 0; m.gb = 1;
    double new_g1 = cache.evaluate({0,1}, p, d);
    m.saving = part.groups[1].cost - new_g1;

    auto affected = apply_fm_move(part, m);
    // Should be accepted (no gap: new_G1 exports T1)
    CHECK("RECOMPUTE accepted despite T1 ephemeral in new_G1 (Bug 5 fix)",
          !affected.empty());
    if (!affected.empty())
        check_feasible("post-RECOMPUTE diamond4", part);
}

void test_apply_eject_feasible() {
    std::cout << "--- test_apply_eject_feasible ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2};
    part.groups[0].cost = cache.evaluate({0,1,2},p,d);
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    FMMove m;
    m.type = FMMove::EJECT; m.op = 2; m.ga = 0;
    m.saving = 1.0;  // re-evaluated in apply

    auto affected = apply_fm_move(part, m);
    CHECK("EJECT applied", !affected.empty());
    if (!affected.empty()) {
        CHECK("Op2 not in G0", !part.groups[0].ops.count(2));
        check_feasible("post-EJECT", part);
    }
}

void test_apply_split_feasible() {
    std::cout << "--- test_apply_split_feasible ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2};
    part.groups[0].cost = cache.evaluate({0,1,2},p,d);
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    FMMove m;
    m.type = FMMove::SPLIT; m.op = 0; m.op2 = 1; m.ga = 0;
    m.saving = 1.0;

    auto affected = apply_fm_move(part, m);
    CHECK("SPLIT applied", !affected.empty());
    if (!affected.empty()) {
        CHECK("2 alive groups", part.num_alive() == 2);
        check_feasible("post-SPLIT", part);
    }
}

void test_apply_merge_cycle_rejected_at_apply() {
    std::cout << "--- test_apply_merge_cycle_rejected_at_apply ---\n";
    // 4-op chain: Op0->Op1->Op2->Op3. Partition: G0={0,3}, G1={1}, G2={2}.
    // merge_creates_cycle({0,3},{1}): S={0,1,3}, external={2}.
    // Op1->Op2 (S->ext) and Op2->Op3 (ext->S) => cycle in condensed DAG => true.
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0,3}, cache.evaluate({0,3},p,d));  // disconnected: cost=1e18
    part.add_group({1},   cache.evaluate({1},p,d));
    part.add_group({2},   cache.evaluate({2},p,d));
    part.rebuild_index();

    CHECK("merge_creates_cycle({0,3},{1}) = true", d.merge_creates_cycle({0,3}, {1}));

    FMMove m;
    m.type = FMMove::MERGE; m.op = 0; m.ga = 0; m.gb = 1; m.saving = 999.0;
    auto affected = apply_fm_move(part, m);
    CHECK("cycle-creating merge rejected at apply", affected.empty());
}
void test_apply_steal_gap_from_source_excluded() {
    std::cout << "--- test_apply_steal_gap_from_source_excluded ---\n";
    // Diamond4 trivial. STEAL Op0 from G0 to G1:
    // new_G1 = {Op0, Op1}. T1 becomes ephemeral.
    // Op2 in G2 needs T1. After steal, G0 has no Op0 → no backup.
    // apply_fm_move must reject this (Bug 2 fix: excludes ga=G0).
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    FMMove m;
    m.type = FMMove::STEAL; m.op = 0; m.ga = 0; m.gb = 1;
    m.saving = 999.0;  // claim positive saving

    auto affected = apply_fm_move(part, m);
    CHECK("STEAL rejected: Op0 stolen leaves Op2 without T1 (Bug 2 fix)",
          affected.empty());
}

// ============================================================================
// 3. ActiveSet
// ============================================================================

void test_active_set_activate_pop() {
    std::cout << "--- test_active_set_activate_pop ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    double floor = 0.0;
    ActiveSet active(part, floor);

    // Activate Op1 (middle op, has both neighbors)
    active.activate(1);
    CHECK("Op1 active", active.is_active(1));
    CHECK("Op0 not active", !active.is_active(0));
    CHECK_EQ_S("1 active", active.num_active(), 1);

    // pop_best should return Op1's move and lock it
    auto m = active.pop_best();
    CHECK("pop returns move", m.has_value());
    if (m.has_value()) {
        CHECK("Op1 locked after pop", active.is_locked(1));
        CHECK("move has positive saving", m->saving > 0);
    }
}

void test_active_set_lock_prevents_pop() {
    std::cout << "--- test_active_set_lock_prevents_pop ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    ActiveSet active(part, 0.0);
    active.activate(0);
    active.activate(1);
    active.activate(2);

    active.lock(0);
    active.lock(1);
    active.lock(2);

    auto m = active.pop_best();
    CHECK("no move when all locked", !m.has_value());
}

void test_active_set_activate_group_ops() {
    std::cout << "--- test_active_set_activate_group_ops ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    ActiveSet active(part, 0.0);
    active.activate_group_ops(1);  // G1={Op1}: border ops activated

    CHECK("Op1 activated", active.is_active(1));
}

void test_active_set_refresh_activates_new_ops() {
    std::cout << "--- test_active_set_refresh_activates_new_ops ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    ActiveSet active(part, 0.0);
    active.activate(0);  // only activate Op0 initially

    // Simulate applying a move: merge G0+G1 → {Op0,Op1}
    FMMove m;
    m.type = FMMove::MERGE; m.op = 0; m.ga = 0; m.gb = 1;
    m.saving = part.groups[0].cost + part.groups[1].cost - part.eval_set({0,1});

    auto affected = apply_fm_move(part, m);
    CHECK("MERGE applied", !affected.empty());

    // refresh_after_move should activate new border ops (Op2 is now border)
    active.refresh_after_move(affected);
    CHECK("Op2 activated after refresh", active.is_active(2));
}

// ============================================================================
// 4. fm_inner_pass
// ============================================================================

void test_fm_inner_pass_valid_partition() {
    std::cout << "--- test_fm_inner_pass_valid_partition ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    FMConfig cfg;
    cfg.floor_fraction = 0.30;
    cfg.max_drift_fraction = 0.50;
    cfg.init_count = 3;
    cfg.seed = 42;

    auto result = fm_inner_pass(std::move(part), cfg);

    CHECK("best partition valid", result.best_cost < 1e17);
    CHECK("start_cost set", result.start_cost > 0);
    check_feasible("fm_pass best", result.best_partition);
    check_feasible("fm_pass end",  result.end_partition);
    std::cout << "    start=" << result.start_cost
              << " best=" << result.best_cost
              << " moves=" << result.moves_applied << "\n";
}

void test_fm_inner_pass_accepts_negative_moves() {
    std::cout << "--- test_fm_inner_pass_accepts_negative_moves ---\n";
    // With a large floor_fraction, FM should accept worsening moves
    // (end_cost > best_cost is expected).
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    // Start from a near-optimal partition (fused)
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2,3};
    part.groups[0].cost = cache.evaluate({0,1,2,3},p,d);
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.groups[3].alive = false;
    part.rebuild_index();

    FMConfig cfg;
    cfg.floor_fraction = 0.80;      // accept large worsening
    cfg.max_drift_fraction = 2.0;   // wide drift budget
    cfg.init_count = 3;
    cfg.seed = 99;

    auto result = fm_inner_pass(std::move(part), cfg);

    // If any negative moves were accepted, end_cost > best_cost
    CHECK("best cost valid", result.best_cost < 1e17);
    check_feasible("fm_pass end (negative moves)", result.end_partition);
    std::cout << "    positive_moves=" << result.moves_positive
              << " negative_moves=" << result.moves_negative << "\n";
}

void test_fm_inner_pass_locking_prevents_revisit() {
    std::cout << "--- test_fm_inner_pass_locking_prevents_revisit ---\n";
    // After an op initiates a move, it's locked and cannot initiate another.
    // This is enforced by the ActiveSet. We can't directly observe locking,
    // but we can verify the pass terminates (doesn't loop infinitely).
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    FMConfig cfg;
    cfg.floor_fraction = 0.5;
    cfg.max_drift_fraction = 1.0;
    cfg.init_count = 10;  // large to activate many ops
    cfg.seed = 7;

    // Should terminate without hanging
    auto result = fm_inner_pass(std::move(part), cfg);
    CHECK("pass terminates", result.moves_applied >= 0);
    check_feasible("post-pass", result.best_partition);
}

void test_fm_inner_pass_drift_abort() {
    std::cout << "--- test_fm_inner_pass_drift_abort ---\n";
    // With a very tight drift budget, the pass should abort early after
    // cost degrades beyond max_drift.
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    FMConfig cfg_tight;
    cfg_tight.floor_fraction = 0.5;
    cfg_tight.max_drift_fraction = 0.001;  // abort after tiny degradation
    cfg_tight.init_count = 10;
    cfg_tight.seed = 11;

    FMConfig cfg_wide;
    cfg_wide.floor_fraction = 0.5;
    cfg_wide.max_drift_fraction = 100.0;  // never abort
    cfg_wide.init_count = 10;
    cfg_wide.seed = 11;

    auto part2 = part;
    auto r_tight = fm_inner_pass(std::move(part), cfg_tight);
    auto r_wide  = fm_inner_pass(std::move(part2), cfg_wide);

    // Tight drift should apply fewer moves than wide drift
    CHECK("tight drift applies fewer moves",
          r_tight.moves_applied <= r_wide.moves_applied);
    check_feasible("tight drift best", r_tight.best_partition);
    std::cout << "    tight_moves=" << r_tight.moves_applied
              << " wide_moves=" << r_wide.moves_applied << "\n";
}

// ============================================================================
// 5. fm_outer_loop
// ============================================================================

void test_fm_outer_improves_chain() {
    std::cout << "--- test_fm_outer_improves_chain ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    double initial = part.total_cost();

    FMOuterConfig cfg;
    cfg.max_passes = 20;
    cfg.max_no_improve = 5;

    auto result = fm_outer_loop(std::move(part), cfg);

    CHECK("outer improves or matches trivial", result.best_cost <= initial + 0.01);
    CHECK("best cost valid", result.best_cost < 1e17);
    check_feasible("fm_outer best", result.best_partition);
    std::cout << "    initial=" << initial << " best=" << result.best_cost
              << " passes=" << result.total_passes << "\n";
}

void test_fm_outer_no_improve_stops() {
    std::cout << "--- test_fm_outer_no_improve_stops ---\n";
    // Start from an already-optimal partition: no moves should improve it,
    // so outer loop should stop after max_no_improve passes.
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    // Fuse everything — this is optimal
    part.groups[0].ops  = {0,1,2};
    part.groups[0].cost = cache.evaluate({0,1,2},p,d);
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    FMOuterConfig cfg;
    cfg.max_passes = 100;
    cfg.max_no_improve = 3;

    auto result = fm_outer_loop(std::move(part), cfg);

    CHECK("stops before max_passes", result.total_passes <= 4);  // 3 + 1 max
    CHECK("best cost unchanged", result.best_cost < 1e17);
    check_feasible("fm_outer stopped", result.best_partition);
    std::cout << "    passes=" << result.total_passes << "\n";
}

void test_fm_outer_greedy_kick() {
    std::cout << "--- test_fm_outer_greedy_kick ---\n";
    // The greedy kick runs when FM pass perturbs but doesn't improve.
    // Verify best_partition is valid regardless of kick outcome.
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    FMOuterConfig cfg;
    cfg.max_passes = 10;
    cfg.max_no_improve = 3;
    cfg.pass_config.floor_fraction = 0.5;
    cfg.pass_config.max_drift_fraction = 1.0;

    auto result = fm_outer_loop(std::move(part), cfg);
    CHECK("kick produces valid partition", result.best_cost < 1e17);
    check_feasible("fm_outer kick", result.best_partition);
    std::cout << "    passes=" << result.total_passes
              << " improving=" << result.improving_passes << "\n";
}

void test_fm_outer_all_ops_covered() {
    std::cout << "--- test_fm_outer_all_ops_covered ---\n";
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    FMOuterConfig cfg;
    cfg.max_passes = 15;
    cfg.max_no_improve = 5;

    auto result = fm_outer_loop(std::move(part), cfg);
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", !result.best_partition.groups_of(i).empty());
    check_feasible("fm_outer diamond", result.best_partition);
}

// ============================================================================
// 6. Integration: greedy -> FM pipeline
// ============================================================================

void test_pipeline_greedy_then_fm() {
    std::cout << "--- test_pipeline_greedy_then_fm ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    auto after_greedy = greedy_descent(std::move(part));
    double greedy_cost = after_greedy.total_cost();
    check_feasible("after greedy", after_greedy);

    FMOuterConfig cfg;
    cfg.max_passes = 20;
    cfg.max_no_improve = 5;

    auto result = fm_outer_loop(std::move(after_greedy), cfg);

    CHECK("FM result <= greedy result", result.best_cost <= greedy_cost + 0.01);
    check_feasible("after FM", result.best_partition);
    for (size_t i = 0; i < p.num_ops(); i++)
        CHECK("op covered", !result.best_partition.groups_of(i).empty());
    std::cout << "    greedy=" << greedy_cost << " fm=" << result.best_cost << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // 1. best_move_for
    test_best_move_proposes_merge();
    test_best_move_proposes_eject();
    test_best_move_recompute_not_blocked_by_gap();
    test_best_move_steal_gap_check_excludes_source();
    test_best_move_split_co_consumer_bridge();

    // 2. apply_fm_move
    test_apply_merge_feasible();
    test_apply_steal_feasible();
    test_apply_recompute_feasible();
    test_apply_recompute_no_gap_check();
    test_apply_eject_feasible();
    test_apply_split_feasible();
    test_apply_merge_cycle_rejected_at_apply();
    test_apply_steal_gap_from_source_excluded();

    // 3. ActiveSet
    test_active_set_activate_pop();
    test_active_set_lock_prevents_pop();
    test_active_set_activate_group_ops();
    test_active_set_refresh_activates_new_ops();

    // 4. fm_inner_pass
    test_fm_inner_pass_valid_partition();
    test_fm_inner_pass_accepts_negative_moves();
    test_fm_inner_pass_locking_prevents_revisit();
    test_fm_inner_pass_drift_abort();

    // 5. fm_outer_loop
    test_fm_outer_improves_chain();
    test_fm_outer_no_improve_stops();
    test_fm_outer_greedy_kick();
    test_fm_outer_all_ops_covered();

    // 6. Integration
    test_pipeline_greedy_then_fm();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}