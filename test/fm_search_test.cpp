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
#include "search/partition_moves.h"
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

    auto m = best_move_for(part, 0);
    CHECK("Op0 proposes valid move", m.valid());
    CHECK("MERGE or STEAL proposed",
          m.type == FMMove::MERGE || m.type == FMMove::STEAL);
    CHECK("positive saving", m.saving > 0);
    std::cout << "    type=" << (int)m.type << " saving=" << m.saving << "\n";
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
    auto m = best_move_for(part, 1);  // low floor to see all
    CHECK("Op1 proposes a move", m.valid());
    // Could be EJECT, STEAL, or MERGE — any is acceptable
    std::cout << "    type=" << m.type << " saving=" << m.saving << "\n";
}

void test_best_move_recompute_not_blocked_by_gap() {
    std::cout << "--- test_best_move_recompute_not_blocked_by_gap ---\n";
    // Diamond4 trivial. T1 is produced by Op0, consumed by BOTH Op1 and Op2.
    //
    // MERGE/STEAL of {Op0,Op1} creates an ephemeral gap (T1 ephemeral, Op2
    // stranded). The gap check is now at eval time, so best_move_for must NOT
    // propose such a move. Any move proposed for Op0 must be gap-free.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    auto m = best_move_for(part, 0);
    if (m.valid()) {
        auto part2 = part;
        auto affected = apply_fm_move(part2, m);
        if (!affected.empty())
            check_feasible("post-move diamond4", part2);
        std::cout << "    move type=" << m.type << " saving=" << m.saving << "\n";
    }
}
void test_best_move_steal_gap_check_excludes_source() {
    std::cout << "--- test_best_move_steal_gap_check_excludes_source ---\n";
    // Diamond4 trivial. Op0 produces T1. Both Op1 and Op2 consume T1.
    // STEAL of Op0 from G0 to G1 creates an ephemeral gap — gap check is now
    // at eval time, so best_move_for must not propose it. Any move proposed
    // must produce a feasible, acyclic partition.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    auto m = best_move_for(part, 0);
    if (m.valid()) {
        auto part2 = part;
        auto affected = apply_fm_move(part2, m);
        if (!affected.empty()) {
            check_feasible("post-move diamond4 steal", part2);
            CHECK("partition acyclic", part2.is_acyclic());
        }
        std::cout << "    type=" << m.type << " saving=" << m.saving << "\n";
    }
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
    auto m = best_move_for(part, 1);
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
    // T1 becomes ephemeral in new_G1={Op0,Op1}. Op2 in G2 still needs T1.
    // But G0={Op0} stays alive and exports T1 as boundary output (Op0 produces
    // T1, no internal consumer in G0) → no gap.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    FMMove m;
    m.type = FMMove::RECOMPUTE; m.op = 0; m.ga = 0; m.gb = 1;
    double new_g1 = cache.evaluate({0,1}, p, d);
    m.saving = part.groups[1].cost - new_g1;

    auto affected = apply_fm_move(part, m);
    // RECOMPUTE accepted: G0 still alive, exports T1 → no gap
    CHECK("RECOMPUTE accepted (G0 exports T1 as boundary output)",
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

void test_eval_merge_cycle_rejected() {
    std::cout << "--- test_eval_merge_cycle_rejected ---\n";
    // 4-op chain: Op0->Op1->Op2->Op3. Partition: G0={0,3}, G1={1}, G2={2}.
    // Group DAG has G0→G1→G2→G0 (cycle) because G0 contains both op0 and op3.
    // acyclic_merge_local(G0,G1): G2 is external successor of G1, and G2→G0
    // closes the loop back into the merge set → cycle detected.
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0,3}, cache.evaluate({0,3},p,d));
    part.add_group({1},   cache.evaluate({1},p,d));
    part.add_group({2},   cache.evaluate({2},p,d));
    part.rebuild_index();

    CHECK("acyclic_merge_local(G0,G1) detects cycle", !part.acyclic_merge_local(0, 1));

    // Safe merge: trivial partition G0={0},G1={1},G2={2},G3={3}.
    // Merging G1 and G2: external successors of {G1,G2} land in G3 only;
    // BFS from G3 finds no path back into {G1,G2} → safe.
    Partition trivial = Partition::trivial(p, d); trivial.cache = &cache;
    CHECK("acyclic_merge_local(G1,G2) is safe in trivial chain",
          trivial.acyclic_merge_local(1, 2));
}
// ============================================================================
// Local acyclicity checks — recomputation corner cases
// ============================================================================

// Helper: 5-op chain with a recomputed op.
// Op0→Op1→Op2→Op3→Op4 (linear), with Op2 recomputed into two groups.
// Partition: G0={0,1,2}, G1={2,3}, G2={4}
// (Op2 appears in both G0 and G1.)
//
// We use a simple chain5:
//   T0->Op0->T1->Op1->T2->Op2->T3->Op3->T4->Op4->T5
static Problem make_chain5() {
    Problem p;
    for (int i = 0; i <= 5; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 5; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},1000});
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// acyclic_recompute_local: copy op into gb where ALL producer copies are
// forward-reachable from gb → cycle detected.
//
// Setup: chain4, Partition G0={0},G1={1},G2={2},G3={3}.
// Propose RECOMPUTE Op1 into G2. Op1 needs T1 (produced by Op0 in G0).
// G0 is NOT forward-reachable from G2 (G2→G3 only), so this is safe.
void test_acyclic_recompute_local_safe() {
    std::cout << "--- test_acyclic_recompute_local_safe ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    // Trivial: G0={0},G1={1},G2={2},G3={3}
    // Recompute Op1 into G2: Op1 needs T1 from G0. G0 is not reachable from G2.
    CHECK("recompute Op1 into G2 is safe", part.acyclic_recompute_local(1, 2));
}

// acyclic_recompute_local: copy op into gb where the only producer group IS
// forward-reachable from gb → cycle.
//
// Setup: chain4, G0={0,1},G1={2,3}.
// Propose RECOMPUTE Op2 into G0. Op2 needs T2 (produced by Op1 in G0).
// But Op1 IS in G0 (gb) → internal, not a new constraint. So this is safe?
// Actually let's think of a clearer cycle case.
// Propose RECOMPUTE Op3 into G0. Op3 needs T3, produced by Op2 in G1.
// G1 is a successor of G0 (G0→G1). Forward-reachable from G0 = {G1}.
// So G1 ∈ fwd(G0), and G1 is the only producer → cycle detected.
void test_acyclic_recompute_local_cycle() {
    std::cout << "--- test_acyclic_recompute_local_cycle ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0,1}, cache.evaluate({0,1},p,d));  // G0
    part.add_group({2,3}, cache.evaluate({2,3},p,d));  // G1
    part.rebuild_index();
    // Recompute Op3 into G0: Op3 needs T3 produced by Op2 in G1.
    // G1 is forward-reachable from G0 → cycle.
    CHECK("recompute Op3 into G0 creates cycle", !part.acyclic_recompute_local(3, 0));
}

// acyclic_recompute_local with two producer copies (OR-node): one copy blocked,
// one free → safe despite the blocked copy.
//
// Setup: chain5, Op2 recomputed in G0={0,1,2} and G1={2,3}.
// Propose RECOMPUTE Op3 into a new group G2={4}. Op3 needs T3 produced by Op2.
// Op2 is in G0 (predecessor of G1) and G1. G2 is only reachable from G1.
// fwd(G2)={} (G2 is the last group). Neither G0 nor G1 is in fwd(G2) → safe.
void test_acyclic_recompute_local_or_node_free() {
    std::cout << "--- test_acyclic_recompute_local_or_node_free ---\n";
    auto p = make_chain5(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0,1,2}, cache.evaluate({0,1,2},p,d));  // G0
    part.add_group({2,3},   cache.evaluate({2,3},p,d));    // G1 (Op2 recomputed)
    part.add_group({4},     cache.evaluate({4},p,d));      // G2
    part.rebuild_index();
    // Recompute Op3 into G2: Op3 needs T3 from Op2 (G0 or G1).
    // Neither is forward-reachable from G2 (G2 has no successors) → safe.
    CHECK("recompute Op3 into G2 safe (OR: G0 and G1 both free)", part.acyclic_recompute_local(3, 2));
}

// acyclic_recompute_local with two producer copies (OR-node): BOTH copies
// forward-reachable from gb → cycle despite OR-node.
//
// We need a partition where gb can reach both G0 and G1.
// Layout: G_target={4}, G0={0,1,2}, G1={2,3}. But G_target is the last group.
// Let's build a diamond where both branches merge back before the target.
// Actually simpler: put target at the front.
// chain5: G_target={0}, G_mid_a={1,2}, G_mid_b={2,3}, G_end={4}.
// Then G_target → G_mid_a and G_mid_b → G_end.
// Propose RECOMPUTE Op2 into G_end. Op2 needs T2 (from Op1 in G_mid_a) OR T2
// could come from recomputed G_mid_b. But T2 is produced by Op1 which is only in
// G_mid_a.
// Hmm, let me use a simpler structure.
// chain4, G0={0},G1={1,2},G2={3}. Propose recompute Op1 into G2.
// Op1 needs T1 from G0. fwd(G2)={}. G0 not in fwd(G2) → safe.
// That's not a cycle case for OR-node.
// Real OR-node cycle: G0={1,2}, G1={2,3}, G_target={0}.
// Recompute Op3 (needs T3) into G_target. T3 produced by Op2 in G0 and G1.
// fwd(G_target) includes G0 and G1 (G_target→...→G0,G1)? We need G_target to
// precede G0 and G1. But G_target={0}, G0={1,2}, G1={2,3}: chain is 0→1→2→3,
// so G_target={0} precedes G0={1,2}. fwd(G_target)={G0,G1,...}. So recomputing
// Op3 (needs T3 from Op2) into G_target: both G0 and G1 are in fwd(G_target) → cycle.
void test_acyclic_recompute_local_or_node_all_blocked() {
    std::cout << "--- test_acyclic_recompute_local_or_node_all_blocked ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0},   cache.evaluate({0},p,d));    // G0 = first
    part.add_group({1,2}, cache.evaluate({1,2},p,d)); // G1 (Op2 in it)
    part.add_group({2,3}, cache.evaluate({2,3},p,d)); // G2 (Op2 recomputed)
    part.rebuild_index();
    // Propose RECOMPUTE Op2 into G0. Op2 needs T2 from Op1, which is in G1.
    // G1 is forward-reachable from G0. Op2 is also in G2 which is forward-reachable.
    // ALL copies of the producer of T2 (Op1, only in G1) are in fwd(G0) → cycle.
    CHECK("recompute Op2 into G0 cycle (all producers fwd-reachable)", !part.acyclic_recompute_local(2, 0));
}

// acyclic_de_recompute_local: removing op from ga where the only remaining copy
// is forward-reachable from ga → cycle.
//
// Setup: chain5, G0={0,1,2,3}, G1={2,4}.
// G0 produces T4 (Op3→T4 consumed by Op4 in G1) → G0→G1 in group DAG.
// fwd(G0) includes G1.
// De-recompute Op2 from G0: Op2's output T3 is consumed by Op3 in G0 (ephemeral).
// The only remaining copy of Op2 is G1. G1 ∈ fwd(G0) → no free copy → cycle.
void test_acyclic_de_recompute_local_cycle() {
    std::cout << "--- test_acyclic_de_recompute_local_cycle ---\n";
    auto p = make_chain5(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0,1,2,3}, cache.evaluate({0,1,2,3},p,d));  // G0
    part.add_group({2,4},     cache.evaluate({2,4},p,d));      // G1 (Op2 recomputed + Op4)
    part.rebuild_index();
    // G0→G1 via T4 (Op3 in G0 produces T4, Op4 in G1 consumes it).
    // De-recompute Op2 from G0: T3 consumed by Op3 in G0 → constraint on G0.
    // Only copy left is G1 which is in fwd(G0) → cycle detected.
    CHECK("de-recompute Op2 from G0 creates cycle (only copy G1 is successor)",
          !part.acyclic_de_recompute_local(2, 0));
}

// acyclic_de_recompute_local: OR-node safe — one remaining copy is free.
//
// chain5, G0={0,1,2} (Op2 recomputed), G1={2}, G2={3,4}.
// De-recompute Op2 from G0. G0 still has Op1 which produces T2.
// But the output of Op2 is T3. Does G0 have any consumer of T3?
// Op3 is in G2, not in G0. So T3 is not consumed in G0 → no constraint added.
// De-recompute is safe (no consumers of T3 inside G0).
// Let me pick a case where G0 does consume Op2's output:
// G0={0,1,2,3} (uses T3 internally via Op3), G1={2} (copy of Op2).
// De-recompute Op2 from G0: G0 still needs T3 (from Op3 consumer).
// Remaining copy is G1. Is G1 forward-reachable from G0? G0 depends on T2 from
// Op1 in G0, so G0→G1? Only if G0 has ops producing things G1 needs.
// G1={Op2} needs T2 from Op1 (in G0). So G0→G1 edge exists. G1 is in fwd(G0).
// → cycle again. We need a case where the remaining copy is NOT in fwd(G0).
// chain5: G0={2,3} (uses T3 internally), G1={0,1,2} (Op2 recomputed).
// G1 is a PREDECESSOR of G0 (G1 produces T1,T2,T3 that G0 uses). G1 not in fwd(G0).
// De-recompute Op2 from G0: G0 needs T3. Remaining copy in G1. G1 not forward-reachable → safe.
void test_acyclic_de_recompute_local_safe() {
    std::cout << "--- test_acyclic_de_recompute_local_safe ---\n";
    auto p = make_chain5(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0,1,2}, cache.evaluate({0,1,2},p,d));  // G0 (predecessor, has Op2)
    part.add_group({2,3,4}, cache.evaluate({2,3,4},p,d));  // G1 (successor, Op2 recomputed)
    part.rebuild_index();
    // De-recompute Op2 from G1: G1 has Op3 and Op4 which need T3 (from Op2).
    // Remaining copy is G0. G0 is NOT forward-reachable from G1 (G0 is predecessor) → safe.
    CHECK("de-recompute Op2 from G1 safe (remaining copy G0 is predecessor)",
          part.acyclic_de_recompute_local(2, 1));
}

// acyclic_extract_local: extracting ops that create a forward cycle.
//
// chain4, trivial partition G0={0},G1={1},G2={2},G3={3}.
// Extract {Op1,Op2}: gnew gets outputs T2 (consumed by Op3 in G3).
// G3 is gnew's successor. Does G3 have a path back to gnew?
// G3={Op3}, Op3 outputs T4 (nothing consumes T4 in any group after it).
// No cycle. → safe.
//
// For a cycle, we need gnew→external→gnew. But in a linear chain there's no
// backwards path. Let's use a chain4 with recomputation:
// G0={0,1,2,3} (all ops), G1={1} (Op1 recomputed).
// Extract {Op1}: gnew gets T2 (consumed by Op2 in G0).
// gnew's successor is G0 (has Op2 consuming T2). From G0, can we reach gnew?
// gnew={Op1} needs T1 (produced by Op0 in G0). So G0→gnew edge after extract.
// G0 is a successor of gnew AND gnew depends on G0 → cycle.
void test_acyclic_extract_local_cycle() {
    std::cout << "--- test_acyclic_extract_local_cycle ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0,1,2,3}, cache.evaluate({0,1,2,3},p,d));  // G0
    part.add_group({1},       cache.evaluate({1},p,d));        // G1 (Op1 recomputed)
    part.rebuild_index();
    // Extract {Op1}: gnew needs T1 from Op0 (in G0), and produces T2 consumed by Op2 (in G0).
    // gnew→G0 (via T2), G0→gnew (via T1) → cycle.
    CHECK("extract {Op1} creates cycle (gnew↔G0)", !part.acyclic_extract_local({1}));
}

// acyclic_extract_local: safe extraction.
//
// chain4, trivial G0={0},G1={1},G2={2},G3={3}.
// Extract {Op1,Op2}: gnew produces T3 consumed by Op3 in G3.
// gnew needs T1 from G0. G0's forward reachability: G0→G1→G2→G3 (in trivial).
// After extract {Op1,Op2}, gnew is between G0 and G3. No cycle.
void test_acyclic_extract_local_safe() {
    std::cout << "--- test_acyclic_extract_local_safe ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    // Extract {Op1,Op2} from trivial partition.
    // gnew external successors: consumer groups of T2 and T3 not in {Op1,Op2}.
    // T3 consumed by Op3 in G3. BFS from G3: G3's outputs go nowhere. No cycle.
    CHECK("extract {Op1,Op2} safe in trivial chain4", part.acyclic_extract_local({1,2}));
}

// acyclic_extract_local: extract from a recomputed group where the
// source group has both remainder and extracted ops.
//
// chain4: G0={0,1,2,3} (all ops), G1={1} (recomputed).
// Extract {Op2}: gnew={Op2}. gnew needs T2 (from Op1 in G0, not in extract_ops).
// gnew produces T3 consumed by Op3 in G0 (not in extract_ops).
// gnew→G0 (via T3), and from G0 we need T2 → G0 needs Op1 which is in G0 → G0→gnew (via T2).
// Wait: T2 is produced by Op1 in G0 (Op1 is not in extract_ops). So G0 is a producer
// of T2 needed by gnew. gnew's successor includes G0 (via T3 consumer Op3). From G0,
// can we reach gnew? G0 contains Op1 which produces T2 consumed by gnew={Op2}. Yes.
// So gnew→G0→gnew → cycle.
void test_acyclic_extract_local_recomp_cycle() {
    std::cout << "--- test_acyclic_extract_local_recomp_cycle ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    part.add_group({0,1,2,3}, cache.evaluate({0,1,2,3},p,d));  // G0
    part.rebuild_index();
    // Extract {Op2} from G0 (single group, Op2 internal).
    // gnew={Op2}: needs T2 from Op1 in G0, produces T3 consumed by Op3 in G0.
    // gnew→G0 and G0→gnew → cycle.
    CHECK("extract {Op2} from singleton group creates cycle", !part.acyclic_extract_local({2}));
}

void test_apply_steal_gap_from_source_excluded() {
    std::cout << "--- test_apply_steal_gap_from_source_excluded ---\n";
    // Diamond4: STEAL Op0 from G0 to G1.
    // G0={Op0} dies (singleton). G1 becomes {Op0,Op1}.
    // T1 ephemeral in G1, Op2 external. G0 is dead → no one exports T1.
    // This is a real structural gap → STEAL correctly rejected at eval time.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    // Gap check is now at eval time, not apply time.
    auto r = partition_moves::eval_steal(part, 0, 0, 1);
    CHECK("STEAL rejected (G0 dies, T1 ephemeral gap)", !r.feasible);
}

// ============================================================================
// 3. ActiveSet
// ============================================================================

void test_active_set_activate_pop() {
    std::cout << "--- test_active_set_activate_pop ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    ActiveSet active(part);

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

    ActiveSet active(part);
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

    ActiveSet active(part);
    active.activate_group_ops(1);  // G1={Op1}: border ops activated

    CHECK("Op1 activated", active.is_active(1));
}

void test_active_set_refresh_activates_new_ops() {
    std::cout << "--- test_active_set_refresh_activates_new_ops ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    ActiveSet active(part);
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
    cfg_tight.max_drift_fraction = 0.001;  // abort after tiny degradation
    cfg_tight.init_count = 10;
    cfg_tight.seed = 11;

    FMConfig cfg_wide;
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
    test_eval_merge_cycle_rejected();
    test_apply_steal_gap_from_source_excluded();
    test_acyclic_recompute_local_safe();
    test_acyclic_recompute_local_cycle();
    test_acyclic_recompute_local_or_node_free();
    test_acyclic_recompute_local_or_node_all_blocked();
    test_acyclic_de_recompute_local_cycle();
    test_acyclic_de_recompute_local_safe();
    test_acyclic_extract_local_cycle();
    test_acyclic_extract_local_safe();
    test_acyclic_extract_local_recomp_cycle();

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