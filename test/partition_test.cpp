// partition_test.cpp — Unit tests for Partition.
//
// Covers:
//   1.  Construction: trivial(), costs, group-level DAG
//   2.  Queries: groups_of, adjacent_groups, border_ops
//   3.  eval_set with and without cache
//   4.  connected_components
//   5.  eval_eject (basic, mid-split, recomputation, singleton guard)
//   6.  eval_split (bridge, non-bridge, 2-op group after Bug 1 fix)
//   7.  bridge_edges (chain, no-bridge fan-in, co-consumer bridge after Bug 2 fix)
//   8.  creates_ephemeral_gap (no-gap, gap, recomputation, self-produce, overload)
//   9.  acyclic_merge_local (chain, diamond)
//  10.  add_group / rebuild_index
//
// Bug fixes exercised:
//   Bug 1: eval_split size<3 guard removed. A 2-op group {A,B} where the edge
//          A-B is a bridge is now correctly split into {A} and {B}.
//   Bug 2: bridge_edges now iterates op_neighbors instead of op_succs. This
//          catches co-consumer bridges — edges between ops that share a common
//          boundary input tensor but have no direct DAG producer-consumer edge.
//          These are real bridges when they are the ONLY path between two parts
//          of a group.
//
// creates_ephemeral_gap semantics:
//   Subgraph marks tensor T as ephemeral if BOTH produced AND consumed within
//   the group, even if external consumers also exist. A gap occurs when T would
//   be ephemeral in the proposed group AND an external consumer has no other
//   source (no recomputation group containing the producer, no separate group
//   that produces T as a boundary output).
//
// finalize() and feasibility invariants:
//   Phase 1 local search maintains Group::ops and Group::cost only. Group::sg,
//   Group::best_cfg, and the group-level DAG are NOT maintained during search
//   (too expensive per move). finalize() rebuilds them once after Phase 1.
//   The three feasibility invariants enforced per move:
//     (a) Subgraph valid: eval_set(merged) < 1e17
//     (b) No ephemeral gap: !creates_ephemeral_gap(proposed)
//     (c) No cycle: !would_create_cycle(gi, gj)
//   would_create_cycle works directly on op-set reachability (from DAG bitmasks),
//   not the group-level DAG, so it is valid during Phase 1.
//
// Build: make partition_test
// Run:   ./partition_test

#include "partition/partition.h"
#include "core/cost_cache.h"
#include "core/dag.h"
#include "core/types.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

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

// T0->Op0->T1->Op1->T2->Op2->T3->Op3->T4
static Problem make_chain4() {
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},1000});
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    return p;
}

// T0->Op0->T1, T1->{Op1->T2, Op2->T3}, T2+T3->Op3->T4
static Problem make_diamond() {
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

// T0 shared by Op0 and Op1 (co-consumers), outputs merged by Op2
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
// 1. Construction
// ============================================================================
void test_trivial_construction() {
    std::cout << "--- test_trivial_construction ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK_EQ_S("4 groups", part.groups.size(), 4);
    CHECK_EQ_S("4 alive",  part.num_alive(), 4);
    for (size_t i = 0; i < 4; i++) {
        CHECK("alive",       part.groups[i].alive);
        CHECK_EQ_S("1 op",   part.groups[i].ops.size(), 1);
        CHECK("correct op",  part.groups[i].ops.count(i));
        // trivial() intentionally skips sg/best_cfg; finalize() populates them later.
        CHECK("cost finite", part.groups[i].cost < 1e17 && part.groups[i].cost > 0);
    }
    double sum = 0; for (auto& g : part.groups) sum += g.cost;
    CHECK_EQ("total_cost", part.total_cost(), sum);
}

void test_trivial_costs_match_best_cost() {
    std::cout << "--- test_trivial_costs_match_best_cost ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    for (size_t i = 0; i < 4; i++) {
        auto sg = Subgraph::create(p, d, {i});
        CHECK_EQ("cost==best_cost", part.groups[i].cost, sg->best_cost().latency);
    }
}

void test_trivial_group_dag() {
    std::cout << "--- test_trivial_group_dag ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK("g0 no preds",  part.group_preds[0].empty());
    CHECK("g1 pred=g0",   part.group_preds[1].count(0));
    CHECK("g2 pred=g1",   part.group_preds[2].count(1));
    CHECK("g3 pred=g2",   part.group_preds[3].count(2));
    CHECK("g0 succ=g1",   part.group_succs[0].count(1));
    CHECK("g3 no succs",  part.group_succs[3].empty());
    CHECK("g0 in_deg=0",  part.group_in_deg[0] == 0);
    CHECK("g1 in_deg=1",  part.group_in_deg[1] == 1);
}

// ============================================================================
// 2. Queries
// ============================================================================
void test_groups_of() {
    std::cout << "--- test_groups_of ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    for (size_t op = 0; op < 4; op++) {
        auto& gs = part.groups_of(op);
        CHECK_EQ_S("in 1 group", gs.size(), 1);
        CHECK("correct group",   part.groups[gs[0]].ops.count(op));
    }
}

void test_adjacent_groups_chain() {
    std::cout << "--- test_adjacent_groups_chain ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK_EQ_S("g0: 1 adj", part.adjacent_groups(0).size(), 1);
    CHECK("g0 adj g1",      part.adjacent_groups(0).count(1));
    CHECK_EQ_S("g1: 2 adj", part.adjacent_groups(1).size(), 2);
    CHECK("g1 adj g0",      part.adjacent_groups(1).count(0));
    CHECK("g1 adj g2",      part.adjacent_groups(1).count(2));
    CHECK_EQ_S("g3: 1 adj", part.adjacent_groups(3).size(), 1);
}

void test_adjacent_groups_co_consumer() {
    std::cout << "--- test_adjacent_groups_co_consumer ---\n";
    // In diamond trivial partition, Op1 and Op2 are co-consumers of T1.
    // Groups 1 and 2 must be adjacent (connected via co-consumer edge).
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    auto adj1 = part.adjacent_groups(1);
    CHECK("g1 adj g0 (producer)",    adj1.count(0));
    CHECK("g1 adj g2 (co-consumer)", adj1.count(2));
    CHECK("g1 adj g3 (consumer)",    adj1.count(3));
}

void test_border_ops() {
    std::cout << "--- test_border_ops ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d);
    part.cache = &cache;
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    size_t gi = part.add_group({1,2}, cache.evaluate({1,2}, p, d));
    part.rebuild_index();
    // Op1 borders Op0 (outside), Op2 borders Op3 (outside). Both are border.
    CHECK_EQ_S("2 border ops", part.border_ops(gi).size(), 2);
    CHECK("no internal ops",   part.internal_ops(gi).empty());
}

// ============================================================================
// 3. eval_set
// ============================================================================
void test_eval_set_no_cache() {
    std::cout << "--- test_eval_set_no_cache ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    Partition part; part.prob = &p; part.dag = &d; part.cache = nullptr;
    auto sg = Subgraph::create(p, d, {0});
    CHECK_EQ("no-cache == best_cost", part.eval_set({0}), sg->best_cost().latency);
}

void test_eval_set_with_cache() {
    std::cout << "--- test_eval_set_with_cache ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    double c1 = part.eval_set({0}), c2 = part.eval_set({0});
    CHECK("feasible",    c1 < 1e17);
    CHECK_EQ("same",     c1, c2);
    CHECK_EQ_S("1 hit",  cache.hits(), 1);
}

void test_eval_set_empty() {
    std::cout << "--- test_eval_set_empty ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    Partition part; part.prob = &p; part.dag = &d; part.cache = nullptr;
    CHECK("empty -> 1e18", part.eval_set({}) >= 1e17);
}

// ============================================================================
// 4. connected_components
// ============================================================================
void test_cc_chain() {
    std::cout << "--- test_cc_chain ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    auto comps = part.connected_components({0,1,2,3});
    CHECK_EQ_S("1 component", comps.size(), 1);
    CHECK_EQ_S("4 ops",       comps[0].size(), 4);
}

void test_cc_gap() {
    std::cout << "--- test_cc_gap ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK_EQ_S("{0,2,3} -> 2", part.connected_components({0,2,3}).size(), 2);
}

void test_cc_isolated() {
    std::cout << "--- test_cc_isolated ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK_EQ_S("{0,3} -> 2", part.connected_components({0,3}).size(), 2);
}

// ============================================================================
// 5. eval_eject
// ============================================================================
void test_eject_border() {
    std::cout << "--- test_eject_border ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    for (auto& g : part.groups) g.alive = false;
    double fc = cache.evaluate({0,1,2,3}, p, d);
    size_t gi = part.add_group({0,1,2,3}, fc); part.rebuild_index();
    auto r = part.eval_eject(0, gi);
    CHECK("feasible",           r.feasible);
    CHECK_EQ_S("1 remainder",   r.remainder_components.size(), 1);
    CHECK_EQ("saving", r.saving, fc - cache.evaluate({1,2,3},p,d) - cache.evaluate({0},p,d));
}

void test_eject_middle_splits() {
    std::cout << "--- test_eject_middle_splits ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    for (auto& g : part.groups) g.alive = false;
    double fc = cache.evaluate({0,1,2,3}, p, d);
    size_t gi = part.add_group({0,1,2,3}, fc); part.rebuild_index();
    auto r = part.eval_eject(2, gi);  // eject middle -> {0,1} and {3} split
    CHECK("feasible",           r.feasible);
    CHECK_EQ_S("2 remainders",  r.remainder_components.size(), 2);
    CHECK_EQ("saving", r.saving,
             fc - cache.evaluate({0,1},p,d) - cache.evaluate({3},p,d) - cache.evaluate({2},p,d));
}

void test_eject_recomputation_op() {
    std::cout << "--- test_eject_recomputation_op ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    size_t ga = part.add_group({0,1}, cache.evaluate({0,1},p,d));
    part.add_group({0,2}, cache.evaluate({0,2},p,d));
    part.add_group({3},   cache.evaluate({3},p,d));
    part.rebuild_index();
    auto r = part.eval_eject(0, ga);
    CHECK("feasible",             r.feasible);
    CHECK_EQ("singleton_cost=0",  r.singleton_cost, 0.0, 0.01);
}

void test_eject_singleton_infeasible() {
    std::cout << "--- test_eject_singleton_infeasible ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK("infeasible", !part.eval_eject(0, 0).feasible);
}

// ============================================================================
// 6. eval_split
// ============================================================================
void test_split_bridge() {
    std::cout << "--- test_split_bridge ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    double fc = cache.evaluate({0,1,2}, p, d);
    size_t gi = part.add_group({0,1,2}, fc); part.rebuild_index();
    auto r = part.eval_split(0, 1, gi);
    CHECK("feasible",        r.feasible);
    CHECK_EQ_S("side_a=1",   r.side_a.size(), 1);
    CHECK("Op0 in side_a",   r.side_a.count(0));
    CHECK_EQ_S("side_b=2",   r.side_b.size(), 2);
    CHECK_EQ("saving",       r.saving, fc - r.cost_a - r.cost_b);
}

void test_split_non_bridge() {
    std::cout << "--- test_split_non_bridge ---\n";
    // Fan-in: Op0 and Op1 share T0 (co-consumer). The Op0↔Op1 edge is NOT a bridge
    // because both ops also connect to Op2 via separate DAG edges.
    auto p = make_fanin(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    size_t gi = part.add_group({0,1,2}, cache.evaluate({0,1,2},p,d)); part.rebuild_index();
    CHECK("non-bridge infeasible", !part.eval_split(0, 1, gi).feasible);
}

void test_split_two_op_group() {
    std::cout << "--- test_split_two_op_group ---\n";
    // Bug 1 fix: {Op0,Op1} has one bridge edge. The old size<3 guard blocked
    // this. After removing the guard, the BFS correctly determines it is a bridge.
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    double c01 = cache.evaluate({0,1}, p, d);
    size_t gi = part.add_group({0,1}, c01); part.rebuild_index();
    auto r = part.eval_split(0, 1, gi);
    CHECK("2-op bridge feasible (Bug 1 fix)", r.feasible);
    CHECK_EQ_S("side_a=1", r.side_a.size(), 1);
    CHECK_EQ_S("side_b=1", r.side_b.size(), 1);
    CHECK_EQ("saving",     r.saving, c01 - r.cost_a - r.cost_b);
    std::cout << "    saving=" << r.saving << "\n";
}

// ============================================================================
// 7. bridge_edges
// ============================================================================
void test_bridge_chain() {
    std::cout << "--- test_bridge_chain ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    size_t gi = part.add_group({0,1,2}, cache.evaluate({0,1,2},p,d)); part.rebuild_index();
    auto br = part.bridge_edges(gi);
    CHECK_EQ_S("2 bridges", br.size(), 2);
    bool h01=false, h12=false;
    for (auto& e : br) {
        if ((e.first==0&&e.second==1)||(e.first==1&&e.second==0)) h01=true;
        if ((e.first==1&&e.second==2)||(e.first==2&&e.second==1)) h12=true;
    }
    CHECK("bridge (0,1)", h01);
    CHECK("bridge (1,2)", h12);
}

void test_bridge_no_bridges_fanin() {
    std::cout << "--- test_bridge_no_bridges_fanin ---\n";
    // Fan-in {0,1,2}: every edge has an alternate path. No bridges.
    // Op0 and Op1 share T0 (co-consumer). Op0->T1->Op2, Op1->T2->Op2.
    // Any edge removal still leaves the remaining ops connected.
    auto p = make_fanin(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    size_t gi = part.add_group({0,1,2}, cache.evaluate({0,1,2},p,d)); part.rebuild_index();
    CHECK("no bridges", part.bridge_edges(gi).empty());
}

void test_bridge_co_consumer_bridge() {
    std::cout << "--- test_bridge_co_consumer_bridge ---\n";
    // Bug 2 fix: Op1's ONLY connection to {Op0,Op2} is the co-consumer edge Op0<->Op1.
    // Op0->T1->Op2. Op1->T2 (boundary, not consumed internally). No DAG edge Op1->Op0 or Op1->Op2.
    // Old code (op_succs only) never proposed Op0<->Op1 as a split candidate.
    // New code (op_neighbors) finds this bridge.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},   // Op0: T0->T1
             {OpType::Pointwise,{0},{2},1000},   // Op1: T0->T2 (co-consumer with Op0)
             {OpType::Pointwise,{1},{3},1000}};  // Op2: T1->T3 (connects only to Op0)
    p.fast_memory_capacity = 200000; p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    bool co = false;
    for (auto v : d.op_neighbors[1]) if (v == 0) { co = true; break; }
    CHECK("co-consumer edge exists",           co);
    CHECK("Op1 has no DAG succ to Op2",        std::find(d.op_succs[1].begin(), d.op_succs[1].end(), (size_t)2) == d.op_succs[1].end());
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    size_t gi = part.add_group({0,1,2}, cache.evaluate({0,1,2},p,d)); part.rebuild_index();
    auto br = part.bridge_edges(gi);
    bool found = false;
    for (auto& e : br)
        if ((e.first==0&&e.second==1)||(e.first==1&&e.second==0)) { found=true; break; }
    CHECK("co-consumer bridge found (Bug 2 fix)", found);
    std::cout << "    bridges=" << br.size() << ": ";
    for (auto& e : br) std::cout << "(" << e.first << "," << e.second << ") ";
    std::cout << "\n";
}

// ============================================================================
// 8. creates_ephemeral_gap
// ============================================================================
void test_eph_chain_no_gap() {
    std::cout << "--- test_eph_chain_no_gap ---\n";
    // Merging {Op1,Op2} in chain trivial: T2 is ephemeral (Op1->Op2 internal).
    // Op3 needs T3 (from Op2, boundary output), not T2. No gap.
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK("no gap", !part.creates_ephemeral_gap({1,2}, 1, 2));
}

void test_eph_diamond_gap() {
    std::cout << "--- test_eph_diamond_gap ---\n";
    // Merging {Op0,Op1}: T1 produced by Op0 AND consumed by Op1 internally → ephemeral.
    // External consumer Op2 has no other source for T1 → gap.
    // (New rule: ephemeral iff produced+consumed inside, regardless of external consumers.)
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK("{0,1} has gap (T1 ephemeral, Op2 stranded)", part.creates_ephemeral_gap({0,1}, 0, 1));
    CHECK("{0,2} has gap (T1 ephemeral, Op1 stranded)", part.creates_ephemeral_gap({0,2}, 0, 2));
}

void test_eph_diamond_full_no_gap() {
    std::cout << "--- test_eph_diamond_full_no_gap ---\n";
    // Merging all four ops: no external consumers exist. No gap possible.
    // Must use vector overload to exclude all 4 groups.
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK("all merged: no gap",
          !part.creates_ephemeral_gap({0,1,2,3}, std::vector<size_t>{0,1,2,3}));
}

void test_eph_recomputation_resolves() {
    std::cout << "--- test_eph_recomputation_resolves ---\n";
    // ga={Op0,Op1}, gb={Op0,Op2}, gc={Op3}. Merge ga+gb -> {Op0,Op1,Op2}.
    // T1 ephemeral inside merged group. Op3 needs T2 (from Op1) and T3 (from Op2),
    // both still boundary outputs of merged group. T1 has no external consumer. No gap.
    auto p = make_diamond(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    size_t ga = part.add_group({0,1}, cache.evaluate({0,1},p,d));
    size_t gb = part.add_group({0,2}, cache.evaluate({0,2},p,d));
    part.add_group({3}, cache.evaluate({3},p,d));
    part.rebuild_index();
    CHECK("merge ga+gb: no gap", !part.creates_ephemeral_gap({0,1,2}, ga, gb));
}

void test_eph_external_has_self_producer() {
    std::cout << "--- test_eph_external_has_self_producer ---\n";
    // ga={Op0,Op1}, gb={Op0,Op2}, gc={Op3}. Propose {Op0,Op1} (exclude only ga).
    // T1 ephemeral in proposed. Op2 (external consumer of T1) is in gb.
    // gb contains Op0 (producer of T1) -> gb self-produces T1. No gap.
    auto p = make_diamond(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    size_t ga = part.add_group({0,1}, cache.evaluate({0,1},p,d));
    part.add_group({0,2}, cache.evaluate({0,2},p,d));
    part.add_group({3},   cache.evaluate({3},p,d));
    part.rebuild_index();
    CHECK("gb self-produces T1: no gap", !part.creates_ephemeral_gap({0,1}, ga, SIZE_MAX));
}

void test_eph_vector_overload() {
    std::cout << "--- test_eph_vector_overload ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    bool pair_result   = part.creates_ephemeral_gap({0,1}, 0, 1);
    bool vector_result = part.creates_ephemeral_gap({0,1}, std::vector<size_t>{0,1});
    CHECK("overloads agree", pair_result == vector_result);
    // Both return true: T1 is ephemeral in {Op0,Op1} (produced+consumed internally),
    // and external consumer Op2 has no other source → gap.
    CHECK("both detect gap", pair_result && vector_result);
}

// ============================================================================
// 9. acyclic_merge_local
// ============================================================================
void test_cycle_chain() {
    std::cout << "--- test_cycle_chain ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    // Adjacent merges never create cycles.
    CHECK("adj {0,1}: safe", part.acyclic_merge_local(0, 1));
    CHECK("adj {1,2}: safe", part.acyclic_merge_local(1, 2));
    CHECK("adj {2,3}: safe", part.acyclic_merge_local(2, 3));
    // Skipping a group creates a cycle: Op0 can reach Op2 through Op1.
    CHECK("skip {0,2}: cycle", !part.acyclic_merge_local(0, 2));
    CHECK("skip {0,3}: cycle", !part.acyclic_merge_local(0, 3));
    CHECK("skip {1,3}: cycle", !part.acyclic_merge_local(1, 3));
}

void test_cycle_diamond() {
    std::cout << "--- test_cycle_diamond ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    // All direct neighbors: fine.
    CHECK("{0,1}: safe", part.acyclic_merge_local(0, 1));
    CHECK("{0,2}: safe", part.acyclic_merge_local(0, 2));
    CHECK("{1,3}: safe", part.acyclic_merge_local(1, 3));
    CHECK("{2,3}: safe", part.acyclic_merge_local(2, 3));
    // Op1 and Op2 are siblings (co-consumers): no DAG path between them.
    CHECK("{1,2}: safe", part.acyclic_merge_local(1, 2));
    // Op0 can reach Op3 through the diamond: merging them creates a cycle.
    CHECK("{0,3}: cycle", !part.acyclic_merge_local(0, 3));
}

// ============================================================================
// 10. add_group / rebuild_index
// ============================================================================
void test_add_group_index() {
    std::cout << "--- test_add_group_index ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    size_t g0 = part.add_group({0,1}, cache.evaluate({0,1},p,d));
    size_t g1 = part.add_group({2,3}, cache.evaluate({2,3},p,d));
    auto chk = [&](size_t op, size_t eg) {
        auto& gs = part.groups_of(op);
        return gs.size() == 1 && gs[0] == eg;
    };
    CHECK("Op0->g0", chk(0,g0)); CHECK("Op1->g0", chk(1,g0));
    CHECK("Op2->g1", chk(2,g1)); CHECK("Op3->g1", chk(3,g1));
}

void test_add_group_recomputation() {
    std::cout << "--- test_add_group_recomputation ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    CostCache cache;
    Partition part; part.prob = &p; part.dag = &d; part.cache = &cache;
    size_t ga = part.add_group({0,1}, cache.evaluate({0,1},p,d));
    size_t gb = part.add_group({0,2}, cache.evaluate({0,2},p,d));
    auto& gs = part.groups_of(0);
    CHECK_EQ_S("Op0 in 2 groups", gs.size(), 2);
    bool a = std::find(gs.begin(),gs.end(),ga)!=gs.end();
    bool b = std::find(gs.begin(),gs.end(),gb)!=gs.end();
    CHECK("in ga", a); CHECK("in gb", b);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    test_trivial_construction();
    test_trivial_costs_match_best_cost();
    test_trivial_group_dag();

    test_groups_of();
    test_adjacent_groups_chain();
    test_adjacent_groups_co_consumer();
    test_border_ops();

    test_eval_set_no_cache();
    test_eval_set_with_cache();
    test_eval_set_empty();

    test_cc_chain();
    test_cc_gap();
    test_cc_isolated();

    test_eject_border();
    test_eject_middle_splits();
    test_eject_recomputation_op();
    test_eject_singleton_infeasible();

    test_split_bridge();
    test_split_non_bridge();
    test_split_two_op_group();

    test_bridge_chain();
    test_bridge_no_bridges_fanin();
    test_bridge_co_consumer_bridge();

    test_eph_chain_no_gap();
    test_eph_diamond_gap();
    test_eph_diamond_full_no_gap();
    test_eph_recomputation_resolves();
    test_eph_external_has_self_producer();
    test_eph_vector_overload();

    test_cycle_chain();
    test_cycle_diamond();

    test_add_group_index();
    test_add_group_recomputation();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}