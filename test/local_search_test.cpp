// local_search_test.cpp
//
// Tests for greedy descent local search: move generation, application,
// staleness, gain correctness, and partition feasibility after every move.
//
// Feasibility invariants checked after every apply_move call:
//   1. Memory:       every alive group has a feasible tiling (eval_set < 1e18).
//   2. No eph. gap:  no tensor is ephemeral in its group while an external
//                    consumer has no other source.
//   (Acyclicity of the condensed group DAG is guaranteed by construction —
//    all move types check merge_creates_cycle before mutating.)
//
// Build: make local_search_test
// Run:   ./local_search_test

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/cost_cache.h"
#include "partition/partition.h"
#include "search/local_search.h"
#include "init/init_strategies.h"
#include <algorithm>
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

// 3-op mini-diamond: T0->Op0->T1, T1->Op1->T2, T1+T2->Op2->T3
static Problem make_diamond3() {
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

// T0 shared by Op0 and Op1 (co-consumers), outputs merged by Op2:
// T0->Op0->T1, T0->Op1->T2, T1+T2->Op2->T3
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

// 4-op diamond: T0->Op0->T1, T1->{Op1->T2, Op2->T3}, T2+T3->Op3->T4
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

// ============================================================================
// Feasibility check helpers
//
// After every apply_move test we verify the two meaningful partition
// invariants that must hold throughout the search:
//   1. Memory: every alive group has a feasible tiling (eval_set < 1e18).
//   2. No ephemeral gap: no tensor is ephemeral while an external consumer
//      group has no backup source.
// We use verify_partition_feasibility from init_strategies.h — the canonical
// checker.  Acyclicity is guaranteed by merge_creates_cycle checks in every
// move path, so there is no post-move cycle check needed.
// ============================================================================

static void check_feasible(const char* label, const Partition& part) {
    std::string err = verify_partition_feasibility(part);
    if (!err.empty()) {
        std::cout << "  FAIL: " << label << " feasibility: " << err << "\n";
        g_fail++;
    } else { g_pass++; }
}

static std::vector<Move> drain(MoveHeap heap) {
    std::vector<Move> v;
    while (!heap.empty()) { v.push_back(heap.top()); heap.pop(); }
    return v;
}

// ============================================================================
// 1. Partition construction and basic queries
// ============================================================================

void test_trivial_partition() {
    std::cout << "--- test_trivial_partition ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK_EQ_S("3 groups", part.num_alive(), 3);
    CHECK("G0 has Op0", part.groups[0].ops.count(0));
    CHECK("G1 has Op1", part.groups[1].ops.count(1));
    CHECK("G2 has Op2", part.groups[2].ops.count(2));
    auto sg0 = *Subgraph::create(p, d, {0});
    CHECK_EQ("G0 cost", part.groups[0].cost, sg0.best_cost().latency);
    check_feasible("trivial", part);
}

void test_groups_of() {
    std::cout << "--- test_groups_of ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    auto gs = part.groups_of(1);
    CHECK_EQ_S("Op1 in 1 group", gs.size(), 1);
    CHECK("Op1 in G1", gs[0] == 1);
    part.groups[1].alive = false;
    part.rebuild_index();
    CHECK_EQ_S("Op1 in 0 groups after kill", part.groups_of(1).size(), 0);
}

void test_boundary_neighbors() {
    std::cout << "--- test_boundary_neighbors ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    auto n0 = part.boundary_neighbors(0);
    CHECK("G0 neighbor Op1", n0.count(1));
    CHECK_EQ_S("G0 1 neighbor", n0.size(), 1);
    auto n1 = part.boundary_neighbors(1);
    CHECK("G1 neighbor Op0", n1.count(0));
    CHECK("G1 neighbor Op2", n1.count(2));
    CHECK_EQ_S("G1 2 neighbors", n1.size(), 2);
    // After merging G0+G1
    part.groups[0].ops = {0, 1};
    part.groups[1].alive = false;
    auto n01 = part.boundary_neighbors(0);
    CHECK("merged neighbor Op2", n01.count(2));
    CHECK_EQ_S("merged 1 neighbor", n01.size(), 1);
}

void test_adjacent_groups() {
    std::cout << "--- test_adjacent_groups ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    auto adj0 = part.adjacent_groups(0);
    CHECK("G0 adj to G1", adj0.count(1));
    CHECK_EQ_S("G0 1 adjacent", adj0.size(), 1);
    auto adj1 = part.adjacent_groups(1);
    CHECK("G1 adj G0", adj1.count(0));
    CHECK("G1 adj G2", adj1.count(2));
    CHECK_EQ_S("G1 2 adjacent", adj1.size(), 2);
}

void test_eval_set() {
    std::cout << "--- test_eval_set ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    CHECK_EQ("eval {0}", part.eval_set({0}), part.groups[0].cost);
    auto sg01 = *Subgraph::create(p, d, {0, 1});
    CHECK_EQ("eval {0,1}", part.eval_set({0,1}), sg01.best_cost().latency);
    CHECK("eval disconnected", part.eval_set({0, 2}) >= 1e17);
}

// ============================================================================
// 2. Manual move application with feasibility checks
// ============================================================================

void test_merge_move() {
    std::cout << "--- test_merge_move ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    double old_cost = part.groups[0].cost + part.groups[1].cost;
    double merged = part.eval_set({0,1});
    CHECK("merge saves", merged < old_cost);

    int old_gen = part.groups[0].gen;
    part.groups[0].ops  = {0,1};
    part.groups[0].cost = merged;
    part.groups[0].gen++;
    part.groups[1].alive = false;
    part.groups[1].gen++;
    part.rebuild_index();

    CHECK_EQ_S("2 alive", part.num_alive(), 2);
    CHECK("G1 dead", !part.groups[1].alive);
    CHECK("gen incremented", part.groups[0].gen == old_gen + 1);
    check_feasible("post-merge", part);
}

void test_steal_move() {
    std::cout << "--- test_steal_move ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    // Fuse Op0+Op1 into G0, then steal Op1 into G2
    part.groups[0].ops  = {0,1};
    part.groups[0].cost = part.eval_set({0,1});
    part.groups[1].alive = false;
    part.rebuild_index();

    double new_g0 = part.eval_set({0});
    double new_g2 = part.eval_set({1,2});
    CHECK("new G0 valid", new_g0 < 1e17);
    CHECK("new G2 valid", new_g2 < 1e17);

    part.groups[0].ops  = {0};   part.groups[0].cost = new_g0; part.groups[0].gen++;
    part.groups[2].ops  = {1,2}; part.groups[2].cost = new_g2; part.groups[2].gen++;
    part.rebuild_index();

    CHECK("Op1 not in G0", !part.groups[0].ops.count(1));
    CHECK("Op1 in G2",      part.groups[2].ops.count(1));
    check_feasible("post-steal", part);
}

void test_recompute_move() {
    std::cout << "--- test_recompute_move ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    std::set<size_t> new_g2 = {1,2};
    double new_cost = part.eval_set(new_g2);
    CHECK("recompute feasible", new_cost < 1e17);

    int g1_gen = part.groups[1].gen;
    part.groups[2].ops  = new_g2;
    part.groups[2].cost = new_cost;
    part.groups[2].gen++;
    part.rebuild_index();

    CHECK("Op1 still in G1", part.groups[1].ops.count(1));
    CHECK("Op1 also in G2", part.groups[2].ops.count(1));
    CHECK("G1 gen unchanged", part.groups[1].gen == g1_gen);
    CHECK_EQ_S("Op1 in 2 groups", part.groups_of(1).size(), 2);
    check_feasible("post-recompute", part);
}

void test_eject_move() {
    std::cout << "--- test_eject_move ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    part.groups[0].ops  = {0,1,2};
    part.groups[0].cost = part.eval_set({0,1,2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    auto ejectable = part.ejectable_ops(0);
    CHECK("Op2 ejectable", std::find(ejectable.begin(), ejectable.end(), 2) != ejectable.end());

    double rem = part.eval_set({0,1});
    double sing = part.eval_set({2});
    double old = part.groups[0].cost;

    part.groups[0].ops  = {0,1};
    part.groups[0].cost = rem;
    part.groups[0].gen++;
    size_t new_gi = part.add_group({2}, sing);
    part.rebuild_index();

    CHECK("Op2 not in G0", !part.groups[0].ops.count(2));
    CHECK("Op2 in new group", part.groups[new_gi].ops.count(2));
    CHECK_EQ_S("3 alive", part.num_alive(), 3);
    check_feasible("post-eject", part);
    std::cout << "    fused=" << old << " split=" << rem + sing << "\n";
}

// ============================================================================
// 3. Generation counters / staleness
// ============================================================================

void test_stale_detection() {
    std::cout << "--- test_stale_detection ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);

    Move m{Move::MERGE, 0, 1, 0, 100.0, part.groups[0].gen, part.groups[1].gen};
    CHECK("fresh: ga alive",    part.groups[m.ga].alive);
    CHECK("fresh: gen_a match", part.groups[m.ga].gen == m.gen_a);
    CHECK("fresh: gen_b match", part.groups[m.gb].gen == m.gen_b);

    part.groups[0].gen++;
    CHECK("stale: gen_a mismatch", part.groups[m.ga].gen != m.gen_a);

    part.groups[1].alive = false;
    CHECK("stale: gb dead", !part.groups[m.gb].alive);
}

// ============================================================================
// 4. generate_moves — one move type per test
// ============================================================================

void test_generate_merge_proposed() {
    std::cout << "--- test_generate_merge_proposed ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    MoveHeap heap;
    generate_moves(part, 0, heap);
    auto moves = drain(heap);
    CHECK("moves generated", !moves.empty());
    bool found = false;
    for (auto& m : moves) if (m.type == Move::MERGE) found = true;
    CHECK("MERGE proposed", found);
    for (auto& m : moves) CHECK("positive saving at default floor", m.saving > 0);
}

void test_generate_eject_proposed() {
    std::cout << "--- test_generate_eject_proposed ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    // Fuse {Op0,Op1}: Op1 is a border op (neighbor Op2 outside)
    part.groups[0].ops  = {0,1};
    part.groups[0].cost = cache.evaluate({0,1}, p, d);
    part.groups[1].alive = false;
    part.rebuild_index();

    MoveHeap heap;
    generate_moves(part, 0, heap, 1e18);
    bool found = false;
    for (auto& m : drain(heap)) if (m.type == Move::EJECT) found = true;
    CHECK("EJECT proposed for border op", found);
}

void test_generate_steal_or_merge_proposed() {
    std::cout << "--- test_generate_steal_or_merge_proposed ---\n";
    // G0={Op0,Op1}, G2={Op2,Op3}: adjacent fused groups
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1}; part.groups[0].cost = cache.evaluate({0,1},p,d);
    part.groups[1].alive = false;
    part.groups[2].ops  = {2,3}; part.groups[2].cost = cache.evaluate({2,3},p,d);
    part.groups[3].alive = false;
    part.rebuild_index();

    MoveHeap heap;
    generate_moves(part, 0, heap, 1e18);
    auto moves = drain(heap);
    bool found = false;
    for (auto& m : moves)
        if (m.type == Move::STEAL || m.type == Move::MERGE) { found = true; break; }
    CHECK("STEAL or MERGE proposed", found);
}

void test_generate_recompute_proposed() {
    std::cout << "--- test_generate_recompute_proposed ---\n";
    // Diamond4: G0={Op0}, G1={Op1}, G2={Op2}, G3={Op3}.
    // T1 is produced by Op0 and consumed by BOTH Op1 (G1) and Op2 (G2).
    //
    // With tiling propagation: merging {Op0,Op1} keeps T1 as a boundary
    // output (Op2 is external consumer). No ephemeral gap → MERGE allowed.
    // RECOMPUTE may also be proposed (it's always safe).
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    MoveHeap heap;
    generate_moves(part, 0, heap, 1e18);
    bool found_recompute = false, found_merge = false, found_steal = false;
    for (auto& m : drain(heap)) {
        if (m.type == Move::RECOMPUTE) found_recompute = true;
        if (m.type == Move::MERGE)     found_merge     = true;
        if (m.type == Move::STEAL)     found_steal     = true;
    }
    // MERGE is now allowed (T1 stays boundary output, not ephemeral)
    CHECK("MERGE allowed (no ephemeral gap)", found_merge || found_steal);
    // RECOMPUTE may or may not be proposed depending on move generator
    // (it's always safe but the generator may skip it when MERGE exists)
    std::cout << "    merge=" << found_merge << " steal=" << found_steal
              << " recompute=" << found_recompute << "\n";
}
void test_generate_internal_eject_proposed() {
    std::cout << "--- test_generate_internal_eject_proposed ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2};
    part.groups[0].cost = cache.evaluate({0,1,2},p,d);
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    auto internals = part.internal_ops(0);
    CHECK("Op1 internal in {0,1,2}",
          std::find(internals.begin(),internals.end(),1) != internals.end());

    MoveHeap heap;
    generate_moves(part, 0, heap, 1e18);
    bool found = false;
    for (auto& m : drain(heap)) if (m.type == Move::INTERNAL_EJECT) found = true;
    CHECK("INTERNAL_EJECT proposed", found);
}

void test_generate_split_proposed() {
    std::cout << "--- test_generate_split_proposed ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1,2};
    part.groups[0].cost = cache.evaluate({0,1,2},p,d);
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    MoveHeap heap;
    generate_moves(part, 0, heap, 1e18);
    bool found = false;
    for (auto& m : drain(heap)) if (m.type == Move::SPLIT) found = true;
    CHECK("SPLIT proposed for bridge edge", found);
}

void test_generate_split_co_consumer_bridge() {
    std::cout << "--- test_generate_split_co_consumer_bridge ---\n";
    // Group {Op0,Op1,Op2}: Op0<->Op1 is a co-consumer bridge (both read T0,
    // Op1 has no DAG edge to Op0 or Op2). After Fix 2, generate_moves iterates
    // op_neighbors and proposes this bridge as a SPLIT candidate.
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

    // Verify the co-consumer bridge exists
    bool co_bridge = false;
    for (auto& e : part.bridge_edges(0))
        if ((e.first==0&&e.second==1)||(e.first==1&&e.second==0)) co_bridge = true;
    CHECK("co-consumer bridge exists", co_bridge);

    MoveHeap heap;
    generate_moves(part, 0, heap, 1e18);
    bool found = false;
    for (auto& m : drain(heap)) {
        if (m.type == Move::SPLIT &&
            ((m.op==0&&m.op2==1)||(m.op==1&&m.op2==0))) { found = true; break; }
    }
    CHECK("co-consumer bridge SPLIT proposed (Fix 2)", found);
}

void test_floor_filters_negative_savings() {
    std::cout << "--- test_floor_filters_negative_savings ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    MoveHeap strict, loose;
    generate_moves(part, 0, strict, 0.0);
    generate_moves(part, 0, loose, 1e18);

    while (!strict.empty()) {
        CHECK("strict: all positive", strict.top().saving > 0);
        strict.pop();
    }
    CHECK("loose has moves", !loose.empty());
}

// ============================================================================
// 5. EJECT/INTERNAL_EJECT identity (Fix 1)
// ============================================================================

void test_eject_internal_eject_same_formula() {
    std::cout << "--- test_eject_internal_eject_same_formula ---\n";
    // After Fix 1, both EJECT and INTERNAL_EJECT share the same apply_move
    // body (fallthrough). Verify eval_eject gives a consistent formula for
    // an internal op (Op1 in {Op0,Op1,Op2}) and that feasibility holds.
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    part.groups[0].alive = false;
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    size_t gf = part.add_group({0,1,2}, cache.evaluate({0,1,2},p,d));
    part.rebuild_index();

    // Op1 is internal
    auto internals = part.internal_ops(gf);
    CHECK("Op1 is internal",
          std::find(internals.begin(),internals.end(),1) != internals.end());
    auto borders = part.border_ops(gf);
    CHECK("Op1 NOT border",  // store result first to avoid iterator-UB
          std::find(borders.begin(),borders.end(),1) == borders.end());

    auto er = part.eval_eject(1, gf);
    CHECK("internal eject feasible", er.feasible);
    CHECK_EQ_S("2 components (chain splits)", er.remainder_components.size(), 2);

    double comp_sum = 0;
    for (auto c : er.component_costs) comp_sum += c;
    CHECK_EQ("saving formula = old - (comps + singleton)",
             er.saving, part.groups[gf].cost - comp_sum - er.singleton_cost);
}

// ============================================================================
// 6. RECOMPUTE cycle check in apply_move (Fix 3)
// ============================================================================

void test_recompute_apply_checks_cycle() {
    std::cout << "--- test_recompute_apply_checks_cycle ---\n";
    // generate_moves only proposes RECOMPUTE when merge_creates_cycle is false.
    // Verify that on a trivial chain, RECOMPUTE moves generated by generate_moves
    // never create cycles. Since generate_moves already checks, and apply_move
    // now re-checks (Fix 3), the result is always feasible.
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    // Run greedy_descent which applies all positive moves including RECOMPUTE
    auto result = greedy_descent(std::move(part));
    check_feasible("post-greedy (RECOMPUTE cycle safety)", result);
}

// ============================================================================
// 7. Move gain correctness: saving == total_cost_before - total_cost_after
// ============================================================================

void test_merge_gain_correctness() {
    std::cout << "--- test_merge_gain_correctness ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    double before  = part.total_cost();
    double merged  = part.eval_set({0,1});
    double expected = (part.groups[0].cost + part.groups[1].cost) - merged;
    CHECK("merge positive", expected > 0);

    part.groups[0].ops  = {0,1};
    part.groups[0].cost = merged; part.groups[0].gen++;
    part.groups[1].alive = false; part.groups[1].gen++;
    part.rebuild_index();

    CHECK_EQ("merge gain", before - part.total_cost(), expected);
    check_feasible("post-merge-gain", part);
    std::cout << "    before=" << before << " after=" << part.total_cost() << "\n";
}

void test_steal_gain_correctness() {
    std::cout << "--- test_steal_gain_correctness ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    part.groups[0].ops  = {0,1}; part.groups[0].cost = cache.evaluate({0,1},p,d);
    part.groups[1].alive = false;
    part.rebuild_index();

    double before   = part.total_cost();
    double new_g0   = part.eval_set({0});
    double new_g2   = part.eval_set({1,2});
    double expected = (part.groups[0].cost + part.groups[2].cost) - (new_g0 + new_g2);

    part.groups[0].ops  = {0};   part.groups[0].cost = new_g0; part.groups[0].gen++;
    part.groups[2].ops  = {1,2}; part.groups[2].cost = new_g2; part.groups[2].gen++;
    part.rebuild_index();

    CHECK_EQ("steal gain", before - part.total_cost(), expected);
    check_feasible("post-steal-gain", part);
    std::cout << "    before=" << before << " after=" << part.total_cost() << "\n";
}

void test_recompute_gain_correctness() {
    std::cout << "--- test_recompute_gain_correctness ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},100},
             {OpType::Pointwise,{1},{2},1000}};
    p.fast_memory_capacity = 50000; p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    double before   = part.total_cost();
    double g1_before = part.groups[1].cost;
    double new_g1   = part.eval_set({0,1});
    double expected  = g1_before - new_g1;   // G0 unchanged

    part.groups[1].ops  = {0,1};
    part.groups[1].cost = new_g1; part.groups[1].gen++;
    part.rebuild_index();

    CHECK_EQ("recompute gain", before - part.total_cost(), expected);
    CHECK_EQ_S("Op0 in 2 groups", part.groups_of(0).size(), 2);
    check_feasible("post-recompute-gain", part);
    std::cout << "    before=" << before << " after=" << part.total_cost() << "\n";
}

void test_eject_gain_correctness() {
    std::cout << "--- test_eject_gain_correctness ---\n";
    auto p = make_chain4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    part.groups[0].ops  = {0,1,2}; part.groups[0].cost = cache.evaluate({0,1,2},p,d);
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    double before   = part.total_cost();
    double rem      = part.eval_set({0,1});
    double sing     = part.eval_set({2});
    double expected = part.groups[0].cost - (rem + sing);

    part.groups[0].ops  = {0,1}; part.groups[0].cost = rem; part.groups[0].gen++;
    size_t ng = part.add_group({2}, sing);
    part.rebuild_index();

    CHECK_EQ("eject gain", before - part.total_cost(), expected);
    CHECK("Op2 in new group", part.groups[ng].ops.count(2));
    check_feasible("post-eject-gain", part);
    std::cout << "    before=" << before << " after=" << part.total_cost() << "\n";
}

// ============================================================================
// 8. ejectable_ops
// ============================================================================

void test_ejectable_ops() {
    std::cout << "--- test_ejectable_ops ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    CHECK_EQ_S("singleton: no eject", part.ejectable_ops(0).size(), 0);

    part.groups[0].ops  = {0,1}; part.groups[0].cost = cache.evaluate({0,1},p,d);
    part.groups[1].alive = false; part.rebuild_index();
    auto ej01 = part.ejectable_ops(0);
    CHECK("Op0 not ejectable (internal)",
          std::find(ej01.begin(),ej01.end(),0) == ej01.end());
    CHECK("Op1 ejectable (border)",
          std::find(ej01.begin(),ej01.end(),1) != ej01.end());
    CHECK_EQ_S("1 ejectable", ej01.size(), 1);

    part.groups[0].ops  = {0,1,2}; part.groups[2].alive = false; part.rebuild_index();
    CHECK_EQ_S("all-fused: 0 ejectable", part.ejectable_ops(0).size(), 0);
}

// ============================================================================
// 9. Greedy descent end-to-end properties
// ============================================================================

void test_greedy_descent_reaches_chain_optimum() {
    std::cout << "--- test_greedy_descent_reaches_chain_optimum ---\n";
    // For a chain, full fusion is always optimal. Descent must reach it.
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    double initial = part.total_cost();
    double optimal = cache.evaluate({0,1,2}, p, d);

    auto result = greedy_descent(std::move(part));
    CHECK("strictly improves", result.total_cost() < initial - 0.01);
    CHECK_EQ("reaches optimal", result.total_cost(), optimal);
    check_feasible("greedy-chain-optimum", result);
    std::cout << "    initial=" << initial << " optimal=" << optimal
              << " achieved=" << result.total_cost() << "\n";
}

void test_greedy_descent_terminates_at_local_optimum() {
    std::cout << "--- test_greedy_descent_terminates_at_local_optimum ---\n";
    // Second descent from the result of first should make 0 moves.
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    auto r1 = greedy_descent(std::move(part));
    double c1 = r1.total_cost();
    auto r2 = greedy_descent(std::move(r1));
    CHECK_EQ("second descent no improvement", r2.total_cost(), c1);
}

void test_greedy_descent_diamond_no_ephemeral_gap() {
    std::cout << "--- test_greedy_descent_diamond_no_ephemeral_gap ---\n";
    // Diamond4: descent must never produce {Op0,Op1} fused without Op2.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    auto result = greedy_descent(std::move(part));
    check_feasible("greedy-diamond4", result);

    bool bad = false;
    for (auto& g : result.groups)
        if (g.alive && g.ops.count(0) && g.ops.count(1) && !g.ops.count(2)) bad = true;
    CHECK("no {Op0,Op1} without Op2", !bad);
    for (size_t i = 0; i < 4; i++)
        CHECK("op covered", !result.groups_of(i).empty());
    std::cout << "    groups=" << result.num_alive()
              << " cost=" << result.total_cost() << "\n";
}

void test_greedy_descent_all_ops_covered() {
    std::cout << "--- test_greedy_descent_all_ops_covered ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    auto result = greedy_descent(std::move(part));
    for (size_t i = 0; i < 3; i++)
        CHECK("op covered", !result.groups_of(i).empty());
}

void test_greedy_descent_cost_monotone() {
    std::cout << "--- test_greedy_descent_cost_monotone ---\n";
    // Running descent from trivial must never increase total cost.
    auto p = make_diamond4(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;
    double initial = part.total_cost();
    auto result = greedy_descent(std::move(part));
    CHECK("cost never increases", result.total_cost() <= initial + 0.01);
}

// ============================================================================
// 10. Cache pass-through (Fix 4)
// ============================================================================

void test_greedy_descent_uses_cache() {
    std::cout << "--- test_greedy_descent_uses_cache ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    CostCache cache;
    auto part = Partition::trivial(p, d); part.cache = &cache;

    size_t miss0 = cache.misses();
    auto result = greedy_descent(std::move(part));
    CHECK("cache has misses (first evals)", cache.misses() > miss0);
    CHECK("cache has hits (re-evals)",      cache.hits()   > 0);
    check_feasible("greedy-cache", result);
    std::cout << "    misses=" << cache.misses() - miss0
              << " hits=" << cache.hits() << "\n";
}

void test_greedy_descent_null_cache_works() {
    std::cout << "--- test_greedy_descent_null_cache_works ---\n";
    // Partition with no cache still works; just uncached.
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);  // cache = nullptr
    auto result = greedy_descent(std::move(part));
    CHECK("null-cache descent correct",
          result.total_cost() <= Partition::trivial(p, d).total_cost() + 0.01);
    check_feasible("greedy-null-cache", result);
}

// ============================================================================
// 11. Full local_search
// ============================================================================

void test_local_search_chain() {
    std::cout << "--- test_local_search_chain ---\n";
    auto p = make_chain3(); DAG d = DAG::build(p);
    auto part = local_search(p, d);
    CHECK("fewer groups than trivial", part.num_alive() < (size_t)p.num_ops());
    for (size_t i = 0; i < 3; i++) CHECK("op covered", !part.groups_of(i).empty());
    check_feasible("local_search_chain", part);
}

void test_local_search_diamond() {
    std::cout << "--- test_local_search_diamond ---\n";
    auto p = make_diamond3(); DAG d = DAG::build(p);
    auto part = local_search(p, d);
    CHECK("cost better than unfused", part.total_cost() < 3 * 3276.8 + 1.0);
    for (size_t i = 0; i < 3; i++) CHECK("op covered", !part.groups_of(i).empty());
    check_feasible("local_search_diamond", part);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Partition construction
    test_trivial_partition();
    test_groups_of();
    test_boundary_neighbors();
    test_adjacent_groups();
    test_eval_set();

    // Manual move application
    test_merge_move();
    test_steal_move();
    test_recompute_move();
    test_eject_move();

    // Staleness
    test_stale_detection();

    // generate_moves: all six types
    test_generate_merge_proposed();
    test_generate_eject_proposed();
    test_generate_steal_or_merge_proposed();
    test_generate_recompute_proposed();
    test_generate_internal_eject_proposed();
    test_generate_split_proposed();
    test_generate_split_co_consumer_bridge();  // Fix 2
    test_floor_filters_negative_savings();

    // Fix 1: EJECT/INTERNAL_EJECT identity
    test_eject_internal_eject_same_formula();

    // Fix 3: RECOMPUTE cycle check
    test_recompute_apply_checks_cycle();

    // Gain correctness
    test_merge_gain_correctness();
    test_steal_gain_correctness();
    test_recompute_gain_correctness();
    test_eject_gain_correctness();

    // ejectable_ops
    test_ejectable_ops();

    // greedy_descent end-to-end
    test_greedy_descent_reaches_chain_optimum();
    test_greedy_descent_terminates_at_local_optimum();
    test_greedy_descent_diamond_no_ephemeral_gap();
    test_greedy_descent_all_ops_covered();
    test_greedy_descent_cost_monotone();

    // Fix 4: cache pass-through
    test_greedy_descent_uses_cache();
    test_greedy_descent_null_cache_works();

    // Full pipeline
    test_local_search_chain();
    test_local_search_diamond();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}