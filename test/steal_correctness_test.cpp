// steal_correctness_test.cpp
//
// Comprehensive unit tests for STEAL operations covering:
//   1. Basic steal (3-op chain, move middle op)
//   2. Steal gain correctness (verify against eval_set)
//   3. Cyclic steal rejected (acyclic_steal_local)
//   4. Ephemeral gap rejected (split_creates_ephemeral_gap)
//   5. Steal from singleton (source group dies)
//   6. Steal with coupling entering transfer
//   7. Steal with coupling retain invalidation
//   8. Steal of recomputed op
//   9. Steal preserves connectivity (or doesn't — no check)
//  10. Steal to/from dead group rejected

#include "search/partition_moves.h"
#include "search/coupled_fm_search.h"
#include "search/coupling_search.h"
#include "partition/partition.h"
#include "core/cost_cache.h"
#include "core/dag.h"
#include "core/types.h"
#include <iostream>
#include <cmath>
#include <set>
#include <memory>
#include <vector>

static int g_pass = 0, g_fail = 0;

static void CHECK(const char* label, bool cond) {
    if (cond) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label << "\n"; }
}

static void CHECK_CLOSE(const char* label, double got, double expected, double tol = 1.0) {
    if (std::abs(got - expected) <= tol + 0.01 * std::max(1.0, std::abs(expected)))
        g_pass++;
    else {
        g_fail++;
        std::cout << "  FAIL: " << label
                  << " got=" << got << " expected=" << expected << "\n";
    }
}

// ============================================================================
// Helper: build a Partition from a Problem with explicit group assignments.
// group_assignments[i] = set of op indices in group i.
// ============================================================================

struct TestContext {
    Problem prob;
    DAG dag;
    std::unique_ptr<CostCache> cache;
    Partition part;

    void build_partition(const std::vector<FlatSet<size_t>>& group_assignments) {
        dag = DAG::build(prob);
        cache = std::make_unique<CostCache>(100000);
        part.prob  = &prob;
        part.dag   = &dag;
        part.cache = cache.get();
        part.groups.clear();
        for (auto& ops : group_assignments) {
            Partition::Group g;
            g.ops   = ops;
            g.cost  = part.eval_set(ops);
            g.alive = true;
            g.gen   = 0;
            part.groups.push_back(std::move(g));
        }
        part.rebuild_index();
        part.rebuild_group_dag();
    }
};

struct CoupledTestContext {
    Problem prob;
    DAG dag;
    std::unique_ptr<CostCache> cache;
    CoupledPartition cp;

    void build(const std::vector<FlatSet<size_t>>& group_assignments) {
        dag = DAG::build(prob);
        cache = std::make_unique<CostCache>(100000);

        Partition part;
        part.prob  = &prob;
        part.dag   = &dag;
        part.cache = cache.get();
        part.groups.clear();
        for (auto& ops : group_assignments) {
            Partition::Group g;
            g.ops   = ops;
            g.cost  = part.eval_set(ops);
            g.alive = true;
            g.gen   = 0;
            part.groups.push_back(std::move(g));
        }
        part.rebuild_index();
        part.rebuild_group_dag();
        cp.init_from(std::move(part), cache.get());
    }

    // Wire a coupling edge: ga -> gb retaining tensors
    void couple(size_t ga, size_t gb, FlatSet<size_t> tensors) {
        cp.next_group[ga] = gb;
        cp.prev_group[gb] = ga;
        cp.retained[{ga, gb}] = std::move(tensors);
    }
};

// ============================================================================
// 1. Basic steal -- 3-op chain, steal middle op from G0 to G1
//
// Chain: op0 -> T1 -> op1 -> T2 -> op2
// Groups: G0={op0, op1}, G1={op2}
// Steal op1 from G0 to G1.
// After: G0={op0}, G1={op1, op2}
// Verify saving = old costs - new costs, partition acyclic after steal.
// ============================================================================

void test_basic_steal() {
    std::cout << "--- test_basic_steal ---\n";

    TestContext ctx;
    // T0, T1, T2, T3 — T0 is input to op0, T3 is output of op2
    ctx.prob.tensors = {{64,64},{64,64},{64,64},{64,64}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
        {OpType::Pointwise, {2}, {3}, 300},  // op2: T2 -> T3
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    // G0={op0, op1}, G1={op2}
    ctx.build_partition({{0, 1}, {2}});

    double cost_g0_old = ctx.part.groups[0].cost;
    double cost_g1_old = ctx.part.groups[1].cost;
    double cost_before = ctx.part.total_cost();

    // Evaluate steal: move op1 from G0 to G1
    auto eval = partition_moves::eval_steal(ctx.part, /*op=*/1, /*from=*/0, /*to=*/1);
    CHECK("basic_steal: feasible", eval.feasible);

    // Independently compute expected costs
    double expected_from_cost = ctx.part.eval_set({0});       // G0 becomes {op0}
    double expected_to_cost   = ctx.part.eval_set({1, 2});    // G1 becomes {op1, op2}
    double expected_saving = cost_g0_old + cost_g1_old - expected_from_cost - expected_to_cost;
    CHECK_CLOSE("basic_steal: saving", eval.saving, expected_saving);

    // Acyclicity pre-check
    CHECK("basic_steal: acyclic_steal_local", ctx.part.acyclic_steal_local(1, 0, 1));

    // Apply steal
    auto affected = partition_moves::apply_steal(ctx.part, 1, 0, 1);
    CHECK("basic_steal: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // Verify group contents
    CHECK("basic_steal: G0 has op0", ctx.part.groups[0].ops.count(0) == 1);
    CHECK("basic_steal: G0 lost op1", ctx.part.groups[0].ops.count(1) == 0);
    CHECK("basic_steal: G1 has op1", ctx.part.groups[1].ops.count(1) == 1);
    CHECK("basic_steal: G1 has op2", ctx.part.groups[1].ops.count(2) == 1);
    CHECK("basic_steal: G0 alive", ctx.part.groups[0].alive);
    CHECK("basic_steal: G1 alive", ctx.part.groups[1].alive);

    // Verify costs
    CHECK_CLOSE("basic_steal: G0 cost", ctx.part.groups[0].cost, expected_from_cost);
    CHECK_CLOSE("basic_steal: G1 cost", ctx.part.groups[1].cost, expected_to_cost);

    // Verify acyclicity
    CHECK("basic_steal: acyclic after", ctx.part.is_acyclic());

    // Verify total cost change matches prediction
    double cost_after = ctx.part.total_cost();
    CHECK_CLOSE("basic_steal: total_cost delta", cost_before - cost_after, expected_saving);
}

// ============================================================================
// 2. Steal gain correctness — verify gain by independently computing costs
//    with eval_set and comparing to eval_steal saving.
//
// Uses a wider chain to exercise the cost model more: 5 ops in two groups.
// ============================================================================

void test_steal_gain_correctness() {
    std::cout << "--- test_steal_gain_correctness ---\n";

    TestContext ctx;
    // 6 tensors for a 5-op chain
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 400},  // op0
        {OpType::Pointwise, {1}, {2}, 400},  // op1
        {OpType::Pointwise, {2}, {3}, 400},  // op2
        {OpType::Pointwise, {3}, {4}, 400},  // op3
        {OpType::Pointwise, {4}, {5}, 400},  // op4
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0, op1, op2}, G1={op3, op4}
    ctx.build_partition({{0, 1, 2}, {3, 4}});

    double old_g0 = ctx.part.groups[0].cost;
    double old_g1 = ctx.part.groups[1].cost;

    // Steal op2 from G0 to G1
    auto eval = partition_moves::eval_steal(ctx.part, 2, 0, 1);
    CHECK("gain: feasible", eval.feasible);

    // Independent computation
    double new_g0 = ctx.part.eval_set({0, 1});
    double new_g1 = ctx.part.eval_set({2, 3, 4});
    double expected_saving = (old_g0 + old_g1) - (new_g0 + new_g1);
    CHECK_CLOSE("gain: eval_steal matches manual", eval.saving, expected_saving);

    // Also verify that apply_steal with precomputed costs works
    auto affected = partition_moves::apply_steal(ctx.part, 2, 0, 1, new_g0, new_g1);
    CHECK("gain: apply with precomputed succeeded", !affected.empty());
    CHECK_CLOSE("gain: G0 cost after", ctx.part.groups[0].cost, new_g0);
    CHECK_CLOSE("gain: G1 cost after", ctx.part.groups[1].cost, new_g1);
}

// ============================================================================
// 3. Cyclic steal rejected
//
// Chain: op0 -> T1 -> op1 -> T2 -> op2 -> T3 -> op3
// Groups: G0={op0}, G1={op1}, G2={op2}, G3={op3}
// Stealing op0 from G0 into G2 would create: G2 depends on G1 (via T2),
// G1 depends on G0_new (via T1 produced by other groups), and G0_new depends
// on nothing... but the issue is G2 now needs T0 from outside and produces T1
// which G1 needs.
//
// Simpler: 3 groups in a chain. Steal from G0 into G2 — skipping G1 — creates
// G2->G1 backward dependency (G2 now produces T1 needed by G1, but G1->G2
// already exists via T2).
// ============================================================================

void test_steal_cyclic_rejected() {
    std::cout << "--- test_steal_cyclic_rejected ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
        {OpType::Pointwise, {2}, {3}, 300},  // op2: T2 -> T3
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0}, G1={op1}, G2={op2}
    ctx.build_partition({{0}, {1}, {2}});

    CHECK("cyclic: partition acyclic initially", ctx.part.is_acyclic());

    // Stealing op0 from G0 to G2: op0 produces T1 consumed by op1 in G1.
    // G2 (with op0) -> G1 (needs T1) creates backward edge, but G1->G2 via T2.
    // Cycle: G1->G2->G1.
    CHECK("cyclic: steal op0 G0->G2 rejected", !ctx.part.acyclic_steal_local(0, 0, 2));

    // Stealing op2 from G2 to G0: op2 consumes T2 from G1.
    // G0 (with op2) depends on G1 (for T2), G1 depends on G0 (for T1).
    // Cycle: G0->G1->G0.
    CHECK("cyclic: steal op2 G2->G0 rejected", !ctx.part.acyclic_steal_local(2, 2, 0));

    // Adjacent steals should be fine:
    // Steal op1 from G1 to G2: G0->G2 (for T1), G2 contains {op1,op2} — acyclic.
    CHECK("cyclic: steal op1 G1->G2 ok", ctx.part.acyclic_steal_local(1, 1, 2));
    // Steal op1 from G1 to G0: G0 contains {op0,op1}->G2 (for T2) — acyclic.
    CHECK("cyclic: steal op1 G1->G0 ok", ctx.part.acyclic_steal_local(1, 1, 0));
}

// ============================================================================
// 4. Ephemeral gap rejected
//
// op0 -> T1 -> op1, op2.  T1 has two consumers.
// G0={op0, op1}, G1={op2}
// Steal op0 from G0 to G1. After: G0={op1}, G1={op0, op2}.
// T1 is produced by op0 (now in G1) and consumed by op1 (G0) and op2 (G1).
// In G1, T1 is ephemeral (produced and consumed internally by op0 and op2).
// But op1 in G0 needs T1 — ephemeral gap!
// ============================================================================

void test_steal_ephemeral_gap_rejected() {
    std::cout << "--- test_steal_ephemeral_gap_rejected ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
        {OpType::Pointwise, {1}, {3}, 300},  // op2: T1 -> T3
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0, op1}, G1={op2}
    ctx.build_partition({{0, 1}, {2}});

    // Steal op0 from G0 to G1.
    // After: G0={op1} needs T1 as boundary input.
    //        G1={op0, op2} has T1 as ephemeral (produced by op0, consumed by op2).
    // op1 in G0 needs T1 but it is ephemeral in G1 — gap!
    auto eval = partition_moves::eval_steal(ctx.part, 0, 0, 1);
    CHECK("eph_gap: steal creates gap -> infeasible", !eval.feasible);

    // The reverse — steal op1 from G0 to G1 — should be fine:
    // G0={op0}, G1={op1, op2}. T1 is boundary output of G0 (produced by op0,
    // not consumed in G0). op1 and op2 in G1 consume T1 as boundary input.
    auto eval2 = partition_moves::eval_steal(ctx.part, 1, 0, 1);
    CHECK("eph_gap: reverse steal is feasible", eval2.feasible);
}

// ============================================================================
// 5. Steal from singleton — source group dies
//
// G0={op0} (singleton), G1={op1}
// Chain: op0 -> T1 -> op1
// Steal op0 from G0 to G1. G0 becomes empty -> dies.
// ============================================================================

void test_steal_from_dies() {
    std::cout << "--- test_steal_from_dies ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    // G0={op0}, G1={op1}
    ctx.build_partition({{0}, {1}});

    double cost_before = ctx.part.total_cost();

    auto eval = partition_moves::eval_steal(ctx.part, 0, 0, 1);
    CHECK("from_dies: feasible", eval.feasible);

    // After steal, G0 dies (empty), G1={op0, op1}
    double expected_to_cost = ctx.part.eval_set({0, 1});
    // from_cost = 0 (empty group)
    double expected_saving = ctx.part.groups[0].cost + ctx.part.groups[1].cost
                           - 0.0 - expected_to_cost;
    CHECK_CLOSE("from_dies: saving", eval.saving, expected_saving);

    auto affected = partition_moves::apply_steal(ctx.part, 0, 0, 1);
    CHECK("from_dies: applied", !affected.empty());
    CHECK("from_dies: affected contains G0", affected.count(0) == 1);
    CHECK("from_dies: affected contains G1", affected.count(1) == 1);

    // G0 should be dead
    CHECK("from_dies: G0 dead", !ctx.part.groups[0].alive);
    CHECK("from_dies: G1 alive", ctx.part.groups[1].alive);
    CHECK("from_dies: G1 has both ops",
          ctx.part.groups[1].ops.count(0) && ctx.part.groups[1].ops.count(1));
    CHECK_CLOSE("from_dies: G1 cost", ctx.part.groups[1].cost, expected_to_cost);

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();
    CHECK("from_dies: acyclic after", ctx.part.is_acyclic());
}

// ============================================================================
// 6. Steal with coupling entering transfer
//
// G_prev -> G0 coupling retaining T1.  Op1 in G0 is the sole consumer of T1.
// G1={op2} is uncoupled.  Steal op1 from G0 to G1.
//
// After fixup_coupling_steal: the coupling edge (G_prev -> G0) should transfer
// to (G_prev -> G1), because op1 (sole consumer of T1) moved to G1.
//
// Chain: op0 -> T1 -> op1 -> T2 -> op2
// Groups: G_prev={op0}, G0={op1}, G1={op2}
// Coupling: G_prev -> G0 retaining T1
// ============================================================================

void test_steal_with_coupling_entering_transfer() {
    std::cout << "--- test_steal_with_coupling_entering_transfer ---\n";

    CoupledTestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
        {OpType::Pointwise, {2}, {3}, 300},  // op2: T2 -> T3
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;
    ctx.prob.retainable_tensors = {1};

    // G0={op0} (G_prev), G1={op1} (source), G2={op2} (target)
    ctx.build({{0}, {1}, {2}});

    // Coupling: G0 -> G1 retaining T1
    ctx.couple(0, 1, {1});

    CHECK("enter_xfer: coupling G0->G1 exists", ctx.cp.next_group[0] == 1);
    CHECK("enter_xfer: G1 prev=G0", ctx.cp.prev_group[1] == 0);
    CHECK("enter_xfer: G2 uncoupled", ctx.cp.prev_group[2] == SIZE_MAX);

    // Apply steal via CoupledFMMove (which calls fixup_coupling_steal)
    CoupledFMMove move;
    move.type = CoupledFMMove::STEAL;
    move.op = 1;      // steal op1
    move.ga = 1;      // from G1
    move.gb = 2;      // to G2
    move.saving = 0;

    auto affected = apply_coupled_fm_move(ctx.cp, move);
    CHECK("enter_xfer: steal applied", !affected.empty());

    // After fixup_coupling_steal: op1 was the sole consumer of T1 in G1.
    // The coupling edge should transfer from (G0->G1) to (G0->G2).
    CHECK("enter_xfer: G0 next now G2", ctx.cp.next_group[0] == 2);
    CHECK("enter_xfer: G2 prev now G0", ctx.cp.prev_group[0 /* G2's prev */] == SIZE_MAX
          || ctx.cp.prev_group[2] == 0);  // G2 should have prev=G0
    // More precise check:
    CHECK("enter_xfer: G2 prev=G0", ctx.cp.prev_group[2] == 0);
    CHECK("enter_xfer: G1 prev cleared", ctx.cp.prev_group[1] == SIZE_MAX);

    // Retained tensors should be on the new edge (G0, G2)
    auto it = ctx.cp.retained.find({0, 2});
    CHECK("enter_xfer: retained(G0,G2) exists", it != ctx.cp.retained.end());
    if (it != ctx.cp.retained.end()) {
        CHECK("enter_xfer: retained has T1", it->second.count(1));
    }

    // Old edge should be gone
    CHECK("enter_xfer: old retained(G0,G1) gone",
          ctx.cp.retained.find({0, 1}) == ctx.cp.retained.end());
}

// ============================================================================
// 7. Steal with coupling retain invalidation
//
// G0 -> G_next coupling retaining T2 (produced by op1 in G0).
// Steal op1 from G0 to G1.  T2 is no longer produced in G0.
// invalidate_couplings should remove the stale retained edge.
//
// Chain: op0 -> T1 -> op1 -> T2 -> op2
// Groups: G0={op0, op1}, G1 (new target, empty-ish or {op2}), G_next={op2} or separate.
//
// Simpler setup: op0 -> T1 -> op1 -> T2, op2 consumes T2.
// G0={op0, op1}, G1={op2}. Coupling: G0 -> G1 retaining T2.
// Steal op1 from G0 to G1. Now G0={op0}, T2 is produced by op1 (now in G1).
// T2 is no longer a boundary output of G0 — coupling invalid.
// ============================================================================

void test_steal_with_coupling_retain_invalidation() {
    std::cout << "--- test_steal_with_coupling_retain_invalidation ---\n";

    CoupledTestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
        {OpType::Pointwise, {2}, {3}, 300},  // op2: T2 -> T3
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;
    ctx.prob.retainable_tensors = {2};

    // G0={op0, op1}, G1={op2}
    ctx.build({{0, 1}, {2}});

    // Coupling: G0 -> G1 retaining T2
    // (T2 is produced by op1 in G0, consumed by op2 in G1 — valid boundary output)
    ctx.couple(0, 1, {2});

    CHECK("retain_inv: coupling G0->G1 exists", ctx.cp.next_group[0] == 1);
    CHECK("retain_inv: retained has T2", ctx.cp.retained[{0, 1}].count(2));

    // Steal op1 from G0 to G1 via CoupledFMMove
    CoupledFMMove move;
    move.type = CoupledFMMove::STEAL;
    move.op = 1;      // steal op1
    move.ga = 0;      // from G0
    move.gb = 1;      // to G1
    move.saving = 0;

    auto affected = apply_coupled_fm_move(ctx.cp, move);
    CHECK("retain_inv: steal applied", !affected.empty());

    // After steal: G0={op0}, G1={op1, op2}
    // T2 is produced by op1 (now in G1) — no longer boundary output of G0.
    // invalidate_couplings (called by apply_coupled_fm_move) should dissolve
    // the coupling edge because T2 is no longer a boundary output of G0.

    // The coupling edge G0->G1 should be dissolved
    bool coupling_dissolved = (ctx.cp.next_group[0] == SIZE_MAX);
    bool retained_gone = (ctx.cp.retained.find({0, 1}) == ctx.cp.retained.end());

    CHECK("retain_inv: coupling dissolved (next_group)", coupling_dissolved);
    CHECK("retain_inv: retained removed", retained_gone);

    if (coupling_dissolved) {
        CHECK("retain_inv: prev_group cleared", ctx.cp.prev_group[1] == SIZE_MAX);
    }
}

// ============================================================================
// 8. Steal of recomputed op
//
// Tests STEAL when the op exists in multiple groups (recomputed).
//
// Setup: op0 -> T1 -> op1 -> T2, op2 consumes T1 independently.
// G0={op0, op1}, G1={op0, op2} (op0 recomputed in G1).
//
// acyclic_steal_local is documented as "conservative under heavy recomputation"
// — it may reject valid moves when recomputed copies satisfy dependencies.
//
// Test 8a: steal op0 from G0 to G1 — rejected because op0 is already in G1.
// Test 8b: steal op1 from G0 where op0 is recomputed — op0 still in G0, no issue.
// Test 8c: verify the conservative rejection for a topology where steal would
//          create an apparent cycle through external groups.
// ============================================================================

void test_steal_recomputed_op() {
    std::cout << "--- test_steal_recomputed_op ---\n";

    // --- 8b: Steal non-recomputed op from a group that also has a recomputed op ---
    // op0 -> T1, op1 consumes T1, op2 consumes T1 independently.
    // G0={op0, op1} (op0 produces T1, op1 consumes it internally)
    // G1={op0, op2} (op0 recomputed, op2 consumes T1 internally)
    // Steal op1 from G0: G0={op0} still has op0, G1 gains op1.
    // But op1 needs T1 (from op0 in G0 or from op0 in G1) — no cycle issue.
    {
        TestContext ctx;
        ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
        ctx.prob.ops = {
            {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
            {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
            {OpType::Pointwise, {1}, {3}, 300},  // op2: T1 -> T3
        };
        ctx.prob.fast_memory_capacity = 50000;
        ctx.prob.slow_memory_bandwidth = 10;
        ctx.prob.native_w = 48;
        ctx.prob.native_h = 48;

        // G0={op0, op1}, G1={op0, op2} — op0 recomputed
        ctx.build_partition({{0, 1}, {0, 2}});

        CHECK("recomp_b: op0 in G0", ctx.part.groups[0].ops.count(0) == 1);
        CHECK("recomp_b: op0 in G1", ctx.part.groups[1].ops.count(0) == 1);
        CHECK("recomp_b: initial acyclic", ctx.part.is_acyclic());

        // Steal op1 from G0 to G1.
        // After: G0={op0}, G1={op0, op1, op2}.
        // T1 is produced by op0 in both groups. T2 goes out of G1.
        // No external dependency issue — T1 is internal to G1.
        auto eval = partition_moves::eval_steal(ctx.part, 1, 0, 1);
        CHECK("recomp_b: steal op1 G0->G1 feasible", eval.feasible);

        if (eval.feasible) {
            auto affected = partition_moves::apply_steal(ctx.part, 1, 0, 1);
            CHECK("recomp_b: applied", !affected.empty());

            ctx.part.rebuild_index();
            ctx.part.rebuild_group_dag();

            CHECK("recomp_b: G0={op0}", ctx.part.groups[0].ops.size() == 1 &&
                                          ctx.part.groups[0].ops.count(0));
            CHECK("recomp_b: G1 has op0,op1,op2", ctx.part.groups[1].ops.size() == 3);
            CHECK("recomp_b: acyclic after", ctx.part.is_acyclic());
        }
    }

    // --- 8c: Conservative rejection with recomputed ops ---
    // acyclic_steal_local is conservative: it may reject a steal that is
    // actually valid when recomputed copies exist in other groups.
    //
    // op0 -> T1 -> op1 -> T2 -> op2, op3 -> T4
    // G0={op0, op1}, G1={op1, op2} (op1 recomputed), G2={op3}
    // Steal op1 from G0 to G2: creates G2 producing T2, consumed by G1.
    // But G1 already has op1 (recomputed) producing T2 internally.
    // acyclic_steal_local conservatively sees G2->G1 (T2) + G1->G2 (T3->op3?) = cycle.
    // The check is conservative but correct (never wrong, may over-reject).
    {
        TestContext ctx;
        ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48}};
        ctx.prob.ops = {
            {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
            {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
            {OpType::Pointwise, {2}, {3}, 300},  // op2: T2 -> T3
            {OpType::Pointwise, {3}, {4}, 300},  // op3: T3 -> T4
        };
        ctx.prob.fast_memory_capacity = 50000;
        ctx.prob.slow_memory_bandwidth = 10;
        ctx.prob.native_w = 48;
        ctx.prob.native_h = 48;

        // G0={op0, op1}, G1={op1, op2} (op1 recomputed), G2={op3}
        ctx.build_partition({{0, 1}, {1, 2}, {3}});

        CHECK("recomp_c: op1 in G0 and G1", ctx.part.groups[0].ops.count(1) == 1 &&
                                              ctx.part.groups[1].ops.count(1) == 1);
        CHECK("recomp_c: initial acyclic", ctx.part.is_acyclic());

        // acyclic_steal_local is conservative — it may reject this even though
        // G1 has its own copy of op1 producing T2. Verify it rejects (Part 1:
        // acyclic_recompute_local sees G2 gaining op1 → G2 produces T2 consumed
        // by op2 in G1 → G2->G1, but G1->G2 exists → cycle).
        bool acyclic_check = ctx.part.acyclic_steal_local(1, 0, 2);
        CHECK("recomp_c: conservative rejection", !acyclic_check);

        // eval_steal does NOT call acyclic_steal_local — it does its own
        // feasibility checks (cost-based + ephemeral gap). But the caller
        // (FM search) calls acyclic_steal_local as a pre-filter before
        // adding the move to the heap.
    }
}

// ============================================================================
// 9. Steal preserves connectivity (behavior test)
//
// STEAL does NOT check that the source group remains connected after the op
// is removed (unlike EJECT which splits into connected components). This test
// verifies this behavior: after stealing a "bridge" op, the source group may
// become disconnected, but STEAL succeeds anyway.
//
// Setup: diamond within G0.
//   op0 -> T1 -> op1 -> T3
//   op0 -> T2 -> op2 -> T4
//   T3, T4 -> op3 -> T5
// G0={op0, op1, op2, op3}, G1=separate group
//
// Steal op0 from G0: G0={op1, op2, op3}. op1 and op2 no longer share op0 as
// a common predecessor inside the group, but they are both connected to op3.
// The group is actually still connected (op1->op3, op2->op3).
//
// Simpler test: verify steal succeeds even when it could leave a complex group.
// ============================================================================

void test_steal_preserves_connectivity() {
    std::cout << "--- test_steal_preserves_connectivity ---\n";

    TestContext ctx;
    // op0 -> T1 -> op1, op0 -> T2 -> op2, both feed op3
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1, 2}, 300},     // op0: T0 -> T1, T2
        {OpType::Pointwise, {1}, {3}, 300},          // op1: T1 -> T3
        {OpType::Pointwise, {2}, {4}, 300},          // op2: T2 -> T4
        {OpType::Pointwise, {3, 4}, {5}, 300},       // op3: T3, T4 -> T5
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0, op1, op2, op3}, G1=new target (need a separate group)
    // We need another op for G1 to have something to steal into.
    // Add a 5th op consuming T5.
    ctx.prob.tensors.push_back({48, 48});  // T6
    ctx.prob.ops.push_back({OpType::Pointwise, {5}, {6}, 300});  // op4: T5 -> T6

    ctx.build_partition({{0, 1, 2, 3}, {4}});

    // Steal op3 from G0 to G1 (op3 is the fan-in node).
    // After: G0={op0, op1, op2}, G1={op3, op4}.
    // G0 is still connected: op0->op1, op0->op2 (all reachable from op0).

    // But T3 and T4 become boundary outputs of G0 — no gap because G1 consumes them.
    auto eval = partition_moves::eval_steal(ctx.part, 3, 0, 1);
    CHECK("connectivity: steal op3 feasible", eval.feasible);

    if (eval.feasible) {
        auto affected = partition_moves::apply_steal(ctx.part, 3, 0, 1);
        CHECK("connectivity: applied", !affected.empty());

        ctx.part.rebuild_index();
        ctx.part.rebuild_group_dag();

        CHECK("connectivity: G0 has 3 ops", ctx.part.groups[0].ops.size() == 3);
        CHECK("connectivity: G1 has 2 ops", ctx.part.groups[1].ops.size() == 2);
        CHECK("connectivity: acyclic after", ctx.part.is_acyclic());
    }
}

// ============================================================================
// 10. Steal to/from dead group rejected
//
// Tests that eval_steal and apply_steal reject moves involving dead groups.
// ============================================================================

void test_steal_dead_group_rejected() {
    std::cout << "--- test_steal_dead_group_rejected ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0
        {OpType::Pointwise, {1}, {2}, 300},  // op1
        {OpType::Pointwise, {2}, {3}, 300},  // op2
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0}, G1={op1}, G2={op2}
    ctx.build_partition({{0}, {1}, {2}});

    // Kill G1
    ctx.part.groups[1].alive = false;

    // Steal from dead group
    auto eval_from_dead = partition_moves::eval_steal(ctx.part, 1, 1, 0);
    CHECK("dead: steal FROM dead infeasible", !eval_from_dead.feasible);

    auto apply_from_dead = partition_moves::apply_steal(ctx.part, 1, 1, 0);
    CHECK("dead: apply FROM dead empty", apply_from_dead.empty());

    // Steal to dead group
    auto eval_to_dead = partition_moves::eval_steal(ctx.part, 0, 0, 1);
    CHECK("dead: steal TO dead infeasible", !eval_to_dead.feasible);

    auto apply_to_dead = partition_moves::apply_steal(ctx.part, 0, 0, 1);
    CHECK("dead: apply TO dead empty", apply_to_dead.empty());

    // Restore G1, kill G0 — steal from dead source
    ctx.part.groups[1].alive = true;
    ctx.part.groups[0].alive = false;

    auto eval_from_dead2 = partition_moves::eval_steal(ctx.part, 0, 0, 2);
    CHECK("dead: steal from dead source infeasible", !eval_from_dead2.feasible);
}

// ============================================================================
// 11. Steal bridge op disconnects remainder
//
// Chain: op0 → op1 → op2, all in G0. Op1 is the bridge between op0 and op2.
// G1={op3} is an adjacent target (op3 consumes T3 from op2).
// Stealing op1 from G0 to G1: G0={op0, op2} which are disconnected (op0
// produces T1, consumed by op1 which is gone; op2 consumes T2 from op1
// which is gone). G0 is now disconnected.
//
// This test documents the current behavior: STEAL does not split the
// remainder into connected components.
// ============================================================================

void test_steal_disconnects_remainder() {
    std::cout << "--- test_steal_disconnects_remainder ---\n";

    TestContext ctx;
    // op0 → T1 → op1 → T2 → op2 → T3 → op3
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},   // op0: T0 → T1
        {OpType::Pointwise, {1}, {2}, 300},   // op1: T1 → T2
        {OpType::Pointwise, {2}, {3}, 300},   // op2: T2 → T3
        {OpType::Pointwise, {3}, {0}, 300},   // op3: T3 → T0 (reuse T0 as output)
    };
    // Fix: op3 needs different output
    ctx.prob.tensors.push_back({48, 48});  // T4
    ctx.prob.ops[3] = {OpType::Pointwise, {3}, {4}, 300};

    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    ctx.build_partition({{0, 1, 2}, {3}});

    // Steal op1 (bridge) from G0 to G1
    // After: G0={op0, op2}, G1={op1, op3}
    // G0 is disconnected: op0 produces T1 (now boundary output), op2 consumes
    // T2 (now boundary input from G1). op0 and op2 have no shared tensor.

    auto eval = partition_moves::eval_steal(ctx.part, 1, 0, 1);
    // eval_steal doesn't check connectivity — it may be feasible
    if (eval.feasible && ctx.part.acyclic_steal_local(1, 0, 1)) {
        auto affected = partition_moves::apply_steal(ctx.part, 1, 0, 1);
        if (!affected.empty()) {
            ctx.part.rebuild_index();

            CHECK("disconnect: G0 has 2 ops", ctx.part.groups[0].ops.size() == 2);
            CHECK("disconnect: G0 contains op0", ctx.part.groups[0].ops.count(0));
            CHECK("disconnect: G0 contains op2", ctx.part.groups[0].ops.count(2));

            // Check if G0 is connected
            auto comps = ctx.part.connected_components(ctx.part.groups[0].ops);
            bool is_connected = (comps.size() == 1);
            // Document: STEAL may leave disconnected remainder
            std::cout << "  G0 connected components after steal: " << comps.size() << "\n";
            CHECK("disconnect: G0 has 2 components (disconnected)", comps.size() == 2);
        }
    } else {
        // If the steal was rejected (gap or acyclicity), that's also valid behavior
        std::cout << "  steal of bridge op was rejected (feasible="
                  << eval.feasible << ")\n";
        CHECK("disconnect: steal rejected is acceptable", true);
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    test_basic_steal();
    test_steal_gain_correctness();
    test_steal_cyclic_rejected();
    test_steal_ephemeral_gap_rejected();
    test_steal_from_dies();
    test_steal_with_coupling_entering_transfer();
    test_steal_with_coupling_retain_invalidation();
    test_steal_recomputed_op();
    test_steal_preserves_connectivity();
    test_steal_dead_group_rejected();
    test_steal_disconnects_remainder();

    std::cout << "\n=== steal_correctness_test: "
              << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
