// recompute_correctness_test.cpp
//
// Comprehensive unit tests for RECOMPUTE operations covering:
//   1. Basic recompute (3-op chain)
//   2. Recompute gain correctness (independent eval_set verification)
//   3. Recompute cyclic rejected (acyclic_recompute_local)
//   4. Recompute input unavailable rejected (ephemeral input gap)
//   5. Recompute makes tensor internal (boundary input eliminated)
//   6. Recompute source unchanged
//   7. Recompute with coupling (invalidate_couplings removes stale edge)
//   8. Recompute into dead group
//   9. Recompute op already in group
//  10. Recompute precomputed cost

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
// 1. test_basic_recompute
//
// 3-op chain: op0 -> T1 -> op1 -> T2 -> op2
// Groups: G0={op0, op1}, G1={op2}
// op0 produces T1 consumed by op1 (both in G0).
// Recompute op0 into G1.
// Verify: G1 now contains {op0, op2}, G0 unchanged at {op0, op1},
// op0 is in both groups, saving = G1.old_cost - G1.new_cost,
// partition is acyclic after.
// ============================================================================

void test_basic_recompute() {
    std::cout << "--- test_basic_recompute ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64},{64,64}};
    // op0: T0 -> T1
    // op1: T1 -> T2
    // op2: T2 -> T3
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
        {OpType::Pointwise, {2}, {3}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    // G0={op0, op1}, G1={op2}
    ctx.build_partition({{0, 1}, {2}});

    double cost_g0_before = ctx.part.groups[0].cost;
    double cost_g1_before = ctx.part.groups[1].cost;
    FlatSet<size_t> g0_ops_before = ctx.part.groups[0].ops;

    // Evaluate recompute: copy op0 into G1
    auto eval = partition_moves::eval_recompute(ctx.part, 0, 1);
    CHECK("basic_recompute: feasible", eval.feasible);

    // Expected: new G1 = {op0, op2}
    double new_g1_cost = ctx.part.eval_set({0, 2});
    double expected_saving = cost_g1_before - new_g1_cost;
    CHECK_CLOSE("basic_recompute: saving", eval.saving, expected_saving);

    // Acyclicity check
    CHECK("basic_recompute: acyclic_local", ctx.part.acyclic_recompute_local(0, 1));

    // Apply
    auto affected = partition_moves::apply_recompute(ctx.part, 0, 1);
    CHECK("basic_recompute: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // G0 unchanged
    CHECK("basic_recompute: G0 ops unchanged",
          ctx.part.groups[0].ops == g0_ops_before);
    CHECK_CLOSE("basic_recompute: G0 cost unchanged",
                ctx.part.groups[0].cost, cost_g0_before);

    // G1 now has {op0, op2}
    CHECK("basic_recompute: G1 has op0", ctx.part.groups[1].ops.count(0));
    CHECK("basic_recompute: G1 has op2", ctx.part.groups[1].ops.count(2));
    CHECK("basic_recompute: G1 size=2", ctx.part.groups[1].ops.size() == 2);

    // op0 is in two groups
    auto groups_of_op0 = ctx.part.groups_of(0);
    CHECK("basic_recompute: op0 in 2 groups", groups_of_op0.size() == 2);

    // Both groups alive
    CHECK("basic_recompute: G0 alive", ctx.part.groups[0].alive);
    CHECK("basic_recompute: G1 alive", ctx.part.groups[1].alive);

    // Acyclicity after
    CHECK("basic_recompute: acyclic_after", ctx.part.is_acyclic());
}

// ============================================================================
// 2. test_recompute_gain_correctness
//
// Verify saving by independently computing eval_set on the new group.
// Setup: op0 -> T1 -> op1, op0 -> T1 -> op2 (fan-out)
// G0={op0, op1}, G1={op2}
// Recompute op0 into G1.
// ============================================================================

void test_recompute_gain_correctness() {
    std::cout << "--- test_recompute_gain_correctness ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64},{64,64}};
    // op0: T0 -> T1
    // op1: T1 -> T2
    // op2: T1 -> T3
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 400},
        {OpType::Pointwise, {1}, {2}, 300},
        {OpType::Pointwise, {1}, {3}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    // G0={op0, op1}, G1={op2}
    ctx.build_partition({{0, 1}, {2}});

    double cost_g1_before = ctx.part.groups[1].cost;

    auto eval = partition_moves::eval_recompute(ctx.part, 0, 1);
    CHECK("gain_correctness: feasible", eval.feasible);

    // Independently compute what G1 should cost after recompute
    double new_g1_cost = ctx.part.eval_set({0, 2});
    double expected_saving = cost_g1_before - new_g1_cost;
    CHECK_CLOSE("gain_correctness: saving matches independent eval",
                eval.saving, expected_saving);

    // Apply and verify final cost
    partition_moves::apply_recompute(ctx.part, 0, 1);
    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    CHECK_CLOSE("gain_correctness: G1 cost after apply",
                ctx.part.groups[1].cost, new_g1_cost);
}

// ============================================================================
// 3. test_recompute_cyclic_rejected
//
// Create a topology where recomputing op into a target group would create
// a cycle. All producer groups of op's inputs are forward-reachable from
// the target group.
//
// Chain: op0 -> T1 -> op1 -> T2 -> op2
// Groups: G0={op0}, G1={op1}, G2={op2}
// Recomputing op2 into G0 would require G2's input (T2) from G1,
// creating edge G1->G0. But G0->G1 already exists -> cycle.
// ============================================================================

void test_recompute_cyclic_rejected() {
    std::cout << "--- test_recompute_cyclic_rejected ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    // op0: T0 -> T1
    // op1: T1 -> T2
    // op2: T2 -> T3
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
        {OpType::Pointwise, {2}, {3}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0}, G1={op1}, G2={op2}
    ctx.build_partition({{0}, {1}, {2}});

    CHECK("cyclic: partition acyclic", ctx.part.is_acyclic());

    // Recomputing op2 into G0: op2 consumes T2 produced by op1 in G1.
    // This adds edge G1->G0. But G0->G1 already exists (T1). -> cycle.
    CHECK("cyclic: recompute op2 into G0 rejected by acyclic_local",
          !ctx.part.acyclic_recompute_local(2, 0));

    // eval_recompute should also reject (it would pass ephemeral checks
    // but the caller checks acyclicity separately; the eval itself checks
    // creates_ephemeral_gap but not acyclicity — however let's verify
    // the acyclic check catches it)
    // Note: eval_recompute does NOT check acyclicity — that's caller's job.
    // The assertion is in apply_recompute. So we just verify acyclic_local rejects.

    // Also: recomputing op1 into G0 would create edge G0->G0 which is fine
    // (op1 consumes T1 from G0 itself). But forward: op1 produces T2 consumed
    // by op2 in G2 — edge G0->G2 which is new but no cycle.
    CHECK("cyclic: recompute op1 into G0 is acyclic",
          ctx.part.acyclic_recompute_local(1, 0));

    // Recomputing op0 into G2: op0 consumes T0 (graph input, no producer group).
    // op0 produces T1 consumed by op1 in G1. Adds edge G2->G1? No — G2 would
    // produce T1 internally; the question is whether T0 is available.
    // T0 has no producer op (graph input), so no new incoming edges for G2.
    // But G0 would still produce T1 -> G1, and G2 now also has op0 producing T1.
    // No cycle: G2 is a pure successor. acyclic check should pass.
    CHECK("cyclic: recompute op0 into G2 is acyclic",
          ctx.part.acyclic_recompute_local(0, 2));
}

// ============================================================================
// 4. test_recompute_input_unavailable_rejected
//
// Op consumes a tensor T that is ephemeral in all groups (never a boundary
// output). Recomputing op into a new group would create unsatisfiable demand
// for T. Verify eval_recompute rejects.
//
// Setup:
//   op0: T0 -> T1
//   op1: T1 -> T2
//   op2: T0 -> T3
// Groups: G0={op0, op1} (T1 is produced and consumed internally = ephemeral),
//         G1={op2}
// Recomputing op1 into G1: op1 needs T1. T1 is produced by op0 in G0.
// But T1 is NOT a boundary output of G0 (it's consumed by op1 inside G0).
// And T1 is not produced by G1. So T1 is unavailable. -> rejected.
// ============================================================================

void test_recompute_input_unavailable_rejected() {
    std::cout << "--- test_recompute_input_unavailable_rejected ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    // op0: T0 -> T1
    // op1: T1 -> T2
    // op2: T0 -> T3
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
        {OpType::Pointwise, {0}, {3}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0, op1}: T1 is produced by op0 and consumed by op1 -> ephemeral
    // G1={op2}
    ctx.build_partition({{0, 1}, {2}});

    // Verify T1 is not a boundary output of G0
    CHECK("input_unavail: T1 not boundary output of G0",
          !is_boundary_output_of(ctx.part.groups[0].ops, 1, ctx.dag));

    // Recompute op1 into G1: op1 needs T1 but T1 is not available
    auto eval = partition_moves::eval_recompute(ctx.part, 1, 1);
    CHECK("input_unavail: rejected", !eval.feasible);
}

// ============================================================================
// 5. test_recompute_makes_tensor_internal
//
// T was a boundary input of `into` (loaded from slow memory). After
// recomputing op (which produces T) into `into`, T becomes internal.
// Verify: T is no longer a boundary input of `into`, `into`'s cost
// decreases (no load cost for T).
//
// Setup:
//   op0: T0 -> T1
//   op1: T1 -> T2
// Groups: G0={op0}, G1={op1}
// T1 is a boundary output of G0 and boundary input of G1.
// Recompute op0 into G1 -> G1={op0, op1}, T1 is now internal.
// G1 no longer needs to load T1 from slow memory.
// ============================================================================

void test_recompute_makes_tensor_internal() {
    std::cout << "--- test_recompute_makes_tensor_internal ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64}};
    // op0: T0 -> T1
    // op1: T1 -> T2
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    // G0={op0}, G1={op1}
    ctx.build_partition({{0}, {1}});

    // T1 is boundary input of G1 before recompute
    CHECK("makes_internal: T1 boundary input of G1 before",
          is_boundary_input_of(ctx.part.groups[1].ops, 1, ctx.dag));

    // T1 is boundary output of G0
    CHECK("makes_internal: T1 boundary output of G0",
          is_boundary_output_of(ctx.part.groups[0].ops, 1, ctx.dag));

    double cost_g1_before = ctx.part.groups[1].cost;

    // Recompute op0 into G1
    auto eval = partition_moves::eval_recompute(ctx.part, 0, 1);
    CHECK("makes_internal: feasible", eval.feasible);

    auto affected = partition_moves::apply_recompute(ctx.part, 0, 1);
    CHECK("makes_internal: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // T1 is no longer a boundary input of G1 (op0 is now in G1, producing T1)
    CHECK("makes_internal: T1 NOT boundary input of G1 after",
          !is_boundary_input_of(ctx.part.groups[1].ops, 1, ctx.dag));

    // G1's cost should have decreased (no load penalty for T1)
    // The saving should be positive since we avoid loading T1 from slow memory
    CHECK("makes_internal: G1 cost decreased",
          ctx.part.groups[1].cost <= cost_g1_before);

    // Acyclicity
    CHECK("makes_internal: acyclic", ctx.part.is_acyclic());
}

// ============================================================================
// 6. test_recompute_source_unchanged
//
// After recompute, verify source groups are completely unchanged (same ops,
// same cost, same alive status).
//
// Setup: 4-op chain across 3 groups. Recompute op0 into G2.
// Verify G0 and G1 are completely unmodified.
// ============================================================================

void test_recompute_source_unchanged() {
    std::cout << "--- test_recompute_source_unchanged ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48}};
    // op0: T0 -> T1
    // op1: T1 -> T2
    // op2: T2 -> T3
    // op3: T3 -> T4
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
        {OpType::Pointwise, {2}, {3}, 300},
        {OpType::Pointwise, {3}, {4}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0}, G1={op1, op2}, G2={op3}
    ctx.build_partition({{0}, {1, 2}, {3}});

    // Snapshot before
    FlatSet<size_t> g0_ops_before = ctx.part.groups[0].ops;
    FlatSet<size_t> g1_ops_before = ctx.part.groups[1].ops;
    double g0_cost_before = ctx.part.groups[0].cost;
    double g1_cost_before = ctx.part.groups[1].cost;
    bool g0_alive_before = ctx.part.groups[0].alive;
    bool g1_alive_before = ctx.part.groups[1].alive;
    int g0_gen_before = ctx.part.groups[0].gen;
    int g1_gen_before = ctx.part.groups[1].gen;

    // Recompute op0 into G2 (op0 consumes T0 which is graph input, so available)
    auto eval = partition_moves::eval_recompute(ctx.part, 0, 2);
    CHECK("source_unchanged: feasible", eval.feasible);

    auto affected = partition_moves::apply_recompute(ctx.part, 0, 2);
    CHECK("source_unchanged: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // G0 unchanged
    CHECK("source_unchanged: G0 ops same", ctx.part.groups[0].ops == g0_ops_before);
    CHECK_CLOSE("source_unchanged: G0 cost same", ctx.part.groups[0].cost, g0_cost_before);
    CHECK("source_unchanged: G0 alive same", ctx.part.groups[0].alive == g0_alive_before);
    CHECK("source_unchanged: G0 gen same", ctx.part.groups[0].gen == g0_gen_before);

    // G1 unchanged
    CHECK("source_unchanged: G1 ops same", ctx.part.groups[1].ops == g1_ops_before);
    CHECK_CLOSE("source_unchanged: G1 cost same", ctx.part.groups[1].cost, g1_cost_before);
    CHECK("source_unchanged: G1 alive same", ctx.part.groups[1].alive == g1_alive_before);
    CHECK("source_unchanged: G1 gen same", ctx.part.groups[1].gen == g1_gen_before);

    // Only G2 should be affected
    CHECK("source_unchanged: only G2 affected",
          affected.size() == 1 && affected.count(2));

    CHECK("source_unchanged: acyclic", ctx.part.is_acyclic());
}

// ============================================================================
// 7. test_recompute_with_coupling
//
// `into` (G1) has incoming coupling from G0 retaining tensor T1.
// Recomputing op0 (which produces T1) into G1 makes T1 internal
// to G1 -> the entering coupling edge becomes stale.
// Verify invalidate_couplings removes the stale edge.
//
// Setup:
//   op0: T0 -> T1
//   op1: T1 -> T2
// Groups: G0={op0}, G1={op1}
// Coupling: G0 -> G1 retaining T1
// After recomputing op0 into G1: G1={op0, op1}, T1 produced internally.
// T1 is no longer a boundary output of G0 (op0 still in G0, but T1 is
// also consumed internally by op1... wait, op1 is in G1 not G0).
// Actually: G0={op0} still. T1 is produced by op0 in G0. Consumers of T1:
// op1 which is in G1. So T1 IS still a boundary output of G0.
// BUT T1 is no longer a boundary input of G1 (op0 is now in G1, producing T1).
// So invalidate_couplings should remove the edge because the consumer side
// check fails (T1 not boundary input of G1).
// ============================================================================

void test_recompute_with_coupling() {
    std::cout << "--- test_recompute_with_coupling ---\n";

    CoupledTestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64}};
    // op0: T0 -> T1
    // op1: T1 -> T2
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;
    ctx.prob.retainable_tensors = {1};

    // G0={op0}, G1={op1}
    ctx.build({{0}, {1}});

    // Wire coupling: G0 -> G1 retaining T1
    ctx.couple(0, 1, {1});

    CHECK("coupling: exists pre", ctx.cp.next_group[0] == 1);
    CHECK("coupling: prev_group set", ctx.cp.prev_group[1] == 0);
    CHECK("coupling: retained T1", ctx.cp.retained[{0, 1}].count(1));

    // Recompute op0 into G1
    auto eval = partition_moves::eval_recompute(ctx.cp.part, 0, 1);
    CHECK("coupling: eval feasible", eval.feasible);

    auto affected = partition_moves::apply_recompute(ctx.cp.part, 0, 1);
    CHECK("coupling: applied", !affected.empty());

    ctx.cp.part.rebuild_index();
    ctx.cp.part.rebuild_group_dag();

    // T1 is no longer boundary input of G1 (op0 now in G1 produces T1)
    CHECK("coupling: T1 not boundary input of G1",
          !is_boundary_input_of(ctx.cp.part.groups[1].ops, 1, *ctx.cp.part.dag));

    // invalidate_couplings should dissolve the stale coupling
    ctx.cp.invalidate_couplings();

    CHECK("coupling: dissolved next", ctx.cp.next_group[0] == SIZE_MAX);
    CHECK("coupling: dissolved prev", ctx.cp.prev_group[1] == SIZE_MAX);
    CHECK("coupling: retained removed",
          ctx.cp.retained.find({0, 1}) == ctx.cp.retained.end());
}

// ============================================================================
// 8. test_recompute_dead_group
//
// Recomputing into a dead group returns infeasible.
// ============================================================================

void test_recompute_dead_group() {
    std::cout << "--- test_recompute_dead_group ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48}};
    // op0: T0 -> T1
    // op1: T1 -> T2
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    ctx.build_partition({{0}, {1}});

    // Kill G1
    ctx.part.groups[1].alive = false;

    auto eval = partition_moves::eval_recompute(ctx.part, 0, 1);
    CHECK("dead_group: not feasible", !eval.feasible);

    // apply_recompute should also fail
    auto affected = partition_moves::apply_recompute(ctx.part, 0, 1);
    CHECK("dead_group: not applied", affected.empty());
}

// ============================================================================
// 9. test_recompute_already_in_group
//
// Op is already in `into`. Verify eval_recompute handles this gracefully
// (no-op or infeasible — the op set doesn't change).
// ============================================================================

void test_recompute_already_in_group() {
    std::cout << "--- test_recompute_already_in_group ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48}};
    // op0: T0 -> T1
    // op1: T1 -> T2
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0, op1}
    ctx.build_partition({{0, 1}});

    double cost_before = ctx.part.groups[0].cost;

    // Recomputing op0 into G0 — op0 is already in G0
    auto eval = partition_moves::eval_recompute(ctx.part, 0, 0);

    if (eval.feasible) {
        // If it's feasible, saving should be 0 (no change)
        CHECK_CLOSE("already_in: saving is zero", eval.saving, 0.0);

        // Apply should succeed but not change anything meaningful
        auto affected = partition_moves::apply_recompute(ctx.part, 0, 0);
        CHECK_CLOSE("already_in: cost unchanged", ctx.part.groups[0].cost, cost_before);
    } else {
        // If eval returns infeasible, that's also acceptable
        CHECK("already_in: gracefully rejected", true);
    }

    // Either way, partition should remain valid
    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();
    CHECK("already_in: acyclic", ctx.part.is_acyclic());
}

// ============================================================================
// 10. test_recompute_precomputed
//
// Verify apply_recompute with precomputed cost gives the same result as
// without precomputed cost.
// ============================================================================

void test_recompute_precomputed() {
    std::cout << "--- test_recompute_precomputed ---\n";

    // We'll build two identical partitions, apply recompute to both:
    // one with precomputed cost, one without.

    auto make_ctx = []() {
        TestContext ctx;
        ctx.prob.tensors = {{64,64},{64,64},{64,64},{64,64}};
        // op0: T0 -> T1
        // op1: T1 -> T2
        // op2: T1 -> T3
        ctx.prob.ops = {
            {OpType::Pointwise, {0}, {1}, 400},
            {OpType::Pointwise, {1}, {2}, 300},
            {OpType::Pointwise, {1}, {3}, 300},
        };
        ctx.prob.fast_memory_capacity = 50000;
        ctx.prob.slow_memory_bandwidth = 10;
        ctx.prob.native_w = 64;
        ctx.prob.native_h = 64;

        // G0={op0, op1}, G1={op2}
        ctx.build_partition({{0, 1}, {2}});
        return ctx;
    };

    // First: without precomputed cost
    auto ctx1 = make_ctx();
    auto eval1 = partition_moves::eval_recompute(ctx1.part, 0, 1);
    CHECK("precomputed: eval feasible", eval1.feasible);

    auto affected1 = partition_moves::apply_recompute(ctx1.part, 0, 1);
    CHECK("precomputed: applied without precomputed", !affected1.empty());
    double cost_without = ctx1.part.groups[1].cost;

    // Second: with precomputed cost
    auto ctx2 = make_ctx();
    // Precompute the cost manually
    double precomputed = ctx2.part.eval_set({0, 2});
    auto affected2 = partition_moves::apply_recompute(ctx2.part, 0, 1, precomputed);
    CHECK("precomputed: applied with precomputed", !affected2.empty());
    double cost_with = ctx2.part.groups[1].cost;

    // Both should give the same cost
    CHECK_CLOSE("precomputed: same cost", cost_with, cost_without);

    // Both should have the same ops
    CHECK("precomputed: same ops",
          ctx1.part.groups[1].ops == ctx2.part.groups[1].ops);

    // Both affected sets should be the same
    CHECK("precomputed: same affected", affected1 == affected2);
}

// ============================================================================
// main
// ============================================================================

int main() {
    test_basic_recompute();
    test_recompute_gain_correctness();
    test_recompute_cyclic_rejected();
    test_recompute_input_unavailable_rejected();
    test_recompute_makes_tensor_internal();
    test_recompute_source_unchanged();
    test_recompute_with_coupling();
    test_recompute_dead_group();
    test_recompute_already_in_group();
    test_recompute_precomputed();

    std::cout << "\n=== recompute_correctness_test: "
              << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
