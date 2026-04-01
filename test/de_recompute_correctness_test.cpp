// de_recompute_correctness_test.cpp
//
// Comprehensive unit tests for DE_RECOMPUTE operations covering:
//   1. Basic de_recompute (3-op chain, remove recomputed op from one group)
//   2. Gain correctness (saving = ga.cost - eval_set(ga.ops - {op}))
//   3. Cyclic rejection (acyclic_de_recompute_local)
//   4. Gap rejection (output tensor unavailable after removal)
//   5. Not-recomputed rejection (op in only one group)
//   6. ga dies (singleton group with recomputed op)
//   7. Coupling invalidation (removing op makes retained tensor stale)
//   8. Source unchanged (other groups containing op are unmodified)
//   9. Dead group (eval returns infeasible for dead group)
//  10. Precomputed cost (apply with precomputed gives same result)

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
// 1. test_basic_de_recompute
//
// 3-op fan-out: op0 -> T1, op1 consumes T0, op2 consumes T0.
// op0 is recomputed in G0 and G1 where it produces T1.
// G0={op0, op1}, G1={op0, op2}
// Remove op0 from G1. op0's output T1 is not consumed by anyone in G1
// (op2 consumes T0). T1 is a graph output (or consumed only in G0).
// So no external consumer issue. Verify: G1={op2}, G0 unchanged, saving
// correct, acyclic after.
//
// Actually simpler: op0 consumes graph input T0 and produces T1 (graph output).
// op1 consumes graph input T0 and produces T2. Unrelated ops.
// G0={op0, op1}, G1={op0}. Remove op0 from G0.
// op0's output T1 has no consumers -> no Part 2 issues. Passes acyclic check.
// ============================================================================

void test_basic_de_recompute() {
    std::cout << "--- test_basic_de_recompute ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64}};
    // op0: T0 -> T1  (T1 is graph output -- no consumers)
    // op1: T0 -> T2  (T2 is graph output -- no consumers)
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0
        {OpType::Pointwise, {0}, {2}, 300},  // op1
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    // G0={op0, op1}, G1={op0}  -- op0 recomputed in both
    // op0's output T1 has no consumers -> no acyclicity issues.
    // No cross-group edges. Group DAG is trivial.
    ctx.build_partition({{0, 1}, {0}});

    double cost_g0_before = ctx.part.groups[0].cost;
    FlatSet<size_t> g1_ops_before = ctx.part.groups[1].ops;
    double cost_g1_before = ctx.part.groups[1].cost;

    // op0 should be in two groups
    CHECK("basic: op0 in 2 groups before", ctx.part.groups_of(0).size() == 2);

    // Evaluate de_recompute: remove op0 from G0
    auto eval = partition_moves::eval_de_recompute(ctx.part, 0, 0);
    CHECK("basic: feasible", eval.feasible);

    // Expected saving = G0.cost - eval_set(G0.ops - {op0}) = G0.cost - eval_set({op1})
    double remaining_cost = ctx.part.eval_set({1});
    double expected_saving = cost_g0_before - remaining_cost;
    CHECK_CLOSE("basic: saving", eval.saving, expected_saving);

    // Acyclicity check
    CHECK("basic: acyclic_de_recompute_local",
          ctx.part.acyclic_de_recompute_local(0, 0));

    // Apply
    auto affected = partition_moves::apply_de_recompute(ctx.part, 0, 0);
    CHECK("basic: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // G0 now has only {op1}
    CHECK("basic: G0 has op1", ctx.part.groups[0].ops.count(1));
    CHECK("basic: G0 does not have op0", !ctx.part.groups[0].ops.count(0));
    CHECK("basic: G0 size=1", ctx.part.groups[0].ops.size() == 1);
    CHECK("basic: G0 alive", ctx.part.groups[0].alive);

    // G1 unchanged
    CHECK("basic: G1 ops unchanged", ctx.part.groups[1].ops == g1_ops_before);
    CHECK_CLOSE("basic: G1 cost unchanged", ctx.part.groups[1].cost, cost_g1_before);

    // Acyclicity after
    CHECK("basic: acyclic_after", ctx.part.is_acyclic());
}

// ============================================================================
// 2. test_de_recompute_gain_correctness
//
// Verify saving = ga.cost - eval_set(ga.ops - {op}) by computing independently.
//
// Setup: op0: T0 -> T1, op1: T0 -> T2, op2: T0 -> T3
// G0={op0, op1, op2}, G1={op0}  (op0 recomputed)
// Remove op0 from G0. Remaining = {op1, op2}.
// ============================================================================

void test_de_recompute_gain_correctness() {
    std::cout << "--- test_de_recompute_gain_correctness ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64},{64,64}};
    // op0: T0 -> T1
    // op1: T0 -> T2
    // op2: T0 -> T3
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 400},  // op0
        {OpType::Pointwise, {0}, {2}, 300},  // op1
        {OpType::Pointwise, {0}, {3}, 350},  // op2
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    // G0={op0, op1, op2}, G1={op0}
    ctx.build_partition({{0, 1, 2}, {0}});

    double cost_ga_before = ctx.part.groups[0].cost;

    auto eval = partition_moves::eval_de_recompute(ctx.part, 0, 0);
    CHECK("gain: feasible", eval.feasible);

    // Independently compute the remaining set cost
    double remaining_cost = ctx.part.eval_set({1, 2});
    double expected_saving = cost_ga_before - remaining_cost;
    CHECK_CLOSE("gain: saving matches independent eval", eval.saving, expected_saving);

    // Apply and verify
    partition_moves::apply_de_recompute(ctx.part, 0, 0);
    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    CHECK_CLOSE("gain: G0 cost after apply", ctx.part.groups[0].cost, remaining_cost);
}

// ============================================================================
// 3. test_de_recompute_cyclic_rejected
//
// Construct topology where removing the recomputed op creates a cycle.
//
//   op0: T0 -> T1, T2    (two outputs)
//   op1: T1 -> T3
//   op2: T2, T3 -> T4
// Groups: G0={op0, op1}, G1={op0, op2}
// op0 recomputed in G0 and G1.
//
// Group DAG:
//   T1: produced by op0 (G0, G1). Consumed by op1 (G0 only).
//     G0: op0 produces T1, op1 consumes T1 -> ephemeral.
//     G1: op0 produces T1. op1 NOT in G1. -> boundary output of G1.
//     For op1 in G0: producer op0 in G0 (internal). No edge.
//   T2: produced by op0 (G0, G1). Consumed by op2 (G1 only).
//     G1: op0 produces T2, op2 consumes T2 -> ephemeral.
//     G0: op0 produces T2. op2 NOT in G0. -> boundary output.
//     For op2 in G1: producer op0 in G1 (internal). No edge.
//   T3: produced by op1 (G0 only). Consumed by op2 (G1 only).
//     G0->G1 edge for T3.
//
// Group DAG: G0->G1. Acyclic.
//
// Remove op0 from G0: G0={op1}. op1 consumes T1 produced by op0 (now only
// in G1). G1 is forward-reachable from G0 (G0->G1 via T3). So all remaining
// copies of op0 (G1) are forward-reachable from G0 -> CYCLE.
// acyclic_de_recompute_local should reject.
// ============================================================================

void test_de_recompute_cyclic_rejected() {
    std::cout << "--- test_de_recompute_cyclic_rejected ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48}};
    // op0: T0 -> T1, T2    (two outputs)
    // op1: T1 -> T3
    // op2: T2, T3 -> T4
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1, 2}, 300},  // op0
        {OpType::Pointwise, {1}, {3}, 300},      // op1
        {OpType::Pointwise, {2, 3}, {4}, 300},   // op2
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0, op1}, G1={op0, op2}  -- op0 recomputed
    ctx.build_partition({{0, 1}, {0, 2}});

    CHECK("cyclic: partition acyclic", ctx.part.is_acyclic());
    CHECK("cyclic: op0 in 2 groups", ctx.part.groups_of(0).size() == 2);

    // Removing op0 from G0 would make G0={op1}. op1 needs T1 from G1.
    // But G0->G1 exists (T3). G1 is forward-reachable from G0.
    // All remaining copies of op0 (just G1) are forward-reachable from G0. -> CYCLE.
    CHECK("cyclic: de_recompute op0 from G0 rejected by acyclic_local",
          !ctx.part.acyclic_de_recompute_local(0, 0));

    // Removing op0 from G1 is also cyclic: G1={op2}. op2 needs T2 (produced
    // by op0). After removing op0 from G1, T2 must come from G0. But G0 is
    // forward-reachable from G1 (via T1 -> op1 in G0). -> CYCLE.
    CHECK("cyclic: de_recompute op0 from G1 also rejected",
          !ctx.part.acyclic_de_recompute_local(0, 1));

    // For a valid (non-cyclic) de_recompute, we need a setup where the
    // remaining copy is NOT forward-reachable from ga. Use independent groups:
    //   op3: T5 -> T6 (independent op)
    //   G3={op0, op3}, G4={op0}
    // Remove op0 from G3: G3={op3}. No consumer of op0's outputs in G3.
    // S={G4}. No forward path from G3 to G4. -> acyclic.
    // (We already test this pattern in test_basic_de_recompute.)
}

// ============================================================================
// 4. test_de_recompute_gap_rejected
//
// Op's output tensor T is ephemeral in all other groups containing op
// (produced + consumed internally). After removing op from ga, T is not
// available as boundary output anywhere. Verify eval_de_recompute rejects.
//
// Setup:
//   op0: T0 -> T1
//   op1: T1 -> T2
//   op2: T1 -> T3           (external consumer of T1)
// Groups: G0={op0, op1}, G1={op0}, G2={op2}
// op0 recomputed in G0 and G1.
//
// T1: produced by op0.
//   G0: produced by op0, consumed by op1 -> ephemeral (NOT boundary output).
//   G1: produced by op0, no consumer of T1 in G1 -> boundary output.
//   G2 gets T1 from G1.
//
// Removing op0 from G1: G1 dies. T1 is no longer produced by G1. The only
// remaining source is G0 where T1 is ephemeral (not boundary output).
// External consumer op2 in G2 has no source for T1. -> GAP -> rejected.
// ============================================================================

void test_de_recompute_gap_rejected() {
    std::cout << "--- test_de_recompute_gap_rejected ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    // op0: T0 -> T1
    // op1: T1 -> T2
    // op2: T1 -> T3
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0
        {OpType::Pointwise, {1}, {2}, 300},  // op1
        {OpType::Pointwise, {1}, {3}, 300},  // op2
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0, op1}, G1={op0}, G2={op2}
    ctx.build_partition({{0, 1}, {0}, {2}});

    CHECK("gap: partition acyclic", ctx.part.is_acyclic());
    CHECK("gap: op0 in 2 groups", ctx.part.groups_of(0).size() == 2);

    // T1: ephemeral in G0, boundary output of G1.
    CHECK("gap: T1 not boundary output of G0",
          !is_boundary_output_of(ctx.part.groups[0].ops, 1, ctx.dag));
    CHECK("gap: T1 is boundary output of G1",
          is_boundary_output_of(ctx.part.groups[1].ops, 1, ctx.dag));

    // Removing op0 from G1: G1 dies. T1 loses its only boundary output source.
    // op2 in G2 still needs T1. -> rejected by ephemeral gap check.
    auto eval = partition_moves::eval_de_recompute(ctx.part, 1, 0);
    CHECK("gap: eval_de_recompute rejects", !eval.feasible);
}

// ============================================================================
// 5. test_de_recompute_not_recomputed_rejected
//
// Op exists in only one group. Verify eval_de_recompute rejects (op must
// be in multiple groups to de_recompute).
// ============================================================================

void test_de_recompute_not_recomputed_rejected() {
    std::cout << "--- test_de_recompute_not_recomputed_rejected ---\n";

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

    // op0 is only in G0, op1 is only in G1
    CHECK("not_recomp: op0 in 1 group", ctx.part.groups_of(0).size() == 1);
    CHECK("not_recomp: op1 in 1 group", ctx.part.groups_of(1).size() == 1);

    // eval_de_recompute should reject: op must be in at least one other alive group
    auto eval0 = partition_moves::eval_de_recompute(ctx.part, 0, 0);
    CHECK("not_recomp: op0 from G0 rejected", !eval0.feasible);

    auto eval1 = partition_moves::eval_de_recompute(ctx.part, 1, 1);
    CHECK("not_recomp: op1 from G1 rejected", !eval1.feasible);
}

// ============================================================================
// 6. test_de_recompute_ga_dies
//
// ga has only op (singleton with recomputed op). After de_recompute, ga
// dies. Verify alive=false, saving = ga.cost.
//
// Setup:
//   op0: T0 -> T1
//   op1: T1 -> T2
// Groups: G0={op0}, G1={op0, op1}  -- op0 recomputed
// Remove op0 from G0 (singleton). G0 dies. saving = G0.cost (- 0).
// ============================================================================

void test_de_recompute_ga_dies() {
    std::cout << "--- test_de_recompute_ga_dies ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64}};
    // op0: T0 -> T1  (T1 is graph output)
    // op1: T0 -> T2  (T2 is graph output)
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {0}, {2}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    // G0={op0}, G1={op0, op1}
    ctx.build_partition({{0}, {0, 1}});

    double cost_g0 = ctx.part.groups[0].cost;

    CHECK("ga_dies: op0 in 2 groups", ctx.part.groups_of(0).size() == 2);
    CHECK("ga_dies: G0 singleton", ctx.part.groups[0].ops.size() == 1);

    // Eval
    auto eval = partition_moves::eval_de_recompute(ctx.part, 0, 0);
    CHECK("ga_dies: feasible", eval.feasible);

    // Saving = ga.cost - 0 (ga dies, new cost is 0)
    CHECK_CLOSE("ga_dies: saving = ga.cost", eval.saving, cost_g0);

    // Apply
    auto affected = partition_moves::apply_de_recompute(ctx.part, 0, 0);
    CHECK("ga_dies: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // G0 should be dead
    CHECK("ga_dies: G0 dead", !ctx.part.groups[0].alive);

    // G1 should still be alive and unchanged
    CHECK("ga_dies: G1 alive", ctx.part.groups[1].alive);
    CHECK("ga_dies: G1 has op0", ctx.part.groups[1].ops.count(0));
    CHECK("ga_dies: G1 has op1", ctx.part.groups[1].ops.count(1));

    // Acyclicity
    CHECK("ga_dies: acyclic_after", ctx.part.is_acyclic());
}

// ============================================================================
// 7. test_de_recompute_with_coupling
//
// ga has outgoing coupling retaining tensor T produced by op. After removing
// op from ga, T is no longer boundary output -> coupling becomes stale.
// Verify invalidate_couplings dissolves it.
//
// Setup:
//   op0: T0 -> T1
//   op1: T0 -> T2
//   op2: T1 -> T3
// Groups: G0={op0, op1}, G1={op0, op2}
// op0 recomputed in G0 and G1.
// T1 is boundary output of G0 (produced by op0, consumed by op2 NOT in G0).
// Coupling: G0 -> G2 (some group) retaining T1.
//
// Actually: T1 consumed by op2 which is in G1. op0 also in G1 so T1 is
// ephemeral in G1. But T1 IS boundary output of G0 (op0 produces, no
// consumer of T1 in G0 -- op1 consumes T0 not T1).
//
// We couple G0 -> G1 retaining T1. T1 is boundary output of G0 and boundary
// input of G1 (produced externally by op0 in G0, consumed by op2 in G1...
// but op0 is also in G1 producing T1 internally. So T1 is NOT a boundary
// input of G1).
//
// Hmm. For the coupling to be valid, T must be boundary output of source
// AND boundary input of destination. With op0 in G1 producing T1 internally,
// T1 is not boundary input of G1. We need a destination group where T1 is
// genuinely a boundary input.
//
// Setup with 3 groups:
//   op0: T0 -> T1
//   op1: T0 -> T2          (independent)
//   op2: T1 -> T3          (consumes T1)
// Groups: G0={op0, op1}, G1={op0}, G2={op2}
// op0 recomputed in G0 and G1.
// T1: boundary output of G0 (produced by op0, consumed by op2 NOT in G0).
// T1: boundary output of G1 (produced by op0, consumed by op2 NOT in G1).
// T1: boundary input of G2 (consumed by op2, produced by op0 NOT in G2).
// Coupling: G0 -> G2 retaining T1.
//
// Remove op0 from G0: G0={op1}. T1 no longer produced by G0.
// Coupling G0->G2 retaining T1 becomes stale (T1 not boundary output of G0).
// invalidate_couplings dissolves.
// ============================================================================

void test_de_recompute_with_coupling() {
    std::cout << "--- test_de_recompute_with_coupling ---\n";

    CoupledTestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    // op0: T0 -> T1
    // op1: T0 -> T2
    // op2: T1 -> T3
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0
        {OpType::Pointwise, {0}, {2}, 300},  // op1
        {OpType::Pointwise, {1}, {3}, 300},  // op2
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;
    ctx.prob.retainable_tensors = {1};

    // G0={op0, op1}, G1={op0}, G2={op2}
    ctx.build({{0, 1}, {0}, {2}});

    // T1 is boundary output of G0 (produced by op0, op1 does not consume T1,
    // op2 is external)
    CHECK("coupling: T1 boundary output of G0",
          is_boundary_output_of(ctx.cp.part.groups[0].ops, 1, *ctx.cp.part.dag));
    // T1 is boundary input of G2
    CHECK("coupling: T1 boundary input of G2",
          is_boundary_input_of(ctx.cp.part.groups[2].ops, 1, *ctx.cp.part.dag));

    // Wire coupling: G0 -> G2 retaining T1
    ctx.couple(0, 2, {1});

    CHECK("coupling: coupling exists pre", ctx.cp.next_group[0] == 2);
    CHECK("coupling: retained T1", ctx.cp.retained[{0, 2}].count(1));

    // De-recompute: remove op0 from G0
    auto eval = partition_moves::eval_de_recompute(ctx.cp.part, 0, 0);
    CHECK("coupling: eval feasible", eval.feasible);

    auto affected = partition_moves::apply_de_recompute(ctx.cp.part, 0, 0);
    CHECK("coupling: applied", !affected.empty());

    ctx.cp.part.rebuild_index();
    ctx.cp.part.rebuild_group_dag();

    // G0={op1}. T1 no longer produced by G0.
    CHECK("coupling: G0 does not have op0", !ctx.cp.part.groups[0].ops.count(0));
    CHECK("coupling: T1 not boundary output of G0",
          !is_boundary_output_of(ctx.cp.part.groups[0].ops, 1, *ctx.cp.part.dag));

    // invalidate_couplings should dissolve the stale coupling
    ctx.cp.invalidate_couplings();

    CHECK("coupling: dissolved next", ctx.cp.next_group[0] == SIZE_MAX);
    CHECK("coupling: dissolved prev", ctx.cp.prev_group[2] == SIZE_MAX);
    CHECK("coupling: retained removed",
          ctx.cp.retained.find({0, 2}) == ctx.cp.retained.end());
}

// ============================================================================
// 8. test_de_recompute_source_unchanged
//
// Other groups containing op are completely unchanged after de_recompute.
//
// Setup: op0: T0 -> T1, op1: T0 -> T2. G0={op0, op1}, G1={op0}.
// Remove op0 from G0. Verify G1 is completely unmodified.
// ============================================================================

void test_de_recompute_source_unchanged() {
    std::cout << "--- test_de_recompute_source_unchanged ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64}};
    // op0: T0 -> T1
    // op1: T0 -> T2
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0
        {OpType::Pointwise, {0}, {2}, 300},  // op1
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    // G0={op0, op1}, G1={op0}
    ctx.build_partition({{0, 1}, {0}});

    // Snapshot G1 before
    FlatSet<size_t> g1_ops_before = ctx.part.groups[1].ops;
    double g1_cost_before = ctx.part.groups[1].cost;
    bool g1_alive_before = ctx.part.groups[1].alive;
    int g1_gen_before = ctx.part.groups[1].gen;

    // Remove op0 from G0
    auto eval = partition_moves::eval_de_recompute(ctx.part, 0, 0);
    CHECK("source_unchanged: feasible", eval.feasible);

    auto affected = partition_moves::apply_de_recompute(ctx.part, 0, 0);
    CHECK("source_unchanged: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // G1 should be completely unchanged
    CHECK("source_unchanged: G1 ops same", ctx.part.groups[1].ops == g1_ops_before);
    CHECK_CLOSE("source_unchanged: G1 cost same", ctx.part.groups[1].cost, g1_cost_before);
    CHECK("source_unchanged: G1 alive same", ctx.part.groups[1].alive == g1_alive_before);
    CHECK("source_unchanged: G1 gen same", ctx.part.groups[1].gen == g1_gen_before);

    // Only G0 should be affected
    CHECK("source_unchanged: only G0 affected",
          affected.size() == 1 && affected.count(0));

    CHECK("source_unchanged: acyclic", ctx.part.is_acyclic());
}

// ============================================================================
// 9. test_de_recompute_dead_group
//
// Verify eval_de_recompute returns infeasible for a dead group.
// ============================================================================

void test_de_recompute_dead_group() {
    std::cout << "--- test_de_recompute_dead_group ---\n";

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

    // G0={op0}, G1={op0, op1}  -- op0 recomputed
    ctx.build_partition({{0}, {0, 1}});

    // Kill G0
    ctx.part.groups[0].alive = false;

    auto eval = partition_moves::eval_de_recompute(ctx.part, 0, 0);
    CHECK("dead: not feasible", !eval.feasible);

    // apply should also fail
    auto affected = partition_moves::apply_de_recompute(ctx.part, 0, 0);
    CHECK("dead: not applied", affected.empty());
}

// ============================================================================
// 10. test_de_recompute_precomputed
//
// Verify apply_de_recompute with precomputed cost gives same result as
// without precomputed cost.
// ============================================================================

void test_de_recompute_precomputed() {
    std::cout << "--- test_de_recompute_precomputed ---\n";

    auto make_ctx = []() {
        TestContext ctx;
        ctx.prob.tensors = {{64,64},{64,64},{64,64},{64,64}};
        // op0: T0 -> T1
        // op1: T0 -> T2
        // op2: T0 -> T3
        ctx.prob.ops = {
            {OpType::Pointwise, {0}, {1}, 400},  // op0
            {OpType::Pointwise, {0}, {2}, 300},  // op1
            {OpType::Pointwise, {0}, {3}, 300},  // op2
        };
        ctx.prob.fast_memory_capacity = 50000;
        ctx.prob.slow_memory_bandwidth = 10;
        ctx.prob.native_w = 64;
        ctx.prob.native_h = 64;

        // G0={op0, op1, op2}, G1={op0}  -- op0 recomputed
        ctx.build_partition({{0, 1, 2}, {0}});
        return ctx;
    };

    // First: without precomputed cost
    auto ctx1 = make_ctx();
    auto eval1 = partition_moves::eval_de_recompute(ctx1.part, 0, 0);
    CHECK("precomputed: eval feasible", eval1.feasible);

    auto affected1 = partition_moves::apply_de_recompute(ctx1.part, 0, 0);
    CHECK("precomputed: applied without precomputed", !affected1.empty());
    double cost_without = ctx1.part.groups[0].cost;
    FlatSet<size_t> ops_without = ctx1.part.groups[0].ops;

    // Second: with precomputed cost
    auto ctx2 = make_ctx();
    // Precompute the remaining cost: G0.ops - {op0} = {op1, op2}
    double precomputed = ctx2.part.eval_set({1, 2});
    auto affected2 = partition_moves::apply_de_recompute(ctx2.part, 0, 0, precomputed);
    CHECK("precomputed: applied with precomputed", !affected2.empty());
    double cost_with = ctx2.part.groups[0].cost;
    FlatSet<size_t> ops_with = ctx2.part.groups[0].ops;

    // Both should give the same cost
    CHECK_CLOSE("precomputed: same cost", cost_with, cost_without);

    // Both should have the same ops
    CHECK("precomputed: same ops", ops_with == ops_without);

    // Both affected sets should be the same
    CHECK("precomputed: same affected", affected1 == affected2);
}

// ============================================================================
// main
// ============================================================================

int main() {
    test_basic_de_recompute();
    test_de_recompute_gain_correctness();
    test_de_recompute_cyclic_rejected();
    test_de_recompute_gap_rejected();
    test_de_recompute_not_recomputed_rejected();
    test_de_recompute_ga_dies();
    test_de_recompute_with_coupling();
    test_de_recompute_source_unchanged();
    test_de_recompute_dead_group();
    test_de_recompute_precomputed();

    std::cout << "\n=== de_recompute_correctness_test: "
              << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
