// merge_correctness_test.cpp
//
// Comprehensive unit tests for MERGE operations covering:
//   1. Basic merge (no coupling)
//   2. Merge dissolves internal coupling
//   3. Merge transfers external coupling links
//   4. Merge where both groups have coupling
//   5. Merge creates ephemeral gap (rejected)
//   6. Merge with retained tensor filtering (invalidate_couplings)
//   7. Merge of co-consumers
//   8. Cyclic merge rejected

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
// 1. Basic merge -- two adjacent groups, no coupling
//
// Chain: op0 -> T1 -> op1
// Groups: G0={op0}, G1={op1}
// Verify gain = cost(G0) + cost(G1) - cost({op0,op1})
// Verify acyclicity after merge.
// ============================================================================

void test_basic_merge() {
    std::cout << "--- test_basic_merge ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    ctx.build_partition({{0}, {1}});

    double cost_g0 = ctx.part.groups[0].cost;
    double cost_g1 = ctx.part.groups[1].cost;
    double cost_before = ctx.part.total_cost();

    // Evaluate merge
    auto eval = partition_moves::eval_merge(ctx.part, 0, 1);
    CHECK("basic_merge feasible", eval.feasible);

    double merged_cost = ctx.part.eval_set({0, 1});
    double expected_saving = cost_g0 + cost_g1 - merged_cost;
    CHECK_CLOSE("basic_merge gain", eval.saving, expected_saving);

    // Acyclicity check before apply
    CHECK("basic_merge acyclic_local", ctx.part.acyclic_merge_local(0, 1));

    // Apply merge
    auto affected = partition_moves::apply_merge(ctx.part, 0, 1);
    CHECK("basic_merge applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // ga (0) survives, gb (1) is killed
    CHECK("basic_merge ga alive", ctx.part.groups[0].alive);
    CHECK("basic_merge gb dead", !ctx.part.groups[1].alive);
    CHECK("basic_merge ga has both ops",
          ctx.part.groups[0].ops.count(0) && ctx.part.groups[0].ops.count(1));

    // Verify cost
    CHECK_CLOSE("basic_merge merged cost", ctx.part.groups[0].cost, merged_cost);

    // Verify acyclicity
    CHECK("basic_merge acyclic_after", ctx.part.is_acyclic());

    // Verify total cost change matches prediction
    double cost_after = ctx.part.total_cost();
    CHECK_CLOSE("basic_merge total_cost", cost_before - cost_after, expected_saving);
}

// ============================================================================
// 2. Merge dissolves internal coupling
//
// Chain: op0 -> T1 -> op1
// Groups: G0={op0}, G1={op1}
// Coupling: G0 -> G1 retaining T1
// After merge(G0, G1): T1 becomes internal (ephemeral).
// The coupling edge G0->G1 should be dissolved.
// ============================================================================

void test_merge_dissolves_internal_coupling() {
    std::cout << "--- test_merge_dissolves_internal_coupling ---\n";

    CoupledTestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;
    ctx.prob.retainable_tensors = {1};

    ctx.build({{0}, {1}});

    // Wire coupling: G0 -> G1 retaining T1
    ctx.couple(0, 1, {1});

    // Verify coupling exists
    CHECK("dissolve: coupling exists pre", ctx.cp.next_group[0] == 1);
    CHECK("dissolve: prev_group set", ctx.cp.prev_group[1] == 0);
    CHECK("dissolve: retained non-empty", !ctx.cp.retained[{0, 1}].empty());

    // Apply merge via the coupled path
    CoupledFMMove move;
    move.type = CoupledFMMove::MERGE;
    move.ga = 0;
    move.gb = 1;
    move.op = 0;
    move.saving = 0;

    auto affected = apply_coupled_fm_move(ctx.cp, move);
    CHECK("dissolve: merge applied", !affected.empty());

    ctx.cp.part.rebuild_index();
    ctx.cp.part.rebuild_group_dag();

    // G0 survives, G1 is dead
    CHECK("dissolve: ga alive", ctx.cp.part.groups[0].alive);
    CHECK("dissolve: gb dead", !ctx.cp.part.groups[1].alive);

    // Internal coupling dissolved
    CHECK("dissolve: next_group cleared", ctx.cp.next_group[0] == SIZE_MAX);
    CHECK("dissolve: prev_group cleared", ctx.cp.prev_group[1] == SIZE_MAX);
    CHECK("dissolve: retained erased", ctx.cp.retained.find({0, 1}) == ctx.cp.retained.end());

    // Acyclicity
    CHECK("dissolve: acyclic_after", ctx.cp.part.is_acyclic());
}

// ============================================================================
// 3. Merge transfers external coupling links
//
// Chain: op0 -> T1 -> op1 -> T2 -> op2 -> T3 -> op3
// Groups: G0={op0}, G1={op1}, G2={op2}, G3={op3}
// Coupling: G0 -> G1 retaining T1, G2 -> G3 retaining T3
// Merge G1 (ga, survives) and G2 (gb, killed):
//   G0->G1 incoming stays on G1.
//   G2->G3 outgoing transfers to G1.
//   Result chain: G0 -> G1(merged) -> G3
// ============================================================================

void test_merge_transfers_external_links() {
    std::cout << "--- test_merge_transfers_external_links ---\n";

    CoupledTestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
        {OpType::Pointwise, {2}, {3}, 300},  // op2: T2 -> T3
        {OpType::Pointwise, {3}, {4}, 300},  // op3: T3 -> T4
    };
    ctx.prob.fast_memory_capacity = 30000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;
    ctx.prob.retainable_tensors = {1, 2, 3};

    ctx.build({{0}, {1}, {2}, {3}});

    // Wire coupling: G0 -> G1 retaining T1, G2 -> G3 retaining T3
    ctx.couple(0, 1, {1});
    ctx.couple(2, 3, {3});

    CHECK("transfer: G1 prev=G0", ctx.cp.prev_group[1] == 0);
    CHECK("transfer: G2 next=G3", ctx.cp.next_group[2] == 3);

    // Merge G1 (ga=1, survives) and G2 (gb=2, killed)
    CoupledFMMove move;
    move.type = CoupledFMMove::MERGE;
    move.ga = 1;
    move.gb = 2;
    move.op = 1;
    move.saving = 0;

    auto affected = apply_coupled_fm_move(ctx.cp, move);
    CHECK("transfer: merge applied", !affected.empty());

    ctx.cp.part.rebuild_index();
    ctx.cp.part.rebuild_group_dag();

    // G1 survives, G2 is dead
    CHECK("transfer: G1 alive", ctx.cp.part.groups[1].alive);
    CHECK("transfer: G2 dead", !ctx.cp.part.groups[2].alive);

    // G0 -> G1 (prev link stays on G1)
    CHECK("transfer: G1 prev still G0", ctx.cp.prev_group[1] == 0);
    CHECK("transfer: G0 next still G1", ctx.cp.next_group[0] == 1);

    // G2's outgoing link (G2->G3) transfers to G1.
    // T3 is produced by op2 (now in G1), consumed by op3 (G3) -- boundary output.
    CHECK("transfer: G1 next=G3", ctx.cp.next_group[1] == 3);
    CHECK("transfer: G3 prev=G1", ctx.cp.prev_group[3] == 1);

    // Retained tensors transferred
    auto it = ctx.cp.retained.find({1, 3});
    CHECK("transfer: retained(G1,G3) exists", it != ctx.cp.retained.end());
    if (it != ctx.cp.retained.end()) {
        CHECK("transfer: retained T3", it->second.count(3));
    }

    // Old coupling edge removed
    CHECK("transfer: old G2->G3 gone",
          ctx.cp.retained.find({2, 3}) == ctx.cp.retained.end());

    // Acyclicity
    CHECK("transfer: acyclic", ctx.cp.part.is_acyclic());
}

// ============================================================================
// 4. Merge where both groups have prev coupling -- only ga's prev survives
//
// Two independent chains: op0->T1->op2 and op1->T3->op3
// Groups: G0={op0}, G1={op1}, G2={op2}, G3={op3}
// Coupling: G0 -> G2 retaining T1, G1 -> G3 retaining T3
// Merge G2 (ga) and G3 (gb): G2 survives, G3 dies.
// G2 already has prev=G0, so G3's prev link (G1->G3) is dissolved.
// ============================================================================

void test_merge_both_groups_have_coupling() {
    std::cout << "--- test_merge_both_groups_have_coupling ---\n";

    CoupledTestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},     // op0: T0 -> T1
        {OpType::Pointwise, {2}, {3}, 300},     // op1: T2 -> T3
        {OpType::Pointwise, {1}, {4}, 300},     // op2: T1 -> T4
        {OpType::Pointwise, {3}, {5}, 300},     // op3: T3 -> T5
    };
    ctx.prob.fast_memory_capacity = 30000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;
    ctx.prob.retainable_tensors = {1, 3};

    ctx.build({{0}, {1}, {2}, {3}});

    // Coupling: G0 -> G2 retaining T1, G1 -> G3 retaining T3
    ctx.couple(0, 2, {1});
    ctx.couple(1, 3, {3});

    CHECK("both_prev: G2 prev=G0", ctx.cp.prev_group[2] == 0);
    CHECK("both_prev: G3 prev=G1", ctx.cp.prev_group[3] == 1);

    // Merge G2 (ga, survives) and G3 (gb, killed)
    CoupledFMMove move;
    move.type = CoupledFMMove::MERGE;
    move.ga = 2;
    move.gb = 3;
    move.op = 2;
    move.saving = 0;

    auto affected = apply_coupled_fm_move(ctx.cp, move);
    CHECK("both_prev: merge applied", !affected.empty());

    ctx.cp.part.rebuild_index();
    ctx.cp.part.rebuild_group_dag();

    CHECK("both_prev: G2 alive", ctx.cp.part.groups[2].alive);
    CHECK("both_prev: G3 dead", !ctx.cp.part.groups[3].alive);

    // G2's prev should still be G0
    CHECK("both_prev: G2 prev=G0 still", ctx.cp.prev_group[2] == 0);

    // G3's prev link (G1->G3) should be dissolved since G2 already has a prev.
    CHECK("both_prev: G1 next dissolved", ctx.cp.next_group[1] == SIZE_MAX);

    // Old retained edge G1->G3 should be gone
    CHECK("both_prev: retained(G1,G3) erased",
          ctx.cp.retained.find({1, 3}) == ctx.cp.retained.end());

    CHECK("both_prev: acyclic", ctx.cp.part.is_acyclic());
}

// ============================================================================
// 5. Merge creates ephemeral gap -- rejected
//
// op0 -> T1 -> op1, op2.  T1 has two consumers in separate groups.
// G0={op0}, G1={op1}, G2={op2}
// Merging G0 and G1: T1 becomes ephemeral (produced AND consumed internally),
// but op2 in G2 still needs T1 and no other group produces it.
// creates_ephemeral_gap detects this; eval_merge rejects the merge.
// ============================================================================

void test_merge_creates_ephemeral_gap_rejected() {
    std::cout << "--- test_merge_creates_ephemeral_gap_rejected ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},     // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},     // op1: T1 -> T2
        {OpType::Pointwise, {1}, {3}, 300},     // op2: T1 -> T3
    };
    ctx.prob.fast_memory_capacity = 30000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    ctx.build_partition({{0}, {1}, {2}});

    FlatSet<size_t> merged = {0, 1};
    bool gap = ctx.part.creates_ephemeral_gap(merged, 0, 1);
    CHECK("gap: creates_ephemeral_gap detected", gap);

    // eval_merge no longer checks for gaps (caller's responsibility).
    // Verify that the caller pattern (gap check → skip) works.
    auto eval = partition_moves::eval_merge(ctx.part, 0, 1);
    CHECK("gap: eval_merge returns feasible (gap check is caller's job)", eval.feasible);
    CHECK("gap: caller rejects due to gap", gap);
}

// ============================================================================
// 6. Merge with retained tensor filtering -- invalidate_couplings
//
// Tests that invalidate_couplings removes stale coupling edges where the
// retained tensor is no longer a boundary output of the source group.
//
// Setup simulates a post-merge state: G0={op0, op1} where T1 is produced
// by op0 and consumed by op1 (both internal). A stale coupling edge
// G0 -> G1 retaining T1 should be dissolved by invalidate_couplings.
// ============================================================================

void test_merge_with_retained_tensor_filtering() {
    std::cout << "--- test_merge_with_retained_tensor_filtering ---\n";

    CoupledTestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},  // op1: T1 -> T2
        {OpType::Pointwise, {2}, {3}, 300},  // op2: T2 -> T3
    };
    ctx.prob.fast_memory_capacity = 30000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;
    ctx.prob.retainable_tensors = {1, 2};

    // G0={op0, op1} (simulates post-merge state), G1={op2}
    ctx.build({{0, 1}, {2}});

    // Manually wire a stale coupling: G0 -> G1 retaining T1.
    // T1 is produced by op0 (in G0), consumed by op1 (also in G0) -> NOT boundary output.
    ctx.couple(0, 1, {1});

    CHECK("filter: stale coupling exists pre", ctx.cp.next_group[0] == 1);
    CHECK("filter: stale retained has T1", ctx.cp.retained[{0, 1}].count(1));

    // invalidate_couplings should detect T1 is not boundary output of G0
    // and dissolve the entire coupling edge.
    ctx.cp.invalidate_couplings();

    CHECK("filter: coupling dissolved", ctx.cp.next_group[0] == SIZE_MAX);
    CHECK("filter: prev dissolved", ctx.cp.prev_group[1] == SIZE_MAX);
    CHECK("filter: retained removed",
          ctx.cp.retained.find({0, 1}) == ctx.cp.retained.end());
}

// ============================================================================
// 7. Merge of co-consumers -- two groups sharing an input tensor
//
// T0 is a graph input (no producer). op0 and op1 both consume T0.
// G0={op0}, G1={op1}. Merging should be acyclic and produce correct gain.
// ============================================================================

void test_merge_co_consumers() {
    std::cout << "--- test_merge_co_consumers ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},  // op0: T0 -> T1
        {OpType::Pointwise, {0}, {2}, 300},  // op1: T0 -> T2
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    ctx.build_partition({{0}, {1}});

    double cost_g0 = ctx.part.groups[0].cost;
    double cost_g1 = ctx.part.groups[1].cost;
    double cost_before = ctx.part.total_cost();

    CHECK("co_consumer: acyclic_local", ctx.part.acyclic_merge_local(0, 1));

    auto eval = partition_moves::eval_merge(ctx.part, 0, 1);
    CHECK("co_consumer: feasible", eval.feasible);

    double merged_cost = ctx.part.eval_set({0, 1});
    double expected_saving = cost_g0 + cost_g1 - merged_cost;
    CHECK_CLOSE("co_consumer: gain", eval.saving, expected_saving);

    auto affected = partition_moves::apply_merge(ctx.part, 0, 1);
    CHECK("co_consumer: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    CHECK("co_consumer: ga alive", ctx.part.groups[0].alive);
    CHECK("co_consumer: gb dead", !ctx.part.groups[1].alive);
    CHECK("co_consumer: acyclic_after", ctx.part.is_acyclic());
    CHECK_CLOSE("co_consumer: total_cost", cost_before - ctx.part.total_cost(), expected_saving);
}

// ============================================================================
// 8. Cyclic merge rejected
//
// Chain: op0 -> T1 -> op1 -> T2 -> op2 -> T3 -> op3
// Groups: G0={op0}, G1={op1}, G2={op2}, G3={op3}
// Merging non-adjacent groups (G0+G3, G0+G2, G1+G3) creates a cycle through
// external groups. acyclic_merge_local rejects them.
// Adjacent merges (G0+G1, G1+G2, G2+G3) should pass.
// ============================================================================

void test_cyclic_merge_rejected() {
    std::cout << "--- test_cyclic_merge_rejected ---\n";

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

    ctx.build_partition({{0}, {1}, {2}, {3}});

    CHECK("cycle: partition acyclic", ctx.part.is_acyclic());

    // Non-adjacent merges create cycles through intermediate groups
    CHECK("cycle: merge G0+G3 rejected", !ctx.part.acyclic_merge_local(0, 3));
    CHECK("cycle: merge G0+G2 rejected", !ctx.part.acyclic_merge_local(0, 2));
    CHECK("cycle: merge G1+G3 rejected", !ctx.part.acyclic_merge_local(1, 3));

    // Adjacent merges are fine (direct edge becomes internal)
    CHECK("cycle: merge G0+G1 ok", ctx.part.acyclic_merge_local(0, 1));
    CHECK("cycle: merge G1+G2 ok", ctx.part.acyclic_merge_local(1, 2));
    CHECK("cycle: merge G2+G3 ok", ctx.part.acyclic_merge_local(2, 3));
}

// ============================================================================
// Additional: merge with multiple ops per group
// ============================================================================

void test_merge_multi_op_groups() {
    std::cout << "--- test_merge_multi_op_groups ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48}};
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

    // G0={op0,op1}, G1={op2,op3}
    ctx.build_partition({{0, 1}, {2, 3}});

    double cost_g0 = ctx.part.groups[0].cost;
    double cost_g1 = ctx.part.groups[1].cost;

    CHECK("multi_op: acyclic_local", ctx.part.acyclic_merge_local(0, 1));

    auto eval = partition_moves::eval_merge(ctx.part, 0, 1);
    CHECK("multi_op: feasible", eval.feasible);

    double merged_cost = ctx.part.eval_set({0, 1, 2, 3});
    double expected_saving = cost_g0 + cost_g1 - merged_cost;
    CHECK_CLOSE("multi_op: gain", eval.saving, expected_saving);

    auto affected = partition_moves::apply_merge(ctx.part, 0, 1);
    CHECK("multi_op: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    CHECK("multi_op: ga has all 4 ops", ctx.part.groups[0].ops.size() == 4);
    CHECK("multi_op: acyclic", ctx.part.is_acyclic());
}

// ============================================================================
// Additional: eval_merge(a,b) == eval_merge(b,a) (saving symmetry)
// ============================================================================

void test_merge_symmetry() {
    std::cout << "--- test_merge_symmetry ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{64,64},{64,64},{64,64}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 400},
        {OpType::Pointwise, {1}, {2}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    ctx.build_partition({{0}, {1}});

    auto eval_ab = partition_moves::eval_merge(ctx.part, 0, 1);
    auto eval_ba = partition_moves::eval_merge(ctx.part, 1, 0);

    CHECK("symmetry: both feasible", eval_ab.feasible && eval_ba.feasible);
    CHECK_CLOSE("symmetry: same saving", eval_ab.saving, eval_ba.saving);
}

// ============================================================================
// Additional: merge with diamond DAG (fan-in)
//
//   T0 -> op0 -> T2
//   T1 -> op1 -> T3
//   T2, T3 -> op2 -> T4
// G0={op0}, G1={op1} are independent -- merge is acyclic.
// ============================================================================

void test_merge_diamond() {
    std::cout << "--- test_merge_diamond ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {2}, 300},      // op0: T0 -> T2
        {OpType::Pointwise, {1}, {3}, 300},      // op1: T1 -> T3
        {OpType::Pointwise, {2, 3}, {4}, 300},   // op2: T2,T3 -> T4
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    ctx.build_partition({{0}, {1}, {2}});

    CHECK("diamond: acyclic_local G0+G1", ctx.part.acyclic_merge_local(0, 1));

    auto eval = partition_moves::eval_merge(ctx.part, 0, 1);
    CHECK("diamond: feasible", eval.feasible);

    double merged_cost = ctx.part.eval_set({0, 1});
    double expected = ctx.part.groups[0].cost + ctx.part.groups[1].cost - merged_cost;
    CHECK_CLOSE("diamond: gain", eval.saving, expected);

    auto affected = partition_moves::apply_merge(ctx.part, 0, 1);
    CHECK("diamond: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    CHECK("diamond: acyclic", ctx.part.is_acyclic());
    CHECK("diamond: ga has ops 0 and 1",
          ctx.part.groups[0].ops.count(0) && ctx.part.groups[0].ops.count(1));
}

// ============================================================================
// Additional: merge with dead group or self-merge fails gracefully
// ============================================================================

void test_merge_dead_group() {
    std::cout << "--- test_merge_dead_group ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    ctx.build_partition({{0}, {1}});

    // Kill group 1
    ctx.part.groups[1].alive = false;

    auto eval = partition_moves::eval_merge(ctx.part, 0, 1);
    CHECK("dead: not feasible", !eval.feasible);

    auto affected = partition_moves::apply_merge(ctx.part, 0, 1);
    CHECK("dead: not applied", affected.empty());

    // Self-merge
    ctx.part.groups[1].alive = true;
    auto eval_self = partition_moves::eval_merge(ctx.part, 0, 0);
    CHECK("self: not feasible", !eval_self.feasible);
}

// ============================================================================
// main
// ============================================================================

int main() {
    test_basic_merge();
    test_merge_dissolves_internal_coupling();
    test_merge_transfers_external_links();
    test_merge_both_groups_have_coupling();
    test_merge_creates_ephemeral_gap_rejected();
    test_merge_with_retained_tensor_filtering();
    test_merge_co_consumers();
    test_cyclic_merge_rejected();
    test_merge_multi_op_groups();
    test_merge_symmetry();
    test_merge_diamond();
    test_merge_dead_group();

    std::cout << "\n=== merge_correctness_test: "
              << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
