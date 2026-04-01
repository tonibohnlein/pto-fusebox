// tensor_merge_correctness_test.cpp
//
// Comprehensive unit tests for TENSOR_MERGE operations covering:
//   1. Basic tensor merge (3 groups sharing a tensor)
//   2. Gain correctness (independent eval_set verification)
//   3. Cyclic merge rejected (chain through external group)
//   4. Ephemeral gap rejected (external group needs tensor)
//   5. Tensor merge with coupling (chain link transfer + invalidate)
//   6. Degenerate 2-group tensor merge (equivalent to regular merge)
//   7. Dead group in group list (returns infeasible)
//   8. Precomputed cost (apply_tensor_merge with precomputed matches without)

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
// TestContext: build a Partition from a Problem with explicit group assignments.
// ============================================================================

struct TestContext {
    Problem prob;
    DAG dag;
    std::unique_ptr<CostCache> cache;
    Partition part;

    void build_partition(const std::vector<std::set<size_t>>& group_assignments) {
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

    void build(const std::vector<std::set<size_t>>& group_assignments) {
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
    void couple(size_t ga, size_t gb, std::set<size_t> tensors) {
        cp.next_group[ga] = gb;
        cp.prev_group[gb] = ga;
        cp.retained[{ga, gb}] = std::move(tensors);
    }
};

// ============================================================================
// 1. test_basic_tensor_merge
//
// 3 groups consuming the same tensor T1.  Merge all 3.
// Verify: survivor has all ops, dead groups killed,
//         saving = sum(old) - merged, acyclic.
//
// DAG:  op0 -> T1 -> op1 -> T3
//              T1 -> op2 -> T4
// T0 is input to op0.  T2 is a separate input to op2 for variety.
// We put op0 in G0, op1 in G1, op2 in G2.
// T1 is produced by G0, consumed by G1 and G2 -- all 3 touch T1.
// ============================================================================

void test_basic_tensor_merge() {
    std::cout << "--- test_basic_tensor_merge ---\n";

    TestContext ctx;
    // T0: input, T1: shared, T2: input to op2, T3: output of op1, T4: output of op2
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},     // op0: T0 -> T1
        {OpType::Pointwise, {1}, {3}, 300},     // op1: T1 -> T3
        {OpType::Pointwise, {1, 2}, {4}, 300},  // op2: T1,T2 -> T4
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    ctx.build_partition({{0}, {1}, {2}});

    double cost_g0 = ctx.part.groups[0].cost;
    double cost_g1 = ctx.part.groups[1].cost;
    double cost_g2 = ctx.part.groups[2].cost;
    double cost_before = ctx.part.total_cost();

    // Evaluate tensor merge of all 3 groups
    std::vector<size_t> group_list = {0, 1, 2};
    auto eval = partition_moves::eval_tensor_merge(ctx.part, group_list);
    CHECK("basic_tm feasible", eval.feasible);

    double merged_cost = ctx.part.eval_set({0, 1, 2});
    double expected_saving = cost_g0 + cost_g1 + cost_g2 - merged_cost;
    CHECK_CLOSE("basic_tm saving", eval.saving, expected_saving);

    // Acyclicity check
    CHECK("basic_tm acyclic_local", ctx.part.acyclic_merge_local(group_list));

    // Apply tensor merge
    auto affected = partition_moves::apply_tensor_merge(ctx.part, group_list);
    CHECK("basic_tm applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // group_list[0] (G0) survives, G1 and G2 are killed
    CHECK("basic_tm G0 alive", ctx.part.groups[0].alive);
    CHECK("basic_tm G1 dead", !ctx.part.groups[1].alive);
    CHECK("basic_tm G2 dead", !ctx.part.groups[2].alive);

    // Survivor has all ops
    CHECK("basic_tm survivor has op0", ctx.part.groups[0].ops.count(0));
    CHECK("basic_tm survivor has op1", ctx.part.groups[0].ops.count(1));
    CHECK("basic_tm survivor has op2", ctx.part.groups[0].ops.count(2));
    CHECK("basic_tm survivor size=3", ctx.part.groups[0].ops.size() == 3);

    // Verify cost
    CHECK_CLOSE("basic_tm merged cost", ctx.part.groups[0].cost, merged_cost);

    // Verify acyclicity
    CHECK("basic_tm acyclic_after", ctx.part.is_acyclic());

    // Verify total cost change matches prediction
    double cost_after = ctx.part.total_cost();
    CHECK_CLOSE("basic_tm total_cost delta", cost_before - cost_after, expected_saving);
}

// ============================================================================
// 2. test_tensor_merge_gain_correctness
//
// Verify saving by independently computing eval_set on the merged op set.
// Uses a slightly more complex graph (4 groups, merge 3 of them).
// ============================================================================

void test_tensor_merge_gain_correctness() {
    std::cout << "--- test_tensor_merge_gain_correctness ---\n";

    TestContext ctx;
    // T0 input, T1 shared tensor, T2 input, T3-T5 outputs
    ctx.prob.tensors = {{64,64},{64,64},{64,64},{64,64},{64,64},{64,64}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 400},     // op0: T0 -> T1  (producer of T1)
        {OpType::Pointwise, {1}, {3}, 300},     // op1: T1 -> T3  (consumer of T1)
        {OpType::Pointwise, {1, 2}, {4}, 350},  // op2: T1,T2 -> T4 (consumer of T1)
        {OpType::Pointwise, {4}, {5}, 200},     // op3: T4 -> T5  (not touching T1)
    };
    ctx.prob.fast_memory_capacity = 80000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 64;
    ctx.prob.native_h = 64;

    // G0={op0}, G1={op1}, G2={op2}, G3={op3}
    ctx.build_partition({{0}, {1}, {2}, {3}});

    // Groups touching T1: G0 (produces), G1 (consumes), G2 (consumes)
    std::vector<size_t> group_list = {0, 1, 2};

    double old_sum = ctx.part.groups[0].cost + ctx.part.groups[1].cost
                   + ctx.part.groups[2].cost;

    // Independently compute merged cost
    std::set<size_t> merged_ops = {0, 1, 2};
    double independent_merged_cost = ctx.part.eval_set(merged_ops);
    double independent_saving = old_sum - independent_merged_cost;

    // eval_tensor_merge should give the same saving
    auto eval = partition_moves::eval_tensor_merge(ctx.part, group_list);
    CHECK("gain_correct feasible", eval.feasible);
    CHECK_CLOSE("gain_correct saving matches", eval.saving, independent_saving);

    // Also verify the saving is positive (merging co-consumers should help)
    CHECK("gain_correct saving >= 0", eval.saving >= 0.0);
}

// ============================================================================
// 3. test_tensor_merge_cyclic_rejected
//
// Groups form a chain: G0 -> G1 -> G2 (via tensors).
// Merging non-adjacent G0 + G2 creates a cycle through G1.
// acyclic_merge_local should reject.
// ============================================================================

void test_tensor_merge_cyclic_rejected() {
    std::cout << "--- test_tensor_merge_cyclic_rejected ---\n";

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

    CHECK("cyclic: partition acyclic", ctx.part.is_acyclic());

    // Merging G0 + G2 (non-adjacent) creates cycle through G1
    std::vector<size_t> gl_02 = {0, 2};
    CHECK("cyclic: merge G0+G2 rejected", !ctx.part.acyclic_merge_local(gl_02));

    // Merging all three is fine (no external group to route through)
    std::vector<size_t> gl_all = {0, 1, 2};
    CHECK("cyclic: merge G0+G1+G2 ok", ctx.part.acyclic_merge_local(gl_all));

    // Adjacent merges are fine
    std::vector<size_t> gl_01 = {0, 1};
    std::vector<size_t> gl_12 = {1, 2};
    CHECK("cyclic: merge G0+G1 ok", ctx.part.acyclic_merge_local(gl_01));
    CHECK("cyclic: merge G1+G2 ok", ctx.part.acyclic_merge_local(gl_12));
}

// ============================================================================
// 4. test_tensor_merge_ephemeral_gap_rejected
//
// Merging groups makes tensor T1 ephemeral (produced+consumed internally),
// but an external group G3 needs T1. eval_tensor_merge should reject.
//
// DAG:
//   op0 -> T1 -> op1, op2, op3
// G0={op0}, G1={op1}, G2={op2}, G3={op3}
// Merging G0+G1+G2: T1 is produced by op0 and consumed by op1,op2 (all in
// merged set). But op3 in G3 also consumes T1 -- ephemeral gap.
// ============================================================================

void test_tensor_merge_ephemeral_gap_rejected() {
    std::cout << "--- test_tensor_merge_ephemeral_gap_rejected ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},   // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},   // op1: T1 -> T2
        {OpType::Pointwise, {1}, {3}, 300},   // op2: T1 -> T3
        {OpType::Pointwise, {1}, {4}, 300},   // op3: T1 -> T4
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0}, G1={op1}, G2={op2}, G3={op3}
    ctx.build_partition({{0}, {1}, {2}, {3}});

    // Merge G0+G1+G2 but leave G3 out -- T1 becomes ephemeral but G3 needs it
    std::vector<size_t> group_list = {0, 1, 2};
    auto eval = partition_moves::eval_tensor_merge(ctx.part, group_list);
    CHECK("eph_gap: eval_tensor_merge rejected", !eval.feasible);

    // But merging all 4 groups should be feasible (T1 fully internal)
    std::vector<size_t> all_groups = {0, 1, 2, 3};
    auto eval_all = partition_moves::eval_tensor_merge(ctx.part, all_groups);
    CHECK("eph_gap: merge all 4 feasible", eval_all.feasible);
}

// ============================================================================
// 5. test_tensor_merge_with_coupling
//
// Survivor (G0) has incoming coupling from G_ext1.
// Dead group (G2) has outgoing coupling to G_ext2.
// After tensor merge + fixup, verify:
//   - G0's incoming coupling (from G_ext1) survives
//   - G2's outgoing coupling transfers to G0
//   - invalidate_couplings removes stale tensors that are now internal
//
// DAG:
//   op_e1 -> T1 -> op0 -> T3 -> op1
//                  T3 -> op2 -> T5 -> op_e2
//   op_e1 is in G_ext1, op_e2 is in G_ext2.
//   G0={op0}, G1={op1}, G2={op2} are merged via shared tensor T3.
//   Coupling: G_ext1 -> G0 retaining T1, G2 -> G_ext2 retaining T5.
// ============================================================================

void test_tensor_merge_with_coupling() {
    std::cout << "--- test_tensor_merge_with_coupling ---\n";

    CoupledTestContext ctx;
    // T0:input, T1:e1->op0, T2:input, T3:shared, T4:input, T5:op2->e2, T6:output
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},   // op0(e1): T0 -> T1
        {OpType::Pointwise, {1}, {3}, 300},   // op1: T1 -> T3
        {OpType::Pointwise, {3}, {4}, 300},   // op2: T3 -> T4
        {OpType::Pointwise, {3}, {5}, 300},   // op3: T3 -> T5
        {OpType::Pointwise, {5}, {6}, 300},   // op4(e2): T5 -> T6
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;
    ctx.prob.retainable_tensors = {1, 5};

    // G0={op0(e1)}, G1={op1}, G2={op2}, G3={op3}, G4={op4(e2)}
    // Groups touching T3: G1 (produces), G2 (consumes), G3 (consumes)
    ctx.build({{0}, {1}, {2}, {3}, {4}});

    // Coupling: G0 -> G1 retaining T1, G3 -> G4 retaining T5
    ctx.couple(0, 1, {1});
    ctx.couple(3, 4, {5});

    CHECK("coupling: G0->G1 exists", ctx.cp.next_group[0] == 1);
    CHECK("coupling: G1 prev=G0", ctx.cp.prev_group[1] == 0);
    CHECK("coupling: G3->G4 exists", ctx.cp.next_group[3] == 4);
    CHECK("coupling: G4 prev=G3", ctx.cp.prev_group[4] == 3);

    // Tensor merge G1, G2, G3 (all touch T3). G1 survives, G2+G3 die.
    CoupledFMMove m;
    m.type = CoupledFMMove::TENSOR_MERGE;
    m.op = 1;
    m.saving = 0;
    m.tensor_groups = {1, 2, 3};

    auto affected = apply_coupled_fm_move(ctx.cp, m);
    CHECK("coupling: tm applied", !affected.empty());

    ctx.cp.part.rebuild_index();
    ctx.cp.part.rebuild_group_dag();

    // G1 survives with all merged ops, G2 and G3 are dead
    CHECK("coupling: G1 alive", ctx.cp.part.groups[1].alive);
    CHECK("coupling: G2 dead", !ctx.cp.part.groups[2].alive);
    CHECK("coupling: G3 dead", !ctx.cp.part.groups[3].alive);

    // G1 has ops from G1, G2, G3
    CHECK("coupling: G1 has op1", ctx.cp.part.groups[1].ops.count(1));
    CHECK("coupling: G1 has op2", ctx.cp.part.groups[1].ops.count(2));
    CHECK("coupling: G1 has op3", ctx.cp.part.groups[1].ops.count(3));

    // G0->G1 incoming coupling should still exist (G1 is the survivor)
    CHECK("coupling: G0->G1 preserved", ctx.cp.next_group[0] == 1);
    CHECK("coupling: G1 prev still G0", ctx.cp.prev_group[1] == 0);

    // G3's outgoing coupling (G3->G4) should transfer to G1 (the survivor)
    CHECK("coupling: G1 next=G4 (transferred)", ctx.cp.next_group[1] == 4);
    CHECK("coupling: G4 prev=G1 (transferred)", ctx.cp.prev_group[4] == 1);

    // Old coupling edge G3->G4 should be gone
    CHECK("coupling: old G3->G4 gone",
          ctx.cp.retained.find({3, 4}) == ctx.cp.retained.end());

    // Acyclicity
    CHECK("coupling: acyclic", ctx.cp.part.is_acyclic());
}

// ============================================================================
// 6. test_tensor_merge_two_groups
//
// Degenerate case: tensor_merge with exactly 2 groups.
// Should produce the same result as a regular merge.
// ============================================================================

void test_tensor_merge_two_groups() {
    std::cout << "--- test_tensor_merge_two_groups ---\n";

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

    // Build partition twice: once for tensor_merge, once for regular merge
    ctx.build_partition({{0}, {1}});

    double cost_g0 = ctx.part.groups[0].cost;
    double cost_g1 = ctx.part.groups[1].cost;

    // Evaluate tensor merge (2-group)
    std::vector<size_t> group_list = {0, 1};
    auto tm_eval = partition_moves::eval_tensor_merge(ctx.part, group_list);

    // Evaluate regular merge
    auto reg_eval = partition_moves::eval_merge(ctx.part, 0, 1);

    CHECK("two_group: both feasible", tm_eval.feasible && reg_eval.feasible);
    CHECK_CLOSE("two_group: same saving", tm_eval.saving, reg_eval.saving);

    // Apply tensor_merge
    auto affected = partition_moves::apply_tensor_merge(ctx.part, group_list);
    CHECK("two_group: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // G0 survives, G1 dead
    CHECK("two_group: G0 alive", ctx.part.groups[0].alive);
    CHECK("two_group: G1 dead", !ctx.part.groups[1].alive);

    // G0 has both ops
    CHECK("two_group: G0 has op0", ctx.part.groups[0].ops.count(0));
    CHECK("two_group: G0 has op1", ctx.part.groups[0].ops.count(1));

    // Merged cost matches expected
    double merged_cost = cost_g0 + cost_g1 - reg_eval.saving;
    CHECK_CLOSE("two_group: merged cost", ctx.part.groups[0].cost, merged_cost);

    CHECK("two_group: acyclic", ctx.part.is_acyclic());
}

// ============================================================================
// 7. test_tensor_merge_dead_group
//
// One group in the list is dead. Verify returns infeasible.
// ============================================================================

void test_tensor_merge_dead_group() {
    std::cout << "--- test_tensor_merge_dead_group ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},   // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},   // op1: T1 -> T2
        {OpType::Pointwise, {1}, {3}, 300},   // op2: T1 -> T3
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    ctx.build_partition({{0}, {1}, {2}});

    // Kill G2
    ctx.part.groups[2].alive = false;

    // eval_tensor_merge should reject (G2 is dead)
    std::vector<size_t> group_list = {0, 1, 2};
    auto eval = partition_moves::eval_tensor_merge(ctx.part, group_list);
    CHECK("dead: eval not feasible", !eval.feasible);

    // apply_tensor_merge should also reject
    auto affected = partition_moves::apply_tensor_merge(ctx.part, group_list);
    CHECK("dead: apply returns empty", affected.empty());

    // Surviving groups should be unmodified
    CHECK("dead: G0 still alive", ctx.part.groups[0].alive);
    CHECK("dead: G1 still alive", ctx.part.groups[1].alive);
    CHECK("dead: G2 still dead", !ctx.part.groups[2].alive);

    // Also test: group_list with fewer than 2 groups
    std::vector<size_t> single = {0};
    auto eval_single = partition_moves::eval_tensor_merge(ctx.part, single);
    CHECK("dead: single group infeasible", !eval_single.feasible);

    std::vector<size_t> empty_list = {};
    auto eval_empty = partition_moves::eval_tensor_merge(ctx.part, empty_list);
    CHECK("dead: empty list infeasible", !eval_empty.feasible);
}

// ============================================================================
// 8. test_tensor_merge_precomputed
//
// Verify apply_tensor_merge with precomputed cost matches without.
// Compute cost ourselves via eval_set, pass to apply_tensor_merge, then
// verify the survivor's cost matches.
// ============================================================================

void test_tensor_merge_precomputed() {
    std::cout << "--- test_tensor_merge_precomputed ---\n";

    // Build two identical contexts to compare with/without precomputed cost
    auto make_ctx = []() {
        TestContext ctx;
        ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48}};
        ctx.prob.ops = {
            {OpType::Pointwise, {0}, {1}, 300},    // op0: T0 -> T1
            {OpType::Pointwise, {1}, {3}, 300},    // op1: T1 -> T3
            {OpType::Pointwise, {1, 2}, {4}, 300}, // op2: T1,T2 -> T4
        };
        ctx.prob.fast_memory_capacity = 50000;
        ctx.prob.slow_memory_bandwidth = 10;
        ctx.prob.native_w = 48;
        ctx.prob.native_h = 48;
        ctx.build_partition({{0}, {1}, {2}});
        return ctx;
    };

    auto ctx1 = make_ctx();
    auto ctx2 = make_ctx();

    std::vector<size_t> group_list = {0, 1, 2};

    // Precompute the merged cost
    std::set<size_t> merged_ops = {0, 1, 2};
    double precomputed_cost = ctx1.part.eval_set(merged_ops);
    CHECK("precomputed: cost is finite", precomputed_cost < 1e17);

    // Apply with precomputed cost
    auto affected1 = partition_moves::apply_tensor_merge(ctx1.part, group_list, precomputed_cost);
    CHECK("precomputed: with precomputed applied", !affected1.empty());

    ctx1.part.rebuild_index();
    ctx1.part.rebuild_group_dag();

    // Apply without precomputed cost
    auto affected2 = partition_moves::apply_tensor_merge(ctx2.part, group_list);
    CHECK("precomputed: without precomputed applied", !affected2.empty());

    ctx2.part.rebuild_index();
    ctx2.part.rebuild_group_dag();

    // Both survivors should have the same cost
    CHECK_CLOSE("precomputed: costs match",
                ctx1.part.groups[0].cost, ctx2.part.groups[0].cost);

    // Specifically, the precomputed path should store exactly what we gave it
    CHECK_CLOSE("precomputed: survivor cost = precomputed",
                ctx1.part.groups[0].cost, precomputed_cost);

    // Both should have the same ops
    CHECK("precomputed: same ops",
          ctx1.part.groups[0].ops == ctx2.part.groups[0].ops);

    // Both should be acyclic
    CHECK("precomputed: ctx1 acyclic", ctx1.part.is_acyclic());
    CHECK("precomputed: ctx2 acyclic", ctx2.part.is_acyclic());

    // Same affected set
    CHECK("precomputed: same affected", affected1 == affected2);
}

// ============================================================================
// main
// ============================================================================

int main() {
    test_basic_tensor_merge();
    test_tensor_merge_gain_correctness();
    test_tensor_merge_cyclic_rejected();
    test_tensor_merge_ephemeral_gap_rejected();
    test_tensor_merge_with_coupling();
    test_tensor_merge_two_groups();
    test_tensor_merge_dead_group();
    test_tensor_merge_precomputed();

    std::cout << "\n=== tensor_merge_correctness_test: "
              << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
