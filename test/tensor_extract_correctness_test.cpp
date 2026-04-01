// tensor_extract_correctness_test.cpp
//
// Comprehensive unit tests for TENSOR_EXTRACT operations covering:
//   1. Basic tensor extract (consumers + producer into new group)
//   2. Gain correctness (independent eval_set verification)
//   3. Cyclic extract rejected (acyclic_extract_local)
//   4. Ephemeral gap rejected (external group needs tensor)
//   5. Source group dies (becomes empty after extract)
//   6. Multiple sources (ops from 3 groups into one new group)
//   7. Dead group in source list (returns infeasible)
//   8. Precomputed cost (apply with precomputed matches without)

#include "search/partition_moves.h"
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

// ============================================================================
// 1. test_basic_tensor_extract
//
// 2 groups sharing consumers of tensor T1. Extract the consumers + producer
// into a new group. Verify: new group created, source groups shrunk,
// saving correct, acyclic.
//
// DAG:
//   op0 -> T1 -> op2 -> T3
//   op1 -> T2 -> op3 -> T4
//   T1 also -> op3 (op3 consumes both T1 and T2)
//
// Groups: G0={op0, op2}, G1={op1, op3}
// Both groups touch T1: G0 produces it (op0) and consumes it (op2),
//                        G1 consumes it (op3).
// Extract ops: {op0, op2, op3} -- producer + all consumers of T1.
// After extract: G0 loses op0 and op2 (dies), G1 loses op3 (keeps op1),
//                new group = {op0, op2, op3}.
// ============================================================================

void test_basic_tensor_extract() {
    std::cout << "--- test_basic_tensor_extract ---\n";

    TestContext ctx;
    // T0: input to op0, T1: shared tensor, T2: op1->op3, T3: output of op2, T4: output of op3
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},     // op0: T0 -> T1
        {OpType::Pointwise, {2}, {3}, 300},     // op1: T2 -> T3  (independent chain)
        {OpType::Pointwise, {1}, {4}, 300},     // op2: T1 -> T4  (consumer of T1)
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0, op2} (producer + consumer of T1), G1={op1} (independent)
    ctx.build_partition({{0, 2}, {1}});

    double cost_g0 = ctx.part.groups[0].cost;
    double cost_g1 = ctx.part.groups[1].cost;
    double cost_before = ctx.part.total_cost();

    // Extract ops: {op0, op2} from G0 into a new group
    // (G0 has only these ops, so G0 dies; G1 is untouched)
    std::set<size_t> extract_ops = {0, 2};
    std::vector<size_t> source_groups = {0};

    auto eval = partition_moves::eval_tensor_extract(ctx.part, extract_ops, source_groups);
    CHECK("basic_te feasible", eval.feasible);

    double extract_cost = ctx.part.eval_set(extract_ops);
    // G0 remainder is empty (all ops extracted), so remainder_cost = 0
    double expected_saving = cost_g0 - extract_cost;
    CHECK_CLOSE("basic_te saving", eval.saving, expected_saving);

    // Acyclicity check
    CHECK("basic_te acyclic_extract_local", ctx.part.acyclic_extract_local(extract_ops));

    // Apply tensor extract
    auto affected = partition_moves::apply_tensor_extract(ctx.part, extract_ops, source_groups);
    CHECK("basic_te applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // G0 should die (all ops extracted), G1 untouched, new group created
    CHECK("basic_te G0 dead", !ctx.part.groups[0].alive);
    CHECK("basic_te G1 alive", ctx.part.groups[1].alive);

    // New group is the last one added
    size_t new_gi = ctx.part.groups.size() - 1;
    CHECK("basic_te new group alive", ctx.part.groups[new_gi].alive);
    CHECK("basic_te new group has op0", ctx.part.groups[new_gi].ops.count(0));
    CHECK("basic_te new group has op2", ctx.part.groups[new_gi].ops.count(2));
    CHECK("basic_te new group size=2", ctx.part.groups[new_gi].ops.size() == 2);

    // Verify new group cost
    CHECK_CLOSE("basic_te new group cost", ctx.part.groups[new_gi].cost, extract_cost);

    // Verify acyclicity
    CHECK("basic_te acyclic_after", ctx.part.is_acyclic());

    // Verify total cost change matches prediction
    double cost_after = ctx.part.total_cost();
    CHECK_CLOSE("basic_te total_cost delta", cost_before - cost_after, expected_saving);
}

// ============================================================================
// 2. test_tensor_extract_gain_correctness
//
// Verify saving = sum(old_costs) - (extract_cost + sum(remainder_costs))
// by computing each term independently via eval_set.
//
// DAG:
//   op0 -> T1 -> op1 -> T2
//   op0 -> T1 -> op2 -> T3
//   op3 -> T4 (independent op in G0 that stays)
//
// Groups: G0={op0, op3}, G1={op1}, G2={op2}
// Extract ops: {op0, op1, op2} from G0, G1, G2.
// Remainder: G0 keeps {op3}, G1 empty (dies), G2 empty (dies).
// ============================================================================

void test_tensor_extract_gain_correctness() {
    std::cout << "--- test_tensor_extract_gain_correctness ---\n";

    TestContext ctx;
    // T0: input to op0, T1: shared, T2: out op1, T3: out op2, T4: out op3, T5: input op3
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},     // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},     // op1: T1 -> T2
        {OpType::Pointwise, {1}, {3}, 300},     // op2: T1 -> T3
        {OpType::Pointwise, {5}, {4}, 300},     // op3: T5 -> T4 (independent)
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0, op3}, G1={op1}, G2={op2}
    ctx.build_partition({{0, 3}, {1}, {2}});

    double cost_g0 = ctx.part.groups[0].cost;
    double cost_g1 = ctx.part.groups[1].cost;
    double cost_g2 = ctx.part.groups[2].cost;
    double old_sum = cost_g0 + cost_g1 + cost_g2;

    std::set<size_t> extract_ops = {0, 1, 2};
    std::vector<size_t> source_groups = {0, 1, 2};

    // Independently compute each term
    double independent_extract_cost = ctx.part.eval_set(extract_ops);
    // G0 remainder: {op3}
    double independent_rem_g0 = ctx.part.eval_set({3});
    // G1 remainder: empty -> cost 0
    // G2 remainder: empty -> cost 0
    double independent_saving = old_sum - (independent_extract_cost + independent_rem_g0);

    auto eval = partition_moves::eval_tensor_extract(ctx.part, extract_ops, source_groups);
    CHECK("gain: feasible", eval.feasible);
    CHECK_CLOSE("gain: saving matches independent", eval.saving, independent_saving);
    CHECK("gain: saving >= 0", eval.saving >= -1.0);  // extraction of co-consumers should help or be neutral
}

// ============================================================================
// 3. test_tensor_extract_cyclic_rejected
//
// Create topology where extracting ops would create a cycle.
// Verify acyclic_extract_local rejects.
//
// Chain: op0 -> T1 -> op1 -> T2 -> op2 -> T3 -> op3
// Groups: G0={op0, op1}, G1={op2, op3}
//
// Extracting {op1, op2}: the new group gnew has:
//   - inputs: T1 (from G0's op0), T2 is internal
//   - outputs: T2 consumed by gnew (internal), T3 consumed by G1's op3
// So gnew depends on G0 (via T1) and G1 depends on gnew (via T3).
// But G0 also lost op1, and G0's remaining op0 produces T1 -> still fine.
// Actually that doesn't create a cycle. Let me construct a proper cycle.
//
// Diamond:
//   op0 -> T1 -> op1 -> T3
//   op0 -> T2 -> op2 -> T3 -> op3
//
// Groups: G0={op0, op2}, G1={op1, op3}
// Extracting {op2, op1}: gnew needs T1 (from G0's op0) and T2 (from G0's op0).
//   gnew produces T3 consumed by G1's op3.
//   G0 (remainder {op0}) -> gnew -> G1 (remainder {op3}).
//   No cycle here either.
//
// Proper cycle setup:
//   op0 -> T1 -> op1 -> T2 -> op2 -> T3 -> op3
//   Groups: G0={op0}, G1={op1}, G2={op2}, G3={op3}
//   Extract {op0, op2}: gnew produces T1 (consumed by G1's op1) and T3 (consumed by G3's op3).
//     gnew also consumes T2 (produced by G1's op1).
//     So: gnew -> G1 (via T1), G1 -> gnew (via T2) -- CYCLE!
// ============================================================================

void test_tensor_extract_cyclic_rejected() {
    std::cout << "--- test_tensor_extract_cyclic_rejected ---\n";

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

    // Extract {op0, op2}: gnew produces T1 (consumed by G1) and T3.
    // gnew also needs T2 as input (produced by G1's op1).
    // So: gnew -> G1 (via T1) and G1 -> gnew (via T2) -- cycle.
    std::set<size_t> extract_cyclic = {0, 2};
    CHECK("cyclic: extract {op0,op2} rejected", !ctx.part.acyclic_extract_local(extract_cyclic));

    // eval_tensor_extract should also fail or at least the acyclicity check rejects
    // (eval_tensor_extract itself does not check acyclicity -- caller does that)
    // But we can verify that the caller pattern works.

    // Extracting adjacent ops is fine: {op0, op1}
    std::set<size_t> extract_adjacent = {0, 1};
    CHECK("cyclic: extract {op0,op1} ok", ctx.part.acyclic_extract_local(extract_adjacent));

    // Extracting all ops is fine (no external group to route through)
    std::set<size_t> extract_all = {0, 1, 2};
    CHECK("cyclic: extract all ok", ctx.part.acyclic_extract_local(extract_all));
}

// ============================================================================
// 4. test_tensor_extract_ephemeral_gap_rejected
//
// Extracting ops makes a tensor ephemeral in the extract group while an
// external group still needs it. eval_tensor_extract rejects.
//
// DAG:
//   op0 -> T1 -> op1, op2, op3
// Groups: G0={op0, op1}, G1={op2}, G2={op3}
//
// Extract {op0, op1} from G0: new group has op0 (produces T1) and op1
// (consumes T1). T1 becomes ephemeral inside the new group.
// But G1's op2 and G2's op3 still need T1 -- ephemeral gap.
// ============================================================================

void test_tensor_extract_ephemeral_gap_rejected() {
    std::cout << "--- test_tensor_extract_ephemeral_gap_rejected ---\n";

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

    // G0={op0, op1}, G1={op2}, G2={op3}
    ctx.build_partition({{0, 1}, {2}, {3}});

    // Extract {op0, op1} from G0. T1 produced by op0, consumed by op1 --
    // both in extract set, so T1 becomes ephemeral. But op2 (G1) and op3 (G2)
    // still need T1 from outside. Ephemeral gap.
    std::set<size_t> extract_ops = {0, 1};
    std::vector<size_t> source_groups = {0};

    auto eval = partition_moves::eval_tensor_extract(ctx.part, extract_ops, source_groups);
    CHECK("eph_gap: eval_tensor_extract rejected", !eval.feasible);

    // But extracting just the producer is fine -- T1 stays boundary output
    std::set<size_t> extract_producer_only = {0};
    auto eval2 = partition_moves::eval_tensor_extract(ctx.part, extract_producer_only, source_groups);
    CHECK("eph_gap: extract producer only feasible", eval2.feasible);
}

// ============================================================================
// 5. test_tensor_extract_source_dies
//
// Source group has only extracted ops (becomes empty after extract).
// Verify it dies.
//
// DAG:
//   op0 -> T1 -> op1 -> T2
//   op2 -> T3 (independent)
// Groups: G0={op0}, G1={op1}, G2={op2}
//
// Extract {op0, op1} from G0 and G1. Both source groups lose all ops -> die.
// New group = {op0, op1}.
// ============================================================================

void test_tensor_extract_source_dies() {
    std::cout << "--- test_tensor_extract_source_dies ---\n";

    TestContext ctx;
    ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48}};
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},   // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},   // op1: T1 -> T2
        {OpType::Pointwise, {3}, {3}, 300},   // op2: T3 -> T3 (self-loop-ish, independent)
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0}, G1={op1}, G2={op2}
    ctx.build_partition({{0}, {1}, {2}});

    double cost_g0 = ctx.part.groups[0].cost;
    double cost_g1 = ctx.part.groups[1].cost;

    std::set<size_t> extract_ops = {0, 1};
    std::vector<size_t> source_groups = {0, 1};

    auto eval = partition_moves::eval_tensor_extract(ctx.part, extract_ops, source_groups);
    CHECK("dies: feasible", eval.feasible);

    // Both remainders are empty, so saving = old_costs - extract_cost
    double extract_cost = ctx.part.eval_set(extract_ops);
    double expected_saving = cost_g0 + cost_g1 - extract_cost;
    CHECK_CLOSE("dies: saving", eval.saving, expected_saving);

    // Apply
    auto affected = partition_moves::apply_tensor_extract(ctx.part, extract_ops, source_groups);
    CHECK("dies: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // Both source groups should be dead
    CHECK("dies: G0 dead", !ctx.part.groups[0].alive);
    CHECK("dies: G1 dead", !ctx.part.groups[1].alive);
    CHECK("dies: G2 alive", ctx.part.groups[2].alive);

    // New group should be alive
    size_t new_gi = ctx.part.groups.size() - 1;
    CHECK("dies: new group alive", ctx.part.groups[new_gi].alive);
    CHECK("dies: new group has op0", ctx.part.groups[new_gi].ops.count(0));
    CHECK("dies: new group has op1", ctx.part.groups[new_gi].ops.count(1));
    CHECK("dies: new group size=2", ctx.part.groups[new_gi].ops.size() == 2);

    // Verify cost
    CHECK_CLOSE("dies: new group cost", ctx.part.groups[new_gi].cost, extract_cost);

    // Acyclicity
    CHECK("dies: acyclic", ctx.part.is_acyclic());

    // Verify alive count: G2 + new group = 2
    CHECK("dies: num_alive=2", ctx.part.num_alive() == 2);
}

// ============================================================================
// 6. test_tensor_extract_multiple_sources
//
// Ops extracted from 3 different source groups into one new group.
//
// DAG:
//   op0 -> T1, op1 -> T2, op2 -> T3
//   op3 consumes T1, T2, T3 -> T4
//   op4 (independent, stays in G0)
//   op5 (independent, stays in G1)
//   op6 (independent, stays in G2)
//
// Groups: G0={op0, op4}, G1={op1, op5}, G2={op2, op6}, G3={op3}
// Extract {op0, op1, op2, op3} from G0, G1, G2, G3.
// Remainders: G0={op4}, G1={op5}, G2={op6}, G3=empty (dies).
// ============================================================================

void test_tensor_extract_multiple_sources() {
    std::cout << "--- test_tensor_extract_multiple_sources ---\n";

    TestContext ctx;
    // T0-T3: inputs, T1-T3: intermediate, T4: output of op3
    // T5-T10: inputs/outputs for independent ops
    ctx.prob.tensors = {
        {48,48},{48,48},{48,48},{48,48},{48,48},  // T0-T4
        {48,48},{48,48},{48,48},{48,48},{48,48},  // T5-T9
        {48,48}                                    // T10
    };
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},          // op0: T0 -> T1
        {OpType::Pointwise, {5}, {2}, 300},          // op1: T5 -> T2
        {OpType::Pointwise, {6}, {3}, 300},          // op2: T6 -> T3
        {OpType::Pointwise, {1, 2, 3}, {4}, 300},   // op3: T1,T2,T3 -> T4
        {OpType::Pointwise, {7}, {8}, 300},          // op4: T7 -> T8 (independent, G0)
        {OpType::Pointwise, {9}, {10}, 300},         // op5: T9 -> T10 (independent, G1)
    };
    ctx.prob.fast_memory_capacity = 80000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    // G0={op0, op4}, G1={op1, op5}, G2={op2}, G3={op3}
    ctx.build_partition({{0, 4}, {1, 5}, {2}, {3}});

    double cost_g0 = ctx.part.groups[0].cost;
    double cost_g1 = ctx.part.groups[1].cost;
    double cost_g2 = ctx.part.groups[2].cost;
    double cost_g3 = ctx.part.groups[3].cost;
    double old_sum = cost_g0 + cost_g1 + cost_g2 + cost_g3;

    std::set<size_t> extract_ops = {0, 1, 2, 3};
    std::vector<size_t> source_groups = {0, 1, 2, 3};

    // Compute expected costs independently
    double extract_cost = ctx.part.eval_set(extract_ops);
    double rem_g0 = ctx.part.eval_set({4});
    double rem_g1 = ctx.part.eval_set({5});
    // G2 remainder: empty (dies)
    // G3 remainder: empty (dies)
    double expected_saving = old_sum - (extract_cost + rem_g0 + rem_g1);

    auto eval = partition_moves::eval_tensor_extract(ctx.part, extract_ops, source_groups);
    CHECK("multi: feasible", eval.feasible);
    CHECK_CLOSE("multi: saving", eval.saving, expected_saving);

    // Apply
    auto affected = partition_moves::apply_tensor_extract(ctx.part, extract_ops, source_groups);
    CHECK("multi: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // G0 and G1 survive with remainders, G2 and G3 die
    CHECK("multi: G0 alive", ctx.part.groups[0].alive);
    CHECK("multi: G1 alive", ctx.part.groups[1].alive);
    CHECK("multi: G2 dead", !ctx.part.groups[2].alive);
    CHECK("multi: G3 dead", !ctx.part.groups[3].alive);

    // Verify remainders
    CHECK("multi: G0 has op4", ctx.part.groups[0].ops.count(4));
    CHECK("multi: G0 size=1", ctx.part.groups[0].ops.size() == 1);
    CHECK("multi: G1 has op5", ctx.part.groups[1].ops.count(5));
    CHECK("multi: G1 size=1", ctx.part.groups[1].ops.size() == 1);

    // New group
    size_t new_gi = ctx.part.groups.size() - 1;
    CHECK("multi: new group alive", ctx.part.groups[new_gi].alive);
    CHECK("multi: new group size=4", ctx.part.groups[new_gi].ops.size() == 4);
    CHECK("multi: new group has op0", ctx.part.groups[new_gi].ops.count(0));
    CHECK("multi: new group has op1", ctx.part.groups[new_gi].ops.count(1));
    CHECK("multi: new group has op2", ctx.part.groups[new_gi].ops.count(2));
    CHECK("multi: new group has op3", ctx.part.groups[new_gi].ops.count(3));

    // Verify costs
    CHECK_CLOSE("multi: new group cost", ctx.part.groups[new_gi].cost, extract_cost);
    CHECK_CLOSE("multi: G0 remainder cost", ctx.part.groups[0].cost, rem_g0);
    CHECK_CLOSE("multi: G1 remainder cost", ctx.part.groups[1].cost, rem_g1);

    // Acyclicity
    CHECK("multi: acyclic", ctx.part.is_acyclic());
}

// ============================================================================
// 7. test_tensor_extract_dead_group
//
// Source group is dead. Verify returns infeasible.
// ============================================================================

void test_tensor_extract_dead_group() {
    std::cout << "--- test_tensor_extract_dead_group ---\n";

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

    // G0={op0}, G1={op1}, G2={op2}
    ctx.build_partition({{0}, {1}, {2}});

    // Kill G2
    ctx.part.groups[2].alive = false;

    // Extract with dead group in source list
    std::set<size_t> extract_ops = {0, 1, 2};
    std::vector<size_t> source_groups = {0, 1, 2};

    auto eval = partition_moves::eval_tensor_extract(ctx.part, extract_ops, source_groups);
    CHECK("dead: eval not feasible", !eval.feasible);

    // apply should also reject
    auto affected = partition_moves::apply_tensor_extract(ctx.part, extract_ops, source_groups);
    CHECK("dead: apply returns empty", affected.empty());

    // Surviving groups should be unmodified
    CHECK("dead: G0 still alive", ctx.part.groups[0].alive);
    CHECK("dead: G1 still alive", ctx.part.groups[1].alive);
    CHECK("dead: G2 still dead", !ctx.part.groups[2].alive);

    // Test edge cases: empty extract_ops
    std::set<size_t> empty_ops;
    std::vector<size_t> sg = {0};
    auto eval_empty_ops = partition_moves::eval_tensor_extract(ctx.part, empty_ops, sg);
    CHECK("dead: empty ops infeasible", !eval_empty_ops.feasible);

    // Empty source_groups
    std::set<size_t> some_ops = {0};
    std::vector<size_t> empty_sg;
    auto eval_empty_sg = partition_moves::eval_tensor_extract(ctx.part, some_ops, empty_sg);
    CHECK("dead: empty sources infeasible", !eval_empty_sg.feasible);
}

// ============================================================================
// 8. test_tensor_extract_precomputed
//
// Verify apply with precomputed extract_cost matches without.
// ============================================================================

void test_tensor_extract_precomputed() {
    std::cout << "--- test_tensor_extract_precomputed ---\n";

    auto make_ctx = []() {
        TestContext ctx;
        ctx.prob.tensors = {{48,48},{48,48},{48,48},{48,48},{48,48},{48,48}};
        ctx.prob.ops = {
            {OpType::Pointwise, {0}, {1}, 300},     // op0: T0 -> T1
            {OpType::Pointwise, {1}, {2}, 300},     // op1: T1 -> T2
            {OpType::Pointwise, {1}, {3}, 300},     // op2: T1 -> T3
            {OpType::Pointwise, {5}, {4}, 300},     // op3: T5 -> T4 (independent, stays in G0)
        };
        ctx.prob.fast_memory_capacity = 50000;
        ctx.prob.slow_memory_bandwidth = 10;
        ctx.prob.native_w = 48;
        ctx.prob.native_h = 48;
        // G0={op0, op3}, G1={op1}, G2={op2}
        ctx.build_partition({{0, 3}, {1}, {2}});
        return ctx;
    };

    auto ctx1 = make_ctx();
    auto ctx2 = make_ctx();

    std::set<size_t> extract_ops = {0, 1, 2};
    std::vector<size_t> source_groups = {0, 1, 2};

    // Precompute the extract cost
    double precomputed_cost = ctx1.part.eval_set(extract_ops);
    CHECK("precomputed: cost is finite", precomputed_cost < 1e17);

    // Apply with precomputed cost
    auto affected1 = partition_moves::apply_tensor_extract(
        ctx1.part, extract_ops, source_groups, precomputed_cost);
    CHECK("precomputed: with precomputed applied", !affected1.empty());

    ctx1.part.rebuild_index();
    ctx1.part.rebuild_group_dag();

    // Apply without precomputed cost
    auto affected2 = partition_moves::apply_tensor_extract(
        ctx2.part, extract_ops, source_groups);
    CHECK("precomputed: without precomputed applied", !affected2.empty());

    ctx2.part.rebuild_index();
    ctx2.part.rebuild_group_dag();

    // Both should have same number of groups
    CHECK("precomputed: same group count",
          ctx1.part.groups.size() == ctx2.part.groups.size());

    // Find the new groups (last added)
    size_t new_gi1 = ctx1.part.groups.size() - 1;
    size_t new_gi2 = ctx2.part.groups.size() - 1;

    // Both new groups should have the same cost
    CHECK_CLOSE("precomputed: new group costs match",
                ctx1.part.groups[new_gi1].cost, ctx2.part.groups[new_gi2].cost);

    // Specifically, the precomputed path should store exactly what we gave it
    CHECK_CLOSE("precomputed: new group cost = precomputed",
                ctx1.part.groups[new_gi1].cost, precomputed_cost);

    // Both should have the same ops in the new group
    CHECK("precomputed: same ops",
          ctx1.part.groups[new_gi1].ops == ctx2.part.groups[new_gi2].ops);

    // Both should have same remainder in G0
    CHECK("precomputed: G0 same ops",
          ctx1.part.groups[0].ops == ctx2.part.groups[0].ops);
    CHECK_CLOSE("precomputed: G0 same cost",
                ctx1.part.groups[0].cost, ctx2.part.groups[0].cost);

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
    test_basic_tensor_extract();
    test_tensor_extract_gain_correctness();
    test_tensor_extract_cyclic_rejected();
    test_tensor_extract_ephemeral_gap_rejected();
    test_tensor_extract_source_dies();
    test_tensor_extract_multiple_sources();
    test_tensor_extract_dead_group();
    test_tensor_extract_precomputed();

    std::cout << "\n=== tensor_extract_correctness_test: "
              << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
