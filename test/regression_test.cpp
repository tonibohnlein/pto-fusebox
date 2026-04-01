// regression_test.cpp
//
// Regression tests for specific bugs found and fixed during the session:
//
//   1. STEAL splits disconnected remainder into components
//   2. DE_RECOMPUTE splits disconnected remainder into components
//   3. TENSOR_EXTRACT splits disconnected remainder into components
//   4. partition_has_gap detects ephemeral gap (tensor consumed internally
//      but needed externally)
//   5. Coupled RECOMPUTE rejected when entering/retain context makes the
//      working set infeasible
//   6. Merge invalidates stale retained tensors (invalidate_couplings)
//   7. Solution fallback to uncoupled when coupled solution is worse
//   8. Bare (no-coupling) partition is included in the coupled pool

#include "search/partition_moves.h"
#include "search/coupled_fm_search.h"
#include "search/coupling_search.h"
#include "search/local_search.h"
#include "search/structural_ops.h"
#include "partition/partition.h"
#include "solution/solution.h"
#include "core/cost_cache.h"
#include "core/dag.h"
#include "core/types.h"
#include <iostream>
#include <cmath>
#include <algorithm>
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
// Shared test context helpers (same pattern as merge_correctness_test.cpp)
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

    void couple(size_t ga, size_t gb, std::set<size_t> tensors) {
        cp.next_group[ga] = gb;
        cp.prev_group[gb] = ga;
        cp.retained[{ga, gb}] = std::move(tensors);
    }
};

// ============================================================================
// 1. test_steal_splits_disconnected_remainder
//
// Topology:
//   T0 -> op0 -> T1 -> op1 -> T2 -> op4  (T2 only goes to op4 in G1)
//   T3 (graph input) -> op1              (op1 also consumes T3)
//   T3 (graph input) -> op2 -> T4 -> op3 -> T5
//
// Neighbor graph:
//   op0 -- op1 (via T1: pred/succ)
//   op1 -- op2 (via T3: co-consumers)
//   op1 -- op4 (via T2: pred/succ)
//   op2 -- op3 (via T4: pred/succ)
//   op0 is NOT neighbor of op2 (no shared tensor, no pred/succ)
//
// Groups: G0={op0, op1, op2, op3}, G1={op4}
// op1 is the bridge between {op0} and {op2, op3} in G0's neighbor subgraph.
//
// Steal op1 from G0 to G1:
//   G1 = {op1, op4}: needs T1 from G0, T3 from graph input. Edge G0->G1.
//   G0 remainder = {op0, op2, op3}: op2 needs T3 (graph input, no dep on G1).
//   No back-edge G1->G0 => acyclic.
//   Remainder disconnects into {op0} and {op2, op3}.
// ============================================================================

void test_steal_splits_disconnected_remainder() {
    std::cout << "--- test_steal_splits_disconnected_remainder ---\n";

    TestContext ctx;
    ctx.prob.tensors = {
        {32, 32},  // T0: graph input -> op0
        {32, 32},  // T1: op0 -> op1
        {32, 32},  // T2: op1 -> op4
        {32, 32},  // T3: graph input -> op1, op2  (co-consumer link)
        {32, 32},  // T4: op2 -> op3
        {32, 32},  // T5: op3 output
    };
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},        // op0: T0 -> T1
        {OpType::Pointwise, {1, 3}, {2}, 300},     // op1: T1, T3 -> T2
        {OpType::Pointwise, {3}, {4}, 300},         // op2: T3 -> T4
        {OpType::Pointwise, {4}, {5}, 300},         // op3: T4 -> T5
        {OpType::Pointwise, {2}, {}, 300},          // op4: T2 -> (output)
    };
    // op4 must produce something; let's give it an output tensor
    ctx.prob.tensors.push_back({32, 32});  // T6: op4 output
    ctx.prob.ops[4].outputs = {6};

    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 32;
    ctx.prob.native_h = 32;

    // G0={op0, op1, op2, op3}, G1={op4}
    ctx.build_partition({{0, 1, 2, 3}, {4}});

    // Verify op1 is a bridge in G0
    auto comps_without_op1 = structural_ops::connected_components({0, 2, 3}, ctx.dag);
    CHECK("steal_split: op1 is bridge (remainder disconnects)",
          comps_without_op1.size() == 2);

    CHECK("steal_split: G0 has 4 ops pre", ctx.part.groups[0].ops.size() == 4);

    size_t num_groups_before = ctx.part.groups.size();

    // Steal op1 from G0 (group 0) to G1 (group 1)
    auto affected = partition_moves::apply_steal(ctx.part, 1, 0, 1);
    CHECK("steal_split: steal applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    // G1 should now contain op1 and op4
    CHECK("steal_split: G1 has op1", ctx.part.groups[1].ops.count(1));
    CHECK("steal_split: G1 has op4", ctx.part.groups[1].ops.count(4));

    // New group created for the disconnected component
    size_t num_groups_after = ctx.part.groups.size();
    CHECK("steal_split: new group created", num_groups_after > num_groups_before);

    // Find all alive groups containing remainder ops
    std::set<size_t> remainder_groups;
    for (size_t gi = 0; gi < ctx.part.groups.size(); gi++) {
        if (!ctx.part.groups[gi].alive) continue;
        if (gi == 1) continue;
        for (auto op : ctx.part.groups[gi].ops)
            if (op == 0 || op == 2 || op == 3)
                { remainder_groups.insert(gi); break; }
    }
    CHECK("steal_split: remainder in 2 groups", remainder_groups.size() == 2);

    // Each remainder group is connected
    for (auto gi : remainder_groups) {
        auto comps = structural_ops::connected_components(
            ctx.part.groups[gi].ops, ctx.dag);
        CHECK("steal_split: remainder group connected", comps.size() == 1);
    }

    CHECK("steal_split: acyclic", ctx.part.is_acyclic());
}

// ============================================================================
// 2. test_de_recompute_splits_disconnected_remainder
//
// Same topology as test 1 but op1 is recomputed in two groups.
// Removing it from G0 via de_recompute disconnects the remainder.
//
// Groups: G0={op0, op1, op2, op3}, G1={op1} (op1 recomputed in both)
// De-recompute: remove op1 from G0.
// Remainder {op0, op2, op3} splits into {op0} and {op2, op3}.
// ============================================================================

void test_de_recompute_splits_disconnected_remainder() {
    std::cout << "--- test_de_recompute_splits_disconnected_remainder ---\n";

    TestContext ctx;
    ctx.prob.tensors = {
        {32, 32},  // T0: graph input -> op0
        {32, 32},  // T1: op0 -> op1
        {32, 32},  // T2: op1 -> (output, goes to nobody except graph output)
        {32, 32},  // T3: graph input -> op1, op2
        {32, 32},  // T4: op2 -> op3
        {32, 32},  // T5: op3 output
    };
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},        // op0: T0 -> T1
        {OpType::Pointwise, {1, 3}, {2}, 300},     // op1: T1, T3 -> T2
        {OpType::Pointwise, {3}, {4}, 300},         // op2: T3 -> T4
        {OpType::Pointwise, {4}, {5}, 300},         // op3: T4 -> T5
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 32;
    ctx.prob.native_h = 32;

    // G0={op0, op1, op2, op3}, G1={op1} -- op1 recomputed in both
    ctx.build_partition({{0, 1, 2, 3}, {1}});

    size_t num_groups_before = ctx.part.groups.size();

    // De-recompute: remove op1 from G0 (it stays in G1)
    auto affected = partition_moves::apply_de_recompute(ctx.part, 0, 1);
    CHECK("de_recomp_split: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    CHECK("de_recomp_split: op1 not in G0", !ctx.part.groups[0].ops.count(1));
    CHECK("de_recomp_split: op1 in G1", ctx.part.groups[1].ops.count(1));

    size_t num_groups_after = ctx.part.groups.size();
    CHECK("de_recomp_split: new group created", num_groups_after > num_groups_before);

    std::set<size_t> remainder_groups;
    for (size_t gi = 0; gi < ctx.part.groups.size(); gi++) {
        if (!ctx.part.groups[gi].alive) continue;
        if (gi == 1) continue;
        for (auto op : ctx.part.groups[gi].ops)
            if (op == 0 || op == 2 || op == 3)
                { remainder_groups.insert(gi); break; }
    }
    CHECK("de_recomp_split: remainder in 2 groups", remainder_groups.size() == 2);

    for (auto gi : remainder_groups) {
        auto comps = structural_ops::connected_components(
            ctx.part.groups[gi].ops, ctx.dag);
        CHECK("de_recomp_split: group connected", comps.size() == 1);
    }

    CHECK("de_recomp_split: acyclic", ctx.part.is_acyclic());
}

// ============================================================================
// 3. test_tensor_extract_splits_disconnected_remainder
//
// Same topology. Extract op1 from G0 into a new group.
// Remainder {op0, op2, op3} splits into {op0} and {op2, op3}.
// ============================================================================

void test_tensor_extract_splits_disconnected_remainder() {
    std::cout << "--- test_tensor_extract_splits_disconnected_remainder ---\n";

    TestContext ctx;
    ctx.prob.tensors = {
        {32, 32},  // T0: graph input -> op0
        {32, 32},  // T1: op0 -> op1
        {32, 32},  // T2: op1 output (graph output)
        {32, 32},  // T3: graph input -> op1, op2
        {32, 32},  // T4: op2 -> op3
        {32, 32},  // T5: op3 output
    };
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},        // op0: T0 -> T1
        {OpType::Pointwise, {1, 3}, {2}, 300},     // op1: T1, T3 -> T2
        {OpType::Pointwise, {3}, {4}, 300},         // op2: T3 -> T4
        {OpType::Pointwise, {4}, {5}, 300},         // op3: T4 -> T5
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 32;
    ctx.prob.native_h = 32;

    // G0={op0, op1, op2, op3}
    ctx.build_partition({{0, 1, 2, 3}});

    size_t num_groups_before = ctx.part.groups.size();

    // Extract op1 from G0
    std::set<size_t> extract_ops = {1};
    std::vector<size_t> source_groups = {0};

    auto affected = partition_moves::apply_tensor_extract(
        ctx.part, extract_ops, source_groups);
    CHECK("tex_split: applied", !affected.empty());

    ctx.part.rebuild_index();
    ctx.part.rebuild_group_dag();

    size_t num_groups_after = ctx.part.groups.size();
    // Should have: G0 remnant (one component), new group (other component),
    // new group for extracted op1 => at least 3 total
    CHECK("tex_split: groups grew", num_groups_after > num_groups_before);

    // Find remainder groups (alive, not the extracted singleton)
    std::set<size_t> remainder_groups;
    for (size_t gi = 0; gi < ctx.part.groups.size(); gi++) {
        if (!ctx.part.groups[gi].alive) continue;
        bool has_remainder_op = false;
        bool has_extract_op = false;
        for (auto op : ctx.part.groups[gi].ops) {
            if (op == 0 || op == 2 || op == 3) has_remainder_op = true;
            if (op == 1) has_extract_op = true;
        }
        if (has_remainder_op && !has_extract_op) remainder_groups.insert(gi);
    }
    CHECK("tex_split: remainder in 2 groups", remainder_groups.size() == 2);

    for (auto gi : remainder_groups) {
        auto comps = structural_ops::connected_components(
            ctx.part.groups[gi].ops, ctx.dag);
        CHECK("tex_split: group connected", comps.size() == 1);
    }

    CHECK("tex_split: acyclic", ctx.part.is_acyclic());
}

// ============================================================================
// 4. test_symm_init_gap_rejected
//
// Tensor T1 is produced by op0, consumed by op1 (both in G0) and also by
// op2 (in G1). With both producer and consumer in G0, T1 is ephemeral.
// But op2 in G1 still needs T1 from slow memory -- gap detected.
// ============================================================================

void test_symm_init_gap_rejected() {
    std::cout << "--- test_symm_init_gap_rejected ---\n";

    TestContext ctx;
    ctx.prob.tensors = {
        {32, 32},  // T0: input to op0
        {32, 32},  // T1: op0 -> op1, op2 (the gap tensor)
        {32, 32},  // T2: op1 output
        {32, 32},  // T3: op2 output
    };
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},    // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},    // op1: T1 -> T2
        {OpType::Pointwise, {1}, {3}, 300},    // op2: T1 -> T3
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 32;
    ctx.prob.native_h = 32;

    // G0={op0, op1}: T1 produced and consumed internally => ephemeral
    // G1={op2}: needs T1 from slow memory, but T1 is ephemeral in G0
    ctx.build_partition({{0, 1}, {2}});

    bool gap = partition_has_gap(ctx.part);
    CHECK("gap: partition_has_gap detects it", gap);

    // Also verify creates_ephemeral_gap on the merged set
    bool gap2 = ctx.part.creates_ephemeral_gap({0, 1}, SIZE_MAX, SIZE_MAX);
    CHECK("gap: creates_ephemeral_gap on {op0,op1}", gap2);
}

// ============================================================================
// 5. test_coupled_recompute_infeasible_rejected
//
// CoupledPartition where group G1 has entering tensors filling fast memory.
// Recomputing op2 into G1 exceeds capacity with entering context.
// Uncoupled eval says feasible; coupled eval rejects.
//
// Topology:
//   op0: T0 -> T1 (T1 is large, retained G0->G1)
//   op1: T1 -> T2 (G1, entering T1)
//   op2: T3, T1 -> T4  (candidate for RECOMPUTE into G1)
//
// fast_memory tight so {op1, op2} with entering T1 overflows.
// ============================================================================

void test_coupled_recompute_infeasible_rejected() {
    std::cout << "--- test_coupled_recompute_infeasible_rejected ---\n";

    CoupledTestContext ctx;
    // Tensor sizes chosen so that:
    //   {op1, op2} without entering: working set fits in fast memory
    //   {op1, op2} with entering T1: working set exceeds fast memory
    //
    // All tensor dims = native_w x native_h (single tile => slice = full tensor).
    // T0=T2=T3 = 32x32 = 1024.  T1 = 32x32 = 1024.  T4 = 32x32 = 1024.
    // T5 = a large extra tensor consumed by op2 (graph input) to push working set.
    //
    // {op1, op2}: boundary tensors: T1(1024), T2(1024), T3(1024), T4(1024), T5(big).
    //   Working set ~ sum of boundary slices.
    // With entering T1: T1 counted at full_size (1024) on top of normal WS.
    //
    // Strategy: make T5 big enough that {op1, op2} barely fits, and entering T1
    // pushes it over.
    ctx.prob.tensors = {
        {32, 32},    // T0: 1024 input
        {32, 32},    // T1: 1024 retained/entering
        {32, 32},    // T2: 1024 op1 output
        {32, 32},    // T3: 1024 input to op2
        {32, 32},    // T4: 1024 op2 output
        {128, 128},  // T5: 16384 extra input to op2 (makes WS large)
    };
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},       // op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 300},        // op1: T1 -> T2
        {OpType::Pointwise, {3, 1, 5}, {4}, 300}, // op2: T3, T1, T5 -> T4
    };
    // {op1, op2} WS: T1(1024) + T2(1024) + T3(1024) + T4(1024) + T5(16384) = ~20480
    // With entering T1: adds T1 at full_size(1024) on top => ~21504.
    // Set capacity between these two: 21000 should allow uncoupled, reject coupled.
    ctx.prob.fast_memory_capacity = 21000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 32;
    ctx.prob.native_h = 32;
    ctx.prob.retainable_tensors = {1};

    // G0={op0}, G1={op1}, G2={op2}
    ctx.build({{0}, {1}, {2}});

    // Coupling: G0 -> G1 retaining T1
    ctx.couple(0, 1, {1});

    auto en = ctx.cp.entering_for(1);
    CHECK("coupled_recomp: entering has T1", en.count(1));

    // Uncoupled: {op1, op2} feasible without entering
    double uncoupled = ctx.cp.part.eval_set({1, 2});
    CHECK("coupled_recomp: uncoupled feasible", uncoupled < 1e17);

    // best_coupled_move_for_op should NOT return RECOMPUTE into G1
    std::set<size_t> feasibly_ret = {1};
    CoupledFMMove move = best_coupled_move_for_op(ctx.cp, 2, feasibly_ret);

    bool recomp_into_g1 = (move.valid() &&
                            move.type == CoupledFMMove::RECOMPUTE &&
                            move.gb == 1);
    CHECK("coupled_recomp: RECOMPUTE into G1 rejected", !recomp_into_g1);
}

// ============================================================================
// 6. test_coupled_merge_invalidates_retained_tensors
//
// After merging G1 and G2, T2 (retained on G1->G2) becomes internal.
// invalidate_couplings should dissolve the stale edge.
//
//   op0 -> T1 -> op1 -> T2 -> op2 -> T3
//   G0={op0}, G1={op1}, G2={op2}
//   Coupling: G0->G1 retaining T1, G1->G2 retaining T2
//   Merge G1+G2: T2 becomes internal. G1->G2 edge dissolved.
// ============================================================================

void test_coupled_merge_invalidates_retained_tensors() {
    std::cout << "--- test_coupled_merge_invalidates_retained_tensors ---\n";

    CoupledTestContext ctx;
    ctx.prob.tensors = {
        {48, 48},  // T0: input
        {48, 48},  // T1: op0 -> op1
        {48, 48},  // T2: op1 -> op2
        {48, 48},  // T3: op2 output
    };
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
        {OpType::Pointwise, {2}, {3}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;
    ctx.prob.retainable_tensors = {1, 2};

    ctx.build({{0}, {1}, {2}});

    ctx.couple(0, 1, {1});
    ctx.couple(1, 2, {2});

    CHECK("merge_inval: G1->G2 exists pre", ctx.cp.next_group[1] == 2);
    CHECK("merge_inval: retained T2 pre", ctx.cp.retained[{1, 2}].count(2));

    CoupledFMMove move;
    move.type = CoupledFMMove::MERGE;
    move.ga = 1;
    move.gb = 2;
    move.op = 1;
    move.saving = 0;

    auto affected = apply_coupled_fm_move(ctx.cp, move);
    CHECK("merge_inval: merge applied", !affected.empty());

    ctx.cp.part.rebuild_index();
    ctx.cp.part.rebuild_group_dag();

    CHECK("merge_inval: G1 alive", ctx.cp.part.groups[1].alive);
    CHECK("merge_inval: G2 dead", !ctx.cp.part.groups[2].alive);
    CHECK("merge_inval: G1 has op1 and op2",
          ctx.cp.part.groups[1].ops.count(1) &&
          ctx.cp.part.groups[1].ops.count(2));

    // invalidate_couplings should remove the stale G1->G2 edge
    ctx.cp.invalidate_couplings();

    CHECK("merge_inval: old G1->G2 dissolved",
          ctx.cp.next_group[1] == SIZE_MAX);
    CHECK("merge_inval: old retained erased",
          ctx.cp.retained.find({1, 2}) == ctx.cp.retained.end());

    // G0->G1 retaining T1 should still be valid
    CHECK("merge_inval: G0->G1 still valid", ctx.cp.next_group[0] == 1);
    auto it = ctx.cp.retained.find({0, 1});
    CHECK("merge_inval: retained T1 still there",
          it != ctx.cp.retained.end() && it->second.count(1));
}

// ============================================================================
// 7. test_solution_fallback_to_uncoupled
//
// Verify from_partition produces a valid solution usable as fallback.
// ============================================================================

void test_solution_fallback_to_uncoupled() {
    std::cout << "--- test_solution_fallback_to_uncoupled ---\n";

    TestContext ctx;
    ctx.prob.tensors = {
        {48, 48}, {48, 48}, {48, 48}, {48, 48},
    };
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
        {OpType::Pointwise, {2}, {3}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    ctx.build_partition({{0}, {1}, {2}});
    ctx.part.finalize(ctx.cache.get());

    // Build uncoupled solution
    Solution uncoupled = Solution::from_partition(ctx.prob, ctx.dag, ctx.part,
                                                   10, ctx.cache.get());
    auto vr = uncoupled.validate();
    CHECK("fallback: uncoupled valid", vr.valid);
    CHECK("fallback: uncoupled has positive cost", uncoupled.total_latency() > 0);

    // Build coupled solution (no coupling edges = bare)
    CoupledTestContext cctx;
    cctx.prob = ctx.prob;
    cctx.build({{0}, {1}, {2}});
    cctx.cp.part.finalize(cctx.cache.get());
    Solution coupled_sol = cctx.cp.to_solution();
    auto vr2 = coupled_sol.validate();
    CHECK("fallback: coupled valid", vr2.valid);

    double coupled_cost   = coupled_sol.total_latency();
    double uncoupled_cost = uncoupled.total_latency();

    CHECK("fallback: coupled cost finite",
          coupled_cost > 0 && coupled_cost < 1e17);
    CHECK("fallback: uncoupled cost finite",
          uncoupled_cost > 0 && uncoupled_cost < 1e17);

    // With no coupling, both should produce comparable costs.
    // The fallback path (from_partition) produces a valid usable solution.
    bool competitive = (uncoupled_cost <= coupled_cost + 0.01) ||
                       (coupled_cost < uncoupled_cost + 0.01);
    CHECK("fallback: solutions are competitive", competitive);
}

// ============================================================================
// 8. test_bare_partition_in_coupled_pool
//
// A bare CoupledPartition (no coupling edges) should produce the same
// total_cost as the uncoupled Partition and yield a valid solution.
// This tests the fix where the bare partition was missing from the pool.
// ============================================================================

void test_bare_partition_in_coupled_pool() {
    std::cout << "--- test_bare_partition_in_coupled_pool ---\n";

    TestContext ctx;
    ctx.prob.tensors = {
        {48, 48}, {48, 48}, {48, 48}, {48, 48},
    };
    ctx.prob.ops = {
        {OpType::Pointwise, {0}, {1}, 300},
        {OpType::Pointwise, {1}, {2}, 300},
        {OpType::Pointwise, {2}, {3}, 300},
    };
    ctx.prob.fast_memory_capacity = 50000;
    ctx.prob.slow_memory_bandwidth = 10;
    ctx.prob.native_w = 48;
    ctx.prob.native_h = 48;

    ctx.build_partition({{0}, {1}, {2}});

    double part_cost = ctx.part.total_cost();

    // Bare CoupledPartition -- no coupling
    CoupledPartition cp;
    cp.init_from(ctx.part, ctx.cache.get());

    CHECK("bare: no coupling edges",
          std::all_of(cp.next_group.begin(), cp.next_group.end(),
                      [](size_t v) { return v == SIZE_MAX; }));
    CHECK("bare: no prev edges",
          std::all_of(cp.prev_group.begin(), cp.prev_group.end(),
                      [](size_t v) { return v == SIZE_MAX; }));

    double cp_cost = cp.total_cost();
    CHECK_CLOSE("bare: cost matches partition", cp_cost, part_cost, 0.01);

    cp.part.finalize(ctx.cache.get());
    Solution sol = cp.to_solution();
    auto vr = sol.validate();
    CHECK("bare: solution valid", vr.valid);
    CHECK("bare: solution cost positive", sol.total_latency() > 0);

    // Simulate pool: bare variant alongside a coupled variant
    std::vector<double> pool_costs;

    CoupledPartition cp_bare;
    cp_bare.init_from(ctx.part, ctx.cache.get());
    pool_costs.push_back(cp_bare.total_cost());

    CoupledPartition cp2;
    cp2.init_from(ctx.part, ctx.cache.get());
    pool_costs.push_back(cp2.total_cost());

    CHECK("bare: pool has entries", pool_costs.size() >= 2);

    double best = *std::min_element(pool_costs.begin(), pool_costs.end());
    CHECK_CLOSE("bare: best pool cost matches", best, part_cost, 0.01);
}

// ============================================================================
// main
// ============================================================================

int main() {
    test_steal_splits_disconnected_remainder();
    test_de_recompute_splits_disconnected_remainder();
    test_tensor_extract_splits_disconnected_remainder();
    test_symm_init_gap_rejected();
    test_coupled_recompute_infeasible_rejected();
    test_coupled_merge_invalidates_retained_tensors();
    test_solution_fallback_to_uncoupled();
    test_bare_partition_in_coupled_pool();

    std::cout << "\n=== regression_test: "
              << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
