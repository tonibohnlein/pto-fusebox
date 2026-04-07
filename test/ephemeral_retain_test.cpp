// ephemeral_retain_test.cpp
// Tests for retained ephemeral tensors across subgraph boundaries.
//
// Competition ruling: ephemeral tensors (produced+consumed within a subgraph)
// CAN be retained in fast memory for the next subgraph.  When retained, the
// full tensor size counts in the working set but there is no eviction IO.

#include "core/dag.h"
#include "core/subgraph.h"
#include "core/types.h"
#include "partition/partition.h"
#include "search/coupling_search.h"
#include "search/local_search.h"
#include "solution/solution.h"
#include <cmath>
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, double got, double exp, double tol = 0.5) {
    if (std::abs(got - exp) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l
           << " got=" << got << " exp=" << exp << "\n"; }
}

static TileConfig TC(int64_t w, int64_t h, int64_t k, SnakeDir s = SnakeDir::None) {
    return {w, h, k, s};
}
static Subgraph make_sg(const Problem& p, const DAG& d, std::vector<size_t> ops) {
    auto sg = Subgraph::create(p, d, std::move(ops));
    if (!sg) { std::cerr << "FATAL: Subgraph::create failed\n"; std::exit(1); }
    return std::move(*sg);
}

// ============================================================================
// Diamond graph used by most tests:
//
//   T0(128x128) -> Op0(PW,1000) -> T1(128x128) -> Op1(PW,100) -> T2(128x128)
//                                       |
//                                       +-------> Op2(PW,100) -> T3(128x128)
//
// Subgraph {Op0, Op1}: T1 is ephemeral (produced by Op0, consumed by Op1).
//   Op2 is external and also consumes T1.
// ============================================================================

static Problem make_diamond() {
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {
        {OpType::Pointwise, {0}, {1}, 1000},  // Op0: T0 -> T1
        {OpType::Pointwise, {1}, {2}, 100},    // Op1: T1 -> T2
        {OpType::Pointwise, {1}, {3}, 100},    // Op2: T1 -> T3
    };
    p.fast_memory_capacity = 60000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    // Precompute retainable_tensors (same logic as io.cpp)
    for (size_t i = 0; i < p.tensors.size(); i++) {
        if (p.tensors[i].size() > p.fast_memory_capacity) continue;
        bool has_consumer = false;
        for (auto& op : p.ops)
            for (auto t : op.inputs)
                if (t == i) { has_consumer = true; break; }
        if (has_consumer) p.retainable_tensors.insert(i);
    }
    return p;
}

// ============================================================================
// Test 1: Ephemeral classification is correct
// ============================================================================

void test_ephemeral_classification() {
    std::cout << "=== test_ephemeral_classification ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);

    // Subgraph {Op0, Op1}: T1 is produced (Op0) and consumed (Op1) internally.
    auto sg01 = make_sg(p, d, {0, 1});
    CHECK("T1 is ephemeral in {Op0,Op1}", sg01.ephemeral().count(1));
    CHECK("T1 is NOT boundary output in {Op0,Op1}", !sg01.boundary_outputs().count(1));
    CHECK("T0 is boundary input", sg01.boundary_inputs().count(0));
    CHECK("T2 is boundary output", sg01.boundary_outputs().count(2));

    // Subgraph {Op0}: T1 is produced but NOT consumed internally -> boundary output.
    auto sg0 = make_sg(p, d, {0});
    CHECK("T1 is boundary output in {Op0}", sg0.boundary_outputs().count(1));
    CHECK("T1 is NOT ephemeral in {Op0}", !sg0.ephemeral().count(1));
}

// ============================================================================
// Test 2: Working set with retained ephemeral
// ============================================================================

void test_working_set_with_retained_ephemeral() {
    std::cout << "=== test_working_set_with_retained_ephemeral ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // T1 is ephemeral -> not in boundary_tensor_info_ -> zero ws contribution
    // without retention.
    int64_t ws_no_retain = sg.working_set(TC(128,128,1));
    // Working set = T0 slice (128*128) + T2 slice (128*128) = 32768
    CHECK_EQ("ws without retain", (double)ws_no_retain, 32768.0);

    // With T1 in retain_these: full_size of T1 (128*128=16384) is added.
    FlatSet<size_t> retain_t1 = {1};
    int64_t ws_retain = sg.working_set(TC(128,128,1), {}, retain_t1);
    // = 32768 (boundary) + 16384 (retained ephemeral T1) = 49152
    CHECK_EQ("ws with retained ephemeral T1", (double)ws_retain, 49152.0);
}

// ============================================================================
// Test 3: Cost with retained ephemeral — no eviction IO for T1
// ============================================================================

void test_cost_with_retained_ephemeral() {
    std::cout << "=== test_cost_with_retained_ephemeral ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // Without retention: T1 is ephemeral, zero IO.
    // IO = load T0 (16384/10=1638.4) + evict T2 (16384/10=1638.4) = 3276.8
    // Compute = 1000+100 = 1100
    // Latency = max(1100, 3276.8) = 3276.8
    auto c_no = sg.compute_cost(TC(128,128,1));
    CHECK("no-retain feasible", c_no.feasible);
    CHECK_EQ("no-retain latency", c_no.latency, 3276.8);

    // With T1 retained: T1 stays in fast memory (no eviction IO for T1).
    // IO = load T0 (1638.4) + evict T2 (1638.4) = 3276.8  (same — T1 has no IO)
    // Working set increases but latency doesn't change (T1 adds no IO).
    FlatSet<size_t> retain_t1 = {1};
    auto c_ret = sg.compute_cost(TC(128,128,1), {}, retain_t1);
    CHECK("retain feasible", c_ret.feasible);
    CHECK_EQ("retain latency same (no eviction IO for ephemeral)",
             c_ret.latency, 3276.8);
    // Working set is larger though:
    CHECK_EQ("retain working_set includes T1",
             (double)c_ret.working_set, 49152.0);
}

// ============================================================================
// Test 4: Solution with retained ephemeral validates
// ============================================================================

void test_solution_validates_retained_ephemeral() {
    std::cout << "=== test_solution_validates_retained_ephemeral ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);

    // Step 0: {Op0, Op1} with T1 retained
    auto sg0 = make_sg(p, d, {0, 1});
    auto c0 = sg0.best_cost();

    // Step 1: {Op2} consumes T1 (retained from step 0)
    auto sg1 = make_sg(p, d, {2});
    auto c1 = sg1.best_cost();

    std::vector<ScheduleStep> steps;
    steps.push_back({std::move(sg0), c0.config, {1}});  // retain T1
    steps.push_back({std::move(sg1), c1.config, {}});

    Solution sol(p, d, std::move(steps));
    auto vr = sol.validate();
    if (!vr.valid)
        std::cout << "  validation error: " << vr.error << "\n";
    CHECK("solution with retained ephemeral is valid", vr.valid);
}

// ============================================================================
// Test 5: Retained ephemeral can make working set exceed capacity
// ============================================================================

void test_retained_ephemeral_oom() {
    std::cout << "=== test_retained_ephemeral_oom ===\n";
    // Tight memory: capacity barely fits T0+T2 slices but NOT T1 full size.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {
        {OpType::Pointwise, {0}, {1}, 1000},
        {OpType::Pointwise, {1}, {2}, 100},
        {OpType::Pointwise, {1}, {3}, 100},
    };
    p.fast_memory_capacity = 40000;  // tight: 32768 (T0+T2) fits, 49152 doesn't
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1});

    // Without retention: ws=32768 < 40000 -> feasible
    auto c_no = sg.compute_cost(TC(128,128,1));
    CHECK("no-retain feasible", c_no.feasible);

    // With T1 retained: ws=49152 > 40000 -> infeasible
    FlatSet<size_t> retain_t1 = {1};
    auto c_ret = sg.compute_cost(TC(128,128,1), {}, retain_t1);
    CHECK("retain infeasible (OOM)", !c_ret.feasible);
}

// ============================================================================
// Test 6: partition_has_gap detects gap without retention
// ============================================================================

void test_partition_has_gap_without_retention() {
    std::cout << "=== test_partition_has_gap_without_retention ===\n";
    // Diamond: {Op0,Op1} fused, {Op2} separate.
    // T1 is ephemeral in {Op0,Op1}, Op2 needs T1 → gap.
    auto p = make_diamond();
    auto d = DAG::build(p);

    Partition part = Partition::trivial(p, d);
    // Merge Op0 and Op1 into one group
    FlatSet<size_t> merged = {0, 1};
    double mc = part.eval_set(merged);
    part.groups[0].ops = merged;
    part.groups[0].cost = mc;
    part.groups[1].alive = false;
    part.rebuild_index();

    // Without retention info → gap detected
    CHECK("gap without retention", partition_has_gap(part));
    // With no-op callback → still gap
    CHECK("gap with false callback", partition_has_gap(part, [](size_t) { return false; }));
}

// ============================================================================
// Test 7: partition_has_gap accepts retained ephemeral
// ============================================================================

void test_partition_has_gap_with_retention() {
    std::cout << "=== test_partition_has_gap_with_retention ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);

    Partition part = Partition::trivial(p, d);
    FlatSet<size_t> merged = {0, 1};
    double mc = part.eval_set(merged);
    part.groups[0].ops = merged;
    part.groups[0].cost = mc;
    part.groups[1].alive = false;
    part.rebuild_index();

    // T1 retained → no gap
    CHECK("no gap with T1 retained",
          !partition_has_gap(part, [](size_t t) { return t == 1; }));
}

// ============================================================================
// Test 8: CoupledPartition::is_retained
// ============================================================================

void test_coupled_partition_is_retained() {
    std::cout << "=== test_coupled_partition_is_retained ===\n";
    auto p = make_diamond();
    auto d = DAG::build(p);

    CoupledPartition cp;
    cp.init_from(Partition::trivial(p, d));

    // No couplings → nothing retained
    CHECK("nothing retained initially", !cp.is_retained(0));
    CHECK("nothing retained initially", !cp.is_retained(1));

    // Add a coupling: group 0 retains T1 for group 2
    size_t ga = 0, gb = 2;
    cp.next_group[ga] = gb;
    cp.prev_group[gb] = ga;
    cp.retained[{ga, gb}].insert(1);

    CHECK("T1 is retained after coupling", cp.is_retained(1));
    CHECK("T0 is not retained", !cp.is_retained(0));
    CHECK("T2 is not retained", !cp.is_retained(2));
}

// ============================================================================
// Test 9: eval_couple accepts ephemeral tensor
// ============================================================================

void test_eval_couple_ephemeral() {
    std::cout << "=== test_eval_couple_ephemeral ===\n";
    // Build partition where {Op0,Op1} is one group and {Op2} is another.
    // T1 is ephemeral in {Op0,Op1}. eval_couple should accept coupling T1
    // from {Op0,Op1} to {Op2}.
    auto p = make_diamond();
    p.fast_memory_capacity = 100000;  // plenty of room for retention
    auto d = DAG::build(p);

    Partition part = Partition::trivial(p, d);
    // Merge Op0+Op1 into group 0
    FlatSet<size_t> merged = {0, 1};
    double mc = part.eval_set(merged);
    part.groups[0].ops = merged;
    part.groups[0].cost = mc;
    part.groups[1].alive = false;
    part.rebuild_index();
    part.finalize();

    CoupledPartition cp;
    cp.init_from(std::move(part));

    // ga = group with {Op0,Op1}, gb = group with {Op2}
    size_t ga = 0, gb = 2;
    auto ev = eval_couple(cp, ga, gb, 1);  // T1
    CHECK("eval_couple accepts ephemeral T1", ev.feasible);
}

// ============================================================================
// Test 10: invalidate_couplings keeps ephemeral tensor
// ============================================================================

void test_invalidate_keeps_ephemeral() {
    std::cout << "=== test_invalidate_keeps_ephemeral ===\n";
    auto p = make_diamond();
    p.fast_memory_capacity = 100000;
    auto d = DAG::build(p);

    Partition part = Partition::trivial(p, d);
    FlatSet<size_t> merged = {0, 1};
    double mc = part.eval_set(merged);
    part.groups[0].ops = merged;
    part.groups[0].cost = mc;
    part.groups[1].alive = false;
    part.rebuild_index();

    CoupledPartition cp;
    cp.init_from(std::move(part));

    // Manually set up coupling with ephemeral T1
    size_t ga = 0, gb = 2;
    cp.next_group[ga] = gb;
    cp.prev_group[gb] = ga;
    cp.retained[{ga, gb}].insert(1);

    // invalidate_couplings should keep T1 (produced in ga, consumed in gb)
    cp.invalidate_couplings();
    CHECK("coupling edge still exists", cp.next_group[ga] == gb);
    auto it = cp.retained.find({ga, gb});
    CHECK("T1 still retained after invalidate",
          it != cp.retained.end() && it->second.count(1));
}

int main() {
    test_ephemeral_classification();
    test_working_set_with_retained_ephemeral();
    test_cost_with_retained_ephemeral();
    test_solution_validates_retained_ephemeral();
    test_retained_ephemeral_oom();
    test_partition_has_gap_without_retention();
    test_partition_has_gap_with_retention();
    test_coupled_partition_is_retained();
    test_eval_couple_ephemeral();
    test_invalidate_keeps_ephemeral();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail ? 1 : 0;
}
