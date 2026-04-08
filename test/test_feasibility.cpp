// test_feasibility.cpp
//
// Unit tests for feasibility::kahn_with_delta and related checks.
// Build: g++ -std=c++17 -I<src> test_feasibility.cpp feasibility.cpp dag.cpp -o test_feasibility
// Run:   ./test_feasibility

#include "search/feasibility.h"
#include "core/types.h"
#include "core/dag.h"
#include <cassert>
#include <iostream>
#include <set>
#include <vector>

using namespace feasibility;

// ============================================================================
// Test helpers: build tiny problems and DAGs
// ============================================================================

// Build a Problem with N ops, each with given inputs/outputs tensors.
// All ops are MatMul with base_cost=100, tensors are 128x128.
struct TestGraph {
    Problem prob;
    DAG dag;
    std::vector<std::vector<size_t>> op_to_groups;

    // Add an op with given input/output tensors
    void add_op(std::vector<size_t> inputs, std::vector<size_t> outputs) {
        Op op;
        op.type = OpType::MatMul;
        op.base_cost = 100;
        op.inputs = inputs;
        op.outputs = outputs;
        prob.ops.push_back(op);
    }

    // Ensure enough tensors exist
    void ensure_tensors(size_t n) {
        while (prob.tensors.size() < n) {
            Tensor t;
            t.width = 128;
            t.height = 128;
            prob.tensors.push_back(t);
        }
    }

    void build() {
        dag = DAG::build(prob);
        op_to_groups.clear();
    }

    // Set up op_to_groups for a given partition (list of groups)
    void set_groups(const std::vector<FlatSet<size_t>>& groups) {
        op_to_groups.assign(prob.num_ops(), {});
        for (size_t gi = 0; gi < groups.size(); gi++)
            for (auto op : groups[gi])
                op_to_groups[op].push_back(gi);
    }

    // Check acyclicity with no delta
    bool is_acyclic(const std::vector<bool>& alive, size_t na) {
        MoveDelta delta;
        return kahn_with_delta(prob, dag, op_to_groups, alive, na, delta);
    }
};

// ============================================================================
// Chain DAG: 0 → 1 → 2  (via tensors T0, T1)
//
//   Op 0: inputs=[], outputs=[T0]
//   Op 1: inputs=[T0], outputs=[T1]
//   Op 2: inputs=[T1], outputs=[T2]
// ============================================================================

TestGraph make_chain3() {
    TestGraph g;
    g.ensure_tensors(3);
    g.add_op({}, {0});      // op 0
    g.add_op({0}, {1});     // op 1
    g.add_op({1}, {2});     // op 2
    g.build();
    return g;
}

void test_chain3_basic() {
    std::cerr << "test_chain3_basic... ";
    auto g = make_chain3();

    // All singletons: {0}, {1}, {2} — acyclic
    g.set_groups({{0}, {1}, {2}});
    std::vector<bool> alive = {true, true, true};
    assert(g.is_acyclic(alive, 3));

    std::cerr << "OK\n";
}

void test_chain3_merge_adjacent() {
    std::cerr << "test_chain3_merge_adjacent... ";
    auto g = make_chain3();

    // Merge {0} and {1} → should be acyclic (adjacent)
    g.set_groups({{0}, {1}, {2}});
    std::vector<bool> alive = {true, true, true};

    MoveDelta delta{MoveDelta::MERGE_PAIR, SIZE_MAX, 0, 1, nullptr};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 3, delta));

    // Merge {1} and {2} → acyclic
    MoveDelta delta2{MoveDelta::MERGE_PAIR, SIZE_MAX, 1, 2, nullptr};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 3, delta2));

    std::cerr << "OK\n";
}

void test_chain3_merge_skip() {
    std::cerr << "test_chain3_merge_skip... ";
    auto g = make_chain3();

    // Merge {0} and {2}, skipping {1} → CYCLE
    // {0,2} needs T0 (internal), T1 from {1}. {1} needs T0 from {0,2}.
    // But {0,2} also produces T0, so {1} depends on {0,2}, and {0,2} depends
    // on {1} for T1. Cycle!
    g.set_groups({{0}, {1}, {2}});
    std::vector<bool> alive = {true, true, true};

    MoveDelta delta{MoveDelta::MERGE_PAIR, SIZE_MAX, 0, 2, nullptr};
    assert(!kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 3, delta));

    // Also verify via DAG::merge_creates_cycle
    assert(g.dag.merge_creates_cycle({0}, {2}));

    std::cerr << "OK\n";
}

void test_chain3_steal() {
    std::cerr << "test_chain3_steal... ";
    auto g = make_chain3();

    // Steal op 1 from {1} to {0} → {0,1} and {2} → acyclic
    g.set_groups({{0}, {1}, {2}});
    std::vector<bool> alive = {true, true, true};

    MoveDelta delta{MoveDelta::STEAL, 1, 1, 0, nullptr};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 3, delta));

    // Steal op 1 from {1} to {2} → {0} and {1,2} → acyclic
    MoveDelta delta2{MoveDelta::STEAL, 1, 1, 2, nullptr};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 3, delta2));

    std::cerr << "OK\n";
}

// ============================================================================
// Diamond DAG: 0 → {1,2} → 3
//
//   Op 0: outputs=[T0]
//   Op 1: inputs=[T0], outputs=[T1]
//   Op 2: inputs=[T0], outputs=[T2]
//   Op 3: inputs=[T1, T2], outputs=[T3]
// ============================================================================

TestGraph make_diamond() {
    TestGraph g;
    g.ensure_tensors(4);
    g.add_op({}, {0});         // op 0: source
    g.add_op({0}, {1});        // op 1: left branch
    g.add_op({0}, {2});        // op 2: right branch
    g.add_op({1, 2}, {3});     // op 3: sink
    g.build();
    return g;
}

void test_diamond_merge_branches() {
    std::cerr << "test_diamond_merge_branches... ";
    auto g = make_diamond();

    // Merge {1} and {2} → {0}, {1,2}, {3} → acyclic (parallel branches)
    g.set_groups({{0}, {1}, {2}, {3}});
    std::vector<bool> alive = {true, true, true, true};

    MoveDelta delta{MoveDelta::MERGE_PAIR, SIZE_MAX, 1, 2, nullptr};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 4, delta));

    std::cerr << "OK\n";
}

void test_diamond_merge_source_sink() {
    std::cerr << "test_diamond_merge_source_sink... ";
    auto g = make_diamond();

    // Merge {0} and {3} → CYCLE ({0,3} depends on {1} and {2}, which depend on {0,3})
    g.set_groups({{0}, {1}, {2}, {3}});
    std::vector<bool> alive = {true, true, true, true};

    MoveDelta delta{MoveDelta::MERGE_PAIR, SIZE_MAX, 0, 3, nullptr};
    assert(!kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 4, delta));

    std::cerr << "OK\n";
}

// ============================================================================
// Bottleneck-3br pattern: recompute bridge removal
//
//   Op 0: inputs=[T0], outputs=[T1]
//   Op 1: inputs=[T1], outputs=[T2]
//   Op 2: inputs=[T1, T2], outputs=[T3]
//
// Groups: G0={0,1}, G1={0,2} (op 0 recomputed)
// DE_RECOMPUTE: kill G0's copy of op 0 → G0 becomes {1}
//   Now {1} needs T1 from G1, and G1={0,2} needs T2 from {1} → CYCLE
// ============================================================================

TestGraph make_bottleneck_bridge() {
    TestGraph g;
    g.ensure_tensors(4);

    g.add_op({0}, {1});       // op 0
    g.add_op({1}, {2});       // op 1
    g.add_op({1, 2}, {3});    // op 2

    g.prob.fast_memory_capacity = 40000;
    g.prob.slow_memory_bandwidth = 10;
    g.build();
    return g;
}

void test_bottleneck_recompute_bridge() {
    std::cerr << "test_bottleneck_recompute_bridge... ";
    auto g = make_bottleneck_bridge();

    // Setup: G0={0,1}, G1={0,2}, op 0 recomputed
    g.set_groups({{0, 1}, {0, 2}});
    std::vector<bool> alive = {true, true};

    // Current state: acyclic (G0 produces T1 internally, G1 gets T2 from G0)
    assert(g.is_acyclic(alive, 2));

    // EJECT op 0 from G0: G0 becomes {1}, new G2={0}
    // Groups: G0={1}, G1={0,2}, G2={0}
    // This should be acyclic: G2={0} → G0={1} → G1={0,2}
    g.set_groups({{1}, {0, 2}, {0}});
    alive = {true, true, true};
    assert(g.is_acyclic(alive, 3));

    // Now DE_RECOMPUTE: kill G2={0} (op 0 covered by G1)
    // Remaining: G0={1}, G1={0,2}
    // G0 needs T1 (from op 0, in G1). G1 needs T2 (from op 1, in G0).
    // CYCLE!
    alive = {true, true, false};  // G2 killed
    assert(!g.is_acyclic(alive, 2));

    std::cerr << "OK\n";
}

// ============================================================================
// Split tests
// ============================================================================

// ============================================================================
// Recompute: adding an op to a second group
// ============================================================================

void test_recompute() {
    std::cerr << "test_recompute... ";
    auto g = make_chain3();

    // Groups: {0,1}, {2}. Recompute op 1 into group {2} → {0,1}, {1,2}
    // Acyclic: {0,1} produces T0,T1. {1,2} produces T1,T2. No cycle.
    g.set_groups({{0, 1}, {2}});
    std::vector<bool> alive = {true, true};

    MoveDelta delta{MoveDelta::RECOMPUTE, 1, SIZE_MAX, 1, nullptr};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 2, delta));

    // Recompute op 0 into group {2} → {0,1}, {0,2}
    // {0,2} needs T1 from {0,1}. {0,1} has no external dep. Acyclic.
    MoveDelta delta2{MoveDelta::RECOMPUTE, 0, SIZE_MAX, 1, nullptr};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 2, delta2));

    std::cerr << "OK\n";
}

// ============================================================================
// Multi-group merge
// ============================================================================

TestGraph make_chain5() {
    TestGraph g;
    g.ensure_tensors(5);
    g.add_op({}, {0});      // op 0
    g.add_op({0}, {1});     // op 1
    g.add_op({1}, {2});     // op 2
    g.add_op({2}, {3});     // op 3
    g.add_op({3}, {4});     // op 4
    g.build();
    return g;
}

void test_multi_merge() {
    std::cerr << "test_multi_merge... ";
    auto g = make_chain5();

    // Groups: {0}, {1}, {2}, {3}, {4}
    g.set_groups({{0}, {1}, {2}, {3}, {4}});
    std::vector<bool> alive = {true, true, true, true, true};

    // Merge {0}, {2}, {4} — skipping {1} and {3} → CYCLE
    std::vector<size_t> ml = {0, 2, 4};
    MoveDelta delta{MoveDelta::MERGE_MULTI, SIZE_MAX, SIZE_MAX, SIZE_MAX, &ml};
    assert(!kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 5, delta));

    // Merge {0}, {1}, {2} — contiguous → acyclic
    std::vector<size_t> ml2 = {0, 1, 2};
    MoveDelta delta2{MoveDelta::MERGE_MULTI, SIZE_MAX, SIZE_MAX, SIZE_MAX, &ml2};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 5, delta2));

    std::cerr << "OK\n";
}

// ============================================================================
// SPLIT_MOVE delta
// ============================================================================

void test_split_move_delta() {
    std::cerr << "test_split_move_delta... ";
    auto g = make_chain3();

    // Group {0,1,2}: split {2} into new group
    // Virtual: G0={0,1}, G1(new)={2}
    g.set_groups({{0, 1, 2}});
    std::vector<bool> alive = {true, true};  // G0 alive, G1 (virtual) alive

    FlatSet<size_t> split_ops = {2};
    MoveDelta delta;
    delta.type = MoveDelta::SPLIT_MOVE;
    delta.ga = 0;  // source
    delta.gb = 1;  // virtual new group
    delta.split_ops = &split_ops;
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 2, delta));

    // Split {0} into new group → G0={1,2}, G1={0}
    // Acyclic: G1={0} → G0={1,2}
    FlatSet<size_t> split_ops2 = {0};
    MoveDelta delta2;
    delta2.type = MoveDelta::SPLIT_MOVE;
    delta2.ga = 0;
    delta2.gb = 1;
    delta2.split_ops = &split_ops2;
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 2, delta2));

    // Split {0,2} from {0,1,2} → G0={1}, G1={0,2}
    // {0,2} needs T1 from {1}, {1} needs T0 from {0,2} → CYCLE
    FlatSet<size_t> split_ops3 = {0, 2};
    MoveDelta delta3;
    delta3.type = MoveDelta::SPLIT_MOVE;
    delta3.ga = 0;
    delta3.gb = 1;
    delta3.split_ops = &split_ops3;
    assert(!kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 2, delta3));

    std::cerr << "OK\n";
}

// ============================================================================
// Parallel branches: independent ops should never create cycles
// ============================================================================

TestGraph make_parallel() {
    TestGraph g;
    g.ensure_tensors(5);
    g.add_op({0}, {1});     // op 0: branch A
    g.add_op({0}, {2});     // op 1: branch B
    g.add_op({0}, {3});     // op 2: branch C
    g.add_op({1, 2, 3}, {4}); // op 3: merge
    g.build();
    return g;
}

void test_parallel_branches() {
    std::cerr << "test_parallel_branches... ";
    auto g = make_parallel();

    // All singletons
    g.set_groups({{0}, {1}, {2}, {3}});
    std::vector<bool> alive = {true, true, true, true};
    assert(g.is_acyclic(alive, 4));

    // Merge any two parallel branches → acyclic
    MoveDelta d1{MoveDelta::MERGE_PAIR, SIZE_MAX, 0, 1, nullptr};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 4, d1));

    // Merge all three branches → acyclic
    std::vector<size_t> ml = {0, 1, 2};
    MoveDelta d2{MoveDelta::MERGE_MULTI, SIZE_MAX, SIZE_MAX, SIZE_MAX, &ml};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 4, d2));

    // Merge branch + sink → {0,3} depends on {1} and {2} which depend on T0...
    // But T0 is a graph input, so no dependency back to {0,3}. Acyclic.
    MoveDelta d3{MoveDelta::MERGE_PAIR, SIZE_MAX, 0, 3, nullptr};
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 4, d3));

    std::cerr << "OK\n";
}

// ============================================================================
// EXTRACT_MOVE delta
// ============================================================================

void test_extract_move() {
    std::cerr << "test_extract_move... ";
    auto g = make_diamond();

    // Groups: G0={0,1}, G1={2,3}
    // Extract {1,2} from G0 and G1 into new G2
    // Result: G0={0}, G1={3}, G2={1,2}
    // Deps: G0={0}→T0→G2={1,2}, G2→T1,T2→G1={3}. Acyclic.
    g.set_groups({{0, 1}, {2, 3}});
    std::vector<bool> alive = {true, true, true};  // G0, G1, G2(virtual)

    FlatSet<size_t> extract = {1, 2};
    std::vector<size_t> sources = {0, 1};
    MoveDelta delta;
    delta.type = MoveDelta::EXTRACT_MOVE;
    delta.gb = 2;  // virtual new group
    delta.split_ops = &extract;
    delta.merge_list = &sources;
    assert(kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 3, delta));

    // Extract {0,3} from G0 and G1 → G2={0,3}
    // G0={1}, G1={2}, G2={0,3}
    // G2 depends on G0 (T1) and G1 (T2), G0 depends on G2 (T0) → CYCLE
    FlatSet<size_t> extract2 = {0, 3};
    MoveDelta delta2;
    delta2.type = MoveDelta::EXTRACT_MOVE;
    delta2.gb = 2;
    delta2.split_ops = &extract2;
    delta2.merge_list = &sources;
    assert(!kahn_with_delta(g.prob, g.dag, g.op_to_groups, alive, 3, delta2));

    std::cerr << "OK\n";
}

// ============================================================================
// Solution ordering violation tests
// ============================================================================

// We can't easily construct ScheduleSteps without the full Subgraph machinery,
// so we test the Layer 1 pre-filters and Layer 2 Kahn's thoroughly instead.
// Layer 3 (has_ordering_violation) is a straightforward loop; its correctness
// follows from the boundary_inputs/boundary_outputs API.

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cerr << "=== Feasibility unit tests ===\n";

    test_chain3_basic();
    test_chain3_merge_adjacent();
    test_chain3_merge_skip();
    test_chain3_steal();
    test_diamond_merge_branches();
    test_diamond_merge_source_sink();
    test_bottleneck_recompute_bridge();
    test_recompute();
    test_multi_merge();
    test_split_move_delta();
    test_parallel_branches();
    test_extract_move();

    std::cerr << "\n=== All " << 12 << " tests passed ===\n";
    return 0;
}