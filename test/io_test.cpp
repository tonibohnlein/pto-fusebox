// io_test.cpp — Unit tests for read_problem / write_solution / retainable_tensors
//
// Tests are self-contained: they build Problem objects directly (matching what
// read_problem would produce) and call the classification logic inline, so no
// JSON files need to be present. Round-trip JSON tests write temp files and
// re-read them using read_problem.
//
// Build: make io_test
// Run:   ./io_test

#include "io/io.h"
#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "solution/solution.h"
#include <nlohmann/json.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;

static int g_pass = 0, g_fail = 0;

static void CHECK(const char* label, bool cond) {
    if (cond) { g_pass++; }
    else { g_fail++; std::cout << "  FAIL: " << label << "\n"; }
}
static void CHECK_EQ(const char* label, double got, double exp, double tol = 0.1) {
    if (std::abs(got - exp) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label
           << " got=" << got << " exp=" << exp << "\n"; }
}
static void CHECK_EQ_S(const char* label, size_t got, size_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label
           << " got=" << got << " exp=" << exp << "\n"; }
}

// ============================================================================
// Helper: build the retainable_tensors set using the same two-rule logic as
// read_problem, without going through JSON. This lets us unit-test the rules
// directly on hand-crafted Problem objects.
// ============================================================================

static std::set<size_t> compute_retainable(const Problem& p) {
    const size_t nt = p.num_tensors();
    std::vector<size_t> consumer_count(nt, 0);
    for (auto& op : p.ops)
        for (auto t : op.inputs)
            consumer_count[t]++;

    std::set<size_t> result;
    for (size_t i = 0; i < nt; i++) {
        if (p.tensors[i].size() > p.fast_memory_capacity) continue;
        if (consumer_count[i] == 0)                        continue;
        result.insert(i);
    }
    return result;
}

// ============================================================================
// 1. Retainable tensor classification
// ============================================================================

void test_retainable_basic_chain() {
    std::cout << "--- test_retainable_basic_chain ---\n";
    // T0 -> Op0 -> T1 -> Op1 -> T2
    // T0: graph input, 1 consumer
    // T1: produced tensor, 1 consumer
    // T2: graph output,  0 consumers
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    auto r = compute_retainable(p);
    // T0: 1 consumer, fits → retainable (rule 3 removed)
    CHECK("T0 retainable (graph input, 1 consumer)", r.count(0));
    // T1: 1 consumer, fits → retainable
    CHECK("T1 retainable (produced, 1 consumer)", r.count(1));
    // T2: 0 consumers → NOT retainable
    CHECK("T2 NOT retainable (graph output)", !r.count(2));
    CHECK_EQ_S("total retainable", r.size(), 2);
}

void test_retainable_graph_input_multi_consumer() {
    std::cout << "--- test_retainable_graph_input_multi_consumer ---\n";
    // T0 (graph input) consumed by Op0 and Op1 — classic reuse candidate.
    // T0 -> Op0 -> T1
    // T0 -> Op1 -> T2
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{0},{2},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    auto r = compute_retainable(p);
    CHECK("T0 retainable (2 consumers)", r.count(0));
    CHECK("T1 NOT retainable (0 consumers)", !r.count(1));
    CHECK("T2 NOT retainable (0 consumers)", !r.count(2));
}

void test_retainable_graph_input_single_consumer_recomputation() {
    std::cout << "--- test_retainable_graph_input_single_consumer_recomputation ---\n";
    // Diamond: T0 -> Op0 -> T1 -> Op1 -> T2
    //          T0 -> Op0 -> T1 -> Op2 -> T3
    //          T2, T3 -> Op3 -> T4
    // T0 is a graph input consumed only by Op0. But in a recomputation schedule,
    // Op0 appears in two subgraphs, so T0 must be loaded twice unless retained.
    // After removing rule 3, T0 is retainable.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},   // Op0
             {OpType::Pointwise,{1},{2},1000},   // Op1
             {OpType::Pointwise,{1},{3},1000},   // Op2
             {OpType::Pointwise,{2,3},{4},1000}}; // Op3
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    auto r = compute_retainable(p);
    // T0: graph input, 1 consumer (Op0). Old rule 3 would exclude this.
    CHECK("T0 retainable after rule-3 removal", r.count(0));
    // T1: produced, 2 consumers → retainable
    CHECK("T1 retainable (2 consumers)", r.count(1));
    // T2, T3: produced, 1 consumer each → retainable
    CHECK("T2 retainable", r.count(2));
    CHECK("T3 retainable", r.count(3));
    // T4: graph output, 0 consumers → NOT retainable
    CHECK("T4 NOT retainable (graph output)", !r.count(4));
    CHECK_EQ_S("total", r.size(), 4);
}

void test_retainable_too_large() {
    std::cout << "--- test_retainable_too_large ---\n";
    // T0 is 1024×1024 = 1M elements. Fast memory = 100 (tiny).
    Problem p;
    p.tensors = {{1024,1024},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 100;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    auto r = compute_retainable(p);
    // T0: too large → excluded
    CHECK("T0 NOT retainable (too large)", !r.count(0));
    // T1: 128×128 = 16384 > 100 → also too large
    CHECK("T1 NOT retainable (too large)", !r.count(1));
    CHECK_EQ_S("total", r.size(), 0);
}

void test_retainable_exactly_capacity() {
    std::cout << "--- test_retainable_exactly_capacity ---\n";
    // T0.size() == fast_memory_capacity exactly → fits (≤), retainable.
    // T1.size() == capacity + 1 → doesn't fit, not retainable.
    Problem p;
    p.tensors = {{128,128},{129,128}};   // 16384, 16512
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 16384;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    auto r = compute_retainable(p);
    CHECK("T0 retainable (size == capacity)", r.count(0));
    CHECK("T1 NOT retainable (size > capacity)", !r.count(1));
}

void test_retainable_graph_output_excluded() {
    std::cout << "--- test_retainable_graph_output_excluded ---\n";
    // T1 is a graph output (no consuming op). Even though it fits, it cannot
    // be usefully retained — it will be evicted at the end of the schedule and
    // no later subgraph will read it.
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    auto r = compute_retainable(p);
    CHECK("T0 retainable (graph input, 1 consumer)", r.count(0));
    CHECK("T1 NOT retainable (graph output)", !r.count(1));
}

void test_retainable_multi_output_op() {
    std::cout << "--- test_retainable_multi_output_op ---\n";
    // Op0 produces T1 and T2 (multi-output). Both consumed downstream.
    // T0 -> Op0 -> T1 -> Op1 -> T3
    //           -> T2 -> Op2 -> T4
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{0},{2},1000},
             {OpType::Pointwise,{1},{3},1000},
             {OpType::Pointwise,{2},{4},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    auto r = compute_retainable(p);
    CHECK("T0 retainable (graph input)", r.count(0));
    CHECK("T1 retainable", r.count(1));
    CHECK("T2 retainable", r.count(2));
    CHECK("T3 NOT retainable (output)", !r.count(3));
    CHECK("T4 NOT retainable (output)", !r.count(4));
}

void test_retainable_no_tensors_if_all_too_large() {
    std::cout << "--- test_retainable_no_tensors_if_all_too_large ---\n";
    Problem p;
    p.tensors = {{512,512},{512,512},{512,512}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 1000;  // all 512×512=262144 >> 1000
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;

    auto r = compute_retainable(p);
    CHECK("empty set when all too large", r.empty());
}

// ============================================================================
// 2. JSON round-trip: write a problem as JSON, read it back, verify fields
//
// We can't test read_problem without files, so we write minimal JSON manually
// and call read_problem on them.
// ============================================================================

static std::string tmp_path(const std::string& name) {
    return "/tmp/io_test_" + name + ".json";
}

static void write_json_file(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

void test_read_basic_chain() {
    std::cout << "--- test_read_basic_chain ---\n";
    std::string path = tmp_path("chain");
    write_json_file(path, R"({
  "widths":  [128, 128, 128],
  "heights": [128, 128, 128],
  "inputs":  [[0], [1]],
  "outputs": [[1], [2]],
  "base_costs": [1000, 500],
  "op_types": ["Pointwise", "Pointwise"],
  "fast_memory_capacity": 50000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
})");
    auto p = read_problem(path);
    CHECK_EQ_S("num_tensors", p.num_tensors(), 3);
    CHECK_EQ_S("num_ops",     p.num_ops(),     2);
    CHECK("T0 width=128",  p.tensors[0].width == 128);
    CHECK("T0 height=128", p.tensors[0].height == 128);
    CHECK("Op0 PW",   p.ops[0].type == OpType::Pointwise);
    CHECK("Op0 in=0", p.ops[0].inputs[0] == 0);
    CHECK("Op0 out=1",p.ops[0].outputs[0] == 1);
    CHECK("Op1 cost=500", p.ops[1].base_cost == 500);
    CHECK("cap=50000", p.fast_memory_capacity == 50000);
    CHECK("bw=10",     p.slow_memory_bandwidth == 10);
    CHECK("native_w=128", p.native_w == 128);
    CHECK("native_h=128", p.native_h == 128);

    // Retainable: T0(input,1consumer)→yes, T1(produced,1consumer)→yes,
    // T2(output,0consumers)→no
    CHECK("T0 retainable", p.retainable_tensors.count(0));
    CHECK("T1 retainable", p.retainable_tensors.count(1));
    CHECK("T2 NOT retainable", !p.retainable_tensors.count(2));
}

void test_read_matmul_op() {
    std::cout << "--- test_read_matmul_op ---\n";
    std::string path = tmp_path("matmul");
    write_json_file(path, R"({
  "widths":  [128, 128, 128],
  "heights": [128, 128, 128],
  "inputs":  [[0, 1]],
  "outputs": [[2]],
  "base_costs": [2000],
  "op_types": ["MatMul"],
  "fast_memory_capacity": 100000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
})");
    auto p = read_problem(path);
    CHECK("Op0 is MatMul", p.ops[0].type == OpType::MatMul);
    // MatMul: inputs[0] = LHS, inputs[1] = RHS — order must be preserved
    CHECK("Op0 LHS = T0", p.ops[0].inputs[0] == 0);
    CHECK("Op0 RHS = T1", p.ops[0].inputs[1] == 1);
    CHECK("Op0 out = T2", p.ops[0].outputs[0] == 2);
    // T0, T1 are graph inputs (1 consumer each) — retainable after rule-3 removal
    CHECK("T0 retainable (LHS graph input)", p.retainable_tensors.count(0));
    CHECK("T1 retainable (RHS graph input)", p.retainable_tensors.count(1));
    // T2 is graph output → not retainable
    CHECK("T2 NOT retainable", !p.retainable_tensors.count(2));
}

void test_read_retainable_large_tensor_excluded() {
    std::cout << "--- test_read_retainable_large_tensor_excluded ---\n";
    // T0 is 512×512 = 262144, capacity = 50000 → too large to retain.
    std::string path = tmp_path("largetensor");
    write_json_file(path, R"({
  "widths":  [512, 128],
  "heights": [512, 128],
  "inputs":  [[0]],
  "outputs": [[1]],
  "base_costs": [1000],
  "op_types": ["Pointwise"],
  "fast_memory_capacity": 50000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
})");
    auto p = read_problem(path);
    CHECK("T0 NOT retainable (too large)", !p.retainable_tensors.count(0));
    // T1 is graph output → not retainable
    CHECK("T1 NOT retainable (output)", !p.retainable_tensors.count(1));
    CHECK("retainable set empty", p.retainable_tensors.empty());
}

void test_read_diamond_retainable() {
    std::cout << "--- test_read_diamond_retainable ---\n";
    // Diamond: T0->Op0->T1->Op1->T2, T1->Op2->T3, T2+T3->Op3->T4
    // T0: graph input, 1 consumer (Op0). Previously excluded by rule 3.
    //     After rule-3 removal: retainable. A recomputation schedule that runs
    //     {Op0,Op1} and {Op0,Op2} separately would benefit from retaining T0.
    std::string path = tmp_path("diamond");
    write_json_file(path, R"({
  "widths":  [128, 128, 128, 128, 128],
  "heights": [128, 128, 128, 128, 128],
  "inputs":  [[0], [1], [1], [2, 3]],
  "outputs": [[1], [2], [3], [4]],
  "base_costs": [1000, 1000, 1000, 1000],
  "op_types": ["Pointwise", "Pointwise", "Pointwise", "Pointwise"],
  "fast_memory_capacity": 50000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
})");
    auto p = read_problem(path);
    // T0: 1 consumer → retainable (rule 3 removed)
    CHECK("T0 retainable (recomputation candidate)", p.retainable_tensors.count(0));
    // T1: 2 consumers → retainable
    CHECK("T1 retainable", p.retainable_tensors.count(1));
    // T2: 1 consumer → retainable
    CHECK("T2 retainable", p.retainable_tensors.count(2));
    // T3: 1 consumer → retainable
    CHECK("T3 retainable", p.retainable_tensors.count(3));
    // T4: 0 consumers (graph output) → NOT retainable
    CHECK("T4 NOT retainable", !p.retainable_tensors.count(4));
    CHECK_EQ_S("total retainable", p.retainable_tensors.size(), 4);
}

// ============================================================================
// 3. write_solution round-trip: verify output JSON has correct structure
//    and that the written latencies match what compute_cost returns.
// ============================================================================

void test_write_solution_structure() {
    std::cout << "--- test_write_solution_structure ---\n";
    // Simple chain: {Op0} then {Op1}, no retention, raster order.
    std::string in_path  = tmp_path("chain");  // reuse the file created above
    std::string out_path = tmp_path("sol_out");

    auto p = read_problem(in_path);
    DAG d = DAG::build(p);
    auto sg0 = *Subgraph::create(p, d, {0});
    auto sg1 = *Subgraph::create(p, d, {1});
    auto c0 = sg0.best_cost();
    auto c1 = sg1.best_cost();

    Solution sol(p, d, {
        {std::move(sg0), c0.config, {}},
        {std::move(sg1), c1.config, {}},
    });
    write_solution(out_path, sol);

    // Read back and validate JSON structure
    std::ifstream f(out_path);
    CHECK("output file opened", f.is_open());
    auto j = json::parse(f);

    CHECK("has subgraphs",         j.contains("subgraphs"));
    CHECK("has granularities",     j.contains("granularities"));
    CHECK("has tensors_to_retain", j.contains("tensors_to_retain"));
    CHECK("has traversal_orders",  j.contains("traversal_orders"));
    CHECK("has subgraph_latencies",j.contains("subgraph_latencies"));

    CHECK_EQ_S("subgraphs length",         j["subgraphs"].size(),         2);
    CHECK_EQ_S("granularities length",     j["granularities"].size(),     2);
    CHECK_EQ_S("tensors_to_retain length", j["tensors_to_retain"].size(), 2);
    CHECK_EQ_S("traversal_orders length",  j["traversal_orders"].size(),  2);
    CHECK_EQ_S("latencies length",         j["subgraph_latencies"].size(),2);

    // Op indices present in each subgraph entry
    int sg0_op0 = j["subgraphs"][0][0].get<int>();
    int sg1_op0 = j["subgraphs"][1][0].get<int>();
    CHECK("step0 subgraph=[0]", sg0_op0 == 0);
    CHECK("step1 subgraph=[1]", sg1_op0 == 1);

    // Granularity is [w, h, k]
    CHECK("step0 gran length=3", j["granularities"][0].size() == 3);

    // No retention → empty lists
    CHECK("step0 retain empty", j["tensors_to_retain"][0].empty());
    CHECK("step1 retain empty", j["tensors_to_retain"][1].empty());

    // No snake → null traversal orders
    CHECK("step0 order null", j["traversal_orders"][0].is_null());
    CHECK("step1 order null", j["traversal_orders"][1].is_null());

    // Latencies are positive
    double lat0 = j["subgraph_latencies"][0].get<double>();
    double lat1 = j["subgraph_latencies"][1].get<double>();
    CHECK("step0 lat > 0", lat0 > 0);
    CHECK("step1 lat > 0", lat1 > 0);
}

void test_write_solution_latency_values() {
    std::cout << "--- test_write_solution_latency_values ---\n";
    // Use Example 1B from PROBLEM.md: fused {Op0, Op1} at [128,128,1].
    // Expected latency = 3276.8.
    std::string in_path  = tmp_path("chain");
    std::string out_path = tmp_path("sol_lat");

    auto p = read_problem(in_path);
    DAG d = DAG::build(p);
    auto sg = *Subgraph::create(p, d, {0, 1});
    TileConfig cfg{128, 128, 1, SnakeDir::None};
    Solution sol(p, d, {{std::move(sg), cfg, {}}});
    write_solution(out_path, sol);

    std::ifstream f(out_path);
    auto j = json::parse(f);
    double lat = j["subgraph_latencies"][0].get<double>();
    CHECK_EQ("fused latency=3276.8", lat, 3276.8);
}

void test_write_solution_snake_traversal() {
    std::cout << "--- test_write_solution_snake_traversal ---\n";
    // Example 4B: MatMul [64,64,128] RowMajor on 2×2 grid.
    // Expected traversal_orders[0] = [0, 1, 3, 2]
    std::string mm_path = tmp_path("matmul4");
    write_json_file(mm_path, R"({
  "widths":  [128, 128, 128],
  "heights": [128, 128, 128],
  "inputs":  [[0, 1]],
  "outputs": [[2]],
  "base_costs": [1500],
  "op_types": ["MatMul"],
  "fast_memory_capacity": 25000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
})");
    auto p = read_problem(mm_path);
    DAG d = DAG::build(p);
    auto sg = *Subgraph::create(p, d, {0});
    TileConfig cfg{64, 64, 128, SnakeDir::RowMajor};
    Solution sol(p, d, {{std::move(sg), cfg, {}}});

    std::string out_path = tmp_path("sol_snake");
    write_solution(out_path, sol);

    std::ifstream f(out_path);
    auto j = json::parse(f);

    // Non-null traversal order for RowMajor on 2×2 grid
    CHECK("traversal order not null", !j["traversal_orders"][0].is_null());
    auto& order = j["traversal_orders"][0];
    CHECK_EQ_S("traversal length=4", order.size(), 4);
    int o0 = order[0].get<int>(), o1 = order[1].get<int>();
    int o2 = order[2].get<int>(), o3 = order[3].get<int>();
    CHECK("order[0]=0", o0 == 0);
    CHECK("order[1]=1", o1 == 1);
    CHECK("order[2]=3", o2 == 3);
    CHECK("order[3]=2", o3 == 2);

    // Latency matches 4B expected value
    double lat4b = j["subgraph_latencies"][0].get<double>();
    CHECK_EQ("4B latency=6548", lat4b, 6548.0);
}

void test_write_solution_retain() {
    std::cout << "--- test_write_solution_retain ---\n";
    // Example 3C: {Op0} retains T1, then {Op1, Op2}.
    std::string ex3_path = tmp_path("ex3");
    write_json_file(ex3_path, R"({
  "widths":  [128, 128, 128, 128],
  "heights": [128, 128, 128, 128],
  "inputs":  [[0], [1], [1, 2]],
  "outputs": [[1], [2], [3]],
  "base_costs": [1500, 1500, 1500],
  "op_types": ["Pointwise", "Pointwise", "Pointwise"],
  "fast_memory_capacity": 50000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
})");
    auto p = read_problem(ex3_path);
    DAG d = DAG::build(p);
    auto sg0 = *Subgraph::create(p, d, {0});
    auto sg1 = *Subgraph::create(p, d, {1, 2});
    TileConfig cfg{128, 128, 1, SnakeDir::None};
    auto c0 = sg0.compute_cost(cfg, {}, {1});
    auto c1 = sg1.compute_cost(cfg, {1}, {});
    Solution sol(p, d, {
        {std::move(sg0), cfg, {1}},
        {std::move(sg1), cfg, {}},
    });

    std::string out_path = tmp_path("sol_retain");
    write_solution(out_path, sol);

    std::ifstream f(out_path);
    auto j = json::parse(f);

    // Step 0 retains T1
    auto& ret0 = j["tensors_to_retain"][0];
    CHECK("step0 retains exactly 1 tensor", ret0.size() == 1);
    int retained_tensor = ret0[0].get<int>();
    CHECK("step0 retains T1", retained_tensor == 1);
    // Step 1 retains nothing
    CHECK("step1 retains nothing", j["tensors_to_retain"][1].empty());

    // Latencies match Example 3C: 1638.4 + 3000.0
    double rlat0 = j["subgraph_latencies"][0].get<double>();
    double rlat1 = j["subgraph_latencies"][1].get<double>();
    CHECK_EQ("3C step0 lat=1638.4", rlat0, 1638.4);
    CHECK_EQ("3C step1 lat=3000.0", rlat1, 3000.0);
}

void test_write_single_tile_snake_null() {
    std::cout << "--- test_write_single_tile_snake_null ---\n";
    // A single-tile MatMul with snake set should emit null traversal_order,
    // not [0] — a 1-element permutation adds no information.
    std::string path = tmp_path("single_tile_mm");
    write_json_file(path, R"({
  "widths":  [128, 128, 128],
  "heights": [128, 128, 128],
  "inputs":  [[0, 1]],
  "outputs": [[2]],
  "base_costs": [2000],
  "op_types": ["MatMul"],
  "fast_memory_capacity": 100000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
})");
    auto p = read_problem(path);
    DAG d = DAG::build(p);
    auto sg = *Subgraph::create(p, d, {0});
    // [128,128,128]: ntw=ntm=1. Single tile.
    TileConfig cfg{128, 128, 128, SnakeDir::RowMajor};
    Solution sol(p, d, {{std::move(sg), cfg, {}}});

    std::string out_path = tmp_path("sol_single_tile");
    write_solution(out_path, sol);
    std::ifstream f(out_path);
    auto j = json::parse(f);
    CHECK("single-tile snake emits null", j["traversal_orders"][0].is_null());
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Retainable classification (no files needed)
    test_retainable_basic_chain();
    test_retainable_graph_input_multi_consumer();
    test_retainable_graph_input_single_consumer_recomputation();
    test_retainable_too_large();
    test_retainable_exactly_capacity();
    test_retainable_graph_output_excluded();
    test_retainable_multi_output_op();
    test_retainable_no_tensors_if_all_too_large();

    // JSON read tests (write temp files, call read_problem)
    test_read_basic_chain();
    test_read_matmul_op();
    test_read_retainable_large_tensor_excluded();
    test_read_diamond_retainable();

    // write_solution round-trip
    test_write_solution_structure();
    test_write_solution_latency_values();
    test_write_solution_snake_traversal();
    test_write_solution_retain();
    test_write_single_tile_snake_null();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}