#include "io/verify.h"
#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include <cmath>
#include <iostream>

static int g_pass = 0, g_total = 0;

static void check(const char* label, double got, double expected) {
    g_total++;
    bool ok = std::abs(got - expected) < 0.1;
    if (ok) g_pass++;
    std::cout << "  " << (ok ? "✓" : "✗") << " " << label
              << ": got=" << got << " exp=" << expected << "\n";
    if (!ok) std::cout << "    *** MISMATCH ***\n";
}

static Subgraph make_sg(const Problem& p, const DAG& d, std::vector<size_t> ops) {
    auto sg = Subgraph::create(p, d, std::move(ops));
    if (!sg) { std::cerr << "FATAL: Subgraph::create failed\n"; std::exit(1); }
    return std::move(*sg);
}

bool verify_examples() {
    g_pass = g_total = 0;

    std::cout << "=== EXAMPLE 1: Chained Pointwise 128x128 ===\n";
    {
        Problem p;
        p.tensors = {{128,128},{128,128},{128,128}};
        p.ops = {{OpType::Pointwise,{0},{1},1000},{OpType::Pointwise,{1},{2},100}};
        p.fast_memory_capacity = 35000; p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128;
        DAG d = DAG::build(p);

        auto s0 = make_sg(p, d, {0});
        auto s1 = make_sg(p, d, {1});
        TileConfig c128{128,128,1,SnakeDir::None};
        check("1A separate",
              s0.compute_cost(c128).latency + s1.compute_cost(c128).latency, 6553.6);

        auto sf = make_sg(p, d, {0,1});
        check("1B fused 128", sf.compute_cost(c128).latency, 3276.8);
        TileConfig c64{64,64,1,SnakeDir::None};
        check("1C fused 64", sf.compute_cost(c64).latency, 4400.0);
    }

    std::cout << "=== EXAMPLE 2: Chained Pointwise 256x256 ===\n";
    {
        Problem p;
        p.tensors = {{256,256},{256,256},{256,256}};
        p.ops = {{OpType::Pointwise,{0},{1},1000},{OpType::Pointwise,{1},{2},100}};
        p.fast_memory_capacity = 25000; p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128;
        DAG d = DAG::build(p);

        TileConfig c128{128,128,1,SnakeDir::None};
        auto s0 = make_sg(p, d, {0});
        auto s1 = make_sg(p, d, {1});
        check("2A separate",
              s0.compute_cost(c128).latency + s1.compute_cost(c128).latency, 26214.4);

        auto sf = make_sg(p, d, {0,1});
        check("2B fused", sf.compute_cost(c128).latency, 13107.2);
    }

    std::cout << "=== EXAMPLE 3: Diamond graph ===\n";
    {
        Problem p;
        p.tensors = {{128,128},{128,128},{128,128},{128,128}};
        p.ops = {{OpType::Pointwise,{0},{1},1500},
                 {OpType::Pointwise,{1},{2},1500},
                 {OpType::Pointwise,{1,2},{3},1500}};
        p.fast_memory_capacity = 50000; p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128;
        DAG d = DAG::build(p);
        TileConfig c128{128,128,1,SnakeDir::None};

        auto s0 = make_sg(p, d, {0});
        auto s1 = make_sg(p, d, {1});
        auto s2 = make_sg(p, d, {2});
        check("3A spill",
              s0.compute_cost(c128).latency + s1.compute_cost(c128).latency
              + s2.compute_cost(c128).latency, 11468.8);

        auto sb0 = make_sg(p, d, {0,1});
        auto sb1 = make_sg(p, d, {0,2});
        check("3B recompute",
              sb0.compute_cost(c128, {}, {2}).latency
              + sb1.compute_cost(c128, {2}, {}).latency, 6276.8);

        auto s12 = make_sg(p, d, {1,2});
        check("3C hybrid",
              s0.compute_cost(c128, {}, {1}).latency
              + s12.compute_cost(c128, {1}, {}).latency, 4638.4);
    }

    std::cout << "=== EXAMPLE 4: MatMul traversal ===\n";
    {
        Problem p;
        p.tensors = {{128,128},{128,128},{128,128}};
        p.ops = {{OpType::MatMul,{0,1},{2},1500}};
        p.fast_memory_capacity = 25000; p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128;
        DAG d = DAG::build(p);
        auto sg = make_sg(p, d, {0});

        check("4A null", sg.compute_cost({64,64,128,SnakeDir::None}).latency, 8192.0);
        check("4B snake", sg.compute_cost({64,64,128,SnakeDir::RowMajor}).latency, 6548.0);
    }

    std::cout << "=== EXAMPLE 5: Chained MatMul Split-K ===\n";
    {
        Problem p;
        p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
        p.ops = {{OpType::MatMul,{0,1},{3},2000},{OpType::MatMul,{3,2},{4},2000}};
        p.fast_memory_capacity = 45000; p.slow_memory_bandwidth = 10;
        p.native_w = 128; p.native_h = 128;
        DAG d = DAG::build(p);
        auto sg = make_sg(p, d, {0,1});

        g_total++;
        int64_t ws = sg.working_set({128,128,128,SnakeDir::None});
        if (ws > 45000) { g_pass++; std::cout << "  ✓ 5A OOM: ws=" << ws << "\n"; }
        else std::cout << "  ✗ 5A OOM: ws=" << ws << "\n";

        check("5B split-K", sg.compute_cost({128,128,32,SnakeDir::None}).latency, 6915.2);
    }

    std::cout << "\n" << g_pass << "/" << g_total << " passed\n";
    return g_pass == g_total;
}
