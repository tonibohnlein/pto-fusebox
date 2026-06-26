// dag_test.cpp — Targeted tests for DAG construction, reachability,
// topo_position, and reachable_from_mask.
//
// Build: make dag_test
// Run:   ./dag_test

#include "core/dag.h"
#include "core/types.h"
#include <cstdio>
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ_S(const char* l, size_t got, size_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l
           << " got=" << got << " exp=" << exp << "\n"; }
}

static Problem make_chain(int n) {
    Problem p;
    for (int i = 0; i <= n; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < n; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)}});
    p.fast_memory_capacity = 500000;

    return p;
}

// ============================================================================
// 1. can_reach
// ============================================================================

void test_can_reach_self() {
    std::cout << "--- test_can_reach_self ---\n";
    auto p = make_chain(3);
    DAG d = DAG::build(p);
    for (size_t i = 0; i < d.num_ops; i++) {
        char buf[64]; snprintf(buf, sizeof(buf), "can_reach(%zu,%zu)", i, i);
        CHECK(buf, d.can_reach(i, i));
    }
}

void test_can_reach_oob() {
    std::cout << "--- test_can_reach_oob ---\n";
    auto p = make_chain(3);
    DAG d = DAG::build(p);
    CHECK("OOB u",    !d.can_reach(100, 0));
    CHECK("OOB v",    !d.can_reach(0, 100));
    CHECK("OOB both", !d.can_reach(100, 200));
}

void test_can_reach_chain() {
    std::cout << "--- test_can_reach_chain ---\n";
    auto p = make_chain(4);
    DAG d = DAG::build(p);
    CHECK("0→1",   d.can_reach(0,1));
    CHECK("0→3",   d.can_reach(0,3));
    CHECK("2→3",   d.can_reach(2,3));
    CHECK("1!→0", !d.can_reach(1,0));
    CHECK("3!→0", !d.can_reach(3,0));
}

void test_can_reach_diamond() {
    std::cout << "--- test_can_reach_diamond ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1}},
             {OpType::Pointwise,{0},{2}},
             {OpType::Pointwise,{1,2},{3}}};
    p.fast_memory_capacity = 500000;

    DAG d = DAG::build(p);
    CHECK("Op0→Op2", d.can_reach(0,2));
    CHECK("Op1→Op2", d.can_reach(1,2));
    CHECK("Op0!→Op1", !d.can_reach(0,1));
    CHECK("Op2!→Op0", !d.can_reach(2,0));
}

// ============================================================================
// 2. op_neighbors — co-consumer edges
// ============================================================================

void test_op_neighbors_co_consumer() {
    std::cout << "--- test_op_neighbors_co_consumer ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1}},
             {OpType::Pointwise,{0},{2}}};
    p.fast_memory_capacity = 500000;

    DAG d = DAG::build(p);
    auto& n0 = d.op_neighbors[0];
    auto& n1 = d.op_neighbors[1];
    CHECK("Op0 nbr Op1", std::find(n0.begin(),n0.end(),(size_t)1)!=n0.end());
    CHECK("Op1 nbr Op0", std::find(n1.begin(),n1.end(),(size_t)0)!=n1.end());
}

void test_op_neighbors_dag_edges() {
    std::cout << "--- test_op_neighbors_dag_edges ---\n";
    auto p = make_chain(3);
    DAG d = DAG::build(p);
    auto& n0 = d.op_neighbors[0];
    auto& n1 = d.op_neighbors[1];
    auto& n2 = d.op_neighbors[2];
    CHECK("Op0 nbr Op1", std::find(n0.begin(),n0.end(),(size_t)1)!=n0.end());
    CHECK("Op1 nbr Op0", std::find(n1.begin(),n1.end(),(size_t)0)!=n1.end());
    CHECK("Op1 nbr Op2", std::find(n1.begin(),n1.end(),(size_t)2)!=n1.end());
    // Op0 and Op2 are NOT direct neighbors
    CHECK("Op0 not nbr Op2", std::find(n0.begin(),n0.end(),(size_t)2)==n0.end());
}

// ============================================================================
// 3. tensor_producer / tensor_consumers / graph_inputs / graph_outputs
// ============================================================================

void test_tensor_classification() {
    std::cout << "--- test_tensor_classification ---\n";
    auto p = make_chain(2);
    DAG d = DAG::build(p);
    CHECK("T0 producer=-1", d.tensor_producer[0] == -1);
    CHECK("T1 producer=0",  d.tensor_producer[1] == 0);
    CHECK("T2 producer=1",  d.tensor_producer[2] == 1);
    CHECK_EQ_S("T0 1 consumer",  d.tensor_consumers[0].size(), 1);
    CHECK("T0 consumed by Op0", d.tensor_consumers[0][0] == 0);
    CHECK_EQ_S("T2 0 consumers", d.tensor_consumers[2].size(), 0);
    CHECK_EQ_S("1 graph input",  d.graph_inputs.size(),  1);
    CHECK("T0 is graph input",   d.graph_inputs[0] == 0);
    CHECK_EQ_S("1 graph output", d.graph_outputs.size(), 1);
    CHECK("T2 is graph output",  d.graph_outputs[0] == 2);
}

void test_tensor_multi_consumer() {
    std::cout << "--- test_tensor_multi_consumer ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1}},
             {OpType::Pointwise,{0},{2}}};
    p.fast_memory_capacity = 500000;

    DAG d = DAG::build(p);
    CHECK_EQ_S("T0 has 2 consumers", d.tensor_consumers[0].size(), 2);
    auto& c = d.tensor_consumers[0];
    CHECK("T0 consumed by Op0", std::find(c.begin(),c.end(),(size_t)0)!=c.end());
    CHECK("T0 consumed by Op1", std::find(c.begin(),c.end(),(size_t)1)!=c.end());
    // T1 and T2 are graph outputs
    CHECK_EQ_S("2 graph outputs", d.graph_outputs.size(), 2);
}

void test_tensor_matmul() {
    std::cout << "--- test_tensor_matmul ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2}}};
    p.fast_memory_capacity = 100000;

    DAG d = DAG::build(p);
    CHECK_EQ_S("2 graph inputs", d.graph_inputs.size(), 2);
    CHECK_EQ_S("1 graph output", d.graph_outputs.size(), 1);
    CHECK("T0 producer=-1", d.tensor_producer[0] == -1);
    CHECK("T1 producer=-1", d.tensor_producer[1] == -1);
    CHECK("T2 producer=0",  d.tensor_producer[2] == 0);
}

// ============================================================================
// 4. topo_sort
// ============================================================================

void test_topo_sort_disconnected() {
    std::cout << "--- test_topo_sort_disconnected ---\n";
    // Two independent chains Op0→Op1 and Op2→Op3.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1}},
             {OpType::Pointwise,{1},{2}},
             {OpType::Pointwise,{3},{4}},
             {OpType::Pointwise,{4},{5}}};
    p.fast_memory_capacity = 500000;

    DAG d = DAG::build(p);
    auto order = d.topo_sort();
    CHECK_EQ_S("covers all 4 ops", order.size(), 4);
    FlatSet<size_t> seen(order.begin(), order.end());
    CHECK_EQ_S("no duplicates", seen.size(), 4);
    // Within each chain, relative order preserved
    auto pos = [&](size_t op){ return std::find(order.begin(),order.end(),op)-order.begin(); };
    CHECK("Op0 before Op1", pos(0) < pos(1));
    CHECK("Op2 before Op3", pos(2) < pos(3));
}

void test_topo_sort_valid_order() {
    std::cout << "--- test_topo_sort_valid_order ---\n";
    // Fan-in: T1→Op1→T2, T1→Op2→T3, T2+T3→Op3→T4
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1}},
             {OpType::Pointwise,{1},{2}},
             {OpType::Pointwise,{1},{3}},
             {OpType::Pointwise,{2,3},{4}}};
    p.fast_memory_capacity = 500000;

    DAG d = DAG::build(p);
    auto order = d.topo_sort();
    std::vector<int> pos(d.num_ops);
    for (int i = 0; i < (int)order.size(); i++) pos[order[i]] = i;
    bool valid = true;
    for (size_t u = 0; u < d.num_ops; u++)
        for (auto v : d.op_succs[u])
            if (pos[u] >= pos[v]) { valid = false; break; }
    CHECK("topo order respects all edges", valid);
}

// ============================================================================
// 5. topo_position / topological_order (cached)
// ============================================================================

void test_topo_position_chain() {
    std::cout << "--- test_topo_position_chain ---\n";
    auto p = make_chain(4);
    DAG d = DAG::build(p);
    CHECK("pos(0)<pos(1)", d.topo_position(0)<d.topo_position(1));
    CHECK("pos(1)<pos(2)", d.topo_position(1)<d.topo_position(2));
    CHECK("pos(2)<pos(3)", d.topo_position(2)<d.topo_position(3));
    // topo_order_ is the inverse
    const auto& order = d.topological_order();
    CHECK_EQ_S("order size=4", order.size(), 4);
    for (size_t i = 0; i < order.size(); i++)
        CHECK("pos(order[i])==i", d.topo_position(order[i]) == i);
}

void test_topo_position_diamond() {
    std::cout << "--- test_topo_position_diamond ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1}},
             {OpType::Pointwise,{1},{2}},
             {OpType::Pointwise,{1},{3}},
             {OpType::Pointwise,{2,3},{4}}};
    p.fast_memory_capacity = 500000;

    DAG d = DAG::build(p);
    CHECK("Op0 before Op1", d.topo_position(0)<d.topo_position(1));
    CHECK("Op0 before Op2", d.topo_position(0)<d.topo_position(2));
    CHECK("Op1 before Op3", d.topo_position(1)<d.topo_position(3));
    CHECK("Op2 before Op3", d.topo_position(2)<d.topo_position(3));
}

// ============================================================================

// ============================================================================
// 7. merge_creates_cycle cross-check with topo_position
// ============================================================================

void test_merge_cycle_vs_topo_pos() {
    std::cout << "--- test_merge_cycle_vs_topo_pos ---\n";
    // On a strict linear chain, merge(a,b) creates a cycle iff |pos(a)-pos(b)| > 1.
    auto p = make_chain(5);
    DAG d = DAG::build(p);
    bool all_ok = true;
    for (size_t a = 0; a < d.num_ops; a++) {
        for (size_t b = 0; b < d.num_ops; b++) {
            if (a == b) continue;
            bool cycle = d.merge_creates_cycle({a},{b});
            size_t pa = d.topo_position(a), pb = d.topo_position(b);
            size_t gap = (pa>pb)?(pa-pb):(pb-pa);
            bool expect = (gap > 1);
            if (cycle != expect) {
                char buf[128];
                snprintf(buf, sizeof(buf),
                         "merge({%zu},{%zu}) cycle=%d exp=%d gap=%zu", a,b,(int)cycle,(int)expect,gap);
                g_fail++; std::cout << "  FAIL: " << buf << "\n";
                all_ok = false;
            }
        }
    }
    if (all_ok) { g_pass++; std::cout << "  OK: merge_creates_cycle consistent with topo_pos\n"; }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    test_can_reach_self();
    test_can_reach_oob();
    test_can_reach_chain();
    test_can_reach_diamond();

    test_op_neighbors_co_consumer();
    test_op_neighbors_dag_edges();

    test_tensor_classification();
    test_tensor_multi_consumer();
    test_tensor_matmul();

    test_topo_sort_disconnected();
    test_topo_sort_valid_order();

    test_topo_position_chain();
    test_topo_position_diamond();


    test_merge_cycle_vs_topo_pos();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass+g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}