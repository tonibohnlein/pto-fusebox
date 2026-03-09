// unit_tests.cpp — Focused unit tests for Subgraph, DAG, cost, Solution
// Build: make unit_tests

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "solution/solution.h"
#include "core/cost.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>

static int g_pass = 0, g_fail = 0;

static void CHECK(const char* label, bool cond) {
    if (cond) g_pass++; else { g_fail++; std::cout << "  FAIL: " << label << "\n"; }
}
static void CHECK_EQ(const char* label, double got, double exp, double tol = 0.1) {
    if (std::abs(got - exp) < tol) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label << " got=" << got << " exp=" << exp << "\n"; }
}
static void CHECK_EQ_I(const char* label, int64_t got, int64_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label << " got=" << got << " exp=" << exp << "\n"; }
}
static void CHECK_EQ_S(const char* label, size_t got, size_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label << " got=" << got << " exp=" << exp << "\n"; }
}

// Helpers
static Problem make_chain_pw() {
    Problem p; p.tensors={{128,128},{128,128},{128,128}};
    p.ops={{OpType::Pointwise,{0},{1},1000},{OpType::Pointwise,{1},{2},500}};
    p.fast_memory_capacity=50000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
    return p;
}
static Problem make_diamond() {
    Problem p; p.tensors={{128,128},{128,128},{128,128},{128,128}};
    p.ops={{OpType::Pointwise,{0},{1},1500},{OpType::Pointwise,{1},{2},1500},{OpType::Pointwise,{1,2},{3},1500}};
    p.fast_memory_capacity=50000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
    return p;
}
static Problem make_single_mm(int64_t sz=256, int64_t cap=100000) {
    Problem p; p.tensors={{sz,sz},{sz,sz},{sz,sz}};
    p.ops={{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity=cap;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
    return p;
}

TileConfig N(int64_t w, int64_t h, int64_t k) { return {w,h,k,SnakeDir::None}; }
TileConfig RS(int64_t w, int64_t h, int64_t k) { return {w,h,k,SnakeDir::RowMajor}; }
TileConfig TC(int64_t w, int64_t h, int64_t k, SnakeDir s = SnakeDir::None) { return {w,h,k,s}; }

// ==================== Subgraph creation ====================

void test_create_single_op() {
    std::cout << "--- test_create_single_op ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto sg0 = Subgraph::create(p, d, {0});
    CHECK("Op0 valid", sg0.has_value());
    CHECK_EQ_S("Op0 sink", sg0->sink_tensor(), 1);
    auto sg1 = Subgraph::create(p, d, {1});
    CHECK("Op1 valid", sg1.has_value());
    CHECK_EQ_S("Op1 sink", sg1->sink_tensor(), 2);
}

void test_create_fused_chain() {
    std::cout << "--- test_create_fused_chain ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0, 1});
    CHECK("fused valid", sg.has_value());
    CHECK_EQ_S("fused sink", sg->sink_tensor(), 2);
    CHECK("T1 ephemeral", sg->ephemeral().count(1));
    CHECK("T0 boundary", sg->boundary_inputs().count(0));
}

void test_create_multi_sink_rejected() {
    std::cout << "--- test_create_multi_sink_rejected ---\n";
    Problem p; p.tensors={{128,128},{128,128},{128,128}};
    p.ops={{OpType::Pointwise,{0},{1},1000},{OpType::Pointwise,{0},{2},1000}};
    p.fast_memory_capacity=50000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
    DAG d = DAG::build(p);
    CHECK("multi-sink rejected", !Subgraph::create(p, d, {0, 1}).has_value());
    CHECK("Op0 alone valid", Subgraph::create(p, d, {0}).has_value());
}

void test_create_disconnected_rejected() {
    std::cout << "--- test_create_disconnected_rejected ---\n";
    // T0->Op0->T1, T2->Op1->T3 — disconnected
    Problem p; p.tensors={{128,128},{128,128},{128,128},{128,128},{128,128}};
    // Op0: T0->T1, Op1: T2->T3, Op2: T1,T3->T4 (to make single sink if fusing 0,1,2)
    p.ops={{OpType::Pointwise,{0},{1},100},{OpType::Pointwise,{2},{3},100},{OpType::Pointwise,{1,3},{4},100}};
    p.fast_memory_capacity=50000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
    DAG d = DAG::build(p);
    // {0,1} are disconnected (no edge between them)
    CHECK("disconnected rejected", !Subgraph::create(p, d, {0, 1}).has_value());
    // {0,2} are connected via T1
    CHECK("{0,2} connected", Subgraph::create(p, d, {0, 2}).has_value());
    // {0,1,2} is connected (0-2 via T1, 1-2 via T3)
    CHECK("{0,1,2} connected", Subgraph::create(p, d, {0, 1, 2}).has_value());
}

void test_create_diamond_fusions() {
    std::cout << "--- test_create_diamond_fusions ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);

    auto sg01 = Subgraph::create(p, d, {0, 1});
    CHECK("{0,1} valid", sg01.has_value());
    CHECK("T1 ephemeral in {0,1}", sg01->ephemeral().count(1));

    auto sg12 = Subgraph::create(p, d, {1, 2});
    CHECK("{1,2} valid", sg12.has_value());
    CHECK("T2 ephemeral in {1,2}", sg12->ephemeral().count(2));
    CHECK("T1 boundary of {1,2}", sg12->boundary_inputs().count(1));

    auto sg02 = Subgraph::create(p, d, {0, 2});
    CHECK("{0,2} valid", sg02.has_value());
    CHECK("T1 ephemeral in {0,2}", sg02->ephemeral().count(1));

    auto sg012 = Subgraph::create(p, d, {0, 1, 2});
    CHECK("{0,1,2} valid", sg012.has_value());
    CHECK("T1 eph in {0,1,2}", sg012->ephemeral().count(1));
    CHECK("T2 eph in {0,1,2}", sg012->ephemeral().count(2));
}

void test_create_empty_rejected() {
    std::cout << "--- test_create_empty_rejected ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    CHECK("empty rejected", !Subgraph::create(p, d, {}).has_value());
}

// ==================== Working set ====================

void test_ws_pointwise() {
    std::cout << "--- test_ws_pointwise ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});
    CHECK_EQ_I("PW ws", sg->working_set(N(128,128,1)), 128*128);
}

void test_ws_matmul() {
    std::cout << "--- test_ws_matmul ---\n";
    auto p = make_single_mm(256, 100000); DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});
    // h*K + k*w + h*w = 128*256 + 128*128 + 128*128
    CHECK_EQ_I("MM ws [128,128,128]", sg->working_set(N(128,128,128)), 65536);
    CHECK_EQ_I("MM ws [128,128,64]", sg->working_set(N(128,128,64)), 57344);
    CHECK_EQ_I("MM ws [64,64,128]", sg->working_set(N(64,64,128)), 28672);
}

void test_ws_retained() {
    std::cout << "--- test_ws_retained ---\n";
    // 256x256 MatMul: T0(LHS), T1(RHS), T2(output). All 256x256 = 65536.
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{128,128}};  // T3 = small external tensor
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // Base at [128,128,128]: T0_lhs(128*256=32768) + T1_rhs(128*128=16384) + T2_out(128*128=16384) = 65536
    int64_t base = sg->working_set(N(128,128,128));
    CHECK_EQ_I("base ws", base, 65536);

    // Retain T3 (128x128=16384, not a boundary input): adds FULL tensor size
    CHECK_EQ_I("retained adds full size", sg->working_set(N(128,128,128), {3}), base + 128*128);

    // Retain T0 (boundary input, LHS): full size 256×256 = 65536 replaces tile strip 128×256 = 32768
    // New ws = 65536 + 16384 + 16384 = 98304 (was 65536 with tile strip)
    CHECK_EQ_I("retain boundary uses full size", sg->working_set(N(128,128,128), {0}),
               256*256 + 128*128 + 128*128);

    // Retain T1 (boundary input, RHS): full size 256×256 = 65536 replaces tile strip 128×128 = 16384
    CHECK_EQ_I("retain RHS uses full size", sg->working_set(N(128,128,128), {1}),
               128*256 + 256*256 + 128*128);
}

// ==================== Feasibility ====================

void test_feasibility() {
    std::cout << "--- test_feasibility ---\n";
    auto p = make_single_mm(256, 30000); DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});
    CHECK("infeasible [128,128,128]", !sg->is_feasible(N(128,128,128)));
    CHECK("feasible [64,64,64]", sg->is_feasible(N(64,64,64)));
    auto r = sg->compute_cost(N(128,128,128));
    CHECK("cost infeasible", !r.feasible);
}

// ==================== Cost properties ====================

void test_snake_leq_null() {
    std::cout << "--- test_snake_leq_null ---\n";
    auto p = make_single_mm(512, 200000); DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});
    for (int64_t kk : {32,64,128,256,512}) {
        if (!sg->is_feasible(N(128,128,kk))) continue;
        auto cn = sg->compute_cost(N(128,128,kk));
        auto cr = sg->compute_cost(RS(128,128,kk));
        char buf[64]; snprintf(buf,64,"row<=null k=%ld",(long)kk);
        CHECK(buf, cr.latency <= cn.latency + 0.01);
    }
}

void test_pw_snake_irrelevant() {
    std::cout << "--- test_pw_snake_irrelevant ---\n";
    Problem p; p.tensors={{512,512},{512,512}};
    p.ops={{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity=100000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
    DAG d = DAG::build(p); auto sg = Subgraph::create(p, d, {0});
    auto cn = sg->compute_cost({128,128,1,SnakeDir::None});
    auto cr = sg->compute_cost({128,128,1,SnakeDir::RowMajor});
    auto cc = sg->compute_cost({128,128,1,SnakeDir::ColMajor});
    CHECK_EQ("PW null==row", cn.latency, cr.latency);
    CHECK_EQ("PW null==col", cn.latency, cc.latency);
}

void test_fusion_saves() {
    std::cout << "--- test_fusion_saves ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto s0 = Subgraph::create(p, d, {0}); auto s1 = Subgraph::create(p, d, {1});
    auto sf = Subgraph::create(p, d, {0, 1});
    double separate = s0->best_cost().latency + s1->best_cost().latency;
    double fused = sf->best_cost().latency;
    CHECK("fusion <= separate", fused <= separate + 0.01);
}

void test_retain_saves() {
    std::cout << "--- test_retain_saves ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto s0 = Subgraph::create(p, d, {0}); auto s1 = Subgraph::create(p, d, {1});
    double base = s0->best_cost({},{}).latency + s1->best_cost({},{}).latency;
    double retained = s0->best_cost({},{1}).latency + s1->best_cost({1},{}).latency;
    CHECK("retain saves", retained <= base + 0.01);
}

void test_best_cost() {
    std::cout << "--- test_best_cost ---\n";
    auto p = make_single_mm(256, 100000); DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});
    auto best = sg->best_cost();
    CHECK("best feasible", best.feasible);
    CHECK("best w>0", best.config.w > 0);
    CHECK("best h>0", best.config.h > 0);
    CHECK("best k>0", best.config.k > 0);
}

void test_best_cost_tight() {
    std::cout << "--- test_best_cost_tight ---\n";
    auto p = make_single_mm(512, 10000); DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});
    auto best = sg->best_cost();
    CHECK("tight feasible", best.feasible);
    CHECK("tight ws fits", best.working_set <= 10000);
}

// ==================== DAG ====================

void test_dag_chain() {
    std::cout << "--- test_dag_chain ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    CHECK("Op0 no preds", d.op_preds[0].empty());
    CHECK("Op0→Op1", d.op_succs[0].count(1));
    CHECK("Op1←Op0", d.op_preds[1].count(0));
    CHECK_EQ_S("graph inputs", d.graph_inputs.size(), 1);
    CHECK_EQ_S("graph outputs", d.graph_outputs.size(), 1);
    auto topo = d.topo_sort();
    CHECK("topo order",
        std::find(topo.begin(),topo.end(),(size_t)0) < std::find(topo.begin(),topo.end(),(size_t)1));
}

void test_dag_cycle() {
    std::cout << "--- test_dag_cycle ---\n";
    Problem p; p.tensors={{128,128},{128,128},{128,128},{128,128}};
    p.ops={{OpType::Pointwise,{0},{1},100},{OpType::Pointwise,{1},{2},100},{OpType::Pointwise,{2},{3},100}};
    p.fast_memory_capacity=50000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
    DAG d = DAG::build(p);
    CHECK("merge {0},{2} cycle", d.merge_creates_cycle({0}, {2}));
    CHECK("merge {0},{1} no cycle", !d.merge_creates_cycle({0}, {1}));
    CHECK("merge {1},{2} no cycle", !d.merge_creates_cycle({1}, {2}));
}

// ==================== Traversal ====================

void test_traversal() {
    std::cout << "--- test_traversal ---\n";
    auto tn = make_traversal(3, 2, SnakeDir::None);
    CHECK_EQ_S("none size", tn.size(), 6);
    for (int i=0;i<6;i++) CHECK("none order", tn[i]==i);

    auto tr = make_traversal(3, 2, SnakeDir::RowMajor);
    CHECK("row r0", tr[0]==0 && tr[1]==1 && tr[2]==2);
    CHECK("row r1 reversed", tr[3]==5 && tr[4]==4 && tr[5]==3);

    auto tc = make_traversal(2, 3, SnakeDir::ColMajor);
    CHECK("col c0", tc[0]==0 && tc[1]==2 && tc[2]==4);
    CHECK("col c1 reversed", tc[3]==5 && tc[4]==3 && tc[5]==1);

    for (auto dir : {SnakeDir::None, SnakeDir::RowMajor, SnakeDir::ColMajor}) {
        auto t = make_traversal(4, 4, dir);
        std::set<int> s(t.begin(), t.end());
        CHECK("is permutation", (int)s.size()==16 && *s.begin()==0 && *s.rbegin()==15);
    }
}

// ==================== Solution validation ====================

void test_solution_valid() {
    std::cout << "--- test_solution_valid ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto sg0 = *Subgraph::create(p, d, {0});
    auto sg1 = *Subgraph::create(p, d, {1});
    auto c0 = sg0.best_cost(); auto c1 = sg1.best_cost();
    std::vector<ScheduleStep> steps = {
        {std::move(sg0), c0.config, {}},
        {std::move(sg1), c1.config, {}},
    };
    Solution sol(p, d, std::move(steps));
    auto vr = sol.validate();
    CHECK("valid solution", vr.valid);
    CHECK("positive latency", sol.total_latency() > 0);
}

void test_solution_missing_op() {
    std::cout << "--- test_solution_missing_op ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto sg0 = *Subgraph::create(p, d, {0});
    auto c0 = sg0.best_cost();
    // Only Op0, missing Op1
    std::vector<ScheduleStep> steps = {{std::move(sg0), c0.config, {}}};
    Solution sol(p, d, std::move(steps));
    auto vr = sol.validate();
    CHECK("invalid: missing op", !vr.valid);
}

void test_solution_wrong_order() {
    std::cout << "--- test_solution_wrong_order ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto sg0 = *Subgraph::create(p, d, {0});
    auto sg1 = *Subgraph::create(p, d, {1});
    auto c0 = sg0.best_cost(); auto c1 = sg1.best_cost();
    // Wrong order: Op1 before Op0
    std::vector<ScheduleStep> steps = {
        {std::move(sg1), c1.config, {}},
        {std::move(sg0), c0.config, {}},
    };
    Solution sol(p, d, std::move(steps));
    auto vr = sol.validate();
    CHECK("invalid: wrong order", !vr.valid);
}

void test_solution_with_retain() {
    std::cout << "--- test_solution_with_retain ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto sg0 = *Subgraph::create(p, d, {0});
    auto sg1 = *Subgraph::create(p, d, {1});
    // Retain T1 from step 0
    auto c0 = sg0.best_cost({}, {1});
    auto c1 = sg1.best_cost({1}, {});
    std::vector<ScheduleStep> steps = {
        {std::move(sg0), c0.config, {1}},
        {std::move(sg1), c1.config, {}},
    };
    Solution sol(p, d, std::move(steps));
    auto vr = sol.validate();
    CHECK("valid with retain", vr.valid);
    CHECK("retained entering step 1", sol.retained_entering(1).count(1));
}

void test_solution_invalid_retain() {
    std::cout << "--- test_solution_invalid_retain ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto sg0 = *Subgraph::create(p, d, {0});
    auto sg1 = *Subgraph::create(p, d, {1});
    auto c0 = sg0.best_cost(); auto c1 = sg1.best_cost();
    // Retain T2 from step 0 — but T2 is not a boundary tensor of sg0
    std::vector<ScheduleStep> steps = {
        {std::move(sg0), c0.config, {2}},
        {std::move(sg1), c1.config, {}},
    };
    Solution sol(p, d, std::move(steps));
    auto vr = sol.validate();
    CHECK("invalid: bad retain", !vr.valid);
}

void test_solution_recomputation() {
    std::cout << "--- test_solution_recomputation ---\n";
    auto p = make_diamond(); DAG d = DAG::build(p);
    // 3B strategy: {0,1} then {0,2} — Op0 appears twice
    auto sg01 = *Subgraph::create(p, d, {0, 1});
    auto sg02 = *Subgraph::create(p, d, {0, 2});
    auto c01 = sg01.best_cost({}, {2});
    auto c02 = sg02.best_cost({2}, {});
    std::vector<ScheduleStep> steps = {
        {std::move(sg01), c01.config, {2}},
        {std::move(sg02), c02.config, {}},
    };
    Solution sol(p, d, std::move(steps));
    auto vr = sol.validate();
    CHECK("valid recomputation", vr.valid);
}

void test_solution_latency_consistent() {
    std::cout << "--- test_solution_latency_consistent ---\n";
    auto p = make_chain_pw(); DAG d = DAG::build(p);
    auto sg0 = *Subgraph::create(p, d, {0});
    auto sg1 = *Subgraph::create(p, d, {1});
    auto c0 = sg0.best_cost(); auto c1 = sg1.best_cost();
    std::vector<ScheduleStep> steps = {
        {std::move(sg0), c0.config, {}},
        {std::move(sg1), c1.config, {}},
    };
    Solution sol(p, d, std::move(steps));
    CHECK_EQ("total == sum of steps",
             sol.total_latency(),
             sol.step_latency(0) + sol.step_latency(1));
}

// ==================== Edge cases ====================

void test_single_tile() {
    std::cout << "--- test_single_tile ---\n";
    Problem p; p.tensors={{128,128},{128,128}};
    p.ops={{OpType::Pointwise,{0},{1},500}};
    p.fast_memory_capacity=50000;p.slow_memory_bandwidth=10;p.native_w=128;p.native_h=128;
    DAG d = DAG::build(p); auto sg = Subgraph::create(p, d, {0});
    auto r = sg->compute_cost(N(128,128,1));
    CHECK("single tile feasible", r.feasible);
    CHECK_EQ_I("tiles", r.num_spatial_tiles, 1);
    CHECK_EQ("latency", r.latency, 3276.8);
}

void test_mixed_K() {
    std::cout << "--- test_mixed_K ---\n";
    // Two MatMuls with different K dimensions, tested separately.
    // Op0: T0(256x128) @ T1(128x256) → T2(128x128), K_0 = T0.width = 256
    // Op1: T2(128x128) @ T3(128x128) → T4(128x128), K_1 = T2.width = 128
    Problem p;
    p.tensors = {{256,128},   // T0: LHS of Op0 (K=256)
                 {128,256},   // T1: RHS of Op0
                 {128,128},   // T2: output of Op0 / LHS of Op1 (K=128)
                 {128,128},   // T3: RHS of Op1
                 {128,128}};  // T4: output of Op1
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},   // K_0 = 256
             {OpType::MatMul, {2,3}, {4}, 1000}};   // K_1 = 128
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    // Each op separately gets the correct K
    auto sg0 = Subgraph::create(p, d, {0});
    CHECK("Op0 valid", sg0.has_value());
    CHECK_EQ_I("Op0 K=256", sg0->op_K(0), 256);
    CHECK_EQ_I("Op0 max_K=256", sg0->max_K(), 256);

    auto sg1 = Subgraph::create(p, d, {1});
    CHECK("Op1 valid", sg1.has_value());
    CHECK_EQ_I("Op1 K=128", sg1->op_K(1), 128);
    CHECK_EQ_I("Op1 max_K=128", sg1->max_K(), 128);

    // Working sets use per-op K
    // Op0 at [128,128,64]: LHS strip h*K_0 = 128*256 = 32768
    //   RHS strip k*w = 64*128 = 8192. Output 128*128 = 16384.
    //   Total = 57344
    CHECK_EQ_I("Op0 ws [128,128,64]", sg0->working_set(N(128,128,64)), 57344);

    // Op1 at [128,128,64]: LHS strip h*K_1 = 128*128 = 16384
    //   RHS strip k*w = 64*128 = 8192. Output 128*128 = 16384.
    //   Total = 40960
    CHECK_EQ_I("Op1 ws [128,128,64]", sg1->working_set(N(128,128,64)), 40960);

    // Different num_k_passes
    auto c0 = sg0->compute_cost(N(128,128,64));
    CHECK_EQ_I("Op0 k_passes", c0.num_k_passes, 4);  // 256/64=4

    auto c1 = sg1->compute_cost(N(128,128,64));
    CHECK_EQ_I("Op1 k_passes", c1.num_k_passes, 2);  // 128/64=2

    // Per-op compute scaling
    CHECK_EQ("Op0 comp/step", c0.compute_per_step, 2000.0 * 64.0 / 256.0);  // 500
    CHECK_EQ("Op1 comp/step", c1.compute_per_step, 1000.0 * 64.0 / 128.0);  // 500
}

// Test: MatMul + Pointwise fusion (different "op types" but PW has no K)
void test_matmul_pw_fusion_K() {
    std::cout << "--- test_matmul_pw_fusion_K ---\n";
    // Op0: MatMul T0 @ T1 → T2, K=256
    // Op1: Pointwise T2 → T3 (no K)
    Problem p;
    p.tensors = {{256,128}, {128,256}, {128,128}, {128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::Pointwise, {2}, {3}, 500}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    auto sg = Subgraph::create(p, d, {0, 1});
    CHECK("MM+PW fused valid", sg.has_value());
    CHECK_EQ_I("fused max_K", sg->max_K(), 256);
    CHECK("T2 ephemeral", sg->ephemeral().count(2));

    // num_k_passes from the MatMul's K=256
    auto c = sg->compute_cost(N(128,128,64));
    CHECK_EQ_I("fused k_passes", c.num_k_passes, 4);

    // Compute per step: MatMul = 2000*64/256=500 per k-step.
    // PW = 500 but runs once per tile (added at last k-step only).
    CHECK_EQ("fused comp/step (MM only)", c.compute_per_step, 500.0);
}

// ==================== Non-power-of-2 tiling ====================

void test_nonpow2_tiling() {
    std::cout << "--- test_nonpow2_tiling ---\n";
    // 768x768 MatMul. 768 = 256*3.
    // Power-of-2 divisors of 768: {1,2,4,8,16,32,64,128,256}.
    // All divisors also include: 3,6,12,24,48,96,192,384,768.
    // e.g. w=192 divides 768 but is NOT a power of 2.
    Problem p;
    p.tensors = {{768,768},{768,768},{768,768}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // Verify non-pow2 divisors are valid
    CHECK("192 divides 768", 768 % 192 == 0);
    CHECK("384 divides 768", 768 % 384 == 0);

    // Compare pow2 tile (128x128) vs non-pow2 tile (192x128)
    // At [128,128,128]: ws = 128*768 + 128*128 + 128*128 = 98304+16384+16384 = 131072
    // At [192,128,128]: ws = 128*768 + 128*192 + 128*192 = 98304+24576+24576 = 147456
    // Both fit in 200k.
    auto c_128 = sg->compute_cost(TC(128,128,128));
    auto c_192 = sg->compute_cost(TC(192,128,128));
    CHECK("128 feasible", c_128.feasible);
    CHECK("192 feasible", c_192.feasible);

    // 192 has fewer spatial tiles: (768/192)*(768/128) = 4*6 = 24 vs 6*6 = 36
    CHECK_EQ_I("128 tiles", c_128.num_spatial_tiles, 36);
    CHECK_EQ_I("192 tiles", c_192.num_spatial_tiles, 24);

    // best_cost should find something at least as good as 128x128
    auto best = sg->best_cost();
    CHECK("best feasible", best.feasible);
    CHECK("best <= 128x128", best.latency <= c_128.latency + 0.01);
    std::cout << "    128x128: tiles=" << c_128.num_spatial_tiles << " lat=" << c_128.latency << "\n";
    std::cout << "    192x128: tiles=" << c_192.num_spatial_tiles << " lat=" << c_192.latency << "\n";
    std::cout << "    best:    [" << best.config.w << "," << best.config.h << ","
              << best.config.k << "] tiles=" << best.num_spatial_tiles
              << " lat=" << best.latency << "\n";
}

// Test: non-power-of-2 K dimension
void test_nonpow2_K() {
    std::cout << "--- test_nonpow2_K ---\n";
    // K=96 = 32*3. Pow2 divisors of 96: {1,2,4,8,16,32}. Misses 3,6,12,24,48,96.
    // k=48 divides 96 but is not power of 2.
    Problem p;
    p.tensors = {{96,128},{128,96},{128,128}};  // LHS 96-wide → K=96
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    CHECK_EQ_I("K=96", sg->max_K(), 96);

    // k=48: 96/48=2 passes. k=32: 96/32=3 passes. Fewer passes = less overhead.
    auto c_32 = sg->compute_cost(TC(128,128,32));
    auto c_48 = sg->compute_cost(TC(128,128,48));
    auto c_96 = sg->compute_cost(TC(128,128,96));
    CHECK("k=32 feasible", c_32.feasible);
    CHECK("k=48 feasible", c_48.feasible);
    CHECK("k=96 feasible", c_96.feasible);
    CHECK_EQ_I("k=32 passes", c_32.num_k_passes, 3);
    CHECK_EQ_I("k=48 passes", c_48.num_k_passes, 2);
    CHECK_EQ_I("k=96 passes", c_96.num_k_passes, 1);

    auto best = sg->best_cost();
    CHECK("best uses all divisors", best.feasible);
    // best should be at least as good as k=32
    CHECK("best <= k=32", best.latency <= c_32.latency + 0.01);
    std::cout << "    k=32: " << c_32.num_k_passes << " passes, lat=" << c_32.latency << "\n";
    std::cout << "    k=48: " << c_48.num_k_passes << " passes, lat=" << c_48.latency << "\n";
    std::cout << "    k=96: " << c_96.num_k_passes << " passes, lat=" << c_96.latency << "\n";
    std::cout << "    best: k=" << best.config.k << " lat=" << best.latency << "\n";
}

// ==================== Retain mechanics ====================

// Working set: retained input uses full tensor size, not tile strip
void test_retain_ws_full_tensor() {
    std::cout << "--- test_retain_ws_full_tensor ---\n";
    // MatMul: T0(256x128) @ T1(128x256) -> T2(256x256). K=256.
    Problem p;
    p.tensors = {{256,128},{128,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // Without retain at [128,128,128]:
    // T0 LHS: h*K = 128*256 = 32768 (strip)
    // T1 RHS: k*w = 128*128 = 16384 (strip)
    // T2 out: h*w = 128*128 = 16384
    CHECK_EQ_I("ws no retain", sg->working_set(N(128,128,128)), 32768+16384+16384);

    // With T0 retained: full T0 = 256*128 = 32768 (happens to equal the strip here!)
    // Because T0 is 256x128 and h*K = 128*256 = 32768. Same for this case.
    CHECK_EQ_I("ws retain T0 (same as strip)", sg->working_set(N(128,128,128), {0}), 32768+16384+16384);

    // With T1 retained: full T1 = 128*256 = 32768. Strip was k*w = 128*128 = 16384.
    // Difference: +16384
    CHECK_EQ_I("ws retain T1 (larger)", sg->working_set(N(128,128,128), {1}), 32768+32768+16384);
}

// Transfer cost: retained input has zero load cost
void test_retain_zero_transfer() {
    std::cout << "--- test_retain_zero_transfer ---\n";
    // Single PW: T0(128x128) -> T1(128x128). 1 tile.
    Problem p;
    p.tensors = {{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // Without retain: load T0 (1638.4) + evict T1 (1638.4) = 3276.8
    // lat = max(1000, 3276.8) = 3276.8
    auto c_no = sg->compute_cost(N(128,128,1));
    CHECK_EQ("no retain lat", c_no.latency, 3276.8);

    // With T0 retained: no load, evict T1 (1638.4)
    // lat = max(1000, 1638.4) = 1638.4
    auto c_ret_in = sg->compute_cost(N(128,128,1), {0});
    CHECK_EQ("retain input: skip load", c_ret_in.latency, 1638.4);

    // With T1 retained (no evict): load T0 (1638.4), no evict
    // lat = max(1000, 1638.4) = 1638.4
    auto c_ret_out = sg->compute_cost(N(128,128,1), {}, {1});
    CHECK_EQ("retain output: skip evict", c_ret_out.latency, 1638.4);

    // Both retained: no load, no evict. lat = compute only = 1000
    auto c_both = sg->compute_cost(N(128,128,1), {0}, {1});
    CHECK_EQ("retain both: compute only", c_both.latency, 1000.0);
}

// Retain forces smaller tiles due to memory pressure
void test_retain_forces_smaller_tiles() {
    std::cout << "--- test_retain_forces_smaller_tiles ---\n";
    // MatMul T0(256x256) @ T1(256x256) -> T2(256x256). K=256.
    // cap = 70000.
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000}};
    p.fast_memory_capacity = 70000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // Without retain at [128,128,128]:
    // ws = 128*256 + 128*128 + 128*128 = 32768+16384+16384 = 65536 <= 70000. Feasible.
    auto c_no = sg->best_cost();
    CHECK("no retain feasible", c_no.feasible);
    std::cout << "    no retain: [" << c_no.config.w << "," << c_no.config.h << ","
              << c_no.config.k << "] lat=" << c_no.latency << "\n";

    // With T1 retained from prev: full T1 = 256*256 = 65536.
    // ws = 65536 + h*256 + h*w + h*w. At h=128,w=128,k=128:
    // 65536 + 32768 + 16384 + 16384 = 131072 > 70000. Infeasible!
    CHECK("retain T1 infeasible at [128,128,128]",
          !sg->is_feasible(N(128,128,128), {1}));

    // best_cost should find a smaller tiling
    auto c_ret = sg->best_cost({1});
    std::cout << "    retain T1: [" << c_ret.config.w << "," << c_ret.config.h << ","
              << c_ret.config.k << "] lat=" << c_ret.latency
              << " feasible=" << c_ret.feasible << "\n";

    if (c_ret.feasible) {
        // The retained version should have smaller tiles
        CHECK("forced smaller w or h or k",
              c_ret.config.w < c_no.config.w || c_ret.config.h < c_no.config.h ||
              c_ret.config.k < c_no.config.k);
    }
}

// Retain output: tiles accumulate, increasing working set at last tile
void test_retain_output_accumulation() {
    std::cout << "--- test_retain_output_accumulation ---\n";
    // PW: T0(256x256) -> T1(256x256). At [128,128,1]: 4 spatial tiles.
    // cap = 40000.
    Problem p;
    p.tensors = {{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000}};
    p.fast_memory_capacity = 40000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = Subgraph::create(p, d, {0});

    // Without retain: ws = h*w + h*w = 16384+16384 = 32768. Feasible.
    // (PW output is free/in-place, so actually ws = h*w(input) = 16384... 
    //  let me check)
    auto ws_no = sg->working_set(N(128,128,1));
    std::cout << "    ws no retain: " << ws_no << "\n";

    // With retain T1 (output): tiles accumulate. Full T1 = 65536.
    // PW output NOT counted in base ws, so retain adds full 65536.
    // ws = ws_no(16384) + full(65536) = 81920.
    auto ws_ret = sg->working_set(N(128,128,1), {}, {1});
    std::cout << "    ws retain output: " << ws_ret << "\n";
    // Retain adds full tensor size for PW output (current tile was not in ws)
    CHECK_EQ_I("accumulation cost", ws_ret - ws_no, 256*256);

    CHECK("retain output infeasible at cap=40000", 
          !sg->is_feasible(N(128,128,1), {}, {1}));
}

// End-to-end: retain decision with tiling re-optimization
void test_retain_reoptimization() {
    std::cout << "--- test_retain_reoptimization ---\n";
    // Chain: T0->Op0->T1->Op1->T2. All 128x128 (single tile). Cap=50000.
    // Perfect retain scenario: T1 fits, single tile, no accumulation issue.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    p.retainable_tensors = {0, 1, 2};
    DAG d = DAG::build(p);

    auto sg0 = Subgraph::create(p, d, {0});
    auto sg1 = Subgraph::create(p, d, {1});

    // Step 0 without retain: load T0 + evict T1 = 3276.8
    auto c0_no = sg0->best_cost();
    // Step 1 without retain: load T1 + evict T2 = 3276.8
    auto c1_no = sg1->best_cost();
    double total_no = c0_no.latency + c1_no.latency;

    // Step 0 retaining T1: load T0, no evict. lat = max(1000, 1638.4) = 1638.4
    auto c0_ret = sg0->best_cost({}, {1});
    // Step 1 with T1 retained: no load, evict T2. lat = max(1000, 1638.4) = 1638.4
    auto c1_ret = sg1->best_cost({1});
    double total_ret = c0_ret.latency + c1_ret.latency;

    CHECK("retain saves", total_ret < total_no - 0.01);
    CHECK_EQ("step0 retain", c0_ret.latency, 1638.4);
    CHECK_EQ("step1 retained", c1_ret.latency, 1638.4);
    CHECK_EQ("total no retain", total_no, 6553.6);
    CHECK_EQ("total with retain", total_ret, 3276.8);
    std::cout << "    no retain: " << total_no << " with retain: " << total_ret
              << " saved " << (int)(100*(total_no-total_ret)/total_no) << "%\n";
}

// ==================== Main ====================

int main() {
    test_create_single_op();
    test_create_fused_chain();
    test_create_multi_sink_rejected();
    test_create_disconnected_rejected();
    test_create_diamond_fusions();
    test_create_empty_rejected();

    test_ws_pointwise();
    test_ws_matmul();
    test_ws_retained();

    test_feasibility();

    test_snake_leq_null();
    test_pw_snake_irrelevant();
    test_fusion_saves();
    test_retain_saves();
    test_best_cost();
    test_best_cost_tight();

    test_dag_chain();
    test_dag_cycle();

    test_traversal();

    test_solution_valid();
    test_solution_missing_op();
    test_solution_wrong_order();
    test_solution_with_retain();
    test_solution_invalid_retain();
    test_solution_recomputation();
    test_solution_latency_consistent();

    test_single_tile();
    test_mixed_K();
    test_matmul_pw_fusion_K();
    test_nonpow2_tiling();
    test_nonpow2_K();

    test_retain_ws_full_tensor();
    test_retain_zero_transfer();
    test_retain_forces_smaller_tiles();
    test_retain_output_accumulation();
    test_retain_reoptimization();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}