// large_subgraph_test.cpp
//
// Feasibility-focused tests for larger subgraphs:
//   * 3-MM and 4-MM chains with various granularities
//   * Diamond / fan-out patterns
//   * Interaction between native bound and working-set capacity
//   * Cases where only a specific granule fits memory
//
// Tests both feasible and infeasible configurations explicitly, and uses
// FMC sweeps to verify the feasibility boundary.

#include "core/dag.h"
#include "core/subgraph.h"
#include "core/types.h"
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ_I(const char* l, int64_t got, int64_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l
           << " got=" << got << " exp=" << exp << "\n"; }
}

static TileConfig TC(int64_t w, int64_t h, int64_t k,
                     SnakeDir s = SnakeDir::None) { return {w, h, k, s}; }

static Subgraph make_sg(const Problem& p, const DAG& d,
                        std::vector<size_t> ops) {
    auto sg = Subgraph::create(p, d, std::move(ops));
    if (!sg) { std::cerr << "FATAL: Subgraph::create failed\n"; std::exit(1); }
    return std::move(*sg);
}

static size_t entries_for(const Subgraph& sg, size_t tensor_id) {
    return sg.boundary_entries_for(tensor_id);
}

// ============================================================================
// 3-MM chain: A → B → C.
// All tensors 128×128. Verify structure, entry counts, feasibility.
// ============================================================================

void test_3mm_chain_basic() {
    std::cout << "--- test_3mm_chain_basic ---\n";
    // Op0: T0 × T1 → T2  (non-sink, ephemeral T2)
    // Op1: T2 × T3 → T4  (non-sink, ephemeral T4)
    // Op2: T4 × T5 → T6  (sink)
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},
                 {128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
             {OpType::MatMul, {2,3}, {4}, 2000},
             {OpType::MatMul, {4,5}, {6}, 2000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // Structure
    CHECK("T0 boundary input",  sg.boundary_inputs().count(0));
    CHECK("T1 boundary input",  sg.boundary_inputs().count(1));
    CHECK("T3 boundary input",  sg.boundary_inputs().count(3));
    CHECK("T5 boundary input",  sg.boundary_inputs().count(5));
    CHECK("T2 ephemeral",       sg.ephemeral().count(2));
    CHECK("T4 ephemeral",       sg.ephemeral().count(4));
    CHECK("T6 boundary output", sg.boundary_outputs().count(6));

    // All 128×128 within native granule — valid at cfg=(128,128,128).
    CHECK("cfg=(128,128,128) valid", sg.is_valid_tiling(TC(128,128,128)));
    CHECK("cfg=(64,64,64) valid",    sg.is_valid_tiling(TC(64,64,64)));

    // Super-native rejected.
    CHECK("cfg=(256,128,128) super-native rejected",
          !sg.is_valid_tiling(TC(256,128,128)));
}

// Feasibility sweep: same subgraph, vary FMC. At sufficient FMC, cfg=native
// feasible; at tight FMC, only sub-native may fit.
void test_3mm_feasibility_fmc_sweep() {
    std::cout << "--- test_3mm_feasibility_fmc_sweep ---\n";
    // Tensors 128×128 so every op's native-size cfg fits the granule-fit
    // bound. WS at cfg=(128,128,128) is small.
    Problem base;
    base.tensors = {{128,128},{128,128},{128,128},
                    {128,128},{128,128},{128,128},{128,128}};
    base.ops = {{OpType::MatMul, {0,1}, {2}, 2000},
                {OpType::MatMul, {2,3}, {4}, 2000},
                {OpType::MatMul, {4,5}, {6}, 2000}};
    base.slow_memory_bandwidth = 10;
    base.native_w = 128; base.native_h = 128;

    // Large FMC — everything fits at cfg=(128,128,128).
    Problem p_large = base;
    p_large.fast_memory_capacity = 500000;
    DAG d_large = DAG::build(p_large);
    auto sg_large = make_sg(p_large, d_large, {0, 1, 2});
    auto c_large = sg_large.compute_cost(TC(128,128,128));
    CHECK("large FMC: cfg=(128,128,128) feasible", c_large.feasible);

    // Tight FMC — 3-MM chain has a FULL-role tensor (128² = 16384 always
    // resident), so FMC below that floor forces infeasibility at native.
    // FMC just above the floor: native may fail (larger WS) but sub-native
    // fits.
    Problem p_tight = base;
    p_tight.fast_memory_capacity = 40000;  // enough for FULL + 1-2 slices
    DAG d_tight = DAG::build(p_tight);
    auto sg_tight = make_sg(p_tight, d_tight, {0, 1, 2});
    auto c_native_tight = sg_tight.compute_cost(TC(128,128,128));
    auto best = sg_tight.best_cost();
    // native may or may not fit (depends on exact WS). What we verify:
    // best_cost always finds SOME feasible config as long as any exists.
    if (!c_native_tight.feasible) {
        CHECK("tight FMC: best_cost succeeds at sub-native",
              best.feasible);
    } else {
        // Native fits → also a valid outcome at this FMC.
        CHECK("tight FMC: best_cost feasible", best.feasible);
    }
}

// ============================================================================
// 4-MM chain, deeper propagation.
// ============================================================================

void test_4mm_chain() {
    std::cout << "--- test_4mm_chain ---\n";
    // Op0: T0 × T1 → T_01
    // Op1: T_01 × T2 → T_012
    // Op2: T_012 × T3 → T_0123
    // Op3: T_0123 × T4 → T5  (sink)
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128},
                 {128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {5}, 2000},
             {OpType::MatMul, {5,2}, {6}, 2000},
             {OpType::MatMul, {6,3}, {7}, 2000},
             {OpType::MatMul, {7,4}, {8}, 2000}};
    p.fast_memory_capacity = 1000000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg_opt = Subgraph::create(p, d, {0, 1, 2, 3});
    CHECK("4-MM chain subgraph created", sg_opt.has_value());
    if (!sg_opt) return;
    auto& sg = *sg_opt;

    CHECK_EQ_I("T5 ephemeral (0 boundary entries)", entries_for(sg, 5), 0);
    CHECK_EQ_I("T6 ephemeral",                      entries_for(sg, 6), 0);
    CHECK_EQ_I("T7 ephemeral",                      entries_for(sg, 7), 0);
    CHECK_EQ_I("T8 boundary output",                entries_for(sg, 8), 1);

    CHECK("cfg=(128,128,128) valid", sg.is_valid_tiling(TC(128,128,128)));
    CHECK("cfg=(64,64,64) valid",    sg.is_valid_tiling(TC(64,64,64)));
}

// ============================================================================
// Diamond: one input feeds two branches that merge at a final op.
// ============================================================================

void test_diamond_subgraph() {
    std::cout << "--- test_diamond_subgraph ---\n";
    // PW0: T0 → T1 (ephemeral, fan-out to two consumers)
    // PW_L: T1 → T2 (left branch)
    // PW_R: T1 → T3 (right branch)
    // PW_merge: T2, T3 → T4 (sink)
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise, {0},   {1}, 500},
             {OpType::Pointwise, {1},   {2}, 500},
             {OpType::Pointwise, {1},   {3}, 500},
             {OpType::Pointwise, {2,3}, {4}, 500}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2, 3});

    // T1 consumed by two PWs (same role → same tensor_tiling_ shape → dedups).
    // T1 is ephemeral (produced + consumed internally) → not in boundary_info_.
    CHECK("T1 ephemeral", sg.ephemeral().count(1));
    // T2, T3 are ephemeral (produced by PW_L/R, consumed by merge).
    CHECK("T2 ephemeral", sg.ephemeral().count(2));
    CHECK("T3 ephemeral", sg.ephemeral().count(3));
    // T0 boundary in, T4 boundary out.
    CHECK_EQ_I("T0 boundary entries", entries_for(sg, 0), 1);
    CHECK_EQ_I("T4 boundary entries", entries_for(sg, 4), 1);

    CHECK("cfg=(128,128,1) valid", sg.is_valid_tiling(TC(128,128,1)));
    CHECK("cfg=(64,64,1) valid",   sg.is_valid_tiling(TC(64,64,1)));
}

// ============================================================================
// Feasibility-flip via cfg: subgraph may be infeasible at native granule
// due to memory, but feasible at sub-native (fewer concurrent slices).
// Not always — but verify we're not accidentally pinning to a single cfg.
// ============================================================================

void test_feasibility_via_cfg_choice() {
    std::cout << "--- test_feasibility_via_cfg_choice ---\n";
    // Simple MM with large tensors. At cfg=(128,128,128) working set is
    // huge; at cfg=(128,128,32) split-K reduces RHS/LHS slice size.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // Should have a feasible config via best_cost.
    auto best = sg.best_cost();
    CHECK("MM feasible via best_cost", best.feasible);
    CHECK("best.w ≤ native", best.config.w <= 128);
    CHECK("best.k ≤ native", best.config.k <= 128);
}

// ============================================================================
// Infeasible subgraph: tensors too large to fit in fast memory at any
// native-or-smaller granule. best_cost returns infeasible.
// ============================================================================

void test_infeasible_specific_cfg() {
    std::cout << "--- test_infeasible_specific_cfg ---\n";
    // At cfg=(128,128,128) with 128² tensors, working set for a MM is
    // LHS+RHS+output = 3 * 16384 = 49152. Set FMC below this → cfg is
    // rejected as infeasible via working_set > FMC in compute_cost.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000}};
    p.fast_memory_capacity = 30000;  // < 49152
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    auto c_native = sg.compute_cost(TC(128,128,128));
    CHECK("cfg=(128,128,128) infeasible (WS > FMC)", !c_native.feasible);
    // Smaller cfg reduces slice size and may fit.
    auto best = sg.best_cost();
    CHECK("best_cost feasible at smaller cfg", best.feasible);
    CHECK("best cfg is smaller than native",
          best.config.w < 128 || best.config.h < 128 || best.config.k < 128);
}

// ============================================================================
// Mixed MM + PW at various granule levels.
// ============================================================================

void test_mixed_mm_pw_chain() {
    std::cout << "--- test_mixed_mm_pw_chain ---\n";
    // PW_pre: T0 → T1
    // MM:     T1 × T2 → T3
    // PW_post: T3 → T4 (sink)
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise, {0},   {1}, 500},
             {OpType::MatMul,    {1,2}, {3}, 2000},
             {OpType::Pointwise, {3},   {4}, 500}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    // T1, T3 ephemeral; T0 + T2 inputs; T4 output.
    CHECK("T1 ephemeral (PW-produced)", sg.ephemeral().count(1));
    CHECK("T3 ephemeral (MM-produced)", sg.ephemeral().count(3));
    CHECK_EQ_I("T0 boundary",  entries_for(sg, 0), 1);
    CHECK_EQ_I("T2 boundary",  entries_for(sg, 2), 1);
    CHECK_EQ_I("T4 boundary",  entries_for(sg, 4), 1);

    // cfg at native should work (modulo PW-produced ephemeral granule-fit).
    CHECK("cfg=(128,128,128) valid", sg.is_valid_tiling(TC(128,128,128)));
}

// ============================================================================
// Multiple sinks (fan-out producing two boundary outputs). Verify each sink
// gets its own boundary_out entry.
// ============================================================================

void test_multi_sink_fan_out() {
    std::cout << "--- test_multi_sink_fan_out ---\n";
    // PW: T0 → T1 (ephemeral, fans out to two sink PWs)
    // sink_a: T1 → T2 (boundary output A)
    // sink_b: T1 → T3 (boundary output B)
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise, {0}, {1}, 500},
             {OpType::Pointwise, {1}, {2}, 500},
             {OpType::Pointwise, {1}, {3}, 500}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0, 1, 2});

    CHECK("T2 boundary output", sg.boundary_outputs().count(2));
    CHECK("T3 boundary output", sg.boundary_outputs().count(3));
    CHECK_EQ_I("T2 entries", entries_for(sg, 2), 1);
    CHECK_EQ_I("T3 entries", entries_for(sg, 3), 1);
    CHECK("cfg=(128,128,1) valid", sg.is_valid_tiling(TC(128,128,1)));
}

// ============================================================================
// Non-square tensor chain — heights ≠ widths.
// ============================================================================

void test_non_square_chain() {
    std::cout << "--- test_non_square_chain ---\n";
    // MM: T0(256×128) × T1(128×256) → T2(128×128)
    //   LHS shape: height=256, width=128=K. RHS: height=128=K, width=256.
    //   Wait, output width = RHS.width = 256, output height = LHS.height = 256.
    // Let me redo: T0=256×128 (h=256, w=128), T1=128×256 (h=128, w=256).
    // Convention: output shape is (LHS.h, RHS.w) = (256, 256).
    // (Output tensor T2 width=256, height=256.)
    Problem p;
    p.tensors = {{128,256},   // T0: w=128, h=256
                 {256,128},   // T1: w=256, h=128 (height=K=128)
                 {256,256}};  // T2: w=256, h=256
    p.ops = {{OpType::MatMul, {0,1}, {2}, 2000}};
    p.fast_memory_capacity = 500000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto sg = make_sg(p, d, {0});

    // cfg at native on all axes.
    CHECK("cfg=(128,128,128) valid",  sg.is_valid_tiling(TC(128,128,128)));
    // Super-native any axis → rejected.
    CHECK("cfg=(256,128,128) rejected (super-native)",
          !sg.is_valid_tiling(TC(256,128,128)));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    test_3mm_chain_basic();
    test_3mm_feasibility_fmc_sweep();
    test_4mm_chain();
    test_diamond_subgraph();
    test_feasibility_via_cfg_choice();
    test_infeasible_specific_cfg();
    test_mixed_mm_pw_chain();
    test_multi_sink_fan_out();
    test_non_square_chain();

    std::cout << "\n=== " << g_pass << " passed, " << g_fail
              << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
