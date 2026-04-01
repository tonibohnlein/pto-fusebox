// move_gain_test.cpp — Thorough gain verification for all move types.
// For every generated move: apply to a copy, verify reported saving = actual delta.
// Tests cover: different tensor sizes, MatMul, compute/memory bound, positive/negative
// savings, tiling changes from moves, and multi-group interactions.

#include "core/types.h"
#include "core/dag.h"
#include "partition/partition.h"
#include "search/local_search.h"
#include "test_move_helpers.h"
#include "search/fm_search.h"
#include <iostream>
#include <cmath>

static int g_pass = 0, g_fail = 0;
static void CHECK_EQ(const char* l, double g, double e, double t = 0.5) {
    if (std::abs(g - e) < t) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << g << " exp=" << e << "\n"; }
}
static void CHECK(const char* l, bool c) {
    if (c) g_pass++; else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}

// Apply an FMMove to a copy and return (before, after) costs — uses apply_fm_move
static GainCheck verify_heap_move(Partition part, const FMMove& m) {
    return verify_move_gain(std::move(part), m);
}

// Verify ALL heap moves from a partition. Returns (checked, failed) counts.
static std::pair<int,int> verify_all_moves(const char* label, Partition& part,
                                            double floor = 1e18) {
    int checked = 0, bad = 0;
    // FMMove::Type ordering: STEAL=0, EJECT=1, RECOMPUTE=2, MERGE=3, INT_EJECT=4, SPLIT=5
    const char* names[] = {"STEAL", "EJECT", "RECOMPUTE", "MERGE", "INT_EJECT", "SPLIT"};
    int type_counts[6] = {};
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        auto moves = all_moves_for_group(part, gi, floor);
        for (auto& m : moves) {
            auto gc = verify_heap_move(part, m);
            if (!gc.applied) continue;
            checked++;
            type_counts[m.type]++;
            double actual = gc.before - gc.after;
            if (std::abs(gc.reported - actual) >= 0.5) {
                bad++;
                std::cout << "  FAIL: " << label << " " << names[m.type]
                          << " op=" << m.op << " ga=G" << m.ga << " gb=G" << m.gb
                          << " reported=" << gc.reported << " actual=" << actual << "\n";
            }
        }
    }
    std::cout << "  " << label << ": " << checked << " moves ("
              << type_counts[3] << "M " << type_counts[0] << "S "
              << type_counts[2] << "R " << type_counts[1] << "E "
              << type_counts[4] << "IE " << type_counts[5] << "SP) "
              << bad << " bad\n";
    return {checked, bad};
}

// Verify FM moves for all border ops
static std::pair<int,int> verify_all_fm(const char* label, Partition& part) {
    int checked = 0, bad = 0;
    const char* names[] = {"STEAL", "EJECT", "RECOMPUTE", "MERGE", "INT_EJECT", "SPLIT"};
    int type_counts[6] = {};

    // Check both border ops AND internal ops from large groups
    FlatSet<size_t> ops_to_check;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        for (auto op : part.border_ops(gi)) ops_to_check.insert(op);
        if (part.groups[gi].ops.size() >= 3)
            for (auto op : part.internal_ops(gi)) ops_to_check.insert(op);
    }

    for (auto op : ops_to_check) {
        auto m = best_move_for(part, op);
        if (!m.valid()) continue;
        Partition p2 = part;
        p2.rebuild_index();
        double before = p2.total_cost();
        auto affected = apply_fm_move(p2, m);
        if (affected.empty()) continue;
        double after = p2.total_cost();
        checked++;
        int ti = (int)m.type;
        if (ti >= 0 && ti < 6) type_counts[ti]++;
        double actual = before - after;
        if (std::abs(m.saving - actual) >= 0.5) {
            bad++;
            std::cout << "  FAIL: " << label << " FM " << names[ti]
                      << " op=" << m.op << " saving=" << m.saving
                      << " actual=" << actual << "\n";
        }
    }
    std::cout << "  " << label << " FM: " << checked << " moves ("
              << type_counts[0] << "S " << type_counts[1] << "E "
              << type_counts[2] << "R " << type_counts[3] << "M "
              << type_counts[4] << "IE " << type_counts[5] << "SP) "
              << bad << " bad\n";
    return {checked, bad};
}

static void tally(std::pair<int,int> r) {
    if (r.first > 0 && r.second == 0) g_pass++;
    else if (r.first > 0) g_fail++;
    // 0 moves checked = not counted (informational)
}

// ==================== Test scenarios ====================

// 1. Different tensor sizes → merge actually changes cost
void test_asymmetric_chain() {
    std::cout << "=== test_asymmetric_chain ===\n";
    // T0(128x128) → Op0 → T1(256x128) → Op1 → T2(256x256) → Op2 → T3(128x128)
    // Different sizes → different load/evict costs → merges have real savings/costs
    Problem p;
    p.tensors = {{128,128},{256,128},{256,256},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000}};
    p.fast_memory_capacity = 100000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    tally(verify_all_moves("asym_trivial", part));
    tally(verify_all_fm("asym_trivial", part));
}

// 2. MatMul chain: steal/merge changes split-K behavior
void test_matmul_chain_varied_K() {
    std::cout << "=== test_matmul_chain_varied_K ===\n";
    // Op0: T0(256x128) @ T1(128x256) → T2(256x256). K=256 (T0.width)
    // Op1: T2(256x256) @ T3(256x128) → T4(256x128). K=256 (T2.width)
    // Singletons → merges/steals change tiling due to fused working set
    Problem p;
    p.tensors = {{256,128},{128,256},{256,256},{256,128},{256,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},3000},
             {OpType::MatMul,{2,3},{4},3000}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    tally(verify_all_moves("mm_varied_K", part));
    tally(verify_all_fm("mm_varied_K", part));
}

// 3. Merge causes tiling change: fused group needs smaller tiles → negative saving
void test_merge_forces_smaller_tiles() {
    std::cout << "=== test_merge_forces_smaller_tiles ===\n";
    // Tensors 512x512. Cap=280000. Single PW: ws = 128*128 = 16384 → fits at [128,128].
    // Fused 2 PW: ws = 128*128 = 16384 → still fits. Same cost.
    // But with 3 PW fused: still valid. Let me use tight cap.
    // 
    // Actually for PW, ws doesn't grow with fusion (output in-place).
    // For MatMul: ws grows. Let me use MatMul.
    //
    // Op0: MM T0(512x256)@T1(256x512)→T2(512x512). K=256.
    // At [128,128,256]: ws = 128*256 + 256*128 + 128*128 = 65536 ≤ 80000. Feasible.
    // Op1: MM T2(512x512)@T3(512x256)→T4(512x256). K=512.
    // At [128,128,256]: ws = 128*512 + 256*128 + 128*128 = 114688. Too big.
    // Singleton: different tiling.
    //
    // Merge {Op0,Op1}: T2 ephemeral. Output = T4(512x256).
    // ws at [128,128,128]: T0_lhs(128*256)=32768, T1_rhs(128*128)=16384,
    //   T3_rhs(128*128)=16384, T4_out(128*128)=16384 = 81920 > 80000 → OOM!
    // Need even smaller tiles → worse compute.
    Problem p;
    p.tensors = {{512,256},{256,512},{512,512},{512,256},{512,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},3000},
             {OpType::MatMul,{2,3},{4},3000}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    auto [n, bad] = verify_all_moves("tile_change", part);
    tally({n, bad});
    auto [nf, badf] = verify_all_fm("tile_change", part);
    tally({nf, badf});
    
    // Check that some moves are negative (merge forces smaller tiles → worse cost).
    // Under the new cost model (only sink ops divide by nk), non-sink ops in a fused
    // group no longer receive the nk discount, so merging these compute-heavy MM ops
    // is unprofitable — all merge moves from the trivial partition are negative.
    int pos = 0, neg = 0;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        for (auto& m : all_moves_for_group(part, gi, 1e18)) {
            if (m.saving > 0.01) pos++;
            else if (m.saving < -0.01) neg++;
        }
    }
    CHECK("has negative moves", neg > 0);
    std::cout << "  " << pos << " positive, " << neg << " negative moves\n";
}

// 4. Eject from fused group with multiple successors
void test_eject_with_external_successor() {
    std::cout << "=== test_eject_with_external_successor ===\n";
    // Op0→T1→Op1→T2. Op2(T1)→T3. {Op0,Op1} fused, {Op2} separate.
    // Eject Op1: remainder {Op0}, singleton {Op1}. Both valid.
    // Op2 still loads T1 from slow memory regardless.
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},2000},
             {OpType::Pointwise,{1},{2},2000},
             {OpType::Pointwise,{1},{3},2000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;
    part.rebuild_index();
    // G0={Op0,Op1}, G2={Op2}
    tally(verify_all_moves("eject_ext", part));
    tally(verify_all_fm("eject_ext", part));
}

// 5. Recompute in diamond: saves loading large intermediate
void test_recompute_saves_large_tensor() {
    std::cout << "=== test_recompute_saves_large_tensor ===\n";
    // Diamond: Op0: T0→T1. Op1: T1→T2. Op2: T1,T2→T3.
    // {Op0},{Op1},{Op2}. Recompute Op0 in G2 makes T1 ephemeral in {Op0,Op2}.
    // Saves: Op2 no longer loads T1 from slow memory (replaced by T0 load, same size).
    // Net saving depends on whether T1 eviction from G0 is also avoided.
    //
    // Actually for equal-sized PW tensors, recompute has zero saving (same tile loads).
    // Use high compute so merging {Op1,Op2} is compute-dominated and recompute helps:
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},5000},
             {OpType::Pointwise,{1},{2},5000},
             {OpType::Pointwise,{1,2},{3},5000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    // Start with {Op0,Op1} fused, {Op2} alone
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;
    part.rebuild_index();
    // G0={Op0,Op1}, G2={Op2}
    // Recompute Op0 into G2: {Op0,Op2} can make T1 ephemeral
    tally(verify_all_moves("recomp_diamond", part));
    tally(verify_all_fm("recomp_diamond", part));
    
    // Verify at least one recompute exists (even if negative)
    int recomp_count = 0;
    for (auto& m : all_moves_for_group(part, 2, 1e18)) {
        if (m.type == FMMove::RECOMPUTE) {
            recomp_count++;
            std::cout << "  recompute Op" << m.op << " in G" << m.ga 
                      << ": saving=" << m.saving << "\n";
        }
    }
    CHECK("recomputes generated", recomp_count > 0);
}

// 6. Steal with MatMul: moving a PW op changes which tensors are boundary
void test_steal_pw_from_mm_group() {
    std::cout << "=== test_steal_pw_from_mm_group ===\n";
    // Op0: PW T0→T1. Op1: MM T1@T2→T3. Start: {Op0,Op1},{...}
    // Steal Op0 out of the fused group.
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},500},
             {OpType::MatMul,{1,2},{3},3000}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;
    part.rebuild_index();
    // G0={Op0,Op1}. Op0 has no pred outside. But Op0 IS ejectable (it's a
    // source of the chain, boundary because it has no preds → not a boundary op!)
    // Actually boundary_neighbors checks DAG preds/succs OUTSIDE group.
    // Op0 has pred=none (T0 is graph input). Op0's succs: Op1 (inside). Not boundary!
    // So we need a third group for moves to exist.
    
    // Add Op2: PW T3→T4
    p.tensors.push_back({256,256}); // T4
    p.ops.push_back({OpType::Pointwise,{3},{4},500});
    d = DAG::build(p);
    part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;
    part.rebuild_index();
    // G0={Op0,Op1}, G2={Op2}
    tally(verify_all_moves("steal_pw_mm", part));
    tally(verify_all_fm("steal_pw_mm", part));
}

// 7. Multiple groups: verifying moves in a 5-group partition
void test_five_groups() {
    std::cout << "=== test_five_groups ===\n";
    // Linear chain of 5 PW ops with increasing tensor sizes
    Problem p;
    p.tensors = {{128,128},{256,128},{256,256},{512,256},{512,512},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000},
             {OpType::Pointwise,{3},{4},1000},
             {OpType::Pointwise,{4},{5},1000}};
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    tally(verify_all_moves("5groups", part));
    tally(verify_all_fm("5groups", part));
}

// 8. After greedy converges: verify remaining moves at optimum
void test_post_greedy_varied() {
    std::cout << "=== test_post_greedy_varied ===\n";
    // Run greedy on a non-trivial problem, then verify all remaining moves
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{128,128},{128,128},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},3000},
             {OpType::Pointwise,{2},{3},500},
             {OpType::Pointwise,{3},{4},500},
             {OpType::MatMul,{4,5},{0},3000}}; // cycle? no, output is T0 consumed by Op0
    // Fix: no cycles allowed
    p.tensors.push_back({256,256}); // T6
    p.ops[3] = {OpType::MatMul,{4,5},{6},3000};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = greedy_descent(Partition::trivial(p, d));
    std::cout << "  post-greedy: " << part.num_alive() << " groups\n";
    tally(verify_all_moves("post_greedy", part));
    tally(verify_all_fm("post_greedy", part));
}

// 9. Compute-bound: fusion helps (saves compute padding overhead)
void test_compute_bound_merge() {
    std::cout << "=== test_compute_bound_merge ===\n";
    // High base_cost, small tensors. Merging two ops at native granularity
    // costs compute = sum but saves memory transfers of intermediate.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},50000},  // very high compute
             {OpType::Pointwise,{1},{2},50000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    
    // Singleton: max(50000, 3276.8) = 50000 each. Total = 100000.
    // Fused: max(100000, 3276.8) = 100000. Total = 100000.
    // Same! Merge saving = 0. But the T1 transfer is eliminated.
    // Wait: fused loads T0(1638.4), evicts T2(1638.4). Compute = 100000.
    // Separate: loads T0(1638.4)+evicts T1(1638.4)=3276.8 → max(50k,3276.8)=50k
    //         + loads T1(1638.4)+evicts T2(1638.4)=3276.8 → max(50k,3276.8)=50k = 100k
    // Actually same total latency! The compute dominates so transfer doesn't matter.
    // With lower B:
    p.slow_memory_bandwidth = 1;
    d = DAG::build(p);
    part = Partition::trivial(p, d);
    // Singleton: max(50000, 2*16384) = max(50000, 32768) = 50000 each.
    // Fused: max(100000, 32768) = 100000. Same!
    // We need T1 transfer to matter. Use higher bandwidth so compute is bottleneck
    // but T1 transfer still adds to the sum.
    // Actually the issue is: for roofline, it's max(compute, memory).
    // When compute dominates, eliminating T1 transfer doesn't help.
    // Merge ONLY helps when memory dominates, or at least contributes.
    
    // Use balanced case: compute ≈ memory
    p.ops[0].base_cost = 3000;
    p.ops[1].base_cost = 3000;
    p.slow_memory_bandwidth = 10;
    d = DAG::build(p);
    part = Partition::trivial(p, d);
    tally(verify_all_moves("compute_bound", part));
    tally(verify_all_fm("compute_bound", part));
}

// 10. Fan-in: Op2 has two inputs from different ops
void test_fanin_graph() {
    std::cout << "=== test_fanin_graph ===\n";
    // Op0: T0→T1. Op1: T0→T2. Op2: T1,T2→T3. (Y-graph)
    // Merging {Op0,Op2} or {Op1,Op2} saves one input load.
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{0},{2},1000},
             {OpType::Pointwise,{1,2},{3},1000}};
    p.fast_memory_capacity = 200000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    tally(verify_all_moves("fanin", part));
    tally(verify_all_fm("fanin", part));
}

// 11. Mixed MM+PW with split-K: verify gain when fusing changes k
void test_mm_pw_splitk_gain() {
    std::cout << "=== test_mm_pw_splitk_gain ===\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{2},{3},1000}};
    p.fast_memory_capacity = 30000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    tally(verify_all_moves("mm_pw_splitk", part));
    tally(verify_all_fm("mm_pw_splitk", part));
}

// 12. Eject from 3-op fused group — both endpoints ejectable
void test_eject_from_chain3() {
    std::cout << "=== test_eject_from_chain3 ===\n";
    // {Op0,Op1,Op2} with Op3 separate. Op2 has successor Op3 outside group → ejectable.
    // Op0 has no pred outside → not ejectable (not a boundary op).
    // Actually: Op0 has pred=none (no DAG preds). So not boundary.
    // But if there's also {Op4} with Op4 consuming T0 (the same graph input as Op0):
    // We need Op0 to have a DAG neighbor outside the group.
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000},
             {OpType::Pointwise,{3},{4},500}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();
    // G0={0,1,2}, G3={3}. Op2 is boundary (succ Op3 in G3). Ejectable.
    auto [n, bad] = verify_all_moves("eject_chain3", part);
    tally({n, bad});
    tally(verify_all_fm("eject_chain3", part));
    
    // Check eject count
    int eject_count = 0, pos_ej = 0, neg_ej = 0;
    for (auto& m : all_moves_for_group(part, 0, 1e18)) {
        if (m.type == FMMove::EJECT) {
            eject_count++;
            if (m.saving > 0.01) pos_ej++;
            else if (m.saving < -0.01) neg_ej++;
            std::cout << "  eject Op" << m.op << ": saving=" << m.saving << "\n";
        }
    }
    CHECK("ejects generated", eject_count > 0);
    std::cout << "  " << eject_count << " ejects (" << pos_ej << "+ " << neg_ej << "-)\n";
}

// 13. FM steal/eject: create partition where FM prefers steal over merge
void test_fm_steal_preferred() {
    std::cout << "=== test_fm_steal_preferred ===\n";
    // {Op0,Op1},{Op2,Op3}. Op1 borders Op2.
    // Steal Op2 into G0 may be better than merging all 4 (tight memory).
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},1000},
             {OpType::Pointwise,{2},{3},1000},
             {OpType::Pointwise,{3},{4},1000}};
    p.fast_memory_capacity = 30000;  // tight: can't fuse all 4
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;
    part.groups[2].ops = {2, 3};
    part.groups[2].cost = part.eval_set({2, 3});
    part.groups[3].alive = false;
    part.rebuild_index();
    
    tally(verify_all_moves("fm_steal", part));
    auto [nfm, bad] = verify_all_fm("fm_steal", part);
    tally({nfm, bad});
    
    // Check what FM picks for border ops
    for (auto op : {1, 2}) {
        auto m = best_move_for(part, op);
        const char* names[] = {"STEAL", "EJECT", "RECOMPUTE", "MERGE"};
        if (m.valid())
            std::cout << "  Op" << op << " best FM: " << names[m.type]
                      << " saving=" << m.saving << "\n";
    }
}

// 14. FM eject preferred: tight memory where group needs splitting
void test_fm_eject_preferred() {
    std::cout << "=== test_fm_eject_preferred ===\n";
    // {Op0,Op1} with {Op2}. If merging all 3 is infeasible and stealing doesn't help,
    // eject may be the only FM move.
    Problem p;
    p.tensors = {{512,512},{512,512},{512,512},{512,512}};
    p.ops = {{OpType::Pointwise,{0},{1},500},
             {OpType::Pointwise,{1},{2},500},
             {OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 270000;  // fits singleton, barely fits 2-fused
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;
    part.rebuild_index();
    // G0={0,1}, G2={2}

    auto [n, bad] = verify_all_moves("fm_eject", part);
    tally({n, bad});
    tally(verify_all_fm("fm_eject", part));
}

// 15. Negative steal gain: moving op to a group makes it worse (forces bad tiling)
void test_negative_steal_gain() {
    std::cout << "=== test_negative_steal_gain ===\n";
    // MM ops with tight memory. Stealing changes working set → different tiling.
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::MatMul,{0,1},{2},3000},
             {OpType::MatMul,{2,3},{4},3000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    
    auto [n, bad] = verify_all_moves("neg_steal", part);
    tally({n, bad});
    tally(verify_all_fm("neg_steal", part));
    
    // Count positive vs negative
    int pos=0, neg=0;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        for (auto& m : all_moves_for_group(part, gi, 1e18)) {
            if (m.saving > 0.01) pos++; else if (m.saving < -0.01) neg++;
        }
    }
    std::cout << "  " << pos << " positive, " << neg << " negative\n";
}

// 16. Large realistic: 8-op chain with mixed MM+PW, verify at local optimum
void test_mixed_chain_at_optimum() {
    std::cout << "=== test_mixed_chain_at_optimum ===\n";
    Problem p;
    p.tensors = {{256,256},{256,256},{256,256},{256,256},{256,256},
                 {256,256},{256,256},{256,256},{256,256}};
    p.ops = {{OpType::Pointwise,{0},{1},500},
             {OpType::MatMul,{1,2},{3},3000},
             {OpType::Pointwise,{3},{4},500},
             {OpType::MatMul,{4,5},{6},3000},
             {OpType::Pointwise,{6},{7},500},
             {OpType::Pointwise,{7},{8},500}};
    p.fast_memory_capacity = 80000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = greedy_descent(Partition::trivial(p, d));
    
    auto [n, bad] = verify_all_moves("mixed_opt", part);
    tally({n, bad});
    auto [nfm, badfm] = verify_all_fm("mixed_opt", part);
    tally({nfm, badfm});
    std::cout << "  " << part.num_alive() << " groups at optimum\n";
}

// ==================== Internal eject and split tests ====================

// 17. Internal eject from a 4-op fused chain: eject middle op → 3 groups
void test_internal_eject_chain4() {
    std::cout << "=== test_internal_eject_chain4 ===\n";
    // {Op0,Op1,Op2,Op3} as single group. Op1 is internal.
    // Eject Op1 → {Op0} + {Op1} + {Op2,Op3} (3 connected components + singleton)
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({256,256});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2, 3};
    part.groups[0].cost = part.eval_set({0, 1, 2, 3});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.groups[3].alive = false;
    part.rebuild_index();

    // Op1 should be internal (pred Op0 inside, succ Op2 inside)
    auto internals = part.internal_ops(0);
    bool found_op1 = false, found_op2 = false;
    for (auto op : internals) {
        if (op == 1) found_op1 = true;
        if (op == 2) found_op2 = true;
    }
    CHECK("Op1 internal", found_op1);
    CHECK("Op2 internal", found_op2);

    // Verify generates internal eject
    int ie_count = 0;
    for (auto& m : all_moves_for_group(part, 0, 1e18)) {
        if (m.type == FMMove::INTERNAL_EJECT) {
            ie_count++;
            auto gc = verify_heap_move(part, m);
            if (gc.applied) {
                double actual = gc.before - gc.after;
                CHECK_EQ("IE gain", m.saving, actual);
                std::cout << "  INTERNAL_EJECT Op" << m.op << ": saving=" << m.saving << "\n";
            }
        }
    }
    CHECK("has internal ejects", ie_count > 0);
    std::cout << "  " << ie_count << " internal ejects generated\n";

    // Verify FM also finds internal eject
    tally(verify_all_fm("ie_chain4", part));
}

// 18. Split at bridge edge in a 4-op chain
void test_split_chain4() {
    std::cout << "=== test_split_chain4 ===\n";
    // {Op0,Op1,Op2,Op3} chain. Every internal edge is a bridge.
    // Split at Op1→Op2 → {Op0,Op1} + {Op2,Op3}
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({256,256});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2, 3};
    part.groups[0].cost = part.eval_set({0, 1, 2, 3});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.groups[3].alive = false;
    part.rebuild_index();

    // Check bridge edges
    auto bridges = part.bridge_edges(0);
    CHECK("has bridges", bridges.size() >= 3);  // 3 edges in chain of 4
    std::cout << "  bridges:";
    for (auto& [a,b] : bridges) std::cout << " (" << a << "→" << b << ")";
    std::cout << "\n";

    // Verify generates SPLIT moves
    int split_count = 0;
    for (auto& m : all_moves_for_group(part, 0, 1e18)) {
        if (m.type == FMMove::SPLIT) {
            split_count++;
            auto gc = verify_heap_move(part, m);
            if (gc.applied) {
                double actual = gc.before - gc.after;
                CHECK_EQ("SPLIT gain", m.saving, actual);
                std::cout << "  SPLIT Op" << m.op << "↔Op" << m.op2
                          << ": saving=" << m.saving << "\n";
            }
        }
    }
    CHECK("has splits", split_count > 0);
    std::cout << "  " << split_count << " split moves generated\n";

    // Verify FM
    tally(verify_all_fm("split_chain4", part));
}

// 19. Diamond: internal op in fused group, no bridge
void test_internal_diamond_fused() {
    std::cout << "=== test_internal_diamond_fused ===\n";
    // Diamond: Op0→T1→{Op1→T2, Op2(T1,T2)→T3}
    // Fuse all 3: {Op0,Op1,Op2}. Op1 has pred Op0, succs Op2 — all internal.
    // But Op0→Op1→Op2 AND Op0→Op2, so edge Op0→Op2 exists.
    // Removing Op0→Op1 edge: Op0 can still reach Op1 via Op0→Op2→? No, Op2 is a successor of Op1.
    // The undirected graph: Op0-Op1, Op0-Op2, Op1-Op2. It's a triangle → no bridges.
    // So SPLIT won't find anything, but INTERNAL_EJECT should work for Op1.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},5000},
             {OpType::Pointwise,{1},{2},5000},
             {OpType::Pointwise,{1,2},{3},5000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.rebuild_index();

    auto bridges = part.bridge_edges(0);
    std::cout << "  bridges: " << bridges.size() << " (expect 0 for triangle)\n";
    // With Op0→Op2 direct edge, it's a triangle → 0 bridges is correct

    // But internal eject should still work
    int ie=0, sp=0;
    for (auto& m : all_moves_for_group(part, 0, 1e18)) {
        if (m.type == FMMove::INTERNAL_EJECT) { ie++; }
        if (m.type == FMMove::SPLIT) { sp++; }
    }
    // Op0 has no preds outside, succ Op1,Op2 inside → internal
    // But Op0 also has NO preds → it IS internal if all succs are inside.
    // Actually Op0 has no op_preds at all (T0 is graph input, no producing op).
    // And succs Op1,Op2 are inside. So Op0 is internal. ✓
    std::cout << "  internal ejects=" << ie << " splits=" << sp << "\n";
    
    // Verify all moves
    tally(verify_all_moves("diamond_fused_all", part));
    tally(verify_all_fm("diamond_fused_all", part));
}

// 20. 5-op chain with tight memory: internal eject helps (breaks bad fused tiling)
void test_internal_eject_helps() {
    std::cout << "=== test_internal_eject_helps ===\n";
    // 5 PW ops, 512x512 tensors. Fusing all 5 is feasible but needs small tiles.
    // Ejecting the middle op and splitting 5→2+1+2 might allow better tiling.
    Problem p;
    for (int i = 0; i <= 5; i++) p.tensors.push_back({512,512});
    for (int i = 0; i < 5; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 280000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);

    // Start with all fused
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2, 3, 4};
    part.groups[0].cost = part.eval_set({0, 1, 2, 3, 4});
    for (int i = 1; i < 5; i++) part.groups[i].alive = false;
    part.rebuild_index();

    double fused_cost = part.total_cost();
    std::cout << "  fused(5) cost=" << fused_cost << "\n";

    // Check what internal moves find
    int ie=0, sp=0, best_type=-1;
    double best_saving = -1e18;
    for (auto& m : all_moves_for_group(part, 0, 1e18)) {
        if (m.type == FMMove::INTERNAL_EJECT) {
            ie++;
            auto gc = verify_heap_move(part, m);
            if (gc.applied && std::abs(gc.reported - (gc.before-gc.after)) < 0.5)
                if (m.saving > best_saving) { best_saving = m.saving; best_type = 4; }
        }
        if (m.type == FMMove::SPLIT) {
            sp++;
            auto gc = verify_heap_move(part, m);
            if (gc.applied && std::abs(gc.reported - (gc.before-gc.after)) < 0.5)
                if (m.saving > best_saving) { best_saving = m.saving; best_type = 5; }
        }
    }
    std::cout << "  internal ejects=" << ie << " splits=" << sp
              << " best_saving=" << best_saving
              << " type=" << (best_type==4?"IE":best_type==5?"SPLIT":"none") << "\n";
    
    tally(verify_all_moves("ie_helps", part));
    tally(verify_all_fm("ie_helps", part));
}

int main() {
    test_asymmetric_chain();
    test_matmul_chain_varied_K();
    test_merge_forces_smaller_tiles();
    test_eject_with_external_successor();
    test_recompute_saves_large_tensor();
    test_steal_pw_from_mm_group();
    test_five_groups();
    test_post_greedy_varied();
    test_compute_bound_merge();
    test_fanin_graph();
    test_mm_pw_splitk_gain();
    test_eject_from_chain3();
    test_fm_steal_preferred();
    test_fm_eject_preferred();
    test_negative_steal_gain();
    test_mixed_chain_at_optimum();
    test_internal_eject_chain4();
    test_split_chain4();
    test_internal_diamond_fused();
    test_internal_eject_helps();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}