// move_gain_test.cpp — Systematic gain verification for all move types.
// For each move: compute total_cost before and after, verify saving matches.

#include "core/types.h"
#include "core/dag.h"
#include "partition/partition.h"
#include "search/local_search.h"
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

// Apply a move to a copy of the partition and return (before, after, saving) for verification
struct GainCheck {
    double before, after, reported;
    bool applied;
};

static GainCheck verify_heap_move(Partition part, const Move& m) {
    GainCheck gc;
    gc.before = part.total_cost();
    gc.reported = m.saving;
    gc.applied = false;

    switch (m.type) {
        case Move::MERGE: {
            std::set<size_t> merged = part.groups[m.ga].ops;
            merged.insert(part.groups[m.gb].ops.begin(), part.groups[m.gb].ops.end());
            double nc = part.eval_set(merged);
            if (nc >= 1e17) return gc;
            part.groups[m.ga].ops = merged;
            part.groups[m.ga].cost = nc;
            part.groups[m.gb].alive = false;
            gc.applied = true;
            break;
        }
        case Move::STEAL: {
            std::set<size_t> new_ga = part.groups[m.ga].ops;
            new_ga.insert(m.op);
            double nac = part.eval_set(new_ga);
            if (nac >= 1e17) return gc;
            std::set<size_t> new_gb = part.groups[m.gb].ops;
            new_gb.erase(m.op);
            double nbc = new_gb.empty() ? 0 : part.eval_set(new_gb);
            if (!new_gb.empty() && nbc >= 1e17) return gc;
            part.groups[m.ga].ops = new_ga;
            part.groups[m.ga].cost = nac;
            if (new_gb.empty()) part.groups[m.gb].alive = false;
            else { part.groups[m.gb].ops = new_gb; part.groups[m.gb].cost = nbc; }
            gc.applied = true;
            break;
        }
        case Move::RECOMPUTE: {
            std::set<size_t> new_ga = part.groups[m.ga].ops;
            new_ga.insert(m.op);
            double nc = part.eval_set(new_ga);
            if (nc >= 1e17) return gc;
            part.groups[m.ga].ops = new_ga;
            part.groups[m.ga].cost = nc;
            gc.applied = true;
            break;
        }
        case Move::EJECT: {
            std::set<size_t> rem = part.groups[m.ga].ops;
            rem.erase(m.op);
            double rc = part.eval_set(rem);
            double sc = part.eval_set({m.op});
            if (rc >= 1e17 || sc >= 1e17) return gc;
            part.groups[m.ga].ops = rem;
            part.groups[m.ga].cost = rc;
            part.add_group({m.op}, sc);
            gc.applied = true;
            break;
        }
    }
    gc.after = part.total_cost();
    return gc;
}

// Verify ALL moves generated from a partition: each one's reported saving
// must equal actual (before - after).
static int verify_all_moves(const char* label, Partition& part, double floor = 1e18) {
    int checked = 0, ok = 0;
    const char* names[] = {"MERGE", "STEAL", "RECOMPUTE", "EJECT"};

    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        MoveHeap heap;
        generate_moves(part, gi, heap, floor, nullptr);
        while (!heap.empty()) {
            Move m = heap.top(); heap.pop();
            auto gc = verify_heap_move(part, m);
            if (!gc.applied) continue;
            checked++;
            double actual = gc.before - gc.after;
            if (std::abs(gc.reported - actual) < 0.5) {
                ok++;
            } else {
                std::cout << "  FAIL: " << label << " " << names[m.type]
                          << " ga=G" << m.ga << " gb=G" << m.gb << " op=" << m.op
                          << " reported=" << gc.reported << " actual=" << actual << "\n";
                g_fail++;
            }
        }
    }
    if (checked > 0 && ok == checked) g_pass++;
    return checked;
}

// Same for FM moves on all border ops
static int verify_all_fm_moves(const char* label, Partition& part, double floor = 1e18) {
    int checked = 0, ok = 0;
    const char* names[] = {"STEAL", "EJECT", "RECOMPUTE", "MERGE"};

    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        for (auto op : part.border_ops(gi)) {
            auto m = best_move_for(part, op, floor);
            if (!m.valid()) continue;
            Partition p2 = part;
            double before = p2.total_cost();
            auto affected = apply_fm_move(p2, m);
            if (affected.empty()) continue;
            double after = p2.total_cost();
            checked++;
            double actual = before - after;
            if (std::abs(m.saving - actual) < 0.5) {
                ok++;
            } else {
                std::cout << "  FAIL: " << label << " FM " << names[m.type]
                          << " op=" << m.op << " reported=" << m.saving 
                          << " actual=" << actual << "\n";
                g_fail++;
            }
        }
    }
    if (checked > 0 && ok == checked) g_pass++;
    return checked;
}

// ==================== Test cases ====================

void test_chain_pw_trivial() {
    std::cout << "--- test_chain_pw_trivial ---\n";
    // 4-op PW chain, 128x128. All singletons. Merges have positive saving.
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    int n = verify_all_moves("chain_pw", part);
    int nfm = verify_all_fm_moves("chain_pw FM", part);
    std::cout << "  checked " << n << " heap moves, " << nfm << " FM moves\n";
}

void test_chain_pw_fused() {
    std::cout << "--- test_chain_pw_fused ---\n";
    // 4-op chain, partially fused: {Op0,Op1},{Op2,Op3}. Steals/recomputes/ejects.
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 50000;
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
    int n = verify_all_moves("chain_fused", part);
    int nfm = verify_all_fm_moves("chain_fused FM", part);
    std::cout << "  checked " << n << " heap moves, " << nfm << " FM moves\n";
}

void test_diamond_all_moves() {
    std::cout << "--- test_diamond_all_moves ---\n";
    // Diamond: Op0→T1→{Op1,Op2(T1,T2)}. All singletons. High compute → fusion helps.
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
    int n = verify_all_moves("diamond", part);
    int nfm = verify_all_fm_moves("diamond FM", part);
    std::cout << "  checked " << n << " heap moves, " << nfm << " FM moves\n";
}

void test_diamond_partially_fused() {
    std::cout << "--- test_diamond_partially_fused ---\n";
    // Diamond with {Op0,Op1} fused, {Op2} alone. Recompute Op0 in G2 is relevant.
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
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;
    int n = verify_all_moves("diamond_fused", part);
    int nfm = verify_all_fm_moves("diamond_fused FM", part);
    std::cout << "  checked " << n << " heap moves, " << nfm << " FM moves\n";
}

void test_matmul_chain() {
    std::cout << "--- test_matmul_chain ---\n";
    // Op0: MM T0@T1→T2. Op1: MM T2@T3→T4. All singletons.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::MatMul,{2,3},{4},2000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    int n = verify_all_moves("mm_chain", part);
    int nfm = verify_all_fm_moves("mm_chain FM", part);
    std::cout << "  checked " << n << " heap moves, " << nfm << " FM moves\n";
}

void test_matmul_pw_fused() {
    std::cout << "--- test_matmul_pw_fused ---\n";
    // Op0: MM T0@T1→T2. Op1: PW T2→T3. Fused {Op0,Op1}. Eject.
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128},{128,128}};
    p.ops = {{OpType::MatMul,{0,1},{2},2000},
             {OpType::Pointwise,{2},{3},500}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;
    int n = verify_all_moves("mm_pw_fused", part);
    int nfm = verify_all_fm_moves("mm_pw_fused FM", part);
    std::cout << "  checked " << n << " heap moves, " << nfm << " FM moves\n";
}

void test_tight_memory() {
    std::cout << "--- test_tight_memory ---\n";
    // Large tensors, small capacity. Merges may be infeasible (filtered out).
    // Ejects may be beneficial.
    Problem p;
    for (int i = 0; i <= 3; i++) p.tensors.push_back({512,512});
    for (int i = 0; i < 3; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 500});
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    // Start with all fused (may need smaller tiling)
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1, 2};
    part.groups[0].cost = part.eval_set({0, 1, 2});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    int n = verify_all_moves("tight_mem", part);
    int nfm = verify_all_fm_moves("tight_mem FM", part);
    std::cout << "  checked " << n << " heap moves, " << nfm << " FM moves\n";
}

void test_recompute_with_op_in_both() {
    std::cout << "--- test_recompute_with_op_in_both ---\n";
    // Diamond after recompute: Op0 already in both G0 and G2.
    // Further moves from this state.
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
    // Recompute Op0 in G2
    part.groups[2].ops = {0, 2};
    part.groups[2].cost = part.eval_set({0, 2});
    int n = verify_all_moves("recomp_both", part);
    int nfm = verify_all_fm_moves("recomp_both FM", part);
    std::cout << "  checked " << n << " heap moves, " << nfm << " FM moves\n";
}

void test_on_benchmark_partition() {
    std::cout << "--- test_on_benchmark_partition ---\n";
    // Load benchmark 1, run greedy, then verify all remaining moves from the optimum.
    Problem p;
    p.tensors = {{128,128},{128,128},{512,128},{128,512},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},
             {OpType::Pointwise,{1},{2},500},
             {OpType::MatMul,{2,3},{4},3000},
             {OpType::MatMul,{4,3},{5},3000},
             {OpType::Pointwise,{5},{0},200}}; // cycle → won't work
    // Actually just use simple known-structure problem
    for (int i = 0; i <= 6; i++) p.tensors.push_back({256,256});
    p.tensors.clear();
    p.ops.clear();
    for (int i = 0; i <= 5; i++) p.tensors.push_back({256,256});
    for (int i = 0; i < 5; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = local_search_from(Partition::trivial(p, d));
    int n = verify_all_moves("post_greedy", part);
    int nfm = verify_all_fm_moves("post_greedy FM", part);
    std::cout << "  " << part.num_alive() << " groups, " << n << " heap, " << nfm << " FM moves\n";
}

// ==================== Missing move analysis ====================

void test_split_vs_eject() {
    std::cout << "--- test_split_vs_eject ---\n";
    // Chain of 4 with 256x256 tensors. Fused {0,1,2,3}.
    // SPLIT {0,1}+{2,3} vs best EJECT.
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({256,256});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 1000});
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0,1,2,3};
    part.groups[0].cost = part.eval_set({0,1,2,3});
    part.groups[1].alive = false;
    part.groups[2].alive = false;
    part.groups[3].alive = false;
    
    double fused_cost = part.total_cost();
    
    // Try all 2+2 splits
    double best_split = 1e18;
    std::string best_desc;
    for (auto& [a, b] : std::vector<std::pair<std::set<size_t>,std::set<size_t>>>{
        {{0,1},{2,3}}, {{0,2},{1,3}}, {{0,3},{1,2}}}) {
        double ca = part.eval_set(a), cb = part.eval_set(b);
        if (ca < 1e17 && cb < 1e17 && ca + cb < best_split) {
            best_split = ca + cb;
        }
    }
    
    // Best eject (singleton)
    MoveHeap heap;
    generate_moves(part, 0, heap, 1e18, nullptr);
    double best_eject_cost = 1e18;
    while (!heap.empty()) {
        Move m = heap.top(); heap.pop();
        if (m.type == Move::EJECT) {
            double cost = fused_cost - m.saving;
            if (cost < best_eject_cost) best_eject_cost = cost;
        }
    }
    
    std::cout << "  fused=" << fused_cost << " best_split(2+2)=" << best_split
              << " best_eject=" << best_eject_cost << "\n";
    
    if (best_split < best_eject_cost - 0.01 && best_split < fused_cost - 0.01) {
        std::cout << "  ** SPLIT(2+2) beats both EJECT and STAY by "
                  << (best_eject_cost - best_split) << "\n";
        std::cout << "  → Consider adding SPLIT move type\n";
    } else {
        std::cout << "  EJECT sufficient (or fused is already best)\n";
    }
    g_pass++;
}


void test_single_group_no_moves() {
    std::cout << "--- test_single_group_no_moves ---\n";
    Problem p;
    p.tensors = {{128,128},{128,128},{128,128}};
    p.ops = {{OpType::Pointwise,{0},{1},1000},{OpType::Pointwise,{1},{2},1000}};
    p.fast_memory_capacity = 50000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    auto part = Partition::trivial(p, d);
    part.groups[0].ops = {0, 1};
    part.groups[0].cost = part.eval_set({0, 1});
    part.groups[1].alive = false;
    auto ej = part.ejectable_ops(0);
    CHECK("no ejectable (single group)", ej.empty());
    MoveHeap heap;
    generate_moves(part, 0, heap, 1e18, nullptr);
    CHECK("no moves at all", heap.empty());
    std::cout << "  ejectable=" << ej.size() << " moves=" << heap.size() << "\n";
}

void test_split_vs_eject_tiled() {
    std::cout << "--- test_split_vs_eject_tiled ---\n";
    Problem p;
    for (int i = 0; i <= 4; i++) p.tensors.push_back({512,512});
    for (int i = 0; i < 4; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 2000});
    p.fast_memory_capacity = 300000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    DAG d = DAG::build(p);
    Partition tmp; tmp.prob = &p; tmp.dag = &d;

    double cost_4 = tmp.eval_set({0,1,2,3});
    double cost_01 = tmp.eval_set({0,1});
    double cost_23 = tmp.eval_set({2,3});
    double cost_012 = tmp.eval_set({0,1,2});
    double cost_123 = tmp.eval_set({1,2,3});
    double cost_0 = tmp.eval_set({0});
    double cost_3 = tmp.eval_set({3});

    double split_22 = cost_01 + cost_23;
    double eject_0 = cost_123 + cost_0;
    double eject_3 = cost_012 + cost_3;
    double best_eject = std::min(eject_0, eject_3);

    std::cout << "  fused(4)=" << cost_4 << " split(2+2)=" << split_22
              << " eject_0=" << eject_0 << " eject_3=" << eject_3 << "\n";
    
    if (split_22 < best_eject - 0.01 && split_22 < cost_4 - 0.01)
        std::cout << "  ** SPLIT(2+2) beats EJECT by " << (best_eject - split_22) << "\n";
    else
        std::cout << "  EJECT sufficient\n";
    g_pass++;
}

int main() {
    test_chain_pw_trivial();
    test_chain_pw_fused();
    test_diamond_all_moves();
    test_diamond_partially_fused();
    test_matmul_chain();
    test_matmul_pw_fused();
    test_tight_memory();
    test_recompute_with_op_in_both();
    test_on_benchmark_partition();
    test_split_vs_eject();
    test_single_group_no_moves();
    test_split_vs_eject_tiled();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}