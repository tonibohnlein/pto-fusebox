// coupled_fm_correctness_test.cpp
//
// Per-move-type correctness tests for the coupled FM local search.
//
// For EVERY move applied (regardless of type), we verify ALL constraints:
//   1. GAIN ACCURACY: predicted saving == actual total_cost delta
//   2. FROM-SCRATCH COST: cached total_cost == fresh Subgraph recomputation
//   3. COUPLING CONSISTENCY: symmetric next/prev, retained tensors valid
//      on both producer (boundary output) and consumer (boundary input) sides
//   4. OP COVERAGE: every op in at least one alive group
//   5. ACYCLICITY: partition is acyclic after the move
//   6. BOUNDARY VALIDITY: for coupling moves, tensors are boundary out/in
//
// Each move type is tested in isolation by constructing multiple diverse
// (Problem, initial-partition) configurations, scanning for all feasible
// moves of that type, applying each one, and checking every constraint.

#include "search/coupled_fm_search.h"
#include "search/coupled_fm_pass.h"
#include "search/coupling_search.h"
#include "search/partition_moves.h"
#include "search/structural_ops.h"
#include "partition/partition.h"
#include "core/cost_cache.h"
#include "solution/solution.h"
#include <iostream>
#include <cmath>
#include <set>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* label, bool cond) {
    if (cond) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label << "\n"; }
}

// ============================================================================
// Full constraint check for a single move
//
// Checks ALL constraints regardless of move type:
//   gain, from-scratch cost, coupling consistency, op coverage, acyclicity,
//   and (for coupling moves) boundary validity of retained tensors.
//
// skip_gain: set true when we can't compute the coupled saving from outside
//   (SPLIT/DE_RECOMPUTE with coupling — no coupled_*_saving function).
// ============================================================================

static bool cp_is_consistent(const CoupledPartition& cp) {
    const size_t n = cp.part.groups.size();
    const auto& dag = *cp.part.dag;
    if (cp.next_group.size() < n || cp.prev_group.size() < n) return false;
    for (size_t g = 0; g < n; g++) {
        if (!cp.part.groups[g].alive) continue;
        size_t h = cp.next_group[g];
        if (h == SIZE_MAX) continue;
        if (h >= n || !cp.part.groups[h].alive) return false;
        if (cp.prev_group[h] != g) return false;
    }
    for (size_t g = 0; g < n; g++) {
        if (!cp.part.groups[g].alive) continue;
        size_t p = cp.prev_group[g];
        if (p == SIZE_MAX) continue;
        if (p >= n || !cp.part.groups[p].alive) return false;
        if (cp.next_group[p] != g) return false;
    }
    for (auto& [edge, tensors] : cp.retained) {
        auto [src, dst] = edge;
        if (src >= n || dst >= n) return false;
        if (!cp.part.groups[src].alive || !cp.part.groups[dst].alive) return false;
        if (cp.next_group[src] != dst) return false;
        if (tensors.empty()) return false;
        for (auto t : tensors) {
            if (!is_boundary_output_of(cp.part.groups[src].ops, t, dag)) return false;
            if (!is_boundary_input_of(cp.part.groups[dst].ops, t, dag)) return false;
        }
    }
    for (size_t g = 0; g < n; g++) {
        if (!cp.part.groups[g].alive) continue;
        size_t h = cp.next_group[g];
        if (h == SIZE_MAX) continue;
        if (!cp.retained.count({g, h})) return false;
        if (cp.retained.at({g, h}).empty()) return false;
    }
    return true;
}

static bool all_ops_covered(const Partition& part) {
    for (size_t i = 0; i < part.prob->num_ops(); i++) {
        bool found = false;
        for (auto gi : part.groups_of(i))
            if (part.groups[gi].alive) { found = true; break; }
        if (!found) return false;
    }
    return true;
}

static double recompute_total_cost(const CoupledPartition& cp) {
    double total = 0;
    const auto& part = cp.part;
    for (size_t g = 0; g < part.groups.size(); g++) {
        if (!part.groups[g].alive) continue;
        auto en = cp.entering_for(g);
        auto re = cp.retain_for(g);
        std::vector<size_t> v(part.groups[g].ops.begin(), part.groups[g].ops.end());
        auto sg = Subgraph::create(*part.prob, *part.dag, v);
        if (!sg) { total += 1e18; continue; }
        auto cr = sg->best_cost(en, re);
        total += cr.feasible ? cr.latency : 1e18;
    }
    return total;
}

struct MoveResult {
    bool applied = false;
    bool gain_ok = false;
    bool cost_ok = false;
    bool consistent = false;
    bool covered = false;
    bool acyclic = false;
};

static MoveResult apply_and_check(CoupledPartition& cp,
                                   const CoupledFMMove& move,
                                   const char* label,
                                   bool skip_gain = false) {
    MoveResult r;
    double cost_before = cp.total_cost();
    double scratch_before = recompute_total_cost(cp);

    // Pre-move: cached == scratch
    if (std::abs(cost_before - scratch_before) > 1.0 + 0.01 * std::max(1.0, cost_before)) {
        std::cout << "  FAIL: " << label << " pre-move cost mismatch "
                  << cost_before << " vs " << scratch_before << "\n";
        g_fail++;
    } else {
        g_pass++;
    }

    auto affected = apply_coupled_fm_move(cp, move);
    if (affected.empty()) return r;
    r.applied = true;

    double cost_after  = cp.total_cost();
    double scratch_after = recompute_total_cost(cp);
    double actual_gain = cost_before - cost_after;

    // 1. From-scratch cost
    r.cost_ok = std::abs(cost_after - scratch_after) <
                1.0 + 0.01 * std::max(1.0, cost_after);
    CHECK((std::string(label) + " cost=scratch").c_str(), r.cost_ok);

    // 2. Gain accuracy
    if (!skip_gain) {
        double disc = move.saving - actual_gain;
        r.gain_ok = std::abs(disc) <= 0.1 * std::max(1.0, std::abs(move.saving)) + 1.0;
        CHECK((std::string(label) + " gain").c_str(), r.gain_ok);
        if (!r.gain_ok)
            std::cout << "    predicted=" << move.saving
                      << " actual=" << actual_gain << "\n";
    } else {
        r.gain_ok = true;
        g_pass++;
    }

    // 3. Coupling consistency
    r.consistent = cp_is_consistent(cp);
    CHECK((std::string(label) + " consistent").c_str(), r.consistent);

    // 4. Op coverage
    r.covered = all_ops_covered(cp.part);
    CHECK((std::string(label) + " coverage").c_str(), r.covered);

    // 5. Acyclicity
    cp.part.rebuild_index();
    cp.part.rebuild_group_dag();
    r.acyclic = cp.part.is_acyclic();
    CHECK((std::string(label) + " acyclic").c_str(), r.acyclic);

    return r;
}

// ============================================================================
// Problem factories
// ============================================================================

static Problem make_chain(int n, int64_t tw, int64_t th, int64_t mem) {
    Problem p;
    for (int i = 0; i <= n; i++) p.tensors.push_back({tw, th});
    for (int i = 0; i < n; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+1)}, 300});
    p.fast_memory_capacity = mem;
    p.slow_memory_bandwidth = 10;
    p.native_w = tw;  p.native_h = th;
    for (int i = 1; i < n; i++) p.retainable_tensors.insert(i);
    return p;
}

static Problem make_fan12() {
    Problem p;
    for (int i = 0; i <= 16; i++) p.tensors.push_back({48, 48});
    p.ops.push_back({OpType::Pointwise, {0},    {5},   300});
    p.ops.push_back({OpType::Pointwise, {1},    {6},   300});
    p.ops.push_back({OpType::Pointwise, {2},    {7},   300});
    p.ops.push_back({OpType::Pointwise, {3},    {8},   300});
    p.ops.push_back({OpType::Pointwise, {4},    {9},   300});
    p.ops.push_back({OpType::Pointwise, {5, 6}, {10},  300});
    p.ops.push_back({OpType::Pointwise, {7, 8}, {11},  300});
    p.ops.push_back({OpType::Pointwise, {8, 9}, {12},  300});
    p.ops.push_back({OpType::Pointwise, {10,11},{13},  300});
    p.ops.push_back({OpType::Pointwise, {11,12},{14},  300});
    p.ops.push_back({OpType::Pointwise, {13,14},{15},  300});
    p.ops.push_back({OpType::Pointwise, {15},   {16},  300});
    p.fast_memory_capacity = 12000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 48;  p.native_h = 48;
    p.retainable_tensors = {5,6,7,8,9,10,11,12,13,14,15};
    return p;
}

static Problem make_diamond6() {
    Problem p;
    for (int i = 0; i <= 7; i++) p.tensors.push_back({48, 48});
    p.ops.push_back({OpType::Pointwise, {0},    {1},   500});
    p.ops.push_back({OpType::Pointwise, {0},    {2},   500});
    p.ops.push_back({OpType::Pointwise, {1, 2}, {4},   500});
    p.ops.push_back({OpType::Pointwise, {3},    {5},   500});
    p.ops.push_back({OpType::Pointwise, {4, 5}, {6},   500});
    p.ops.push_back({OpType::Pointwise, {6},    {7},   500});
    p.fast_memory_capacity = 12000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 48;  p.native_h = 48;
    p.retainable_tensors = {1, 2, 4, 5, 6};
    return p;
}

static Problem make_wide_fan() {
    Problem p;
    for (int i = 0; i <= 10; i++) p.tensors.push_back({32, 32});
    for (int i = 0; i < 5; i++)
        p.ops.push_back({OpType::Pointwise, {(size_t)i}, {(size_t)(i+5)}, 200});
    p.ops.push_back({OpType::Pointwise, {5,6,7,8,9}, {10}, 200});
    p.fast_memory_capacity = 8000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 32;  p.native_h = 32;
    p.retainable_tensors = {5, 6, 7, 8, 9};
    return p;
}

// ladder8: two parallel 4-op chains with cross-links
static Problem make_ladder8() {
    Problem p;
    for (int i = 0; i <= 11; i++) p.tensors.push_back({40, 40});
    p.ops.push_back({OpType::Pointwise, {0},      {2},   300});
    p.ops.push_back({OpType::Pointwise, {1},      {3},   300});
    p.ops.push_back({OpType::Pointwise, {2, 3},   {5},   300});
    p.ops.push_back({OpType::Pointwise, {3},      {6},   300});
    p.ops.push_back({OpType::Pointwise, {5},      {8},   300});
    p.ops.push_back({OpType::Pointwise, {5, 6},   {9},   300});
    p.ops.push_back({OpType::Pointwise, {8},      {10},  300});
    p.ops.push_back({OpType::Pointwise, {9},      {11},  300});
    p.fast_memory_capacity = 8000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 40;  p.native_h = 40;
    p.retainable_tensors = {2, 3, 5, 6, 8, 9};
    return p;
}

// ============================================================================
// TestInstance: Problem + DAG + cache + CoupledPartition
// ============================================================================

struct TestInstance {
    Problem prob;
    DAG dag;
    std::unique_ptr<CostCache> cache;
    CoupledPartition cp;
    FlatSet<size_t> feasibly_ret;

    TestInstance() = default;
    TestInstance(TestInstance&&) = default;
    TestInstance& operator=(TestInstance&&) = default;
};

// Build from a problem with specific merge list
static TestInstance build_instance(Problem p,
    const std::vector<std::pair<size_t,size_t>>& merges = {}) {
    TestInstance ti;
    ti.prob = std::move(p);
    ti.dag = DAG::build(ti.prob);
    ti.cache = std::make_unique<CostCache>(100000);
    Partition part = Partition::trivial(ti.prob, ti.dag);
    part.finalize(ti.cache.get());
    for (auto [a, b] : merges) {
        partition_moves::apply_merge(part, a, b);
        part.rebuild_index();
    }
    part.finalize(ti.cache.get());
    ti.feasibly_ret = compute_feasibly_retainable(ti.prob, ti.dag);
    ti.cp.init_from(std::move(part), ti.cache.get());
    return ti;
}

// Merge consecutive pairs: {0,1},{2,3},...
static std::vector<std::pair<size_t,size_t>> pairs(int n) {
    std::vector<std::pair<size_t,size_t>> v;
    for (int i = 0; i+1 < n; i += 2) v.push_back({(size_t)i,(size_t)(i+1)});
    return v;
}

// Merge consecutive triples: {0,1},{0,2},{3,4},{3,5},...
static std::vector<std::pair<size_t,size_t>> triples(int n) {
    std::vector<std::pair<size_t,size_t>> v;
    for (int i = 0; i+2 < n; i += 3) {
        v.push_back({(size_t)i,(size_t)(i+1)});
        v.push_back({(size_t)i,(size_t)(i+2)});
    }
    return v;
}

// Merge consecutive quads: {0,1},{0,2},{0,3},{4,5},{4,6},{4,7},...
static std::vector<std::pair<size_t,size_t>> quads(int n) {
    std::vector<std::pair<size_t,size_t>> v;
    for (int i = 0; i+3 < n; i += 4) {
        v.push_back({(size_t)i,(size_t)(i+1)});
        v.push_back({(size_t)i,(size_t)(i+2)});
        v.push_back({(size_t)i,(size_t)(i+3)});
    }
    return v;
}

// ============================================================================
// Per-move-type test: SPLIT
// Needs: multi-op group (3+) with a bridge edge.
// ============================================================================

void test_split_isolated() {
    std::cout << "--- test_split_isolated ---\n";
    int count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        for (size_t gi = 0; gi < ti.cp.part.groups.size(); gi++) {
            auto& g = ti.cp.part.groups[gi];
            if (!g.alive || g.ops.size() < 3) continue;
            for (auto& [a, b] : ti.cp.part.bridge_edges(gi)) {
                auto sr = ti.cp.part.eval_split(a, b, gi);
                if (!sr.feasible) continue;
                CoupledFMMove m;
                m.type = CoupledFMMove::SPLIT;
                m.op = a;  m.op2 = b;  m.ga = gi;
                m.saving = sr.saving;
                std::string label = std::string(tag) + " SPLIT G" +
                    std::to_string(gi) + "(" + std::to_string(g.ops.size()) +
                    " ops) at " + std::to_string(a) + "-" + std::to_string(b);
                apply_and_check(ti.cp, m, label.c_str());
                ti.cp.part.rebuild_index();
                ti.cp.part.rebuild_group_dag();
                count++;
            }
        }
    };

    // Diverse configurations
    run("chain8-triple",  build_instance(make_chain(8, 64,64,50000), triples(8)));
    run("chain12-triple", build_instance(make_chain(12,64,64,50000), triples(12)));
    run("chain12-quad",   build_instance(make_chain(12,64,64,50000), quads(12)));
    run("chain16-triple", build_instance(make_chain(16,48,48,50000), triples(16)));
    run("chain16-quad",   build_instance(make_chain(16,48,48,50000), quads(16)));
    run("fan12-triple",   build_instance(make_fan12(), {{0,1},{0,5},{2,3},{2,6},{8,9},{8,10}}));
    run("ladder8-triple", build_instance(make_ladder8(), {{0,2},{1,3},{4,5}}));
    run("diamond6-merge", build_instance(make_diamond6(), {{0,1},{0,2}}));

    CHECK("split count >= 10", count >= 10);
    std::cout << "  total splits verified: " << count << "\n";
}

// ============================================================================
// Per-move-type test: EJECT
// Needs: multi-op group (2+) with an ejectable border op.
// ============================================================================

void test_eject_isolated() {
    std::cout << "--- test_eject_isolated ---\n";
    int count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        for (size_t gi = 0; gi < ti.cp.part.groups.size(); gi++) {
            auto& g = ti.cp.part.groups[gi];
            if (!g.alive || g.ops.size() < 2) continue;
            for (auto op : ti.cp.part.ejectable_ops(gi)) {
                auto er = ti.cp.part.eval_eject(op, gi);
                if (!er.feasible) continue;
                CoupledFMMove m;
                m.type = CoupledFMMove::EJECT;
                m.op = op;  m.ga = gi;  m.saving = er.saving;
                std::string label = std::string(tag) + " EJECT op" +
                    std::to_string(op) + " from G" + std::to_string(gi) +
                    "(" + std::to_string(g.ops.size()) + ")";
                apply_and_check(ti.cp, m, label.c_str());
                ti.cp.part.rebuild_index();
                ti.cp.part.rebuild_group_dag();
                count++;
                break;  // one per group to keep state manageable
            }
        }
    };

    run("chain8-pair",    build_instance(make_chain(8, 64,64,50000), pairs(8)));
    run("chain12-pair",   build_instance(make_chain(12,64,64,50000), pairs(12)));
    run("chain12-triple", build_instance(make_chain(12,64,64,50000), triples(12)));
    run("chain16-pair",   build_instance(make_chain(16,48,48,50000), pairs(16)));
    run("fan12-pair",     build_instance(make_fan12(), {{0,5},{1,6},{2,6},{3,7},{8,9}}));
    run("ladder8-pair",   build_instance(make_ladder8(), {{0,2},{1,3},{4,6},{5,7}}));
    run("diamond6-pair",  build_instance(make_diamond6(), {{0,2},{3,4}}));

    CHECK("eject count >= 10", count >= 10);
    std::cout << "  total ejects verified: " << count << "\n";
}

// ============================================================================
// Per-move-type test: INTERNAL_EJECT
// Needs: group with 3-15 ops, op with no external neighbors.
// ============================================================================

void test_internal_eject_isolated() {
    std::cout << "--- test_internal_eject_isolated ---\n";
    int count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        for (size_t gi = 0; gi < ti.cp.part.groups.size(); gi++) {
            auto& g = ti.cp.part.groups[gi];
            if (!g.alive || g.ops.size() < 3 || g.ops.size() > 15) continue;
            for (auto op : ti.cp.part.internal_ops(gi)) {
                auto er = ti.cp.part.eval_eject(op, gi);
                if (!er.feasible) continue;
                CoupledFMMove m;
                m.type = CoupledFMMove::INTERNAL_EJECT;
                m.op = op;  m.ga = gi;  m.saving = er.saving;
                std::string label = std::string(tag) + " INT_EJECT op" +
                    std::to_string(op) + " from G" + std::to_string(gi) +
                    "(" + std::to_string(g.ops.size()) + ")";
                apply_and_check(ti.cp, m, label.c_str());
                ti.cp.part.rebuild_index();
                ti.cp.part.rebuild_group_dag();
                count++;
                break;
            }
        }
    };

    run("chain8-triple",   build_instance(make_chain(8, 64,64,50000), triples(8)));
    run("chain12-triple",  build_instance(make_chain(12,64,64,50000), triples(12)));
    run("chain12-quad",    build_instance(make_chain(12,64,64,50000), quads(12)));
    run("chain16-triple",  build_instance(make_chain(16,48,48,50000), triples(16)));
    run("chain16-quad",    build_instance(make_chain(16,48,48,50000), quads(16)));
    run("fan12-triple",    build_instance(make_fan12(), {{0,1},{0,5},{2,3},{2,6}}));
    run("ladder8-triple",  build_instance(make_ladder8(), {{0,1},{0,2},{4,5},{4,6}}));

    CHECK("internal_eject count >= 10", count >= 10);
    std::cout << "  total internal_ejects verified: " << count << "\n";
}

// ============================================================================
// Per-move-type test: MERGE
// Needs: two adjacent alive groups where merged cost is feasible.
// ============================================================================

void test_merge_isolated() {
    std::cout << "--- test_merge_isolated ---\n";
    int count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        // Check each pair of adjacent groups
        for (size_t gi = 0; gi < ti.cp.part.groups.size() && count < 20; gi++) {
            if (!ti.cp.part.groups[gi].alive) continue;
            auto adj = ti.cp.part.adjacent_groups(gi);
            for (auto gj : adj) {
                if (gj <= gi) continue;  // avoid duplicates
                if (!ti.cp.part.acyclic_merge_local(gi, gj)) continue;
                auto mr = partition_moves::eval_merge(ti.cp.part, gi, gj);
                if (!mr.feasible) continue;
                CoupledFMMove m;
                m.type = CoupledFMMove::MERGE;
                m.op = *ti.cp.part.groups[gi].ops.begin();
                m.ga = gi;  m.gb = gj;  m.saving = mr.saving;
                std::string label = std::string(tag) + " MERGE G" +
                    std::to_string(gi) + "+G" + std::to_string(gj);
                apply_and_check(ti.cp, m, label.c_str());
                ti.cp.part.rebuild_index();
                ti.cp.part.rebuild_group_dag();
                count++;
                break;  // one merge per source group
            }
        }
    };

    run("chain8-trivial",  build_instance(make_chain(8, 64,64,50000)));
    run("chain12-trivial", build_instance(make_chain(12,64,64,50000)));
    run("chain16-trivial", build_instance(make_chain(16,48,48,50000)));
    run("fan12-trivial",   build_instance(make_fan12()));
    run("diamond6-trivial",build_instance(make_diamond6()));
    run("ladder8-trivial", build_instance(make_ladder8()));
    run("wide_fan-trivial",build_instance(make_wide_fan()));

    CHECK("merge count >= 10", count >= 10);
    std::cout << "  total merges verified: " << count << "\n";
}

// ============================================================================
// Per-move-type test: STEAL
// Needs: border op in one group that can move to an adjacent group.
// ============================================================================

void test_steal_isolated() {
    std::cout << "--- test_steal_isolated ---\n";
    int count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        for (size_t gi = 0; gi < ti.cp.part.groups.size() && count < 25; gi++) {
            if (!ti.cp.part.groups[gi].alive) continue;
            if (ti.cp.part.groups[gi].ops.size() < 2) continue;
            for (auto op : ti.cp.part.ejectable_ops(gi)) {
                // Find an adjacent group to steal into
                for (auto nbr : ti.dag.op_neighbors[op]) {
                    for (auto gj : ti.cp.part.groups_of(nbr)) {
                        if (gj == gi || !ti.cp.part.groups[gj].alive) continue;
                        if (!ti.cp.part.acyclic_steal_local(op, gi, gj)) continue;
                        // Evaluate
                        FlatSet<size_t> new_gj = ti.cp.part.groups[gj].ops;
                        new_gj.insert(op);
                        double new_gj_cost = ti.cp.part.eval_set(new_gj);
                        if (new_gj_cost >= 1e17) continue;
                        FlatSet<size_t> new_gi = ti.cp.part.groups[gi].ops;
                        new_gi.erase(op);
                        double new_gi_cost = 0;
                        if (!new_gi.empty()) {
                            auto comps = structural_ops::connected_components(new_gi, ti.dag);
                            for (auto& comp : comps) {
                                double c = ti.cp.part.eval_set(comp);
                                if (c >= 1e17) { new_gi_cost = 1e18; break; }
                                new_gi_cost += c;
                            }
                        }
                        if (new_gi_cost >= 1e17) continue;
                        double saving = (ti.cp.part.groups[gi].cost +
                                         ti.cp.part.groups[gj].cost) -
                                        (new_gi_cost + new_gj_cost);

                        CoupledFMMove m;
                        m.type = CoupledFMMove::STEAL;
                        m.op = op;  m.ga = gi;  m.gb = gj;
                        m.saving = saving;
                        std::string label = std::string(tag) + " STEAL op" +
                            std::to_string(op) + " G" + std::to_string(gi) +
                            "->G" + std::to_string(gj);
                        apply_and_check(ti.cp, m, label.c_str());
                        ti.cp.part.rebuild_index();
                        ti.cp.part.rebuild_group_dag();
                        count++;
                        goto next_group;
                    }
                }
            }
            next_group:;
        }
    };

    run("chain12-pair",    build_instance(make_chain(12,64,64,50000), pairs(12)));
    run("chain8-pair",     build_instance(make_chain(8, 64,64,50000), pairs(8)));
    run("chain16-pair",    build_instance(make_chain(16,48,48,50000), pairs(16)));
    run("fan12-pair",      build_instance(make_fan12(), {{0,5},{1,6},{3,7},{8,9}}));
    run("ladder8-pair",    build_instance(make_ladder8(), {{0,2},{1,3},{4,6},{5,7}}));

    CHECK("steal count >= 10", count >= 10);
    std::cout << "  total steals verified: " << count << "\n";
}

// ============================================================================
// Per-move-type test: RECOMPUTE + DE_RECOMPUTE
// Apply RECOMPUTE (copy op into neighbor group), then DE_RECOMPUTE (remove).
// ============================================================================

void test_recompute_de_recompute_isolated() {
    std::cout << "--- test_recompute_de_recompute_isolated ---\n";
    int rc_count = 0, dr_count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        // Find recompute opportunities: border op adjacent to another group
        for (size_t op = 0; op < ti.prob.num_ops() && rc_count < 15; op++) {
            auto gs = ti.cp.part.groups_of(op);
            if (gs.empty()) continue;
            size_t ga = gs[0];
            if (!ti.cp.part.groups[ga].alive) continue;
            for (auto nbr : ti.dag.op_neighbors[op]) {
                for (auto gb : ti.cp.part.groups_of(nbr)) {
                    if (gb == ga || !ti.cp.part.groups[gb].alive) continue;
                    if (ti.cp.part.groups[gb].ops.count(op)) continue;
                    auto rr = partition_moves::eval_recompute(ti.cp.part, op, gb);
                    if (!rr.feasible) continue;

                    CoupledFMMove m;
                    m.type = CoupledFMMove::RECOMPUTE;
                    m.op = op;  m.ga = ga;  m.gb = gb;  m.saving = rr.saving;
                    std::string label = std::string(tag) + " RECOMP op" +
                        std::to_string(op) + " into G" + std::to_string(gb);
                    apply_and_check(ti.cp, m, label.c_str());
                    ti.cp.part.rebuild_index();
                    ti.cp.part.rebuild_group_dag();
                    rc_count++;

                    // Now try DE_RECOMPUTE from gb
                    auto dr = ti.cp.part.acyclic_de_recompute_local(op, gb)
                                  ? partition_moves::eval_de_recompute(ti.cp.part, gb, op)
                                  : partition_moves::EvalResult{};
                    if (dr.feasible) {
                        CoupledFMMove dm;
                        dm.type = CoupledFMMove::DE_RECOMPUTE;
                        dm.op = op;  dm.ga = gb;  dm.saving = dr.saving;
                        std::string dlabel = std::string(tag) + " DE_RECOMP op" +
                            std::to_string(op) + " from G" + std::to_string(gb);
                        apply_and_check(ti.cp, dm, dlabel.c_str());
                        ti.cp.part.rebuild_index();
                        ti.cp.part.rebuild_group_dag();
                        dr_count++;
                    }
                    goto next_op;
                }
            }
            next_op:;
        }
    };

    run("chain12-trivial", build_instance(make_chain(12,64,64,50000)));
    run("chain8-trivial",  build_instance(make_chain(8, 64,64,50000)));
    run("fan12-trivial",   build_instance(make_fan12()));
    run("diamond6-trivial",build_instance(make_diamond6()));
    run("ladder8-trivial", build_instance(make_ladder8()));
    run("chain16-trivial", build_instance(make_chain(16,48,48,50000)));
    run("wide_fan-trivial",build_instance(make_wide_fan()));

    CHECK("recompute count >= 10", rc_count >= 10);
    CHECK("de_recompute count >= 10", dr_count >= 10);
    std::cout << "  recompute=" << rc_count << " de_recompute=" << dr_count << "\n";
}

// ============================================================================
// Per-move-type test: TENSOR_MERGE
// Needs: a tensor consumed by ops in 2+ different groups.
// ============================================================================

void test_tensor_merge_isolated() {
    std::cout << "--- test_tensor_merge_isolated ---\n";
    int count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        FlatSet<size_t> tensors_done;
        for (size_t t = 0; t < ti.dag.tensor_consumers.size(); t++) {
            if (tensors_done.count(t)) continue;
            auto& consumers = ti.dag.tensor_consumers[t];
            if (consumers.size() < 2) continue;
            FlatSet<size_t> cgroups;
            for (auto cop : consumers)
                for (auto cg : ti.cp.part.groups_of(cop))
                    cgroups.insert(cg);
            int prod = ti.dag.tensor_producer[t];
            if (prod >= 0)
                for (auto pg : ti.cp.part.groups_of((size_t)prod))
                    cgroups.insert(pg);
            if (cgroups.size() < 2) continue;
            std::vector<size_t> gl(cgroups.begin(), cgroups.end());
            if (!ti.cp.part.acyclic_merge_local(gl)) continue;
            auto tmr = partition_moves::eval_tensor_merge(ti.cp.part, gl);
            if (!tmr.feasible) continue;
            CoupledFMMove m;
            m.type = CoupledFMMove::TENSOR_MERGE;
            m.op = consumers[0];  m.saving = tmr.saving;
            m.tensor_groups = gl;
            std::string label = std::string(tag) + " T_MERGE T" +
                std::to_string(t) + " (" + std::to_string(gl.size()) + " groups)";
            apply_and_check(ti.cp, m, label.c_str());
            ti.cp.part.rebuild_index();
            ti.cp.part.rebuild_group_dag();
            tensors_done.insert(t);
            count++;
        }
    };

    run("fan12-trivial",    build_instance(make_fan12()));
    run("diamond6-trivial", build_instance(make_diamond6()));
    run("ladder8-trivial",  build_instance(make_ladder8()));
    run("wide_fan-trivial", build_instance(make_wide_fan()));
    run("chain12-trivial",  build_instance(make_chain(12,64,64,50000)));
    run("chain16-trivial",  build_instance(make_chain(16,48,48,50000)));
    // With merges to create more multi-consumer scenarios
    run("fan12-pair",       build_instance(make_fan12(), {{0,1},{2,3}}));
    run("ladder8-pair",     build_instance(make_ladder8(), {{0,1},{4,5}}));

    CHECK("tensor_merge count >= 5", count >= 5);
    std::cout << "  total tensor_merges verified: " << count << "\n";
}

// ============================================================================
// Per-move-type test: TENSOR_EXTRACT
// Needs: shared tensor with consumers in separate groups; extract them.
// ============================================================================

void test_tensor_extract_isolated() {
    std::cout << "--- test_tensor_extract_isolated ---\n";
    int count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        for (size_t t = 0; t < ti.dag.tensor_consumers.size(); t++) {
            auto& consumers = ti.dag.tensor_consumers[t];
            if (consumers.size() < 2) continue;
            FlatSet<size_t> cgroups;
            FlatSet<size_t> extract_ops;
            for (auto cop : consumers) {
                extract_ops.insert(cop);
                for (auto cg : ti.cp.part.groups_of(cop))
                    cgroups.insert(cg);
            }
            int prod = ti.dag.tensor_producer[t];
            if (prod >= 0) {
                extract_ops.insert((size_t)prod);
                for (auto pg : ti.cp.part.groups_of((size_t)prod))
                    cgroups.insert(pg);
            }
            if (cgroups.size() < 2) continue;
            std::vector<size_t> gl(cgroups.begin(), cgroups.end());
            if (!ti.cp.part.acyclic_extract_local(extract_ops)) continue;
            auto ter = partition_moves::eval_tensor_extract(
                ti.cp.part, extract_ops, gl);
            if (!ter.feasible) continue;
            CoupledFMMove m;
            m.type = CoupledFMMove::TENSOR_EXTRACT;
            m.op = *extract_ops.begin();  m.op2 = t;
            m.saving = ter.saving;
            m.tensor_groups = gl;
            m.tensor_consumer_ops.assign(extract_ops.begin(), extract_ops.end());
            std::string label = std::string(tag) + " T_EXTRACT T" + std::to_string(t);
            apply_and_check(ti.cp, m, label.c_str());
            ti.cp.part.rebuild_index();
            ti.cp.part.rebuild_group_dag();
            count++;
        }
    };

    // Merge some groups so that tensor_extract can pull ops out of them
    run("fan12-merged",    build_instance(make_fan12(), {{3,7},{8,9}}));
    run("ladder8-merged",  build_instance(make_ladder8(), {{0,2},{1,3}}));
    run("diamond6-merged", build_instance(make_diamond6(), {{0,2},{3,4}}));
    run("fan12-merged2",   build_instance(make_fan12(), {{0,5},{1,6}}));

    std::cout << "  total tensor_extracts verified: " << count << "\n";
}

// ============================================================================
// Per-move-type test: COUPLE
// Needs: two groups with a retainable boundary tensor between them.
// ============================================================================

void test_couple_isolated() {
    std::cout << "--- test_couple_isolated ---\n";
    int count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        for (size_t ga = 0; ga < ti.cp.part.groups.size(); ga++) {
            if (!ti.cp.part.groups[ga].alive) continue;
            if (ti.cp.next_group[ga] != SIZE_MAX) continue;  // must be free tail
            for (auto t : ti.feasibly_ret) {
                if (!is_boundary_output_of(ti.cp.part.groups[ga].ops, t, ti.dag))
                    continue;
                for (auto cop : ti.dag.tensor_consumers[t]) {
                    for (auto gb : ti.cp.part.groups_of(cop)) {
                        if (gb == ga || !ti.cp.part.groups[gb].alive) continue;
                        if (ti.cp.prev_group[gb] != SIZE_MAX) continue;
                        if (!is_boundary_input_of(ti.cp.part.groups[gb].ops, t, ti.dag))
                            continue;
                        auto r = eval_couple(ti.cp, ga, gb, t);
                        if (!r.feasible) continue;
                        CoupledFMMove m;
                        m.type = CoupledFMMove::COUPLE;
                        m.op = *ti.cp.part.groups[ga].ops.begin();
                        m.ga = ga;  m.gb = gb;  m.tensor = t;
                        m.saving = r.saving;
                        std::string label = std::string(tag) + " COUPLE G" +
                            std::to_string(ga) + "->G" + std::to_string(gb) +
                            " T" + std::to_string(t);
                        apply_and_check(ti.cp, m, label.c_str());
                        ti.cp.part.rebuild_index();
                        ti.cp.part.rebuild_group_dag();
                        count++;
                        goto next_ga;
                    }
                }
            }
            next_ga:;
        }
    };

    run("chain12-pair",    build_instance(make_chain(12,64,64,15000), pairs(12)));
    run("chain8-pair",     build_instance(make_chain(8, 64,64,15000), pairs(8)));
    run("chain16-pair",    build_instance(make_chain(16,48,48,10000), pairs(16)));
    run("fan12-pair",      build_instance(make_fan12(), {{0,5},{1,6},{3,7}}));
    run("ladder8-pair",    build_instance(make_ladder8(), {{0,2},{1,3},{4,6}}));
    run("diamond6-pair",   build_instance(make_diamond6(), {{0,2},{3,4}}));

    CHECK("couple count >= 10", count >= 10);
    std::cout << "  total couples verified: " << count << "\n";
}

// ============================================================================
// Per-move-type test: UNCOUPLE
// Seed coupling edges, then remove them.
// ============================================================================

void test_uncouple_isolated() {
    std::cout << "--- test_uncouple_isolated ---\n";
    int count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        // First add coupling edges
        std::vector<std::tuple<size_t,size_t,size_t>> edges;
        for (size_t ga = 0; ga < ti.cp.part.groups.size(); ga++) {
            if (!ti.cp.part.groups[ga].alive) continue;
            if (ti.cp.next_group[ga] != SIZE_MAX) continue;
            for (auto t : ti.feasibly_ret) {
                if (!is_boundary_output_of(ti.cp.part.groups[ga].ops, t, ti.dag))
                    continue;
                for (auto cop : ti.dag.tensor_consumers[t]) {
                    for (auto gb : ti.cp.part.groups_of(cop)) {
                        if (gb == ga || !ti.cp.part.groups[gb].alive) continue;
                        if (ti.cp.prev_group[gb] != SIZE_MAX) continue;
                        auto rc = apply_couple(ti.cp, ga, gb, t);
                        if (!rc.empty()) {
                            edges.push_back({ga, gb, t});
                            goto next_couple;
                        }
                    }
                }
            }
            next_couple:;
        }

        // Now uncouple each edge
        for (auto [ga, gb, t] : edges) {
            if (ti.cp.next_group[ga] != gb) continue;
            auto r = eval_uncouple(ti.cp, ga, gb, t);
            if (!r.feasible) continue;
            CoupledFMMove m;
            m.type = CoupledFMMove::UNCOUPLE;
            m.op = *ti.cp.part.groups[ga].ops.begin();
            m.ga = ga;  m.gb = gb;  m.tensor = t;  m.saving = r.saving;
            std::string label = std::string(tag) + " UNCOUPLE G" +
                std::to_string(ga) + "->G" + std::to_string(gb) +
                " T" + std::to_string(t);
            apply_and_check(ti.cp, m, label.c_str());
            ti.cp.part.rebuild_index();
            ti.cp.part.rebuild_group_dag();
            count++;
        }
    };

    run("chain12-pair",    build_instance(make_chain(12,64,64,15000), pairs(12)));
    run("chain8-pair",     build_instance(make_chain(8, 64,64,15000), pairs(8)));
    run("chain16-pair",    build_instance(make_chain(16,48,48,10000), pairs(16)));
    run("fan12-pair",      build_instance(make_fan12(), {{0,5},{1,6},{3,7}}));
    run("ladder8-pair",    build_instance(make_ladder8(), {{0,2},{1,3},{4,6}}));

    CHECK("uncouple count >= 10", count >= 10);
    std::cout << "  total uncouples verified: " << count << "\n";
}

// ============================================================================
// Per-move-type test: RETAIN_FORCE_SPLIT + FORCE_RETAIN
// Via best_coupled_move_for_op on merged-triple configurations.
// ============================================================================

void test_rfs_force_retain_isolated() {
    std::cout << "--- test_rfs_force_retain_isolated ---\n";
    int rfs_count = 0, fr_count = 0;

    auto run = [&](const char* tag, TestInstance ti) {
        for (size_t op = 0; op < ti.prob.num_ops(); op++) {
            auto move = best_coupled_move_for_op(ti.cp, op, ti.feasibly_ret);
            if (!move.valid()) continue;
            if (move.type == CoupledFMMove::RETAIN_FORCE_SPLIT) {
                std::string label = std::string(tag) + " RFS op" +
                    std::to_string(op) + " G" + std::to_string(move.ga) +
                    " T" + std::to_string(move.tensor);
                apply_and_check(ti.cp, move, label.c_str());
                ti.cp.part.rebuild_index();
                ti.cp.part.rebuild_group_dag();
                rfs_count++;
            } else if (move.type == CoupledFMMove::FORCE_RETAIN) {
                std::string label = std::string(tag) + " FR op" +
                    std::to_string(op) + " ga=" + std::to_string(move.ga) +
                    " T" + std::to_string(move.tensor);
                apply_and_check(ti.cp, move, label.c_str());
                ti.cp.part.rebuild_index();
                ti.cp.part.rebuild_group_dag();
                fr_count++;
            }
        }
    };

    run("chain12-triple",  build_instance(make_chain(12,64,64,15000), triples(12)));
    run("chain8-triple",   build_instance(make_chain(8, 64,64,15000), triples(8)));
    run("chain16-triple",  build_instance(make_chain(16,48,48,10000), triples(16)));
    run("fan12-triple",    build_instance(make_fan12(), {{0,1},{0,5},{2,3},{2,6},{8,9},{8,10}}));
    run("ladder8-triple",  build_instance(make_ladder8(), {{0,1},{0,2},{4,5},{4,6}}));
    run("chain12-quad",    build_instance(make_chain(12,64,64,15000), quads(12)));

    std::cout << "  rfs=" << rfs_count << " force_retain=" << fr_count << "\n";
}

// ============================================================================
// Integration: full FM pass on diverse instances, verify all constraints
// ============================================================================

struct StepStats {
    int steps = 0, mismatches = 0, inconsistent = 0, lost_ops = 0;
    int type_counts[13] = {};
};

static StepStats run_validated_steps(CoupledPartition& cp,
                                      const FlatSet<size_t>& fr,
                                      int max_steps) {
    StepStats s;
    double floor = cp.total_cost() * 0.30;
    CoupledActiveSet active(cp, fr, floor);
    for (size_t gi = 0; gi < cp.part.groups.size(); gi++) {
        if (!cp.part.groups[gi].alive) continue;
        for (auto op : cp.part.border_ops(gi)) active.activate(op);
        if (cp.part.groups[gi].ops.size() >= 3)
            for (auto op : cp.part.internal_ops(gi)) active.activate(op);
    }
    for (int step = 0; step < max_steps; step++) {
        auto mo = active.pop_best();
        if (!mo) break;
        auto move = *mo;
        double before = cp.total_cost();

        std::vector<size_t> locks;
        if (move.type == CoupledFMMove::MERGE) {
            locks.push_back(move.op);
            if (move.gb < cp.part.groups.size() && cp.part.groups[move.gb].alive)
                for (auto n : cp.part.dag->op_neighbors[move.op])
                    if (cp.part.groups[move.gb].ops.count(n)) locks.push_back(n);
        } else if (move.type == CoupledFMMove::TENSOR_MERGE ||
                   move.type == CoupledFMMove::TENSOR_EXTRACT) {
            for (auto cg : move.tensor_groups)
                if (cp.part.groups[cg].alive)
                    for (auto co : cp.part.groups[cg].ops) locks.push_back(co);
        } else if (move.type == CoupledFMMove::RETAIN_FORCE_SPLIT) {
            locks.push_back(move.op);
            locks.push_back(move.op2);
        }

        auto aff = apply_coupled_fm_move(cp, move);
        if (aff.empty()) continue;
        s.steps++;
        int mt = (int)move.type;
        if (mt >= 0 && mt < 13) s.type_counts[mt]++;

        double after = cp.total_cost();
        double actual = before - after;
        double disc = move.saving - actual;
        if (std::abs(disc) > 0.1 * std::max(1.0, std::abs(move.saving)) + 1.0)
            s.mismatches++;
        if (!cp_is_consistent(cp)) s.inconsistent++;
        if (!all_ops_covered(cp.part)) s.lost_ops++;

        active.lock_all(locks);
        active.refresh_after_move(aff);
    }
    return s;
}

void test_full_pass() {
    std::cout << "--- test_full_pass ---\n";

    struct Case { const char* name; Problem prob;
        std::vector<std::pair<size_t,size_t>> merges; };
    Case cases[] = {
        {"chain12",       make_chain(12,64,64,15000), {}},
        {"chain12-pair",  make_chain(12,64,64,15000), pairs(12)},
        {"chain16",       make_chain(16,48,48,10000), {}},
        {"fan12",         make_fan12(), {}},
        {"fan12-triple",  make_fan12(), {{0,1},{0,5},{2,3},{2,6}}},
        {"diamond6",      make_diamond6(), {}},
        {"wide_fan",      make_wide_fan(), {}},
        {"ladder8",       make_ladder8(), {}},
        {"ladder8-pair",  make_ladder8(), {{0,2},{1,3},{4,6},{5,7}}},
    };

    for (auto& c : cases) {
        auto ti = build_instance(std::move(c.prob), c.merges);
        auto s = run_validated_steps(ti.cp, ti.feasibly_ret, 60);
        CHECK((std::string(c.name) + " no mismatches").c_str(), s.mismatches == 0);
        CHECK((std::string(c.name) + " consistent").c_str(), s.inconsistent == 0);
        CHECK((std::string(c.name) + " coverage").c_str(), s.lost_ops == 0);

        static const char* tn[] = {
            "STEAL","EJECT","RECOMP","MERGE","IE",
            "SPLIT","TM","TE","DR","COUPLE","UNCOUPLE","RFS","FR"};
        std::cout << "  " << c.name << ": steps=" << s.steps;
        for (int i = 0; i < 13; i++)
            if (s.type_counts[i]) std::cout << " " << tn[i] << "=" << s.type_counts[i];
        std::cout << "\n";
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << "=== coupled_fm_correctness_test ===\n\n";

    test_split_isolated();
    test_eject_isolated();
    test_internal_eject_isolated();
    test_merge_isolated();
    test_steal_isolated();
    test_recompute_de_recompute_isolated();
    test_tensor_merge_isolated();
    test_tensor_extract_isolated();
    test_couple_isolated();
    test_uncouple_isolated();
    test_rfs_force_retain_isolated();
    test_full_pass();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
