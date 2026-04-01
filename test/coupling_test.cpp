// coupling_test.cpp
//
// Unit tests for CoupledPartition, eval/apply_couple, eval/apply_uncouple,
// to_solution, and the coupling_search greedy descent.
//
// Problem topology used throughout: chain6_tight — 6 PW ops in a linear chain
//   Op0:T0→T1, Op1:T1→T2, ..., Op5:T5→T6.  retainable = {T1..T5}.
// Trivial partition: G0={Op0}, G1={Op1}, ..., G5={Op5}.
// After finalize: G0 produces T1 as boundary output; G1 consumes T1 as boundary input.

#include "search/coupling_search.h"
#include "search/partition_moves.h"
#include "core/cost_cache.h"
#include "solution/solution.h"
#include <chrono>
#include <iostream>

static int g_pass = 0, g_fail = 0;
static void CHECK(const char* l, bool c) {
    if (c) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << "\n"; }
}
static void CHECK_EQ(const char* l, size_t got, size_t exp) {
    if (got == exp) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << got << " exp=" << exp << "\n"; }
}
static void CHECK_LE(const char* l, double got, double exp) {
    if (got <= exp + 0.5) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << l << " got=" << got << " exp≤" << exp << "\n"; }
}

using Clock = std::chrono::steady_clock;

// ============================================================================
// Problem + partition helpers
// ============================================================================

static Problem make_chain6_tight() {
    Problem p;
    for (int i = 0; i <= 6; i++) p.tensors.push_back({128,128});
    for (int i = 0; i < 6; i++)
        p.ops.push_back({OpType::Pointwise,{(size_t)i},{(size_t)(i+1)},500});
    p.fast_memory_capacity = 40000;
    p.slow_memory_bandwidth = 10;
    p.native_w = 128; p.native_h = 128;
    p.retainable_tensors = {1,2,3,4,5};
    return p;
}

// Trivial partition (one group per op), finalized.
// Returns (part, feasibly_ret).
static std::pair<Partition, FlatSet<size_t>>
make_trivial_finalized(const Problem& p, const DAG& d) {
    Partition part = Partition::trivial(p, d);
    part.finalize();
    auto fr = compute_feasibly_retainable(p, d);
    return {std::move(part), fr};
}

// Build a CoupledPartition from a trivial partition of chain6_tight.
// Problem and DAG use static storage so pointers in Partition remain valid.
static CoupledPartition make_chain_cp() {
    static Problem p = make_chain6_tight();
    static DAG    d = DAG::build(p);
    Partition part = Partition::trivial(p, d);
    part.finalize();
    CoupledPartition cp;
    cp.init_from(std::move(part));
    return cp;
}

// ============================================================================
// 1. init_from
// ============================================================================

void test_init_from() {
    std::cout << "--- test_init_from ---\n";
    auto cp = make_chain_cp();
    size_t n = cp.part.groups.size();
    CHECK_EQ("6 groups", n, 6);
    for (size_t g = 0; g < n; g++) {
        CHECK("next_group[g] = SIZE_MAX", cp.next_group[g] == SIZE_MAX);
        CHECK("prev_group[g] = SIZE_MAX", cp.prev_group[g] == SIZE_MAX);
    }
    CHECK("retained empty", cp.retained.empty());
}

// ============================================================================
// 2. Chain helpers
// ============================================================================

void test_chain_helpers_single() {
    std::cout << "--- test_chain_helpers_single ---\n";
    auto cp = make_chain_cp();
    // No coupling — every group is its own chain of length 1.
    for (size_t g = 0; g < cp.part.groups.size(); g++) {
        CHECK("head = g", cp.chain_head(g) == g);
        CHECK("tail = g", cp.chain_tail(g) == g);
        auto ch = cp.chain_of(g);
        CHECK_EQ("chain_of len=1", ch.size(), 1);
        CHECK("chain_of[0]=g", ch[0] == g);
    }
}

void test_chain_helpers_after_couple() {
    std::cout << "--- test_chain_helpers_after_couple ---\n";
    auto cp = make_chain_cp();
    // Manually build a chain G0→G1→G2 by setting next/prev directly.
    // (We test the helpers, not apply_couple here.)
    cp.next_group[0] = 1;  cp.prev_group[1] = 0;
    cp.next_group[1] = 2;  cp.prev_group[2] = 1;

    CHECK("head of 0 = 0", cp.chain_head(0) == 0);
    CHECK("head of 1 = 0", cp.chain_head(1) == 0);
    CHECK("head of 2 = 0", cp.chain_head(2) == 0);
    CHECK("tail of 0 = 2", cp.chain_tail(0) == 2);
    CHECK("tail of 2 = 2", cp.chain_tail(2) == 2);

    auto ch = cp.chain_of(1);  // start from middle — should go to head first
    CHECK_EQ("chain_of(1) len=3", ch.size(), 3);
    CHECK("chain_of[0]=0", ch[0] == 0);
    CHECK("chain_of[1]=1", ch[1] == 1);
    CHECK("chain_of[2]=2", ch[2] == 2);

    // G3 is still independent
    CHECK("head of 3 = 3", cp.chain_head(3) == 3);
    CHECK_EQ("chain_of(3) len=1", cp.chain_of(3).size(), 1);
}

// ============================================================================
// 3. apply_couple / apply_uncouple state transitions
// ============================================================================

void test_apply_couple() {
    std::cout << "--- test_apply_couple ---\n";
    auto cp = make_chain_cp();
    // G0 produces T1 (tensor index 1), G1 consumes T1.
    auto affected = apply_couple(cp, 0, 1, 1);
    CHECK("affected non-empty", !affected.empty());
    CHECK("next_group[0]=1",   cp.next_group[0] == 1);
    CHECK("prev_group[1]=0",   cp.prev_group[1] == 0);
    CHECK("retained has (0,1)", cp.retained.count({0,1}) > 0);
    CHECK("T1 in retained",    cp.retained[{0,1}].count(1) > 0);
    // Other groups unchanged.
    CHECK("next_group[1]=SIZE_MAX", cp.next_group[1] == SIZE_MAX);
    CHECK("prev_group[0]=SIZE_MAX", cp.prev_group[0] == SIZE_MAX);
}

void test_apply_uncouple_restores() {
    std::cout << "--- test_apply_uncouple_restores ---\n";
    auto cp = make_chain_cp();
    apply_couple(cp, 0, 1, 1);
    auto affected = apply_uncouple(cp, 0, 1, 1);
    CHECK("affected non-empty", !affected.empty());
    CHECK("next_group[0]=SIZE_MAX", cp.next_group[0] == SIZE_MAX);
    CHECK("prev_group[1]=SIZE_MAX", cp.prev_group[1] == SIZE_MAX);
    CHECK("retained empty",         cp.retained.empty());
}

void test_apply_couple_idempotent_guard() {
    std::cout << "--- test_apply_couple_idempotent_guard ---\n";
    auto cp = make_chain_cp();
    apply_couple(cp, 0, 1, 1);
    // G0 already has outgoing coupling — second apply_couple must fail.
    auto r = apply_couple(cp, 0, 2, 2);
    CHECK("second couple fails", r.empty());
    CHECK("next_group[0] still 1", cp.next_group[0] == 1);
}

void test_apply_couple_multiple_tensors() {
    std::cout << "--- test_apply_couple_multiple_tensors ---\n";
    // The retained set for one edge can hold multiple tensors.
    // We need a problem where two tensors flow from G0 to G1.
    // chain6_tight only has one tensor per boundary, so test
    // by manually forcing — just verify state is consistent after
    // a second apply_couple on the same edge (adds to retained).
    auto cp = make_chain_cp();
    apply_couple(cp, 0, 1, 1);
    // Manually add a second tensor to the same edge (as if T2 also flows G0→G1)
    // This bypasses the checks but tests the retained map behaviour.
    cp.retained[{0,1}].insert(2);
    CHECK("two tensors in retained", cp.retained[{0,1}].size() == 2);
    // Uncoupling T1 only removes T1; T2 remains; edge survives.
    auto r = apply_uncouple(cp, 0, 1, 1);
    CHECK("uncouple T1 succeeded", !r.empty());
    CHECK("edge still exists",      cp.next_group[0] == 1);
    CHECK("only T2 remains",        cp.retained[{0,1}].count(2) && cp.retained[{0,1}].size() == 1);
    // Uncoupling T2 dissolves the edge.
    apply_uncouple(cp, 0, 1, 2);
    CHECK("edge dissolved",         cp.next_group[0] == SIZE_MAX);
    CHECK("retained empty",         cp.retained.empty());
}

// ============================================================================
// 4. eval_couple
// ============================================================================

void test_eval_couple_feasible() {
    std::cout << "--- test_eval_couple_feasible ---\n";
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    auto [part, fr] = make_trivial_finalized(p, d);
    CoupledPartition cp; cp.init_from(std::move(part));

    // T1 (index 1) flows G0→G1 and is retainable.
    if (!fr.count(1)) { std::cout << "  SKIP: T1 not feasibly retainable\n"; return; }

    auto ev = eval_couple(cp, 0, 1, 1);
    CHECK("eval_couple feasible", ev.feasible);
    // Saving may be negative (retaining costs more than reloading in some configs) —
    // we only require feasibility here.
    std::cout << "    saving=" << ev.saving << "\n";
}

void test_eval_couple_infeasible_ga_has_next() {
    std::cout << "--- test_eval_couple_infeasible_ga_has_next ---\n";
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    auto [part, fr] = make_trivial_finalized(p, d);
    CoupledPartition cp; cp.init_from(std::move(part));

    // Couple G0→G1 first, then try to couple G0→G2 — must fail.
    apply_couple(cp, 0, 1, 1);
    auto ev = eval_couple(cp, 0, 2, 2);
    CHECK("infeasible when ga has outgoing", !ev.feasible);
}

void test_eval_couple_infeasible_gb_has_prev() {
    std::cout << "--- test_eval_couple_infeasible_gb_has_prev ---\n";
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    auto [part, fr] = make_trivial_finalized(p, d);
    CoupledPartition cp; cp.init_from(std::move(part));

    // Couple G0→G1, then try to couple G2→G1 — G1 already has prev.
    apply_couple(cp, 0, 1, 1);
    auto ev = eval_couple(cp, 2, 1, 2);
    CHECK("infeasible when gb has incoming", !ev.feasible);
}

void test_eval_couple_infeasible_wrong_tensor() {
    std::cout << "--- test_eval_couple_infeasible_wrong_tensor ---\n";
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    auto [part, fr] = make_trivial_finalized(p, d);
    CoupledPartition cp; cp.init_from(std::move(part));

    // T2 is boundary output of G1 but NOT of G0 — must fail.
    auto ev = eval_couple(cp, 0, 1, 2);
    CHECK("infeasible: T not boundary output of ga", !ev.feasible);
    // T1 is boundary output of G0 but NOT boundary input of G2 in chain6_tight.
    auto ev2 = eval_couple(cp, 0, 2, 1);
    CHECK("infeasible: T not boundary input of gb", !ev2.feasible);
}

// ============================================================================
// 5. eval_uncouple
// ============================================================================

void test_eval_uncouple_feasible() {
    std::cout << "--- test_eval_uncouple_feasible ---\n";
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    auto [part, fr] = make_trivial_finalized(p, d);
    CoupledPartition cp; cp.init_from(std::move(part));
    if (!fr.count(1)) { std::cout << "  SKIP: T1 not feasibly retainable\n"; return; }

    auto ev_c = eval_couple(cp, 0, 1, 1);
    if (!ev_c.feasible) { std::cout << "  SKIP: couple not feasible\n"; return; }
    apply_couple(cp, 0, 1, 1);

    auto ev_u = eval_uncouple(cp, 0, 1, 1);
    CHECK("uncouple feasible", ev_u.feasible);
    // Saving of uncouple should be negation of couple saving (approximate).
    double diff = ev_c.saving + ev_u.saving;
    CHECK("saving is symmetric", diff < 1.0 && diff > -1.0);
    std::cout << "    couple.saving=" << ev_c.saving
              << " uncouple.saving=" << ev_u.saving << "\n";
}

void test_eval_uncouple_infeasible_no_edge() {
    std::cout << "--- test_eval_uncouple_infeasible_no_edge ---\n";
    auto cp = make_chain_cp();
    // No coupling exists — uncouple must fail.
    auto ev = eval_uncouple(cp, 0, 1, 1);
    CHECK("infeasible: no edge", !ev.feasible);
}

// ============================================================================
// 6. to_solution
// ============================================================================

void test_to_solution_no_coupling() {
    std::cout << "--- test_to_solution_no_coupling ---\n";
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    auto [part, fr] = make_trivial_finalized(p, d);
    CoupledPartition cp; cp.init_from(std::move(part));

    auto sol = cp.to_solution();
    auto vr = sol.validate();
    CHECK("valid (no coupling)", vr.valid);
    if (!vr.valid) std::cout << "    error: " << vr.error << "\n";
    CHECK_EQ("6 steps (trivial)", sol.num_steps(), 6);
    // No retain_these anywhere.
    for (size_t i = 0; i < sol.num_steps(); i++)
        CHECK("retain_these empty", sol.step(i).retain_these.empty());
    // All ops covered.
    FlatSet<size_t> cov;
    for (size_t i = 0; i < sol.num_steps(); i++)
        for (auto op : sol.step(i).subgraph.ops()) cov.insert(op);
    CHECK_EQ("all 6 ops", cov.size(), 6);
}

void test_to_solution_with_coupling() {
    std::cout << "--- test_to_solution_with_coupling ---\n";
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    auto [part, fr] = make_trivial_finalized(p, d);
    CoupledPartition cp; cp.init_from(std::move(part));
    if (!fr.count(1)) { std::cout << "  SKIP\n"; return; }

    // Couple G0→G1 with T1.
    apply_couple(cp, 0, 1, 1);

    auto sol = cp.to_solution();
    auto vr = sol.validate();
    CHECK("valid (with coupling)", vr.valid);
    if (!vr.valid) std::cout << "    error: " << vr.error << "\n";
    CHECK_EQ("still 6 steps", sol.num_steps(), 6);

    // Find the step that corresponds to G0 (the step that retains T1).
    // It must come immediately before the step that has T1 as boundary input.
    bool found_retain = false;
    for (size_t i = 0; i + 1 < sol.num_steps(); i++) {
        if (sol.step(i).retain_these.count(1)) {
            found_retain = true;
            // The next step must have T1 as a boundary input.
            CHECK("next step has T1 as boundary input",
                  sol.step(i+1).subgraph.boundary_inputs().count(1) > 0);
        }
    }
    CHECK("retain_these T1 found", found_retain);
}

void test_to_solution_chain_order() {
    std::cout << "--- test_to_solution_chain_order ---\n";
    // Couple G0→G1→G2 (chain of 3). to_solution must emit them in order.
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    auto [part, fr] = make_trivial_finalized(p, d);
    CoupledPartition cp; cp.init_from(std::move(part));

    if (fr.count(1)) apply_couple(cp, 0, 1, 1);
    if (fr.count(2)) apply_couple(cp, 1, 2, 2);

    auto sol = cp.to_solution();
    CHECK("valid (chain)", sol.validate().valid);

    // The three chained groups must appear in consecutive positions and in
    // chain order (G0 before G1 before G2).
    std::vector<size_t> step_for_op(6, SIZE_MAX);
    for (size_t i = 0; i < sol.num_steps(); i++)
        for (auto op : sol.step(i).subgraph.ops())
            step_for_op[op] = i;

    if (fr.count(1) && fr.count(2)) {
        CHECK("Op0 before Op1", step_for_op[0] < step_for_op[1]);
        CHECK("Op1 before Op2", step_for_op[1] < step_for_op[2]);
        CHECK("G0,G1 adjacent", step_for_op[1] == step_for_op[0] + 1);
        CHECK("G1,G2 adjacent", step_for_op[2] == step_for_op[1] + 1);
    }
}

// ============================================================================
// 7. coupling_search end-to-end
// ============================================================================

void test_search_no_retain() {
    std::cout << "--- test_search_no_retain ---\n";
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    auto [part, fr] = make_trivial_finalized(p, d);
    FlatSet<size_t> empty_fr;

    auto sol = coupling_search(p, d, std::move(part), empty_fr,
                               Clock::now() + std::chrono::milliseconds(100));
    CHECK("valid (no retain)", sol.validate().valid);
    // No retainment should be added.
    for (size_t i = 0; i < sol.num_steps(); i++)
        CHECK("no retain_these", sol.step(i).retain_these.empty());
}

void test_search_valid_with_retain() {
    std::cout << "--- test_search_valid_with_retain ---\n";
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    auto [part, fr] = make_trivial_finalized(p, d);

    auto sol = coupling_search(p, d, std::move(part), fr,
                               Clock::now() + std::chrono::milliseconds(200));
    auto vr = sol.validate();
    CHECK("valid", vr.valid);
    if (!vr.valid) std::cout << "    error: " << vr.error << "\n";

    FlatSet<size_t> cov;
    for (size_t i = 0; i < sol.num_steps(); i++)
        for (auto op : sol.step(i).subgraph.ops()) cov.insert(op);
    CHECK_EQ("all 6 ops", cov.size(), 6);
    std::cout << "    cost=" << sol.total_latency() << "\n";
}

void test_search_not_worse_than_baseline() {
    std::cout << "--- test_search_not_worse_than_baseline ---\n";
    auto p = make_chain6_tight();
    auto d = DAG::build(p);

    CostCache cache;
    Partition base_part = Partition::trivial(p, d);
    base_part.cache = &cache;
    base_part.finalize(&cache);

    double base_cost = Solution::from_partition(p, d, base_part, 1, &cache).total_latency();

    auto fr = compute_feasibly_retainable(p, d);
    auto sol = coupling_search(p, d, base_part, fr,
                               Clock::now() + std::chrono::milliseconds(200));
    CHECK("valid", sol.validate().valid);
    CHECK_LE("coupling ≤ baseline", sol.total_latency(), base_cost);
    std::cout << "    baseline=" << base_cost << " after=" << sol.total_latency() << "\n";
}

// ============================================================================
// RETAIN_FORCE_SPLIT tests
//
// Fixture: chain6_tight with G0={Op0,Op1} (merged), G1={Op2}, ..., G4={Op5}.
// T1 is internal to G0 (Op0 produces it, Op1 consumes it).
// ============================================================================

// Build a CoupledPartition where G0 = {Op0, Op1} (merged).
// Returns (cp, merged_group_index).
static std::pair<CoupledPartition, size_t> make_merged_cp() {
    static Problem p = make_chain6_tight();
    static DAG    d = DAG::build(p);
    Partition part = Partition::trivial(p, d);
    // Merge G0 (Op0) and G1 (Op1): G0 absorbs G1.
    partition_moves::apply_merge(part, 0, 1);
    part.finalize();  // sets .sg for all alive groups
    CoupledPartition cp;
    cp.init_from(std::move(part));
    return {std::move(cp), 0};
}

void test_eval_retain_force_split_feasible() {
    std::cout << "--- test_eval_retain_force_split_feasible ---\n";
    auto [cp, gm] = make_merged_cp();

    // Verify setup: T1 is internal to gm (Op0 produces it, Op1 consumes it).
    CHECK("T1 not boundary output of merged", !is_boundary_output_of(cp.part.groups[gm].ops, 1, *cp.part.dag));
    CHECK("Op0 and Op1 in merged group", cp.part.groups[gm].ops.count(0) && cp.part.groups[gm].ops.count(1));

    // eval: split at (Op0=0, Op1=1), retain T1.
    // Saving is 0 here: merged cost == split+coupled cost (no memory pressure in
    // this problem, T1 was already ephemeral so no extra I/O either way).
    auto ev = eval_retain_force_split(cp, gm, 0, 1, 1);
    CHECK("eval feasible", ev.feasible);
    CHECK("saving >= 0", ev.saving >= 0);
    std::cout << "    saving=" << ev.saving << "\n";
}

void test_eval_retain_force_split_infeasible_op_not_in_group() {
    std::cout << "--- test_eval_retain_force_split_infeasible_op_not_in_group ---\n";
    auto [cp, gm] = make_merged_cp();
    // Op2 is not in gm (it's in a different group)
    auto ev = eval_retain_force_split(cp, gm, 0, 2, 1);
    CHECK("op_b not in group → infeasible", !ev.feasible);
}

void test_eval_retain_force_split_infeasible_t_already_boundary() {
    std::cout << "--- test_eval_retain_force_split_infeasible_t_already_boundary ---\n";
    auto cp = make_chain_cp();  // trivial: T1 is already a boundary output of G0
    // G0={Op0}, T1 is boundary output. eval_retain_force_split should bail early.
    auto ev = eval_retain_force_split(cp, 0, 0, 1, 1);  // op_b=1 not in G0 anyway
    CHECK("t already boundary or op_b missing → infeasible", !ev.feasible);
}

void test_apply_retain_force_split_basic() {
    std::cout << "--- test_apply_retain_force_split_basic ---\n";
    auto [cp, gm] = make_merged_cp();
    size_t n_before = cp.part.groups.size();

    auto aff = apply_retain_force_split(cp, gm, 0, 1, 1);
    CHECK("apply succeeded", !aff.empty());

    // One extra group slot created
    CHECK("group count increased", cp.part.groups.size() > n_before);

    // Find the two resulting groups (both in aff)
    size_t ga = SIZE_MAX, gb = SIZE_MAX;
    for (auto idx : aff) {
        if (idx == gm) ga = idx;
        else           gb = idx;
    }
    CHECK("ga = gm", ga == (size_t)gm);
    CHECK("gb found", gb != SIZE_MAX);

    // T1 is now a boundary output of ga and boundary input of gb
    const DAG& d2 = *cp.part.dag;
    CHECK("T1 boundary output of ga", is_boundary_output_of(cp.part.groups[ga].ops, 1, d2));
    CHECK("T1 boundary input of gb",  is_boundary_input_of(cp.part.groups[gb].ops,  1, d2));

    // Chain: ga → gb via T1
    CHECK("ga.next = gb",     cp.next_group[ga] == gb);
    CHECK("gb.prev = ga",     cp.prev_group[gb] == ga);
    CHECK("retained T1",      cp.retained.count({ga, gb}) && cp.retained.at({ga,gb}).count(1));
}

void test_apply_retain_force_split_inherits_chain() {
    std::cout << "--- test_apply_retain_force_split_inherits_chain ---\n";
    // Build merged partition, then also couple G2→G3 via T3 so that G2 has a next.
    // Then split G0 (merged) — verifies ga inherits G0's prev (none here) and
    // gb inherits G0's next (none here).  Slightly simpler case — just verify
    // that an existing chain link on the *next side* transfers to gb.
    auto [cp, gm] = make_merged_cp();

    // Couple the *next* group (whatever has T2 as boundary output) to another group.
    // In the merged fixture: Op2 is in group index 1 (G2 after merge killed G1).
    // Find group containing Op2.
    size_t g2 = SIZE_MAX;
    for (size_t g = 0; g < cp.part.groups.size(); g++) {
        if (cp.part.groups[g].alive && cp.part.groups[g].ops.count(2))
            g2 = g;
    }
    if (g2 == SIZE_MAX) { CHECK("g2 found", false); return; }
    // Find group containing Op3 (T2 consumer in the next group).
    size_t g3 = SIZE_MAX;
    for (size_t g = 0; g < cp.part.groups.size(); g++) {
        if (cp.part.groups[g].alive && cp.part.groups[g].ops.count(3))
            g3 = g;
    }
    if (g3 == SIZE_MAX) { CHECK("g3 found", false); return; }

    // We can't couple gm to g2 yet (T2 is already a boundary output of gm after merge?
    // Actually in merged gm={Op0,Op1}, T2 is NOT produced there — Op1 produces T2.
    // Wait: Op1: T1→T2. So T2 is a boundary output of gm. Let's couple gm→g2 via T2...
    // But first we need gm to be a free tail and g2 to be a free head.
    // Actually, let's just test a simpler scenario: the existing chain link on gm's
    // outgoing side gets transferred to gb after the split.
    // couple gm → g2 via T2 (T2 = boundary output of gm since Op1 is in gm).
    const DAG& d3 = *cp.part.dag;
    CHECK("T2 boundary output of gm", is_boundary_output_of(cp.part.groups[gm].ops, 2, d3));
    CHECK("T2 boundary input of g2",  is_boundary_input_of(cp.part.groups[g2].ops,  2, d3));
    apply_couple(cp, gm, g2, 2);
    CHECK("gm→g2 coupled", cp.next_group[gm] == g2);

    // Now split gm at (Op0, Op1) with T1.
    auto aff = apply_retain_force_split(cp, gm, 0, 1, 1);
    CHECK("apply succeeded", !aff.empty());

    size_t ga = gm, gb = SIZE_MAX;
    for (auto idx : aff) if (idx != ga) gb = idx;
    CHECK("gb found", gb != SIZE_MAX);

    // Chain should now be: ga → gb → g2
    CHECK("ga.next = gb",  cp.next_group[ga] == gb);
    CHECK("gb.next = g2",  cp.next_group[gb] == g2);
    CHECK("g2.prev = gb",  cp.prev_group[g2] == gb);
    // T2 retained across gb→g2
    CHECK("retained T2 on gb→g2", cp.retained.count({gb, g2}) && cp.retained.at({gb,g2}).count(2));
    // T1 retained across ga→gb
    CHECK("retained T1 on ga→gb", cp.retained.count({ga, gb}) && cp.retained.at({ga,gb}).count(1));
}

void test_search_with_merged_groups() {
    std::cout << "--- test_search_with_merged_groups ---\n";
    // Use a partition where G0={Op0,Op1} so T1 is internal.
    // coupling_search should find RETAIN_FORCE_SPLIT and improve cost.
    auto p = make_chain6_tight();
    auto d = DAG::build(p);
    Partition part = Partition::trivial(p, d);
    partition_moves::apply_merge(part, 0, 1);  // merge Op0 and Op1
    part.finalize();

    // Baseline: no coupling
    CostCache cache;
    part.finalize(&cache);
    double baseline = 0;
    for (auto& g : part.groups)
        if (g.alive) baseline += g.cost;

    auto fr = compute_feasibly_retainable(p, d);
    auto sol = coupling_search(p, d, part, fr,
                               Clock::now() + std::chrono::milliseconds(200));
    CHECK("valid", sol.validate().valid);
    CHECK_LE("coupling ≤ baseline", sol.total_latency(), baseline);
    std::cout << "    baseline=" << baseline << " after=" << sol.total_latency() << "\n";
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << "=== coupling_test ===\n";

    test_init_from();
    test_chain_helpers_single();
    test_chain_helpers_after_couple();

    test_apply_couple();
    test_apply_uncouple_restores();
    test_apply_couple_idempotent_guard();
    test_apply_couple_multiple_tensors();

    test_eval_couple_feasible();
    test_eval_couple_infeasible_ga_has_next();
    test_eval_couple_infeasible_gb_has_prev();
    test_eval_couple_infeasible_wrong_tensor();

    test_eval_uncouple_feasible();
    test_eval_uncouple_infeasible_no_edge();

    test_to_solution_no_coupling();
    test_to_solution_with_coupling();
    test_to_solution_chain_order();

    test_search_no_retain();
    test_search_valid_with_retain();
    test_search_not_worse_than_baseline();

    test_eval_retain_force_split_feasible();
    test_eval_retain_force_split_infeasible_op_not_in_group();
    test_eval_retain_force_split_infeasible_t_already_boundary();
    test_apply_retain_force_split_basic();
    test_apply_retain_force_split_inherits_chain();
    test_search_with_merged_groups();

    std::cout << "\n=== " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail ? 1 : 0;
}
