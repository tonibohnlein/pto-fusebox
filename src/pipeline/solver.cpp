#include "pipeline/solver.h"
#include "partition/partition.h"
#include "search/parallel_search.h"
#include "search/solution_search.h"
#include "postopt/post_opt.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <optional>
#include <set>
#include <algorithm>
#include <vector>

// ============================================================================
// Shared group-level DAG structure for ordering algorithms
// ============================================================================

struct GroupDAGInfo {
    struct Group {
        std::vector<size_t> ops;
        Subgraph sg;
        CostResult base_cost;  // cost without retain
    };

    const Problem* prob;
    const DAG* dag;
    std::vector<Group> groups;
    std::vector<std::set<size_t>> preds, succs;
    std::map<size_t, size_t> tensor_producer;  // tensor → group index
    std::vector<int> in_deg;

    static std::optional<GroupDAGInfo> build(const Problem& prob, const DAG& dag,
                                             const Partition& part) {
        GroupDAGInfo gd;
        gd.prob = &prob;
        gd.dag = &dag;

        for (auto& g : part.groups) {
            if (!g.alive) continue;
            std::vector<size_t> ops(g.ops.begin(), g.ops.end());
            auto sg = Subgraph::create(prob, dag, ops);
            if (!sg) continue;
            auto cost = sg->best_cost();
            gd.groups.push_back({ops, std::move(*sg), cost});
        }

        size_t n = gd.groups.size();
        if (n == 0) return std::nullopt;

        gd.preds.resize(n);
        gd.succs.resize(n);

        // Build group DAG based on tensor dependencies.
        // For each boundary input T of group G, find which group contains
        // the op that produces T (via the op-level DAG). This correctly
        // handles recomputed ops where T may be ephemeral in its producing
        // group but still needed as boundary input by another group.
        
        // Map: op → group index (for recomputed ops, track all groups)
        std::vector<std::vector<size_t>> op_groups(prob.num_ops());
        for (size_t i = 0; i < n; i++)
            for (auto op : gd.groups[i].ops)
                op_groups[op].push_back(i);

        for (size_t i = 0; i < n; i++) {
            for (auto t : gd.groups[i].sg.boundary_inputs()) {
                int prod_op = dag.tensor_producer[t];
                if (prod_op < 0) continue;  // graph input, no producer
                for (auto pg : op_groups[prod_op]) {
                    if (pg != i) {
                        gd.preds[i].insert(pg);
                        gd.succs[pg].insert(i);
                    }
                }
            }
        }

        // Also keep tensor_producer map for retain scoring (boundary outputs only)
        for (size_t i = 0; i < n; i++)
            for (auto t : gd.groups[i].sg.boundary_outputs())
                gd.tensor_producer[t] = i;

        gd.in_deg.resize(n, 0);
        for (size_t i = 0; i < n; i++)
            gd.in_deg[i] = (int)gd.preds[i].size();

        return gd;
    }

    size_t size() const { return groups.size(); }

    // Check if any unscheduled group needs tensor t
    bool future_needs(size_t t, const std::vector<bool>& scheduled) const {
        for (size_t i = 0; i < groups.size(); i++) {
            if (scheduled[i]) continue;
            if (groups[i].sg.boundary_inputs().count(t)) return true;
        }
        return false;
    }
};

// ============================================================================
// DFS chain-first ordering (fast, deterministic)
// ============================================================================

static std::vector<size_t> dfs_ordering(const GroupDAGInfo& gd) {
    size_t n = gd.size();
    std::vector<size_t> order;
    std::vector<bool> scheduled(n, false);
    std::set<size_t> ready;
    auto in_deg = gd.in_deg;

    for (size_t i = 0; i < n; i++)
        if (in_deg[i] == 0) ready.insert(i);

    auto retain_score = [&](size_t cand, size_t prev) -> int64_t {
        if (prev >= n) return 0;
        int64_t score = 0;
        size_t prev_sink = gd.groups[prev].sg.sink_tensor();
        if (gd.groups[cand].sg.boundary_inputs().count(prev_sink) &&
            gd.prob->retainable_tensors.count(prev_sink))
            score += gd.prob->tensors[prev_sink].width * gd.prob->tensors[prev_sink].height;
        for (auto t : gd.groups[cand].sg.boundary_inputs())
            if (gd.groups[prev].sg.boundary_inputs().count(t) &&
                gd.prob->retainable_tensors.count(t))
                score += gd.prob->tensors[t].width * gd.prob->tensors[t].height;
        return score;
    };

    auto input_size = [&](size_t gi) -> int64_t {
        int64_t s = 0;
        for (auto t : gd.groups[gi].sg.boundary_inputs())
            s += gd.prob->tensors[t].width * gd.prob->tensors[t].height;
        return s;
    };

    size_t prev = SIZE_MAX;
    while (!ready.empty()) {
        size_t chosen = SIZE_MAX;
        int64_t best_score = -1, best_input = -1;

        // Prefer direct successors of prev
        if (prev < n) {
            for (auto s : gd.succs[prev]) {
                if (!scheduled[s] && ready.count(s)) {
                    int64_t sc = retain_score(s, prev);
                    int64_t inp = input_size(s);
                    if (sc > best_score || (sc == best_score && inp > best_input)) {
                        chosen = s; best_score = sc; best_input = inp;
                    }
                }
            }
        }

        if (chosen == SIZE_MAX) {
            for (auto gi : ready) {
                int64_t sc = retain_score(gi, prev);
                int64_t inp = input_size(gi);
                if (sc > best_score || (sc == best_score && inp > best_input)) {
                    chosen = gi; best_score = sc; best_input = inp;
                }
            }
        }

        ready.erase(chosen);
        order.push_back(chosen);
        scheduled[chosen] = true;
        prev = chosen;

        for (auto v : gd.succs[chosen])
            if (--in_deg[v] == 0) ready.insert(v);
    }

    return order;
}

// ============================================================================
// Beam search ordering with true memory state tracking.
//
// Each beam state tracks which tensors are currently resident in fast memory.
// This enables multi-step retention: a tensor produced at step 0 can remain
// resident through steps 1,2,... and be consumed at step 3 for free.
//
// The expansion evaluates each candidate with the actual resident set,
// decides what to retain, and propagates the memory state forward.
// ============================================================================

struct BeamResult {
    std::vector<size_t> order;
    std::vector<std::set<size_t>> retain_per_step;  // what each step retains
    double total_latency = 0;
};

static BeamResult beam_search_ordering(const GroupDAGInfo& gd, int beam_width) {
    size_t n = gd.size();
    const Problem& prob = *gd.prob;

    struct State {
        std::vector<size_t> order;
        std::vector<int> in_deg;
        std::set<size_t> resident;   // tensors currently in fast memory
        std::vector<std::set<size_t>> retain_per_step;
        double total_latency = 0;
    };

    State init;
    init.in_deg = gd.in_deg;
    std::vector<State> beam = {init};

    for (size_t step = 0; step < n; step++) {
        std::vector<State> candidates;

        for (auto& state : beam) {
            std::vector<bool> scheduled(n, false);
            for (auto g : state.order) scheduled[g] = true;

            std::vector<size_t> ready;
            for (size_t i = 0; i < n; i++)
                if (!scheduled[i] && state.in_deg[i] == 0)
                    ready.push_back(i);

            for (auto gi : ready) {
                // Mark gi as scheduled for future_needs check
                scheduled[gi] = true;

                // Step 1: Prune resident set — drop tensors no future step needs
                std::set<size_t> useful_resident;
                for (auto t : state.resident)
                    if (gd.future_needs(t, scheduled))
                        useful_resident.insert(t);

                // Step 2: Determine what to retain after this step.
                // Candidate: this group's sink tensor, if future steps need it.
                std::set<size_t> retain_these;
                size_t sink = gd.groups[gi].sg.sink_tensor();
                bool want_retain_sink = prob.retainable_tensors.count(sink) &&
                                        gd.future_needs(sink, scheduled);

                // Step 3: Evaluate with full resident set + output retain.
                // Pass ALL useful_resident (including pass-through tensors that
                // this group doesn't use — working_set counts them at full size).
                CostResult cost;
                bool retained_sink = false;

                // Try: all resident + retain output
                if (want_retain_sink) {
                    retain_these.insert(sink);
                    cost = gd.groups[gi].sg.best_cost(useful_resident, retain_these);
                    if (cost.feasible) {
                        retained_sink = true;
                    } else {
                        retain_these.clear();
                    }
                }

                // Fallback 1: all resident, no output retain
                if (!cost.feasible) {
                    cost = gd.groups[gi].sg.best_cost(useful_resident, {});
                }

                // Fallback 2: drop pass-through tensors (only keep what this group uses)
                if (!cost.feasible) {
                    std::set<size_t> only_used;
                    for (auto t : useful_resident)
                        if (gd.groups[gi].sg.boundary_inputs().count(t))
                            only_used.insert(t);
                    useful_resident = only_used;
                    cost = gd.groups[gi].sg.best_cost(useful_resident, {});
                }

                // Fallback 3: no retained tensors at all
                if (!cost.feasible) {
                    useful_resident.clear();
                    cost = gd.groups[gi].sg.best_cost({}, {});
                }

                // Fallback 4: base cost (should always work)
                if (!cost.feasible) {
                    cost = gd.groups[gi].base_cost;
                    useful_resident.clear();
                    retain_these.clear();
                    retained_sink = false;
                }

                // Build next state
                State next;
                next.order = state.order;
                next.order.push_back(gi);
                next.in_deg = state.in_deg;
                for (auto v : gd.succs[gi])
                    next.in_deg[v]--;
                next.total_latency = state.total_latency + cost.latency;

                // Propagate memory state: useful pass-through + newly retained output
                next.resident = useful_resident;
                if (retained_sink)
                    next.resident.insert(sink);

                // Record retain decision for this step:
                // For the solution output, only retain tensors that this step
                // actually produces or consumes (boundary tensors). Pass-through
                // tensors are tracked in the beam state for scoring but may not
                // be supported by the evaluator in tensors_to_retain.
                std::set<size_t> exportable_retain;
                for (auto t : next.resident) {
                    // Keep if it's a boundary output or boundary input of this group
                    if (gd.groups[gi].sg.boundary_outputs().count(t) ||
                        gd.groups[gi].sg.boundary_inputs().count(t))
                        exportable_retain.insert(t);
                }
                next.retain_per_step = state.retain_per_step;
                next.retain_per_step.push_back(exportable_retain);

                candidates.push_back(std::move(next));

                // Restore scheduled flag
                scheduled[gi] = false;
            }
        }

        if (candidates.empty()) break;

        std::sort(candidates.begin(), candidates.end(),
                  [](const State& a, const State& b) {
                      return a.total_latency < b.total_latency;
                  });
        if ((int)candidates.size() > beam_width)
            candidates.resize(beam_width);

        beam = std::move(candidates);
    }

    if (beam.empty()) {
        BeamResult r;
        r.order = dfs_ordering(gd);
        return r;
    }

    BeamResult r;
    r.order = beam[0].order;
    r.retain_per_step = beam[0].retain_per_step;
    r.total_latency = beam[0].total_latency;
    return r;
}

// ============================================================================
// Build Solution from a Partition — tries both DFS and beam search
// ============================================================================

static Solution build_solution(const Problem& prob, const DAG& dag,
                                const Partition& part) {
    auto gd_opt = GroupDAGInfo::build(prob, dag, part);
    if (!gd_opt) return Solution(prob, dag, {});
    auto& gd = *gd_opt;
    size_t n = gd.size();

    // --- Strategy 1: DFS ordering + post-hoc retain ---
    auto dfs_order = dfs_ordering(gd);
    std::vector<ScheduleStep> dfs_steps;
    for (auto idx : dfs_order)
        dfs_steps.push_back({Subgraph(gd.groups[idx].sg), gd.groups[idx].base_cost.config, {}});
    Solution dfs_sol(prob, dag, std::move(dfs_steps));

    // --- Strategy 2: Beam search with true memory state tracking ---
    int beam_width = std::min(20, std::max(5, (int)n));
    auto beam_result = beam_search_ordering(gd, beam_width);

    // Build solution using beam's ordering AND retain decisions.
    // beam_result.retain_per_step[i] contains ALL tensors that step i keeps
    // resident for future steps (both new outputs and pass-through tensors).
    std::vector<ScheduleStep> beam_steps;
    std::set<size_t> beam_entering;  // retained from previous step

    for (size_t i = 0; i < beam_result.order.size(); i++) {
        size_t idx = beam_result.order[i];
        std::set<size_t> retain_these;
        if (i < beam_result.retain_per_step.size())
            retain_these = beam_result.retain_per_step[i];

        // Compute cost with the actual retained state
        auto cost = gd.groups[idx].sg.best_cost(beam_entering, retain_these);
        if (!cost.feasible) {
            // Fallback: try without retain
            retain_these.clear();
            cost = gd.groups[idx].sg.best_cost(beam_entering, {});
        }
        if (!cost.feasible) {
            // Fallback: no resident tensors
            beam_entering.clear();
            retain_these.clear();
            cost = gd.groups[idx].base_cost;
        }

        beam_steps.push_back({Subgraph(gd.groups[idx].sg), cost.config, retain_these});
        beam_entering = retain_these;  // next step sees our full retain set
    }
    Solution beam_sol(prob, dag, std::move(beam_steps));

    // Pick the better one AFTER running retain+recompute on both.
    double dfs_raw = dfs_sol.total_latency();
    double beam_raw = beam_sol.total_latency();

    auto dfs_opt = optimize_retain(prob, dag, std::move(dfs_sol));
    dfs_opt = optimize_recompute(prob, dag, std::move(dfs_opt));
    double dfs_lat = dfs_opt.total_latency();

    auto beam_opt = optimize_retain(prob, dag, std::move(beam_sol));
    beam_opt = optimize_recompute(prob, dag, std::move(beam_opt));
    double beam_lat = beam_opt.total_latency();

    std::cerr << "  DFS:  raw=" << dfs_raw << " → opt=" << dfs_lat;
    if (dfs_lat < dfs_raw - 0.01) 
        std::cerr << " (-" << std::fixed << std::setprecision(1) 
                  << 100.0*(dfs_raw-dfs_lat)/dfs_raw << "%)";
    std::cerr << "\n";
    std::cerr << "  Beam: raw=" << beam_raw << " → opt=" << beam_lat;
    if (beam_lat < beam_raw - 0.01) 
        std::cerr << " (-" << std::fixed << std::setprecision(1) 
                  << 100.0*(beam_raw-beam_lat)/beam_raw << "%)";
    std::cerr << "\n";
    std::cerr << "  Winner: " << (beam_lat < dfs_lat - 0.01 ? "Beam" : "DFS") << "\n";

    if (beam_lat < dfs_lat - 0.01) {
        return beam_opt;
    }
    return dfs_opt;
}

// ============================================================================
// Full pipeline
// ============================================================================

Solution solve(const Problem& prob, TimePoint deadline) {
    DAG dag = DAG::build(prob);

    std::cerr << "Phase 1: Parallel search (init + greedy + FM + evo)...\n";
    ParallelConfig pcfg;
    pcfg.fm.deadline = deadline;
    auto best_part = parallel_search(prob, dag, pcfg);
    double partition_cost = best_part.total_cost();
    std::cerr << "  Partition cost: " << partition_cost 
              << " (" << best_part.num_alive() << " groups)\n";

    std::cerr << "Phase 2: Build solution (DFS + beam + retain/recompute)...\n";
    auto sol = build_solution(prob, dag, best_part);
    double after_build = sol.total_latency();
    std::cerr << "  After build+opt: " << after_build 
              << " (" << sol.num_steps() << " steps)\n";

    // Additional retain+recompute passes
    for (int pass = 1; pass <= 2; pass++) {
        double before = sol.total_latency();
        sol = optimize_retain(prob, dag, std::move(sol));
        sol = optimize_recompute(prob, dag, std::move(sol));
        sol = optimize_retain(prob, dag, std::move(sol));
        if (sol.total_latency() >= before - 0.01) break;
    }
    double after_retain_recomp = sol.total_latency();

    // Phase 3: Solution-level FM search
    std::cerr << "Phase 3: Solution-level FM search...\n";
    SolutionFMConfig sfm_cfg;
    sfm_cfg.deadline = deadline;
    sol = solution_fm_search(prob, dag, std::move(sol), sfm_cfg);
    double after_sol_opt = sol.total_latency();
    if (after_sol_opt < after_build - 0.01)
        std::cerr << "  Solution opt: " << after_build << " → " << after_sol_opt
                  << " (-" << std::fixed << std::setprecision(1)
                  << 100.0*(after_build - after_sol_opt)/after_build << "%)\n";
    else
        std::cerr << "  Solution opt: no improvement\n";

    // Final retain+recompute cleanup
    sol = optimize_retain(prob, dag, std::move(sol));
    sol = optimize_recompute(prob, dag, std::move(sol));
    sol = optimize_retain(prob, dag, std::move(sol));

    auto vr = sol.validate();
    if (!vr.valid)
        std::cerr << "  WARNING: " << vr.error << "\n";

    // Summary
    double final_cost = sol.total_latency();
    double improve_pct = 100.0 * (partition_cost - final_cost) / partition_cost;
    std::cerr << "  === Summary ===\n";
    std::cerr << "  Partition:  " << partition_cost << "\n";
    std::cerr << "  Build+opt:  " << after_build;
    if (after_build < partition_cost - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << 100.0*(partition_cost - after_build)/partition_cost << "%)";
    std::cerr << "\n";
    std::cerr << "  Retain+rec: " << after_retain_recomp;
    if (after_retain_recomp < after_build - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << 100.0*(after_build - after_retain_recomp)/after_build << "%)";
    std::cerr << "\n";
    std::cerr << "  Sol-FM:     " << after_sol_opt;
    if (after_sol_opt < after_retain_recomp - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << 100.0*(after_retain_recomp - after_sol_opt)/after_retain_recomp << "%)";
    std::cerr << "\n";
    std::cerr << "  Final:      " << final_cost;
    if (final_cost < partition_cost - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << improve_pct << "% total)";
    std::cerr << "\n";

    return sol;
}