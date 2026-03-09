#include "pipeline/solver.h"
#include "partition/partition.h"
#include "search/parallel_search.h"
#include "postopt/post_opt.h"
#include <iostream>
#include <map>
#include <optional>
#include <set>

// ============================================================================
// Build Solution from a Partition
//
// If no tensors are retainable, any topological order gives the same cost.
// If retainable tensors exist, we greedily order steps to place consecutive
// steps that share retainable boundary tensors adjacent to each other.
// ============================================================================

static Solution build_solution(const Problem& prob, const DAG& dag,
                                const Partition& part) {
    // Collect alive groups and build their Subgraphs
    struct GroupInfo {
        std::vector<size_t> ops;
        std::optional<Subgraph> sg;
        CostResult cost;
    };

    std::vector<GroupInfo> infos;
    for (auto& g : part.groups) {
        if (!g.alive) continue;
        GroupInfo gi;
        gi.ops = {g.ops.begin(), g.ops.end()};
        gi.sg = Subgraph::create(prob, dag, gi.ops);
        if (!gi.sg) continue;
        gi.cost = gi.sg->best_cost();
        infos.push_back(std::move(gi));
    }

    size_t n = infos.size();

    // Group-level DAG based on TENSOR dependencies (correct with recompute).
    std::map<size_t, size_t> tensor_producer_grp;
    for (size_t i = 0; i < n; i++)
        for (auto t : infos[i].sg->boundary_outputs())
            tensor_producer_grp[t] = i;

    std::vector<std::set<size_t>> grp_preds(n), grp_succs(n);
    for (size_t i = 0; i < n; i++) {
        for (auto t : infos[i].sg->boundary_inputs()) {
            auto it = tensor_producer_grp.find(t);
            if (it != tensor_producer_grp.end() && it->second != i) {
                grp_preds[i].insert(it->second);
                grp_succs[it->second].insert(i);
            }
        }
    }

    std::vector<int> in_deg(n, 0);
    for (size_t i = 0; i < n; i++) in_deg[i] = (int)grp_preds[i].size();

    // DFS chain-first topological sort with retain-aware scoring.
    //
    // After scheduling group G, prefer its direct successors (chain-first).
    // Among ready candidates, score by the byte size of retainable tensors
    // shared with G's output — this maximizes the chance that optimize_retain
    // can keep data resident.
    //
    // If no successor is ready, fall back to any ready group scored by:
    //   1. retainable overlap with any recently scheduled group
    //   2. tie-break: prefer groups consuming the largest boundary inputs
    //      (schedule big consumers early to shorten tensor lifetimes)

    std::vector<size_t> order;
    std::vector<bool> scheduled(n, false);

    // Ready set
    std::set<size_t> ready;
    for (size_t i = 0; i < n; i++)
        if (in_deg[i] == 0) ready.insert(i);

    // Score a candidate group based on retain potential with the previously
    // scheduled group. Returns the total byte size of retainable shared tensors.
    auto retain_score = [&](size_t candidate, size_t prev) -> int64_t {
        if (prev >= n) return 0;  // no previous group
        int64_t score = 0;

        // Does candidate consume prev's sink tensor?
        size_t prev_sink = infos[prev].sg->sink_tensor();
        if (infos[candidate].sg->boundary_inputs().count(prev_sink) &&
            prob.retainable_tensors.count(prev_sink))
            score += prob.tensors[prev_sink].width * prob.tensors[prev_sink].height;

        // Shared boundary inputs (both groups load the same tensor)
        for (auto t : infos[candidate].sg->boundary_inputs())
            if (infos[prev].sg->boundary_inputs().count(t) &&
                prob.retainable_tensors.count(t))
                score += prob.tensors[t].width * prob.tensors[t].height;

        return score;
    };

    // Total boundary input size (for tie-breaking)
    auto input_size = [&](size_t gi) -> int64_t {
        int64_t s = 0;
        for (auto t : infos[gi].sg->boundary_inputs())
            s += prob.tensors[t].width * prob.tensors[t].height;
        return s;
    };

    size_t prev = SIZE_MAX;

    while (!ready.empty()) {
        size_t chosen = SIZE_MAX;
        int64_t best_score = -1;
        int64_t best_input = -1;

        // DFS priority: if prev has ready successors, strongly prefer them
        if (prev < n) {
            for (auto s : grp_succs[prev]) {
                if (!scheduled[s] && ready.count(s)) {
                    int64_t sc = retain_score(s, prev);
                    int64_t inp = input_size(s);
                    if (sc > best_score || (sc == best_score && inp > best_input)) {
                        chosen = s;
                        best_score = sc;
                        best_input = inp;
                    }
                }
            }
        }

        // If no successor is ready, pick best from all ready groups
        if (chosen == SIZE_MAX) {
            for (auto gi : ready) {
                int64_t sc = retain_score(gi, prev);
                int64_t inp = input_size(gi);
                if (sc > best_score || (sc == best_score && inp > best_input)) {
                    chosen = gi;
                    best_score = sc;
                    best_input = inp;
                }
            }
        }

        ready.erase(chosen);
        order.push_back(chosen);
        scheduled[chosen] = true;
        prev = chosen;

        // Update in-degrees and ready set
        for (auto v : grp_succs[chosen]) {
            if (--in_deg[v] == 0)
                ready.insert(v);
        }
    }

    // Build ScheduleSteps
    std::vector<ScheduleStep> steps;
    for (auto idx : order)
        steps.push_back({std::move(*infos[idx].sg), infos[idx].cost.config, {}});

    return Solution(prob, dag, std::move(steps));
}

// ============================================================================
// Full pipeline
// ============================================================================

Solution solve(const Problem& prob) {
    DAG dag = DAG::build(prob);

    std::cerr << "Phase 1: Parallel search (init + greedy + FM)...\n";
    ParallelConfig pcfg;
    adapt_fm_budget(pcfg, prob.num_ops());
    auto best_part = parallel_search(prob, dag, pcfg);

    std::cerr << "Phase 2: Build solution...\n";
    auto sol = build_solution(prob, dag, best_part);
    std::cerr << "  " << sol.num_steps() << " steps, lat=" << sol.total_latency() << "\n";

    for (int pass = 1; pass <= 3; pass++) {
        std::cerr << "Phase 3." << pass << ": Retain...\n";
        sol = optimize_retain(prob, dag, std::move(sol));
        std::cerr << "  lat=" << sol.total_latency() << "\n";

        double before = sol.total_latency();
        std::cerr << "Phase 3." << pass << ": Recompute...\n";
        sol = optimize_recompute(prob, dag, std::move(sol));
        std::cerr << "  lat=" << sol.total_latency() << "\n";

        if (sol.total_latency() >= before - 0.01) break;
    }

    auto vr = sol.validate();
    if (!vr.valid)
        std::cerr << "  WARNING: " << vr.error << "\n";
    std::cerr << "  Final: " << sol.num_steps() << " steps, lat="
              << sol.total_latency() << "\n";

    return sol;
}