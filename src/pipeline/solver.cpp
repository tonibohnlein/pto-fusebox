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
        std::set<size_t> all_tensors;  // boundary inputs + boundary outputs
    };

    std::vector<GroupInfo> infos;
    for (auto& g : part.groups) {
        if (!g.alive) continue;
        GroupInfo gi;
        gi.ops = {g.ops.begin(), g.ops.end()};
        gi.sg = Subgraph::create(prob, dag, gi.ops);
        if (!gi.sg) continue;
        gi.cost = gi.sg->best_cost();
        gi.all_tensors = gi.sg->boundary_inputs();
        gi.all_tensors.insert(gi.sg->sink_tensor());
        infos.push_back(std::move(gi));
    }

    size_t n = infos.size();

    // Group-level DAG based on TENSOR dependencies (correct with recompute).
    // Group A depends on Group B if A has a boundary input that is a boundary
    // output of B. We must NOT use op-level predecessors because recomputed ops
    // in multiple groups would create false edges/cycles.
    std::map<size_t, size_t> tensor_producer_grp;  // tensor -> group index that outputs it
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

    // Greedy topological order: at each step pick the ready group sharing the
    // most retainable tensors with the previously scheduled group.
    std::vector<size_t> order;
    std::vector<bool> scheduled(n, false);

    // Seed ready set
    std::vector<size_t> ready;
    for (size_t i = 0; i < n; i++)
        if (in_deg[i] == 0) ready.push_back(i);

    size_t prev = SIZE_MAX;
    while (!ready.empty()) {
        size_t best_idx = 0;
        int best_shared = -1;

        if (prev != SIZE_MAX && !prob.retainable_tensors.empty()) {
            // Score each ready group by shared retainable tensors with prev
            for (size_t ri = 0; ri < ready.size(); ri++) {
                size_t gi = ready[ri];
                int shared = 0;
                for (auto t : infos[gi].all_tensors)
                    if (prob.retainable_tensors.count(t) &&
                        infos[prev].all_tensors.count(t))
                        shared++;
                if (shared > best_shared) {
                    best_shared = shared;
                    best_idx = ri;
                }
            }
        }

        size_t chosen = ready[best_idx];
        ready.erase(ready.begin() + best_idx);
        order.push_back(chosen);
        scheduled[chosen] = true;
        prev = chosen;

        for (auto v : grp_succs[chosen]) {
            if (--in_deg[v] == 0)
                ready.push_back(v);
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
    ParallelConfig pcfg;  // auto: threads=hw_concurrency, tasks=~1 per thread
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