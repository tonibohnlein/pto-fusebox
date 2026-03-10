#include "postopt/post_opt.h"
#include "partition/partition.h"
#include <algorithm>
#include <map>

Solution optimize_retain(const Problem& prob, const DAG& dag, Solution sol) {
    auto steps = sol.steps();
    bool improved = true;
    while (improved) {
        improved = false;
        Solution current(prob, dag, steps);

        for (size_t i = 0; i + 1 < current.num_steps(); i++) {
            const auto& si = current.step(i);
            const auto& sj = current.step(i + 1);

            std::vector<size_t> candidates;
            for (auto t : si.subgraph.boundary_inputs())
                if (sj.subgraph.boundary_inputs().count(t)) candidates.push_back(t);
            if (sj.subgraph.boundary_inputs().count(si.subgraph.sink_tensor()))
                candidates.push_back(si.subgraph.sink_tensor());

            for (auto t : candidates) {
                if (si.retain_these.count(t)) continue;

                // Only tensors whose full size fits in fast memory can be retained
                if (!prob.retainable_tensors.count(t)) continue;

                // Check total retained size fits
                int64_t total_retained = prob.tensors[t].width * prob.tensors[t].height;
                for (auto rt : si.retain_these)
                    total_retained += prob.tensors[rt].width * prob.tensors[rt].height;
                if (total_retained > prob.fast_memory_capacity) continue;

                auto new_retain_i = si.retain_these;
                new_retain_i.insert(t);
                auto new_retained_j = current.retained_entering(i + 1);
                new_retained_j.insert(t);

                auto ci = si.subgraph.best_cost(current.retained_entering(i), new_retain_i);
                if (!ci.feasible) continue;
                auto cj = sj.subgraph.best_cost(new_retained_j, sj.retain_these);
                if (!cj.feasible) continue;

                double old_cost = current.step_latency(i) + current.step_latency(i + 1);
                double new_cost = ci.latency + cj.latency;

                if (new_cost < old_cost - 0.01) {
                    steps[i].retain_these = new_retain_i;
                    steps[i].config = ci.config;
                    steps[i + 1].config = cj.config;
                    improved = true;
                    break;
                }
            }
            if (improved) break;
        }
    }
    return Solution(prob, dag, std::move(steps));
}

Solution optimize_recompute(const Problem& prob, const DAG& dag, Solution sol) {
    auto steps = sol.steps();
    bool improved = true;
    while (improved) {
        improved = false;
        Solution current(prob, dag, steps);

        for (size_t i = 0; i < current.num_steps(); i++) {
            const auto& step = current.step(i);
            std::set<size_t> current_ops(step.subgraph.ops().begin(),
                                         step.subgraph.ops().end());

            for (auto t : step.subgraph.boundary_inputs()) {
                int prod = dag.tensor_producer[t];
                if (prod < 0) continue;
                if (current_ops.count((size_t)prod)) continue;

                auto expanded_ops = step.subgraph.ops();
                expanded_ops.push_back((size_t)prod);

                auto esg = Subgraph::create(prob, dag, expanded_ops);
                if (!esg) continue;

                auto ec = esg->best_cost(current.retained_entering(i), step.retain_these);
                if (!ec.feasible) continue;

                if (ec.latency < current.step_latency(i) - 0.01) {
                    steps[i] = {std::move(*esg), ec.config, step.retain_these};
                    improved = true;
                    break;
                }
            }
            if (improved) break;
        }
    }
    return Solution(prob, dag, std::move(steps));
}

// ============================================================================
// Solution-level optimizer: run solution FM in parallel with different seeds
// ============================================================================

#include "search/solution_search.h"

Solution optimize_solution(const Problem& prob, const DAG& dag, Solution sol,
                            std::chrono::steady_clock::time_point deadline) {
    SolutionFMConfig cfg;
    cfg.deadline = deadline;
    return solution_fm_search(prob, dag, std::move(sol), cfg);
}