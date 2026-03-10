#include "postopt/post_opt.h"
#include "partition/partition.h"
#include <algorithm>
#include <map>

Solution optimize_retain(const Problem& prob, const DAG& dag, Solution sol) {
    auto steps = sol.steps();
    bool improved = true;

    while (improved) {
        improved = false;
        // Build perfectly accurate baseline state
        Solution current(prob, dag, steps);

        for (size_t i = 0; i + 1 < current.num_steps(); i++) {
            const auto& si = current.step(i);
            const auto& sj = current.step(i + 1);

            // 1. Gather candidates: must be boundary_in/out of `i`, and boundary_in of `i+1`
            std::vector<size_t> candidates;
            for (auto t : si.subgraph.boundary_inputs()) {
                if (sj.subgraph.boundary_inputs().count(t)) candidates.push_back(t);
            }
            size_t sink = si.subgraph.sink_tensor();
            if (sj.subgraph.boundary_inputs().count(sink)) {
                candidates.push_back(sink);
            }

            // 2. Sort candidates by size (Largest first = maximum bandwidth saved)
            std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
                return prob.tensors[a].width * prob.tensors[a].height >
                       prob.tensors[b].width * prob.tensors[b].height;
            });

            // 3. Test retaining them safely
            for (auto t : candidates) {
                // Skip if already retained or inherently unretainable
                if (steps[i].retain_these.count(t)) continue;
                if (!prob.retainable_tensors.count(t)) continue;

                auto trial_retain_i = steps[i].retain_these;
                trial_retain_i.insert(t);

                // Re-evaluate Step i with accurate entering state
                std::set<size_t> entering_i = current.retained_entering(i);
                auto cost_i = steps[i].subgraph.best_cost(entering_i, trial_retain_i);
                if (!cost_i.feasible) continue;

                // Re-evaluate Step i+1 (it receives whatever i retained)
                std::set<size_t> entering_j = trial_retain_i; 
                auto trial_retain_j = steps[i+1].retain_these;
                auto cost_j = steps[i+1].subgraph.best_cost(entering_j, trial_retain_j);
                if (!cost_j.feasible) continue;

                // CRITICAL FIX 1: Did we ACTUALLY save time? 
                // (Prevents shrinking tiles to fit a tensor if it costs more compute latency)
                double old_latency = current.step_latency(i) + current.step_latency(i + 1);
                double new_latency = cost_i.latency + cost_j.latency;

                if (new_latency < old_latency - 0.01) {
                    steps[i].retain_these = trial_retain_i;
                    steps[i].config = cost_i.config;
                    steps[i+1].config = cost_j.config;
                    improved = true;
                    break; // Break candidate loop
                }
            }
            
            // CRITICAL FIX 2: If we changed memory state, break step loop to rebuild `current`!
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

                if (!shapes_match(&prob, (size_t)prod, current_ops)) continue;

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