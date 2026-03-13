#include "postopt/post_opt.h"
#include "partition/partition.h"
#include <algorithm>
#include <map>

Solution optimize_retain(const Problem& prob, const DAG& dag, Solution sol) {
    auto steps = sol.steps();
    bool global_improved = true;

    while (global_improved) {
        global_improved = false;

        // Forward sweep: track entering state incrementally so changes at
        // pair (i, i+1) cascade correctly into pairs (i+1, i+2), etc.
        // This avoids breaking out and rebuilding the Solution on every change.
        std::set<size_t> entering_state;

        for (size_t i = 0; i + 1 < steps.size(); i++) {
            // Compute step i and i+1 costs with the current (possibly updated)
            // entering state — accurate even after prior cascade changes.
            auto actual_i = steps[i].subgraph.compute_cost(
                steps[i].config, entering_state, steps[i].retain_these);
            std::set<size_t> entering_j = steps[i].retain_these;
            auto actual_j = steps[i+1].subgraph.compute_cost(
                steps[i+1].config, entering_j, steps[i+1].retain_these);
            double old_pair = actual_i.latency + actual_j.latency;

            // Gather candidates: boundary tensors of step i that step i+1 needs
            std::vector<size_t> candidates;
            for (auto t : steps[i].subgraph.boundary_inputs()) {
                if (steps[i+1].subgraph.boundary_inputs().count(t))
                    candidates.push_back(t);
            }
            for (auto t : steps[i].subgraph.boundary_outputs()) {
                if (steps[i+1].subgraph.boundary_inputs().count(t))
                    candidates.push_back(t);
            }

            // Sort by size descending (largest bandwidth savings first)
            std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
                return prob.tensors[a].size() > prob.tensors[b].size();
            });

            for (auto t : candidates) {
                if (steps[i].retain_these.count(t)) continue;
                if (!prob.retainable_tensors.count(t)) continue;

                auto trial_retain_i = steps[i].retain_these;
                trial_retain_i.insert(t);

                // Re-evaluate step i with trial retain
                auto cost_i = steps[i].subgraph.best_cost(entering_state, trial_retain_i);
                if (!cost_i.feasible) continue;

                // Re-evaluate step i+1 with new entering state
                auto cost_j = steps[i+1].subgraph.best_cost(
                    trial_retain_i, steps[i+1].retain_these);
                if (!cost_j.feasible) continue;

                double new_pair = cost_i.latency + cost_j.latency;
                if (new_pair < old_pair - 0.01) {
                    steps[i].retain_these = trial_retain_i;
                    steps[i].config = cost_i.config;
                    steps[i+1].config = cost_j.config;
                    global_improved = true;
                    // Don't break — cascade forward to check (i+1, i+2) next
                    break; // break candidate loop, continue step loop
                }
            }

            // Advance entering state for the next pair
            entering_state = steps[i].retain_these;
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

                // Check: would this create an ephemeral gap?
                std::set<size_t> expanded_set(expanded_ops.begin(), expanded_ops.end());
                if (Solution::creates_ephemeral_gap(prob, dag, expanded_set, steps, i))
                    continue;

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