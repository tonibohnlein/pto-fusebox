#include "postopt/post_opt.h"
#include "partition/partition.h"
#include <algorithm>
#include <map>

Solution optimize_retain(const Problem& prob, const DAG& dag, Solution sol) {
    auto steps = sol.steps();
    size_t n = steps.size();

    // 1. Calculate intervals: tensor -> {producer_step, last_consumer_step}
    std::map<size_t, std::pair<int, int>> intervals;

    for (size_t i = 0; i < n; i++) {
        const auto& sg = steps[i].subgraph;
        
        // Producer (Sink Tensor)
        size_t sink = sg.sink_tensor();
        if (prob.retainable_tensors.count(sink)) {
            if (intervals.find(sink) == intervals.end()) {
                intervals[sink] = {(int)i, (int)i};
            }
        }
        
        // Consumers (Boundary Inputs)
        for (auto t : sg.boundary_inputs()) {
            if (prob.retainable_tensors.count(t)) {
                if (intervals.find(t) == intervals.end()) {
                    intervals[t] = {0, (int)i}; // Graph input, born at step 0
                } else {
                    intervals[t].second = std::max(intervals[t].second, (int)i);
                }
            }
        }
    }

    // 2. Clear existing retain_these to rebuild optimally from scratch
    for (auto& step : steps) step.retain_these.clear();

    // 3. Sort candidates by size (Largest tensors save the most bandwidth)
    std::vector<size_t> candidates;
    for (auto& kv : intervals) {
        if (kv.second.first < kv.second.second) { // Spans at least across to the next step
            candidates.push_back(kv.first);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
        return prob.tensors[a].width * prob.tensors[a].height >
               prob.tensors[b].width * prob.tensors[b].height;
    });

    // 4. Greedily allocate tensor lifespans into the fast memory
    for (auto t : candidates) {
        int start = intervals[t].first;
        int end = intervals[t].second;
        
        bool fits_everywhere = true;
        
        // Check if it fits in every step's fast memory during its lifespan
        for (int i = start; i < end; i++) {
            auto trial_retain = steps[i].retain_these;
            trial_retain.insert(t);

            std::set<size_t> entering;
            if (i > 0) entering = steps[i-1].retain_these; // Approximate entering set

            auto cost = steps[i].subgraph.best_cost(entering, trial_retain);
            if (!cost.feasible) {
                fits_everywhere = false;
                break;
            }
        }

        // If it fits across the whole gap, commit it!
        if (fits_everywhere) {
            for (int i = start; i < end; i++) {
                steps[i].retain_these.insert(t);
            }
        }
    }

    // 5. Return a new Solution object, which natively re-evaluates exact costs
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