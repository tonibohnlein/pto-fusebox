#include "solution/solution.h"
#include <sstream>

// ============================================================================
// Construction: evaluate costs with inter-step retain propagation
// ============================================================================

Solution::Solution(const Problem& prob, const DAG& dag, std::vector<ScheduleStep> steps)
    : prob_(&prob), dag_(&dag), steps_(std::move(steps))
{
    size_t n = steps_.size();
    step_costs_.resize(n);
    retained_entering_.resize(n);

    std::set<size_t> currently_retained;

    for (size_t i = 0; i < n; i++) {
        retained_entering_[i] = currently_retained;

        step_costs_[i] = steps_[i].subgraph.compute_cost(
            steps_[i].config,
            currently_retained,
            steps_[i].retain_these);

        total_latency_ += step_costs_[i].latency;
        currently_retained = steps_[i].retain_these;
    }
}

// ============================================================================
// Validation
// ============================================================================

Solution::ValidationResult Solution::validate() const {
    ValidationResult vr;
    auto fail = [&](const std::string& msg) {
        vr.valid = false;
        vr.error = msg;
    };

    // 1. Non-empty
    if (steps_.empty()) {
        fail("Solution has no steps");
        return vr;
    }

    // 2. Coverage: every op appears in at least one subgraph
    std::set<size_t> covered_ops;
    for (auto& step : steps_)
        for (auto op : step.subgraph.ops())
            covered_ops.insert(op);

    for (size_t i = 0; i < prob_->num_ops(); i++) {
        if (!covered_ops.count(i)) {
            fail("Op " + std::to_string(i) + " not covered by any subgraph");
            return vr;
        }
    }

    // 3. Per-subgraph: check tile feasibility with retained tensors.
    //    (Connectivity and boundary output validity are enforced by Subgraph::create.)
    for (size_t i = 0; i < steps_.size(); i++) {
        const auto& step = steps_[i];
        if (!step.subgraph.is_feasible(step.config,
                                       retained_entering_[i],
                                       step.retain_these)) {
            fail("Step " + std::to_string(i) + ": tile config infeasible "
                 "(working set exceeds fast memory)");
            return vr;
        }
    }

    // 4. Topological order: for each step, all boundary input tensors must
    //    have been produced by an earlier step or be graph inputs.
    std::set<size_t> available_tensors;
    for (auto t : dag_->graph_inputs)
        available_tensors.insert(t);

    for (size_t i = 0; i < steps_.size(); i++) {
        const auto& sg = steps_[i].subgraph;

        for (auto t : sg.boundary_inputs()) {
            if (!available_tensors.count(t)) {
                fail("Step " + std::to_string(i) + ": boundary input tensor "
                     + std::to_string(t) + " not yet produced");
                return vr;
            }
        }

        // All output tensors of all ops in this step become available
        for (auto op_idx : sg.ops())
            for (auto t : prob_->ops[op_idx].outputs)
                available_tensors.insert(t);
    }

    // 5. Retain validity: retained tensors must be boundary tensors of the step
    for (size_t i = 0; i < steps_.size(); i++) {
        const auto& step = steps_[i];
        const auto& sg = step.subgraph;
        for (auto t : step.retain_these) {
            bool is_boundary_in = sg.boundary_inputs().count(t);
            bool is_boundary_out = sg.boundary_outputs().count(t);
            if (!is_boundary_in && !is_boundary_out) {
                fail("Step " + std::to_string(i) + ": retained tensor "
                     + std::to_string(t) + " is not a boundary tensor");
                return vr;
            }
        }
    }

    // 6. Check that computed costs are feasible
    for (size_t i = 0; i < steps_.size(); i++) {
        if (!step_costs_[i].feasible) {
            fail("Step " + std::to_string(i) + ": cost evaluation returned infeasible");
            return vr;
        }
    }

    return vr;
}