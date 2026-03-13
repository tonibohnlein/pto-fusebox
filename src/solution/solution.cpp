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

    // 7. Check for ephemeral gaps: no step should need a tensor that's
    //    ephemeral in its producing step without a recomputation path
    for (size_t si = 0; si < steps_.size(); si++) {
        for (auto t : steps_[si].subgraph.boundary_inputs()) {
            // Is T available from slow memory?
            if (dag_->tensor_producer[t] < 0) continue;  // graph input, always available
            bool found = false;
            for (size_t sj = 0; sj < steps_.size(); sj++) {
                if (steps_[sj].subgraph.boundary_outputs().count(t)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                fail("Step " + std::to_string(si) + ": boundary input T" +
                     std::to_string(t) + " is not available from slow memory "
                     "(ephemeral in producing step, no recomputation)");
                return vr;
            }
        }
    }

    return vr;
}

// ============================================================================
// Ephemeral gap check for solution-level moves
// ============================================================================

bool Solution::creates_ephemeral_gap(const Problem& prob, const DAG& dag,
                                      const std::set<size_t>& proposed_ops,
                                      const std::vector<ScheduleStep>& steps,
                                      size_t exclude_step,
                                      size_t exclude_step2) {
    // Find tensors that would be ephemeral in proposed_ops
    for (auto op : proposed_ops) {
        for (auto t : prob.ops[op].outputs) {
            // Is T consumed internally?
            bool consumed_internally = false;
            for (auto cop : dag.tensor_consumers[t])
                if (proposed_ops.count(cop)) { consumed_internally = true; break; }
            if (!consumed_internally) continue;

            // T would be ephemeral. Check external consumers.
            int prod_op = dag.tensor_producer[t];
            if (prod_op < 0) continue;

            // Is T available as boundary output from another step?
            bool available = false;
            for (size_t si = 0; si < steps.size(); si++) {
                if (si == exclude_step || si == exclude_step2) continue;
                if (steps[si].subgraph.boundary_outputs().count(t)) {
                    available = true;
                    break;
                }
            }
            if (available) continue;

            // Check each external consumer's step for recomputation
            for (auto cop : dag.tensor_consumers[t]) {
                if (proposed_ops.count(cop)) continue;
                for (size_t si = 0; si < steps.size(); si++) {
                    if (si == exclude_step || si == exclude_step2) continue;
                    bool in_step = false;
                    for (auto o : steps[si].subgraph.ops())
                        if (o == cop) { in_step = true; break; }
                    if (!in_step) continue;
                    bool has_prod = false;
                    for (auto o : steps[si].subgraph.ops())
                        if (o == (size_t)prod_op) { has_prod = true; break; }
                    if (!has_prod) return true;  // GAP
                }
            }
        }
    }
    return false;
}