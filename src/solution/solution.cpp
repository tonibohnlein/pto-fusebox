#include "solution/solution.h"
#include "partition/partition.h"
#include <algorithm>
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

        // retain_these may overlap with currently_retained (pass-through).
        // working_set handles this correctly by not double-counting.
        // Only validate that retained tensors are boundary tensors of this subgraph.
        std::set<size_t> valid_retain;
        for (auto t : steps_[i].retain_these)
            if (steps_[i].subgraph.boundary_inputs().count(t) ||
                steps_[i].subgraph.boundary_outputs().count(t))
                valid_retain.insert(t);
        steps_[i].retain_these = valid_retain;

        step_costs_[i] = steps_[i].subgraph.compute_cost(
            steps_[i].config,
            currently_retained,
            steps_[i].retain_these);

        total_latency_   += step_costs_[i].latency;
        currently_retained = steps_[i].retain_these;
    }
}

// ============================================================================
// steps_from_ordering
//
// Converts an OrderingResult into concrete ScheduleSteps.
// For each group in the ordering:
//   1. Build the intended retain_these (filter: only tensors needed by the
//      immediately following step are actually retained).
//   2. Call best_cost with the current entering set and retain_these.
//   3. If infeasible, progressively fall back:
//        a. Clear retain_these, try again with entering.
//        b. Clear entering too, use the cached no-retention baseline.
// ============================================================================

std::vector<ScheduleStep> Solution::steps_from_ordering(
        const Problem& /*prob*/, const DAG& /*dag*/,
        const Partition& part,
        const OrderingResult& res) {

    std::vector<ScheduleStep> steps;
    steps.reserve(res.order.size());
    std::set<size_t> entering;

    for (size_t i = 0; i < res.order.size(); i++) {
        size_t gi = res.order[i];
        const Partition::Group& g = part.groups[gi];
        if (!g.sg) continue;
        const Subgraph& sg = *g.sg;

        // Build retain_these: keep tensors the next step will actually read.
        // Two sources:
        //   a) tensors from retain_per_step that the next step needs (new retains)
        //   b) tensors from entering that the next step also needs (pass-through)
        // Both can overlap with entering; working_set handles that correctly.
        std::set<size_t> retain_these;
        if (i + 1 < res.order.size()) {
            size_t next_gi = res.order[i + 1];
            if (part.groups[next_gi].sg) {
                const auto& next_inputs = part.groups[next_gi].sg->boundary_inputs();
                // (a) New retains from ordering hint
                if (i < res.retain_per_step.size()) {
                    for (auto t : res.retain_per_step[i])
                        if (next_inputs.count(t)) retain_these.insert(t);
                }
                // (b) Pass-through: entering tensors needed by next step
                for (auto t : entering)
                    if (next_inputs.count(t) &&
                        (sg.boundary_inputs().count(t) || sg.boundary_outputs().count(t)))
                        retain_these.insert(t);
            }
        }

        // Attempt 1: full retention context
        auto cost = sg.best_cost(entering, retain_these);

        // Attempt 2: drop retain_these, keep entering
        if (!cost.feasible) {
            retain_these.clear();
            cost = sg.best_cost(entering, {});
        }

        // Attempt 3: clear everything, use cached no-retain baseline
        if (!cost.feasible) {
            entering.clear();
            retain_these.clear();
            // Use the stored baseline to avoid a full best_cost() enumeration
            cost.feasible = (g.cost < 1e17);
            cost.latency  = g.cost;
            cost.config   = g.best_cfg;
        }

        steps.push_back({Subgraph(sg), cost.config, retain_these});
        entering = retain_these;
    }

    return steps;
}

// ============================================================================
// from_partition
//
// Runs DFS and beam search, builds a Solution from each, returns the better.
// ============================================================================

Solution Solution::from_partition(const Problem& prob, const DAG& dag,
                                   Partition part) {
    // Re-populate Group::sg / best_cfg and rebuild the group-level DAG.
    // Phase 1 search mutates ops/costs but never touches these fields.
    part.finalize();

    auto dfs_res  = dfs_ordering(part);
    int  bw       = std::min(20, std::max(5, (int)part.num_alive()));
    auto beam_res = beam_search_ordering(part, bw);

    auto dfs_steps  = steps_from_ordering(prob, dag, part, dfs_res);
    auto beam_steps = steps_from_ordering(prob, dag, part, beam_res);

    Solution dfs_sol (prob, dag, std::move(dfs_steps));
    Solution beam_sol(prob, dag, std::move(beam_steps));

    return (beam_sol.total_latency() < dfs_sol.total_latency() - 0.01)
           ? std::move(beam_sol)
           : std::move(dfs_sol);
}

// ============================================================================
// Validation
// ============================================================================

Solution::ValidationResult Solution::validate() const {
    ValidationResult vr;
    auto fail = [&](const std::string& msg) { vr.valid = false; vr.error = msg; };

    if (steps_.empty()) { fail("Solution has no steps"); return vr; }

    // Coverage
    std::set<size_t> covered;
    for (auto& step : steps_)
        for (auto op : step.subgraph.ops()) covered.insert(op);
    for (size_t i = 0; i < prob_->num_ops(); i++)
        if (!covered.count(i)) { fail("Op " + std::to_string(i) + " not covered"); return vr; }

    // Per-subgraph feasibility
    for (size_t i = 0; i < steps_.size(); i++)
        if (!steps_[i].subgraph.is_feasible(steps_[i].config,
                                             retained_entering_[i],
                                             steps_[i].retain_these)) {
            fail("Step " + std::to_string(i) + ": working set exceeds fast memory");
            return vr;
        }

    // Topological order
    std::set<size_t> available(dag_->graph_inputs.begin(), dag_->graph_inputs.end());
    for (size_t i = 0; i < steps_.size(); i++) {
        for (auto t : steps_[i].subgraph.boundary_inputs())
            if (!available.count(t)) {
                fail("Step " + std::to_string(i) + ": boundary input T"
                     + std::to_string(t) + " not yet produced");
                return vr;
            }
        for (auto op : steps_[i].subgraph.ops())
            for (auto t : prob_->ops[op].outputs) available.insert(t);
    }

    // Retain validity
    for (size_t i = 0; i < steps_.size(); i++) {
        const auto& sg = steps_[i].subgraph;
        for (auto t : steps_[i].retain_these)
            if (!sg.boundary_inputs().count(t) && !sg.boundary_outputs().count(t)) {
                fail("Step " + std::to_string(i) + ": retained T"
                     + std::to_string(t) + " is not a boundary tensor");
                return vr;
            }
    }

    // Cost feasibility
    for (size_t i = 0; i < steps_.size(); i++)
        if (!step_costs_[i].feasible) {
            fail("Step " + std::to_string(i) + ": cost evaluation infeasible");
            return vr;
        }

    // Ephemeral gap check
    for (size_t si = 0; si < steps_.size(); si++) {
        for (auto t : steps_[si].subgraph.boundary_inputs()) {
            if (dag_->tensor_producer[t] < 0) continue;
            bool found = false;
            for (size_t sj = 0; sj < steps_.size(); sj++)
                if (steps_[sj].subgraph.boundary_outputs().count(t)) { found = true; break; }
            if (!found) {
                fail("Step " + std::to_string(si) + ": T" + std::to_string(t)
                     + " not available from slow memory (ephemeral gap)");
                return vr;
            }
        }
    }

    return vr;
}

// ============================================================================
// Ephemeral gap check (solution-level)
// ============================================================================

bool Solution::creates_ephemeral_gap(const Problem& prob, const DAG& dag,
                                      const std::set<size_t>& proposed_ops,
                                      const std::vector<ScheduleStep>& steps,
                                      size_t exclude_step,
                                      size_t exclude_step2) {
    for (auto op : proposed_ops) {
        for (auto t : prob.ops[op].outputs) {
            // New ephemeral rule: T is ephemeral only if ALL DAG consumers
            // are inside proposed_ops. If any consumer is external to the
            // proposed group, T becomes a boundary output (materialized).
            bool all_consumers_internal = true;
            bool any_consumer_internal = false;
            for (auto cop : dag.tensor_consumers[t]) {
                if (proposed_ops.count(cop))
                    any_consumer_internal = true;
                else
                    all_consumers_internal = false;
            }
            if (!any_consumer_internal) continue;  // pure boundary output
            if (!all_consumers_internal) continue;  // external consumer → boundary output

            // T would be purely ephemeral. Is it available from some other step?
            int prod_op = dag.tensor_producer[t];
            if (prod_op < 0) continue;

            bool available = false;
            for (size_t si = 0; si < steps.size(); si++) {
                if (si == exclude_step || si == exclude_step2) continue;
                if (steps[si].subgraph.boundary_outputs().count(t)) { available = true; break; }
            }
            if (available) continue;

            // T is ephemeral everywhere. Any consumer in a non-excluded step
            // that doesn't recompute the producer is stranded.
            for (auto cop : dag.tensor_consumers[t]) {
                for (size_t si = 0; si < steps.size(); si++) {
                    if (si == exclude_step || si == exclude_step2) continue;
                    bool in_step = false, has_prod = false;
                    for (auto o : steps[si].subgraph.ops()) {
                        if (o == cop)              in_step  = true;
                        if (o == (size_t)prod_op)  has_prod = true;
                    }
                    if (in_step && !has_prod) return true;
                }
            }
        }
    }
    return false;
}