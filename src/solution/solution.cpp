#include "solution/solution.h"
#include "partition/partition.h"
#include "core/cost_cache.h"
#include <algorithm>
#include <sstream>
#include <iostream>

// ============================================================================
// One-time feasibility check: which tensors can physically be retained?
// ============================================================================
std::set<size_t> compute_feasibly_retainable(const Problem& prob, const DAG& dag) {
    std::set<size_t> result;
    for (auto t : prob.retainable_tensors) {
        if (prob.tensors[t].size() > prob.fast_memory_capacity)
            continue;
        bool ok = true;
        // Producer singleton must be feasible while retaining t
        int prod = dag.tensor_producer[t];
        if (prod >= 0) {
            auto sg = Subgraph::create(prob, dag, {(size_t)prod});
            if (!sg) { ok = false; }
            else if (!sg->best_cost({}, {t}).feasible) { ok = false; }
        }
        if (!ok) continue;
        // Each consumer singleton must be feasible with t entering
        for (size_t op = 0; op < prob.num_ops() && ok; op++) {
            bool consumes = false;
            for (auto inp : prob.ops[op].inputs)
                if (inp == t) { consumes = true; break; }
            if (!consumes) continue;
            auto sg = Subgraph::create(prob, dag, {op});
            if (!sg) { ok = false; break; }
            if (!sg->best_cost({t}, {}).feasible) { ok = false; break; }
        }
        if (ok) result.insert(t);
    }
    return result;
}

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

        // Only boundary OUTPUTS can be retained (organizer ruling).
        // Overlap with currently_retained is possible (recomputation case)
        // and handled correctly by working_set.
        std::set<size_t> valid_retain;
        for (auto t : steps_[i].retain_these)
            if (steps_[i].subgraph.boundary_outputs().count(t))
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
        const OrderingResult& res,
        CostCache* cache) {

    // Helper: route best_cost through cache when available
    auto best_cost_cached = [&](const Subgraph& sg,
                                const std::set<size_t>& entering,
                                const std::set<size_t>& retain) -> CostResult {
        if (cache) return cache->evaluate_with_context(sg, entering, retain);
        return sg.best_cost(entering, retain);
    };

    std::vector<ScheduleStep> steps;
    steps.reserve(res.order.size());
    std::set<size_t> entering;

    for (size_t i = 0; i < res.order.size(); i++) {
        size_t gi = res.order[i];
        const Partition::Group& g = part.groups[gi];
        if (!g.sg) continue;
        const Subgraph& sg = *g.sg;

        // Build retain_these: only boundary OUTPUTS of this step that the next
        // step needs as input.
        std::set<size_t> retain_these;
        if (i + 1 < res.order.size()) {
            size_t next_gi = res.order[i + 1];
            if (part.groups[next_gi].sg) {
                const auto& next_inputs = part.groups[next_gi].sg->boundary_inputs();
                if (i < res.retain_per_step.size()) {
                    for (auto t : res.retain_per_step[i])
                        if (next_inputs.count(t) && sg.boundary_outputs().count(t))
                            retain_these.insert(t);
                }
            }
        }

        // Standalone baseline: no entering, no retain (= partition cost)
        CostResult baseline;
        baseline.feasible = (g.cost < 1e17);
        baseline.latency  = g.cost;
        baseline.config   = g.best_cfg;

        // Attempt 1: full retention context (entering + retain)
        auto cost = best_cost_cached(sg, entering, retain_these);

        // Attempt 2: keep entering, drop retain — ONLY if infeasible.
        // A local latency increase from retention is often worth it when it
        // saves the next step a massive slow-memory load.
        if (!cost.feasible) {
            auto cost2 = best_cost_cached(sg, entering, {});
            if (cost2.feasible) {
                cost = cost2;
                retain_these.clear();
            }
        }

        // Attempt 3: drop entering too — ONLY if still infeasible.
        if (!cost.feasible && baseline.feasible) {
            cost = baseline;
            retain_these.clear();
            entering.clear();
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
                                   const Partition& part, int max_beam_width,
                                   CostCache* cache) {
    // Caller must have called part.finalize() already.
    // No deep copy, no redundant finalize.

    auto dfs_res  = dfs_ordering(part);

    // Scale beam width inversely with partition size.
    // 35+ groups with beam=10 → ~15K best_cost calls → 10+ seconds.
    // Cap to keep Phase 2 fast.
    int n_alive = (int)part.num_alive();
    int bw = max_beam_width;
    if (n_alive > 25) bw = std::min(bw, 3);
    else if (n_alive > 15) bw = std::min(bw, 5);
    bw = std::max(2, bw);

    auto beam_res = beam_search_ordering(part, bw);

    auto dfs_steps  = steps_from_ordering(prob, dag, part, dfs_res, cache);
    auto beam_steps = steps_from_ordering(prob, dag, part, beam_res, cache);

    // No-retention baseline: DFS order, each step uses standalone best_cost.
    // Guarantees solution cost ≤ partition cost.
    std::vector<ScheduleStep> bare_steps;
    bare_steps.reserve(dfs_res.order.size());
    for (size_t gi : dfs_res.order) {
        const auto& g = part.groups[gi];
        if (!g.sg) continue;
        bare_steps.push_back({Subgraph(*g.sg), g.best_cfg, {}});
    }

    Solution dfs_sol (prob, dag, std::move(dfs_steps));
    Solution beam_sol(prob, dag, std::move(beam_steps));
    Solution bare_sol(prob, dag, std::move(bare_steps));

    // Pick the best valid solution
    Solution* best = nullptr;
    for (auto* s : {&bare_sol, &dfs_sol, &beam_sol}) {
        if (s->validate().valid &&
            (!best || s->total_latency() < best->total_latency() - 0.01))
            best = s;
    }
    if (!best) best = &bare_sol;  // fallback (all invalid)

#ifndef NDEBUG
    // Diagnostic: if best is invalid, dump group DAG info
    if (!best->validate().valid) {
        auto vr = best->validate();
        std::cerr << "  DIAG: " << vr.error << "\n";
        std::cerr << "  DIAG: " << part.num_alive() << " alive groups\n";
        for (size_t gi = 0; gi < part.groups.size(); gi++) {
            if (!part.groups[gi].alive) continue;
            if (part.group_in_deg[gi] != 0) continue;
            if (!part.groups[gi].sg) continue;
            for (auto t : part.groups[gi].sg->boundary_inputs()) {
                int prod = dag.tensor_producer[t];
                if (prod < 0) continue;
                std::cerr << "  DIAG: G" << gi << " (in_deg=0) needs T" << t
                          << " from op" << prod << "\n";
            }
        }
        std::cerr << "  DIAG dfs_order: ";
        for (auto gi : dfs_res.order) std::cerr << gi << " ";
        std::cerr << "\n";
    }
#endif

    return std::move(*best);
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

    // Retain validity: only boundary OUTPUTS can be retained
    for (size_t i = 0; i < steps_.size(); i++) {
        const auto& sg = steps_[i].subgraph;
        for (auto t : steps_[i].retain_these)
            if (!sg.boundary_outputs().count(t)) {
                fail("Step " + std::to_string(i) + ": retained T"
                     + std::to_string(t) + " is not a boundary output");
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

    // Mixed-consumer check (matches reference evaluator's checkEphemeralization):
    // If tensor T is produced AND consumed inside step si (ephemeral), every
    // external consumer must have T's producer recomputed in its own step.
    // Without recomputation, the external consumer can't access T (it's ephemeral).
    for (size_t si = 0; si < steps_.size(); si++) {
        const auto& ops = steps_[si].subgraph.ops();
        std::set<size_t> op_set(ops.begin(), ops.end());
        for (auto op : ops) {
            for (auto t : prob_->ops[op].outputs) {
                // Is T consumed by any op inside this step?
                bool consumed_internal = false;
                for (auto cop : dag_->tensor_consumers[t])
                    if (cop != op && op_set.count(cop)) { consumed_internal = true; break; }
                if (!consumed_internal) continue;

                // T is ephemeral in step si. If T is a boundary output of
                // some OTHER step, external consumers can read it from slow
                // memory — no recomputation needed.
                bool avail_elsewhere = false;
                for (size_t sj = 0; sj < steps_.size(); sj++) {
                    if (sj == si) continue;
                    if (steps_[sj].subgraph.boundary_outputs().count(t))
                        { avail_elsewhere = true; break; }
                }
                if (avail_elsewhere) continue;

                // T not available elsewhere — each external consumer must
                // have the producer recomputed in its own step.
                for (auto cop : dag_->tensor_consumers[t]) {
                    if (op_set.count(cop)) continue;  // internal → OK

                    bool covered = false;
                    for (size_t sj = 0; sj < steps_.size(); sj++) {
                        if (sj == si) continue;
                        bool has_cop = false, has_prod = false;
                        for (auto o : steps_[sj].subgraph.ops()) {
                            if (o == cop) has_cop = true;
                            if (o == op)  has_prod = true;
                        }
                        if (has_cop && has_prod) { covered = true; break; }
                    }
                    if (!covered) {
                        fail("Step " + std::to_string(si) + ": T" + std::to_string(t)
                             + " is ephemeral but external consumer op" + std::to_string(cop)
                             + " has no recomputation of producer op" + std::to_string(op));
                        return vr;
                    }
                }
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
            // New ephemeral rule: T is ephemeral in proposed_ops if ANY
            // consumer is internal (produced + consumed = ephemeral).
            bool any_consumer_internal = false;
            for (auto cop : dag.tensor_consumers[t])
                if (proposed_ops.count(cop)) { any_consumer_internal = true; break; }
            if (!any_consumer_internal) continue;  // pure boundary output → safe

            // T is ephemeral → external consumers need it from elsewhere
            for (auto cop : dag.tensor_consumers[t]) {
                if (proposed_ops.count(cop)) continue;  // internal → served

                int prod_op = dag.tensor_producer[t];
                if (prod_op < 0) continue;

                // Is T available as boundary output from some non-excluded step?
                // Under new rule: boundary_outputs() excludes internally-consumed tensors
                bool available = false;
                for (size_t si = 0; si < steps.size(); si++) {
                    if (si == exclude_step || si == exclude_step2) continue;
                    if (steps[si].subgraph.boundary_outputs().count(t))
                        { available = true; break; }
                }
                if (available) continue;

                // Check if cop's step recomputes the producer
                bool cop_served = false;
                for (size_t si = 0; si < steps.size(); si++) {
                    if (si == exclude_step || si == exclude_step2) continue;
                    bool in_step = false, has_prod = false;
                    for (auto o : steps[si].subgraph.ops()) {
                        if (o == cop)              in_step  = true;
                        if (o == (size_t)prod_op)  has_prod = true;
                    }
                    if (in_step && has_prod) { cop_served = true; break; }
                }
                if (!cop_served) return true;
            }
        }
    }
    return false;
}