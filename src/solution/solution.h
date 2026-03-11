#pragma once

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include <set>
#include <string>
#include <vector>

// ============================================================================
// ScheduleStep: a subgraph with its chosen tile configuration and retain set
// ============================================================================

struct ScheduleStep {
    Subgraph subgraph;
    TileConfig config;
    std::set<size_t> retain_these;  // tensors to keep in fast memory after this step
};

// ============================================================================
// Solution: an ordered sequence of ScheduleSteps covering the DAG.
//
// Feasibility criteria:
//   1. Coverage: every op in the DAG appears in at least one subgraph
//   2. Per-subgraph: connected, valid boundary outputs (enforced by Subgraph::create)
//   3. Tile feasibility: working set fits in fast memory (including retained)
//   4. Topological order: if step j depends on step i, then i < j
//   5. Retain validity: retained tensors are boundary tensors of the step
// ============================================================================

class Solution {
public:
    // --- Construction ---

    // Build from an ordered list of steps. Evaluates costs accounting for
    // inter-step retained tensors.
    Solution(const Problem& prob, const DAG& dag, std::vector<ScheduleStep> steps);

    // --- Validation ---

    struct ValidationResult {
        bool valid = true;
        std::string error;         // first error found (empty if valid)
    };

    ValidationResult validate() const;

    // --- Cost ---

    double total_latency() const { return total_latency_; }
    double step_latency(size_t i) const { return step_costs_[i].latency; }
    const CostResult& step_cost(size_t i) const { return step_costs_[i]; }

    // --- Accessors ---

    const Problem& problem() const { return *prob_; }
    const DAG& dag() const { return *dag_; }
    size_t num_steps() const { return steps_.size(); }
    const ScheduleStep& step(size_t i) const { return steps_[i]; }
    const std::vector<ScheduleStep>& steps() const { return steps_; }

    // What's retained entering step i (computed from step i-1's retain set)
    const std::set<size_t>& retained_entering(size_t i) const { return retained_entering_[i]; }

private:
    const Problem* prob_;
    const DAG* dag_;
    std::vector<ScheduleStep> steps_;

    // Computed at construction
    std::vector<CostResult> step_costs_;
    std::vector<std::set<size_t>> retained_entering_;  // per step
    double total_latency_ = 0;
};