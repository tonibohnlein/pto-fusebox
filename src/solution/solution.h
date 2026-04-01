#pragma once

#include "core/types.h"
#include "core/dag.h"
#include "core/subgraph.h"
#include "solution/ordering.h"
#include <set>
#include <string>
#include <vector>

// Forward declaration — avoids pulling all of partition.h into every translation
// unit that includes solution.h.
struct Partition;

// ============================================================================
// One-time feasibility check: which tensors can physically be retained?
// Depends only on Problem + DAG. Checks that singleton producer/consumer
// subgraphs have feasible tilings with the tensor retained/entering.
// ============================================================================
FlatSet<size_t> compute_feasibly_retainable(const Problem& prob, const DAG& dag);

// ============================================================================
// ScheduleStep: a subgraph with its chosen tile configuration and retain set
// ============================================================================

struct ScheduleStep {
    Subgraph         subgraph;
    TileConfig       config;
    FlatSet<size_t> retain_these;
};

// ============================================================================
// Solution: an ordered sequence of ScheduleSteps covering the DAG.
// ============================================================================

class Solution {
public:
    // --- Construction from raw steps (used by solution_search) ---
    Solution(const Problem& prob, const DAG& dag, std::vector<ScheduleStep> steps);

    // --- Construction from a Partition ---
    //
    // Takes a const reference to a FINALIZED partition (caller must have called
    // finalize() already). Runs DFS and beam search ordering, returns the
    // lower-latency result. No deep copy, no redundant finalize.
    static Solution from_partition(const Problem& prob, const DAG& dag,
                                   const Partition& part, int max_beam_width = 10,
                                   class CostCache* cache = nullptr);

    // Lower-level: build ScheduleSteps from a pre-computed OrderingResult.
    // Performs per-step feasibility fallback (drop retains until feasible).
    // Exposed for callers (e.g. solver.cpp random variant) that compute their
    // own ordering.  When cache is non-null, retention-aware best_cost calls
    // are routed through it, pre-warming Phase 3's retention cache.
    static std::vector<ScheduleStep> steps_from_ordering(
        const Problem& prob, const DAG& dag,
        const Partition& part,
        const OrderingResult& res,
        class CostCache* cache = nullptr);

    // --- Validation ---

    struct ValidationResult {
        bool        valid = true;
        std::string error;
    };

    ValidationResult validate() const;

    // --- Cost ---

    double            total_latency()  const { return total_latency_; }
    double            step_latency(size_t i) const { return step_costs_[i].latency; }
    const CostResult& step_cost(size_t i)    const { return step_costs_[i]; }

    // --- Accessors ---

    const Problem&                   problem()           const { return *prob_; }
    const DAG&                       dag()               const { return *dag_; }
    size_t                           num_steps()         const { return steps_.size(); }
    const ScheduleStep&              step(size_t i)      const { return steps_[i]; }
    const std::vector<ScheduleStep>& steps()             const { return steps_; }
    const FlatSet<size_t>&          retained_entering(size_t i) const { return retained_entering_[i]; }

    // --- Ephemeral gap check (solution-level) ---

    static bool creates_ephemeral_gap(const Problem& prob, const DAG& dag,
                                       const FlatSet<size_t>& proposed_ops,
                                       const std::vector<ScheduleStep>& steps,
                                       size_t exclude_step,
                                       size_t exclude_step2 = SIZE_MAX);

private:
    const Problem*           prob_;
    const DAG*               dag_;
    std::vector<ScheduleStep> steps_;
    std::vector<CostResult>  step_costs_;
    std::vector<FlatSet<size_t>> retained_entering_;
    double                   total_latency_ = 0;
};