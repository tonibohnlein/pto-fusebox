#pragma once

#include "core/types.h"
#include "core/dag.h"
#include <set>
#include <vector>

// ============================================================================
// Feasibility: cycle checks for partition search.
//
// Two layers:
//   1. O(degree) pre-filters — cheap heuristics that catch most infeasible moves
//   2. Partition-level Kahn's — full topological sort with virtual group mapping
//
// Rule: every acyclicity check in the partition search must live here.
// Search files (fm_search, local_search, evolution) call these functions;
// they NEVER implement cycle checks inline.
// ============================================================================

namespace feasibility {

// ============================================================================
// Partition-level Kahn's algorithm
//
// MoveDelta describes a hypothetical move so we can check acyclicity
// WITHOUT copying op_to_groups.  kahn_with_delta does a full topological
// sort of the group DAG "as if" the move had been applied.
//
// Returns true if the group DAG remains acyclic after the hypothetical move.
// ============================================================================

struct MoveDelta {
    enum Type {
        NONE,           // no move — check current state
        MERGE_PAIR,     // merge ga and gb into ga
        MERGE_MULTI,    // merge all groups in merge_list into merge_list[0]
        STEAL,          // move op from ga to gb
        RECOMPUTE,      // copy op into gb (op stays in ga)
        SPLIT_MOVE,     // move split_ops from ga to new virtual group gb
        EXTRACT_MOVE    // move split_ops from source groups (merge_list) to gb
    } type = NONE;

    size_t op = SIZE_MAX;
    size_t ga = SIZE_MAX;
    size_t gb = SIZE_MAX;
    const std::vector<size_t>* merge_list = nullptr;
    const FlatSet<size_t>* split_ops = nullptr;
};

// Core algorithm: Kahn's topological sort with virtual group mapping.
// All partition-level acyclicity methods delegate to this.
bool kahn_with_delta(
    const Problem& prob,
    const DAG& dag,
    const std::vector<std::vector<size_t>>& op_to_groups,
    const std::vector<bool>& alive,
    size_t num_alive_groups,
    const MoveDelta& delta);

} // namespace feasibility