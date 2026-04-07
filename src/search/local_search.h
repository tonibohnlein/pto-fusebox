#pragma once

#include "partition/partition.h"

// ============================================================================
// Greedy descent: repeatedly apply the best positive move until no
// improving move exists. Returns a local optimum.
// Uses best_move_for() and apply_fm_move() from fm_search.
// ============================================================================

Partition greedy_descent(Partition part);

// ============================================================================
// Post-search validation
// ============================================================================

// Quick full gap check: returns true if ANY ephemeral gap exists
// OR the group DAG has a cycle. Used as post-move safety check.
//
// retained_tensors: tensors retained across coupling edges.  An ephemeral
// tensor in this set is not considered a gap (it is materialized in fast
// memory for the next step).
bool partition_has_gap(const Partition& part,
                       const FlatSet<size_t>& retained_tensors = {});
