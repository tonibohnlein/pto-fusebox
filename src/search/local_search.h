#pragma once

#include "partition/partition.h"
#include <functional>

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
// is_retained: optional callback to query whether a specific tensor is
// retained across a coupling edge.  When provided, an ephemeral tensor
// that is retained is not considered a gap.
bool partition_has_gap(const Partition& part,
                       std::function<bool(size_t)> is_retained = nullptr);
