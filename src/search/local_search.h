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
bool partition_has_gap(const Partition& part);
