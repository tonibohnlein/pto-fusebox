#pragma once

#include "search/coupling_search.h"
#include "search/coupled_fm_search.h"
#include "search/fm_pass.h"   // FMConfig (reused as-is)
#include <set>

// ============================================================================
// Coupled FM pass result
// ============================================================================

struct CoupledFMPassResult {
    CoupledPartition best_cp;
    CoupledPartition end_cp;
    double best_cost  = 1e18;
    double end_cost   = 1e18;
    double start_cost = 1e18;
    int moves_applied = 0;
    int moves_positive = 0;
    int moves_negative = 0;
};

// ============================================================================
// One FM inner pass on a CoupledPartition.
//
// Mirrors fm_inner_pass but the active set uses CoupledFMMove, evaluating
// both partition moves and coupling moves (COUPLE / UNCOUPLE /
// RETAIN_FORCE_SPLIT).
//
// Checkpoint saves groups + coupling arrays; restored at end.
// ============================================================================

CoupledFMPassResult coupled_fm_inner_pass(
    CoupledPartition               cp,
    const FlatSet<size_t>&        feasibly_ret,
    const FMConfig&                cfg = {});
