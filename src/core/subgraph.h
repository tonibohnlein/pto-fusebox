#pragma once

// ============================================================================
// `Subgraph` — the active per-architecture subgraph handle used throughout the
// solver pipeline. It is an alias for the compile-time-selected cost model.
//
// To retarget the solver to another architecture, swap Ascend910BCost for that
// architecture's CostModel implementation (e.g. a future Ascend950Cost); every
// pipeline type that names `Subgraph` follows automatically. This is the single
// compile-time switch point.
//
// New code that is genuinely architecture-independent should prefer
// SubgraphStructure (subgraph_structure.h) directly.
// ============================================================================

#include "core/ascend910b_cost.h"
#include "core/cost_model.h"

using Subgraph = SubgraphT<Ascend910BCost>;
