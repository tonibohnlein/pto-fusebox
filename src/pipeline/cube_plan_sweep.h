#pragma once

#include <string>

#include "core/dag.h"
#include "core/types.h"

// Serialize every feasible candidate considered by one homogeneous cube
// subgraph. Each candidate carries an ordinary solution.v6 payload, so model
// validation can test the production typed/source-readiness boundary without
// a benchmark-only lowering path. Analytic model-ahead candidates may still
// be rejected by that source boundary.
std::string cube_plan_sweep_json(const Problem& problem, const DAG& dag);
