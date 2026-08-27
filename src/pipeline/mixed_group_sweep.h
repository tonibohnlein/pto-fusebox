#pragma once

#include <string>

#include "core/dag.h"
#include "core/types.h"

// Serialize every feasible active-group divisor for the model-selected tile of
// one connected generic mixed region. The payload is diagnostic only: every
// component comes from the production cost path, while local search continues
// to retain only compact CostResult values.
std::string mixed_group_sweep_json(const Problem& problem, const DAG& dag);
