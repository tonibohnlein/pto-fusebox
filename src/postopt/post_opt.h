#pragma once

#include "solution/solution.h"

// Retain optimization: try retaining tensors between consecutive steps
Solution optimize_retain(const Problem& prob, const DAG& dag, Solution sol);

// Recompute optimization: try adding predecessor ops to eliminate boundary loads
Solution optimize_recompute(const Problem& prob, const DAG& dag, Solution sol);
