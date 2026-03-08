#pragma once

#include "core/types.h"
#include <vector>

// Generate tile index permutation for a given snake direction.
// Used by Subgraph::compute_cost internals and by io.cpp for output.
std::vector<int> make_traversal(int num_tw, int num_th, SnakeDir dir);
