#pragma once

#include "core/types.h"
#include "solution/solution.h"
#include <string>

Problem read_problem(const std::string& filename);
void write_solution(const std::string& filename, const Solution& sol);
