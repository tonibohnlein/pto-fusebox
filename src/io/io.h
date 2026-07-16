#pragma once

#include "core/types.h"
#include "solution/solution.h"
#include <string>

Problem read_problem(const std::string& filename);
std::string solution_json(const Solution& sol);
void write_solution(const std::string& filename, const Solution& sol);
