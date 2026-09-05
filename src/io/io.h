#pragma once

#include "core/types.h"
#include "solution/solution.h"
#include <string>
#include <vector>

Problem read_problem(const std::string& filename);
std::string solution_json(const Solution& sol);
void write_solution(const std::string& filename, const Solution& sol);
std::string source_candidate_summaries_json(const std::vector<Solution>& candidates);
void write_source_candidate_summaries(const std::string& filename,
                                      const std::vector<Solution>& candidates);
