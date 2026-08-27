#include "pipeline/mixed_group_sweep.h"

#include <fstream>
#include <iostream>

#include "core/dag.h"
#include "io/io.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <input-problem.json> <output-sweep.json>\n";
    return 1;
  }
  try {
    const Problem problem = read_problem(argv[1]);
    const DAG dag = DAG::build(problem);
    std::ofstream output(argv[2]);
    if (!output.is_open()) {
      std::cerr << "Error: cannot write '" << argv[2] << "'\n";
      return 1;
    }
    output << mixed_group_sweep_json(problem, dag);
    if (!output) {
      std::cerr << "Error: write failed for '" << argv[2] << "'\n";
      return 1;
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
