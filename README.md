# PTO Fusebox

PTO Fusebox is a solver-driven fusion and tiling engine for tensor-operation
graphs. It partitions a DAG into convex kernel groups, selects each group's
tile/grid/split strategy, and exposes the execution schedule needed by a
faithful kernel emitter.

The current hardware model targets the Ascend 910B. It includes:

- grounded vector primitive, reduction, DMA, and launch costs;
- `VectorStreamPlan` schedules for materialized, streamed, and online
  multi-stat reductions;
- recursive `CubeSchedulePlan` schedules for matmuls and matmul DAGs;
- an experimental mixed cube/vector model; and
- compact cost evaluation suitable for local-search enumeration.

The repository began as a Track A entry for the MLSys 2026 Scheduling Contest
and was subsequently developed into the scheduling and cost-model component of
PyPTO AutoFuse. The historical `mlsys` and `mlsys_mixed` executable names are
retained for command-line compatibility; embedders normally link `solver_lib`.

## Requirements

- Linux x86_64 (Ubuntu 22.04 LTS or compatible)
- A C++20 compiler (`g++-12` or newer is recommended)
- CMake 3.16 or newer
- `nlohmann-json3-dev` on Debian/Ubuntu, or `nlohmann-json-devel` on Fedora/RHEL

On Ubuntu 22.04:

```bash
sudo apt install build-essential cmake g++-12 nlohmann-json3-dev
```

## Build

Use at most two parallel compilation jobs; the search and cost-model sources
are memory intensive.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

The primary targets are:

- `solver_lib`: embeddable static library;
- `mlsys`: standalone solver using the homogeneous 910B model;
- `mlsys_mixed`: standalone solver using the experimental mixed model; and
- `ascend_910b_test`: grounded cost and schedule-plan regression suite.

For a portable standalone binary with static libstdc++ and libgcc:

```bash
CXX=g++-12 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTATIC_STDLIB=ON
cmake --build build --parallel 2 --target mlsys
```

## Run

```bash
./build/mlsys input.json output.json
```

The input is a JSON computation graph plus hardware constraints. The output
records the selected subgraph grouping, tile granularities, retention choices,
and traversal order.

## Test

```bash
cmake --build build --parallel 2 --target ascend_910b_test
./build/tests/ascend_910b_test
```

The suite intentionally reports a small documented baseline of model research
failures while checking the implemented vector, cube, and mixed schedule-plan
surface.

## Repository layout

```text
src/core/       DAG, subgraph, hardware costs, and schedule plans
src/partition/  Partition state and group bookkeeping
src/search/     Greedy, FM, evolution, and parallel search
src/solution/   Schedule construction and traversal ordering
src/io/         JSON input and output
src/pipeline/   Library and standalone entry points
test/           Unit, integration, and 910B grounding tests
doc/            Cost-model and solver design notes
scripts/        Validation, rendering, and benchmark helpers
```

## License

PTO Fusebox is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
