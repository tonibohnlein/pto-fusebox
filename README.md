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

The repository began as a Track A entry for the MLSys 2026 Scheduling Contest.
It now owns a standalone source-to-source path: capture and normalize a tensor
DAG, choose fusion and tiling with PTO Fusebox, and emit the selected schedule
as explicit PyPTO DSL. PTO Fusebox is not a PyPTO submodule, and the generated
program does not invoke a compiler-side AutoFuse or AutoTile pass. The
historical `mlsys` and `mlsys_mixed` executable names are retained for
command-line compatibility; embedders normally link `solver_lib`.

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
- `mlsys_mixed`: standalone solver using the experimental mixed model;
- `cube_plan_sweep`: enumerate every finite, fixed one-matmul cube candidate
  with its modeled cost and ordinary `solution.v3` replay payload; and
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

### Torch Export frontend

The optional Python package captures and normalizes Torch programs without
adding a C++ dependency on Torch:

```bash
python -m pip install -e ".[torch]"
```

```python
from pto_fusebox import (
    can_emit_region,
    emit_pypto_region,
    export_and_normalize,
    solve_graph,
)

graph = export_and_normalize(module, example_args)
result = solve_graph(
    graph,
    solver_binary="build/mlsys_mixed",
    solver_workers=2,
)
region = result.regions[0]
assert can_emit_region(graph, region)
source = emit_pypto_region(graph, region)
print(source.source)
```

The current Torch reader schedules **static-shape tensor DAGs only**. It records
symbolic tensor dimensions and their bounds for diagnostics, but any symbol
that determines solver geometry or participates in tensor arithmetic is an
explicit scheduling boundary. The reader does not yet specialize shape
buckets, emit runtime dispatch, or plan dynamic physical tiles.

Unsupported operations remain explicit graph boundaries. The source backend
replays supported homogeneous vector and cube steps from the selected schedule.
Vector source covers materialized/pointwise replay, versioned two-pass online
softmax, and one-reduction folded or spanning streams. Cube source covers
uniform non-split spatial plans, nested matmul DAGs, sequential outer-K windows,
on-chip produced values, and solver-selected retained boundary panels. Other
schedules raise a precise `SourceEmissionError` rather than being approximated.
Each maximal supported region is one solver input DAG, not one mandatory fused
kernel: when the solver selects several homogeneous steps, source emission
materializes their cut edges through explicit GM tensors and emits the steps as
dependency-linked `pl.spmd` launches in solver order. Mixed steps and split-K
source remain follow-ups.

The C++/Python boundary combines the typed problem descriptor with
`pto_fusebox.solution.v3`: C++ owns the selected launch, order, loops, physical
frames, lifetimes, and memory policy, while the problem retains the region ABI
and output-allocation lineage. Python builds one typed emission context from
both halves and renders it without searching again. The same graph-aware path
implements `can_emit_region` and actual emission, so readiness cannot accept a
program that the renderer later rejects. The renderer is organized as separate
API, shared-mechanics, vector, and cube modules. The online-softmax path consumes
the solver's versioned semantic state/substitution recipe; it does not recognize
a program or model name. Vector plans also serialize their spatial replay
policy: the current 910B model prices one maximum static tile per work unit and
clamps ragged-edge origins backwards, so generated source preserves static tile
shapes instead of turning known extents into runtime scalar operands. A
homogeneous step is emitted as one `pl.spmd(work_units)` launch rather than a
host loop of single-block submissions, matching the grid priced by the model.
Welford/multi-stat vector schedules, singleton-column normalization, nonuniform
cube spatial partitions, split-K, and mixed schedules remain explicitly not
source-ready until their complete state or transport contracts are serialized.

Runnable capture examples include the basic positive contracts plus
shape-reduced Torch forms of DeepSeek V4-Pro RMSNorm/MTP projection and the
Qwen3 final RMSNorm/LM head. The
[PyPTO PR #2335](https://github.com/hw-native-sys/pypto/pull/2335) reproduction
surface retains the exact four softmax shapes plus RMSNorm, LayerNorm, and SiLU
formulas used by the closed in-compiler AutoTile comparison:

```bash
python -m examples.torch_frontend.basic
python -m examples.torch_frontend.deepseek_v4
python -m examples.torch_frontend.qwen3
python -m examples.torch_frontend.pr2335_vector
```

Pass `--json` to inspect the normalized graph or `--solver build/mlsys_mixed`
to also submit its supported regions to an existing solver build. The model
examples preserve the relevant tensor algebra from `pypto-lib`; their shapes
and, where documented in the source, dtypes are reduced for practical local
execution. Every example matmul still satisfies `[M,K] @ [K,N] -> [M,N]`, and
all three axes span at least one legal 910B cube tile. They are coherent
reduced computations, not full checkpoint implementations.

## Test

```bash
cmake --build build --parallel 2 --target ascend_910b_test
./build/tests/ascend_910b_test
```

The suite intentionally reports a small documented baseline of model research
failures while checking the implemented vector, cube, and mixed schedule-plan
surface.

For model-versus-silicon ranking work, build `cube_plan_sweep` and feed it the
same lowered problem JSON used by the production solver:

```bash
cmake --build build --parallel 2 --target cube_plan_sweep
./build/cube_plan_sweep problem.json sweep.json
```

The sweep uses the production candidate grid and fixed-plan evaluator; it does
not fit a second model in Python. Each entry records the model cost and embeds
the exact forced solution. The predeclared device surface in
`test/device/cube_model_cases.py` covers underfill, balanced, ragged-K,
outer-K, rectangular-reuse, and split-K-positive shapes. Candidates beyond the
current source backend—most notably split-K and mixed cross-core plans—remain
useful analytic evidence but fail closed instead of being approximated by
source.

The generated-source silicon matrix is opt-in and is not part of the default
host suite. It covers 14 vector and 10 single-matmul cube programs, compiles
each emitted PyPTO program once, and checks five seeded executions:

```bash
PTO_FUSEBOX_RUN_DEVICE_TESTS=1 \
PTO_FUSEBOX_DEVICE_ID=<physical-id> \
PTO_FUSEBOX_SOLVER=build/mlsys_mixed \
PYTHONPATH=python:<pypto-checkout>/python \
python -m pytest test/device/test_source_silicon.py -v
```

## Design notes

- [Torch/Hugging Face to PyPTO source generation](doc/torch_to_pypto_frontend.md)
  records the proposed PTO-Fusebox-owned external frontend, source backend, and
  staged dynamic-shape scope.

## Repository layout

```text
src/core/       DAG, subgraph, hardware costs, and schedule plans
src/partition/  Partition state and group bookkeeping
src/search/     Greedy, FM, evolution, and parallel search
src/solution/   Schedule construction and traversal ordering
src/io/         JSON input and output
src/pipeline/   Library and standalone entry points
test/           Unit, integration, and 910B grounding tests
python/         Torch Export normalization and solver subprocess frontend
doc/            Cost-model and solver design notes
scripts/        Validation, rendering, and benchmark helpers
```

## License

PTO Fusebox is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
