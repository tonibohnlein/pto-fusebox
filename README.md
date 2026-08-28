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
- `cube_plan_sweep`: enumerate every finite, fixed homogeneous cube-DAG candidate
  with its modeled cost and ordinary `solution.v6` replay payload; and
- `mixed_group_sweep`: enumerate every uniform active-group assignment for the
  model-selected mixed tile with its production pipe/stage cost breakdown; and
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
    emit_pypto_callable,
    export_and_normalize,
    solve_graph,
)

graph = export_and_normalize(module, example_args)
result = solve_graph(
    graph,
    solver_binary="build/mlsys_mixed",
    solver_workers=2,
    require_source_codegen=True,
)
region = result.regions[0]
assert can_emit_region(graph, region)
callable_source = emit_pypto_callable(
    graph,
    region,
    function_name="fused_region",
)
print(callable_source.source)
```

The current Torch reader schedules **static-shape tensor DAGs only**. It records
symbolic tensor dimensions and their bounds for diagnostics, but any symbol
that determines solver geometry or participates in tensor arithmetic is an
explicit scheduling boundary. The reader does not yet specialize shape
buckets, emit runtime dispatch, or plan dynamic physical tiles. Native PyPTO
orchestration owns that dynamic behavior. For an eligible single-step vector
region, import `RuntimeValidShapeSpec` and pass
`runtime_valid_shape=RuntimeValidShapeSpec()` to
`emit_pypto_callable`. This adds one runtime logical-row scalar while retaining the
solver's fixed tensor annotations, physical tiles, grid, and memory plan.
Cube, mixed, multi-step, and non-row dynamic geometry still fail closed.

`emit_pypto_callable` is the integration-oriented API: it returns one
module-level `@pl.inline` function with stable named tensor and output
arguments, ordered normalized value IDs for binding, and optional runtime
logical-shape metadata. Native PyPTO orchestration imports that function and
owns loops, metadata, dispatch, and dependencies outside the static region.
`emit_pypto_region` remains available when a standalone generated
`@pl.program` is more convenient.

Unsupported operations remain explicit graph boundaries. The source backend
replays supported vector, cube, and mixed steps from the selected schedule.
Vector source covers materialized/pointwise replay, versioned two-pass online
softmax, and one-reduction folded or spanning streams. Cube source covers
uniform non-split spatial plans, nested matmul DAGs, sequential outer-K windows,
on-chip produced values, solver-selected retained boundary panels, and
single-sink split-K through dependency-linked PyPTO tasks. Other
schedules raise a precise `SourceEmissionError` rather than being approximated.
Each maximal supported region is one solver input DAG, not one mandatory fused
kernel: when the solver selects several homogeneous steps, source emission
materializes their cut edges through explicit GM tensors and emits the steps as
dependency-linked `pl.spmd` launches in solver order. A split cube DAG may
split only its unique sink: upstream matmuls replay their serial K plans inside
each share, while resident inputs and retained panels stay local to that share.
Multiple split accumulators and multiple-output split groups fail closed.
Mixed source initially covers generic one-way `C -> V`, generic one-way
`V -> C` with an in-memory or online-softmax vector producer, generic
`C -> V -> C`, dense
`C,C -> V -> C`, and one linear `C -> V -> C -> V` plan through PyPTO's
public `pl.split(UP_DOWN)` mechanism. Fusebox retains its logical FIFO
descriptors for planning, validation, and costing, and emits each one through
PyPTO's generic `pl.cross_core_pipe(...)` contract. PyPTO validates the ordered
unidirectional descriptors against the actual crossings, then owns AIC/AIV
outlining, pipe setup, and pipeline lowering. Different pipes may therefore
carry different slot sizes and depths without hand-written core programs. The
four-stage form is deliberately
sequential: its three FIFO crossings replay in topological order and receive no
skew-overlap credit. Online softmax-to-PV replays the serialized statistics and
apply phases, publishes each normalized K chunk through one V2C pipe, and
accumulates the sink matmul over the same K windows. A complete square produced
panel may also serve both operands of a single-region sink matmul through one
FIFO-owned L1 ring. Partitioned dual-role values, branched/deeper round trips,
and mixed multi-step composition still fail closed.
Mixed source readiness combines the serialized cube-stage L1 peak with V2C
ring reservations and the vector-stage Vec peak with C2V ring reservations.
The initial split-K task bundle must be the region's only selected step; the
multi-step composer fails closed rather than splicing its internal dependency
into a larger launch sequence.

The earlier fork-only explicit-pipe contract was silicon-closed for one-way
`C -> V`, generic `C -> V -> C`, dense `C,C -> V -> C`, and sequential
`C -> V -> C -> V` source families. It is historical validation evidence, not
the compatibility target. PTO Fusebox now targets upstream PyPTO `main` and
emits only its public automatic-pipe contract. On the current upstream-main
host matrix, the callable integration surface covers multi-step cube,
single-sink split-K, mixed attention, and dense SwiGLU inside independent
native orchestration. The dense case still exposes a general
nested-accumulator join defect in PyPTO's memory-reuse lowering until the
corresponding generic PyPTO repair is present. Current-main runtime-valid and
PyPTO-lib comparison silicon revalidation is in progress; historical evidence
is not presented as current-main closure.
An earlier reported residual in the generic attention case was retracted: the
device harness passed `(query, key, value)` positionally to an emitted
`(key, query, value)` ABI. Generated source now publishes its ordered normalized
input value IDs, and the checked-in device harness binds tensors through those
IDs. Dense SwiGLU transport is also closed: generated and independently
hand-written PyPTO sources are bit-identical. In the closure fixture the direct
Torch comparison has zero tolerance misses and one of 8192 outputs differs by
one BF16 ULP, so BF16 narrowing remains an explicitly reported numerical-oracle
caveat rather than a FIFO or source-emission defect.

Generic one-way `V -> C` passes model and typed-plan checks for LHS, RHS,
online-softmax-to-PV, and the complete-square dual-role case. Source and silicon
status are evaluated against upstream PyPTO `main`, rather than inherited from
the historical explicit-pipe fork.
Partitioned dual-role schedules are rejected until the plan defines replication
and FIFO ownership. PyPTO already supports multiple dependency-
linked `pl.spmd` tasks, so deeper composition does not require a new PyPTO
primitive. Fusebox still needs an ordered mixed task-bundle contract that
preserves GM cuts, task dependencies, internal pipes, and result ownership.
Branched replay additionally needs explicit fan-out, lifetime, and per-consumer
FIFO ownership and must not be inferred by the emitter.

The C++/Python boundary combines the typed problem descriptor with
`pto_fusebox.solution.v6`: C++ owns the selected launch, order, loops, physical
frames, lifetimes, and memory policy, while the problem retains the region ABI
and output-allocation lineage. Python builds one typed emission context from
both halves and renders it without searching again. The same graph-aware path
implements `can_emit_region` and actual emission, so readiness cannot accept a
program that the renderer later rejects. The renderer is organized as separate
API, shared-mechanics, vector, cube, and mixed modules. The online-softmax path consumes
the solver's versioned semantic state/substitution recipe; it does not recognize
a program or model name. Vector plans also serialize their spatial replay
policy: the current 910B model prices one maximum static tile per work unit and
clamps ragged-edge origins backwards, so generated source preserves static tile
shapes instead of turning known extents into runtime scalar operands. A
homogeneous step is emitted as one `pl.spmd(work_units)` launch rather than a
host loop of single-block submissions, matching the grid priced by the model.
Welford/multi-stat vector schedules, singleton-column normalization, nonuniform
cube spatial partitions, and mixed schedules outside the admitted stage
patterns remain explicitly outside source readiness.
Single-sink split-K is source-ready through the model-selected
`FirstPartialThenAtomic` or `AivZeroSeedThenAtomic` PyPTO mechanism.

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
python -m examples.torch_frontend.static_mixed
```

Pass `--json` to inspect the normalized graph or `--solver build/mlsys_mixed`
to also submit its supported regions to an existing solver build. Add
`--emit-source` with `--solver` to constrain selection to the external source
contract and print a standalone PyPTO program. Use `--emit-callable` instead to
print the importable `@pl.inline` form intended for native PyPTO orchestration.
Plain solver runs keep the broader analytic search surface. The model
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
current source backend—including deeper mixed cross-core plans—remain useful
analytic evidence but fail closed instead of being approximated by source.

The generated-source silicon matrix is opt-in and is not part of the default
host suite. Its reusable base matrix covers 14 vector and 10 single-matmul cube
programs, compiles each emitted PyPTO program once, and checks five seeded
executions:

```bash
PTO_FUSEBOX_RUN_DEVICE_TESTS=1 \
PTO_FUSEBOX_DEVICE_ID=<physical-id> \
PTO_FUSEBOX_SOLVER=build/mlsys_mixed \
PYTHONPATH=python:<pypto-checkout>/python \
python -m pytest test/device/test_source_silicon.py -v
```

The opt-in source-integration suite separately exercises callable expansion
inside native PyPTO orchestration, including multi-step cube, split-K, mixed
attention, dense SwiGLU, Qwen RMSNorm/LM-head components, and runtime-valid
vector lowering. Those tests establish source and compiler contracts; focused
two-device campaigns establish silicon correctness, stability, and
like-for-like performance.

## Design notes

- [Torch/Hugging Face to PyPTO source generation](doc/torch_to_pypto_frontend.md)
  documents the implemented external frontend, callable source backend, native
  orchestration boundary, and staged static-region validation targets.

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
