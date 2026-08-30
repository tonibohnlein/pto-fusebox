# Torch/Hugging Face to PyPTO source generation

## Status

The first frontend milestone is implemented. PTO-Fusebox can capture a
`torch.nn.Module` through `torch.export`, normalize an existing
`torch.export.ExportedProgram`, preserve unsupported operations as explicit
boundaries, extract statically schedulable regions, lower them into a versioned
solver problem, and invoke an already-built C++ solver.

The implemented reader is currently **static-shape only for scheduling**.
Symbolic tensor dimensions and bounds are retained in the normalized graph so
capture remains faithful and future extensions have a stable boundary, but no
symbolic region is lowered to the solver today. Shape-derived scalar values
used by computation are preserved as explicit opaque boundaries rather than
being approximated. PTO-Fusebox does not synthesize dynamic orchestration:
native PyPTO orchestration remains responsible for runtime loops, metadata,
dispatch, and logical extents, while Fusebox generates the callable static
kernels used by that orchestration.

The PyPTO DSL source backend validates a solved region as a typed schedule and
can emit either an ordinary standalone `@pl.program` or a module-level
`@pl.inline` callable for native orchestration. Both forms contain the same
grids, logical ownership, physical frames, loops, pipeline stages, operation
order, and GM traffic. Each
homogeneous group is one `pl.spmd(work_units)` launch; its block index selects
one solver-owned output region. If the solver cuts a region into several
homogeneous groups, the program creates explicit GM tensors for the cut edges
and submits dependency-linked SPMD tasks in solver order.

Vector execution replays one maximum compile-time tile per work unit and clamps
ragged-edge origins backwards, preserving static shapes while recomputing the
small overlap already charged by the model. The implemented vector set is
materialized or pointwise replay, versioned two-pass online softmax, and
one-reduction folded or spanning streams. Cube execution covers uniform
non-split spatial schedules, nested matmul DAGs, sequential outer-K windows,
produced values resident in L1, and solver-selected retained boundary panels.

Mixed source covers generic one-way `C -> V`, generic one-way `V -> C` with an
in-memory or online-softmax vector producer, generic `C -> V -> C`, dense
`C,C -> V -> C`, and linear `C -> V -> C -> V` schedules. It replays the
serialized stages and tensor DAG through one `pl.spmd` grid with
`pl.split(UP_DOWN)`; PyPTO inserts
the concrete push/pop/free pipeline. The four-stage form is an ordinary ordered
loop and receives no skew-overlap credit. Online softmax-to-PV publishes
normalized K chunks into the sink's matching accumulation windows. One complete
square panel may serve both sink operands without replication; partitioned
dual-role values, branched/deeper round trips, and mixed multi-step composition
fail closed. Welford/multi-stat
vector plans, singleton-column normalization, and nonuniform cube spatial
partitions remain outside source readiness. A uniform cube DAG may split only
its unique sink through the selected dependency-linked PyPTO task protocol.
Dynamic-shape classes are retained but declined when they affect solver
geometry.

## Python API

Install the optional Torch frontend dependencies and import the public package:

```bash
python -m pip install -e ".[torch]"
```

```python
from pto_fusebox import (
    RuntimeValidShapeSpec,
    emit_pypto_callable,
    export_and_normalize,
    solve_graph,
)

graph = export_and_normalize(module, example_args, dynamic_shapes=constraints)
result = solve_graph(
    graph,
    target="ascend910b",
    solver_binary="build/mlsys_mixed",
    solver_workers=2,
    require_source_codegen=True,
)
source = emit_pypto_callable(
    graph,
    result.regions[0],
    function_name="fused_region",
    runtime_valid_shape=RuntimeValidShapeSpec(),  # optional vector free-axis ABI
)
print(source.source)
```

The core entry points are:

- `export_and_normalize(module, args, ...) -> NormalizedGraph`;
- `normalize_exported(program) -> NormalizedGraph`;
- `extract_solver_regions(graph, target) -> list[SolverRegion]`;
- `solve_graph(graph, target=..., solver_binary=...) -> SolveResult`;
- `scheduled_region(result) -> ScheduledRegion`;
- `can_emit_region(graph, result) -> bool`;
- `emit_pypto_callable(graph, result, ...) -> EmittedPyPTOCallable`; and
- `emit_pypto_region(graph, result, ...) -> EmittedPyPTOSource`.

`normalize_exported` is authoritative; `export_and_normalize` is a convenience
wrapper around `torch.export.export`. `solve_graph` never builds the solver. A
caller must provide an executable or set `PTO_FUSEBOX_SOLVER`.
The optional `require_source_codegen=True` flag applies the external PyPTO
source topology, stage-kind, and physical FIFO capacity contract to the first
candidate search. Leave it false to preserve the unrestricted analytic search.
`scheduled_region`
rejects incomplete or internally inconsistent solution
arrays before emission. `can_emit_region` and `emit_pypto_region` build the same
typed emission context and run the same graph-aware renderer validation; the
readiness query is not a weaker schedule-family approximation. The emitter
accepts one or more selected homogeneous steps or one supported mixed step and
raises `SourceEmissionError` for every unimplemented algorithm or unsafe cut
edge.
`EmittedPyPTOSource.kinds` preserves the ordered engine kind of every emitted
step; its compatibility `kind` property is populated only when all steps use
the same engine kind.
`EmittedPyPTOCallable.input_arguments` and `output_arguments` expose the stable
normalized value ID and generated Python name for every ABI argument. The
callable is an inline orchestration fragment rather than an `InCore` shortcut:
calling it expands the exact solver-selected SPMD task graph, so multi-step,
split-K, and mixed schedules retain their launch and dependency structure.
Callable extraction requires exactly one generated program class containing
one `main` function and no other class members. Unexpected members, multiple
generated functions, and runtime argument names that collide with generated
identifiers fail closed instead of being silently dropped or shadowed.

`runtime_valid_shape=RuntimeValidShapeSpec()` adds a named
`valid_rows: pl.Scalar[pl.INDEX]` argument to a single homogeneous vector
callable. The tensor annotations and every physical tile remain the concrete
shapes priced by the solver. Only non-broadcast GM loads receive
`max(min(valid_rows - row_offset, planned_valid_rows), 0)`; PyPTO propagates
that logical shape through the vector DAG and its stores. Axis 1, cube or mixed
steps, and multi-step regions fail closed because those cases can change a
schedule-defining contraction or crossing geometry.

Native orchestration imports and calls the generated function normally:

```python
from generated_region import fused_region

@pl.program
class Model:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: InputType,
        valid_rows: pl.Scalar[pl.INDEX],
        output: pl.Out[OutputType],
    ) -> OutputType:
        # Runtime loops, metadata, and dispatch remain here.
        output = fused_region(x, valid_rows, output)
        return output
```

The example runner accepts `--emit-callable` to print this importable form;
`--emit-source` retains the standalone `@pl.program` form.

### Source-backend structure and validation

The source backend is schedule-family-driven, not model- or pattern-driven.
It selects a vector, cube, or mixed emitter from the typed solver step, replays
the solver's operation order and serialized geometry, and dispatches
individual operations by normalized operator kind. Names such as softmax,
RMSNorm, attention, SwiGLU, or a source `nn.Module` class are never emission
inputs. The cube and mixed families are driven by typed request/stage DAGs
rather than by example or model names. Cube replay preserves the selected
topological order, propagated regions, outer-K windows, local produced-value
lifetimes, retained boundary panels, and drains. Mixed replay preserves the
selected engine stages, propagated crossing frames, cube K windows, generic
vector DAG, group loop, and pipeline depth.

The replay structure follows the earlier PyPTO fusion-scheduler prototype:
one solver-owned grid, propagated regions, planned physical frames and
partitions, lifetime-respecting topological replay, and pipelined strip or K
windows. The implementations are not shared code: the prototype builds PyPTO
IR in C++, while this repository emits readable PyPTO DSL source. The
serialized schedule is the common contract.

Reference checks use three levels:

- PyPTO-lib programs establish ordinary DSL structure, including DeepSeek
  RMSNorm and Qwen decode examples;
- PTO-ISA/PTOAS examples establish lowered data paths and tile constraints; and
- opt-in source integration tests parse the DSL with an independently selected
  PyPTO checkout and compile it through PTOAS.

The integration gate is deliberately outside the default unit-test dependency
set. Run it in the target validation environment with a built Fusebox solver,
the intended PyPTO import on `PYTHONPATH`, and a valid `PTOAS_ROOT`:

```bash
PTO_FUSEBOX_PYPTO_INTEGRATION=1 \
PTO_FUSEBOX_TEST_SOLVER=build/mlsys_mixed \
PYPTO_CODEGEN_MAX_WORKERS=2 \
python -m pytest test/python/test_source_pypto_integration.py -q
```

## Versioned boundaries

The frontend publishes three schemas:

- `pto_fusebox.normalized_graph.v1`: semantics-preserving normalized capture data;
- `pto_fusebox.problem.v1`: a statically lowered solver region; and
- `pto_fusebox.solution.v6`: the C++ schedule response. Cross-kernel values are
  always materialized through GM. Fast-memory residence and retained panels are
  cube-step-local policies, not promises spanning separate launches.

The normalized graph records stable topological IDs, ordered operands, exact
normalized operator kinds, attributes, input roles, target names for parameters
and buffers, logical shapes, strides, storage offsets, dtypes, alias
relationships, ordered duplicate outputs, input/output PyTree structures,
explicit output kinds and mutation targets, opaque reasons, symbolic bounds,
diagnostics, and exact algorithm patterns. Parameter and buffer payloads are
never embedded. Canonical JSON is independent of temporary paths, object
addresses, and PyTorch's process-global symbolic names. Deserialization checks
producer consistency, topological order, alias cycles, interface IDs, positive
static dimensions, pattern references, and output-spec ordering before a graph
can reach the scheduler.

The problem schema is intentionally narrower. It contains one supported static
region in the existing solver's two-dimensional tensor representation plus a
`frontend_mapping` that maps every solver tensor/op back to normalized graph
IDs and carries the canonical normalized-graph SHA-256. Source emission checks
that digest, so a solution cannot be combined with a same-shaped graph carrying
different operations or attributes. Missing `schema_version` remains accepted
for legacy benchmark inputs; unknown explicit versions fail closed.

A zero-valued granularity and infinite latency are internal C++ cost-search
sentinels meaning that no feasible tile was found. They are never a valid
schedule. The Python bridge classifies such a response as `infeasible` and
must not pass it to the PyPTO source backend. The solution uses one typed,
nested step descriptor. Its common portion records launch geometry, selected
topological order, sequential tile counts, and latency.
The common launch tile is the optimizer's selected configuration and a
diagnostic summary; the nested family plan is authoritative wherever lowering
derives different replay frames, as it does for vector reductions.
Vector plans additionally record phase operation order, input lifetimes,
logical/physical tensor frames, the static clamped-overlap spatial replay
policy, reduction workspaces, and exact loop bounds.
Cube plans record propagated regions and axis bindings, execution order,
resident-boundary lifetimes, K/L0 loops, retained panels, drains, and split
policy. These fields are solver output, not choices rediscovered by Python
emission.

The Python boundary decodes `problem.v1` into `LoweredRegion` and `solution.v6`
into immutable `ScheduledRegion`/`KernelStep` types before rendering. The
lowered half owns region inputs, outputs, and output-allocation lineage; the
scheduled half owns execution. Together with the normalized graph they form a
single `EmissionContext`, so neither readiness nor rendering reconstructs an
ABI from a partial view. Unknown, missing, or internally inconsistent fields
fail closed. The source package is split by responsibility: `source/api.py`
owns context construction and dispatch, `source/common.py` owns typed
ABI/naming/partition mechanics, and `source/vector.py`, `source/cube.py`, and
`source/mixed.py` replay their respective plans.

## v1 normalization and admission

The v1 reader accepts a closed, static tensor-DAG subset. All tensor extents
that determine two-dimensional solver geometry must be positive compile-time
integers. Parameters and buffers may remain external tensors, but scalar shape
programs, data-dependent indexing, Python control flow, mutations, and dynamic
work counts are outside the schedulable subset. Unsupported semantics remain
visible as opaque operations and split solver regions; they are never silently
dropped.

The implemented closed set includes casts; tensor/scalar arithmetic; exp, log,
abs, sqrt, rsqrt and negation; last-axis sum/max/mean with `keepdim=True`;
rank-2 matmul/mm; rank-2-or-higher linear with contiguous leading dimensions
collapsed into solver height; metadata-only shape aliases; and an immediately
consumed rank-2 transpose represented as a zero-copy view.

Softmax is decomposed into its ordinary max/sub/exp/sum/div tensor DAG. The
graph also carries the exact P4 op set and semantically named substitutions
that permit the existing online softmax implementation. There is no attention, RMSNorm, or
model-name recognizer. Mean is similarly lowered to sum plus reciprocal
multiplication, and linear becomes matmul plus optional broadcast bias.

Ascend 910B admission keeps FP16/FP32 vector arithmetic, BF16 storage/cube/cast
endpoints, and the supported cube dtypes distinct. It accepts dense storage at
zero offset and only the equal/scalar/row-singleton/column-singleton broadcast
geometries represented by the existing two-dimensional problem. A rank-2
transpose is retained only as an explicit external matmul operand; an internal
transpose remains a scheduling boundary until transformed-alias edges exist in
the solver schema. Unsupported operations, mutations, copying views,
non-default operator semantics, data-dependent results, batched matmul, and
schedule-defining symbolic dimensions are never approximated. They remain in
the normalized graph with a stable reason and delimit deterministic maximal
convex solver regions. Same-dtype `to(copy=False)` is a metadata alias, while
`copy=True` declines. Real casts are expanded at the target-lowering boundary
into the 910B native conversion path (for example FP16 -> FP32 -> BF16 and FP32
-> FP16 -> INT8), so every intermediate dtype and lifetime is visible to solver
UB accounting without changing the public normalized graph. Source emission
rejects Torch float-to-INT8 casts because the native multi-hop conversion does
not yet encode direct Torch truncation; analytic scheduling remains available.

The first contract suite fixes the exact normalized and solver DAG for RMSNorm,
softmax, rank-3 linear with bias, plain matmul, generic `QK -> softmax -> PV`,
and a TopK boundary. Additional adversarial tests cover opaque bypass diamonds,
metadata aliases between computations, structured duplicate outputs, layout
and storage-offset rejection, broadcast admission, native cast chains, and
near misses for exact mixed-algorithm scalar semantics.

The runnable model examples use reduced, hardware-representative static shapes
rather than production checkpoint sizes. Their tests independently verify each
matmul contraction and output shape, require every `M`, `N`, and `K` to span at
least one legal cube tile, and run every complete supported region through an
existing `mlsys_mixed` build when one is available.

The standalone target admits the complete analytic schedule surface by
default. It does not restrict partition search to schedules supported by the
historical in-compiler AutoFuse emitter: PTO-Fusebox will ultimately generate
tensor/tile PyPTO source from its own selected schedule. Split cube DAGs,
multi-reduction streaming, non-uniform cube-DAG grids, and deeper serial mixed
topologies therefore remain eligible even when no source
emitter path exists. Solution metadata records the stages, directional
transfers, and protocol for every analytic mixed plan. Plans with a complete
stage-local geometry, vector stream, cube-window, and FIFO contract also set
their internal `source_codegen_ready=true` contract-completeness bit.
When the caller requests `require_source_codegen=True`, source constraints
participate in candidate costing and selection from the first solve. This can
intentionally select a different source-realizable tile than the unrestricted
analytic optimum. With the flag false, analytic admission and costing are
unchanged.
The Python backend separately admits generic one-way `C -> V`, generic one-way
`V -> C` with an in-memory or online-softmax vector producer, generic
`C -> V -> C`, dense
`C,C -> V -> C`, and linear sequential `C -> V -> C -> V` today. Broader
analytic winners remain valid research results but are not presented as
source-emittable.
`regions_solved` therefore reports solver success for the requested analytic or
source-oriented policy separately from `whole_graph_codegen_ready`, which
additionally requires no opaque graph boundaries and a successful exact
`can_emit_region(graph, region)` check for every selected step.

## Goal and ownership

PTO-Fusebox owns the static source-to-source scheduling path:

```text
PyTorch/Hugging Face static tensor function + concrete physical shapes
        |
        v
torch.export / FX graph capture
        |
        v
PTO-Fusebox Torch graph adapter
        |
        v
PTO-Fusebox AutoFuse + AutoTile planning
        |
        v
PTO-Fusebox PyPTO DSL source backend
        |
        v
callable static PyPTO kernel with a stable named ABI
        |
        v
native PyPTO orchestration + ordinary PyPTO compiler and runtime
```

The way forward is deliberately hybrid:

- keep native PyPTO orchestration;
- generate callable static PyPTO kernels with stable named ABIs;
- use Fusebox for every vector, cube, or mixed static region;
- treat `valid_shape` as a runtime parameter over a statically planned physical
  frame; and
- compare generated kernels against the corresponding hand-written PyPTO-lib
  implementations.

Torch functions are therefore tensor-algorithm specifications for static call
sites, not a second orchestration language. Existing PyPTO orchestration keeps
ownership of runtime dimensions, loops, cache metadata, indirect accesses,
dispatch, state, and output placement. A generated kernel may accept a runtime
logical extent through its tensor view or named scalar ABI, but its physical
tiles, allocations, grid, pipeline, and engine assignment are fixed by the
serialized Fusebox solution.

No compiler-integrated AutoFuse pass is required for this path. The generated
PyPTO source will contain the selected fusion boundaries, grid, propagated
regions, topological order, physical tiles, loops, pipelines, lifetimes,
cross-core FIFOs, and valid-shape handling. PyPTO parses, verifies, lowers, and
executes that explicit program without rerunning the Fusebox planner.

Consequently, generated functions must not carry `auto_fuse` or `auto_tile`
attributes. Those attributes belong to the earlier compiler-integrated
prototype. Adding either attribute to already planned source would ask a second
planner to reinterpret the selected partition and would break the
solution-to-emission fidelity contract.

The frontend and backend may be Python components because `torch.export` and FX
are Python APIs, but they belong to the PTO-Fusebox repository. PyPTO remains a
generated-language target rather than a dependency of the scheduling model.

## Components

### Torch/Hugging Face capture adapter

Capture `torch.nn.Module` programs through `torch.export` or FX. Do not attempt
to translate arbitrary Python source. The adapter converts the captured graph
to a frontend-neutral Fusebox problem containing:

- tensor operations and exact input order;
- dtypes, logical shapes, layouts, views, aliases, and mutations;
- symbolic shape constraints and representative shape information;
- outputs and explicit state updates;
- opaque nodes for unsupported control flow, custom operations, and
  data-dependent access.

Hugging Face code is normally a PyTorch module plus configuration and weights.
It describes model semantics, but deployment choices such as KV-cache layout,
quantization, and runtime dependencies may live outside the module. Such
choices must be present in the exported graph or supplied as explicit frontend
configuration; Fusebox must not guess them.

### Scheduler core

The imported problem feeds the existing unified subgraph planner. For each
connected candidate group, Fusebox:

1. chooses fusion or cut boundaries;
2. assigns one common grid to the selected group;
3. propagates output regions backwards through the tensor DAG;
4. selects a legal topological and pebbling order;
5. accounts for boundary and produced-value lifetimes;
6. checks UB, L1, L0, and cross-core FIFO feasibility;
7. prices compute, transfers, drains, and implementable overlap; and
8. returns a complete analytic schedule, plus the typed replay contract for
   schedule families implemented by the source backend.

The frontend passes each maximal supported region as one complete solver input
DAG. This does not require the region to become one kernel: AutoFuse partition
search chooses any number of feasible groups that cover it. The Ascend 910B
profile enables the restricted buildable mixed model by default; unsupported
mixed topologies remain candidate cuts rather than making the whole region an
AutoTile-style all-or-nothing request.

Fusebox does not replace the runtime scheduler. It forms good kernels and
preserves their dependency graph. The PyPTO runtime remains responsible for
launching ready kernels and overlapping independent AIC and AIV work.

### PyPTO source backend

The backend deterministically serializes the selected solution as readable
PyPTO DSL. The installed homogeneous slice emits materialized/pointwise vector,
`softmax_flash.v1`, one-reduction folded/spanning vector streams, uniform
non-split cube DAG schedules, or single-sink split-K as explicit `pl.spmd`
tasks, with
static physical and valid shapes, explicit pipelines, and GM boundaries. It
never expands a grid into a host-side loop of single-block submissions.
Several homogeneous steps are composed in one orchestration program using
dependency-linked GM cut values; this preserves the solver partition without
claiming cross-kernel fast-memory retention.
Single-sink split-K currently requires a one-step solution because its internal
two-task dependency is not yet representable by the multi-step composer.
ABI inputs use an `arg_` namespace, so captured names cannot shadow generated
names.
Materialized/pointwise schedules execute `body.ops`; online softmax consumes the
typed stats/apply loops, frames, workspaces, carry state, and substitutions.
Selection is independent of program, module, model, or shape names.

Cube source replays the selected request order, outer spatial and K-window
schedule, L1 resident values, retained panels, and drains. In a split-K cube
DAG, only the unique sink is divided; every share replays upstream matmuls in
the selected order and accumulates the sink's disjoint K interval. Multiple
split sinks, ambiguous accumulators, and multi-output split groups fail closed.
The general non-split path then lets PyPTO's
`AutoTileMatmulL0` choose child-L0 `(m,n,k)`, stationarity, and buffer depths.
The ordinary DSL cannot pin that complete design point. Exact L0 replay is an
optional future extension via a PyPTO schedule directive or explicit low-level
tile loops; current source makes no exact child-L0 performance claim.

Mixed source validates every logical transfer/FIFO descriptor in solver order.
Generated source targets PyPTO's generic explicit-pipe API: it emits
`pl.split(pl.SplitMode.UP_DOWN)` plus one generic
`pl.cross_core_pipe(...)` entry per logical transfer. The entries preserve
direction, physical frame, slot geometry, pipe ID, and protocol bundle, while
ordinary tensor def-use expresses each crossing. PyPTO's `ExpandMixedKernel` and
`SkewCrossCorePipeline` own the AIC/AIV split and the physical
initialize/push/pop/free lowering; the external emitter does not hand-code
either core program. Repeated loop/tail realizations reuse the same ordered
pipe protocol rather than creating extra logical FIFOs. Source readiness
charges the serialized cube-stage L1 peak together
with every V2C consumer ring, and the vector-stage peak together with every
C2V consumer ring; neither direction can hide its physical FIFO reservation.
No attention or SwiGLU recognizer is involved in this transport contract.

### Mixed-source silicon status

The historical fork-only explicit-descriptor path was silicon-closed for
one-way `C -> V`, generic `C -> V -> C`, dense `C,C -> V -> C`, and sequential
`C -> V -> C -> V`. That evidence established the model and generic replay but
is not treated as closure for upstream PyPTO `main`. The main-compatible source
surface is checked independently because PyPTO now owns automatic physical
pipe construction. The current upstream-main host matrix lowers every mixed
family except dense `C,C -> V -> C` through real PTOAS. The remaining case
exposes a general nested-accumulator join defect in PyPTO's memory-reuse pass,
not a missing Fusebox transport API. With the generic PyPTO regression and fix
applied, the complete 38-case opt-in source-integration surface passes; a
current-main silicon campaign remains pending.

Generic one-way `V -> C` passes the model and typed-plan contracts for both a
produced LHS (`M`-owned FIFO) and produced RHS (`N`-owned FIFO). Online
softmax-to-PV serializes chunked publication and matching
cube K windows. A complete square value may serve both sink operands from one
FIFO-owned L1 ring; partitioned dual-role values are rejected until the plan
defines replication and per-role FIFO ownership. Lowering and device closure
are assessed specifically against the pinned upstream-main revision.

Two ABI/numerical findings must remain distinct from that transport result:

- The old generic-attention "lane-0 residual" is retracted. The campaign
  harness bound Torch inputs by position even though the emitted signature had
  deterministically reordered `query` and `key`. `EmittedPyPTOSource` now
  carries `input_value_ids` in signature order, and device execution binds
  tensors by normalized value ID.
- Dense SwiGLU generated and independently hand-written PyPTO outputs are
  bit-identical. In the closure fixture the direct Torch comparison has zero
  `2e-2` tolerance misses and the ordered ULP histogram is `{0: 8191, 1: 1}`.
  BF16 vector-stage narrowing therefore remains a reported numerical-oracle
  caveat, not a mixed FIFO defect; the tolerance is not silently weakened.

Analytic support is broader than the renderer. Welford/multi-stat P4 and mixed
plans outside the admitted stage patterns remain valid solver results but
are not source-ready. P4
descriptors preserve named roles; each new recipe must version its carry and
publication semantics. Future plan classes can add those contracts and
cross-core transport without changing the ownership boundary.

The backend must not redo planning. Every emitted loop, lifetime, transfer, and
FIFO must be traceable to the solution descriptor. It should publish the
schedule report and pseudocode beside the source so users can inspect the
decision.

## Native orchestration and static scheduling boundary

PTO-Fusebox owns **static physical schedules**, not serving orchestration. This
matches PyPTO's existing compilation model: tensor extents, loop bounds, work
counts, offsets, and `valid_shape` may be runtime values, but physical
UB/L1/L0 tiles, allocations, pipeline depths, and engine assignments remain
compile-time decisions.

The boundary is therefore not "static shape versus dynamic shape." It is:

```text
native PyPTO model / serving orchestration
  dynamic extents, chunk and window loops, cache metadata, rank dispatch,
  opaque or external operations, data-dependent routing and sampling
                             |
                             v
generated PTO-Fusebox kernel call
  stable named ABI, one connected affine tensor DAG, one selected fusion
  partition, static physical tiles and memories, fixed pipeline protocols,
  runtime logical valid extents over that fixed physical frame
                             |
                             v
ordinary PyPTO compiler and runtime
  verify and lower the explicit DSL, submit ready tasks, and overlap independent
  AIC and AIV work according to the preserved dependency graph
```

Fusebox does not globally schedule all ready kernels or choose how unrelated
kernel grids co-reside on the device. It forms and prices individual fused
groups and preserves their dependencies; the PyPTO runtime scheduler decides
when ready groups execute.

The source-level boundary is not always an existing Python function boundary.
PyPTO-lib `@pl.jit` functions commonly contain orchestration statements around
several static `pl.at` or `pl.spmd` regions. Integration identifies those
static call sites, gives each one an explicit named ABI, and replaces only its
implementation with generated source. The surrounding PyPTO caller remains
explicit and authoritative.

Classify a model fragment as follows:

| Fragment property | Initial Fusebox treatment |
| --- | --- |
| Dense pointwise, reduction, or matmul DAG with static physical geometry | Plan and emit with the vector, cube, or mixed model. |
| Runtime outer/free extent changes only region count, offsets, or the final valid tail | Native orchestration calls one static physical plan with a runtime logical `valid_shape`. |
| A bounded range needs materially different physical tiles or pipelines | Native orchestration may select among explicitly generated static kernels; Fusebox plans each member independently. |
| `pl.jit.extern` or another independently implemented device operation | Preserve as an opaque call and cut the Fusebox region at its tensor interface. |
| Block tables, slot mappings, TopK indices, or routing values select addresses or work | Preserve as a data-dependent opaque boundary until the access and cost semantics are modeled. |
| Distributed rank loops, cache-pool management, recurrent serving state, or token-generation control | Keep in orchestration. |

An opaque boundary does not imply that the operation runs on the host. Paged
cache gathers, TopK, and routing can remain device kernels; they are opaque
because their access graph or work cardinality is data-dependent and is not
represented by the current dense affine model.

This boundary lets Fusebox cover substantial static portions of a model without
synthesizing the model program. It also prevents dynamic metadata from
contaminating local tile selection: cache-pool capacity may be dynamic while
the compute performed for each fixed-size cache block remains statically tiled.

## Dynamic-shape baseline

This section classifies dynamic behavior that native PyPTO orchestration may
place around generated kernels. It is not a Fusebox orchestration roadmap.
**No dynamic-shape class below is currently admitted by the Torch-to-solver
path.** The current implementation retains symbolic metadata and then declines
schedule-defining symbolic regions.

PyPTO supports extent-polymorphic programs, not runtime-sized hardware tiles:

- tensor dimensions may be runtime values;
- runtime dimensions may control offsets, loop bounds, work counts, and
  `valid_shape`;
- one compiled artifact may accept different values of declared dynamic
  dimensions;
- physical UB/L1/L0 tile dimensions and allocations remain compile-time
  constants; and
- dynamic outputs are supplied explicitly rather than automatically allocated
  by the current runtime helper.

The callable-kernel ABI must preserve this contract. Fusebox receives only the
static physical work; native orchestration owns the runtime logical extents.

## Native-orchestration case: dynamic extent, static physical chunk

This is the first integration case, but it does not make the Fusebox solver
dynamic:

```text
runtime:       M, number of regions, offsets, final valid extent
compile time:  CHUNK, physical tiles, grid policy, pipeline depth, allocation
```

Native PyPTO orchestration fixes `CHUNK` and calls a generated kernel. Fusebox
plans the per-chunk DAG as a static problem. Runtime `M` changes only the number
of calls and the logical size of the last region:

```python
m = pl.tensor.dim(x, 0)
for m0 in pl.range(0, m, CHUNK):
    valid_m = pl.min(CHUNK, m - m0)
    x_tile = pl.slice(x, [CHUNK, D], [m0, 0], valid_shape=[valid_m, D])
    # Statically planned DAG over physical [CHUNK, D].
```

The static kernel contract requires:

- the dynamic dimension is an outer/free axis whose chunks are independent;
- all physical tile extents and memory footprints are static;
- region propagation is affine and preserves the chunk boundary;
- the tail fits the same physical frame through `valid_shape`; and
- no data-dependent address or branch changes the per-chunk DAG.

Capacity is checked for a full chunk. Fusebox does not price the surrounding
runtime loop or choose its trip count. If native orchestration needs several
physical variants, each variant is a separate concrete Fusebox input and the
orchestration owns selection among their stable ABIs.

Existing examples include dynamic-token RMSNorm, `hc_head` with a clamped final
token tile, Qwen RMSNorm/LM-head with a dynamically trimmed cube result, and
`hc_post` with a runtime-derived SPMD work count.

## Native-orchestration dynamic classes

The following patterns exist in PyPTO and PyPTO-lib. They remain native
orchestration responsibilities rather than Fusebox scheduling milestones.

### Type 2: bounded active prefix

A statically bounded tensor has a runtime active length. The program processes
the prefix and may skip or deterministically fill the inactive suffix. DeepSeek
`hc_post_prefill` follows this form. Native orchestration owns the
active/inactive work; any invoked Fusebox kernel retains a static frame.

### Type 3: ragged packed batch

Each request has its own runtime chunk length and tail. Total work is a sum over
per-request extents rather than `ceildiv` of one global extent. Qwen prefill
uses `chunk_lens`, `chunk_offsets`, and per-request `valid_shape` this way.
Fusebox plans only the concrete static kernel called for each chunk.

### Type 4: dynamic recurrence length

A runtime reduction or stream length controls how often a fixed physical chunk
updates loop-carried state. Examples are paged attention's online-softmax
tuple, Welford statistics, and a persistent cube accumulator. Native
orchestration owns the recurrence state and call sequence. A Fusebox region may
implement one statically shaped initialization, update, or finalization call,
but does not merge the dynamic recurrence into its local schedule.

### Type 5: runtime configuration selecting a static physical family

Some logical dimensions, such as attention head size or cache block size,
change the required physical kernel. Native orchestration may dispatch among a
bounded family whose members Fusebox planned independently. Fusebox does not
emit that dispatcher or model it as a runtime-sized tile.

### Dynamic GM capacity and metadata

KV-cache rows, block-table sizes, and output extents may be runtime dimensions
while local work remains statically tiled. They remain in the orchestration
ABI and affect offsets, work counts, tails, or recurrence there; they do not
enter the Fusebox problem unless concretized as one physical kernel shape.

### Indirect and data-dependent access

Block tables, slot mappings, gathers, and TopK indices choose addresses or
active elements from tensor values. This is not merely dynamic shape: it changes
the access graph or result cardinality. Initially these operations remain
explicit opaque boundaries. Fusebox must not infer affine region propagation
through them until their semantics and costs are represented directly.

Dynamic physical tile extents and dynamic rank remain unsupported. Hardware
allocations need static sizes, and the PyPTO compiler correctly rejects a
dynamic physical tile that reaches allocation or tile-flattening.

## Implementation sequence

The homogeneous vector/cube matrix, single-sink cube-DAG split-K, explicit
mixed pipes, and the first sequential CVCV source contract are silicon-closed.
Generic one-way V2C is source-ready for materialized LHS/RHS producers, online
softmax-to-PV K streaming, and one complete-square value used in both matmul
roles. The last two paths have real PyPTO lowering coverage but still require
focused silicon closure.
The remaining source capabilities are ordered by contract complexity:

1. Silicon-close online softmax-to-PV and complete-square dual-role V2C replay.
   Keep partitioned dual-role cases rejected until an explicit replication
   policy exists.
2. Replace the fixed four-stage sequential replay with a generic renderer over
   a serialized linear alternating engine chain. Keep deeper chains sequential
   until the model explicitly grants a realizable overlap schedule.
3. Allow a mixed step to participate in dependency-linked multi-step source,
   preserving GM cuts, task order, and the mixed step's internal SPMD/pipe
   contract. PyPTO already supports dependency-linked `pl.spmd` tasks; this is
   a Fusebox task-bundle and ownership-contract gap, not a missing PyPTO
   execution primitive.
4. Add branched mixed graphs only after defining fan-out replication versus
   retention, per-consumer FIFO ownership, lifetimes, and backpressure in the
   serialized plan. Continue failing closed until that contract exists.
5. Preserve unsupported nodes as explicit graph cuts and verify every value
   crossing those boundaries.
6. Ground and compose the new callable ABI. A single homogeneous vector step
   can now receive a runtime logical row extent over one fixed planned physical
   frame; cube, mixed, multi-step, and non-row variation fail closed. Next,
   compare the generated Qwen RMSNorm/LM-head, attention, and dense-SwiGLU
   callables with same-shape PyPTO-lib-derived controls and compose independently
   emitted regions under native PyPTO orchestration. Keep Types 2-5 in
   orchestration and outside Fusebox planning.

## PyPTO-lib validation targets

The following experiments ground the standalone pipeline in current Qwen and
DeepSeek programs. They are ordered so that each experiment adds one new
contract instead of combining every missing feature at once. Reduced fixtures
must preserve the production contraction dimensions, operation order, dtypes,
and boundary semantics; model or function names must never affect planning.

1. **Qwen RMSNorm and LM head as separate regions.** The reduced
   `qwen3_rms_norm_chunk` and `qwen3_lm_head_chunk` fixtures now capture, solve,
   and emit independent callable vector and cube regions. The RMSNorm callable
   carries the first runtime-valid-row ABI. The LM-head callable preserves the
   production `[VOCAB, HIDDEN]` weight layout through a zero-copy
   `pl.tile.transpose_view`. Silicon comparison with
   `models/qwen3_14b/rms_lm_head.py` remains outstanding.
2. **Qwen RMSNorm to LM head.** Solve the connected `V -> C` graph, emit the
   selected boundary, and compare it with `models/qwen3_14b/rms_lm_head.py`.
   This is the smallest model-derived one-way mixed-source target.
3. **DeepSeek V4-Flash MTP projection at fixed token extents.** Cover the two
   normalizations, activation quantization, two projections, and their sum
   without recognizing an MTP pattern. Start with concrete decode and prefill
   token extents; later use the same case for Type-1 dynamic-token support.
4. **Qwen layer with paged attention kept opaque.** Preserve the CANN
   `pl.jit.extern` attention call and its cache metadata interface. Schedule
   the QKV preprocessing and MLP sides as independent Fusebox regions and
   verify that generated source preserves every crossing value and dependency.
5. **Static QK to softmax to PV.** Capture the ordinary matmul, reduction, and
   pointwise DAG and exercise generic `C -> V -> C` planning and source
   emission. The experiment must not use an attention recognizer and initially
   excludes paged or sparse cache addressing.
   `examples/torch_frontend/static_mixed.py` exercises this complete capture ->
   solve -> callable-source path together with the dense SwiGLU core from the
   next target, using reduced but legal static dimensions. Both are compiled
   as imports inside independent native PyPTO orchestration by the opt-in
   integration suite.
6. **DeepSeek MoE cut at data-dependent routing.** Schedule the dense
   normalization/router prefix and the bounded expert-local
   matmul/SwiGLU/matmul computation separately. Keep TopK, token-to-expert
   dispatch, variable expert counts, and distributed exchange explicit and
   opaque.

For every target, record four independent outcomes:

- normalized Torch DAG fidelity and explicit opaque boundaries;
- analytic partition, tile, memory, and latency result;
- exact solution-to-PyPTO source replay and generated-kernel compilation; and
- silicon correctness and performance against the corresponding hand-written
  PyPTO-lib implementation when the target platform is calibrated.

Performance comparisons use identical reduced shapes, dtypes, layouts,
operation order, narrowing points, runtime/compiler revisions, and
input/output residency. The hand-written control is frozen before silicon and
must not copy generated source or solver-selected tile sizes. Generated and
control arms run in a balanced paired order on the same devices. Full
production functions with different shapes or a larger fused scope are
reported separately and are not used to form a speed ratio. Consequently,
reduced-region results characterize schedule quality for matching static work;
they are not end-to-end model performance claims.

Qwen3-14B and DeepSeek V4-Flash are the initial A2/A3 performance references.
DeepSeek V4-Pro targets A5 and is suitable for structural capture tests, but no
performance conclusion is valid until the A5 model is independently calibrated
and device-verified.

## Non-goals

- translating arbitrary Python control flow;
- generating or reproducing native PyPTO orchestration from Torch;
- recognizing model names or hard-coding FlashAttention or SwiGLU algorithms;
- choosing quantization precision for the model author;
- replacing the PyPTO compiler, verifier, PTOAS, or runtime scheduler;
- silently approximating unsupported aliases, mutations, views, indirect
  accesses, or data-dependent control.
