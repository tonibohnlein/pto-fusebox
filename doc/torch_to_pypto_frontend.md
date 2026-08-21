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
being approximated. Static specialization families, runtime dispatch, and
dynamic physical tiles are not implemented.

The PyPTO DSL source backend validates a solved region as a typed schedule and
emits an ordinary `@pl.program` with its grids, logical ownership, physical
frames, loops, pipeline stages, operation order, and GM traffic. Each
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

Welford/multi-stat vector plans, singleton-column normalization, nonuniform cube
spatial partitions, split cube DAGs, and mixed cross-core schedules fail closed.
Single-sink split-K is emitted through the selected dependency-linked PyPTO task
protocol.
Dynamic-shape classes are retained but declined when they affect solver
geometry.

## Python API

Install the optional Torch frontend dependencies and import the public package:

```bash
python -m pip install -e ".[torch]"
```

```python
from pto_fusebox import emit_pypto_region, export_and_normalize, solve_graph

graph = export_and_normalize(module, example_args, dynamic_shapes=constraints)
result = solve_graph(
    graph,
    target="ascend910b",
    solver_binary="build/mlsys_mixed",
    solver_workers=2,
)
source = emit_pypto_region(graph, result.regions[0], program_name="FusedRegion")
print(source.source)
```

The core entry points are:

- `export_and_normalize(module, args, ...) -> NormalizedGraph`;
- `normalize_exported(program) -> NormalizedGraph`;
- `extract_solver_regions(graph, target) -> list[SolverRegion]`;
- `solve_graph(graph, target=..., solver_binary=...) -> SolveResult`;
- `scheduled_region(result) -> ScheduledRegion`;
- `can_emit_region(graph, result) -> bool`; and
- `emit_pypto_region(graph, result, ...) -> EmittedPyPTOSource`.

`normalize_exported` is authoritative; `export_and_normalize` is a convenience
wrapper around `torch.export.export`. `solve_graph` never builds the solver. A
caller must provide an executable or set `PTO_FUSEBOX_SOLVER`.
`scheduled_region` rejects incomplete or internally inconsistent solution
arrays before emission. `can_emit_region` and `emit_pypto_region` build the same
typed emission context and run the same graph-aware renderer validation; the
readiness query is not a weaker schedule-family approximation. The emitter
accepts one or more selected homogeneous steps and raises
`SourceEmissionError` for every unimplemented algorithm or unsafe cut edge.
`EmittedPyPTOSource.kinds` preserves the ordered engine kind of every emitted
step; its compatibility `kind` property is populated only when all steps use
the same engine kind.

### Source-backend structure and validation

The source backend is schedule-family-driven, not model- or pattern-driven.
It selects a vector or cube emitter from the typed solver step, replays the
solver's operation order and serialized geometry, and dispatches individual
operations by normalized operator kind. Names such as softmax, RMSNorm,
attention, or a source `nn.Module` class are never emission inputs. The cube
family is driven by the typed request DAG rather than by example or model
names. It replays the selected topological order, propagated regions, outer-K
windows, local produced-value lifetimes, retained boundary panels, and drains
for non-split plans.

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
- `pto_fusebox.solution.v4`: the C++ schedule response. Cross-kernel values are
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

The Python boundary decodes `problem.v1` into `LoweredRegion` and `solution.v4`
into immutable `ScheduledRegion`/`KernelStep` types before rendering. The
lowered half owns region inputs, outputs, and output-allocation lineage; the
scheduled half owns execution. Together with the normalized graph they form a
single `EmissionContext`, so neither readiness nor rendering reconstructs an
ABI from a partial view. Unknown, missing, or internally inconsistent fields
fail closed. The source package is split by responsibility: `source/api.py`
owns context construction and dispatch, `source/common.py` owns typed
ABI/naming/partition mechanics, and `source/vector.py` and `source/cube.py`
replay their respective homogeneous plans.

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

The standalone target admits the complete analytic schedule surface. It does
not restrict partition search to schedules supported by the historical
in-compiler AutoFuse emitter: PTO-Fusebox will ultimately generate tensor/tile
PyPTO source from its own selected schedule. Split cube DAGs, multi-reduction
streaming, non-uniform cube-DAG grids, one-way `V -> C`, complete
`C -> V -> C`, and deeper serial mixed topologies therefore remain eligible
even when no legacy emitter path exists. Solution metadata records the stages,
directional transfers, and protocol for every analytic mixed plan. Plans with
a complete stage-local geometry, vector stream, cube-window, and FIFO contract
also set their internal `source_codegen_ready=true` contract-completeness bit;
that mixed-plan field does not by itself claim support in the installed Python
source backend. Broader analytic winners remain valid research results but are
not presented as source-emittable. `regions_solved` therefore reports analytic
solver success separately from `whole_graph_codegen_ready`, which additionally
requires no opaque graph boundaries and a successful exact
`can_emit_region(graph, region)` check for every selected step.

## Goal and ownership

PTO-Fusebox should own the complete source-to-source scheduling path:

```text
PyTorch/Hugging Face module + configuration + shape constraints
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
ordinary PyPTO compiler and runtime
```

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
schedule, L1 resident values, retained panels, and drains, then lets PyPTO's
`AutoTileMatmulL0` choose child-L0 `(m,n,k)`, stationarity, and buffer depths.
The ordinary DSL cannot pin that complete design point. Exact L0 replay is an
optional future extension via a PyPTO schedule directive or explicit low-level
tile loops; current source makes no exact child-L0 performance claim.

Analytic support is broader than the renderer. Welford/multi-stat P4, split
cube DAGs, and mixed plans remain valid solver results but are not source-ready. P4
descriptors preserve named roles; each new recipe must version its carry and
publication semantics. Future plan classes can add those contracts and
cross-core transport without changing the ownership boundary.

The backend must not redo planning. Every emitted loop, lifetime, transfer, and
FIFO must be traceable to the solution descriptor. It should publish the
schedule report and pseudocode beside the source so users can inspect the
decision.

## Static scheduling and dynamic orchestration boundary

PTO-Fusebox initially owns **static physical schedules**, not complete serving
orchestration. This matches PyPTO's existing compilation model: tensor extents,
loop bounds, work counts, offsets, and `valid_shape` may be runtime values, but
physical UB/L1/L0 tiles, allocations, pipeline depths, and engine assignments
remain compile-time decisions.

The boundary is therefore not "static shape versus dynamic shape." It is:

```text
model / serving orchestration
  dynamic extents, chunk and window loops, cache metadata, rank dispatch,
  opaque or external operations, data-dependent routing and sampling
                             |
                             v
PTO-Fusebox scheduled regions
  one connected affine tensor DAG, one selected fusion partition per region,
  static physical tiles and memories, fixed pipeline protocols, runtime valid
  extents and work counts where one plan is valid for the declared range
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

The source-level boundary is not always a Python function boundary. Existing
PyPTO-lib `@pl.jit` functions commonly contain orchestration statements around
several static `pl.at` or `pl.spmd` regions. A future whole-model frontend must
extract the static regions while retaining the surrounding runtime program.
The first implementation may instead emit scheduled inline functions and leave
their callers explicit.

Classify a model fragment as follows:

| Fragment property | Initial Fusebox treatment |
| --- | --- |
| Dense pointwise, reduction, or matmul DAG with static physical geometry | Plan and emit with the vector, cube, or mixed model. |
| Runtime outer/free extent changes only region count, offsets, or the final valid tail | Use one static physical plan with runtime logical extents (planned Type 1 below). |
| A bounded range needs materially different physical tiles or pipelines | Compile a small static family and dispatch outside the scheduled region; defer until one-plan Type 1 works. |
| `pl.jit.extern` or another independently implemented device operation | Preserve as an opaque call and cut the Fusebox region at its tensor interface. |
| Block tables, slot mappings, TopK indices, or routing values select addresses or work | Preserve as a data-dependent opaque boundary until the access and cost semantics are modeled. |
| Distributed rank loops, cache-pool management, recurrent serving state, or token-generation control | Keep in orchestration. |

An opaque boundary does not imply that the operation runs on the host. Paged
cache gathers, TopK, and routing can remain device kernels; they are opaque
because their access graph or work cardinality is data-dependent and is not
represented by the current dense affine model.

This boundary lets Fusebox cover substantial static portions of a model before
it can synthesize the entire model program. It also prevents dynamic metadata
from contaminating local tile selection: cache-pool capacity may be dynamic
while the compute performed for each fixed-size cache block remains statically
tiled.

## Dynamic-shape baseline

This section records future design constraints. **No dynamic-shape class below
is currently admitted by the Torch-to-solver path.** The current implementation
retains symbolic metadata and then declines schedule-defining symbolic regions.

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

Fusebox should preserve this contract. A dynamic problem is schedulable only
when it can be expressed as static physical work plus runtime logical extents.

## Planned Type 1: dynamic independent extent, static physical chunk

This is the first dynamic-shape class to support:

```text
runtime:       M, number of regions, offsets, final valid extent
compile time:  CHUNK, physical tiles, grid policy, pipeline depth, allocation
```

For fixed `CHUNK`, Fusebox plans the per-chunk DAG as a static problem. Runtime
`M` changes only the number of independent regions and the logical size of the
last region:

```python
m = pl.tensor.dim(x, 0)
for m0 in pl.range(0, m, CHUNK):
    valid_m = pl.min(CHUNK, m - m0)
    x_tile = pl.slice(x, [CHUNK, D], [m0, 0], valid_shape=[valid_m, D])
    # Statically planned DAG over physical [CHUNK, D].
```

Initial admission requires:

- the dynamic dimension is an outer/free axis whose chunks are independent;
- all physical tile extents and memory footprints are static;
- region propagation is affine and preserves the chunk boundary;
- the tail fits the same physical frame through `valid_shape`; and
- no data-dependent address or branch changes the per-chunk DAG.

Capacity is checked for a full chunk. Runtime work is
`ceildiv(M, CHUNK)` regions plus the normal clamped-tail accounting. The planner
must apply the existing wave model rather than serially multiplying a per-region
latency when regions execute concurrently.

The programmer may fix `CHUNK`, or Fusebox may enumerate a small static set for
a representative extent, bounded range, or supplied shape distribution. The
generated program remains extent-polymorphic. Emit multiple variants only when
different static tiles, grids, or pipelines are materially better in different
shape regimes.

Existing examples include dynamic-token RMSNorm, `hc_head` with a clamped final
token tile, Qwen RMSNorm/LM-head with a dynamically trimmed cube result, and
`hc_post` with a runtime-derived SPMD work count.

## Deferred dynamic classes

The following patterns exist in PyPTO and pypto-lib, but are not part of the
first frontend milestone.

### Type 2: bounded active prefix

A statically bounded tensor has a runtime active length. The program processes
the prefix and may skip or deterministically fill the inactive suffix. DeepSeek
`hc_post_prefill` follows this form. It needs separate active/inactive region
accounting, but not runtime-sized physical tiles.

### Type 3: ragged packed batch

Each request has its own runtime chunk length and tail. Total work is a sum over
per-request extents rather than `ceildiv` of one global extent. Qwen prefill
uses `chunk_lens`, `chunk_offsets`, and per-request `valid_shape` this way. A
future model must represent the distribution of ragged work across waves.

### Type 4: dynamic recurrence length

A runtime reduction or stream length controls how often a fixed physical chunk
updates loop-carried state. Examples are paged attention's online-softmax
tuple, Welford statistics, and a persistent cube accumulator. Supporting this
requires an explicit recurrence state, initialization, update, finalization,
memory-lifetime, and cost contract. It cannot be treated as independent Type-1
regions.

### Type 5: runtime configuration selecting a static physical family

Some logical dimensions, such as attention head size or cache block size,
change the required physical kernel. Fusebox may plan a bounded family of
static variants and emit a small host dispatcher. This is specialization, not
a runtime-sized tile, and should be added only when one physical plan is not
competitive over the required range.

### Dynamic GM capacity and metadata

KV-cache rows, block-table sizes, and output extents may be runtime dimensions
while local work remains statically tiled. These dimensions can remain symbolic
in the boundary schema when they do not change the local schedule. They become
one of the types above when they affect work count, tails, or recurrence length.

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

1. Silicon-close the expanded homogeneous source matrix: materialized and
   streamed vector schedules, non-split cube DAGs, outer-K replay, retained
   panels, and dependency-linked homogeneous cuts.
2. Preserve the reproduced
   [closed PyPTO PR #2335](https://github.com/hw-native-sys/pypto/pull/2335)
   vector behavior while extending source replay; do not rediscover those
   schedules in the emitter.
3. Extend the single-sink split-K source contract to cube DAGs with resident
   operands and per-share outer-K windows while rejecting ambiguous multi-root
   merges.
4. Implement the generic mixed source backend from one round trip while continuing
   to reject unsupported multi-round-trip groups before emission.
5. Preserve unsupported nodes as explicit graph cuts and verify every value
   crossing those boundaries.
6. Add Type-1 dynamic outer chunks with a static physical tile and runtime
   `valid_shape`.
7. Defer Types 2-5 until the static frontend/backend and Type 1 are correct on
   device.

## PyPTO-lib validation targets

The following experiments ground the standalone pipeline in current Qwen and
DeepSeek programs. They are ordered so that each experiment adds one new
contract instead of combining every missing feature at once. Reduced fixtures
must preserve the production contraction dimensions, operation order, dtypes,
and boundary semantics; model or function names must never affect planning.

1. **Qwen RMSNorm and LM head as separate regions.** Capture, solve, emit, and
   compare the vector RMSNorm and cube LM-head projection independently. This
   is the first direct application of the silicon-closed homogeneous source
   slices to a model component.
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

Qwen3-14B and DeepSeek V4-Flash are the initial A2/A3 performance references.
DeepSeek V4-Pro targets A5 and is suitable for structural capture tests, but no
performance conclusion is valid until the A5 model is independently calibrated
and device-verified.

## Non-goals

- translating arbitrary Python control flow;
- recognizing model names or hard-coding FlashAttention or SwiGLU algorithms;
- choosing quantization precision for the model author;
- replacing the PyPTO compiler, verifier, PTOAS, or runtime scheduler;
- silently approximating unsupported aliases, mutations, views, indirect
  accesses, or data-dependent control.
