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

The first PyPTO DSL source slice is also implemented. A solved homogeneous
region can be validated as a typed schedule and emitted as an ordinary
`@pl.program` containing the selected grid, balanced region partition,
physical vector frame, strip/K-window loops, pipeline stages, operations, and
loads/stores. The initial closed set is materialized/pointwise vector replay,
the versioned two-pass online-softmax schedule, and one spatial,
output-stationary cube matmul. Unsupported schedules fail closed; the backend
does not approximate them or ask PyPTO to plan them again.

Other streamed vector phases, split reductions, split-K, retained panels,
multi-matmul DAGs, mixed schedules, multiple selected kernel steps, and whole
graph orchestration remain source-backend milestones. Dynamic-shape classes
are preserved in the normalized graph but remain unschedulable when they
affect solver geometry. They are catalogued below and deliberately deferred.

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
currently accepts one selected homogeneous step and raises
`SourceEmissionError` for every unimplemented algorithm.

### Source-backend structure and validation

The source backend is schedule-family-driven, not model- or pattern-driven.
It selects a vector or cube emitter from the typed solver step, replays the
solver's operation order and serialized geometry, and dispatches individual
operations by normalized operator kind. Names such as softmax, RMSNorm,
attention, or a source `nn.Module` class are never emission inputs. The initial
cube family is intentionally limited to one spatial output-stationary matmul;
that is an explicit fail-closed schedule-family boundary, not a matmul-example
recognizer.

The replay structure follows the earlier PyPTO fusion-scheduler prototype:
one solver-owned grid, propagated regions, planned physical frames and
partitions, lifetime-respecting topological replay, and pipelined strip or K
windows. The implementations are not shared code: the prototype builds PyPTO
IR in C++, while this repository emits readable PyPTO DSL source. The
serialized schedule is the common contract.

Reference checks use three levels:

- PyPTO-lib programs establish the ordinary DSL structure (`pl.parallel` or
  `pl.spmd`, `pl.pipeline`, row reductions/expands, and
  `pl.matmul` followed by `pl.matmul_acc`), including
  `models/deepseek_v4_flash_dspark/rmsnorm.py` and
  `models/qwen3_32b/decode_4d.py`;
- PTO-ISA and PTOAS examples establish the expected lowered data path and tile
  constraints, including PTO-ISA's A2/A3 `tmatmul_kernel.cpp` and PTOAS's
  `matmul_static_singlecore.pto` and `trowexpandsub_v0_roundtrip.pto`; and
- the opt-in `test_source_pypto_integration.py` gate parses the generated DSL
  with an independently selected PyPTO checkout and compiles it through PTOAS.

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
- `pto_fusebox.solution.v2`: the C++ schedule response.

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
topological order, sequential tile counts, retained tensors, and latency.
The common launch tile is the optimizer's selected configuration and a
diagnostic summary; the nested family plan is authoritative wherever lowering
derives different replay frames, as it does for vector reductions.
Vector plans additionally record phase operation order, input lifetimes,
logical/physical tensor frames, reduction workspaces, and exact loop bounds.
Cube plans record propagated regions and axis bindings, execution order,
resident-boundary lifetimes, K/L0 loops, retained panels, drains, and split
policy. These fields are solver output, not choices rediscovered by Python
emission.

The Python boundary decodes `problem.v1` into `LoweredRegion` and `solution.v2`
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
PyPTO source from its own selected schedule. Analytic split-K, multi-reduction
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

The backend deterministically serializes the selected solution descriptor as
readable PyPTO DSL. The installed homogeneous slice emits one materialized or
pointwise vector step, the `softmax_flash.v1` two-pass online schedule, or one
uniform spatial cube matmul, using `pl.parallel`, static physical tile shapes,
runtime `valid_shape`, `pl.range`/`pl.pipeline`, and explicit GM boundaries.
ABI inputs use an `arg_` namespace, so captured names cannot shadow `pl` or
generated schedule locals.
Materialized/pointwise schedules execute `body.ops` directly and require all
other phases to be empty. Online softmax consumes the typed stats/apply loops,
frames, workspaces, carry-state names, and reduction-output substitutions. Its
selection is recipe-driven and independent of program, module, model, or shape
names.

Cube source replays the outer spatial and K-window schedule, then lets PyPTO's
`AutoTileMatmulL0` choose child-L0 `(m,n,k)`, stationarity, and buffer depths.
The ordinary DSL cannot pin that complete design point. Exact L0 replay is an
optional future extension via a PyPTO schedule directive or explicit low-level
tile loops; current source makes no exact child-L0 performance claim.

Analytic support is deliberately broader than this installed renderer.
General streamed reductions, Welford/multi-stat P4 algorithms, split
reductions, retained cube panels, multi-matmul, split-K, multi-step, and mixed
plans remain valid solver results but are not source-ready. P4 descriptors
preserve named substitution roles rather than reconstructing them from
operation numbers; each additional P4 recipe must version its carry state and
publication semantics before source emission is enabled. Future plan classes
will add those contracts and supported `tpush`/`tpop`/`tfree` transport without
changing this ownership boundary.

The backend must not redo planning. Every emitted loop, lifetime, transfer, and
FIFO must be traceable to the solution descriptor. It should publish the
schedule report and pseudocode beside the source so users can inspect the
decision.

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

1. Silicon-close the installed materialized-vector, online-softmax, and static
   cube source slices.
2. Reproduce [closed PyPTO PR #2335](https://github.com/hw-native-sys/pypto/pull/2335)'s
   seven hand-tiled vector comparisons through standalone Torch capture,
   Fusebox planning, and explicit generated PyPTO.
3. Extend homogeneous source replay to the remaining serialized vector and cube
   schedule families.
4. Expand the generic mixed source backend from one round trip while continuing
   to reject unsupported multi-round-trip groups before emission.
5. Preserve unsupported nodes as explicit graph cuts and verify every value
   crossing those boundaries.
6. Add Type-1 dynamic outer chunks with a static physical tile and runtime
   `valid_shape`.
7. Defer Types 2-5 until the static frontend/backend and Type 1 are correct on
   device.

## Non-goals

- translating arbitrary Python control flow;
- recognizing model names or hard-coding FlashAttention or SwiGLU algorithms;
- choosing quantization precision for the model author;
- replacing the PyPTO compiler, verifier, PTOAS, or runtime scheduler;
- silently approximating unsupported aliases, mutations, views, indirect
  accesses, or data-dependent control.
