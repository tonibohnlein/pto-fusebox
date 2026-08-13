# Torch/Hugging Face to PyPTO source generation

## Status

This document records a future PTO-Fusebox direction. It does not describe a
currently supported API. Dynamic-shape classes beyond the first static-chunk
class are catalogued here and deliberately deferred.

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
PyPTO source already contains the selected fusion boundaries, grid, propagated
regions, topological order, physical tiles, loops, pipelines, lifetimes,
cross-core FIFOs, and valid-shape handling. PyPTO parses, verifies, lowers, and
executes that explicit program without rerunning the Fusebox planner.

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
8. returns a complete, code-generatable solution descriptor.

Fusebox does not replace the runtime scheduler. It forms good kernels and
preserves their dependency graph. The PyPTO runtime remains responsible for
launching ready kernels and overlapping independent AIC and AIV work.

### PyPTO source backend

The backend deterministically serializes the selected solution descriptor as
readable PyPTO DSL. Depending on the plan, it emits `pl.spmd`, static physical
tile shapes, runtime `valid_shape`, `pl.range`, `pl.pipeline`, tensor views,
explicit GM boundaries, and supported `tpush`/`tpop`/`tfree` transport.

The backend must not redo planning. Every emitted loop, lifetime, transfer, and
FIFO must be traceable to the solution descriptor. It should publish the
schedule report and pseudocode beside the source so users can inspect the
decision.

## Dynamic-shape baseline

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

## Type 1: dynamic independent extent, static physical chunk

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

## First implementation sequence

1. Import a static vector RMSNorm FX graph and compare it with the current
   Fusebox vector plan and a hand-written PyPTO reference.
2. Generate readable PyPTO from the solution and compare graph semantics,
   schedule fields, lowered PTO, and device numerics.
3. Repeat for a static cube matmul.
4. Generate a generic `QK -> vector softmax DAG -> PV` program without an
   attention recognizer.
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
