# Cost Model Documentation

This document explains how the solver computes the latency of a subgraph given a
tiling configuration `[w, h, k]`. It traces exactly what the code does, with
references to the implementation in `src/core/subgraph.cpp`.

## Overview

Each subgraph is a set of ops executed together. The tiling `[w, h, k]` splits
the computation into a grid of spatial tiles, each producing a `w × h` output
slice. The total latency is the sum of per-tile-step latencies, where each step's
latency follows a roofline model: `max(compute, memory_in + memory_out)`.

## Tensor classification

When a subgraph is created (`Subgraph::create`), tensors are classified into
three categories:

- **Boundary inputs**: consumed by an op in the subgraph but NOT produced by any
  op in the subgraph. These must be loaded from slow memory.
- **Boundary outputs**: produced by an op in the subgraph but NOT consumed by
  any op in the subgraph. These must be evicted to slow memory (unless retained).
  The subgraph must have exactly one boundary output (the "sink tensor").
- **Ephemeral**: produced AND consumed within the subgraph. These pass directly
  between ops without touching fast memory. They consume **zero capacity** and
  incur **zero transfer cost**. This is the key benefit of fusion.

## The tiling parameters `[w, h, k]`

### Spatial dimensions: `w` and `h`

`w` (width) and `h` (height) define the output tile size. The sink tensor has
dimensions `out_W × out_H`, and the tiling creates a grid:

    num_tile_cols (ntw) = out_W / w
    num_tile_rows (nth) = out_H / h
    num_tiles           = ntw × nth

**Validity**: `w` must evenly divide every output tensor width in the subgraph.
`h` must evenly divide every output tensor height. This ensures all ops in the
subgraph can be tiled compatibly.

**Input slice shapes** depend on op type:

| Input type        | Slice shape | Size      |
|-------------------|-------------|-----------|
| Pointwise input   | `w × h`     | `w × h`   |
| MatMul LHS        | `h × K`     | `h × K`   |
| MatMul RHS        | `k × w`     | `k × w`   |
| MatMul output     | `h × w`     | `h × w`   |

**Important**: The MatMul LHS slice is `h × K` (full reduction dimension), NOT
`h × k`. The spec line 95 says "LHS requires width k and height h," but this
describes the per-k-step *computational* slice. The *memory* model loads the full
`h × K` strip once per tile and keeps it resident across all k-steps. This is
confirmed by Example 5 in PROBLEM.md, where T0(128×128) is kept fully resident
with a working set of 40,960 = 128×128 + 32×128 + 32×128 + 128×128.

### Reduction dimension: `k`

`k` controls the **split-K** depth for MatMul operations:

    num_k_passes (nk) = K / k     (K = full reduction dimension)

With `k < K`, the hardware enters "output stationary" mode: it locks the output
accumulator in fast memory and iterates through the reduction dimension in `k`-
sized steps. At each k-step:
- The LHS strip (`h × K`) is already resident (loaded at k=0)
- A new RHS strip (`k × w`) is loaded from slow memory
- A partial dot product is accumulated into the output

**Validity**: `k` must evenly divide every MatMul's reduction dimension K in the
subgraph. For PW-only subgraphs, `k` is ignored (always 1 k-pass).

**Trade-off**: Smaller `k` → smaller RHS strips → lower working set. But more
k-passes → more RHS loads (paid every step).

## Working set (fast memory usage)

The working set is the peak fast memory occupied during any single tile-step
(`Subgraph::working_set`):

    ws = Σ (boundary input slices) + Σ (boundary output slices)
         + Σ (retained-from-prev full tensors)
         + Σ (retain-these output accumulation)

### Per-input contributions

For each boundary input tensor:

| Input type             | Normal (not retained)  | Retained from prev step |
|------------------------|------------------------|-------------------------|
| PW input               | `h × w`                | `width × height` (full) |
| MatMul LHS             | `h × K`                | `width × height` (full) |
| MatMul RHS             | `k × w`                | `width × height` (full) |

A retained tensor occupies its **full size** (not a tile-sized slice) because
the entire tensor is already resident in fast memory from the previous step.

### Output contribution

- **MatMul output (accumulator)**: `h × w`. Required because the accumulator
  must persist in fast memory across k-steps.
- **PW output**: **NOT counted**. The hardware processes PW element-by-element
  in a streaming fashion; the output does not need a separate buffer in fast
  memory. This is confirmed by Example 2 in PROBLEM.md where a PW singleton
  at `[128,128]` fits in capacity 25,000 (input 16,384 ≤ 25,000; if output
  were counted, 32,768 > 25,000 would be OOM).

### Shared boundary inputs

When multiple ops in a subgraph share the same boundary input tensor, it is
loaded **once** and counted **once** in both working set and transfer cost.
A `counted` set prevents double-charging.

### Retained output accumulation

When the output is retained (not evicted after this step), tiles accumulate in
fast memory. At the last tile, all previous tiles' output is still resident:

    additional_ws = full_output_size - h × w

### Unused retained tensors

A tensor retained from a previous step that is NOT used by this subgraph still
occupies its full `width × height` in fast memory.

### Feasibility check

    ws ≤ fast_memory_capacity

If not, this tiling is infeasible (OOM).

## Compute cost

### Compute scale (padding)

    scale = ceil(w / native_w) × ceil(h / native_h)

- Below native (e.g., w=64, native=128): `ceil(64/128) = 1`. The hardware
  processes a full native-sized block but only produces `w × h` of useful output.
  Compute cost is the same as native, but you need more tiles to cover the
  full output.
- Above native (e.g., w=256, native=128): `ceil(256/128) = 2`. The hardware
  needs 2 native passes per tile direction.

### Per-k-step compute

MatMul and PW compute are separated because they have different k-step behavior:

    mm_comp = scale × Σ (base_cost_i × k / K_i)   for each MatMul op i
    pw_comp = scale × Σ (base_cost_i)              for each PW op i

- **MatMul**: compute is proportional to `k / K` (fraction of reduction done
  this step). Summed across all k-steps, total = `base_cost × scale`.
- **Pointwise**: compute is the full `base_cost × scale`, executed once per
  tile (at the last k-step only), because PW has no reduction dimension.

## Memory transfer cost

### Transfer amounts (per tile, in time units = size / bandwidth)

    lhs_load   = Σ (h × K_i / B)     for each UNIQUE boundary MatMul LHS (not retained)
    rhs_load   = Σ (k × w / B)       for each UNIQUE boundary MatMul RHS (not retained)
    pw_in_load = Σ (h × w / B)       for each UNIQUE boundary PW input (not retained)
    out_evict  = Σ (h × w / B)       for each UNIQUE boundary output (not retained-this-step)

Each tensor is counted at most once across all ops (deduplicated via a `counted`
set). This matters when multiple ops in a subgraph share the same boundary input.

Retained inputs have zero transfer cost (already in fast memory).
Retained outputs have zero eviction cost (stay in fast memory).

### Per-k-step transfer schedule

Within a single tile, transfers are distributed across k-steps:

| k-step | Memory in                          | Memory out  |
|--------|------------------------------------|-------------|
| k = 0  | lhs_load + rhs_load + pw_in_load   | 0           |
| k = 1..nk-2 | rhs_load                     | 0           |
| k = nk-1 | rhs_load                          | out_evict   |

- **k=0 (first)**: loads LHS (once per tile), first RHS strip, and all PW inputs
- **Middle k-steps**: only reload the RHS strip (LHS stays resident)
- **k=nk-1 (last)**: loads final RHS strip and evicts the output

If `nk = 1` (no split-K), k=0 is also the last step, so all transfers happen
together.

## Roofline model

Each k-step's latency is:

    step_latency = max(step_compute, memory_in + memory_out)

Where `step_compute` is `mm_comp` for k-steps 0..nk-2, and `mm_comp + pw_comp`
for the last k-step (PW executes once per tile).

A tile's total latency is the sum of its k-step latencies.

## Snake traversal (tile ordering)

Without snake (`None`): every tile loads all its inputs fresh. The per-tile cost
is `tile_cost(lhs_fresh=true, rhs_fresh=true)` and total = `num_tiles × per_tile`.

With snake, tiles are visited in a zig-zag pattern that enables data reuse:

### RowMajor snake

Traverses left→right on even rows, right→left on odd rows:

    Row 0: (0,0) → (0,1) → (0,2) →
    Row 1:                          (1,2) → (1,1) → (1,0) →
    Row 2: (2,0) → (2,1) → (2,2) →  ...

Consecutive tiles on the **same row** share the LHS strip (no reload needed).
At row transitions (same column), the RHS strip is reused.

Tile classification for RowMajor:
- **ff** (both fresh): 1 tile (the first one)
- **rf** (LHS reused, RHS fresh): `(ntw - 1) × nth` tiles (same-row transitions)
- **fr** (LHS fresh, RHS reused): `nth - 1` tiles (row transitions)

### ColMajor snake

Traverses top→bottom on even columns, bottom→top on odd columns. The counts
are transposed:
- **ff**: 1
- **rf** (LHS reused): `ntw - 1`
- **fr** (RHS reused): `(nth - 1) × ntw`

### Which snake is better?

RowMajor creates more LHS-reuse tiles (`(ntw-1)×nth`), ColMajor creates more
RHS-reuse tiles (`(nth-1)×ntw`). Choose RowMajor when LHS is large relative to
RHS (wide grid), ColMajor when RHS is large (tall grid).

**With LHS retained**: LHS load is always 0 (regardless of reuse), so LHS-reuse
tiles don't save anything. ColMajor may then win because its RHS reuse still
provides savings.

Snake only affects MatMul subgraphs. For PW-only, all boundary inputs have shape
`h × w` regardless of tile position, so there's no spatial reuse pattern.

## The search: `best_cost()`

The optimizer enumerates all valid `[w, h, k]` combinations:
- `w` ∈ divisors of `out_W` that also divide all output widths in the subgraph
- `h` ∈ divisors of `out_H` that also divide all output heights in the subgraph
- `k` ∈ divisors of `max_K` that also divide all K values in the subgraph
- Snake ∈ {None, RowMajor, ColMajor} (only None for PW-only subgraphs)

Minimum tile size: `native_w/4 × native_h/4` (relaxed to 1×1 if nothing fits).

For each combination, it computes working set, checks feasibility, computes
latency, and keeps the minimum.

## Retain mechanics

### Retaining an output (this step)

When a boundary output tensor is in `retain_these`:
- **No eviction**: `out_evict` is 0 for that tensor (transfer savings)
- **Accumulation overhead**: tiles accumulate without being flushed. The working
  set increases by `full_size - tile_size` (all completed tiles stay resident)
- **Tiling impact**: the larger working set may force smaller tiles

### Using a retained input (from previous step)

When a boundary input is in `retained_from_prev`:
- **No load cost**: that input's transfer is 0 (already in fast memory)
- **Full-size capacity**: the tensor occupies `width × height` in working set
  (the full tensor, not a tile-sized slice), because the entire tensor is
  resident from the previous step

### When retain helps

Retain is beneficial when:
    transfer_savings > tiling_penalty

The transfer savings come from eliminating one tensor's load/evict across all
tiles. The penalty comes from larger working set potentially forcing smaller
tiles (more tiles, more overhead, possible padding).

## Complete latency formula

    total_latency = Σ over all tiles of Σ over all k-steps of
                    max(step_compute, memory_in + memory_out)

With snake, this simplifies to:

    total = ff_count × tile_cost(true, true)
          + rf_count × tile_cost(false, true)
          + fr_count × tile_cost(true, false)

Where `tile_cost(lhs_fresh, rhs_fresh)` sums the k-step roofline values for
one tile with the given reuse pattern.
