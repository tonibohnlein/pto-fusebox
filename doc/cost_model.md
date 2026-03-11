# Cost Model Documentation

This document explains how the solver computes the latency of a subgraph given a
tiling configuration `[w, h, k]`, the constraints on forming valid subgraphs, and
the constraints on valid tiling parameters.

## Overview

Each subgraph is a set of ops executed together. The tiling `[w, h, k]` splits
the computation into a grid of spatial tiles, each producing a `w × h` output
slice. The total latency is the sum of per-tile-step latencies, where each step's
latency follows a roofline model: `max(compute, memory_in + memory_out)`.

---

## Subgraph validity

A set of ops forms a valid subgraph if ALL of the following hold:

### 1. Non-empty

At least one op.

### 2. At least one boundary output

Every subgraph must produce at least one tensor that is not consumed internally.

### 3. All boundary outputs have the same dimensions

If a subgraph has multiple boundary outputs, they must all have the same
`(width, height)`. This is because the spatial tile grid is defined by the
boundary output dimensions — all boundary outputs share the same grid.

### 4. Connected

Ops must form a connected group. Two ops are connected if they:
- Have a producer-consumer edge (one produces a tensor the other consumes), OR
- Share a common input tensor (co-consumers of the same boundary input)

The second condition enables fusing parallel ops that share an expensive input
(e.g., 16 MatMuls all reading the same large weight matrix in benchmark 13).

### 5. Ephemeral tensors have exactly one consumer

A tensor that is both produced and consumed within the subgraph is "ephemeral" —
it flows directly between ops without touching fast memory. For this to work,
each ephemeral tensor must have exactly **one** consumer op within the subgraph.
Fan-out of ephemeral tensors is invalid because the data exists only momentarily
and cannot be consumed twice.

Note: fan-out of **boundary** tensors is fine — boundary tensors are materialized
in slow memory (or retained in fast memory) and can be read by multiple subgraphs.

---

## Tensor classification

When a subgraph is created (`Subgraph::create`), tensors are classified into
three categories:

- **Boundary inputs**: consumed by an op in the subgraph but NOT produced by any
  op in the subgraph. These must be loaded from slow memory (unless retained from
  a previous step).
- **Boundary outputs**: produced by an op in the subgraph but NOT consumed by
  any op in the subgraph. These must be evicted to slow memory (unless retained
  for the next step).
- **Ephemeral**: produced AND consumed within the subgraph. These pass directly
  between ops without touching fast memory. They consume **zero capacity** and
  incur **zero transfer cost**. This is the key benefit of fusion.

---

## The tiling parameters `[w, h, k]`

### Role-based tiling constraints

The tiling parameters constrain tensor dimensions based on **how each tensor is
used** (its role), not based on a blanket rule. For each op in the subgraph:

**MatMul** with LHS=T1, RHS=T2, OUT=T3:

| Tensor | Role    | Slice shape | Constraints              |
|--------|---------|-------------|--------------------------|
| T3     | Output  | `w × h`     | `w` divides W3, `h` divides H3 |
| T1     | LHS     | `k × h`     | `k` divides W1, `h` divides H1 |
| T2     | RHS     | `w × k`     | `w` divides W2, `k` divides H2 |

Note: W1 and H2 are the reduction dimensions (they must be equal: W1 = H2 = K).
So `k` divides W1 and `k` divides H2 is the same constraint: `k | K`.

**Pointwise** with inputs T_i and output T_o:

The slice shape for PW ops depends on context. If T_o is a boundary output, all
slices are `w × h`. But if T_o is ephemeral and consumed by a downstream op, the
PW adopts the slice shape demanded by its consumer:

| Context                              | PW output slice | PW input slice |
|--------------------------------------|-----------------|----------------|
| T_o is boundary output               | `w × h`         | `w × h`        |
| T_o feeds MatMul as LHS              | `k × h`         | `k × h`        |
| T_o feeds MatMul as RHS              | `w × k`         | `w × k`        |
| T_o feeds another PW (propagate)     | same as that PW | same as that PW|

This propagation is computed recursively at subgraph creation time.

### Constraint collection

The code collects three sets of values:
- `w_divides_`: w must divide every value in this set
- `h_divides_`: h must divide every value in this set
- `k_divides_`: k must divide every value in this set

Each tensor contributes to these sets based on its role as described above.

### PW sink constraint

**If any boundary output is produced by a Pointwise op, k must be 1.**

Rationale: PW ops execute once per spatial tile and have no reduction dimension.
When k > 1, the subgraph's k-loop drives MatMul accumulation across multiple
steps. If a PW op is a sink of the subgraph, the interaction between the MatMul's
multi-step accumulation and the PW's single-shot execution creates ambiguity in
the cost model (when does the PW fire? what memory does the intermediate occupy
across k-steps?). Rather than model this complex interaction, we enforce k = 1
for any subgraph with a PW sink. This is not restrictive in practice because:

- PW-only subgraphs already have k = 1 (no MatMul, no reduction)
- MatMul chains without PW sinks can still use k > 1
- Mixed subgraphs like MM→PW will use k = 1, which means the MatMul fully
  computes its output in one step, the PW fires immediately, and the intermediate
  is truly ephemeral with zero memory cost — the simplest possible model.

### Candidate generation

Valid tiling candidates are divisors of the GCD of each constraint set:

    w ∈ divisors(gcd(w_divides_))
    h ∈ divisors(gcd(h_divides_))
    k ∈ {1}                          if any boundary output is a PW output
    k ∈ divisors(gcd(k_divides_))    otherwise

This is efficient: the GCD captures the tightest common divisibility requirement.

### Example: benchmark 13 chain

    Op0: T2(4096×128 LHS) @ T0(4096×4096 RHS) → T34(4096×128 eph)
    Op1: T34(4096×128 LHS) @ T1(4096×4096 RHS) → T35(4096×128 eph)
    Op2: T35(4096×128 LHS) @ T18(128×4096 RHS) → T36(128×128 boundary)

Constraints:
- w_divides_: {W of T0=4096 (Op0 RHS), W of T1=4096 (Op1 RHS), W of T18=128 (Op2 RHS), W of T36=128 (boundary out)} → gcd = 128
- h_divides_: {H of T2=128, H of T34=128, H of T35=128, H of T36=128, ...} → gcd = 128
- k_divides_: {W of T2=4096 (Op0 LHS), W of T34=4096 (Op1 LHS), W of T35=4096 (Op2 LHS), H of T0=4096 (Op0 RHS), H of T1=4096 (Op1 RHS), H of T18=4096 (Op2 RHS)} → gcd = 4096

Note that `w` need NOT divide the ephemeral tensor width (4096). The ephemeral
tensors T34 and T35 are consumed as LHS, so their width is constrained by `k`,
not `w`. This allows `w=128` even though ephemeral widths are 4096.

---

## Working set (fast memory usage)

The working set is the peak fast memory occupied during any single tile-step
(`Subgraph::working_set`):

    ws = Σ (boundary input slices) + Σ (boundary output slices)
         + Σ (retained-from-prev full tensors)
         + Σ (retain-these output accumulation)

### Per-input contributions

For each boundary input tensor:

| Input role             | Normal (not retained)  | Retained from prev step |
|------------------------|------------------------|-------------------------|
| PW input               | `h × w`                | `width × height` (full) |
| MatMul LHS             | `h × K`                | `width × height` (full) |
| MatMul RHS             | `k × w`                | `width × height` (full) |

A retained tensor occupies its **full size** (not a tile-sized slice) because
the entire tensor is already resident in fast memory from the previous step.

When a tensor serves multiple roles across different ops in the subgraph
(e.g., used as LHS by one MatMul and as PW input by another), the **maximum**
slice size across all roles is used.

### Output contribution

- **MatMul output (accumulator)**: `h × w`. Required because the accumulator
  must persist in fast memory across k-steps.
- **PW output**: **NOT counted**. The hardware processes PW element-by-element
  in a streaming fashion; the output does not need a separate buffer in fast
  memory. This is confirmed by Example 2 in PROBLEM.md where a PW singleton
  at `[128,128]` fits in capacity 25,000 (input 16,384 ≤ 25,000; if output
  were counted, 32,768 > 25,000 would be OOM).

Note: for mixed MM→PW subgraphs, the PW sink constraint forces k = 1. This
means the MatMul completes its output in a single step, so the ephemeral
intermediate is never an accumulator — it flows directly to the PW. No extra
memory is needed for ephemeral MM→PW intermediates.

### Shared boundary inputs

When multiple ops in a subgraph share the same boundary input tensor, it is
loaded **once** and counted **once** in both working set and transfer cost.

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

### Known conservatism in WS estimation

Our code sums the working set contributions of ALL boundary tensors simultaneously.
In reality, during a chained execution (e.g., Op0 → T_eph → Op1), the inputs of
Op0 and the inputs of Op1 may not all be needed at the same instant. The true peak
WS could be the maximum across ops rather than the sum. Our approach over-estimates,
which means we may reject some technically feasible tilings, but we will never
accept an infeasible one. The solver compensates by finding alternative tiling
parameters (e.g., smaller k or smaller spatial tiles) that fit the conservative
estimate.

---

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

- **MatMul**: `K_i` is the reduction dimension of MatMul op `i` (= width of its
  LHS tensor). Compute per k-step is proportional to `k / K_i`. Summed across
  all `K_i / k` k-steps, total = `base_cost_i × scale`.
- **Pointwise**: compute is the full `base_cost × scale`, executed once per
  tile (at the last k-step only), because PW has no reduction dimension.

---

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

**Note on PW boundary inputs in mixed PW+MatMul subgraphs**: When a PW op's
boundary input is loaded, we charge `h × w` for the slice. In principle, if the
PW is producing ephemeral data for a MatMul LHS (slice `k × h`), the PW input
should also be `k × h`. However, we conservatively use `h × w` because the PW
input is a boundary tensor loaded from slow memory independently. This may
over-estimate transfer cost slightly for mixed subgraphs, but no benchmark
currently exercises this case.

### Per-k-step transfer schedule

Within a single tile, transfers are distributed across k-steps:

| k-step     | Memory in                          | Memory out  |
|------------|------------------------------------|-------------|
| k = 0      | lhs_load + rhs_load + pw_in_load   | 0           |
| k = 1..nk-2 | rhs_load                         | 0           |
| k = nk-1   | rhs_load                           | out_evict   |

- **k=0 (first)**: loads LHS strips (once per tile), first RHS strip, and all
  PW inputs
- **Middle k-steps**: only reload the RHS strips (LHS stays resident)
- **k=nk-1 (last)**: loads final RHS strip and evicts the output(s)

If `nk = 1` (no split-K), k=0 is also the last step, so all transfers happen
together.

**Note on LHS reuse across k-steps**: the MatMul LHS strip (`h × K_i`) is loaded
once at k=0 and stays resident in fast memory for all subsequent k-steps. This is
why `lhs_load` uses the full `h × K_i` (not `h × k`). The RHS strip changes each
k-step (different `k`-wide column), so it's reloaded every time.

---

## Roofline model

Each k-step's latency is:

    step_latency = max(step_compute, memory_in + memory_out)

Where `step_compute` is `mm_comp` for k-steps 0..nk-2, and `mm_comp + pw_comp`
for the last k-step (PW executes once per tile, after all MatMul accumulation
is complete).

A tile's total latency is the sum of its k-step latencies.

---

## Snake traversal (tile ordering)

Without snake (`None`): every tile loads all its inputs fresh. The per-tile cost
is `tile_cost(lhs_fresh=true, rhs_fresh=true)` and total = `num_tiles × per_tile`.

With snake, tiles are visited in a zig-zag pattern that enables data reuse for
**MatMul** subgraphs. For PW-only subgraphs, snake has no effect (all inputs
depend on both row and column, so there's no strip reuse).

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

Snake only affects MatMul subgraphs. For PW-only, all boundary inputs have shape
`h × w` regardless of tile position, so there's no spatial reuse pattern.

---

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

---

## The search: `best_cost()`

The optimizer enumerates all valid `[w, h, k]` combinations:
- `w` ∈ divisors of `gcd(w_divides_)` — values w must divide, derived from
  tensor roles (MatMul RHS widths, boundary output widths, etc.)
- `h` ∈ divisors of `gcd(h_divides_)` — values h must divide (MatMul LHS
  heights, boundary output heights, etc.)
- `k` ∈ `{1}` if any boundary output is a PW output (PW sink constraint);
  otherwise `k` ∈ divisors of `gcd(k_divides_)` (MatMul reduction dimensions)
- Snake ∈ {RowMajor, ColMajor} for MatMul subgraphs, {None} for PW-only

Minimum tile size: `native_w/4 × native_h/4` (relaxed to 1×1 if nothing fits).

For each combination, it computes working set, checks feasibility, computes
latency, and keeps the minimum.

---

## Complete latency formula

    total_latency = Σ over all tiles of Σ over all k-steps of
                    max(step_compute, memory_in + memory_out)

With snake, this simplifies to:

    total = ff_count × tile_cost(true, true)
          + rf_count × tile_cost(false, true)
          + fr_count × tile_cost(true, false)

Where `tile_cost(lhs_fresh, rhs_fresh)` sums the k-step roofline values for
one tile with the given reuse pattern.