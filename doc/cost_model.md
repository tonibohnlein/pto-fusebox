# Cost Model Reference

This document describes the cost model as implemented in the solver.
It covers three levels: subgraph cost (Phase 1), partition cost (sum of
subgraph costs), and solution cost (subgraph costs with inter-step
retention).

---

## 1. Problem Structure

A problem is a DAG of `MatMul` and `Pointwise` ops over 2D tensors.

Each **tensor** has `width` (columns) and `height` (rows), with
`size = width * height`.

Each **op** has typed inputs, outputs, and a `base_cost` (hardware
compute latency at native granularity).

Hardware parameters:
- `fast_memory_capacity` — working set limit (elements)
- `slow_memory_bandwidth` — transfer rate (elements/time)
- `native_w`, `native_h` — hardware array dimensions

Three-tier memory:
- **Slow memory**: infinite capacity, finite bandwidth. Graph inputs
  start here; graph outputs must end here.
- **Fast memory**: finite capacity, zero-cost access. Compute reads/writes
  here.
- **Ephemeral**: intermediates within a fused subgraph. Zero capacity,
  zero transfer cost.

---

## 2. Subgraph (`src/core/ascend910b_cost.{h,cpp}`)

A subgraph is a set of op indices that will execute as a single fused
kernel with a shared tiling grid.

`Subgraph` is a compile-time alias (`src/core/subgraph.h`) for the active
architecture's cost model — today `Ascend910BCost`, which models the
`CostModel` concept (`src/core/cost_model.h`). Its architecture-independent
structural classification (boundary/ephemeral tensors, sinks) is delegated to
`SubgraphStructure` (`src/core/subgraph_structure.{h,cpp}`). All `Subgraph::…`
references below resolve through that alias.

### 2.1 Construction — `Subgraph::create()`

Factory that validates and precomputes everything needed for cost
evaluation. Returns `nullopt` if invalid.

**Validation requirements:**
1. At least one boundary output tensor.
2. All boundary outputs have the same `(width, height)`.
3. Connected sub-DAG (ops linked through internal tensor edges or
   shared-input co-consumer edges).

**Tensor classification:**
- **Boundary input**: consumed by an op in the subgraph, not produced
  internally. Must be loaded from slow memory.
- **Boundary output**: produced by an op in the subgraph and consumed
  only outside the subgraph. Must be evicted to slow memory (unless
  retained).
- **Ephemeral**: produced AND consumed internally, with no external
  consumers. Zero memory cost — passes through the tile pipeline.
  Multiple internal consumers are allowed (fan-out OK).

**Mixed-consumer constraint**: a tensor produced inside a subgraph
cannot have both internal and external consumers. It must be either
fully ephemeral or fully boundary. This means the partition search
must avoid grouping ops such that an internal tensor has consumers
both inside and outside the group.

The one exception is **recomputation**: if every external consumer
lives in a subgraph that also contains the producing op, those
subgraphs compute the tensor independently. The tensor can then be
treated as ephemeral in the original subgraph. The partition
tracks which tensors satisfy this condition via `creates_ephemeral_gap`,
which rejects moves that would violate the constraint.

**Tiling constraints** (precomputed as divisibility sets):
- `w_divides_`: w must divide every boundary and ephemeral output width
- `h_divides_`: h must divide every boundary and ephemeral output height
- `k_divides_`: k must divide every MatMul's K dimension (= LHS width)
- If any boundary output is produced by a Pointwise, `has_pw_sink_ = true`
  and k is forced to 1 (PW cannot participate in k-steps)

**Candidate enumeration**: valid `(w, h, k)` values are precomputed
from the divisors of the relevant dimensions (`ws_cand_`, `hs_cand_`,
`ks_cand_`).

### 2.2 Boundary Tensor Info — Slice Size Propagation

Each boundary tensor's per-tile slice size is determined by backward
propagation through the op chain. The key abstraction is
`BoundaryTensorInfo`, which stores the *source* of the tensor's
h_tiles and v_tiles:

```
h_tiles divides tensor.width  → slice_w = tensor.width / h_tiles
v_tiles divides tensor.height → slice_h = tensor.height / v_tiles
slice_size = slice_w * slice_h
```

**TileSource enum** — how h_tiles/v_tiles are determined:

| Source     | Value                | Meaning                           |
|------------|----------------------|-----------------------------------|
| `FIXED_1`  | 1                    | Full extent (no tiling in this dim) |
| `FROM_NTW` | `out_W / w`          | Tracks output column tiling       |
| `FROM_NTH` | `out_H / h`          | Tracks output row tiling          |
| `FROM_NK`  | `output_K / k`       | Tracks k-step tiling (split-K)    |

**Propagation rules** (matching the reference evaluator):

| Tensor role          | h_source   | v_source   |
|----------------------|------------|------------|
| Boundary output      | `FROM_NTW` | `FROM_NTH` |
| PW input (any)       | inherits output's h | inherits output's v |
| Non-sink MM LHS      | `FIXED_1`  | output's v |
| Non-sink MM RHS      | output's h | `FIXED_1`  |
| Sink MM LHS (nk > 1) | `FROM_NK` | output's v |
| Sink MM RHS (nk > 1) | output's h | `FROM_NK`  |

"Sink MM" = the MatMul whose output is a boundary output (drives the
temporal k-loop). "Non-sink MM" = upstream MatMul whose output is
ephemeral.

**Transfer timing** is derived from the sources:
- `FIXED_1` → resident (loaded once per tile grid)
- `FROM_NTW` → changes with output column position
- `FROM_NTH` → changes with output row position
- `FROM_NK` → streamed (fresh slice every k-step)

---

## 3. Tiling Config (`TileConfig`)

```cpp
struct TileConfig {
    int64_t w, h, k;
    SnakeDir snake;  // None (raster), RowMajor, ColMajor
};
```

The output tensor (`out_W` x `out_H`) is divided into a spatial grid:

```
num_tw = out_W / w      (horizontal tile count)
num_th = out_H / h      (vertical tile count)
num_tiles = num_tw * num_th
nk = output_K / k       (k-passes per tile; 1 if no MatMul or PW sink)
```

### 3.1 Tiling Validity — `is_valid_tiling()`

**Fast path**: checks that w, h, k divide all required dimensions
(stored in `w_divides_`, `h_divides_`, `k_divides_`). Enforces k = 1
if `has_pw_sink_`.

**Slow path** (only for subgraphs with tiling conflicts): full
numerical propagation of h_tiles/v_tiles backward through the chain,
checking for dimension conflicts where a tensor gets incompatible
tiling requirements from different consumers.

---

## 4. Working Set — `working_set()`

Returns the peak fast-memory occupancy for a given config and retention
context:

```
WorkingSet = sum of slice sizes for all boundary tensors
```

For each boundary tensor:
- **Retained from previous step**: charged at full tensor size (not
  slice size) — the entire tensor persists in fast memory.
- **Retained for next step**: NOT charged (retained outputs don't
  occupy additional space — they replace the eviction slot).
- **Normal boundary tensor**: charged at slice size =
  `(width / h_tiles) * (height / v_tiles)`.

The subgraph is **feasible** if `working_set <= fast_memory_capacity`.

---

## 5. Cost Evaluation — `compute_cost()`

Returns a `CostResult` with latency, working set, tile counts, and
the config used. Returns infeasible if working set exceeds capacity.

### 5.1 Compute Cost Per Step

For each op in the subgraph:

```
op_scale = ceil(slice_w / native_w) * ceil(slice_h / native_h)
```

where `(slice_w, slice_h)` are the op's output tile dimensions. At or
below native dimensions, `op_scale = 1` (full `base_cost` per tile).
Above native, multiple hardware passes fire.

```
comp_per_step = sum over ops of: base_cost * op_scale / nk
```

The `/nk` factor reflects that each k-step processes `1/nk` of the
reduction dimension.

### 5.2 Memory I/O Categorization

Each boundary tensor's transfer cost is classified by *when* it needs
to be (re)loaded relative to the tile traversal:

| Category       | When loaded                              | Determined by       |
|----------------|------------------------------------------|---------------------|
| `once_load`    | Once per tile grid (first tile only)     | Both dims `FIXED_1` |
| `row_load`     | On output row change                     | h `FIXED_1`, v varies |
| `col_load`     | On output column change                  | v `FIXED_1`, h varies |
| `tile_load`    | Every tile                               | Both dims vary      |
| `stream_load`  | Every k-step                             | Depends on `FROM_NK` |
| `out_evict`    | Output tensor written per tile           | Boundary outputs    |

Transfer cost per tensor = `slice_size / slow_memory_bandwidth`.

Retained-from-prev tensors skip loading (already in fast memory).
Retained-for-next outputs skip eviction.

### 5.3 Per-Tile Latency — Roofline Model

Each tile's cost is the max of compute and memory (overlapped):

**For nk = 1 (no split-K):**
```
tile_cost = max(comp_per_step, once_load + row_load + col_load +
                                tile_load + stream_load + out_evict)
```
(where `once/row/col` loads are conditional on whether data is fresh
for this tile)

**For nk > 1 (split-K):**
```
step_0   = max(comp, per_tile_io + stream_load)     // load all
mid_steps = (nk - 2) * max(comp, stream_load)       // stream only
step_last = max(comp, stream_load + out_evict)       // evict output
tile_cost = step_0 + mid_steps + step_last
```

The `per_tile_io` term includes `once/row/col/tile` loads, conditional
on whether the tile is the first in its row/column/grid.

### 5.4 Traversal Order

The traversal order determines which loads are "fresh" vs "reused"
between consecutive tiles.

**Raster (`SnakeDir::None`)** — row-major:
- Within a row: column changes, row stays → `col_load` fresh,
  `row_load` reused.
- Row transition: both change → both fresh.

For MatMul subgraphs with multiple columns:
```
latency = first_tile(all fresh)
        + (num_th - 1) * row_start(row + col fresh)
        + (num_tw - 1) * num_th * within_row(col fresh only)
```

**RowMajor snake** — alternating row direction:
- Within a row: column changes (same as raster).
- Row transition: row stays fixed, column resets → `row_load` fresh,
  `col_load` reused (the snake reversal means the last tile of row N
  and the first tile of row N+1 share a column position).

```
latency = first_tile
        + (num_th - 1) * row_transition(row fresh, col reused)
        + (num_tw - 1) * num_th * within_row(col fresh)
```

**ColMajor snake** — alternating column direction:
- Within a column: row changes, column stays.
- Column transition: column changes, row reused.

```
latency = first_tile
        + (num_tw - 1) * col_transition(col fresh, row reused)
        + (num_th - 1) * num_tw * within_col(row fresh)
```

For PW-only subgraphs (no MatMul), every tile pays the full cost
(no directional reuse). Only `SnakeDir::None` is tried.

### 5.5 Best Config Enumeration — `best_cost()`

Iterates over all `(w, h, k, snake)` combinations:
- For MatMul subgraphs: tries `RowMajor` and `ColMajor` snakes.
- For PW-only: tries `None` only.

Returns the feasible config with lowest latency.

---

## 6. Cost Cache (`src/core/cost_cache.h`)

Two-tier memoization:

**Tier 1 — Base map** (Phase 1, partition search):
- Key: sorted op indices.
- Value: `CostResult` from `best_cost()` with no retention.
- Thread-safe (shared_mutex, concurrent reads).
- `evaluate(ops, prob, dag)` → cache lookup or compute + store.

**Tier 2 — Retention map** (Phase 2/3, solution search):
- Key: `[op_set, separator, entering_set, separator, retain_set]`.
- Value: `CostResult` from `best_cost(entering, retain)`.
- Falls back to base map when entering and retain are both empty.
- `evaluate_with_context(ops, prob, dag, entering, retain)`.

The cache is shared across all threads and persists across all solver
phases. Phase 1 populates the base map; Phases 2/3 reuse base entries
and add retention variants.

---

## 7. Partition Cost (`src/partition/partition.h`)

### 7.1 `eval_set(ops)`

Evaluates the latency of a single op set:

```cpp
if (cache) return cache->evaluate(ops, prob, dag);
// else: create Subgraph, call best_cost(), return latency
```

Returns `1e18` if infeasible (can't form a valid subgraph, or no
config fits in memory).

### 7.2 `total_cost()`

Sum of all alive group costs:

```cpp
total = sum of groups[gi].cost for all alive groups
```

Each `groups[gi].cost` is set by `eval_set(groups[gi].ops)` when the
group is created or modified. This is the objective that Phase 1
(greedy descent + FM) optimizes.

**Important**: partition cost assumes no retention (each group's
subgraph is costed independently). Retention savings are only realized
in Phase 2/3 when groups are ordered into a solution.

### 7.3 Partition Constraints

The partition search maintains three structural invariants. Every move
type validates these before mutating the partition.

**Mixed-consumer (ephemeral gap) constraint**: a tensor produced inside
a group cannot have both internal and external consumers. Enforcement:
`creates_ephemeral_gap()` (and `split_creates_ephemeral_gap()` for
splits) reject any move that would create such a mixed-consumer tensor.
The sole exception is recomputation — if every external consumer's
group also contains the producing op, the tensor is independently
computed in each group and can remain ephemeral in the original.

**PW sink k=1**: if any boundary output of a subgraph is produced by a
Pointwise op, k is forced to 1 (Pointwise cannot participate in
k-steps). This is enforced at subgraph creation: when `has_pw_sink_`
is true, `ks_cand_` is set to `{1}`. All cost evaluation and tiling
validity checks flow through these precomputed candidates, so no
Phase 1 code path can bypass this restriction.

**Acyclicity**: the group-level DAG induced by the partition must be
acyclic. This is non-trivial with recomputation because an op can
appear in multiple groups, creating OR-dependencies: a consumer group
is satisfied if *any one* of its producer groups has been scheduled.
Enforcement uses a modified Kahn's algorithm (`kahn_with_delta` in
`feasibility.cpp`) that tracks in-degrees with OR-node semantics —
when a group is removed from the queue, all ops it contains satisfy
their consumers via an OR (not AND) reduction. Move validation
functions (`is_acyclic_after_steal`, `is_acyclic_after_merge`, etc.)
call `kahn_with_delta` with virtual group deltas, simulating the move
without mutating the partition. All 8 move types that change group
membership validate acyclicity before applying.

---

## 8. Solution Cost (`src/solution/solution.h`, `src/solution/solution.cpp`)

### 8.1 Schedule Structure

A solution is an ordered sequence of `ScheduleStep`:

```cpp
struct ScheduleStep {
    Subgraph subgraph;
    TileConfig config;
    std::set<size_t> retain_these;  // outputs to keep for next step
};
```

### 8.2 Total Latency

Computed in the `Solution` constructor by propagating retention state:

```cpp
set<size_t> currently_retained = {};  // fast-memory contents

for each step i:
    retained_entering[i] = currently_retained;

    step_costs[i] = subgraph.compute_cost(
        config[i],
        currently_retained,    // tensors available from step i-1
        retain_these[i]        // tensors to keep for step i+1
    );

    total_latency += step_costs[i].latency;
    currently_retained = retain_these[i];
```

Retention affects cost in two ways:
1. **Producing step**: retained output skips eviction
   (saves `tensor_size / bandwidth`).
2. **Consuming step**: retained tensor skips loading
   (saves `slice_size / bandwidth`, but adds `full_tensor_size`
   to working set capacity).

### 8.3 Retention Lifetime

A retained tensor lives for exactly one step transition:
`retain_these[k]` keeps tensors from step k into step k+1. After
step k+1 completes, they are freed. To keep a tensor for step k+2,
step k+1 must produce it (recomputation) and retain it again.

### 8.4 Ordering Algorithms

**DFS ordering** (`src/solution/ordering.cpp`): greedy topological
sort. Picks the ready group that maximizes `retain_score` (total size
of shared boundary tensors with the previous group). Fast: O(N log N).

**Beam search ordering**: tracks fast-memory residency explicitly.
For each candidate group, evaluates cost with different combinations
of entering and retain sets. Keeps top-K states by total latency.
Quality-focused but O(N^2 * beam_width).

### 8.5 `steps_from_ordering()`

Converts an `OrderingResult` into `ScheduleStep` objects with fallback:
1. Try full retention context (entering + retain).
2. If infeasible, drop retain (entering only).
3. If still infeasible, drop both (baseline, always cached from Phase 1).

---

## 9. Cost Flow Summary

```
Phase 1 (Partition search):
  eval_set(ops) → CostCache::evaluate()
    → Subgraph::create(ops)
    → Subgraph::best_cost(no retention)
      → for each (w, h, k, snake):
          working_set() → feasibility check
          compute_cost() → per-tile roofline
    → cache in base_map
  total_cost = sum of group costs

Phase 2 (Solution construction):
  OrderingResult → steps_from_ordering()
    → Subgraph::best_cost(entering, retain)
      → compute_cost with retention context
    → cache in retention_map
  Solution(steps) → propagate retention → total_latency

Phase 3 (Solution evolution):
  Mutate solution steps → re-evaluate with retention
    → CostCache::evaluate_with_context()
    → total_latency from Solution constructor
```

---

## 10. Verified Against Reference Examples

| Ex | Scenario                  | Strategy              | Expected | Model |
|----|---------------------------|-----------------------|----------|-------|
| 1  | PW chain 128x128          | A: Separate           | 6,553.6  | pass  |
| 1  |                           | B: Fused 128x128      | 3,276.8  | pass  |
| 1  |                           | C: Fused 64x64        | 4,400.0  | pass  |
| 2  | PW chain 256x256          | A: Separate           | 26,214.4 | pass  |
| 2  |                           | B: Fused 128x128      | 13,107.2 | pass  |
| 3  | Diamond PW graph          | A: Spill              | 11,468.8 | pass  |
| 3  |                           | B: Recompute          | 6,276.8  | pass  |
| 3  |                           | C: Retain             | 4,638.4  | pass  |
| 4  | MatMul traversal          | A: Raster             | 7,096    | pass  |
| 4  |                           | B: Zig-zag            | 6,548    | pass  |
| 5  | Chained MatMul split-k    | B: k=32               | 6,915.2  | pass  |
