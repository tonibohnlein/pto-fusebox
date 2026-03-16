# MLSys 2026 — Definitive Cost Model Reference

Based on PROBLEM.md, all 5 worked examples, and organizer responses in
issues #3, #10, #14, #32, #34, #37, #38.

---

## 1. Problem Overview

Given a DAG of MatMul and Pointwise operations over 2D tensors, produce an
ordered sequence of subgraphs with execution granularities that covers every
op at least once, respects memory constraints, and minimizes total latency.

Three-tier memory hierarchy:
- **Slow memory:** infinite capacity, finite bandwidth B. All graph inputs
  start here; all graph outputs must end here.
- **Fast memory:** finite capacity C, zero-cost access. Compute can only
  touch data resident here.
- **Ephemeral:** intermediate data within a fused subgraph. Zero capacity
  cost, zero transfer cost.

---

## 2. Tensors and Operations

Each tensor has `width` (columns) and `height` (rows). Size = width × height.

**MatMul** `C = A @ B`:
- A (LHS): height × K matrix (K = A.width = reduction dimension)
- B (RHS): K × width matrix
- C (output): height × width matrix
- `inputs = [LHS_index, RHS_index]` — order matters

**Pointwise** `C = f(A, ...)`:
- All inputs and the output have the same dimensions
- No reduction dimension

**Graph inputs:** tensors with no producing op. Start in slow memory.
**Graph outputs:** tensors with no consuming op. Must end in slow memory.

---

## 3. Subgraph Rules

### 3.1 Composition

A subgraph is a set of op indices. Requirements:
- **Connected sub-DAG:** ops must be linked through tensors within the
  subgraph (producer→consumer edges or co-consumer edges sharing an input).
  Confirmed by organizer in #34.
- **At least one boundary output.**
- **All boundary outputs must have the same (width, height).**
- **Every op must appear in at least one subgraph** across the schedule.
  Ops MAY appear in multiple subgraphs (recomputation). See Example 3B.

### 3.2 Tensor Classification

For a given subgraph S:
- **Boundary input:** consumed by an op in S but not produced by any op in S.
- **Boundary output:** produced by an op in S and NOT consumed by any op in S.
- **Ephemeral:** produced AND consumed within S. Costs zero memory, zero
  bandwidth. Passes directly between ops in the tile pipeline.

A tensor can be ephemeral in one subgraph and a boundary output of another.
External consumers of an ephemeral tensor must be served by recomputation
(producing op also in their subgraph) or by the tensor being a boundary
output of a different subgraph. (Example 3B: T1 is ephemeral in [0,1] but
Op2 is served by recomputation of Op0 in [0,2].)

### 3.3 Ephemeral Single-Consumer Constraint

An ephemeral tensor may have only ONE consumer within the subgraph. The tile
exists momentarily in the pipeline and cannot fan out.

---

## 4. Execution Granularity [w, h, k]

A single `[w, h, k]` tuple governs the entire subgraph. It determines the
output tile size and the shapes of all input/output slices.

### 4.1 Output Tile

The subgraph's boundary output tensor (W × H) is divided into a grid:
```
num_tiles_w = W / w       (w must divide W)
num_tiles_h = H / h       (h must divide H)
num_spatial_tiles = num_tiles_w × num_tiles_h
```

Each spatial tile produces a `w × h` block of the output.

### 4.2 MatMul Slice Shapes (per the output-producing MatMul)

For the MatMul that directly produces a boundary output:
```
Output tile:   w × h
LHS slice:     h × k     (k columns of the LHS per k-step)
RHS slice:     k × w     (k rows of the RHS per k-step)
```

### 4.3 Upstream MatMul Slice Shapes

For a MatMul whose output is ephemeral (feeding another op in the chain):
the output tile shape is determined by what the downstream consumer needs.
In a two-MatMul chain Op0→(ephemeral)→Op1→(output) with granularity
[w, h, k]:
```
Op0 output (ephemeral):  h × k      (Op1's LHS slice per k-step)
Op0 LHS:                 h × K0     (full reduction, resident)
Op0 RHS:                 K0 × k     (k columns, streamed per k-step)
```
Note: Op0's RHS slice is `K0 × k`, NOT `k × w`. The dimensions are
determined by what Op0 needs to produce its `h × k` output tile.

### 4.4 Pointwise Slice Shapes

Pointwise inputs and outputs all have the same shape as whatever the
downstream chain demands. If the PW output feeds a MatMul LHS, the shape
is `h × k`. If it's a boundary output, the shape is `w × h`.

### 4.5 k Dimension Rules

**k splits the reduction of the output-producing MatMul.** The number of
temporal steps is:
```
nk = K_output / k       (k must divide K_output)
```
where K_output = reduction dimension of the MatMul producing the boundary
output (= its LHS tensor width).

**k > 1 is only valid if the boundary output is produced by a MatMul.**
If a Pointwise produces the boundary output, k must be 1. The PW has no
reduction dimension and cannot participate in k-steps.
Confirmed by organizer in #32:
> "No, this fusion is not valid with [128, 128, 32]. With split-K, the
> MatMul takes 4 passes to accumulate its result, and the Pointwise
> cannot participate in those k-steps."

Valid patterns for k > 1:
- Pure MatMul chain (output produced by MatMul)
- PW → MatMul chain (MatMul produces boundary output, PW produces
  k-sized tiles per step)

Invalid patterns for k > 1:
- MatMul → PW (PW produces boundary output) → k must be 1
- Pure PW chain → k must be 1

MatMul → PW fusion IS permitted with k = 1.
Confirmed by organizer in #14 and #32.

### 4.6 Divisibility Constraints

- w must divide every boundary output width AND every ephemeral output width
- h must divide every boundary output height AND every ephemeral output height
- k must divide every MatMul's reduction dimension (LHS tensor width)

These ensure every op in the subgraph can tile at the unified grid.

---

## 5. Memory Constraint: Working Set

For any chosen `[w, h, k]`, the sum of all input slices and output slices
that must coexist in fast memory during a single step must fit:

```
WorkingSet ≤ fast_memory_capacity
```

Working set components (per spatial tile, during peak step):

| Tensor role                    | Size per step     | Timing          |
|--------------------------------|-------------------|-----------------|
| Output-MM LHS (boundary input) | h × K            | Resident: loaded once per tile, kept across k-steps |
| Output-MM RHS (boundary input) | k × w            | Streamed: fresh slice each k-step |
| Upstream-MM RHS (boundary input) | K_upstream × k  | Streamed: fresh slice each k-step |
| PW input → boundary output     | w × h            | Tile-once: loaded once per tile |
| PW input → feeds MM LHS chain  | h × k            | Streamed: per k-step |
| PW input → feeds MM RHS chain  | k × w            | Streamed: per k-step |
| Boundary output (non-retained) | w × h            | Evicted per tile |
| Boundary output (MM accumulator) | w × h           | Resident across k-steps |
| Retained from previous step    | full tensor size  | Already in fast memory |
| Retained for next step         | full tensor size  | Must accumulate in fast memory |

Ephemeral tensors: **zero** memory cost.

Note: retained tensors (both `retained_from_prev` and `retain_these`) are
charged at **full tensor size**, not tile-slice size, because they persist
across all tiles.

---

## 6. Latency Model

### 6.1 Per-Step Roofline

Each execution step (one tile × one k-step) has latency:
```
StepLatency = max(ComputeTime, MemoryTime_in + MemoryTime_out)
```

Compute and memory are overlapped (roofline model). Memory in and out are
serialized (summed).

### 6.2 Compute Cost

**Spatial tiling:** each tile pays the full `base_cost` per native-array
execution. The hardware fires the full physical array each time.

```
compute_per_tile = base_cost × scale
scale = ceil(w / native_w) × ceil(h / native_h)
```

At or below native: scale = 1, full base_cost per tile.
Above native: scale > 1, multiple hardware passes per tile.
Confirmed by organizer in #38.

**Reduction (k) splitting:** compute scales proportionally. Fewer cycles
streamed through the array.
```
MatMul compute per k-step = base_cost × (k / K_output) × scale
```
All MatMul ops in a fused chain share the same fraction `k / K_output`
per step, because each step produces k / K_output of the chain's output.

**Pointwise compute:** fires once per spatial tile (after all k-steps
complete, if k > 1 and PW is upstream of a MatMul).
```
PW compute per tile = base_cost × scale
```

### 6.3 Memory Transfer Cost

```
MemoryTime = tensor_slice_size / slow_memory_bandwidth
```

Sizes are in elements (not bytes). Bandwidth B is in elements/time.

### 6.4 Per-Tile Latency Structure (with k-steps)

For nk > 1 (split-k):
```
Step 0:         max(mm_comp, resident_load + stream_load + tile_load)
Steps 1..nk-2:  max(mm_comp, stream_load)
Step nk-1:      max(mm_comp + pw_comp, stream_load + out_evict)
```

For nk = 1:
```
Single step:    max(mm_comp + pw_comp, all_loads + out_evict)
```

Where:
- `resident_load` = cost to load resident tensors (once per tile, first
  k-step only)
- `stream_load` = cost to load streamed tensors (every k-step)
- `tile_load` = cost to load tile-once PW inputs
- `out_evict` = cost to evict non-retained boundary outputs
- `mm_comp` = MatMul compute per k-step
- `pw_comp` = Pointwise compute per tile (added to last k-step)

### 6.5 Intra-Subgraph Data Reuse (Traversal Order)

The hardware automatically reuses input strips between consecutive spatial
tiles that share them. This is implicit — no explicit control needed.
Confirmed by organizer in #37:
> "Intra-subgraph data reuse is managed automatically by the hardware."

For MatMul with spatial tiling, consecutive tiles sharing a row index
reuse the LHS row strip; tiles sharing a column index reuse the RHS
column strip.

**Raster order** (default, or `null`): row-major. Within each row, the
LHS row strip is reused.
```
Raster [0,1,2,3] on 2×2 grid:
  TL(0,0) → TR(0,1): reuse row strip 0
  BL(1,0) → BR(1,1): reuse row strip 1
  BUT TL→BL transition: new row, reload both strips
  Result: 2 reuses
```

**Snake/zig-zag order**: alternating direction. Reuses both row and column
strips across row boundaries.
```
Zig-zag [0,1,3,2]:
  TL(0,0) → TR(0,1): reuse row strip 0
  TR(0,1) → BR(1,1): reuse col strip 1
  BR(1,1) → BL(1,0): reuse row strip 1
  Result: 3 reuses
```

Note: when nk > 1 (split-k), stream strips from the previous tile's last
k-step cover a different k-range than the new tile's first k-step. So
stream reuse does NOT carry across tiles — stream_load is always fresh
per tile. Resident reuse IS valid across tiles (same h×K stays resident).

### 6.6 Total Subgraph Latency

```
SubgraphLatency = Σ tile_latency(tile_i)  for all spatial tiles
```

Each tile's latency depends on whether resident/stream data is fresh or
reused, which depends on the traversal order.

### 6.7 Total Schedule Latency

```
TotalLatency = Σ SubgraphLatency(step_i)  for all steps in order
```

Execution between subgraphs is strictly serialized. Step k+1 cannot begin
until step k has fully completed.

---

## 7. Inter-Subgraph Tensor Retention

### 7.1 Mechanism

`tensors_to_retain[k]` lists output tensors from step k to keep in fast
memory for step k+1.

### 7.2 Lifetime = Exactly One Step

Confirmed by organizer in #34:
> "The lifetime of a retained tensor extends exactly one step:
> tensors_to_retain[k] keeps tensors resident from step k into step k+1.
> After step k+1 completes, they are freed."

To keep a tensor alive for step k+2, step k+1 must produce it (via
recomputation) and explicitly retain it in tensors_to_retain[k+1].

**No accumulation of retained tensors across multiple steps.** The set
of tensors in fast memory at the start of step k+1 is exactly
`tensors_to_retain[k]` and nothing else from earlier steps.

### 7.3 Effects on Cost

**Producing step (k):** retained output is NOT evicted to slow memory.
Saves `evict_cost = tensor_size / B`.

**Consuming step (k+1):** retained tensor is already in fast memory.
Saves `load_cost = tensor_slice_size / B` (or full tensor if used as
tile-once or resident input). The full tensor size occupies fast memory
capacity throughout step k+1.

**Non-consuming step:** if step k+1 doesn't use the retained tensor, it
still occupies fast memory capacity during step k+1 (then freed after).

### 7.4 Graph Outputs

Graph outputs (tensors with no consumers) must end in slow memory. If
retained at step k, they are freed after step k+1 but were never evicted
to slow memory — this may be a validity issue. Retention is typically
only useful for tensors that will be consumed by a later subgraph.

---

## 8. Schedule Validity Requirements

1. Every op in the graph appears in at least one subgraph.
2. Each subgraph is a connected sub-DAG.
3. Each subgraph has a valid `[w, h, k]` (divisibility, PW sink → k=1).
4. Each subgraph's working set fits in fast memory.
5. Graph outputs end in slow memory (evicted at some point).
6. The schedule's subgraph order respects data dependencies: if step j
   needs a boundary input produced by step i (or retained from step i),
   then i < j.
7. The condensed DAG of subgraphs must be acyclic (no circular
   dependencies between groups).
8. Ephemeral tensors with external consumers must be served (either by
   recomputation or by availability as a boundary output from another
   subgraph).

---

## 9. Verified Against Examples

| Ex | Scenario                  | Strategy              | Expected | Model |
|----|---------------------------|-----------------------|----------|-------|
| 1  | PW chain 128×128          | A: Separate           | 6,553.6  | ✓     |
| 1  |                           | B: Fused 128×128      | 3,276.8  | ✓     |
| 1  |                           | C: Fused 64×64        | 4,400.0  | ✓     |
| 2  | PW chain 256×256          | A: Separate           | 26,214.4 | ✓     |
| 2  |                           | B: Fused 128×128      | 13,107.2 | ✓     |
| 3  | Diamond PW graph          | A: Spill              | 11,468.8 | ✓     |
| 3  |                           | B: Recompute          | 6,276.8  | ✓     |
| 3  |                           | C: Retain             | 4,638.4  | ✓     |
| 4  | MatMul traversal          | A: Raster (7,096*)    | 7,096    | ✓     |
| 4  |                           | B: Zig-zag            | 6,548    | ✓     |
| 5  | Chained MatMul split-k    | B: k=32               | 6,915.2  | ✓     |

*Example 4A was corrected from 8,192 to 7,096 per issue #37.

---

## 10. Key Organizer Clarifications

| Issue | Topic                          | Ruling |
|-------|--------------------------------|--------|
| #3,#37 | Raster order has implicit reuse | Yes. Hardware always reuses strips between consecutive same-row tiles. PROBLEM.md corrected. |
| #10,#38 | Spatial vs k-split compute scaling | Spatial: full base_cost per tile (array fires fully). K-split: proportional (fewer cycles). |
| #14   | MatMul + PW fusion allowed?    | Yes, with single [w,h,k]. Must be "mathematically valid." |
| #32   | MatMul→PW with k>1?           | No. PW can't participate in k-steps. k must be 1 if PW produces boundary output. |
| #34   | Retained tensor lifetime       | Exactly one step. Freed after step k+1 completes. No multi-step accumulation. |
| #38   | Native granularity update      | All benchmarks updated to have same native granularity across all dimensions (square). |