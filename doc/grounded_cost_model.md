# 910B Grounded Cost Model + SpatialSchedule

This document describes the Ascend-910B cost model: the pto-isa-grounded
coefficients, the wave/LPT makespan, and the **grid-only** SpatialSchedule that
drives **both** the cube and the vector path. It extends `doc/910b_cost_model.md`
(backpropagation, memory feasibility, shared-input reuse) and supersedes its §4
roofline and §8 calibration.

Everything grounded is gated on `cube_freq_hz > 0`. When unset, every term
collapses to the legacy behavior (single `slow_memory_bandwidth`, flat
`cube_compute_cost`, no extract, double-buffer floors inert), so
competition / single-context instances are byte-for-byte unchanged.

**One enumeration.** On the 910B path (`num_cores > 1`) the model enumerates only
`(parts_m, parts_n, split_k)` **triples** over the SpatialSchedule grid — for the
cube *and* the vector. Uniform exact-divisor tiles survive only as the
single-context (`num_cores ≤ 1`) legacy fallback.

---

## 1. Units and grounded coefficients

**Work is in core CYCLES.** Source: pto-isa `include/pto/costmodel/arch_config.hpp`
and `cce_costmodel_cube.hpp` (the A2/A3 reference cost model).

Per-direction effective bandwidths (GiB/s) and the clock:

| direction | field | GiB/s |
| --- | --- | --- |
| GM→L1  (operand reload / TLOAD)   | `bw_gm_l1`  | 135.0 |
| L0C→GM (output store / TSTORE)    | `bw_l0c_gm` | 70.0 |
| L1→L0A (lhs extract / TEXTRACT)   | `bw_l1_l0a` | 441.0 |
| L1→L0B (rhs extract)              | `bw_l1_l0b` | 220.5 |
| GM→UB  (vector load)              | `bw_gm_ub`  | 100.9 |
| UB→GM  (vector store)             | `bw_ub_gm`  | 188.46 |
| clock | `cube_freq_hz` | 1.85e9 |

A transfer of `bytes` over a direction costs, in **cycles** (pto-isa
`EstimateBandwidthCycles`):

```
transfer_cycles = (bytes / 2^30) / bw_GiBps * cube_freq_hz
```

i.e. `MakeByteCost()` precomputes `cycles_per_byte = cube_freq_hz / (2^30 *
bw_GiBps)`. Legacy fallback: `1 / slow_memory_bandwidth` for every direction.

**Cube fractals** (pto-isa `mad`): a matmul of an `M×N` output with contraction `K`:

```
repeats = ceil(M/16) * ceil(N/16) * ceil(K/kF)      kF = 32 / dtype_bytes   (fp32:8, fp16/bf16:16)
cube_cycles = repeats * cyc(dtype)                  cyc = 2 (fp32) else 1
```

so fp32 costs 4× fp16. `cube_compute_cost` is a calibration multiplier (grounded
value 1). → `CubeMacCycles()`.

**Vector repeats** (pto-isa `cce_costmodel_vector_compute.hpp`): one vector op over
`elems` elements costs `head + slope·ceil(elems/epr) + tail`, where the SIMD repeat
is `REPEAT_BYTE = 256 B` (`epr = vec_reg_bytes / dtype_bytes` = 128 fp16 / 64 fp32),
`slope` is `vec_slope_pw` (pointwise) or `vec_slope_reduce` (reduction, ~14× pw).
→ the vector branch of `compute_cost`.

---

## 2. The cube core's work is hierarchical and double-buffered

Producing an output tile needs both the **cube MACs** (Matrix pipe) and the
**L1→L0 operand extract** (MTE1 pipe). They run concurrently (double buffering),
so per-region work is the steady-state pipeline, NOT just the MACs. (This is the
*inner* L1↔L0 double-buffer; the *outer* DDR↔L1 reload double-buffer — and its
roofline floor — is in §3.)

**Extract** (`CubeExtractCycles`): the L1→L0 operand reload, reusing the L0 base
tile `(l0_tile_m, l0_tile_n) = (128, 256)`:

```
lhs_bytes = M*N*K / l0_tile_n * dtype_bytes      (lhs reloaded once per L0 N-block)
rhs_bytes = M*N*K / l0_tile_m * dtype_bytes      (rhs reloaded once per L0 M-block)
extract_cycles = lhs_bytes * cyc_per_byte(L1→L0A) + rhs_bytes * cyc_per_byte(L1→L0B)
```

**Double-buffer overlap (Phase D)** over `L = ceil(m/128)*ceil(n/256)*ceil(K/64)`
L0-MAD steps:

```
T_region = ( MAC + extract + (L-1)*max(MAC, extract) ) / L
```

`L = 1` (tiny region): `MAC + extract` (no steady state). `L ≫ 1`: → `max(MAC,
extract)` (full ping-pong, the common case). **Chained regions** run sequentially,
so per-matmul `T_region` **sum**.

**Validation**: with the 128×256 L0 tile the extract lands at ≈ 0.6× the MACs for a
square fp16 GEMM (matching pto-isa's 7680³: TEXTRACT 63% / Cube 80.6%); fp32 = 2×
fp16 on a 2048³ → correctly reload-bound (TLOAD 98.4%).

---

## 3. The cube roofline — `compute_cost`

```
lat = max( compute_makespan , ddr )    # when the reload can double-buffer
    = compute_makespan + ddr           # otherwise (no overlap)
```

**Double-buffer floor.** The `max()` overlap is physical only when the operand
reload can **ping-pong** — the per-core contraction halvable into ≥2 seq-K
sub-strips (`per_core_K ≥ 32`; the emit's implicit halving needs that). A tiny
contraction, or an over-aggressive split-K with `K/S < 32`, can't overlap →
reload and compute **serialize** (`compute + ddr`). This caps split-K at `S ≤
K/32`. It is a cost, not a hard reject. Grounded only; legacy keeps `max`.

**Double-buffering does NOT reserve L1/UB.** The two ping-pong buffers *together*
are the pool, so `derive_exec` / `vector_stream` use the **full** `l1_capacity` /
`vec_capacity`. The model keeps the full seq-K strip; the emit halves the
*per-load* k (`kk → 2×kk/2`, both in the full L1). No `double_buffer` flag.

**DDR** is the shared HBM floor, charged PER DIRECTION, discounted by active-core
saturation:

```
ddr = ( reload * cyc_per_byte(GM→L1) + out_store * cyc_per_byte(L0C→GM) ) * sat(active)
sat(c) = max(1, n_cores / c)          # too few DMA engines underfill HBM
reload = cube_operand_reload(cfg)     # distribution-aware MNK*(1/w + 1/h), per (tensor,role)
```

`cfg.w / cfg.h` carry the **physical (max) region extent** in grid mode, so
`cube_operand_reload` / `fits_on_chip` are unchanged. The vector roofline is §7.

---

## 4. SpatialSchedule — the grid (cube *and* vector)

The spatial schedule **is the enumerated tiling of the sink output tensor**.
Operand and intermediate tiles are backpropagated from it (base doc §2). Each
candidate is a **`(parts_m, parts_n, split_k)` triple** — one enumeration of all
three core-fill levers, evaluated as a fixed config (no internal sweep).

**`partition_axis(dim, parts, granule)`** splits an axis into `parts` near-equal
regions of `granule`-aligned extent: `F = ceil(dim/granule)`, `rem = F % parts`
regions get `(base+1)` blocks, the rest `base`. Regions differ by ≤1 block → an
axis has ≤2 extents, a grid ≤4 region shapes. `w,h` = the physical (max) extent.

**Granularity is per-path** (the key distinction):

| path | rows (height) | contiguous (width) | why |
| --- | --- | --- | --- |
| cube  | 16 | 16 | the 16×16 MAC fractal (hardware, both axes) |
| vector | **1** | **16** | rows have no fractal constraint; width = the 32-byte `BLOCK_BYTE_SIZE` DMA block |

The 16-on-both is a *cube* requirement — `is_valid_tiling` already cube-gates the
16 check, so sub-16 vector tiles are valid. Fine row tiling is what lets a few-row
reduction fill all `C` cores from the grid (a `[W, 128]` softmax tiles to `[W, 3]`,
48 regions; the 16-fractal grid would cap at 8).

**Candidates** (`gen_grid(C, maxP, maxQ, s_vals)`): spatial `P·Q` over the
**divisors of {C, 2C}** (including `(1,1)`), bounded by each axis's
granule count; `s_vals` is the split set. The **work units `P·Q·S` range freely**
— filling all `C` cores is a strong SOFT preference, but the cost (merge barrier vs
streaming gain) drives it. So a power-of-two shape that can't form `C` spatial
regions still fills the cores via split (64² cube = 4×4 fractals → `(4,3) × S=2`).

- **Cube**: `C = num_cube_cores`, `s_vals = divisors(kfrac)` (the sink split-K).
  Gated on all matmuls sharing the sink M; else falls back to uniform.
- **Vector**: `C = num_vector_cores`; a reduced axis pins to 1 part (it can't be
  spatially tiled — the whole row/col must be present to reduce). `s_vals =
  divisors(2C)` capped by the reduced extent **iff the sink is a reduction**, else
  `{1}` (an internal reduction is tiled like pointwise — see §7).

**Chained backpropagation** (cube): the sink M-partition slices every matmul's
rows; a consumed intermediate is the next matmul's contraction, so it is a
full-width `[m_ext, N_int]` row-band recomputed once per N-region.

---

## 5. The makespan — wave (uniform) / LPT (grid)

The independent work units are `spatial_regions × split-partials`.

**Wave** (uniform / equal units, and the vector path):

```
U = num_tiles * S
T = ceil(U / C) * (W_total / U)          # NOT W/min(U,C): U not a multiple of C costs extra
```

e.g. 32 units on 24 cores → `ceil(32/24)=2` waves → `W/16`, not `W/24`.

**LPT** (cube grid — regions are unequal by ±1 fractal):

```
LptMakespan: enumerate the ≤4 region shapes with their counts (and ksplit S),
             sort descending, assign each to the least-loaded of C cores;
             makespan = busiest core
```

With `parts == C` this is one wave → the largest region; it also captures the
±1-fractal imbalance and multi-wave grids.

---

## 6. The split taxonomy

Two distinct k-splits — do not conflate them:

| split | scope | cost |
| --- | --- | --- |
| **serial seq-k** (single-core streaming of the contraction) | **universal** — every matmul | free; `derive_exec` sizes it to fit L1 |
| **parallel split** (across cores, cross-core merge) | **SINK-ONLY** — the boundary-output op | streaming + an additive merge barrier |

The parallel split is sink-only because the cross-core merge (atomic-add / DDR
reduction) is clean only at the boundary output. The cube and vector parallel
splits are **analogous**:

**Cube split-K** (sink matmul). `S | kfrac` (`kfrac = output_K/16`), capped by the
double-buffer floor `K/S ≥ 32`. On a grid, LPT-consistent: `LptMakespan(...,
ksplit=S)` splits each region's K. Merge barrier (`ddr_atomic_add`): with
SetAtomicAdd just the S partial writes (sat-discounted); without, a serial DDR
read-back + sum (∝ S).

**Vector reduced-axis split** (reduction **sink**). The `[H,1]`/`[1,W]` partials
reduce across cores. `S` lets `P_spatial · S` fill the cores when the non-reduced
axis alone can't (a few-row `rowmax`). Same two merge regimes via `ddr_atomic_add`.

The single-core seq-k is **derived** (`derive_exec`), never enumerated.

---

## 7. The vector roofline — `compute_cost` (vector branch)

```
compute_mk = WaveComputeCycles( Σ_op (head + slope·ceil(elems/epr) + tail), num_tiles, C )
lat        = max( compute_mk , io * sat(eff) )    # when the tile can double-buffer
           = compute_mk + io * sat(eff)           # otherwise
io = Σ boundary tensors * cyc_per_byte(GM↔UB)
```

**Wave-aware** like the cube (a balanced `num_tiles == C` grid, one wave, beats an
over-tiled count). `compute_mk` is the full-op work distributed over the tiles.

**Double-buffer floor (vector).** The `max()` overlap holds only when the per-core
tile streams in **≥2 SIMD-repeat chunks** (`tile_bytes ≥ 2·vec_reg_bytes`), so the
load of chunk s+1 overlaps the compute of chunk s. A smaller tile can't ping-pong
→ load and compute **serialize** (`compute + io`). **Binary**: crossing the
threshold grants `max` and nothing more — a larger tile gets no extra (unreal)
overlap credit. The vector analog of the cube's `K ≥ 32`.

**Reduction recompute.** When a reduction's coupled band overflows UB
(`vector_peak_ub > vec_capacity`), the feasible schedule STREAMS the reduced axis
and recomputes the reused ephemerals once per pass: `compute *= n_passes`,
`io *= n_passes`, `n_passes = #reductions + 1` (softmax = 3). Pessimistic upper
bound.

**Internal vs sink reduction.** A reduction **sink** (output `[H,1]`) pins the
reduced axis spatially and splits it across cores (§6). An **internal** reduction
(softmax: reductions feed a pointwise `div`, output full `[H,W]`) is tiled like
pointwise — the reduction is a recompute *cost*, not a tiling restriction — and
fills the cores by the fine sub-16 row tiling of §4.

---

## 8. Feasibility (recap; full detail in base doc §3)

Feasibility **never depends on the spatial tile** — which is *why* the grid can
replace uniform on the 910B path.

- **Cube** → `derive_exec ≠ INT64_MAX`: the red-blue pebble peak over the fixed DFS
  order — live ephemeral bands + per-op **greedy seq-k** operand strips fit the
  **full** L1. The output drains L0c→DDR (**not** charged to L1); **L0c sizing is
  deferred to `AutoTileMatmulL0`**. Infeasible ⇔ no fitting k exists; the derived
  k is written to `config.k` for the emit.
- **Vector** → `vector_stream(cfg).chunk > 0`: the tile streams through UB to a
  min-chunk (free for pointwise, recompute-costed for a reduction).
- **Mixed** (cube+vector) → both: `derive_exec` (L1) *and* `vector_stream` (UB);
  the cube↔vector crossing rides the GM ring, not a resident band.

---

## 9. Enumeration tiebreak — `best_cost`

Among equal-latency configs, lexicographic:

1. **fewer parallel-split partials** — a balanced grid that fills the cores beats a
   split (less merge, no atomic serialization, simpler emit).
2. **lower DDR traffic** — matmul reuse; flat for pointwise.
3. **more cores used** — fill the unit.
4. **evenly-dividing tile** — a tile whose extents divide the output (identical
   regions) lowers cleanly; an imbalanced grid (±1-block extents the emit can't
   realize) is used only when **strictly** faster (power-of-two / few-row fills
   with no dividing C-tiling). Uniform tiles always divide; a grid sometimes does.
5. **larger tile area** — vectorization / least per-tile overhead.
6. **larger k** — fewer L1 passes.

---

## 10. Worked examples (grounded; standalone `mlsys`)

| shape (fp16) | path | result |
| --- | --- | --- |
| 256³ / 1024³ / 4096³ | cube | `3×4` grid × split-2 → 24 units, 1 wave |
| 2048³ | cube | `3×4` grid × split-2 (bigger tiles, ~8% < `4×6` spatial) |
| 64² (K=64) | cube | only 4×4 fractals; floor caps `S ≤ 2` → 16 units (DDR-bound) |
| 16² (K=512) | cube | `(1,1) × split-32` → fills 24 purely via split-K |
| `[512,512]` pointwise | vector | balanced grid → 48 regions, 1 wave |
| `[W,128]` softmax (small rows) | vector | `[W, 3]` fine-row grid → 48 regions (sub-16 rows) |
| `[1024,512]→[1,512]` rowmax | vector | spatial rows fill; few-row → reduced-axis split |

---

## 11. Known limitations / open calibration

- **Vector double-buffer threshold** (`2·vec_reg_bytes`) and the reduction
  **`n_passes` recompute** are the least-calibrated terms — both are reasoned
  bounds, not measured.
- **Per-tile overhead.** The vector compute is tiling-invariant (full-op cost); it
  does not charge a per-tile SIMD pipeline-fill, so "favor larger tiles" is a
  *tiebreak* (§9.5), not cost-driven. Fine once tiles can be arbitrarily fine.
- **Emit gap (downstream).** The solver chooses grids and split-K; the AutoFuse
  cube **emit** does not yet realize split-K for a lone matmul. Solver-side
  complete; this is Phase-C codegen work, independent of the cost model.

---

## 12. Source map (`src/core/`)

| concept | location |
| --- | --- |
| grounded coefficients, fields | `types.h` (`Problem`), `io.cpp` |
| per-direction byte cost | `ascend910b_cost.cpp` `MakeByteCost` |
| cube MACs / extract | `CubeMacCycles` / `CubeExtractCycles` |
| wave makespan | `WaveComputeCycles` |
| grid partition (granule) | `types.h` `AxisPartition` / `partition_axis` |
| LPT makespan (+ ksplit) | `LptMakespan` |
| grid candidates (triples, granularity) | `Ascend910BCost::create` (`gen_grid`, `grid_gran_*`) |
| cube roofline + split-K + Phase D | `Ascend910BCost::compute_cost` (matmul branch) |
| vector roofline + double-buffer floor + reduced split | `compute_cost` (vector branch) |
| feasibility | `derive_exec` / `cube_peak_l1` / `vector_stream` |
| enumeration + tiebreak | `Ascend910BCost::best_cost` |
