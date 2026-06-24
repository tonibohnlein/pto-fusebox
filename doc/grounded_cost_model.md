# 910B Grounded Cost Model + SpatialSchedule

This document outlines the **new** Ascend-910B cube cost model: the pto-isa-grounded
coefficients, the wave/LPT makespan, and the non-uniform SpatialSchedule grid. It
extends `doc/910b_cost_model.md` (backpropagation, memory feasibility, shared-input
reuse — unchanged) and supersedes its §4 roofline and §8 calibration.

Everything is gated on `cube_freq_hz > 0`. When unset, every term below collapses
to the legacy behavior (single `slow_memory_bandwidth`, flat `cube_compute_cost`,
no extract), so competition / single-context instances are byte-for-byte unchanged.

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

i.e. `MakeByteCost()` precomputes a per-direction `cycles_per_byte =
cube_freq_hz / (2^30 * bw_GiBps)`. Legacy fallback: `1 / slow_memory_bandwidth`
for every direction.

**Cube fractals** (pto-isa `mad`): a matmul of an `M×N` output with contraction `K`
issues

```
repeats = ceil(M/16) * ceil(N/16) * ceil(K/kF)      kF = 32 / dtype_bytes   (fp32:8, fp16/bf16:16)
cube_cycles = repeats * cyc(dtype)                  cyc = 2 (fp32) else 1
```

so fp32 costs 4× fp16 (kF halves AND cyc doubles). `cube_compute_cost` is a
calibration multiplier (grounded value: 1). → `CubeMacCycles()`.

---

## 2. The cube core's work is hierarchical and double-buffered

Producing an output tile needs both the **cube MACs** (Matrix pipe) and the
**L1→L0 operand extract** (MTE1 pipe). They run concurrently (double buffering), so
the per-region work is the steady-state pipeline, NOT just the MACs.

**Extract** (`CubeExtractCycles`): the L1→L0 operand reload, reusing the L0 base
tile `(l0_tile_m, l0_tile_n) = (128, 256)` the same way `cube_operand_reload`
reuses the L1 tile one level up:

```
lhs_bytes = M*N*K / l0_tile_n * dtype_bytes      (lhs reloaded once per L0 N-block)
rhs_bytes = M*N*K / l0_tile_m * dtype_bytes      (rhs reloaded once per L0 M-block)
extract_cycles = lhs_bytes * cyc_per_byte(L1→L0A) + rhs_bytes * cyc_per_byte(L1→L0B)
```

**Double-buffer overlap (Phase D)** over `L` L0-MAD steps:

```
L = ceil(m/128) * ceil(n/256) * ceil(K/64)                 (baseK = 64)
T_region = ( MAC + extract + (L-1)*max(MAC, extract) ) / L
```

- `L = 1` (one L0 tile, tiny region): `MAC + extract` — no steady state to overlap.
- `L ≫ 1`: → `max(MAC, extract)` — full ping-pong (the common case).

**Chained regions**: the matmuls run SEQUENTIALLY (the intermediate feeds the sink),
so their per-matmul `T_region` **sum** (not max-of-summed-totals).

**Validation**: with the 128×256 L0 tile the extract lands at ≈ **0.6× the MACs** for
a square fp16 GEMM — matching pto-isa's measured 7680³ run (TEXTRACT 63% / Cube
80.6% pipe utilisation). And fp32 = **2× fp16** on a 2048³ → correctly reload-bound
(captures "input feed dominates", TLOAD 98.4%).

---

## 3. The roofline — `compute_cost`

```
lat = max( compute_makespan , ddr )
```

**DDR** is the shared HBM floor, charged PER DIRECTION (reload and store ride
different ports at different bandwidths), discounted by active-core saturation:

```
ddr = ( reload * cyc_per_byte(GM→L1) + out_store * cyc_per_byte(L0C→GM) ) * sat(active)
sat(c) = max(1, n_cores / c)          # too few DMA engines underfill HBM
reload = cube_operand_reload(cfg)     # distribution-aware MNK*(1/w + 1/h), per (tensor,role)
```

`cfg.w / cfg.h` carry the **physical (max) region extent** in both uniform and grid
mode, so `cube_operand_reload` / `fits_on_chip` / `cube_peak_l1` are unchanged.

**`compute_makespan`** depends on the spatial schedule (§4–§5).

---

## 4. SpatialSchedule — the output-tensor tiling

The spatial schedule **is the enumerated tiling of the sink output tensor**.
Operand and intermediate tiles are backpropagated from it (§5 of the base doc).

- **Uniform mode** (legacy): `TileConfig{w,h,k}` — an exact-divisor tile; the region
  count is `floor(out_W/w) * floor(out_H/h)`. 16-aligned divisors of a
  power-of-two dim give only power-of-two counts — never a multiple of 24.
- **Grid mode**: `TileConfig{w,h,k, parts_m, parts_n}` — the output is partitioned
  into `parts_m × parts_n` regions whose extents are an **even split of the
  16-fractal counts** (`partition_axis`): `F = ceil(dim/16)` fractals split so
  `rem = F % parts` regions get `(base+1)` fractals and the rest get `base`. Regions
  differ by ≤ 1 fractal, so an axis has at most **two** extents and a grid at most
  **four** region shapes. `w,h` = the physical (max) extent.

**Why**: a grid can land exactly `n_cores` balanced regions where divisor tiling
can't. e.g. 1024² → `4×6` grid, N-extents `[176,176,176,176,160,160]` — 24
non-empty 16-aligned regions covering 1024 columns with **no padding**.

**Candidates** (`grid_cand_`, cube path): factor pairs of `n_cores` and `2*n_cores`
(both orientations), bounded by each axis's fractal count, gated on all matmuls
sharing the sink M (`tensors[o].height == out_H_`). A DAG that tiles M differently
falls back to uniform.

**Chained backpropagation**: the sink M-partition slices every matmul's rows; a
consumed intermediate is the next matmul's contraction, so it is a full-width
`[m_ext, N_int]` row-band recomputed once per N-region (reproducing the `num_tw`
recompute factor).

---

## 5. The makespan — wave (uniform) / LPT (grid)

The independent work units are `spatial_regions × split-K_partials`.

**Uniform** — equal units, so the makespan is the wave model:

```
U = num_tiles * S
T_compute = ceil(U / C) * (W_total / U)          # NOT W/min(U,C): U not a multiple of C costs extra
```

e.g. 32 units on 24 cores → `ceil(32/24)=2` waves → `W/16`, not the optimistic `W/24`.

**Grid** — regions are unequal (±1 fractal), so LPT (longest-processing-time-first):

```
LptMakespan: per-region work T_region(m_ext, n_ext)   (§2, sink + intermediate bands)
             enumerate the ≤4 shapes with their counts, sort descending,
             assign each to the least-loaded of C cores; makespan = busiest core
```

With `parts == n_cores` this is one wave → the largest region; the LPT also captures
the ±1-fractal imbalance and multi-wave grids.

---

## 6. Split-K (secondary core-filler)

Splits the sink contraction into `S` per-tile partials (atomic-add merge). On a grid
it is LPT-consistent: `LptMakespan(..., ksplit=S)` splits each region's K into S
partials (`P*Q*S` units of `work/S`) — honest about the region imbalance, where the
equal-unit wave would be optimistic. `S` ranges over divisors of `K/16`.

Merge barrier (`ddr_atomic_add`): with SetAtomicAdd, just the S partial writes
(sat-discounted); without, a serial DDR read-back + sum (∝ S).

**Tiebreak** (`best_cost`): among equal-latency configs, prefer **fewer split-K
partials** — a balanced grid that fills the cores at the same latency beats a
K-split (less merge, no atomic serialization, simpler emit). Split-K only wins when
**strictly** faster.

---

## 7. Worked examples (grounded; standalone `mlsys`)

| shape (fp) | result | regions | waves | eff_par |
| --- | --- | --- | --- | --- |
| 2048² | GRID 4×6, tile `512×352`, split=1 | 24 | 1 | ~24 |
| 512²  | GRID 4×6, tile `128×96`, split=1  | 24 | 1 | ~24 |
| 1536² | uniform 3×4 + split-2             | 24 | 1 | 24 |
| 64²   | uniform 4×4 + split-2 (only 4×4 fractals exist) | 16 | 2 | 16 |

Before the grid, 2048²/512² were stuck at a power-of-two tile count (16) → eff_par
16. The grid lands 24 balanced regions in one wave.

---

## 8. Known limitations / open calibration question

**The split-K runaway.** For shapes the rectangular grid can't balance into 24 —
`GEOM` (128² = 8×8 fractals → grid regions differ 2×) and `KVAR` (a chained group
whose huge-K=8192 internal dominates) — the solver can reach eff_par ~24 by
splitting into many small **equal** K-partials that pack better than the coarse
grid. This is **latency-optimal under the model but emit-heavy** (many atomic-add
partials), and the per-partial launch/sync overhead is **not yet charged** (counting
it via `kernel_fill` work-units perturbs fusion decisions — reverted). Two
`ascend_910b_test` cases (`GEOM`, `KVAR`) assert the older clean-spatial-fill /
exact-`W/cores` behavior and are pending reconciliation against this.

This is the main item to settle in the example/sanity-check pass: whether to charge
a (calibrated) per-partial overhead so the clean grid wins, or accept the split-K
makespan as the model's answer and update those expectations.

---

## 9. Source map

| concept | location (`src/core/`) |
| --- | --- |
| grounded coefficients, fields | `types.h` (`Problem`), `io.cpp` |
| per-direction byte cost | `ascend910b_cost.cpp` `MakeByteCost` |
| cube MACs / extract | `CubeMacCycles` / `CubeExtractCycles` |
| wave makespan | `WaveComputeCycles` |
| grid partition | `types.h` `AxisPartition` / `partition_axis` |
| LPT makespan (+ ksplit) | `LptMakespan` |
| grid candidates | `Ascend910BCost::create` (`grid_cand_`) |
| roofline + split-K + Phase D | `Ascend910BCost::compute_cost` |
| enumeration + tiebreak | `Ascend910BCost::best_cost` |
