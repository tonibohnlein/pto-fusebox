# 910B Grounded Cost Model + SpatialSchedule

This document describes the Ascend-910B cost model: the pto-isa-grounded
coefficients, the wave/LPT makespan, and the **grid-only** SpatialSchedule that
drives **both** the cube and the vector path. It extends `doc/910b_cost_model.md`
(backpropagation, memory feasibility, shared-input reuse) and supersedes its §4
roofline and §8 calibration.

The 910B is the **only** cost model: every term uses the grounded pto-isa
coefficients (per-direction bandwidths, fractal-cycle compute, L1↔L0 extract,
double-buffer floors). `cube_freq_hz` is the core clock; `cube_compute_cost` is a
calibration multiplier (default 1). The earlier single-`slow_memory_bandwidth` /
flat-`cube_compute_cost` competition path has been removed.

**One enumeration.** The model enumerates only `(parts_m, parts_n, split_k)`
**triples** over the SpatialSchedule grid — for the cube *and* the vector —
including the `(1,1)` whole-output region.

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
bw_GiBps)`, per direction.

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
K/32`. It is a cost, not a hard reject.

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
lat = Σ_phase roofline(phase)
roofline(p) = max(compute_mk[p], io[p])   # stage-2 rolled loop
            = compute_mk[p] + io[p]       # serial body/init/tail/finalize
compute_mk[p] = WaveComputeCycles(Σ_op∈p VecOpCompute(op), num_tiles, C)
```

**Wave-aware** like the cube (a balanced `num_tiles == C` grid, one wave, beats an
over-tiled count). `compute_mk` is the full-op work distributed over the tiles.

**Per-op compute — `VecOpCompute` (grounded, pto-isa `vec_tile_study`).**

- *Pointwise:* `slope·ceil(elems/epr)` + `head+tail` **once per back-to-back stream** (Fix 3:
  the perf-sim pays startup only when the VEC queue is empty, so a fused elementwise chain
  overlaps its startup — reductions / matmuls break the stream). `elems` = the op's largest
  tensor, tiling-invariant; `epr = vec_reg_bytes/dtype_bytes`.
- *Reduction (Fix 1):* a reduction is **not** a single `slope_reduce·repeat` op — it lowers
  to a barrier-separated **tree** of count-mode passes, so its cost tracks the **reduced
  axis**, not `ROWS·COLS`. Reduce-W (row reduction): `45·(K−1)+51`, `K=W/epr` — **ROWS-
  independent**. Reduce-H (col reduction, binary): `16·(H−1)+30·log₂H`. The old
  `slope_reduce·(ROWS·COLS/epr)` overcounted a tall row-reduce **up to 19×** (`vec_reduce`).

**Double-buffer floor (vector).** The `max()` overlap holds only for an emitted
rolled loop with at least two trips whose per-strip transfer spans one vector
register. Then load s+1 overlaps compute s; a short or sub-register loop cannot
ping-pong and uses `compute + io`. **Binary**: crossing both guards grants `max`
and nothing more — a larger strip gets no extra (unreal) overlap credit. This is
the vector analog of the cube's `K ≥ 32` floor, applied independently per phase.

**DMA-shape penalty.** GM↔UB moves a tile as `h` **strided segments** of `w`
contiguous elements (one tile-row). The DMA reaches peak bandwidth only when that
contiguous run spans ≥ one transfer burst; below it the per-descriptor setup
dominates (a narrow rectangle issues many tiny strided transfers). So `io` carries
a `max(1, vec_reg_bytes / (w·dtype_bytes))` factor: a **wide / full-width row-strip**
tile is unpenalized (factor 1), only sub-burst widths pay. This makes a
row/column-friendly layout **cost-favored** (not just tiebreak-favored) — e.g. a
256² pointwise lands on a full-width `[256, 6]` row-strip instead of a `[32, 64]`
rectangle. (pto-isa `BLOCK_BYTE_SIZE = 32` is the DMA block; contiguity below the
burst is descriptor-bound.) Threshold form so the dividing tiebreak still picks the
emit-friendly tile among efficient ones.

**Vector stream plan (Fix 2/A5).** A stack-local derivation per candidate owns the full pebble peak (including the
reduction source/work pair), materialize/stream choice, materialized/pointwise strips, reduction chunks, algorithm kind,
and serial-init/rolled/tail/finalize phases. Stage 2 duplicates source-DAG transient bands while
accumulator/assemble/online-stat bands stay persistent. Each phase has its own roofline; only a stage-2 rolled
loop earns `max(compute,DDR)`, while barriers and sub-register/short loops use `compute+DDR`. Boundary
input phase masks price stats-only, apply-only, and shared inputs separately (including repeated
broadcast chunks). `CostResult` retains only cost/config metadata; AutoFuse re-derives and consumes
the same plan for the final or explicitly forced configuration.

**Internal vs sink reduction.** A reduction **sink** (output `[H,1]`) pins the
reduced axis spatially and splits it across cores (§6). An **internal** reduction
(softmax: reductions feed a pointwise `div`, output full `[H,W]`) is tiled like
pointwise — the reduction is a recompute *cost*, not a tiling restriction — and
fills the cores by the fine sub-16 row tiling of §4.

---

## 8. The mixed roofline — `Ascend910BMixed::compute_cost`

A **mixed** subgraph fuses cube (matmul) and vector (pointwise/reduction) ops into one
kernel. On the 910B there is **no direct Acc→Vec pipe** — the cube↔vector handoff
round-trips **GM** (`ExpandMixedKernel` splits the kernel into AIC + AIV functions joined
by a GM-backed `tpush`/`tpop` FIFO; `SkewCrossCorePipeline` software-pipelines them). So the
two units run **concurrently**, overlapping compute with each other and with the GM traffic.
Admissible only when the model opts in (`allow_mixed` = `Ascend910BMixed`); the base model
routes a cube↔vector group as two separate kernels.

```
cube_stage = makespan( max(cube_mac,cube_extract) )       # AIC busiest-core wall — LPT grid / wave uniform (§5)
vec_stage  = makespan( Σ_op VecOpCompute(op) ) / 2         # AIV busiest-core wall — 2 cores/unit, ALL vec ops (§7)
ddr_lat    = max over the 4 GM ports (each par()-capped)   # cross-unit HBM contention
one_cube_tile = max(cube_mac,cube_extract)/num_tiles ;  one_vec_tile = Σ VecOpCompute /(2·num_tiles)

2-stage: wall = max( cube_stage + one_vec_tile,           # producer-bound → + one consumer drain
                     vec_stage  + one_cube_tile,           # consumer-bound → + one producer fill
                     ddr_lat )
3-stage: wall = max( cube_stage, vec_stage, ddr_lat )     # fill absorbed (output unit busy from t=0)

lat       = wall + rounds · kernel_fill_cost              # per-LAUNCH fill — added to BOTH shapes
rounds    = ceil(num_tiles / num_cube_cores)              # unit-rounds over the grid
eff_units = min(num_tiles, num_cube_cores)                # atomic resource = 1 cube : 2 vector unit
```

Grounded by pto-isa **`mixed_tile_study`** (7 experiments; the study is the *evidence*, this
section the *model*).

**Overlap `max` — real and near-universal.** The cube and vector stages overlap (the
producer skew), so the wall is a `max`, not a sum. **Every single-round-trip shape overlaps**:
`c→v` (epilogue), `v→c` (prologue), `v→c→v`, `c→v→c` (flash-decode — #1900's depth-2 per-stage
buffers let its two cube matmuls pipeline). A `max` is wrong only for a genuine cross-tile
**carry** or a **multi**-round-trip loop (the skew demotes those to Sequential = the sum, and
worse). The **multi-round-trip** case is excluded at admission — `create()` rejects a group
whose cube↔vector alternation depth exceeds 2 (the exact dual of the emit's `num_tpush!=1`
demote), so it never reaches this cost. The **tile-carry** case needs no separate guard: a
cross-tile reduction carry is prevented by the reduced-axis **pinning** in the tiling (§6) — the
mixed grid pins it (`parts_n=1`/`parts_m=1`) in **both** its cube- and vector-led branches, so a
**mid-kernel** reduction stays resident on one unit and, when its coupled band overflows UB,
**streams online** (fused flash attention `QK→softmax`; priced by §7's Fix-2 surcharge on the
pooled `vec_stage`) — a matmul's own contraction rides a *separate* axis, chosen by the tiling. A
split-K merge is instead priced as atomic-add traffic, so the `max` is only ever applied to
genuinely skewable groups. `vec_stage` pools
**all** Pointwise/Reduction ops (a `v→c→v`'s prologue *and* epilogue run on the same AIV pool).

**The symmetric fill (grounded: shape sweep `mixed_vcv`/`vc`/`cvc`).** A 2-stage wall is the
**symmetric cross-term** `max(cube_stage + one_vec_tile, vec_stage + one_cube_tile)`: the
bottleneck unit runs its full stage plus **one tile of the other** — the un-overlapped fill
(the output unit's first-tile wait) or drain (the last tile after the producer). Matches the
sim to ~1 cycle (`v→c`) / ~2.7% (`c→v`). Because it lives *inside* the `max`, the fill is
**absorbed** when DDR-bound or when the other unit dominates — so an imbalanced fusion (matmul
+ tiny epilogue) pays only one **tiny** non-bottleneck tile, never a full cube tile. A
**3-stage** kernel (`v→c→v`, `c→v→c`) has the output unit already running an earlier stage —
busy from `t=0` — so the fill is **0** (plain `max`). Detection is **structural**, not a count:
the fill is absorbed iff the sink unit (the boundary output's producing unit, `is_mm_out`) has
an *early-stage* op whose input cone is same-unit + boundary — independent of the opposite unit
(a `v→c→v` prologue, a `c→v→c` first matmul). Counting sink-unit ops is wrong: a same-unit tail
(`c→v→v`) has >1 sink op yet still idles at `t=0`, so it pays the 2-stage fill.

Absorption additionally requires **`num_tiles >= 2`**: a single tile has no second tile to skew
against (the sweep's `NTILES=1` row measures `overlap_factor 0.00`), so a 3-stage kernel there
takes the cross-term too — which at one tile collapses to the sequential sum `cube_stage +
vec_stage`. This matters for the low-batch flash-decode corner (whole output as one tile), which
the plain `max` would otherwise credit full overlap the hardware runs serially. *Mid-band
caveat:* `one_*_tile` does not shrink while `rounds == 1`, so partial overlap between 2 and
`num_cube_cores` tiles is slightly over-credited when compute-bound — a bounded level-1
imprecision; only the unambiguous, sim-contradicted `NTILES=1` case is corrected.

**Four-port DDR — `max`, not sum, each HBM-capped.** The GM ring is four independent per-unit
pipes that **overlap**, so `ddr_lat` is the `max` over them — not the summed
`ddr_bytes · bc.reload` of the old flat model:

| port | traffic | cyc/byte · peak |
| ---- | ------- | --------------- |
| `GM→L1` (cube reads) | operand reload (§3) + vec→cube crossing reads | `bc.reload` · `bw_gm_l1` |
| `GM→UB` (vector reads) | vector boundary inputs + cube→vec crossing reads | `bc.ub_in` · `bw_gm_ub` |
| `L0C→GM` (cube writes) | cube→vec crossing writes + a MatMul-sink output | `bc.store` · `bw_l0c_gm` |
| `UB→GM` (vector writes) | vec→cube crossing writes + a vector-sink output | `bc.ub_out` · `bw_ub_gm` |

A cube↔vector crossing intermediate round-trips **write + read on two SEPARATE overlapping
ports** (not `2×` summed onto one). Fusion does **not** make it free — it still round-trips GM
— but write and read ride distinct pipes. Each port divides across the active units and caps at
the HBM ceiling: `port_lat = bytes · cyc_per_byte / par(eff_units, peak)` — the **same `par()`**
the cube/vector paths use (§3, §7). Grounded by **`mixed_ddr_bound`** (single-core GM subsumed
into the per-unit stages — `max(cube,vec,ddr) == max(cube,vec)` across a K-sweep) and
**`mixed_contention`** (multi-core: the cube `GM→L1` and vector `GM→UB` **read** ports are each
throttled to `min(peak, 900/B)` by the shared HBM read ceiling — a per-port **rate** cap, not a
summed byte budget — both collapsing to `900/B` past the knee, matching `par()` to **0%**).

**Scope of the four-port `max` (two untested corners).** (i) `mixed_contention` validates
**read↔read** pooling only (`GM→L1` + `GM→UB`); the write ports (`L0C→GM`, `UB→GM`) and
read↔write overlap follow the *same* per-port `par()` model but are **untested** — if real HBM
shares R/W bandwidth, a write-heavy *and* read-heavy mixed kernel could under-count. (ii) mixed
`GM↔UB` traffic assumes **wide** vector tiles: unlike §7 it does **not** apply the DMA-shape
penalty (`max(1, vec_reg_bytes / (w·dtype_bytes))`), so a narrow-tile mixed epilogue is
under-charged on those two ports. Both are second-order for the balanced tiles the partitioner
emits; revisit with a write-contention experiment / a narrow-tile mixed sweep.

**Fusion economics (910B).** Fusion saves the **overlap** (`max` vs the separated `cube + vec`)
and one kernel launch — **not** the DDR (the intermediate round-trips GM either way). So it
wins when the vector work is substantial or the kernel is memory-bound, and is neutral-to-losing
for a compute-bound matmul with a trivial epilogue (small overlap saving).

**Sink split-K (cube sink, single matmul).** When the sink IS the matmul (`v→c`, `v→v→c`) it may
split-K exactly like the separated cube: split the contraction across idle cube cores, atomic-add
`S` partials (no merge barrier — the cores stay independent), the vector prologue overlapping
orthogonally. So `parallel_split ≥ 1` for a single-matmul cube sink; the sweep recruits
`eff_cube = min(num_tiles·S, num_cube_cores)` cores and grows the L0C→GM write-back to `S`
partials, `S=1` reproducing the spatial-only cost. Split-K stays **off** (`= 1`) for a **vector
sink** (`c→v` — the epilogue needs the fully-reduced C, so the matmul is never the sink) and for a
**multi-matmul** cube sink (`c→v→c` — only the sink may split, never a mid-kernel matmul).

---

## 9. Feasibility (recap; full detail in base doc §3)

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

## 10. Enumeration tiebreak — `best_cost`

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

## 11. Worked examples (grounded; standalone `mlsys`)

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

## 12. Known limitations / open calibration

- **Calibration constants (perf-sim-grounded, device-eval-pending).** The reduction tree
  (`45/51/16/30`, §7 Fix 1), streaming surcharge (Fix 2), and per-op startup (Fix 3, once-per-
  stream via `pw_stream_start`) are perf-sim-grounded (`pto-isa vec_tile_study`, R²≈1.0) but
  device-eval-pending (count-mode flat-per-pass is itself coarse vs real HW). The vector
  double-buffer threshold (`2·vec_reg_bytes`) and DMA-shape burst (`vec_reg_bytes`) remain
  reasoned bounds, not measured. The stream-break heuristic (reductions/matmuls reset the chain)
  approximates the true VEC-queue-empty condition; exact only when the op order matches the emit.
- **Per-tile compute overhead.** The vector *compute* is still tiling-invariant
  (full-op cost) — it charges no per-tile SIMD pipeline-fill. So among same-width
  tiles, "favor larger tiles" is a *tiebreak* (§10.5), not cost-driven. (The DMA
  shape *is* now cost-driven, via §7.)
- **Split-K is model-ahead of the emit (base *and* mixed) — behind a buildable flag.** The solver
  credits `parallel_split > 1` for a lone matmul and a single-matmul cube sink (`v→c`), which the
  AutoFuse auto-emit does not realize yet. `Problem::allow_model_ahead_split_k` gates it: `true`
  (default) credits the split; `false` forces `S=1` for **both** base and mixed, so `best_cost`
  never picks an unemittable split (`CostResult::uses_model_ahead_split_k` flags one). Flip when Phase-C lands.
- **Mixed cube stage — makespan + floors.** The cube/vector stages route through the base
  `LptMakespan` (grid) / `WaveComputeCycles` (uniform) — the busiest-unit makespan, not the flat
  `eff_units` average — so an imbalanced grid no longer under-predicts its biggest region, and the
  double-buffer floor is ported (a thin-K cube `output_K/S < 32` serializes compute with its GM
  reload). The **cube** region work is recomputed per region — `max(Σ MAC, Σ extract)` at the
  region extent — so it captures fractal/extract padding (the per-region `ceil`) and per-region
  MAC-vs-extract; the **vector** region work stays an output-area fraction (a documented
  approximation — a region-aware `VecOpCompute` is the follow-up). A multi-matmul group (`c→v→c`)
  uses the sink region extent for every matmul — a bounded approximation. *Scope:* a
  matmul→reduction sink gets neither cube split-K (matmul-sink-gated) nor the §6
  reduced-axis split, so few-row reductions under-fill.

---

## 13. Source map (`src/core/`)

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
| mixed roofline (overlap max + symmetric fill + 4-port par ddr + cube-sink split-K) | `Ascend910BMixed::compute_cost` |
| feasibility | `derive_exec` / `cube_peak_l1` / `vector_stream` (mixed: both) |
| enumeration + tiebreak | `Ascend910BCost::best_cost` |
