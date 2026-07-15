# 910B Grounded Cost Model + SpatialSchedule

This document describes the Ascend-910B cost model: the pto-isa-grounded
coefficients, the queue/LPT makespan, the shared SpatialSchedule grid, and the
engine-specific `VectorStreamPlan` / `CubeSchedulePlan` derived for each candidate. It extends
`doc/910b_cost_model.md` (backpropagation, memory feasibility, shared-input reuse) and supersedes
its §4 roofline and §8 calibration.

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
| L0C→L1 (internal FIXPIPE drain)  | child `L0MatmulPlan` | 128.0 |
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

The L0C→L1 value is PTO A2/A3 `PipeKey::L0C_TO_L1` from `arch_config.hpp`. The shared L0
planner stores its equivalent 74.3 bytes/cycle. `chain_fused_kernel.cpp` and the PTO FIXPIPE
reference ground the operation itself: the completed FP32 accumulator narrows and drains directly
to a BF16/FP16 L1 Mat tile. There is no intermediate L1→GM leg in the buildable cube schedule.

**Cube fractals** (pto-isa `mad`): a matmul of an `M×N` output with contraction `K`;
`dtype` is the operand precision (not the often-FP32 accumulator/output):

```
repeats = ceil(M/16) * ceil(N/16) * ceil(K/kF)      kF = 32 / dtype_bytes   (fp32:8, fp16/bf16:16)
cube_cycles = repeats * cyc(dtype)                  cyc = 2 (fp32) else 1
```

so fp32 costs 4× fp16. `cube_compute_cost` is a calibration multiplier (grounded
value 1). → `CubeMacCycles()`.

**Vector primitives** (PTO A2/A3): a flat instruction over `elems` elements costs
`head + slope·ceil(elems/epr) + tail`, where `REPEAT_BYTE = 256 B`
(`epr = vec_reg_bytes / dtype_bytes` = 128 fp16 / 64 fp32). Row/column reductions
instead use PTO's profiled formula tables at their valid 2D frame. The adapter records which exact
lowering applies and `VectorStreamPlan` replays it per emitted invocation. → §7 and
`GroundedVectorOpCompute()`.

---

## 2. The cube core's work is hierarchical and double-buffered

Producing an output tile needs both the **cube MACs** (Matrix pipe) and the
**L1→L0 operand extract** (MTE1 pipe). They run concurrently (double buffering),
so per-region work is the steady-state pipeline, NOT just the MACs. (This is the
*inner* L1↔L0 double-buffer; the *outer* DDR↔L1 reload double-buffer — and its
roofline floor — is in §3.)

**Extract** (`CubeExtractCycles`): the analytic L1→L0 operand reload, using the L0 base tile
`(l0_tile_m, l0_tile_n) = (128, 256)`:

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

For a buildable uniform schedule, exact/co-optimized mode derives a backend-parameterized
`L0MatmulPlan` through the same pure chooser used by `AutoTileMatmulL0`. It records actual L0 M/N/K,
stationarity, buffer depths, the K loop, traffic, and serial-init/rolled/tail/drain phases. Candidate
ranking composes these child phase walls into the outer GM→L1 schedule. Analytic mode instead uses
the aggregate fixed-base-tile formula above and delegates detailed L0 selection until after the
outer winner is known. The aggregate formula also remains the geometry-selection oracle: using the
phase wall itself to re-select baseK before a per-iteration event/synchronization cost is grounded
falsely favors baseK=16.

The PTO A2/A3 GEMM and fused-chain kernels establish the accumulator contract. One output tile stays
in FP32 L0C across its complete sequential-K stream, then drains exactly once. A root drains to GM;
a buildable internal BF16/FP16 result narrows from FP32 Acc to an L1 Mat tile for its consumer. PTO
does not provide a same-type FP32 Acc→Mat plus Mat→Acc chain, so FP32 internal matmul results are a
capability decline, not a costed fused schedule.

---

## 3. The cube roofline — `compute_cost`

```
lat = max( compute_makespan , ddr )    # when the reload can double-buffer
    = compute_makespan + ddr           # otherwise (no overlap)
```

**Double-buffer floor.** The `max()` overlap requires every boundary-loading
request to reconstruct an emitted stage-2 K loop with at least two rolled chunks.
Otherwise compute and reload **serialize** (`compute + ddr`); the old scalar
`K/S ≥ 32` test did not describe upstream loops.

**Double-buffering does NOT reserve L1/UB.** The two ping-pong buffers *together*
are the pool, so `derive_exec` / `vector_stream` use the **full** `l1_capacity` /
`vec_capacity`. The model keeps the full seq-K strip; the emit halves the
*per-load* k (`kk → 2×kk/2`, both in the full L1). No `double_buffer` flag.

**DDR** is the shared HBM floor, charged PER DIRECTION, discounted by active-core
saturation:

```
ddr = ( reload * cyc_per_byte(GM→L1) + out_store * cyc_per_byte(L0C→GM) ) * sat(active)
sat(c) = max(1, n_cores / c)          # too few DMA engines underfill HBM
reload = cube_request_reload(cfg,S)   # exact boundary requests per work unit
```

`cfg.w / cfg.h` carry the **physical (max) region extent**. Requests cover produced operands and
fan-out roles. For buildable uniform cube plans, traffic follows the emitted output-tile order: each
GM K window is loaded once for every output tile that consumes it. Thus a second N tile reloads its
LHS unless a future plan explicitly represents a retained-panel lifetime. Produced intermediates
cost zero GM traffic but occupy L1. A final Acc→L1/GM drain is charged once per output tile, using
the stored dtype; split-K additionally charges the explicit seed launch and atomic root stores.

The outer phase composition is local, not a subgraph-wide roofline:

```
first = GM feed + child L0 wall
rolled = feed + child + (R-1) * max(feed, child)  # only an actual stage-2 loop
tail = GM feed + child L0 wall
final = one Acc→L1/GM drain
```

First/tail/final work remains serial. The vector roofline is §7.

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

- **Cube**: `C = num_cube_cores`, `s_vals = divisors(kfrac)` for one sink; multiple
  sinks use `{1}`. Request propagation handles arbitrary internal M/N. AutoFuse
  can require uniform multi-matmul grids for emit buildability.
- **Vector**: `C = num_vector_cores`; a reduced axis pins to 1 part (it can't be
  spatially tiled — the whole row/col must be present to reduce). `s_vals =
  divisors(2C)` capped by the reduced extent **iff the sink is a reduction**, else
  `{1}` (an internal reduction is tiled like pointwise — see §7).

**Recursive request propagation** (cube): `O[rows,cols]` for `O=A@B` induces
`A[rows,K]` and `B[K,cols]`. Binding-based memoization shares identical requests;
different fan-out roles are recomputed. This covers arbitrary matmul DAGs.

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
LptMakespanPerUnit: enumerate each region shape × split partial, evaluate its
                    full recursive request DAG, sort descending, assign each
                    to the least-loaded of C cores; makespan = busiest core
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

**Cube split-K** (single sink matmul). `S | kfrac` (`kfrac = output_K/16`). On a
grid, LPT-consistent request evaluation shrinks every `ParallelK` region and
creates `P·Q·S` work units. A split whose concrete K loops cannot ping-pong is
legal but pays serialized compute+DDR. With SetAtomicAdd, merge traffic is the S
partial writes; without it, add a serial DDR read-back + sum (∝ S).

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

**Queue-makespan aware** like the cube. `num_tiles` is the logical task/SPMD count; each task owns
one large logical region and streams its inner strips/chunks on one core. `compute_mk` distributes
the sum of those valid-region costs through `ceil(num_tiles/C)` queue rounds. There is no emitted
wave identity or affinity.

**Per-op compute — `GroundedVectorOpCompute` (grounded in PTO-ISA).**

- The PyPTO adapter records a compact primitive and emitted geometry for exact one-instruction
  lowerings: arithmetic/transcendentals, abs/sqrt/neg, scalar and row/column broadcast forms,
  exact part add/mul/max/min, and supported reductions. Candidate costing replays each descriptor at
  every emitted strip/chunk/task. Composite or alias-sensitive operations stay visibly `Generic`.
- *Pointwise:* `slope·ceil(elems/epr) + head+tail`, with startup shared only inside one actual
  back-to-back vector run. Barrier-bearing row expansion starts a new run. This charges startup per
  emitted invocation rather than fractionally scaling one whole-tensor call.
- *Reduction:* FP32/FP16 row and column sum/extrema use the PTO A2/A3 formula tables. Each anchor is
  evaluated as `round(slope(cols)·valid_rows·cols + bias(cols))`; legal between-anchor shapes
  interpolate adjacent total-cycle values. Unsupported dtypes and descriptor-free research inputs
  use the named structural-tree fallback. Generated P1/P2 merges and P4 online-stat work use the
  same grounded primitive table, so work absent from the source DAG is still charged.
- *Split seed:* a terminal materialized `col_sum` split records a separate serial seed launch.
  `TEXPANDS` contributes 24 cycles for each nonempty seed tile, followed by its UB→GM store; seed
  tasks and kernel-fill rounds are additive and cannot hide beneath the atomic-body roofline.

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

**Candidate-evaluation complexity.** `create()` computes graph-only facts once: UB band intervals,
flattened transient tensor references, and ordered op/input lists for each phase. A candidate still
derives its shape-dependent bytes, chunk, loop stages, and cost, but it does not rebuild
producer/consumer positions or phase cones. The UB sweep uses inline delta storage through 64 ops
and falls back to a local vector for deeper groups. Thus evaluation is linear in the selected
subgraph plus a constant number of P1/P2/P4 phases; special streaming kinds do not multiply the tile
enumeration.
The local-search cache continues to retain only `CostResult`, not a stream plan.

**Logical-region identity (A1/G5).** `parts_m * parts_n` is both the solver work-unit and SPMD-block
count. Candidate generation may use the DMA granule to bound useful counts, but
`VectorStreamPlan::{m,n}_partition` distributes the selected count over elements. `tile_h/tile_w`
are maximum logical extents; `free_tile_alloc` is independent UB padding. Pointwise strips and
reduction chunks stay inside a region/task. A “wave” is only the `ceil(work_units/cores)` queue
makespan; neither plan nor emitter invents affinities or a second launch-block count.

**Internal vs sink reduction.** A reduction **sink** (output `[H,1]`) pins the
reduced axis spatially and splits it across cores (§6). An **internal** reduction
(softmax: reductions feed a pointwise `div`, output full `[H,W]`) is tiled like
pointwise — the reduction is a recompute *cost*, not a tiling restriction — and
fills the cores by the fine sub-16 row tiling of §4.

---

## 8. The mixed roofline — `MixedSchedulePlan` / `compute_mixed_cost`

A **mixed** subgraph fuses cube (matmul) and vector (pointwise/reduction) ops into one kernel. On the
910B there is **no direct Acc→Vec pipe**: the handoff round-trips **GM**. `ExpandMixedKernel` splits
the kernel into AIC/AIV functions joined by a GM-backed `tpush`/`tpop` FIFO, and
`SkewCrossCorePipeline` pipelines them. Research opts in; production defaults
`Problem::fuse_cube_vector` off.

```
cube_stage = makespan( max(cube_mac,cube_extract) )       # AIC busiest-core wall — LPT grid / wave uniform (§5)
vec_stage  = makespan( Σ_op VecOpCompute(op) ) / 2         # AIV busiest-core wall — 2 cores/unit, ALL vec ops (§7)
ddr_lat    = max over the 4 GM ports (each par()-capped)   # cross-unit HBM contention
one_cube_tile = max(cube_mac,cube_extract)/num_tiles ;  one_vec_tile = Σ VecOpCompute /(2·num_tiles)

2-stage: wall = max( cube_stage + one_vec_tile,           # producer-bound → + one consumer drain
                     vec_stage  + one_cube_tile,           # consumer-bound → + one producer fill
                     ddr_lat )
3-stage: wall = max( cube_stage, vec_stage, ddr_lat )     # fill absorbed (output unit busy from t=0)
serial:  wall = max( cube_stage + vec_stage, ddr_lat )    # unsupported FIFO topology; no skewed max

lat       = wall + rounds · kernel_fill_cost              # per-LAUNCH fill — added to BOTH shapes
rounds    = ceil(num_tiles / num_cube_cores)              # unit-rounds over the grid
eff_units = min(num_tiles, num_cube_cores)                # atomic resource = 1 cube : 2 vector unit
```

Grounded by pto-isa **`mixed_tile_study`** (7 experiments; the study is the *evidence*, this
section the *model*).

**Overlap `max` — real only for the recorded loop.** Producer skew makes the wall a `max`, not a sum,
for exact `c→v`, `v→c`, `v→c→v`, and `c→v→c` chains (#1900's depth-2 buffers pipeline the two cube
matmuls). A cross-tile carry or multi-round-trip FIFO demotes to Sequential. `create()` caches
maximal same-engine stages and GM transfers once; one-way/three-stage chains are emit-compatible,
while compiler mode rejects serial multi-message/`c→v→c→v` topologies retained for analytic use.
Reduced-axis pinning (`parts_n=1`/`parts_m=1`) prevents a cross-tile reduction carry in either grid
branch. A mid-kernel reduction stays on one unit and streams online if its band exceeds UB (fused
`QK→softmax`); a matmul contraction uses a separate tiling axis. Split-K is atomic-add traffic, so
the `max` applies only to skewable groups. `vec_stage` pools every vector op, including both sides
of `v→c→v`.

**The symmetric fill (grounded: `mixed_vcv`/`vc`/`cvc`).** A 2-stage wall uses
`max(cube_stage + one_vec_tile, vec_stage + one_cube_tile)`: the bottleneck runs its full stage plus
one opposite-unit tile for the unoverlapped fill/drain. It matches simulation to ~1 cycle (`v→c`)
and ~2.7% (`c→v`). Inside the `max`, this fill is absorbed when DDR or the other unit dominates, so
an imbalanced fusion pays only one tiny non-bottleneck tile. In `v→c→v`/`c→v→c`, the output unit is
busy from `t=0`, making fill zero. Detection is structural: the sink unit must have an early-stage
op independent of the opposite unit. Counting sink-unit ops is wrong: `c→v→v` still idles initially.

Absorption also requires **`num_tiles >= 2`**: one tile has no successor to skew and the measured
overlap factor is zero, so its cross-term becomes `cube_stage + vec_stage`. This matters for
low-batch flash decode. *Mid-band caveat:* `one_*_tile` does not shrink while `rounds == 1`, slightly
over-crediting compute-bound partial overlap. More fundamentally, total tiles are not per-group
trips; `MixedSchedulePlan` must choose active groups and a real inner loop. Full attention needs a
key-chunk axis distinct from its query grid.

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

- **Cube** → `derive_exec ≠ INT64_MAX`: a red-blue pebble peak over the
  producer-before-consumer request instances. Live exact intermediate regions +
  each node's greedy sequential-K boundary strips must fit full L1. The root
  output drains L0c→DDR (not charged to L1). Each emitted phase receives a child L0 plan selected
  from backend capacities, but the GM/L1 pebble problem remains in `CubeSchedulePlan`. Region sizes
  and a root K split can change feasibility.
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
| small K | cube | a split that leaves no two-trip K loop serializes compute+DDR; it receives no fictional overlap |
| 16² (K=512) | cube | `(1,1) × split-32` → fills 24 purely via split-K |
| `[512,512]` pointwise | vector | balanced grid → 48 regions, 1 wave |
| `[W,128]` softmax (small rows) | vector | `[W, 3]` fine-row grid → 48 regions (sub-16 rows) |
| `[1024,512]→[1,512]` rowmax | vector | spatial rows fill; few-row → reduced-axis split |

---

## 12. Known limitations / open calibration

- **Remaining vector calibration.** Row/column reduction work comes directly from PTO A2/A3 fit
  tables, and the G7/G8/G9 phase model is silicon-validated. The double-buffer threshold
  (`2·vec_reg_bytes`) and DMA-shape burst (`vec_reg_bytes`) remain reasoned machine bounds rather
  than separately fitted device constants. Count-mode startup sharing is exact only when the
  descriptor order matches the emitted vector run; barriers deliberately break the run.
- **Short-grid ranking.** Grounded valid-row work fixed the tall-softmax under-parallel plan, but
  `[128,8192]` mildly over-splits to 48 tasks (about 7.1% wall regret versus 16 tasks). This is a
  bounded ranking/calibration issue, not a task-count, traffic, or emitted-grid mismatch.
- **Pure-cube plan buildability.** AutoFuse emits uniform multi-matmul grids from
  `CubeSchedulePlan`, including exact output-tile variants, sequential K streams, one final drain,
  and split seed/atomic stores. Unequal multi-op grids and identical deduplicated boundary requests
  are declined. Lone split=1 ceil+clamp is an explicit `ClampedOverlap` plan and charges every
  maximum-shape task; ragged split-K declines because overlapping atomic owners are invalid.
- **Pure-cube inner/outer integration.** AutoFuse owns the GM↔L1 request DAG, L1 lifetimes, K
  windows, output-tile order, split ownership, and final drains. It attaches the shared backend L0
  descriptor; `AutoTileMatmulL0` validates and lowers L1↔L0 geometry, stationarity, buffering, and
  accumulation. The model charges the emitted panel-reload multiplicity and composes child
  init/rolled/tail/drain walls per outer phase. The geometry chooser intentionally retains its
  aggregate oracle until the per-baseK event/synchronization cost is grounded. FP32 internal chains
  decline because PTO has no supported L1 handoff; BF16/FP16 chains use FP32 accumulation and one
  narrowing drain. Remaining work is device validation of nested pipe overlap/FIXPIPE behavior,
  forced-plan ranking, optional retained-panel schedules, and candidate-evaluation performance. The
  production nested schedule uses PTOAS because the host PyPTO allocator cannot pack every disjoint
  ping-pong/scratch lifetime.
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
| grid, wave, and LPT | `partition_axis` / `WaveComputeCycles` / `LptMakespanPerUnit` |
| recursive cube requests | `Ascend910BCost::create` / `CubeRequestNode` |
| grid candidates (triples, granularity) | `Ascend910BCost::create` (`gen_grid`, `grid_gran_*`) |
| cube cost and final schedule | `compute_cost` / `cube_schedule_plan` / `CubeSchedulePlan` |
| vector roofline + double-buffer floor + reduced split | `compute_cost` (vector branch) |
| mixed roofline (plan-gated overlap/serial + fill + 4-port DDR + split-K) | `MixedSchedulePlan` / `compute_mixed_cost` |
| feasibility | `derive_exec` / `cube_peak_l1` / `vector_stream` (mixed: both) |
| enumeration + tiebreak | `Ascend910BCost::best_cost` |
