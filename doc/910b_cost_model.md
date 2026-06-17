# 910B Cost Model

The cost model used by the partitioner for the Ascend 910B. This is the
**level-1** model: it fuses ops into subgraphs, tiles each subgraph's **output
tensor**, and runs every output tile as one independent kernel on one core.
(Level-2 — scheduling tile-ops onto cores within a kernel — is a separate
concern handled downstream.)

A subgraph is **unit-homogeneous**: all cube (MatMul) or all vector
(Pointwise/Reduction). Cube↔vector handoff routes through DDR, so the two never
fuse into one grid.

---

## 1. Parameterization

The output tensor is **M×N** (M rows, N cols). One global decision:

- **Spatial tiling** of the output: a tile `(w, h)` — `w` along N, `h` along M.
  → `num_tiles = (N/w)·(M/h)`. Each tile is an **independent kernel** on one
  core; tiles share only read-only inputs, so there are **no dependencies
  between tiles**.
- **Temporal tiling** = the contraction `k`. Two variants:
  - **single-core sequential k** — one core accumulates the contraction in its
    L0c, no cross-core merge. Allowed on any matmul. **Cost-free** (§6).
  - **parallel split-K** (factor `S`) — the contraction is split across `S` cores,
    the partials reduced through DDR. **Sink-only**; the merge cost depends on the
    target's atomic-add capability (§4.2).

**`(w, h)` and the split-K factor `S` are searched**: `best_cost` loops
`ws_cand_ × hs_cand_`, and for each tile `compute_cost` enumerates `S` over its
useful range and keeps the best (§4.2). The single-core contraction `k` is **not**
searched — it is derived **greedily** per matmul (§3). The cube spatial candidates
are bounded by the sink output dim (a tile never exceeds the output it tiles; an
intermediate matmul's wider output width is a contraction, not a spatial axis).

---

## 2. Backpropagation (slicing)

Fixing `(w, h)` on the output determines the slice of **every** tensor each tile
touches. For one tile `C[h rows, w cols]` of `C=(A·B)·D`:

| op | needs |
| --- | --- |
| `C = T·D` | `T[h rows, full N1]` and `D[w cols, full N1]` |
| `T = A·B` | `A[h rows, full K1]` and `B[full N1, full K1]` |

So `h` (the M-band) propagates to the M-axis of **A, T, C**; `w` tiles only the
final output's N; the contraction axes (K1, N1) stay **full** — they are not
spatial, they are handled by `k`.

Implemented in `Subgraph::create()` as the **TileSource role propagation** (a
reverse-topo sweep): the sink output is `FROM_NTW × FROM_NTH`; an intermediate's
width becomes `FROM_NK` (the next matmul's contraction), never a spatial `w`.

---

## 3. Memory feasibility — `derive_exec` / `cube_peak_l1`

A red–blue pebble peak over the fixed DFS execution order. At each step the
**L1/Mat working set** has two charged contributors; feasible iff
`peak ≤ l1_capacity`:

1. **Ephemeral intermediate bands (charged).** A fused intermediate T must become
   fully L1-resident for its consumer matmul, so it occupies a `[full N1, h]`
   band charged across `[producer .. last consumer]`. It is charged **at the
   producer step too** — deliberately conservative: T's band routinely exceeds
   **L0c** (`cube_capacity`, 128 KB) and spills into L1 as it is produced, and
   even when it fits, at the producer→consumer transition it coexists with the
   producer's operand strips. *Ephemeral = no DDR round-trip, not zero memory.*
2. **The boundary output is NOT charged.** On the 910B the L0c accumulator drains
   **directly to DDR** (no L1 staging of the result), so a tile's own output
   needs no L1 slot. (L0c sizing is `AutoTileMatmulL0`'s job, not this model.)

On top of the live bands, each matmul step adds its **boundary-operand strip**,
sized by the greedy `k`: per unit of k the strip is `lhs_bytes·h + rhs_bytes·w`
(an ephemeral operand is already a band → contributes 0). The **greedy k** is the
largest 16-aligned divisor of that op's contraction that fits the headroom
`L1 − live_bands`. If a huge K cannot fit in one pass it is sliced — the
**internal single-core k-split**.

**Arbitrary DAGs (trees, diamonds).** The band loop is interval-overlap, so it
generalises with no special-casing: every ephemeral is live `[producer .. last
consumer]`, and each step sums *all* bands covering it. A **tree** join (e.g.
`(A·B)·(E·F)`) therefore has **both** child bands co-resident at the root; the
DFS post-order (one subtree fully, then the sibling, then the join) keeps the
live-band count = the DFS frontier width. At a join the root op's `per_unit = 0`
— both its operands are ephemeral bands already charged, so it adds no strip.

---

## 4. Roofline cost — `compute_cost`

`num_tw = N/w`, `num_th = M/h`, `num_tiles = num_tw·num_th`,
`eff1 = min(num_tiles, n_cores)`, `inv_B = 1/bandwidth`,
`sat(c) = max(1, n_cores/c)`.

**Compute** `= Σ_matmul fractals · cube_compute_cost · recompute`, with
`fractals = ⌈N/16⌉·⌈M/16⌉·⌈K/16⌉` (16³ cube fractals; CEIL pads sub-16 dims).
`recompute = num_tw` for an **intermediate** matmul (its band is recomputed per
sink N-column tile when the sink tiles over N), `1` for a boundary output.

**Reload (DDR)** `= Σ` over **boundary** operands: LHS `M·N·K/w_i`, RHS
`M·N·K/h_i` (each operand re-streamed across the perpendicular output tiling;
`w_i = min(w, N_i)` for a boundary op, else the full intermediate width `N_i`).
Ephemeral operands are skipped (on-chip). **Shared-input dedup:** an operand used
by several ops in the **same role** (`(tensor, orientation, is_boundary_op)`) is
charged **once** — within a subgraph all outputs share `(M,N)`, so the same key
is the same per-tile slice, loaded once into L1 and reused (§5.1). A role *switch*
(LHS in one op, RHS in another) keys differently and is charged twice (rows vs
columns — different data). `out_store = Σ boundary-output bytes`.

### 4.1 Spatial-only (S=1)

```
lat1 = max( total_compute / eff1 , (reload + out_store) · inv_B · sat(eff1) )
       + kernel_fill
```
`kernel_fill = ⌈num_tiles/n_cores⌉ · kernel_fill_cost` — the dual of `eff`:
penalises over-tiling, so the optimum sits at ~one kernel per core. Compute
parallelises over cores; **DDR is a shared floor** (`sat` penalises too few
cores to saturate HBM).

### 4.2 Parallel split-K (sink contraction split across `S` cores)

Used when the spatial tiles can't fill the cores (`num_tiles < n_cores`). The
sink contraction is split into `S` per-tile partials. `S` is **enumerated** over
its useful range and the best kept (the optimum is interior when the merge is
serial — below):

```
kfrac  = output_K / 16                          # 16-fractal contraction slices
S_max  = min(kfrac, ⌈n_cores / num_tiles⌉)      # >=1 fractal/partial; beyond this effS caps at n_cores
for S in 2 .. S_max:
  effS    = min(num_tiles · S, n_cores)
  streamS = max( total_compute / effS , reload · inv_B · sat(effS) )       # decreases with S
  merge   = S · out_store · inv_B · sat(effS)                              # (1) write the S partials
          + (ddr_atomic_add ? 0
                            : (S+1) · out_store · inv_B · sat(num_tiles))  # (2) serial read-back + sum
  latS    = streamS + merge
# result = min( lat1 , best latS ) + kernel_fill
```

The merge is an **additive barrier** (the reduction can't start until all `S`
partials exist). It has two regimes, set by the target capability
`Problem::ddr_atomic_add` (§8):

- **With `SetAtomicAdd`** (`ddr_atomic_add = true`; the 910B): the partials are
  atomically accumulated into the DDR output **during write-back** — no read-back.
  The merge is just (1), which is `sat`-discounted and **cancels below full-fill**
  (`sat(effS) = n_cores/(num_tiles·S)` → the `S` cancels), so split-K is cheap and
  the optimal `S` is max-fill.
- **Without it** (`false`): add (2) — read the `S` partials back and sum them
  **serially per output tile** (one DDR accumulator/tile, so the `S` adds
  serialise; parallelism is across `num_tiles`, not `S` → `sat(num_tiles)`, grows
  ∝ `S`). Split-K is punished and the optimal `S` is **interior** — which is why
  `S` is enumerated rather than derived as a single max-fill value.

**Sink-only:** only a boundary-output matmul may split this way (its output goes
to DDR anyway); an *internal* matmul splitting would round-trip the ephemeral
through DDR — expressed as a **subgraph cut** at the partition level, not an
in-subgraph split. The displayed per-core `k` composes the single-core L1-fit `k`
with the split share `⌈kfrac/S⌉·16`.

---

## 5. Shared inputs & multiple outputs

A subgraph may have **several boundary outputs** as long as they share the same
`(W,H)` (`create()` rejects mismatched sinks). One unified grid tiles them all;
per tile-position the kernel runs the DFS order over the ops, producing every
output's tile in that kernel — they are computed **together**, not in separate
passes.

### 5.1 Shared-input reuse

When a boundary input feeds several ops in the same role (e.g. a shared LHS `t0`
in `{t0·t1, t0·t3}`), every op wants the *same* per-tile slice `t0[K, m-band]`
(independent of the N-column), so it is loaded **once** into L1 and reused. The
reload dedup (§4) credits this: `t0` is charged once, not once per op. **This is
only visible when the subgraph is memory-bound** — see §8. A *role switch* needs
different slices (rows vs columns) and is correctly charged twice.

---

## 6. Single-core k-split is cost-free

Splitting one matmul's contraction into `nk` single-core passes changes neither
compute nor IO:

- The **L0c accumulator persists** across all `nk` passes (the output tile fits
  L0c, or spills to L1 — §3). Each pass streams its k-chunk operands and
  **accumulates in place**; the partials merge *in L0c*. So the cube fragment
  count is the full `⌈N/16⌉⌈M/16⌉⌈K/16⌉` regardless of chunking, and each operand
  is loaded once (in chunks). `nk · max(compute/nk, io/nk) = max(compute, io)`.
- **No `k²` in a chain.** With both matmuls split, the per-op contractions just
  walk independently; the op1-chunks *partition* N1, so op0 produces each T-slice
  once — `Σ` over the partition is op0's full count, once. (The fused-pipeline
  variant that *would* re-read A per chunk is not used; the model materialises
  the full T band instead — §3.)

This is exactly what distinguishes single-core (free, L0c merge) from parallel
split-K (the DDR merge — free with `SetAtomicAdd`, else a serial reduction; §4.2).

---

## 7. Worked example (verified end-to-end)

`A[256,3072]·B[256,256]→T[256,3072]` ; `T·D[128,256]→C[128,3072]`
(M=3072, K1=256, N1=256, N2=128; FP32; L1=512 KB, L0c=128 KB, 24 cube cores,
`cube_compute_cost=1000`, bandwidth=10, `kernel_fill_cost=10000`). Solver picks
`(w=128, h=128)`, S=1, `seq_k=[256,256]`.

**Memory** — order `[op0, op1]`, T band `= 256·128·4 = 128 KB`:

| step | live band | operand strip (greedy k) | total |
| --- | --- | --- | --- |
| op0 `A·B→T` | T 128 KB | `(4·128+4·128)·256` = 256 KB | 384 KB |
| op1 `T·D→C` | T 128 KB | `(4·128)·256` = 128 KB (T ephemeral, no strip) | 256 KB |

`peak = 384 KB ≤ 512 KB` → feasible, both k full (256).

**Roofline** (`num_tw=1, num_th=24, num_tiles=24`):
compute `= 49152·1000 + 24576·1000 = 73.73M`; reload `= 9.44M + 3.15M = 12.58M`,
out_store `1.57M`; `eff=24, sat=1` →
`lat = max(73.73M/24, 14.16M·0.1) + 10000 = max(3.072M, 1.42M) + 10000 =`
**3,082,000** (compute-bound at this `cube_compute_cost`).

---

## 8. Calibration

The machine params are **placeholders** (relative units) until grounded from real
910B specs; the per-op `base_cost` is ignored.

- `cube_compute_cost` — time per 16³ cube fractal. **Default is memory-bound**
  (64 in `set_910b`): real 910B matmul is memory-bound for typical shapes. With
  `bandwidth=10` the roofline knee sits around a few hundred of these units (the
  fp16 hardware estimate is ~5). The old 4096 placeholder was compute-bound and
  hid every IO-side effect (the shared-input reuse, the DDR floor). Tests that
  specifically exercise a *compute-bound* property (split-K compute speedup,
  fractal-quant decode, the compute-bound `fused==sum`) pin a compute-bound value.
- `vector_compute_cost`, `vector_lanes`, `kernel_fill_cost` — also placeholders.
- `ddr_atomic_add` — **target capability** (true on the 910B). Selects the split-K
  merge regime (§4.2): `true` = `SetAtomicAdd` write-back (cheap merge, max-fill
  split-K); `false` = serial per-tile read-back + sum (split-K punished ∝ S,
  interior optimum). The scheduler's analog of a compiler/target flag.
- `slow_memory_bandwidth`, capacities (`l1`/`cube`/`vec`), `num_*_cores`,
  `double_buffer` — the per-die 910B layout (24 cube / 48 vector cores; L1 512 KB,
  L0c 128 KB, UB 192 KB).

---

## 9. Key modeling choices

1. **The intermediate is ephemeral, full-width (N1), tiled only over M.** It never
   round-trips DDR (on-chip), but it *is* **recomputed** once per sink N-column
   tile (`recompute = num_tw`). Fusion's win = "no DDR round-trip for T," paid for
   by recompute when the sink tiles over N.
2. **k is per-matmul greedy, not searched** — each matmul independently takes the
   largest 16-aligned divisor of its own contraction that fits the L1 headroom.
3. **Compute parallelises over `min(num_tiles·split, cores)`; DDR is a shared
   floor.** Single-core k-split merges in L0c (free); parallel split-K merges
   through DDR (additive barrier).

---

## 10. Known limitations

- **Single roofline over a fused subgraph (`Σmax ≥ maxΣ`).** `compute_cost` takes
  one roofline `max(Σcompute/eff, Σio·inv_B·sat)` over all ops rather than summing
  per-op rooflines `Σ max(c_i, i_i)`. This is optimistic when fused stages differ
  in bottleneck (one compute-bound, one IO-bound) — it assumes cross-tile
  pipelining hides the per-stage sequential dependency. Invisible when uniformly
  compute- or memory-bound.
- **Conservative L0c-aware band.** An ephemeral is charged at its producer step
  unconditionally (§3). A refinement would charge it there only when its band
  exceeds L0c (else it sits in the accumulator, not L1).
- **Two places.** The per-tensor *slicing* (TileSource in `create()`) and the
  *memory peak* (`derive_exec`, which re-derives bands/strips from `cfg`) are
  consistent but not shared code — a candidate for unification, along with
  computing the cube reload from `boundary_tensor_info_` (as the vector path
  does) instead of the per-op dedup loop.
