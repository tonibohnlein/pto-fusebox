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

The output tensor **C is M×N** (M rows, N cols). One global decision:

- **Spatial tiling** of the output: a tile `(w, h)` — `w` along N, `h` along M.
  → `num_tiles = (N/w)·(M/h)`. Each tile is an **independent kernel** on one
  cube core; tiles share only the read-only weights, so there are **no
  dependencies between tiles**.
- **Temporal tiling** = the contraction `k`. Two variants:
  - **single-core sequential k** — one core accumulates the contraction in its
    L0c, no cross-core merge. Allowed on any matmul.
  - **parallel split-K** — the contraction is split across cores, partials merged
    through DDR. **Sink-only** (the merge is a DDR barrier).

**Only `(w, h)` are searched** (`best_cost` loops `ws_cand_ × hs_cand_`). The
contraction `k` is **not** searched — it is derived **greedily** per matmul (see
§3). This document focuses on the single-core (S=1) path.

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

---

## 4. Roofline cost — `compute_cost` (S=1)

`num_tw = N/w`, `num_th = M/h`, `num_tiles = num_tw·num_th`.

- **Compute** `= Σ_matmul fractals · cube_compute_cost · recompute`, with
  `fractals = ⌈N/16⌉·⌈M/16⌉·⌈K/16⌉` (16³ cube fractals; CEIL pads sub-16 dims).
  `recompute = num_tw` for an **intermediate** matmul (its band is recomputed per
  sink N-column tile when the sink tiles over N), `1` for a boundary output.
- **Reload (DDR)** `= Σ_matmul` over **boundary** operands only: LHS `M·N·K/w_i`,
  RHS `M·N·K/h_i` (each operand re-streamed across the perpendicular output
  tiling). **Ephemeral operands are skipped** — they stay on-chip. Plus
  `out_store = Σ boundary-output bytes`.
- **Latency** `= max(compute/eff, (reload + out_store)·inv_B·sat) + kernel_fill`,
  where `eff = min(num_tiles, n_cores)`, `sat = max(1, n_cores/eff)` (HBM is a
  **shared** floor; too few cores can't saturate it), and
  `kernel_fill = ⌈num_tiles/n_cores⌉ · kernel_fill_cost` (the dual of `eff`:
  penalizes over-tiling, so the optimum sits at ~one kernel per core).

---

## 5. Worked example (verified end-to-end)

`A[256,3072]·B[256,256]→T[256,3072]` ; `T·D[128,256]→C[128,3072]`
(M=3072, K1=256, N1=256, N2=128; FP32; L1=512 KB, L0c=128 KB, 24 cube cores,
`cube_compute_cost`=1000, bandwidth=10, `kernel_fill_cost`=10000). Solver picks
`(w=128, h=128)`, S=1, `seq_k=[256,256]`.

**Memory** — order `[op0, op1]`, T band `= 256·128·4 = 128 KB`:

| step | live band | operand strip (greedy k) | total |
| --- | --- | --- | --- |
| op0 `A·B→T` | T 128 KB | `(4·128+4·128)·256` = 256 KB | 384 KB |
| op1 `T·D→C` | T 128 KB | `(4·128)·256` = 128 KB (T ephemeral, no strip) | 256 KB |

`peak = 384 KB ≤ 512 KB` → feasible, both k full (256).

**Roofline**: `num_tw=1, num_th=24, num_tiles=24`.
compute `= 49152·1000·1 + 24576·1000·1 = 73.73M`; reload `= 9.44M + 3.15M =
12.58M`, out_store `1.57M`; `eff=24`, `sat=1` →
`lat = max(73.73M/24, 14.16M·0.1) + 1·10000 = max(3.072M, 1.42M) + 10000 =`
**3,082,000**.

---

## 6. Key modeling choices

1. **The intermediate is ephemeral, full-width (N1), tiled only over M.** It never
   round-trips DDR (on-chip), but it *is* **recomputed** once per sink N-column
   tile (the `recompute = num_tw` factor). Fusion's win = "no DDR round-trip for
   T," paid for by recompute when the sink tiles over N.
2. **k is per-matmul greedy, not searched** — each matmul independently takes the
   largest 16-aligned divisor of its own contraction that fits the L1 headroom.
3. **Compute parallelizes over `min(num_tiles, cores)`; DDR is a shared floor.**

## 7. Implementation note — two places

The **slicing** (which slice of each tensor a tile needs) lives in
`create()`'s TileSource propagation; the **memory peak** is computed separately in
`derive_exec` (it re-derives bands/strips from `cfg` + tensor dims). They are
consistent but not shared code — a candidate for future unification.
