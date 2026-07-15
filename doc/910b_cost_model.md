# 910B cost model

This document describes the level-1 scheduler for Ascend 910B. It partitions a tensor-op DAG,
enumerates a spatial grid and optional cross-core split for each subgraph, and estimates one-kernel
latency. AutoFuse then consumes a solver-owned schedule plan for the selected configuration.
`grounded_cost_model.md` is the authority for coefficients, phase costs, and mixed-unit scheduling.

A homogeneous subgraph is either pure cube (`MatMul`) or pure vector (pointwise/reduction). This is
the production default. `Problem::fuse_cube_vector` enables the same base model's mixed-plan path at
runtime; the research `Ascend910BMixed` convenience opts in unconditionally.

---

## 1. Cube decisions

For a pure-matmul subgraph whose root output is `M×N`, a candidate is:

- `parts_m × parts_n`: a balanced spatial partition of root M/N;
- `split_k`: an optional cross-core split of the one supported root contraction;
- one derived, single-core sequential K window for every matmul request instance.

`partition_axis(dim, parts, 16)` splits an axis into near-equal 16-aligned regions. An axis has at
most two extents and a 2D grid at most four region shapes. `TileConfig.h/w` hold the largest physical
region, while `parts_m/parts_n` hold the exact counts.

The solver enumerates `(parts_m, parts_n, split_k)` triples. The sequential K windows are not search
dimensions: they are greedily derived from L1 headroom. Candidate evaluation keeps only the compact
`CostResult`; it does not cache a full schedule plan.

---

## 2. Recursive request DAG

The old shared-M/full-intermediate-width chain shortcut is replaced by consumer-driven request
propagation. If a consumer requests `O[rows,cols]` from `O=A@B`, it requests:

```
A[rows,K]
B[K,cols]
```

The rule is applied recursively from every root. Axis extents are represented symbolically:

- `SpatialM` / `SpatialN`: this work unit's root region;
- `ParallelK`: this work unit's root K share;
- `SequentialK`: an operation's local streamed contraction;
- `Full`: the complete tensor axis.

The request topology is candidate-invariant and is built once in `Ascend910BCost::create()`. A
request is memoized by `(tensor,height_binding,width_binding)`. Identical fan-out requests share one
producer instance; different roles create different instances and explicitly pay recomputation.
Nodes are stored producer-before-consumer, so one order drives liveness, cost, plan reconstruction,
and emission.

This representation supports non-square deep chains, trees, a produced RHS, both inputs produced,
fan-out role switches, and multiple roots. With multiple roots `split_k` is forced to one: a single
split coordinate cannot identify several atomic output targets.

---

## 3. L1 feasibility and sequential K

`derive_exec()` plays a red-blue pebble game over request instances. At each step it charges:

1. every produced region live from its producer through its last requesting consumer;
2. the current operation's boundary-operand K strip;
3. retained tensors supplied by the outer solver.

An internally produced operand is already a live region and is not charged again as a boundary
strip. A boundary root output is not charged to L1 because its accumulator drains through L0C to GM.
An output that is both a root and an internal value is retained for its consumer and also stored.

For an operation with requested output `m×n`, the bytes per K element are:

```
lhs_boundary ? m * lhs_dtype_bytes : 0
+ rhs_boundary ? n * rhs_dtype_bytes : 0
```

The derived window is the largest 16-aligned divisor of that operation's effective contraction that
fits beside the live regions. The peak is the maximum of `live + window strip` over the schedule.
The candidate is infeasible if no 16-element window fits. Every operation derives independently;
the sink's K value is never reused for an upstream contraction.

For a split root, `ParallelK` regions and the root contraction shrink to `K/split_k`; upstream local
contractions remain full. The lone-matmul path deliberately preserves its historical result: derive
the L1 divisor against full K, then cap that divisor to the selected K share.

---

## 4. Per-request cube work and the L0 child plan

The legacy analytic work for one request instance has two inner pipes:

- Matrix-pipe `CubeMacCycles(m,n,k,input_dtype)`;
- MTE1 `CubeExtractCycles(m,n,k,input_dtype)` for L1→L0A/B extraction. Precision and operand bytes
  follow the matmul inputs, not an FP32 accumulator/output.

The requested region is subdivided using `Problem::l0_tile_m/l0_tile_n` (normally 128×256) and a
64-wide K step. With `L` L0-MAD steps, Phase D is:

```
(MAC + extract + (L - 1) * max(MAC, extract)) / L
```

One step serializes MAC and extraction; a long sequence approaches their overlapped maximum.
Matmul request instances have data dependencies, so their Phase-D times sum within a work unit.

The model deliberately exposes two policies. Analytic mode (the default `Problem` setting) uses the
aggregate fixed-base-tile formula above. Exact/co-optimized mode sets
`Problem::use_hierarchical_cube_cost=true` and derives a backend-parameterized `L0MatmulPlan` for
every init/rolled/tail tensor phase. The shared chooser records actual L0 M/N/K, stationarity,
buffer depths, an L0 K loop, and serial-init/rolled/tail/drain costs; the uniform candidate cost then
composes those child phase walls directly. The aggregate formula remains both the cheap policy and
the already-grounded geometry-selection oracle: using phase granularity to re-select baseK before
adding a grounded per-iteration event cost incorrectly prefers baseK=16.

Exact enumeration uses an ephemeral memo keyed by local M/N/K, operand/storage dtypes,
accumulator-read state, and output target. It is scoped to one `best_cost()` or plan dump: neither
the `L0MatmulPlan` nor the memo enters `CostResult` or the global op-set cost cache.

For a balanced grid, `LptMakespanPerUnit` creates every concrete
`region-shape × split-partial` task and assigns longest tasks first to the least-loaded cube core.
This preserves unequal region work and upstream recomputation. The lone-matmul path retains the
pre-refactor LPT formula as a fixed cost anchor.

---

## 5. GM traffic and outer roofline

Only boundary inputs and boundary outputs touch GM. The buildable hierarchical cost charges each
boundary request once per emitted output tile and GM K window. This includes the emitter's actual
reload multiplicity: another N output tile reloads its LHS unless an explicit retained-panel
lifetime is added. Produced intermediates cost zero GM traffic but non-zero L1 capacity.

GM→L1 operand reads and L0C→GM output writes are separate concurrent pipes. Each divides across
active cores up to its aggregate bandwidth cap. Split-K writes the root output `S` times using
atomic add.

For each output variant, the outer phase cost is:

```
first window = GM feed + child L0 wall
rolled       = feed + child + (R-1) * max(feed, child)
tail         = GM feed + child L0 wall
final        = one Acc->L1/GM drain
```

Only an actual stage-2 rolled phase receives `max`; first/tail/final work is additive. Up to four
ragged output variants carry exact counts. The split seed is a separate vector fill/store/task
phase and contributes another kernel-fill wave.

---

## 6. Split taxonomy

Two mechanisms must remain distinct:

| mechanism | scope | merge |
|---|---|---|
| sequential K streaming | every matmul request instance | persistent local accumulator |
| parallel split-K | the single supported root only | zero seed + atomic-add GM stores |

Sequential K changes memory feasibility and overlap but not mathematical work. Parallel split-K
creates `parts_m × parts_n × S` independent work units, shrinks the root K request, and multiplies
root stores by S. It is enumerated only when the root contraction divides into aligned shares.

Multiple roots use `S=1`. An internal cross-core split would require a synchronization/materialized
merge and is represented by cutting the subgraph, not by this in-kernel plan.

---

## 7. `CubeSchedulePlan`

For the winning or forced configuration, `cube_schedule_plan()` re-runs the lightweight derivation
and returns:

- exact M/N partitions, split, work units, and peak L1;
- one `CubeMatmulSchedule` per request instance;
- producer-instance dependencies and root markers;
- concrete output/LHS/RHS regions plus their symbolic bindings;
- full/effective contraction;
- accumulator dtype and result-storage dtype;
- L1 K window, emitted chunk, rolled trips, tail, and pipeline stage;
- output/L0C variants with shared-backend init/rolled/tail child plans;
- one explicit final Acc→L1/GM drain;
- the split seed plus overlap/buildability flags.

The plan is intentionally absent from `CostResult`, which is stored in the local-search cache. The
topology is built once per subgraph; O(nodes) candidate derivations remain stack-local. This keeps
local search compact while ensuring the emitter does not rediscover roles or K loops independently.

---

## 8. AutoFuse buildability boundary

The current plan-driven emitter exactly replays uniform multi-matmul grids. AutoFuse sets
`Problem::require_uniform_cube_dag_grid`, which filters non-uniform M/N partitions only for groups
with more than one matmul. This is a buildability restriction, not an analytic limitation of the
cost model.

AutoFuse emits the planned tensor-level output tiles and K windows and attaches each child
`L0MatmulPlan`; it does not create an L0 tile. `AutoTileMatmulL0` validates the descriptor against
the active backend chooser and builds Left/Right/Acc IR. Output tiles remain in L0C through the
complete K stream and drain once. The outer GM→L1 pipeline is marked so its stage depth is not
multiplied into nested L0 buffers.

Floating operands accumulate in FP32. A2/A3's fused-chain path drains an internal FP32 Acc to a
BF16/FP16 Mat tile. Same-type FP32 Acc→Mat and Mat→Acc do not exist, so an explicitly FP32 internal
result is not a buildable fused chain and is partitioned into standalone matmuls.

Lone matmuls retain the established balanced non-uniform search and ceil+clamp emitter. That emitter
is numerically idempotent, but it can execute more max-size tiles than the LPT-balanced cost prices.
General multi-op non-uniform emission is therefore not claimed complete.

The plan emitter also declines a deduplicated identical boundary request until an explicit shared L1
panel and lifetime are represented. In non-strict production mode, declined groups fall back to
dependency-ordered standalone matmuls; strict mode reports the violated contract.

---

## 9. Validation and remaining work

Host coverage includes:

- lone-matmul fixed costs, plan reconstruction, exact/ragged K, split and no-split;
- produced LHS/RHS, both inputs produced, non-square trees, deep chains, fan-out role switches, and
  multiple sinks;
- role-aware L1 peak/reload, one-trip no-overlap, uniform-grid buildability, and compact cache state;
- Torch numerics and PTOAS-backed default-stack lowering, including a 192 KiB BF16 intermediate,
  explicit FP32 accumulation/narrowing, and FP32-chain decline.

Before device ranking is trusted, complete:

1. implement or consistently price non-uniform multi-op grids and lone ceil+clamp work;
2. add shared boundary-panel retention only with an explicit lifetime and matching traffic change;
3. ground a per-baseK event/synchronization term before phase cost changes geometry selection;
4. extend the Acc→Mat capability table beyond BF16/FP16 only when PTO supports it;
5. validate descriptor consumption, actual reload multiplicity, nested MTE2/MTE1/Matrix/FIXPIPE
   overlap, narrowing, atomic stores, wall time, and forced-plan regret on Ascend 910B2 with the
   latest PTOAS;
6. profile and optimize buildable candidate evaluation.

The production nested schedule uses the PTOAS memory planner. The host PyPTO allocator cannot yet
interval-pack every disjoint ping-pong and full-region scratch lifetime required by this path.

---

## 10. Source map

| concept | location |
|---|---|
| request DAG construction | `Ascend910BCost::create` |
| L1 pebbling / per-node K | `derive_exec`, `cube_peak_l1` |
| boundary reload | `cube_request_reload` |
| MAC/extract grounding | `CubeMacCycles`, `CubeExtractCycles` |
| LPT scheduling | `LptMakespan`, `LptMakespanPerUnit` |
| cube candidate cost | `Ascend910BCost::compute_cost` |
| final plan reconstruction | `cube_schedule_plan` |
| plan data structures | `types.h` |
| shared L0 chooser | `l0_matmul_plan.h`, `l0_matmul_plan.cpp` |
| JSON interchange | `io.cpp` and AutoFuse `DumpProblemJson` |
