# Partition Search

## Overview

The partition search finds the optimal assignment of operations to subgroups.
It operates on a `Partition` — a mutable collection of groups, where each group
is a set of op indices that will execute as one subgraph step.

The architecture has four layers:

```
greedy_descent (hill climb, strictly monotonic)
    └── used standalone AND as subroutine inside:

fm_inner_pass (one FM pass: random seed → locking → drift control)
    └── called repeatedly by:

fm_outer_loop (adaptive cooling, greedy-kick escape)
    └── called per-task by:

parallel_search (multi-start + evolutionary pool management)
```

---

## 1. Move Types

All six moves operate on groups (sets of ops). Each move is evaluated by calling
`eval_set(ops)` which computes `Subgraph::create + best_cost` (memoized via
CostCache).

`Subgraph::create` is the authoritative validity check. It verifies:
connectivity (DAG edges + shared-input co-consumer edges), ephemeral fan-out
(single consumer), boundary output dimension consistency, and the PW-sink k=1
constraint. If any check fails, `eval_set` returns 1e18 and the move is rejected.

### STEAL
Move a border op from group A to adjacent group B.

- **Precondition**: op is on the boundary of A (has a DAG neighbor or
  co-consumer outside A). Must not create a cycle (`merge_creates_cycle`
  check on `{op}` vs B's ops).
- **Cost**: `eval_set(A - {op}) + eval_set(B ∪ {op})`
- **Saving**: `cost(A) + cost(B) - new_cost(A) - new_cost(B)`
- **Note**: if removing op disconnects A, `eval_set` returns 1e18, rejecting
  the move silently. No explicit connectivity check needed.

### MERGE
Fuse two adjacent groups A and B into one group.

- **Precondition**: A and B are adjacent (share a DAG edge or a common input
  tensor via co-consumer relationship). Merging must not create a cycle in the
  condensed DAG (`merge_creates_cycle` check).
- **Cost**: `eval_set(A ∪ B)`
- **Saving**: `cost(A) + cost(B) - cost(A ∪ B)`
- **Effect**: Group B is marked dead. All ops move to A.
- **Dedup**: In greedy's `generate_moves`, a `merge_checked` set prevents
  evaluating the same (A, B) pair multiple times when multiple boundary ops
  connect to the same neighbor group.

### RECOMPUTE
Duplicate an op into a neighbor group (keep in both groups).

- **Precondition**: op is in group A, group B is adjacent (DAG edge or
  shared input), op not already in B. Must not create a cycle.
- **Cost**: `eval_set(B ∪ {op})` (group A unchanged)
- **Saving**: `cost(B) - new_cost(B)`
- **Effect**: op now exists in both A and B. B recomputes the op locally
  instead of loading its output tensor from slow memory.

### EJECT
Remove a border op from its group. The remainder may split into
multiple connected components, each becoming a separate group.

- **Precondition**: op is on the boundary of group A, group has ≥2 ops.
- **Evaluation**: `eval_eject(op, A)` which calls `connected_components`
  on the remainder and `eval_set` on each component plus the singleton.
- **Saving**: `cost(A) - Σ cost(component_i) - cost({op})`
- **Effect**: Group A is replaced by its remainder component(s). A new
  singleton group is created for the ejected op. If the remainder splits
  into multiple components, each becomes its own group.

### INTERNAL_EJECT
Remove an internal op from its group. Identical to EJECT in evaluation
and application, but generated for internal ops (no DAG neighbors outside
the group).

- **Precondition**: op is internal (no DAG neighbors or co-consumers outside
  group A). Group has 3–15 ops (capped to avoid expensive evaluation).
- **Evaluation**: same `eval_eject` as EJECT.
- **Effect**: same as EJECT.
- **Generation**: competes with SPLIT — one best move per internal op is
  pushed to the heap.

### SPLIT
Split a group at a bridge edge into two non-trivial parts.

- **Precondition**: Group A has a bridge edge (u, v) — a DAG edge whose removal
  disconnects the group (using the full `op_neighbors` graph including
  co-consumer edges). Group has 3–15 ops.
- **Evaluation**: `eval_split(u, v, A)` — BFS from u skipping edge u↔v,
  ops reached form side_A; unreached ops form side_B.
- **Cost**: `eval_set(side_A) + eval_set(side_B)`
- **Saving**: `cost(A) - cost(side_A) - cost(side_B)`
- **Effect**: Group A keeps side_A, a new group gets side_B.

### TENSOR_MERGE (FM only)
Merge all groups containing consumers of a shared tensor, optionally
including the producer group.

- **Precondition**: tensor t has ≥2 consumers in ≥2 different groups.
  No locked ops in any involved group. No pairwise cycles.
- **Cost**: `eval_set(union of all groups' ops)`
- **Saving**: `Σ cost(group_i) - cost(merged)`
- **Effect**: first group absorbs all ops, remaining groups are killed.

### TENSOR_EXTRACT (FM only)
Pull just the consumer ops (and optionally producer op) of a shared
tensor into a new group, leaving remainders in their original groups.

- **Precondition**: same as TENSOR_MERGE. Used as fallback when full
  merge is infeasible.
- **Cost**: `eval_set(extracted_ops) + Σ eval_set(remainder_i)`
- **Saving**: `Σ cost(group_i) - cost(extracted) - Σ cost(remainder_i)`
- **Effect**: each original group keeps its non-extracted ops. A new
  group is created for the extracted ops.

---

## 2. Greedy Hill Climb (`greedy_descent`)

Pure descent — only accepts improving moves. Guaranteed to converge.

### Data structure

Priority heap (`std::priority_queue<Move>`) with lazy staleness detection via
per-group generation counters.

### Move generation

`generate_moves(gi, heap, floor=0)` evaluates moves from group gi's perspective.
To minimize heap size and eval_set calls:

- **Per neighbor op** (border ops adjacent to gi): evaluate MERGE, STEAL,
  RECOMPUTE across all neighbor groups. Only the single best move per
  neighbor op is pushed. MERGE results are deduped by a `merge_checked` set
  so each (gi, gj) pair is evaluated at most once.
- **Per ejectable op** (border ops of gi, group ≥2): evaluate EJECT via
  `eval_eject`. One move per op.
- **Per internal op** (internal ops of gi, group 3–15): evaluate
  INTERNAL_EJECT and SPLIT at all incident DAG edges. Only the single best
  move per internal op is pushed.

STEAL and RECOMPUTE share the `eval_set(gi ∪ {adj_op})` call — the
gi-expansion cost is computed once and used for both.

### Algorithm

```
1. Seed heap: generate_moves for all alive groups.

2. While heap not empty:
   a. Pop top move m (highest saving).
   b. Staleness check:
      - m.gen_a must match groups[m.ga].gen
      - For MERGE/STEAL/RECOMPUTE: m.gen_b must match groups[m.gb].gen
      - groups[m.ga] and groups[m.gb] must be alive
      If stale → discard, continue.
   c. apply_move(m):
      - Re-evaluates cost at apply time (cost guard)
      - If actual saving < -0.001 → reject (return empty dirty set)
      - Otherwise: mutate partition, bump gen counters, rebuild index
      - Return dirty set = affected groups + their adjacent groups
   d. For each dirty group: generate_moves(gi, heap)
```

### Cost guards in apply_move

Each move type re-evaluates its cost at apply time and rejects if the
actual saving has flipped negative. This prevents stale moves from
worsening the partition:

| Move | Guard |
|------|-------|
| MERGE | `new_cost >= cost(A) + cost(B) - 0.001` → reject |
| STEAL | `actual_saving < -0.001` → reject |
| RECOMPUTE | `actual_saving < -0.001` → reject |
| EJECT | `actual_saving < -0.001` → reject |
| INTERNAL_EJECT | `eval_eject.saving < -0.001` → reject |
| SPLIT | `eval_split.saving < -0.001` → reject |

### Dirty set propagation

After a move is applied, moves are regenerated for all groups in the
dirty set. The dirty set includes:

- The directly modified groups (ga, gb, any new groups)
- All groups adjacent to the modified groups

This ensures that any move whose cost estimate depends on the changed
groups gets re-evaluated with fresh data.

---

## 3. FM Inner Pass (`fm_inner_pass`)

One pass of the FM algorithm: random initialization, locking, drift control.
Allows worsening moves to escape local minima.

### Active Set

Flat vector of `{op, best_move}` entries. Each entry represents one op's
current best move across all its groups and neighbors. Uses O(N) scan for
max selection (N ≤ num_ops, fast for practical sizes).

### best_move_for(op)

For a single op, evaluates all candidate moves across all its groups and
neighbor groups. Unlike greedy's `generate_moves` (which iterates per-group),
this iterates per-op.

For each group `gx` containing op:
- If op is a border op of `gx`:
  - Collect `neighbor_groups` (a `std::set`, automatically deduped)
  - For each neighbor group gy: evaluate STEAL, MERGE, RECOMPUTE
  - Evaluate EJECT from gx (via `eval_eject`)
  - Evaluate TENSOR_MERGE / TENSOR_EXTRACT for shared input tensors
- If op is internal in `gx` (and group size 3–15):
  - Evaluate INTERNAL_EJECT (via `eval_eject`)
  - Evaluate SPLIT at each DAG edge incident to op

Returns the single best move. Moves with `saving < -floor` or
`|saving| < 1.0` (epsilon to filter noise) are filtered.

### Algorithm

```
1. INITIALIZE
   Collect all activatable ops (border + internal for groups 3–15 ops).
   Select random subset (size = cfg.init_count, typically 3).
   For each selected op:
     activate(op) → compute best_move_for(op, floor, locked={})
                   → add {op, move} entry to active set

2. MAIN LOOP (until exhaustion or max_drift):

   a. POP BEST
      Linear scan of entries for unlocked op with highest saving.
      Lock the initiating op.

   b. PRE-LOCK PARTNERS
      For MERGE: lock initiating op + boundary ops of partner group
        that are DAG neighbors of the initiating op.
      For TENSOR_MERGE/EXTRACT: lock ALL ops in ALL involved groups.
      This prevents conflicting moves from being applied in the same pass.

   c. APPLY MOVE
      apply_fm_move(partition, move)
        → mutates partition, rebuilds index
        → returns affected group indices (empty if move failed)
      Note: apply_fm_move has NO cost guards (intentional for FM).
      Only feasibility is checked (eval_set < 1e17).
      Exception: TENSOR_MERGE and TENSOR_EXTRACT do re-verify savings.

   d. SNAPSHOT
      If total_cost < best_seen → snapshot partition as new pass-best.

   e. DRIFT CHECK
      cumulative_gain = start_cost - current_cost
      If (best_cumulative_gain - cumulative_gain) > max_drift → abort pass.

   f. REFRESH ACTIVE SET (refresh_after_move)
      Three steps, combined in one pass over affected + adjacent groups:
        1. Collect relevant = affected_groups ∪ adjacent_groups(affected)
        2. For each active, unlocked entry whose op touches a relevant group:
           Recompute best_move_for(op, floor, locked)
        3. For each relevant alive group:
           Activate all border + internal ops (new entries added to set)

3. RETURN best snapshot + perturbed end state
```

### What gets locked

| Move applied | Ops locked |
|---|---|
| STEAL(op) | op |
| MERGE(op) | op + DAG neighbors of op in the absorbed group |
| RECOMPUTE(op) | op |
| EJECT(op) | op |
| INTERNAL_EJECT(op) | op |
| SPLIT(op) | op |
| TENSOR_MERGE(op) | ALL ops in ALL involved groups |
| TENSOR_EXTRACT(op) | ALL ops in ALL involved groups |

Locked ops cannot initiate new moves but CAN be affected by other ops'
moves (e.g., a locked op may end up in a merged group initiated by an
unlocked op).

### Floor and drift

- **Floor**: minimum saving to accept. `floor = start_cost × floor_fraction`.
  Moves with `saving < -floor` are rejected by `best_move_for`. Computed once
  at the start of the pass.
- **Drift**: maximum allowed degradation below the pass-best.
  `max_drift = start_cost × max_drift_fraction`.
  If the cumulative gain drops below `best_cumulative_gain - max_drift`,
  the pass aborts.

---

## 4. FM Outer Loop (`fm_outer_loop`)

Repeated FM passes with adaptive exploration control.

### Algorithm

```
best_partition = initial partition
heat = 1.0

For each pass (up to max_passes or deadline):

  1. TEMPERATURE (cosine annealing)
     progress = pass / max_passes
     temperature = 0.1 + 0.9 × 0.5 × (1 + cos(π × progress))
     Decays smoothly from 1.0 → 0.1 over the pass budget.

  2. EFFECTIVE PARAMETERS
     effective_floor = base_floor × temperature × heat  (clamped [0.02, 1.0])
     effective_drift = base_drift × temperature × heat  (clamped [0.05, 2.0])

  3. RUN FM PASS
     Start from best_partition with effective parameters.
     Each pass gets a different seed: base_seed + pass × 7.

  4. EVALUATE RESULT
     If pass found new best:
       Update best_partition.
       Cool down: heat × 0.7 (narrow search near this basin).

     Else if pass applied moves (perturbed state available):
       GREEDY-KICK: run greedy_descent on the perturbed end state.
       If kick found new best:
         Update best_partition.
         Moderate cool: heat × 0.9.
       Else:
         Heat up: heat × 1.3 (explore wider next time).

     Else:
       Heat up: heat × 1.3.

  5. CAPTURE END STATE
     Store the last perturbed end_partition in the result for
     diversity seeding in the evolutionary pool.

  6. STOPPING
     Stop after max_no_improve consecutive non-improving passes.
     Also stop on wall-clock deadline.
```

### Heat dynamics

Heat controls how aggressively the search explores:
- High heat → large floor/drift → accepts bad moves → broad exploration
- Low heat → small floor/drift → only good moves → local refinement
- Range: [0.1, 3.0]

The greedy-kick is the escape mechanism: after a non-improving FM pass,
the perturbed end state may be near a different local minimum. Full
greedy descent from there finds the bottom of that basin.

---

## 5. Parallel Search + Evolution (`parallel_search`)

Multi-start search with evolutionary pool management. Spawns threads
running independent (init → greedy → FM) pipelines, maintains a diverse
pool of partitions, and evolves new candidates via mutation and crossover.

### Pool Management

The pool holds up to `pool_size` (default 8) partitions. Insertion is
diversity-aware:

**Distance metric**: Rand index via contingency table. For two partitions
A and B, counts pairs of ops that are co-grouped in A but not B (or vice
versa), normalized to [0, 1]. O(n + ga×gb) computation.

**Insertion rules**:
- **Near-duplicate** (distance < 0.01): replace only if strictly better cost.
- **Pool not full**: always add.
- **Pool full**: find the least-unique entry (smallest nearest-neighbor
  distance, excluding the best-cost entry which is protected). Replace it if
  the candidate brings more diversity or better cost with decent diversity
  (distance > 0.02).

### Generation 0 (initialization)

```
For each (strategy, seed) task in parallel:
  1. init = strategy.init(prob, dag)       // trivial, random, chain+edge, etc.
  2. greedy = greedy_descent(init)
  3. fm = fm_outer_loop(greedy)
  4. Insert fm.best_partition into pool
  5. Insert fm.end_partition into pool     // diversity candidate
```

### Generation 1+ (evolution)

```
While deadline not reached:
  For each task in parallel:
    1. SELECT OPERATOR
       - 1/3 of tasks: crossover (pick 2 random parents from pool)
       - 2/3 of tasks: mutation (pick 1 random parent, apply 2–6 mutations)

    2. REFINE
       child = greedy_descent(mutated/crossed)
       fm = fm_outer_loop(child)

    3. INSERT
       pool_insert(fm.best_partition)      // best result
       pool_insert(fm.end_partition)       // diversity candidate

  Stop after 25 generations without improvement.
```

### Mutation Operators

All mutations are structurally random — they ignore cost and only check
feasibility. This provides the diversity that gain-guided FM cannot.

| Operator | Description |
|----------|-------------|
| `mutate_merge` | Pick two random adjacent groups, fuse them. |
| `mutate_split` | Pick a random group (≥3 ops), cut at a random bridge edge. |
| `mutate_reassign` | Pick a random border op, move to a random neighbor group. Rejects if remainder disconnects. |
| `mutate_block_move` | Grow a connected block of 2–3 ops from a border op seed, move the block to a random neighbor group (or eject as new group). |
| `mutate_eject` | Pick a random op from a random group (≥2 ops), eject as singleton. Handles disconnection via `eval_eject`. |
| `mutate_tensor_merge` | Pick a random tensor with ≥2 consumers in different groups. Try full merge of all consumer+producer groups; fall back to extracting just the consumer+producer ops. |

`mutate_compound(N)` applies N random mutations (uniform choice among
the six operators). Typical N = 2–6.

### Crossover

Agreement-based crossover (Sanders & Schulz style):

```
1. Map each op to its group in parent A and parent B.
2. Union-Find: ops that share the same group in BOTH parents are
   clustered together (they "agree" on grouping).
3. Shuffle clusters randomly.
4. Greedily assign each cluster to the child partition:
   - Try merging with each existing child group adjacent via DAG edges.
   - Pick the cheapest feasible merge.
   - If merging is worse than standalone by >100, create a new group.
   - Fall back to standalone or forced merge if needed.
5. Rebuild index after each cluster insertion.
```

This preserves structural decisions that both parents agree on while
allowing disagreed ops to be reassigned freely.

---

## 6. Validity and Subgraph Constraints

All validity checking is delegated to `Subgraph::create` via `eval_set`.
The search does NOT maintain separate shape-matching or connectivity checks.
`Subgraph::create` enforces:

- **Connectivity**: ops must be connected via DAG edges or shared-input
  co-consumer edges.
- **Ephemeral fan-out**: each ephemeral tensor (produced and consumed
  internally) has exactly one consumer within the subgraph.
- **Boundary output consistency**: all boundary outputs must have the same
  (width, height) dimensions.
- **PW sink constraint**: if any boundary output is produced by a Pointwise
  op, k is forced to 1 in the tiling parameter search.

The only pre-filter is `merge_creates_cycle`, which cheaply rejects merges
that would create scheduling cycles, avoiding an expensive `eval_set` call.

---

## 7. Index Maintenance

The Partition maintains an `op_to_groups_` index mapping each op to its
containing group(s). This enables O(1) `groups_of(op)` and O(neighbors)
`adjacent_groups(gi)` lookups.

Both `apply_move` (greedy) and `apply_fm_move` (FM) call
`part.rebuild_index()` after every successful mutation. This ensures all
subsequent lookups see consistent state.

---

## 8. CostCache

`eval_set` results are memoized in a thread-safe `CostCache` keyed by the
sorted op-index set. All partitions within a `parallel_search` invocation
share a single cache instance. This is safe because subgraph cost depends
only on the op set and the immutable Problem/DAG — not on the partition
context.

Cache statistics (entries, hits, misses) are logged at the end of
parallel_search for tuning visibility.