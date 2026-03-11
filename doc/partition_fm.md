# Partition FM Search

## Overview

The partition FM search finds the optimal assignment of operations to subgroups.
It operates on a `Partition` — a mutable collection of groups, where each group
is a set of op indices that will execute as one subgraph step.

The architecture has three layers:

```
greedy_descent (hill climb, no worsening moves)
    └── used standalone AND as subroutine inside:

fm_inner_pass (one FM pass: random init → locking → drift control)
    └── called repeatedly by:

fm_outer_loop (adaptive cooling, greedy-kick escape)
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
  co-consumer outside A). Removing op from A must leave A as a valid
  subgraph (connected, feasible). Checked implicitly by `eval_set(A - {op})`.
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

### RECOMPUTE
Duplicate an op into a neighbor group (keep in both groups).

- **Precondition**: op is in group A, group B is adjacent (DAG edge or
  shared input), op not already in B.
- **Cost**: `eval_set(B ∪ {op})` (group A unchanged)
- **Saving**: `cost(B) - new_cost(B)`
- **Effect**: op now exists in both A and B. B recomputes the op locally
  instead of loading its output tensor from slow memory.

### EJECT
Remove a border op from its group into a new singleton group.

- **Precondition**: op is on the boundary of group A. Removing it must leave
  A as a valid subgraph (checked by `eval_eject` which calls `eval_set`
  on each connected component of the remainder).
- **Cost**: `eval_set(components of A - {op}) + eval_set({op})`
- **Saving**: `cost(A) - Σ cost(component_i) - cost({op})`
- **Effect**: Group A is replaced by its remainder component(s). A new
  singleton group is created for the ejected op. If the remainder splits
  into multiple components, each becomes its own group.

### INTERNAL_EJECT
Remove an internal op from its group. The remainder may split into
multiple connected components, each becoming a separate group.

- **Precondition**: op is internal (no DAG neighbors or co-consumers outside
  group A). Group has 3–15 ops (capped to avoid expensive evaluation).
- **Cost**: same as EJECT — `eval_eject` handles both uniformly.
- **Saving**: same formula as EJECT.
- **Effect**: same as EJECT. The distinction is only in move generation:
  EJECT is generated for border ops, INTERNAL_EJECT for internal ops.

### SPLIT
Split a group at a bridge edge into two non-trivial parts.

- **Precondition**: Group A has a bridge edge (u, v) — a DAG edge whose removal
  disconnects the group (checked using the full connectivity graph including
  co-consumer edges). Group has 3–15 ops.
- **Cost**: `eval_set(side_A) + eval_set(side_B)` where `side_A` contains u,
  `side_B` contains v.
- **Saving**: `cost(A) - cost(side_A) - cost(side_B)`
- **Effect**: Group A keeps `side_A`, a new group gets `side_B`.
- **Note**: Split candidates are DAG edges (producer-consumer), but the bridge
  detection BFS uses the full `op_neighbors` graph (including co-consumer edges)
  to determine connectivity.

---

## 2. Greedy Hill Climb (`greedy_phase`)

Pure descent — only accepts improving moves.

### Data structure

Priority heap (`std::priority_queue<Move>`) with lazy staleness detection via
per-group generation counters.

### Algorithm

```
1. Generate moves for ALL groups, push to heap
   For each group gi:
     generate_moves(gi, heap, floor=0, tabu=null)
       → evaluates MERGE, STEAL, RECOMPUTE for each neighbor op
       → evaluates EJECT for each border op
       → evaluates INTERNAL_EJECT for each internal op (groups ≤ 15 ops)
       → evaluates SPLIT for each bridge edge (groups ≤ 15 ops)

2. While heap not empty:
   a. Pop top move m
   b. Staleness check: verify m.gen_a == groups[m.ga].gen
      For STEAL, MERGE, RECOMPUTE: also verify m.gen_b == groups[m.gb].gen
      If stale → skip (O(1) rejection)
   c. Apply move (re-verifies: if actual saving flipped sign, reject)
   d. Rebuild op_to_groups_ index
   e. Bump generation counters of affected groups
   f. Regenerate moves for dirty groups (affected + adjacent)
```

### Staleness mechanism

Each group has a `gen` counter, incremented whenever the group changes.
Moves store `gen_a` and `gen_b` at evaluation time. On pop, if the stored
gen doesn't match the current gen, the move is stale and skipped.
This avoids re-evaluating all moves after each change — only moves
touching modified groups are lazily invalidated.

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
neighbor groups. For each group `gx` containing op:
- If op is a border op of `gx`: evaluate STEAL, MERGE, RECOMPUTE for each
  neighbor group, and EJECT from `gx`.
- If op is internal in `gx` (and group size 3-15): evaluate INTERNAL_EJECT
  and SPLIT at DAG edges incident to op.

Returns the single best move across all candidates. Moves with
`saving < -floor` or `|saving| < 1.0` (epsilon) are filtered.

### Algorithm

```
1. INITIALIZE
   Collect all activatable ops (border + internal for groups 3-15 ops)
   Select random subset (size = cfg.init_count)
   For each selected op:
     activate(op) → compute best_move_for(partition, op, floor, locked)
                   → add entry to active set

2. MAIN LOOP (until exhaustion or max_drift):
   a. POP BEST
      Scan entries for unlocked op with highest saving
      Lock the initiating op
      For MERGE: also lock boundary ops of the partner group

   b. APPLY MOVE
      apply_fm_move(partition, move) → mutates partition, rebuilds index,
                                        returns affected group indices

   c. SNAPSHOT
      If current_cost < best_seen → snapshot partition as new best

   d. DRIFT CHECK
      If (best_cumulative_gain - cumulative_gain) > max_drift → abort pass

   e. UPDATE NEIGHBORS
      update_affected(affected_groups):
        For each active, unlocked entry touching affected or adjacent groups:
          Recompute best_move_for(op)
      activate_neighbors_of(affected_groups):
        For each affected + adjacent group:
          Activate all border + internal ops (adds new entries)

3. RETURN best snapshot + perturbed end state
```

### What gets locked

| Move applied | What is locked |
|---|---|
| STEAL(op) | op (the initiating/moved op) |
| MERGE(op) | op + neighbor ops of op in the absorbed group |
| RECOMPUTE(op) | op |
| EJECT(op) | op |
| INTERNAL_EJECT(op) | op |
| SPLIT(op) | op |

Locked ops cannot initiate new moves but CAN be moved by other ops' moves
(e.g., a locked op can be part of a MERGE initiated by an unlocked op).

### Floor and drift

- **Floor**: minimum saving to accept. `floor = start_cost × floor_fraction`.
  Moves with `saving < -floor` are rejected. Note: floor is computed once at
  the start of the pass and does not adapt as cost changes during the pass.
- **Drift**: maximum allowed degradation below the pass-best.
  `max_drift = start_cost × max_drift_fraction`.
  If cumulative gain drops below `best_cumulative_gain - max_drift`, the pass
  aborts.

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
     Decays from 1.0 → 0.1 over the pass budget.

  2. EFFECTIVE PARAMETERS
     effective_floor = base_floor × temperature × heat  (clamped [0.02, 1.0])
     effective_drift = base_drift × temperature × heat  (clamped [0.05, 2.0])

  3. RUN FM PASS
     Start from best_partition with effective parameters.
     Each pass gets a different seed: base_seed + pass × 7.

  4. EVALUATE RESULT
     If pass found new best:
       Update best_partition
       Cool down: heat × 0.7 (narrow search near this basin)

     Else if pass applied moves (perturbed state available):
       GREEDY-KICK: run greedy_descent on the perturbed end state
       If kick found new best:
         Update best_partition
         Moderate cool: heat × 0.9
       Else:
         Heat up: heat × 1.3 (explore wider next time)

     Else:
       Heat up: heat × 1.3

  5. STOPPING
     Stop after max_no_improve consecutive non-improving passes.
     Also stop on wall-clock deadline (cfg.deadline).
```

### Heat dynamics

Heat controls how aggressively the search explores:
- High heat → large floor/drift → accepts bad moves → broad exploration
- Low heat → small floor/drift → only good moves → local refinement
- Range: [0.1, 3.0]

The greedy-kick is the escape mechanism: after a non-improving FM pass,
the perturbed end state may be near a different local minimum. Greedy
descent from there can find a basin that FM's locking prevented it from
reaching during the pass.

---

## 5. Validity and Subgraph Constraints

All validity checking is delegated to `Subgraph::create` via `eval_set`.
The FM search does NOT maintain separate shape-matching or connectivity checks.
`Subgraph::create` enforces:

- **Connectivity**: ops must be connected via DAG edges or shared-input
  co-consumer edges.
- **Ephemeral fan-out**: each ephemeral tensor has exactly one consumer
  within the subgraph.
- **Boundary output consistency**: all boundary outputs must have the same
  (width, height) dimensions.
- **PW sink constraint**: if any boundary output is produced by a Pointwise
  op, k is forced to 1 in the tiling parameter search.

The only pre-filter is `merge_creates_cycle`, which cheaply rejects merges
that would create scheduling cycles, avoiding an expensive `eval_set` call.

---

## 6. Index Maintenance

The Partition maintains an `op_to_groups_` index mapping each op to its
containing group(s). This enables O(1) `groups_of(op)` and O(neighbors)
`adjacent_groups(gi)` lookups.

`apply_fm_move` calls `part.rebuild_index()` after every successful mutation.
This ensures all subsequent `ActiveSet` operations (update_affected,
activate_neighbors_of) see consistent index state.