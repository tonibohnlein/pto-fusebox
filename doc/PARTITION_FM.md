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
`eval_set(ops)` which computes `Subgraph::create + best_cost` (memoized via CostCache).

### STEAL
Move a border op from group A to adjacent group B.

- **Precondition**: op is on the boundary of A (has DAG neighbor outside A).
  Removing op from A must leave A connected and non-empty.
- **Cost**: `eval_set(A - {op}) + eval_set(B ∪ {op})`
- **Saving**: `cost(A) + cost(B) - new_cost(A) - new_cost(B)`

### MERGE
Fuse two adjacent groups A and B into one group.

- **Precondition**: A and B share a DAG edge (via the initiating op).
- **Cost**: `eval_set(A ∪ B)`
- **Saving**: `cost(A) + cost(B) - cost(A ∪ B)`
- **Effect**: Group B is marked dead. All ops move to A.

### RECOMPUTE
Duplicate an op into a neighbor group (keep in both groups).

- **Precondition**: op is in group A, group B is adjacent, op not already in B.
- **Cost**: `eval_set(B ∪ {op})` (group A unchanged)
- **Saving**: `cost(B) - new_cost(B)`
- **Effect**: op now exists in both A and B. B recomputes the op locally
  instead of loading its output tensor from slow memory.

### EJECT
Remove a border op from its group into a new singleton group.

- **Precondition**: op is on the boundary of group A. Removing it leaves A
  connected (checked via `ejectable_ops`).
- **Cost**: `eval_set(A - {op}) + eval_set({op})`
- **Saving**: `cost(A) - cost(A - {op}) - cost({op})`
- **Effect**: Creates a new group containing just the ejected op.

### INTERNAL_EJECT
Remove an internal op from its group. The remainder may split into
multiple connected components, each becoming a separate group.

- **Precondition**: op is internal (no DAG neighbors outside group A).
  Group has 3–15 ops.
- **Cost**: `eval_set({op}) + Σ eval_set(component_i)` for each connected
  component of `A - {op}`.
- **Saving**: `cost(A) - cost({op}) - Σ cost(component_i)`
- **Effect**: Group A is replaced by 1 singleton + N component groups.

### SPLIT
Split a group at a bridge edge into two non-trivial parts.

- **Precondition**: Group A has a bridge edge (u, v) — an edge whose removal
  disconnects the group. Group has 3–15 ops.
- **Cost**: `eval_set(side_A) + eval_set(side_B)` where `side_A` contains u,
  `side_B` contains v.
- **Saving**: `cost(A) - cost(side_A) - cost(side_B)`
- **Effect**: Group A keeps `side_A`, a new group gets `side_B`.

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
   d. Bump generation counters of affected groups
   e. Regenerate moves for dirty groups (affected + adjacent)
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
current best move across all its groups and neighbors.

```
ActiveSet:
  entries: [{op=3, move=STEAL(3→g2, saving=+50)},
            {op=7, move=MERGE(g1∪g3, saving=-20)},
            ...]
  active_ops: {3, 7, ...}   (set of activated ops)
  locked: {1, 5, ...}       (set of locked ops)
```

### Algorithm

```
1. INITIALIZE
   Collect all activatable ops (border + internal for groups ≤ 15)
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
      apply_fm_move(partition, move) → returns affected group indices
      Track cumulative gain = start_cost - current_cost

   c. SNAPSHOT
      If current_cost < best_seen → snapshot partition as new best

   d. DRIFT CHECK
      If (best_cumulative_gain - cumulative_gain) > max_drift → abort pass

   e. UPDATE NEIGHBORS
      update_affected(affected_groups):
        For each active, unlocked entry touching affected groups:
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
| MERGE(op) | op + boundary ops of the absorbed group |
| RECOMPUTE(op) | op |
| EJECT(op) | op |
| INTERNAL_EJECT(op) | op |
| SPLIT(op) | op |

Locked ops cannot initiate new moves but CAN be moved by other ops' moves
(e.g., a locked op can be part of a MERGE initiated by an unlocked op).

### Floor and drift

- **Floor**: minimum saving to accept. `floor = start_cost × floor_fraction`.
  Moves with `saving < -floor` are rejected.
- **Drift**: maximum allowed degradation below the pass-best.
  `max_drift = start_cost × max_drift_fraction`.
  If cumulative gain drops below `best_cumulative_gain - max_drift`, the pass aborts.

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