# Solution FM Search

## Overview

The solution FM search optimizes the full solution: partition structure AND
retain decisions simultaneously. It operates on a `Solution` — an ordered
sequence of `ScheduleStep` objects, each containing a subgraph, tiling
configuration, and retain set.

The architecture mirrors the partition FM with the same three layers:

```
solution_greedy_descent (hill climb, heap-based, no worsening)
    └── used standalone AND as subroutine inside:

solution_fm_pass (one FM pass: random init → active set → locking → drift)
    └── called repeatedly by:

solution_fm_search (parallel outer loops, adaptive cooling, greedy-kick)
```

---

## 1. Move Types

Eight move types. Six operate on partition structure (same vocabulary as
partition FM), two operate on retain decisions (unique to solution FM).

All moves are evaluated locally — only the 1–2 affected steps are
recomputed. The `filter_retain` helper ensures retain sets stay valid
after structural changes (no retaining non-boundary tensors).

### Partition moves (operate on step structure)

#### STEAL
Move a border op from step A to adjacent step B.

- **Precondition**: op is on the boundary of step A. Removing op leaves
  A connected. Step B is at position A±1.
- **Cost**: `best_cost(A - {op}) + best_cost(B ∪ {op})`
- **Saving**: `cost(A) + cost(B) - new_cost(A) - new_cost(B)`
- **Retain handling**: `filter_retain` on both new subgraphs. If the stolen
  op was the producer of a retained tensor, the tensor is dropped from the
  retain set.

#### MERGE
Fuse two adjacent dependent steps into one.

- **Precondition**: step B depends on step A (B consumes a tensor produced by A).
  B is at position A+1.
- **Cost**: `best_cost(A ∪ B)`
- **Saving**: `cost(A) + cost(B) - cost(merged)`
- **Effect**: Step B is removed. Step count decreases by 1 (structural change).
- **Retain handling**: merged step inherits B's retain set, filtered for the
  new subgraph boundaries.

#### RECOMPUTE
Duplicate an op into an adjacent step (keep in both steps).

- **Precondition**: op is in step B, produces a tensor consumed by step A
  as a boundary input. Op not already in A.
- **Cost**: `best_cost(A ∪ {op})` (step B unchanged)
- **Saving**: `cost(A) - new_cost(A)`
- **Effect**: Step A now recomputes the op locally, avoiding a slow-memory
  load of the tensor. The tensor is no longer a boundary input of A.
- **Retain handling**: `filter_retain` on expanded A (the tensor may no longer
  be a boundary, so its retain entry is dropped).

#### EJECT
Extract a border op from a step into a new singleton step.

- **Precondition**: op is on the boundary. Removing it leaves the step
  connected. Step has ≥ 2 ops.
- **Cost**: `best_cost(A - {op}) + best_cost({op})`
- **Saving**: `cost(A) - cost(remainder) - cost(singleton)`
- **Effect**: New step inserted after A. Step count increases (structural change).
- **Retain handling**: remainder's retain set is filtered. Singleton has no retain.

#### INTERNAL_EJECT
Extract an internal op. The remainder may split into multiple components.

- **Precondition**: op is internal (no DAG neighbors outside step). Step has
  3–15 ops.
- **Cost**: `eval_eject(op)` via temporary Partition → singleton + components.
- **Saving**: `cost(A) - cost(singleton) - Σ cost(components)`
- **Effect**: Step A is replaced by 1 + N new steps (structural change).
- **Retain handling**: all new steps start with empty retain sets.

#### SPLIT
Split a step at a bridge edge. Optionally retain the bridge tensor.

- **Precondition**: step has a bridge edge (u, v). Step has 3–15 ops.
- **Cost**: `best_cost(side_A, retain=bridge_tensors) + best_cost(side_B)`
- **Saving**: `cost(A) - cost(side_A) - cost(side_B)`
- **Effect**: Step A becomes side_A, new step for side_B (structural change).
- **Retain handling**: bridge tensors (outputs of first side that are inputs
  of second side) are automatically added to the first side's retain set.
  Second side inherits A's original retain, filtered.

### Retain moves (operate on data residency)

#### RETAIN_ADD
Add a tensor to a step's retain set (keep in fast memory for the next step).

- **Precondition**: tensor T is a boundary tensor of step A (input, output,
  or sink). Step A+1 has T as a boundary input. T is retainable (fits in
  fast memory). T not already retained.
- **Cost**: `best_cost(A, retain ∪ {T}) + best_cost(A+1, entering ∪ {T})`
- **Saving**: eliminates slow-memory load of T at step A+1, but may force
  a smaller tiling at step A (T occupies fast memory).

#### RETAIN_REMOVE
Remove a tensor from a step's retain set.

- **Precondition**: tensor T is currently in step A's retain set.
- **Cost**: `best_cost(A, retain - {T}) + best_cost(A+1, entering - {T})`
- **Saving**: frees fast memory at step A (may enable better tiling), but
  step A+1 must reload T from slow memory.

---

## 2. Greedy Hill Climb (`solution_greedy_descent`)

Pure descent — only accepts improving moves. Uses a priority heap with
per-step generation counters for lazy staleness detection.

### Data structure

Priority heap (`std::priority_queue<SolutionMove>`) where moves store
`gen_a` and `gen_b` (step generation counters at evaluation time).

Each step in `SolState` has a `gen` counter incremented when that step changes.

### Algorithm

```
1. INITIALIZE
   For each step si:
     generate_step_moves(si, heap, floor=0):
       For each op in step si:
         best_move_for_op(op) → push to heap with gen_a, gen_b
       For each boundary/retained tensor of step si:
         best_move_for_tensor(t) → push to heap with gen_a, gen_b

2. While heap not empty:
   a. Pop top move m
   b. If m.saving ≤ 0 → stop (greedy: only improving)
   c. Staleness check (is_stale):
      - m.step_a out of range → stale
      - m.gen_a ≠ state.gen[step_a] → stale
      - m.step_b exists and gen_b ≠ state.gen[step_b] → stale
      - RETAIN moves: gen_b ≠ state.gen[step_a + 1] → stale
      If stale → skip
   d. Apply move → returns affected range [lo, hi)
   e. Bump generations:
      Structural change (SPLIT/MERGE/EJECT/INTERNAL_EJECT) → bump ALL
        (step indices shift, all heap entries are stale)
      Non-structural (STEAL/RECOMPUTE/RETAIN) → bump affected + neighbors
   f. Rebuild costs from lo onwards (incremental)
   g. Regenerate moves for steps in [lo-1, hi+1]
```

### Staleness mechanism

Same principle as partition greedy: generation counters make stale heap
entries free to reject. Structural moves (which change step count and shift
indices) use `bump_all()` to invalidate everything — a necessary cost since
all step indices change. Non-structural moves only bump the 2–4 affected steps.

---

## 3. FM Inner Pass (`solution_fm_pass`)

One pass: random initialization, active set with locking, drift control.

### Active Set

Flat vector of entries. Each entry is either an op-entry or a tensor-entry:

```
SolActiveSet:
  entries: [{is_tensor=false, id=op3,  move=STEAL(...)},
            {is_tensor=true,  id=T42,  move=RETAIN_ADD(...)},
            {is_tensor=false, id=op7,  move=MERGE(...)},
            ...]
  active_ops: {3, 7, ...}
  active_tensors: {42, ...}
  locked_ops: {1, ...}
  locked_tensors: {15, ...}
```

### Algorithm

```
1. INITIALIZE (small random subset, 1:2 ops:tensors)
   total = max(1, √(num_ops + num_retainable_tensors))
   init_tensors = min(available, 2/3 × total)   ← tensor-biased
   init_ops = min(available, total - init_tensors)

   Can start with just 1 tensor or 1 op.
   Different seeds → different random subsets → different exploration.

   For each selected tensor t:
     activate_tensor(t) → compute best_move_for_tensor(t) → add entry
   For each selected op:
     activate_op(op) → compute best_move_for_op(op) → add entry

2. MAIN LOOP (until exhaustion or max_drift):
   a. POP BEST
      Scan entries for best unlocked move
      Lock the initiating entity (op or tensor)

   b. APPLY MOVE
      apply_move(state, move) → returns affected range [lo, hi)
      Rebuild costs from lo onwards

   c. POST-MOVE LOCKING (prevent reversal):
      See locking table below

   d. SNAPSHOT
      If total_cost < best_seen → snapshot steps

   e. DRIFT CHECK
      If (best_cumul_gain - cumul_gain) > max_drift → abort

   f. UPDATE NEIGHBORS
      update_affected(state, lo, hi):
        Collect ops and tensors in affected step range [lo-1, hi+1]
        Recompute best_move for matching unlocked entries
      activate_step(si) for each affected step:
        Add all ops and tensors of the step as new entries

3. RETURN best snapshot + perturbed end state
```

### What gets locked

| Move applied | Initiator locked by pop_best | Additional locks |
|---|---|---|
| **STEAL(op)** | op | — |
| **MERGE(op)** | op | — |
| **RECOMPUTE(op)** | op | — |
| **EJECT(op)** | op | — |
| **INTERNAL_EJECT(op)** | op | — |
| **SPLIT(op, op2)** | op | op2 |
| **RETAIN_ADD(tensor T)** | T | producer op of T, consumer ops of T in step A+1 |
| **RETAIN_REMOVE(tensor T)** | T | — |

The RETAIN_ADD locking is critical:
- **Lock T**: prevents RETAIN_REMOVE from undoing it.
- **Lock producer op**: prevents STEAL/EJECT from breaking the boundary that T
  lives on. If the producer moves to another step, T is no longer a boundary
  output and the retain becomes invalid.
- **Lock consumer ops in step A+1**: prevents STEAL from removing the
  beneficiary. If the consumer moves out, step A+1 no longer needs T.

### Floor and drift

Same semantics as partition FM:
- **Floor**: `start_cost × floor_fraction`. Moves below `-floor` are rejected.
- **Drift**: `start_cost × max_drift_fraction`. Pass aborts if cumulative gain
  drops below `best - max_drift`.

---

## 4. FM Outer Loop (`solution_fm_search`)

Parallel outer loops with adaptive cooling. Each thread runs independently
with a different seed.

### Parallelism

```
N = hardware_concurrency() threads
Each thread t:
  seed_offset = t × 1000
  Runs solution_fm_search_single(init_steps, seed + seed_offset)
  Reports best to shared result under mutex
```

Different seeds → different random active set initialization → different
first moves → different exploration trajectories. The `update_affected`
propagation means each thread explores a different region of the solution
space even from the same starting point.

### Single-thread outer loop

```
best_steps = initial solution steps
heat = 1.0

For each pass (up to max_passes or deadline):

  1. TEMPERATURE (cosine annealing)
     temperature = 0.1 + 0.9 × 0.5 × (1 + cos(π × progress))

  2. EFFECTIVE PARAMETERS
     effective_floor = base_floor × temperature × heat  [0.02, 1.0]
     effective_drift = base_drift × temperature × heat  [0.05, 2.0]

  3. RUN FM PASS
     seed = base_seed + pass × 7
     Start from best_steps with effective floor/drift.

  4. EVALUATE
     If improved:
       Update best_steps. Cool down: heat × 0.7

     Else if perturbed state available:
       GREEDY-KICK: solution_greedy_descent(perturbed) + optimize_retain
       If kick improved:
         Update best_steps. Moderate cool: heat × 0.9
       Else:
         Heat up: heat × 1.3

     Else:
       Heat up: heat × 1.3

  5. STOP after max_no_improve non-improving passes or deadline
```

---

## 5. Comparison: Partition FM vs Solution FM

### Identical

- Three-layer architecture (greedy → FM pass → outer loop)
- Heap-based greedy with generation-counter staleness
- Active set with pop_best / lock / update_affected / activate_neighbors
- Floor and drift control
- Cosine temperature annealing with adaptive heat
- Greedy-kick escape mechanism
- Deadline awareness at all levels

### Different

| Aspect | Partition FM | Solution FM |
|---|---|---|
| **Operates on** | Partition (groups of ops) | Solution (ordered steps with retain) |
| **Move types** | 6 (partition only) | 8 (6 partition + 2 retain) |
| **Active entity** | Op | Op + Tensor |
| **Init bias** | All border/internal ops | 1:2 ops:tensors (tensor-biased) |
| **Locking** | Per-op | Per-op + per-tensor + RETAIN producer/consumer |
| **Structural changes** | Groups use stable indices (dead = gap) | Steps shift on SPLIT/MERGE → bump_all |
| **Greedy-kick** | `greedy_descent(perturbed)` | `solution_greedy_descent + optimize_retain` |
| **Parallelism** | Via `parallel_search` (evolutionary) | Built-in: N threads with seed offsets |
| **Cost model** | `eval_set(ops)` (no retain) | `best_cost(retained_entering, retain_these)` |