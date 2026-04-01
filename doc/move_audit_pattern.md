# Move Audit Pattern

Step-by-step pattern for auditing and fixing local search move types
(MERGE, STEAL, EJECT, SPLIT, RECOMPUTE, DE_RECOMPUTE, TENSOR_MERGE, TENSOR_EXTRACT).

## Step 1: Identify ALL code paths

Every move type can appear in up to 7 code paths:

1. **FM eval** — `best_move_for()` in `fm_search.cpp`. Evaluates candidate moves for an op, returns best.
2. **FM pass** — `fm_pass.cpp`. Pops from heap, applies via `apply_fm_move()`, refreshes heap.
3. **Coupled FM eval** — `best_coupled_move_for_op()` in `coupled_fm_search.cpp`. Delegates partition moves to `best_move_for()`, adds chain-level checks, recomputes gain with coupling context.
4. **Coupled FM pass** — `coupled_fm_pass.cpp`. Pops from coupled heap, applies via `apply_coupled_fm_move()` which calls `apply_fm_move()` then runs the coupling fixup.
5. **Evolution mutation** — `evolution.cpp`. Random move selection + apply for partition mutations.
6. **Coupled evolution** — `mutate_compound_coupled()` in `evolution.cpp`. Calls `mutate_compound()` (which runs partition mutations), then `invalidate_couplings()` + `fix_chain_couplings()`.
7. **Coupling greedy descent** — `coupling_greedy_pass()` in `coupling_search.cpp`. For coupling-specific moves (COUPLE, UNCOUPLE, RETAIN_FORCE_SPLIT).

List every call site for `eval_<move>`, `apply_<move>`, and the coupling prediction/fixup functions.

## Step 2: Describe the move logic

Before verifying code, write down the complete semantics of the move. This is the specification that all code paths must implement.

### 2a: Group changes

Describe exactly which groups change and how:
- Which groups exist BEFORE the move? Which are modified, killed, or created?
- What are the new op sets for each resulting group?
- Under what conditions does the move apply? (e.g., bridge edge exists for SPLIT, op is border for EJECT)

Example for SPLIT(ga, op_a, op_b):
- Before: ga = {op_a, ..., op_b, ...}
- Condition: (op_a, op_b) is a bridge edge in ga's internal subgraph
- After: ga becomes side_a (containing op_a), new group gb = side_b (containing op_b)
- side_a and side_b are the two connected components after removing edge (op_a, op_b)

### 2b: Operator-level effects

For each op in the affected groups, describe what changes:
- Does the op change group membership? (e.g., in STEAL, the moved op changes group)
- Does the op's border/internal status change? (e.g., in SPLIT, ops at the cut boundary become border ops)
- Does the op gain or lose DAG neighbors in its group? (affects which moves are available)

This determines which ops need their heap entries refreshed (Step 6).

### 2c: Tensor-level effects

For each tensor on the boundary of the affected groups:
- Does the tensor's boundary status change? (boundary output → internal, or internal → boundary output)
- Which tensors become newly ephemeral? (produced AND consumed within the same new group)
- Which tensors become newly exposed as boundary outputs? (were internal, now cross the split/steal boundary)

This determines:
- Ephemeral gap risk (Step 5)
- Which ops in external groups have stale cached moves (their group's boundary changed because a tensor they consume/produce changed status)

### 2d: Coupling effects

For coupled partitions, describe how the move interacts with chain links:
- Does the move kill a group that has incoming/outgoing chain links? → links must transfer or dissolve
- Does the move create a new group? → it starts with no chain links
- Does the move change a group's ops such that retained tensors become internal? → retained edge becomes stale
- Does the move change which tensors are boundary outputs/inputs of coupled groups? → entering/retain sets for neighbors change

Specifically enumerate:
- **Which retained tensors survive?** A retained tensor T on edge (G_prev → G) survives iff T is still a boundary output of G_prev AND a boundary input of G (after the move). Use `is_boundary_output_of` and `is_boundary_input_of` on the NEW op sets.
- **Which chain links transfer?** When a group dies (MERGE kills gb), its chain links must go somewhere. The prediction function must simulate this exactly.
- **Which chain links dissolve?** When transfer is impossible (e.g., survivor already has a link in that direction, or transfer would create a chain cycle).

## Step 3: Verify gain computation

The gain must be computed **locally** — only the groups directly affected by the move change cost. All other groups keep their existing cost.

### Uncoupled gain (FM search, evolution)

```
saving = sum(old_cost[g] for g in affected_groups) - sum(new_cost[g'] for g' in resulting_groups)
```

Each group's cost is `eval_set(group.ops)` which calls `Subgraph::create` + `best_cost()` (usually a cache hit). Only the groups that change membership need re-evaluation.

For each move type, identify exactly which groups change:
- **MERGE(ga, gb)**: old = {ga, gb}, new = {merged}. Saving = ga.cost + gb.cost - eval_set(ga.ops ∪ gb.ops).
- **STEAL(op, from, to)**: old = {from, to}, new = {from−op, to+op}. Saving = from.cost + to.cost - eval_set(from.ops−op) - eval_set(to.ops+op).
- **EJECT(op, ga)**: old = {ga}, new = {singleton(op), remainder_components...}. Saving = ga.cost - singleton.cost - sum(component.cost).
- **SPLIT(ga → a, b)**: old = {ga}, new = {a, b}. Saving = ga.cost - eval_set(a) - eval_set(b).
- **RECOMPUTE(op, gb)**: old = {gb}, new = {gb+op}. Saving = gb.cost - eval_set(gb.ops+op). Note: source groups are NOT affected (op stays in them).
- **DE_RECOMPUTE(op, ga)**: old = {ga}, new = {ga−op}. Saving = ga.cost - eval_set(ga.ops−op).

Verify: the gain formula in the code matches this. No extra groups should be included. The cost of unaffected groups cancels out in old−new.

### Coupled gain (coupled FM search)

In coupled partitions, each group's cost depends on its **coupling context**: entering tensors (retained from predecessor) and retain tensors (kept for successor). The gain must account for how the move changes this context.

```
coupled_saving = sum(group_cost(g) for g in affected) - sum(new_coupled_cost(g') for g' in resulting)
```

Where `group_cost(g)` = `eval_coupled_group_cost(g.ops, entering_for(g), retain_for(g))`.

The prediction function (e.g., `coupled_merge_saving`) must:
1. **Simulate the coupling fixup**: using the logic described in Step 2d.
2. **Determine the new entering/retain sets** for each resulting group.
3. **Filter retained tensors for boundary validity**: using the tensor analysis from Step 2c.
4. **Evaluate cost with the correct context**: `eval_coupled_group_cost(new_ops, filtered_entering, filtered_retain)`.

Verify: the prediction matches the actual state after apply by checking:
- The prediction's entering/retain sets match what `entering_for(g)` / `retain_for(g)` return after `fixup_coupling_<move>` + `invalidate_couplings()`
- The cost evaluation uses the same cache path (`evaluate_with_context`)

## Step 4: Verify feasibility — acyclicity

Using the specification from [doc/acyclicity_check.md](acyclicity_check.md), perform these checks:

1. **Identify which new edges the move introduces** in the group DAG (from Step 2c — which tensors change boundary status). Refer to the per-move-type section in acyclicity_check.md.
2. **Identify the correct `acyclic_*_local` function** for this move type. Verify it matches the new-edge analysis.
3. **For each code path** (from Step 1): verify the acyclicity check is called BEFORE apply. Check:
   - FM eval: called in `best_move_for`?
   - Evolution: called before `apply_<move>`?
   - `apply_<move>`: debug-only assert (not a runtime safety net)?
4. **For coupled paths**: is `acyclic_chain_merge_groups` needed? (Only for MERGE/TENSOR_MERGE that combine groups from different chains.)
5. **No redundant checks**: the acyclicity should be checked exactly once per code path (at eval time), not again inside `eval_<move>` or `apply_<move>`.

## Step 5: Verify feasibility — ephemeral gap

Using the specification from [doc/ephemeral_gap_check.md](ephemeral_gap_check.md), perform these checks:

1. **Identify which tensors change boundary status** after the move (from Step 2c). Which tensors become newly ephemeral? Which external groups could lose access?
2. **Identify the correct gap check function** for this move type (`creates_ephemeral_gap`, `split_creates_ephemeral_gap`, `eject_creates_ephemeral_gap`).
3. **For each code path** (from Step 1): verify the gap check is called BEFORE apply. Check:
   - FM eval: called in `best_move_for`?
   - Evolution: called before `apply_<move>`?
   - NOT inside `eval_<move>` (caller's responsibility, matching MERGE/SPLIT/STEAL pattern).
4. **No redundant checks**: gap should be checked exactly once per code path.

## Step 6: Verify heap updates after apply

The heap stores one `(op, best_move)` entry per op. After a move, some ops' cached best moves become stale. We must refresh exactly the ops whose best move could have changed.

### What makes an op's cached move stale?

An op's best move depends on:
1. **Its own group(s)** — group membership, group cost, border/internal status
2. **Adjacent groups** — groups reachable via the op's DAG neighbors (merge/steal targets)
3. **Acyclicity** — whether proposed moves are acyclic (depends on group DAG topology near the op)

A move invalidates an op's cached best move iff the move changes any of these for that op. Use the analysis from Step 2b (which ops change status) and Step 2c (which tensors change boundary status, affecting adjacent groups).

### Deriving the refresh set from Step 2

1. **Ops whose group changed** (Step 2a): all ops in modified/new groups → refresh
2. **Ops whose border status changed** (Step 2b): ops at the boundary of the move → refresh
3. **Ops in external groups that had a DAG neighbor in a changed group** (Step 2b+2c): their merge/steal candidates changed → refresh
4. **For coupled FM**: ops in chain neighbors of changed groups (Step 2d): their coupling context may have changed → refresh

### The refresh mechanism

```
refresh_after_move(affected_groups):
    relevant = affected ∪ adjacent_groups(each alive affected group)
    ops_to_refresh = all ops in relevant groups + their DAG neighbors
    for each op in ops_to_refresh:
        recompute_and_update(op)  // calls best_move_for, pushes to heap
    activate_group_ops(each relevant group)  // add newly exposed border ops
```

### Verifying completeness

For each move type, use the Step 2 analysis to check:
1. **`affected` set**: includes all modified + killed groups?
2. **Adjacent coverage**: does `adjacent_groups(surviving_group)` cover groups that were adjacent to the killed group? (Yes, if the surviving group now contains the killed group's ops.)
3. **No missed ops**: is there any op outside the refresh radius whose best move references a group that changed? Cross-check against Step 2b/2c.

### For coupled FM

`CoupledActiveSet::refresh_after_move` additionally includes **chain neighbors** (prev_group/next_group of affected groups) in the relevant set. Verify this covers all groups whose coupling context changed (Step 2d).

### Locking — preventing reversal

After refresh, lock ops to prevent the move from being immediately undone or reversed in the same FM pass. The lock set must be broad enough that NO op on the heap can initiate the exact reverse move.

For each move type, identify the reverse move and what ops could initiate it:
- **MERGE(ga, gb → ga)**: reverse = SPLIT of the merged group at any bridge edge. Lock all ops in the merged group (= old ga ∪ old gb). No op in the merged group can initiate a SPLIT.
- **STEAL(op, from, to)**: reverse = STEAL(op, to, from). To prevent: lock op (can't be stolen back). Also lock ops in `from` that are DAG neighbors of op (they could initiate a MERGE of `from` and `to` which effectively reverses the steal). Lock op + DAG neighbors of op in both `from` and `to`.
- **EJECT(op, ga → singleton + remainder)**: reverse = MERGE(singleton, remainder) or STEAL(op, singleton, remainder). Lock op.
- **SPLIT(ga → side_a, side_b)**: reverse = MERGE(side_a, side_b). Lock all ops in both sides (= all ops in old ga). No op can initiate a re-merge.

### Disconnected remainder check

Some moves can leave a group disconnected (ops that are not connected via DAG edges or shared input tensors):
- **STEAL(op, from, to)**: If op was the sole bridge between two halves of `from`, the remainder becomes disconnected. Unlike EJECT (which splits the remainder into connected components), STEAL does NOT split the remainder.
- **DE_RECOMPUTE(op, ga)**: Same issue — removing a recomputed op can disconnect the remainder.

For each move type, check:
1. Can the move disconnect the source group?
2. If yes, does the code split the remainder into connected components (like EJECT does)?
3. If not, should it? A disconnected group is wasteful (its cost model treats all ops as one subgraph, but they can't actually share tiles).

**Recommended**: After STEAL/DE_RECOMPUTE, check if the `from` group's remainder is still connected. If not, split it into connected components (additional groups). This should be tested with a unit test that steals a bridge op and verifies the remainder is split.

## Step 7: Fix flaws

After Steps 1–6 identify discrepancies between the Step 2 specification and the actual code, fix them:

### Gain computation flaws
- **Wrong groups in the formula**: code includes/excludes groups that shouldn't be affected
- **Coupled prediction doesn't simulate fixup correctly**: entering/retain sets differ from what the apply path produces. Fix by aligning the prediction with the fixup logic, including boundary filtering.
- **Double evaluation**: `apply_<move>` re-computes costs that `eval_<move>` already computed. Fix by passing precomputed values.

### Feasibility check flaws
- **Missing check**: a code path calls `apply_<move>` without checking acyclicity or gap first. Fix by adding the check before apply.
- **Wrong check**: the check doesn't match the move's semantics (e.g., checking `acyclic_extract_local` when `acyclic_split_local` is needed). Fix by using the correct check from Step 4/5.
- **Redundant check**: the same condition is checked in `eval_<move>`, in the caller, and in `apply_<move>`. Fix by assigning responsibility to one place: callers check feasibility, eval computes cost, apply mutates (with debug-only assert).

### Heap update flaws
- **Missing ops in refresh**: an op outside the refresh radius has a stale cached move that references a changed group. Fix by expanding the refresh set or moving the check to the correct scope.
- **Missing activation**: a newly exposed border op (after split/eject) isn't activated on the heap. Fix by ensuring `activate_group_ops` covers the new groups.
- **Wrong lock set**: too narrow (allows immediate reversal) or too broad (prevents useful follow-up moves). Fix by matching the lock set to the ops directly involved.

### Coupling flaws
- **Missing `invalidate_couplings`**: after apply, retained tensors that became internal are not cleaned up. Fix by adding `invalidate_couplings()` after the fixup.
- **Missing boundary filtering in prediction**: the prediction uses stale retained tensors. Fix by filtering with `is_boundary_output_of` / `is_boundary_input_of` on the new op sets.
- **Missing chain-level cycle check**: MERGE/TENSOR_MERGE of groups in different chains needs `acyclic_chain_merge_groups`. Other partition moves that don't create chain edges don't need it.

## Step 8: Remove redundancies and optimize

### Redundant computation
- **Don't compute the same set twice** — if the caller builds the new op sets for a gap check, pass them downstream
- **Don't check the same condition twice** — eval computes cost, caller checks feasibility (acyclicity, gaps), apply mutates
- **Single responsibility**: separate cost evaluation from feasibility checking from mutation
- Pass precomputed values via function parameters (e.g., `apply_merge(p, ga, gb, precomputed_cost)`, `apply_split(p, op_a, op_b, ga, &sr)`)

### Performance optimization
- **Avoid redundant `eval_set` calls**: `eval_set` → `Subgraph::create` + `best_cost()` is often a cache hit but still allocates and hashes. If the caller already has the cost, pass it through.
- **Avoid redundant `eval_split`/`eval_merge` calls**: these rebuild op sets, re-check feasibility, and re-compute costs. When `apply_<move>` is called immediately after eval with no intervening partition changes, the re-evaluation is wasted.
- **Lock set scope**: too broad a lock wastes search opportunities; too narrow risks thrashing. Match to the ops directly involved in the move.
- **`acyclic_split_local` uses `is_acyclic()` (full Kahn's)**: this is O(V+E) per call. For the FM inner loop where splits are evaluated frequently, consider a cheaper local BFS check if possible.

### Verification
After making changes:
1. **Build** — `cmake --build . -j$(nproc)` must succeed with zero errors
2. **Run ALL tests** — `ctest --output-on-failure -j$(nproc)` must pass 100%. This includes both old tests (regression) and new move-specific tests (correctness).
3. **Run benchmarks** — run 2-3 representative benchmarks and verify no warnings (BUG, GAIN MISMATCH, ephemeral gap, cycle). Compare solution quality (total latency) to ensure optimizations don't degrade results.

## Step 9: Write unit tests

Tests are part of the audit, not a separate follow-up. Write them as part of the same pass.

For each move type, create a test file `test/<move>_correctness_test.cpp` and add it to `CMakeLists.txt`. Cover:
- **Basic move** — gain correctness, partition valid after apply
- **Move with coupling** — coupled prediction matches actual cost change after apply + fixup + invalidate_couplings
- **Acyclicity rejection** — construct topology where the move would create a cycle, verify rejection
- **Ephemeral gap rejection** — construct topology where the move would strand a tensor, verify rejection
- **Coupling invalidation** — retained tensors that become internal after the move are removed
- **Edge cases** — dead groups, singleton groups, symmetric arguments
- **Disconnected remainder** — for STEAL/DE_RECOMPUTE: steal a bridge op and verify the remainder is split into connected components (or verify the disconnection is detected/handled)
- **Locking prevents reversal** — after applying a move, verify the reverse move is not immediately available on the heap (the relevant ops are locked)
- **Move-specific corner cases** — e.g., for SPLIT: non-bridge edge rejected, recomputed ops crossing the cut; for EJECT: remainder splits into multiple components; for STEAL: remainder becomes disconnected

### Verification after writing tests
Run both the new tests AND the full test suite to ensure nothing regresses:
```
ctest -R <move>_correctness -V    # new tests pass
ctest --output-on-failure -j$(nproc)  # all tests still pass
```
