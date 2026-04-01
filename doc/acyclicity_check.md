# Acyclicity Check for Partition Moves

## The group DAG

Each alive group is a node. An edge G_a → G_b exists when G_a produces a tensor T as a **boundary output** and G_b consumes T as a **boundary input**. The group DAG must be acyclic — otherwise no valid execution order exists.

The group DAG is an implicit structure derived from:
- `groups[g].ops` — which ops are in each group
- The op-level DAG — `tensor_producer[t]`, `tensor_consumers[t]`
- `groups_of(op)` — which groups contain each op (from `op_to_groups_`)

An edge G_a → G_b exists iff: there exists a tensor T such that `is_boundary_output_of(G_a.ops, T, dag)` AND some consumer of T is in G_b.

## Local BFS algorithm

All `acyclic_*_local` functions follow the same pattern:

1. Identify the **new edges** that the move would introduce
2. For each new edge A → B: check if B can already reach A in the current group DAG
3. If yes → the new edge closes a cycle → reject

The BFS uses the raw op/tensor DAG + `groups_of()`. It does NOT use `group_preds/group_succs` (which may be stale). The traversal:

```
for each op in group:
    for each output tensor T of op:
        for each consumer cop of T:
            if cop is internal to the group: skip
            for each group gj containing cop:
                if gj is alive and not visited: visit gj, enqueue
```

This is O(reachable_groups × ops_per_group × tensors_per_op).

## Per-move-type: which new edges appear?

### MERGE(ga, gb)
Internal edges between ga and gb disappear (become internal). But a cycle can exist through external groups: if there's a path ga → ext₁ → ... → extₖ → gb AND gb → ext₁' → ... → ga, the merge closes the loop.

**Check**: `acyclic_merge_local(ga, gb)` — BFS from external successors of {ga, gb}. Seed: for each op in ga∪gb, follow outputs to consumers NOT in ga∪gb, collect their groups. BFS forward through those groups. Cycle if BFS reaches back into {ga, gb}.

### STEAL(op, from, to)
Three sources of new edges:
1. **Op enters `to`**: op's input tensors create edges producer_group → to (same as RECOMPUTE check)
2. **`from` loses op**: if op produced T consumed internally by `from`, `from` now needs T from elsewhere → edge source_of_T → from
3. **External consumers lose `from` as source**: if op produced T that was boundary output of `from`, external consumers must get T from `to` or op's other groups → edge new_source → consumer_group

**Check**: `acyclic_steal_local(op, from, to)` — three-part BFS using `forward_reachable()`.

### EJECT(op, ga) with recomputed op
Equivalent to DE_RECOMPUTE when op exists in multiple groups. The remainder of ga loses op's outputs as ephemeral sources → needs them from op's other groups.

**Check**: `acyclic_de_recompute_local(op, ga)` — for each output T of op consumed in ga, check that at least one source group (containing another copy of op) is NOT forward-reachable from ga.

### SPLIT(ga → side_a, side_b)
Internal tensors become boundary edges between side_a and side_b. If side_a → side_b is a new edge AND side_b can reach side_a through external groups → cycle.

**Check**: `acyclic_split_local(side_a, side_b, ga)` — temporarily applies the split, runs full `is_acyclic()`. (A local BFS approach is possible but tricky with recomputed ops.)

### RECOMPUTE(op, gb)
Op is copied into gb. New input edges: for each input T of op not already consumed in gb, edges from T's producer groups → gb.

**Check**: `acyclic_recompute_local(op, gb)` — for each new input dependency, check that at least one producer group is NOT forward-reachable from gb.

### DE_RECOMPUTE(op, ga)
Op is removed from ga. Internal consumers in ga lose their ephemeral source. External consumers of op's boundary outputs from ga lose ga as a source.

**Check**: `acyclic_de_recompute_local(op, ga)` — two parts: (1) internal consumers need source not forward-reachable from ga, (2) external consumers need source not forward-reachable from them.

## The coupled case: chain-level DAG

Coupling chains group multiple groups into ordered sequences: G_a → G_b → G_c (linked via `next_group/prev_group`). The chain-level DAG has one node per chain. An edge chain₁ → chain₂ exists if any group in chain₁ produces a tensor consumed by a group in chain₂.

A move can be acyclic at the group level but cyclic at the chain level. This happens when merging groups from different chains — the merged group inherits both chains' positions, and the cross-chain dependency closes a loop.

### When to check chain-level acyclicity
- **MERGE/TENSOR_MERGE**: `acyclic_chain_merge_groups(cp, {ga, gb})` — collects all groups from the chains of ga and gb, then runs the same BFS pattern but at chain granularity.
- **COUPLE(ga, gb, t)**: The eval function `eval_couple` checks `acyclic_chain_merge(chain_of(ga), chain_of(gb))` before creating the coupling edge.
- **Other partition moves** (STEAL, EJECT, SPLIT): These don't create new coupling edges, so they can't create chain-level cycles directly. They may break existing coupling edges (handled by `invalidate_couplings` / `fix_chain_couplings`), but breaking edges can only remove cycles, not create them.

### The `acyclic_chain_merge` algorithm
1. Collect all groups in both chains → set G
2. BFS from external successors of G (groups not in G that are reachable via tensor edges)
3. Walk each external chain fully (follow next_group pointers)
4. Cycle if BFS reaches a group that produces a tensor consumed by any group in G

This uses the raw op/tensor DAG (not `group_succs`) and chain pointers (`next_group`). It's O(reachable_chains × groups_per_chain × ops_per_group).

## Post-move cleanup for coupled partitions

After applying a partition move in the coupled context:
1. `fixup_coupling_<move>(cp, ...)` — updates `next_group/prev_group/retained` to reflect the partition change
2. `invalidate_couplings()` — removes retained tensors no longer valid (not boundary output of producer OR not boundary input of consumer)
3. `fix_chain_couplings()` — removes coupling edges that now form chain-level cycles (called after mutations in evolution, not after individual FM moves)

The prediction function (e.g., `coupled_merge_saving`) must simulate steps 1+2 to predict the correct coupling context. Step 3 is only needed after compound mutations (evolution), not after single FM moves.
