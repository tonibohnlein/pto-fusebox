# Ephemeral Gap Check

## What is an ephemeral gap?

A tensor T is **ephemeral** in group G if T is produced by an op in G AND consumed by another op in G. Ephemeral tensors are computed as intermediate results inside the subgraph and never written to slow memory. They don't appear as boundary outputs.

An **ephemeral gap** occurs when:
1. An external group G_ext needs tensor T as a boundary input
2. After a move, every group that produces T also consumes T internally (T is ephemeral everywhere)
3. T is never written to slow memory → G_ext cannot load it → invalid schedule

## The boundary output definition

`is_boundary_output_of(ops, t, dag)`:
- T is produced by an op inside `ops` (producer ∈ ops)
- T is NOT consumed by any op inside `ops` (all consumers are external)
- If even ONE consumer is inside `ops`, T is NOT a boundary output — it's ephemeral

This is strict: a tensor with both internal and external consumers is classified as ephemeral. The external consumers must get T from another group or it's a gap.

## How to check for a gap after a move

Given a proposed set of new groups (components) replacing some excluded groups:

```
creates_ephemeral_gap(proposed_ops, exclude_ga, exclude_gb)
```

For each tensor T:
1. Check if T is a boundary output of `proposed_ops`
2. If not (T is internal/ephemeral in proposed_ops), check if any OTHER alive group (not in the excluded set) needs T as a boundary input
3. If such a group exists and no other alive group (not excluded) produces T as a boundary output → gap

The `split_creates_ephemeral_gap(components, excluded_groups)` variant checks multiple proposed component sets simultaneously — important for SPLIT/EJECT where the old group is replaced by multiple new groups.

## Which moves can create a gap?

**MERGE(ga, gb)**: Tensor T was boundary output of ga, consumed by gb. After merge, T is internal. If external group G_ext also consumed T from ga, G_ext loses access.
→ Check: `creates_ephemeral_gap(merged_ops, ga, gb)`

**STEAL(op, from, to)**: Tensor T was boundary output of `from` (produced by op). After steal, T is produced by op in `to`. If remaining ops in `from` also consume T, T becomes ephemeral in `from` and only available from `to`. But if `to` also consumes T internally, T is ephemeral in `to` too → gap for external consumers.
→ Check: `split_creates_ephemeral_gap({new_to, new_from}, {from, to})`

**EJECT(op, ga)**: Same pattern — remainder of ga might lose a boundary output.
→ Check: `split_creates_ephemeral_gap({singleton, remainder_components...}, {ga})`

**SPLIT(ga → side_a, side_b)**: Internal tensors of ga become boundary between sides. But tensors that were boundary outputs of ga might become internal to one side.
→ Check: `split_creates_ephemeral_gap({side_a, side_b}, {ga})`

**RECOMPUTE(op, gb)**: Adding op to gb can make a tensor that was a boundary input of gb become internal (if op produces it). Generally safe since recompute only adds producers (more sources, not fewer). But should verify.

**DE_RECOMPUTE(op, ga)**: Removing op from ga — if op was the sole producer of T in ga and T was a boundary output, after removal T is no longer produced by ga. External groups lose access. Similar to EJECT.

**MERGE cannot create a gap for tensors that were already ephemeral.** Only tensors that WERE boundary outputs before the move and become ephemeral after can cause gaps.

## Locality

The gap check requires scanning all alive groups to find consumers of the affected tensors. However, only tensors whose boundary status CHANGES need checking — these are tensors on the boundary between the two groups being merged/split. The number of such tensors is bounded by the size of the boundary between the affected groups.
