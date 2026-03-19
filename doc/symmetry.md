# Symmetric Pattern Discovery for DAG Partitioning

## Goal

Given an ML computation DAG, discover maximal groups of isomorphic
connected components. Each group is a "pattern" — solve the partition
problem for one representative component, replicate the solution to
all others.

## Overview

```
Input:  Problem (ops, tensors, shapes) + DAG (edges, reachability)
Output: List of SymmetricPattern { components[], symmetry }
        sorted by (symmetry desc, component_size desc)
```

The algorithm has 5 stages:

```
1. Merkle hashing         → per-op structural fingerprints
2. Forward-only orbits    → groups of locally-equivalent ops
3. Orbit DAG + levels     → directed graph over orbits with distance labels
4. Greedy orbit merging   → grow symmetric regions (Phase 1 + Phase 2)
5. Dominance filter       → emit only maximal, non-redundant patterns
```

## Stage 1: Merkle Hashing

Compute structural fingerprints for each op via two passes over the DAG:

- **Forward pass** (bottom-up, 3 iterations): each op's hash encodes
  its type, cost, tensor shapes, and the hashes of its predecessors.
  Graph inputs are hashed by shape + fan-out count.

- **Backward pass** (top-down, 3 iterations): each op's hash encodes
  the hashes of its successors. Graph outputs are hashed by shape.

- **Combined**: `combined[op] = hash(fwd[op], bwd[op])`

The forward hash is used for orbit formation (Stage 2).
The combined hash is used by `MerkleHashes::equiv_classes` for
general structural equivalence queries outside the symmetry detector.

## Stage 2: Forward-Only Orbits

Group ops by their **forward hash only**.

Why not the combined hash? The backward pass sees "chain-end effects":
ops that feed into position 2 of a sequential aggregation chain hash
differently from ops feeding position 7, even though their local
structure is identical. The forward hash captures input-side structure
(type, shapes, predecessor topology) without this pollution.

The local `component_hash` (Stage 4) handles fine-grained isomorphism
during merging, so coarser orbits are safe — they give the merge
algorithm more material to work with.

Example (benchmark 17):
```
Combined hash: Op4,Op9 in one orbit, Op14 alone (backward sees different chain depths)
Forward hash:  Op4,Op9,Op14 all in one orbit (same input-side structure)
```

## Stage 3: Orbit DAG + Levels

Build a directed graph over orbits from tensor flow:

```
For each tensor T:
  if producer_orbit ≠ consumer_orbit:
    add edge producer_orbit → consumer_orbit
```

Compute two distance labels per orbit:
- **top_dist**: longest path from any source orbit (forward)
- **bot_dist**: longest path to any sink orbit (backward)

These are used by the level constraint in Stage 4.

## Stage 4: Greedy Orbit Merging

### Data Structure: Super-Nodes

A super-node is a merged set of orbits. It tracks:
```
struct SuperNode {
    orbit_ids:   which original orbits are included
    all_ops:     union of all ops
    components:  connected components (tensor-flow edges only)
    symmetry:    size of largest isomorphism class among components
    top_min/max: range of top_dist across included orbits
    bot_min/max: range of bot_dist across included orbits
}
```

Initially, each orbit is its own super-node with `symmetry = orbit_size`.

### Level Constraint

Two super-nodes may merge only if their level ranges are adjacent or
overlapping in BOTH dimensions:

```
top_ok = (A.top_max >= B.top_min - 1) AND (B.top_max >= A.top_min - 1)
bot_ok = (A.bot_max >= B.bot_min - 1) AND (B.bot_max >= A.bot_min - 1)
can_merge = top_ok AND bot_ok
```

This prevents merging orbits separated by intermediate orbits not in
either super-node, which would create cycles in the condensed DAG:
`merged_SN → intermediate → merged_SN`.

### Connected Components (Tensor-Flow Only)

When merging two super-nodes, compute connected components of the
union using **tensor-flow edges only** (producer → consumer), NOT
`dag.op_neighbors` (which includes co-consumer edges).

Co-consumer edges would collapse parallel branches that read a shared
input (e.g., 8 attention heads all reading T0) into one component,
destroying the symmetry.

### Local Component Hash (Isomorphism Check)

To check if two components are isomorphic, compute a **local Merkle
hash** for each component:

1. Build local predecessor/successor maps restricted to ops in the component
2. Count external inputs/outputs per op (boundary role)
3. Run 3 iterations of forward + backward hashing using only local edges
4. External inputs/outputs are hashed by tensor shape only (not global identity)
5. Component fingerprint = sorted local combined hashes

This makes ops locally isomorphic even when their global context differs.

Example: Op4 → Op15 vs Op14 → Op16 in benchmark 5.
Globally different (Op15 ≠ Op16), but within their respective 5-op
components, both Op4 and Op14 are "the PW join op at the bottom with
one external output" — same local hash.

### Phase 1: Symmetry-Preserving Merges

```
repeat until no merge found:
    for each adjacent super-node pair (A, B):
        skip if A.symmetry ≠ B.symmetry
        skip if A.symmetry ≤ 1
        compute CC(A ∪ B), symmetry of union
        skip if symmetry < A.symmetry  (would lose symmetry)
        score = total ops in union (prefer largest)
    apply best merge
```

This grows regions without losing any symmetry factor. In benchmark 5,
Phase 1 alone grows from individual orbits all the way to sym=3×5.

### Phase 2: Symmetry-Reducing Merges

```
repeat until no merge found:
    for each adjacent super-node pair (A, B):
        compute CC(A ∪ B), symmetry of union
        skip if symmetry ≤ 1
        score = (symmetry desc, component_size desc)
    apply best merge
```

Accepts symmetry drops. In the 4-head attention example, Phase 2
merges the sym=4×2 super-node with the sym=2 reduce orbit, producing
sym=2×5.

### Tracking

Throughout both phases, every super-node state with `symmetry > 1`
and `component_size > 1` is recorded in a flat list. This captures
all intermediate and final states for the dominance filter.

## Stage 5: Dominance Filter

Pattern P is dominated (removed) if there exists pattern Q where:
- Q.symmetry ≥ P.symmetry
- Q.component_size ≥ P.component_size
- Every op in P also appears in Q

This removes:
- **Intermediate growth stages**: sym=3×3 dominated by sym=3×5
  (same sym, larger components, same ops)
- **Singleton patterns**: filtered before tracking (component_size ≤ 1)

It preserves:
- **Independent patterns at the same symmetry**: attention sym=8×6 and
  MLP sym=8×5 cover disjoint ops, neither dominates the other
- **Nested patterns at different symmetry**: sym=4×2 and sym=2×5 in
  the 4-head example — higher sym offers more search reduction even
  though the components are smaller

After filtering, deduplicate patterns with identical symmetry and ops.
Sort by (symmetry desc, component_size desc).

## Example Traces

### Benchmark 5 (3 attention heads + aggregation)

```
Orbits:  {0,5,10}(3)  {1,6,11}(3)  {2,7,12}(3)  {3,8,13}(3)  {4,9,14}(3)  singletons...
Phase 1: merge {0,5,10}+{1,6,11} → sym=3×2
         merge +{2,7,12}         → sym=3×3
         merge +{3,8,13}         → sym=3×4
         merge +{4,9,14}         → sym=3×5
Output:  [sym=3 × {0,1,2,3,4} {5,6,7,8,9} {10,11,12,13,14}]
```

### 4-Head Attention (pairwise reduce)

```
Orbits:  {0,1,2,3}(4)  {4,5,6,7}(4)  {8,9}(2)  {10}(1)
Phase 1: merge {0..3}+{4..7} → sym=4×2 (each head chains with its activation)
Phase 2: merge sym4+{8,9}    → sym=2×5 (heads 0+1 and 2+3 pair up)
Output:  [sym=4×2ops, sym=2×5ops]  (both Pareto-optimal)
```

### Benchmark 17 (8 heads + 8 MLPs + aggregation chain)

```
Phase 1: attention orbits merge → sym=8×6
         MLP orbits merge       → sym=8×5
Phase 2: no further merges preserve sym > 1
Output:  [sym=8×6ops (attention), sym=8×5ops (MLP)]
         Coverage: 88/103 ops = 85%
```

## Complexity

- Merkle hashing: O(iterations × (V + E)) = O(V + E)
- Orbit formation: O(V)
- Orbit DAG + levels: O(orbits + orbit_edges)
- Merge loop: O(merge_steps × adjacent_pairs × CC_cost)
  - merge_steps ≤ num_orbits
  - adjacent_pairs ≤ orbit_edges
  - CC_cost = O(V + E) per trial merge
  - Total: O(orbits² × (V + E)) worst case
  - In practice: orbits ~ 10-30, V ~ 100-500, very fast (< 5ms)
- Dominance filter: O(patterns² × V) — patterns typically < 10