#pragma once

#include "core/types.h"
#include <cstdint>
#include <set>
#include <vector>

// ============================================================================
// DAG: op-level dependency graph derived from the problem
// ============================================================================

struct DAG {
    size_t num_ops;

    std::vector<int> tensor_producer;                  // tensor -> producing op (-1 = graph input)
    std::vector<std::vector<size_t>> tensor_consumers; // tensor -> consuming ops
    std::vector<std::set<size_t>> op_preds;            // op -> predecessor ops
    std::vector<std::set<size_t>> op_succs;            // op -> successor ops

    // Expanded adjacency: DAG edges + co-consumer edges (undirected).
    // op_neighbors[i] includes all ops j such that i→j, j→i, or i and j
    // share a common input tensor. Precomputed at build time.
    std::vector<std::vector<size_t>> op_neighbors;

    std::vector<size_t> graph_inputs;                  // tensors with no producer
    std::vector<size_t> graph_outputs;                 // tensors with no consumer

    // Precomputed transitive reachability via reverse-topo DP.
    //
    // Stored as a flat dynamic bit matrix sized at build time:
    //   reachable_[u * words_per_row_ + (v/64)]  bit (v%64)  == 1
    //   iff there is a directed path u ->* v  (including u == v: self-bit is set).
    //
    // Using uint64_t words instead of std::bitset<N> avoids any compile-time
    // size assumption — the matrix is sized to num_ops at runtime.
    //
    // Memory: num_ops * ceil(num_ops/64) * 8 bytes.
    // For 512 ops: 32 KB.  For 4096 ops: 2 MB.  Both are fine.
    std::vector<uint64_t> reachable_;
    size_t words_per_row_ = 0;   // ceil(num_ops / 64)

    // Cached topological order (output of Kahn's algorithm, computed once in
    // build()).  topo_order_[i] is the op processed at step i.
    // topo_pos_[op] is the position of op in topo_order_ (inverse map).
    // Enables O(1) "is op A topologically before op B?" queries.
    std::vector<size_t> topo_order_;
    std::vector<size_t> topo_pos_;

    // Build the DAG from a problem specification.
    // Also calls precompute_reachability().
    static DAG build(const Problem& prob);

    // Topological sort of all ops.
    std::vector<size_t> topo_sort() const;

    // Would merging two op-sets create a cycle in the condensed DAG?
    // O(|a|*|b| / 64) word operations — no BFS.
    bool merge_creates_cycle(const std::set<size_t>& a,
                             const std::set<size_t>& b) const;

    // Single-pair reachability test.  O(1).
    // Returns true iff there is a directed path u ->* v in the op DAG.
    // can_reach(u, u) is always true (self-reachability).
    // Returns false if either index is out of bounds.
    bool can_reach(size_t u, size_t v) const {
        if (u >= num_ops || v >= num_ops) return false;
        return (reachable_[u * words_per_row_ + v / 64] >> (v % 64)) & 1u;
    }

    // Return the cached topological position of op. O(1).
    // topo_position(a) < topo_position(b) iff a comes before b in topo order.
    size_t topo_position(size_t op) const { return topo_pos_[op]; }

    // Return the cached topological order (topo_order_[i] = op at step i). O(1).
    const std::vector<size_t>& topological_order() const { return topo_order_; }

    // Compute the union of everything reachable from all ops in the given
    // bitmask. Returns a bitmask of the same width (words_per_row_ words).
    // Used by the partition layer to answer "what lies between group A and B?"
    // without the set → bitmask conversion overhead.
    std::vector<uint64_t> reachable_from_mask(const std::vector<uint64_t>& mask) const;

private:
    // Fills reachable_ using a reverse-topo DP pass.  Called once by build().
    void precompute_reachability();

    // OR row src into row dst in place.
    void row_or(size_t dst, size_t src) {
        uint64_t*       d = reachable_.data() + dst * words_per_row_;
        const uint64_t* s = reachable_.data() + src * words_per_row_;
        for (size_t w = 0; w < words_per_row_; w++) d[w] |= s[w];
    }

};