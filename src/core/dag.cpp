#include "core/dag.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <queue>

// ============================================================================
// Build
// ============================================================================

DAG DAG::build(const Problem& prob) {
    DAG d;
    d.num_ops = prob.num_ops();

    size_t nt = prob.num_tensors();
    d.tensor_producer.assign(nt, -1);
    d.tensor_consumers.resize(nt);

    d.op_preds.resize(d.num_ops);
    d.op_succs.resize(d.num_ops);

    // Map tensors to producers / consumers
    for (size_t i = 0; i < d.num_ops; i++) {
        size_t t = prob.ops[i].output();
        assert(d.tensor_producer[t] < 0 && "tensor produced by multiple ops");
        d.tensor_producer[t] = (int)i;
    }
    for (size_t i = 0; i < d.num_ops; i++) {
        for (auto t : prob.ops[i].inputs) {
            d.tensor_consumers[t].push_back(i);
            int prod = d.tensor_producer[t];
            if (prod >= 0) {
                d.op_preds[i].push_back((size_t)prod);
                d.op_succs[(size_t)prod].push_back(i);
            }
        }
    }

    // Sort and deduplicate op_preds/op_succs
    for (size_t i = 0; i < d.num_ops; i++) {
        auto& p = d.op_preds[i];
        std::sort(p.begin(), p.end());
        p.erase(std::unique(p.begin(), p.end()), p.end());
        auto& s = d.op_succs[i];
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
    }

    // Graph inputs / outputs.
    //
    // A graph input is a tensor that has no producing op AND is consumed by
    // at least one op. A graph output is a tensor that has a producing op
    // AND is not consumed by any op.
    //
    // Tensors with neither a producer nor a consumer are isolated and are
    // excluded from both lists. They play no role in the computation.
    for (size_t t = 0; t < nt; t++) {
        bool has_producer = (d.tensor_producer[t] >= 0);
        bool has_consumer = !d.tensor_consumers[t].empty();
        if (!has_producer &&  has_consumer)  d.graph_inputs.push_back(t);
        if ( has_producer && !has_consumer)  d.graph_outputs.push_back(t);
    }

    // op_neighbors: DAG edges + co-consumer edges (undirected)
    d.op_neighbors.resize(d.num_ops);
    for (size_t i = 0; i < d.num_ops; i++) {
        FlatSet<size_t> nbrs;
        for (auto j : d.op_preds[i])  nbrs.insert(j);
        for (auto j : d.op_succs[i])  nbrs.insert(j);
        // Co-consumer edges: share a common input tensor
        for (auto t : prob.ops[i].inputs)
            for (auto j : d.tensor_consumers[t])
                if (j != i) nbrs.insert(j);
        d.op_neighbors[i] = {nbrs.begin(), nbrs.end()};
    }

    // Cache topological order and position map, then use it for reachability.
    d.topo_order_ = d.topo_sort();
    d.topo_pos_.resize(d.num_ops);
    for (size_t i = 0; i < d.topo_order_.size(); i++)
        d.topo_pos_[d.topo_order_[i]] = i;

    d.precompute_reachability();

    // Longest directed path (number of edges) via forward DP on topo order.
    // dist[u] = length of longest path ending at u.
    {
        std::vector<size_t> dist(d.num_ops, 0);
        for (auto u : d.topo_order_)
            for (auto v : d.op_succs[u])
                if (dist[u] + 1 > dist[v])
                    dist[v] = dist[u] + 1;
        d.longest_chain_ = 0;
        for (auto v : dist)
            if (v > d.longest_chain_) d.longest_chain_ = v;
    }

    return d;
}

// ============================================================================
// Topological sort (Kahn's algorithm)
// ============================================================================

std::vector<size_t> DAG::topo_sort() const {
    std::vector<int> in_deg(num_ops, 0);
    for (size_t i = 0; i < num_ops; i++)
        in_deg[i] = (int)op_preds[i].size();

    std::queue<size_t> q;
    for (size_t i = 0; i < num_ops; i++)
        if (in_deg[i] == 0) q.push(i);

    std::vector<size_t> order;
    order.reserve(num_ops);
    while (!q.empty()) {
        auto u = q.front(); q.pop();
        order.push_back(u);
        for (auto v : op_succs[u])
            if (--in_deg[v] == 0) q.push(v);
    }
    return order;
}

// ============================================================================
// Precompute reachability via reverse-topo DP
//
// reachable_[u * words_per_row_ + (v/64)]  bit (v%64)  == 1
//   iff there is a directed path u ->* v  (including u->u, i.e., self).
//
// Algorithm: process ops in REVERSE topological order.
// For each u, start with just {u} in its row, then OR in all successors' rows.
// ============================================================================

void DAG::precompute_reachability() {
    if (num_ops == 0) { words_per_row_ = 0; return; }
    words_per_row_ = (num_ops + 63) / 64;

    reachable_.assign(num_ops * words_per_row_, 0ULL);

    // Use the already-computed topo_order_ (avoids a redundant Kahn's pass).
    const auto& order = topo_order_;

    // Process in reverse topological order: successors already done
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        size_t u = order[i];
        // Set self bit
        reachable_[u * words_per_row_ + u / 64] |= (1ULL << (u % 64));
        // OR in all immediate successor rows
        for (auto v : op_succs[u])
            row_or(u, v);
    }
}

// ============================================================================
// merge_creates_cycle
//
// Returns true iff merging op-sets A and B into a single group S = A∪B would
// create a cycle in the condensed DAG.
//
// A cycle is created when the merged super-node S can reach some external op x
// (x ∉ S), AND x can reach back into S — forming a loop: S → ... → x → ... → S.
//
// Equivalently: ∃ a ∈ S, x ∉ S, b ∈ S such that a →* x and x →* b.
//
// Algorithm (using precomputed reachability, all O(words_per_row)):
//   1. Build mask_S = bitmask of all ops in A∪B.
//   2. combined_out = OR(reachable_[a] for a ∈ S) = everything S can reach.
//   3. external_out = combined_out & ~mask_S = external ops reachable from S.
//   4. back_reach   = OR(reachable_[x] for x ∈ external_out) = everything
//                     reachable from those external ops.
//   5. Cycle iff back_reach & mask_S ≠ 0.
//
// Correctness: adjacent groups (direct edge A→B, nothing between them) do NOT
// produce a cycle because the external_out set has no ops that loop back.
//
// Example — chain 0→1→2:
//   merge({0},{1}): external_out = {2}, reachable_[2] & {0,1} = ∅ → false ✓
//   merge({0},{2}): external_out = {1}, reachable_[1] & {0,2} = {2} ≠ ∅ → true ✓
//   merge({1},{2}): external_out = ∅ → false ✓
// ============================================================================

bool DAG::merge_creates_cycle(const FlatSet<size_t>& a,
                               const FlatSet<size_t>& b) const {
    if (words_per_row_ == 0) return false;

    // Thread-local workspace to avoid heap allocations on every call.
    // mask_S and combined_out are reused across calls on the same thread.
    thread_local std::vector<uint64_t> mask_S, combined_out;
    mask_S.assign(words_per_row_, 0ULL);
    combined_out.assign(words_per_row_, 0ULL);

    // Step 1: mask for S = A∪B
    for (auto op : a) if (op < num_ops) mask_S[op / 64] |= (1ULL << (op % 64));
    for (auto op : b) if (op < num_ops) mask_S[op / 64] |= (1ULL << (op % 64));

    // Step 2: combined forward reach of every op in S
    for (auto op : a) if (op < num_ops) {
        const uint64_t* row = reachable_.data() + op * words_per_row_;
        for (size_t w = 0; w < words_per_row_; w++) combined_out[w] |= row[w];
    }
    for (auto op : b) if (op < num_ops) {
        const uint64_t* row = reachable_.data() + op * words_per_row_;
        for (size_t w = 0; w < words_per_row_; w++) combined_out[w] |= row[w];
    }

    // Steps 3-5 fused: for each external op x reachable from S, check if x
    // can reach back into S.  Early exit on the first cycle found — no need
    // to compute the full back_reach bitmask.
    for (size_t w = 0; w < words_per_row_; w++) {
        uint64_t ext_bits = combined_out[w] & ~mask_S[w];
        while (ext_bits) {
            int bit = std::countr_zero(ext_bits);
            size_t x = w * 64 + bit;
            if (x < num_ops) {
                const uint64_t* rx = reachable_.data() + x * words_per_row_;
                for (size_t w2 = 0; w2 < words_per_row_; w2++)
                    if (rx[w2] & mask_S[w2]) return true;
            }
            ext_bits &= ext_bits - 1;
        }
    }
    return false;
}
