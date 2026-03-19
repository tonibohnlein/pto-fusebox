#pragma once

#include "core/types.h"
#include "core/dag.h"
#include <vector>
#include <algorithm>
#include <map>

// ============================================================================
// Merkle hashing for DAG ops
//
// Computes structural fingerprints via two passes:
//   Forward:  hash encodes "what my inputs look like" (bottom-up)
//   Backward: hash encodes "what my outputs feed into" (top-down)
//   Combined: hash encodes full structural context
//
// Ops with the same combined hash are structurally interchangeable.
// This is a directed-graph Merkle hash — more precise than WL color
// refinement because it respects edge direction and tensor dimensions.
//
// Iteration count controls the propagation depth:
//   iter=0: only local info (type, cost, shapes) — "init hash"
//   iter=k: encodes depth-k predecessor/successor trees
//   iter=longest_chain: full propagation, every op gets a unique
//          fingerprint if its global position is unique.
//
// For parallel symmetry detection, the forward hash with full
// propagation is used for orbit formation. For series pattern
// detection, init_hash (iter=0) is used.
// ============================================================================

struct MerkleHashes {
    std::vector<size_t> init;      // iteration-0 hash (type + cost + shapes only)
    std::vector<size_t> fwd;       // forward (input-side) hash per op
    std::vector<size_t> bwd;       // backward (output-side) hash per op
    std::vector<size_t> combined;  // fwd ⊕ bwd

    // Equivalence classes: ops with same combined hash
    std::map<size_t, std::vector<size_t>> equiv_classes;

    // Number of distinct structural roles
    size_t num_classes() const { return equiv_classes.size(); }

    // Symmetry factor: how many ops are in non-trivial equivalence classes
    size_t symmetric_ops() const {
        size_t count = 0;
        for (auto& [h, ops] : equiv_classes)
            if (ops.size() > 1) count += ops.size();
        return count;
    }

    // Compute Merkle hashes.
    // num_iters: number of propagation iterations.
    //   0 = use dag.longest_chain() (full propagation, default).
    //   Any positive value = use exactly that many iterations.
    static MerkleHashes compute(const Problem& prob, const DAG& dag,
                                size_t num_iters = 0) {
        size_t n = prob.num_ops();
        MerkleHashes mh;
        mh.init.resize(n);
        mh.fwd.resize(n);
        mh.bwd.resize(n);
        mh.combined.resize(n);

        int iters = (num_iters > 0)
                    ? (int)num_iters
                    : std::max((int)dag.longest_chain(), 1);

        auto hash_combine = [](size_t seed, size_t v) -> size_t {
            return seed ^ (v * 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
        };

        // Initial hash per op: type + input/output tensor shapes
        for (size_t i = 0; i < n; i++) {
            size_t h = (prob.ops[i].type == OpType::MatMul) ? 0xAA55AA55ULL : 0x55AA55AAULL;
            h = hash_combine(h, (size_t)prob.ops[i].base_cost);

            // Hash input tensor shapes (sorted for permutation invariance)
            std::vector<std::pair<int64_t,int64_t>> in_shapes;
            for (auto t : prob.ops[i].inputs)
                in_shapes.push_back({prob.tensors[t].width, prob.tensors[t].height});
            std::sort(in_shapes.begin(), in_shapes.end());
            for (auto [w, ht] : in_shapes) {
                h = hash_combine(h, (size_t)w);
                h = hash_combine(h, (size_t)ht);
            }

            // Hash output tensor shapes
            std::vector<std::pair<int64_t,int64_t>> out_shapes;
            for (auto t : prob.ops[i].outputs)
                out_shapes.push_back({prob.tensors[t].width, prob.tensors[t].height});
            std::sort(out_shapes.begin(), out_shapes.end());
            h = hash_combine(h, 0xDEADBEEF);  // separator
            for (auto [w, ht] : out_shapes) {
                h = hash_combine(h, (size_t)w);
                h = hash_combine(h, (size_t)ht);
            }
            mh.init[i] = h;
        }

        // ================================================================
        // Forward pass: process in topological order (inputs → outputs)
        // fwd[op] = hash(init[op], sorted fwd hashes of predecessor ops)
        //
        // For graph-input tensors (no producer), hash their dimensions
        // as a "virtual predecessor" so ops reading different graph inputs
        // get different forward hashes.
        // ================================================================
        auto topo = dag.topo_sort();
        mh.fwd = mh.init;

        for (int iter = 0; iter < iters; iter++) {
            std::vector<size_t> new_fwd(n);
            for (auto op : topo) {
                size_t h = mh.init[op];

                // Collect predecessor hashes via input tensors
                std::vector<size_t> pred_hashes;
                for (auto t : prob.ops[op].inputs) {
                    int prod = dag.tensor_producer[t];
                    if (prod >= 0) {
                        // Internal tensor: hash = fwd[producer] + tensor shape
                        size_t th = hash_combine(mh.fwd[prod], (size_t)prob.tensors[t].width);
                        th = hash_combine(th, (size_t)prob.tensors[t].height);
                        pred_hashes.push_back(th);
                    } else {
                        // Graph input: hash tensor identity (shape + number of consumers)
                        size_t th = hash_combine(0xFEEDFACEULL, (size_t)prob.tensors[t].width);
                        th = hash_combine(th, (size_t)prob.tensors[t].height);
                        th = hash_combine(th, dag.tensor_consumers[t].size());
                        pred_hashes.push_back(th);
                    }
                }
                std::sort(pred_hashes.begin(), pred_hashes.end());
                for (auto ph : pred_hashes)
                    h = hash_combine(h, ph);

                new_fwd[op] = h;
            }
            mh.fwd = new_fwd;
        }

        // ================================================================
        // Backward pass: process in reverse topological order (outputs → inputs)
        // bwd[op] = hash(init[op], sorted bwd hashes of successor ops)
        //
        // For graph-output tensors (no consumer), hash their dimensions
        // as a "virtual successor".
        // ================================================================
        mh.bwd = mh.init;

        for (int iter = 0; iter < iters; iter++) {
            std::vector<size_t> new_bwd(n);
            for (int idx = (int)topo.size() - 1; idx >= 0; idx--) {
                size_t op = topo[idx];
                size_t h = mh.init[op];

                // Collect successor hashes via output tensors
                std::vector<size_t> succ_hashes;
                for (auto t : prob.ops[op].outputs) {
                    auto& consumers = dag.tensor_consumers[t];
                    if (consumers.empty()) {
                        // Graph output: terminal tensor
                        size_t th = hash_combine(0xCAFEBABEULL, (size_t)prob.tensors[t].width);
                        th = hash_combine(th, (size_t)prob.tensors[t].height);
                        succ_hashes.push_back(th);
                    } else {
                        // Internal: hash each consumer's bwd hash + tensor shape
                        for (auto cons : consumers) {
                            size_t th = hash_combine(mh.bwd[cons], (size_t)prob.tensors[t].width);
                            th = hash_combine(th, (size_t)prob.tensors[t].height);
                            succ_hashes.push_back(th);
                        }
                    }
                }
                std::sort(succ_hashes.begin(), succ_hashes.end());
                for (auto sh : succ_hashes)
                    h = hash_combine(h, sh);

                new_bwd[op] = h;
            }
            mh.bwd = new_bwd;
        }

        // ================================================================
        // Combined: fwd ⊕ bwd
        // ================================================================
        for (size_t i = 0; i < n; i++)
            mh.combined[i] = hash_combine(mh.fwd[i], mh.bwd[i]);

        // Build equivalence classes
        for (size_t i = 0; i < n; i++)
            mh.equiv_classes[mh.combined[i]].push_back(i);

        return mh;
    }
};