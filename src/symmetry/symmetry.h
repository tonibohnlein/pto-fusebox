#pragma once

#include "core/types.h"
#include "core/dag.h"
#include "symmetry/merkle_hash.h"
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <queue>
#include <iostream>
#include <sstream>

// ============================================================================
// SymmetricPattern: a set of isomorphic connected components discovered by
// orbit merging.  The solver can solve partitioning for components[0] and
// replicate the solution to components[1..symmetry-1].
// ============================================================================

struct SymmetricPattern {
    std::vector<std::set<size_t>> components;   // isomorphic components
    size_t symmetry;                             // == components.size()

    size_t component_size() const {
        return components.empty() ? 0 : components[0].size();
    }

    std::string to_string() const {
        std::ostringstream os;
        os << "sym=" << symmetry << " x " << component_size() << "ops:";
        for (size_t ci = 0; ci < components.size(); ci++) {
            os << " {";
            bool first = true;
            for (auto op : components[ci]) {
                if (!first) os << ",";
                os << op;
                first = false;
            }
            os << "}";
        }
        return os.str();
    }
};

// ============================================================================
// SymmetryDetector
//
// Discovers symmetric sub-DAG patterns by:
//   1. Starting from Merkle orbits (structurally equivalent ops)
//   2. Building an orbit-level DAG with top/bottom distance labels
//   3. Phase 1: merging same-symmetry adjacent orbits (preserving symmetry)
//   4. Phase 2: greedy extension accepting symmetry drops (down to >1)
//
// Level constraint: two super-nodes may merge only if their top-distance
// ranges AND bottom-distance ranges differ by at most 1.  This prevents
// merging orbits with intermediate orbits between them, which would create
// cycles in the condensed super-node DAG.
// ============================================================================

class SymmetryDetector {
public:
    static std::vector<SymmetricPattern> discover(
        const Problem& prob, const DAG& dag, const MerkleHashes& merkle,
        bool verbose = false)
    {
        const size_t num_ops = prob.num_ops();
        if (num_ops == 0) return {};

        // ================================================================
        // 1. Build orbits from Merkle equivalence classes
        // ================================================================

        size_t num_orbits = 0;
        std::vector<size_t> op_to_orbit(num_ops, SIZE_MAX);
        std::vector<std::vector<size_t>> orbit_ops;

        for (auto& [hash, ops] : merkle.equiv_classes) {
            size_t oid = num_orbits++;
            orbit_ops.push_back(ops);
            for (auto op : ops)
                op_to_orbit[op] = oid;
        }

        if (verbose) {
            std::cerr << "[symmetry] " << num_orbits << " orbits from "
                      << num_ops << " ops\n";
            for (size_t oid = 0; oid < num_orbits; oid++) {
                std::cerr << "  orbit " << oid << " (size "
                          << orbit_ops[oid].size() << "):";
                for (auto op : orbit_ops[oid]) std::cerr << " " << op;
                std::cerr << "\n";
            }
        }

        // ================================================================
        // 2. Build orbit-level DAG
        // ================================================================

        std::vector<std::set<size_t>> orbit_succs(num_orbits);
        std::vector<std::set<size_t>> orbit_preds(num_orbits);

        for (size_t t = 0; t < prob.num_tensors(); t++) {
            int prod = dag.tensor_producer[t];
            if (prod < 0) continue;
            size_t prod_orbit = op_to_orbit[(size_t)prod];
            for (auto cons : dag.tensor_consumers[t]) {
                size_t cons_orbit = op_to_orbit[cons];
                if (prod_orbit != cons_orbit) {
                    orbit_succs[prod_orbit].insert(cons_orbit);
                    orbit_preds[cons_orbit].insert(prod_orbit);
                }
            }
        }

        // ================================================================
        // 3. Compute orbit levels (longest-path from sources / to sinks)
        // ================================================================

        std::vector<int> top_dist(num_orbits, 0);
        std::vector<int> bot_dist(num_orbits, 0);

        {
            // Kahn's for topological order + forward longest path
            std::vector<int> in_deg(num_orbits, 0);
            for (size_t i = 0; i < num_orbits; i++)
                in_deg[i] = (int)orbit_preds[i].size();

            std::queue<size_t> q;
            for (size_t i = 0; i < num_orbits; i++)
                if (in_deg[i] == 0) q.push(i);

            std::vector<size_t> topo;
            topo.reserve(num_orbits);
            while (!q.empty()) {
                auto u = q.front(); q.pop();
                topo.push_back(u);
                for (auto v : orbit_succs[u]) {
                    top_dist[v] = std::max(top_dist[v], top_dist[u] + 1);
                    if (--in_deg[v] == 0) q.push(v);
                }
            }

            // Backward longest path (from sinks)
            for (int i = (int)topo.size() - 1; i >= 0; i--) {
                size_t u = topo[i];
                for (auto p : orbit_preds[u])
                    bot_dist[p] = std::max(bot_dist[p], bot_dist[u] + 1);
            }
        }

        if (verbose) {
            for (size_t oid = 0; oid < num_orbits; oid++)
                std::cerr << "  orbit " << oid << ": top=" << top_dist[oid]
                          << " bot=" << bot_dist[oid] << "\n";
        }

        // ================================================================
        // 4. Super-node data structure
        // ================================================================

        struct SuperNode {
            size_t id;
            std::set<size_t> orbit_ids;
            std::set<size_t> all_ops;
            std::vector<std::set<size_t>> components;
            size_t symmetry;
            int top_min, top_max;
            int bot_min, bot_max;
            bool alive = true;
        };

        std::vector<SuperNode> snodes(num_orbits);
        std::vector<size_t> op_to_snode(num_ops);

        for (size_t oid = 0; oid < num_orbits; oid++) {
            auto& sn = snodes[oid];
            sn.id = oid;
            sn.orbit_ids = {oid};
            sn.all_ops.insert(orbit_ops[oid].begin(), orbit_ops[oid].end());
            for (auto op : orbit_ops[oid])
                sn.components.push_back({op});
            sn.symmetry = orbit_ops[oid].size();
            sn.top_min = sn.top_max = top_dist[oid];
            sn.bot_min = sn.bot_max = bot_dist[oid];
            for (auto op : orbit_ops[oid])
                op_to_snode[op] = oid;
        }

        // ================================================================
        // 5. Helpers
        // ================================================================

        // Connected components of an op set via tensor FLOW edges only.
        //
        // Two ops are connected iff one produces a tensor consumed by the
        // other (directly, not through co-consumer adjacency).  This is
        // critical: co-consumer edges (ops sharing a common input) would
        // collapse parallel heads that read the same shared tensor into
        // one component, destroying the symmetry we're trying to detect.
        auto compute_cc = [&](const std::set<size_t>& ops)
            -> std::vector<std::set<size_t>>
        {
            // Build undirected adjacency from tensor producer→consumer
            // restricted to ops in the set.
            std::vector<std::vector<size_t>> adj(num_ops);
            for (auto op : ops) {
                for (auto t : prob.ops[op].outputs) {
                    for (auto cons : dag.tensor_consumers[t]) {
                        if (ops.count(cons)) {
                            adj[op].push_back(cons);
                            adj[cons].push_back(op);
                        }
                    }
                }
            }

            std::vector<std::set<size_t>> result;
            std::set<size_t> visited;
            for (auto seed : ops) {
                if (visited.count(seed)) continue;
                std::set<size_t> comp;
                std::vector<size_t> stack = {seed};
                visited.insert(seed);
                while (!stack.empty()) {
                    size_t u = stack.back(); stack.pop_back();
                    comp.insert(u);
                    for (auto v : adj[u]) {
                        if (!visited.count(v)) {
                            visited.insert(v);
                            stack.push_back(v);
                        }
                    }
                }
                result.push_back(std::move(comp));
            }
            return result;
        };

        auto hash_combine = [](size_t seed, size_t v) -> size_t {
            return seed ^ (v * 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
        };

        // ============================================================
        // Local structural hash for a component.
        //
        // Uses a mini Merkle hash that only considers edges WITHIN
        // the component. External inputs/outputs are hashed by their
        // tensor shape (not by which specific op produces/consumes
        // them globally). This makes Op4 in {0,1,2,3,4} hash the
        // same as Op14 in {10,11,12,13,14} because their LOCAL roles
        // are identical — the difference (Op4→Op15 vs Op14→Op16) is
        // external and irrelevant for partition isomorphism.
        // ============================================================
        auto component_hash = [&](const std::set<size_t>& comp) -> size_t {
            if (comp.empty()) return 0;

            // Build local init hash per op (type + cost + shapes)
            std::map<size_t, size_t> init_h;
            for (auto op : comp) {
                size_t h = (prob.ops[op].type == OpType::MatMul)
                           ? 0xAA55AA55ULL : 0x55AA55AAULL;
                h = hash_combine(h, (size_t)prob.ops[op].base_cost);
                std::vector<std::pair<int64_t,int64_t>> in_shapes;
                for (auto t : prob.ops[op].inputs)
                    in_shapes.push_back({prob.tensors[t].width,
                                         prob.tensors[t].height});
                std::sort(in_shapes.begin(), in_shapes.end());
                for (auto [w, ht] : in_shapes) {
                    h = hash_combine(h, (size_t)w);
                    h = hash_combine(h, (size_t)ht);
                }
                h = hash_combine(h, 0xDEADBEEF);
                std::vector<std::pair<int64_t,int64_t>> out_shapes;
                for (auto t : prob.ops[op].outputs)
                    out_shapes.push_back({prob.tensors[t].width,
                                          prob.tensors[t].height});
                std::sort(out_shapes.begin(), out_shapes.end());
                for (auto [w, ht] : out_shapes) {
                    h = hash_combine(h, (size_t)w);
                    h = hash_combine(h, (size_t)ht);
                }
                init_h[op] = h;
            }

            // Build local predecessor/successor maps (only within comp)
            std::map<size_t, std::vector<std::pair<size_t,size_t>>> local_preds;
            std::map<size_t, std::vector<std::pair<size_t,size_t>>> local_succs;
            // Track external input/output counts per op (for boundary role)
            std::map<size_t, size_t> ext_in_count, ext_out_count;
            for (auto op : comp) {
                ext_in_count[op] = 0;
                ext_out_count[op] = 0;
                for (auto t : prob.ops[op].inputs) {
                    int prod = dag.tensor_producer[t];
                    if (prod >= 0 && comp.count((size_t)prod)) {
                        local_preds[op].push_back({(size_t)prod, t});
                    } else {
                        ext_in_count[op]++;
                    }
                }
                for (auto t : prob.ops[op].outputs) {
                    for (auto cons : dag.tensor_consumers[t]) {
                        if (comp.count(cons)) {
                            local_succs[op].push_back({cons, t});
                        }
                    }
                    // Count external consumers
                    bool has_ext = false;
                    for (auto cons : dag.tensor_consumers[t])
                        if (!comp.count(cons)) { has_ext = true; break; }
                    if (has_ext || dag.tensor_consumers[t].empty())
                        ext_out_count[op]++;
                }
            }

            // Topo sort within the component
            std::vector<size_t> local_topo;
            {
                std::map<size_t, int> in_deg;
                for (auto op : comp) in_deg[op] = 0;
                for (auto op : comp)
                    for (auto& [pred, t] : local_preds[op])
                        in_deg[op]++;
                std::queue<size_t> q;
                for (auto op : comp)
                    if (in_deg[op] == 0) q.push(op);
                while (!q.empty()) {
                    auto u = q.front(); q.pop();
                    local_topo.push_back(u);
                    for (auto& [succ, t] : local_succs[u])
                        if (--in_deg[succ] == 0) q.push(succ);
                }
            }

            // Forward local Merkle: 3 iterations
            std::map<size_t, size_t> fwd;
            for (auto op : comp) fwd[op] = init_h[op];
            for (int iter = 0; iter < 3; iter++) {
                std::map<size_t, size_t> new_fwd;
                for (auto op : local_topo) {
                    size_t h = init_h[op];
                    // External input count as structural feature
                    h = hash_combine(h, ext_in_count[op] + 0xEE00);
                    std::vector<size_t> pred_hashes;
                    for (auto& [pred, t] : local_preds[op]) {
                        size_t th = hash_combine(fwd[pred],
                                    (size_t)prob.tensors[t].width);
                        th = hash_combine(th,
                                    (size_t)prob.tensors[t].height);
                        pred_hashes.push_back(th);
                    }
                    // Hash external inputs by shape (not identity)
                    for (auto t : prob.ops[op].inputs) {
                        int prod = dag.tensor_producer[t];
                        if (prod < 0 || !comp.count((size_t)prod)) {
                            size_t th = hash_combine(0xFEEDFACEULL,
                                        (size_t)prob.tensors[t].width);
                            th = hash_combine(th,
                                        (size_t)prob.tensors[t].height);
                            pred_hashes.push_back(th);
                        }
                    }
                    std::sort(pred_hashes.begin(), pred_hashes.end());
                    for (auto ph : pred_hashes) h = hash_combine(h, ph);
                    new_fwd[op] = h;
                }
                fwd = new_fwd;
            }

            // Backward local Merkle: 3 iterations
            std::map<size_t, size_t> bwd;
            for (auto op : comp) bwd[op] = init_h[op];
            for (int iter = 0; iter < 3; iter++) {
                std::map<size_t, size_t> new_bwd;
                for (int i = (int)local_topo.size() - 1; i >= 0; i--) {
                    size_t op = local_topo[i];
                    size_t h = init_h[op];
                    h = hash_combine(h, ext_out_count[op] + 0xFF00);
                    std::vector<size_t> succ_hashes;
                    for (auto& [succ, t] : local_succs[op]) {
                        size_t th = hash_combine(bwd[succ],
                                    (size_t)prob.tensors[t].width);
                        th = hash_combine(th,
                                    (size_t)prob.tensors[t].height);
                        succ_hashes.push_back(th);
                    }
                    // Hash external outputs by shape
                    for (auto t : prob.ops[op].outputs) {
                        bool has_ext = false;
                        for (auto cons : dag.tensor_consumers[t])
                            if (!comp.count(cons)) { has_ext = true; break; }
                        if (has_ext || dag.tensor_consumers[t].empty()) {
                            size_t th = hash_combine(0xCAFEBABEULL,
                                        (size_t)prob.tensors[t].width);
                            th = hash_combine(th,
                                        (size_t)prob.tensors[t].height);
                            succ_hashes.push_back(th);
                        }
                    }
                    std::sort(succ_hashes.begin(), succ_hashes.end());
                    for (auto sh : succ_hashes) h = hash_combine(h, sh);
                    new_bwd[op] = h;
                }
                bwd = new_bwd;
            }

            // Component fingerprint = sorted(fwd ⊕ bwd) for all ops
            std::vector<size_t> combined;
            combined.reserve(comp.size());
            for (auto op : comp)
                combined.push_back(hash_combine(fwd[op], bwd[op]));
            std::sort(combined.begin(), combined.end());
            size_t h = combined.size();
            for (auto v : combined)
                h = hash_combine(h, v);
            return h;
        };

        // Compute symmetry = size of the largest isomorphism class
        auto compute_symmetry = [&](const std::vector<std::set<size_t>>& comps)
            -> size_t
        {
            if (comps.empty()) return 0;
            std::map<size_t, size_t> freq;
            for (auto& comp : comps)
                freq[component_hash(comp)]++;
            size_t max_freq = 0;
            for (auto& [h, f] : freq)
                max_freq = std::max(max_freq, f);
            return max_freq;
        };

        // Level constraint: ranges must be adjacent or overlapping in BOTH
        // top-distance and bottom-distance dimensions.
        //
        // This prevents merging orbits separated by intermediate orbits
        // that are not part of either super-node, which would create
        // a cycle:  merged_SN → intermediate → merged_SN.
        auto can_merge = [](const SuperNode& a, const SuperNode& b) -> bool {
            bool top_ok = (a.top_max >= b.top_min - 1) &&
                          (b.top_max >= a.top_min - 1);
            bool bot_ok = (a.bot_max >= b.bot_min - 1) &&
                          (b.bot_max >= a.bot_min - 1);
            return top_ok && bot_ok;
        };

        // Find all adjacent super-node pairs (tensor flows between them,
        // both alive, level constraint satisfied)
        auto get_adjacent_pairs = [&]()
            -> std::vector<std::pair<size_t, size_t>>
        {
            std::set<std::pair<size_t, size_t>> pairs;
            for (size_t t = 0; t < prob.num_tensors(); t++) {
                int prod = dag.tensor_producer[t];
                if (prod < 0) continue;
                size_t sn_prod = op_to_snode[(size_t)prod];
                for (auto cons : dag.tensor_consumers[t]) {
                    size_t sn_cons = op_to_snode[cons];
                    if (sn_prod == sn_cons) continue;
                    if (!snodes[sn_prod].alive || !snodes[sn_cons].alive) continue;
                    if (!can_merge(snodes[sn_prod], snodes[sn_cons])) continue;
                    auto p = std::minmax(sn_prod, sn_cons);
                    pairs.insert(p);
                }
            }
            return {pairs.begin(), pairs.end()};
        };

        // Merge two super-nodes, creating a new one
        auto do_merge = [&](size_t id_a, size_t id_b,
                            std::vector<std::set<size_t>> comps,
                            size_t sym) -> size_t
        {
            auto& a = snodes[id_a];
            auto& b = snodes[id_b];

            size_t new_id = snodes.size();
            SuperNode sn;
            sn.id        = new_id;
            sn.orbit_ids = a.orbit_ids;
            sn.orbit_ids.insert(b.orbit_ids.begin(), b.orbit_ids.end());
            sn.all_ops   = a.all_ops;
            sn.all_ops.insert(b.all_ops.begin(), b.all_ops.end());
            sn.components = std::move(comps);
            sn.symmetry   = sym;
            sn.top_min    = std::min(a.top_min, b.top_min);
            sn.top_max    = std::max(a.top_max, b.top_max);
            sn.bot_min    = std::min(a.bot_min, b.bot_min);
            sn.bot_max    = std::max(a.bot_max, b.bot_max);

            a.alive = false;
            b.alive = false;

            snodes.push_back(std::move(sn));

            for (auto op : snodes[new_id].all_ops)
                op_to_snode[op] = new_id;

            return new_id;
        };

        // Extract the largest isomorphism class from a super-node's components
        auto extract_pattern = [&](const SuperNode& sn) -> SymmetricPattern {
            std::map<size_t, std::vector<size_t>> hash_to_idx;
            for (size_t ci = 0; ci < sn.components.size(); ci++)
                hash_to_idx[component_hash(sn.components[ci])].push_back(ci);

            // Find largest class
            size_t best_hash = 0;
            size_t best_count = 0;
            for (auto& [h, idxs] : hash_to_idx) {
                if (idxs.size() > best_count) {
                    best_count = idxs.size();
                    best_hash = h;
                }
            }

            SymmetricPattern pat;
            pat.symmetry = best_count;
            for (auto ci : hash_to_idx[best_hash])
                pat.components.push_back(sn.components[ci]);
            return pat;
        };

        // ================================================================
        // 6. Phase 1: merge same-symmetry adjacent orbits, preserving sym
        // ================================================================
        //
        // Only consider pairs where both have the same symmetry and the
        // merged result keeps it.  Pick the merge that produces the
        // largest region (most ops).  Record each merge as a pattern.

        std::vector<SymmetricPattern> patterns;

        {
            bool merged = true;
            while (merged) {
                merged = false;
                auto pairs = get_adjacent_pairs();

                size_t best_a = SIZE_MAX, best_b = SIZE_MAX;
                std::vector<std::set<size_t>> best_comps;
                size_t best_sym = 0;
                size_t best_ops = 0;

                for (auto [ia, ib] : pairs) {
                    auto& a = snodes[ia];
                    auto& b = snodes[ib];
                    if (a.symmetry != b.symmetry) continue;

                    size_t target_sym = a.symmetry;
                    if (target_sym <= 1) continue;

                    std::set<size_t> union_ops = a.all_ops;
                    union_ops.insert(b.all_ops.begin(), b.all_ops.end());
                    auto comps = compute_cc(union_ops);
                    size_t sym = compute_symmetry(comps);

                    if (sym < target_sym) continue;

                    size_t score = union_ops.size();
                    if (score > best_ops) {
                        best_a = ia;
                        best_b = ib;
                        best_comps = std::move(comps);
                        best_sym = sym;
                        best_ops = score;
                    }
                }

                if (best_a != SIZE_MAX) {
                    size_t new_id = do_merge(best_a, best_b,
                                             std::move(best_comps), best_sym);
                    auto pat = extract_pattern(snodes[new_id]);
                    if (verbose)
                        std::cerr << "[symmetry] phase1 merge: "
                                  << pat.to_string() << "\n";
                    if (pat.symmetry > 1)
                        patterns.push_back(std::move(pat));
                    merged = true;
                }
            }
        }

        // ================================================================
        // 7. Phase 2: greedy extension, accepting symmetry drops > 1
        // ================================================================
        //
        // For every adjacent pair, compute merged symmetry.  Accept the
        // merge with the best (symmetry desc, component_size desc).
        // Record each intermediate pattern.  Stop when every remaining
        // merge would collapse symmetry to 1.

        {
            bool merged = true;
            while (merged) {
                merged = false;
                auto pairs = get_adjacent_pairs();

                size_t best_a = SIZE_MAX, best_b = SIZE_MAX;
                std::vector<std::set<size_t>> best_comps;
                size_t best_sym = 0;
                size_t best_comp_size = 0;

                for (auto [ia, ib] : pairs) {
                    std::set<size_t> union_ops = snodes[ia].all_ops;
                    union_ops.insert(snodes[ib].all_ops.begin(),
                                    snodes[ib].all_ops.end());
                    auto comps = compute_cc(union_ops);
                    size_t sym = compute_symmetry(comps);

                    if (sym <= 1) continue;

                    // Find the representative component size for this sym
                    size_t comp_size = 0;
                    {
                        std::map<size_t, std::vector<size_t>> hg;
                        for (size_t ci = 0; ci < comps.size(); ci++)
                            hg[component_hash(comps[ci])].push_back(ci);
                        for (auto& [h, idxs] : hg) {
                            if (idxs.size() == sym) {
                                comp_size = comps[idxs[0]].size();
                                break;
                            }
                        }
                    }

                    // Lexicographic: symmetry desc, then component size desc
                    if (sym > best_sym ||
                        (sym == best_sym && comp_size > best_comp_size)) {
                        best_a = ia;
                        best_b = ib;
                        best_comps = std::move(comps);
                        best_sym = sym;
                        best_comp_size = comp_size;
                    }
                }

                if (best_a != SIZE_MAX) {
                    size_t new_id = do_merge(best_a, best_b,
                                             std::move(best_comps), best_sym);
                    auto pat = extract_pattern(snodes[new_id]);
                    if (verbose)
                        std::cerr << "[symmetry] phase2 merge: "
                                  << pat.to_string() << "\n";
                    if (pat.symmetry > 1)
                        patterns.push_back(std::move(pat));
                    merged = true;
                }
            }
        }

        // ================================================================
        // 8. Sort and deduplicate
        // ================================================================

        std::sort(patterns.begin(), patterns.end(),
                  [](const SymmetricPattern& a, const SymmetricPattern& b) {
            if (a.symmetry != b.symmetry) return a.symmetry > b.symmetry;
            return a.component_size() > b.component_size();
        });

        // Remove subsumed patterns: if P has the same symmetry as Q and
        // every component of P is a subset of a component of Q, drop P.
        // After sorting (largest first), check each pattern against all
        // earlier (larger) patterns at the same symmetry level.
        std::vector<bool> subsumed(patterns.size(), false);
        for (size_t i = 1; i < patterns.size(); i++) {
            for (size_t j = 0; j < i; j++) {
                if (subsumed[j]) continue;
                if (patterns[j].symmetry != patterns[i].symmetry) continue;
                if (patterns[j].component_size() <= patterns[i].component_size())
                    continue;

                // Check if every component of P[i] is a subset of
                // some component of P[j].
                bool all_subsets = true;
                for (auto& ci : patterns[i].components) {
                    bool found = false;
                    for (auto& cj : patterns[j].components) {
                        if (std::includes(cj.begin(), cj.end(),
                                          ci.begin(), ci.end())) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) { all_subsets = false; break; }
                }
                if (all_subsets) { subsumed[i] = true; break; }
            }
        }
        {
            std::vector<SymmetricPattern> filtered;
            for (size_t i = 0; i < patterns.size(); i++)
                if (!subsumed[i])
                    filtered.push_back(std::move(patterns[i]));
            patterns = std::move(filtered);
        }

        if (verbose) {
            std::cerr << "[symmetry] discovered " << patterns.size()
                      << " patterns:\n";
            for (auto& p : patterns)
                std::cerr << "  " << p.to_string() << "\n";
        }

        return patterns;
    }
};