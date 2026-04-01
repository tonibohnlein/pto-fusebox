#include "init/symm_init.h"
#include "core/dag.h"
#include "symmetry/merkle_hash.h"
#include "symmetry/symmetry.h"
#include "symmetry/series.h"
#include "search/local_search.h"
#include <iostream>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <queue>

// ============================================================================
// Restricted greedy merge descent.
//
// Only merges groups whose ops are entirely within `allowed_ops`.
// Simple O(N^2) per pass — fine because the allowed set is small
// (one pattern component, typically 3–20 ops).
//
// Only does MERGE moves: the main init-time optimization is fusing
// singletons into multi-op subgraphs.  STEAL/EJECT/SPLIT/RECOMPUTE
// are refinement moves applied later by the full greedy_descent.
// ============================================================================

static Partition greedy_merge_restricted(
    Partition part, const FlatSet<size_t>& allowed_ops)
{
    bool improved = true;
    int passes = 0;
    while (improved && passes < 200) {
        improved = false;
        passes++;

        // Collect alive groups that are fully within the allowed set
        std::vector<size_t> eligible;
        for (size_t gi = 0; gi < part.groups.size(); gi++) {
            if (!part.groups[gi].alive) continue;
            bool all_in = true;
            for (auto op : part.groups[gi].ops)
                if (!allowed_ops.count(op)) { all_in = false; break; }
            if (all_in) eligible.push_back(gi);
        }

        // Try all pairs of eligible adjacent groups
        double best_saving = 0.001;  // minimum threshold
        size_t best_ga = SIZE_MAX, best_gb = SIZE_MAX;
        FlatSet<size_t> best_merged;
        double best_cost = 0;

        for (size_t i = 0; i < eligible.size(); i++) {
            size_t ga = eligible[i];
            if (!part.groups[ga].alive) continue;

            // Find adjacent eligible groups via boundary ops
            FlatSet<size_t> adj;
            for (auto op : part.groups[ga].ops)
                for (auto nbr : part.dag->op_neighbors[op])
                    if (allowed_ops.count(nbr))
                        for (auto gj : part.groups_of(nbr))
                            if (gj != ga && part.groups[gj].alive)
                                adj.insert(gj);

            for (auto gb : adj) {
                // Check gb is eligible (all ops in allowed set)
                bool gb_ok = true;
                for (auto op : part.groups[gb].ops)
                    if (!allowed_ops.count(op)) { gb_ok = false; break; }
                if (!gb_ok) continue;

                if (!part.acyclic_merge_local(ga, gb)) continue;

                // Evaluate merge
                FlatSet<size_t> merged = part.groups[ga].ops;
                merged.insert(part.groups[gb].ops.begin(),
                              part.groups[gb].ops.end());
                double mc = part.eval_set(merged);
                if (mc >= 1e17) continue;

                double saving = (part.groups[ga].cost + part.groups[gb].cost) - mc;
                if (saving > best_saving) {
                    best_saving = saving;
                    best_ga = ga;
                    best_gb = gb;
                    best_merged = std::move(merged);
                    best_cost = mc;
                }
            }
        }

        if (best_ga != SIZE_MAX) {
            part.groups[best_ga].ops = std::move(best_merged);
            part.groups[best_ga].cost = best_cost;
            part.groups[best_ga].gen++;
            part.groups[best_gb].alive = false;
            part.groups[best_gb].gen++;
            part.rebuild_index();
            improved = true;
        }
    }

    return part;
}

// ============================================================================
// Compute bijection between two isomorphic op sets.
//
// Assigns each op a local fingerprint (init hash + local topo depth),
// then matches by sorted fingerprint.  Ties broken by local topo order.
// ============================================================================

static std::unordered_map<size_t, size_t> compute_bijection(
    const FlatSet<size_t>& src, const FlatSet<size_t>& dst,
    const Problem& prob, const DAG& dag, const MerkleHashes& merkle)
{
    // Compute local topo depth within a component (BFS from sources)
    auto local_depths = [&](const FlatSet<size_t>& comp)
        -> std::unordered_map<size_t, size_t>
    {
        std::unordered_map<size_t, size_t> depth;
        depth.reserve(comp.size());
        std::queue<size_t> q;
        for (auto op : comp) {
            bool is_source = true;
            for (auto p : dag.op_preds[op])
                if (comp.count(p)) { is_source = false; break; }
            if (is_source) {
                depth[op] = 0;
                q.push(op);
            }
        }

        while (!q.empty()) {
            size_t u = q.front(); q.pop();
            for (auto v : dag.op_succs[u]) {
                if (!comp.count(v)) continue;
                if (!depth.count(v) || depth[u] + 1 > depth[v]) {
                    depth[v] = depth[u] + 1;
                    q.push(v);
                }
            }
        }

        for (auto op : comp)
            if (!depth.count(op)) depth[op] = 0;

        return depth;
    };

    // Build (init_hash, depth, combined_hash) tuples for stable sorting.
    // Uses merkle.combined (local structural context) instead of global
    // topo position to avoid cross-wiring when global topo order interleaves
    // parallel components.
    auto make_ranked = [&](const FlatSet<size_t>& comp)
        -> std::vector<std::pair<std::tuple<size_t, size_t, size_t>, size_t>>
    {
        auto depths = local_depths(comp);
        std::vector<std::pair<std::tuple<size_t, size_t, size_t>, size_t>> ranked;
        for (auto op : comp) {
            ranked.push_back({
                {merkle.init[op], depths[op], merkle.combined[op]},
                op
            });
        }
        std::sort(ranked.begin(), ranked.end());
        return ranked;
    };

    auto src_ranked = make_ranked(src);
    auto dst_ranked = make_ranked(dst);

    std::unordered_map<size_t, size_t> bij;
    bij.reserve(src_ranked.size());
    size_t n = std::min(src_ranked.size(), dst_ranked.size());
    for (size_t i = 0; i < n; i++)
        bij[src_ranked[i].second] = dst_ranked[i].second;

    return bij;
}

// ============================================================================
// init_from_patterns: main entry point
// ============================================================================

std::vector<Partition> init_from_patterns(
    const Problem& prob, const DAG& dag, CostCache* cache)
{
    auto merkle = MerkleHashes::compute(prob, dag);
    auto parallel = SymmetryDetector::discover(prob, dag, merkle, false);
    auto series   = SeriesDetector::discover(prob, dag, merkle, false);

    if (parallel.empty() && series.empty())
        return {};

    std::cerr << "  [symm_init] patterns: " << parallel.size()
              << " parallel, " << series.size() << " series\n";

    std::vector<Partition> results;

    // ================================================================
    // Process parallel patterns
    // ================================================================

    for (size_t pi = 0; pi < parallel.size(); pi++) {
        auto& pat = parallel[pi];
        if (pat.symmetry < 2 || pat.component_size() < 2) continue;

        std::cerr << "  [symm_init] parallel " << pi
                  << ": sym=" << pat.symmetry
                  << " x " << pat.component_size() << " ops\n";

        // Step 1: Solve the representative component
        const auto& rep = pat.components[0];

        Partition rep_part = Partition::trivial(prob, dag);
        rep_part.cache = cache;
        rep_part = greedy_merge_restricted(rep_part, rep);

        // Extract the groups found for the representative
        struct GroupInfo {
            FlatSet<size_t> ops;
            double cost;
        };
        std::vector<GroupInfo> rep_groups;
        for (size_t gi = 0; gi < rep_part.groups.size(); gi++) {
            if (!rep_part.groups[gi].alive) continue;
            bool in_rep = false;
            for (auto op : rep_part.groups[gi].ops)
                if (rep.count(op)) { in_rep = true; break; }
            if (!in_rep) continue;
            // Only include groups fully within rep
            bool all_in = true;
            for (auto op : rep_part.groups[gi].ops)
                if (!rep.count(op)) { all_in = false; break; }
            if (!all_in) continue;
            if (rep_part.groups[gi].ops.size() <= 1) continue;  // skip singletons
            rep_groups.push_back({rep_part.groups[gi].ops,
                                  rep_part.groups[gi].cost});
        }

        if (rep_groups.empty()) continue;

        // Step 2: Build full partition with replicated groups
        Partition full = Partition::trivial(prob, dag);
        full.cache = cache;

        // Kill singleton groups for ops covered by pattern groups
        FlatSet<size_t> covered_ops;

        // Add representative groups (already evaluated)
        for (auto& rg : rep_groups) {
            full.add_group(rg.ops, rg.cost);
            for (auto op : rg.ops) covered_ops.insert(op);
        }

        // Replicate to each copy
        for (size_t ci = 1; ci < pat.components.size(); ci++) {
            auto bij = compute_bijection(rep, pat.components[ci],
                                         prob, dag, merkle);

            for (auto& rg : rep_groups) {
                FlatSet<size_t> mapped;
                for (auto op : rg.ops) {
                    auto it = bij.find(op);
                    if (it != bij.end())
                        mapped.insert(it->second);
                }

                if (mapped.size() != rg.ops.size()) continue;

                double cost = full.eval_set(mapped);
                if (cost < 1e17) {
                    full.add_group(mapped, cost);
                    for (auto op : mapped) covered_ops.insert(op);
                }
            }
        }

        // Kill trivial singletons for covered ops
        for (size_t gi = 0; gi < full.groups.size(); gi++) {
            if (!full.groups[gi].alive) continue;
            if (full.groups[gi].ops.size() != 1) continue;
            size_t op = *full.groups[gi].ops.begin();
            if (covered_ops.count(op))
                full.groups[gi].alive = false;
        }
        full.rebuild_index();

        std::cerr << "    after replication: " << full.num_alive()
                  << " groups, cost=" << full.total_cost() << "\n";

        // Step 3: Global greedy refinement
        full = greedy_descent(std::move(full));

        std::cerr << "    after greedy: " << full.num_alive()
                  << " groups, cost=" << full.total_cost() << "\n";

        results.push_back(std::move(full));
    }

    // ================================================================
    // Process series patterns
    // ================================================================

    for (size_t si = 0; si < series.size(); si++) {
        auto& pat = series[si];
        if (pat.num_blocks < 2 || pat.block_size < 2) continue;

        std::cerr << "  [symm_init] series " << si
                  << ": " << pat.num_blocks << " blocks x "
                  << pat.block_size << " ops\n";

        // Step 1: Solve the first block
        FlatSet<size_t> rep_set(pat.representative.begin(),
                                  pat.representative.end());

        Partition rep_part = Partition::trivial(prob, dag);
        rep_part.cache = cache;
        rep_part = greedy_merge_restricted(rep_part, rep_set);

        // Extract groups
        struct GroupInfo {
            FlatSet<size_t> ops;
            double cost;
        };
        std::vector<GroupInfo> rep_groups;
        for (size_t gi = 0; gi < rep_part.groups.size(); gi++) {
            if (!rep_part.groups[gi].alive) continue;
            bool all_in = true;
            for (auto op : rep_part.groups[gi].ops)
                if (!rep_set.count(op)) { all_in = false; break; }
            if (!all_in || rep_part.groups[gi].ops.size() <= 1) continue;
            rep_groups.push_back({rep_part.groups[gi].ops,
                                  rep_part.groups[gi].cost});
        }

        if (rep_groups.empty()) continue;

        // Step 2: Build bijections and replicate
        // Series bijection: block[0][j] ↔ block[i][j]
        Partition full = Partition::trivial(prob, dag);
        full.cache = cache;
        FlatSet<size_t> covered_ops;

        for (size_t bi = 0; bi < pat.num_blocks; bi++) {
            std::map<size_t, size_t> bij;
            for (size_t j = 0; j < pat.block_size; j++)
                bij[pat.blocks[0][j]] = pat.blocks[bi][j];

            for (auto& rg : rep_groups) {
                FlatSet<size_t> mapped;
                for (auto op : rg.ops) {
                    auto it = bij.find(op);
                    if (it != bij.end())
                        mapped.insert(it->second);
                }
                if (mapped.size() != rg.ops.size()) continue;

                double cost = (bi == 0) ? rg.cost : full.eval_set(mapped);
                if (cost < 1e17) {
                    full.add_group(mapped, cost);
                    for (auto op : mapped) covered_ops.insert(op);
                }
            }
        }

        // Kill trivial singletons for covered ops
        for (size_t gi = 0; gi < full.groups.size(); gi++) {
            if (!full.groups[gi].alive) continue;
            if (full.groups[gi].ops.size() != 1) continue;
            size_t op = *full.groups[gi].ops.begin();
            if (covered_ops.count(op))
                full.groups[gi].alive = false;
        }
        full.rebuild_index();

        std::cerr << "    after replication: " << full.num_alive()
                  << " groups, cost=" << full.total_cost() << "\n";

        // Step 3: Global greedy refinement
        full = greedy_descent(std::move(full));

        std::cerr << "    after greedy: " << full.num_alive()
                  << " groups, cost=" << full.total_cost() << "\n";

        results.push_back(std::move(full));
    }

    return results;
}