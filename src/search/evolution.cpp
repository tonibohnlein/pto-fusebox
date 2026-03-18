#include "search/evolution.h"
#include "search/verbose.h"
#include "search/local_search.h"  // partition_has_gap (cycle check)
#include "search/merkle_hash.h"
#include <algorithm>
#include <iostream>

// Under the new ephemeral rule, gap checks are unnecessary.
// Only acyclicity matters. partition_has_gap now checks cycles only.

#include <functional>

// ============================================================================
// Merkle-aware ARI canonicalisation
//
// Within each Merkle equivalence class (structurally symmetric ops), sort ops
// by their assignment in map_a, then match them rank-for-rank to ops sorted
// by their assignment in map_b.  This makes symmetric variants have distance 0
// instead of a spurious non-zero ARI distance.
//
// Time: O(sum_over_classes(k log k)) — negligible for typical ML graphs.
// ============================================================================
static void merkle_canonicalise(
        const MerkleHashes& mh,
        const std::vector<int>& map_a,
        std::vector<int>& map_b)   // modified in-place
{
    for (auto& [hash, ops] : mh.equiv_classes) {
        if (ops.size() <= 1) continue;

        // Sort ops by their assignment in A (break ties by op index for stability)
        std::vector<size_t> by_a(ops.begin(), ops.end());
        std::sort(by_a.begin(), by_a.end(), [&](size_t x, size_t y){
            int gx = (x < map_a.size()) ? map_a[x] : -1;
            int gy = (y < map_a.size()) ? map_a[y] : -1;
            return gx != gy ? gx < gy : x < y;
        });

        // Sort ops by their assignment in B
        std::vector<size_t> by_b(ops.begin(), ops.end());
        std::sort(by_b.begin(), by_b.end(), [&](size_t x, size_t y){
            int gx = (x < map_b.size()) ? map_b[x] : -1;
            int gy = (y < map_b.size()) ? map_b[y] : -1;
            return gx != gy ? gx < gy : x < y;
        });

        // Match rank-for-rank: A's i-th op gets B's i-th op's assignment
        std::vector<int> new_b(ops.size());
        for (size_t i = 0; i < ops.size(); i++)
            new_b[i] = (by_b[i] < map_b.size()) ? map_b[by_b[i]] : -1;
        for (size_t i = 0; i < ops.size(); i++)
            if (by_a[i] < map_b.size()) map_b[by_a[i]] = new_b[i];
    }
}

#include <numeric>
#include <map>
#include <set>

// ============================================================================
// Helper: find alive groups and their adjacencies
// ============================================================================

static std::vector<size_t> alive_groups(const Partition& part) {
    std::vector<size_t> result;
    for (size_t i = 0; i < part.groups.size(); i++)
        if (part.groups[i].alive) result.push_back(i);
    return result;
}

// ============================================================================
// Mutate: random merge
// ============================================================================

Partition mutate_merge(Partition part, std::mt19937& rng) {
    auto ag = alive_groups(part);
    if (ag.size() < 2) return part;
    
    // Collect all pairs of adjacent alive groups
    std::vector<std::pair<size_t,size_t>> adj_pairs;
    for (auto gi : ag) {
        auto neighbors = part.adjacent_groups(gi);
        for (auto gj : neighbors) {
            if (part.groups[gj].alive && gi < gj)
                adj_pairs.push_back({gi, gj});
        }
    }
    if (adj_pairs.empty()) return part;
    
    // Pick random pair
    auto [ga, gb] = adj_pairs[rng() % adj_pairs.size()];

    if (part.dag->merge_creates_cycle(part.groups[ga].ops, part.groups[gb].ops)) return part;
    
    // Merge: add all ops from gb into ga
    std::set<size_t> merged = part.groups[ga].ops;
    for (auto op : part.groups[gb].ops)
        merged.insert(op);
    
    double cost = part.eval_set(merged);
    if (cost >= 1e17) return part;
    
    Partition saved = part;
    part.groups[ga].ops = merged;
    part.groups[ga].cost = cost;
    part.groups[ga].gen++;
    part.groups[gb].alive = false;
    part.rebuild_index();
    if (partition_has_gap(part)) return saved;
    return part;
}

// ============================================================================
// Mutate: random split
// ============================================================================

Partition mutate_split(Partition part, std::mt19937& rng) {
    auto ag = alive_groups(part);
    
    // Collect groups that can be split (have bridge edges)
    std::vector<size_t> splittable;
    for (auto gi : ag)
        if (part.groups[gi].ops.size() >= 3)
            splittable.push_back(gi);
    if (splittable.empty()) return part;
    
    // Pick random splittable group
    size_t gi = splittable[rng() % splittable.size()];
    auto bridges = part.bridge_edges(gi);
    if (bridges.empty()) return part;
    
    // Pick random bridge edge and split
    auto [op_a, op_b] = bridges[rng() % bridges.size()];
    auto sr = part.eval_split(op_a, op_b, gi);
    if (!sr.feasible) return part;
    
    // Apply split
    Partition saved = part;
    part.groups[gi].ops = sr.side_a;
    part.groups[gi].cost = sr.cost_a;
    part.groups[gi].gen++;
    part.add_group(sr.side_b, sr.cost_b);
    part.rebuild_index();
    if (partition_has_gap(part)) return saved;
    return part;
}

// ============================================================================
// Mutate: random reassign (NOT gain-guided)
// ============================================================================

Partition mutate_reassign(Partition part, std::mt19937& rng) {
    auto ag = alive_groups(part);
    if (ag.empty()) return part;
    
    // Collect all border ops with their group
    std::vector<std::pair<size_t, size_t>> border_ops;  // (op, group)
    for (auto gi : ag)
        for (auto op : part.border_ops(gi))
            border_ops.push_back({op, gi});
    if (border_ops.empty()) return part;
    
    // Pick random border op
    auto [op, src_gi] = border_ops[rng() % border_ops.size()];
    
    // Find neighbor groups (via all neighbor edges including co-consumers)
    std::vector<size_t> targets;
    for (auto nbr : part.dag->op_neighbors[op])
        for (auto gi : part.groups_of(nbr))
            if (gi != src_gi) targets.push_back(gi);
    
    // Deduplicate
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    if (targets.empty()) return part;
    
    // Pick random target
    size_t dst_gi = targets[rng() % targets.size()];

    if (part.dag->merge_creates_cycle({op}, part.groups[dst_gi].ops)) return part;
    
    // Check: removing op from src must leave it connected and non-empty
    if (part.groups[src_gi].ops.size() <= 1) return part;
    
    std::set<size_t> new_src = part.groups[src_gi].ops;
    new_src.erase(op);
    auto comps = part.connected_components(new_src);
    if (comps.size() != 1) return part;  // would disconnect
    
    double src_cost = part.eval_set(new_src);
    if (src_cost >= 1e17) return part;
    
    std::set<size_t> new_dst = part.groups[dst_gi].ops;
    new_dst.insert(op);
    double dst_cost = part.eval_set(new_dst);
    if (dst_cost >= 1e17) return part;
    
    // Apply, then validate full partition. Revert if gap.
    Partition saved = part;
    part.groups[src_gi].ops = new_src;
    part.groups[src_gi].cost = src_cost;
    part.groups[src_gi].gen++;
    part.groups[dst_gi].ops = new_dst;
    part.groups[dst_gi].cost = dst_cost;
    part.groups[dst_gi].gen++;
    part.rebuild_index();
    
    if (partition_has_gap(part)) return saved;
    return part;
}

// ============================================================================
// Mutate: block move (move a connected subpath between groups)
// ============================================================================

Partition mutate_block_move(Partition part, std::mt19937& rng) {
    auto ag = alive_groups(part);
    if (ag.empty()) return part;
    
    // Pick a random group with ≥2 ops
    std::vector<size_t> candidates;
    for (auto gi : ag)
        if (part.groups[gi].ops.size() >= 2) candidates.push_back(gi);
    if (candidates.empty()) return part;
    
    size_t src_gi = candidates[rng() % candidates.size()];
    auto& src_ops = part.groups[src_gi].ops;
    
    // Pick a random border op as seed, grow a connected block of 2-3 ops
    auto borders = part.border_ops(src_gi);
    if (borders.empty()) return part;
    
    size_t seed_op = borders[rng() % borders.size()];
    std::set<size_t> block = {seed_op};
    int block_size = 2 + (rng() % 2);  // 2 or 3
    
    // Grow block by adding random neighbors within the group
    for (int i = 1; i < block_size; i++) {
        std::vector<size_t> growth_candidates;
        for (auto op : block) {
            for (auto p : part.dag->op_preds[op])
                if (src_ops.count(p) && !block.count(p))
                    growth_candidates.push_back(p);
            for (auto s : part.dag->op_succs[op])
                if (src_ops.count(s) && !block.count(s))
                    growth_candidates.push_back(s);
        }
        if (growth_candidates.empty()) break;
        block.insert(growth_candidates[rng() % growth_candidates.size()]);
    }
    
    // Remainder must stay connected and non-empty
    std::set<size_t> remainder = src_ops;
    for (auto op : block) remainder.erase(op);
    if (remainder.empty()) return part;
    auto comps = part.connected_components(remainder);
    if (comps.size() != 1) return part;
    
    // Find a target group adjacent to the block (via all neighbor edges)
    std::set<size_t> target_groups;
    for (auto op : block) {
        for (auto nbr : part.dag->op_neighbors[op])
            for (auto gi : part.groups_of(nbr))
                if (gi != src_gi) target_groups.insert(gi);
    }
    if (target_groups.empty()) {
        // No adjacent group — create new group for the block
        double block_cost = part.eval_set(block);
        if (block_cost >= 1e17) return part;
        double rem_cost = part.eval_set(remainder);
        if (rem_cost >= 1e17) return part;
        
        Partition saved = part;
        part.groups[src_gi].ops = remainder;
        part.groups[src_gi].cost = rem_cost;
        part.groups[src_gi].gen++;
        part.add_group(block, block_cost);
        part.rebuild_index();
        if (partition_has_gap(part)) return saved;
        return part;
    }
    
    // Pick random target
    std::vector<size_t> tgt_vec(target_groups.begin(), target_groups.end());
    size_t dst_gi = tgt_vec[rng() % tgt_vec.size()];
    
    if (part.dag->merge_creates_cycle(block, part.groups[dst_gi].ops)) return part;

    // Evaluate
    double rem_cost = part.eval_set(remainder);
    if (rem_cost >= 1e17) return part;
    
    std::set<size_t> new_dst = part.groups[dst_gi].ops;
    for (auto op : block) new_dst.insert(op);
    double dst_cost = part.eval_set(new_dst);
    if (dst_cost >= 1e17) return part;
    
    // Apply, then validate full partition
    Partition saved = part;
    part.groups[src_gi].ops = remainder;
    part.groups[src_gi].cost = rem_cost;
    part.groups[src_gi].gen++;
    part.groups[dst_gi].ops = new_dst;
    part.groups[dst_gi].cost = dst_cost;
    part.groups[dst_gi].gen++;
    part.rebuild_index();
    
    if (partition_has_gap(part)) return saved;
    return part;
}

// ============================================================================
// Mutate: random eject
// ============================================================================

Partition mutate_eject(Partition part, std::mt19937& rng) {
    auto ag = alive_groups(part);
    
    // Collect groups with ≥2 ops
    std::vector<size_t> candidates;
    for (auto gi : ag)
        if (part.groups[gi].ops.size() >= 2) candidates.push_back(gi);
    if (candidates.empty()) return part;
    
    // Pick random group
    size_t gi = candidates[rng() % candidates.size()];
    
    // Pick random op from the group
    std::vector<size_t> ops_vec(part.groups[gi].ops.begin(),
                                part.groups[gi].ops.end());
    size_t op = ops_vec[rng() % ops_vec.size()];
    
    auto er = part.eval_eject(op, gi);
    if (!er.feasible) return part;
    
    // Apply: replace group with components + singleton
    Partition saved = part;
    if (er.remainder_components.size() == 1) {
        part.groups[gi].ops = er.remainder_components[0];
        part.groups[gi].cost = er.component_costs[0];
    } else {
        part.groups[gi].ops = er.remainder_components[0];
        part.groups[gi].cost = er.component_costs[0];
        for (size_t c = 1; c < er.remainder_components.size(); c++)
            part.add_group(er.remainder_components[c], er.component_costs[c]);
    }
    part.groups[gi].gen++;
    
    if (er.singleton_cost > 0)
        part.add_group({op}, er.singleton_cost);
    
    part.rebuild_index();
    if (partition_has_gap(part)) return saved;
    return part;
}

// ============================================================================
// Mutate: tensor-centric merge
// ============================================================================

Partition mutate_tensor_merge(Partition part, std::mt19937& rng) {
    const Problem& prob = *part.prob;
    const DAG& dag = *part.dag;
    
    // Collect tensors with ≥2 consumers in different groups
    std::vector<size_t> candidate_tensors;
    for (size_t t = 0; t < prob.num_tensors(); t++) {
        auto& consumers = dag.tensor_consumers[t];
        if (consumers.size() < 2) continue;
        
        // Check if consumers are in ≥2 different groups
        std::set<size_t> groups_seen;
        for (auto cop : consumers)
            for (auto gi : part.groups_of(cop))
                groups_seen.insert(gi);
        if (groups_seen.size() >= 2)
            candidate_tensors.push_back(t);
    }
    if (candidate_tensors.empty()) return part;
    
    // Pick random tensor
    size_t t = candidate_tensors[rng() % candidate_tensors.size()];
    
    // Collect all consumer groups + optionally the producer group
    std::set<size_t> target_groups;
    for (auto cop : dag.tensor_consumers[t])
        for (auto gi : part.groups_of(cop))
            target_groups.insert(gi);
    
    int prod = dag.tensor_producer[t];
    if (prod >= 0)
        for (auto gi : part.groups_of((size_t)prod))
            target_groups.insert(gi);
    
    if (target_groups.size() < 2) return part;
    
    // Check for cycles (pairwise — conservative)
    std::vector<size_t> group_list(target_groups.begin(), target_groups.end());
    for (size_t a = 0; a < group_list.size(); a++)
        for (size_t b = a + 1; b < group_list.size(); b++)
            if (dag.merge_creates_cycle(part.groups[group_list[a]].ops,
                                        part.groups[group_list[b]].ops))
                return part;
    
    // Try full merge first
    std::set<size_t> merged_ops;
    for (auto gi : group_list)
        merged_ops.insert(part.groups[gi].ops.begin(),
                          part.groups[gi].ops.end());
    
    double merged_cost = part.eval_set(merged_ops);
    if (merged_cost < 1e17) {
        // Apply full merge: first group absorbs all, rest are killed
        Partition saved = part;
        part.groups[group_list[0]].ops = merged_ops;
        part.groups[group_list[0]].cost = merged_cost;
        part.groups[group_list[0]].gen++;
        for (size_t i = 1; i < group_list.size(); i++) {
            part.groups[group_list[i]].alive = false;
            part.groups[group_list[i]].gen++;
        }
        part.rebuild_index();
        if (!partition_has_gap(part)) return part;
        part = std::move(saved);  // revert, fall through to extract
    }
    
    // Fallback: extract just the consumer ops (+ producer) into a new group.
    std::set<size_t> extract_ops;
    for (auto cop : dag.tensor_consumers[t])
        extract_ops.insert(cop);
    if (prod >= 0)
        extract_ops.insert((size_t)prod);
    
    // Note: no merge_creates_cycle check here — extract_ops and remainders are
    // separate groups, not being merged. Acyclicity is validated at the end
    // via partition_has_gap (which calls is_acyclic).
    double extract_cost = part.eval_set(extract_ops);
    if (extract_cost >= 1e17) return part;
    
    // Compute remainder costs
    bool feasible = true;
    struct RemInfo { size_t gi; std::set<size_t> ops; double cost; };
    std::vector<RemInfo> remainders;
    
    for (auto gi : group_list) {
        std::set<size_t> rem;
        for (auto op : part.groups[gi].ops)
            if (!extract_ops.count(op))
                rem.insert(op);
        double rc = 0;
        if (!rem.empty()) {
            rc = part.eval_set(rem);
            if (rc >= 1e17) { feasible = false; break; }
        }
        remainders.push_back({gi, rem, rc});
    }
    if (!feasible) return part;
    
    // Apply extract
    Partition saved = part;
    for (auto& ri : remainders) {
        if (ri.ops.empty()) {
            part.groups[ri.gi].alive = false;
        } else {
            part.groups[ri.gi].ops = ri.ops;
            part.groups[ri.gi].cost = ri.cost;
        }
        part.groups[ri.gi].gen++;
    }
    part.add_group(extract_ops, extract_cost);
    part.rebuild_index();
    if (partition_has_gap(part)) return saved;
    return part;
}

// ============================================================================
// Compound mutation: mix of random operators
// ============================================================================

// ============================================================================
// Mutate: remove a recomputation group
// ============================================================================

Partition mutate_de_recompute(Partition part, std::mt19937& rng) {
    std::vector<size_t> candidates;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        bool all_covered = true;
        for (auto op : part.groups[gi].ops) {
            bool in_other = false;
            for (auto gj : part.groups_of(op))
                if (gj != gi && part.groups[gj].alive) { in_other = true; break; }
            if (!in_other) { all_covered = false; break; }
        }
        if (all_covered) candidates.push_back(gi);
    }
    if (candidates.empty()) return part;
    std::shuffle(candidates.begin(), candidates.end(), rng);

    // Greedy removal: try removing each candidate, re-checking coverage
    // after each removal (since removing one group may invalidate another).
    bool any_removed = false;
    for (auto gi : candidates) {
        if (!part.groups[gi].alive) continue;
        // Re-check: all ops in gi still covered by other alive groups?
        bool still_redundant = true;
        for (auto op : part.groups[gi].ops) {
            bool in_other = false;
            for (auto gj : part.groups_of(op))
                if (gj != gi && part.groups[gj].alive) { in_other = true; break; }
            if (!in_other) { still_redundant = false; break; }
        }
        if (still_redundant) {
            part.groups[gi].alive = false;
            part.groups[gi].gen++;
            any_removed = true;
        }
    }
    if (any_removed) part.rebuild_index();
    return part;
}

Partition mutate_compound(Partition part, int num_mutations, std::mt19937& rng) {
    const int max_attempts = num_mutations * 10;
    int applied  = 0;
    int attempts = 0;

    while (applied < num_mutations && attempts < max_attempts) {
        attempts++;
        size_t before_groups = part.num_alive();
        double before_cost   = part.total_cost();

        int choice = rng() % 7;
        switch (choice) {
            case 0: part = mutate_merge(std::move(part), rng);        break;
            case 1: part = mutate_split(std::move(part), rng);        break;
            case 2: part = mutate_reassign(std::move(part), rng);     break;
            case 3: part = mutate_block_move(std::move(part), rng);   break;
            case 4: part = mutate_eject(std::move(part), rng);        break;
            case 5: part = mutate_tensor_merge(std::move(part), rng); break;
            case 6: part = mutate_de_recompute(std::move(part), rng); break;
        }

        if (part.num_alive() != before_groups || part.total_cost() != before_cost)
            applied++;

#ifndef NDEBUG
        // Verify cost consistency: each group's stored cost matches eval_set
        if (g_verbose) {
            for (size_t gi = 0; gi < part.groups.size(); gi++) {
                if (!part.groups[gi].alive) continue;
                double stored = part.groups[gi].cost;
                double fresh = part.eval_set(part.groups[gi].ops);
                if (std::abs(stored - fresh) > 0.1 * std::max(1.0, stored) + 1.0) {
                    std::cerr << "    MUTATE COST INCONSISTENCY: G" << gi
                              << " stored=" << stored << " eval=" << fresh
                              << " mutation=" << choice << "\n";
                }
            }
        }
#endif
    }
    return part;
}

// ============================================================================
// Crossover: agreement-based (Sanders & Schulz style)
//
// 1. Map each op to its group in both parents
// 2. Find op-pairs that are in the same group in BOTH parents ("agreed")
// 3. These form clusters that must stay together in the child
// 4. Greedily assign clusters to groups using eval_set
// ============================================================================

Partition crossover(const Partition& parent_a, const Partition& parent_b,
                    std::mt19937& rng,
                    const MerkleHashes* mh) {
    const Problem& prob = *parent_a.prob;
    const DAG& dag = *parent_a.dag;
    size_t n_ops = prob.num_ops();
    
    // Step 1: Map each op to its group index in each parent
    std::vector<int> grp_a(n_ops, -1), grp_b(n_ops, -1);
    for (size_t gi = 0; gi < parent_a.groups.size(); gi++)
        if (parent_a.groups[gi].alive)
            for (auto op : parent_a.groups[gi].ops)
                grp_a[op] = (int)gi;
    for (size_t gi = 0; gi < parent_b.groups.size(); gi++)
        if (parent_b.groups[gi].alive)
            for (auto op : parent_b.groups[gi].ops)
                grp_b[op] = (int)gi;
    
    // Step 2: Cluster ops that agree in both parents.
    // Apply Merkle canonicalisation so symmetric op permutations are treated as
    // identical: within each Merkle class, sort by grp_a then match rank-for-rank
    // in grp_b.  Without this, two symmetric parents would produce no agreement
    // clusters even though they encode the same partition structure.
    if (mh) merkle_canonicalise(*mh, grp_a, grp_b);

    // Two ops are in the same cluster iff they share the same group in BOTH parents.
    // O(n) via hashing (grp_a, grp_b) pairs — no pairwise comparison needed.
    std::map<std::pair<int,int>, std::set<size_t>> clusters;
    for (size_t i = 0; i < n_ops; i++) {
        if (grp_a[i] < 0 || grp_b[i] < 0) continue;
        clusters[{grp_a[i], grp_b[i]}].insert(i);
    }
    
    // Step 3: Shuffle clusters and greedily assign to groups.
    std::vector<std::pair<int,int>> cluster_keys;
    for (auto& [k, _] : clusters) cluster_keys.push_back(k);
    std::shuffle(cluster_keys.begin(), cluster_keys.end(), rng);
    
    Partition child;
    child.prob = &prob;
    child.dag = &dag;
    child.cache = parent_a.cache;
    child.rebuild_index();
    
    for (auto ck : cluster_keys) {
        auto& cluster = clusters[ck];
        
        // Find existing child groups adjacent to this cluster
        // Uses op_to_groups_ which is maintained incrementally below
        std::set<size_t> adj_child_groups;
        for (auto op : cluster) {
            for (auto nbr : dag.op_neighbors[op])
                for (auto gi : child.groups_of(nbr))
                    adj_child_groups.insert(gi);
        }
        
        // Try merging with each adjacent group, pick cheapest
        double best_cost = 1e18;
        size_t best_gi = SIZE_MAX;
        
        for (auto gi : adj_child_groups) {
            if (child.dag->merge_creates_cycle(cluster, child.groups[gi].ops)) continue;
            std::set<size_t> merged = child.groups[gi].ops;
            for (auto op : cluster) merged.insert(op);
            double c = child.eval_set(merged);
            if (c < 1e17 && c < best_cost) {
                best_cost = c;
                best_gi = gi;
            }
        }
        
        // Also evaluate the cluster as standalone
        double standalone = child.eval_set(cluster);
        
        if (best_gi != SIZE_MAX && best_cost < standalone + 100) {
            // Merge into existing group — incremental index update
            child.merge_ops_into(best_gi, cluster);
            child.groups[best_gi].cost = best_cost;
            child.groups[best_gi].gen++;
        } else if (standalone < 1e17) {
            child.add_group(cluster, standalone);  // add_group updates index
        } else if (best_gi != SIZE_MAX) {
            child.merge_ops_into(best_gi, cluster);
            child.groups[best_gi].cost = best_cost;
            child.groups[best_gi].gen++;
        } else {
            // Last resort: singletons
            for (auto op : cluster) {
                double c = child.eval_set({op});
                child.add_group({op}, c);
            }
        }
    }
    
    // Single rebuild at the end for cleanup (removes dead entries, etc.)
    child.rebuild_index();
    
    // Safety: crossover can accidentally create cycles when splicing clusters
    // from two parents. Reject cyclic children — fall back to better parent.
    if (!child.is_acyclic()) {
        return (parent_a.total_cost() <= parent_b.total_cost()) ? parent_a : parent_b;
    }
    
#ifndef NDEBUG
    // Verify cost consistency after crossover
    if (g_verbose) {
        for (size_t gi = 0; gi < child.groups.size(); gi++) {
            if (!child.groups[gi].alive) continue;
            double stored = child.groups[gi].cost;
            double fresh = child.eval_set(child.groups[gi].ops);
            if (std::abs(stored - fresh) > 0.1 * std::max(1.0, stored) + 1.0) {
                std::cerr << "    CROSSOVER COST INCONSISTENCY: G" << gi
                          << " stored=" << stored << " eval=" << fresh << "\n";
            }
        }
    }
#endif

    return child;
}