#include "search/evolution.h"
#include "search/verbose.h"
#include "search/local_search.h"  // partition_has_gap
#include "search/feasibility.h"
#include "search/partition_moves.h"
#include "search/structural_ops.h"
#include "symmetry/merkle_hash.h"
#include <algorithm>
#include <cassert>
#include <iostream>

// Under the new ephemeral rule, a tensor produced AND consumed inside a group
// is always ephemeral. partition_has_gap() checks BOTH acyclicity AND
// mixed-consumer violations (external consumers not covered by recomputation).

#include <functional>
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

    if (!part.acyclic_merge_local(ga, gb)) return part;
    
    // Eval feasibility
    auto r = partition_moves::eval_merge(part, ga, gb);
    if (!r.feasible) return part;

    // Apply (includes Kahn's check)
    auto affected = partition_moves::apply_merge(part, ga, gb);
    if (affected.empty()) return part;
    part.rebuild_index();
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
    
    // Pick random bridge edge
    auto [op_a, op_b] = bridges[rng() % bridges.size()];

    // Eval feasibility
    auto sr = part.eval_split(op_a, op_b, gi);
    if (!sr.feasible) return part;
    if (part.split_creates_ephemeral_gap({sr.side_a, sr.side_b}, {gi})) return part;

    // Apply (includes cheap pre-filter + Kahn's)
    auto affected = partition_moves::apply_split(part, op_a, op_b, gi);
    if (affected.empty()) return part;
    part.rebuild_index();
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

    // Check: removing op from src must leave it connected and non-empty
    if (part.groups[src_gi].ops.size() <= 1) return part;
    if (!structural_ops::is_connected_without(part.groups[src_gi].ops, op, *part.dag))
        return part;  // would disconnect

    // Eval feasibility (from=src_gi, to=dst_gi)
    auto r = partition_moves::eval_steal(part, op, src_gi, dst_gi);
    if (!r.feasible) return part;

    // Apply (includes Kahn's check)
    auto affected = partition_moves::apply_steal(part, op, src_gi, dst_gi);
    if (affected.empty()) return part;
    part.rebuild_index();
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
    
    // Eval feasibility
    auto er = part.eval_eject(op, gi);
    if (!er.feasible) return part;

    // Gap check: ejecting may make T ephemeral in remainder while op needs it
    {
        std::vector<std::set<size_t>> components;
        components.push_back({op});
        for (auto& comp : er.remainder_components)
            components.push_back(comp);
        if (part.split_creates_ephemeral_gap(components, {gi})) return part;
    }

    // Apply (includes cheap pre-filter + Kahn's)
    auto affected = partition_moves::apply_eject(part, op, gi);
    if (affected.empty()) return part;
    part.rebuild_index();
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
    std::vector<size_t> group_list(target_groups.begin(), target_groups.end());

    // Try full merge first
    auto mr = partition_moves::eval_tensor_merge(part, group_list);
    if (mr.feasible) {
        auto affected = partition_moves::apply_tensor_merge(part, group_list);
        if (!affected.empty()) {
            part.rebuild_index();
            return part;
        }
    }
    
    // Fallback: extract consumer ops (+ producer) into a new group
    std::set<size_t> extract_ops;
    for (auto cop : dag.tensor_consumers[t])
        extract_ops.insert(cop);
    if (prod >= 0)
        extract_ops.insert((size_t)prod);

    auto er = partition_moves::eval_tensor_extract(part, extract_ops, group_list);
    if (!er.feasible) return part;

    auto affected = partition_moves::apply_tensor_extract(part, extract_ops, group_list);
    if (affected.empty()) return part;
    part.rebuild_index();
    return part;
}

// ============================================================================
// Compound mutation: mix of random operators
// ============================================================================

// ============================================================================
// Mutate: remove recomputed ops
// ============================================================================

Partition mutate_de_recompute(Partition part, std::mt19937& rng) {
    // Collect (group, op) pairs where op is recomputed in that group.
    std::vector<std::pair<size_t, size_t>> candidates;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        for (auto op : part.groups[gi].ops) {
            auto r = partition_moves::eval_de_recompute(part, gi, op);
            if (r.feasible) candidates.emplace_back(gi, op);
        }
    }
    if (candidates.empty()) return part;
    std::shuffle(candidates.begin(), candidates.end(), rng);

    // Greedy removal: try each candidate, re-checking after each removal.
    for (auto [gi, op] : candidates) {
        if (!part.groups[gi].alive) continue;
        if (!part.groups[gi].ops.count(op)) continue;
        auto r = partition_moves::eval_de_recompute(part, gi, op);
        if (!r.feasible) continue;
        auto affected = partition_moves::apply_de_recompute(part, gi, op);
        if (!affected.empty()) {
            part.rebuild_index();
        }
    }
    return part;
}

CoupledPartition mutate_compound_coupled(CoupledPartition cp,
                                          int num_mutations,
                                          std::mt19937& rng) {
    cp.part = mutate_compound(std::move(cp.part), num_mutations, rng);

    // Extend coupling arrays for any new groups created by splits
    while (cp.next_group.size() < cp.part.groups.size()) cp.next_group.push_back(SIZE_MAX);
    while (cp.prev_group.size() < cp.part.groups.size()) cp.prev_group.push_back(SIZE_MAX);

    // Remove coupling edges broken by the mutation (dead groups, tensors no
    // longer boundary outputs, etc.). This is called within the mutation so
    // coupling state remains valid before local search begins.
    cp.invalidate_couplings();

    // Mutations can introduce new group DAG edges that make existing coupling
    // links cycle at the chain level (even though the group DAG is acyclic).
    // Remove any such links before handing the result to FM / greedy descent.
    cp.fix_chain_couplings();

    return cp;
}

Partition mutate_compound(Partition part, int num_mutations, std::mt19937& rng) {
    const int max_attempts = num_mutations * 10;
    int applied  = 0;
    int attempts = 0;

    while (applied < num_mutations && attempts < max_attempts) {
        attempts++;
        size_t before_groups = part.num_alive();
        double before_cost   = part.total_cost();

        int choice = rng() % 6;
        switch (choice) {
            case 0: part = mutate_merge(std::move(part), rng);        break;
            case 1: part = mutate_split(std::move(part), rng);        break;
            case 2: part = mutate_reassign(std::move(part), rng);     break;
            case 3: part = mutate_eject(std::move(part), rng);        break;
            case 4: part = mutate_tensor_merge(std::move(part), rng); break;
            case 5: part = mutate_de_recompute(std::move(part), rng); break;
        }

        if (part.num_alive() != before_groups || part.total_cost() != before_cost)
            applied++;

        // Verify coverage: every op in at least one alive group.
        // If a mutation lost an op (rare edge case in split/de_recompute),
        // the partition is invalid — stop mutating and let the caller's
        // partition_has_gap check reject it.
        {
            bool coverage_ok = true;
            for (size_t i = 0; i < part.prob->num_ops(); i++) {
                bool found = false;
                for (auto gi : part.groups_of(i))
                    if (part.groups[gi].alive) { found = true; break; }
                if (!found) { coverage_ok = false; break; }
            }
            if (!coverage_ok) break;  // bail out of mutation loop
        }
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
            if (!child.acyclic_add_ops_into(cluster, gi)) continue;
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
    
    // Safety: crossover can create cycles or mixed-consumer violations
    // when splicing clusters from two parents.
    if (partition_has_gap(child)) {
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