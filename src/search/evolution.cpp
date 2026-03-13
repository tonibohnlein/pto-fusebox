#include "search/evolution.h"
#include <algorithm>
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

    if (part.dag->merge_creates_cycle(part.groups[ga].ops, part.groups[gb].ops)) return part;
    
    // Merge: add all ops from gb into ga
    std::set<size_t> merged = part.groups[ga].ops;
    for (auto op : part.groups[gb].ops)
        merged.insert(op);
    
    if (part.creates_ephemeral_gap(merged, ga, gb)) return part;
    double cost = part.eval_set(merged);
    if (cost >= 1e17) return part;  // infeasible merge
    
    part.groups[ga].ops = merged;
    part.groups[ga].cost = cost;
    part.groups[ga].gen++;
    part.groups[gb].alive = false;
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
    
    // Pick random bridge edge and split
    auto [op_a, op_b] = bridges[rng() % bridges.size()];
    auto sr = part.eval_split(op_a, op_b, gi);
    if (!sr.feasible) return part;
    
    // Apply split
    part.groups[gi].ops = sr.side_a;
    part.groups[gi].cost = sr.cost_a;
    part.groups[gi].gen++;
    part.add_group(sr.side_b, sr.cost_b);
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
    if (part.creates_ephemeral_gap(new_dst, dst_gi, SIZE_MAX)) return part;
    double dst_cost = part.eval_set(new_dst);
    if (dst_cost >= 1e17) return part;
    
    // Apply regardless of cost (this is random mutation, not optimization)
    part.groups[src_gi].ops = new_src;
    part.groups[src_gi].cost = src_cost;
    part.groups[src_gi].gen++;
    part.groups[dst_gi].ops = new_dst;
    part.groups[dst_gi].cost = dst_cost;
    part.groups[dst_gi].gen++;
    part.rebuild_index();
    
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
        
        part.groups[src_gi].ops = remainder;
        part.groups[src_gi].cost = rem_cost;
        part.groups[src_gi].gen++;
        part.add_group(block, block_cost);
        part.rebuild_index();
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
    if (part.creates_ephemeral_gap(new_dst, dst_gi, SIZE_MAX)) return part;
    double dst_cost = part.eval_set(new_dst);
    if (dst_cost >= 1e17) return part;
    
    // Apply
    part.groups[src_gi].ops = remainder;
    part.groups[src_gi].cost = rem_cost;
    part.groups[src_gi].gen++;
    part.groups[dst_gi].ops = new_dst;
    part.groups[dst_gi].cost = dst_cost;
    part.groups[dst_gi].gen++;
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
    
    // Evaluate eject (handles disconnection into components)
    auto er = part.eval_eject(op, gi);
    if (!er.feasible) return part;
    
    // Apply: replace group with components + singleton
    if (er.remainder_components.size() == 1) {
        part.groups[gi].ops = std::move(er.remainder_components[0]);
        part.groups[gi].cost = er.component_costs[0];
    } else {
        part.groups[gi].ops = std::move(er.remainder_components[0]);
        part.groups[gi].cost = er.component_costs[0];
        for (size_t c = 1; c < er.remainder_components.size(); c++)
            part.add_group(std::move(er.remainder_components[c]),
                           er.component_costs[c]);
    }
    part.groups[gi].gen++;
    
    if (er.singleton_cost > 0)
        part.add_group({op}, er.singleton_cost);
    
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
    
    std::vector<size_t> group_list_vec(group_list.begin(), group_list.end());
    if (part.creates_ephemeral_gap(merged_ops, group_list_vec)) return part;
    double merged_cost = part.eval_set(merged_ops);
    if (merged_cost < 1e17) {
        // Apply full merge: first group absorbs all, rest are killed
        part.groups[group_list[0]].ops = std::move(merged_ops);
        part.groups[group_list[0]].cost = merged_cost;
        part.groups[group_list[0]].gen++;
        for (size_t i = 1; i < group_list.size(); i++) {
            part.groups[group_list[i]].alive = false;
            part.groups[group_list[i]].gen++;
        }
        part.rebuild_index();
        return part;
    }
    
    // Fallback: extract just the consumer ops (+ producer) into a new group
    std::set<size_t> extract_ops;
    for (auto cop : dag.tensor_consumers[t])
        extract_ops.insert(cop);
    if (prod >= 0)
        extract_ops.insert((size_t)prod);
    
    if (part.creates_ephemeral_gap(extract_ops, group_list_vec)) return part;
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
        remainders.push_back({gi, std::move(rem), rc});
    }
    if (!feasible) return part;
    
    // Apply extract
    for (auto& ri : remainders) {
        if (ri.ops.empty()) {
            part.groups[ri.gi].alive = false;
        } else {
            part.groups[ri.gi].ops = std::move(ri.ops);
            part.groups[ri.gi].cost = ri.cost;
        }
        part.groups[ri.gi].gen++;
    }
    part.add_group(std::move(extract_ops), extract_cost);
    part.rebuild_index();
    return part;
}

// ============================================================================
// Compound mutation: mix of random operators
// ============================================================================

Partition mutate_compound(Partition part, int num_mutations, std::mt19937& rng) {
    for (int i = 0; i < num_mutations; i++) {
        int choice = rng() % 6;
        switch (choice) {
            case 0: part = mutate_merge(std::move(part), rng); break;
            case 1: part = mutate_split(std::move(part), rng); break;
            case 2: part = mutate_reassign(std::move(part), rng); break;
            case 3: part = mutate_block_move(std::move(part), rng); break;
            case 4: part = mutate_eject(std::move(part), rng); break;
            case 5: part = mutate_tensor_merge(std::move(part), rng); break;
        }
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
                    std::mt19937& rng) {
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
        // Use op_neighbors + groups_of (requires index to be current)
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
            if (child.creates_ephemeral_gap(merged, gi, SIZE_MAX)) continue;
            double c = child.eval_set(merged);
            if (c < 1e17 && c < best_cost) {
                best_cost = c;
                best_gi = gi;
            }
        }
        
        // Also evaluate the cluster as standalone
        double standalone = child.eval_set(cluster);
        
        if (best_gi != SIZE_MAX && best_cost < standalone + 100) {
            for (auto op : cluster) child.groups[best_gi].ops.insert(op);
            child.groups[best_gi].cost = best_cost;
            child.groups[best_gi].gen++;
        } else if (standalone < 1e17) {
            child.add_group(cluster, standalone);
        } else {
            if (best_gi != SIZE_MAX) {
                for (auto op : cluster) child.groups[best_gi].ops.insert(op);
                child.groups[best_gi].cost = best_cost;
                child.groups[best_gi].gen++;
            } else {
                child.add_group(cluster, 1e18);
            }
        }
        // Keep index current for next cluster iteration
        child.rebuild_index();
    }
    
    return child;
}