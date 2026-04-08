#include "search/evolution.h"
#include "search/verbose.h"
#include "search/local_search.h"  // partition_has_gap
#include "search/feasibility.h"
#include "search/partition_moves.h"
#include "search/structural_ops.h"
#include "search/coupling_search.h"  // eval_couple, apply_couple, apply_uncouple
#include "symmetry/merkle_hash.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>

// --- Mutation statistics (thread-safe) ---
static std::atomic<uint64_t> g_mut_total_requested{0};
static std::atomic<uint64_t> g_mut_total_applied{0};
static std::atomic<uint64_t> g_mut_total_attempts{0};
static std::atomic<uint64_t> g_mut_per_type_attempts[6]{};
static std::atomic<uint64_t> g_mut_per_type_applied[6]{};

// Crossover statistics
static std::atomic<uint64_t> g_xover_calls{0};
static std::atomic<uint64_t> g_xover_clusters{0};
static std::atomic<uint64_t> g_xover_eval_set_calls{0};
static std::atomic<uint64_t> g_xover_gap_rejects{0};

void print_mutation_stats() {
    uint64_t req = g_mut_total_requested.load();
    uint64_t app = g_mut_total_applied.load();
    uint64_t att = g_mut_total_attempts.load();
    if (att == 0 && g_xover_calls.load() == 0) return;
    if (att > 0) {
        std::cerr << "  Mutation stats: " << app << "/" << req << " applied/requested"
                  << "  (" << att << " attempts, "
                  << (att > 0 ? (int)(100.0 * app / att) : 0) << "% success rate)\n";
        const char* names[] = {"merge", "split", "reassign", "eject",
                               "tensor_merge", "force_recompute"};
        for (int i = 0; i < 6; i++) {
            uint64_t a = g_mut_per_type_attempts[i].load();
            uint64_t s = g_mut_per_type_applied[i].load();
            if (a > 0)
                std::cerr << "    " << names[i] << ": " << s << "/" << a
                          << " (" << (int)(100.0 * s / a) << "%)\n";
        }
    }
    uint64_t xc = g_xover_calls.load();
    if (xc > 0) {
        std::cerr << "  Crossover stats: " << xc << " calls"
                  << ", avg clusters=" << g_xover_clusters.load() / xc
                  << ", avg evals=" << g_xover_eval_set_calls.load() / xc
                  << ", gap_rejects=" << g_xover_gap_rejects.load()
                  << " (" << (int)(100.0 * g_xover_gap_rejects.load() / xc) << "%)\n";
    }
}

void reset_mutation_stats() {
    g_mut_total_requested = 0;
    g_mut_total_applied = 0;
    g_mut_total_attempts = 0;
    for (int i = 0; i < 6; i++) {
        g_mut_per_type_attempts[i] = 0;
        g_mut_per_type_applied[i] = 0;
    }
    g_xover_calls = 0;
    g_xover_clusters = 0;
    g_xover_eval_set_calls = 0;
    g_xover_gap_rejects = 0;
}

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
    FlatSet<size_t> merged_ops = part.groups[ga].ops;
    merged_ops.insert(part.groups[gb].ops.begin(), part.groups[gb].ops.end());
    if (part.creates_ephemeral_gap(merged_ops, ga, gb)) return part;
    double merged_cost = part.eval_set(merged_ops);
    if (merged_cost >= 1e17) return part;
    double saving = part.groups[ga].cost + part.groups[gb].cost - merged_cost;
    if (saving < -1e15) return part;  // infeasible

    auto affected = partition_moves::apply_merge(part, ga, gb, merged_cost);
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
    if (!part.acyclic_split_local(sr.side_a, sr.side_b, gi)) return part;

    auto affected = partition_moves::apply_split(part, op_a, op_b, gi, &sr);
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
    if (!part.acyclic_steal_local(op, src_gi, dst_gi)) return part;

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
        std::vector<FlatSet<size_t>> components;
        components.push_back({op});
        for (auto& comp : er.remainder_components)
            components.push_back(comp);
        if (part.split_creates_ephemeral_gap(components, {gi})) return part;
    }

    // Acyclicity: ejecting a recomputed op is a de_recompute.
    {
        bool in_other = false;
        for (auto gj : part.groups_of(op))
            if (gj != gi && part.groups[gj].alive) { in_other = true; break; }
        if (in_other && !part.acyclic_de_recompute_local(op, gi)) return part;
    }

    auto affected = partition_moves::apply_eject(part, op, gi, &er);
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
        
        FlatSet<size_t> groups_seen;
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
    FlatSet<size_t> target_groups;
    for (auto cop : dag.tensor_consumers[t])
        for (auto gi : part.groups_of(cop))
            target_groups.insert(gi);
    
    int prod = dag.tensor_producer[t];
    if (prod >= 0)
        for (auto gi : part.groups_of((size_t)prod))
            target_groups.insert(gi);
    
    if (target_groups.size() < 2) return part;
    std::vector<size_t> group_list(target_groups.begin(), target_groups.end());

    // Try full merge first (check acyclicity before eval — cheaper, filters early)
    if (part.acyclic_merge_local(group_list)) {
        auto mr = partition_moves::eval_tensor_merge(part, group_list);
        if (mr.feasible) {
            double old_cost = 0;
            for (auto gi : group_list) old_cost += part.groups[gi].cost;
            double merged_cost = old_cost - mr.saving;
            auto affected = partition_moves::apply_tensor_merge(part, group_list, merged_cost);
            if (!affected.empty()) {
                part.rebuild_index();
                return part;
            }
        }
    }

    // Fallback 1: extract consumer ops (+ producer) into a new group
    FlatSet<size_t> extract_ops;
    std::vector<size_t> consumer_ops_vec;
    for (auto cop : dag.tensor_consumers[t]) {
        extract_ops.insert(cop);
        consumer_ops_vec.push_back(cop);
    }
    if (prod >= 0)
        extract_ops.insert((size_t)prod);

    if (part.acyclic_extract_local(extract_ops)) {
        auto er = partition_moves::eval_tensor_extract(part, extract_ops, group_list);
        if (er.feasible) {
            auto affected = partition_moves::apply_tensor_extract(part, extract_ops, group_list);
            if (!affected.empty()) {
                part.rebuild_index();
                return part;
            }
        }
    }

    // Fallback 2: split consumers into balanced sub-groups with producer recomputed
    if (consumer_ops_vec.size() >= 2) {
        auto sr = partition_moves::eval_tensor_extract_split(
            part, t, consumer_ops_vec, group_list);
        if (sr.feasible) {
            auto affected = partition_moves::apply_tensor_extract_split(
                part, sr, group_list);
            if (!affected.empty()) {
                part.rebuild_index();
                return part;
            }
        }
    }

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
            if (!part.acyclic_de_recompute_local(op, gi)) continue;
            auto r = partition_moves::eval_de_recompute(part, gi, op);
            if (r.feasible) candidates.emplace_back(gi, op);
        }
    }
    if (candidates.empty()) return part;
    std::shuffle(candidates.begin(), candidates.end(), rng);

    // Greedy removal: try each candidate, re-checking after each removal.
    bool any_applied = false;
    for (auto [gi, op] : candidates) {
        if (!part.groups[gi].alive) continue;
        if (!part.groups[gi].ops.count(op)) continue;
        // rebuild_index deferred — must refresh before acyclicity check
        if (any_applied) { part.rebuild_index(); any_applied = false; }
        if (!part.acyclic_de_recompute_local(op, gi)) continue;
        auto r = partition_moves::eval_de_recompute(part, gi, op);
        if (!r.feasible) continue;
        double new_ga_cost = part.groups[gi].cost - r.saving;
        auto affected = partition_moves::apply_de_recompute(part, gi, op, new_ga_cost);
        if (!affected.empty()) any_applied = true;
    }
    if (any_applied) part.rebuild_index();
    return part;
}

// ============================================================================
// Mutate: force recomputation around a multi-consumer tensor
//
// Pick a tensor T with ≥2 consumers, produced by op P (not a model input).
// Extract P and each consumer C_i from their current groups and form new
// groups {P, C_i}.  T becomes ephemeral in each new group.
// ============================================================================

Partition mutate_force_recompute(Partition part, std::mt19937& rng) {
    const auto& dag = *part.dag;
    const auto& prob = *part.prob;

    // Collect candidate tensors: produced by an op, ≥2 consumers
    std::vector<size_t> candidates;
    for (size_t t = 0; t < dag.tensor_producer.size(); t++) {
        int prod = dag.tensor_producer[t];
        if (prod < 0) continue;  // model input, no producer op
        if (dag.tensor_consumers[t].size() < 2) continue;
        candidates.push_back(t);
    }
    if (candidates.empty()) return part;

    // Pick a random candidate tensor
    size_t t = candidates[rng() % candidates.size()];
    size_t prod_op = (size_t)dag.tensor_producer[t];
    const auto& consumers = dag.tensor_consumers[t];

    // For each consumer C_i, form a new group {P, C_i}.
    // First check that each {P, C_i} is feasible (fits in fast memory).
    std::vector<std::pair<size_t, double>> new_groups;  // (consumer_op, cost)
    for (auto cop : consumers) {
        FlatSet<size_t> ops = {prod_op, cop};
        double cost = part.eval_set(ops);
        if (cost >= 1e17) continue;  // doesn't fit → skip this consumer
        new_groups.push_back({cop, cost});
    }
    if (new_groups.empty()) return part;  // no feasible {P, C_i} pair

    // Collect group memberships BEFORE mutating ops (groups_of reads op_to_groups_).
    std::vector<std::pair<size_t, size_t>> erase_list;  // (group, op)
    for (auto& [cop, cost] : new_groups)
        for (auto gi : part.groups_of(cop))
            if (part.groups[gi].alive) erase_list.push_back({gi, cop});
    for (auto gi : part.groups_of(prod_op))
        if (part.groups[gi].alive) erase_list.push_back({gi, prod_op});

    // Now erase ops from groups and collect affected set.
    FlatSet<size_t> affected_groups;
    for (auto [gi, op] : erase_list) {
        part.groups[gi].ops.erase(op);
        affected_groups.insert(gi);
    }

    // Handle affected groups: re-evaluate costs, split disconnected components,
    // kill empty groups.  No rebuild_index needed — this loop only uses
    // groups[gi].ops directly, not groups_of().
    for (auto gi : affected_groups) {
        if (!part.groups[gi].alive) continue;
        if (part.groups[gi].ops.empty()) {
            part.groups[gi].alive = false;
            continue;
        }
        // Split disconnected components
        auto comps = structural_ops::connected_components(part.groups[gi].ops, dag);
        if (comps.size() <= 1) {
            // Still connected — just re-evaluate cost
            double nc = part.eval_set(part.groups[gi].ops);
            part.groups[gi].cost = (nc < 1e17) ? nc : 1e18;
        } else {
            // Keep first component in the original group, create new groups for the rest
            part.groups[gi].ops = comps[0];
            double nc = part.eval_set(comps[0]);
            part.groups[gi].cost = (nc < 1e17) ? nc : 1e18;
            for (size_t c = 1; c < comps.size(); c++) {
                double cc = part.eval_set(comps[c]);
                part.add_group(std::move(comps[c]), (cc < 1e17) ? cc : 1e18);
            }
        }
    }

    // Create the new {P, C_i} groups
    for (auto& [cop, cost] : new_groups) {
        part.add_group({prod_op, cop}, cost);
    }

    part.rebuild_index();
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
    // longer boundary outputs, etc.).
    cp.invalidate_couplings();

    // Remove coupling links that would cycle at the chain level.
    cp.fix_chain_couplings();

    // --- Coupling mutations: randomly perturb coupling structure ---
    // 4 move types: COUPLE, UNCOUPLE, RETAIN_FORCE_SPLIT, FORCE_RETAIN.
    const DAG& dag = *cp.part.dag;
    const Problem& prob = *cp.part.prob;
    int coupling_muts = 1 + (int)(rng() % 2);
    for (int cm = 0; cm < coupling_muts; cm++) {
        int choice = rng() % 12;  // 0-2: uncouple, 3-6: couple, 7-8: rfs, 9: fr, 10-11: eph_fuse

        if (choice <= 2) {
            // --- UNCOUPLE: remove a random retained tensor from a coupling edge ---
            std::vector<std::pair<size_t,size_t>> edges;
            for (size_t g = 0; g < cp.next_group.size(); g++)
                if (cp.next_group[g] != SIZE_MAX)
                    edges.push_back({g, cp.next_group[g]});
            if (edges.empty()) continue;
            auto [ga, gb] = edges[rng() % edges.size()];
            auto it = cp.retained.find({ga, gb});
            if (it == cp.retained.end() || it->second.empty()) continue;
            std::vector<size_t> tensors(it->second.begin(), it->second.end());
            size_t t = tensors[rng() % tensors.size()];
            apply_uncouple(cp, ga, gb, t);

        } else if (choice <= 6) {
            // --- COUPLE: add a coupling edge for a random boundary tensor ---
            std::vector<size_t> boundary_tensors;
            for (size_t t = 0; t < dag.tensor_producer.size(); t++) {
                int prod = dag.tensor_producer[t];
                if (prod < 0) continue;
                if (!dag.tensor_consumers[t].empty())
                    boundary_tensors.push_back(t);
            }
            if (boundary_tensors.empty()) continue;
            size_t t = boundary_tensors[rng() % boundary_tensors.size()];
            int prod_op = dag.tensor_producer[t];
            if (prod_op < 0) continue;

            auto& prod_groups = cp.part.groups_of((size_t)prod_op);
            if (prod_groups.empty()) continue;
            size_t ga = prod_groups[rng() % prod_groups.size()];
            if (!cp.part.groups[ga].alive) continue;
            // Allow boundary outputs and ephemeral tensors (retained coupling).
            if (prod_op < 0 || !cp.part.groups[ga].ops.count((size_t)prod_op)) continue;

            auto& consumers = dag.tensor_consumers[t];
            if (consumers.empty()) continue;
            size_t cons_op = consumers[rng() % consumers.size()];
            auto& cons_groups = cp.part.groups_of(cons_op);
            if (cons_groups.empty()) continue;
            size_t gb = cons_groups[rng() % cons_groups.size()];
            if (!cp.part.groups[gb].alive || ga == gb) continue;
            if (!is_boundary_input_of(cp.part.groups[gb].ops, t, dag)) continue;

            auto ev = eval_couple(cp, ga, gb, t);
            if (ev.feasible && ev.saving > -1e17)
                apply_couple(cp, ga, gb, t);

        } else if (choice <= 8) {
            // --- RETAIN_FORCE_SPLIT: split a group so the two halves can couple ---
            // Pick a random group with ≥3 ops and a bridge edge
            std::vector<size_t> alive;
            for (size_t gi = 0; gi < cp.part.groups.size(); gi++)
                if (cp.part.groups[gi].alive && cp.part.groups[gi].ops.size() >= 3)
                    alive.push_back(gi);
            if (alive.empty()) continue;
            size_t g = alive[rng() % alive.size()];
            auto bridges = structural_ops::bridge_edges(cp.part.groups[g].ops, dag);
            if (bridges.empty()) continue;
            auto [op_a, op_b] = bridges[rng() % bridges.size()];

            // Find a tensor produced by one side and consumed by the other
            auto sa = structural_ops::analyze_split(op_a, op_b,
                cp.part.groups[g].ops, dag);
            if (!sa.is_bridge) continue;
            // Look for internal tensors that become boundary after split
            size_t split_tensor = SIZE_MAX;
            for (auto op : sa.side_a) {
                { size_t out = prob.ops[op].output();
                    // Check if any consumer is in side_b
                    for (auto cop : dag.tensor_consumers[out])
                        if (sa.side_b.count(cop)) { split_tensor = out; break; }
                }
                if (split_tensor != SIZE_MAX) break;
            }
            if (split_tensor == SIZE_MAX) continue;

            auto ev = eval_retain_force_split(cp, g, op_a, op_b, split_tensor);
            if (ev.feasible && ev.saving > -1e17) {
                auto aff = apply_retain_force_split(cp, g, op_a, op_b, split_tensor);
                if (!aff.empty()) {
                    cp.part.rebuild_index();
                    cp.part.rebuild_group_dag();
                }
            }

        } else if (choice <= 9) {
            // --- FORCE_RETAIN: split a consumer group to enable coupling ---
            // Pick a random boundary tensor, find producer (ga) and consumer (g_dst)
            // where COUPLE fails, then split g_dst
            std::vector<size_t> boundary_tensors;
            for (size_t t = 0; t < dag.tensor_producer.size(); t++) {
                int prod = dag.tensor_producer[t];
                if (prod < 0) continue;
                if (!dag.tensor_consumers[t].empty())
                    boundary_tensors.push_back(t);
            }
            if (boundary_tensors.empty()) continue;
            size_t t = boundary_tensors[rng() % boundary_tensors.size()];
            int prod_op = dag.tensor_producer[t];
            if (prod_op < 0) continue;

            auto& prod_groups = cp.part.groups_of((size_t)prod_op);
            if (prod_groups.empty()) continue;
            size_t ga = prod_groups[rng() % prod_groups.size()];
            if (!cp.part.groups[ga].alive) continue;
            // T must be produced in ga (boundary output or ephemeral).
            if (prod_op < 0 || !cp.part.groups[ga].ops.count((size_t)prod_op)) continue;

            // Find a consumer group that's too large for direct COUPLE
            auto& consumers = dag.tensor_consumers[t];
            if (consumers.empty()) continue;
            size_t cons_op = consumers[rng() % consumers.size()];
            auto& cons_groups = cp.part.groups_of(cons_op);
            if (cons_groups.empty()) continue;
            size_t g_dst = cons_groups[rng() % cons_groups.size()];
            if (!cp.part.groups[g_dst].alive || ga == g_dst) continue;
            if (cp.part.groups[g_dst].ops.size() < 3) continue;

            // Find a bridge edge in g_dst that separates the consumer from the rest
            auto bridges = structural_ops::bridge_edges(cp.part.groups[g_dst].ops, dag);
            if (bridges.empty()) continue;
            // Pick a bridge where cons_op is on one side
            for (auto& [ba, bb] : bridges) {
                auto sa = structural_ops::analyze_split(ba, bb,
                    cp.part.groups[g_dst].ops, dag);
                if (!sa.is_bridge) continue;
                // Determine which side has the consumer
                size_t op_a_dst, op_b_dst;
                if (sa.side_a.count(cons_op)) {
                    op_a_dst = ba; op_b_dst = bb;
                } else if (sa.side_b.count(cons_op)) {
                    op_a_dst = bb; op_b_dst = ba;
                } else continue;

                auto ev = eval_force_retain(cp, ga, g_dst, op_a_dst, op_b_dst, t);
                if (ev.feasible && ev.saving > -1e17) {
                    auto aff = apply_force_retain(cp, ga, g_dst, op_a_dst, op_b_dst, t);
                    if (!aff.empty()) {
                        cp.part.rebuild_index();
                        cp.part.rebuild_group_dag();
                    }
                }
                break;  // tried one bridge, move on
            }

        } else {
            // --- EPHEMERAL_FUSE: extract {P, C1} into a new group, couple T to C2 ---
            // Pick a random tensor with ≥2 consumers
            std::vector<size_t> candidate_tensors;
            for (size_t t = 0; t < dag.tensor_producer.size(); t++) {
                int prod = dag.tensor_producer[t];
                if (prod < 0) continue;
                if (dag.tensor_consumers[t].size() >= 2)
                    candidate_tensors.push_back(t);
            }
            if (candidate_tensors.empty()) continue;
            size_t t = candidate_tensors[rng() % candidate_tensors.size()];
            size_t op_p = (size_t)dag.tensor_producer[t];
            auto& consumers = dag.tensor_consumers[t];

            // Pick random c1 that's in the same group as producer
            std::vector<size_t> c1_candidates;
            for (auto c : consumers)
                for (auto g : cp.part.groups_of(op_p))
                    if (cp.part.groups[g].alive && cp.part.groups[g].ops.count(c))
                        { c1_candidates.push_back(c); break; }
            if (c1_candidates.empty()) continue;
            size_t c1 = c1_candidates[rng() % c1_candidates.size()];

            // Pick random c2 in a group that is a free chain head
            std::vector<std::pair<size_t, size_t>> c2_targets;  // (c2, g_c2)
            for (auto c2 : consumers) {
                if (c2 == c1) continue;
                for (auto g_c2 : cp.part.groups_of(c2)) {
                    if (!cp.part.groups[g_c2].alive) continue;
                    if (g_c2 >= cp.prev_group.size() || cp.prev_group[g_c2] != SIZE_MAX) continue;
                    c2_targets.push_back({c2, g_c2});
                }
            }
            if (c2_targets.empty()) continue;
            auto [c2, g_c2] = c2_targets[rng() % c2_targets.size()];

            auto ev = eval_ephemeral_fuse(cp, op_p, c1, g_c2, t);
            if (ev.feasible && ev.saving > -1e17) {
                auto aff = apply_ephemeral_fuse(cp, op_p, c1, g_c2, t);
                if (!aff.empty()) {
                    cp.part.rebuild_index();
                    cp.part.rebuild_group_dag();
                }
            }
        }
    }

    return cp;
}

Partition mutate_compound(Partition part, int num_mutations, std::mt19937& rng) {
    int applied  = 0;
    int attempts = 0;
    int consecutive_fails = 0;

    g_mut_total_requested.fetch_add(num_mutations, std::memory_order_relaxed);

    // --- Pre-filter: determine which mutation types can fire on this partition ---
    // 0=merge, 1=split, 2=reassign, 3=eject, 4=tensor_merge, 5=force_recompute
    const auto& dag = *part.dag;
    const auto& prob = *part.prob;

    auto compute_eligible = [&](const Partition& p) {
        std::vector<int> eligible;

        bool has_multi_op_group = false;
        bool has_large_group = false;  // ≥3 ops
        bool has_adjacent_pair = false;
        for (size_t gi = 0; gi < p.groups.size(); gi++) {
            if (!p.groups[gi].alive) continue;
            size_t sz = p.groups[gi].ops.size();
            if (sz >= 2) has_multi_op_group = true;
            if (sz >= 3) has_large_group = true;
            if (has_multi_op_group && has_adjacent_pair && has_large_group) break;
            if (!has_adjacent_pair) {
                auto nbrs = p.adjacent_groups(gi);
                for (auto gj : nbrs)
                    if (p.groups[gj].alive) { has_adjacent_pair = true; break; }
            }
        }

        if (has_adjacent_pair) eligible.push_back(0);  // merge
        if (has_large_group) eligible.push_back(1);    // split
        if (has_adjacent_pair) eligible.push_back(2);  // reassign
        if (has_multi_op_group) eligible.push_back(3); // eject

        // tensor_merge: any tensor with ≥2 consumers in different groups?
        bool has_tensor_merge_candidate = false;
        for (size_t t = 0; t < prob.num_tensors() && !has_tensor_merge_candidate; t++) {
            auto& consumers = dag.tensor_consumers[t];
            if (consumers.size() < 2) continue;
            size_t first_group = SIZE_MAX;
            for (auto cop : consumers) {
                auto& gs = p.groups_of(cop);
                for (auto gi : gs) {
                    if (first_group == SIZE_MAX) first_group = gi;
                    else if (gi != first_group) { has_tensor_merge_candidate = true; break; }
                }
                if (has_tensor_merge_candidate) break;
            }
        }
        if (has_tensor_merge_candidate) eligible.push_back(4);

        // force_recompute: any tensor with ≥2 consumers and a producer op?
        bool has_force_recompute_candidate = false;
        for (size_t t = 0; t < dag.tensor_producer.size() && !has_force_recompute_candidate; t++) {
            if (dag.tensor_producer[t] < 0) continue;
            if (dag.tensor_consumers[t].size() >= 2) has_force_recompute_candidate = true;
        }
        if (has_force_recompute_candidate) eligible.push_back(5);

        return eligible;
    };

    auto eligible = compute_eligible(part);
    if (eligible.empty()) return part;

    const int max_attempts = num_mutations * 5;
    const int max_consecutive_fails = std::max(num_mutations * 2, 10);

    while (applied < num_mutations && attempts < max_attempts
           && consecutive_fails < max_consecutive_fails) {
        attempts++;
        size_t before_groups = part.num_alive();
        double before_cost   = part.total_cost();

        int choice = eligible[rng() % eligible.size()];
        switch (choice) {
            case 0: part = mutate_merge(std::move(part), rng);            break;
            case 1: part = mutate_split(std::move(part), rng);            break;
            case 2: part = mutate_reassign(std::move(part), rng);         break;
            case 3: part = mutate_eject(std::move(part), rng);            break;
            case 4: part = mutate_tensor_merge(std::move(part), rng);     break;
            case 5: part = mutate_force_recompute(std::move(part), rng);  break;
        }

        g_mut_per_type_attempts[choice].fetch_add(1, std::memory_order_relaxed);
        bool did_apply = (part.num_alive() != before_groups || part.total_cost() != before_cost);
        if (did_apply) {
            applied++;
            consecutive_fails = 0;
            g_mut_per_type_applied[choice].fetch_add(1, std::memory_order_relaxed);

            // Recompute eligible types — partition structure changed
            eligible = compute_eligible(part);
            if (eligible.empty()) break;
        } else {
            consecutive_fails++;
        }

        // Verify coverage only after successful mutations (the only case
        // where ops can be lost).
        if (did_apply) {
            bool coverage_ok = true;
            for (size_t i = 0; i < prob.num_ops(); i++) {
                bool found = false;
                for (auto gi : part.groups_of(i))
                    if (part.groups[gi].alive) { found = true; break; }
                if (!found) { coverage_ok = false; break; }
            }
            if (!coverage_ok) break;
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

    g_mut_total_attempts.fetch_add(attempts, std::memory_order_relaxed);
    g_mut_total_applied.fetch_add(applied, std::memory_order_relaxed);

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
    std::map<std::pair<int,int>, FlatSet<size_t>> clusters;
    for (size_t i = 0; i < n_ops; i++) {
        if (grp_a[i] < 0 || grp_b[i] < 0) continue;
        clusters[{grp_a[i], grp_b[i]}].insert(i);
    }
    
    // Step 3: Shuffle clusters and greedily assign to groups.
    std::vector<std::pair<int,int>> cluster_keys;
    for (auto& [k, _] : clusters) cluster_keys.push_back(k);
    std::shuffle(cluster_keys.begin(), cluster_keys.end(), rng);

    g_xover_calls.fetch_add(1, std::memory_order_relaxed);
    g_xover_clusters.fetch_add(cluster_keys.size(), std::memory_order_relaxed);
    uint64_t local_evals = 0;
    
    Partition child;
    child.prob = &prob;
    child.dag = &dag;
    child.cache = parent_a.cache;
    child.rebuild_index();
    
    for (auto ck : cluster_keys) {
        auto& cluster = clusters[ck];
        
        // Find existing child groups adjacent to this cluster
        // Uses op_to_groups_ which is maintained incrementally below
        FlatSet<size_t> adj_child_groups;
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
            FlatSet<size_t> merged = child.groups[gi].ops;
            for (auto op : cluster) merged.insert(op);
            double c = child.eval_set(merged);
            local_evals++;
            if (c < 1e17 && c < best_cost) {
                best_cost = c;
                best_gi = gi;
            }
        }

        // Also evaluate the cluster as standalone
        double standalone = child.eval_set(cluster);
        local_evals++;
        
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
                local_evals++;
                child.add_group({op}, c);
            }
        }
    }

    g_xover_eval_set_calls.fetch_add(local_evals, std::memory_order_relaxed);

    // Single rebuild at the end for cleanup (removes dead entries, etc.)
    child.rebuild_index();

    // Safety: crossover can create cycles or mixed-consumer violations
    // when splicing clusters from two parents.
    if (partition_has_gap(child)) {
        g_xover_gap_rejects.fetch_add(1, std::memory_order_relaxed);
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