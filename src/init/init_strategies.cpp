#include "init/init_strategies.h"
#include "core/cost_cache.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <random>
#include <atomic>

// ============================================================================
// Helpers
// ============================================================================

// Map each op to its current alive group index. Assumes no overlap.
// Uses Partition's index if available, otherwise builds from scratch.
static std::vector<int> build_op_to_group(const Partition& p) {
    std::vector<int> m(p.prob->num_ops(), -1);
    for (size_t op = 0; op < p.prob->num_ops(); op++) {
        auto& gs = p.groups_of(op);
        if (!gs.empty()) m[op] = (int)gs[0];
    }
    return m;
}

static bool try_merge(Partition& p, size_t ga, size_t gb,
                      std::vector<int>& op_grp) {
    if (ga == gb || !p.groups[ga].alive || !p.groups[gb].alive) return false;

    if (p.dag->merge_creates_cycle(p.groups[ga].ops, p.groups[gb].ops)) return false;

    FlatSet<size_t> merged = p.groups[ga].ops;
    merged.insert(p.groups[gb].ops.begin(), p.groups[gb].ops.end());

    if (p.creates_ephemeral_gap(merged, ga, gb)) return false;

    double new_cost = p.eval_set(merged);

    if (new_cost < p.groups[ga].cost + p.groups[gb].cost - 0.01) {
        p.groups[ga].ops = std::move(merged);
        p.groups[ga].cost = new_cost;
        p.groups[ga].gen++;
        p.groups[gb].alive = false;
        p.groups[gb].gen++;
        p.rebuild_index();
        // Incremental op_grp update: all ops in the merged group now map to ga.
        for (auto op : p.groups[ga].ops)
            op_grp[op] = (int)ga;
        return true;
    }
    return false;
}

// ============================================================================
// Trivial: one op per group
// ============================================================================

Partition init_trivial(const Problem& prob, const DAG& dag, CostCache* cache) {
    auto p = Partition::trivial(prob, dag);
    p.cache = cache;
    return p;
}

// ============================================================================
// Chain + edge greedy
//
// Phase A: detect maximal chains (op with single successor whose single
//          predecessor is that op). Fuse entire chains.
// Phase B: sort intermediate tensors by size descending. For each, try
//          merging producer and consumer groups.
// ============================================================================

Partition init_chain_then_edge(const Problem& prob, const DAG& dag, CostCache* cache) {
    auto p = Partition::trivial(prob, dag);
    p.cache = cache;
    auto op_grp = build_op_to_group(p);

    // Phase A: chain detection
    std::vector<bool> in_chain(prob.num_ops(), false);
    for (size_t i = 0; i < prob.num_ops(); i++) {
        if (in_chain[i]) continue;
        std::vector<size_t> chain = {i};
        size_t cur = i;
        while (true) {
            if (dag.op_succs[cur].size() != 1) break;
            size_t next = *dag.op_succs[cur].begin();
            if (dag.op_preds[next].size() != 1) break;
            if (in_chain[next]) break;
            chain.push_back(next);
            cur = next;
        }
        if (chain.size() > 1) {
            for (size_t j = 1; j < chain.size(); j++) {
                size_t ga = op_grp[chain[0]];
                size_t gb = op_grp[chain[j]];
                if (try_merge(p, ga, gb, op_grp)) {}
            }
            for (auto op : chain) in_chain[op] = true;
        }
    }

    // Phase B: edge greedy by tensor size
    struct Edge {
        size_t tensor, producer, consumer;
        int64_t size;
    };
    std::vector<Edge> edges;
    for (size_t t = 0; t < prob.tensors.size(); t++) {
        int prod = dag.tensor_producer[t];
        if (prod < 0) continue;
        for (auto c : dag.tensor_consumers[t]) {
            edges.push_back({t, (size_t)prod, c,
                prob.tensors[t].width * prob.tensors[t].height});
        }
    }
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return a.size > b.size; });

    for (auto& e : edges) {
        int ga = op_grp[e.producer];
        int gb = op_grp[e.consumer];
        if (ga < 0 || gb < 0) continue;
        try_merge(p, (size_t)ga, (size_t)gb, op_grp);
    }

    // Phase C: co-consumer merging by shared tensor size
    struct SharedTensor {
        size_t tensor;
        int64_t size;
    };
    std::vector<SharedTensor> shared_tensors;
    for (size_t t = 0; t < prob.tensors.size(); t++) {
        if (dag.tensor_consumers[t].size() > 1) {
            shared_tensors.push_back({t,
                prob.tensors[t].width * prob.tensors[t].height});
        }
    }
    std::sort(shared_tensors.begin(), shared_tensors.end(),
              [](const SharedTensor& a, const SharedTensor& b) {
                  return a.size > b.size;
              });

    for (auto& st : shared_tensors) {
        auto& consumers = dag.tensor_consumers[st.tensor];
        for (size_t i = 0; i < consumers.size(); i++) {
            for (size_t j = i + 1; j < consumers.size(); j++) {
                int ga = op_grp[consumers[i]];
                int gb = op_grp[consumers[j]];
                if (ga < 0 || gb < 0 || ga == gb) continue;
                try_merge(p, (size_t)ga, (size_t)gb, op_grp);
            }
        }
    }

    return p;
}

// ============================================================================
// Seed-and-grow
//
// Sort ops by base_cost descending. For each unclaimed op, start a new
// group and repeatedly add the unassigned neighbor giving the biggest
// cost reduction. Stop when no neighbor improves.
// ============================================================================

Partition init_seed_and_grow(const Problem& prob, const DAG& dag, CostCache* cache) {
    Partition p;
    p.prob = &prob;
    p.dag = &dag;
    p.cache = cache;

    std::vector<size_t> order(prob.num_ops());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return prob.ops[a].base_cost > prob.ops[b].base_cost;
    });

    std::vector<bool> assigned(prob.num_ops(), false);

    for (auto seed : order) {
        if (assigned[seed]) continue;

        FlatSet<size_t> group = {seed};
        double group_cost = p.eval_set(group);
        assigned[seed] = true;

        bool grew = true;
        while (grew) {
            grew = false;
            size_t best_op = SIZE_MAX;
            double best_cost = group_cost;

            FlatSet<size_t> neighbors;
            for (auto op : group) {
                for (auto v : dag.op_neighbors[op])
                    if (!group.count(v) && !assigned[v])
                        neighbors.insert(v);
            }

            for (auto cand : neighbors) {
                if (p.dag->merge_creates_cycle({cand}, group)) continue;

                // Insert candidate in place, evaluate, then erase (avoids
                // heap-allocating a copy of the entire set for each candidate).
                group.insert(cand);

                // Ephemeral gap check: T is ephemeral if produced AND consumed
                // inside group. If T also has external consumers, those are
                // stranded (T never in slow memory). During init, no
                // recomputation exists, so this is always invalid.
                bool has_gap = false;
                for (auto op_i : group) {
                    {
                        size_t t = prob.ops[op_i].output();
                        bool any_in = false;
                        bool any_out = false;
                        for (auto cop : dag.tensor_consumers[t]) {
                            if (group.count(cop)) any_in = true;
                            else any_out = true;
                        }
                        if (any_in && any_out) {
                            int prod_op_t = dag.tensor_producer[t];
                            if (prod_op_t >= 0) {
                                for (auto cop : dag.tensor_consumers[t]) {
                                    if (has_gap) break;
                                    if (group.count(cop)) continue;
                                    bool served = false;
                                    for (auto gj : p.groups_of(cop)) {
                                        if (!p.groups[gj].alive) continue;
                                        if (p.groups[gj].ops.count((size_t)prod_op_t))
                                            { served = true; break; }
                                    }
                                    if (!served) has_gap = true;
                                }
                            }
                        }
                    }
                    if (has_gap) break;
                }

                double cost = has_gap ? 1e18 : p.eval_set(group);
                if (!has_gap && cost < best_cost - 0.01) {
                    best_cost = cost;
                    best_op = cand;
                }

                group.erase(cand);  // backtrack
            }

            if (best_op != SIZE_MAX) {
                group.insert(best_op);
                group_cost = best_cost;
                assigned[best_op] = true;
                grew = true;
            }
        }

        p.groups.push_back({group, group_cost, true, 0});
    }

    for (size_t i = 0; i < prob.num_ops(); i++) {
        if (!assigned[i]) {
            double cost = p.eval_set({i});
            p.groups.push_back({{i}, cost, true, 0});
        }
    }

    p.rebuild_index();
    return p;
}

// ============================================================================
// Reverse topological greedy
//
// Process ops in reverse topo order. For each, try merging into the
// group of each successor. Accept the merge that saves the most.
// ============================================================================

Partition init_reverse_topo(const Problem& prob, const DAG& dag, CostCache* cache) {
    auto p = Partition::trivial(prob, dag);
    p.cache = cache;
    auto op_grp = build_op_to_group(p);

    auto topo = dag.topo_sort();
    std::reverse(topo.begin(), topo.end());

    for (auto op : topo) {
        int gi = op_grp[op];
        if (gi < 0) continue;

        size_t best_gj = SIZE_MAX;
        double best_saving = 0;

        // Collect candidate groups from all neighbors (DAG edges + shared inputs).
        // Use a flat vector + sort/unique — typically only 1-5 entries.
        std::vector<int> candidate_groups;
        for (auto v : dag.op_neighbors[op]) {
            int gj = op_grp[v];
            if (gj >= 0 && (size_t)gj != (size_t)gi)
                candidate_groups.push_back(gj);
        }
        std::sort(candidate_groups.begin(), candidate_groups.end());
        candidate_groups.erase(std::unique(candidate_groups.begin(),
                                            candidate_groups.end()),
                               candidate_groups.end());

        for (int gj : candidate_groups) {

            if (p.dag->merge_creates_cycle(p.groups[gi].ops, p.groups[gj].ops)) continue;

            FlatSet<size_t> merged = p.groups[gi].ops;
            merged.insert(p.groups[gj].ops.begin(), p.groups[gj].ops.end());

            if (p.creates_ephemeral_gap(merged, (size_t)gi, (size_t)gj)) continue;

            double new_cost = p.eval_set(merged);
            double saving = (p.groups[gi].cost + p.groups[gj].cost) - new_cost;

            if (saving > best_saving + 0.01) {
                best_saving = saving;
                best_gj = (size_t)gj;
            }
        }

        if (best_gj != SIZE_MAX) {
            // Inline the merge directly — we already verified cycle/gap/cost above.
            // try_merge would re-evaluate (cache hit but redundant).
            FlatSet<size_t> merged = p.groups[gi].ops;
            merged.insert(p.groups[best_gj].ops.begin(), p.groups[best_gj].ops.end());
            double new_cost = p.eval_set(merged);  // cache hit from above
            if (new_cost < p.groups[gi].cost + p.groups[best_gj].cost - 0.01) {
                p.groups[gi].ops = std::move(merged);
                p.groups[gi].cost = new_cost;
                p.groups[gi].gen++;
                p.groups[best_gj].alive = false;
                p.groups[best_gj].gen++;
                p.rebuild_index();
                // Incremental op_grp update
                for (auto o : p.groups[gi].ops) op_grp[o] = (int)gi;
            }
        }
    }

    return p;
}

// ============================================================================
// Random: start from singletons, randomly merge along DAG edges
// ============================================================================

Partition init_random(const Problem& prob, const DAG& dag, CostCache* cache) {
    static std::atomic<int> call_counter{0};
    int seed = 12345 + call_counter.fetch_add(1) * 77;
    std::mt19937 rng(seed);

    Partition p = Partition::trivial(prob, dag);
    p.cache = cache;

    // Collect all DAG edges: (producer_op, consumer_op) via tensors.
    // Use dag.tensor_consumers[t] directly — O(E) not O(E*V).
    struct Edge { size_t op_a; size_t op_b; size_t tensor; };
    std::vector<Edge> edges;
    for (size_t t = 0; t < prob.num_tensors(); t++) {
        int prod = dag.tensor_producer[t];
        if (prod < 0) continue;
        for (auto consumer : dag.tensor_consumers[t])
            edges.push_back({(size_t)prod, consumer, t});
    }

    // Shuffle edges and try random merges
    std::shuffle(edges.begin(), edges.end(), rng);

    // Try merging a random fraction of edges (30-70%)
    std::uniform_real_distribution<double> frac_dist(0.3, 0.7);
    int target_merges = (int)(edges.size() * frac_dist(rng));
    int merges_done = 0;

    for (auto& e : edges) {
        if (merges_done >= target_merges) break;

        auto& gs_a = p.groups_of(e.op_a);
        auto& gs_b = p.groups_of(e.op_b);
        if (gs_a.empty() || gs_b.empty()) continue;
        size_t ga = gs_a[0], gb = gs_b[0];
        if (ga == gb) continue; // already same group
        if (!p.groups[ga].alive || !p.groups[gb].alive) continue;

        // Check cycle
        if (dag.merge_creates_cycle(p.groups[ga].ops, p.groups[gb].ops))
            continue;

        // Try merge — accept if subgraph is valid (feasible tiling exists)
        FlatSet<size_t> merged = p.groups[ga].ops;
        merged.insert(p.groups[gb].ops.begin(), p.groups[gb].ops.end());
        if (p.creates_ephemeral_gap(merged, ga, gb)) continue;
        double new_cost = p.eval_set(merged);
        if (new_cost >= 1e17) continue; // infeasible

        // Accept unconditionally (random strategy — cost doesn't matter)
        p.groups[ga].ops = std::move(merged);
        p.groups[ga].cost = new_cost;
        p.groups[ga].gen++;
        p.groups[gb].alive = false;
        p.groups[gb].gen++;
        p.rebuild_index();
        merges_done++;
    }

    return p;
}

// ============================================================================
// Tensor-aligned: group ops that share input tensors.
//
// For each tensor T (sorted by size descending), try to merge all consumers
// of T into one group.  If the producer of T is a separate group, try to
// merge it in too.  This produces groups aligned around shared data,
// maximizing internal tensor reuse (ephemeral classification).
// ============================================================================

Partition init_tensor_aligned(const Problem& prob, const DAG& dag,
                               CostCache* cache) {
    Partition p = Partition::trivial(prob, dag);
    p.cache = cache;

    // Sort tensors by size (largest first — merge around big shared tensors)
    std::vector<size_t> tensor_order(prob.num_tensors());
    std::iota(tensor_order.begin(), tensor_order.end(), 0);
    std::sort(tensor_order.begin(), tensor_order.end(),
        [&](size_t a, size_t b) {
            int64_t sa = prob.tensors[a].width * prob.tensors[a].height;
            int64_t sb = prob.tensors[b].width * prob.tensors[b].height;
            return sa > sb;
        });

    for (auto t : tensor_order) {
        const auto& consumers = dag.tensor_consumers[t];
        if (consumers.size() < 2) continue;
        int prod = dag.tensor_producer[t];

        // Find the group of the first alive consumer
        size_t target_gi = SIZE_MAX;
        for (auto cop : consumers) {
            for (auto gi : p.groups_of(cop))
                if (p.groups[gi].alive) { target_gi = gi; break; }
            if (target_gi != SIZE_MAX) break;
        }
        if (target_gi == SIZE_MAX) continue;

        // Try to merge other consumers' groups into target
        for (auto cop : consumers) {
            for (auto gi : p.groups_of(cop)) {
                if (gi == target_gi || !p.groups[gi].alive) continue;
                FlatSet<size_t> merged = p.groups[target_gi].ops;
                merged.insert(p.groups[gi].ops.begin(), p.groups[gi].ops.end());
                if (p.creates_ephemeral_gap(merged, target_gi, gi)) continue;
                if (!p.acyclic_merge_local(target_gi, gi)) continue;
                double cost = p.eval_set(merged);
                if (cost >= 1e17) continue;
                p.groups[target_gi].ops = std::move(merged);
                p.groups[target_gi].cost = cost;
                p.groups[target_gi].gen++;
                p.groups[gi].alive = false;
                p.groups[gi].gen++;
                p.rebuild_index();
                break;  // gi invalidated, restart consumer loop
            }
        }

        // Try to merge producer into the target group
        if (prod >= 0) {
            for (auto gi : p.groups_of((size_t)prod)) {
                if (gi == target_gi || !p.groups[gi].alive) continue;
                FlatSet<size_t> merged = p.groups[target_gi].ops;
                merged.insert(p.groups[gi].ops.begin(), p.groups[gi].ops.end());
                if (p.creates_ephemeral_gap(merged, target_gi, gi)) continue;
                if (!p.acyclic_merge_local(target_gi, gi)) continue;
                double cost = p.eval_set(merged);
                if (cost >= 1e17) continue;
                p.groups[target_gi].ops = std::move(merged);
                p.groups[target_gi].cost = cost;
                p.groups[target_gi].gen++;
                p.groups[gi].alive = false;
                p.groups[gi].gen++;
                p.rebuild_index();
                break;
            }
        }
    }

    return p;
}

// ============================================================================
// Registry and selection
// ============================================================================

std::vector<InitStrategy> all_init_strategies() {
    return {
        {"trivial",       init_trivial},
        {"chain+edge",    init_chain_then_edge},
        {"seed+grow",     init_seed_and_grow},
        {"rev-topo",      init_reverse_topo},
        {"tensor-align",  init_tensor_aligned},
        {"random",        init_random},
    };
}

// Run every strategy once and return the lowest-cost partition.
// All strategies share a single CostCache so evaluations from one strategy
// benefit subsequent ones, and the warm cache carries into Phase 1 FM search
// when the caller reuses the same CostCache instance.
Partition best_initial(const Problem& prob, const DAG& dag, CostCache* cache) {
    auto strategies = all_init_strategies();

    size_t best_idx = 0;
    std::vector<Partition> results;
    for (auto& s : strategies) {
        results.push_back(s.init(prob, dag, cache));
        std::cerr << "    " << s.name << ": "
                  << results.back().num_alive() << " groups, cost="
                  << results.back().total_cost() << "\n";
        if (results.back().total_cost() < results[best_idx].total_cost())
            best_idx = results.size() - 1;
    }
    std::cerr << "    -> using " << strategies[best_idx].name << "\n";

    return std::move(results[best_idx]);
}
// ============================================================================
// Feasibility validator
// ============================================================================

std::string verify_partition_feasibility(const Partition& part) {
    const Problem& prob = *part.prob;
    const DAG&     dag  = *part.dag;

    // ── 0. Coverage: every op must be in at least one alive group ─────
    for (size_t op = 0; op < prob.num_ops(); op++) {
        bool covered = false;
        for (auto gi : part.groups_of(op))
            if (part.groups[gi].alive) { covered = true; break; }
        if (!covered)
            return "Op " + std::to_string(op) + " not covered by any alive group";
    }

    // ── 1. Memory feasibility: every alive group must have a valid tiling ──
    for (size_t i = 0; i < part.groups.size(); i++) {
        if (!part.groups[i].alive) continue;
        const auto& g = part.groups[i];
        // eval_set returns 1e18 if no feasible tiling exists.
        // Re-evaluate directly via Subgraph to be independent of cache state.
        auto sg = Subgraph::create(prob, dag,
                      std::vector<size_t>(g.ops.begin(), g.ops.end()));
        if (!sg) {
            return "G" + std::to_string(i) + ": Subgraph::create failed "
                   "(disconnected or invalid structure)";
        }
        auto c = sg->best_cost();
        if (!c.feasible) {
            return "G" + std::to_string(i) + ": no feasible tiling "
                   "(working set exceeds fast_memory_capacity for all candidates)";
        }
    }

    // ── 2. No cycles in the condensed group DAG ───────────────────────────
    if (!part.is_acyclic())
        return "Partition has a cycle in the condensed group DAG";

    // ── 3. No mixed-consumer violations (ephemeral gap) ─────────────────
    // A tensor produced AND consumed inside a group is ephemeral (never in
    // slow memory). Any external consumer must either:
    //   (a) have the producer recomputed in its own group, OR
    //   (b) another alive group exports the tensor as a boundary output
    //       (produced but not consumed internally → written to slow memory).
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        for (auto op : part.groups[gi].ops) {
            {
                size_t t = prob.ops[op].output();
                bool consumed_internal = false;
                for (auto cop : dag.tensor_consumers[t])
                    if (cop != op && part.groups[gi].ops.count(cop))
                        { consumed_internal = true; break; }
                if (!consumed_internal) continue;

                for (auto cop : dag.tensor_consumers[t]) {
                    if (part.groups[gi].ops.count(cop)) continue;
                    bool covered = false;

                    // (a) consumer's group recomputes the producer
                    for (auto gj : part.groups_of(cop)) {
                        if (!part.groups[gj].alive) continue;
                        if (part.groups[gj].ops.count(op)) { covered = true; break; }
                    }

                    // (b) another group exports T as boundary output
                    if (!covered) {
                        for (auto gj : part.groups_of(op)) {
                            if (gj == gi || !part.groups[gj].alive) continue;
                            // T is boundary output of gj if gj produces T
                            // but doesn't consume T internally
                            bool consumed_in_gj = false;
                            for (auto c2 : dag.tensor_consumers[t])
                                if (part.groups[gj].ops.count(c2))
                                    { consumed_in_gj = true; break; }
                            if (!consumed_in_gj) { covered = true; break; }
                        }
                    }

                    if (!covered)
                        return "G" + std::to_string(gi) + ": T" + std::to_string(t)
                             + " is ephemeral but has uncovered external consumer op"
                             + std::to_string(cop);
                }
            }
        }
    }

    return "";  // all invariants satisfied
}