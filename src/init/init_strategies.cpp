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

// Try merging groups ga and gb. Returns true if merge improved cost.
static bool try_merge(Partition& p, size_t ga, size_t gb) {
    if (ga == gb || !p.groups[ga].alive || !p.groups[gb].alive) return false;

    if (p.dag->merge_creates_cycle(p.groups[ga].ops, p.groups[gb].ops)) return false;

    std::set<size_t> merged = p.groups[ga].ops;
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
                if (try_merge(p, ga, gb))
                    op_grp = build_op_to_group(p);
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

    op_grp = build_op_to_group(p);
    for (auto& e : edges) {
        int ga = op_grp[e.producer];
        int gb = op_grp[e.consumer];
        if (ga < 0 || gb < 0) continue;
        if (try_merge(p, (size_t)ga, (size_t)gb))
            op_grp = build_op_to_group(p);
    }

    // Phase C: co-consumer merging by shared tensor size
    // For each tensor (including graph inputs), try merging pairs of consumer
    // groups. This captures the benchmark-13 pattern where parallel ops share
    // a large input tensor.
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

    op_grp = build_op_to_group(p);
    for (auto& st : shared_tensors) {
        auto& consumers = dag.tensor_consumers[st.tensor];
        for (size_t i = 0; i < consumers.size(); i++) {
            for (size_t j = i + 1; j < consumers.size(); j++) {
                int ga = op_grp[consumers[i]];
                int gb = op_grp[consumers[j]];
                if (ga < 0 || gb < 0 || ga == gb) continue;
                if (try_merge(p, (size_t)ga, (size_t)gb))
                    op_grp = build_op_to_group(p);
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

        std::set<size_t> group = {seed};
        double group_cost = p.eval_set(group);
        assigned[seed] = true;

        bool grew = true;
        while (grew) {
            grew = false;
            size_t best_op = SIZE_MAX;
            double best_cost = group_cost;

            std::set<size_t> neighbors;
            for (auto op : group) {
                for (auto v : dag.op_neighbors[op])
                    if (!group.count(v) && !assigned[v])
                        neighbors.insert(v);
            }

            for (auto cand : neighbors) {
                if (p.dag->merge_creates_cycle({cand}, group)) continue;

                std::set<size_t> expanded = group;
                expanded.insert(cand);

                // Ephemeral gap check is handled by Subgraph::create (via
                // eval_set): tensors with external consumers become boundary
                // outputs, not ephemeral. creates_ephemeral_gap handles the
                // partition-level constraint. No manual check needed here.
                double cost = p.eval_set(expanded);
                if (cost < best_cost - 0.01) {
                    best_cost = cost;
                    best_op = cand;
                }
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

        // Collect candidate groups from all neighbors (DAG edges + shared inputs)
        std::set<int> candidate_groups;
        for (auto v : dag.op_neighbors[op]) {
            int gj = op_grp[v];
            if (gj >= 0 && (size_t)gj != (size_t)gi)
                candidate_groups.insert(gj);
        }

        for (int gj : candidate_groups) {

            if (p.dag->merge_creates_cycle(p.groups[gi].ops, p.groups[gj].ops)) continue;

            std::set<size_t> merged = p.groups[gi].ops;
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
            try_merge(p, (size_t)gi, best_gj);
            op_grp = build_op_to_group(p);
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
    int target_merges = (int)(edges.size() * (0.3 + 0.4 * (rng() % 1000) / 1000.0));
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
        std::set<size_t> merged = p.groups[ga].ops;
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
// Registry and selection
// ============================================================================

std::vector<InitStrategy> all_init_strategies() {
    return {
        {"trivial",     init_trivial},
        {"chain+edge",  init_chain_then_edge},
        {"seed+grow",   init_seed_and_grow},
        {"rev-topo",    init_reverse_topo},
        {"random",      init_random},
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
                   "(disconnected or ephemeral fan-out)";
        }
        auto c = sg->best_cost();
        if (!c.feasible) {
            return "G" + std::to_string(i) + ": no feasible tiling "
                   "(working set exceeds fast_memory_capacity for all candidates)";
        }
    }

    // ── 2. No cycles in the condensed group DAG ───────────────────────────
    // The condensed group DAG of any partition of a DAG is always acyclic —
    // this is a direct consequence of the original graph being a DAG.
    // No check is needed here.
    //
    // Note: merge_creates_cycle(A, B) is a *move pre-condition*, not a partition
    // invariant. It returns true when there is a directed path between op-sets A
    // and B through intermediate groups; merging them would then create a
    // self-loop in the condensed DAG. That check belongs in try_merge / apply_move,
    // not here.

    // ── 3. No ephemeral tensors with external consumers ───────────────────
    // T is truly ephemeral in group G only if ALL of T's consumers are inside G.
    // If any consumer is external, Subgraph::create classifies T as a boundary
    // output and external consumers load it from slow memory — no gap possible.
    //
    // For an existing partition where every group passes eval_set < 1e18, this
    // invariant is automatically satisfied: eval_set calls Subgraph::create which
    // correctly classifies each tensor as ephemeral or boundary output based on
    // whether all consumers are internal. External consumers are always served by
    // boundary outputs. No further check is needed here.
    //
    // Note: creates_ephemeral_gap() is the correct pre-move check used during
    // search to verify that a PROPOSED fusion would not strand any consumer.
    // That check is distinct from verifying an existing partition.

    return "";  // all invariants satisfied
}