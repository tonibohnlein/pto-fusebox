#include "partition/partition.h"
#include "core/cost_cache.h"
#include "search/feasibility.h"
#include "search/structural_ops.h"
#include <iostream>
#include <queue>
#include <functional>
#include <deque>
#include <algorithm>

// ============================================================================
// finalize()
// ============================================================================

void Partition::finalize(CostCache* ext_cache) {
    rebuild_index();

    for (size_t gi = 0; gi < groups.size(); gi++) {
        auto& g = groups[gi];
        if (!g.alive) continue;
        g.sg = std::nullopt;

        auto sg_opt = Subgraph::create(*prob, *dag,
                          std::vector<size_t>(g.ops.begin(), g.ops.end()));
        if (sg_opt) {
            CostResult c;
            if (ext_cache)
                c = ext_cache->evaluate_with_context(*sg_opt, {}, {});
            else
                c = sg_opt->best_cost();
            if (c.feasible) {
                g.cost     = c.latency;
                g.best_cfg = c.config;
            } else {
                g.cost = 1e18;
            }
            g.sg = std::move(*sg_opt);
        } else {
            g.cost = 1e18;
            std::cerr << "    WARNING: finalize: Subgraph::create failed for group with "
                      << g.ops.size() << " ops:";
            for (auto op : g.ops) std::cerr << " " << op;
            std::cerr << "\n";
        }
    }

    rebuild_group_dag();
}

void Partition::rebuild_index() {
    op_to_groups_.assign(prob->num_ops(), {});
    for (size_t i = 0; i < groups.size(); i++)
        if (groups[i].alive)
            for (auto op : groups[i].ops)
                op_to_groups_[op].push_back(i);
    invalidate_fwd_cache();
}

const std::vector<bool>& Partition::cached_forward_reachable(size_t g) const {
    size_t ng = groups.size();
    if (fwd_cache_.size() < ng) {
        fwd_cache_.resize(ng);
        fwd_cache_gen_.resize(ng, 0);
    }
    if (fwd_cache_gen_[g] == fwd_gen_ && !fwd_cache_[g].empty())
        return fwd_cache_[g];

    // Compute via BFS
    fwd_cache_[g].assign(ng, false);
    std::queue<size_t> q;
    fwd_cache_[g][g] = true;
    q.push(g);
    while (!q.empty()) {
        size_t cur = q.front(); q.pop();
        for (auto op : groups[cur].ops) {
            for (auto t : prob->ops[op].outputs) {
                for (auto cop : dag->tensor_consumers[t]) {
                    if (groups[cur].ops.count(cop)) continue;
                    for (auto gj : groups_of(cop)) {
                        if (!groups[gj].alive || fwd_cache_[g][gj]) continue;
                        fwd_cache_[g][gj] = true;
                        q.push(gj);
                    }
                }
            }
        }
    }
    fwd_cache_gen_[g] = fwd_gen_;
    return fwd_cache_[g];
}

void Partition::rebuild_group_dag() {
    size_t ng = groups.size();
    group_preds.assign(ng, {});
    group_succs.assign(ng, {});
    group_in_deg.assign(ng, 0);
    tensor_to_group.clear();

    for (size_t i = 0; i < ng; i++) {
        if (!groups[i].alive) continue;
        for (size_t t = 0; t < dag->tensor_producer.size(); t++)
            if (is_boundary_output_of(groups[i].ops, t, *dag))
                tensor_to_group[t] = i;
    }

    // Step 1: Get a valid topological order using OP-BASED OR-node Kahn's.
    // Used to disambiguate which source group to draw an edge from when a
    // producer op is in multiple alive groups (recomputation).
    std::vector<int> topo_pos(ng, -1);
    {
        std::vector<int> unsatisfied(ng, 0);
        std::vector<bool> dep_met;
        std::vector<std::vector<std::pair<size_t, size_t>>> frees(ng);

        for (size_t op_idx = 0; op_idx < prob->ops.size(); op_idx++) {
            for (auto t : prob->ops[op_idx].inputs) {
                int prod = dag->tensor_producer[t];
                if (prod < 0) continue;
                for (auto target_gi : op_to_groups_[op_idx]) {
                    if (!groups[target_gi].alive) continue;
                    bool prod_internal = false;
                    for (auto g : op_to_groups_[(size_t)prod])
                        if (g == target_gi) { prod_internal = true; break; }
                    if (prod_internal) continue;

                    // Check if ANY alive source exists for this dep.
                    // If not, the dep is permanently unsatisfied — skip it
                    // to avoid poisoning the Kahn's traversal.
                    bool has_source = false;
                    for (auto source_gj : op_to_groups_[(size_t)prod])
                        if (groups[source_gj].alive) { has_source = true; break; }
                    if (!has_source) continue;

                    size_t dep_id = dep_met.size();
                    dep_met.push_back(false);
                    unsatisfied[target_gi]++;
                    for (auto source_gj : op_to_groups_[(size_t)prod]) {
                        if (groups[source_gj].alive)
                            frees[source_gj].push_back({target_gi, dep_id});
                    }
                }
            }
        }

        std::vector<size_t> q;
        q.reserve(ng);
        std::vector<bool> enqueued(ng, false);
        for (size_t i = 0; i < ng; i++)
            if (groups[i].alive && unsatisfied[i] == 0) {
                q.push_back(i);
                enqueued[i] = true;
            }
        int pos = 0;
        size_t head = 0;
        while (head < q.size()) {
            size_t u = q[head++];
            topo_pos[u] = pos++;
            for (auto [gi, dep_id] : frees[u]) {
                if (!dep_met[dep_id]) {
                    dep_met[dep_id] = true;
                    unsatisfied[gi]--;
                    if (unsatisfied[gi] == 0 && !enqueued[gi]) {
                        q.push_back(gi);
                        enqueued[gi] = true;
                    }
                }
            }
        }

    }

    // Step 2: Build group DAG edges.
    //
    // For each group i, for each op's input tensor T, if T's producer is
    // external to i, find the best source group and draw an edge.
    //
    // Source priority (best to worst):
    //   1. Kahn's topo_pos (available when Step 1 resolves the group)
    //   2. DAG-level topo position of the producer op within the group
    //      (always available — the op DAG is guaranteed acyclic)
    //   3. Arbitrary (smallest group index)
    for (size_t i = 0; i < ng; i++) {
        if (!groups[i].alive) continue;
        for (auto op : groups[i].ops) {
            for (auto t : prob->ops[op].inputs) {
                int prod = dag->tensor_producer[t];
                if (prod < 0) continue;
                if (groups[i].ops.count((size_t)prod)) continue;  // internal

                size_t best_gj = SIZE_MAX;
                int best_pos = INT_MAX;
                for (auto gj : op_to_groups_[(size_t)prod]) {
                    if (!groups[gj].alive || gj == i) continue;
                    // Use Kahn's topo_pos if available; otherwise fall back
                    // to DAG topo position of the producer op (deterministic,
                    // always valid). The offset by ng ensures resolved groups
                    // are always preferred over unresolved ones.
                    int pos;
                    if (topo_pos[gj] >= 0) {
                        pos = topo_pos[gj];
                    } else {
                        pos = (int)ng + (int)dag->topo_position((size_t)prod);
                    }
                    if (pos < best_pos) {
                        best_gj = gj;
                        best_pos = pos;
                    }
                }
                if (best_gj != SIZE_MAX) {
                    if (group_preds[i].insert(best_gj).second)
                        group_succs[best_gj].insert(i);
                }
            }
        }
    }

    for (size_t i = 0; i < ng; i++)
        group_in_deg[i] = (int)group_preds[i].size();

#ifndef NDEBUG
    // Validation: for every exported tensor, check that consuming groups have
    // the corresponding DAG edge.  Uses tensor_to_group (freshly built above)
    // and is_boundary_input_of (DAG-based) to avoid stale .sg reads.
    for (auto& [t, exporting_gj] : tensor_to_group) {
        int prod = (int)dag->tensor_producer[t];
        for (size_t i = 0; i < ng; i++) {
            if (!groups[i].alive || i == exporting_gj) continue;
            if (!is_boundary_input_of(groups[i].ops, t, *dag)) continue;
            if (!group_preds[i].count(exporting_gj)) {
                std::cerr << "  EDGE_MISSING: G" << i << " needs T" << t
                          << " (prod=op" << prod << "), G" << exporting_gj
                          << " exports it, but no edge G" << exporting_gj
                          << "→G" << i << " exists.\n";
            }
        }
    }
#endif
}


// ============================================================================
// Construction
// ============================================================================

Partition Partition::trivial(const Problem& prob, const DAG& dag) {
    Partition p;
    p.prob = &prob;
    p.dag  = &dag;
    p.groups.resize(prob.num_ops());

    for (size_t i = 0; i < prob.num_ops(); i++) {
        p.groups[i].ops   = {i};
        p.groups[i].alive = true;
        // Compute cost only — sg/best_cfg populated later by finalize()
        // when the partition is actually used for solution building.
        // This avoids Subgraph::create overhead for strategies that
        // immediately mutate the partition (chain+edge, seed+grow, etc.).
        p.groups[i].cost = p.eval_set({i});
    }

    p.rebuild_index();
    p.rebuild_group_dag();
    return p;
}

// ============================================================================
// Queries
// ============================================================================

double Partition::total_cost() const {
    double t = 0;
    for (auto& g : groups) if (g.alive) t += g.cost;
    return t;
}

size_t Partition::num_alive() const {
    size_t n = 0;
    for (auto& g : groups) if (g.alive) n++;
    return n;
}

bool Partition::is_border_op(size_t op, size_t gi) const {
    for (auto v : dag->op_neighbors[op])
        if (!groups[gi].ops.count(v)) return true;
    return false;
}

std::vector<size_t> Partition::border_ops(size_t gi) const {
    std::vector<size_t> result;
    for (auto op : groups[gi].ops)
        if (is_border_op(op, gi))
            result.push_back(op);
    return result;
}

FlatSet<size_t> Partition::boundary_neighbors(size_t gi) const {
    FlatSet<size_t> result;
    for (auto op : groups[gi].ops)
        for (auto v : dag->op_neighbors[op])
            if (!groups[gi].ops.count(v))
                result.insert(v);
    return result;
}

FlatSet<size_t> Partition::adjacent_groups(size_t gi) const {
    FlatSet<size_t> result;
    for (auto op : boundary_neighbors(gi))
        for (auto gj : op_to_groups_[op])
            if (gj != gi) result.insert(gj);
    return result;
}

bool Partition::future_needs(size_t t, const std::vector<bool>& scheduled) const {
    for (size_t i = 0; i < groups.size(); i++) {
        if (!groups[i].alive || scheduled[i]) continue;
        if (groups[i].sg && groups[i].sg->boundary_inputs().count(t)) return true;
    }
    return false;
}

// ============================================================================
// is_acyclic — reference full Kahn's check used for debug/validation.
// ============================================================================

bool Partition::is_acyclic() const {
    if (!prob || !dag) return true;
    if (op_to_groups_.size() < prob->num_ops()) return false;  // stale index

    using feasibility::MoveDelta;
    using feasibility::kahn_with_delta;

    std::vector<bool> alive(groups.size(), false);
    size_t na = 0;
    for (size_t i = 0; i < groups.size(); i++)
        if (groups[i].alive) { alive[i] = true; na++; }

    MoveDelta delta;  // NONE
    return kahn_with_delta(*prob, *dag, op_to_groups_, alive, na, delta);
}

// ============================================================================
// Local acyclicity checks
//
// Exploit the invariant that the current group DAG is acyclic: adding edge
// A→B creates a cycle iff B can already reach A (B →* A).
//
// With OR-node recomputation semantics: a group gB needs tensor T from AT
// LEAST ONE of its producer groups.  A new constraint on gB is only
// unsatisfiable (cyclic) if ALL producer options are forward-reachable from
// gB (meaning every option is downstream of gB).  If even one producer is
// not reachable from gB, we can place it before gB without creating a cycle.
// ============================================================================

bool Partition::acyclic_merge_local(const std::vector<size_t>& G) const {
    if (!prob || !dag || G.size() < 2) return true;

    // A merge creates a cycle iff there is a directed path from G to G that
    // passes through at least one external group (a direct gi→gj edge between
    // two members of G just becomes internal after merge — not a cycle).
    //
    // Algorithm: BFS from external successors of all groups in G.  If we ever
    // reach a group that is itself in G, the merge would create a cycle.

    std::vector<bool> in_G(groups.size(), false);
    for (auto g : G)
        if (g < groups.size()) in_G[g] = true;

    std::vector<bool> vis(groups.size(), false);
    std::queue<size_t> q;

    // Seed with external direct successors of each group in G.
    for (auto gi : G) {
        if (!groups[gi].alive) continue;
        for (auto op : groups[gi].ops) {
            for (auto t : prob->ops[op].outputs) {
                for (auto cop : dag->tensor_consumers[t]) {
                    if (groups[gi].ops.count(cop)) continue;  // internal to gi
                    for (auto gj : groups_of(cop)) {
                        if (!groups[gj].alive || vis[gj]) continue;
                        if (in_G[gj]) continue;  // direct G→G edge, fine
                        vis[gj] = true;
                        q.push(gj);
                    }
                }
            }
        }
    }

    // BFS through external groups; cycle if we reach back into G.
    while (!q.empty()) {
        size_t cur = q.front(); q.pop();
        for (auto op : groups[cur].ops) {
            for (auto t : prob->ops[op].outputs) {
                for (auto cop : dag->tensor_consumers[t]) {
                    if (groups[cur].ops.count(cop)) continue;
                    for (auto gj : groups_of(cop)) {
                        if (!groups[gj].alive) continue;
                        if (in_G[gj]) return false;  // external path back into G
                        if (!vis[gj]) { vis[gj] = true; q.push(gj); }
                    }
                }
            }
        }
    }
    return true;
}

bool Partition::acyclic_merge_local(size_t ga, size_t gb) const {
    return acyclic_merge_local(std::vector<size_t>{ga, gb});
}

bool Partition::acyclic_add_ops_into(const FlatSet<size_t>& new_ops, size_t gi) const {
    if (!prob || !dag || new_ops.empty() || gi >= groups.size()) return true;
    // Treat new_ops as a virtual group gnew (no group index yet).
    // Merged set = new_ops ∪ groups[gi].ops.
    // Cycle iff there is a directed path from the merged set back into itself
    // through at least one external group.
    // Same BFS as acyclic_merge_local but seeds from both new_ops and gi.ops.
    auto in_merged = [&](size_t op) {
        return new_ops.count(op) || groups[gi].ops.count(op);
    };

    std::vector<bool> vis(groups.size(), false);
    vis[gi] = true;  // gi is in the merged set, skip it during BFS
    std::queue<size_t> q;

    auto seed = [&](size_t op) {
        for (auto t : prob->ops[op].outputs) {
            for (auto cop : dag->tensor_consumers[t]) {
                if (in_merged(cop)) continue;
                for (auto gc : groups_of(cop)) {
                    if (!groups[gc].alive || vis[gc]) continue;
                    vis[gc] = true;
                    q.push(gc);
                }
            }
        }
    };
    for (auto op : new_ops)         seed(op);
    for (auto op : groups[gi].ops)  seed(op);

    while (!q.empty()) {
        size_t cur = q.front(); q.pop();
        for (auto cur_op : groups[cur].ops) {
            for (auto t : prob->ops[cur_op].outputs) {
                for (auto cop : dag->tensor_consumers[t]) {
                    if (in_merged(cop)) return false;  // path back into merged set
                    for (auto gc : groups_of(cop)) {
                        if (!groups[gc].alive || vis[gc]) continue;
                        vis[gc] = true;
                        q.push(gc);
                    }
                }
            }
        }
    }
    return true;
}

bool Partition::acyclic_extract_local(const FlatSet<size_t>& extract_ops) const {
    if (!prob || !dag || extract_ops.empty()) return true;
    // gnew = virtual group containing extract_ops.
    // Seed BFS with external successors of gnew: consumer groups of extract_ops outputs
    // that are NOT themselves extract_ops.
    std::vector<bool> vis(groups.size(), false);
    std::queue<size_t> q;

    for (auto op : extract_ops) {
        for (auto t : prob->ops[op].outputs) {
            for (auto cop : dag->tensor_consumers[t]) {
                if (extract_ops.count(cop)) continue;  // internal to gnew
                for (auto gc : groups_of(cop)) {
                    if (!groups[gc].alive || vis[gc]) continue;
                    vis[gc] = true;
                    q.push(gc);
                }
            }
        }
    }

    // BFS through external groups. A cycle exists if we find a group that
    // contains an op (not in extract_ops) whose output is consumed by extract_ops —
    // meaning gnew depends on that group, completing a cycle.
    while (!q.empty()) {
        size_t cur = q.front(); q.pop();
        for (auto cur_op : groups[cur].ops) {
            if (extract_ops.count(cur_op)) continue;
            for (auto t : prob->ops[cur_op].outputs) {
                for (auto cop : dag->tensor_consumers[t]) {
                    if (extract_ops.count(cop)) return false;  // path back into gnew
                    for (auto gc : groups_of(cop)) {
                        if (!groups[gc].alive || vis[gc]) continue;
                        vis[gc] = true;
                        q.push(gc);
                    }
                }
            }
        }
    }
    return true;
}

bool Partition::acyclic_recompute_local(size_t op, size_t gb) const {
    if (!prob || !dag) return true;
    // Adding op to gb: for each input T of op not already consumed in gb,
    // gb gains a new OR-constraint: at least one producer group must precede gb.
    // Cycle iff ALL producer groups are forward-reachable from gb.
    // Outputs are only new options for consumers (never forced) → no new cycle.

    const auto& fwd_gb = cached_forward_reachable(gb);

    for (auto t : prob->ops[op].inputs) {
        // Skip if T already consumed by another op in gb (constraint exists).
        bool already = false;
        for (auto cop : dag->tensor_consumers[t])
            if (cop != op && groups[gb].ops.count(cop)) { already = true; break; }
        if (already) continue;

        int prod = dag->tensor_producer[t];
        if (prod < 0) continue;
        if (groups[gb].ops.count((size_t)prod)) continue;  // internal

        // OR constraint: need at least one producer group NOT in fwd(gb).
        bool any_free = false;
        for (auto gp : groups_of((size_t)prod)) {
            if (gp != gb && groups[gp].alive && !fwd_gb[gp]) { any_free = true; break; }
        }
        if (!any_free) return false;
    }
    return true;
}

bool Partition::acyclic_de_recompute_local(size_t op, size_t ga) const {
    if (!prob || !dag) return true;
    // Removing op from ga: for each output T of op consumed in ga (T ephemeral),
    // ga gains a new OR-constraint: must come after at least one remaining copy.
    // Cycle iff ALL remaining groups containing op are forward-reachable from ga.

    // Build source set S = groups_of(op) − {ga}  (remaining copies of op)
    std::vector<size_t> S;
    for (auto gother : groups_of(op))
        if (gother != ga && groups[gother].alive) S.push_back(gother);

    const auto& fwd_ga = cached_forward_reachable(ga);

    for (auto t : prob->ops[op].outputs) {
        // Part 1: internal consumers in ga lose their ephemeral source.
        bool consumed_in_ga = false;
        for (auto cop : dag->tensor_consumers[t])
            if (cop != op && groups[ga].ops.count(cop)) { consumed_in_ga = true; break; }
        if (consumed_in_ga) {
            // OR constraint on ga: need at least one gP ∈ S not in fwd(ga).
            bool any_free = false;
            for (auto gP : S)
                if (!fwd_ga[gP]) { any_free = true; break; }
            if (!any_free) return false;
        }

        // Part 2: external consumers that were getting T from ga now must get
        // it from S.  For each such consumer group gc, at least one gP ∈ S
        // must not be forward-reachable from gc (can be placed before gc).
        // This mirrors Part 3 of acyclic_steal_local.
        if (S.empty()) continue;
        for (auto cop : dag->tensor_consumers[t]) {
            if (groups[ga].ops.count(cop)) continue;  // internal, handled above
            for (auto gc : groups_of(cop)) {
                if (!groups[gc].alive || gc == ga) continue;
                const auto& fwd_gc = cached_forward_reachable(gc);
                bool any_free = false;
                for (auto gP : S)
                    if (!fwd_gc[gP]) { any_free = true; break; }
                if (!any_free) return false;
            }
        }
    }
    return true;
}

bool Partition::acyclic_steal_local(size_t op, size_t ga, size_t gb) const {
    if (!prob || !dag) return true;

    // Part 1: gb gains op → check gb's new input OR-constraints (same as recompute).
    if (!acyclic_recompute_local(op, gb)) return false;

    // Part 2: ga loses op → for each output T of op consumed in ga (T was ephemeral),
    // ga now needs T from {gb} ∪ (groups_of(op) − {ga}).
    // Cycle iff ALL sources in that set are forward-reachable from ga.
    {
        const auto& fwd_ga = cached_forward_reachable(ga);
        for (auto t : prob->ops[op].outputs) {
            bool consumed_in_ga = false;
            for (auto cop : dag->tensor_consumers[t])
                if (cop != op && groups[ga].ops.count(cop)) { consumed_in_ga = true; break; }
            if (!consumed_in_ga) continue;

            // Source options: gb (op's new home) ∪ any other existing copy.
            bool any_free = (gb < groups.size() && groups[gb].alive && !fwd_ga[gb]);
            if (!any_free) {
                for (auto gother : groups_of(op))
                    if (gother != ga && groups[gother].alive && !fwd_ga[gother])
                        { any_free = true; break; }
            }
            if (!any_free) return false;
        }
    }

    // Part 3: external consumers of op's boundary outputs from ga lose ga as a source.
    // After steal, T is available from S = {gb} ∪ (groups_of(op) − {ga}).
    // For each external consumer group gc: need at least one gP ∈ S that gc cannot
    // forward-reach (i.e., gP can be placed before gc without forming a cycle).
    // Cycle iff ALL gP ∈ S are forward-reachable from gc (gc →* all options).
    {
        // Build S once.
        std::vector<size_t> S;
        if (gb < groups.size() && groups[gb].alive) S.push_back(gb);
        for (auto gother : groups_of(op))
            if (gother != ga && groups[gother].alive) S.push_back(gother);
        if (S.empty()) return false;  // op has no copy left → no source for T

        for (auto t : prob->ops[op].outputs) {
            // Check external consumers of T (not in ga, not in gb).
            for (auto cop : dag->tensor_consumers[t]) {
                for (auto gc : groups_of(cop)) {
                    if (!groups[gc].alive || gc == ga || gc == gb) continue;
                    // Is there at least one gP ∈ S that gc cannot reach?
                    const auto& fwd_gc = cached_forward_reachable(gc);
                    bool any_free = false;
                    for (auto gP : S)
                        if (!fwd_gc[gP]) { any_free = true; break; }
                    if (!any_free) return false;
                }
            }
        }
    }

    return true;
}

bool Partition::acyclic_split_local(const FlatSet<size_t>& side_a,
                                     const FlatSet<size_t>& side_b,
                                     size_t ga) const {
    if (!prob || !dag) return true;

    // Temporarily apply the split, check acyclicity, then undo.
    // The split only touches groups[ga] and adds a new group — O(V+E) check.
    Partition& mut = const_cast<Partition&>(*this);
    auto saved_ops = mut.groups[ga].ops;
    double saved_cost = mut.groups[ga].cost;

    mut.groups[ga].ops = side_a;
    size_t gb = mut.groups.size();
    mut.groups.push_back({side_b, 0.0, true, 0, std::nullopt, {}});
    mut.rebuild_index();

    bool ok = mut.is_acyclic();

    // Undo: remove the temporary group and restore ga.
    mut.groups.pop_back();
    mut.groups[ga].ops = std::move(saved_ops);
    mut.groups[ga].cost = saved_cost;
    mut.rebuild_index();

    return ok;
}

double Partition::eval_set(const FlatSet<size_t>& ops) const {
    if (ops.empty()) return 1e18;
    if (cache) return cache->evaluate(ops, *prob, *dag);
    auto sg = Subgraph::create(*prob, *dag, {ops.begin(), ops.end()});
    if (!sg) return 1e18;
    auto c = sg->best_cost();
    return c.feasible ? c.latency : 1e18;
}

// ============================================================================
// Mutation
// ============================================================================

size_t Partition::add_group(FlatSet<size_t> ops, double cost,
                             std::optional<Subgraph> sg, TileConfig cfg) {
    size_t idx = groups.size();
    Group g;
    g.ops      = std::move(ops);
    g.cost     = cost;
    g.alive    = true;
    g.gen      = 0;
    g.best_cfg = cfg;
    g.sg       = std::move(sg);
    groups.push_back(std::move(g));

    // Update op_to_groups_ incrementally
    if (op_to_groups_.size() < prob->num_ops())
        op_to_groups_.resize(prob->num_ops());
    for (auto op : groups[idx].ops)
        op_to_groups_[op].push_back(idx);

    return idx;
}

// ============================================================================
// Eject / split helpers
// ============================================================================

std::vector<size_t> Partition::ejectable_ops(size_t gi) const {
    if (groups[gi].ops.size() <= 1) return {};
    std::vector<size_t> result;
    for (auto op : groups[gi].ops)
        if (is_border_op(op, gi))
            result.push_back(op);
    return result;
}

std::vector<size_t> Partition::internal_ops(size_t gi) const {
    if (!groups[gi].alive) return {};
    std::vector<size_t> result;
    for (auto op : groups[gi].ops)
        if (!is_border_op(op, gi))
            result.push_back(op);
    return result;
}

// ============================================================================
// Connected components
// ============================================================================

std::vector<FlatSet<size_t>> Partition::connected_components(
        const FlatSet<size_t>& ops) const {
    return structural_ops::connected_components(ops, *dag);
}

// ============================================================================
// Eject evaluation
// ============================================================================

Partition::EjectResult Partition::eval_eject(size_t op, size_t gi) const {
    EjectResult result;
    if (!groups[gi].alive || !groups[gi].ops.count(op)) return result;
    if (groups[gi].ops.size() <= 1) return result;

    auto analysis = structural_ops::analyze_eject(op, groups[gi].ops, *dag);
    result.remainder_components = std::move(analysis.remainder_components);

    double total_remainder = 0;
    result.component_costs.reserve(result.remainder_components.size());
    for (auto& comp : result.remainder_components) {
        double c = eval_set(comp);
        if (c >= 1e17) return result;
        result.component_costs.push_back(c);
        total_remainder += c;
    }

    bool op_in_other = false;
    for (auto gj : op_to_groups_[op])
        if (gj != gi) { op_in_other = true; break; }

    result.singleton_cost = op_in_other ? 0.0 : eval_set({op});
    if (!op_in_other && result.singleton_cost >= 1e17) return result;

    result.feasible = true;
    result.saving   = groups[gi].cost - (total_remainder + result.singleton_cost);
    return result;
}

// ============================================================================
// Split evaluation
// ============================================================================

Partition::SplitResult Partition::eval_split(size_t op_a, size_t op_b, size_t gi) const {
    SplitResult result;
    if (!groups[gi].alive) return result;
    if (!groups[gi].ops.count(op_a) || !groups[gi].ops.count(op_b)) return result;

    auto analysis = structural_ops::analyze_split(op_a, op_b, groups[gi].ops, *dag);
    if (!analysis.is_bridge) return result;

    result.side_a = std::move(analysis.side_a);
    result.side_b = std::move(analysis.side_b);
    if (result.side_a.empty() || result.side_b.empty()) return result;

    result.cost_a = eval_set(result.side_a);
    if (result.cost_a >= 1e17) return result;
    result.cost_b = eval_set(result.side_b);
    if (result.cost_b >= 1e17) return result;

    result.feasible = true;
    result.saving   = groups[gi].cost - (result.cost_a + result.cost_b);
    return result;
}

// ============================================================================
// Bridge edges
// ============================================================================

std::vector<std::pair<size_t,size_t>> Partition::bridge_edges(size_t gi) const {
    if (!groups[gi].alive || groups[gi].ops.size() < 2) return {};
    return structural_ops::bridge_edges(groups[gi].ops, *dag);
}

// ============================================================================
// Ephemeral gap check
// ============================================================================

bool Partition::creates_ephemeral_gap(const FlatSet<size_t>& proposed_ops,
                                      size_t exclude_ga,
                                      size_t exclude_gb) const {
    if (!prob || !dag) return false;

    for (auto op : proposed_ops) {
        for (auto t : prob->ops[op].outputs) {
            // New ephemeral rule: T is ephemeral in proposed_ops iff it is
            // produced AND consumed internally. External consumers are irrelevant
            // to the ephemeral classification — but they DO need T from somewhere.
            bool any_consumer_internal = false;
            for (auto cop : dag->tensor_consumers[t])
                if (proposed_ops.count(cop)) { any_consumer_internal = true; break; }
            if (!any_consumer_internal) continue;  // pure boundary output → safe

            // T is ephemeral in proposed_ops → not written to slow memory.
            // Any external consumer needs T from another group.
            for (auto cop : dag->tensor_consumers[t]) {
                if (proposed_ops.count(cop)) continue;  // internal → served

                // cop is external — does it have access to T?
                // T is available if some alive group (not excluded) produces T
                // as a boundary output (= produces T but does NOT consume T internally)
                int prod_op = dag->tensor_producer[t];
                if (prod_op < 0) continue;

                bool available = false;
                for (auto gj : groups_of((size_t)prod_op)) {
                    if (!groups[gj].alive) continue;
                    if (gj == exclude_ga || gj == exclude_gb) continue;

                    // Is T a boundary output of gj?
                    // Under new rule: T is ephemeral in gj iff gj consumes T internally.
                    // T is boundary output of gj iff gj produces T but does NOT consume it.
                    bool consumed_in_gj = false;
                    for (auto c2 : dag->tensor_consumers[t])
                        if (groups[gj].ops.count(c2)) { consumed_in_gj = true; break; }
                    if (!consumed_in_gj) { available = true; break; }
                }

                // Also check if cop's own group recomputes the producer
                if (!available) {
                    for (auto gj : groups_of(cop)) {
                        if (!groups[gj].alive) continue;
                        if (gj == exclude_ga || gj == exclude_gb) continue;
                        if (groups[gj].ops.count((size_t)prod_op)) { available = true; break; }
                    }
                }

                if (!available) return true;
            }
        }
    }
    return false;
}

bool Partition::creates_ephemeral_gap(const FlatSet<size_t>& proposed_ops,
                                      const std::vector<size_t>& exclude_groups) const {
    if (!prob || !dag) return false;
    FlatSet<size_t> excluded(exclude_groups.begin(), exclude_groups.end());

    for (auto op : proposed_ops) {
        for (auto t : prob->ops[op].outputs) {
            bool any_consumer_internal = false;
            for (auto cop : dag->tensor_consumers[t])
                if (proposed_ops.count(cop)) { any_consumer_internal = true; break; }
            if (!any_consumer_internal) continue;  // pure boundary output → safe

            // T is ephemeral → external consumers need it from elsewhere
            for (auto cop : dag->tensor_consumers[t]) {
                if (proposed_ops.count(cop)) continue;

                int prod_op = dag->tensor_producer[t];
                if (prod_op < 0) continue;

                bool available = false;
                // Check non-excluded groups for boundary output of T
                for (auto gj : groups_of((size_t)prod_op)) {
                    if (!groups[gj].alive || excluded.count(gj)) continue;
                    bool consumed_in_gj = false;
                    for (auto c2 : dag->tensor_consumers[t])
                        if (groups[gj].ops.count(c2)) { consumed_in_gj = true; break; }
                    if (!consumed_in_gj) { available = true; break; }
                }
                // Check if cop's group recomputes producer
                if (!available) {
                    for (auto gj : groups_of(cop)) {
                        if (!groups[gj].alive || excluded.count(gj)) continue;
                        if (groups[gj].ops.count((size_t)prod_op)) { available = true; break; }
                    }
                }
                if (!available) return true;
            }
        }
    }
    return false;
}

// ============================================================================
// split_creates_ephemeral_gap
//
// Checks whether replacing excluded groups with a set of new component
// op-sets creates an ephemeral gap.
//
// Unlike creates_ephemeral_gap (single proposed group vs existing groups),
// this checks multiple sibling components simultaneously.  A tensor might be
// available as a boundary output from a sibling component even if no existing
// group provides it.
// ============================================================================

bool Partition::split_creates_ephemeral_gap(
        const std::vector<FlatSet<size_t>>& components,
        const FlatSet<size_t>& excluded) const {
    if (!prob || !dag) return false;

    for (size_t ci = 0; ci < components.size(); ci++) {
        const auto& comp = components[ci];

        for (auto op : comp) {
            for (auto t : prob->ops[op].outputs) {
                // T is ephemeral in this component if any consumer is internal
                bool any_consumer_in_comp = false;
                for (auto cop : dag->tensor_consumers[t])
                    if (comp.count(cop)) { any_consumer_in_comp = true; break; }
                if (!any_consumer_in_comp) continue;  // pure boundary output → safe

                int prod_op = dag->tensor_producer[t];
                if (prod_op < 0) continue;

                // External consumers need T from elsewhere
                for (auto cop : dag->tensor_consumers[t]) {
                    if (comp.count(cop)) continue;  // internal → served

                    bool served = false;

                    // Check sibling components: T is boundary output of cj if
                    // cj produces T and does NOT consume T internally
                    for (size_t cj = 0; cj < components.size() && !served; cj++) {
                        if (cj == ci) continue;
                        if (!components[cj].count((size_t)prod_op)) continue;
                        bool consumed_in_cj = false;
                        for (auto c2 : dag->tensor_consumers[t])
                            if (components[cj].count(c2)) { consumed_in_cj = true; break; }
                        if (!consumed_in_cj) served = true;
                    }

                    // Check if cop's sibling component recomputes producer
                    for (size_t cj = 0; cj < components.size() && !served; cj++)
                        if (components[cj].count(cop) && components[cj].count((size_t)prod_op))
                            served = true;

                    // Check existing non-excluded groups
                    for (auto gj : groups_of((size_t)prod_op)) {
                        if (served) break;
                        if (!groups[gj].alive || excluded.count(gj)) continue;
                        bool consumed_in_gj = false;
                        for (auto c2 : dag->tensor_consumers[t])
                            if (groups[gj].ops.count(c2)) { consumed_in_gj = true; break; }
                        if (!consumed_in_gj) served = true;
                    }

                    // Check if cop's existing group recomputes producer
                    for (auto gj : groups_of(cop)) {
                        if (served) break;
                        if (!groups[gj].alive || excluded.count(gj)) continue;
                        if (groups[gj].ops.count((size_t)prod_op)) served = true;
                    }

                    if (!served) return true;
                }
            }
        }
    }
    return false;
}