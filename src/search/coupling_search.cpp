#include "search/coupling_search.h"
#include "search/partition_moves.h"
#include "core/cost_cache.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>

using SteadyClock = std::chrono::steady_clock;

// Forward declaration — defined in the Evaluation section below.
static bool acyclic_chain_merge(const CoupledPartition& cp,
                                 const std::vector<size_t>& chain_a,
                                 const std::vector<size_t>& chain_b);

// ============================================================================
// CoupledPartition: construction and maintenance
// ============================================================================

void CoupledPartition::init_from(Partition p, CostCache* cache) {
    part = std::move(p);
    // Store the cache pointer for cost evaluation during search.
    // finalize() is NOT called here: coupling search uses the cache directly
    // for cost evaluation and does not need stored .sg.  Callers that need .sg
    // (ordering functions in solver.cpp Phase 2) call part.finalize() themselves.
    if (cache) part.cache = cache;
    size_t n = part.groups.size();
    next_group.assign(n, SIZE_MAX);
    prev_group.assign(n, SIZE_MAX);
    retained.clear();
}

void CoupledPartition::invalidate_couplings() {
    const DAG& dag = *part.dag;
    for (size_t g = 0; g < next_group.size(); g++) {
        size_t h = next_group[g];
        if (h == SIZE_MAX) continue;

        bool ok = (g < part.groups.size() && h < part.groups.size() &&
                   part.groups[g].alive && part.groups[h].alive);
        if (!ok) {
            next_group[g] = SIZE_MAX;
            if (h < prev_group.size()) prev_group[h] = SIZE_MAX;
            retained.erase({g, h});
            continue;
        }

        // Remove retained tensors no longer valid on either side:
        //   producer side: t must still be a boundary output of g
        //   consumer side: t must still be a boundary input  of h
        // (After EJECT/STEAL, the consuming op may have left h, making the
        //  retained edge stale and causing incorrect working-set inflation.)
        auto it = retained.find({g, h});
        if (it == retained.end()) continue;
        std::set<size_t> valid;
        for (auto t : it->second)
            if (is_boundary_output_of(part.groups[g].ops, t, dag) &&
                is_boundary_input_of(part.groups[h].ops, t, dag))
                valid.insert(t);
        if (valid.empty()) {
            next_group[g] = SIZE_MAX;
            prev_group[h] = SIZE_MAX;
            retained.erase(it);
        } else {
            it->second = std::move(valid);
        }
    }
}

void CoupledPartition::fix_chain_couplings() {
    // Mutations can create new group DAG edges that make existing coupling links
    // form a cycle in the chain-level DAG.  Detect and remove them.
    //
    // For each coupling link (ga→gb): temporarily disconnect it, compute the
    // two sub-chains, and check chain-level acyclicity.  Remove the link if it
    // creates a cycle.  Repeat until no more cycles are found.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t ga = 0; ga < next_group.size(); ga++) {
            size_t gb = next_group[ga];
            if (gb == SIZE_MAX) continue;
            if (!part.groups[ga].alive || !part.groups[gb].alive) continue;

            // Temporarily disconnect ga→gb so chain_of correctly splits.
            next_group[ga] = SIZE_MAX;
            prev_group[gb] = SIZE_MAX;

            auto chain_a = chain_of(ga);
            auto chain_b = chain_of(gb);

            // Restore the link.
            next_group[ga] = gb;
            prev_group[gb] = ga;

            if (!acyclic_chain_merge(*this, chain_a, chain_b)) {
                // Remove this coupling link.
                next_group[ga] = SIZE_MAX;
                prev_group[gb] = SIZE_MAX;
                retained.erase({ga, gb});
                changed = true;
                break;  // Restart — chain heads changed.
            }
        }
    }
}

// ============================================================================
// Chain helpers
// ============================================================================

size_t CoupledPartition::chain_head(size_t g) const {
    size_t n = part.groups.size() + 1;
    for (size_t steps = 0; steps < n && g < prev_group.size() && prev_group[g] != SIZE_MAX; steps++)
        g = prev_group[g];
    return g;
}

size_t CoupledPartition::chain_tail(size_t g) const {
    size_t n = part.groups.size() + 1;
    for (size_t steps = 0; steps < n && g < next_group.size() && next_group[g] != SIZE_MAX; steps++)
        g = next_group[g];
    return g;
}

std::vector<size_t> CoupledPartition::chain_of(size_t g) const {
    std::vector<size_t> result;
    size_t cur = chain_head(g);
    size_t n = part.groups.size() + 1;
    for (size_t steps = 0; steps < n && cur != SIZE_MAX; steps++) {
        result.push_back(cur);
        cur = (cur < next_group.size()) ? next_group[cur] : SIZE_MAX;
    }
    return result;
}

std::set<size_t> CoupledPartition::entering_for(size_t g) const {
    if (g >= prev_group.size() || prev_group[g] == SIZE_MAX) return {};
    auto it = retained.find({prev_group[g], g});
    return (it != retained.end()) ? it->second : std::set<size_t>{};
}

std::set<size_t> CoupledPartition::retain_for(size_t g) const {
    if (g >= next_group.size() || next_group[g] == SIZE_MAX) return {};
    auto it = retained.find({g, next_group[g]});
    return (it != retained.end()) ? it->second : std::set<size_t>{};
}

// ============================================================================
// Cost
// ============================================================================

double CoupledPartition::group_cost(size_t g) const {
    if (!part.groups[g].alive) return 0.0;
    auto en = entering_for(g);
    auto re = retain_for(g);
    if (en.empty() && re.empty()) return part.groups[g].cost;
    // Use cache: no .sg required.
    if (part.cache) {
        auto cr = part.cache->evaluate_with_context(part.groups[g].ops, en, re,
                                                     *part.prob, *part.dag);
        return cr.feasible ? cr.latency : 1e18;
    }
    // Fallback: use stored .sg if available.
    if (!part.groups[g].sg) return 1e18;
    auto bc = part.groups[g].sg->best_cost(en, re);
    return bc.feasible ? bc.latency : 1e18;
}

double CoupledPartition::total_cost() const {
    double total = 0;
    for (size_t g = 0; g < part.groups.size(); g++)
        if (part.groups[g].alive)
            total += group_cost(g);
    return total;
}

// ============================================================================
// to_solution: topological sort of chains → ScheduleSteps
// ============================================================================

Solution CoupledPartition::to_solution() const {
    const auto& prob = *part.prob;
    const auto& dag  = *part.dag;
    size_t n = part.groups.size();

    // Map each alive group to its chain head.
    std::vector<size_t> head_of(n, SIZE_MAX);
    for (size_t g = 0; g < n; g++)
        if (part.groups[g].alive)
            head_of[g] = chain_head(g);

    // Collect unique chain heads (alive groups with no incoming coupling).
    std::vector<size_t> heads;
    for (size_t g = 0; g < n; g++)
        if (part.groups[g].alive && (g >= prev_group.size() || prev_group[g] == SIZE_MAX))
            heads.push_back(g);

    // Build chain-level DAG from the op/tensor DAG directly (not
    // part.group_succs, which may be stale after coupling mutations).
    // Edge h1→h2 if any group in chain(h1) produces a tensor consumed by
    // a group in chain(h2).  Edges within the same chain are skipped.
    std::map<size_t, std::set<size_t>> chain_succs;
    std::map<size_t, int> in_deg;
    for (auto h : heads) { chain_succs[h]; in_deg[h] = 0; }

    for (size_t g = 0; g < n; g++) {
        if (!part.groups[g].alive) continue;
        size_t hg = head_of[g];
        for (auto op : part.groups[g].ops) {
            for (auto t : prob.ops[op].outputs) {
                if (!is_boundary_output_of(part.groups[g].ops, t, dag)) continue;
                for (auto cop : dag.tensor_consumers[t]) {
                    for (auto gs : part.groups_of(cop)) {
                        if (!part.groups[gs].alive || gs == g) continue;
                        size_t hs = head_of[gs];
                        if (hs == hg) continue;  // same chain
                        if (chain_succs[hg].insert(hs).second)
                            in_deg[hs]++;
                    }
                }
            }
        }
    }

    // Kahn's topological sort of chain heads.
    std::vector<size_t> topo_order;
    std::queue<size_t> q;
    for (auto h : heads)
        if (in_deg[h] == 0) q.push(h);
    while (!q.empty()) {
        size_t h = q.front(); q.pop();
        topo_order.push_back(h);
        for (auto hs : chain_succs[h])
            if (--in_deg[hs] == 0) q.push(hs);
    }

    // If cycle guard fires (should not happen after fix_chain_couplings / the
    // chain-level eval_couple check), fall back to a group-level topological
    // sort and emit steps without any retention.  This produces a valid
    // (though unoptimised) solution rather than an invalid one.
    if (topo_order.size() < heads.size()) {
        std::cerr << "WARNING: to_solution: chain DAG cycle detected — "
                     "falling back to group-level sort (no retention)\n";

        // Group-level Kahn's from op/tensor DAG (not stale group_succs).
        std::vector<size_t> all_g;
        for (size_t g = 0; g < n; g++)
            if (part.groups[g].alive) all_g.push_back(g);

        std::map<size_t, std::set<size_t>> g_succs;
        std::map<size_t, int> g_indeg;
        for (auto g : all_g) { g_succs[g]; g_indeg[g] = 0; }
        for (auto g : all_g) {
            for (auto op : part.groups[g].ops) {
                for (auto t : prob.ops[op].outputs) {
                    if (!is_boundary_output_of(part.groups[g].ops, t, dag)) continue;
                    for (auto cop : dag.tensor_consumers[t]) {
                        for (auto gs : part.groups_of(cop)) {
                            if (!part.groups[gs].alive || gs == g) continue;
                            if (g_succs[g].insert(gs).second)
                                g_indeg[gs]++;
                        }
                    }
                }
            }
        }
        std::queue<size_t> gq;
        std::vector<size_t> g_topo;
        for (auto g : all_g)
            if (g_indeg[g] == 0) gq.push(g);
        while (!gq.empty()) {
            size_t g = gq.front(); gq.pop();
            g_topo.push_back(g);
            for (auto s : g_succs[g])
                if (part.groups[s].alive && --g_indeg[s] == 0) gq.push(s);
        }
        for (auto g : all_g)
            if (g_indeg[g] > 0) g_topo.push_back(g);

        std::vector<ScheduleStep> fallback_steps;
        for (auto g : g_topo) {
            const auto& grp = part.groups[g];
            auto sg_opt = Subgraph::create(prob, dag,
                              std::vector<size_t>(grp.ops.begin(), grp.ops.end()));
            if (!sg_opt) continue;
            ScheduleStep step;
            step.subgraph = *sg_opt;
            CostResult cr;
            if (part.cache)
                cr = part.cache->evaluate_with_context(grp.ops, {}, {}, prob, dag);
            else
                cr = sg_opt->best_cost({}, {});
            step.config = cr.feasible ? cr.config : grp.best_cfg;
            fallback_steps.push_back(std::move(step));
        }
        return Solution(prob, dag, std::move(fallback_steps));
    }

    // Emit ScheduleSteps: walk each chain head-to-tail.
    std::vector<ScheduleStep> steps;
    for (size_t h : topo_order) {
        size_t g = h;
        while (g != SIZE_MAX) {
            const auto& grp = part.groups[g];
            // Build subgraph inline — .sg may be absent or stale.
            auto sg_opt = Subgraph::create(prob, dag,
                              std::vector<size_t>(grp.ops.begin(), grp.ops.end()));
            if (!sg_opt) break;

            ScheduleStep step;
            step.subgraph = *sg_opt;
            size_t nxt = (g < next_group.size()) ? next_group[g] : SIZE_MAX;
            if (nxt != SIZE_MAX) {
                auto it = retained.find({g, nxt});
                if (it != retained.end())
                    step.retain_these = it->second;
            }
            // Pick config that accounts for entering + retain sets.
            // Always evaluate via cache/subgraph so we get the actual TileConfig
            // back — grp.best_cfg may be unset ({0,0,0}) for groups created by
            // mutations that never called finalize().
            auto enter = entering_for(g);
            auto ret   = retain_for(g);
            CostResult cr;
            if (part.cache)
                cr = part.cache->evaluate_with_context(grp.ops, enter, ret, prob, dag);
            else
                cr = sg_opt->best_cost(enter, ret);
            if (!cr.feasible && (!enter.empty() || !ret.empty())) {
                // Retention context infeasible — fall back to standalone config.
                if (part.cache)
                    cr = part.cache->evaluate_with_context(grp.ops, {}, {}, prob, dag);
                else
                    cr = sg_opt->best_cost({}, {});
            }
            step.config = cr.feasible ? cr.config : grp.best_cfg;
            steps.push_back(std::move(step));
            g = nxt;
        }
    }

    return Solution(prob, dag, std::move(steps));
}

// ============================================================================
// Chain-level acyclicity check for coupling moves.
//
// acyclic_merge_local operates at the GROUP level: it BFS's through individual
// external groups.  This misses chain-level cycles where two valid couplings
// (e.g. A={G1→G2}, B={G3→G4}) create a cycle via inter-chain group edges
// (e.g. G3→G2 and G1→G4): the group-level BFS from G2 never traverses G1
// (a different group in the same chain), so it misses the G1→G4 edge.
//
// The chain-level check BFS's through entire external CHAINS (following
// next_group from the entry group all the way to the tail), catching exactly
// these cross-chain dependencies.
// ============================================================================

static bool acyclic_chain_merge(const CoupledPartition& cp,
                                 const std::vector<size_t>& chain_a,
                                 const std::vector<size_t>& chain_b) {
    if (!cp.part.prob || !cp.part.dag) return true;

    std::set<size_t> in_G(chain_a.begin(), chain_a.end());
    for (auto g : chain_b) in_G.insert(g);

    size_t n = cp.part.groups.size();

    // For external groups, compute their chain head once.
    std::vector<size_t> head_of(n, SIZE_MAX);
    for (size_t g = 0; g < n; g++)
        if (cp.part.groups[g].alive && !in_G.count(g))
            head_of[g] = cp.chain_head(g);

    // BFS over external chain heads reachable from G.
    std::set<size_t> vis_heads;
    std::queue<size_t> q;

    // Seed: external direct successors of each group in G.
    for (auto gi : in_G) {
        if (!cp.part.groups[gi].alive) continue;
        for (auto op : cp.part.groups[gi].ops) {
            for (auto t : cp.part.prob->ops[op].outputs) {
                for (auto cop : cp.part.dag->tensor_consumers[t]) {
                    if (cp.part.groups[gi].ops.count(cop)) continue;
                    for (auto gj : cp.part.groups_of(cop)) {
                        if (!cp.part.groups[gj].alive || in_G.count(gj)) continue;
                        size_t hj = head_of[gj];
                        if (hj == SIZE_MAX || vis_heads.count(hj)) continue;
                        vis_heads.insert(hj);
                        q.push(hj);
                    }
                }
            }
        }
    }

    // BFS through external chains; cycle if any group in an external chain
    // produces a tensor consumed by a group in G.
    while (!q.empty()) {
        size_t h_cur = q.front(); q.pop();
        // Walk all groups in this external chain.
        size_t g_cur = h_cur;
        for (size_t steps = 0; steps <= n && g_cur != SIZE_MAX; steps++) {
            if (!cp.part.groups[g_cur].alive) break;
            for (auto op : cp.part.groups[g_cur].ops) {
                for (auto t : cp.part.prob->ops[op].outputs) {
                    for (auto cop : cp.part.dag->tensor_consumers[t]) {
                        if (cp.part.groups[g_cur].ops.count(cop)) continue;
                        for (auto gj : cp.part.groups_of(cop)) {
                            if (!cp.part.groups[gj].alive) continue;
                            if (in_G.count(gj)) return false;  // external path back into G
                            size_t hj = head_of[gj];
                            if (hj == SIZE_MAX || vis_heads.count(hj)) continue;
                            vis_heads.insert(hj);
                            q.push(hj);
                        }
                    }
                }
            }
            g_cur = (g_cur < cp.next_group.size()) ? cp.next_group[g_cur] : SIZE_MAX;
        }
    }
    return true;
}

// ============================================================================
// Public wrapper: chain-level acyclicity for an arbitrary set of groups.
//
// Collects all groups from the chains of each input group, then checks via
// the same BFS as acyclic_chain_merge.  Used by best_coupled_move_for_op to
// filter MERGE / TENSOR_MERGE that would form chain-level cycles.
// ============================================================================

bool acyclic_chain_merge_groups(const CoupledPartition& cp,
                                  const std::vector<size_t>& groups) {
    std::vector<size_t> all_in_chains;
    for (auto g : groups) {
        if (g >= cp.part.groups.size() || !cp.part.groups[g].alive) continue;
        auto ch = cp.chain_of(g);
        all_in_chains.insert(all_in_chains.end(), ch.begin(), ch.end());
    }
    return acyclic_chain_merge(cp, all_in_chains, {});
}

// ============================================================================
// Evaluation
// ============================================================================

CouplingEvalResult eval_couple(const CoupledPartition& cp,
                                size_t ga, size_t gb, size_t t) {
    if (ga >= cp.part.groups.size() || gb >= cp.part.groups.size()) return {};
    if (!cp.part.groups[ga].alive || !cp.part.groups[gb].alive) return {};
    // Allowed in two cases:
    //   new edge:      ga is a free chain tail AND gb is a free chain head.
    //   existing edge: ga→gb already set (adding another tensor to the retained set).
    const bool existing_edge = (ga < cp.next_group.size() && cp.next_group[ga] == gb) &&
                                (gb < cp.prev_group.size() && cp.prev_group[gb] == ga);
    const bool new_edge      = (ga < cp.next_group.size() && cp.next_group[ga] == SIZE_MAX) &&
                                (gb < cp.prev_group.size() && cp.prev_group[gb] == SIZE_MAX);
    if (!existing_edge && !new_edge) return {};
    const DAG& dag = *cp.part.dag;
    if (!is_boundary_output_of(cp.part.groups[ga].ops, t, dag)) return {};
    if (!is_boundary_input_of(cp.part.groups[gb].ops, t, dag))  return {};

    // Refuse to add a tensor already in the retained set for this edge.
    if (existing_edge) {
        auto it = cp.retained.find({ga, gb});
        if (it != cp.retained.end() && it->second.count(t)) return {};
    }

    // Chain-level acyclicity: the merged chain (chain_of(ga) ++ chain_of(gb))
    // must not create a cycle in the chain-level DAG.  The stronger chain-level
    // check (vs the old acyclic_merge_local) BFS's through entire external
    // chains, catching cross-chain dependency cycles.
    auto chain_a = cp.chain_of(ga);
    auto chain_b = cp.chain_of(gb);
    if (!acyclic_chain_merge(cp, chain_a, chain_b)) return {};

    // Cost delta: (cost before) - (cost after adding retention).
    auto ga_enter  = cp.entering_for(ga);
    auto ga_retain = cp.retain_for(ga);   // {} (ga is chain tail)
    auto gb_enter  = cp.entering_for(gb); // {} (gb is chain head)
    auto gb_retain = cp.retain_for(gb);

    double cost_before = cp.group_cost(ga) + cp.group_cost(gb);

    auto new_ga_retain = ga_retain; new_ga_retain.insert(t);
    auto new_gb_enter  = gb_enter;  new_gb_enter.insert(t);

    CostResult r_ga, r_gb;
    if (cp.part.cache) {
        r_ga = cp.part.cache->evaluate_with_context(cp.part.groups[ga].ops,
                                                     ga_enter, new_ga_retain,
                                                     *cp.part.prob, dag);
        r_gb = cp.part.cache->evaluate_with_context(cp.part.groups[gb].ops,
                                                     new_gb_enter, gb_retain,
                                                     *cp.part.prob, dag);
    } else {
        if (!cp.part.groups[ga].sg || !cp.part.groups[gb].sg) return {};
        r_ga = cp.part.groups[ga].sg->best_cost(ga_enter, new_ga_retain);
        r_gb = cp.part.groups[gb].sg->best_cost(new_gb_enter, gb_retain);
    }
    if (!r_ga.feasible || !r_gb.feasible) return {};

    return {true, cost_before - (r_ga.latency + r_gb.latency)};
}

CouplingEvalResult eval_uncouple(const CoupledPartition& cp,
                                  size_t ga, size_t gb, size_t t) {
    if (ga >= cp.next_group.size() || cp.next_group[ga] != gb) return {};
    auto it = cp.retained.find({ga, gb});
    if (it == cp.retained.end() || !it->second.count(t)) return {};

    auto ga_enter  = cp.entering_for(ga);
    auto ga_retain = cp.retain_for(ga);
    auto gb_enter  = cp.entering_for(gb);
    auto gb_retain = cp.retain_for(gb);

    double cost_before = cp.group_cost(ga) + cp.group_cost(gb);

    auto new_ga_retain = ga_retain; new_ga_retain.erase(t);
    auto new_gb_enter  = gb_enter;  new_gb_enter.erase(t);

    CostResult r_ga, r_gb;
    if (cp.part.cache) {
        r_ga = cp.part.cache->evaluate_with_context(cp.part.groups[ga].ops,
                                                     ga_enter, new_ga_retain,
                                                     *cp.part.prob, *cp.part.dag);
        r_gb = cp.part.cache->evaluate_with_context(cp.part.groups[gb].ops,
                                                     new_gb_enter, gb_retain,
                                                     *cp.part.prob, *cp.part.dag);
    } else {
        if (!cp.part.groups[ga].sg || !cp.part.groups[gb].sg) return {};
        r_ga = cp.part.groups[ga].sg->best_cost(ga_enter, new_ga_retain);
        r_gb = cp.part.groups[gb].sg->best_cost(new_gb_enter, gb_retain);
    }
    if (!r_ga.feasible || !r_gb.feasible) return {};

    return {true, cost_before - (r_ga.latency + r_gb.latency)};
}

// ============================================================================
// RETAIN_FORCE_SPLIT
// ============================================================================

CouplingEvalResult eval_retain_force_split(const CoupledPartition& cp,
                                            size_t g,
                                            size_t op_a, size_t op_b,
                                            size_t t) {
    if (g >= cp.part.groups.size() || !cp.part.groups[g].alive) return {};
    if (!cp.part.groups[g].ops.count(op_a)) return {};
    if (!cp.part.groups[g].ops.count(op_b)) return {};

    // op_a must produce t; op_b must consume t
    if (cp.part.dag->tensor_producer[t] != (int)op_a) return {};
    bool b_consumes = false;
    for (auto inp : cp.part.prob->ops[op_b].inputs)
        if (inp == t) { b_consumes = true; break; }
    if (!b_consumes) return {};

    // t already a boundary output? COUPLE handles that case.
    if (is_boundary_output_of(cp.part.groups[g].ops, t, *cp.part.dag)) return {};

    // Bridge check + side feasibility (no retention yet)
    auto sr = cp.part.eval_split(op_a, op_b, g);
    if (!sr.feasible) return {};

    // Verify t becomes a boundary output of side_a and boundary input of side_b.
    if (!is_boundary_output_of(sr.side_a, t, *cp.part.dag)) return {};
    if (!is_boundary_input_of(sr.side_b, t, *cp.part.dag))  return {};

    // Cost with retention, inheriting G's current chain context:
    //   Ga: entering = enter_g, retain = {t}  (Ga → Gb via t)
    //   Gb: entering = {t},     retain = retain_g  (Gb → G's old next, if any)
    auto enter_g  = cp.entering_for(g);
    auto retain_g = cp.retain_for(g);

    CostResult r_a, r_b;
    if (cp.part.cache) {
        r_a = cp.part.cache->evaluate_with_context(sr.side_a, enter_g, {t},
                                                    *cp.part.prob, *cp.part.dag);
        r_b = cp.part.cache->evaluate_with_context(sr.side_b, {t}, retain_g,
                                                    *cp.part.prob, *cp.part.dag);
    } else {
        auto sg_a = Subgraph::create(*cp.part.prob, *cp.part.dag,
                                      {sr.side_a.begin(), sr.side_a.end()});
        if (!sg_a) return {};
        auto sg_b = Subgraph::create(*cp.part.prob, *cp.part.dag,
                                      {sr.side_b.begin(), sr.side_b.end()});
        if (!sg_b) return {};
        r_a = sg_a->best_cost(enter_g, {t});
        r_b = sg_b->best_cost({t}, retain_g);
    }
    if (!r_a.feasible || !r_b.feasible) return {};

    return {true, cp.group_cost(g) - (r_a.latency + r_b.latency)};
}

// ============================================================================
// FORCE_RETAIN
// ============================================================================

CouplingEvalResult eval_force_retain(const CoupledPartition& cp,
                                      size_t ga, size_t g_dst,
                                      size_t op_a_dst, size_t op_b_dst,
                                      size_t t) {
    if (ga >= cp.part.groups.size() || g_dst >= cp.part.groups.size()) return {};
    if (!cp.part.groups[ga].alive || !cp.part.groups[g_dst].alive) return {};
    // ga must be a free chain tail; g_dst must be a free chain head.
    if (ga >= cp.next_group.size() || cp.next_group[ga] != SIZE_MAX) return {};
    if (g_dst >= cp.prev_group.size() || cp.prev_group[g_dst] != SIZE_MAX) return {};

    const DAG& dag = *cp.part.dag;

    // t must already be a boundary output of ga.
    if (!is_boundary_output_of(cp.part.groups[ga].ops, t, dag)) return {};
    // op_a_dst (consumer of t) must be in g_dst, and t must be a boundary input there.
    if (!cp.part.groups[g_dst].ops.count(op_a_dst)) return {};
    if (!is_boundary_input_of(cp.part.groups[g_dst].ops, t, dag)) return {};

    // Chain-level acyclicity: same check as COUPLE.
    auto chain_a = cp.chain_of(ga);
    auto chain_b = cp.chain_of(g_dst);
    if (!acyclic_chain_merge(cp, chain_a, chain_b)) return {};

    // Split g_dst at bridge (op_a_dst, op_b_dst): side_a gets op_a_dst.
    auto sr = cp.part.eval_split(op_a_dst, op_b_dst, g_dst);
    if (!sr.feasible) return {};

    // op_a_dst (consumer of t) must end up in side_a so it can receive t from ga.
    if (!sr.side_a.count(op_a_dst)) return {};
    // t must remain a boundary input of the smaller side_a.
    if (!is_boundary_input_of(sr.side_a, t, dag)) return {};

    // Cost before = group_cost(ga) + group_cost(g_dst).
    double cost_before = cp.group_cost(ga) + cp.group_cost(g_dst);

    // Cost after:
    //   ga:     same entering, now retains t additionally
    //   side_a: enters with {t} (from ga coupling), no outgoing retain
    //   side_b: enters with nothing, inherits g_dst's retain context (for
    //           g_dst's old chain successor, if any)
    auto ga_enter     = cp.entering_for(ga);
    auto ga_retain    = cp.retain_for(ga);    // {} — ga is chain tail
    auto g_dst_retain = cp.retain_for(g_dst); // tensors retained for g_dst's next (if any)

    auto new_ga_retain = ga_retain;
    new_ga_retain.insert(t);
    const std::set<size_t> side_a_enter = {t};

    CostResult r_ga, r_side_a, r_side_b;
    if (cp.part.cache) {
        r_ga = cp.part.cache->evaluate_with_context(
            cp.part.groups[ga].ops, ga_enter, new_ga_retain,
            *cp.part.prob, dag);
        r_side_a = cp.part.cache->evaluate_with_context(
            sr.side_a, side_a_enter, {},
            *cp.part.prob, dag);
        r_side_b = cp.part.cache->evaluate_with_context(
            sr.side_b, {}, g_dst_retain,
            *cp.part.prob, dag);
    } else {
        if (!cp.part.groups[ga].sg) return {};
        r_ga = cp.part.groups[ga].sg->best_cost(ga_enter, new_ga_retain);
        auto sg_a = Subgraph::create(*cp.part.prob, dag,
                                     {sr.side_a.begin(), sr.side_a.end()});
        if (!sg_a) return {};
        r_side_a = sg_a->best_cost(side_a_enter, {});
        auto sg_b = Subgraph::create(*cp.part.prob, dag,
                                     {sr.side_b.begin(), sr.side_b.end()});
        if (!sg_b) return {};
        r_side_b = sg_b->best_cost({}, g_dst_retain);
    }
    if (!r_ga.feasible || !r_side_a.feasible || !r_side_b.feasible) return {};

    return {true, cost_before - (r_ga.latency + r_side_a.latency + r_side_b.latency)};
}

std::set<size_t> apply_force_retain(CoupledPartition& cp,
                                     size_t ga, size_t g_dst,
                                     size_t op_a_dst, size_t op_b_dst,
                                     size_t t) {
    if (ga >= cp.part.groups.size() || g_dst >= cp.part.groups.size()) return {};
    if (!cp.part.groups[ga].alive || !cp.part.groups[g_dst].alive) return {};

    // Save g_dst's outgoing chain link (g_dst is chain head, so no prev to save).
    size_t old_next = (g_dst < cp.next_group.size()) ? cp.next_group[g_dst] : SIZE_MAX;
    std::set<size_t> old_retain_to_next;
    if (old_next != SIZE_MAX) {
        auto it = cp.retained.find({g_dst, old_next});
        if (it != cp.retained.end())
            old_retain_to_next = it->second;
        cp.next_group[g_dst] = SIZE_MAX;
        if (old_next < cp.prev_group.size())
            cp.prev_group[old_next] = SIZE_MAX;
        cp.retained.erase({g_dst, old_next});
    }

    // Apply partition split: slot_a = g_dst (side_a, contains op_a_dst),
    //                        slot_b = new slot (side_b, contains op_b_dst).
    auto affected = partition_moves::apply_split(cp.part, op_a_dst, op_b_dst, g_dst);
    if (affected.empty()) {
        // Restore on failure.
        if (old_next != SIZE_MAX) {
            cp.next_group[g_dst] = old_next;
            if (old_next < cp.prev_group.size())
                cp.prev_group[old_next] = g_dst;
            if (!old_retain_to_next.empty())
                cp.retained[{g_dst, old_next}] = old_retain_to_next;
        }
        return {};
    }

    size_t slot_a = g_dst, slot_b = SIZE_MAX;
    for (auto idx : affected)
        if (idx != slot_a) { slot_b = idx; break; }
    if (slot_b == SIZE_MAX) return {};

    // Rebuild index + group DAG for the new slot.
    cp.part.rebuild_index();
#ifndef NDEBUG
    if (!cp.part.is_acyclic()) {
        std::cerr << "  BUG: apply_force_retain: split created cyclic partition"
                  << " ga=" << ga << " g_dst=" << g_dst
                  << " op_a=" << op_a_dst << " op_b=" << op_b_dst
                  << " t=" << t << "\n";
        for (size_t gi2 = 0; gi2 < cp.part.groups.size(); gi2++) {
            if (!cp.part.groups[gi2].alive) continue;
            std::cerr << "    G" << gi2 << " ops={";
            for (auto o : cp.part.groups[gi2].ops) {
                std::cerr << o;
                auto gs = cp.part.groups_of(o);
                int ac = 0; for (auto g : gs) if (cp.part.groups[g].alive) ac++;
                if (ac > 1) std::cerr << "*";
                std::cerr << ",";
            }
            std::cerr << "}\n";
        }
    }
#endif
    cp.part.rebuild_group_dag();

    // Extend coupling arrays for the new slot.
    while (cp.next_group.size() <= slot_b) cp.next_group.push_back(SIZE_MAX);
    while (cp.prev_group.size() <= slot_b) cp.prev_group.push_back(SIZE_MAX);

    // Create coupling: ga → slot_a via t.
    cp.next_group[ga]    = slot_a;
    cp.prev_group[slot_a] = ga;
    cp.retained[{ga, slot_a}].insert(t);

    // slot_b inherits g_dst's old outgoing chain link.
    if (old_next != SIZE_MAX) {
        cp.next_group[slot_b] = old_next;
        if (old_next < cp.prev_group.size())
            cp.prev_group[old_next] = slot_b;
        if (!old_retain_to_next.empty())
            cp.retained[{slot_b, old_next}] = old_retain_to_next;
    }

    affected.insert(ga);
    return affected;
}

std::set<size_t> apply_retain_force_split(CoupledPartition& cp,
                                           size_t g,
                                           size_t op_a, size_t op_b,
                                           size_t t) {
    if (g >= cp.part.groups.size() || !cp.part.groups[g].alive) return {};

    // Save G's outgoing chain link; incoming link stays with ga (reuses G's slot).
    size_t old_next = (g < cp.next_group.size()) ? cp.next_group[g] : SIZE_MAX;
    std::set<size_t> old_retain_to_next;
    if (old_next != SIZE_MAX) {
        auto it = cp.retained.find({g, old_next});
        if (it != cp.retained.end())
            old_retain_to_next = it->second;
        cp.next_group[g] = SIZE_MAX;
        if (old_next < cp.prev_group.size())
            cp.prev_group[old_next] = SIZE_MAX;
        cp.retained.erase({g, old_next});
    }

    // Apply partition split: ga = g (side_a), gb = new slot (side_b).
    auto affected = partition_moves::apply_split(cp.part, op_a, op_b, g);
    if (affected.empty()) {
        // Restore on failure
        if (old_next != SIZE_MAX) {
            cp.next_group[g] = old_next;
            if (old_next < cp.prev_group.size())
                cp.prev_group[old_next] = g;
            if (!old_retain_to_next.empty())
                cp.retained[{g, old_next}] = old_retain_to_next;
        }
        return {};
    }

    size_t ga = g, gb = SIZE_MAX;
    for (auto idx : affected)
        if (idx != ga) { gb = idx; break; }
    if (gb == SIZE_MAX) return {};

    // Rebuild index + group DAG for the new group slot.
    // Note: .sg is NOT built here — coupling eval uses the cache directly.
    cp.part.rebuild_index();
#ifndef NDEBUG
    if (!cp.part.is_acyclic()) {
        std::cerr << "  BUG: apply_retain_force_split: split created cyclic partition"
                  << " g=" << g << " op_a=" << op_a << " op_b=" << op_b
                  << " t=" << t << "\n";
        for (size_t gi2 = 0; gi2 < cp.part.groups.size(); gi2++) {
            if (!cp.part.groups[gi2].alive) continue;
            std::cerr << "    G" << gi2 << " ops={";
            for (auto o : cp.part.groups[gi2].ops) {
                std::cerr << o;
                auto gs = cp.part.groups_of(o);
                int ac = 0; for (auto g2 : gs) if (cp.part.groups[g2].alive) ac++;
                if (ac > 1) std::cerr << "*";
                std::cerr << ",";
            }
            std::cerr << "}\n";
        }
    }
#endif
    cp.part.rebuild_group_dag();

    // Extend coupling arrays for the new slot.
    while (cp.next_group.size() <= gb) cp.next_group.push_back(SIZE_MAX);
    while (cp.prev_group.size() <= gb) cp.prev_group.push_back(SIZE_MAX);

    // Chain: ... [G's prev] → ga → gb → old_next (if any)
    // ga's prev link is already correct (same slot as old G).
    cp.next_group[ga] = gb;
    cp.prev_group[gb] = ga;
    cp.retained[{ga, gb}].insert(t);

    if (old_next != SIZE_MAX) {
        cp.next_group[gb] = old_next;
        if (old_next < cp.prev_group.size())
            cp.prev_group[old_next] = gb;
        if (!old_retain_to_next.empty())
            cp.retained[{gb, old_next}] = old_retain_to_next;
    }

    return affected;
}

// ============================================================================
// Application
// ============================================================================

std::set<size_t> apply_couple(CoupledPartition& cp,
                               size_t ga, size_t gb, size_t t) {
    const bool existing_edge = (ga < cp.next_group.size() && cp.next_group[ga] == gb) &&
                                (gb < cp.prev_group.size() && cp.prev_group[gb] == ga);
    const bool new_edge      = (ga < cp.next_group.size() && cp.next_group[ga] == SIZE_MAX) &&
                                (gb < cp.prev_group.size() && cp.prev_group[gb] == SIZE_MAX);
    if (!existing_edge && !new_edge) return {};
    if (existing_edge) {
        auto it = cp.retained.find({ga, gb});
        if (it != cp.retained.end() && it->second.count(t)) return {};
    }
    cp.retained[{ga, gb}].insert(t);
    if (new_edge) {
        cp.next_group[ga] = gb;
        cp.prev_group[gb] = ga;
    }
    return {ga, gb};
}

std::set<size_t> apply_uncouple(CoupledPartition& cp,
                                 size_t ga, size_t gb, size_t t) {
    if (ga >= cp.next_group.size() || cp.next_group[ga] != gb) return {};
    auto it = cp.retained.find({ga, gb});
    if (it == cp.retained.end() || !it->second.count(t)) return {};
    it->second.erase(t);
    if (it->second.empty()) {
        cp.retained.erase(it);
        cp.next_group[ga] = SIZE_MAX;
        cp.prev_group[gb] = SIZE_MAX;
    }
    return {ga, gb};
}

// ============================================================================
// Greedy coupling descent — inner logic extracted for reuse
// ============================================================================

// Run one best-improving sweep over all coupling moves.
// Returns the number of moves applied (0 = local optimum reached).
static int coupling_greedy_pass(CoupledPartition& cp,
                                 const std::set<size_t>& feasibly_ret,
                                 size_t& n,
                                 CouplingTimePoint deadline) {
    const DAG& dag = *cp.part.dag;
    int n_moves = 0;

    bool improved = true;
    while (improved && SteadyClock::now() < deadline) {
        improved = false;

        CouplingEvalResult best;
        size_t best_ga = SIZE_MAX, best_gb = SIZE_MAX, best_t = SIZE_MAX;
        size_t best_g  = SIZE_MAX, best_op_a = SIZE_MAX, best_op_b = SIZE_MAX;
        enum class MoveType { Couple, Uncouple, RetainForceSplit } best_type = MoveType::Couple;

        // --- Enumerate COUPLE candidates ---
        for (size_t ga = 0; ga < n; ga++) {
            if (!cp.part.groups[ga].alive) continue;
            if (ga >= cp.next_group.size() || cp.next_group[ga] != SIZE_MAX) continue;

            for (auto t : feasibly_ret) {
                if (!is_boundary_output_of(cp.part.groups[ga].ops, t, dag)) continue;

                // Find alive groups that have t as a boundary input and are free chain heads.
                for (auto cop : dag.tensor_consumers[t]) {
                    for (auto gc : cp.part.groups_of(cop)) {
                        if (!cp.part.groups[gc].alive) continue;
                        if (gc >= cp.prev_group.size()) continue;
                        if (cp.prev_group[gc] != SIZE_MAX) continue;
                        if (gc == ga) continue;
                        if (!is_boundary_input_of(cp.part.groups[gc].ops, t, dag)) continue;

                        auto ev = eval_couple(cp, ga, gc, t);
                        if (ev.feasible && ev.saving > best.saving) {
                            best = ev;
                            best_ga = ga; best_gb = gc; best_t = t;
                            best_type = MoveType::Couple;
                        }
                    }
                }
            }
        }

        // --- Enumerate UNCOUPLE candidates ---
        for (auto& [edge, tensors] : cp.retained) {
            size_t ga = edge.first, gb = edge.second;
            for (auto t : tensors) {
                auto ev = eval_uncouple(cp, ga, gb, t);
                if (ev.feasible && ev.saving > best.saving) {
                    best = ev;
                    best_ga = ga; best_gb = gb; best_t = t;
                    best_type = MoveType::Uncouple;
                }
            }
        }

        // --- Enumerate RETAIN_FORCE_SPLIT candidates ---
        // For each retainable tensor t: find groups where t is internal
        // (producer and a consumer both in the same group), then try splitting
        // at the (producer, consumer) edge to expose t as a boundary output.
        for (auto t : feasibly_ret) {
            int prod_op = dag.tensor_producer[t];
            if (prod_op < 0) continue;
            size_t op_a = (size_t)prod_op;
            for (auto g : cp.part.groups_of(op_a)) {
                if (!cp.part.groups[g].alive) continue;
                // t already a boundary output → COUPLE handles it
                if (is_boundary_output_of(cp.part.groups[g].ops, t, dag)) continue;

                // Try each internal consumer as the cut target
                for (auto op_b : dag.tensor_consumers[t]) {
                    if (!cp.part.groups[g].ops.count(op_b)) continue;
                    auto ev = eval_retain_force_split(cp, g, op_a, op_b, t);
                    if (ev.feasible && ev.saving > best.saving) {
                        best = ev;
                        best_g = g; best_op_a = op_a; best_op_b = op_b; best_t = t;
                        best_type = MoveType::RetainForceSplit;
                    }
                }
            }
        }

        // Apply if improving.
        if (best.feasible && best.saving > 0.01) {
            if (best_type == MoveType::Uncouple)
                apply_uncouple(cp, best_ga, best_gb, best_t);
            else if (best_type == MoveType::Couple)
                apply_couple(cp, best_ga, best_gb, best_t);
            else {
                auto aff = apply_retain_force_split(cp, best_g, best_op_a, best_op_b, best_t);
                if (aff.empty()) { improved = false; break; }
                // n was computed before; update it since a new group was added.
                n = cp.part.groups.size();
            }
            n_moves++;
            improved = true;
        }
    }
    return n_moves;
}

// Run sweeps until no improvement. Returns final total_cost().
double coupling_greedy_descent(CoupledPartition& cp,
                                const std::set<size_t>& feasibly_ret,
                                CouplingTimePoint deadline) {
    size_t n = cp.part.groups.size();
    while (SteadyClock::now() < deadline)
        if (coupling_greedy_pass(cp, feasibly_ret, n, deadline) == 0) break;
    return cp.total_cost();
}

Solution coupling_search(const Problem& prob, const DAG& dag,
                          Partition part,
                          const std::set<size_t>& feasibly_ret,
                          CouplingTimePoint deadline) {
    (void)prob; (void)dag;  // embedded in part

    if (!part.groups.empty() && !part.groups[0].sg)
        part.finalize();

    CoupledPartition cp;
    cp.init_from(std::move(part));

    double init_cost = cp.total_cost();
    std::cerr << "  Coupling search: initial cost=" << init_cost
              << " retainable=" << feasibly_ret.size() << "\n";

    double final_cost = coupling_greedy_descent(cp, feasibly_ret, deadline);
    (void)final_cost;

    std::cerr << "  Coupling search: final cost=" << cp.total_cost() << "\n";
    return cp.to_solution();
}
