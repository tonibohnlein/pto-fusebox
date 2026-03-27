#include "search/coupled_fm_search.h"
#include "search/feasibility.h"
#include "core/cost_cache.h"
#include <algorithm>
#include <cassert>
#include <iostream>

// ============================================================================
// Coupled cost evaluation helpers (forward-declared; defined after in_forward_chain)
// ============================================================================
static bool in_forward_chain(const CoupledPartition& cp, size_t start, size_t target);
static double coupled_steal_saving(const CoupledPartition& cp, size_t op, size_t ga, size_t gb);
static double coupled_merge_saving(const CoupledPartition& cp, size_t ga, size_t gb);
static double coupled_tensor_merge_saving(const CoupledPartition& cp,
                                           const std::vector<size_t>& groups);

// ============================================================================
// best_coupled_move_for_op
// ============================================================================

CoupledFMMove best_coupled_move_for_op(const CoupledPartition& cp,
                                        size_t op,
                                        const std::set<size_t>& feasibly_ret,
                                        const std::set<size_t>& locked) {
    CoupledFMMove best;

    // --- Partition moves (delegate to existing best_move_for) ---
    // Group-level acyclicity checks in best_move_for are sufficient for most
    // moves.  MERGE and TENSOR_MERGE additionally need a chain-level cycle
    // check: when the two groups are already in different non-trivial chains,
    // merging them can create a chain-level cycle that acyclic_merge_local
    // misses.  Other partition moves cannot create chain-level cycles (they
    // only remove or invalidate existing coupling edges).
    {
        FMMove pm = best_move_for(cp.part, op, locked);
        if (pm.valid()) {
            bool chain_ok = true;
            if (pm.type == FMMove::MERGE) {
                chain_ok = acyclic_chain_merge_groups(cp, {pm.ga, pm.gb});
            } else if (pm.type == FMMove::TENSOR_MERGE) {
                chain_ok = acyclic_chain_merge_groups(cp, pm.tensor_groups);
            }
            if (chain_ok) {
                best.type                = static_cast<CoupledFMMove::Type>(static_cast<int>(pm.type));
                best.op                  = pm.op;
                best.ga                  = pm.ga;
                best.gb                  = pm.gb;
                best.op2                 = pm.op2;
                best.tensor_groups       = pm.tensor_groups;
                best.tensor_consumer_ops = pm.tensor_consumer_ops;

                // Recompute saving using coupled costs: partition-level eval uses
                // uncoupled group costs, but total_cost() uses evaluate_with_context.
                // Groups in coupling chains have entering/retain tensors that
                // constrain tile configs — ignoring them causes large gain mismatches.
                if (pm.type == FMMove::STEAL) {
                    double cs = coupled_steal_saving(cp, pm.op, pm.ga, pm.gb);
                    best.saving = (cs > -1e17) ? cs : pm.saving;
                } else if (pm.type == FMMove::EJECT || pm.type == FMMove::INTERNAL_EJECT) {
                    double cs = coupled_steal_saving(cp, pm.op, pm.ga, SIZE_MAX);
                    best.saving = (cs > -1e17) ? cs : pm.saving;
                } else if (pm.type == FMMove::MERGE) {
                    double cs = coupled_merge_saving(cp, pm.ga, pm.gb);
                    best.saving = (cs > -1e17) ? cs : pm.saving;
                } else if (pm.type == FMMove::TENSOR_MERGE) {
                    double cs = coupled_tensor_merge_saving(cp, pm.tensor_groups);
                    best.saving = (cs > -1e17) ? cs : pm.saving;
                } else {
                    best.saving = pm.saving;
                }
            }
        }
    }

    if (locked.count(op)) return best;

    const auto& part        = cp.part;
    const auto& dag         = *part.dag;
    const auto& prob        = *part.prob;
    auto groups_of_x        = part.groups_of(op);

    for (auto ga : groups_of_x) {
        if (!part.groups[ga].alive) continue;
        if (ga >= cp.next_group.size()) continue;

        // --- COUPLE: create a new coupling edge OR add a tensor to an existing one. ---
        {
            size_t h = cp.next_group[ga];
            if (h == SIZE_MAX) {
                // New edge: ga is a free chain tail — find candidate gb groups.
                for (auto t : feasibly_ret) {
                    if (!is_boundary_output_of(part.groups[ga].ops, t, dag)) continue;
                    for (auto cop : dag.tensor_consumers[t]) {
                        auto cg_list = part.groups_of(cop);
                        for (auto gb : cg_list) {
                            if (gb == ga) continue;
                            if (!part.groups[gb].alive) continue;
                            if (gb >= cp.prev_group.size()) continue;
                            if (cp.prev_group[gb] != SIZE_MAX) continue;
                            if (!is_boundary_input_of(part.groups[gb].ops, t, dag)) continue;

                            auto r = eval_couple(cp, ga, gb, t);
                            if (r.feasible && r.saving > best.saving) {
                                best.type   = CoupledFMMove::COUPLE;
                                best.op     = op;
                                best.ga     = ga;
                                best.gb     = gb;
                                best.tensor = t;
                                best.saving = r.saving;
                            }
                        }
                    }
                }
            } else {
                // Existing outgoing edge ga→h: try adding more retained tensors.
                if (part.groups[h].alive && h < cp.prev_group.size()) {
                    auto rit = cp.retained.find({ga, h});
                    for (auto t : feasibly_ret) {
                        if (rit != cp.retained.end() && rit->second.count(t)) continue;
                        if (!is_boundary_output_of(part.groups[ga].ops, t, dag)) continue;
                        if (!is_boundary_input_of(part.groups[h].ops, t, dag)) continue;
                        auto r = eval_couple(cp, ga, h, t);
                        if (r.feasible && r.saving > best.saving) {
                            best.type   = CoupledFMMove::COUPLE;
                            best.op     = op;
                            best.ga     = ga;
                            best.gb     = h;
                            best.tensor = t;
                            best.saving = r.saving;
                        }
                    }
                }
            }
        }

        // --- COUPLE (incoming edge): add more tensors to the existing p→ga edge. ---
        {
            size_t p = cp.prev_group[ga];
            if (p != SIZE_MAX && part.groups[p].alive && p < cp.next_group.size()) {
                auto rit = cp.retained.find({p, ga});
                for (auto t : feasibly_ret) {
                    if (rit != cp.retained.end() && rit->second.count(t)) continue;
                    if (!is_boundary_output_of(part.groups[p].ops, t, dag)) continue;
                    if (!is_boundary_input_of(part.groups[ga].ops, t, dag)) continue;
                    auto r = eval_couple(cp, p, ga, t);
                    if (r.feasible && r.saving > best.saving) {
                        best.type   = CoupledFMMove::COUPLE;
                        best.op     = op;
                        best.ga     = p;
                        best.gb     = ga;
                        best.tensor = t;
                        best.saving = r.saving;
                    }
                }
            }
        }

        // --- UNCOUPLE: remove tensor from an existing coupling edge adjacent to ga ---
        // Outgoing edge: ga → next
        {
            size_t h = cp.next_group[ga];
            if (h != SIZE_MAX) {
                auto it = cp.retained.find({ga, h});
                if (it != cp.retained.end()) {
                    for (auto t : it->second) {
                        auto r = eval_uncouple(cp, ga, h, t);
                        if (r.feasible && r.saving > best.saving) {
                            best.type   = CoupledFMMove::UNCOUPLE;
                            best.op     = op;
                            best.ga     = ga;
                            best.gb     = h;
                            best.tensor = t;
                            best.saving = r.saving;
                        }
                    }
                }
            }
        }
        // Incoming edge: prev → ga
        {
            size_t p = cp.prev_group[ga];
            if (p != SIZE_MAX) {
                auto it = cp.retained.find({p, ga});
                if (it != cp.retained.end()) {
                    for (auto t : it->second) {
                        auto r = eval_uncouple(cp, p, ga, t);
                        if (r.feasible && r.saving > best.saving) {
                            best.type   = CoupledFMMove::UNCOUPLE;
                            best.op     = op;
                            best.ga     = p;
                            best.gb     = ga;
                            best.tensor = t;
                            best.saving = r.saving;
                        }
                    }
                }
            }
        }

        // --- RETAIN_FORCE_SPLIT: op produces a feasibly-retainable tensor
        //     that is internal to ga (not yet a boundary output). ---
        for (auto t : prob.ops[op].outputs) {
            if (!feasibly_ret.count(t)) continue;
            if (is_boundary_output_of(part.groups[ga].ops, t, dag)) continue;  // use COUPLE

            // t is internal to ga. Look for consumers of t also in ga.
            for (auto cop : dag.tensor_consumers[t]) {
                if (!part.groups[ga].ops.count(cop)) continue;
                // Try split at bridge edge (op → cop)
                auto r = eval_retain_force_split(cp, ga, op, cop, t);
                if (r.feasible && r.saving > best.saving) {
                    best.type   = CoupledFMMove::RETAIN_FORCE_SPLIT;
                    best.op     = op;
                    best.op2    = cop;
                    best.ga     = ga;
                    best.tensor = t;
                    best.saving = r.saving;
                }
            }
        }

        // --- FORCE_RETAIN: t is already a boundary output of ga.
        //     COUPLE(ga, g_dst) may be infeasible because g_dst is too large
        //     to hold all of t in fast memory. Split g_dst at a bridge adjacent
        //     to t's consumer so the smaller side_a can be coupled to ga via t. ---
        if (cp.next_group[ga] == SIZE_MAX) {  // ga must be chain tail
            for (auto t : prob.ops[op].outputs) {
                if (!feasibly_ret.count(t)) continue;
                if (!is_boundary_output_of(part.groups[ga].ops, t, dag)) continue;

                for (auto cop : dag.tensor_consumers[t]) {
                    for (auto g_dst : part.groups_of(cop)) {
                        if (g_dst == ga) continue;
                        if (!part.groups[g_dst].alive) continue;
                        if (g_dst >= cp.prev_group.size()) continue;
                        if (cp.prev_group[g_dst] != SIZE_MAX) continue;  // must be chain head
                        if (!part.groups[g_dst].ops.count(cop)) continue;
                        if (part.groups[g_dst].ops.size() < 2) continue;
                        if (!is_boundary_input_of(part.groups[g_dst].ops, t, dag)) continue;

                        // Try splitting g_dst at each bridge (cop, v) where cop ends
                        // up in side_a (receives the coupling from ga via t).
                        for (auto v : dag.op_neighbors[cop]) {
                            if (!part.groups[g_dst].ops.count(v)) continue;
                            auto r = eval_force_retain(cp, ga, g_dst, cop, v, t);
                            if (r.feasible && r.saving > best.saving) {
                                best.type   = CoupledFMMove::FORCE_RETAIN;
                                best.op     = op;
                                best.ga     = ga;
                                best.gb     = g_dst;
                                best.tensor = t;
                                best.op2    = cop;
                                best.op3    = v;
                                best.saving = r.saving;
                            }
                        }
                    }
                }
            }
        }
    }

    return best;
}

// ============================================================================
// Coupling fixup helpers for partition move application
// ============================================================================

// Returns true if 'target' appears in the forward chain starting at 'start'
// (follows next_group; bounded to prevent infinite loops on malformed state).
static bool in_forward_chain(const CoupledPartition& cp, size_t start, size_t target) {
    size_t cur = start;
    size_t n   = cp.part.groups.size() + 1;
    for (size_t steps = 0; steps < n && cur != SIZE_MAX; steps++) {
        if (cur == target) return true;
        cur = (cur < cp.next_group.size()) ? cp.next_group[cur] : SIZE_MAX;
    }
    return false;
}

// ============================================================================
// Coupled cost helpers: compute move savings using evaluate_with_context so
// they match what CoupledPartition::total_cost() will observe.
// ============================================================================

static double eval_coupled_group_cost(const CoupledPartition& cp,
                                       const std::set<size_t>& ops,
                                       const std::set<size_t>& entering,
                                       const std::set<size_t>& retain) {
    if (ops.empty()) return 0.0;
    const Partition& part = cp.part;
    if (entering.empty() && retain.empty()) return part.eval_set(ops);
    if (!part.cache) return part.eval_set(ops);
    auto cr = part.cache->evaluate_with_context(ops, entering, retain, *part.prob, *part.dag);
    return cr.feasible ? cr.latency : 1e18;
}

// Coupled saving for STEAL(op, from=ga, to=gb) or EJECT [gb==SIZE_MAX].
// Evaluates new group costs with coupling contexts and simulates the
// coupling-edge transfer that fixup_coupling_steal would perform.
// Also accounts for chain-neighbor cost changes when coupling edges are
// dissolved (predecessor loses retain, successor loses entering).
static double coupled_steal_saving(const CoupledPartition& cp,
                                    size_t op, size_t ga, size_t gb) {
    const Partition& part = cp.part;
    bool is_eject = (gb == SIZE_MAX);

    std::set<size_t> new_ga = part.groups[ga].ops;  new_ga.erase(op);
    std::set<size_t> new_gb = is_eject ? std::set<size_t>{op} : part.groups[gb].ops;
    if (!is_eject) new_gb.insert(op);

    auto orig_en_ga = cp.entering_for(ga);
    auto orig_re_ga = cp.retain_for(ga);
    auto en_ga = orig_en_ga;
    auto re_ga = orig_re_ga;
    auto en_gb = is_eject ? std::set<size_t>{} : cp.entering_for(gb);
    auto re_gb = is_eject ? std::set<size_t>{} : cp.retain_for(gb);

    // Simulate fixup_coupling_steal: if op is the last consumer of ALL retained
    // entering tensors in ga, and gb has no incoming coupling, transfer the edge.
    if (!en_ga.empty() && cp.prev_group[ga] != SIZE_MAX &&
        (is_eject || cp.prev_group[gb] == SIZE_MAX)) {
        bool any_keeps = false;
        for (auto t : en_ga) {
            bool op_uses = false;
            for (auto inp : part.prob->ops[op].inputs)
                if (inp == t) { op_uses = true; break; }
            if (!op_uses) { any_keeps = true; break; }
            for (auto cop : part.dag->tensor_consumers[t]) {
                if (cop == op) continue;
                if (part.groups[ga].ops.count(cop)) { any_keeps = true; break; }
            }
            if (any_keeps) break;
        }
        if (!any_keeps) {
            if (!is_eject) en_gb = en_ga;  // full transfer to gb
            en_ga = {};                      // ga loses entering
        }
    }

    // Simulate: if op is the only producer of a retained output in ga,
    // that retention becomes invalid (invalidate_couplings will remove it).
    if (!re_ga.empty()) {
        std::set<size_t> valid_re;
        for (auto t : re_ga) {
            bool op_produces = false;
            for (auto out : part.prob->ops[op].outputs)
                if (out == t) { op_produces = true; break; }
            if (!op_produces) { valid_re.insert(t); continue; }
            bool other_producer = false;
            for (auto cop : new_ga) {
                for (auto out2 : part.prob->ops[cop].outputs)
                    if (out2 == t) { other_producer = true; break; }
                if (other_producer) break;
            }
            if (other_producer) valid_re.insert(t);
        }
        re_ga = std::move(valid_re);
    }

    // Simulate invalidate_couplings: filter en_ga/re_ga for boundary validity
    // on the post-move group ops (consumer side for entering, producer side for
    // retain). Also filter en_gb for tensors that are no longer boundary inputs
    // of new_gb (e.g. tensor became internal after absorbing its producer).
    if (!en_ga.empty()) {
        std::set<size_t> v;
        for (auto t : en_ga)
            if (is_boundary_input_of(new_ga, t, *part.dag)) v.insert(t);
        en_ga = std::move(v);
    }
    if (!re_ga.empty()) {
        std::set<size_t> v;
        for (auto t : re_ga)
            if (is_boundary_output_of(new_ga, t, *part.dag)) v.insert(t);
        re_ga = std::move(v);
    }
    if (!en_gb.empty()) {
        std::set<size_t> v;
        for (auto t : en_gb)
            if (is_boundary_input_of(new_gb, t, *part.dag)) v.insert(t);
        en_gb = std::move(v);
    }
    if (!re_gb.empty() && !is_eject) {
        std::set<size_t> v;
        for (auto t : re_gb)
            if (is_boundary_output_of(new_gb, t, *part.dag)) v.insert(t);
        re_gb = std::move(v);
    }

    bool ga_dies = new_ga.empty();

    // If ga dies and gb's outgoing edge pointed to ga, gb loses its retain.
    if (ga_dies && !is_eject && cp.next_group[gb] == ga)
        re_gb = {};

    // Chain-neighbor cost deltas: when coupling edges are dissolved, the
    // predecessor (loses retain) and successor (loses entering) change cost.
    double neighbor_delta = 0;

    // ga_prev: if ga's incoming coupling dissolves (en_ga became empty)
    size_t ga_prev = (ga < cp.prev_group.size()) ? cp.prev_group[ga] : SIZE_MAX;
    if (ga_prev != SIZE_MAX && ga_prev != gb &&
        !orig_en_ga.empty() && en_ga.empty()) {
        double old_prev = cp.group_cost(ga_prev);
        double new_prev = eval_coupled_group_cost(
            cp, part.groups[ga_prev].ops, cp.entering_for(ga_prev), {});
        neighbor_delta += old_prev - new_prev;
    }

    // ga_next: if ga's outgoing coupling dissolves (re_ga became empty)
    size_t ga_next = (ga < cp.next_group.size()) ? cp.next_group[ga] : SIZE_MAX;
    if (ga_next != SIZE_MAX && ga_next != gb &&
        !orig_re_ga.empty() && re_ga.empty()) {
        double old_next = cp.group_cost(ga_next);
        double new_next = eval_coupled_group_cost(
            cp, part.groups[ga_next].ops, {}, cp.retain_for(ga_next));
        neighbor_delta += old_next - new_next;
    }

    double old_cost = cp.group_cost(ga) + (is_eject ? 0.0 : cp.group_cost(gb));
    double new_ga_cost = eval_coupled_group_cost(cp, new_ga, en_ga, re_ga);
    double new_gb_cost = eval_coupled_group_cost(cp, new_gb, en_gb, re_gb);
    if (new_ga_cost >= 1e17 || new_gb_cost >= 1e17) return -1e18;
    return old_cost - (new_ga_cost + new_gb_cost) + neighbor_delta;
}

// Coupled saving for MERGE(ga absorbs gb).
// Simulates fixup_coupling_merge to determine the merged group's coupling context.
static double coupled_merge_saving(const CoupledPartition& cp, size_t ga, size_t gb) {
    const Partition& part = cp.part;

    std::set<size_t> merged = part.groups[ga].ops;
    for (auto op : part.groups[gb].ops) merged.insert(op);

    size_t ga_prev = (ga < cp.prev_group.size()) ? cp.prev_group[ga] : SIZE_MAX;
    size_t ga_next = (ga < cp.next_group.size()) ? cp.next_group[ga] : SIZE_MAX;
    size_t gb_prev = (gb < cp.prev_group.size()) ? cp.prev_group[gb] : SIZE_MAX;
    size_t gb_next = (gb < cp.next_group.size()) ? cp.next_group[gb] : SIZE_MAX;

    // Dissolve internal ga↔gb coupling
    if (ga_next == gb) { ga_next = SIZE_MAX; gb_prev = SIZE_MAX; }
    if (ga_prev == gb) { ga_prev = SIZE_MAX; gb_next = SIZE_MAX; }

    // Merged entering: ga's prev (if any), else gb's prev (if no cycle)
    std::set<size_t> entering;
    if (ga_prev != SIZE_MAX) {
        auto it = cp.retained.find({ga_prev, ga});
        if (it != cp.retained.end()) entering = it->second;
    } else if (gb_prev != SIZE_MAX && gb_prev != ga && gb_prev != gb &&
               !in_forward_chain(cp, ga, gb_prev)) {
        auto it = cp.retained.find({gb_prev, gb});
        if (it != cp.retained.end()) entering = it->second;
    }

    // Merged retain: ga's next (if any), else gb's next (if no cycle)
    std::set<size_t> retain;
    if (ga_next != SIZE_MAX) {
        auto it = cp.retained.find({ga, ga_next});
        if (it != cp.retained.end()) retain = it->second;
    } else if (gb_next != SIZE_MAX && gb_next != ga && gb_next != gb &&
               !in_forward_chain(cp, gb_next, ga)) {
        auto it = cp.retained.find({gb, gb_next});
        if (it != cp.retained.end()) retain = it->second;
    }

    double old_cost = cp.group_cost(ga) + cp.group_cost(gb);
    double merged_cost = eval_coupled_group_cost(cp, merged, entering, retain);
    if (merged_cost >= 1e17) return -1e18;
    return old_cost - merged_cost;
}

// Coupled saving for TENSOR_MERGE(groups[0] survives, rest die).
static double coupled_tensor_merge_saving(const CoupledPartition& cp,
                                           const std::vector<size_t>& groups) {
    if (groups.empty()) return -1e18;
    size_t survivor = groups[0];

    std::set<size_t> merged = cp.part.groups[survivor].ops;
    for (size_t i = 1; i < groups.size(); i++)
        for (auto op : cp.part.groups[groups[i]].ops) merged.insert(op);

    double old_cost = 0;
    for (auto g : groups) old_cost += cp.group_cost(g);

    std::set<size_t> merge_set(groups.begin(), groups.end());

    // Start with survivor's current coupling context
    std::set<size_t> entering, retain;
    {
        size_t sv_prev = (survivor < cp.prev_group.size()) ? cp.prev_group[survivor] : SIZE_MAX;
        size_t sv_next = (survivor < cp.next_group.size()) ? cp.next_group[survivor] : SIZE_MAX;
        if (sv_prev != SIZE_MAX && !merge_set.count(sv_prev)) {
            auto it = cp.retained.find({sv_prev, survivor});
            if (it != cp.retained.end()) entering = it->second;
        }
        if (sv_next != SIZE_MAX && !merge_set.count(sv_next)) {
            auto it = cp.retained.find({survivor, sv_next});
            if (it != cp.retained.end()) retain = it->second;
        }
    }

    // Check dead groups for external links that would transfer to survivor
    for (size_t i = 1; i < groups.size(); i++) {
        size_t g = groups[i];
        size_t g_prev = (g < cp.prev_group.size()) ? cp.prev_group[g] : SIZE_MAX;
        size_t g_next = (g < cp.next_group.size()) ? cp.next_group[g] : SIZE_MAX;
        if (entering.empty() && g_prev != SIZE_MAX && !merge_set.count(g_prev) &&
            !in_forward_chain(cp, g_prev, survivor)) {
            auto it = cp.retained.find({g_prev, g});
            if (it != cp.retained.end()) entering = it->second;
        }
        if (retain.empty() && g_next != SIZE_MAX && !merge_set.count(g_next) &&
            !in_forward_chain(cp, g_next, survivor)) {
            auto it = cp.retained.find({g, g_next});
            if (it != cp.retained.end()) retain = it->second;
        }
    }

    double merged_cost = eval_coupled_group_cost(cp, merged, entering, retain);
    if (merged_cost >= 1e17) return -1e18;
    return old_cost - merged_cost;
}

// After MERGE(ga, gb → ga): transfer gb's external chain links to ga.
static void fixup_coupling_merge(CoupledPartition& cp,
                                  size_t ga, size_t gb) {
    if (ga >= cp.next_group.size() || gb >= cp.next_group.size()) return;

    // Save gb's external links before we clear them
    size_t gb_next = cp.next_group[gb];
    size_t gb_prev = cp.prev_group[gb];

    // Dissolve internal ga↔gb coupling edge
    if (cp.next_group[ga] == gb) {
        cp.next_group[ga] = SIZE_MAX;
        cp.prev_group[gb] = SIZE_MAX;
        cp.retained.erase({ga, gb});
        gb_prev = SIZE_MAX;  // already cleared above
    }
    if (cp.prev_group[ga] == gb) {
        cp.prev_group[ga] = SIZE_MAX;
        cp.next_group[gb] = SIZE_MAX;
        cp.retained.erase({gb, ga});
        gb_next = SIZE_MAX;
    }

    // Transfer gb's external next link to ga (if ga has none and won't create a cycle).
    // A cycle would occur if ga is already in gb_next's forward chain
    // (e.g. chain was: gb → gb_next → ... → ga).
    if (gb_next != SIZE_MAX && gb_next != ga && gb_next != gb) {
        if (cp.next_group[ga] == SIZE_MAX && !in_forward_chain(cp, gb_next, ga)) {
            cp.next_group[ga] = gb_next;
            cp.prev_group[gb_next] = ga;
            auto it = cp.retained.find({gb, gb_next});
            if (it != cp.retained.end()) {
                cp.retained[{ga, gb_next}] = std::move(it->second);
                cp.retained.erase(it);
            }
        } else {
            // ga already has a next, or transfer would create a cycle — dissolve gb's outgoing edge
            cp.prev_group[gb_next] = SIZE_MAX;
            cp.retained.erase({gb, gb_next});
        }
    }

    // Transfer gb's external prev link to ga (if ga has none and won't create a cycle).
    // A cycle would occur if gb_prev is already in ga's forward chain
    // (e.g. chain was: ga → ... → gb_prev → gb).
    if (gb_prev != SIZE_MAX && gb_prev != ga && gb_prev != gb) {
        if (cp.prev_group[ga] == SIZE_MAX && !in_forward_chain(cp, ga, gb_prev)) {
            cp.prev_group[ga] = gb_prev;
            cp.next_group[gb_prev] = ga;
            auto it = cp.retained.find({gb_prev, gb});
            if (it != cp.retained.end()) {
                cp.retained[{gb_prev, ga}] = std::move(it->second);
                cp.retained.erase(it);
            }
        } else {
            // ga already has a prev, or transfer would create a cycle — dissolve gb's incoming edge
            cp.next_group[gb_prev] = SIZE_MAX;
            cp.retained.erase({gb_prev, gb});
        }
    }

    // Clear gb's now-dead coupling slots
    cp.next_group[gb] = SIZE_MAX;
    cp.prev_group[gb] = SIZE_MAX;
}

// After STEAL(op from ga to gb): if op is the sole consumer of a retained
// input tensor t in ga, transfer the coupling edge (prev → ga) to (prev → gb).
static void fixup_coupling_steal(CoupledPartition& cp,
                                  size_t op, size_t ga, size_t gb) {
    const auto& dag = *cp.part.dag;
    const auto& prob = *cp.part.prob;

    if (ga >= cp.prev_group.size() || gb >= cp.prev_group.size()) return;

    size_t ga_prev = cp.prev_group[ga];
    if (ga_prev == SIZE_MAX) return;

    auto it = cp.retained.find({ga_prev, ga});
    if (it == cp.retained.end()) return;

    std::set<size_t> keep, transfer;
    for (auto t : it->second) {
        // Check if op is the only consumer of t in ga (after steal, op is in gb)
        bool op_consumes_t = false;
        for (auto inp : prob.ops[op].inputs)
            if (inp == t) { op_consumes_t = true; break; }
        if (!op_consumes_t) { keep.insert(t); continue; }

        // Check if any other alive op in ga also consumes t
        bool others_consume = false;
        for (auto cop : dag.tensor_consumers[t]) {
            if (cop == op) continue;
            if (cp.part.groups[ga].ops.count(cop)) { others_consume = true; break; }
        }

        if (others_consume) {
            keep.insert(t);  // t is still needed by ga
        } else {
            transfer.insert(t);  // op was the sole consumer → follow op to gb
        }
    }

    if (transfer.empty()) return;

    // Can only transfer if gb has no incoming coupling
    if (cp.prev_group[gb] != SIZE_MAX) {
        // gb already coupled to something else — keep all on ga
        return;
    }

    if (keep.empty()) {
        // Move the entire edge from (ga_prev → ga) to (ga_prev → gb)
        cp.next_group[ga_prev] = gb;
        cp.prev_group[gb]      = ga_prev;
        cp.prev_group[ga]      = SIZE_MAX;
        cp.retained[{ga_prev, gb}] = std::move(transfer);
        cp.retained.erase(it);
    } else {
        // Split: keep on ga, also add edge to gb for transferred tensors
        it->second = std::move(keep);
        cp.prev_group[gb]      = ga_prev;
        // ga_prev can only have one next → conflict: keep ga_prev → ga
        // since ga still needs it; don't create the gb edge
        // (transferred tensors are unfortunately lost here)
    }
}

// After SPLIT(ga → side_a kept at ga, side_b in gb_new):
// Transfer coupling edges to whichever side now owns the relevant endpoint.
//
// Incoming edge (prev → ga): if the consumer of the retained tensor moved to
//   side_b, redirect prev → gb_new.  If tensors split between sides, keep all
//   on ga (can't have two outgoing links from prev).
// Outgoing edge (ga → next): if the producer of the retained tensor moved to
//   side_b, redirect gb_new → next.
static void fixup_coupling_split(CoupledPartition& cp, size_t ga, size_t gb_new) {
    const auto& dag = *cp.part.dag;

    // ── Incoming edge: ga_prev → ga ──
    if (ga < cp.prev_group.size()) {
        size_t ga_prev = cp.prev_group[ga];
        if (ga_prev != SIZE_MAX) {
            auto it = cp.retained.find({ga_prev, ga});
            if (it != cp.retained.end()) {
                std::set<size_t> keep, move;
                for (auto t : it->second) {
                    if (is_boundary_input_of(cp.part.groups[ga].ops, t, dag))
                        keep.insert(t);
                    else if (is_boundary_input_of(cp.part.groups[gb_new].ops, t, dag))
                        move.insert(t);
                    // else: consumer vanished — let invalidate_couplings clean up
                }
                if (!keep.empty()) {
                    it->second = std::move(keep);
                    // tensors in 'move' are lost (prev can only have one outgoing link)
                } else if (!move.empty()) {
                    // All consumers moved to gb_new — redirect entire edge
                    cp.next_group[ga_prev] = gb_new;
                    cp.prev_group[gb_new]  = ga_prev;
                    cp.prev_group[ga]      = SIZE_MAX;
                    cp.retained[{ga_prev, gb_new}] = std::move(move);
                    cp.retained.erase(it);
                } else {
                    // Nothing valid remains — dissolve
                    cp.next_group[ga_prev] = SIZE_MAX;
                    cp.prev_group[ga]      = SIZE_MAX;
                    cp.retained.erase(it);
                }
            }
        }
    }

    // ── Outgoing edge: ga → ga_next ──
    if (ga < cp.next_group.size()) {
        size_t ga_next = cp.next_group[ga];
        if (ga_next != SIZE_MAX) {
            auto it = cp.retained.find({ga, ga_next});
            if (it != cp.retained.end()) {
                std::set<size_t> keep, move;
                for (auto t : it->second) {
                    if (is_boundary_output_of(cp.part.groups[ga].ops, t, dag))
                        keep.insert(t);
                    else if (is_boundary_output_of(cp.part.groups[gb_new].ops, t, dag))
                        move.insert(t);
                    // else: producer vanished — let invalidate_couplings clean up
                }
                if (!keep.empty()) {
                    it->second = std::move(keep);
                    // tensors in 'move' are lost (ga_next already has incoming from ga)
                } else if (!move.empty()) {
                    // All producers moved to gb_new — redirect entire edge
                    cp.next_group[ga]      = SIZE_MAX;
                    cp.next_group[gb_new]  = ga_next;
                    cp.prev_group[ga_next] = gb_new;
                    cp.retained[{gb_new, ga_next}] = std::move(move);
                    cp.retained.erase(it);
                } else {
                    // Nothing valid remains — dissolve
                    cp.next_group[ga]      = SIZE_MAX;
                    cp.prev_group[ga_next] = SIZE_MAX;
                    cp.retained.erase(it);
                }
            }
        }
    }
}

// ============================================================================
// apply_coupled_fm_move
// ============================================================================

std::set<size_t> apply_coupled_fm_move(CoupledPartition& cp,
                                        const CoupledFMMove& m) {
    if (!m.valid()) return {};

    if (m.is_partition_move()) {
        FMMove fm = m.as_fm_move();

        const bool is_steal  = (m.type == CoupledFMMove::STEAL ||
                                 m.type == CoupledFMMove::INTERNAL_EJECT ||
                                 m.type == CoupledFMMove::EJECT);
        const size_t steal_ga = (is_steal) ? m.ga : SIZE_MAX;
        const size_t steal_gb = (m.type == CoupledFMMove::STEAL) ? m.gb : SIZE_MAX;

        auto affected = apply_fm_move(cp.part, fm);
        if (affected.empty()) return {};

        // Extend coupling arrays for any new groups created by SPLIT /
        // RETAIN_FORCE_SPLIT / RECOMPUTE (new group appended to groups vector).
        while (cp.next_group.size() < cp.part.groups.size())
            cp.next_group.push_back(SIZE_MAX);
        while (cp.prev_group.size() < cp.part.groups.size())
            cp.prev_group.push_back(SIZE_MAX);

        if (m.type == CoupledFMMove::MERGE) {
            // Two-group merge: transfer gb's chain links to ga.
            fixup_coupling_merge(cp, m.ga, m.gb);
        } else if (m.type == CoupledFMMove::TENSOR_MERGE) {
            // N-group merge: tensor_groups[0] survives, rest die.
            // Call fixup_coupling_merge once per dead group so their chain
            // links are transferred to the survivor.
            for (size_t i = 1; i < m.tensor_groups.size(); i++)
                fixup_coupling_merge(cp, m.tensor_groups[0], m.tensor_groups[i]);
        } else if (m.type == CoupledFMMove::STEAL) {
            fixup_coupling_steal(cp, m.op, steal_ga, steal_gb);
            // Then clean up anything else broken (e.g. ga's outgoing edge if
            // op produced a retained tensor)
            cp.invalidate_couplings();
        } else if (m.type == CoupledFMMove::SPLIT) {
            // Find the new group created by the split (the group in affected
            // that is not ga).
            size_t gb_new = SIZE_MAX;
            for (auto g : affected)
                if (g != m.ga) { gb_new = g; break; }
            if (gb_new != SIZE_MAX)
                fixup_coupling_split(cp, m.ga, gb_new);
            cp.invalidate_couplings();
        } else {
            // EJECT, INTERNAL_EJECT, RECOMPUTE, TENSOR_EXTRACT, DE_RECOMPUTE
            cp.invalidate_couplings();
        }

        return affected;
    }

    // --- Coupling moves ---
    std::set<size_t> affected;

    switch (m.type) {
        case CoupledFMMove::COUPLE:
            affected = apply_couple(cp, m.ga, m.gb, m.tensor);
            break;
        case CoupledFMMove::UNCOUPLE:
            affected = apply_uncouple(cp, m.ga, m.gb, m.tensor);
            break;
        case CoupledFMMove::RETAIN_FORCE_SPLIT:
            affected = apply_retain_force_split(cp, m.ga, m.op, m.op2, m.tensor);
            if (!affected.empty()) {
                while (cp.next_group.size() < cp.part.groups.size())
                    cp.next_group.push_back(SIZE_MAX);
                while (cp.prev_group.size() < cp.part.groups.size())
                    cp.prev_group.push_back(SIZE_MAX);
            }
            break;
        case CoupledFMMove::FORCE_RETAIN:
            affected = apply_force_retain(cp, m.ga, m.gb, m.op2, m.op3, m.tensor);
            if (!affected.empty()) {
                while (cp.next_group.size() < cp.part.groups.size())
                    cp.next_group.push_back(SIZE_MAX);
                while (cp.prev_group.size() < cp.part.groups.size())
                    cp.prev_group.push_back(SIZE_MAX);
            }
            break;
        default:
            break;
    }

    return affected;
}

// ============================================================================
// CoupledActiveSet
// ============================================================================

CoupledActiveSet::CoupledActiveSet(const CoupledPartition& cp,
                                    const std::set<size_t>& feasibly_ret,
                                    double floor)
    : cp_(&cp),
      feasibly_ret_(&feasibly_ret),
      heap_(cp.part.prob->num_ops()),
      floor_(floor) {}

void CoupledActiveSet::recompute_and_update(size_t op) {
    if (locked_.count(op)) return;
    auto move = best_coupled_move_for_op(*cp_, op, *feasibly_ret_, locked_);
    if (move.valid() && move.saving > -floor_)
        heap_.push_or_update(op, move);
    else
        heap_.remove(op);
}

void CoupledActiveSet::activate(size_t op) {
    if (locked_.count(op)) return;
    recompute_and_update(op);
}

void CoupledActiveSet::activate_group_ops(size_t gi) {
    const auto& part = cp_->part;
    if (!part.groups[gi].alive) return;
    for (auto op : part.border_ops(gi))
        recompute_and_update(op);
    if (part.groups[gi].ops.size() >= 3 && part.groups[gi].ops.size() <= 15) {
        for (auto op : part.internal_ops(gi))
            recompute_and_update(op);
    }
}

void CoupledActiveSet::activate_border(size_t gi) {
    const auto& part = cp_->part;
    if (!part.groups[gi].alive) return;
    for (auto op : part.border_ops(gi))
        recompute_and_update(op);
}

std::optional<CoupledFMMove> CoupledActiveSet::pop_best() {
    while (!heap_.empty()) {
        auto m = heap_.pop_best();
        if (!m) return std::nullopt;
        if (!locked_.count(m->op)) {
            locked_.insert(m->op);
            return m;
        }
    }
    return std::nullopt;
}

void CoupledActiveSet::lock(size_t op) {
    locked_.insert(op);
    heap_.remove(op);
}

void CoupledActiveSet::lock_all(const std::vector<size_t>& ops) {
    for (auto op : ops) {
        locked_.insert(op);
        heap_.remove(op);
    }
}

void CoupledActiveSet::refresh_after_move(const std::set<size_t>& affected_groups) {
    const auto& part = cp_->part;

    // Collect relevant groups: affected + their adjacents
    std::set<size_t> relevant;
    for (auto gi : affected_groups) {
        relevant.insert(gi);
        if (gi < part.groups.size() && part.groups[gi].alive) {
            auto adj = part.adjacent_groups(gi);
            relevant.insert(adj.begin(), adj.end());
        }
    }

    // Also include chain neighbours (groups coupled to affected groups)
    for (auto gi : affected_groups) {
        if (gi < cp_->next_group.size() && cp_->next_group[gi] != SIZE_MAX)
            relevant.insert(cp_->next_group[gi]);
        if (gi < cp_->prev_group.size() && cp_->prev_group[gi] != SIZE_MAX)
            relevant.insert(cp_->prev_group[gi]);
    }

    // Collect ops to refresh
    std::set<size_t> ops_to_refresh;
    for (auto gi : relevant) {
        if (gi >= part.groups.size() || !part.groups[gi].alive) continue;
        for (auto op : part.groups[gi].ops) {
            ops_to_refresh.insert(op);
            for (auto nbr : part.dag->op_neighbors[op])
                ops_to_refresh.insert(nbr);
        }
    }

    // Expand to cover full groups: a DAG neighbor may share a group with ops
    // that have no direct DAG connection to the affected area, yet can have stale
    // cached moves because acyclicity uses global forward_reachable.
    {
        std::set<size_t> groups_to_cover;
        for (auto op : ops_to_refresh)
            for (auto gi : part.groups_of(op))
                if (gi < part.groups.size() && part.groups[gi].alive)
                    groups_to_cover.insert(gi);
        for (auto gi : groups_to_cover)
            for (auto op : part.groups[gi].ops)
                ops_to_refresh.insert(op);
    }

    for (auto op : ops_to_refresh)
        recompute_and_update(op);

    for (auto gi : relevant)
        activate_group_ops(gi);
}
