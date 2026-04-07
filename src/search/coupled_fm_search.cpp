#include "search/coupled_fm_search.h"
#include "search/feasibility.h"
#include "search/structural_ops.h"
#include "core/cost_cache.h"
#include <algorithm>
#include <cassert>
#include <iostream>

// Helper: tensor t is produced inside ops (regardless of internal consumers).
// Unlike is_boundary_output_of, this also matches ephemeral tensors that are
// produced+consumed internally — they are eligible for retention coupling.
//
// In practice, ephemeral tensors with external consumers rarely exist in valid
// uncoupled partitions (partition_has_gap rejects them).  The main path for
// creating such couplings is RETAIN_FORCE_SPLIT, which splits a group to
// expose an internal tensor.  Using is_produced_in here (instead of the
// stricter is_boundary_output_of) keeps the COUPLE move consistent with the
// relaxed gap check and handles edge cases from mutations.
static bool is_produced_in(const FlatSet<size_t>& ops, size_t t, const DAG& dag) {
    int prod = dag.tensor_producer[t];
    return prod >= 0 && ops.count((size_t)prod);
}

// ============================================================================
// Coupled cost evaluation helpers (forward-declared; defined after in_forward_chain)
// ============================================================================
static bool in_forward_chain(const CoupledPartition& cp, size_t start, size_t target);
static double coupled_recompute_saving(const CoupledPartition& cp, size_t op, size_t gb);
static double coupled_steal_saving(const CoupledPartition& cp, size_t op, size_t ga, size_t gb);
static double coupled_merge_saving(const CoupledPartition& cp, size_t ga, size_t gb);
static double coupled_tensor_merge_saving(const CoupledPartition& cp,
                                           const std::vector<size_t>& groups);

// ============================================================================
// best_coupled_move_for_op
// ============================================================================

CoupledFMMove best_coupled_move_for_op(const CoupledPartition& cp,
                                        size_t op,
                                        const FlatSet<size_t>& feasibly_ret,
                                        const FlatSet<size_t>& locked) {
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
#ifndef NDEBUG
                    if (cs > -1e17) {
                        CoupledPartition cp_copy = cp;
                        double before = cp_copy.total_cost();
                        CoupledFMMove cm;
                        cm.type = CoupledFMMove::STEAL;
                        cm.op = pm.op; cm.ga = pm.ga; cm.gb = pm.gb;
                        cm.saving = cs;
                        auto aff = apply_coupled_fm_move(cp_copy, cm);
                        double actual = aff.empty() ? -1e18 : before - cp_copy.total_cost();
                        double disc = cs - actual;
                        if (std::abs(disc) > 0.1 * std::max(1.0, std::abs(cs)) + 1.0) {
                            std::cerr << "  EVAL-TIME STEAL MISMATCH: predicted=" << cs
                                      << " simulated=" << actual
                                      << " op=" << pm.op << " ga=" << pm.ga
                                      << " gb=" << pm.gb << "\n";
                        }
                    }
#endif
                    best.saving = (cs > -1e17) ? cs : pm.saving;
                } else if (pm.type == FMMove::EJECT || pm.type == FMMove::INTERNAL_EJECT) {
                    double cs = coupled_steal_saving(cp, pm.op, pm.ga, SIZE_MAX);
                    best.saving = (cs > -1e17) ? cs : pm.saving;
                } else if (pm.type == FMMove::MERGE) {
                    double cs = coupled_merge_saving(cp, pm.ga, pm.gb);
#ifndef NDEBUG
                    // Verify prediction at eval time by simulating on a copy.
                    if (cs > -1e17) {
                        CoupledPartition cp_copy = cp;
                        double before = cp_copy.total_cost();
                        CoupledFMMove cm;
                        cm.type = CoupledFMMove::MERGE;
                        cm.op = pm.op; cm.ga = pm.ga; cm.gb = pm.gb;
                        cm.saving = cs;
                        auto aff = apply_coupled_fm_move(cp_copy, cm);
                        double actual = aff.empty() ? -1e18 : before - cp_copy.total_cost();
                        double disc = cs - actual;
                        if (std::abs(disc) > 0.1 * std::max(1.0, std::abs(cs)) + 1.0) {
                            std::cerr << "  EVAL-TIME MERGE MISMATCH: predicted=" << cs
                                      << " simulated=" << actual
                                      << " op=" << pm.op << " ga=" << pm.ga
                                      << " gb=" << pm.gb << "\n";
                        }
                    }
#endif
                    best.saving = (cs > -1e17) ? cs : pm.saving;
                } else if (pm.type == FMMove::TENSOR_MERGE) {
                    double cs = coupled_tensor_merge_saving(cp, pm.tensor_groups);
                    best.saving = (cs > -1e17) ? cs : pm.saving;
                } else if (pm.type == FMMove::RECOMPUTE) {
                    double cs = coupled_recompute_saving(cp, pm.op, pm.gb);
                    if (cs <= -1e17) { best = CoupledFMMove{}; }  // infeasible with coupling — reject
                    else best.saving = cs;
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
                    if (!is_produced_in(part.groups[ga].ops, t, dag)) continue;
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
                        if (!is_produced_in(part.groups[ga].ops, t, dag)) continue;
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
                    if (!is_produced_in(part.groups[p].ops, t, dag)) continue;
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
                if (!is_produced_in(part.groups[ga].ops, t, dag)) continue;

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
                                       const FlatSet<size_t>& ops,
                                       const FlatSet<size_t>& entering,
                                       const FlatSet<size_t>& retain) {
    if (ops.empty()) return 0.0;
    const Partition& part = cp.part;
    if (entering.empty() && retain.empty()) return part.eval_set(ops);
    if (!part.cache) return part.eval_set(ops);
    auto cr = part.cache->evaluate_with_context(ops, entering, retain, *part.prob, *part.dag);
    return cr.feasible ? cr.latency : 1e18;
}

// Coupled saving for RECOMPUTE(op into gb).
// Only gb changes — evaluate it with its coupling context.
// Source groups are unchanged (op stays in them).
static double coupled_recompute_saving(const CoupledPartition& cp,
                                        size_t op, size_t gb) {
    const Partition& part = cp.part;
    if (!part.groups[gb].alive) return -1e18;

    FlatSet<size_t> new_gb = part.groups[gb].ops;
    new_gb.insert(op);

    auto en_gb = cp.entering_for(gb);
    auto re_gb = cp.retain_for(gb);

    // After adding op, some entering tensors may become internal
    // (op produces them → now produced inside gb). Filter.
    if (!en_gb.empty()) {
        FlatSet<size_t> valid;
        for (auto t : en_gb)
            if (is_boundary_input_of(new_gb, t, *part.dag)) valid.insert(t);
        en_gb = std::move(valid);
    }
    // Retain tensors: op might consume a retained tensor internally,
    // but that doesn't change its boundary-output status — still produced
    // by gb for the next group. No filtering needed for retain.

    double old_cost = cp.group_cost(gb);
    double new_cost = eval_coupled_group_cost(cp, new_gb, en_gb, re_gb);
    if (new_cost >= 1e17) return -1e18;
    return old_cost - new_cost;
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

    FlatSet<size_t> new_ga = part.groups[ga].ops;  new_ga.erase(op);
    FlatSet<size_t> new_gb = is_eject ? FlatSet<size_t>{op} : part.groups[gb].ops;
    if (!is_eject) new_gb.insert(op);

    auto orig_en_ga = cp.entering_for(ga);
    auto orig_re_ga = cp.retain_for(ga);
    auto en_ga = orig_en_ga;
    auto re_ga = orig_re_ga;
    auto en_gb = is_eject ? FlatSet<size_t>{} : cp.entering_for(gb);
    auto re_gb = is_eject ? FlatSet<size_t>{} : cp.retain_for(gb);

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
        FlatSet<size_t> valid_re;
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
        FlatSet<size_t> v;
        for (auto t : en_ga)
            if (is_boundary_input_of(new_ga, t, *part.dag)) v.insert(t);
        en_ga = std::move(v);
    }
    if (!re_ga.empty()) {
        FlatSet<size_t> v;
        for (auto t : re_ga)
            if (is_boundary_output_of(new_ga, t, *part.dag)) v.insert(t);
        re_ga = std::move(v);
    }
    if (!en_gb.empty()) {
        FlatSet<size_t> v;
        for (auto t : en_gb)
            if (is_boundary_input_of(new_gb, t, *part.dag)) v.insert(t);
        en_gb = std::move(v);
    }
    if (!re_gb.empty() && !is_eject) {
        FlatSet<size_t> v;
        for (auto t : re_gb)
            if (is_boundary_output_of(new_gb, t, *part.dag)) v.insert(t);
        re_gb = std::move(v);
    }

    bool ga_dies = new_ga.empty();

    // If ga dies and gb's outgoing edge pointed to ga, gb loses its retain.
    if (ga_dies && !is_eject && cp.next_group[gb] == ga)
        re_gb = {};

    // Chain-neighbor cost deltas: when coupling context changes (entering/retain
    // shrank or dissolved), neighbors' costs change.  Compute precise deltas.
    double neighbor_delta = 0;

    size_t ga_prev = (ga < cp.prev_group.size()) ? cp.prev_group[ga] : SIZE_MAX;
    size_t ga_next = (ga < cp.next_group.size()) ? cp.next_group[ga] : SIZE_MAX;
    size_t gb_prev_orig = (!is_eject && gb < cp.prev_group.size()) ? cp.prev_group[gb] : SIZE_MAX;
    size_t gb_next_orig = (!is_eject && gb < cp.next_group.size()) ? cp.next_group[gb] : SIZE_MAX;

    // ga_prev: if ga's entering changed, ga_prev's retain context changed
    if (ga_prev != SIZE_MAX && ga_prev != gb && en_ga != orig_en_ga) {
        double old_prev = cp.group_cost(ga_prev);
        // ga_prev retains tensors for ga. If en_ga is now empty, prev loses all retain.
        // If en_ga shrank, prev retains only the remaining tensors.
        auto prev_retain = en_ga.empty() ? FlatSet<size_t>{} : en_ga;
        double new_prev = eval_coupled_group_cost(
            cp, part.groups[ga_prev].ops, cp.entering_for(ga_prev), prev_retain);
        neighbor_delta += old_prev - new_prev;
    }

    // ga_next chain cascade: if ga's retain changed, walk the chain forward
    // until the change stops propagating (a group's cost doesn't change,
    // or the chain ends).
    if (ga_next != SIZE_MAX && ga_next != gb && re_ga != orig_re_ga) {
        // Walk forward from ga_next along the chain.
        FlatSet<size_t> cur_enter = re_ga.empty() ? FlatSet<size_t>{} : re_ga;
        size_t cur = ga_next;
        size_t max_steps = part.groups.size();
        for (size_t step = 0; step < max_steps && cur != SIZE_MAX; step++) {
            if (cur == ga || cur == gb || !part.groups[cur].alive) break;
            double old_cur = cp.group_cost(cur);
            auto cur_retain = cp.retain_for(cur);
            double new_cur = eval_coupled_group_cost(
                cp, part.groups[cur].ops, cur_enter, cur_retain);
            if (std::abs(old_cur - new_cur) < 0.001) break;  // no change, stop
            neighbor_delta += old_cur - new_cur;
            // If cur retains for next, the next group's entering is cur's retain
            // (which didn't change — only cur's entering changed).
            // Cost change only propagates if cur's retain depends on its entering.
            // Actually, cur's retain set doesn't change — only its cost changes.
            // The next group's entering (= cur's retained tensors) is unchanged.
            // So the cascade STOPS here — next group's entering is the same.
            break;
        }
    }

    // gb's edges: if gb gains op, its retained tensors may change boundary status
    if (!is_eject) {
        auto orig_re_gb = cp.retain_for(gb);
        auto orig_en_gb = cp.entering_for(gb);

        // gb_next: if gb's retain changed (re_gb != orig_re_gb), next's entering changed
        if (gb_next_orig != SIZE_MAX && gb_next_orig != ga && re_gb != orig_re_gb) {
            double old_next = cp.group_cost(gb_next_orig);
            auto next_enter = re_gb.empty() ? FlatSet<size_t>{} : re_gb;
            double new_next = eval_coupled_group_cost(
                cp, part.groups[gb_next_orig].ops, next_enter, cp.retain_for(gb_next_orig));
            neighbor_delta += old_next - new_next;
        }

        // gb_prev: if gb's entering changed (en_gb != orig_en_gb), prev's retain changed
        if (gb_prev_orig != SIZE_MAX && gb_prev_orig != ga && en_gb != orig_en_gb) {
            double old_prev = cp.group_cost(gb_prev_orig);
            auto prev_retain = en_gb.empty() ? FlatSet<size_t>{} : en_gb;
            double new_prev = eval_coupled_group_cost(
                cp, part.groups[gb_prev_orig].ops, cp.entering_for(gb_prev_orig), prev_retain);
            neighbor_delta += old_prev - new_prev;
        }
    }

    double old_cost = cp.group_cost(ga) + (is_eject ? 0.0 : cp.group_cost(gb));

    // Account for disconnected components in the remainder.
    // First component keeps ga's coupling context, additional components are uncoupled.
    double new_ga_cost = 0;
    if (!new_ga.empty()) {
        auto comps = structural_ops::connected_components(new_ga, *part.dag);
        new_ga_cost = eval_coupled_group_cost(cp, comps[0], en_ga, re_ga);
        for (size_t i = 1; i < comps.size(); i++) {
            double c = part.eval_set(comps[i]);
            if (c >= 1e17) return -1e18;
            new_ga_cost += c;
        }
    }

    double new_gb_cost = eval_coupled_group_cost(cp, new_gb, en_gb, re_gb);
    if (new_ga_cost >= 1e17 || new_gb_cost >= 1e17) return -1e18;
    return old_cost - (new_ga_cost + new_gb_cost) + neighbor_delta;
}

// Coupled saving for MERGE(ga absorbs gb).
// Simulates fixup_coupling_merge to determine the merged group's coupling context.
static double coupled_merge_saving(const CoupledPartition& cp, size_t ga, size_t gb) {
    const Partition& part = cp.part;

    FlatSet<size_t> merged = part.groups[ga].ops;
    for (auto op : part.groups[gb].ops) merged.insert(op);

    size_t ga_prev = (ga < cp.prev_group.size()) ? cp.prev_group[ga] : SIZE_MAX;
    size_t ga_next = (ga < cp.next_group.size()) ? cp.next_group[ga] : SIZE_MAX;
    size_t gb_prev = (gb < cp.prev_group.size()) ? cp.prev_group[gb] : SIZE_MAX;
    size_t gb_next = (gb < cp.next_group.size()) ? cp.next_group[gb] : SIZE_MAX;

    // Dissolve internal ga↔gb coupling
    if (ga_next == gb) { ga_next = SIZE_MAX; gb_prev = SIZE_MAX; }
    if (ga_prev == gb) { ga_prev = SIZE_MAX; gb_next = SIZE_MAX; }

    // Merged entering: ga's prev (if any), else gb's prev (if no cycle)
    FlatSet<size_t> entering;
    if (ga_prev != SIZE_MAX) {
        auto it = cp.retained.find({ga_prev, ga});
        if (it != cp.retained.end()) entering = it->second;
    } else if (gb_prev != SIZE_MAX && gb_prev != ga && gb_prev != gb &&
               !in_forward_chain(cp, ga, gb_prev)) {
        auto it = cp.retained.find({gb_prev, gb});
        if (it != cp.retained.end()) entering = it->second;
    }

    // Merged retain: ga's next (if any), else gb's next (if no cycle)
    FlatSet<size_t> retain;
    if (ga_next != SIZE_MAX) {
        auto it = cp.retained.find({ga, ga_next});
        if (it != cp.retained.end()) retain = it->second;
    } else if (gb_next != SIZE_MAX && gb_next != ga && gb_next != gb &&
               !in_forward_chain(cp, gb_next, ga)) {
        auto it = cp.retained.find({gb, gb_next});
        if (it != cp.retained.end()) retain = it->second;
    }

    // Filter entering/retain to only include tensors still valid for the
    // merged group.  A tensor that was boundary between ga and gb becomes
    // internal after merge → no longer valid as entering or retained.
    const DAG& dag = *part.dag;
    {
        FlatSet<size_t> valid_entering;
        for (auto t : entering)
            if (is_boundary_input_of(merged, t, dag))
                valid_entering.insert(t);
        entering = std::move(valid_entering);
    }
    {
        FlatSet<size_t> valid_retain;
        for (auto t : retain)
            if (is_boundary_output_of(merged, t, dag))
                valid_retain.insert(t);
        retain = std::move(valid_retain);
    }

    double old_cost = cp.group_cost(ga) + cp.group_cost(gb);
    double merged_cost = eval_coupled_group_cost(cp, merged, entering, retain);
    if (merged_cost >= 1e17) return -1e18;

    // Chain-neighbor cost deltas: dissolved/stale coupling edges change neighbor costs.
    double neighbor_delta = 0;

    // Determine which prev/next the merged group ends up with
    size_t merged_prev = (ga_prev != SIZE_MAX) ? ga_prev :
                          ((gb_prev != SIZE_MAX && gb_prev != ga && gb_prev != gb &&
                            !in_forward_chain(cp, ga, gb_prev)) ? gb_prev : SIZE_MAX);
    size_t merged_next = (ga_next != SIZE_MAX) ? ga_next :
                          ((gb_next != SIZE_MAX && gb_next != ga && gb_next != gb &&
                            !in_forward_chain(cp, gb_next, ga)) ? gb_next : SIZE_MAX);

    // gb's prev loses its retain if NOT transferred (ga already had a prev)
    if (gb_prev != SIZE_MAX && gb_prev != ga && gb_prev != gb &&
        merged_prev != gb_prev) {
        double old_p = cp.group_cost(gb_prev);
        double new_p = eval_coupled_group_cost(
            cp, part.groups[gb_prev].ops, cp.entering_for(gb_prev), {});
        if (new_p >= 1e17) return -1e18;
        neighbor_delta += old_p - new_p;
    }

    // gb's next loses its entering if NOT transferred (ga already had a next)
    if (gb_next != SIZE_MAX && gb_next != ga && gb_next != gb &&
        merged_next != gb_next) {
        double old_n = cp.group_cost(gb_next);
        double new_n = eval_coupled_group_cost(
            cp, part.groups[gb_next].ops, {}, cp.retain_for(gb_next));
        if (new_n >= 1e17) return -1e18;
        neighbor_delta += old_n - new_n;
    }

    // Merged prev loses retain if entering became empty after filtering
    if (merged_prev != SIZE_MAX && entering.empty()) {
        // Check if prev originally HAD retain to the merged group
        bool had_retain = false;
        if (merged_prev == ga_prev && !cp.entering_for(ga).empty()) had_retain = true;
        if (merged_prev == gb_prev && !cp.entering_for(gb).empty()) had_retain = true;
        if (had_retain) {
            double old_p = cp.group_cost(merged_prev);
            double new_p = eval_coupled_group_cost(
                cp, part.groups[merged_prev].ops, cp.entering_for(merged_prev), {});
            if (new_p >= 1e17) return -1e18;
            neighbor_delta += old_p - new_p;
        }
    }

    // Merged next loses entering if retain became empty after filtering
    if (merged_next != SIZE_MAX && retain.empty()) {
        bool had_entering = false;
        if (merged_next == ga_next && !cp.retain_for(ga).empty()) had_entering = true;
        if (merged_next == gb_next && !cp.retain_for(gb).empty()) had_entering = true;
        if (had_entering) {
            double old_n = cp.group_cost(merged_next);
            double new_n = eval_coupled_group_cost(
                cp, part.groups[merged_next].ops, {}, cp.retain_for(merged_next));
            if (new_n >= 1e17) return -1e18;
            neighbor_delta += old_n - new_n;
        }
    }

    return old_cost - merged_cost + neighbor_delta;
}

// Coupled saving for TENSOR_MERGE(groups[0] survives, rest die).
static double coupled_tensor_merge_saving(const CoupledPartition& cp,
                                           const std::vector<size_t>& groups) {
    if (groups.empty()) return -1e18;
    size_t survivor = groups[0];

    FlatSet<size_t> merged = cp.part.groups[survivor].ops;
    for (size_t i = 1; i < groups.size(); i++)
        for (auto op : cp.part.groups[groups[i]].ops) merged.insert(op);

    double old_cost = 0;
    for (auto g : groups) old_cost += cp.group_cost(g);

    FlatSet<size_t> merge_set(groups.begin(), groups.end());

    // Start with survivor's current coupling context
    FlatSet<size_t> entering, retain;
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

    // Filter entering/retain to only include tensors still valid for the
    // merged group (tensors between merged groups become internal).
    const DAG& dag = *cp.part.dag;
    {
        FlatSet<size_t> valid;
        for (auto t : entering)
            if (is_boundary_input_of(merged, t, dag)) valid.insert(t);
        entering = std::move(valid);
    }
    {
        FlatSet<size_t> valid;
        for (auto t : retain)
            if (is_boundary_output_of(merged, t, dag)) valid.insert(t);
        retain = std::move(valid);
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

    const DAG& dag = *cp.part.dag;

    // Helper: filter retained tensors for boundary validity on the merged group.
    auto filter_retained = [&](const std::pair<size_t,size_t>& edge) {
        auto it = cp.retained.find(edge);
        if (it == cp.retained.end()) return;
        FlatSet<size_t> valid;
        for (auto t : it->second) {
            // Producer side: must still be boundary output of the producer group
            if (edge.first == ga) {
                if (!is_boundary_output_of(cp.part.groups[ga].ops, t, dag)) continue;
            } else if (cp.part.groups[edge.first].alive) {
                if (!is_boundary_output_of(cp.part.groups[edge.first].ops, t, dag)) continue;
            }
            // Consumer side: must still be boundary input of the consumer group
            if (edge.second == ga) {
                if (!is_boundary_input_of(cp.part.groups[ga].ops, t, dag)) continue;
            } else if (cp.part.groups[edge.second].alive) {
                if (!is_boundary_input_of(cp.part.groups[edge.second].ops, t, dag)) continue;
            }
            valid.insert(t);
        }
        if (valid.empty()) {
            cp.retained.erase(it);
            cp.next_group[edge.first] = SIZE_MAX;
            cp.prev_group[edge.second] = SIZE_MAX;
        } else {
            it->second = std::move(valid);
        }
    };

    // Transfer gb's external next link to ga (if ga has none and won't create a cycle).
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
            cp.prev_group[gb_next] = SIZE_MAX;
            cp.retained.erase({gb, gb_next});
        }
    }

    // Transfer gb's external prev link to ga (if ga has none and won't create a cycle).
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
            cp.next_group[gb_prev] = SIZE_MAX;
            cp.retained.erase({gb_prev, gb});
        }
    }

    // Validate all edges involving ga (the merged group) — retained tensors
    // that became internal after merge must be removed.
    if (cp.next_group[ga] != SIZE_MAX)
        filter_retained({ga, cp.next_group[ga]});
    if (cp.prev_group[ga] != SIZE_MAX)
        filter_retained({cp.prev_group[ga], ga});

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

    FlatSet<size_t> keep, transfer;
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
                FlatSet<size_t> keep, move;
                for (auto t : it->second) {
                    if (is_boundary_input_of(cp.part.groups[ga].ops, t, dag))
                        keep.insert(t);
                    else if (is_boundary_input_of(cp.part.groups[gb_new].ops, t, dag))
                        move.insert(t);
                    // else: consumer vanished — dissolve below
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
                FlatSet<size_t> keep, move;
                for (auto t : it->second) {
                    if (is_boundary_output_of(cp.part.groups[ga].ops, t, dag))
                        keep.insert(t);
                    else if (is_boundary_output_of(cp.part.groups[gb_new].ops, t, dag))
                        move.insert(t);
                    // else: producer vanished — dissolve below
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

FlatSet<size_t> apply_coupled_fm_move(CoupledPartition& cp,
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

        // ── Coupling fixup: move-specific logic + targeted edge validation ──
        //
        // Each move type handles coupling precisely:
        // 1. Transfer/redirect chain links as needed (MERGE/STEAL/SPLIT)
        // 2. Validate retained tensors on edges involving changed groups
        // No global invalidate_couplings — each fixup is surgical.

        // Save chain neighbors BEFORE fixup (they need refreshing).
        auto save_chain_neighbors = [&](size_t g) {
            if (g < cp.prev_group.size() && cp.prev_group[g] != SIZE_MAX)
                affected.insert(cp.prev_group[g]);
            if (g < cp.next_group.size() && cp.next_group[g] != SIZE_MAX)
                affected.insert(cp.next_group[g]);
        };

        // Validate a single coupling edge: remove retained tensors no longer
        // boundary-valid. Dissolve edge if retained set becomes empty.
        const DAG& dag = *cp.part.dag;
        auto validate_edge = [&](size_t from, size_t to) {
            auto it = cp.retained.find({from, to});
            if (it == cp.retained.end()) return;
            if (!cp.part.groups[from].alive || !cp.part.groups[to].alive) {
                cp.retained.erase(it);
                cp.next_group[from] = SIZE_MAX;
                cp.prev_group[to] = SIZE_MAX;
                affected.insert(from);
                affected.insert(to);
                return;
            }
            FlatSet<size_t> valid;
            for (auto t : it->second)
                if (is_produced_in(cp.part.groups[from].ops, t, dag) &&
                    is_boundary_input_of(cp.part.groups[to].ops, t, dag))
                    valid.insert(t);
            if (valid.empty()) {
                cp.retained.erase(it);
                cp.next_group[from] = SIZE_MAX;
                cp.prev_group[to] = SIZE_MAX;
                affected.insert(from);
                affected.insert(to);
            } else if (valid.size() < it->second.size()) {
                it->second = std::move(valid);
            }
        };

        // Validate all edges involving a specific group.
        auto validate_group_edges = [&](size_t g) {
            if (g < cp.next_group.size() && cp.next_group[g] != SIZE_MAX)
                validate_edge(g, cp.next_group[g]);
            if (g < cp.prev_group.size() && cp.prev_group[g] != SIZE_MAX)
                validate_edge(cp.prev_group[g], g);
        };

        if (m.type == CoupledFMMove::MERGE) {
            save_chain_neighbors(m.ga);
            save_chain_neighbors(m.gb);
            fixup_coupling_merge(cp, m.ga, m.gb);
            // fixup_coupling_merge already validates ga's edges inline
        } else if (m.type == CoupledFMMove::TENSOR_MERGE) {
            for (auto g : m.tensor_groups) save_chain_neighbors(g);
            for (size_t i = 1; i < m.tensor_groups.size(); i++)
                fixup_coupling_merge(cp, m.tensor_groups[0], m.tensor_groups[i]);
        } else if (m.type == CoupledFMMove::STEAL) {
            save_chain_neighbors(steal_ga);
            fixup_coupling_steal(cp, m.op, steal_ga, steal_gb);
            // Validate edges on from (steal_ga) — op's tensors may have changed
            // boundary status. Also validate to (steal_gb) if it gained coupling.
            validate_group_edges(steal_ga);
            if (steal_gb != SIZE_MAX) validate_group_edges(steal_gb);
        } else if (m.type == CoupledFMMove::SPLIT) {
            size_t gb_new = SIZE_MAX;
            for (auto g : affected)
                if (g != m.ga) { gb_new = g; break; }
            if (gb_new != SIZE_MAX)
                fixup_coupling_split(cp, m.ga, gb_new);
            // fixup_coupling_split already handles edge redirection
        } else {
            // EJECT, INTERNAL_EJECT, RECOMPUTE, TENSOR_EXTRACT, DE_RECOMPUTE, TENSOR_EXTRACT_SPLIT
            // No link transfer needed — just validate edges on changed groups.
            for (auto g : affected)
                validate_group_edges(g);
        }

        return affected;
    }

    // --- Coupling moves ---
    FlatSet<size_t> affected;

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
                                    const FlatSet<size_t>& feasibly_ret,
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

void CoupledActiveSet::refresh_after_move(const FlatSet<size_t>& affected_groups) {
    const auto& part = cp_->part;

    // Collect relevant groups: affected + their adjacents
    FlatSet<size_t> relevant;
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
    FlatSet<size_t> ops_to_refresh;
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
        FlatSet<size_t> groups_to_cover;
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
