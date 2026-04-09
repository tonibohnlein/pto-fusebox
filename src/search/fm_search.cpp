#include "search/fm_search.h"
#include "search/partition_moves.h"
#include "search/structural_ops.h"
#include "core/group_dag.h"
#include <algorithm>
#include <iostream>
#include <cassert>

// ============================================================================
// eject_creates_ephemeral_gap: local check for EJECT / INTERNAL_EJECT.
//
// When ejecting op from group gx, the remainder may have tensors that are
// ephemeral (produced + consumed internally) but now have a NEW external
// consumer (the ejected op). We only check tensors consumed by the ejected
// op — fully local, O(|inputs of op| * small) via groups_of().
// ============================================================================

static bool eject_creates_ephemeral_gap(const Partition& part, size_t op, size_t gx) {
    const auto& dag = *part.dag;
    const auto& prob = *part.prob;
    const auto& gx_ops = part.groups[gx].ops;

    for (auto t : prob.ops[op].inputs) {
        int prod = dag.tensor_producer[t];
        if (prod < 0) continue;
        if (!gx_ops.count((size_t)prod)) continue;  // producer outside gx → unaffected

        // Producer is in gx. Will T be ephemeral in the remainder?
        // Only if another op in the remainder also consumes T.
        bool consumed_in_remainder = false;
        for (auto cop : dag.tensor_consumers[t])
            if (cop != op && gx_ops.count(cop)) { consumed_in_remainder = true; break; }
        if (!consumed_in_remainder) continue;  // T becomes boundary output → safe

        // T will be ephemeral in remainder. The ejected op needs T externally.
        // Check if T is available as boundary output from another group.
        bool available = false;
        for (auto gj : part.groups_of((size_t)prod)) {
            if (gj == gx || !part.groups[gj].alive) continue;
            bool consumed_in_gj = false;
            for (auto c2 : dag.tensor_consumers[t])
                if (part.groups[gj].ops.count(c2)) { consumed_in_gj = true; break; }
            if (!consumed_in_gj) { available = true; break; }
        }
        if (!available) return true;
    }
    return false;
}

// ============================================================================
// split_creates_ephemeral_gap_local: local check for SPLIT.
//
// When splitting group gx into side_a and side_b, tensors that cross the
// boundary may become ephemeral on one side with external consumers on the
// other. Only checks tensors at the split boundary — O(|side| * small).
// ============================================================================

static bool split_creates_ephemeral_gap_local(
    const Partition& part,
    const FlatSet<size_t>& side_a, const FlatSet<size_t>& side_b,
    size_t gx) {
    const auto& dag = *part.dag;
    const auto& prob = *part.prob;

    // Check each side: does it have an ephemeral tensor needed by the other side?
    auto check_side = [&](const FlatSet<size_t>& inside,
                          const FlatSet<size_t>& outside) -> bool {
        for (auto op : outside) {
            for (auto t : prob.ops[op].inputs) {
                int prod = dag.tensor_producer[t];
                if (prod < 0 || !inside.count((size_t)prod)) continue;

                // Producer in inside, consumer (op) in outside.
                // Is T ephemeral in inside (another consumer also in inside)?
                bool consumed_in_inside = false;
                for (auto cop : dag.tensor_consumers[t])
                    if (cop != op && inside.count(cop))
                        { consumed_in_inside = true; break; }
                if (!consumed_in_inside) continue;

                // T ephemeral in inside, op in outside needs it.
                // Available from another group?
                bool available = false;
                for (auto gj : part.groups_of((size_t)prod)) {
                    if (gj == gx || !part.groups[gj].alive) continue;
                    bool consumed_in_gj = false;
                    for (auto c2 : dag.tensor_consumers[t])
                        if (part.groups[gj].ops.count(c2))
                            { consumed_in_gj = true; break; }
                    if (!consumed_in_gj) { available = true; break; }
                }
                if (!available) return true;
            }
        }
        return false;
    };

    return check_side(side_a, side_b) || check_side(side_b, side_a);
}

// ============================================================================
// best_move_for: evaluate all candidate moves for op
// ============================================================================

FMMove best_move_for(const Partition& part, size_t op,
                     const FlatSet<size_t>& locked,
                     const GroupDAG* gdag) {
    if (locked.count(op)) return {};

    FMMove best;
    const auto& dag = *part.dag;
    auto groups_of_x = part.groups_of(op);
    if (groups_of_x.empty()) return {};

    bool op_in_multiple = groups_of_x.size() > 1;

    // --- DE_RECOMPUTE: if op is recomputed (in multiple groups), evaluate
    // removing it from each group ---
    if (op_in_multiple) {
        for (auto gx : groups_of_x) {
            if (!part.groups[gx].alive) continue;
            if (gdag ? gdag->eval_de_recompute(part, op, gx)
                    : !part.acyclic_de_recompute_local(op, gx)) continue;
            auto dr = partition_moves::eval_de_recompute(part, gx, op);
            if (dr.feasible && dr.saving > best.saving)
                best = FMMove{FMMove::DE_RECOMPUTE, op, gx, SIZE_MAX, SIZE_MAX, dr.saving};
        }
    }

    // --- Border moves: EJECT ---
    for (auto gx : groups_of_x) {
        if (!part.is_border_op(op, gx)) continue;
        if (eject_creates_ephemeral_gap(part, op, gx)) continue;
        // Ejecting a recomputed op is equivalent to DE_RECOMPUTE: the op's
        // outputs become external deps for the remainder.  Must check
        // acyclicity to prevent cycles through the op's other group(s).
        if (op_in_multiple && (gdag ? gdag->eval_de_recompute(part, op, gx)
                                    : !part.acyclic_de_recompute_local(op, gx))) continue;

        auto er = part.eval_eject(op, gx);
        if (er.feasible && er.saving > best.saving)
            best = FMMove{FMMove::EJECT, op, gx, SIZE_MAX, SIZE_MAX, er.saving};
    }

    // --- Internal moves: INTERNAL_EJECT and SPLIT ---
    for (auto gx : groups_of_x) {
        if (part.is_border_op(op, gx)) continue;
        if (part.groups[gx].ops.size() < 3 || part.groups[gx].ops.size() > 15) continue;

        // INTERNAL_EJECT
        if (!eject_creates_ephemeral_gap(part, op, gx)) {
            if (op_in_multiple && (gdag ? gdag->eval_de_recompute(part, op, gx)
                                    : !part.acyclic_de_recompute_local(op, gx))) continue;
            auto er = part.eval_eject(op, gx);
            if (er.feasible && er.saving > best.saving) {
                FMMove candidate;
                candidate.type = FMMove::INTERNAL_EJECT;
                candidate.op = op; candidate.ga = gx;
                candidate.saving = er.saving;
                best = candidate;
            }
        }

        // SPLIT: cheap O(|side|) pre-filter only
        for (auto v : dag.op_neighbors[op]) {
            if (!part.groups[gx].ops.count(v)) continue;
            size_t u_lo = std::min(op, v), u_hi = std::max(op, v);

            auto sr = part.eval_split(u_lo, u_hi, gx);
            if (sr.feasible && sr.saving > best.saving) {
                if (!split_creates_ephemeral_gap_local(part, sr.side_a, sr.side_b, gx)
                    && (gdag ? !gdag->eval_split(part, sr.side_a, sr.side_b, gx)
                             : part.acyclic_split_local(sr.side_a, sr.side_b, gx)))
                    best = FMMove{FMMove::SPLIT, op, gx, SIZE_MAX, v, sr.saving};
            }
        }
    }

    // --- STEAL / RECOMPUTE / MERGE: op pulled into adjacent group ---
    // Structured as (target → source) with pre-computed costs for efficiency.
    FlatSet<size_t> adj_groups;
    for (auto nbr : dag.op_neighbors[op])
        for (auto gi : part.groups_of(nbr))
            if (part.groups[gi].alive && !part.groups[gi].ops.count(op))
                adj_groups.insert(gi);

    // Pre-compute cost of each source group after losing op.
    // Account for disconnected components in the remainder.
    struct GxCost { size_t gx; double cost; bool valid; };
    // groups_of_x is typically 1-2 entries; flat vector beats unordered_map
    std::vector<GxCost> gx_without_op;
    gx_without_op.reserve(groups_of_x.size());
    for (auto gx : groups_of_x) {
        if (!part.groups[gx].alive) continue;
        FlatSet<size_t> rem = part.groups[gx].ops;
        rem.erase(op);
        if (rem.empty()) {
            gx_without_op.push_back({gx, 0.0, true});
        } else {
            auto comps = structural_ops::connected_components(rem, *part.dag);
            double total = 0;
            bool ok = true;
            for (auto& comp : comps) {
                double c = part.eval_set(comp);
                if (c >= 1e17) { ok = false; break; }
                total += c;
            }
            gx_without_op.push_back({gx, total, ok});
        }
    }

    // Typically <10 pairs; flat vector with linear scan beats std::set
    std::vector<std::pair<size_t,size_t>> merge_checked;
    for (auto gi : adj_groups) {
        FlatSet<size_t> new_gi = part.groups[gi].ops;
        new_gi.insert(op);
        double new_gi_cost = part.eval_set(new_gi);
        if (new_gi_cost >= 1e17) continue;

        for (auto gx : groups_of_x) {
            if (gx == gi || !part.groups[gx].alive) continue;

            // STEAL: move op from gx into gi
            {
                const GxCost* gxc = nullptr;
                for (auto& e : gx_without_op)
                    if (e.gx == gx) { gxc = &e; break; }
                if (gxc && gxc->valid) {
                    double new_gx_cost = gxc->cost;
                    double saving = (part.groups[gi].cost + part.groups[gx].cost)
                                    - (new_gi_cost + new_gx_cost);
                    if (saving > best.saving &&
                        (gdag ? !gdag->eval_steal(part, op, gx, gi)
                              : part.acyclic_steal_local(op, gx, gi))) {
                        // Gap check: reuse new_gi (already built) + reconstruct new_gx
                        FlatSet<size_t> new_gx = part.groups[gx].ops;
                        new_gx.erase(op);
                        std::vector<FlatSet<size_t>> comps;
                        comps.push_back(new_gi);
                        if (!new_gx.empty()) comps.push_back(new_gx);
                        if (!part.split_creates_ephemeral_gap(comps, {gx, gi}))
                            best = FMMove{FMMove::STEAL, op, gx, gi, SIZE_MAX, saving};
                    }
                }
            }

            // MERGE: dedup via canonical pair key
            auto pair_key = std::make_pair(std::min(gi, gx), std::max(gi, gx));
            if (std::find(merge_checked.begin(), merge_checked.end(), pair_key)
                == merge_checked.end()) {
                merge_checked.push_back(pair_key);
                if (gdag ? !gdag->eval_merge(gi, gx)
                        : part.acyclic_merge_local(gi, gx)) {
                    FlatSet<size_t> merged_ops = part.groups[gi].ops;
                    merged_ops.insert(part.groups[gx].ops.begin(),
                                      part.groups[gx].ops.end());
                    if (!part.creates_ephemeral_gap(merged_ops, gi, gx)) {
                        double merged_cost = part.eval_set(merged_ops);
                        if (merged_cost < 1e17) {
                            double saving = part.groups[gi].cost + part.groups[gx].cost - merged_cost;
                            if (saving > best.saving)
                                best = FMMove{FMMove::MERGE, op, gi, gx, SIZE_MAX, saving};
                        }
                    }
                }
            }
        }

        // RECOMPUTE: copy op into gi (stays in source groups too)
        if (gdag ? !gdag->eval_recompute(part, op, gi)
                : part.acyclic_recompute_local(op, gi)) {
            auto rr = partition_moves::eval_recompute(part, op, gi);
            size_t gx = groups_of_x.empty() ? 0 : groups_of_x[0];
            if (rr.feasible && rr.saving > best.saving)
                best = FMMove{FMMove::RECOMPUTE, op, gx, gi, SIZE_MAX, rr.saving};
        }
    }

    // --- Tensor-centric moves: TENSOR_MERGE and TENSOR_EXTRACT ---
    {
        FlatSet<size_t> tensors_checked;
        for (auto gx : groups_of_x) {
            for (auto t : part.prob->ops[op].inputs) {
                if (tensors_checked.count(t)) continue;
                tensors_checked.insert(t);

                auto& consumers = dag.tensor_consumers[t];
                if (consumers.size() < 2) continue;

                FlatSet<size_t> consumer_groups;
                std::vector<size_t> consumer_ops_vec;
                for (auto cop : consumers) {
                    for (auto cg : part.groups_of(cop))
                        consumer_groups.insert(cg);
                    consumer_ops_vec.push_back(cop);
                }

                int prod = dag.tensor_producer[t];
                if (prod >= 0) {
                    auto& pg = part.groups_of((size_t)prod);
                    if (!pg.empty())
                        consumer_groups.insert(pg[0]);
                }

                if (consumer_groups.size() < 2) continue;

                bool has_locked = false;
                for (auto cg : consumer_groups) {
                    if (!part.groups[cg].alive) { has_locked = true; break; }
                    for (auto cop : part.groups[cg].ops) {
                        if (locked.count(cop)) { has_locked = true; break; }
                    }
                    if (has_locked) break;
                }
                if (has_locked) continue;

                std::vector<size_t> group_list(consumer_groups.begin(),
                                               consumer_groups.end());

                // TENSOR_MERGE: local group-level BFS at eval time
                if (gdag ? !gdag->merge_creates_cycle(group_list)
                        : part.acyclic_merge_local(group_list)) {
                    auto tmr = partition_moves::eval_tensor_merge(part, group_list);
                    if (tmr.feasible && tmr.saving > best.saving) {
                        FMMove candidate;
                        candidate.type = FMMove::TENSOR_MERGE;
                        candidate.op = op;
                        candidate.op2 = t;
                        candidate.saving = tmr.saving;
                        candidate.tensor_groups = group_list;
                        best = candidate;
                    }
                }

                // TENSOR_EXTRACT: Kahn's at eval time (rare move, high rejection)
                {
                    FlatSet<size_t> extract_ops(consumer_ops_vec.begin(),
                                                 consumer_ops_vec.end());
                    if (prod >= 0) extract_ops.insert((size_t)prod);

                    if (gdag ? !gdag->eval_extract(part, extract_ops)
                            : part.acyclic_extract_local(extract_ops)) {
                        auto ter = partition_moves::eval_tensor_extract(
                            part, extract_ops, group_list);
                        if (ter.feasible && ter.saving > best.saving) {
                            FMMove candidate;
                            candidate.type = FMMove::TENSOR_EXTRACT;
                            candidate.op = op;
                            candidate.op2 = t;
                            candidate.saving = ter.saving;
                            candidate.tensor_groups = group_list;
                            candidate.tensor_consumer_ops.assign(
                                extract_ops.begin(), extract_ops.end());
                            best = candidate;
                        }
                    }
                }

                // TENSOR_EXTRACT_SPLIT: split consumers into balanced sub-groups
                if (consumer_ops_vec.size() >= 2) {
                    auto sr = partition_moves::eval_tensor_extract_split(
                        part, t, consumer_ops_vec, group_list);
                    if (sr.feasible && sr.saving > best.saving) {
                        FMMove candidate;
                        candidate.type = FMMove::TENSOR_EXTRACT_SPLIT;
                        candidate.op = op;
                        candidate.op2 = t;
                        candidate.saving = sr.saving;
                        candidate.tensor_groups = group_list;
                        candidate.split_extract_result = std::move(sr);
                        best = candidate;
                    }
                }
            }
        }
    }

    // --- FORCE_RECOMPUTE: for each input tensor of op with ≥2 consumers,
    // evaluate extracting producer + consumers into {P, C_i} pairs ---
    {
        FlatSet<size_t> fr_tensors_checked;
        for (auto gx : groups_of_x) {
            for (auto t : part.prob->ops[op].inputs) {
                if (fr_tensors_checked.count(t)) continue;
                fr_tensors_checked.insert(t);

                int prod = dag.tensor_producer[t];
                if (prod < 0) continue;
                if (dag.tensor_consumers[t].size() < 2) continue;

                auto frr = partition_moves::eval_force_recompute(part, t);
                if (!frr.feasible) continue;
                if (frr.saving > best.saving) {
                    bool has_locked = false;
                    if (locked.count(frr.prod_op)) has_locked = true;
                    for (auto cop : frr.consumer_ops)
                        if (locked.count(cop)) { has_locked = true; break; }
                    if (has_locked) continue;

                    FMMove candidate;
                    candidate.type = FMMove::FORCE_RECOMPUTE;
                    candidate.op = op;
                    candidate.op2 = t;
                    candidate.saving = frr.saving;
                    candidate.tensor_consumer_ops.assign(
                        frr.consumer_ops.begin(), frr.consumer_ops.end());
                    best = candidate;
                }
            }
        }
    }

    return best;
}

// ============================================================================
// apply_fm_move
//
// Pre-mutation acyclicity checks for MERGE/STEAL/RECOMPUTE/TENSOR_MERGE.
// TENSOR_EXTRACT: all remainder costs validated before any mutation.
// No post-hoc is_acyclic() needed — no partial mutation on failure.
// ============================================================================

FlatSet<size_t> apply_fm_move(Partition& part, const FMMove& m) {
    if (!m.valid()) return {};
    FlatSet<size_t> affected;

    switch (m.type) {
        case FMMove::STEAL: {
            // FMMove::STEAL: op moves FROM m.ga INTO m.gb
            affected = partition_moves::apply_steal(part, m.op, m.ga, m.gb);
            if (affected.empty()) return {};
            break;
        }
        case FMMove::MERGE: {
            // ga survives, gb killed
            affected = partition_moves::apply_merge(part, m.ga, m.gb);
            if (affected.empty()) return {};
            break;
        }
        case FMMove::RECOMPUTE: {
            affected = partition_moves::apply_recompute(part, m.op, m.gb);
            if (affected.empty()) return {};
            break;
        }
        case FMMove::EJECT:
        case FMMove::INTERNAL_EJECT: {
            affected = partition_moves::apply_eject(part, m.op, m.ga);
            if (affected.empty()) return {};
            break;
        }
        case FMMove::SPLIT: {
            affected = partition_moves::apply_split(part, m.op, m.op2, m.ga);
            if (affected.empty()) return {};
            break;
        }
        case FMMove::TENSOR_MERGE: {
            affected = partition_moves::apply_tensor_merge(part, m.tensor_groups);
            if (affected.empty()) return {};
            break;
        }
        case FMMove::TENSOR_EXTRACT: {
            FlatSet<size_t> extract_ops(m.tensor_consumer_ops.begin(),
                                         m.tensor_consumer_ops.end());
            affected = partition_moves::apply_tensor_extract(part, extract_ops,
                                                              m.tensor_groups);
            if (affected.empty()) return {};
            break;
        }
        case FMMove::DE_RECOMPUTE: {
            affected = partition_moves::apply_de_recompute(part, m.ga, m.op);
            if (affected.empty()) return {};
            break;
        }
        case FMMove::FORCE_RECOMPUTE: {
            auto frr = partition_moves::eval_force_recompute(part, m.op2);
            affected = partition_moves::apply_force_recompute(part, m.op2, frr);
            if (affected.empty()) return {};
            break;
        }
        case FMMove::TENSOR_EXTRACT_SPLIT: {
            affected = partition_moves::apply_tensor_extract_split(
                part, m.split_extract_result, m.tensor_groups);
            if (affected.empty()) return {};
            break;
        }
        default:
            return {};
    }

    part.rebuild_index();

#ifndef NDEBUG
    // Debug: verify no op was lost by this move
    for (size_t i = 0; i < part.prob->num_ops(); i++) {
        bool found = false;
        for (auto gi : part.groups_of(i))
            if (part.groups[gi].alive) { found = true; break; }
        if (!found) {
            std::cerr << "  BUG: apply_fm_move lost op " << i
                      << " (move type=" << (int)m.type
                      << " op=" << m.op << " ga=" << m.ga << " gb=" << m.gb << ")\n";
            assert(false && "apply_fm_move lost an op");
        }
    }
#endif

    return affected;
}