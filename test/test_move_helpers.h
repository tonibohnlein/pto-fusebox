#pragma once
// test_move_helpers.h — Test utilities for move enumeration and gain verification.
// Uses FMMove and apply_fm_move (no legacy Move struct).

#include "partition/partition.h"
#include "search/fm_search.h"
#include <vector>
#include <set>

// Enumerate ALL candidate FMMove objects involving group gi.
// Returns every valid (MERGE, STEAL, RECOMPUTE, EJECT, INTERNAL_EJECT, SPLIT)
// with saving > -floor. Used by tests that need exhaustive move enumeration.
inline std::vector<FMMove> all_moves_for_group(const Partition& part, size_t gi,
                                                double floor = 1e18) {
    std::vector<FMMove> result;
    if (!part.groups[gi].alive) return result;

    auto neighbors = part.boundary_neighbors(gi);
    for (auto adj_op : neighbors) {
        for (auto gj : part.groups_of(adj_op)) {
            if (gj == gi) continue;

            // MERGE
            if (!part.dag->merge_creates_cycle(part.groups[gi].ops, part.groups[gj].ops)) {
                FlatSet<size_t> merged = part.groups[gi].ops;
                merged.insert(part.groups[gj].ops.begin(), part.groups[gj].ops.end());
                double nc = part.eval_set(merged);
                double saving = (part.groups[gi].cost + part.groups[gj].cost) - nc;
                if (saving > -floor)
                    result.push_back({FMMove::MERGE, adj_op, gi, gj, SIZE_MAX, saving});
            }

            // STEAL (from gj to gi)
            if (!part.groups[gi].ops.count(adj_op) &&
                !part.dag->merge_creates_cycle({adj_op}, part.groups[gi].ops)) {
                FlatSet<size_t> new_gi = part.groups[gi].ops;
                new_gi.insert(adj_op);
                double new_gi_cost = part.eval_set(new_gi);
                if (new_gi_cost < 1e17) {
                    FlatSet<size_t> new_gj = part.groups[gj].ops;
                    new_gj.erase(adj_op);
                    double new_gj_cost = new_gj.empty() ? 0 : part.eval_set(new_gj);
                    if (new_gj.empty() || new_gj_cost < 1e17) {
                        double saving = (part.groups[gi].cost + part.groups[gj].cost)
                                        - (new_gi_cost + new_gj_cost);
                        // STEAL convention: ga=source (where op leaves), gb=target (where op goes)
                        if (saving > -floor)
                            result.push_back({FMMove::STEAL, adj_op, gj, gi, SIZE_MAX, saving});
                    }
                }
            }

            // RECOMPUTE (add adj_op to gi, keep in gj)
            if (!part.groups[gi].ops.count(adj_op) &&
                !part.dag->merge_creates_cycle({adj_op}, part.groups[gi].ops)) {
                FlatSet<size_t> new_gi = part.groups[gi].ops;
                new_gi.insert(adj_op);
                double nc = part.eval_set(new_gi);
                double saving = part.groups[gi].cost - nc;
                if (saving > -floor)
                    result.push_back({FMMove::RECOMPUTE, adj_op, gj, gi, SIZE_MAX, saving});
            }
        }
    }

    // EJECT
    if (part.groups[gi].ops.size() >= 2) {
        auto ejectable = part.ejectable_ops(gi);
        for (auto op : ejectable) {
            auto er = part.eval_eject(op, gi);
            if (er.feasible && er.saving > -floor)
                result.push_back({FMMove::EJECT, op, gi, SIZE_MAX, SIZE_MAX, er.saving});
        }
    }

    // INTERNAL_EJECT
    if (part.groups[gi].ops.size() >= 3 && part.groups[gi].ops.size() <= 15) {
        auto internals = part.internal_ops(gi);
        for (auto op : internals) {
            auto er = part.eval_eject(op, gi);
            if (er.feasible && er.saving > -floor)
                result.push_back({FMMove::INTERNAL_EJECT, op, gi, SIZE_MAX, SIZE_MAX, er.saving});
        }
    }

    // SPLIT
    if (part.groups[gi].ops.size() >= 3 && part.groups[gi].ops.size() <= 15) {
        auto bridges = part.bridge_edges(gi);
        for (auto& [a, b] : bridges) {
            auto sr = part.eval_split(a, b, gi);
            if (sr.feasible && sr.saving > -floor)
                result.push_back({FMMove::SPLIT, a, gi, SIZE_MAX, b, sr.saving});
        }
    }

    return result;
}

// Apply FMMove to a copy, return (before, after, reported saving, applied).
struct GainCheck { double before, after, reported; bool applied; };

inline GainCheck verify_move_gain(Partition part, const FMMove& m) {
    GainCheck gc;
    gc.before = part.total_cost();
    gc.reported = m.saving;
    gc.applied = false;

    auto affected = apply_fm_move(part, m);
    if (affected.empty()) return gc;
    gc.applied = true;
    gc.after = part.total_cost();
    return gc;
}