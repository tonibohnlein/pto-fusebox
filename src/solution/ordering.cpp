#include "solution/ordering.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <set>

// ============================================================================
// DFS ordering
// ============================================================================

OrderingResult dfs_ordering(const Partition& part) {
    const Problem& prob = *part.prob;
    const DAG& dag = *part.dag;
    size_t ng = part.groups.size();

    std::vector<bool> scheduled(ng, false);
    for (size_t i = 0; i < ng; i++)
        if (!part.groups[i].alive) scheduled[i] = true;

    // Track physical tensors in DRAM
    std::set<size_t> available(dag.graph_inputs.begin(), dag.graph_inputs.end());

    // A group is physically ready if ALL its boundary inputs are in DRAM
    auto is_ready = [&](size_t gi) -> bool {
        if (!part.groups[gi].sg) return true;
        for (auto t : part.groups[gi].sg->boundary_inputs()) {
            if (!available.count(t)) return false;
        }
        return true;
    };

    std::set<size_t> ready;
    for (size_t i = 0; i < ng; i++) {
        if (part.groups[i].alive && !scheduled[i] && is_ready(i))
            ready.insert(i);
    }

    auto retain_score = [&](size_t cand, size_t prev) -> int64_t {
        if (prev >= ng || !part.groups[prev].sg || !part.groups[cand].sg) return 0;
        const auto& prev_sg = *part.groups[prev].sg;
        const auto& cand_sg = *part.groups[cand].sg;
        int64_t score = 0;
        for (auto t : prev_sg.boundary_outputs())
            if (cand_sg.boundary_inputs().count(t) && prob.retainable_tensors.count(t))
                score += prob.tensors[t].size();
        for (auto t : cand_sg.boundary_inputs())
            if (prev_sg.boundary_inputs().count(t) && prob.retainable_tensors.count(t))
                score += prob.tensors[t].size();
        return score;
    };

    auto input_size = [&](size_t gi) -> int64_t {
        if (!part.groups[gi].sg) return 0;
        int64_t s = 0;
        for (auto t : part.groups[gi].sg->boundary_inputs())
            s += prob.tensors[t].size();
        return s;
    };

    OrderingResult result;
    size_t prev = SIZE_MAX;
    std::mt19937 rng(42);

    while (!ready.empty()) {
        size_t chosen    = SIZE_MAX;
        int64_t best_sc  = -1;
        int64_t best_inp = -1;

        for (auto gi : ready) {
            int64_t sc  = retain_score(gi, prev);
            int64_t inp = input_size(gi);
            if (sc > best_sc || (sc == best_sc && inp > best_inp) || 
                (sc == best_sc && inp == best_inp && (rng() % 2))) {
                chosen = gi; best_sc = sc; best_inp = inp;
            }
        }

        ready.erase(chosen);
        result.order.push_back(chosen);
        scheduled[chosen] = true;
        prev = chosen;

        // When scheduled, this group physically writes its boundary outputs to DRAM
        if (part.groups[chosen].sg) {
            for (auto t : part.groups[chosen].sg->boundary_outputs()) {
                available.insert(t);
            }
        }

        // Check if any blocked groups are now physically ready
        for (size_t i = 0; i < ng; i++) {
            if (!scheduled[i] && !ready.count(i) && part.groups[i].alive && is_ready(i)) {
                ready.insert(i);
            }
        }
    }

    for (size_t i = 0; i < ng; i++) {
        if (part.groups[i].alive && !scheduled[i]) {
            std::cerr << "WARNING: dfs_ordering: group " << i << " stuck, force-adding\n";
            result.order.push_back(i);
            scheduled[i] = true;
        }
    }

    size_t n_steps = result.order.size();
    result.retain_per_step.resize(n_steps);
    for (size_t i = 0; i + 1 < n_steps; i++) {
        size_t ga = result.order[i];
        size_t gb = result.order[i + 1];
        if (!part.groups[ga].sg || !part.groups[gb].sg) continue;
        const auto& out_a = part.groups[ga].sg->boundary_outputs();
        const auto& in_b  = part.groups[gb].sg->boundary_inputs();
        for (auto t : out_a)
            if (in_b.count(t) && prob.retainable_tensors.count(t))
                result.retain_per_step[i].insert(t);
    }

    return result;
}

// ============================================================================
// Beam search ordering
// ============================================================================

OrderingResult beam_search_ordering(const Partition& part, int beam_width) {
    const Problem& prob = *part.prob;
    const DAG& dag = *part.dag;
    size_t ng      = part.groups.size();
    size_t n_alive = part.num_alive();

    struct State {
        std::vector<size_t>           order;
        std::set<size_t>              available;        
        std::set<size_t>              resident;         
        std::vector<std::set<size_t>> retain_per_step;
        double                        total_latency = 0;
    };

    State init;
    init.available.insert(dag.graph_inputs.begin(), dag.graph_inputs.end());

    std::vector<State> beam = {init};

    for (size_t step = 0; step < n_alive; step++) {
        std::vector<State> candidates;

        for (auto& state : beam) {
            std::vector<bool> scheduled(ng, false);
            for (size_t i = 0; i < ng; i++)
                if (!part.groups[i].alive) scheduled[i] = true;
            for (auto g : state.order) scheduled[g] = true;

            std::vector<size_t> ready;
            for (size_t i = 0; i < ng; i++) {
                if (scheduled[i] || !part.groups[i].alive) continue;
                bool rdy = true;
                if (part.groups[i].sg) {
                    for (auto t : part.groups[i].sg->boundary_inputs()) {
                        if (!state.available.count(t)) { rdy = false; break; }
                    }
                }
                if (rdy) ready.push_back(i);
            }

            for (auto gi : ready) {
                if (!part.groups[gi].sg) continue;
                const Subgraph& sg = *part.groups[gi].sg;
                scheduled[gi] = true;

                std::set<size_t> useful_resident;
                for (auto t : state.resident)
                    if (part.future_needs(t, scheduled)) useful_resident.insert(t);

                std::set<size_t> retainable_out;
                for (auto t : sg.boundary_outputs())
                    if (prob.retainable_tensors.count(t) && part.future_needs(t, scheduled))
                        retainable_out.insert(t);

                CostResult best;
                best.feasible = true;
                best.latency  = part.groups[gi].cost;
                best.config   = part.groups[gi].best_cfg;

                std::set<size_t> best_resident_in;
                std::set<size_t> best_retain_these;

                std::set<size_t> only_used;
                for (auto t : useful_resident)
                    if (sg.boundary_inputs().count(t)) only_used.insert(t);

                auto try_option = [&](const std::set<size_t>& res_in, const std::set<size_t>& ret_out) {
                    auto c = sg.best_cost(res_in, ret_out);
                    if (c.feasible && c.latency < best.latency) {
                        best             = c;
                        best_resident_in = res_in;
                        best_retain_these = ret_out;
                    }
                };

                try_option(only_used,       {});
                try_option(useful_resident, {});
                if (!retainable_out.empty()) {
                    try_option(only_used,       retainable_out);
                    try_option(useful_resident, retainable_out);
                }

                State next;
                next.order = state.order;
                next.order.push_back(gi);
                next.available = state.available;
                for (auto t : sg.boundary_outputs()) {
                    next.available.insert(t);
                }
                next.total_latency = state.total_latency + best.latency;

                std::set<size_t> exportable;
                for (auto t : best_retain_these)
                    if (sg.boundary_outputs().count(t)) exportable.insert(t);

                next.resident = exportable;
                next.retain_per_step = state.retain_per_step;
                next.retain_per_step.push_back(exportable);

                candidates.push_back(std::move(next));
                scheduled[gi] = false; 
            }
        }

        if (candidates.empty()) break;

        std::sort(candidates.begin(), candidates.end(),
                  [](const State& a, const State& b) { return a.total_latency < b.total_latency; });
        if ((int)candidates.size() > beam_width)
            candidates.resize(beam_width);
        beam = std::move(candidates);
    }

    if (beam.empty() || beam[0].order.size() < n_alive) 
        return dfs_ordering(part);

    OrderingResult r;
    r.order           = beam[0].order;
    r.retain_per_step = beam[0].retain_per_step;
    r.total_latency   = beam[0].total_latency;
    return r;
}

// ============================================================================
// Random ordering (for solution-pool diversity)
// ============================================================================

OrderingResult random_ordering(const Partition& part,
                               const std::set<size_t>& feasibly_ret,
                               std::mt19937& rng) {
    const Problem& prob = *part.prob;
    const DAG& dag = *part.dag;
    size_t ng      = part.groups.size();
    size_t n_alive = part.num_alive();

    struct RetainCandidate { size_t tensor, prod_group, cons_group; };
    std::vector<RetainCandidate> cands;
    for (size_t pi = 0; pi < ng; pi++) {
        if (!part.groups[pi].alive || !part.groups[pi].sg) continue;
        for (auto t : part.groups[pi].sg->boundary_outputs()) {
            if (!feasibly_ret.count(t)) continue;
            auto it = part.tensor_to_group.find(t);
            if (it == part.tensor_to_group.end()) continue;
            for (size_t ci = 0; ci < ng; ci++) {
                if (ci == pi || !part.groups[ci].alive || !part.groups[ci].sg) continue;
                if (part.groups[ci].sg->boundary_inputs().count(t))
                    cands.push_back({t, pi, ci});
            }
        }
    }

    std::shuffle(cands.begin(), cands.end(), rng);
    double frac = 0.3 + 0.4 * (rng() % 1000) / 1000.0;
    cands.resize((size_t)(cands.size() * frac));

    std::map<size_t, std::vector<std::pair<size_t, size_t>>> wants_from;
    for (auto& rc : cands)
        wants_from[rc.cons_group].push_back({rc.prod_group, rc.tensor});

    std::vector<bool> scheduled(ng, false);
    for (size_t i = 0; i < ng; i++)
        if (!part.groups[i].alive) scheduled[i] = true;

    std::set<size_t> available(dag.graph_inputs.begin(), dag.graph_inputs.end());

    auto is_ready = [&](size_t gi) -> bool {
        if (!part.groups[gi].sg) return true;
        for (auto t : part.groups[gi].sg->boundary_inputs()) {
            if (!available.count(t)) return false;
        }
        return true;
    };

    std::set<size_t> ready;
    for (size_t i = 0; i < ng; i++)
        if (part.groups[i].alive && !scheduled[i] && is_ready(i))
            ready.insert(i);

    std::vector<size_t> order;
    order.reserve(n_alive);
    size_t last = SIZE_MAX;

    while (!ready.empty()) {
        size_t chosen = SIZE_MAX;
        int best_score = -1;
        
        for (auto gi : ready) {
            int score = 0;
            if (last != SIZE_MAX) {
                auto it = wants_from.find(gi);
                if (it != wants_from.end())
                    for (auto& [pg, t] : it->second)
                        if (pg == last) score++;
            }
            if (score > best_score || (score == best_score && rng() % 2)) {
                best_score = score;
                chosen = gi;
            }
        }

        ready.erase(chosen);
        order.push_back(chosen);
        scheduled[chosen] = true;
        last = chosen;

        if (part.groups[chosen].sg) {
            for (auto t : part.groups[chosen].sg->boundary_outputs()) {
                available.insert(t);
            }
        }

        for (size_t i = 0; i < ng; i++) {
            if (!scheduled[i] && !ready.count(i) && part.groups[i].alive && is_ready(i)) {
                ready.insert(i);
            }
        }
    }

    std::vector<size_t> group_to_step(ng, SIZE_MAX);
    for (size_t i = 0; i < order.size(); i++) group_to_step[order[i]] = i;

    std::vector<std::set<size_t>> retain_per_step(order.size());
    for (auto& rc : cands) {
        size_t sp = group_to_step[rc.prod_group];
        size_t sc = group_to_step[rc.cons_group];
        if (sp != SIZE_MAX && sc == sp + 1)
            retain_per_step[sp].insert(rc.tensor);
    }

    OrderingResult result;
    result.order           = std::move(order);
    result.retain_per_step = std::move(retain_per_step);
    return result;
}