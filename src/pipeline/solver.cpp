#include "pipeline/solver.h"
#include "partition/partition.h"
#include "search/parallel_search.h"
#include "postopt/post_opt.h"
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <algorithm>
#include <vector>

// ============================================================================
// Shared group-level DAG structure for ordering algorithms
// ============================================================================

struct GroupDAGInfo {
    struct Group {
        std::vector<size_t> ops;
        Subgraph sg;
        CostResult base_cost;  // cost without retain
    };

    const Problem* prob;
    const DAG* dag;
    std::vector<Group> groups;
    std::vector<std::set<size_t>> preds, succs;
    std::map<size_t, size_t> tensor_producer;  // tensor → group index
    std::vector<int> in_deg;

    static std::optional<GroupDAGInfo> build(const Problem& prob, const DAG& dag,
                                             const Partition& part) {
        GroupDAGInfo gd;
        gd.prob = &prob;
        gd.dag = &dag;

        for (auto& g : part.groups) {
            if (!g.alive) continue;
            std::vector<size_t> ops(g.ops.begin(), g.ops.end());
            auto sg = Subgraph::create(prob, dag, ops);
            if (!sg) continue;
            auto cost = sg->best_cost();
            gd.groups.push_back({ops, std::move(*sg), cost});
        }

        size_t n = gd.groups.size();
        if (n == 0) return std::nullopt;

        gd.preds.resize(n);
        gd.succs.resize(n);

        for (size_t i = 0; i < n; i++)
            for (auto t : gd.groups[i].sg.boundary_outputs())
                gd.tensor_producer[t] = i;

        for (size_t i = 0; i < n; i++) {
            for (auto t : gd.groups[i].sg.boundary_inputs()) {
                auto it = gd.tensor_producer.find(t);
                if (it != gd.tensor_producer.end() && it->second != i) {
                    gd.preds[i].insert(it->second);
                    gd.succs[it->second].insert(i);
                }
            }
        }

        gd.in_deg.resize(n, 0);
        for (size_t i = 0; i < n; i++)
            gd.in_deg[i] = (int)gd.preds[i].size();

        return gd;
    }

    size_t size() const { return groups.size(); }

    // Check if any unscheduled group needs tensor t
    bool future_needs(size_t t, const std::vector<bool>& scheduled) const {
        for (size_t i = 0; i < groups.size(); i++) {
            if (scheduled[i]) continue;
            if (groups[i].sg.boundary_inputs().count(t)) return true;
        }
        return false;
    }
};

// ============================================================================
// DFS chain-first ordering (fast, deterministic)
// ============================================================================

static std::vector<size_t> dfs_ordering(const GroupDAGInfo& gd) {
    size_t n = gd.size();
    std::vector<size_t> order;
    std::vector<bool> scheduled(n, false);
    std::set<size_t> ready;
    auto in_deg = gd.in_deg;

    for (size_t i = 0; i < n; i++)
        if (in_deg[i] == 0) ready.insert(i);

    auto retain_score = [&](size_t cand, size_t prev) -> int64_t {
        if (prev >= n) return 0;
        int64_t score = 0;
        size_t prev_sink = gd.groups[prev].sg.sink_tensor();
        if (gd.groups[cand].sg.boundary_inputs().count(prev_sink) &&
            gd.prob->retainable_tensors.count(prev_sink))
            score += gd.prob->tensors[prev_sink].width * gd.prob->tensors[prev_sink].height;
        for (auto t : gd.groups[cand].sg.boundary_inputs())
            if (gd.groups[prev].sg.boundary_inputs().count(t) &&
                gd.prob->retainable_tensors.count(t))
                score += gd.prob->tensors[t].width * gd.prob->tensors[t].height;
        return score;
    };

    auto input_size = [&](size_t gi) -> int64_t {
        int64_t s = 0;
        for (auto t : gd.groups[gi].sg.boundary_inputs())
            s += gd.prob->tensors[t].width * gd.prob->tensors[t].height;
        return s;
    };

    size_t prev = SIZE_MAX;
    while (!ready.empty()) {
        size_t chosen = SIZE_MAX;
        int64_t best_score = -1, best_input = -1;

        // Prefer direct successors of prev
        if (prev < n) {
            for (auto s : gd.succs[prev]) {
                if (!scheduled[s] && ready.count(s)) {
                    int64_t sc = retain_score(s, prev);
                    int64_t inp = input_size(s);
                    if (sc > best_score || (sc == best_score && inp > best_input)) {
                        chosen = s; best_score = sc; best_input = inp;
                    }
                }
            }
        }

        if (chosen == SIZE_MAX) {
            for (auto gi : ready) {
                int64_t sc = retain_score(gi, prev);
                int64_t inp = input_size(gi);
                if (sc > best_score || (sc == best_score && inp > best_input)) {
                    chosen = gi; best_score = sc; best_input = inp;
                }
            }
        }

        ready.erase(chosen);
        order.push_back(chosen);
        scheduled[chosen] = true;
        prev = chosen;

        for (auto v : gd.succs[chosen])
            if (--in_deg[v] == 0) ready.insert(v);
    }

    return order;
}

// ============================================================================
// Beam search ordering (explores K orderings simultaneously)
//
// State: ordered groups so far + accumulated cost (simulating retain).
// At each step: expand all ready groups for each beam state, compute cost
// with simulated retain from previous group, keep top K.
// ============================================================================

static std::vector<size_t> beam_search_ordering(
        const GroupDAGInfo& gd, int beam_width) {

    size_t n = gd.size();
    const Problem& prob = *gd.prob;

    // Beam state: just ordering + accumulated latency.
    // Retain is handled by post-optimization, not here.
    // But we simulate retain to score orderings: if the previous group's
    // sink is a boundary input of the current group and retainable,
    // we compute cost as-if retained (to prefer retain-friendly orderings).
    struct State {
        std::vector<size_t> order;
        std::vector<int> in_deg;
        double total_latency = 0;
        size_t last = SIZE_MAX;
    };

    State init;
    init.in_deg = gd.in_deg;
    std::vector<State> beam = {init};

    for (size_t step = 0; step < n; step++) {
        std::vector<State> candidates;

        for (auto& state : beam) {
            std::vector<bool> scheduled(n, false);
            for (auto g : state.order) scheduled[g] = true;

            std::vector<size_t> ready;
            for (size_t i = 0; i < n; i++)
                if (!scheduled[i] && state.in_deg[i] == 0)
                    ready.push_back(i);

            for (auto gi : ready) {
                // Simulate retain from previous step: if prev's sink
                // is a boundary input of gi and retainable, compute
                // cost as-if retained. This steers ordering towards
                // producer→consumer adjacency.
                std::set<size_t> sim_retained;
                if (state.last < n) {
                    size_t prev_sink = gd.groups[state.last].sg.sink_tensor();
                    if (gd.groups[gi].sg.boundary_inputs().count(prev_sink) &&
                        prob.retainable_tensors.count(prev_sink)) {
                        // Check feasibility with retain
                        auto c_ret = gd.groups[gi].sg.best_cost({prev_sink}, {});
                        if (c_ret.feasible)
                            sim_retained.insert(prev_sink);
                    }
                }

                auto cost = gd.groups[gi].sg.best_cost(sim_retained, {});
                if (!cost.feasible)
                    cost = gd.groups[gi].base_cost;

                State next;
                next.order = state.order;
                next.order.push_back(gi);
                next.in_deg = state.in_deg;
                for (auto v : gd.succs[gi])
                    next.in_deg[v]--;
                next.total_latency = state.total_latency + cost.latency;
                next.last = gi;

                candidates.push_back(std::move(next));
            }
        }

        if (candidates.empty()) break;

        std::sort(candidates.begin(), candidates.end(),
                  [](const State& a, const State& b) {
                      return a.total_latency < b.total_latency;
                  });
        if ((int)candidates.size() > beam_width)
            candidates.resize(beam_width);

        beam = std::move(candidates);
    }

    if (beam.empty()) return dfs_ordering(gd);
    return beam[0].order;
}

// ============================================================================
// Build Solution from a Partition — tries both DFS and beam search
// ============================================================================

static Solution build_solution(const Problem& prob, const DAG& dag,
                                const Partition& part) {
    auto gd_opt = GroupDAGInfo::build(prob, dag, part);
    if (!gd_opt) return Solution(prob, dag, {});
    auto& gd = *gd_opt;
    size_t n = gd.size();

    // --- Strategy 1: DFS ordering + post-hoc retain ---
    auto dfs_order = dfs_ordering(gd);
    std::vector<ScheduleStep> dfs_steps;
    for (auto idx : dfs_order)
        dfs_steps.push_back({Subgraph(gd.groups[idx].sg), gd.groups[idx].base_cost.config, {}});
    Solution dfs_sol(prob, dag, std::move(dfs_steps));

    // --- Strategy 2: Beam search ordering (retain-aware scoring) ---
    int beam_width = std::min(20, std::max(5, (int)n));
    auto beam_order = beam_search_ordering(gd, beam_width);

    std::vector<ScheduleStep> beam_steps;
    for (auto idx : beam_order)
        beam_steps.push_back({Subgraph(gd.groups[idx].sg), gd.groups[idx].base_cost.config, {}});
    Solution beam_sol(prob, dag, std::move(beam_steps));

    // Pick the better one
    double dfs_lat = dfs_sol.total_latency();
    double beam_lat = beam_sol.total_latency();

    std::cerr << "  DFS ordering: " << dfs_lat << ", Beam ordering: " << beam_lat << "\n";

    if (beam_lat < dfs_lat - 0.01) {
        return beam_sol;
    }
    return dfs_sol;
}

// ============================================================================
// Full pipeline
// ============================================================================

Solution solve(const Problem& prob, TimePoint deadline) {
    DAG dag = DAG::build(prob);

    std::cerr << "Phase 1: Parallel search (init + greedy + FM)...\n";
    ParallelConfig pcfg;
    pcfg.fm.deadline = deadline;
    auto best_part = parallel_search(prob, dag, pcfg);

    std::cerr << "Phase 2: Build solution (DFS + beam search)...\n";
    auto sol = build_solution(prob, dag, best_part);
    std::cerr << "  " << sol.num_steps() << " steps, lat=" << sol.total_latency() << "\n";

    for (int pass = 1; pass <= 3; pass++) {
        std::cerr << "Phase 3." << pass << ": Retain...\n";
        sol = optimize_retain(prob, dag, std::move(sol));
        std::cerr << "  lat=" << sol.total_latency() << "\n";

        double before = sol.total_latency();
        std::cerr << "Phase 3." << pass << ": Recompute...\n";
        sol = optimize_recompute(prob, dag, std::move(sol));
        std::cerr << "  lat=" << sol.total_latency() << "\n";

        if (sol.total_latency() >= before - 0.01) break;
    }

    auto vr = sol.validate();
    if (!vr.valid)
        std::cerr << "  WARNING: " << vr.error << "\n";
    std::cerr << "  Final: " << sol.num_steps() << " steps, lat="
              << sol.total_latency() << "\n";

    return sol;
}