#include "pipeline/solver.h"
#include "partition/partition.h"
#include "search/parallel_search.h"
#include "search/solution_search.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <optional>
#include <set>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>

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

        // Build group DAG based on tensor dependencies.
        // For each boundary input T of group G, find which group contains
        // the op that produces T (via the op-level DAG). This correctly
        // handles recomputed ops where T may be ephemeral in its producing
        // group but still needed as boundary input by another group.
        
        // Map: op → group index (for recomputed ops, track all groups)
        std::vector<std::vector<size_t>> op_groups(prob.num_ops());
        for (size_t i = 0; i < n; i++)
            for (auto op : gd.groups[i].ops)
                op_groups[op].push_back(i);

        for (size_t i = 0; i < n; i++) {
            for (auto t : gd.groups[i].sg.boundary_inputs()) {
                int prod_op = dag.tensor_producer[t];
                if (prod_op < 0) continue;  // graph input, no producer
                for (auto pg : op_groups[prod_op]) {
                    if (pg != i) {
                        gd.preds[i].insert(pg);
                        gd.succs[pg].insert(i);
                    }
                }
            }
        }

        // Also keep tensor_producer map for retain scoring (boundary outputs only)
        for (size_t i = 0; i < n; i++)
            for (auto t : gd.groups[i].sg.boundary_outputs())
                gd.tensor_producer[t] = i;

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
        // Check if any boundary output of prev is needed by cand
        for (auto prev_out : gd.groups[prev].sg.boundary_outputs()) {
            if (gd.groups[cand].sg.boundary_inputs().count(prev_out) &&
                gd.prob->retainable_tensors.count(prev_out))
                score += gd.prob->tensors[prev_out].width * gd.prob->tensors[prev_out].height;
        }
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
// Beam search ordering with true memory state tracking.
//
// Each beam state tracks which tensors are currently resident in fast memory.
// This enables multi-step retention: a tensor produced at step 0 can remain
// resident through steps 1,2,... and be consumed at step 3 for free.
//
// The expansion evaluates each candidate with the actual resident set,
// decides what to retain, and propagates the memory state forward.
// ============================================================================

struct BeamResult {
    std::vector<size_t> order;
    std::vector<std::set<size_t>> retain_per_step;  // what each step retains
    double total_latency = 0;
};

static BeamResult beam_search_ordering(const GroupDAGInfo& gd, int beam_width) {
    size_t n = gd.size();
    const Problem& prob = *gd.prob;

    struct State {
        std::vector<size_t> order;
        std::vector<int> in_deg;
        std::set<size_t> resident;   // tensors currently in fast memory
        std::vector<std::set<size_t>> retain_per_step;
        double total_latency = 0;
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
                // Mark gi as scheduled for future_needs check
                scheduled[gi] = true;

                // Step 1: Prune resident set — drop tensors no future step needs
                std::set<size_t> useful_resident;
                for (auto t : state.resident)
                    if (gd.future_needs(t, scheduled))
                        useful_resident.insert(t);

                // Step 2: Determine what to retain after this step.
                // Candidates: boundary outputs that future steps need.
                std::set<size_t> retainable_outputs;
                for (auto t : gd.groups[gi].sg.boundary_outputs()) {
                    if (prob.retainable_tensors.count(t) &&
                        gd.future_needs(t, scheduled))
                        retainable_outputs.insert(t);
                }

                // Step 3: Evaluate memory states and strictly enforce latency improvement
                CostResult best_cost = gd.groups[gi].base_cost;
                std::set<size_t> best_resident;
                std::set<size_t> best_retain_these;

                // Option A: Keep only what this group strictly uses
                std::set<size_t> only_used;
                for (auto t : useful_resident)
                    if (gd.groups[gi].sg.boundary_inputs().count(t))
                        only_used.insert(t);
                
                auto c_used = gd.groups[gi].sg.best_cost(only_used, {});
                if (c_used.feasible && c_used.latency < best_cost.latency) {
                    best_cost = c_used; best_resident = only_used; best_retain_these.clear();
                }

                // Option B: Keep all pass-through residents
                auto c_all = gd.groups[gi].sg.best_cost(useful_resident, {});
                if (c_all.feasible && c_all.latency < best_cost.latency) {
                    best_cost = c_all; best_resident = useful_resident; best_retain_these.clear();
                }

                if (!retainable_outputs.empty()) {
                    // Option C: Only used + retain outputs
                    auto c_used_ret = gd.groups[gi].sg.best_cost(only_used, retainable_outputs);
                    if (c_used_ret.feasible && c_used_ret.latency < best_cost.latency) {
                        best_cost = c_used_ret; best_resident = only_used; best_retain_these = retainable_outputs;
                    }
                    
                    // Option D: All resident + retain outputs
                    auto c_all_ret = gd.groups[gi].sg.best_cost(useful_resident, retainable_outputs);
                    if (c_all_ret.feasible && c_all_ret.latency < best_cost.latency) {
                        best_cost = c_all_ret; best_resident = useful_resident; best_retain_these = retainable_outputs;
                    }
                }

                CostResult cost = best_cost;
                useful_resident = best_resident;

                // Build next state
                State next;
                next.order = state.order;
                next.order.push_back(gi);
                next.in_deg = state.in_deg;
                for (auto v : gd.succs[gi])
                    next.in_deg[v]--;
                next.total_latency = state.total_latency + cost.latency;

                // Apply intermediate residency
                std::set<size_t> intermediate_resident = useful_resident;
                for (auto t : best_retain_these)
                    intermediate_resident.insert(t);

                // Determine strictly what can physically stay in memory
                std::set<size_t> exportable_retain;
                for (auto t : intermediate_resident) {
                    // Keep if it's a boundary output or boundary input of this group
                    if (gd.groups[gi].sg.boundary_outputs().count(t) ||
                        gd.groups[gi].sg.boundary_inputs().count(t))
                        exportable_retain.insert(t);
                }
                
                // STRICT RESIDENCY FIX: Only propagate what can physically be retained
                next.resident = exportable_retain;
                next.retain_per_step = state.retain_per_step;
                next.retain_per_step.push_back(exportable_retain);

                candidates.push_back(std::move(next));

                // Restore scheduled flag
                scheduled[gi] = false;
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

    if (beam.empty()) {
        BeamResult r;
        r.order = dfs_ordering(gd);
        return r;
    }

    BeamResult r;
    r.order = beam[0].order;
    r.retain_per_step = beam[0].retain_per_step;
    r.total_latency = beam[0].total_latency;
    return r;
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

    // --- Strategy 2: Beam search with true memory state tracking ---
    int beam_width = std::min(20, std::max(5, (int)n));
    auto beam_result = beam_search_ordering(gd, beam_width);

    // Build solution using beam's ordering AND retain decisions.
    std::vector<ScheduleStep> beam_steps;
    std::set<size_t> beam_entering;  // retained from previous step

    for (size_t i = 0; i < beam_result.order.size(); i++) {
        size_t idx = beam_result.order[i];
        std::set<size_t> retain_these;
        
        if (i < beam_result.retain_per_step.size()) {
            // Filter: only retain if the NEXT step actually uses it
            if (i + 1 < beam_result.order.size()) {
                size_t next_idx = beam_result.order[i + 1];
                for (auto t : beam_result.retain_per_step[i]) {
                    if (gd.groups[next_idx].sg.boundary_inputs().count(t)) {
                        retain_these.insert(t);
                    }
                }
            }
        }

        // Compute cost with the actual retained state
        auto cost = gd.groups[idx].sg.best_cost(beam_entering, retain_these);
        if (!cost.feasible) {
            // Fallback: try without retain
            retain_these.clear();
            cost = gd.groups[idx].sg.best_cost(beam_entering, {});
        }
        if (!cost.feasible) {
            // Fallback: no resident tensors
            beam_entering.clear();
            retain_these.clear();
            cost = gd.groups[idx].base_cost;
        }

        beam_steps.push_back({Subgraph(gd.groups[idx].sg), cost.config, retain_these});
        beam_entering = retain_these;  // next step sees our full retain set
    }
    Solution beam_sol(prob, dag, std::move(beam_steps));

    double dfs_lat = dfs_sol.total_latency();
    double beam_lat = beam_sol.total_latency();

    std::cerr << "  DFS:  " << dfs_lat << "\n";
    std::cerr << "  Beam: " << beam_lat << "\n";
    std::cerr << "  Winner: " << (beam_lat < dfs_lat - 0.01 ? "Beam" : "DFS") << "\n";

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

    auto now = SteadyClock::now();
    auto effective_deadline = deadline;
    // If no real deadline was given (tests), cap at 10s
    if (deadline == no_deadline() || (deadline - now) > std::chrono::seconds(300))
        effective_deadline = now + std::chrono::seconds(5);
    auto total_budget = effective_deadline - now;

    // Precompute once: which tensors can physically be retained?
    auto feasibly_ret = compute_feasibly_retainable(prob, dag);
    bool has_retain = !feasibly_ret.empty();

    std::cerr << "  Retainable tensors: " << feasibly_ret.size()
              << " / " << prob.retainable_tensors.size() << "\n";

    // Time allocation depends on whether retain is useful.
    // No retainable tensors → solution cost = partition cost for any topo order,
    // so solution FM/evo is pointless. Give all time to partition search.
    TimePoint phase1_deadline, phase2_deadline, phase3_deadline;
    if (has_retain) {
        // Normal: 35% partition, 5% build, 60% solution evo+FM
        phase1_deadline = now + std::chrono::duration_cast<SteadyClock::duration>(total_budget * 35 / 100);
        phase2_deadline = phase1_deadline + std::chrono::duration_cast<SteadyClock::duration>(total_budget * 5 / 100);
        phase3_deadline = now + total_budget;
    } else {
        // No retain: 95% partition, 5% build (topo sort only), no phase 3
        phase1_deadline = now + std::chrono::duration_cast<SteadyClock::duration>(total_budget * 95 / 100);
        phase2_deadline = now + total_budget;
        phase3_deadline = now;  // skip
    }

    // ================================================================
    // Phase 1: Partition pool via parallel search
    // ================================================================
    std::cerr << "Phase 1: Parallel search (init + greedy + FM + evo)...\n";
    ParallelConfig pcfg;
    pcfg.fm.deadline = phase1_deadline;
    auto partition_pool = parallel_search(prob, dag, pcfg);

    std::cerr << "  Partition pool: " << partition_pool.size() << " entries, best="
              << partition_pool[0].total_cost() << "\n";

    // ================================================================
    // Phase 2: Build solutions from each partition in the pool
    // ================================================================

    std::cerr << "Phase 2: Build solutions from " << partition_pool.size() << " partitions"
              << (has_retain ? "" : " (no-retain fast path)") << "...\n";

    std::vector<Solution> solution_pool;
    std::mutex sol_mutex;

    {
        int hw_threads = (int)std::thread::hardware_concurrency();
        if (hw_threads <= 0) hw_threads = 4;
        int n_tasks = (int)partition_pool.size();
        std::atomic<int> next_task{0};

        auto build_worker = [&]() {
            while (true) {
                int tid = next_task.fetch_add(1);
                if (tid >= n_tasks) break;
                if (SteadyClock::now() >= phase2_deadline) break;

                auto& part = partition_pool[tid];
                auto gd_opt = GroupDAGInfo::build(prob, dag, part);
                if (!gd_opt) continue;
                auto& gd = *gd_opt;

                if (!has_retain) {
                    // No-retain fast path: any topological order gives the same cost.
                    // Just build a DFS solution without retain/recompute optimization.
                    auto dfs_order = dfs_ordering(gd);
                    std::vector<ScheduleStep> dfs_steps;
                    for (auto idx : dfs_order)
                        dfs_steps.push_back({Subgraph(gd.groups[idx].sg), gd.groups[idx].base_cost.config, {}});
                    Solution dfs_sol(prob, dag, std::move(dfs_steps));
                    {
                        std::lock_guard<std::mutex> lock(sol_mutex);
                        if (dfs_sol.validate().valid)
                            solution_pool.push_back(std::move(dfs_sol));
                    }
                    continue;
                }

                // DFS solution
                auto dfs_order = dfs_ordering(gd);
                std::vector<ScheduleStep> dfs_steps;
                for (auto idx : dfs_order)
                    dfs_steps.push_back({Subgraph(gd.groups[idx].sg), gd.groups[idx].base_cost.config, {}});
                Solution dfs_sol(prob, dag, std::move(dfs_steps));

                // Beam solution
                int beam_width = std::min(20, std::max(5, (int)gd.size()));
                auto beam_result = beam_search_ordering(gd, beam_width);
                std::vector<ScheduleStep> beam_steps;
                std::set<size_t> beam_entering;
                for (size_t i = 0; i < beam_result.order.size(); i++) {
                    size_t idx = beam_result.order[i];
                    std::set<size_t> retain_these;
                    if (i < beam_result.retain_per_step.size() && i + 1 < beam_result.order.size()) {
                        size_t next_idx = beam_result.order[i + 1];
                        for (auto t : beam_result.retain_per_step[i])
                            if (gd.groups[next_idx].sg.boundary_inputs().count(t))
                                retain_these.insert(t);
                    }
                    auto cost = gd.groups[idx].sg.best_cost(beam_entering, retain_these);
                    if (!cost.feasible) { retain_these.clear(); cost = gd.groups[idx].sg.best_cost(beam_entering, {}); }
                    if (!cost.feasible) { beam_entering.clear(); retain_these.clear(); cost = gd.groups[idx].base_cost; }
                    beam_steps.push_back({Subgraph(gd.groups[idx].sg), cost.config, retain_these});
                    beam_entering = retain_these;
                }
                Solution beam_sol(prob, dag, std::move(beam_steps));

                // Random-retain variant: pick tensors to retain first, then
                // build an ordering that supports those retain decisions.
                // Diversity comes from which tensors are retained, not from ordering.
                std::mt19937 rand_rng(42 + tid * 1337);

                // 1. Find retainable tensors at group boundaries
                //    T is retainable between groups if:
                //    - T is in feasibly_ret (singleton feasibility check)
                //    - T is boundary_output of some group (producer)
                //    - T is boundary_input of some other group (consumer)
                struct RetainCandidate {
                    size_t tensor;
                    size_t prod_group;   // group that produces T
                    size_t cons_group;   // a group that consumes T
                };
                std::vector<RetainCandidate> retain_cands;
                for (size_t gi = 0; gi < gd.size(); gi++) {
                    for (auto t : gd.groups[gi].sg.boundary_outputs()) {
                        if (!feasibly_ret.count(t)) continue;
                        for (size_t gj = 0; gj < gd.size(); gj++) {
                            if (gi == gj) continue;
                            if (gd.groups[gj].sg.boundary_inputs().count(t))
                                retain_cands.push_back({t, gi, gj});
                        }
                    }
                }

                // 2. Randomly pick ~50% of candidates
                std::shuffle(retain_cands.begin(), retain_cands.end(), rand_rng);
                int n_retain = (int)(retain_cands.size() * (0.3 + 0.4 * (rand_rng() % 1000) / 1000.0));
                retain_cands.resize(std::min(n_retain, (int)retain_cands.size()));

                // Build adjacency preferences: prod_group → cons_group for each retained tensor
                // Map: for each group, which retained tensors want to enter it from which predecessor?
                // wants_from[cons_group] = set of (prod_group, tensor) pairs
                std::map<size_t, std::vector<std::pair<size_t, size_t>>> wants_from;
                for (auto& rc : retain_cands)
                    wants_from[rc.cons_group].push_back({rc.prod_group, rc.tensor});

                // 3. Greedy topological sort: prefer groups that receive retained tensors
                //    from the just-scheduled group
                size_t n_groups = gd.size();
                std::vector<int> in_deg = gd.in_deg;
                std::vector<bool> scheduled(n_groups, false);
                std::vector<size_t> order;
                order.reserve(n_groups);
                size_t last_group = SIZE_MAX;

                while (order.size() < n_groups) {
                    // Collect ready groups (in_deg == 0 and not scheduled)
                    std::vector<size_t> ready;
                    for (size_t i = 0; i < n_groups; i++)
                        if (!scheduled[i] && in_deg[i] == 0)
                            ready.push_back(i);
                    if (ready.empty()) break; // shouldn't happen if DAG is valid

                    // Score each ready group: how many retained tensors enter from last_group?
                    size_t best = ready[0];
                    int best_score = -1;
                    for (auto gi : ready) {
                        int score = 0;
                        if (last_group != SIZE_MAX) {
                            auto it = wants_from.find(gi);
                            if (it != wants_from.end())
                                for (auto& [pg, t] : it->second)
                                    if (pg == last_group) score++;
                        }
                        if (score > best_score || (score == best_score && rand_rng() % 2)) {
                            best_score = score;
                            best = gi;
                        }
                    }

                    order.push_back(best);
                    scheduled[best] = true;
                    last_group = best;
                    for (auto succ : gd.succs[best])
                        in_deg[succ]--;
                }

                // 4. Build steps with retain decisions based on adjacency
                std::vector<ScheduleStep> rand_steps;
                for (auto idx : order)
                    rand_steps.push_back({Subgraph(gd.groups[idx].sg), gd.groups[idx].base_cost.config, {}});

                // Map group_index → step_position for quick lookup
                std::vector<size_t> group_to_step(n_groups, SIZE_MAX);
                for (size_t i = 0; i < order.size(); i++)
                    group_to_step[order[i]] = i;

                // For each retained tensor: if producer and consumer are adjacent, retain it
                for (auto& rc : retain_cands) {
                    size_t sp = group_to_step[rc.prod_group];
                    size_t sc = group_to_step[rc.cons_group];
                    if (sp != SIZE_MAX && sc == sp + 1)
                        rand_steps[sp].retain_these.insert(rc.tensor);
                }

                // 5. Forward pass: verify feasibility and set configs
                std::set<size_t> rand_entering;
                for (size_t i = 0; i < rand_steps.size(); i++) {
                    auto& si = rand_steps[i];
                    auto cost = si.subgraph.best_cost(rand_entering, si.retain_these);
                    if (!cost.feasible) {
                        // Drop retains one by one until feasible
                        auto ret = si.retain_these;
                        std::vector<size_t> ret_vec(ret.begin(), ret.end());
                        std::shuffle(ret_vec.begin(), ret_vec.end(), rand_rng);
                        for (auto t : ret_vec) {
                            ret.erase(t);
                            cost = si.subgraph.best_cost(rand_entering, ret);
                            if (cost.feasible) break;
                        }
                        if (!cost.feasible) { ret.clear(); cost = si.subgraph.best_cost(rand_entering, {}); }
                        if (!cost.feasible) { rand_entering.clear(); ret.clear(); cost = si.subgraph.best_cost({}, {}); }
                        si.retain_these = ret;
                    }
                    si.config = cost.config;
                    rand_entering = si.retain_these;
                }
                Solution rand_sol(prob, dag, std::move(rand_steps));

                {
                    std::lock_guard<std::mutex> lock(sol_mutex);
                    if (dfs_sol.validate().valid)
                        solution_pool.push_back(std::move(dfs_sol));
                    if (beam_sol.validate().valid)
                        solution_pool.push_back(std::move(beam_sol));
                    if (rand_sol.validate().valid)
                        solution_pool.push_back(std::move(rand_sol));
                }
            }
        };

        std::vector<std::thread> threads;
        int nt = std::min(hw_threads, n_tasks);
        for (int i = 0; i < nt; i++) threads.emplace_back(build_worker);
        for (auto& t : threads) t.join();
    }

    // Sort solution pool by latency
    std::sort(solution_pool.begin(), solution_pool.end(),
              [](const Solution& a, const Solution& b) {
                  return a.total_latency() < b.total_latency();
              });

    if (solution_pool.empty()) {
        // Fallback: build from best partition
        auto sol = build_solution(prob, dag, partition_pool[0]);
        solution_pool.push_back(std::move(sol));
    }

    double after_build = solution_pool[0].total_latency();
    std::cerr << "  Solution pool: " << solution_pool.size() << " solutions, best="
              << after_build << "\n";

    // ================================================================
    // Phase 3: Solution-level evolutionary search with FM polish
    // ================================================================
    Solution final_sol(prob, dag, {});
    double after_sol_evo;

    if (has_retain) {
        std::cerr << "Phase 3: Solution evolution + FM from " << solution_pool.size() << " starting points...\n";
        SolutionFMConfig sfm_cfg;
        sfm_cfg.deadline = phase3_deadline;
        final_sol = solution_evo_search(prob, dag, std::move(solution_pool), sfm_cfg);
        after_sol_evo = final_sol.total_latency();
    } else {
        std::cerr << "Phase 3: Skipped (no retainable tensors — solution cost = partition cost)\n";
        final_sol = std::move(solution_pool[0]);
        after_sol_evo = final_sol.total_latency();
    }

    auto vr = final_sol.validate();
    if (!vr.valid)
        std::cerr << "  WARNING: " << vr.error << "\n";

    // Summary
    double partition_cost = partition_pool[0].total_cost();
    double final_cost = final_sol.total_latency();
    std::cerr << "  === Summary ===\n";
    std::cerr << "  Partition:  " << partition_cost << "\n";
    std::cerr << "  Build:      " << after_build;
    if (after_build < partition_cost - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << 100.0*(partition_cost - after_build)/partition_cost << "%)";
    std::cerr << "\n";
    std::cerr << "  Sol-Evo:     " << after_sol_evo;
    if (after_sol_evo < after_build - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << 100.0*(after_build - after_sol_evo)/after_build << "%)";
    std::cerr << "\n";
    std::cerr << "  Final:      " << final_cost;
    if (final_cost < partition_cost - 0.01)
        std::cerr << " (-" << std::fixed << std::setprecision(1)
                  << 100.0*(partition_cost - final_cost)/partition_cost << "% total)";
    std::cerr << "\n";

    return final_sol;
}