#include "search/symm_mutations.h"
#include "search/structural_ops.h"
#include "solution/solution.h"
#include <algorithm>
#include <map>
#include <queue>

namespace symm_mutations {

// ============================================================================
// compute_bijection: reused from symm_init.cpp
// Maps ops from src component to dst component via Merkle fingerprint +
// local topo depth matching.
// ============================================================================

static std::map<size_t, size_t> compute_bijection(
    const FlatSet<size_t>& src, const FlatSet<size_t>& dst,
    const DAG& dag, const MerkleHashes& merkle)
{
    auto local_depths = [&](const FlatSet<size_t>& comp)
        -> std::map<size_t, size_t>
    {
        std::map<size_t, size_t> depth;
        std::queue<size_t> q;
        for (auto op : comp) {
            bool is_source = true;
            for (auto p : dag.op_preds[op])
                if (comp.count(p)) { is_source = false; break; }
            if (is_source) { depth[op] = 0; q.push(op); }
        }
        while (!q.empty()) {
            size_t u = q.front(); q.pop();
            for (auto v : dag.op_succs[u]) {
                if (!comp.count(v)) continue;
                if (!depth.count(v) || depth[u] + 1 > depth[v]) {
                    depth[v] = depth[u] + 1;
                    q.push(v);
                }
            }
        }
        for (auto op : comp)
            if (!depth.count(op)) depth[op] = 0;
        return depth;
    };

    auto make_ranked = [&](const FlatSet<size_t>& comp)
        -> std::vector<std::pair<std::tuple<size_t, size_t, size_t>, size_t>>
    {
        auto depths = local_depths(comp);
        std::vector<std::pair<std::tuple<size_t, size_t, size_t>, size_t>> ranked;
        for (auto op : comp)
            ranked.push_back({{merkle.init[op], depths[op],
                               dag.topo_position(op)}, op});
        std::sort(ranked.begin(), ranked.end());
        return ranked;
    };

    auto src_ranked = make_ranked(src);
    auto dst_ranked = make_ranked(dst);

    std::map<size_t, size_t> bij;
    size_t n = std::min(src_ranked.size(), dst_ranked.size());
    for (size_t i = 0; i < n; i++)
        bij[src_ranked[i].second] = dst_ranked[i].second;
    return bij;
}

// ============================================================================
// Core primitive: extract ops from partition, inject new grouping
//
// 1. Remove target_ops from their current groups
// 2. Split disconnected remainders into components
// 3. Kill empty groups
// 4. Re-evaluate costs on affected groups
// 5. Create new groups from injected configuration
// ============================================================================

static void extract_and_inject(
    Partition& part,
    const FlatSet<size_t>& target_ops,
    const std::vector<FlatSet<size_t>>& new_groups)
{
    // Phase 1: Extract target_ops from existing groups
    FlatSet<size_t> affected_groups;
    for (auto op : target_ops)
        for (auto gi : part.groups_of(op))
            affected_groups.insert(gi);

    for (auto gi : affected_groups) {
        if (!part.groups[gi].alive) continue;

        // Remove target ops from this group
        FlatSet<size_t> remaining;
        for (auto op : part.groups[gi].ops)
            if (!target_ops.count(op))
                remaining.insert(op);

        if (remaining.empty()) {
            // Group entirely consumed — kill it
            part.groups[gi].alive = false;
            part.groups[gi].gen++;
            continue;
        }

        if (remaining.size() == part.groups[gi].ops.size())
            continue;  // no ops removed from this group

        // Check if remainder is still connected
        auto components = structural_ops::connected_components(remaining,
                                                                *part.dag);
        // First component stays in the original group
        part.groups[gi].ops = std::move(components[0]);
        part.groups[gi].cost = part.eval_set(part.groups[gi].ops);
        part.groups[gi].gen++;

        // Additional components become new groups
        for (size_t c = 1; c < components.size(); c++) {
            double cost = part.eval_set(components[c]);
            part.add_group(std::move(components[c]), cost);
        }
    }

    // Phase 2: Inject new groups
    for (auto& ng : new_groups) {
        if (ng.empty()) continue;
        double cost = part.eval_set(ng);
        if (cost >= 1e17) {
            // Infeasible as single group — add as singletons
            for (auto op : ng) {
                double sc = part.eval_set({op});
                part.add_group({op}, sc);
            }
        } else {
            part.add_group(ng, cost);
        }
    }

    // Phase 3+4: Coverage sweep + redundant singleton cleanup.
    // Single rebuild_index covers both phases.
    part.rebuild_index();

    // Phase 3: Ensure every target_op is in at least one alive group.
    bool added_in_phase3 = false;
    for (auto op : target_ops) {
        bool covered = false;
        for (auto gi : part.groups_of(op))
            if (part.groups[gi].alive) { covered = true; break; }
        if (!covered) {
            double cost = part.eval_set({op});
            part.add_group({op}, cost);
            added_in_phase3 = true;
        }
    }

    // Rebuild only if Phase 3 added singletons (add_group doesn't update index)
    if (added_in_phase3) part.rebuild_index();

    // Phase 4: Kill redundant singletons for ops now in multi-op injected groups
    FlatSet<size_t> injected_ops;
    for (auto& ng : new_groups)
        for (auto op : ng)
            injected_ops.insert(op);

    bool killed_in_phase4 = false;
    for (size_t gi = 0; gi < part.groups.size(); gi++) {
        if (!part.groups[gi].alive) continue;
        if (part.groups[gi].ops.size() != 1) continue;
        size_t op = *part.groups[gi].ops.begin();
        if (injected_ops.count(op)) {
            bool in_other = false;
            for (auto gj : part.groups_of(op))
                if (gj != gi && part.groups[gj].alive) { in_other = true; break; }
            if (in_other) {
                part.groups[gi].alive = false;
                part.groups[gi].gen++;
                killed_in_phase4 = true;
            }
        }
    }

    if (killed_in_phase4) part.rebuild_index();
}

// ============================================================================
// Extract current grouping config for a set of ops from a partition
//
// Returns the groups that are fully within the op set.
// Groups that mix pattern ops with non-pattern ops are excluded
// (the extract_and_inject will handle them via surgical extraction).
// ============================================================================

// ============================================================================
// Extract current grouping config for a set of ops from a partition
// ============================================================================

std::vector<FlatSet<size_t>> extract_config_from_partition(
    const Partition& part,
    const FlatSet<size_t>& ops)
{
    std::vector<FlatSet<size_t>> config;
    FlatSet<size_t> seen_groups;
    FlatSet<size_t> covered;

    for (auto op : ops) {
        for (auto gi : part.groups_of(op)) {
            if (seen_groups.count(gi)) continue;
            seen_groups.insert(gi);
            if (!part.groups[gi].alive) continue;

            // Collect ops from this group that are in our target set
            FlatSet<size_t> group_ops;
            for (auto gop : part.groups[gi].ops)
                if (ops.count(gop))
                    group_ops.insert(gop);

            if (group_ops.size() >= 2) {
                config.push_back(std::move(group_ops));
                for (auto op2 : config.back())
                    covered.insert(op2);
            }
        }
    }

    // Any uncovered ops become singletons (implicit — not added to config)
    return config;
}

// ============================================================================
// Map a config from one component to another via bijection
// ============================================================================

static std::vector<FlatSet<size_t>> map_config(
    const std::vector<FlatSet<size_t>>& config,
    const std::map<size_t, size_t>& bij)
{
    std::vector<FlatSet<size_t>> mapped;
    for (auto& group : config) {
        FlatSet<size_t> mg;
        bool complete = true;
        for (auto op : group) {
            auto it = bij.find(op);
            if (it != bij.end())
                mg.insert(it->second);
            else
                complete = false;
        }
        if (complete && mg.size() == group.size())
            mapped.push_back(std::move(mg));
    }
    return mapped;
}

// ============================================================================
// Build pattern context
// ============================================================================

PatternContext build_context(
    const Problem& prob, const DAG& dag,
    const std::vector<SymmetricPattern>& parallel,
    const std::vector<SeriesPattern>& series,
    const MerkleHashes& merkle)
{
    PatternContext ctx;
    ctx.parallel = parallel;
    ctx.series = series;
    ctx.merkle = merkle;

    // For now, solutions are populated externally (from symm_init cache).
    // build_context just assembles the structural data.
    ctx.parallel_solutions.resize(parallel.size());
    ctx.series_solutions.resize(series.size());

    return ctx;
}

// ============================================================================
// Mutation 1: Inject representative solution
// ============================================================================

std::optional<Partition> inject_representative_solution(
    Partition part,
    const PatternContext& ctx,
    const Problem& prob, const DAG& dag,
    std::mt19937& rng)
{
    // Collect usable patterns (have cached solutions)
    // Minimum coverage: patterns covering <5% of ops are too small to be useful.
    size_t min_ops = std::max<size_t>(4, prob.num_ops() / 20);

    std::vector<size_t> usable_parallel;
    for (size_t i = 0; i < ctx.parallel.size(); i++)
        if (!ctx.parallel_solutions[i].empty() &&
            ctx.parallel[i].symmetry >= 2 &&
            ctx.parallel[i].component_size() * ctx.parallel[i].symmetry >= min_ops)
            usable_parallel.push_back(i);

    std::vector<size_t> usable_series;
    for (size_t i = 0; i < ctx.series.size(); i++)
        if (!ctx.series_solutions[i].empty() &&
            ctx.series[i].num_blocks >= 2 &&
            ctx.series[i].total_ops() >= min_ops)
            usable_series.push_back(i);

    size_t total = usable_parallel.size() + usable_series.size();
    if (total == 0) return std::nullopt;

    // Pick pattern weighted by coverage (total ops in pattern).
    std::vector<double> weights;
    for (auto pi : usable_parallel)
        weights.push_back((double)(ctx.parallel[pi].component_size() *
                                    ctx.parallel[pi].symmetry));
    for (auto si : usable_series)
        weights.push_back((double)ctx.series[si].total_ops());
    std::discrete_distribution<size_t> pat_dist(weights.begin(), weights.end());
    size_t pick = pat_dist(rng);

    if (pick < usable_parallel.size()) {
        // Parallel pattern
        size_t pi = usable_parallel[pick];
        auto& pat = ctx.parallel[pi];
        auto& solutions = ctx.parallel_solutions[pi];

        // Pick best cached solution variant by cost on representative
        size_t best_si = 0;
        double best_sol_cost = 1e18;
        for (size_t si = 0; si < solutions.size(); si++) {
            double cost = 0;
            for (auto& g : solutions[si].groups)
                cost += part.eval_set(g);
            if (cost < best_sol_cost) {
                best_sol_cost = cost;
                best_si = si;
            }
        }
        auto& sol = solutions[best_si];

        // Collect all ops in the pattern
        FlatSet<size_t> all_pattern_ops;
        for (auto& comp : pat.components)
            for (auto op : comp)
                all_pattern_ops.insert(op);

        // Build new groups: representative solution replicated to all components
        std::vector<FlatSet<size_t>> new_groups;

        // Representative (component 0) — use solution directly
        for (auto& g : sol.groups)
            new_groups.push_back(g);

        // Other components — map via bijection
        for (size_t ci = 1; ci < pat.components.size(); ci++) {
            auto bij = compute_bijection(pat.components[0],
                                         pat.components[ci],
                                         dag, ctx.merkle);
            auto mapped = map_config(sol.groups, bij);
            for (auto& g : mapped)
                new_groups.push_back(std::move(g));
        }

        extract_and_inject(part, all_pattern_ops, new_groups);
        return part;

    } else {
        // Series pattern
        size_t si = usable_series[pick - usable_parallel.size()];
        auto& pat = ctx.series[si];
        auto& solutions = ctx.series_solutions[si];
        // Pick best cached solution by cost on representative block
        size_t best_si2 = 0;
        double best_sol_cost2 = 1e18;
        for (size_t si2 = 0; si2 < solutions.size(); si2++) {
            double cost = 0;
            for (auto& g : solutions[si2].groups)
                cost += part.eval_set(g);
            if (cost < best_sol_cost2) {
                best_sol_cost2 = cost;
                best_si2 = si2;
            }
        }
        auto& sol = solutions[best_si2];

        // Series: blocks[i] ops correspond positionally
        FlatSet<size_t> all_pattern_ops;
        for (auto& block : pat.blocks)
            for (auto op : block)
                all_pattern_ops.insert(op);

        std::vector<FlatSet<size_t>> new_groups;

        // Representative (block 0) — direct
        for (auto& g : sol.groups)
            new_groups.push_back(g);

        // Other blocks — positional mapping
        for (size_t bi = 1; bi < pat.blocks.size(); bi++) {
            // Build positional bijection: rep[j] → block_i[j]
            std::map<size_t, size_t> bij;
            for (size_t j = 0; j < pat.representative.size() &&
                              j < pat.blocks[bi].size(); j++)
                bij[pat.representative[j]] = pat.blocks[bi][j];

            auto mapped = map_config(sol.groups, bij);
            for (auto& g : mapped)
                new_groups.push_back(std::move(g));
        }

        extract_and_inject(part, all_pattern_ops, new_groups);
        return part;
    }
}

// ============================================================================
// Mutation 2: Cross-representative alignment
// ============================================================================

std::optional<Partition> align_symmetric_reps(
    Partition part,
    const PatternContext& ctx,
    const Problem& prob, const DAG& dag,
    std::mt19937& rng)
{
    // Collect usable patterns (filter tiny ones: <5% coverage)
    size_t min_ops_align = std::max<size_t>(4, prob.num_ops() / 20);

    std::vector<size_t> usable_parallel;
    for (size_t i = 0; i < ctx.parallel.size(); i++)
        if (ctx.parallel[i].symmetry >= 2 &&
            ctx.parallel[i].component_size() >= 2 &&
            ctx.parallel[i].component_size() * ctx.parallel[i].symmetry >= min_ops_align)
            usable_parallel.push_back(i);

    std::vector<size_t> usable_series;
    for (size_t i = 0; i < ctx.series.size(); i++)
        if (ctx.series[i].num_blocks >= 2 &&
            ctx.series[i].block_size >= 2 &&
            ctx.series[i].total_ops() >= min_ops_align)
            usable_series.push_back(i);

    size_t total = usable_parallel.size() + usable_series.size();
    if (total == 0) return std::nullopt;

    // Pick pattern weighted by coverage (total ops in pattern).
    std::vector<double> weights;
    for (auto pi : usable_parallel)
        weights.push_back((double)(ctx.parallel[pi].component_size() *
                                    ctx.parallel[pi].symmetry));
    for (auto si : usable_series)
        weights.push_back((double)ctx.series[si].total_ops());
    std::discrete_distribution<size_t> pat_dist(weights.begin(), weights.end());
    size_t pick = pat_dist(rng);

    if (pick < usable_parallel.size()) {
        // Parallel pattern
        size_t pi = usable_parallel[pick];
        auto& pat = ctx.parallel[pi];
        size_t k = pat.components.size();

        // Pick donor component randomly, biased toward cheaper grouping cost.
        size_t donor = 0;
        std::vector<FlatSet<size_t>> donor_config;
        {
            std::vector<std::pair<double, size_t>> candidates;  // (cost, index)
            std::vector<std::vector<FlatSet<size_t>>> configs(k);
            for (size_t ci = 0; ci < k; ci++) {
                configs[ci] = extract_config_from_partition(part, pat.components[ci]);
                if (configs[ci].empty()) continue;
                double cost = 0;
                for (auto& g : configs[ci]) cost += part.eval_set(g);
                candidates.push_back({cost, ci});
            }
            if (candidates.empty()) return std::nullopt;
            // Softmax selection: weight = exp(-cost / temperature)
            double min_cost = candidates[0].first;
            for (auto& [c, _] : candidates) min_cost = std::min(min_cost, c);
            std::vector<double> weights;
            for (auto& [c, _] : candidates)
                weights.push_back(std::exp(-(c - min_cost) / std::max(1.0, min_cost * 0.1)));
            std::discrete_distribution<size_t> donor_dist(weights.begin(), weights.end());
            size_t picked = donor_dist(rng);
            donor = candidates[picked].second;
            donor_config = std::move(configs[donor]);
        }

        // Collect all ops in the pattern
        FlatSet<size_t> all_pattern_ops;
        for (auto& comp : pat.components)
            for (auto op : comp)
                all_pattern_ops.insert(op);

        // Build new groups: donor config replicated to all components
        std::vector<FlatSet<size_t>> new_groups;

        // Donor keeps its own config
        for (auto& g : donor_config)
            new_groups.push_back(g);

        // Map to other components
        for (size_t ci = 0; ci < k; ci++) {
            if (ci == donor) continue;
            auto bij = compute_bijection(pat.components[donor],
                                         pat.components[ci],
                                         dag, ctx.merkle);
            auto mapped = map_config(donor_config, bij);
            for (auto& g : mapped)
                new_groups.push_back(std::move(g));
        }

        extract_and_inject(part, all_pattern_ops, new_groups);
        return part;

    } else {
        // Series pattern
        size_t si = usable_series[pick - usable_parallel.size()];
        auto& pat = ctx.series[si];
        size_t k = pat.blocks.size();

        // Pick donor block randomly, biased toward cheaper grouping cost.
        size_t donor = 0;
        std::vector<FlatSet<size_t>> donor_config;
        {
            std::vector<std::pair<double, size_t>> candidates;
            std::vector<std::vector<FlatSet<size_t>>> configs(k);
            for (size_t bi = 0; bi < k; bi++) {
                FlatSet<size_t> block_ops(pat.blocks[bi].begin(),
                                           pat.blocks[bi].end());
                configs[bi] = extract_config_from_partition(part, block_ops);
                if (configs[bi].empty()) continue;
                double cost = 0;
                for (auto& g : configs[bi]) cost += part.eval_set(g);
                candidates.push_back({cost, bi});
            }
            if (candidates.empty()) return std::nullopt;
            double min_cost = candidates[0].first;
            for (auto& [c, _] : candidates) min_cost = std::min(min_cost, c);
            std::vector<double> weights;
            for (auto& [c, _] : candidates)
                weights.push_back(std::exp(-(c - min_cost) / std::max(1.0, min_cost * 0.1)));
            std::discrete_distribution<size_t> donor_dist(weights.begin(), weights.end());
            size_t picked = donor_dist(rng);
            donor = candidates[picked].second;
            donor_config = std::move(configs[donor]);
        }

        FlatSet<size_t> all_pattern_ops;
        for (auto& block : pat.blocks)
            for (auto op : block)
                all_pattern_ops.insert(op);

        std::vector<FlatSet<size_t>> new_groups;

        // Donor keeps config
        for (auto& g : donor_config)
            new_groups.push_back(g);

        // Map to other blocks via positional bijection
        for (size_t bi = 0; bi < k; bi++) {
            if (bi == donor) continue;
            std::map<size_t, size_t> bij;
            for (size_t j = 0; j < pat.blocks[donor].size() &&
                              j < pat.blocks[bi].size(); j++)
                bij[pat.blocks[donor][j]] = pat.blocks[bi][j];

            auto mapped = map_config(donor_config, bij);
            for (auto& g : mapped)
                new_groups.push_back(std::move(g));
        }

        extract_and_inject(part, all_pattern_ops, new_groups);
        return part;
    }
}

} // namespace symm_mutations

// ============================================================================
// Solution-level symmetry mutations
// ============================================================================

namespace symm_mutations {

// Core primitive: extract target_ops from steps, inject new_groups as new steps.
// Returns reconstructed step sequence in topological order.
static std::optional<std::vector<ScheduleStep>> extract_and_inject_steps(
    std::vector<ScheduleStep> steps,
    const FlatSet<size_t>& target_ops,
    const std::vector<FlatSet<size_t>>& new_groups,
    const Problem& prob, const DAG& dag)
{
    // Phase 1: Remove target_ops from existing steps
    std::vector<FlatSet<size_t>> rebuilt_groups;

    for (auto& step : steps) {
        FlatSet<size_t> remaining;
        for (auto op : step.subgraph.ops())
            if (!target_ops.count(op))
                remaining.insert(op);

        if (remaining.empty()) continue;

        if (remaining.size() == step.subgraph.ops().size()) {
            // Untouched step — keep as-is
            rebuilt_groups.push_back(std::move(remaining));
            continue;
        }

        // Step was modified — split into connected components
        auto components = structural_ops::connected_components(remaining, dag);
        for (auto& comp : components)
            rebuilt_groups.push_back(std::move(comp));
    }

    // Phase 2: Add injected groups
    for (auto& ng : new_groups) {
        if (!ng.empty())
            rebuilt_groups.push_back(ng);
    }

    // Phase 2b: Ensure every target_op is covered.
    // new_groups may not include all target_ops (singletons are implicit in
    // RepSolution; map_config drops incomplete bijections).
    {
        FlatSet<size_t> covered_by_rebuilt;
        for (auto& g : rebuilt_groups)
            for (auto op : g)
                covered_by_rebuilt.insert(op);

        for (auto op : target_ops) {
            if (!covered_by_rebuilt.count(op))
                rebuilt_groups.push_back({op});
        }
    }

    // Phase 3: Topo-sort all groups into a valid step sequence
    size_t k = rebuilt_groups.size();

    // Build inter-group DAG
    // Map each op to its group index
    std::map<size_t, size_t> op_to_gi;
    for (size_t gi = 0; gi < k; gi++)
        for (auto op : rebuilt_groups[gi])
            op_to_gi[op] = gi;

    std::vector<FlatSet<size_t>> group_succs(k);
    std::vector<int> in_deg(k, 0);
    for (size_t gi = 0; gi < k; gi++) {
        for (auto op : rebuilt_groups[gi]) {
            for (auto succ : dag.op_succs[op]) {
                auto it = op_to_gi.find(succ);
                if (it != op_to_gi.end() && it->second != gi) {
                    if (group_succs[gi].insert(it->second).second)
                        in_deg[it->second]++;
                }
            }
        }
    }

    // Kahn's
    std::vector<size_t> order;
    std::queue<size_t> q;
    for (size_t gi = 0; gi < k; gi++)
        if (in_deg[gi] == 0) q.push(gi);
    while (!q.empty()) {
        size_t u = q.front(); q.pop();
        order.push_back(u);
        for (auto v : group_succs[u])
            if (--in_deg[v] == 0) q.push(v);
    }
    if (order.size() != k) return std::nullopt;  // cycle

    // Phase 4: Build ScheduleSteps in topo order
    std::vector<ScheduleStep> result;
    result.reserve(k);
    for (auto gi : order) {
        auto& ops = rebuilt_groups[gi];
        auto sg = Subgraph::create(prob, dag, {ops.begin(), ops.end()});
        if (!sg) return std::nullopt;

        ScheduleStep ns;
        ns.subgraph = std::move(*sg);
        // retain_these will be rebuilt by the caller's rebuild_from / recompute_costs
        result.push_back(std::move(ns));
    }

    return result;
}

// Extract config from steps (like extract_config_from_partition but for steps)
std::vector<FlatSet<size_t>> extract_config_from_steps(
    const std::vector<ScheduleStep>& steps,
    const FlatSet<size_t>& ops)
{
    std::vector<FlatSet<size_t>> config;
    FlatSet<size_t> seen_steps;

    for (auto op : ops) {
        for (size_t si = 0; si < steps.size(); si++) {
            if (seen_steps.count(si)) continue;
            bool has_op = false;
            for (auto sop : steps[si].subgraph.ops())
                if (sop == op) { has_op = true; break; }
            if (!has_op) continue;
            seen_steps.insert(si);

            FlatSet<size_t> group_ops;
            for (auto sop : steps[si].subgraph.ops())
                if (ops.count(sop))
                    group_ops.insert(sop);

            if (group_ops.size() >= 2)
                config.push_back(std::move(group_ops));
        }
    }
    return config;
}

// ============================================================================
// Solution-level Mutation 1: Inject representative solution
// ============================================================================

std::optional<std::vector<ScheduleStep>> inject_representative_solution_steps(
    std::vector<ScheduleStep> steps,
    const PatternContext& ctx,
    const Problem& prob, const DAG& dag,
    std::mt19937& rng)
{
    // Collect usable parallel patterns with cached solutions
    std::vector<size_t> usable;
    for (size_t i = 0; i < ctx.parallel.size(); i++)
        if (!ctx.parallel_solutions[i].empty() && ctx.parallel[i].symmetry >= 2)
            usable.push_back(i);

    // Also check series
    std::vector<size_t> usable_series;
    for (size_t i = 0; i < ctx.series.size(); i++)
        if (!ctx.series_solutions[i].empty() && ctx.series[i].num_blocks >= 2)
            usable_series.push_back(i);

    size_t total = usable.size() + usable_series.size();
    if (total == 0) return std::nullopt;

    size_t pick = rng() % total;

    FlatSet<size_t> all_pattern_ops;
    std::vector<FlatSet<size_t>> new_groups;

    if (pick < usable.size()) {
        size_t pi = usable[pick];
        auto& pat = ctx.parallel[pi];
        auto& sol = ctx.parallel_solutions[pi][rng() % ctx.parallel_solutions[pi].size()];

        for (auto& comp : pat.components)
            for (auto op : comp) all_pattern_ops.insert(op);

        for (auto& g : sol.groups) new_groups.push_back(g);
        for (size_t ci = 1; ci < pat.components.size(); ci++) {
            auto bij = compute_bijection(pat.components[0], pat.components[ci],
                                         dag, ctx.merkle);
            for (auto& g : map_config(sol.groups, bij))
                new_groups.push_back(std::move(g));
        }
    } else {
        size_t si = usable_series[pick - usable.size()];
        auto& pat = ctx.series[si];
        auto& sol = ctx.series_solutions[si][rng() % ctx.series_solutions[si].size()];

        for (auto& block : pat.blocks)
            for (auto op : block) all_pattern_ops.insert(op);

        for (auto& g : sol.groups) new_groups.push_back(g);
        for (size_t bi = 1; bi < pat.blocks.size(); bi++) {
            std::map<size_t, size_t> bij;
            for (size_t j = 0; j < pat.representative.size() &&
                              j < pat.blocks[bi].size(); j++)
                bij[pat.representative[j]] = pat.blocks[bi][j];
            for (auto& g : map_config(sol.groups, bij))
                new_groups.push_back(std::move(g));
        }
    }

    return extract_and_inject_steps(std::move(steps), all_pattern_ops,
                                     new_groups, prob, dag);
}

// ============================================================================
// Solution-level Mutation 2: Cross-representative alignment
// ============================================================================

std::optional<std::vector<ScheduleStep>> align_symmetric_reps_steps(
    std::vector<ScheduleStep> steps,
    const PatternContext& ctx,
    const Problem& prob, const DAG& dag,
    std::mt19937& rng)
{
    std::vector<size_t> usable_par;
    for (size_t i = 0; i < ctx.parallel.size(); i++)
        if (ctx.parallel[i].symmetry >= 2 && ctx.parallel[i].component_size() >= 2)
            usable_par.push_back(i);

    std::vector<size_t> usable_ser;
    for (size_t i = 0; i < ctx.series.size(); i++)
        if (ctx.series[i].num_blocks >= 2 && ctx.series[i].block_size >= 2)
            usable_ser.push_back(i);

    size_t total = usable_par.size() + usable_ser.size();
    if (total == 0) return std::nullopt;

    size_t pick = rng() % total;

    FlatSet<size_t> all_pattern_ops;
    std::vector<FlatSet<size_t>> new_groups;

    if (pick < usable_par.size()) {
        size_t pi = usable_par[pick];
        auto& pat = ctx.parallel[pi];
        size_t donor = rng() % pat.components.size();

        auto donor_config = extract_config_from_steps(steps, pat.components[donor]);
        if (donor_config.empty()) return std::nullopt;

        for (auto& comp : pat.components)
            for (auto op : comp) all_pattern_ops.insert(op);

        for (auto& g : donor_config) new_groups.push_back(g);
        for (size_t ci = 0; ci < pat.components.size(); ci++) {
            if (ci == donor) continue;
            auto bij = compute_bijection(pat.components[donor], pat.components[ci],
                                         dag, ctx.merkle);
            for (auto& g : map_config(donor_config, bij))
                new_groups.push_back(std::move(g));
        }
    } else {
        size_t si = usable_ser[pick - usable_par.size()];
        auto& pat = ctx.series[si];
        size_t donor = rng() % pat.blocks.size();

        FlatSet<size_t> donor_ops(pat.blocks[donor].begin(), pat.blocks[donor].end());
        auto donor_config = extract_config_from_steps(steps, donor_ops);
        if (donor_config.empty()) return std::nullopt;

        for (auto& block : pat.blocks)
            for (auto op : block) all_pattern_ops.insert(op);

        for (auto& g : donor_config) new_groups.push_back(g);
        for (size_t bi = 0; bi < pat.blocks.size(); bi++) {
            if (bi == donor) continue;
            std::map<size_t, size_t> bij;
            for (size_t j = 0; j < pat.blocks[donor].size() &&
                              j < pat.blocks[bi].size(); j++)
                bij[pat.blocks[donor][j]] = pat.blocks[bi][j];
            for (auto& g : map_config(donor_config, bij))
                new_groups.push_back(std::move(g));
        }
    }

    return extract_and_inject_steps(std::move(steps), all_pattern_ops,
                                     new_groups, prob, dag);
}

} // namespace symm_mutations