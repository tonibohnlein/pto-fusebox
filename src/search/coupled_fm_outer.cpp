#include "search/coupled_fm_outer.h"
#include "search/local_search.h"
#include "search/verbose.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>

CoupledFMOuterResult coupled_fm_outer_loop(CoupledPartition        cp,
                                            const FlatSet<size_t>& feasibly_ret,
                                            const FMOuterConfig&    cfg,
                                            CostCache*              cache) {
    // Ensure the cache pointer is set — coupling eval uses it directly.
    if (cache) cp.part.cache = cache;

    // Ensure coupling arrays are sized
    while (cp.next_group.size() < cp.part.groups.size()) cp.next_group.push_back(SIZE_MAX);
    while (cp.prev_group.size() < cp.part.groups.size()) cp.prev_group.push_back(SIZE_MAX);

    CoupledFMOuterResult result;
    result.best_cp   = cp;
    result.best_cost = cp.total_cost();

    int no_improve  = 0;
    auto outer_start = SteadyClock::now();

    double base_drift = cfg.pass_config.max_drift_fraction;

    for (int pass = 0; pass < cfg.max_passes; pass++) {
        if (SteadyClock::now() >= cfg.deadline) break;

        double progress     = (double)pass / std::max(1, cfg.max_passes - 1);
        double temperature  = 0.1 + 0.9 * 0.5 * (1.0 + std::cos(progress * M_PI));
        double effective_drift = std::clamp(base_drift * temperature, 0.05, 2.0);

        // Each pass starts from the same input (explore different neighbourhood)
        CoupledPartition current = cp;

        FMConfig pass_cfg               = cfg.pass_config;
        pass_cfg.seed                   = (unsigned)(pass_cfg.seed + pass * 7);
        pass_cfg.max_drift_fraction     = effective_drift;
        pass_cfg.deadline               = cfg.deadline;

        auto pass_result = coupled_fm_inner_pass(std::move(current), feasibly_ret, pass_cfg);
        result.total_passes++;
        result.total_moves += pass_result.moves_applied;

        double        pass_best_cost = pass_result.best_cost;
        CoupledPartition pass_best_cp  = std::move(pass_result.best_cp);

        // Greedy descent on end state — explores a different basin
        if (pass_result.moves_applied > 0) {
            result.end_cost = pass_result.end_cost;
            result.end_cp   = std::move(pass_result.end_cp);

            double greedy_cost = coupling_greedy_descent(result.end_cp, feasibly_ret,
                                                          cfg.deadline);
            if (greedy_cost < pass_best_cost - 0.001) {
                pass_best_cost = greedy_cost;
                pass_best_cp   = result.end_cp;
            }
        }

        if (pass_best_cost < result.best_cost - 0.001) {
            double improvement = result.best_cost - pass_best_cost;
            result.best_cost   = pass_best_cost;
            result.best_cp     = std::move(pass_best_cp);
            result.improving_passes++;
            no_improve = 0;

            if (g_verbose)
                std::cerr << "    Coupled FM pass " << pass << ": improved to "
                          << result.best_cost << " (delta=" << improvement << ")\n";
        } else {
            if (++no_improve >= cfg.max_no_improve) {
                if (g_verbose)
                    std::cerr << "    Coupled FM: " << cfg.max_no_improve
                              << " non-improving passes, stopping\n";
                break;
            }
        }
    }

    result.elapsed_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
        SteadyClock::now() - outer_start).count();

    // Rebuild the group DAG (group_succs / group_preds) so to_solution() can
    // topological-sort chains correctly.  partition moves (EJECT, SPLIT, etc.)
    // and the checkpoint restore in inner_pass both skip rebuild_group_dag().
    result.best_cp.part.rebuild_index();
    result.best_cp.part.rebuild_group_dag();
    result.best_cp.invalidate_couplings();
    // Remove any coupling links that now form a chain-level DAG cycle (FM passes
    // may have introduced new group edges via partition moves).
    result.best_cp.fix_chain_couplings();

#ifndef NDEBUG
    // Eval-time checks should guarantee the checkpoint is valid.  Assert to
    // catch regressions — never silently fall back in release builds.
    if (!result.best_cp.part.is_acyclic()) {
        std::cerr << "BUG: coupled_fm_outer_loop: best checkpoint is cyclic\n";
        assert(false && "cyclic checkpoint — eval-time check missed a move");
    }
    if (partition_has_gap(result.best_cp.part, result.best_cp.all_retained_tensors())) {
        std::cerr << "BUG: coupled_fm_outer_loop: best checkpoint has ephemeral gap\n";
        assert(false && "ephemeral gap in checkpoint — eval-time check missed a move");
    }
#endif

    if (g_verbose)
        std::cerr << "    Coupled FM done: " << result.total_passes << " passes, "
                  << result.improving_passes << " improved, cost="
                  << result.best_cost << " (" << result.elapsed_ms << "ms)\n";

    return result;
}
