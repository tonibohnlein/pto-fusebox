#include "search/verbose.h"
#include "search/fm_outer.h"
#include "search/local_search.h"
#include <iostream>
#include <cmath>
#include <algorithm>

FMOuterResult fm_outer_loop(Partition part, const FMOuterConfig& cfg) {
    FMOuterResult result;
    result.best_partition = part;
    result.best_cost = part.total_cost();

    int no_improve = 0;
    auto outer_start = SteadyClock::now();

    double base_drift = cfg.pass_config.max_drift_fraction;

    for (int pass = 0; pass < cfg.max_passes; pass++) {
        if (SteadyClock::now() >= cfg.deadline) break;

        // Cooling schedule: temperature decays from 1.0 to ~0.1 over passes.
        double progress = (double)pass / std::max(1, cfg.max_passes - 1);
        double temperature = 0.1 + 0.9 * 0.5 * (1.0 + std::cos(progress * M_PI));

        double effective_drift = std::clamp(base_drift * temperature, 0.05, 2.0);

        // Each pass starts from the best found so far
        Partition current(part);

        FMConfig pass_cfg = cfg.pass_config;
        pass_cfg.seed = (unsigned)(pass_cfg.seed + pass * 7);
        pass_cfg.max_drift_fraction = effective_drift;
        pass_cfg.deadline = cfg.deadline;

        auto pass_result = fm_inner_pass(std::move(current), pass_cfg);
        result.total_passes++;
        result.total_moves += pass_result.moves_applied;

        if (pass_result.moves_applied == 0) {
            if (++no_improve >= cfg.max_no_improve) break;
            continue;
        }

        // Greedy descent on FM best
        Partition candidate = greedy_descent(std::move(pass_result.best_partition));
        double candidate_cost = candidate.total_cost();

        if (candidate_cost < result.best_cost - 0.001) {
            double improvement = result.best_cost - candidate_cost;
            result.best_cost = candidate_cost;
            result.best_partition = candidate;
            // Start next pass from improved partition
            part = std::move(candidate);
            result.improving_passes++;
            no_improve = 0;

            if (g_verbose)
                std::cerr << "    FM pass " << pass << ": improved to "
                          << result.best_cost << " (delta=" << improvement << ")\n";
        } else {
            if (++no_improve >= cfg.max_no_improve) {
                if (g_verbose)
                    std::cerr << "    FM: " << cfg.max_no_improve
                              << " non-improving passes, stopping\n";
                break;
            }
        }
    }

    result.elapsed_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
        SteadyClock::now() - outer_start).count();

    if (g_verbose)
        std::cerr << "    FM done: " << result.total_passes << " passes, "
                  << result.improving_passes << " improved, cost="
                  << result.best_cost << " (" << result.elapsed_ms << "ms)\n";

    return result;
}