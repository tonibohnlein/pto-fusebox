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

    // Base exploration parameters from config
    double base_floor = cfg.pass_config.floor_fraction;
    double base_drift = cfg.pass_config.max_drift_fraction;

    // Adaptive perturbation: heat controls how aggressively we explore.
    // Increases on failure, decreases on success (ILS-style).
    double heat = 1.0;
    const double heat_min = 0.1, heat_max = 3.0;
    const double heat_up = 1.3;    // widen on failure
    const double heat_down = 0.7;  // narrow on success

    for (int pass = 0; pass < cfg.max_passes; pass++) {
        // Wall-clock cutoff
        if (SteadyClock::now() >= cfg.deadline) {
            if (g_verbose) std::cerr << "    FM: deadline reached at pass " << pass << "\n";
            break;
        }

        // Cooling schedule: temperature decays from 1.0 to ~0.1 over passes.
        // Uses cosine annealing: smooth, well-behaved, no tunable decay rate.
        // t=0 → temp=1.0, t=max_passes → temp=0.1
        double progress = (double)pass / std::max(1, cfg.max_passes - 1);
        double temperature = 0.1 + 0.9 * 0.5 * (1.0 + std::cos(progress * M_PI));

        // Effective exploration parameters: cooling × adaptive heat
        double effective_floor = base_floor * temperature * heat;
        double effective_drift = base_drift * temperature * heat;

        // Clamp to reasonable range
        effective_floor = std::clamp(effective_floor, 0.02, 1.0);
        effective_drift = std::clamp(effective_drift, 0.05, 2.0);

        // Start each pass from the best known partition
        Partition current = result.best_partition;

        // Configure this pass
        FMConfig pass_cfg = cfg.pass_config;
        pass_cfg.seed = (unsigned)(pass_cfg.seed + pass * 7);
        pass_cfg.floor_fraction = effective_floor;
        pass_cfg.max_drift_fraction = effective_drift;

        auto pass_result = fm_inner_pass(std::move(current), pass_cfg);
        result.total_passes++;
        result.total_moves += pass_result.moves_applied;

        // Keep the last perturbed state for diversity seeding
        if (pass_result.moves_applied > 0) {
            result.end_partition = pass_result.end_partition;
            result.end_cost = pass_result.end_cost;
        }

        if (pass_result.best_cost < result.best_cost - 0.001) {
            // FM improved directly — cool down (narrow search near this basin)
            double improvement = result.best_cost - pass_result.best_cost;
            result.best_cost = pass_result.best_cost;
            result.best_partition = std::move(pass_result.best_partition);
            result.improving_passes++;
            no_improve = 0;
            heat = std::clamp(heat * heat_down, heat_min, heat_max);

            if (g_verbose) std::cerr << "    FM pass " << pass << ": improved to "
                      << result.best_cost << " (delta=" << improvement
                      << " temp=" << temperature << " heat=" << heat << ")\n";
        } else {
            // FM didn't improve. Try greedy-kick on the perturbed end state.
            bool kicked = false;
            if (pass_result.moves_applied > 0) {
                auto greedy_kick = greedy_descent(std::move(pass_result.end_partition));
                if (greedy_kick.total_cost() < result.best_cost - 0.001) {
                    double improvement = result.best_cost - greedy_kick.total_cost();
                    result.best_cost = greedy_kick.total_cost();
                    result.best_partition = std::move(greedy_kick);
                    result.improving_passes++;
                    no_improve = 0;
                    kicked = true;
                    // Kick succeeded — moderate cooling (kick found a new basin,
                    // but FM itself didn't improve, so don't cool too aggressively)
                    heat = std::clamp(heat * 0.9, heat_min, heat_max);

                    if (g_verbose) std::cerr << "    FM pass " << pass
                              << ": greedy-kick escaped to " << result.best_cost
                              << " (delta=" << improvement
                              << " temp=" << temperature << " heat=" << heat << ")\n";
                    continue;
                }
            }

            // Neither FM nor kick improved — heat up (explore wider next time)
            no_improve++;
            heat = std::clamp(heat * heat_up, heat_min, heat_max);

            if (no_improve >= cfg.max_no_improve) {
                if (g_verbose) std::cerr << "    FM: " << cfg.max_no_improve
                          << " consecutive non-improving passes, stopping"
                          << " (temp=" << temperature << " heat=" << heat << ")\n";
                break;
            }
        }
    }

    if (g_verbose) std::cerr << "    FM done: " << result.total_passes << " passes, "
              << result.improving_passes << " improved, "
              << result.total_moves << " total moves, cost="
              << result.best_cost << "\n";

    return result;
}