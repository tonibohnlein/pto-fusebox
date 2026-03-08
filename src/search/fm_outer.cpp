#include "search/fm_outer.h"
#include <iostream>

FMOuterResult fm_outer_loop(Partition part, const FMOuterConfig& cfg) {
    FMOuterResult result;
    result.best_partition = part;
    result.best_cost = part.total_cost();

    int no_improve = 0;

    for (int pass = 0; pass < cfg.max_passes; pass++) {
        // Start each pass from the best known partition
        Partition current = result.best_partition;

        // Vary seed each pass for different initial active subsets
        FMConfig pass_cfg = cfg.pass_config;
        pass_cfg.seed = (unsigned)(42 + pass * 7);

        auto pass_result = fm_inner_pass(std::move(current), pass_cfg);
        result.total_passes++;
        result.total_moves += pass_result.moves_applied;

        if (pass_result.best_cost < result.best_cost - 0.001) {
            double improvement = result.best_cost - pass_result.best_cost;
            result.best_cost = pass_result.best_cost;
            result.best_partition = std::move(pass_result.best_partition);
            result.improving_passes++;
            no_improve = 0;

            std::cerr << "    FM pass " << pass << ": improved to "
                      << result.best_cost << " (delta=" << improvement
                      << ", +" << pass_result.moves_positive
                      << " -" << pass_result.moves_negative << ")\n";
        } else {
            no_improve++;
            if (no_improve >= cfg.max_no_improve) {
                std::cerr << "    FM: " << cfg.max_no_improve
                          << " passes without improvement, stopping\n";
                break;
            }
        }
    }

    std::cerr << "    FM done: " << result.total_passes << " passes, "
              << result.improving_passes << " improved, "
              << result.total_moves << " total moves, cost="
              << result.best_cost << "\n";

    return result;
}
