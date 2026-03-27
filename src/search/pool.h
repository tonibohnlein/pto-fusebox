#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <vector>

// ============================================================================
// DiversityPool: Quality-Diversity pool management.
//
// Maintains a bounded pool of entries that is Pareto-optimal in both
// objective cost and structural diversity. Implements:
//
//   - Insertion with near-duplicate detection + diversity-aware eviction
//   - Softmax selection for mutation (cost-biased, non-deterministic)
//   - Negative-assortative mating for crossover (quality × diversity)
//   - Cached pairwise distance matrix (O(n) per insert, not O(n²))
//
// Template parameter Entry must be move-constructible. The pool does not
// interpret entries — cost and distance are provided via function objects.
//
// Thread safety: the pool owns a mutex. Callers lock it externally:
//   { std::lock_guard lock(pool.mutex()); pool.insert(...); }
//
// References:
//   - MAP-Elites (Mouret & Clune 2015) — quality-diversity framework
//   - Deterministic Crowding (Mahfoud 1995) — near-duplicate replacement
//   - Negative Assortative Mating (Fernandes & Rosa 2001) — diverse crossover
// ============================================================================

struct PoolConfig {
    size_t hard_cap            = 16;    // maximum pool size
    double near_dup_threshold  = 0.05;  // distance below this = duplicate
    double evict_div_margin    = 0.05;  // new must beat victim's nn-dist by this
    double min_diversity       = 0.10;  // floor for "decent diversity"
    double cost_eps            = 0.01;  // cost difference threshold
    double selection_temp      = 0.5;   // softmax temperature for mutation
    size_t tournament_k        = 3;     // tournament size for crossover
};

template<typename Entry>
class DiversityPool {
public:
    using DistFn = std::function<double(const Entry&, const Entry&)>;
    using CostFn = std::function<double(const Entry&)>;

    DiversityPool(PoolConfig cfg, DistFn dist, CostFn cost)
        : cfg_(cfg), dist_(std::move(dist)), cost_(std::move(cost)) {
        entries_.reserve(cfg_.hard_cap);
        dist_cache_.resize(cfg_.hard_cap * cfg_.hard_cap, 0.0);
    }

    // ====================================================================
    // Insertion with diversity-aware eviction
    // ====================================================================

    bool insert(Entry entry) {
        double ec = cost_(entry);

        // 1. Compute distances to all existing entries (fills new_dists)
        std::vector<double> new_dists(entries_.size());
        double min_dist = 1.0;
        size_t closest_idx = 0;
        for (size_t i = 0; i < entries_.size(); i++) {
            new_dists[i] = dist_(entry, entries_[i]);
            if (new_dists[i] < min_dist) {
                min_dist = new_dists[i];
                closest_idx = i;
            }
        }

        // 2. Near-duplicate: only replace if strictly better
        if (!entries_.empty() && min_dist < cfg_.near_dup_threshold) {
            if (ec < cost_(entries_[closest_idx]) - cfg_.cost_eps) {
                replace_at(closest_idx, std::move(entry), new_dists);
                return true;
            }
            return false;
        }

        // 3. Pool not full: just add
        if (entries_.size() < cfg_.hard_cap) {
            append(std::move(entry), new_dists);
            return true;
        }

        // 4. Pool full: diversity-aware eviction
        size_t best_ci = find_best_cost_idx();

        // Find least-contributing entry (smallest nn-distance), skip best
        size_t victim = SIZE_MAX;
        double victim_nn = 2.0;
        for (size_t i = 0; i < entries_.size(); i++) {
            if (i == best_ci) continue;
            double nn = nn_dist(i);
            if (nn < victim_nn) {
                victim_nn = nn;
                victim = i;
            }
        }
        if (victim == SIZE_MAX) return false;

        bool more_diverse = (min_dist > victim_nn + cfg_.evict_div_margin);
        bool better_cost  = (ec < cost_(entries_[victim]) - cfg_.cost_eps);
        bool decent_div   = (min_dist > cfg_.min_diversity);

        if (more_diverse || (better_cost && decent_div)) {
            replace_at(victim, std::move(entry), new_dists);
            return true;
        }
        return false;
    }

    // ====================================================================
    // Selection for mutation — softmax on cost rank
    //
    // P(select i) ∝ exp(-temp * rank_i)
    // Lower cost = lower rank = higher probability.
    // temp=0 → uniform, temp→∞ → always best.
    // ====================================================================

    size_t select_for_mutation(std::mt19937& rng) const {
        size_t n = entries_.size();
        if (n <= 1) return 0;

        // Build cost-rank (0 = best cost)
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return cost_(entries_[a]) < cost_(entries_[b]);
        });

        std::vector<double> weights(n);
        for (size_t rank = 0; rank < n; rank++)
            weights[indices[rank]] = std::exp(-cfg_.selection_temp * (double)rank);

        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        return dist(rng);
    }

    // ====================================================================
    // Selection for crossover — negative assortative mating
    //
    // Parent 1: tournament on cost (pick k random, take best cost)
    // Parent 2: tournament on distance from p1 (pick k random, take
    //           most distant from p1)
    //
    // This maximizes structural difference between parents.
    // ====================================================================

    std::pair<size_t, size_t> select_for_crossover(std::mt19937& rng) const {
        size_t n = entries_.size();
        if (n < 2) return {0, 0};

        // Parent 1: tournament on cost
        size_t p1 = tournament_cost(rng);

        // Parent 2: tournament on distance from p1
        size_t p2 = tournament_diversity(rng, p1);

        return {p1, p2};
    }

    // ====================================================================
    // Accessors
    // ====================================================================

    const Entry& operator[](size_t i) const { return entries_[i]; }
    Entry& operator[](size_t i) { return entries_[i]; }
    size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }

    size_t find_best_cost_idx() const {
        size_t best = 0;
        for (size_t i = 1; i < entries_.size(); i++)
            if (cost_(entries_[i]) < cost_(entries_[best])) best = i;
        return best;
    }

    const Entry& best() const { return entries_[find_best_cost_idx()]; }
    double best_cost() const {
        if (entries_.empty()) return 1e18;
        return cost_(entries_[find_best_cost_idx()]);
    }

    // Cached distance between entries i and j
    double cached_dist(size_t i, size_t j) const {
        return dist_cache_[i * cfg_.hard_cap + j];
    }

    // Nearest-neighbor distance for entry i
    double nn_dist(size_t i) const {
        double mn = 2.0;
        for (size_t j = 0; j < entries_.size(); j++) {
            if (i == j) continue;
            double d = dist_cache_[i * cfg_.hard_cap + j];
            if (d < mn) mn = d;
        }
        return mn;
    }

    // Sort pool by cost (for display / logging)
    void sort_by_cost() {
        // Rebuild everything after sort since indices change
        std::vector<size_t> order(entries_.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return cost_(entries_[a]) < cost_(entries_[b]);
        });

        std::vector<Entry> sorted;
        sorted.reserve(entries_.size());
        for (auto i : order) sorted.push_back(std::move(entries_[i]));
        entries_ = std::move(sorted);
        rebuild_dist_cache();
    }

    // Iterate
    auto begin() const { return entries_.begin(); }
    auto end() const { return entries_.end(); }
    auto begin() { return entries_.begin(); }
    auto end() { return entries_.end(); }

    std::shared_mutex& mutex() { return mutex_; }

    const PoolConfig& config() const { return cfg_; }

private:
    PoolConfig cfg_;
    DistFn dist_;
    CostFn cost_;
    std::vector<Entry> entries_;
    mutable std::shared_mutex mutex_;

    // Flat distance cache: dist_cache_[i * hard_cap + j]
    // Only entries [0..size) × [0..size) are valid.
    std::vector<double> dist_cache_;

    // Append new entry (pool not yet full)
    void append(Entry entry, const std::vector<double>& dists_to_existing) {
        size_t idx = entries_.size();
        entries_.push_back(std::move(entry));

        // Fill row and column for new entry
        for (size_t i = 0; i < idx; i++) {
            dist_cache_[idx * cfg_.hard_cap + i] = dists_to_existing[i];
            dist_cache_[i * cfg_.hard_cap + idx] = dists_to_existing[i];
        }
        dist_cache_[idx * cfg_.hard_cap + idx] = 0.0;
    }

    // Replace entry at position `idx`
    void replace_at(size_t idx, Entry entry,
                    const std::vector<double>& dists_to_all) {
        entries_[idx] = std::move(entry);

        // dists_to_all[i] = distance(new_entry, entries_[i]) for i != idx.
        // These were computed before replacement, but entries_[i] (i != idx)
        // haven't changed, so the distances are still valid.
        for (size_t i = 0; i < entries_.size(); i++) {
            double d = (i == idx) ? 0.0 : dists_to_all[i];
            dist_cache_[idx * cfg_.hard_cap + i] = d;
            dist_cache_[i * cfg_.hard_cap + idx] = d;
        }
    }

    // Full rebuild (after sort or bulk modification)
    void rebuild_dist_cache() {
        size_t n = entries_.size();
        for (size_t i = 0; i < n; i++) {
            dist_cache_[i * cfg_.hard_cap + i] = 0.0;
            for (size_t j = i + 1; j < n; j++) {
                double d = dist_(entries_[i], entries_[j]);
                dist_cache_[i * cfg_.hard_cap + j] = d;
                dist_cache_[j * cfg_.hard_cap + i] = d;
            }
        }
    }

    // Tournament selection on cost (pick k random, return best-cost)
    size_t tournament_cost(std::mt19937& rng) const {
        size_t n = entries_.size();
        size_t k = std::min(cfg_.tournament_k, n);
        size_t best = rng() % n;
        for (size_t i = 1; i < k; i++) {
            size_t cand = rng() % n;
            if (cost_(entries_[cand]) < cost_(entries_[best]))
                best = cand;
        }
        return best;
    }

    // Tournament selection on distance from `ref` (pick k random,
    // return most distant from ref)
    size_t tournament_diversity(std::mt19937& rng, size_t ref) const {
        size_t n = entries_.size();
        size_t k = std::min(cfg_.tournament_k, n);

        // First candidate must differ from ref
        size_t best = rng() % n;
        if (best == ref) best = (best + 1) % n;

        for (size_t i = 1; i < k; i++) {
            size_t cand = rng() % n;
            if (cand == ref) continue;
            if (cached_dist(cand, ref) > cached_dist(best, ref))
                best = cand;
        }
        return best;
    }
};