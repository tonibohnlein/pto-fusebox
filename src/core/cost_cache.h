#pragma once

#include "core/subgraph.h"
#include <vector>
#include <unordered_map>
#include <set>
#include <shared_mutex>
#include <mutex>
#include <atomic>

// ============================================================================
// Thread-safe memoization cache for eval_set.
//
// The key insight: best_cost() for a given set of ops is deterministic
// (depends only on the ops and the problem). During local search, the same
// op-sets are evaluated millions of times across greedy, tabu, and FM passes.
// Caching avoids redundant Subgraph::create + tiling enumeration.
//
// NOTE: only the scalar cost is stored here — not the Subgraph or TileConfig.
// finalize() rebuilds those on demand for the small partition pool (O(pool *
// groups) calls, all cache misses converted to Subgraph::create directly).
// Storing Subgraph in this cache would copy a heavy object on every one of
// the millions of cache hits during Phase 1, destroying search performance.
//
// --- Key representation ---
// The cache key is std::vector<size_t> built from std::set<size_t>, which
// always iterates in sorted order. The key is therefore always sorted.
// This is a required invariant: the same op-set must always produce the same
// key regardless of how it was assembled by the caller. Callers MUST pass
// ops as std::set<size_t> to guarantee this.
//
// --- Thread safety ---
// Reads (hits) use a shared lock so multiple threads can read concurrently.
// Writes (misses) use an exclusive lock.
//
// TOCTOU note: two threads can both miss the read check, both compute, and
// one silently overwrites the other. This is safe because best_cost() is
// deterministic — both produce the same value. The only side effect is that
// misses_ may be incremented more than once for a single logical miss under
// high concurrency, so misses() is an upper bound, not an exact count.
// ============================================================================

struct VectorHash {
    size_t operator()(const std::vector<size_t>& v) const {
        size_t h = v.size();
        for (size_t op : v)
            h ^= std::hash<size_t>()(op) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

class CostCache {
public:
    // max_entries: maximum number of entries to store.
    // 0 = unlimited (default — suitable when problem size is known to be small).
    // When the cap is reached, new op-sets are still evaluated and returned but
    // not inserted into the map. Use overcapacity() to detect this.
    //
    // Sizing guidance:
    //   Each entry ≈ avg_group_size*8 + 8 + 64 bytes (key + value + node overhead).
    //   For avg_group_size=4: ~100 bytes/entry.
    //   250K entries ≈ 25 MB  (fine for competition benchmarks, N ≤ ~200 ops).
    //   1M  entries ≈ 100 MB  (safe upper bound for N ≤ ~500 ops).
    explicit CostCache(size_t max_entries = 0) : max_entries_(max_entries) {}

    // Evaluate the cost of a set of ops. Returns cached result if available.
    // ops must be a std::set<size_t> to guarantee sorted key ordering.
    double evaluate(const std::set<size_t>& ops, const Problem& prob, const DAG& dag) {
        std::vector<size_t> key(ops.begin(), ops.end());

        // Fast path: shared (read) lock — multiple threads can hit concurrently.
        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            auto it = map_.find(key);
            if (it != map_.end()) {
                hits_.fetch_add(1, std::memory_order_relaxed);
                return it->second;
            }
        }

        // Cache miss: compute outside the lock.
        // Two threads may both reach here for the same key — see TOCTOU note above.
        misses_.fetch_add(1, std::memory_order_relaxed);
        double cost = 1e18;
        auto sg = Subgraph::create(prob, dag, key);
        if (sg) {
            auto c = sg->best_cost();
            if (c.feasible) cost = c.latency;
        }

        // Write: exclusive lock.
        // Skip insertion if the cap is set and already reached (stop-on-full policy).
        // Early entries (small singleton/pair groups, most frequently re-evaluated)
        // were cached first, so later overflow entries are typically less valuable.
        {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            if (max_entries_ == 0 || map_.size() < max_entries_) {
                // Another thread may have already inserted this key (TOCTOU);
                // harmlessly overwrite with the same deterministic value.
                map_[key] = cost;
            } else {
                overcapacity_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        return cost;
    }

    // Reset all state. Use between independent problem instances.
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        map_.clear();
        hits_.store(0,          std::memory_order_relaxed);
        misses_.store(0,        std::memory_order_relaxed);
        overcapacity_.store(0,  std::memory_order_relaxed);
    }

    // misses() is an upper bound under concurrent access (see TOCTOU note).
    size_t hits()         const { return hits_.load(std::memory_order_relaxed); }
    size_t misses()       const { return misses_.load(std::memory_order_relaxed); }
    // overcapacity(): number of evaluations that were computed but not stored
    // because the cap was reached. Non-zero means the cap should be raised.
    size_t overcapacity() const { return overcapacity_.load(std::memory_order_relaxed); }
    size_t size()         const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return map_.size();
    }
    size_t max_entries()  const { return max_entries_; }

private:
    const size_t max_entries_;
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::vector<size_t>, double, VectorHash> map_;
    std::atomic<size_t> hits_{0}, misses_{0}, overcapacity_{0};
};