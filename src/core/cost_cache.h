#pragma once

#include "core/subgraph.h"
#include <vector>
#include <unordered_map>
#include <set>
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
    // Evaluate the cost of a set of ops. Returns cached result if available.
    double evaluate(const std::set<size_t>& ops, const Problem& prob, const DAG& dag) {
        std::vector<size_t> key(ops.begin(), ops.end());
        
        // Check cache (read lock)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = map_.find(key);
            if (it != map_.end()) {
                hits_.fetch_add(1, std::memory_order_relaxed);
                return it->second;
            }
        }
        
        // Cache miss: compute
        misses_.fetch_add(1, std::memory_order_relaxed);
        double cost = 1e18;
        auto sg = Subgraph::create(prob, dag, key);
        if (sg) {
            auto c = sg->best_cost();
            if (c.feasible) cost = c.latency;
        }
        
        // Store (write lock)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            map_[key] = cost;
        }
        
        return cost;
    }
    
    size_t hits()  const { return hits_.load(std::memory_order_relaxed); }
    size_t misses() const { return misses_.load(std::memory_order_relaxed); }
    size_t size()  const { std::lock_guard<std::mutex> lock(mutex_); return map_.size(); }
    
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::vector<size_t>, double, VectorHash> map_;
    std::atomic<size_t> hits_{0}, misses_{0};
};