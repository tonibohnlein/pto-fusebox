#pragma once

#include "core/subgraph.h"
#include "core/flat_set.h"
#include <vector>
#include <atomic>
#include <memory>
#include <cstring>

// ============================================================================
// Unified cost cache for partition search AND solution search.
//
// Two tiers sharing one object, kept alive across all three solver phases:
//
//   BASE MAP: Lock-free open-addressed hash table.
//     hash(op_set) → CostResult.  Populated during Phase 1 (partition search)
//     via evaluate().  Reads are fully lock-free (single atomic load).
//     Writes use CAS on per-slot atomic hash.
//     ~30-70K entries across all phases.
//
//   RETENTION MAP: hash(op_set, entering, retain) → CostResult.
//     Populated during Phase 2/3 (coupling search) via evaluate_with_context().
//     Also fully lock-free — same design as the base map.
//     When entering={} and retain={}, falls back to the base map instead.
//
// Keys are 64-bit hashes only — no stored key vectors.  With a strong hash
// and <100K entries in a 1M-slot table, collision probability is negligible
// (~1e-9 per lookup).  This eliminates all vector copies and comparisons
// from the hot path.
// ============================================================================

// ============================================================================
// Strong 64-bit hash for sorted op-index sequences.
// Uses wyhash-style multiply-xor mixing for excellent avalanche properties.
// ============================================================================

inline uint64_t wymix(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b;
    return (uint64_t)(r >> 64) ^ (uint64_t)r;
}

// Hash a contiguous array of size_t values.
inline uint64_t hash_ops(const size_t* data, size_t n) {
    // Seed chosen to avoid zero-hash for empty sets.
    uint64_t h = 0x2d358dccaa6c78a5ULL ^ (uint64_t)n;
    for (size_t i = 0; i < n; i++)
        h = wymix(h ^ data[i], 0x9e3779b97f4a7c15ULL);
    // Final mix
    h ^= h >> 32;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 31;
    // Reserve 0 as "empty slot" sentinel
    return h | 1;
}

// Hash a FlatSet<size_t> in place (no copy).
inline uint64_t hash_flatset(const FlatSet<size_t>& s) {
    return hash_ops(s.data(), s.size());
}

// Combine three hashes for the retention key: (ops, entering, retain).
// Uses independent seeds so hash(ops,{},{}) != hash(ops).
inline uint64_t hash_retention(const FlatSet<size_t>& ops,
                                const FlatSet<size_t>& entering,
                                const FlatSet<size_t>& retain) {
    uint64_t h1 = hash_ops(ops.data(), ops.size());
    uint64_t h2 = 0x6c62272e07bb0142ULL;
    for (size_t i = 0; i < entering.size(); i++)
        h2 = wymix(h2 ^ entering.data()[i], 0x517cc1b727220a95ULL);
    uint64_t h3 = 0x6295c58d62b82175ULL;
    for (size_t i = 0; i < retain.size(); i++)
        h3 = wymix(h3 ^ retain.data()[i], 0x3243f6a8885a308dULL);
    uint64_t h = wymix(h1, h2 ^ h3);
    h ^= h >> 31;
    return h | 1;
}

// Hash from a vector<size_t> (e.g. Subgraph::ops()).
inline uint64_t hash_vec(const std::vector<size_t>& v) {
    return hash_ops(v.data(), v.size());
}

// ============================================================================
// Lock-free open-addressed hash table with hash-only keys.
//
// Each slot is: atomic<uint64_t> hash + Value.
// hash == 0 means empty.  hash != 0 means occupied (hash is the key).
//
// Read path:  compute slot index → load hash(acquire) → if match, return value.
// Write path: CAS(0, hash) → write value → done.
//
// No stored keys, no vector comparisons.  ~48 bytes per slot (vs ~100+ before).
// Collision probability with 64-bit hash: ~n²/2^64 where n = entries.
// With n = 100K entries: ~5e-10 — negligible.
//
// No size counter — avoiding shared atomic contention. Size is approximated
// from a per-thread counter that's collected lazily.
// ============================================================================

template<typename Value>
class HashOnlyMap {
    struct Slot {
        std::atomic<uint64_t> hash{0};  // 0 = empty
        Value value{};
    };

    std::unique_ptr<Slot[]> slots_;
    size_t capacity_;
    size_t mask_;
    std::atomic<size_t> size_{0};

public:
    explicit HashOnlyMap(size_t min_capacity = 131072) {
        capacity_ = 1;
        while (capacity_ < min_capacity) capacity_ <<= 1;
        mask_ = capacity_ - 1;
        slots_ = std::make_unique<Slot[]>(capacity_);
    }

    // Lock-free lookup.  Returns pointer to value if hash matches, nullptr otherwise.
    const Value* find(uint64_t h) const {
        size_t idx = (size_t)h & mask_;
        for (size_t i = 0; i < 32; i++) {  // bounded probe length
            uint64_t sh = slots_[idx].hash.load(std::memory_order_acquire);
            if (sh == 0) return nullptr;     // empty → end of chain
            if (sh == h) return &slots_[idx].value;
            idx = (idx + 1) & mask_;
        }
        return nullptr;  // probe limit reached
    }

    // Insert hash→value.  Returns true if inserted.
    bool insert(uint64_t h, const Value& value) {
        size_t idx = (size_t)h & mask_;
        for (size_t i = 0; i < 32; i++) {
            auto& slot = slots_[idx];
            uint64_t expected = 0;
            if (slot.hash.compare_exchange_strong(expected, h,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                slot.value = value;
                size_.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            if (expected == h) return false;  // duplicate
            idx = (idx + 1) & mask_;
        }
        return false;  // probe limit
    }

    size_t size() const { return size_.load(std::memory_order_relaxed); }

    void clear() {
        for (size_t i = 0; i < capacity_; i++) {
            slots_[i].hash.store(0, std::memory_order_relaxed);
            slots_[i].value = Value{};
        }
        size_.store(0, std::memory_order_relaxed);
    }
};

// ============================================================================
// Per-thread cache statistics — avoids cross-thread atomic contention on
// the hot path. Each thread accumulates locally; sums are computed on demand.
// ============================================================================
struct alignas(64) CacheStats {
    uint64_t base_hits = 0;
    uint64_t base_misses = 0;
    uint64_t ret_hits = 0;
    uint64_t ret_misses = 0;
    uint64_t base_overcapacity = 0;
    uint64_t _pad[3];  // pad to 64 bytes (one cache line) to avoid false sharing
};

inline CacheStats& thread_cache_stats() {
    thread_local CacheStats stats;
    return stats;
}

// ============================================================================
// CostCache
// ============================================================================

class CostCache {
public:
    explicit CostCache(size_t max_entries = 0)
        : max_entries_(max_entries)
        , base_map_(std::max<size_t>(1048576, max_entries > 0 ? max_entries * 2 : 1048576))
        , ret_map_(1048576)
    {
        auto& s = thread_cache_stats();
        s.base_hits = s.base_misses = s.ret_hits = s.ret_misses = s.base_overcapacity = 0;
    }

    // ====================================================================
    // Tier 1: Base map — hash(op_set) → CostResult.
    // ====================================================================

    double evaluate(const FlatSet<size_t>& ops, const Problem& prob, const DAG& dag) {
        uint64_t h = hash_flatset(ops);

        auto* hit = base_map_.find(h);
        if (hit) {
            thread_cache_stats().base_hits++;
            return hit->feasible ? hit->latency : 1e18;
        }

        thread_cache_stats().base_misses++;
        CostResult cr;
        auto sg = Subgraph::create(prob, dag,
            std::vector<size_t>(ops.begin(), ops.end()));
        if (sg) cr = sg->best_cost();

        if (max_entries_ > 0 && base_map_.size() >= max_entries_)
            thread_cache_stats().base_overcapacity++;
        else
            base_map_.insert(h, cr);

        return cr.feasible ? cr.latency : 1e18;
    }

    // ====================================================================
    // Tier 2: Retention map — hash(ops, entering, retain) → CostResult.
    // ====================================================================

    CostResult evaluate_with_context(const FlatSet<size_t>& ops,
                                      const FlatSet<size_t>& entering,
                                      const FlatSet<size_t>& retain,
                                      const Problem& prob,
                                      const DAG& dag) {
        if (entering.empty() && retain.empty()) {
            uint64_t h = hash_flatset(ops);
            auto* hit = base_map_.find(h);
            if (hit) {
                thread_cache_stats().base_hits++;
                return *hit;
            }
            thread_cache_stats().base_misses++;
            auto sg_opt = Subgraph::create(prob, dag,
                std::vector<size_t>(ops.begin(), ops.end()));
            CostResult cr;
            if (sg_opt) cr = sg_opt->best_cost({}, {});
            if (max_entries_ == 0 || base_map_.size() < max_entries_)
                base_map_.insert(h, cr);
            return cr;
        }

        uint64_t h = hash_retention(ops, entering, retain);

        auto* hit = ret_map_.find(h);
        if (hit) {
            thread_cache_stats().ret_hits++;
            return *hit;
        }

        thread_cache_stats().ret_misses++;
        auto sg_opt = Subgraph::create(prob, dag,
            std::vector<size_t>(ops.begin(), ops.end()));
        CostResult cr;
        if (sg_opt) cr = sg_opt->best_cost(entering, retain);

        ret_map_.insert(h, cr);
        return cr;
    }

    CostResult evaluate_with_context(const Subgraph& sg,
                                      const FlatSet<size_t>& entering,
                                      const FlatSet<size_t>& retain) {
        if (entering.empty() && retain.empty()) {
            uint64_t h = hash_vec(sg.ops());
            auto* hit = base_map_.find(h);
            if (hit) {
                thread_cache_stats().base_hits++;
                return *hit;
            }
            thread_cache_stats().base_misses++;
            auto cr = sg.best_cost({}, {});
            base_map_.insert(h, cr);
            return cr;
        }

        FlatSet<size_t> ops_set(sg.ops().begin(), sg.ops().end());
        uint64_t h = hash_retention(ops_set, entering, retain);

        auto* hit = ret_map_.find(h);
        if (hit) {
            thread_cache_stats().ret_hits++;
            return *hit;
        }

        thread_cache_stats().ret_misses++;
        auto cr = sg.best_cost(entering, retain);
        ret_map_.insert(h, cr);
        return cr;
    }

    // ====================================================================
    // Diagnostics & management
    // ====================================================================

    void clear() {
        base_map_.clear();
        ret_map_.clear();
        auto& s = thread_cache_stats();
        s.base_hits = s.base_misses = s.ret_hits = s.ret_misses = s.base_overcapacity = 0;
    }

    // These return per-thread counts only (not cross-thread sum).
    // Used for end-of-run diagnostic logging from the main thread.
    size_t base_hits()    const { return thread_cache_stats().base_hits; }
    size_t base_misses()  const { return thread_cache_stats().base_misses; }
    size_t ret_hits()     const { return thread_cache_stats().ret_hits; }
    size_t ret_misses()   const { return thread_cache_stats().ret_misses; }

    size_t hits()         const { return base_hits(); }
    size_t misses()       const { return base_misses(); }
    size_t overcapacity() const { return thread_cache_stats().base_overcapacity; }
    size_t size()         const { return base_map_.size(); }
    size_t ret_size()     const { return ret_map_.size(); }
    size_t max_entries()  const { return max_entries_; }

private:
    const size_t max_entries_;
    HashOnlyMap<CostResult> base_map_;

    HashOnlyMap<CostResult> ret_map_;
};
