#pragma once

#include "core/subgraph.h"
#include <vector>
#include <set>
#include <atomic>
#include <memory>

// ============================================================================
// Unified cost cache for partition search AND solution search.
//
// Two tiers sharing one object, kept alive across all three solver phases:
//
//   BASE MAP: Lock-free open-addressed hash table.
//     (op_set) → CostResult.  Populated during Phase 1 (partition search)
//     via evaluate().  Reads are fully lock-free (atomic state load +
//     key compare).  Writes use CAS on per-slot atomic state.
//     ~30-70K entries across all phases.
//
//   RETENTION MAP: (op_set | SIZE_MAX | entering | SIZE_MAX | retain) → CostResult
//     Populated during Phase 2/3 (coupling search) via evaluate_with_context().
//     Also fully lock-free — same LockFreeMap design as the base map.
//     When entering={} and retain={}, falls back to the base map instead.
// ============================================================================

struct VectorHash {
    size_t operator()(const std::vector<size_t>& v) const {
        size_t h = v.size();
        for (size_t op : v)
            h ^= std::hash<size_t>()(op) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ============================================================================
// Lock-free open-addressed hash table with linear probing.
//
// Slot states: EMPTY → WRITING → READY  (one-way transitions, never recycle)
//
// Read path:  hash → probe → load state(acquire) → if READY, compare key
//             No locks, no atomics beyond the state load.
//
// Write path: hash → probe → CAS(EMPTY, WRITING) → write key+value
//             → store(READY, release).  Single-writer-per-slot via CAS.
//
// Duplicates from races are harmless (cache semantics — any hit is valid).
// Table never rehashes; pre-allocate sufficient capacity.
// ============================================================================

template<typename Value>
class LockFreeMap {
    enum State : uint8_t { EMPTY = 0, WRITING = 1, READY = 2 };

    struct Slot {
        std::atomic<uint8_t> state{EMPTY};
        size_t hash = 0;
        std::vector<size_t> key;
        Value value{};
    };

    std::unique_ptr<Slot[]> slots_;
    size_t capacity_;
    size_t mask_;
    std::atomic<size_t> size_{0};

public:
    explicit LockFreeMap(size_t min_capacity = 131072) {
        // Round up to power of 2
        capacity_ = 1;
        while (capacity_ < min_capacity) capacity_ <<= 1;
        mask_ = capacity_ - 1;
        slots_ = std::make_unique<Slot[]>(capacity_);
    }

    // Lock-free lookup. Returns pointer to value if found, nullptr otherwise.
    // Safe to call concurrently from any number of threads.
    const Value* find(const std::vector<size_t>& key, size_t h) const {
        size_t idx = h & mask_;
        for (size_t i = 0; i < capacity_; i++) {
            const auto& slot = slots_[idx];
            auto s = slot.state.load(std::memory_order_acquire);
            if (s == EMPTY) return nullptr;  // end of probe chain
            if (s == READY && slot.hash == h && slot.key == key)
                return &slot.value;
            idx = (idx + 1) & mask_;
        }
        return nullptr;
    }

    // Insert key→value. Returns true if inserted, false if duplicate or full.
    // Safe to call concurrently; at most one thread wins each slot via CAS.
    bool insert(const std::vector<size_t>& key, size_t h, const Value& value) {
        size_t idx = h & mask_;
        for (size_t i = 0; i < capacity_; i++) {
            auto& slot = slots_[idx];
            uint8_t expected = EMPTY;
            if (slot.state.compare_exchange_strong(expected, WRITING,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                // Won this slot — write key+value, then publish
                slot.hash = h;
                slot.key = key;
                slot.value = value;
                slot.state.store(READY, std::memory_order_release);
                size_.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            if (expected == WRITING) {
                // Another thread is filling this slot — skip (may create duplicate, harmless)
                idx = (idx + 1) & mask_;
                continue;
            }
            // expected == READY: check for duplicate key
            if (slot.hash == h && slot.key == key)
                return false;  // already cached
            idx = (idx + 1) & mask_;
        }
        return false;  // table full
    }

    size_t size() const { return size_.load(std::memory_order_relaxed); }

    // Reset all slots. Only call when no other thread is accessing the table.
    void clear() {
        for (size_t i = 0; i < capacity_; i++) {
            slots_[i].state.store(EMPTY, std::memory_order_relaxed);
            slots_[i].hash = 0;
            slots_[i].key = std::vector<size_t>();  // free heap capacity
            slots_[i].value = Value{};
        }
        size_.store(0, std::memory_order_relaxed);
    }
};

// ============================================================================
// CostCache
// ============================================================================

class CostCache {
public:
    explicit CostCache(size_t max_entries = 0)
        : max_entries_(max_entries)
        // Base map: 2x headroom, minimum 1M slots.  Load factor < 0.5.
        , base_map_(std::max<size_t>(1048576, max_entries > 0 ? max_entries * 2 : 1048576))
        // Retention map: fixed 1M slots → up to ~500K entries at 0.5 load factor.
        , ret_map_(1048576)
    {}

    // ====================================================================
    // Tier 1: Base map — lock-free (op_set) → CostResult.
    // ====================================================================

    double evaluate(const std::set<size_t>& ops, const Problem& prob, const DAG& dag) {
        thread_local std::vector<size_t> key;
        key.assign(ops.begin(), ops.end());
        size_t h = VectorHash{}(key);

        // Lock-free read
        auto* hit = base_map_.find(key, h);
        if (hit) {
            base_hits_.fetch_add(1, std::memory_order_relaxed);
            return hit->feasible ? hit->latency : 1e18;
        }

        // Miss — compute
        base_misses_.fetch_add(1, std::memory_order_relaxed);
        CostResult cr;
        auto sg = Subgraph::create(prob, dag,
            std::vector<size_t>(ops.begin(), ops.end()));
        if (sg) cr = sg->best_cost();

        // Insert (may fail if full or duplicate from race — both harmless)
        if (max_entries_ > 0 && base_map_.size() >= max_entries_)
            base_overcapacity_.fetch_add(1, std::memory_order_relaxed);
        else
            base_map_.insert(key, h, cr);

        return cr.feasible ? cr.latency : 1e18;
    }

    // ====================================================================
    // Tier 2: Retention map — lock-free (op_set|entering|retain) → CostResult.
    //
    // When entering={} and retain={}, falls back to the lock-free base map.
    // ====================================================================

    // Overload: look up or compute (ops, entering, retain) without requiring a
    // pre-built Subgraph.  On cache hit, returns immediately (no Subgraph
    // created).  On miss, builds a Subgraph from ops and evaluates.
    CostResult evaluate_with_context(const std::set<size_t>& ops,
                                      const std::set<size_t>& entering,
                                      const std::set<size_t>& retain,
                                      const Problem& prob,
                                      const DAG& dag) {
        if (entering.empty() && retain.empty()) {
            thread_local std::vector<size_t> okey;
            okey.assign(ops.begin(), ops.end());
            size_t h = VectorHash{}(okey);
            auto* hit = base_map_.find(okey, h);
            if (hit) {
                base_hits_.fetch_add(1, std::memory_order_relaxed);
                return *hit;
            }
            base_misses_.fetch_add(1, std::memory_order_relaxed);
            auto sg_opt = Subgraph::create(prob, dag,
                std::vector<size_t>(ops.begin(), ops.end()));
            CostResult cr;
            if (sg_opt) cr = sg_opt->best_cost({}, {});
            if (max_entries_ == 0 || base_map_.size() < max_entries_)
                base_map_.insert(okey, h, cr);
            return cr;
        }

        thread_local std::vector<size_t> oext_key;
        oext_key.clear();
        oext_key.insert(oext_key.end(), ops.begin(), ops.end());
        oext_key.push_back(SIZE_MAX);
        oext_key.insert(oext_key.end(), entering.begin(), entering.end());
        oext_key.push_back(SIZE_MAX);
        oext_key.insert(oext_key.end(), retain.begin(), retain.end());
        size_t h = VectorHash{}(oext_key);

        // Lock-free read
        auto* hit = ret_map_.find(oext_key, h);
        if (hit) {
            ret_hits_.fetch_add(1, std::memory_order_relaxed);
            return *hit;
        }

        // Miss — compute
        ret_misses_.fetch_add(1, std::memory_order_relaxed);
        auto sg_opt = Subgraph::create(prob, dag,
            std::vector<size_t>(ops.begin(), ops.end()));
        CostResult cr;
        if (sg_opt) cr = sg_opt->best_cost(entering, retain);

        if (ret_map_.size() < 750000)  // stay under ~75% load factor
            ret_map_.insert(oext_key, h, cr);
        return cr;
    }

    CostResult evaluate_with_context(const Subgraph& sg,
                                      const std::set<size_t>& entering,
                                      const std::set<size_t>& retain) {
        // No retention context → lock-free base map
        if (entering.empty() && retain.empty()) {
            thread_local std::vector<size_t> base_key;
            base_key.assign(sg.ops().begin(), sg.ops().end());
            size_t h = VectorHash{}(base_key);

            auto* hit = base_map_.find(base_key, h);
            if (hit) {
                base_hits_.fetch_add(1, std::memory_order_relaxed);
                return *hit;
            }

            base_misses_.fetch_add(1, std::memory_order_relaxed);
            auto cr = sg.best_cost({}, {});
            base_map_.insert(base_key, h, cr);
            return cr;
        }

        // Retention context → lock-free retention map
        thread_local std::vector<size_t> ext_key;
        ext_key.clear();
        ext_key.insert(ext_key.end(), sg.ops().begin(), sg.ops().end());
        ext_key.push_back(SIZE_MAX);
        ext_key.insert(ext_key.end(), entering.begin(), entering.end());
        ext_key.push_back(SIZE_MAX);
        ext_key.insert(ext_key.end(), retain.begin(), retain.end());
        size_t h = VectorHash{}(ext_key);

        auto* hit = ret_map_.find(ext_key, h);
        if (hit) {
            ret_hits_.fetch_add(1, std::memory_order_relaxed);
            return *hit;
        }

        ret_misses_.fetch_add(1, std::memory_order_relaxed);
        auto cr = sg.best_cost(entering, retain);
        if (ret_map_.size() < 750000)  // stay under ~75% load factor
            ret_map_.insert(ext_key, h, cr);
        return cr;
    }

    // ====================================================================
    // Diagnostics & management
    // ====================================================================

    // Only call when no threads are using the cache.
    void clear() {
        base_map_.clear();
        ret_map_.clear();
        base_hits_.store(0, std::memory_order_relaxed);
        base_misses_.store(0, std::memory_order_relaxed);
        base_overcapacity_.store(0, std::memory_order_relaxed);
        ret_hits_.store(0, std::memory_order_relaxed);
        ret_misses_.store(0, std::memory_order_relaxed);
    }

    size_t base_hits()    const { return base_hits_.load(std::memory_order_relaxed); }
    size_t base_misses()  const { return base_misses_.load(std::memory_order_relaxed); }
    size_t ret_hits()     const { return ret_hits_.load(std::memory_order_relaxed); }
    size_t ret_misses()   const { return ret_misses_.load(std::memory_order_relaxed); }

    size_t hits()         const { return base_hits(); }
    size_t misses()       const { return base_misses(); }
    size_t overcapacity() const { return base_overcapacity_.load(std::memory_order_relaxed); }
    size_t size()         const { return base_map_.size(); }
    size_t ret_size()     const { return ret_map_.size(); }
    size_t max_entries()  const { return max_entries_; }

    // freeze/unfreeze are no longer needed (both maps are always lock-free)
    void freeze_base() {}
    void unfreeze_base() {}

private:
    const size_t max_entries_;
    LockFreeMap<CostResult> base_map_;
    std::atomic<size_t> base_hits_{0}, base_misses_{0}, base_overcapacity_{0};

    // Retention map: lock-free (op_set|entering|retain) → CostResult  [Phase 2/3]
    LockFreeMap<CostResult> ret_map_;
    std::atomic<size_t> ret_hits_{0}, ret_misses_{0};
};
