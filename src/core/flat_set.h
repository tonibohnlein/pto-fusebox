#pragma once

#include <algorithm>
#include <initializer_list>
#include <vector>

// ============================================================================
// FlatSet<T>: a sorted std::vector<T> with std::set-compatible API.
//
// Drop-in replacement for std::set<T> when the set is small (< ~100 elements).
// Benefits over std::set:
//   - No per-element heap allocation (single contiguous buffer)
//   - Better cache locality for iteration and lookup
//   - Lower memory overhead (no tree node pointers)
//
// Trade-offs:
//   - insert/erase are O(n) due to element shifting (fine for small sets)
//   - Lookup via binary_search is O(log n), same as std::set
// ============================================================================

template <typename T>
class FlatSet {
public:
    using value_type             = T;
    using iterator               = typename std::vector<T>::iterator;
    using const_iterator         = typename std::vector<T>::const_iterator;
    using reverse_iterator       = typename std::vector<T>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;
    using size_type              = std::size_t;

    // --- Constructors ---

    FlatSet() = default;

    FlatSet(std::initializer_list<T> il) : v_(il) {
        std::sort(v_.begin(), v_.end());
        v_.erase(std::unique(v_.begin(), v_.end()), v_.end());
    }

    template <typename It>
    FlatSet(It first, It last) : v_(first, last) {
        std::sort(v_.begin(), v_.end());
        v_.erase(std::unique(v_.begin(), v_.end()), v_.end());
    }

    // Default copy/move are correct (vector handles everything).

    // --- Capacity ---

    bool      empty()    const { return v_.empty(); }
    size_type size()     const { return v_.size(); }
    void      reserve(size_type n) { v_.reserve(n); }
    void      clear()          { v_.clear(); }

    // --- Lookup ---

    size_type count(const T& val) const {
        return std::binary_search(v_.begin(), v_.end(), val) ? 1 : 0;
    }

    const_iterator find(const T& val) const {
        auto it = std::lower_bound(v_.begin(), v_.end(), val);
        return (it != v_.end() && *it == val) ? it : v_.end();
    }

    iterator find(const T& val) {
        auto it = std::lower_bound(v_.begin(), v_.end(), val);
        return (it != v_.end() && *it == val) ? it : v_.end();
    }

    // --- Modifiers ---

    std::pair<iterator, bool> insert(const T& val) {
        auto it = std::lower_bound(v_.begin(), v_.end(), val);
        if (it != v_.end() && *it == val)
            return {it, false};
        return {v_.insert(it, val), true};
    }

    template <typename It>
    void insert(It first, It last) {
        v_.insert(v_.end(), first, last);
        std::sort(v_.begin(), v_.end());
        v_.erase(std::unique(v_.begin(), v_.end()), v_.end());
    }

    size_type erase(const T& val) {
        auto it = std::lower_bound(v_.begin(), v_.end(), val);
        if (it != v_.end() && *it == val) {
            v_.erase(it);
            return 1;
        }
        return 0;
    }

    iterator erase(const_iterator pos) {
        return v_.erase(pos);
    }

    iterator erase(const_iterator first, const_iterator last) {
        return v_.erase(first, last);
    }

    // --- Iterators ---

    iterator       begin()        { return v_.begin(); }
    iterator       end()          { return v_.end(); }
    const_iterator begin()  const { return v_.begin(); }
    const_iterator end()    const { return v_.end(); }
    const_iterator cbegin() const { return v_.cbegin(); }
    const_iterator cend()   const { return v_.cend(); }

    reverse_iterator       rbegin()        { return v_.rbegin(); }
    reverse_iterator       rend()          { return v_.rend(); }
    const_reverse_iterator rbegin()  const { return v_.rbegin(); }
    const_reverse_iterator rend()    const { return v_.rend(); }

    // --- Comparison ---

    bool operator==(const FlatSet& o) const { return v_ == o.v_; }
    bool operator!=(const FlatSet& o) const { return v_ != o.v_; }
    bool operator<(const FlatSet& o)  const { return v_ < o.v_; }

    // --- Direct access ---

    const T*              data()       const { return v_.data(); }
    const std::vector<T>& underlying() const { return v_; }

private:
    std::vector<T> v_;
};
