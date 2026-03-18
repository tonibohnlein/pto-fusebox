#pragma once
// pairing_heap.h — O(log n) amortized decrease-key / increase-key heap
// Templated on MoveType which must have a `double saving` field.

#include <vector>
#include <optional>
#include <cstddef>

template<typename MoveType>
class PairingHeap {
public:
    explicit PairingHeap(size_t capacity)
        : nodes_(capacity), in_heap_(capacity, false), root_(-1) {
        merge_buf_.reserve(capacity);
    }

    void push_or_update(size_t key, const MoveType& move) {
        if (!in_heap_[key]) {
            nodes_[key].move = move;
            nodes_[key].child = -1;
            nodes_[key].next_sibling = -1;
            nodes_[key].prev = -1;
            in_heap_[key] = true;
            root_ = (root_ == -1) ? (int)key : link(root_, (int)key);
        } else {
            double old = nodes_[key].move.saving;
            nodes_[key].move = move;

            if ((int)key == root_) {
                if (move.saving < old) {
                    int c = nodes_[root_].child;
                    nodes_[root_].child = -1;
                    if (c != -1)
                        root_ = link(root_, merge_peers(c));
                }
            } else {
                cut((int)key);
                if (move.saving < old) {
                    int c = nodes_[key].child;
                    nodes_[key].child = -1;
                    if (c != -1)
                        root_ = link(root_, merge_peers(c));
                }
                root_ = link(root_, (int)key);
            }
        }
    }

    void remove(size_t key) {
        if (!in_heap_[key]) return;
        in_heap_[key] = false;
        if ((int)key == root_) {
            int c = nodes_[key].child;
            root_ = (c == -1) ? -1 : merge_peers(c);
        } else {
            cut((int)key);
            int c = nodes_[key].child;
            nodes_[key].child = -1;
            if (c != -1)
                root_ = link(root_, merge_peers(c));
        }
    }

    std::optional<MoveType> pop_best() {
        if (root_ == -1) return std::nullopt;
        int best = root_;
        MoveType m = nodes_[best].move;
        in_heap_[best] = false;
        int c = nodes_[best].child;
        root_ = (c == -1) ? -1 : merge_peers(c);
        return m;
    }

    bool empty() const { return root_ == -1; }
    bool contains(size_t key) const { return key < in_heap_.size() && in_heap_[key]; }

    // Peek at the best element without removing it.
    // Returns nullopt if heap is empty.
    std::optional<std::pair<size_t, const MoveType&>> peek_best() const {
        if (root_ == -1) return std::nullopt;
        return std::pair<size_t, const MoveType&>{(size_t)root_, nodes_[root_].move};
    }

    void clear() {
        if (root_ == -1) return;
        std::fill(in_heap_.begin(), in_heap_.end(), false);
        root_ = -1;
    }

    const MoveType& peek(size_t key) const { return nodes_[key].move; }

private:
    struct Node {
        MoveType move;
        int child = -1;
        int next_sibling = -1;
        int prev = -1;
    };

    std::vector<Node> nodes_;
    std::vector<bool> in_heap_;
    int root_;
    std::vector<int> merge_buf_;

    int link(int a, int b) {
        if (a == -1) return b;
        if (b == -1) return a;
        if (nodes_[b].move.saving > nodes_[a].move.saving)
            std::swap(a, b);
        nodes_[b].prev = a;
        nodes_[b].next_sibling = nodes_[a].child;
        if (nodes_[a].child != -1)
            nodes_[nodes_[a].child].prev = b;
        nodes_[a].child = b;
        return a;
    }

    void cut(int a) {
        int p = nodes_[a].prev;
        if (p != -1) {
            if (nodes_[p].child == a)
                nodes_[p].child = nodes_[a].next_sibling;
            else
                nodes_[p].next_sibling = nodes_[a].next_sibling;
        }
        if (nodes_[a].next_sibling != -1)
            nodes_[nodes_[a].next_sibling].prev = p;
        nodes_[a].prev = -1;
        nodes_[a].next_sibling = -1;
    }

    int merge_peers(int first) {
        if (first == -1) return -1;
        merge_buf_.clear();
        int cur = first;
        while (cur != -1) {
            int next = nodes_[cur].next_sibling;
            nodes_[cur].prev = -1;
            nodes_[cur].next_sibling = -1;
            merge_buf_.push_back(cur);
            cur = next;
        }
        if (merge_buf_.size() == 1) return merge_buf_[0];
        for (size_t i = 0; i + 1 < merge_buf_.size(); i += 2)
            merge_buf_[i] = link(merge_buf_[i], merge_buf_[i + 1]);
        int last = (merge_buf_.size() % 2 == 1)
                       ? (int)merge_buf_.size() - 1
                       : (int)merge_buf_.size() - 2;
        int result = merge_buf_[last];
        for (int i = last - 2; i >= 0; i -= 2)
            result = link(result, merge_buf_[i]);
        return result;
    }
};