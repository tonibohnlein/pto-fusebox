// pairing_heap_test.cpp
// Tests for PairingHeap<T> — push, pop, update, remove.

#include "util/pairing_heap.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

static int g_pass = 0, g_fail = 0;

static void CHECK(const char* label, bool cond) {
    if (cond) g_pass++;
    else { g_fail++; std::cout << "  FAIL: " << label << "\n"; }
}

// Minimal move type for testing — just needs a `saving` field.
struct TestMove {
    size_t key;
    double saving;
};

// ============================================================================
// Test 1: basic push and pop ordering
// ============================================================================
void test_basic_push_pop() {
    std::cout << "--- test_basic_push_pop ---\n";
    PairingHeap<TestMove> h(10);

    h.push_or_update(0, {0, 100.0});
    h.push_or_update(1, {1, 300.0});
    h.push_or_update(2, {2, 200.0});

    CHECK("not empty", !h.empty());

    auto m1 = h.pop_best();
    CHECK("first pop valid", m1.has_value());
    CHECK("first is 300", m1->saving == 300.0);
    CHECK("first key=1", m1->key == 1);

    auto m2 = h.pop_best();
    CHECK("second is 200", m2->saving == 200.0);
    CHECK("second key=2", m2->key == 2);

    auto m3 = h.pop_best();
    CHECK("third is 100", m3->saving == 100.0);
    CHECK("third key=0", m3->key == 0);

    CHECK("now empty", h.empty());
    CHECK("pop empty returns nullopt", !h.pop_best().has_value());
}

// ============================================================================
// Test 2: update increases saving — should bubble up
// ============================================================================
void test_update_increase() {
    std::cout << "--- test_update_increase ---\n";
    PairingHeap<TestMove> h(10);

    h.push_or_update(0, {0, 100.0});
    h.push_or_update(1, {1, 200.0});
    h.push_or_update(2, {2, 50.0});

    // Increase key=2 from 50 to 500
    h.push_or_update(2, {2, 500.0});

    auto m = h.pop_best();
    CHECK("increased key pops first", m->key == 2);
    CHECK("increased saving=500", m->saving == 500.0);

    m = h.pop_best();
    CHECK("second is key=1", m->key == 1);
    m = h.pop_best();
    CHECK("third is key=0", m->key == 0);
    CHECK("empty after", h.empty());
}

// ============================================================================
// Test 3: update decreases saving — should sink down
// ============================================================================
void test_update_decrease() {
    std::cout << "--- test_update_decrease ---\n";
    PairingHeap<TestMove> h(10);

    h.push_or_update(0, {0, 300.0});
    h.push_or_update(1, {1, 200.0});
    h.push_or_update(2, {2, 100.0});

    // Decrease key=0 from 300 to 5
    h.push_or_update(0, {0, 5.0});

    auto m = h.pop_best();
    CHECK("key=1 now first (200)", m->key == 1);
    m = h.pop_best();
    CHECK("key=2 second (100)", m->key == 2);
    m = h.pop_best();
    CHECK("key=0 last (5)", m->key == 0 && m->saving == 5.0);
}

// ============================================================================
// Test 4: remove
// ============================================================================
void test_remove() {
    std::cout << "--- test_remove ---\n";
    PairingHeap<TestMove> h(10);

    h.push_or_update(0, {0, 100.0});
    h.push_or_update(1, {1, 200.0});
    h.push_or_update(2, {2, 300.0});

    CHECK("contains 1", h.contains(1));
    h.remove(1);
    CHECK("not contains 1", !h.contains(1));

    auto m = h.pop_best();
    CHECK("first after remove is key=2", m->key == 2);
    m = h.pop_best();
    CHECK("second after remove is key=0", m->key == 0);
    CHECK("empty after", h.empty());
}

// ============================================================================
// Test 5: remove root
// ============================================================================
void test_remove_root() {
    std::cout << "--- test_remove_root ---\n";
    PairingHeap<TestMove> h(10);

    h.push_or_update(0, {0, 500.0});
    h.push_or_update(1, {1, 200.0});
    h.push_or_update(2, {2, 300.0});

    h.remove(0);
    auto m = h.pop_best();
    CHECK("after removing root, key=2 is top", m->key == 2);
    CHECK("saving=300", m->saving == 300.0);
}

// ============================================================================
// Test 6: push_or_update same key multiple times
// ============================================================================
void test_repeated_update() {
    std::cout << "--- test_repeated_update ---\n";
    PairingHeap<TestMove> h(5);

    h.push_or_update(0, {0, 10.0});
    h.push_or_update(0, {0, 50.0});
    h.push_or_update(0, {0, 30.0});
    h.push_or_update(0, {0, 90.0});
    h.push_or_update(0, {0, 1.0});

    auto m = h.pop_best();
    CHECK("final saving is 1.0", m->saving == 1.0);
    CHECK("key=0", m->key == 0);
    CHECK("empty after single pop", h.empty());
}

// ============================================================================
// Test 7: single element
// ============================================================================
void test_single_element() {
    std::cout << "--- test_single_element ---\n";
    PairingHeap<TestMove> h(1);

    h.push_or_update(0, {0, 42.0});
    CHECK("not empty", !h.empty());
    CHECK("contains 0", h.contains(0));

    auto m = h.pop_best();
    CHECK("pop returns 42", m->saving == 42.0);
    CHECK("empty after", h.empty());
}

// ============================================================================
// Test 8: clear
// ============================================================================
void test_clear() {
    std::cout << "--- test_clear ---\n";
    PairingHeap<TestMove> h(10);

    for (int i = 0; i < 5; i++)
        h.push_or_update(i, {(size_t)i, (double)i * 10});

    CHECK("not empty before clear", !h.empty());
    h.clear();
    CHECK("empty after clear", h.empty());
    CHECK("pop after clear = nullopt", !h.pop_best().has_value());

    // Can re-use after clear
    h.push_or_update(3, {3, 77.0});
    CHECK("re-use after clear", !h.empty());
    auto m = h.pop_best();
    CHECK("re-use value correct", m->saving == 77.0);
}

// ============================================================================
// Test 9: remove nonexistent key is no-op
// ============================================================================
void test_remove_nonexistent() {
    std::cout << "--- test_remove_nonexistent ---\n";
    PairingHeap<TestMove> h(10);

    h.push_or_update(3, {3, 100.0});
    h.remove(7);  // not in heap
    h.remove(3);
    h.remove(3);  // double remove

    CHECK("empty after removes", h.empty());
}

// ============================================================================
// Test 10: stress test — random insert/update/remove/pop
// ============================================================================
void test_stress() {
    std::cout << "--- test_stress ---\n";
    const int N = 200;
    PairingHeap<TestMove> h(N);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> key_dist(0, N - 1);
    std::uniform_real_distribution<double> val_dist(-1000.0, 1000.0);

    // Ground truth: track what's in the heap
    std::vector<bool> in_heap(N, false);
    std::vector<double> values(N, 0.0);

    for (int iter = 0; iter < 2000; iter++) {
        int action = rng() % 4;
        int key = key_dist(rng);

        if (action <= 1) {
            // push or update
            double v = val_dist(rng);
            h.push_or_update(key, {(size_t)key, v});
            in_heap[key] = true;
            values[key] = v;
        } else if (action == 2) {
            // remove
            h.remove(key);
            in_heap[key] = false;
        } else {
            // pop best
            auto m = h.pop_best();
            // Find expected best
            double best_val = -1e18;
            int best_key = -1;
            for (int i = 0; i < N; i++) {
                if (in_heap[i] && values[i] > best_val) {
                    best_val = values[i];
                    best_key = i;
                }
            }
            if (best_key == -1) {
                CHECK("pop empty consistent", !m.has_value());
            } else {
                if (!m.has_value()) {
                    CHECK("pop should have value", false);
                } else {
                    bool ok = std::abs(m->saving - best_val) < 1e-9;
                    if (!ok) {
                        std::cout << "  FAIL: stress iter=" << iter
                                  << " expected key=" << best_key << " val=" << best_val
                                  << " got key=" << m->key << " val=" << m->saving << "\n";
                        g_fail++;
                    } else {
                        g_pass++;
                    }
                    in_heap[m->key] = false;
                }
            }
        }
    }

    // Drain and verify sorted order
    std::vector<double> drained;
    while (true) {
        auto m = h.pop_best();
        if (!m) break;
        drained.push_back(m->saving);
        in_heap[m->key] = false;
    }
    bool sorted = true;
    for (size_t i = 1; i < drained.size(); i++)
        if (drained[i] > drained[i - 1] + 1e-9) { sorted = false; break; }
    CHECK("drain is sorted descending", sorted);

    // All should be out
    bool all_out = true;
    for (int i = 0; i < N; i++)
        if (in_heap[i]) { all_out = false; break; }
    CHECK("all drained", all_out);
}

// ============================================================================
// Test 11: interleaved push and pop
// ============================================================================
void test_interleaved() {
    std::cout << "--- test_interleaved ---\n";
    PairingHeap<TestMove> h(20);

    // Push 5, pop 2, push 5 more, pop all — verify order
    for (int i = 0; i < 5; i++)
        h.push_or_update(i, {(size_t)i, (double)(i * 10)});

    auto m1 = h.pop_best();
    CHECK("first pop=40", m1->saving == 40.0);
    auto m2 = h.pop_best();
    CHECK("second pop=30", m2->saving == 30.0);

    for (int i = 5; i < 10; i++)
        h.push_or_update(i, {(size_t)i, (double)(i * 10)});

    // Now heap has: 0(0), 1(10), 2(20), 5(50), 6(60), 7(70), 8(80), 9(90)
    std::vector<double> expected = {90, 80, 70, 60, 50, 20, 10, 0};
    bool correct = true;
    for (double exp : expected) {
        auto m = h.pop_best();
        if (!m || std::abs(m->saving - exp) > 1e-9) {
            correct = false;
            std::cout << "  FAIL: expected " << exp << " got "
                      << (m ? m->saving : -999) << "\n";
            g_fail++;
            break;
        }
    }
    if (correct) g_pass++;
    CHECK("empty after drain", h.empty());
}

int main() {
    test_basic_push_pop();
    test_update_increase();
    test_update_decrease();
    test_remove();
    test_remove_root();
    test_repeated_update();
    test_single_element();
    test_clear();
    test_remove_nonexistent();
    test_stress();
    test_interleaved();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed out of "
              << (g_pass + g_fail) << " tests\n";
    return g_fail > 0 ? 1 : 0;
}