#include "core/dag.h"
#include <set>

DAG DAG::build(const Problem& prob) {
    DAG d;
    d.num_ops = prob.num_ops();
    d.tensor_producer.resize(prob.num_tensors(), -1);
    d.tensor_consumers.resize(prob.num_tensors());
    d.op_preds.resize(prob.num_ops());
    d.op_succs.resize(prob.num_ops());

    for (size_t i = 0; i < prob.num_ops(); i++) {
        for (auto t : prob.ops[i].outputs) d.tensor_producer[t] = (int)i;
        for (auto t : prob.ops[i].inputs)  d.tensor_consumers[t].push_back(i);
    }

    // Build op-level adjacency from tensor edges
    for (size_t i = 0; i < prob.num_ops(); i++) {
        for (auto t : prob.ops[i].inputs) {
            int prod = d.tensor_producer[t];
            if (prod >= 0 && (size_t)prod != i) {
                d.op_preds[i].insert((size_t)prod);
                d.op_succs[(size_t)prod].insert(i);
            }
        }
    }

    // Build expanded adjacency: DAG edges + co-consumer edges
    d.op_neighbors.resize(prob.num_ops());
    {
        std::vector<std::set<size_t>> nbr_set(prob.num_ops());
        // DAG edges (undirected)
        for (size_t i = 0; i < prob.num_ops(); i++) {
            for (auto j : d.op_preds[i]) nbr_set[i].insert(j);
            for (auto j : d.op_succs[i]) nbr_set[i].insert(j);
        }
        // Co-consumer edges: ops sharing an input tensor
        for (size_t t = 0; t < prob.num_tensors(); t++) {
            auto& cons = d.tensor_consumers[t];
            for (size_t a = 0; a < cons.size(); a++)
                for (size_t b = a + 1; b < cons.size(); b++) {
                    nbr_set[cons[a]].insert(cons[b]);
                    nbr_set[cons[b]].insert(cons[a]);
                }
        }
        // Flatten to vectors for cache-friendly iteration
        for (size_t i = 0; i < prob.num_ops(); i++)
            d.op_neighbors[i].assign(nbr_set[i].begin(), nbr_set[i].end());
    }

    // Identify graph inputs and outputs
    std::set<size_t> produced, consumed;
    for (auto& op : prob.ops) {
        for (auto t : op.outputs) produced.insert(t);
        for (auto t : op.inputs) consumed.insert(t);
    }
    for (size_t i = 0; i < prob.num_tensors(); i++) {
        if (!produced.count(i)) d.graph_inputs.push_back(i);
        if (!consumed.count(i)) d.graph_outputs.push_back(i);
    }

    d.precompute_reachability();
    return d;
}

std::vector<size_t> DAG::topo_sort() const {
    std::vector<int> in_degree(num_ops, 0);
    for (size_t i = 0; i < num_ops; i++)
        in_degree[i] = (int)op_preds[i].size();

    std::vector<size_t> order, queue;
    for (size_t i = 0; i < num_ops; i++)
        if (in_degree[i] == 0) queue.push_back(i);

    while (!queue.empty()) {
        size_t u = queue.back(); queue.pop_back();
        order.push_back(u);
        for (auto v : op_succs[u])
            if (--in_degree[v] == 0) queue.push_back(v);
    }
    return order;
}

void DAG::precompute_reachability() {
    words_per_row_ = (num_ops + 63) / 64;
    reachable_.assign(num_ops * words_per_row_, 0);

    auto order = topo_sort();
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        size_t u = order[i];
        // u reaches itself
        reachable_[u * words_per_row_ + u / 64] |= (1ull << (u % 64));
        // u reaches everything its successors reach
        for (auto v : op_succs[u])
            row_or(u, v);
    }
}

bool DAG::merge_creates_cycle(const std::set<size_t>& a,
                               const std::set<size_t>& b) const {
    // Build bitmasks for each side
    std::vector<uint64_t> mask_a(words_per_row_, 0), mask_b(words_per_row_, 0);
    for (auto u : a) mask_a[u / 64] |= (1ull << (u % 64));
    for (auto u : b) mask_b[u / 64] |= (1ull << (u % 64));

    // Cycle iff any node in B reaches any node in A, or vice versa
    for (auto u : b)
        if (row_intersects(u, mask_a)) return true;
    for (auto u : a)
        if (row_intersects(u, mask_b)) return true;
    return false;
}