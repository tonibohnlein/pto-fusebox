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

bool DAG::merge_creates_cycle(const std::set<size_t>& a, const std::set<size_t>& b) const {
    // Use vector<bool> for O(1) lookups instead of std::set
    std::vector<bool> in_merged(num_ops, false);
    for (auto op : a) in_merged[op] = true;
    for (auto op : b) in_merged[op] = true;

    auto has_external_path = [&](const std::set<size_t>& from,
                                 const std::set<size_t>& to) {
        std::vector<bool> visited(num_ops, false);
        std::vector<size_t> queue;
        for (auto op : from)
            for (auto s : op_succs[op])
                if (!in_merged[s] && !visited[s])
                    { visited[s] = true; queue.push_back(s); }

        while (!queue.empty()) {
            size_t u = queue.back(); queue.pop_back();
            for (auto v : op_succs[u]) {
                if (to.count(v)) return true;
                if (!in_merged[v] && !visited[v])
                    { visited[v] = true; queue.push_back(v); }
            }
        }
        return false;
    };

    return has_external_path(a, b) || has_external_path(b, a);
}