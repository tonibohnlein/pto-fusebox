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
    std::set<size_t> merged;
    merged.insert(a.begin(), a.end());
    merged.insert(b.begin(), b.end());

    // BFS from `from` through external ops; return true if we reach `to`
    auto has_external_path = [&](const std::set<size_t>& from,
                                 const std::set<size_t>& to) {
        std::set<size_t> visited;
        std::vector<size_t> queue;
        for (auto op : from)
            for (auto s : op_succs[op])
                if (!merged.count(s) && !visited.count(s))
                    { visited.insert(s); queue.push_back(s); }

        while (!queue.empty()) {
            size_t u = queue.back(); queue.pop_back();
            // Check successors: if any lands in `to`, we found an external path
            for (auto v : op_succs[u]) {
                if (to.count(v)) return true;       // reached target through external
                if (!merged.count(v) && !visited.count(v))
                    { visited.insert(v); queue.push_back(v); }
            }
        }
        return false;
    };

    return has_external_path(a, b) || has_external_path(b, a);
}
