#include "core/subgraph_structure.h"

#include <algorithm>
#include <utility>

// ============================================================================
// SubgraphStructure — architecture-independent structural classification.
//
// This mirrors EXACTLY the structural half of Subgraph::create() in
// subgraph.cpp (boundary/ephemeral classification, sink detection, and the
// post-order DFS execution order). The arch-specific cost model composes this
// and adds tiling / feasibility / cost on top; nothing here depends on the
// machine parameters, so every backend shares it verbatim.
// ============================================================================

SubgraphStructure::SubgraphStructure(const Problem &prob, const DAG &dag,
                                     std::vector<size_t> ops)
    : prob_(&prob), dag_(&dag), ops_(std::move(ops)) {
  if (ops_.empty())
    return;  // structurally degenerate — valid_ stays false

  const size_t num_tensors = prob.num_tensors();
  const size_t num_ops = prob.num_ops();

  // Membership + producer/consumer maps over this op set (in-subgraph only).
  std::vector<bool> is_in_sg(num_ops, false);
  std::vector<bool> is_produced(num_tensors, false);
  std::vector<bool> is_consumed(num_tensors, false);
  for (auto i : ops_) {
    is_in_sg[i] = true;
    is_produced[prob.ops[i].output()] = true;
    for (auto t : prob.ops[i].inputs)
      is_consumed[t] = true;
  }

  // Boundary / ephemeral classification (pure DAG facts):
  //   boundary input  = consumed inside, NOT produced inside
  //   ephemeral       = produced AND consumed inside
  //   boundary output = produced inside and either terminal or externally required
  //
  // Consumers in other solver groups remain a partition-layer concern. Function
  // results are different: Problem::required_outputs is an explicit observable
  // boundary, so a returned-and-consumed value is BOTH an on-chip intermediate
  // (ephemeral lifetime) and a DDR live-out.
  std::vector<bool> is_ephemeral(num_tensors, false);
  for (size_t t = 0; t < num_tensors; t++) {
    if (is_consumed[t] && !is_produced[t])
      boundary_inputs_.insert(t);
    if (is_produced[t] && is_consumed[t])
      is_ephemeral[t] = true;
  }
  for (size_t t = 0; t < num_tensors; t++) {
    if (is_produced[t] && (!is_ephemeral[t] || prob.required_outputs.count(t)))
      boundary_outputs_.insert(t);
    if (is_ephemeral[t])
      ephemeral_.insert(t);
  }

  // Execution roots / live-outs: terminal ops plus producers of explicitly
  // required outputs. A required value may still have an internal successor;
  // rooting its producer makes the DFS replay and sink metadata retain both
  // observable paths.
  for (auto i : ops_) {
    bool has_internal_succ = false;
    size_t t = prob.ops[i].output();
    for (auto cop : dag.tensor_consumers[t])
      if (is_in_sg[cop]) { has_internal_succ = true; break; }
    if (!has_internal_succ || prob.required_outputs.count(t))
      sinks_.push_back(i);
  }

  // A non-empty acyclic op set always has a sink, and a sink's output is by
  // definition a boundary output — so non-empty boundary_outputs_ is the
  // structural-validity test (equivalently, non-empty sinks_).
  if (boundary_outputs_.empty())
    return;  // valid_ stays false

  // Fixed depth-first (post-order) execution order — the pebbling order. Mirrors
  // subgraph.cpp: post-order DFS from each sink over in-subgraph producers, with
  // a topo-position tie-break so the same subgraph always yields the same order
  // (required for the cost cache).
  {
    auto by_topo = [&](size_t a, size_t b) {
      return dag.topo_position(a) < dag.topo_position(b);
    };
    auto sg_producers = [&](size_t op) {
      std::vector<size_t> preds;
      for (auto t : prob.ops[op].inputs) {
        int p = dag.tensor_producer[t];
        if (p >= 0 && is_in_sg[(size_t)p]) preds.push_back((size_t)p);
      }
      std::sort(preds.begin(), preds.end(), by_topo);
      preds.erase(std::unique(preds.begin(), preds.end()), preds.end());
      return preds;
    };
    std::vector<size_t> roots = sinks_;
    std::sort(roots.begin(), roots.end(), by_topo);

    std::vector<bool> visited(num_ops, false);
    std::vector<std::pair<size_t, bool>> stack;  // (op, expanded?)
    for (auto root : roots) {
      if (visited[root]) continue;
      stack.push_back({root, false});
      while (!stack.empty()) {
        auto [op, expanded] = stack.back();
        stack.pop_back();
        if (expanded) {  // all producers already emitted
          dfs_order_.push_back(op);
          continue;
        }
        if (visited[op]) continue;
        visited[op] = true;
        stack.push_back({op, true});  // emit after producers
        auto preds = sg_producers(op);
        // Push in reverse so the smallest-topo producer is processed first.
        for (auto it = preds.rbegin(); it != preds.rend(); ++it)
          if (!visited[*it]) stack.push_back({*it, false});
      }
    }
  }

  valid_ = true;
}
