#include "core/pebbling_order.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace {

class DfsPostOrderStrategy final : public PebblingOrderStrategy {
 public:
  [[nodiscard]] std::vector<size_t> Compute(
      const PebblingOrderGraph& graph) const override {
    std::map<size_t, const PebblingOrderNode*> by_id;
    for (const PebblingOrderNode& node : graph.nodes) {
      by_id.emplace(node.id, &node);
    }

    auto stable_position = [&](size_t id) {
      auto it = by_id.find(id);
      return it == by_id.end() ? id : it->second->stable_position;
    };
    auto by_stable_position = [&](size_t lhs, size_t rhs) {
      const size_t lhs_position = stable_position(lhs);
      const size_t rhs_position = stable_position(rhs);
      return lhs_position != rhs_position ? lhs_position < rhs_position
                                          : lhs < rhs;
    };

    std::vector<size_t> roots = graph.roots;
    std::sort(roots.begin(), roots.end(), by_stable_position);
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());

    std::set<size_t> visited;
    std::vector<std::pair<size_t, bool>> stack;
    std::vector<size_t> order;
    order.reserve(graph.nodes.size());
    for (size_t root : roots) {
      if (visited.count(root) != 0 || by_id.count(root) == 0) continue;
      stack.emplace_back(root, false);
      while (!stack.empty()) {
        const auto [node_id, expanded] = stack.back();
        stack.pop_back();
        if (expanded) {
          order.push_back(node_id);
          continue;
        }
        if (!visited.insert(node_id).second) continue;
        stack.emplace_back(node_id, true);

        std::vector<size_t> predecessors = by_id.at(node_id)->predecessors;
        std::sort(predecessors.begin(), predecessors.end(),
                  by_stable_position);
        predecessors.erase(
            std::unique(predecessors.begin(), predecessors.end()),
            predecessors.end());
        for (auto it = predecessors.rbegin(); it != predecessors.rend(); ++it) {
          if (visited.count(*it) == 0 && by_id.count(*it) != 0) {
            stack.emplace_back(*it, false);
          }
        }
      }
    }
    return order;
  }
};

}  // namespace

PebblingOrderGraph BuildSourceOpPebblingGraph(
    const Problem& prob, const DAG& dag, const std::vector<size_t>& ops,
    const std::vector<size_t>& roots) {
  std::vector<bool> in_subgraph(prob.num_ops(), false);
  for (size_t op : ops) in_subgraph[op] = true;

  PebblingOrderGraph graph;
  graph.nodes.reserve(ops.size());
  graph.roots = roots;
  for (size_t op : ops) {
    PebblingOrderNode node;
    node.id = op;
    node.stable_position = dag.topo_position(op);
    for (size_t tensor : prob.ops[op].inputs) {
      const int producer = dag.tensor_producer[tensor];
      if (producer >= 0 && in_subgraph[static_cast<size_t>(producer)]) {
        node.predecessors.push_back(static_cast<size_t>(producer));
      }
    }
    std::sort(node.predecessors.begin(), node.predecessors.end());
    node.predecessors.erase(
        std::unique(node.predecessors.begin(), node.predecessors.end()),
        node.predecessors.end());
    graph.nodes.push_back(std::move(node));
  }
  return graph;
}

const PebblingOrderStrategy& GetPebblingOrderStrategy(
    PebblingOrderKind kind) {
  static const DfsPostOrderStrategy dfs_post_order;
  switch (kind) {
    case PebblingOrderKind::DfsPostOrder:
      return dfs_post_order;
  }
  return dfs_post_order;
}

std::vector<size_t> ComputePebblingOrder(PebblingOrderKind kind,
                                         const PebblingOrderGraph& graph) {
  return GetPebblingOrderStrategy(kind).Compute(graph);
}

PebblingValueLifetimePlan ComputeAlwaysRetainedValueLifetimes(
    size_t step_count, const std::vector<PebblingValueEvent>& events) {
  struct Accumulator {
    size_t first_materialization = std::numeric_limits<size_t>::max();
    size_t first_use = std::numeric_limits<size_t>::max();
    size_t last_use = 0;
    size_t materialization_count = 0;
    size_t use_count = 0;
  };

  PebblingValueLifetimePlan plan;
  std::map<size_t, Accumulator> by_value;
  for (const PebblingValueEvent& event : events) {
    if (event.step >= step_count) return plan;
    Accumulator& accumulator = by_value[event.value_id];
    if (event.kind == PebblingValueEventKind::Materialize) {
      accumulator.first_materialization =
          std::min(accumulator.first_materialization, event.step);
      ++accumulator.materialization_count;
    } else {
      accumulator.first_use = std::min(accumulator.first_use, event.step);
      accumulator.last_use = std::max(accumulator.last_use, event.step);
      ++accumulator.use_count;
    }
  }

  plan.lifetimes.reserve(by_value.size());
  for (const auto& [value_id, accumulator] : by_value) {
    // The initial policy has exactly one materialization and never recomputes.
    // Values without a consumer do not need an on-chip lifetime.
    if (accumulator.materialization_count > 1) return PebblingValueLifetimePlan{};
    if (accumulator.use_count == 0) continue;
    const bool has_materialization = accumulator.materialization_count == 1;
    const size_t first_live = has_materialization
                                  ? accumulator.first_materialization
                                  : accumulator.first_use;
    if (has_materialization &&
        accumulator.first_materialization > accumulator.first_use) {
      return PebblingValueLifetimePlan{};
    }
    plan.lifetimes.push_back({value_id, first_live, accumulator.last_use,
                              accumulator.use_count, has_materialization});
  }
  plan.valid = true;
  return plan;
}
