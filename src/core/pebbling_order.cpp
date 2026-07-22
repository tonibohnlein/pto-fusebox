#include "core/pebbling_order.h"

#include <algorithm>
#include <deque>
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

// Adapt the Gorder greedy objective to an executable DAG schedule. The
// original algorithm may choose any remaining vertex; this implementation may
// choose only a ready node. Pair weights retain the paper's definition:
// direct directed edges plus common in-neighbors. Caller-provided locality
// values project non-schedulable boundary/request values into the same common-
// predecessor score without inventing executable graph nodes.
class DependencyConstrainedGorderStrategy final : public PebblingOrderStrategy {
 public:
  [[nodiscard]] std::vector<size_t> Compute(const PebblingOrderGraph& graph) const override {
    const size_t node_count = graph.nodes.size();
    std::map<size_t, size_t> index_by_id;
    for (size_t index = 0; index < node_count; ++index) {
      if (!index_by_id.emplace(graph.nodes[index].id, index).second) {
        return {};
      }
    }

    std::vector<std::vector<size_t>> predecessors(node_count);
    std::vector<std::vector<size_t>> successors(node_count);
    std::vector<std::vector<size_t>> locality_values(node_count);
    for (size_t index = 0; index < node_count; ++index) {
      for (size_t predecessor_id : graph.nodes[index].predecessors) {
        const auto found = index_by_id.find(predecessor_id);
        if (found == index_by_id.end()) return {};
        predecessors[index].push_back(found->second);
      }
      std::sort(predecessors[index].begin(), predecessors[index].end());
      predecessors[index].erase(std::unique(predecessors[index].begin(), predecessors[index].end()),
                                predecessors[index].end());
      for (size_t predecessor : predecessors[index]) {
        successors[predecessor].push_back(index);
      }

      locality_values[index] = graph.nodes[index].locality_values;
      std::sort(locality_values[index].begin(), locality_values[index].end());
      locality_values[index].erase(std::unique(locality_values[index].begin(), locality_values[index].end()),
                                   locality_values[index].end());
    }

    std::map<size_t, std::vector<size_t>> users_by_locality_value;
    for (size_t node = 0; node < node_count; ++node) {
      for (size_t value : locality_values[node]) {
        users_by_locality_value[value].push_back(node);
      }
    }
    struct ReadyKey {
      size_t score = 0;
      size_t stable_position = 0;
      size_t id = 0;
      size_t index = 0;
    };
    struct ReadyBefore {
      bool operator()(const ReadyKey& lhs, const ReadyKey& rhs) const {
        if (lhs.score != rhs.score) return lhs.score > rhs.score;
        if (lhs.stable_position != rhs.stable_position) {
          return lhs.stable_position < rhs.stable_position;
        }
        if (lhs.id != rhs.id) return lhs.id < rhs.id;
        return lhs.index < rhs.index;
      }
    };

    std::vector<size_t> indegree(node_count, 0);
    std::vector<size_t> score(node_count, 0);
    std::vector<bool> ready(node_count, false);
    std::vector<bool> scheduled(node_count, false);
    std::set<ReadyKey, ReadyBefore> ready_nodes;
    auto make_ready_key = [&](size_t index) {
      return ReadyKey{score[index], graph.nodes[index].stable_position, graph.nodes[index].id, index};
    };
    auto insert_ready = [&](size_t index) {
      ready[index] = true;
      ready_nodes.insert(make_ready_key(index));
    };
    for (size_t node = 0; node < node_count; ++node) {
      indegree[node] = predecessors[node].size();
      if (indegree[node] == 0) insert_ready(node);
    }

    std::vector<size_t> delta(node_count, 0);
    std::vector<size_t> touched;
    auto add_delta = [&](size_t node) {
      if (delta[node] == 0) touched.push_back(node);
      ++delta[node];
    };
    auto adjust_window_score = [&](size_t window_node, bool entering) {
      // Compute only the relationships incident to the entering/leaving node,
      // mirroring Gorder's incremental priority-queue update. `delta` combines
      // multiple common predecessors/locality values into the exact pair
      // weight without materializing the potentially quadratic sibling graph.
      touched.clear();
      for (size_t predecessor : predecessors[window_node]) {
        add_delta(predecessor);  // direct edge
      }
      for (size_t successor : successors[window_node]) {
        add_delta(successor);  // direct edge
      }
      for (size_t predecessor : predecessors[window_node]) {
        for (size_t sibling : successors[predecessor]) {
          if (sibling != window_node) add_delta(sibling);
        }
      }
      for (size_t value : locality_values[window_node]) {
        for (size_t user : users_by_locality_value.at(value)) {
          if (user != window_node) add_delta(user);
        }
      }
      bool valid = true;
      for (size_t other : touched) {
        const size_t weight = delta[other];
        if (scheduled[other]) continue;
        if (ready[other]) ready_nodes.erase(make_ready_key(other));
        if (entering) {
          score[other] += weight;
        } else {
          if (score[other] < weight) {
            valid = false;
          } else {
            score[other] -= weight;
          }
        }
        if (ready[other]) ready_nodes.insert(make_ready_key(other));
      }
      for (size_t other : touched) delta[other] = 0;
      return valid;
    };

    std::deque<size_t> window;
    std::vector<size_t> order;
    order.reserve(node_count);
    while (order.size() < node_count) {
      if (ready_nodes.empty()) return {};
      const ReadyKey chosen = *ready_nodes.begin();
      ready_nodes.erase(ready_nodes.begin());
      ready[chosen.index] = false;
      scheduled[chosen.index] = true;
      order.push_back(chosen.id);

      if (!adjust_window_score(chosen.index, true)) return {};
      window.push_back(chosen.index);
      if (window.size() > kDependencyConstrainedGorderWindow) {
        const size_t leaving = window.front();
        window.pop_front();
        if (!adjust_window_score(leaving, false)) return {};
      }

      for (size_t successor : successors[chosen.index]) {
        if (indegree[successor] == 0) return {};
        --indegree[successor];
        if (indegree[successor] == 0) insert_ready(successor);
      }
    }
    return order;
  }
};

}  // namespace

PebblingOrderGraph BuildSourceOpPebblingGraph(const Problem& prob, const DAG& dag,
                                              const std::vector<size_t>& ops,
                                              const std::vector<size_t>& roots) {
  enum class SourceLocalityRole {
    Generic,
    CubeLhs,
    CubeRhs,
  };
  std::vector<bool> in_subgraph(prob.num_ops(), false);
  for (size_t op : ops) in_subgraph[op] = true;
  std::map<std::pair<size_t, SourceLocalityRole>, size_t> boundary_locality_ids;

  PebblingOrderGraph graph;
  graph.nodes.reserve(ops.size());
  graph.roots = roots;
  for (size_t op : ops) {
    PebblingOrderNode node;
    node.id = op;
    node.stable_position = dag.topo_position(op);
    for (size_t input_index = 0; input_index < prob.ops[op].inputs.size(); ++input_index) {
      const size_t tensor = prob.ops[op].inputs[input_index];
      const int producer = dag.tensor_producer[tensor];
      if (producer >= 0 && in_subgraph[static_cast<size_t>(producer)]) {
        node.predecessors.push_back(static_cast<size_t>(producer));
      } else {
        SourceLocalityRole role = SourceLocalityRole::Generic;
        if (prob.ops[op].type == OpType::MatMul) {
          role = input_index == 0 ? SourceLocalityRole::CubeLhs : SourceLocalityRole::CubeRhs;
        }
        const auto [it, inserted] =
            boundary_locality_ids.emplace(std::make_pair(tensor, role), boundary_locality_ids.size());
        (void)inserted;
        node.locality_values.push_back(it->second);
      }
    }
    std::sort(node.predecessors.begin(), node.predecessors.end());
    node.predecessors.erase(std::unique(node.predecessors.begin(), node.predecessors.end()),
                            node.predecessors.end());
    std::sort(node.locality_values.begin(), node.locality_values.end());
    node.locality_values.erase(std::unique(node.locality_values.begin(), node.locality_values.end()),
                               node.locality_values.end());
    graph.nodes.push_back(std::move(node));
  }
  return graph;
}

const PebblingOrderStrategy& GetPebblingOrderStrategy(PebblingOrderKind kind) {
  static const DfsPostOrderStrategy dfs_post_order;
  static const DependencyConstrainedGorderStrategy dependency_constrained_gorder;
  switch (kind) {
    case PebblingOrderKind::DfsPostOrder:
      return dfs_post_order;
    case PebblingOrderKind::DependencyConstrainedGorder:
      return dependency_constrained_gorder;
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
