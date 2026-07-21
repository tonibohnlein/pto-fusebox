#include "core/dag.h"
#include "core/pebbling_order.h"
#include "core/types.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void Check(const char* label, bool condition) {
  if (condition) {
    ++g_pass;
  } else {
    ++g_fail;
    std::cout << "  FAIL: " << label << "\n";
  }
}

bool IsTopological(const PebblingOrderGraph& graph,
                   const std::vector<size_t>& order) {
  std::map<size_t, size_t> position;
  for (size_t index = 0; index < order.size(); ++index) {
    position.emplace(order[index], index);
  }
  if (position.size() != graph.nodes.size()) return false;
  for (const PebblingOrderNode& node : graph.nodes) {
    auto node_position = position.find(node.id);
    if (node_position == position.end()) return false;
    for (size_t predecessor : node.predecessors) {
      auto predecessor_position = position.find(predecessor);
      if (predecessor_position == position.end() ||
          predecessor_position->second >= node_position->second) {
        return false;
      }
    }
  }
  return true;
}

void TestChainMatchesExistingDfs() {
  Problem problem;
  problem.tensors = {{8, 8}, {8, 8}, {8, 8}, {8, 8}};
  problem.ops = {{OpType::Pointwise, {0}, {1}},
                 {OpType::Pointwise, {1}, {2}},
                 {OpType::Pointwise, {2}, {3}}};
  DAG dag = DAG::build(problem);
  const PebblingOrderGraph graph =
      BuildSourceOpPebblingGraph(problem, dag, {0, 1, 2}, {2});
  const std::vector<size_t> order =
      ComputePebblingOrder(PebblingOrderKind::DfsPostOrder, graph);
  Check("chain DFS remains [0,1,2]", order == std::vector<size_t>({0, 1, 2}));
  Check("chain DFS is topological", IsTopological(graph, order));
}

void TestDiamondFinishesOneBranchBeforeSibling() {
  Problem problem;
  problem.tensors = {{8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8}};
  problem.ops = {{OpType::Pointwise, {0}, {1}},
                 {OpType::Pointwise, {1}, {2}},
                 {OpType::Pointwise, {1}, {3}},
                 {OpType::Pointwise, {2, 3}, {4}}};
  DAG dag = DAG::build(problem);
  const PebblingOrderGraph graph =
      BuildSourceOpPebblingGraph(problem, dag, {0, 1, 2, 3}, {3});
  const std::vector<size_t> order =
      ComputePebblingOrder(PebblingOrderKind::DfsPostOrder, graph);
  Check("diamond DFS preserves stable predecessor order",
        order == std::vector<size_t>({0, 1, 2, 3}));
  Check("diamond DFS is topological", IsTopological(graph, order));
}

void TestMultipleRootsAreDeterministic() {
  Problem problem;
  problem.tensors = {{8, 8}, {8, 8}, {8, 8}, {8, 8}};
  problem.ops = {{OpType::Pointwise, {0}, {1}},
                 {OpType::Pointwise, {1}, {2}},
                 {OpType::Pointwise, {1}, {3}}};
  DAG dag = DAG::build(problem);
  const PebblingOrderGraph graph =
      BuildSourceOpPebblingGraph(problem, dag, {0, 1, 2}, {2, 1});
  const std::vector<size_t> first =
      ComputePebblingOrder(PebblingOrderKind::DfsPostOrder, graph);
  const std::vector<size_t> second =
      ComputePebblingOrder(PebblingOrderKind::DfsPostOrder, graph);
  Check("multi-root DFS uses stable root order",
        first == std::vector<size_t>({0, 1, 2}));
  Check("multi-root DFS is deterministic", first == second);
  Check("multi-root DFS is topological", IsTopological(graph, first));
}

void TestGenericIdsNeedNotBeDense() {
  PebblingOrderGraph graph;
  graph.nodes = {{42, 0, {}}, {7, 1, {42}}, {99, 2, {42}},
                 {13, 3, {7, 99}}};
  graph.roots = {13};
  const std::vector<size_t> order =
      ComputePebblingOrder(PebblingOrderKind::DfsPostOrder, graph);
  Check("generic graph keeps stable non-dense ids",
        order == std::vector<size_t>({42, 7, 99, 13}));
  Check("generic non-dense graph is topological", IsTopological(graph, order));
}

void TestBoundaryValueLivesFromFirstThroughLastUse() {
  const PebblingValueLifetimePlan plan =
      ComputeAlwaysRetainedValueLifetimes(
          4, {{17, 0, PebblingValueEventKind::Use},
              {17, 3, PebblingValueEventKind::Use}});
  Check("repeated boundary lifetime is valid", plan.valid);
  Check("repeated boundary has one canonical lifetime",
        plan.lifetimes.size() == 1);
  if (plan.lifetimes.size() == 1) {
    const PebblingValueLifetime& lifetime = plan.lifetimes.front();
    Check("boundary lifetime starts at first use",
          lifetime.first_live_step == 0);
    Check("boundary lifetime ends at last use", lifetime.last_use_step == 3);
    Check("boundary lifetime records both uses", lifetime.use_count == 2);
    Check("repeated boundary spans steps", lifetime.spans_steps());
  }
}

void TestProducedValueLivesFromMaterialization() {
  const PebblingValueLifetimePlan plan =
      ComputeAlwaysRetainedValueLifetimes(
          4, {{5, 0, PebblingValueEventKind::Materialize},
              {5, 2, PebblingValueEventKind::Use},
              {5, 3, PebblingValueEventKind::Use}});
  Check("produced lifetime is valid", plan.valid);
  Check("produced value has one lifetime", plan.lifetimes.size() == 1);
  if (plan.lifetimes.size() == 1) {
    const PebblingValueLifetime& lifetime = plan.lifetimes.front();
    Check("produced lifetime starts at materialization",
          lifetime.first_live_step == 0);
    Check("produced lifetime reaches final consumer",
          lifetime.last_use_step == 3);
    Check("produced lifetime remembers its definition",
          lifetime.has_materialization);
  }
}

void TestSingleUseBoundaryStaysTransient() {
  const PebblingValueLifetimePlan plan =
      ComputeAlwaysRetainedValueLifetimes(
          2, {{8, 1, PebblingValueEventKind::Use}});
  Check("single-use boundary lifetime is valid", plan.valid);
  Check("single-use boundary is represented", plan.lifetimes.size() == 1);
  if (plan.lifetimes.size() == 1) {
    Check("single-use boundary does not span steps",
          !plan.lifetimes.front().spans_steps());
  }
}

void TestCallerKeepsIncompatibleRolesDistinct() {
  const PebblingValueLifetimePlan plan =
      ComputeAlwaysRetainedValueLifetimes(
          1, {{100, 0, PebblingValueEventKind::Use},
              {101, 0, PebblingValueEventKind::Use}});
  Check("role-expanded lifetime plan is valid", plan.valid);
  Check("LHS and RHS value identities remain separate",
        plan.lifetimes.size() == 2 &&
            plan.lifetimes[0].value_id == 100 &&
            plan.lifetimes[1].value_id == 101);
}

void TestInvalidLifetimeTopologiesDecline() {
  const PebblingValueLifetimePlan late_materialization =
      ComputeAlwaysRetainedValueLifetimes(
          3, {{1, 0, PebblingValueEventKind::Use},
              {1, 1, PebblingValueEventKind::Materialize}});
  Check("materialization after use declines", !late_materialization.valid);

  const PebblingValueLifetimePlan rematerialized =
      ComputeAlwaysRetainedValueLifetimes(
          3, {{2, 0, PebblingValueEventKind::Materialize},
              {2, 1, PebblingValueEventKind::Materialize},
              {2, 2, PebblingValueEventKind::Use}});
  Check("unmodeled rematerialization declines", !rematerialized.valid);

  const PebblingValueLifetimePlan bad_step =
      ComputeAlwaysRetainedValueLifetimes(
          2, {{3, 2, PebblingValueEventKind::Use}});
  Check("out-of-range lifetime event declines", !bad_step.valid);
}

}  // namespace

int main() {
  TestChainMatchesExistingDfs();
  TestDiamondFinishesOneBranchBeforeSibling();
  TestMultipleRootsAreDeterministic();
  TestGenericIdsNeedNotBeDense();
  TestBoundaryValueLivesFromFirstThroughLastUse();
  TestProducedValueLivesFromMaterialization();
  TestSingleUseBoundaryStaysTransient();
  TestCallerKeepsIncompatibleRolesDistinct();
  TestInvalidLifetimeTopologiesDecline();
  std::cout << "=== pass=" << g_pass << " fail=" << g_fail << " ===\n";
  return g_fail == 0 ? 0 : 1;
}
