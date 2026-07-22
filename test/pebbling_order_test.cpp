#include "core/pebbling_order.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include "core/ascend910b_cost.h"
#include "core/dag.h"
#include "core/subgraph_structure.h"
#include "core/types.h"

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

void SetVector910B(Problem* problem) {
  problem->num_cube_cores = 24;
  problem->num_vector_cores = 48;
  problem->fast_memory_capacity = 1 << 26;
  problem->cube_capacity = 128 * 1024;
  problem->l1_capacity = 512 * 1024;
  problem->vec_capacity = 192 * 1024;
  problem->cube_freq_hz = 1.85e9;
  problem->bw_gm_ub = 100.9;
  problem->bw_ub_gm = 188.46;
  problem->vec_reg_bytes = 256;
  problem->vec_op_head = 14.0;
  problem->vec_op_tail = 18.0;
  problem->vec_slope_pw = 2.0;
  problem->vec_slope_reduce = 14.0;
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

size_t GorderObjective(const PebblingOrderGraph& graph, const std::vector<size_t>& order) {
  std::map<size_t, const PebblingOrderNode*> by_id;
  for (const PebblingOrderNode& node : graph.nodes) {
    by_id.emplace(node.id, &node);
  }
  auto pair_score = [&](size_t lhs_id, size_t rhs_id) {
    const PebblingOrderNode& lhs = *by_id.at(lhs_id);
    const PebblingOrderNode& rhs = *by_id.at(rhs_id);
    size_t score = static_cast<size_t>(std::count(lhs.predecessors.begin(), lhs.predecessors.end(), rhs_id)) +
                   static_cast<size_t>(std::count(rhs.predecessors.begin(), rhs.predecessors.end(), lhs_id));
    for (size_t predecessor : lhs.predecessors) {
      score += static_cast<size_t>(std::count(rhs.predecessors.begin(), rhs.predecessors.end(), predecessor));
    }
    for (size_t value : lhs.locality_values) {
      score += static_cast<size_t>(std::count(rhs.locality_values.begin(), rhs.locality_values.end(), value));
    }
    return score;
  };

  size_t objective = 0;
  for (size_t index = 0; index < order.size(); ++index) {
    const size_t first =
        index > kDependencyConstrainedGorderWindow ? index - kDependencyConstrainedGorderWindow : 0;
    for (size_t previous = first; previous < index; ++previous) {
      objective += pair_score(order[previous], order[index]);
    }
  }
  return objective;
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
  const PebblingOrderGraph graph = BuildSourceOpPebblingGraph(problem, dag, {0, 1, 2, 3}, {3});
  const std::vector<size_t> order = ComputePebblingOrder(PebblingOrderKind::DfsPostOrder, graph);
  Check("diamond DFS preserves stable predecessor order", order == std::vector<size_t>({0, 1, 2, 3}));
  Check("diamond DFS is topological", IsTopological(graph, order));
}

void TestMultipleRootsAreDeterministic() {
  Problem problem;
  problem.tensors = {{8, 8}, {8, 8}, {8, 8}, {8, 8}};
  problem.ops = {{OpType::Pointwise, {0}, {1}}, {OpType::Pointwise, {1}, {2}}, {OpType::Pointwise, {1}, {3}}};
  DAG dag = DAG::build(problem);
  const PebblingOrderGraph graph = BuildSourceOpPebblingGraph(problem, dag, {0, 1, 2}, {2, 1});
  const std::vector<size_t> first = ComputePebblingOrder(PebblingOrderKind::DfsPostOrder, graph);
  const std::vector<size_t> second = ComputePebblingOrder(PebblingOrderKind::DfsPostOrder, graph);
  Check("multi-root DFS uses stable root order", first == std::vector<size_t>({0, 1, 2}));
  Check("multi-root DFS is deterministic", first == second);
  Check("multi-root DFS is topological", IsTopological(graph, first));
}

void TestGenericIdsNeedNotBeDense() {
  PebblingOrderGraph graph;
  graph.nodes = {{42, 0, {}}, {7, 1, {42}}, {99, 2, {42}}, {13, 3, {7, 99}}};
  graph.roots = {13};
  const std::vector<size_t> order = ComputePebblingOrder(PebblingOrderKind::DfsPostOrder, graph);
  Check("generic graph keeps stable non-dense ids", order == std::vector<size_t>({42, 7, 99, 13}));
  Check("generic non-dense graph is topological", IsTopological(graph, order));
}

void TestGorderPreservesTopologyAndSiblingLocality() {
  PebblingOrderGraph graph;
  graph.nodes = {{0, 0, {}, {}}, {1, 1, {0}, {}}, {2, 2, {}, {}}, {3, 3, {0}, {}}};
  graph.roots = {1, 2, 3};
  const std::vector<size_t> order =
      ComputePebblingOrder(PebblingOrderKind::DependencyConstrainedGorder, graph);
  Check("Gorder groups siblings before an unrelated ready node", order == std::vector<size_t>({0, 1, 3, 2}));
  Check("Gorder result remains topological", IsTopological(graph, order));
}

void TestGorderGroupsSharedBoundaryValues() {
  Problem problem;
  problem.tensors = {{8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8}};
  problem.ops = {{OpType::Pointwise, {0}, {2}}, {OpType::Pointwise, {1}, {3}}, {OpType::Pointwise, {0}, {4}}};
  DAG dag = DAG::build(problem);
  const PebblingOrderGraph graph = BuildSourceOpPebblingGraph(problem, dag, {0, 1, 2}, {0, 1, 2});
  const std::vector<size_t> order =
      ComputePebblingOrder(PebblingOrderKind::DependencyConstrainedGorder, graph);
  Check("source graph records shared boundary locality",
        graph.nodes[0].locality_values == std::vector<size_t>({0}) &&
            graph.nodes[1].locality_values == std::vector<size_t>({1}) &&
            graph.nodes[2].locality_values == std::vector<size_t>({0}));
  Check("Gorder places shared boundary users together", order == std::vector<size_t>({0, 2, 1}));
  Check("boundary-local Gorder is topological", IsTopological(graph, order));
}

void TestSourceLocalityKeepsCubeRolesDistinct() {
  Problem problem;
  problem.tensors.assign(7, {8, 8});
  problem.ops = {{OpType::MatMul, {0, 1}, {4}}, {OpType::MatMul, {2, 0}, {5}}, {OpType::MatMul, {0, 3}, {6}}};
  DAG dag = DAG::build(problem);
  const PebblingOrderGraph graph = BuildSourceOpPebblingGraph(problem, dag, {0, 1, 2}, {0, 1, 2});
  Check("source locality shares equal cube roles",
        graph.nodes[0].locality_values[0] == graph.nodes[2].locality_values[0]);
  Check("source locality separates LHS and RHS representations",
        graph.nodes[0].locality_values[0] != graph.nodes[1].locality_values[1]);
}

void TestGorderImprovesItsExactWindowObjective() {
  PebblingOrderGraph graph;
  for (size_t id = 0; id < 7; ++id) {
    graph.nodes.push_back({id, id, {}, {}});
  }
  graph.nodes[0].locality_values = {77};
  graph.nodes[6].locality_values = {77};
  graph.roots = {0, 1, 2, 3, 4, 5, 6};
  const std::vector<size_t> stable = {0, 1, 2, 3, 4, 5, 6};
  const std::vector<size_t> order =
      ComputePebblingOrder(PebblingOrderKind::DependencyConstrainedGorder, graph);
  Check("Gorder pulls a scored pair into its window", order == std::vector<size_t>({0, 6, 1, 2, 3, 4, 5}));
  Check("Gorder improves the paper's exact window objective",
        GorderObjective(graph, order) > GorderObjective(graph, stable));
}

void TestGorderUsesARealSlidingWindow() {
  PebblingOrderGraph graph;
  graph.nodes = {{0, 0, {}, {99}},  {1, 1, {}, {}},  {2, 2, {1}, {}},   {3, 3, {2}, {}}, {4, 4, {3}, {}},
                 {5, 5, {4}, {55}}, {6, 6, {5}, {}}, {7, 8, {6}, {99}}, {8, 7, {}, {55}}};
  graph.roots = {7, 8};
  const std::vector<size_t> order =
      ComputePebblingOrder(PebblingOrderKind::DependencyConstrainedGorder, graph);
  Check("Gorder expires locality outside the five-node window",
        order == std::vector<size_t>({0, 1, 2, 3, 4, 5, 6, 8, 7}));
  Check("sliding-window Gorder is topological", IsTopological(graph, order));
}

void TestGorderIsDeterministicAndDeclinesMalformedGraphs() {
  PebblingOrderGraph ties;
  ties.nodes = {{9, 0, {}, {}}, {3, 0, {}, {}}, {7, 0, {}, {}}};
  ties.roots = {9, 3, 7};
  const std::vector<size_t> first =
      ComputePebblingOrder(PebblingOrderKind::DependencyConstrainedGorder, ties);
  const std::vector<size_t> second =
      ComputePebblingOrder(PebblingOrderKind::DependencyConstrainedGorder, ties);
  Check("Gorder tie-breaks by stable id", first == std::vector<size_t>({3, 7, 9}));
  Check("Gorder is deterministic", first == second);

  PebblingOrderGraph cycle;
  cycle.nodes = {{0, 0, {1}, {}}, {1, 1, {0}, {}}};
  Check("Gorder declines a cycle",
        ComputePebblingOrder(PebblingOrderKind::DependencyConstrainedGorder, cycle).empty());

  PebblingOrderGraph missing_predecessor;
  missing_predecessor.nodes = {{0, 0, {42}, {}}};
  Check("Gorder declines an unknown predecessor",
        ComputePebblingOrder(PebblingOrderKind::DependencyConstrainedGorder, missing_predecessor).empty());
}

void TestCompileTimeDefaultSelection() {
#ifdef PTO_FUSEBOX_GORDER
  Check("Gorder is selected by the compile-time option",
        kDefaultPebblingOrderKind == PebblingOrderKind::DependencyConstrainedGorder);
#else
  Check("DFS remains the compile-time default", kDefaultPebblingOrderKind == PebblingOrderKind::DfsPostOrder);
#endif
}

void TestCompileTimeSelectionDrivesSubgraphStructure() {
  Problem problem;
  problem.tensors = {{8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8}};
  problem.ops = {{OpType::Pointwise, {0}, {2}}, {OpType::Pointwise, {1}, {3}}, {OpType::Pointwise, {0}, {4}}};
  DAG dag = DAG::build(problem);
  const SubgraphStructure structure(problem, dag, {0, 1, 2});
  Check("selected production order remains structurally valid", structure.valid());
#ifdef PTO_FUSEBOX_GORDER
  Check("SubgraphStructure uses Gorder when compiled on",
        structure.execution_order() == std::vector<size_t>({0, 2, 1}));
#else
  Check("SubgraphStructure keeps DFS by default",
        structure.execution_order() == std::vector<size_t>({0, 1, 2}));
#endif
}

void TestCompileTimeSelectionDrivesCubeRequestOrder() {
  Problem problem;
  problem.tensors.assign(8, {64, 64});
  problem.ops = {{OpType::MatMul, {0, 1}, {5}}, {OpType::MatMul, {2, 3}, {6}}, {OpType::MatMul, {0, 4}, {7}}};
  problem.fast_memory_capacity = 1 << 26;
  problem.cube_capacity = 128 * 1024;
  problem.l1_capacity = 512 * 1024;
  problem.cube_freq_hz = 1.85e9;
  problem.bw_gm_l1 = 135.0;
  problem.bw_l0c_gm = 70.0;
  problem.bw_l1_l0a = 441.0;
  problem.bw_l1_l0b = 220.5;
  problem.l0_tile_m = 128;
  problem.l0_tile_n = 256;
  problem.use_hierarchical_cube_cost = true;
  problem.require_uniform_cube_dag_grid = true;
  DAG dag = DAG::build(problem);
  const auto subgraph = Ascend910BCost::create(problem, dag, {0, 1, 2});
  Check("selected cube request order builds a subgraph", subgraph.has_value());
  if (!subgraph) return;

  TileConfig config;
  config.w = 32;
  config.h = 32;
  config.k = 64;
  config.parts_m = 2;
  config.parts_n = 2;
  config.split_k = 1;
  const CubeSchedulePlan plan = subgraph->cube_schedule_plan(config, {}, {}, 1);
  Check("selected cube request order remains emit-compatible",
        plan.feasible && plan.emit_compatible && plan.matmuls.size() == 3);
  if (plan.matmuls.size() != 3) return;
#ifdef PTO_FUSEBOX_GORDER
  Check("cube requests use shared-boundary Gorder locality",
        plan.matmuls[0].op == 0 && plan.matmuls[1].op == 2 && plan.matmuls[2].op == 1);
#else
  Check("cube requests keep DFS order by default",
        plan.matmuls[0].op == 0 && plan.matmuls[1].op == 1 && plan.matmuls[2].op == 2);
#endif
}

void TestVectorBoundaryInputsJoinPebblingLifetimes() {
  Problem problem;
  problem.tensors = {{64, 64}, {64, 64}, {64, 64}, {64, 64}};
  // x is consumed before and after two produced intermediates. The emitter's
  // per-strip input cache keeps x resident across both intervening steps.
  problem.ops = {
      {OpType::Pointwise, {0}, {1}}, {OpType::Pointwise, {1}, {2}}, {OpType::Pointwise, {2, 0}, {3}}};
  SetVector910B(&problem);
  DAG dag = DAG::build(problem);
  const auto subgraph = Ascend910BCost::create(problem, dag, {0, 1, 2});
  Check("vector shared-input lifetime builds a subgraph", subgraph.has_value());
  if (!subgraph) return;

  TileConfig config;
  config.w = 64;
  config.h = 64;
  config.parts_m = 1;
  config.parts_n = 1;
  config.split_k = 1;
  const VectorStreamPlan plan = subgraph->vector_stream_plan(config, {}, {});
  Check("vector plan carries candidate-invariant input lifetimes", plan.input_lifetimes != nullptr);
  if (!plan.input_lifetimes) return;
  const auto& body = plan.input_lifetimes->phases[vector_replay_phase_index(VectorReplayPhase::Body)];
  Check("shared vector input has one canonical body lifetime",
        body.size() == 1 && body[0].tensor == 0 && body[0].first_use_step == 0 &&
            body[0].last_use_step == 2 && body[0].use_count == 2 && body[0].uses.size() == 2 &&
            body[0].spans_steps());
  Check("shared input contributes throughout the exact UB peak",
        plan.full_peak_ub_bytes == 3LL * 64 * 64 * 4);
}

void TestVectorInputsHaveSeparatePhaseLifetimes() {
  Problem problem;
  problem.tensors = {{8192, 8}, {1, 8}, {8192, 1}, {8192, 8}};
  problem.ops = {{OpType::Reduction, {0}, {1}}, {OpType::Pointwise, {0, 1, 2}, {3}}};
  problem.ops[0].vector_primitive = VectorPrimitiveFamily::RowExtrema;
  problem.ops[0].vector_geometry = VectorOpGeometry::Flat;
  SetVector910B(&problem);
  DAG dag = DAG::build(problem);
  const auto subgraph = Ascend910BCost::create(problem, dag, {0, 1});
  Check("vector two-pass lifetime builds a subgraph", subgraph.has_value());
  if (!subgraph) return;

  TileConfig config;
  config.w = 8192;
  config.h = 8;
  config.parts_m = 1;
  config.parts_n = 1;
  config.split_k = 1;
  const VectorStreamPlan plan = subgraph->vector_stream_plan(config, {}, {});
  Check("spanning reduction selects a two-pass stream",
        plan.feasible && plan.kind == VectorStreamKind::ReductionSpanning && plan.stream_passes == 2 &&
            plan.input_lifetimes != nullptr);
  if (!plan.input_lifetimes) return;
  const auto& stats = plan.input_lifetimes->phases[vector_replay_phase_index(VectorReplayPhase::Stats)];
  const auto& apply = plan.input_lifetimes->phases[vector_replay_phase_index(VectorReplayPhase::Apply)];
  Check("stats and apply own separate reload lifetimes",
        stats.size() == 1 && apply.size() == 2 && stats[0].tensor == 0 && apply[0].tensor == 0 &&
            stats[0].phase == VectorReplayPhase::Stats && apply[0].phase == VectorReplayPhase::Apply);
  Check("apply-only broadcast is not charged to stats",
        apply.size() == 2 && apply[1].tensor == 2 && apply[1].phase == VectorReplayPhase::Apply);
}

void TestVectorDiamondCombinesBoundaryAndProducedLifetimes() {
  Problem problem;
  problem.tensors.assign(7, {64, 64});
  // x and y have overlapping boundary lifetimes.  Produced value a fans out,
  // and c is both consumed inside the group and required as a function result.
  problem.ops = {
      {OpType::Pointwise, {0}, {2}},     // a = exp(x)
      {OpType::Pointwise, {2, 1}, {3}},  // b = a + y
      {OpType::Pointwise, {2, 0}, {4}},  // c = a * x (also live-out)
      {OpType::Pointwise, {3, 4}, {5}},  // d = b + c
      {OpType::Pointwise, {5, 1}, {6}},  // e = d * y
  };
  problem.required_outputs.insert(4);
  SetVector910B(&problem);
  DAG dag = DAG::build(problem);
  const auto subgraph = Ascend910BCost::create(problem, dag, {0, 1, 2, 3, 4});
  Check("vector diamond lifetime builds a subgraph", subgraph.has_value());
  if (!subgraph) return;

  TileConfig config;
  config.w = 64;
  config.h = 64;
  config.parts_m = 1;
  config.parts_n = 1;
  config.split_k = 1;
  const VectorStreamPlan plan = subgraph->vector_stream_plan(config, {}, {});
  Check("vector diamond remains one materialized replay",
        plan.feasible && plan.kind == VectorStreamKind::Materialized &&
            plan.input_lifetimes != nullptr);
  if (!plan.input_lifetimes) return;

  const auto& body =
      plan.input_lifetimes->phases[vector_replay_phase_index(VectorReplayPhase::Body)];
  const auto& order = subgraph->execution_order();
  auto step_of = [&](size_t op) {
    return static_cast<size_t>(std::find(order.begin(), order.end(), op) -
                               order.begin());
  };
  const VectorInputLifetimePlan* x = nullptr;
  const VectorInputLifetimePlan* y = nullptr;
  for (const VectorInputLifetimePlan& input : body) {
    if (input.tensor == 0) x = &input;
    if (input.tensor == 1) y = &input;
  }
  Check("vector diamond has two canonical boundary values",
        body.size() == 2 && x != nullptr && y != nullptr);
  Check("x spans its first and final fanout consumers",
        x != nullptr && x->first_use_step == std::min(step_of(0), step_of(2)) &&
            x->last_use_step == std::max(step_of(0), step_of(2)) &&
            x->use_count == 2 && x->uses.size() == 2);
  Check("y spans the interleaved middle and terminal consumers",
        y != nullptr && y->first_use_step == std::min(step_of(1), step_of(4)) &&
            y->last_use_step == std::max(step_of(1), step_of(4)) &&
            y->use_count == 2 && y->uses.size() == 2);
  Check("returned-and-consumed c remains both ephemeral and a boundary output",
        subgraph->ephemeral().count(4) == 1 &&
            subgraph->boundary_outputs().count(4) == 1);
}

void TestBoundaryValueLivesFromFirstThroughLastUse() {
  const PebblingValueLifetimePlan plan = ComputeAlwaysRetainedValueLifetimes(
      4, {{17, 0, PebblingValueEventKind::Use}, {17, 3, PebblingValueEventKind::Use}});
  Check("repeated boundary lifetime is valid", plan.valid);
  Check("repeated boundary has one canonical lifetime", plan.lifetimes.size() == 1);
  if (plan.lifetimes.size() == 1) {
    const PebblingValueLifetime& lifetime = plan.lifetimes.front();
    Check("boundary lifetime starts at first use", lifetime.first_live_step == 0);
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
  TestGorderPreservesTopologyAndSiblingLocality();
  TestGorderGroupsSharedBoundaryValues();
  TestSourceLocalityKeepsCubeRolesDistinct();
  TestGorderImprovesItsExactWindowObjective();
  TestGorderUsesARealSlidingWindow();
  TestGorderIsDeterministicAndDeclinesMalformedGraphs();
  TestCompileTimeDefaultSelection();
  TestCompileTimeSelectionDrivesSubgraphStructure();
  TestCompileTimeSelectionDrivesCubeRequestOrder();
  TestVectorBoundaryInputsJoinPebblingLifetimes();
  TestVectorInputsHaveSeparatePhaseLifetimes();
  TestVectorDiamondCombinesBoundaryAndProducedLifetimes();
  TestBoundaryValueLivesFromFirstThroughLastUse();
  TestProducedValueLivesFromMaterialization();
  TestSingleUseBoundaryStaysTransient();
  TestCallerKeepsIncompatibleRolesDistinct();
  TestInvalidLifetimeTopologiesDecline();
  std::cout << "=== pass=" << g_pass << " fail=" << g_fail << " ===\n";
  return g_fail == 0 ? 0 : 1;
}
