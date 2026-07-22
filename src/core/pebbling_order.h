#pragma once

#include <cstddef>
#include <vector>

#include "core/dag.h"
#include "core/types.h"

// A candidate-invariant graph view for choosing one deterministic pebbling
// order. Node ids are stable ids owned by the caller (source-op ids or
// role-expanded cube-request ids). Keeping the ordering algorithm independent
// of the graph owner lets vector/source DAGs and cube request DAGs share the
// same policies without duplicating their semantic graph representation.
struct PebblingOrderNode {
  size_t id = 0;
  size_t stable_position = 0;
  std::vector<size_t> predecessors;
  // Non-schedulable requested values whose proximity improves locality. Two
  // nodes carrying the same id receive the same score contribution as two
  // siblings with one common predecessor in the original Gorder objective.
  // The caller owns identity and must keep incompatible representations (for
  // example LHS and RHS views of A in A@A) distinct.
  std::vector<size_t> locality_values;
};

struct PebblingOrderGraph {
  std::vector<PebblingOrderNode> nodes;
  std::vector<size_t> roots;
};

enum class PebblingOrderKind {
  DfsPostOrder,
  // Gorder's sliding-window locality objective, restricted to topologically
  // ready nodes so the result remains an executable producer-before-consumer
  // schedule. This is an experimental policy; DFS remains the default until
  // model and device evaluation justify changing it.
  DependencyConstrainedGorder,
};

#ifdef PTO_FUSEBOX_GORDER
inline constexpr PebblingOrderKind kDefaultPebblingOrderKind = PebblingOrderKind::DependencyConstrainedGorder;
#else
inline constexpr PebblingOrderKind kDefaultPebblingOrderKind = PebblingOrderKind::DfsPostOrder;
#endif
inline constexpr size_t kDependencyConstrainedGorderWindow = 5;

class PebblingOrderStrategy {
 public:
  virtual ~PebblingOrderStrategy() = default;

  [[nodiscard]] virtual std::vector<size_t> Compute(const PebblingOrderGraph& graph) const = 0;
};

// Build the source-operation view used by SubgraphStructure and by cost-model
// paths whose execution roots have already been refined. Producers outside
// `ops` are role-aware boundary locality values rather than predecessor nodes.
[[nodiscard]] PebblingOrderGraph BuildSourceOpPebblingGraph(const Problem& prob, const DAG& dag,
                                                            const std::vector<size_t>& ops,
                                                            const std::vector<size_t>& roots);

[[nodiscard]] const PebblingOrderStrategy& GetPebblingOrderStrategy(PebblingOrderKind kind);

[[nodiscard]] std::vector<size_t> ComputePebblingOrder(PebblingOrderKind kind,
                                                       const PebblingOrderGraph& graph);

// Candidate-invariant lifetime topology for canonical requested values. The
// caller owns value identity: equal ids mean the same source allocation,
// requested region, dtype, on-chip pool, and compatible representation. This
// deliberately keeps incompatible roles (for example the LHS and RHS views of
// A in A@A) separate even when they originate from the same source tensor.
enum class PebblingValueEventKind {
  Materialize,
  Use,
};

struct PebblingValueEvent {
  size_t value_id = 0;
  size_t step = 0;
  PebblingValueEventKind kind = PebblingValueEventKind::Use;
};

struct PebblingValueLifetime {
  size_t value_id = 0;
  size_t first_live_step = 0;
  size_t last_use_step = 0;
  size_t use_count = 0;
  bool has_materialization = false;

  // A single-use boundary value is a transient operand at its only consumer.
  // A produced value or repeated boundary value spanning steps is resident.
  [[nodiscard]] bool spans_steps() const {
    return first_live_step < last_use_step;
  }
};

struct PebblingValueLifetimePlan {
  bool valid = false;
  std::vector<PebblingValueLifetime> lifetimes;
};

// Build the always-retain lifetime plan used by both vector and cube schedule
// models. A produced value becomes live at its one materialization; a boundary
// value becomes live at its first use. Both remain live through their final
// use. Reload/rematerialization alternatives are intentionally outside this
// initial policy.
[[nodiscard]] PebblingValueLifetimePlan ComputeAlwaysRetainedValueLifetimes(
    size_t step_count, const std::vector<PebblingValueEvent>& events);
