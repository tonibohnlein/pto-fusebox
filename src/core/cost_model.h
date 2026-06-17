#pragma once

#include "core/dag.h"
#include "core/flat_set.h"
#include "core/types.h"   // Problem, DAG, TileConfig, CostResult
#include <concepts>
#include <optional>
#include <vector>

// ============================================================================
// CostModel — the compile-time interface a per-architecture cost model must
// satisfy. A concrete model (Ascend910BCost today; a future Ascend950Cost, …)
// owns everything hardware-specific: admissibility (which ops may share a
// subgraph), the searched tiling knobs, on-chip feasibility, and the roofline
// cost. It does NOT decide partition-level constraints (acyclicity,
// ephemeral-gap, coverage, connectivity) — those are architecture-independent
// and live in the partition layer over SubgraphStructure + DAG.
//
// Architecture selection is a COMPILE-TIME choice: the pipeline names the
// active model through the `Subgraph` alias (see subgraph.h). Swapping that
// alias retargets the whole solver; the hot search loop is monomorphised on the
// concrete model, with no virtual dispatch.
// ============================================================================

// The structural classification (boundary/ephemeral tensors, sinks, the DFS
// pebbling order) is shared and architecture-independent — see
// SubgraphStructure (subgraph_structure.h), which every cost model composes.

template <class CM>
concept CostModel = requires(const Problem& prob, const DAG& dag,
                             std::vector<size_t> ops, const CM& cm) {
  // Factory: build the cost model for an op set, or nullopt if those ops cannot
  // form a valid subgraph on this architecture (homogeneity, same-shape sinks,
  // tiling-representability). Structural validity is folded in.
  { CM::create(prob, dag, ops) } -> std::same_as<std::optional<CM>>;

  // Structural surface the partition/solution layers consume (delegated by the
  // model to its composed SubgraphStructure).
  { cm.ops() }              -> std::convertible_to<const std::vector<size_t>&>;
  { cm.boundary_inputs() }  -> std::convertible_to<const FlatSet<size_t>&>;
  { cm.boundary_outputs() } -> std::convertible_to<const FlatSet<size_t>&>;
  { cm.ephemeral() }        -> std::convertible_to<const FlatSet<size_t>&>;
  { cm.execution_order() }  -> std::convertible_to<const std::vector<size_t>&>;

  // Feasibility folded into cost: best tiling + latency (infeasible ⇒ infinite
  // latency). Pure function of (subgraph, machine params) — safe to memoise.
  { cm.best_cost() } -> std::same_as<CostResult>;
};

// ============================================================================
// SubgraphT<CM> — the per-architecture subgraph handle. Because the cost model
// composes SubgraphStructure and exposes the full subgraph interface, the
// handle IS the cost model itself, constrained to model CostModel. The pipeline
// uses the `Subgraph` alias defined in subgraph.h; selecting a different CM
// there switches the entire solver to another architecture at compile time.
//
// (When a second backend lands and runtime backend dispatch is wanted, the
// pipeline classes can be promoted to explicit `<CM>` templates — this alias is
// a strict prefix of that, so no rework is thrown away.)
// ============================================================================
template <CostModel CM>
using SubgraphT = CM;
