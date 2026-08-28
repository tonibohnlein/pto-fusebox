from __future__ import annotations

import math
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.static_mixed import (
    StaticAttentionCore,
    StaticAttentionResidual,
    StaticC2VEpilogue,
    build_examples as build_static_mixed_examples,
)
from pto_fusebox import (
    MixedGroupCandidate,
    MixedGroupSweep,
    NormalizedGraph,
    RegionSolveResult,
    can_emit_region,
    emit_pypto_region,
    enumerate_mixed_group_plans,
    export_and_normalize,
    region_for_mixed_group_candidate,
    scheduled_region,
    solve_graph,
)
from pto_fusebox.schedule.schema import (
    MixedAlgorithm,
    MixedCrossCoreProtocol,
    MixedKernelPlan,
)
from torch import nn


class V2COnly(nn.Module):
    def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.exp(value), weight)


class StreamingSoftmaxPv(nn.Module):
    def forward(self, scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.softmax(scores, dim=-1), value)


def _solver() -> Path:
    path = Path(__file__).parents[2] / "build" / "mlsys_mixed"
    if not path.is_file():
        pytest.fail(f"mixed solver binary does not exist: {path}")
    return path


def _sweep_binary() -> Path:
    path = Path(__file__).parents[2] / "build" / "mixed_group_sweep"
    if not path.is_file():
        pytest.fail(f"mixed group sweep binary does not exist: {path}")
    return path


def _solve_and_sweep(
    module: nn.Module, shapes: tuple[tuple[int, ...], ...]
) -> tuple[NormalizedGraph, RegionSolveResult, MixedKernelPlan, MixedGroupSweep]:
    graph = export_and_normalize(module, tuple(torch.zeros(shape) for shape in shapes))
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved == len(solved.regions) == 1
    region = solved.regions[0]
    plan = scheduled_region(region).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    sweep = enumerate_mixed_group_plans(region, sweep_binary=_sweep_binary())
    return graph, region, plan, sweep


def _assert_cost_identity(candidate: MixedGroupCandidate) -> None:
    breakdown = candidate.breakdown
    assert breakdown.group_overhead_cycles == 480 * candidate.groups
    ports = (
        (
            breakdown.gm_l1_bytes,
            breakdown.gm_l1_effective_parallelism,
            breakdown.gm_l1_cycles,
        ),
        (
            breakdown.gm_ub_bytes,
            breakdown.gm_ub_effective_parallelism,
            breakdown.gm_ub_cycles,
        ),
        (
            breakdown.l0c_gm_bytes,
            breakdown.l0c_gm_effective_parallelism,
            breakdown.l0c_gm_cycles,
        ),
        (
            breakdown.ub_gm_bytes,
            breakdown.ub_gm_effective_parallelism,
            breakdown.ub_gm_cycles,
        ),
    )
    for issued_bytes, effective_parallelism, cycles in ports:
        assert issued_bytes >= 0
        assert effective_parallelism >= 1
        assert (issued_bytes == 0) == (cycles == 0)
    assert breakdown.ddr_wall_cycles == max(
        breakdown.gm_l1_cycles,
        breakdown.gm_ub_cycles,
        breakdown.l0c_gm_cycles,
        breakdown.ub_gm_cycles,
    )
    assert math.isclose(
        breakdown.total_cycles,
        breakdown.pipeline_wall_cycles
        + breakdown.kernel_fill_cycles
        + breakdown.group_overhead_cycles,
        rel_tol=1e-12,
    )
    assert breakdown.pipeline_wall_cycles >= breakdown.ddr_wall_cycles
    assert [stage["topology_stage"] for stage in candidate.stages] == list(
        range(len(candidate.stages))
    )
    assert all(stage["ops"] for stage in candidate.stages)
    pipe_ids = [fifo["pipe_id"] for fifo in candidate.fifos]
    assert pipe_ids == list(range(len(candidate.fifos)))
    assert all(
        fifo["reserved_bytes"] == fifo["slot_bytes"] * fifo["slot_count"]
        for fifo in candidate.fifos
    )


def _assert_source_pipeline_identity(candidate: MixedGroupCandidate) -> None:
    breakdown = candidate.breakdown
    expected = (
        max(
            breakdown.cube_phase_cycles
            + breakdown.vector_phase_cycles / candidate.trips_per_group,
            breakdown.vector_phase_cycles
            + breakdown.cube_phase_cycles / candidate.trips_per_group,
        )
        if candidate.overlap_implementable
        else breakdown.cube_phase_cycles + breakdown.vector_phase_cycles
    )
    assert math.isclose(
        breakdown.pipeline_wall_cycles,
        expected,
        rel_tol=1e-12,
    )


def _selection_bucket(total_cycles: float, resolution_cycles: float) -> int:
    """Mirror positive ``std::llround`` used by the production selector."""

    return math.floor(total_cycles / resolution_cycles + 0.5)


@pytest.mark.parametrize(
    ("name", "shapes", "selected_groups", "selected_trips"),
    (
        ("C1", ((192, 64), (64, 256), (1, 256)), 4, 3),
        ("C2", ((384, 64), (64, 256), (1, 256)), 6, 4),
        ("H1", ((768, 64), (64, 256), (1, 256)), 8, 6),
        ("H2", ((384, 64), (64, 512), (1, 512)), 8, 6),
    ),
)
def test_calibrated_c2v_group_ranking_uses_production_breakdown(
    name: str,
    shapes: tuple[tuple[int, ...], ...],
    selected_groups: int,
    selected_trips: int,
) -> None:
    graph, region, plan, sweep = _solve_and_sweep(StaticC2VEpilogue(), shapes)

    assert plan.protocol is MixedCrossCoreProtocol.ONE_WAY
    assert sweep.selection_resolution_cycles == 16.0
    assert plan.active_groups == sweep.selected.groups == selected_groups
    assert plan.max_trips_per_group == sweep.selected.trips_per_group == selected_trips
    selected_bucket = _selection_bucket(
        sweep.selected.breakdown.total_cycles,
        sweep.selection_resolution_cycles,
    )
    candidate_buckets = {
        candidate.groups: _selection_bucket(
            candidate.breakdown.total_cycles,
            sweep.selection_resolution_cycles,
        )
        for candidate in sweep.candidates
    }
    assert selected_bucket == min(candidate_buckets.values())
    assert selected_groups == min(
        groups
        for groups, bucket in candidate_buckets.items()
        if bucket == selected_bucket
    )
    assert {candidate.groups for candidate in sweep.candidates} == {
        divisor
        for divisor in range(1, plan.spatial_tiles + 1)
        if plan.spatial_tiles % divisor == 0 and divisor <= plan.group_capacity
    }
    for candidate in sweep.candidates:
        _assert_cost_identity(candidate)
        _assert_source_pipeline_identity(candidate)
        assert candidate.groups * candidate.trips_per_group == plan.spatial_tiles
        assert candidate.pipeline_stages == (2 if candidate.trips_per_group >= 2 else 1)
        assert len(candidate.fifos) == 1
        assert candidate.fifos[0]["direction"] == "cube_to_vector"
        assert [stage["engine"] for stage in candidate.stages] == ["cube", "vector"]

    for field in ("gm_l1_bytes", "gm_ub_bytes", "l0c_gm_bytes", "ub_gm_bytes"):
        assert (
            len({getattr(candidate.breakdown, field) for candidate in sweep.candidates})
            == 1
        )
    by_groups = sorted(sweep.candidates, key=lambda candidate: candidate.groups)
    for field in (
        "gm_l1_effective_parallelism",
        "gm_ub_effective_parallelism",
        "l0c_gm_effective_parallelism",
        "ub_gm_effective_parallelism",
    ):
        values = [getattr(candidate.breakdown, field) for candidate in by_groups]
        assert values == sorted(values)

    forced = region_for_mixed_group_candidate(region, sweep.selected)
    assert can_emit_region(graph, forced)
    source = emit_pypto_region(
        graph, forced, program_name=f"mixed_{name.lower()}"
    ).source
    assert f"pl.spmd({selected_groups}," in source
    assert f"pl.pipeline({selected_trips}, stage=2" in source


@pytest.mark.parametrize(
    ("name", "shapes", "expected_grid", "selected_groups", "selected_trips"),
    (
        ("D1", ((128, 64), (64, 256), (1, 256)), (2, 4), 4, 2),
        ("D2", ((256, 64), (64, 256), (1, 256)), (4, 4), 4, 4),
        ("D3", ((256, 64), (64, 384), (1, 384)), (4, 6), 6, 4),
    ),
)
def test_c2v_descriptor_matched_group_controls(
    name: str,
    shapes: tuple[tuple[int, ...], ...],
    expected_grid: tuple[int, int],
    selected_groups: int,
    selected_trips: int,
) -> None:
    graph, region, plan, sweep = _solve_and_sweep(StaticC2VEpilogue(), shapes)

    assert plan.protocol is MixedCrossCoreProtocol.ONE_WAY
    assert (plan.m_partition.parts, plan.n_partition.parts) == expected_grid
    assert (sweep.tile.height, sweep.tile.width, sweep.tile.contraction) == (64, 64, 64)
    assert sweep.selection_resolution_cycles == 16.0
    assert sweep.selected.groups == plan.active_groups == selected_groups
    assert sweep.selected.trips_per_group == plan.max_trips_per_group == selected_trips
    assert len(plan.fifos) == 1
    fifo = plan.fifos[0]
    assert fifo.direction.value == "cube_to_vector"
    assert (fifo.valid_rows, fifo.valid_cols) == (64, 64)
    assert fifo.slot_bytes == 16384
    assert fifo.slot_count == 8
    assert fifo.reserved_bytes == 131072

    forced = region_for_mixed_group_candidate(region, sweep.selected)
    assert can_emit_region(graph, forced)
    source = emit_pypto_region(
        graph, forced, program_name=f"mixed_descriptor_{name.lower()}"
    ).source
    assert f"pl.spmd({selected_groups}," in source
    assert f"pl.pipeline({selected_trips}, stage=2" in source


def test_cvc_one_trip_candidate_is_serial_and_source_ready() -> None:
    graph, region, plan, sweep = _solve_and_sweep(
        StaticAttentionCore(), ((768, 64), (64, 64), (64, 128))
    )

    assert plan.protocol is MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE
    assert plan.m_partition.parts == 12
    assert plan.n_partition.parts == 1
    assert plan.vector_stage_peak_ub_bytes == 49280
    c2v_ring_bytes = sum(
        fifo.reserved_bytes
        for fifo in plan.fifos
        if fifo.direction.value == "cube_to_vector"
    )
    assert c2v_ring_bytes == 65536
    assert plan.vector_stage_peak_ub_bytes + c2v_ring_bytes == 114816
    one_trip = next(
        candidate for candidate in sweep.candidates if candidate.groups == 12
    )
    assert one_trip.trips_per_group == one_trip.pipeline_stages == 1
    assert not one_trip.overlap_implementable
    assert [fifo["direction"] for fifo in one_trip.fifos] == [
        "cube_to_vector",
        "vector_to_cube",
    ]
    assert [stage["engine"] for stage in one_trip.stages] == [
        "cube",
        "vector",
        "cube",
    ]
    _assert_cost_identity(one_trip)
    _assert_source_pipeline_identity(one_trip)

    forced = region_for_mixed_group_candidate(region, one_trip)
    forced_plan = scheduled_region(forced).steps[0].plan
    assert isinstance(forced_plan, MixedKernelPlan)
    assert forced_plan.pipeline_stages == 1
    assert can_emit_region(graph, forced)
    source = emit_pypto_region(graph, forced, program_name="mixed_cvc_one_trip").source
    assert "pl.range(1, init_values=" in source
    assert "stage=3" not in source


def test_sweep_candidate_is_bound_to_exact_tile_and_solution() -> None:
    _, region, _, sweep = _solve_and_sweep(
        StaticC2VEpilogue(), ((192, 64), (64, 256), (1, 256))
    )
    candidate = sweep.selected

    stale_tile = replace(
        candidate,
        tile=replace(candidate.tile, width=candidate.tile.width + 1),
    )
    with pytest.raises(ValueError, match="tile differs from the solution launch"):
        region_for_mixed_group_candidate(region, stale_tile)

    assert region.solution is not None
    stale_solution = copy.deepcopy(region.solution)
    stale_solution["steps"][0]["latency_cycles"] += 1.0
    with pytest.raises(ValueError, match="belongs to a different solver plan"):
        region_for_mixed_group_candidate(
            replace(region, solution=stale_solution), candidate
        )


@pytest.mark.parametrize(
    ("module", "shapes", "protocol", "directions", "engines"),
    (
        (
            V2COnly(),
            ((96, 64), (64, 128)),
            MixedCrossCoreProtocol.ONE_WAY,
            ("vector_to_cube",),
            ("vector", "cube"),
        ),
        (
            StaticAttentionResidual(),
            ((96, 64), (64, 64), (64, 128), (96, 128)),
            MixedCrossCoreProtocol.MULTI_ROUND_TRIP_SEQUENTIAL,
            ("cube_to_vector", "vector_to_cube", "cube_to_vector"),
            ("cube", "vector", "cube", "vector"),
        ),
        (
            StreamingSoftmaxPv(),
            ((16, 4096), (4096, 64)),
            MixedCrossCoreProtocol.ONE_WAY,
            ("vector_to_cube",),
            ("vector", "cube"),
        ),
    ),
)
def test_sweep_preserves_fifo_and_topology_controls(
    module: nn.Module,
    shapes: tuple[tuple[int, ...], ...],
    protocol: MixedCrossCoreProtocol,
    directions: tuple[str, ...],
    engines: tuple[str, ...],
) -> None:
    _, _, plan, sweep = _solve_and_sweep(module, shapes)

    assert plan.protocol is protocol
    for candidate in sweep.candidates:
        _assert_cost_identity(candidate)
        assert tuple(fifo["direction"] for fifo in candidate.fifos) == directions
        assert tuple(stage["engine"] for stage in candidate.stages) == engines
    if protocol is MixedCrossCoreProtocol.MULTI_ROUND_TRIP_SEQUENTIAL:
        assert len(sweep.candidates) == 1
        assert sweep.selected.pipeline_stages == 1
        assert not sweep.selected.overlap_implementable
    if isinstance(module, StreamingSoftmaxPv):
        assert len(sweep.candidates) == 1


def test_dense_swiglu_sweep_exposes_its_fixed_fifo_and_stage_breakdown() -> None:
    module, args = build_static_mixed_examples()["pypto_lib_static_dense_swiglu"]
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved == len(solved.regions) == 1
    region = solved.regions[0]
    plan = scheduled_region(region).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    assert plan.algorithm is MixedAlgorithm.DENSE_SWIGLU_MLP

    sweep = enumerate_mixed_group_plans(region, sweep_binary=_sweep_binary())
    assert len(sweep.candidates) == 1
    candidate = sweep.selected
    _assert_cost_identity(candidate)
    assert candidate.groups == plan.active_groups
    assert candidate.pipeline_stages == plan.pipeline_stages
    assert tuple(fifo["direction"] for fifo in candidate.fifos) == (
        "cube_to_vector",
        "cube_to_vector",
        "vector_to_cube",
    )
    assert tuple(stage["engine"] for stage in candidate.stages) == (
        "cube",
        "cube",
        "vector",
        "cube",
    )
