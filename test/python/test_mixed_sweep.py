from __future__ import annotations

import copy
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.static_mixed import (
    StaticAttentionCore,
    StaticAttentionResidual,
    StaticC2VEpilogue,
    StaticDenseSwiGlu,
    StaticFp32DeepLinearBlend,
    StaticFp32DenseSwiGlu,
    StaticFp32FeatureBlend,
    build_examples as build_static_mixed_examples,
)
from pto_fusebox import (
    MIXED_GROUP_SWEEP_AVAILABILITY_SCHEMA,
    MixedGroupCandidate,
    MixedGroupSweep,
    MixedGroupSweepUnavailable,
    NormalizedGraph,
    RegionSolveResult,
    ScheduleContractError,
    can_emit_region,
    emit_pypto_region,
    enumerate_mixed_group_plans,
    export_and_normalize,
    region_for_mixed_group_candidate,
    mixed_group_sweep_availability,
    scheduled_region,
    solve_graph,
)
from pto_fusebox.schedule.schema import (
    MixedAlgorithm,
    MixedCrossCoreProtocol,
    MixedKernelPlan,
    MixedTransferDirection,
)
from torch import nn


class V2COnly(nn.Module):
    def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.exp(value), weight)


class V2COnlyRhs(nn.Module):
    def forward(
        self, lhs: torch.Tensor, value: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.mm(lhs, torch.exp(value + bias))


class StreamingSoftmaxPv(nn.Module):
    def forward(self, scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.softmax(scores, dim=-1), value)


class GenericFeatureBlend(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        first_weight: torch.Tensor,
        second_weight: torch.Tensor,
        sink_weight: torch.Tensor,
    ) -> torch.Tensor:
        first = torch.mm(value, first_weight, out_dtype=torch.float32)
        second = torch.mm(value, second_weight, out_dtype=torch.float32)
        blended = (first + second).to(torch.bfloat16)
        return torch.mm(blended, sink_weight, out_dtype=torch.float32)


class GenericFeatureBlendLinearSink(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        first_weight: torch.Tensor,
        second_weight: torch.Tensor,
        sink_weight: torch.Tensor,
    ) -> torch.Tensor:
        blended = torch.mm(value, first_weight, out_dtype=torch.float32) + torch.mm(
            value, second_weight, out_dtype=torch.float32
        )
        return torch.nn.functional.linear(blended, sink_weight)


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


def test_source_costing_changes_attention_plan_before_emission() -> None:
    graph = export_and_normalize(
        StaticAttentionCore(),
        (torch.zeros(96, 64), torch.zeros(64, 64), torch.zeros(64, 128)),
    )
    analytic = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=False,
    ).regions[0]
    source = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    ).regions[0]

    assert analytic.problem is not None and source.problem is not None
    assert analytic.problem["require_source_codegen"] is False
    assert source.problem["require_source_codegen"] is True
    analytic_plan = scheduled_region(analytic).steps[0].plan
    source_plan = scheduled_region(source).steps[0].plan
    assert isinstance(analytic_plan, MixedKernelPlan)
    assert isinstance(source_plan, MixedKernelPlan)
    assert (
        analytic_plan.m_partition.parts,
        analytic_plan.n_partition.parts,
        analytic_plan.active_groups,
        analytic_plan.max_trips_per_group,
    ) == (1, 4, 2, 2)
    assert (
        source_plan.m_partition.parts,
        source_plan.n_partition.parts,
        source_plan.active_groups,
        source_plan.max_trips_per_group,
    ) == (3, 1, 3, 1)
    assert analytic_plan.cube_stage_peak_l0a_bytes == 73_728
    assert analytic_plan.cube_stage_peak_l0b_bytes == 49_152
    assert not analytic_plan.source_codegen_ready
    assert not can_emit_region(graph, analytic)
    assert source_plan.cube_stage_peak_l0a_bytes == 16_384
    assert source_plan.cube_stage_peak_l0b_bytes == 65_536
    assert can_emit_region(graph, source)


def test_attention_source_plan_traffic_matches_emitted_topology() -> None:
    graph, region, plan, sweep = _solve_and_sweep(
        StaticAttentionCore(), ((96, 64), (64, 64), (64, 128))
    )

    assert (plan.m_partition.parts, plan.n_partition.parts) == (3, 1)
    assert (sweep.tile.height, sweep.tile.width, sweep.tile.contraction) == (
        32,
        128,
        64,
    )
    regions = plan.spatial_tiles
    query_bytes = regions * 32 * 64 * 4
    key_bytes = regions * 64 * 64 * 4
    value_bytes = regions * 64 * 128 * 4
    crossing_bytes = regions * 32 * 64 * 4
    output_bytes = 96 * 128 * 4
    assert (
        sweep.selected.breakdown.gm_l1_bytes,
        sweep.selected.breakdown.gm_ub_bytes,
        sweep.selected.breakdown.l0c_gm_bytes,
        sweep.selected.breakdown.ub_gm_bytes,
    ) == (
        query_bytes + key_bytes + value_bytes + crossing_bytes,
        crossing_bytes,
        crossing_bytes + output_bytes,
        crossing_bytes,
    )
    source = emit_pypto_region(graph, region, program_name="attention_source").source
    assert "pl.spmd(3," in source
    assert "pl.range(1, init_values=" in source
    assert source.count("pl.tensor.slice(arg_query, [32, 64]") == 1
    assert source.count("pl.tensor.slice(arg_key, [64, 64]") == 1
    assert source.count("pl.tensor.slice(arg_value, [64, 128]") == 1
    assert source.count("pl.tensor.matmul(") == 2


def test_v2c_rhs_prices_broadcast_bias_load_on_both_aiv_lanes() -> None:
    _, _, plan, sweep = _solve_and_sweep(V2COnlyRhs(), ((96, 64), (64, 128), (1, 128)))

    assert plan.vector_lanes == 2
    assert plan.vector_split.value == "rows"
    expected_value_bytes = 64 * 128 * 4
    expected_bias_bytes = plan.vector_lanes * 1 * 128 * 4
    assert sweep.selected.breakdown.gm_ub_bytes == (
        expected_value_bytes + expected_bias_bytes
    )


def test_feature_round_trip_prices_physical_fp32_c2v_messages() -> None:
    module = StaticDenseSwiGlu(hidden_size=64, intermediate_size=128).eval()
    graph = export_and_normalize(
        module,
        (torch.zeros(128, 64, dtype=torch.bfloat16),),
    )
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
    assert plan.algorithm is MixedAlgorithm.FEATURE_CHUNK_ROUND_TRIP
    assert plan.feature_round_trip is not None
    sweep = enumerate_mixed_group_plans(region, sweep_binary=_sweep_binary())

    c2v_fifos = tuple(
        fifo
        for fifo in plan.fifos
        if fifo.direction is MixedTransferDirection.CUBE_TO_VECTOR
    )
    assert len(c2v_fifos) == 2
    assert all(fifo.wire_dtype == "fp32" for fifo in c2v_fifos)
    assert all(fifo.slot_count == 4 for fifo in plan.fifos)
    crossing_bytes = (
        sum(fifo.slot_bytes for fifo in c2v_fifos)
        * plan.spatial_tiles
        * plan.feature_round_trip.intermediate_chunks
    )
    output = graph.value_map()[graph.outputs[0]]
    assert all(isinstance(dimension, int) for dimension in output.shape)
    output_bytes = (
        math.prod(dimension for dimension in output.shape if isinstance(dimension, int))
        * 4
    )
    assert sweep.selected.breakdown.gm_ub_bytes == crossing_bytes
    assert sweep.selected.breakdown.l0c_gm_bytes == crossing_bytes + output_bytes


def test_deep_feature_round_trip_prices_whole_program_l1_residency() -> None:
    """All task-local Mat allocation families must fit one source program."""

    module = StaticFp32DeepLinearBlend(160).eval()
    graph = export_and_normalize(
        module,
        (
            torch.zeros(256, 160),
            torch.zeros(160, 320),
            torch.zeros(160, 320),
            torch.zeros(320, 160),
        ),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
        collect_candidate_summaries=True,
    )
    repeated = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
        collect_candidate_summaries=True,
    )

    region = solved.regions[0]
    assert region.solution is not None
    assert tuple(step["kind"] for step in region.solution["steps"]) == (
        "mixed",
        "cube",
    )
    assert tuple(tuple(step["ops"]) for step in region.solution["steps"]) == (
        (0, 1, 2, 3),
        (4,),
    )
    assert region.candidate_summaries
    assert tuple(
        (candidate.modeled_cost_cycles, candidate.partition, candidate.schedule)
        for candidate in region.candidate_summaries
    ) == tuple(
        (candidate.modeled_cost_cycles, candidate.partition, candidate.schedule)
        for candidate in repeated.regions[0].candidate_summaries
    )
    assert all(
        sum(
            int(step["plan"].get("source_l1_allocation_bytes", 0))
            for step in candidate.schedule
        )
        <= 524_288
        for candidate in region.candidate_summaries
    )
    selected_l1_by_step = tuple(
        int(step["plan"].get("source_l1_allocation_bytes", 0))
        for step in region.solution["steps"]
    )
    assert selected_l1_by_step == (235_520, 14_336)
    assert sum(selected_l1_by_step) == 249_856
    assert ((0, 1, 2, 3), (4,)) in {
        candidate.partition for candidate in region.candidate_summaries
    }


def test_mixed_cost_breakdown_rejects_stale_or_impossible_evidence() -> None:
    _, region, _, _ = _solve_and_sweep(
        StaticC2VEpilogue(), ((192, 64), (64, 256), (1, 256))
    )
    assert region.solution is not None

    stale_groups = copy.deepcopy(region.solution)
    stale_groups["steps"][0]["plan"]["cost_breakdown"]["active_groups"] += 1
    with pytest.raises(
        ScheduleContractError,
        match="cost_breakdown.active_groups differs from its mixed plan",
    ):
        scheduled_region(replace(region, solution=stale_groups))

    negative_bytes = copy.deepcopy(region.solution)
    negative_bytes["steps"][0]["plan"]["cost_breakdown"]["traffic_bytes"]["gm_l1"] = -1
    with pytest.raises(
        ScheduleContractError,
        match=r"cost_breakdown.traffic_bytes.gm_l1 must be nonnegative",
    ):
        scheduled_region(replace(region, solution=negative_bytes))

    stale_total = copy.deepcopy(region.solution)
    stale_total["steps"][0]["plan"]["cost_breakdown"]["total_cycles"] += 1
    with pytest.raises(
        ScheduleContractError,
        match=r"cost_breakdown.total_cycles differs from",
    ):
        scheduled_region(replace(region, solution=stale_total))

    impossible_pipeline_wall = copy.deepcopy(region.solution)
    breakdown = impossible_pipeline_wall["steps"][0]["plan"]["cost_breakdown"]
    assert breakdown["ddr_wall_cycles"] > 0
    breakdown["pipeline_wall_cycles"] = breakdown["ddr_wall_cycles"] - 1
    with pytest.raises(
        ScheduleContractError,
        match=r"cost_breakdown.pipeline_wall_cycles is below its DDR wall",
    ):
        scheduled_region(replace(region, solution=impossible_pipeline_wall))


@pytest.mark.parametrize(
    "variant",
    ("swiglu", "blend", "linear_sink"),
)
@pytest.mark.parametrize(
    "shape",
    ((64, 96, 192), (128, 128, 256), (256, 160, 320)),
)
def test_feature_round_trip_is_selected_as_one_maximal_static_region(
    shape: tuple[int, int, int],
    variant: str,
) -> None:
    rows, hidden_size, intermediate_size = shape
    if variant == "swiglu":
        module: nn.Module = StaticDenseSwiGlu(
            hidden_size=hidden_size, intermediate_size=intermediate_size
        ).eval()
        args = (torch.zeros(rows, hidden_size, dtype=torch.bfloat16),)
    elif variant == "blend":
        module = GenericFeatureBlend()
        args = (
            torch.zeros(rows, hidden_size, dtype=torch.bfloat16),
            torch.zeros(hidden_size, intermediate_size, dtype=torch.bfloat16),
            torch.zeros(hidden_size, intermediate_size, dtype=torch.bfloat16),
            torch.zeros(intermediate_size, hidden_size, dtype=torch.bfloat16),
        )
    else:
        module = GenericFeatureBlendLinearSink()
        args = (
            torch.zeros(rows, hidden_size, dtype=torch.bfloat16),
            torch.zeros(hidden_size, intermediate_size, dtype=torch.bfloat16),
            torch.zeros(hidden_size, intermediate_size, dtype=torch.bfloat16),
            torch.zeros(hidden_size, intermediate_size),
        )
    graph = export_and_normalize(
        module,
        args,
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
        collect_candidate_summaries=True,
    )

    assert solved.regions_solved == len(solved.regions) == 1
    region = solved.regions[0]
    assert region.region.op_ids == tuple(op.id for op in graph.ops)
    assert region.solution is not None
    assert [step["kind"] for step in region.solution["steps"]] == ["mixed"]
    assert [step["ops"] for step in region.solution["steps"]] == [
        list(range(len(region.solver_op_to_graph)))
    ]
    plan = scheduled_region(region).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    assert plan.algorithm is MixedAlgorithm.FEATURE_CHUNK_ROUND_TRIP
    assert mixed_group_sweep_availability(region).available
    assert (
        enumerate_mixed_group_plans(
            region, sweep_binary=_sweep_binary()
        ).selected.groups
        == plan.active_groups
    )


@pytest.mark.parametrize(
    ("module_factory", "expected_steps"),
    (
        (StaticFp32DenseSwiGlu, ("mixed",)),
        (StaticFp32FeatureBlend, ("mixed",)),
        (StaticFp32DeepLinearBlend, ("mixed", "cube")),
    ),
)
@pytest.mark.parametrize(
    "shape",
    ((64, 96, 192), (128, 128, 256), (256, 160, 320)),
)
def test_broader_fp32_feature_round_trip_graphs_are_source_planned(
    module_factory: type[nn.Module],
    expected_steps: tuple[str, ...],
    shape: tuple[int, int, int],
) -> None:
    rows, hidden_size, intermediate_size = shape
    module = (
        module_factory(hidden_size)
        if module_factory is StaticFp32DeepLinearBlend
        else module_factory()
    )
    args = (
        torch.zeros(rows, hidden_size),
        torch.zeros(hidden_size, intermediate_size),
        torch.zeros(hidden_size, intermediate_size),
        torch.zeros(intermediate_size, hidden_size),
    )
    graph = export_and_normalize(module.eval(), args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
        collect_candidate_summaries=True,
    )

    assert solved.regions_solved
    assert len(solved.regions) == 1
    region = solved.regions[0]
    assert region.region.op_ids == tuple(op.id for op in graph.ops)
    assert region.solution is not None
    selected_steps = expected_steps
    if module_factory is StaticFp32DeepLinearBlend and shape == (256, 160, 320):
        selected_steps = ("mixed", "cube")
    assert tuple(step["kind"] for step in region.solution["steps"]) == selected_steps
    assert region.candidate_summaries
    assert region.candidate_summaries[0].selected
    assert region.candidate_summaries[0].source_ready
    assert all(
        candidate.source_ready or candidate.rejection_reason
        for candidate in region.candidate_summaries
    )
    plans = tuple(step.plan for step in scheduled_region(region).steps)
    mixed_plans = tuple(plan for plan in plans if isinstance(plan, MixedKernelPlan))
    assert len(mixed_plans) == 1
    assert mixed_plans[0].algorithm is MixedAlgorithm.FEATURE_CHUNK_ROUND_TRIP


@pytest.mark.parametrize(
    (
        "shapes",
        "selected_step_kinds",
        "selected_partition",
    ),
    (
        (
            ((160, 64), (2048, 64), (2048, 64)),
            ("cube", "mixed"),
            ((0,), tuple(range(1, 7))),
        ),
        (
            ((320, 128), (2048, 128), (2048, 128)),
            ("cube", "vector", "cube"),
            ((0,), (1, 2, 3, 4, 5), (6,)),
        ),
    ),
)
def test_cvc_sweep_reports_when_source_solver_selects_a_gm_cut(
    shapes: tuple[tuple[int, ...], ...],
    selected_step_kinds: tuple[str, ...],
    selected_partition: tuple[tuple[int, ...], ...],
) -> None:
    graph = export_and_normalize(
        StaticAttentionCore(), tuple(torch.zeros(shape) for shape in shapes)
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved == len(solved.regions) == 1
    region = solved.regions[0]

    availability = mixed_group_sweep_availability(region)
    assert not availability.available
    assert availability.code == "selected_solution_is_not_one_mixed_region"
    assert availability.selected_step_kinds == selected_step_kinds
    assert availability.selected_partition == selected_partition

    probed = mixed_group_sweep_availability(region, sweep_binary=_sweep_binary())
    assert not probed.available
    assert probed.code == "mixed_cube_execution_unrepresentable"
    assert probed.closest_tile is None
    assert probed.rejection_counts is not None
    assert probed.rejection_counts["mixed_cube_execution_unrepresentable"] > 0
    with pytest.raises(
        MixedGroupSweepUnavailable,
        match="mixed_cube_execution_unrepresentable",
    ):
        enumerate_mixed_group_plans(region)


def test_non_capacity_sweep_rejection_has_no_fabricated_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structural rejection carries no zero-sized capacity evidence."""

    graph = export_and_normalize(
        StaticAttentionCore(),
        (
            torch.zeros(160, 64),
            torch.zeros(2048, 64),
            torch.zeros(2048, 64),
        ),
    )
    region = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    ).regions[0]

    def unavailable_process(args, **kwargs):
        Path(args[2]).write_text(
            json.dumps(
                {
                    "schema_version": MIXED_GROUP_SWEEP_AVAILABILITY_SCHEMA,
                    "available": False,
                    "code": "whole_region_has_no_feasible_mixed_candidate",
                    "reason": "the complete op set has no feasible mixed candidate",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", unavailable_process)
    availability = mixed_group_sweep_availability(region, sweep_binary=_sweep_binary())

    assert not availability.available
    assert availability.code == "whole_region_has_no_feasible_mixed_candidate"
    assert availability.closest_tile is None
    assert availability.required_vec_bytes is None
    assert availability.available_vec_bytes is None
    assert availability.required_l1_bytes is None
    assert availability.available_l1_bytes is None
    assert availability.required_l0a_bytes is None
    assert availability.available_l0a_bytes is None
    assert availability.required_l0b_bytes is None
    assert availability.available_l0b_bytes is None


@pytest.mark.parametrize(
    "shapes",
    (
        ((96, 32), (32, 32), (32, 64)),
        ((128, 48), (48, 48), (48, 48)),
        ((192, 32), (64, 32), (64, 32)),
        ((256, 48), (64, 48), (64, 64)),
        ((384, 32), (32, 32), (32, 64)),
        ((512, 32), (64, 32), (64, 64)),
        ((768, 48), (48, 48), (48, 48)),
    ),
)
def test_cvc_realization_corpus_has_rankable_group_candidates(
    shapes: tuple[tuple[int, ...], ...],
) -> None:
    _, _, plan, sweep = _solve_and_sweep(StaticAttentionCore(), shapes)

    assert plan.protocol is MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE
    assert len(sweep.candidates) >= 3
    assert all(
        candidate.groups * candidate.trips_per_group == plan.spatial_tiles
        for candidate in sweep.candidates
    )
    assert all(
        candidate.cube_stage_peak_l0a_bytes <= 65_536
        and candidate.cube_stage_peak_l0b_bytes <= 65_536
        for candidate in sweep.candidates
    )


@pytest.mark.parametrize(
    "shapes",
    (
        pytest.param(((128, 32), (48, 32), (48, 48)), id="short_rectangular"),
        pytest.param(((192, 48), (64, 48), (64, 64)), id="medium_wider_k"),
        pytest.param(((256, 32), (32, 32), (32, 64)), id="long_thin"),
        pytest.param(((384, 48), (48, 48), (48, 48)), id="tall_wider_k"),
        pytest.param(((512, 32), (64, 32), (64, 32)), id="tall_square"),
        pytest.param(((640, 48), (64, 48), (64, 64)), id="deep_wider_k"),
        pytest.param(((768, 32), (48, 32), (48, 48)), id="deep_rectangular"),
    ),
)
def test_cvc_blind_holdout_shapes_have_rankable_source_candidates(
    shapes: tuple[tuple[int, ...], ...],
) -> None:
    """Freeze unseen CVC geometries before using their silicon timings."""

    graph, region, plan, sweep = _solve_and_sweep(StaticAttentionCore(), shapes)

    assert plan.protocol is MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE
    assert len(sweep.candidates) >= 3
    assert all(candidate.breakdown.total_cycles > 0 for candidate in sweep.candidates)
    assert all(
        candidate.groups * candidate.trips_per_group == plan.spatial_tiles
        for candidate in sweep.candidates
    )
    for candidate in sweep.candidates:
        assert candidate.cube_stage_peak_l0a_bytes <= 65_536
        assert candidate.cube_stage_peak_l0b_bytes <= 65_536
        forced = region_for_mixed_group_candidate(region, candidate)
        assert can_emit_region(graph, forced)


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
        StaticAttentionCore(), ((384, 32), (32, 32), (32, 64))
    )

    assert plan.protocol is MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE
    assert plan.m_partition.parts == 12
    assert plan.n_partition.parts == 1
    assert plan.vector_stage_peak_ub_bytes == 41088
    c2v_ring_bytes = sum(
        fifo.reserved_bytes
        for fifo in plan.fifos
        if fifo.direction.value == "cube_to_vector"
    )
    assert c2v_ring_bytes == 16384
    assert plan.vector_stage_peak_ub_bytes + c2v_ring_bytes == 57472
    one_trip = next(
        candidate for candidate in sweep.candidates if candidate.groups == 12
    )
    assert one_trip.trips_per_group == one_trip.pipeline_stages == 1
    assert not one_trip.overlap_implementable
    assert [fifo["slot_count"] for fifo in one_trip.fifos] == [1, 1]
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
    forced_breakdown = forced_plan.cost_breakdown
    assert forced_breakdown.active_groups == one_trip.groups
    assert forced_breakdown.trips_per_group == one_trip.trips_per_group
    assert forced_breakdown.pipeline_stages == one_trip.pipeline_stages
    assert forced_breakdown.overlap_implementable is one_trip.overlap_implementable
    assert forced_breakdown.total_cycles == one_trip.breakdown.total_cycles
    assert dict(forced_breakdown.traffic_bytes) == {
        "gm_l1": one_trip.breakdown.gm_l1_bytes,
        "gm_ub": one_trip.breakdown.gm_ub_bytes,
        "l0c_gm": one_trip.breakdown.l0c_gm_bytes,
        "ub_gm": one_trip.breakdown.ub_gm_bytes,
    }
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
            ((16, 512), (512, 64)),
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
    assert plan.algorithm is MixedAlgorithm.FEATURE_CHUNK_ROUND_TRIP

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
