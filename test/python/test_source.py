from __future__ import annotations

import ast
import copy
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.basic import build_examples
from pto_fusebox import (
    KernelKind,
    ScheduleContractError,
    SourceEmissionError,
    can_emit_region,
    emit_pypto_region,
    export_and_normalize,
    scheduled_region,
    solve_graph,
)
from pto_fusebox.schedule.schema import (
    CubeKernelPlan,
    VectorKernelPlan,
    VectorReplayPhase,
)
from torch import nn


def _solver() -> Path:
    configured = os.environ.get("PTO_FUSEBOX_TEST_SOLVER")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "build" / "mlsys_mixed"


@lru_cache(maxsize=None)
def _solved(name: str):
    module, args = build_examples()[name]
    graph = export_and_normalize(module, args)
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    assert solved.regions_solved
    assert len(solved.regions) == 1
    return graph, solved.regions[0]


def _solve_module(module: nn.Module, args: tuple[torch.Tensor, ...]):
    graph = export_and_normalize(module, args)
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    assert solved.regions_solved == 1
    assert len(solved.regions) == 1
    return graph, solved.regions[0]


pytestmark = pytest.mark.skipif(
    not _solver().is_file(), reason="built solver unavailable"
)


def test_vector_solution_is_a_complete_typed_emission_contract() -> None:
    graph, result = _solved("softmax")
    schedule = scheduled_region(result)
    assert can_emit_region(graph, result)

    assert schedule.tensor_values == result.solver_tensor_to_value
    assert len(schedule.steps) == 1
    step = schedule.steps[0]
    assert step.kind is KernelKind.VECTOR
    assert step.op_order == (0, 1, 2, 3, 4)
    assert isinstance(step.plan, VectorKernelPlan)
    body = step.plan.phase(VectorReplayPhase.BODY)
    assert body.input_lifetimes[0].use_count == 2
    assert body.ops == (0, 1, 2, 3, 4)
    assert step.plan.physical_frame.align_rows
    assert step.plan.physical_frame.element_granule == 8
    assert step.plan.physical_frame.iteration_cols == 1024
    assert step.plan.physical_frame.iteration_rows == 128
    assert step.plan.physical_frame.reduced_axis == 1

    source = emit_pypto_region(graph, result, program_name="softmax_fused").source
    ast.parse(source)
    assert "for region_index in pl.parallel(12):" in source
    assert "for strip_index in pl.pipeline(2, stage=2):" in source
    assert "region_rows = 10 +" in source
    assert "pl.load(value," in source
    assert "[8, 1024], valid_shape=[valid_rows, valid_cols]" in source
    assert source.count("pl.load(") == 1
    assert source.count("pl.store(") == 1
    assert "pl.row_max" in source
    assert "pl.row_sum" in source
    assert "pl.row_expand_div" in source
    assert "auto_fuse" not in source and "auto_tile" not in source
    assert (
        source == emit_pypto_region(graph, result, program_name="softmax_fused").source
    )


def test_cube_solution_emits_exact_spatial_and_k_window_schedule() -> None:
    graph, result = _solved("matmul")
    schedule = scheduled_region(result)
    assert can_emit_region(graph, result)

    step = schedule.steps[0]
    assert step.kind is KernelKind.CUBE
    assert isinstance(step.plan, CubeKernelPlan)
    assert step.plan.m_partition.big == 32
    assert step.plan.m_partition.small == 32
    assert step.plan.m_partition.num_big == 0
    assert step.plan.m_partition.parts == 4
    assert step.plan.n_partition.big == 64
    assert step.plan.n_partition.small == 64
    assert step.plan.n_partition.num_big == 0
    assert step.plan.n_partition.parts == 3
    assert step.plan.execution_order == (0,)
    assert step.plan.resident_boundaries == ()

    source = emit_pypto_region(graph, result, program_name="matmul_fused").source
    ast.parse(source)
    assert "for region_index in pl.parallel(12):" in source
    assert "region_row = m_index * 32" in source
    assert "region_col = n_index * 64" in source
    assert "for k_window in pl.pipeline(1, 3, stage=2):" in source
    assert source.count("pl.tile.matmul(") == 1
    assert source.count("pl.tile.matmul_acc(") == 2
    assert "[region_row, 240], [32, 16]" in source
    assert "[240, region_col], [16, 64]" in source
    assert source.count("pl.store(") == 1
    assert "auto_fuse" not in source and "auto_tile" not in source


def test_vector_emission_is_generic_over_pointwise_operation_order() -> None:
    class PointwiseChain(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.maximum(torch.exp(lhs * 0.5) + rhs, rhs)

    graph, result = _solve_module(
        PointwiseChain(), (torch.zeros(96, 320), torch.ones(96, 320))
    )
    source = emit_pypto_region(graph, result, program_name="pointwise_chain").source

    assert [op.kind for op in graph.ops] == ["mul", "exp", "add", "maximum"]
    assert source.index("pl.mul") < source.index("pl.exp")
    assert source.index("pl.exp") < source.index("pl.add")
    assert source.index("pl.add") < source.index("pl.maximum")
    assert source.count("pl.load(") == 2
    assert source.count("pl.store(") == 1


def test_vector_reduction_uses_pinned_iteration_frame_not_output_extent() -> None:
    class SumOfSquares(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.sum(value * value, dim=-1, keepdim=True)

    graph, result = _solve_module(SumOfSquares(), (torch.ones(128, 1024),))
    source = emit_pypto_region(graph, result, program_name="sum_of_squares").source

    assert [op.kind for op in graph.ops] == ["mul", "sum"]
    assert "valid_cols = pl.max(pl.min(1024 - strip_col, 1024), 0)" in source
    assert "pl.row_sum" in source
    assert source.count("pl.load(") == 1
    assert source.count("pl.store(") == 1


def test_cube_emission_is_generic_over_shape_and_k_tail() -> None:
    class MatmulWithTail(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs)

    graph, result = _solve_module(
        MatmulWithTail(), (torch.zeros(64, 272), torch.zeros(272, 80))
    )
    source = emit_pypto_region(graph, result, program_name="matmul_with_tail").source

    assert "[region_row, 240], [16, 32]" in source
    assert "[240, region_col], [32, 80]" in source
    assert "pl.tile.matmul_acc" in source
    assert source.count("pl.store(") == 1


def test_source_backend_rejects_unimplemented_cube_residency() -> None:
    graph, result = _solved("matmul")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    matmul = solution["steps"][0]["plan"]["matmuls"][0]
    lhs_region = copy.deepcopy(matmul["lhs"])
    matmul["lhs_resident_boundary"] = 0
    solution["steps"][0]["plan"]["resident_boundaries"] = [
        {
            "id": 0,
            "region": lhs_region,
            "role": "lhs",
            "first_use": 0,
            "last_use": 0,
            "use_count": 1,
            "bytes": lhs_region["height"] * lhs_region["width"] * 4,
        }
    ]

    with pytest.raises(SourceEmissionError, match="resident-boundary"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_source_backend_rejects_cube_execution_order_drift() -> None:
    graph, result = _solved("matmul")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["execution_order"] = []

    with pytest.raises(SourceEmissionError, match="execution order"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_schedule_contract_rejects_an_incomplete_operation_order() -> None:
    _, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["op_order"] = solution["steps"][0]["op_order"][:-1]

    with pytest.raises(ScheduleContractError, match="not a permutation"):
        scheduled_region(replace(result, solution=solution))


def test_schedule_contract_rejects_legacy_or_dropped_step_fields() -> None:
    _, result = _solved("softmax")
    assert result.solution is not None

    legacy = copy.deepcopy(result.solution)
    legacy["schema_version"] = "pto_fusebox.solution.v1"
    with pytest.raises(ScheduleContractError, match="solution schema"):
        scheduled_region(replace(result, solution=legacy))

    for field in ("sequential_tiles", "retain"):
        incomplete = copy.deepcopy(result.solution)
        incomplete["steps"][0].pop(field)
        with pytest.raises(ScheduleContractError, match=f"omits fields.*{field}"):
            scheduled_region(replace(result, solution=incomplete))


def test_schedule_contract_rejects_vector_phase_order_or_frame_drift() -> None:
    _, result = _solved("softmax")
    assert result.solution is not None

    reordered = copy.deepcopy(result.solution)
    body = reordered["steps"][0]["plan"]["phases"][0]
    body["ops"][0], body["ops"][1] = body["ops"][1], body["ops"][0]
    with pytest.raises(ScheduleContractError, match="preserve.*operation order"):
        scheduled_region(replace(result, solution=reordered))

    stale_frame = copy.deepcopy(result.solution)
    stale_frame["steps"][0]["plan"]["phases"][0]["tensor_frames"].pop()
    with pytest.raises(ScheduleContractError, match="tensor_frames do not cover"):
        scheduled_region(replace(result, solution=stale_frame))

    stale_workspace = copy.deepcopy(result.solution)
    workspace = stale_workspace["steps"][0]["plan"]["phases"][0]["workspaces"][0]
    workspace["source_tensor"] += 1
    with pytest.raises(ScheduleContractError, match="wrong source tensor"):
        scheduled_region(replace(result, solution=stale_workspace))


def test_schedule_contract_accepts_sparse_cube_resident_identity() -> None:
    _, result = _solved("matmul")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    matmul = solution["steps"][0]["plan"]["matmuls"][0]
    lhs_region = copy.deepcopy(matmul["lhs"])
    matmul["lhs_resident_boundary"] = 0
    solution["steps"][0]["plan"]["resident_boundaries"] = [
        {
            "id": 19,
            "region": lhs_region,
            "role": "lhs",
            "first_use": 0,
            "last_use": 0,
            "use_count": 1,
            "bytes": lhs_region["height"] * lhs_region["width"] * 4,
        }
    ]

    schedule = scheduled_region(replace(result, solution=solution))
    plan = schedule.steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert plan.resident_boundaries[0].id == 19
    assert plan.matmuls[0].lhs_resident_boundary == 0


def test_schedule_contract_rejects_cube_k_or_residency_drift() -> None:
    _, result = _solved("matmul")
    assert result.solution is not None

    stale_k = copy.deepcopy(result.solution)
    stale_k["steps"][0]["plan"]["matmuls"][0]["k_loop"]["tail"] += 1
    with pytest.raises(ScheduleContractError, match="k_loop does not cover K"):
        scheduled_region(replace(result, solution=stale_k))

    stale_variant = copy.deepcopy(result.solution)
    stale_variant["steps"][0]["plan"]["matmuls"][0]["output_variants"][0]["count"] += 1
    with pytest.raises(ScheduleContractError, match="output_variants"):
        scheduled_region(replace(result, solution=stale_variant))

    shortened_k = copy.deepcopy(result.solution)
    matmul = shortened_k["steps"][0]["plan"]["matmuls"][0]
    matmul["contraction"] = 240
    matmul["effective_contraction"] = 240
    matmul["k_loop"]["tail"] = 0
    for variant in matmul["output_variants"]:
        variant["l0_tail"] = None
    with pytest.raises(ScheduleContractError, match="contraction differs"):
        scheduled_region(replace(result, solution=shortened_k))

    stale_region = copy.deepcopy(result.solution)
    stale_region["steps"][0]["plan"]["matmuls"][0]["lhs"]["height"] += 1
    with pytest.raises(ScheduleContractError, match="axis bindings"):
        scheduled_region(replace(result, solution=stale_region))


def test_schedule_contract_rejects_common_sequential_drift() -> None:
    _, vector = _solved("softmax")
    assert vector.solution is not None
    stale_vector = copy.deepcopy(vector.solution)
    stale_vector["steps"][0]["sequential_tiles"][0] = 1
    with pytest.raises(ScheduleContractError, match="nonzero sequential tiles"):
        scheduled_region(replace(vector, solution=stale_vector))

    _, cube = _solved("matmul")
    assert cube.solution is not None
    stale_sequential = copy.deepcopy(cube.solution)
    stale_sequential["steps"][0]["sequential_tiles"][0] -= 16
    with pytest.raises(ScheduleContractError, match="common sequential tile"):
        scheduled_region(replace(cube, solution=stale_sequential))


def test_schedule_contract_rejects_unknown_p4_recipe_version() -> None:
    class Softmax(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.softmax(value, dim=-1)

    _, result = _solve_module(Softmax(), (torch.ones(16, 4096),))
    assert result.solution is not None
    assert result.solution["steps"][0]["plan"]["kind"] == "softmax_flash"
    assert result.solution["steps"][0]["plan"]["p4_recipe"]["apply_substitutions"] == [
        {"op": 0, "value": "running_max"},
        {"op": 3, "value": "running_sum"},
    ]
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["p4_recipe"]["version"] = "softmax_flash.v2"

    with pytest.raises(ScheduleContractError, match="version is unsupported"):
        scheduled_region(replace(result, solution=solution))


def test_schedule_contract_rejects_p4_semantic_role_drift() -> None:
    class Softmax(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.softmax(value, dim=-1)

    _, result = _solve_module(Softmax(), (torch.ones(16, 4096),))
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    bindings = solution["steps"][0]["plan"]["p4_recipe"]["apply_substitutions"]
    bindings[0]["value"], bindings[1]["value"] = (
        bindings[1]["value"],
        bindings[0]["value"],
    )

    with pytest.raises(ScheduleContractError, match="versioned state contract"):
        scheduled_region(replace(result, solution=solution))


def test_materialized_schedule_rejects_an_inactive_phase_replay() -> None:
    _, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["phases"][1]["ops"] = [0]

    with pytest.raises(ScheduleContractError, match="must be empty"):
        scheduled_region(replace(result, solution=solution))


def test_source_backend_rejects_vector_lifetime_drift() -> None:
    graph, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["phases"][0]["input_lifetimes"] = []

    with pytest.raises(SourceEmissionError, match="input lifetimes"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_source_backend_rejects_partition_that_does_not_cover_output() -> None:
    graph, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["m_partition"]["big"] += 1

    with pytest.raises(SourceEmissionError, match="covers .* output axis"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_source_backend_rejects_non_last_axis_reduction() -> None:
    graph, result = _solved("softmax")
    first = replace(graph.ops[0], attributes={"axis": 0, "keepdim": True})
    invalid_graph = replace(graph, ops=(first, *graph.ops[1:]))

    with pytest.raises(SourceEmissionError, match="last-axis keepdim"):
        emit_pypto_region(invalid_graph, result)


def test_source_readiness_rejects_multi_output_region() -> None:
    class TwoOutputs(nn.Module):
        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            first = torch.exp(value)
            return first, first + 1.0

    graph, result = _solve_module(TwoOutputs(), (torch.ones(64, 128),))

    assert not can_emit_region(graph, result)
    with pytest.raises(SourceEmissionError, match="exactly one region output"):
        emit_pypto_region(graph, result)


def test_source_readiness_rejects_transposed_matmul() -> None:
    class TransposedMatmul(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs.t())

    graph, result = _solve_module(
        TransposedMatmul(),
        (torch.ones(64, 96), torch.ones(128, 96)),
    )

    assert not can_emit_region(graph, result)
    with pytest.raises(SourceEmissionError, match="non-transposed matmul"):
        emit_pypto_region(graph, result)


def test_source_backend_rejects_mixed_plan_without_replanning_it() -> None:
    graph, result = _solved("attention_core")
    with pytest.raises(SourceEmissionError, match="mixed PyPTO source emission"):
        emit_pypto_region(graph, result)
