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
    emit_pypto_region,
    export_and_normalize,
    scheduled_region,
    solve_graph,
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

    assert schedule.tensor_values == result.solver_tensor_to_value
    assert len(schedule.steps) == 1
    step = schedule.steps[0]
    assert step.kind is KernelKind.VECTOR
    assert step.op_order == (0, 1, 2, 3, 4)
    assert step.plan["input_lifetimes"]["body"][0]["use_count"] == 2
    assert step.plan["physical_frame"] == {
        "align_rows": True,
        "element_granule": 8,
        "iteration_cols": 1024,
        "iteration_rows": 128,
        "reduced_axis": 1,
    }

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

    step = schedule.steps[0]
    assert step.kind is KernelKind.CUBE
    assert step.plan["m_partition"] == {
        "big": 32,
        "num_big": 0,
        "parts": 4,
        "small": 32,
    }
    assert step.plan["n_partition"] == {
        "big": 64,
        "num_big": 0,
        "parts": 3,
        "small": 64,
    }
    assert step.plan["execution_order"] == [0]
    assert step.plan["resident_boundaries"] == []

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
    solution["cube_schedule"][0]["resident_boundaries"] = [
        {
            "id": 0,
            "region": {},
            "role": "lhs",
            "first_use": 0,
            "last_use": 0,
            "use_count": 1,
            "bytes": 1,
        }
    ]

    with pytest.raises(SourceEmissionError, match="resident-boundary"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_source_backend_rejects_cube_execution_order_drift() -> None:
    graph, result = _solved("matmul")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["cube_schedule"][0]["execution_order"] = []

    with pytest.raises(SourceEmissionError, match="execution order"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_schedule_contract_rejects_an_incomplete_operation_order() -> None:
    _, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["op_order"][0] = solution["op_order"][0][:-1]

    with pytest.raises(ScheduleContractError, match="not a permutation"):
        scheduled_region(replace(result, solution=solution))


def test_source_backend_rejects_vector_lifetime_drift() -> None:
    graph, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["vector_stream"][0]["input_lifetimes"]["body"] = []

    with pytest.raises(SourceEmissionError, match="input lifetimes"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_source_backend_rejects_partition_that_does_not_cover_output() -> None:
    graph, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["vector_stream"][0]["m_partition"]["big"] += 1

    with pytest.raises(SourceEmissionError, match="covers .* output axis"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_source_backend_rejects_non_last_axis_reduction() -> None:
    graph, result = _solved("softmax")
    first = replace(graph.ops[0], attributes={"axis": 0, "keepdim": True})
    invalid_graph = replace(graph, ops=(first, *graph.ops[1:]))

    with pytest.raises(SourceEmissionError, match="last-axis keepdim"):
        emit_pypto_region(invalid_graph, result)


def test_source_backend_rejects_mixed_plan_without_replanning_it() -> None:
    graph, result = _solved("attention_core")
    with pytest.raises(SourceEmissionError, match="mixed PyPTO source emission"):
        emit_pypto_region(graph, result)
