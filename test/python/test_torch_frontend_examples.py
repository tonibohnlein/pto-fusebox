from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.basic import build_examples as build_basic_examples
from examples.torch_frontend.deepseek_v4 import (
    build_examples as build_deepseek_examples,
)
from examples.torch_frontend.qwen3 import build_examples as build_qwen_examples
from examples.torch_frontend.static_mixed import (
    build_examples as build_static_mixed_examples,
)
from examples.torch_frontend.pr2335_vector import (
    build_examples as build_pr2335_examples,
)
from pto_fusebox import (
    can_emit_region,
    emit_pypto_region,
    export_and_normalize,
    extract_solver_regions,
    solve_graph,
)
from torch import nn

Example = tuple[nn.Module, tuple[torch.Tensor, ...]]


def _all_examples() -> dict[str, Example]:
    return {
        **build_basic_examples(),
        **build_deepseek_examples(),
        **build_qwen_examples(),
        **build_pr2335_examples(),
        **build_static_mixed_examples(),
    }


def _test_solver() -> Path:
    configured = os.environ.get("PTO_FUSEBOX_TEST_SOLVER")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "build" / "mlsys_mixed"


@pytest.mark.parametrize(
    ("name", "expected_kinds"),
    [
        ("softmax", ["max", "sub", "exp", "sum", "div"]),
        ("matmul", ["matmul"]),
        (
            "attention_core",
            ["transpose_view", "matmul", "max", "sub", "exp", "sum", "div", "matmul"],
        ),
    ],
)
def test_basic_examples_export_as_expected(
    name: str, expected_kinds: list[str]
) -> None:
    module, args = build_basic_examples()[name]
    graph = export_and_normalize(module, args)

    assert [op.kind for op in graph.ops] == expected_kinds
    assert len(extract_solver_regions(graph)) == 1


@pytest.mark.parametrize(
    ("builder", "name", "expected_kinds"),
    [
        (
            build_deepseek_examples,
            "deepseek_v4_rmsnorm",
            ["cast", "mul", "sum", "mul", "add", "rsqrt", "mul", "mul", "cast"],
        ),
        (
            build_deepseek_examples,
            "deepseek_v4_mtp_projection",
            [
                "cast",
                "mul",
                "sum",
                "mul",
                "add",
                "rsqrt",
                "mul",
                "mul",
                "transpose_view",
                "matmul",
                "view",
                "mul",
                "sum",
                "mul",
                "add",
                "rsqrt",
                "mul",
                "mul",
                "transpose_view",
                "matmul",
                "add",
            ],
        ),
        (
            build_qwen_examples,
            "qwen3_rms_lm_head",
            [
                "cast",
                "mul",
                "sum",
                "mul",
                "add",
                "rsqrt",
                "mul",
                "mul",
                "transpose_view",
                "matmul",
            ],
        ),
    ],
)
def test_model_examples_form_one_supported_region(
    builder, name: str, expected_kinds: list[str]
) -> None:
    module, args = builder()[name]
    graph = export_and_normalize(module, args)

    assert [op.kind for op in graph.ops] == expected_kinds
    assert all(op.supported for op in graph.ops)
    regions = extract_solver_regions(graph)
    assert len(regions) == 1
    assert regions[0].op_ids == tuple(op.id for op in graph.ops)


@pytest.mark.parametrize(
    ("name", "expected_kinds"),
    [
        (
            "pypto_lib_static_attention",
            ["transpose_view", "matmul", "max", "sub", "exp", "sum", "div", "matmul"],
        ),
        (
            "pypto_lib_static_dense_swiglu",
            [
                "matmul",
                "matmul",
                "neg",
                "exp",
                "add",
                "div",
                "mul",
                "mul",
                "cast",
                "matmul",
            ],
        ),
        (
            "pypto_lib_static_attention_residual",
            [
                "transpose_view",
                "matmul",
                "max",
                "sub",
                "exp",
                "sum",
                "div",
                "matmul",
                "add",
            ],
        ),
    ],
)
def test_static_mixed_examples_export_one_coherent_dag(
    name: str, expected_kinds: list[str]
) -> None:
    module, args = build_static_mixed_examples()[name]
    graph = export_and_normalize(module, args)

    assert [op.kind for op in graph.ops] == expected_kinds
    assert all(op.supported for op in graph.ops)
    assert len(extract_solver_regions(graph)) == 1


@pytest.mark.parametrize("name", sorted(_all_examples()))
def test_example_matmuls_are_semantically_coherent_and_cube_sized(name: str) -> None:
    module, args = _all_examples()[name]
    graph = export_and_normalize(module, args)
    values = graph.value_map()

    for op in graph.ops:
        if op.kind != "matmul":
            continue
        lhs = values[op.inputs[0]]
        rhs = values[op.inputs[1]]
        output = values[op.outputs[0]]
        lhs_shape = tuple(
            dimension for dimension in lhs.shape if isinstance(dimension, int)
        )
        rhs_shape = tuple(
            dimension for dimension in rhs.shape if isinstance(dimension, int)
        )
        output_shape = tuple(
            dimension for dimension in output.shape if isinstance(dimension, int)
        )
        assert len(lhs_shape) == len(lhs.shape)
        assert len(rhs_shape) == len(rhs.shape)
        assert len(output_shape) == len(output.shape)
        m = math.prod(lhs_shape[:-1])
        lhs_k = lhs_shape[-1]
        rhs_k, n = rhs_shape
        output_m = math.prod(output_shape[:-1])
        output_n = output_shape[-1]
        assert (lhs_k, output_m, output_n) == (rhs_k, m, n)
        assert min(m, n, lhs_k) >= 16


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_all_examples_solve_as_complete_supported_regions() -> None:
    solver = _test_solver()
    for name, (module, args) in _all_examples().items():
        graph = export_and_normalize(module, args)
        result = solve_graph(graph, solver_binary=solver, solver_workers=2)
        assert result.successful, {
            "example": name,
            "statuses": [region.status for region in result.regions],
            "diagnostics": [region.diagnostics for region in result.regions],
        }


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_attention_solver_selects_complete_cube_vector_cube_group() -> None:
    module, args = build_basic_examples()["attention_core"]
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.successful
    assert result.regions_solved
    assert result.whole_graph_codegen_ready
    region = result.regions[0]
    assert can_emit_region(graph, region)
    assert region.solution is not None
    assert region.solution["steps"][0]["ops"] == list(
        range(len(region.solver_op_to_graph))
    )
    schedule = region.solution["steps"][0]["plan"]
    assert schedule["source_codegen_ready"] is True
    vector_stages = [
        stage for stage in schedule["stages"] if stage["engine"] == "vector"
    ]
    assert len(vector_stages) == 1
    assert vector_stages[0]["vector_stream"]["kind"] == "materialized"


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
@pytest.mark.parametrize(
    ("name", "pipe_count"),
    [
        ("pypto_lib_static_attention", 2),
        ("pypto_lib_static_dense_swiglu", 3),
        ("pypto_lib_static_attention_residual", 3),
    ],
)
def test_static_mixed_examples_solve_and_emit_generic_pypto_source(
    name: str, pipe_count: int
) -> None:
    module, args = build_static_mixed_examples()[name]
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.successful
    assert result.whole_graph_codegen_ready
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.problem is not None
    # Source planning refines the analytic result instead of perturbing it: all
    # three natural winners are already source-ready, so no constrained retry
    # should replace their selected schedules.
    assert region.problem["require_source_codegen"] is False
    assert can_emit_region(graph, region)
    source = emit_pypto_region(graph, region, program_name=name).source
    assert source.count("pl.cross_core_pipe(") == pipe_count
    assert "auto_fuse" not in source and "auto_tile" not in source


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_analytic_success_is_distinct_from_source_codegen_readiness() -> None:
    module, args = build_deepseek_examples()["deepseek_v4_mtp_projection"]
    graph = export_and_normalize(module, args)
    result = solve_graph(graph, solver_binary=_test_solver(), solver_workers=2)

    assert result.regions_solved
    assert result.whole_graph_supported
    assert not result.whole_graph_codegen_ready
    assert any(not can_emit_region(graph, region) for region in result.regions)


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_vector_to_cube_pipeline_is_source_ready() -> None:
    class VectorToCube(nn.Module):
        def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.exp(value), weight)

    graph = export_and_normalize(
        VectorToCube(),
        (torch.zeros(96, 64), torch.zeros(64, 128)),
    )
    assert [op.kind for op in graph.ops] == ["exp", "matmul"]

    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert result.successful
    region = result.regions[0]
    assert region.solution is not None
    assert region.solution["steps"][0]["ops"] == [0, 1]
    schedule = region.solution["steps"][0]["plan"]
    assert schedule["source_codegen_ready"] is True
    assert schedule["split_k"] == 1
    assert schedule["work_units"] == schedule["spatial_tiles"]
    assert schedule["pipeline_extent"] == schedule["spatial_tiles"]
    assert [stage["engine"] for stage in schedule["stages"]] == ["vector", "cube"]
    assert (
        schedule["stages"][0]["valid_rows"] * schedule["vector_lanes"]
        == schedule["fifos"][0]["valid_rows"]
    )
    assert schedule["stages"][0]["valid_cols"] == 64
    assert schedule["stages"][0]["vector_stream"]["kind"] == "pointwise"
    assert schedule["mode"] == "one_way"
    assert schedule["transfers"] == [
        {
            "tensor": 1,
            "producer_stage": 0,
            "consumer_stage": 1,
            "producer_engine": "vector",
            "consumer_engine": "cube",
        }
    ]
    assert len(schedule["fifos"]) == 1
    assert schedule["fifos"][0]["direction"] == "vector_to_cube"
    assert schedule["fifos"][0]["spatial_m"] is True
    assert schedule["fifos"][0]["spatial_n"] is False
    assert result.whole_graph_codegen_ready
    assert can_emit_region(graph, region)


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_vector_to_cube_rhs_pipeline_has_source_ready_k_by_n_geometry() -> None:
    class VectorToCubeRhs(nn.Module):
        def forward(
            self, lhs: torch.Tensor, value: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            return torch.mm(lhs, torch.exp(value + bias))

    graph = export_and_normalize(
        VectorToCubeRhs(),
        (
            torch.zeros(96, 64),
            torch.zeros(64, 128),
            torch.zeros(1, 128),
        ),
    )
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.successful
    region = result.regions[0]
    assert region.solution is not None
    assert region.solution["steps"][0]["ops"] == [0, 1, 2]
    schedule = region.solution["steps"][0]["plan"]
    assert schedule["source_codegen_ready"] is True
    assert schedule["split_k"] == 1
    assert [stage["engine"] for stage in schedule["stages"]] == ["vector", "cube"]
    vector_stage = schedule["stages"][0]
    fifo = schedule["fifos"][0]
    assert vector_stage["valid_rows"] * schedule["vector_lanes"] == 64
    assert vector_stage["valid_cols"] == schedule["n_partition"]["big"]
    assert fifo["valid_rows"] == 64
    assert fifo["valid_cols"] == schedule["n_partition"]["big"]
    assert fifo["direction"] == "vector_to_cube"
    assert fifo["spatial_m"] is False
    assert fifo["spatial_n"] is True
    assert result.whole_graph_codegen_ready
    assert can_emit_region(graph, region)


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_softmax_to_pv_serializes_the_complete_flash_stream() -> None:
    class SoftmaxPv(nn.Module):
        def forward(self, scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.softmax(scores, dim=-1), value)

    graph = export_and_normalize(
        SoftmaxPv(),
        (torch.zeros(16, 4096), torch.zeros(4096, 64)),
    )
    result = solve_graph(graph, solver_binary=_test_solver(), solver_workers=2)

    assert result.regions_solved
    assert not result.whole_graph_codegen_ready
    region = result.regions[0]
    assert not can_emit_region(graph, region)
    assert region.solution is not None
    assert region.solution["steps"][0]["ops"] == list(range(6))
    schedule = region.solution["steps"][0]["plan"]
    vector_stage = schedule["stages"][0]
    stream = vector_stage["vector_stream"]
    assert stream["kind"] == "softmax_flash"
    assert stream["extent"] == 4096
    assert stream["chunk"] < stream["extent"]
    assert stream["full_chunks"] > 1
    assert stream["tail"] > 0
    phases = {phase["name"]: phase for phase in stream["phases"]}
    assert phases["stats"]["loop"]["trip_count"] > 0
    assert phases["apply"]["loop"]["trip_count"] > 0


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_multi_role_vector_to_cube_is_not_source_codegen_ready() -> None:
    class MultiRole(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            produced = torch.exp(value)
            return torch.mm(produced, produced)

    graph = export_and_normalize(MultiRole(), (torch.zeros(64, 64),))
    result = solve_graph(graph, solver_binary=_test_solver(), solver_workers=2)

    assert result.regions_solved
    assert not result.whole_graph_codegen_ready
    assert not can_emit_region(graph, result.regions[0])


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_internal_transpose_boundary_is_not_whole_graph_codegen_ready() -> None:
    class InternalTranspose(nn.Module):
        def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.exp(value).t(), weight)

    graph = export_and_normalize(
        InternalTranspose(),
        (torch.zeros(8, 16), torch.zeros(8, 24)),
    )
    result = solve_graph(graph, solver_binary=_test_solver(), solver_workers=2)

    assert not result.whole_graph_supported
    assert not result.whole_graph_codegen_ready


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_shape_changing_cube_to_vector_uses_the_crossing_frame() -> None:
    class MatmulReduce(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.sum(torch.mm(lhs, rhs), dim=-1, keepdim=True)

    graph = export_and_normalize(
        MatmulReduce(),
        (torch.zeros(96, 64), torch.zeros(64, 128)),
    )
    result = solve_graph(graph, solver_binary=_test_solver(), solver_workers=2)

    assert result.regions_solved
    assert not result.whole_graph_codegen_ready
    region = result.regions[0]
    assert not can_emit_region(graph, region)
    assert region.solution is not None
    assert region.solution["steps"][0]["ops"] == [0, 1]
    schedule = region.solution["steps"][0]["plan"]
    fifo = schedule["fifos"][0]
    vector_stage = schedule["stages"][1]
    assert fifo["direction"] == "cube_to_vector"
    assert fifo["valid_cols"] == 128
    assert vector_stage["valid_cols"] == 128
    assert vector_stage["vector_stream"]["tile"][1] == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
