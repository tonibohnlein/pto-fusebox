from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.basic import build_examples as build_basic_examples
from examples.torch_frontend.deepseek_v4 import build_examples as build_deepseek_examples
from examples.torch_frontend.qwen3 import build_examples as build_qwen_examples
from pto_fusebox import export_and_normalize, extract_solver_regions, solve_graph
from torch import nn

Example = tuple[nn.Module, tuple[torch.Tensor, ...]]


def _all_examples() -> dict[str, Example]:
    return {
        **build_basic_examples(),
        **build_deepseek_examples(),
        **build_qwen_examples(),
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
def test_basic_examples_export_as_expected(name: str, expected_kinds: list[str]) -> None:
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
            ["cast", "mul", "sum", "mul", "add", "rsqrt", "mul", "mul", "transpose_view", "matmul"],
        ),
    ],
)
def test_model_examples_form_one_supported_region(builder, name: str, expected_kinds: list[str]) -> None:
    module, args = builder()[name]
    graph = export_and_normalize(module, args)

    assert [op.kind for op in graph.ops] == expected_kinds
    assert all(op.supported for op in graph.ops)
    regions = extract_solver_regions(graph)
    assert len(regions) == 1
    assert regions[0].op_ids == tuple(op.id for op in graph.ops)


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
        m = math.prod(lhs.shape[:-1])
        lhs_k = lhs.shape[-1]
        rhs_k, n = rhs.shape
        output_m = math.prod(output.shape[:-1])
        output_n = output.shape[-1]
        assert (lhs_k, output_m, output_n) == (rhs_k, m, n)
        assert min(m, n, lhs_k) >= 16


@pytest.mark.skipif(not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
