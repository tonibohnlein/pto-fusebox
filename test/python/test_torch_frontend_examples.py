from __future__ import annotations

import pytest
from examples.torch_frontend.basic import build_examples as build_basic_examples
from examples.torch_frontend.deepseek_v4 import build_examples as build_deepseek_examples
from examples.torch_frontend.qwen3 import build_examples as build_qwen_examples
from pto_fusebox import export_and_normalize, extract_solver_regions


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
