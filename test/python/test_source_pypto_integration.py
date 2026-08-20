"""Opt-in end-to-end checks against an independently installed PyPTO/PTOAS.

The standalone source backend intentionally does not depend on PyPTO.  Set
``PTO_FUSEBOX_PYPTO_INTEGRATION=1`` in an environment whose ``pypto`` import and
``PTOAS_ROOT`` point at the versions under validation to exercise this gate.
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.pr2335_vector import (
    build_examples as build_pr2335_examples,
)
from pto_fusebox import (
    emit_pypto_region,
    export_and_normalize,
    scheduled_region,
    solve_graph,
)
from pto_fusebox.schedule.schema import CubeKernelPlan, VectorKernelPlan
from torch import nn


pytestmark = pytest.mark.skipif(
    os.environ.get("PTO_FUSEBOX_PYPTO_INTEGRATION") != "1",
    reason="set PTO_FUSEBOX_PYPTO_INTEGRATION=1 with PyPTO and PTOAS configured",
)


class _PointwiseChain(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.exp(lhs * 0.5) + rhs, rhs)


class _SumOfSquares(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sum(value * value, dim=-1, keepdim=True)


class _MatmulWithTail(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.mm(lhs, rhs)


class _ChainedMatmul(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(torch.mm(lhs, middle), rhs)


class _DiamondMatmul(nn.Module):
    def forward(
        self,
        shared: torch.Tensor,
        lhs_weight: torch.Tensor,
        rhs_weight: torch.Tensor,
    ) -> torch.Tensor:
        lhs = torch.mm(shared, lhs_weight)
        rhs = torch.mm(shared, rhs_weight)
        return torch.mm(lhs, rhs)


class _FanoutMatmul(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        first_rhs: torch.Tensor,
        second_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.mm(lhs, middle)
        return torch.mm(shared, first_rhs), torch.mm(shared, second_rhs)


class _WideSoftmax(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.softmax(value, dim=-1)


class _Silu(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.reciprocal(torch.exp(value * -1.0) + 1.0)


class _LayerNorm(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        centered = value - value.mean(dim=-1, keepdim=True)
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(variance + 1.0e-5)
        return normalized * gamma + beta


class _StreamedReduction(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sum(value, dim=-1, keepdim=True)


class _StreamedNormalize(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value / torch.sum(value, dim=-1, keepdim=True)


class _NamingCollision(nn.Module):
    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        return torch.exp(pl)


def _solver() -> Path:
    configured = os.environ.get("PTO_FUSEBOX_TEST_SOLVER")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "build" / "mlsys_mixed"


def _assert_static_vector_frames(pto: str) -> None:
    assert re.search(r"partition_tensor_view<[^>]*\?", pto) is None
    assert re.search(r"valid_(?:row|col) = %arg[0-9]+", pto) is None


def _assert_single_spmd_orchestration(source: str, work_units: int) -> None:
    submits = re.findall(r"\brt_submit_ai[cv]_task\(", source)
    assert len(submits) == 1
    assert source.count("launch_spec.set_block_num(") == 1
    assert f"launch_spec.set_block_num({work_units});" in source
    assert "region_index" not in source


def _compile_source(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    skip_ptoas: bool,
) -> tuple[str, int]:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")

    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(module, args)
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    assert solved.regions_solved == 1
    assert len(solved.regions) == 1
    plan = scheduled_region(solved.regions[0]).steps[0].plan
    assert isinstance(plan, (CubeKernelPlan, VectorKernelPlan))

    source = emit_pypto_region(graph, solved.regions[0], program_name=name).source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / name),
        dump_passes=False,
        skip_ptoas=skip_ptoas,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    assert len(pto_files) == 1
    generated_cpp = list((compiled.output_dir / "ptoas").glob("*.cpp"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(orchestration_files) == 1
    _assert_single_spmd_orchestration(
        orchestration_files[0].read_text(encoding="utf-8"),
        plan.work_units,
    )
    return pto_files[0].read_text(encoding="utf-8"), len(generated_cpp)


@pytest.mark.parametrize(
    ("name", "module", "args", "expected_pto_op", "static_vector_frames"),
    [
        (
            "pointwise_chain",
            _PointwiseChain(),
            (torch.zeros(96, 320), torch.ones(96, 320)),
            "pto.texp",
            True,
        ),
        (
            "sum_of_squares",
            _SumOfSquares(),
            (torch.ones(128, 1024),),
            "pto.trowsum",
            True,
        ),
        (
            "matmul_with_tail",
            _MatmulWithTail(),
            (torch.zeros(64, 272), torch.zeros(272, 80)),
            "pto.tmatmul.acc",
            False,
        ),
        (
            "chained_matmul",
            _ChainedMatmul(),
            (
                torch.zeros(64, 128, dtype=torch.bfloat16),
                torch.zeros(128, 96, dtype=torch.bfloat16),
                torch.zeros(96, 80, dtype=torch.bfloat16),
            ),
            "pto.tinsert",
            False,
        ),
        (
            "wide_softmax",
            _WideSoftmax(),
            (torch.zeros(32, 8192),),
            "pto.trowmax",
            True,
        ),
        (
            "silu",
            _Silu(),
            (torch.zeros(512, 256),),
            "pto.trecip",
            True,
        ),
        (
            "layer_norm",
            _LayerNorm(),
            (
                torch.zeros(512, 256),
                torch.ones(1, 256),
                torch.zeros(1, 256),
            ),
            "pto.trowsum",
            True,
        ),
        (
            "naming_collision",
            _NamingCollision(),
            (torch.zeros(64, 128),),
            "pto.texp",
            True,
        ),
        (
            "streamed_reduction",
            _StreamedReduction(),
            (torch.ones(5, 32771),),
            "pto.trowsum",
            True,
        ),
        (
            "streamed_normalize",
            _StreamedNormalize(),
            (torch.ones(5, 32771),),
            "pto.trowsum",
            True,
        ),
    ],
)
def test_generated_source_compiles_through_pypto_and_ptoas(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    expected_pto_op: str,
    static_vector_frames: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pto, generated_cpp = _compile_source(
        name,
        module,
        args,
        tmp_path,
        monkeypatch,
        skip_ptoas=False,
    )
    assert generated_cpp == 1
    assert expected_pto_op in pto
    if static_vector_frames:
        _assert_static_vector_frames(pto)


@pytest.mark.parametrize(
    "name",
    [
        "pr2335_softmax_512x256",
        "pr2335_softmax_256x512",
        "pr2335_softmax_128x1024",
        "pr2335_softmax_32x8192",
        "pr2335_rms_norm",
        "pr2335_layer_norm",
        "pr2335_silu",
    ],
)
def test_pr2335_vector_surface_lowers_static_frames(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = build_pr2335_examples()[name]
    pto, _ = _compile_source(
        name,
        module,
        args,
        tmp_path,
        monkeypatch,
        skip_ptoas=True,
    )
    _assert_static_vector_frames(pto)


def test_cut_fp32_chain_compiles_as_two_dependency_linked_spmd_kernels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    module = _ChainedMatmul()
    args = (
        torch.zeros(64, 128),
        torch.zeros(128, 96),
        torch.zeros(96, 80),
    )
    graph = export_and_normalize(module, args)
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    region = solved.regions[0]
    schedule = scheduled_region(region)
    assert len(schedule.steps) == 2

    source = emit_pypto_region(graph, region, program_name="cut_fp32_chain").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "cut_fp32_chain"),
        dump_passes=False,
        skip_ptoas=True,
    )

    assert len(list(compiled.output_dir.rglob("*.pto"))) == 2
    orchestration = next(
        (compiled.output_dir / "orchestration").glob("*.cpp")
    ).read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_aic_task(") == 2
    assert "launch_spec.set_block_num(12);" in orchestration
    assert "launch_spec.set_block_num(4);" in orchestration
    assert "add_inout(intermediate_tensor_2)" in orchestration
    assert "add_input(intermediate_tensor_2)" in orchestration


def test_cut_fp32_fanout_compiles_with_two_outputs_and_one_shared_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _FanoutMatmul(), tuple(torch.zeros(64, 64) for _ in range(4))
    )
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    region = solved.regions[0]
    schedule = scheduled_region(region)
    assert [step.solver_ops for step in schedule.steps] == [(0,), (1, 2)]

    source = emit_pypto_region(graph, region, program_name="cut_fp32_fanout").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "cut_fp32_fanout"),
        dump_passes=False,
        skip_ptoas=True,
    )

    assert len(list(compiled.output_dir.rglob("*.pto"))) == 2
    orchestration = next(
        (compiled.output_dir / "orchestration").glob("*.cpp")
    ).read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_aic_task(") == 2
    assert "add_inout(ext_output_0)" in orchestration
    assert "add_inout(ext_output_1)" in orchestration
    assert "add_inout(intermediate_tensor_2)" in orchestration
    assert "add_input(intermediate_tensor_2)" in orchestration


@pytest.mark.parametrize(
    ("name", "module", "args", "expected_matmuls"),
    [
        (
            "retained_panel_matmul",
            _MatmulWithTail(),
            (
                torch.zeros(512, 64, dtype=torch.bfloat16),
                torch.zeros(64, 2048, dtype=torch.bfloat16),
            ),
            2,
        ),
        (
            "diamond_matmul",
            _DiamondMatmul(),
            tuple(torch.zeros(32, 32, dtype=torch.bfloat16) for _ in range(3)),
            3,
        ),
    ],
)
def test_non_split_cube_dag_source_compiles_through_pypto(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    expected_matmuls: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pto, _ = _compile_source(
        name,
        module,
        args,
        tmp_path,
        monkeypatch,
        skip_ptoas=True,
    )
    assert pto.count("pto.tmatmul ") == expected_matmuls
