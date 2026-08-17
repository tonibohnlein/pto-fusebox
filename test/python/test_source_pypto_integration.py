"""Opt-in end-to-end checks against an independently installed PyPTO/PTOAS.

The standalone source backend intentionally does not depend on PyPTO.  Set
``PTO_FUSEBOX_PYPTO_INTEGRATION=1`` in an environment whose ``pypto`` import and
``PTOAS_ROOT`` point at the versions under validation to exercise this gate.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
import torch
from pto_fusebox import emit_pypto_region, export_and_normalize, solve_graph
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


class _WideSoftmax(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.softmax(value, dim=-1)


class _Silu(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.reciprocal(torch.exp(value * -1.0) + 1.0)


def _solver() -> Path:
    configured = os.environ.get("PTO_FUSEBOX_TEST_SOLVER")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "build" / "mlsys_mixed"


@pytest.mark.parametrize(
    ("name", "module", "args", "expected_pto_op"),
    [
        (
            "pointwise_chain",
            _PointwiseChain(),
            (torch.zeros(96, 320), torch.ones(96, 320)),
            "pto.texp",
        ),
        (
            "sum_of_squares",
            _SumOfSquares(),
            (torch.ones(128, 1024),),
            "pto.trowsum",
        ),
        (
            "matmul_with_tail",
            _MatmulWithTail(),
            (torch.zeros(64, 272), torch.zeros(272, 80)),
            "pto.tmatmul.acc",
        ),
        (
            "wide_softmax",
            _WideSoftmax(),
            (torch.zeros(32, 8192),),
            "pto.trowmax",
        ),
        (
            "silu",
            _Silu(),
            (torch.zeros(512, 256),),
            "pto.trecip",
        ),
    ],
)
def test_generated_source_compiles_through_pypto_and_ptoas(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    expected_pto_op: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")

    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(module, args)
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    assert solved.regions_solved == 1
    assert len(solved.regions) == 1

    source = emit_pypto_region(graph, solved.regions[0], program_name=name).source
    program = pl.parse_program(source)
    compiled = ir.compile(
        program,
        output_dir=str(tmp_path / name),
        dump_passes=False,
        skip_ptoas=False,
    )

    pto_files = list(compiled.output_dir.rglob("*.pto"))
    generated_cpp = list((compiled.output_dir / "ptoas").glob("*.cpp"))
    assert len(pto_files) == 1
    assert len(generated_cpp) == 1
    assert expected_pto_op in pto_files[0].read_text(encoding="utf-8")
