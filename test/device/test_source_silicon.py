"""Opt-in silicon coverage for Fusebox-generated PyPTO source.

The default host suite does not collect ``test/device``. Run this file against
an explicit PyPTO checkout and physical device by setting
``PTO_FUSEBOX_RUN_DEVICE_TESTS=1`` and ``PTO_FUSEBOX_DEVICE_ID``.

The matrix is deliberately preselected. It spans the currently source-ready
vector algorithms, single-matmul cube emitter, and initial mixed pipelines
without searching device results for favorable shapes or tolerances.
"""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.pr2335_vector import LayerNorm, RmsNorm, Silu, Softmax
from examples.torch_frontend.static_mixed import (
    StaticAttentionCore,
    StaticAttentionResidual,
    StaticDenseSwiGlu,
)
from torch import nn

from pto_fusebox import (
    bind_emitted_inputs,
    can_emit_region,
    emit_pypto_region,
    export_and_normalize,
    scheduled_region,
    solve_graph,
)
from pto_fusebox.schedule.schema import (
    CubeKernelPlan,
    MixedKernelPlan,
    VectorKernelPlan,
)

if os.environ.get("PTO_FUSEBOX_RUN_DEVICE_TESTS") != "1":
    pytest.skip(
        "set PTO_FUSEBOX_RUN_DEVICE_TESTS=1 for Fusebox source silicon tests",
        allow_module_level=True,
    )


ArgsFactory = Callable[[int], tuple[torch.Tensor, ...]]


@dataclass(frozen=True)
class SiliconCase:
    """One statically shaped Torch graph and its numerical contract."""

    name: str
    kind: str
    module: nn.Module
    make_args: ArgsFactory
    rtol: float = 1.0e-4
    atol: float = 1.0e-4


class PointwiseChain(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.exp(lhs * 0.5) + rhs, rhs)


class RaggedPointwise(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.exp(value * 0.5) + value


class BroadcastAdd(nn.Module):
    def forward(self, value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return value + bias


class CastRoundTrip(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.float16).to(torch.float32) + value


class SumOfSquares(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sum(value * value, dim=-1, keepdim=True)


class Matmul(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.mm(lhs, rhs)


class C2VEpilogue(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(value, weight) + bias


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _random_args(
    *shapes: tuple[int, int],
    scale: float = 0.5,
) -> ArgsFactory:
    def make(seed: int) -> tuple[torch.Tensor, ...]:
        generator = _generator(seed)
        return tuple(
            torch.randn(shape, generator=generator, dtype=torch.float32) * scale
            for shape in shapes
        )

    return make


def _random_bf16_args(*shapes: tuple[int, int], scale: float = 0.25) -> ArgsFactory:
    def make(seed: int) -> tuple[torch.Tensor, ...]:
        generator = _generator(seed)
        return tuple(
            (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(
                torch.bfloat16
            )
            for shape in shapes
        )

    return make


def _static_dense_swiglu() -> StaticDenseSwiGlu:
    module = StaticDenseSwiGlu(hidden_size=64, intermediate_size=128)
    generator = _generator(0)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=torch.float32,
                ).to(torch.bfloat16)
                * 0.25
            )
    return module


def _rms_args(seed: int) -> tuple[torch.Tensor, ...]:
    generator = _generator(seed)
    value = torch.randn((512, 512), generator=generator) * 0.5
    gamma = 1.0 + torch.randn((1, 512), generator=generator) * 0.1
    return value, gamma


def _layer_args(seed: int) -> tuple[torch.Tensor, ...]:
    generator = _generator(seed)
    value = torch.randn((512, 256), generator=generator) * 0.5
    gamma = 1.0 + torch.randn((1, 256), generator=generator) * 0.1
    beta = torch.randn((1, 256), generator=generator) * 0.1
    return value, gamma, beta


VECTOR_CASES = (
    SiliconCase(
        "vector_pointwise_chain_96x320",
        "vector",
        PointwiseChain(),
        _random_args((96, 320), (96, 320), scale=0.25),
    ),
    SiliconCase(
        "vector_pointwise_ragged_257x65",
        "vector",
        RaggedPointwise(),
        _random_args((257, 65), scale=0.25),
    ),
    SiliconCase(
        "vector_broadcast_column_64x128",
        "vector",
        BroadcastAdd(),
        _random_args((64, 128), (64, 1), scale=0.5),
    ),
    SiliconCase(
        "vector_broadcast_row_64x128",
        "vector",
        BroadcastAdd(),
        _random_args((64, 128), (1, 128), scale=0.5),
    ),
    SiliconCase(
        "vector_cast_roundtrip_128x512",
        "vector",
        CastRoundTrip(),
        _random_args((128, 512), scale=0.5),
        rtol=1.0e-3,
        atol=1.0e-3,
    ),
    SiliconCase(
        "vector_sum_of_squares_128x1024",
        "vector",
        SumOfSquares(),
        _random_args((128, 1024), scale=0.25),
        rtol=1.0e-3,
        atol=1.0e-3,
    ),
    SiliconCase(
        "vector_softmax_ragged_130x272",
        "vector",
        Softmax(),
        _random_args((130, 272), scale=1.0),
    ),
    *(
        SiliconCase(
            f"vector_softmax_{rows}x{cols}",
            "vector",
            Softmax(),
            _random_args((rows, cols), scale=1.0),
        )
        for rows, cols in ((512, 256), (256, 512), (128, 1024), (32, 8192))
    ),
    SiliconCase(
        "vector_rms_norm_512x512",
        "vector",
        RmsNorm(),
        _rms_args,
        # PR #2335 computes row_sum(x*x) before applying 1/N. The generated
        # source is bit-identical to that hand-written formulation, whose
        # accumulation is less accurate than Torch's mean-first reference and
        # requires the historical 1e-2 tolerance.
        rtol=1.0e-2,
        atol=1.0e-2,
    ),
    SiliconCase(
        "vector_layer_norm_512x256",
        "vector",
        LayerNorm(),
        _layer_args,
        rtol=1.0e-3,
        atol=1.0e-3,
    ),
    SiliconCase(
        "vector_silu_512x256",
        "vector",
        Silu(),
        _random_args((512, 256), scale=0.5),
    ),
)


MATMUL_CASES = tuple(
    SiliconCase(
        f"matmul_{m}x{k}x{n}",
        "cube",
        Matmul(),
        _random_args((m, k), (k, n), scale=0.1),
        rtol=1.0e-3,
        atol=1.0e-3,
    )
    for m, k, n in (
        (16, 64, 32),
        (16, 272, 32),
        (64, 272, 80),
        (128, 256, 192),
        (64, 240, 80),
        (32, 736, 64),
        (256, 256, 256),
        (96, 64, 128),
        (48, 512, 128),
        (64, 512, 256),
    )
)


MIXED_CASES = (
    SiliconCase(
        "mixed_c2v_epilogue_32x64x32",
        "mixed",
        C2VEpilogue(),
        _random_args((32, 64), (64, 32), (1, 32), scale=0.1),
        rtol=1.0e-4,
        atol=1.0e-4,
    ),
    SiliconCase(
        "mixed_qk_softmax_pv_96x64x128",
        "mixed",
        StaticAttentionCore(),
        _random_args((96, 64), (64, 64), (64, 128), scale=0.1),
        rtol=1.0e-4,
        atol=1.0e-4,
    ),
    SiliconCase(
        "mixed_dense_swiglu_128x64x128x64",
        "mixed",
        _static_dense_swiglu(),
        _random_bf16_args((128, 64)),
        rtol=2.0e-2,
        atol=2.0e-2,
    ),
    SiliconCase(
        "mixed_qk_softmax_pv_residual_96x64x128",
        "mixed",
        StaticAttentionResidual(),
        _random_args(
            (96, 64),
            (64, 64),
            (64, 128),
            (96, 128),
            scale=0.1,
        ),
        rtol=1.0e-4,
        atol=1.0e-4,
    ),
)


def _solver() -> Path:
    configured = os.environ.get("PTO_FUSEBOX_SOLVER")
    path = (
        Path(configured)
        if configured
        else Path(__file__).parents[2] / "build" / "mlsys_mixed"
    )
    if not path.is_file():
        pytest.fail(f"Fusebox solver does not exist: {path}")
    return path


def _device_id() -> int:
    raw = os.environ.get("PTO_FUSEBOX_DEVICE_ID")
    if raw is None:
        pytest.fail("PTO_FUSEBOX_DEVICE_ID must name one physical device")
    assert raw is not None
    return int(raw)


def _assert_static_artifact(
    compiled: object,
    case: SiliconCase,
    work_units: int,
) -> None:
    output_dir = Path(getattr(compiled, "output_dir"))
    pto_files = list(output_dir.rglob("*.pto"))
    assert len(pto_files) == 1
    pto = pto_files[0].read_text(encoding="utf-8")
    if case.kind == "mixed":
        assert pto.count("pto.kernel_kind = #pto.kernel_kind<cube>") == 1
        assert pto.count("pto.kernel_kind = #pto.kernel_kind<vector>") == 1
        assert "pto.tpush_to_aiv" in pto
        assert "pto.tpop_from_aic" in pto
        assert "pto.tfree_from_aic" in pto
    else:
        assert "pto.tpush" not in pto
        assert "pto.tpop" not in pto
        assert "pto.tfree" not in pto
    if case.kind == "vector":
        assert "pto.tmatmul" not in pto
        assert re.search(r"partition_tensor_view<[^>]*\?", pto) is None
        assert re.search(r"valid_(?:row|col) = %arg[0-9]+", pto) is None
    elif case.kind == "cube":
        assert "pto.tmatmul" in pto

    orchestration_files = list((output_dir / "orchestration").glob("*.cpp"))
    assert len(orchestration_files) == 1
    orchestration = orchestration_files[0].read_text(encoding="utf-8")
    if case.kind == "mixed":
        assert orchestration.count("rt_submit_task(") == 1
    else:
        assert len(re.findall(r"\brt_submit_ai[cv]_task\(", orchestration)) == 1
    assert orchestration.count("launch_spec.set_block_num(") == 1
    assert f"launch_spec.set_block_num({work_units});" in orchestration
    assert "region_index" not in orchestration


def _run_case(case: SiliconCase, tmp_path: Path) -> None:
    pl = importlib.import_module("pypto.language")
    runtime = importlib.import_module("pypto.runtime")

    initial_args = case.make_args(0)
    graph = export_and_normalize(case.module, initial_args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=int(os.environ.get("PTO_FUSEBOX_SOLVER_WORKERS", "2")),
        require_source_codegen=True,
    )
    assert solved.regions_solved == 1
    assert len(solved.regions) == 1
    region = solved.regions[0]
    assert can_emit_region(graph, region)
    plan = scheduled_region(region).steps[0].plan
    assert isinstance(plan, (CubeKernelPlan, MixedKernelPlan, VectorKernelPlan))
    emitted = emit_pypto_region(graph, region, program_name=case.name)
    assert emitted.kind is not None
    assert emitted.kind.value == case.kind
    assert "auto_fuse" not in emitted.source
    assert "auto_tile" not in emitted.source

    program = pl.parse_program(emitted.source)
    config = runtime.RunConfig(
        platform=os.environ.get("PTO_FUSEBOX_PLATFORM", "a2a3"),
        device_id=_device_id(),
        save_kernels=True,
        save_kernels_dir=str(tmp_path / case.name),
        dump_passes=False,
    )
    seed_count = int(os.environ.get("PTO_FUSEBOX_DEVICE_SEEDS", "5"))
    assert seed_count > 0

    compiled = None
    for seed in range(seed_count):
        args = case.make_args(seed)
        runtime_args = bind_emitted_inputs(case.module, graph, emitted, args)
        with torch.no_grad():
            expected = case.module(*args)
        output = torch.full_like(expected, torch.nan)
        if compiled is None:
            compiled = runtime.run(program, *runtime_args, output, config=config)
            _assert_static_artifact(compiled, case, plan.work_units)
        else:
            compiled(*runtime_args, output, config=config)

        assert torch.isfinite(output).all(), (
            f"{case.name} seed {seed} left invalid output"
        )
        assert torch.allclose(output, expected, rtol=case.rtol, atol=case.atol), (
            f"{case.name} seed {seed}: max abs error "
            f"{torch.max(torch.abs(output - expected)).item():.6g}"
        )


@pytest.mark.parametrize("case", VECTOR_CASES, ids=lambda case: case.name)
def test_generated_vector_source_on_silicon(case: SiliconCase, tmp_path: Path) -> None:
    _run_case(case, tmp_path)


@pytest.mark.parametrize("case", MATMUL_CASES, ids=lambda case: case.name)
def test_generated_matmul_source_on_silicon(case: SiliconCase, tmp_path: Path) -> None:
    _run_case(case, tmp_path)


@pytest.mark.parametrize("case", MIXED_CASES, ids=lambda case: case.name)
def test_generated_mixed_source_on_silicon(case: SiliconCase, tmp_path: Path) -> None:
    _run_case(case, tmp_path)
