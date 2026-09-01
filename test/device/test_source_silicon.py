"""Opt-in silicon coverage for Fusebox-generated PyPTO source.

The default host suite does not collect ``test/device``. Run this file against
an explicit PyPTO checkout and physical device by setting
``PTO_FUSEBOX_RUN_DEVICE_TESTS=1`` and ``PTO_FUSEBOX_DEVICE_ID``.

The matrix is deliberately preselected. It spans the currently source-ready
vector algorithms, single-matmul cube emitter, and initial mixed pipelines
without searching device results for favorable shapes or tolerances.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.pr2335_vector import LayerNorm, RmsNorm, Silu, Softmax
from examples.torch_frontend.qwen3 import build_examples as build_qwen_examples
from examples.torch_frontend.static_mixed import (
    StaticAttentionCore,
    StaticAttentionResidual,
    StaticDenseSwiGlu,
)
from torch import nn

from pto_fusebox import (
    RegionSolveResult,
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
    MixedTransferDirection,
    VectorReplayPhase,
    VectorKernelPlan,
    VectorStreamKind,
)

if os.environ.get("PTO_FUSEBOX_RUN_DEVICE_TESTS") != "1":
    pytest.skip(
        "set PTO_FUSEBOX_RUN_DEVICE_TESTS=1 for Fusebox source silicon tests",
        allow_module_level=True,
    )


ArgsFactory = Callable[[int], tuple[torch.Tensor, ...]]
ReferenceFactory = Callable[[nn.Module, tuple[torch.Tensor, ...]], torch.Tensor]


@dataclass(frozen=True)
class SiliconCase:
    """One statically shaped Torch graph and its numerical contract."""

    name: str
    kind: str
    module: nn.Module
    make_args: ArgsFactory
    rtol: float = 1.0e-4
    atol: float = 1.0e-4
    mixed_contract: str | None = None
    forced_mixed_groups: int | None = None
    expected_mixed_groups: int | None = None
    expected_mixed_trips: int | None = None
    reference: ReferenceFactory | None = None


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


class V2CLhs(nn.Module):
    def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.exp(value), weight)


class V2CRhs(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(lhs, torch.exp(value + bias))


class StreamingSoftmaxPv(nn.Module):
    def forward(self, scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.softmax(scores, dim=-1), value)


class V2CDualRole(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        produced = torch.exp(value)
        return torch.mm(produced, produced)


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


_QWEN_EXAMPLES = build_qwen_examples()


def _qwen_args(seed: int) -> tuple[torch.Tensor, ...]:
    generator = _generator(seed)
    return (
        torch.randn((16, 512), generator=generator, dtype=torch.float32).to(
            torch.bfloat16
        ),
    )


def _qwen_rms_reference(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    value = args[0].float()
    gamma = module.get_parameter("norm_weight")
    inverse_rms = torch.rsqrt(
        torch.sum(value * value, dim=-1, keepdim=True) * (1.0 / value.shape[-1])
        + 1.0e-6
    )
    return (value * inverse_rms * gamma).to(torch.bfloat16)


def _qwen_lm_reference(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    weight = module.get_parameter("lm_head_weight")
    return torch.mm(args[0].float(), weight.float().t())


def _qwen_connected_reference(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    normalized = _qwen_rms_reference(module, args)
    return _qwen_lm_reference(module, (normalized,))


def _dense_swiglu_reference(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """CPU-portable reference for the BF16-storage dense SwiGLU fixture.

    The captured module intentionally uses ``torch.mm(..., out_dtype=FP32)`` to
    describe the device accumulation contract, but that overload is not
    available in every CPU Torch build used by the device harness. Compute the
    same accumulation explicitly in FP32 and preserve the BF16 activation
    boundary before the down projection.
    """

    value = args[0].float()
    gate_weight = module.get_parameter("gate_weight")
    up_weight = module.get_parameter("up_weight")
    down_weight = module.get_parameter("down_weight")
    gate = torch.mm(value, gate_weight.float())
    up = torch.mm(value, up_weight.float())
    activation = (gate * torch.reciprocal(torch.exp(-gate) + 1.0) * up).to(
        torch.bfloat16
    )
    return torch.mm(activation.float(), down_weight.float())


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
    SiliconCase(
        "qwen_rms_norm_chunk_16x512",
        "vector",
        _QWEN_EXAMPLES["qwen3_rms_norm_chunk"][0],
        _qwen_args,
        rtol=1.0e-2,
        atol=1.0e-2,
        reference=_qwen_rms_reference,
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
MATMUL_CASES += (
    SiliconCase(
        "qwen_lm_head_chunk_16x512x192",
        "cube",
        _QWEN_EXAMPLES["qwen3_lm_head_chunk"][0],
        _qwen_args,
        rtol=2.0e-2,
        atol=2.0e-2,
        reference=_qwen_lm_reference,
    ),
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
        "mixed_c2v_resolution_192x64x256",
        "mixed",
        C2VEpilogue(),
        _random_args((192, 64), (64, 256), (1, 256), scale=0.1),
        rtol=1.0e-4,
        atol=1.0e-4,
        mixed_contract="c2v_descriptor_control",
        expected_mixed_groups=4,
        expected_mixed_trips=3,
    ),
    SiliconCase(
        "mixed_c2v_descriptor_128x64x256",
        "mixed",
        C2VEpilogue(),
        _random_args((128, 64), (64, 256), (1, 256), scale=0.1),
        rtol=1.0e-4,
        atol=1.0e-4,
        mixed_contract="c2v_descriptor_control",
        expected_mixed_groups=4,
        expected_mixed_trips=2,
    ),
    SiliconCase(
        "mixed_c2v_descriptor_256x64x256",
        "mixed",
        C2VEpilogue(),
        _random_args((256, 64), (64, 256), (1, 256), scale=0.1),
        rtol=1.0e-4,
        atol=1.0e-4,
        mixed_contract="c2v_descriptor_control",
        expected_mixed_groups=4,
        expected_mixed_trips=4,
    ),
    SiliconCase(
        "mixed_c2v_descriptor_256x64x384",
        "mixed",
        C2VEpilogue(),
        _random_args((256, 64), (64, 384), (1, 384), scale=0.1),
        rtol=1.0e-4,
        atol=1.0e-4,
        mixed_contract="c2v_descriptor_control",
        expected_mixed_groups=6,
        expected_mixed_trips=4,
    ),
    SiliconCase(
        "mixed_c2v_streamed_groups_384x64x256",
        "mixed",
        C2VEpilogue(),
        _random_args((384, 64), (64, 256), (1, 256), scale=0.1),
        rtol=1.0e-4,
        atol=1.0e-4,
        mixed_contract="c2v_streamed_groups",
    ),
    SiliconCase(
        "mixed_v2c_lhs_96x64x128",
        "mixed",
        V2CLhs(),
        _random_args((96, 64), (64, 128), scale=0.1),
        mixed_contract="v2c_lhs",
    ),
    SiliconCase(
        "mixed_v2c_lhs_stage2_96x64x128",
        "mixed",
        V2CLhs(),
        _random_args((96, 64), (64, 128), scale=0.1),
        mixed_contract="v2c_lhs_streamed_groups",
        forced_mixed_groups=1,
    ),
    SiliconCase(
        "mixed_v2c_rhs_96x64x128",
        "mixed",
        V2CRhs(),
        _random_args((96, 64), (64, 128), (1, 128), scale=0.1),
        mixed_contract="v2c_rhs",
    ),
    SiliconCase(
        "mixed_v2c_streaming_softmax_pv_16x4096x64",
        "mixed",
        StreamingSoftmaxPv(),
        _random_args((16, 4096), (4096, 64), scale=0.1),
        mixed_contract="v2c_streaming_softmax_pv",
    ),
    SiliconCase(
        "mixed_v2c_dual_role_64x64",
        "mixed",
        V2CDualRole(),
        _random_args((64, 64), scale=0.1),
        mixed_contract="v2c_dual_role",
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
        "mixed_qk_softmax_pv_streamed_groups_384x64x128",
        "mixed",
        StaticAttentionCore(),
        _random_args((384, 64), (64, 64), (64, 128), scale=0.1),
        rtol=1.0e-4,
        atol=1.0e-4,
        mixed_contract="cvc_streamed_groups",
    ),
    SiliconCase(
        "mixed_dense_swiglu_128x64x128x64",
        "mixed",
        _static_dense_swiglu(),
        _random_bf16_args((128, 64)),
        rtol=2.0e-2,
        atol=2.0e-2,
        reference=_dense_swiglu_reference,
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
    SiliconCase(
        "qwen_rms_lm_head_v2c_16x512x192",
        "mixed",
        _QWEN_EXAMPLES["qwen3_rms_lm_head"][0],
        _qwen_args,
        rtol=2.0e-2,
        atol=2.0e-2,
        mixed_contract="qwen_rms_lm_head_v2c",
        reference=_qwen_connected_reference,
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
    plan: CubeKernelPlan | MixedKernelPlan | VectorKernelPlan,
) -> None:
    work_units = (
        plan.active_groups if isinstance(plan, MixedKernelPlan) else plan.work_units
    )
    output_dir = Path(getattr(compiled, "output_dir"))
    pto_files = list(output_dir.rglob("*.pto"))
    assert len(pto_files) == 1
    pto = pto_files[0].read_text(encoding="utf-8")
    if case.kind == "mixed":
        assert isinstance(plan, MixedKernelPlan)
        assert pto.count("pto.kernel_kind = #pto.kernel_kind<cube>") == 1
        assert pto.count("pto.kernel_kind = #pto.kernel_kind<vector>") == 1
        directions = {fifo.direction for fifo in plan.fifos}
        if MixedTransferDirection.CUBE_TO_VECTOR in directions:
            assert "pto.tpush_to_aiv" in pto
            assert "pto.tpop_from_aic" in pto
            assert "pto.tfree_from_aic" in pto
        if MixedTransferDirection.VECTOR_TO_CUBE in directions:
            assert "pto.tpush_to_aic" in pto
            assert "pto.tpop_from_aiv" in pto
            assert "pto.tfree_from_aiv" in pto
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


def _assert_mixed_contract(
    case: SiliconCase,
    plan: CubeKernelPlan | MixedKernelPlan | VectorKernelPlan,
    source: str,
) -> None:
    if case.mixed_contract is None:
        return
    assert isinstance(plan, MixedKernelPlan)
    assert plan.source_codegen_ready
    if case.mixed_contract == "cvc_streamed_groups":
        assert len(plan.fifos) == 2
        assert plan.spatial_tiles == 4
        assert plan.active_groups == 4
        assert plan.max_trips_per_group == 1
        assert plan.pipeline_stages == 1
        assert plan.requested_skew_depth == 0
        assert not plan.overlap_implementable
        assert "pl.range(1, init_values=" in source
        return
    if case.mixed_contract == "qwen_rms_lm_head_v2c":
        assert plan.active_groups == 3
        assert plan.spatial_tiles == 3
        assert plan.max_trips_per_group == 1
        assert plan.pipeline_stages == 1
        assert len(plan.fifos) == 1
        fifo = plan.fifos[0]
        assert fifo.direction is MixedTransferDirection.VECTOR_TO_CUBE
        assert (fifo.valid_rows, fifo.valid_cols) == (16, 512)
        assert fifo.slot_bytes == 16 * 512 * 2
        assert fifo.reserved_bytes == fifo.slot_bytes * fifo.slot_count
        assert "pl.tensor.cast(" in source
        assert "b_trans=True" in source
        return
    assert len(plan.fifos) == 1
    fifo = plan.fifos[0]
    if case.mixed_contract == "c2v_streamed_groups":
        assert fifo.direction is MixedTransferDirection.CUBE_TO_VECTOR
        assert plan.spatial_tiles == 24
        assert plan.active_groups == 6
        assert plan.max_trips_per_group == 4
        assert plan.pipeline_stages == 2
        assert plan.requested_skew_depth == 1
        assert plan.overlap_implementable
        assert "pl.pipeline(4, stage=2" in source
        return
    if case.mixed_contract == "c2v_descriptor_control":
        assert fifo.direction is MixedTransferDirection.CUBE_TO_VECTOR
        assert (fifo.valid_rows, fifo.valid_cols) == (64, 64)
        assert fifo.slot_bytes == 16384
        assert fifo.slot_count == 8
        assert fifo.reserved_bytes == 131072
        assert case.expected_mixed_groups is not None
        assert case.expected_mixed_trips is not None
        assert plan.active_groups == case.expected_mixed_groups
        assert plan.max_trips_per_group == case.expected_mixed_trips
        assert plan.pipeline_stages == 2
        assert plan.requested_skew_depth == 1
        assert plan.overlap_implementable
        assert f"pl.pipeline({case.expected_mixed_trips}, stage=2" in source
        return
    if case.mixed_contract == "v2c_lhs_streamed_groups":
        assert plan.spatial_tiles == 2
        assert plan.active_groups == 1
        assert plan.max_trips_per_group == 2
        assert plan.pipeline_stages == 2
        assert plan.requested_skew_depth == 1
        assert plan.overlap_implementable
        assert "pl.pipeline(2, stage=2" in source
    assert fifo.direction is MixedTransferDirection.VECTOR_TO_CUBE
    assert fifo.slot_count == 8
    assert fifo.reserved_bytes == fifo.slot_bytes * fifo.slot_count
    assert len(plan.stages) == 2
    assert plan.stages[0].vector_stream is not None

    if case.mixed_contract in {"v2c_lhs", "v2c_lhs_streamed_groups"}:
        assert fifo.spatial_m and not fifo.spatial_n
    elif case.mixed_contract == "v2c_rhs":
        assert not fifo.spatial_m and fifo.spatial_n
    elif case.mixed_contract == "v2c_streaming_softmax_pv":
        stream = plan.stages[0].vector_stream
        assert stream.kind is VectorStreamKind.SOFTMAX_FLASH
        assert fifo.spatial_m and not fifo.spatial_n
        assert fifo.valid_cols == stream.chunk
        assert plan.stages[1].cube_window_k == (stream.chunk,)
        apply = stream.phase(VectorReplayPhase.APPLY)
        assert apply.loop is not None
        assert apply.loop.pipeline_stages == 2
        assert apply.tail is not None and apply.tail.present
        elements_per_slot = fifo.valid_rows * fifo.valid_cols
        assert elements_per_slot > 0
        assert fifo.slot_bytes % elements_per_slot == 0
        element_bytes = fifo.slot_bytes // elements_per_slot
        assert plan.cube_stage_peak_l1_bytes == (
            stream.chunk
            * plan.n_partition.big
            * element_bytes
            * apply.loop.pipeline_stages
        )
        assert "for stats_chunk" in source
        assert "for apply_chunk" in source
        assert source.count("pl.tensor.matmul(") == 1
        assert source.count("pl.tensor.matmul_acc(") == 2
    elif case.mixed_contract == "v2c_dual_role":
        assert fifo.spatial_m and fifo.spatial_n
        assert plan.m_partition.parts == 1
        assert plan.n_partition.parts == 1
        assert re.search(r"pl\.tensor\.matmul\(([^,]+), \1,", source)
    else:
        pytest.fail(f"unknown mixed contract: {case.mixed_contract}")


def _output_signature(output: torch.Tensor) -> str:
    payload = output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _assert_numerics(
    case: SiliconCase,
    seed: int,
    output: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    assert torch.isfinite(output).all(), f"{case.name} seed {seed} left invalid output"
    assert torch.allclose(output, expected, rtol=case.rtol, atol=case.atol), (
        f"{case.name} seed {seed}: max abs error "
        f"{torch.max(torch.abs(output - expected)).item():.6g}"
    )


def _force_one_way_mixed_groups(
    result: RegionSolveResult,
    active_groups: int,
) -> RegionSolveResult:
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    step = solution["steps"][0]
    plan = step["plan"]
    assert plan["protocol"] == "one_way"
    spatial_tiles = plan["spatial_tiles"]
    assert spatial_tiles % active_groups == 0
    trips = spatial_tiles // active_groups
    assert trips >= 2
    plan["active_groups"] = active_groups
    plan["min_trips_per_group"] = trips
    plan["max_trips_per_group"] = trips
    plan["pipeline_stages"] = 2
    plan["requested_skew_depth"] = 1
    plan["model_overlap_granted"] = True
    plan["overlap_implementable"] = True
    step["launch"]["cores"] = active_groups * (1 + plan["vector_lanes"])
    return replace(result, solution=solution)


def _run_case(case: SiliconCase, tmp_path: Path) -> None:
    ir = importlib.import_module("pypto.ir")
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
    if case.forced_mixed_groups is not None:
        region = _force_one_way_mixed_groups(
            region,
            case.forced_mixed_groups,
        )
    assert can_emit_region(graph, region)
    plan = scheduled_region(region).steps[0].plan
    assert isinstance(plan, (CubeKernelPlan, MixedKernelPlan, VectorKernelPlan))
    emitted = emit_pypto_region(graph, region, program_name=case.name)
    assert emitted.kind is not None
    assert emitted.kind.value == case.kind
    assert "auto_fuse" not in emitted.source
    assert "auto_tile" not in emitted.source
    _assert_mixed_contract(case, plan, emitted.source)

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

    compiled = ir.compile(program, **config.compile_kwargs())
    _assert_static_artifact(compiled, case, plan)
    for seed in range(seed_count):
        args = case.make_args(seed)
        runtime_args = bind_emitted_inputs(case.module, graph, emitted, args)
        with torch.no_grad():
            expected = (
                case.module(*args)
                if case.reference is None
                else case.reference(case.module, args)
            )
        output = torch.full_like(expected, torch.nan)
        compiled(*runtime_args, output, config=config)
        _assert_numerics(case, seed, output, expected)

    repeat_count = int(os.environ.get("PTO_FUSEBOX_DEVICE_REPEATS", "1"))
    assert repeat_count > 0
    if repeat_count > 1:
        args = case.make_args(0)
        runtime_args = bind_emitted_inputs(case.module, graph, emitted, args)
        with torch.no_grad():
            expected = (
                case.module(*args)
                if case.reference is None
                else case.reference(case.module, args)
            )
        signatures: set[str] = set()
        for repeat in range(repeat_count):
            output = torch.full_like(expected, torch.nan)
            compiled(*runtime_args, output, config=config)
            _assert_numerics(case, repeat, output, expected)
            signatures.add(_output_signature(output))
        assert len(signatures) == 1, (
            f"{case.name} produced {len(signatures)} signatures across "
            f"{repeat_count} identical launches"
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
