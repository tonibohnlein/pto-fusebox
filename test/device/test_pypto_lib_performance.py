"""Correctness gates and paired timing characterization for PyPTO-lib controls."""

from __future__ import annotations

import importlib
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.deepseek_v4 import (
    build_examples as build_deepseek_examples,
)
from examples.torch_frontend.qwen3 import build_examples as build_qwen_examples
from examples.torch_frontend.static_mixed import (
    build_examples as build_static_mixed_examples,
)
from pypto_lib_static_controls import (
    attention_source,
    deepseek_mtp_unfused_source,
    dense_swiglu_source,
    qwen_lm_head_source,
    qwen_rms_lm_head_source,
    qwen_rms_norm_source,
)
from torch import nn

from pto_fusebox import (
    bind_emitted_inputs,
    emit_pypto_region,
    export_and_normalize,
    solve_graph,
)

if os.environ.get("PTO_FUSEBOX_RUN_DEVICE_TESTS") != "1":
    pytest.skip(
        "set PTO_FUSEBOX_RUN_DEVICE_TESTS=1 for Fusebox source silicon tests",
        allow_module_level=True,
    )


@dataclass(frozen=True)
class Comparison:
    name: str
    family: str
    example_name: str
    control_source: str
    rtol: float
    atol: float
    control_kind: str
    bit_exact: bool = True


COMPARISONS = (
    Comparison(
        "static_attention",
        "static_mixed",
        "pypto_lib_static_attention",
        attention_source(),
        1.0e-4,
        1.0e-4,
        "pypto_lib_style",
    ),
    Comparison(
        "dense_swiglu",
        "static_mixed",
        "pypto_lib_static_dense_swiglu",
        dense_swiglu_source(),
        2.0e-2,
        2.0e-2,
        "pypto_lib_style",
    ),
    Comparison(
        "qwen_rms_norm",
        "qwen",
        "qwen3_rms_norm_chunk",
        qwen_rms_norm_source(),
        1.0e-2,
        1.0e-2,
        "pypto_lib_reduced",
    ),
    Comparison(
        "qwen_lm_head",
        "qwen",
        "qwen3_lm_head_chunk",
        qwen_lm_head_source(),
        2.0e-2,
        2.0e-2,
        "pypto_lib_reduced",
    ),
    Comparison(
        "qwen_rms_lm_head",
        "qwen",
        "qwen3_rms_lm_head",
        qwen_rms_lm_head_source(),
        2.0e-2,
        2.0e-2,
        "pypto_lib_reduced",
    ),
    Comparison(
        "deepseek_mtp_projection",
        "deepseek",
        "deepseek_v4_mtp_projection",
        deepseek_mtp_unfused_source(),
        1.0e-2,
        1.0e-2,
        "independent_unfused",
        # Five frozen silicon seeds happened to be bit-identical, but the
        # independent schedules are allowed to change FP32 accumulation order
        # across compiler revisions. Keep the durable contract tolerance-based.
        bit_exact=False,
    ),
)


def _example(comparison: Comparison) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    builders = {
        "static_mixed": build_static_mixed_examples,
        "qwen": build_qwen_examples,
        "deepseek": build_deepseek_examples,
    }
    return builders[comparison.family]()[comparison.example_name]


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


def _reference(
    comparison: Comparison,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if comparison.name == "static_attention":
        with torch.no_grad():
            return module(*args)
    if comparison.name == "dense_swiglu":
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
    if comparison.name.startswith("qwen_"):
        value = args[0].float()
        if comparison.name != "qwen_lm_head":
            gamma = module.get_parameter("norm_weight")
            inverse_rms = torch.rsqrt(
                torch.sum(value * value, dim=-1, keepdim=True) * (1.0 / value.shape[-1])
                + 1.0e-6
            )
            value = (value * inverse_rms * gamma).to(torch.bfloat16).float()
        weight = (
            module.get_parameter("lm_head_weight")
            if comparison.name != "qwen_rms_norm"
            else None
        )
        if weight is None:
            return value.to(torch.bfloat16)
        return torch.mm(value, weight.float().t())
    if comparison.name == "deepseek_mtp_projection":
        hidden = args[0].float()
        previous = args[1].float()
        enorm = module.get_parameter("enorm_weight")
        hnorm = module.get_parameter("hnorm_weight")
        hidden_inv = torch.rsqrt(
            torch.sum(hidden * hidden, dim=-1, keepdim=True) / hidden.shape[-1] + 1.0e-6
        )
        previous_inv = torch.rsqrt(
            torch.sum(previous * previous, dim=-1, keepdim=True) / previous.shape[-1]
            + 1.0e-6
        )
        embedded = torch.mm(
            hidden * hidden_inv * enorm,
            module.get_parameter("e_proj.weight").float().t(),
        )
        history = torch.mm(
            previous * previous_inv * hnorm,
            module.get_parameter("h_proj.weight").float().t(),
        )
        return embedded + history
    raise AssertionError(f"missing reference for {comparison.name}")


def _control_arguments(
    comparison: Comparison,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if comparison.name == "static_attention":
        return args
    if comparison.name == "dense_swiglu":
        return (
            args[0],
            module.get_parameter("gate_weight"),
            module.get_parameter("up_weight"),
            module.get_parameter("down_weight"),
        )
    if comparison.name == "qwen_rms_norm":
        return args[0], module.get_parameter("norm_weight")
    if comparison.name == "qwen_lm_head":
        return args[0], module.get_parameter("lm_head_weight")
    if comparison.name == "qwen_rms_lm_head":
        return (
            args[0],
            module.get_parameter("norm_weight"),
            module.get_parameter("lm_head_weight"),
        )
    if comparison.name == "deepseek_mtp_projection":
        return (
            args[0],
            module.get_parameter("enorm_weight"),
            module.get_parameter("e_proj.weight"),
            args[1],
            module.get_parameter("hnorm_weight"),
            module.get_parameter("h_proj.weight"),
        )
    raise AssertionError(f"missing control arguments for {comparison.name}")


@pytest.mark.parametrize("comparison", COMPARISONS, ids=lambda item: item.name)
def test_generated_source_matches_control_and_characterizes_performance(
    comparison: Comparison,
    tmp_path: Path,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    runtime = importlib.import_module("pypto.runtime")
    module, args = _example(comparison)
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=int(os.environ.get("PTO_FUSEBOX_SOLVER_WORKERS", "2")),
        require_source_codegen=True,
    )
    assert solved.whole_graph_codegen_ready
    emitted = emit_pypto_region(
        graph,
        solved.regions[0],
        program_name=f"generated_{comparison.name}",
    )
    generated_program = pl.parse_program(emitted.source)
    control_program = pl.parse_program(comparison.control_source)
    generated_args = bind_emitted_inputs(module, graph, emitted, args)
    control_args = _control_arguments(comparison, module, args)
    expected = _reference(comparison, module, args)
    generated_output = torch.full_like(expected, torch.nan)
    control_output = torch.full_like(expected, torch.nan)
    generated_config = runtime.RunConfig(
        platform=os.environ.get("PTO_FUSEBOX_PLATFORM", "a2a3"),
        device_id=_device_id(),
        save_kernels=True,
        save_kernels_dir=str(tmp_path / "generated"),
        dump_passes=False,
    )
    control_config = runtime.RunConfig(
        platform=os.environ.get("PTO_FUSEBOX_PLATFORM", "a2a3"),
        device_id=_device_id(),
        save_kernels=True,
        save_kernels_dir=str(tmp_path / "control"),
        dump_passes=False,
    )

    generated_compiled = ir.compile(
        generated_program, **generated_config.compile_kwargs()
    )
    control_compiled = ir.compile(control_program, **control_config.compile_kwargs())
    generated_compiled(*generated_args, generated_output, config=generated_config)
    control_compiled(*control_args, control_output, config=control_config)
    torch.testing.assert_close(
        generated_output, expected, rtol=comparison.rtol, atol=comparison.atol
    )
    torch.testing.assert_close(
        control_output, expected, rtol=comparison.rtol, atol=comparison.atol
    )
    if comparison.bit_exact:
        assert torch.equal(generated_output, control_output)
    else:
        torch.testing.assert_close(
            generated_output,
            control_output,
            rtol=comparison.rtol,
            atol=comparison.atol,
        )

    # These are characterization samples, not a performance acceptance gate.
    # The device campaign reports order strata and uncertainty before any
    # performance conclusion is drawn; correctness above remains the ST gate.
    rounds = int(os.environ.get("PTO_FUSEBOX_PERF_ROUNDS", "30"))
    warmup = int(os.environ.get("PTO_FUSEBOX_PERF_WARMUP", "5"))
    assert rounds > 0 and warmup >= 0
    generated_dispatch = (*generated_args, generated_output)
    control_dispatch = (*control_args, control_output)
    generated_first = runtime.benchmark(
        generated_compiled,
        generated_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=generated_config,
    )
    control_second = runtime.benchmark(
        control_compiled,
        control_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=control_config,
    )
    control_first = runtime.benchmark(
        control_compiled,
        control_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=control_config,
    )
    generated_second = runtime.benchmark(
        generated_compiled,
        generated_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=generated_config,
    )
    generated_samples = generated_first.device_wall_us + generated_second.device_wall_us
    control_samples = control_first.device_wall_us + control_second.device_wall_us
    assert len(generated_samples) == len(control_samples) == 2 * rounds
    assert min(generated_samples) > 0.0
    assert min(control_samples) > 0.0
    ratio = statistics.median(control_samples) / statistics.median(generated_samples)
    print(
        f"{comparison.name}: control/generated device_wall ratio={ratio:.6f}; "
        f"generated={statistics.median(generated_samples):.6f}us; "
        f"control={statistics.median(control_samples):.6f}us; "
        f"control_kind={comparison.control_kind}"
    )
