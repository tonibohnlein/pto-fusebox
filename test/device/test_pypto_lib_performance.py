"""Correctness gates and paired timing characterization for PyPTO-lib controls."""

from __future__ import annotations

import importlib
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.static_mixed import build_examples
from pypto_lib_static_controls import attention_source, dense_swiglu_source
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
    example_name: str
    control_source: str
    rtol: float
    atol: float


COMPARISONS = (
    Comparison(
        "static_attention",
        "pypto_lib_static_attention",
        attention_source(),
        1.0e-4,
        1.0e-4,
    ),
    Comparison(
        "dense_swiglu",
        "pypto_lib_static_dense_swiglu",
        dense_swiglu_source(),
        2.0e-2,
        2.0e-2,
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


def _reference(
    comparison: Comparison,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if comparison.name == "static_attention":
        with torch.no_grad():
            return module(*args)
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


def _control_arguments(
    comparison: Comparison,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if comparison.name == "static_attention":
        return args
    return (
        args[0],
        module.get_parameter("gate_weight"),
        module.get_parameter("up_weight"),
        module.get_parameter("down_weight"),
    )


@pytest.mark.parametrize("comparison", COMPARISONS, ids=lambda item: item.name)
def test_generated_source_matches_control_and_characterizes_performance(
    comparison: Comparison,
    tmp_path: Path,
) -> None:
    pl = importlib.import_module("pypto.language")
    runtime = importlib.import_module("pypto.runtime")
    module, args = build_examples()[comparison.example_name]
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
    config = runtime.RunConfig(
        platform=os.environ.get("PTO_FUSEBOX_PLATFORM", "a2a3"),
        device_id=_device_id(),
        save_kernels=True,
        save_kernels_dir=str(tmp_path),
        dump_passes=False,
    )

    generated_compiled = runtime.run(
        generated_program,
        *generated_args,
        generated_output,
        config=config,
    )
    control_compiled = runtime.run(
        control_program,
        *control_args,
        control_output,
        config=config,
    )
    torch.testing.assert_close(
        generated_output, expected, rtol=comparison.rtol, atol=comparison.atol
    )
    torch.testing.assert_close(
        control_output, expected, rtol=comparison.rtol, atol=comparison.atol
    )
    assert torch.equal(generated_output, control_output)

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
        config=config,
    )
    control_second = runtime.benchmark(
        control_compiled,
        control_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=config,
    )
    control_first = runtime.benchmark(
        control_compiled,
        control_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=config,
    )
    generated_second = runtime.benchmark(
        generated_compiled,
        generated_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=config,
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
        f"control={statistics.median(control_samples):.6f}us"
    )
