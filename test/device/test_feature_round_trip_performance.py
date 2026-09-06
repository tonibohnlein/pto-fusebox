"""Paired silicon timing for generic feature-round-trip source candidates.

The control is the best independently enumerated source-ready partition with a
different kernel boundary. It is not a copied schedule or a workload-specific
hand implementation. This isolates the solver's fusion decision while keeping
the Torch graph, named ABI, arithmetic, and target capabilities identical.
"""

from __future__ import annotations

import importlib
import json
import os
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.static_mixed import (
    StaticFp32DeepLinearBlend,
    StaticFp32DenseSwiGlu,
    StaticFp32FeatureBlend,
)
from torch import nn

from pto_fusebox import (
    RegionSolveResult,
    bind_emitted_call,
    can_emit_region,
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
class FeatureCase:
    name: str
    module: nn.Module
    shapes: tuple[tuple[int, int], ...]


def _deep_blend() -> StaticFp32DeepLinearBlend:
    module = StaticFp32DeepLinearBlend(hidden_size=160).eval()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        module.sink.weight.copy_(
            torch.randn(module.sink.weight.shape, generator=generator) * 0.1
        )
    return module


CASES = (
    FeatureCase(
        "dense_swiglu_64x96x192",
        StaticFp32DenseSwiGlu().eval(),
        ((64, 96), (96, 192), (96, 192), (192, 96)),
    ),
    FeatureCase(
        "feature_blend_128x128x256",
        StaticFp32FeatureBlend().eval(),
        ((128, 128), (128, 256), (128, 256), (256, 128)),
    ),
    FeatureCase(
        "deep_linear_blend_256x160x320",
        _deep_blend(),
        ((256, 160), (160, 320), (160, 320), (320, 160)),
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
    return int(raw)


def _args(case: FeatureCase, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
        for shape in case.shapes
    )


def _candidate_region(
    selected: RegionSolveResult, solution: Mapping[str, object]
) -> RegionSolveResult:
    return replace(selected, solution=solution, candidate_summaries=())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_selected_feature_plan_against_best_alternative(
    case: FeatureCase,
    tmp_path: Path,
) -> None:
    """Check both partitions on multiple seeds, then time in both orders."""

    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    runtime = importlib.import_module("pypto.runtime")
    initial_args = _args(case, 0)
    graph = export_and_normalize(case.module, initial_args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=int(os.environ.get("PTO_FUSEBOX_SOLVER_WORKERS", "2")),
        require_source_codegen=True,
        collect_candidate_summaries=True,
    )
    assert solved.regions_solved == len(solved.regions) == 1
    region = solved.regions[0]
    assert region.solution is not None
    candidates = tuple(
        candidate for candidate in region.candidate_summaries if candidate.source_ready
    )
    assert candidates and candidates[0].selected
    runner_up = next(
        candidate
        for candidate in candidates[1:]
        if candidate.partition != candidates[0].partition
    )
    selected_region = _candidate_region(region, candidates[0].solution)
    runner_up_region = _candidate_region(region, runner_up.solution)
    assert can_emit_region(graph, selected_region)
    assert can_emit_region(graph, runner_up_region)
    selected_source = emit_pypto_region(
        graph, selected_region, program_name=f"selected_{case.name}"
    )
    runner_up_source = emit_pypto_region(
        graph, runner_up_region, program_name=f"runner_up_{case.name}"
    )

    selected_config = runtime.RunConfig(
        platform=os.environ.get("PTO_FUSEBOX_PLATFORM", "a2a3"),
        device_id=_device_id(),
        save_kernels=True,
        save_kernels_dir=str(tmp_path / "selected"),
        dump_passes=False,
    )
    runner_up_config = runtime.RunConfig(
        platform=os.environ.get("PTO_FUSEBOX_PLATFORM", "a2a3"),
        device_id=_device_id(),
        save_kernels=True,
        save_kernels_dir=str(tmp_path / "runner_up"),
        dump_passes=False,
    )
    selected_compiled = ir.compile(
        pl.parse_program(selected_source.source), **selected_config.compile_kwargs()
    )
    runner_up_compiled = ir.compile(
        pl.parse_program(runner_up_source.source), **runner_up_config.compile_kwargs()
    )

    seed_count = int(os.environ.get("PTO_FUSEBOX_DEVICE_SEEDS", "5"))
    assert seed_count > 0
    selected_dispatch: tuple[object, ...] | None = None
    runner_up_dispatch: tuple[object, ...] | None = None
    for seed in range(seed_count):
        args = _args(case, seed)
        with torch.no_grad():
            expected = case.module(*args)
        selected_output = torch.full_like(expected, torch.nan)
        runner_up_output = torch.full_like(expected, torch.nan)
        selected_dispatch = bind_emitted_call(
            case.module, graph, selected_source, args, (selected_output,)
        )
        runner_up_dispatch = bind_emitted_call(
            case.module, graph, runner_up_source, args, (runner_up_output,)
        )
        selected_compiled(*selected_dispatch, config=selected_config)
        runner_up_compiled(*runner_up_dispatch, config=runner_up_config)
        assert torch.isfinite(selected_output).all()
        assert torch.isfinite(runner_up_output).all()
        torch.testing.assert_close(selected_output, expected, rtol=1.0e-3, atol=1.0e-3)
        torch.testing.assert_close(runner_up_output, expected, rtol=1.0e-3, atol=1.0e-3)

    assert selected_dispatch is not None and runner_up_dispatch is not None
    rounds = int(os.environ.get("PTO_FUSEBOX_PERF_ROUNDS", "30"))
    warmup = int(os.environ.get("PTO_FUSEBOX_PERF_WARMUP", "5"))
    assert rounds > 0 and warmup >= 0
    selected_first = runtime.benchmark(
        selected_compiled,
        selected_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=selected_config,
    )
    runner_up_second = runtime.benchmark(
        runner_up_compiled,
        runner_up_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=runner_up_config,
    )
    runner_up_first = runtime.benchmark(
        runner_up_compiled,
        runner_up_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=runner_up_config,
    )
    selected_second = runtime.benchmark(
        selected_compiled,
        selected_dispatch,
        rounds=rounds,
        warmup=warmup,
        config=selected_config,
    )
    selected_times = selected_first.device_wall_us + selected_second.device_wall_us
    runner_up_times = runner_up_first.device_wall_us + runner_up_second.device_wall_us
    assert len(selected_times) == len(runner_up_times) == 2 * rounds
    assert min(selected_times) > 0.0 and min(runner_up_times) > 0.0
    print(
        "PTO_FUSEBOX_FEATURE_ROUND_TRIP_PERF="
        + json.dumps(
            {
                "device": _device_id(),
                "case": case.name,
                "selected_candidate": candidates[0].id,
                "selected_partition": candidates[0].partition,
                "selected_modeled_cycles": candidates[0].modeled_cost_cycles,
                "runner_up_candidate": runner_up.id,
                "runner_up_partition": runner_up.partition,
                "runner_up_modeled_cycles": runner_up.modeled_cost_cycles,
                "selected_median_device_wall_us": statistics.median(selected_times),
                "runner_up_median_device_wall_us": statistics.median(runner_up_times),
                "runner_up_over_selected": statistics.median(runner_up_times)
                / statistics.median(selected_times),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
