"""Opt-in silicon calibration for source-ready CVC group/trip candidates.

The shapes in this file were frozen using host-only realization checks.  The
test compiles and executes every source-ready candidate; it does not select
shapes or modify model coefficients from device results.  Run the same matrix
on two devices and aggregate the emitted JSON records before changing the
cost model.
"""

from __future__ import annotations

import importlib
import json
import os
import statistics
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.static_mixed import StaticAttentionCore

from pto_fusebox import (
    bind_emitted_call,
    can_emit_region,
    emit_pypto_region,
    export_and_normalize,
    scheduled_region,
    solve_graph,
)
from pto_fusebox.schedule.schema import MixedKernelPlan

if os.environ.get("PTO_FUSEBOX_RUN_DEVICE_TESTS") != "1":
    pytest.skip(
        "set PTO_FUSEBOX_RUN_DEVICE_TESTS=1 for Fusebox source silicon tests",
        allow_module_level=True,
    )


HOLDOUT_SHAPES = (
    pytest.param(((128, 32), (48, 32), (48, 48)), id="short_rectangular"),
    pytest.param(((192, 48), (64, 48), (64, 64)), id="medium_wider_k"),
    pytest.param(((256, 32), (32, 32), (32, 64)), id="long_thin"),
    pytest.param(((384, 48), (48, 48), (48, 48)), id="tall_wider_k"),
    pytest.param(((512, 32), (64, 32), (64, 32)), id="tall_square"),
    pytest.param(((640, 48), (64, 48), (64, 64)), id="deep_wider_k"),
    pytest.param(((768, 32), (48, 32), (48, 48)), id="deep_rectangular"),
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


def _args(shapes: tuple[tuple[int, ...], ...], seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
        for shape in shapes
    )


@pytest.mark.parametrize("shapes", HOLDOUT_SHAPES)
def test_every_source_ready_cvc_candidate_on_silicon(
    shapes: tuple[tuple[int, ...], ...],
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Check correctness and report timing for every frozen CVC candidate."""

    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    runtime = importlib.import_module("pypto.runtime")
    module = StaticAttentionCore().eval()
    initial_args = _args(shapes, 0)
    graph = export_and_normalize(module, initial_args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=int(os.environ.get("PTO_FUSEBOX_SOLVER_WORKERS", "2")),
        require_source_codegen=True,
        collect_candidate_summaries=True,
    )
    assert solved.regions_solved and len(solved.regions) == 1
    region = solved.regions[0]
    candidates = tuple(
        candidate for candidate in region.candidate_summaries if candidate.source_ready
    )
    assert len(candidates) >= 3

    seed_count = int(os.environ.get("PTO_FUSEBOX_DEVICE_SEEDS", "3"))
    rounds = int(os.environ.get("PTO_FUSEBOX_PERF_ROUNDS", "30"))
    warmup = int(os.environ.get("PTO_FUSEBOX_PERF_WARMUP", "5"))
    assert seed_count > 0 and rounds > 0 and warmup >= 0
    records: list[dict[str, object]] = []

    for candidate in candidates:
        forced = replace(region, solution=candidate.solution)
        assert can_emit_region(graph, forced)
        plan = scheduled_region(forced).steps[0].plan
        assert isinstance(plan, MixedKernelPlan)
        emitted = emit_pypto_region(
            graph,
            forced,
            program_name=f"cvc_{request.node.callspec.id}_candidate_{candidate.id}",
        )
        config = runtime.RunConfig(
            platform=os.environ.get("PTO_FUSEBOX_PLATFORM", "a2a3"),
            device_id=_device_id(),
            save_kernels=True,
            save_kernels_dir=str(tmp_path / f"candidate_{candidate.id}"),
            dump_passes=False,
        )
        compiled = ir.compile(
            pl.parse_program(emitted.source), **config.compile_kwargs()
        )

        dispatch = None
        output = None
        for seed in range(seed_count):
            args = _args(shapes, seed)
            with torch.no_grad():
                expected = module(*args)
            output = torch.full_like(expected, torch.nan)
            dispatch = bind_emitted_call(module, graph, emitted, args, (output,))
            compiled(*dispatch, config=config)
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, expected, rtol=1.0e-4, atol=1.0e-4)

        assert dispatch is not None and output is not None
        sample = runtime.benchmark(
            compiled,
            dispatch,
            rounds=rounds,
            warmup=warmup,
            config=config,
        )
        device_wall = sample.device_wall_us
        assert len(device_wall) == rounds and min(device_wall) > 0.0
        records.append(
            {
                "candidate_id": candidate.id,
                "selected": candidate.selected,
                "modeled_cost_cycles": candidate.modeled_cost_cycles,
                "groups": plan.active_groups,
                "trips_per_group": plan.max_trips_per_group,
                "pipeline_stages": plan.pipeline_stages,
                "cube_stage_peak_l0a_bytes": plan.cube_stage_peak_l0a_bytes,
                "cube_stage_peak_l0b_bytes": plan.cube_stage_peak_l0b_bytes,
                "source_l1_allocation_bytes": plan.source_l1_allocation_bytes,
                "median_device_wall_us": statistics.median(device_wall),
                "min_device_wall_us": min(device_wall),
                "max_device_wall_us": max(device_wall),
            }
        )

    print(
        "PTO_FUSEBOX_CVC_CALIBRATION="
        + json.dumps(
            {
                "device": _device_id(),
                "shapes": shapes,
                "records": records,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
