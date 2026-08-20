from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from pto_fusebox import (
    NormalizedGraph,
    RegionSolveResult,
    SourceEmissionError,
    can_emit_region,
    emit_pypto_region,
    enumerate_cube_plans,
    export_and_normalize,
    extract_solver_regions,
    region_for_cube_candidate,
    scheduled_region,
)
from pto_fusebox.schedule import CubeKernelPlan


class Matmul(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.mm(lhs, rhs)


def _sweep_binary() -> Path:
    path = Path(__file__).parents[2] / "build" / "cube_plan_sweep"
    if not path.is_file():
        pytest.fail(f"cube plan sweep binary does not exist: {path}")
    return path


def _lowered_region(
    m: int, k: int, n: int
) -> tuple[NormalizedGraph, RegionSolveResult]:
    graph = export_and_normalize(
        Matmul(),
        (torch.zeros((m, k)), torch.zeros((k, n))),
    )
    regions = extract_solver_regions(graph)
    assert len(regions) == 1
    lowered = regions[0].lower(graph)
    result = RegionSolveResult(
        region=regions[0],
        status="lowered",
        problem=lowered.problem,
        solution=None,
        solver_op_to_graph=lowered.solver_op_to_graph,
        solver_tensor_to_value=lowered.solver_tensor_to_value,
        diagnostics=regions[0].diagnostics,
    )
    return graph, result


@pytest.mark.parametrize(
    ("m", "k", "n", "source_replay_expected"),
    (
        (16, 64, 32, True),
        (256, 256, 256, True),
        (64, 272, 80, True),
        (32, 736, 64, True),
        (64, 512, 256, True),
        # The selected plan is split-K and remains source-model-ahead, while
        # non-split candidates for the same problem are valid source replays.
        (128, 8192, 128, True),
    ),
)
def test_cube_model_surface_enumerates_replayable_forced_solutions(
    m: int, k: int, n: int, source_replay_expected: bool
) -> None:
    graph, region = _lowered_region(m, k, n)
    sweep = enumerate_cube_plans(region, sweep_binary=_sweep_binary())

    assert sweep.selected.selected
    assert sweep.selected.latency_cycles == min(
        candidate.latency_cycles for candidate in sweep.candidates
    )
    assert all(candidate.grid.work_units > 0 for candidate in sweep.candidates)

    source_ready = []
    for candidate in sweep.candidates:
        forced = region_for_cube_candidate(region, candidate)
        if can_emit_region(graph, forced):
            assert scheduled_region(forced).steps[0]
            source_ready.append(candidate)
            source = emit_pypto_region(
                graph,
                forced,
                program_name=f"forced_{candidate.id}",
            ).source
            assert "pl.spmd(" in source
            assert "auto_tile" not in source
            assert "auto_fuse" not in source
        if candidate.grid.split_k > 1:
            assert candidate.uses_model_ahead_split_k
            assert not can_emit_region(graph, forced)
    assert bool(source_ready) is source_replay_expected


def test_deep_k_surface_carries_split_and_no_split_candidates() -> None:
    _, region = _lowered_region(128, 8192, 128)
    sweep = enumerate_cube_plans(region, sweep_binary=_sweep_binary())

    assert sweep.selected.grid.split_k == 16
    split_factors = {candidate.grid.split_k for candidate in sweep.candidates}
    assert {1, 2, 4, 8, 16}.issubset(split_factors)
    for candidate in sweep.candidates:
        step = candidate.solution["steps"][0]
        if candidate.grid.split_k == 1:
            assert step["plan"]["split_merge_policy"] == "none"
        else:
            assert step["plan"]["split_merge_policy"] == "first_partial_then_atomic"

    # The outer L1 window and nested L0 loop are different hierarchy levels.
    # The selected S=16 plan covers a 160-wide L1 window with two 64-wide L0
    # iterations plus a 32-wide L0 tail; typed parsing must preserve that
    # decomposition rather than compare the L0 tile directly with 160.
    selected = region_for_cube_candidate(region, sweep.selected)
    plan = scheduled_region(selected).steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    matmul = plan.matmuls[0]
    assert matmul.k_loop.chunk == 160
    assert matmul.output_variants[0].l0_init.tile[2] == 64
    l0_loop = matmul.output_variants[0].l0_init.k_loop
    assert l0_loop.full_chunks * l0_loop.chunk + l0_loop.tail == 160


def test_balanced_surface_excludes_outer_pipeline_l0_overflow() -> None:
    graph, region = _lowered_region(256, 256, 256)
    sweep = enumerate_cube_plans(region, sweep_binary=_sweep_binary())
    candidates = {candidate.id: candidate for candidate in sweep.candidates}

    # These grids require a 40,960-byte stationary operand frame. PyPTO's
    # stage-2 outer K loop rotates that frame twice, so the lowered 81,920-byte
    # L0A/L0B allocation exceeds the 64-KiB hardware capacity. Keep the
    # analytic candidates, but reject them at the exact source-readiness
    # boundary before PyPTO compilation.
    for candidate_id in ("p2_q4_s1", "p4_q2_s1"):
        forced = region_for_cube_candidate(region, candidates[candidate_id])
        assert not can_emit_region(graph, forced)
        with pytest.raises(
            SourceEmissionError, match="exceeds lowered L0 operand capacity"
        ):
            emit_pypto_region(graph, forced)
    assert "p4_q4_s1" in candidates
    assert sweep.selected.id == "p4_q4_s1"
    assert can_emit_region(
        graph, region_for_cube_candidate(region, candidates["p4_q4_s1"])
    )


def test_candidate_cannot_be_rebound_to_a_different_problem() -> None:
    _, first = _lowered_region(16, 64, 32)
    _, second = _lowered_region(64, 272, 80)
    candidate = enumerate_cube_plans(first, sweep_binary=_sweep_binary()).selected

    with pytest.raises(ValueError, match="different lowered problem"):
        region_for_cube_candidate(second, candidate)


def test_selected_marker_disagreement_fails_closed(tmp_path: Path) -> None:
    _, region = _lowered_region(16, 64, 32)
    invalid = tmp_path / "invalid_sweep.py"
    invalid.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({\n"
        " 'schema_version':'pto_fusebox.cube_plan_sweep.v1',\n"
        " 'selected_candidate_id':'p1_q1_s1',\n"
        " 'candidates':[]\n"
        "}))\n",
        encoding="utf-8",
    )
    invalid.chmod(0o755)

    with pytest.raises(ValueError, match="contains no candidates"):
        enumerate_cube_plans(region, sweep_binary=invalid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
