"""Fixed candidates from the 41bac5ac silent-wrong-answer campaign.

These retain the original dense, transposed weight layout. They are not a
replacement for the latest NZ-weight native Flash-MTP ABI. Candidate identity
and geometry are asserted before compiling so a changed solver pick cannot
turn a reproduction into a test of a different schedule.
"""

from __future__ import annotations

from pathlib import Path

import torch
from examples.torch_frontend.deepseek_v4 import (
    build_production_mtp_history_projection_branch,
)
from pto_fusebox import (
    NormalizedGraph,
    RegionSolveResult,
    enumerate_cube_plans,
    export_and_normalize,
    extract_solver_regions,
    region_for_cube_candidate,
    region_for_source_candidate,
    solve_graph,
)
from torch import nn


CASE_NAMES = (
    "int8_m32_n4096_k4096",
    "flashmtp_history_candidate_0",
    "flashmtp_history_candidate_1",
    "flashmtp_history_candidate_2_control",
)
SEEDS = (11, 23, 37, 53, 71)


class Int8MatmulNK(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(
            torch.empty(4096, 4096, dtype=torch.int8, device="meta"),
            requires_grad=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten._int_mm.default(value, self.w.t())


def frozen_case(
    name: str, solver: Path
) -> tuple[nn.Module, tuple[torch.Tensor, ...], NormalizedGraph, RegionSolveResult]:
    """Recover one named candidate, never substitute the current selected plan."""

    if name not in CASE_NAMES:
        raise ValueError(f"unknown wrong-answer regression {name!r}")
    module: nn.Module
    args: tuple[torch.Tensor, ...]
    if name == CASE_NAMES[0]:
        module = Int8MatmulNK()
        args = (torch.empty(32, 4096, dtype=torch.int8, device="meta"),)
        graph = export_and_normalize(module, args)
        regions = extract_solver_regions(graph)
        assert len(regions) == 1
        lowered = regions[0].lower(graph)
        unsolved = RegionSolveResult(
            region=regions[0],
            status="lowered",
            problem=lowered.problem,
            solution=None,
            solver_op_to_graph=lowered.solver_op_to_graph,
            solver_tensor_to_value=lowered.solver_tensor_to_value,
            diagnostics=regions[0].diagnostics,
        )
        sweep = enumerate_cube_plans(
            unsolved,
            sweep_binary=solver.parent / "cube_plan_sweep",
            source_oriented=True,
        )
        cube_matches = [c for c in sweep.candidates if c.id == "p1_q8_s1_k64_i64"]
        assert len(cube_matches) == 1, (
            "the original wrong-answer cube candidate is absent"
        )
        cube_candidate = cube_matches[0]
        assert tuple(cube_candidate.geometry.physical_l0_tile) == (32, 512, 64)
        region = region_for_cube_candidate(unsolved, cube_candidate)
    else:
        module, args = build_production_mtp_history_projection_branch()
        graph = export_and_normalize(module, args)
        solved = solve_graph(
            graph,
            solver_binary=solver,
            solver_workers=2,
            require_source_codegen=True,
            collect_candidate_summaries=True,
        )
        assert solved.successful and len(solved.regions) == 1
        base = solved.regions[0]
        candidate_id = "candidate_" + name.split("candidate_")[1].split("_")[0]
        matches = [c for c in base.candidate_summaries if c.id == candidate_id]
        assert len(matches) == 1, "the original history candidate is absent"
        candidate = matches[0]
        partitions = {
            "candidate_0": (tuple(range(18)), (18,), (19, 20, 21)),
            "candidate_1": (
                tuple(range(8)),
                tuple(range(8, 13)),
                tuple(range(13, 17)),
                (18,),
                (17, 19, 20, 21),
            ),
            "candidate_2": (tuple(range(22)),),
        }
        assert candidate.partition == partitions[candidate_id]
        launches = {
            "candidate_0": (
                ("vector", 32, (32, 1), (4096, 1, 1)),
                ("cube", 8, (1, 8), (512, 32, 64)),
                ("vector", 8, (1, 8), (512, 32, 1)),
            ),
            "candidate_1": (
                ("vector", 8, (8, 1), (4096, 4, 1)),
                ("vector", 8, (8, 1), (1, 4, 1)),
                ("vector", 8, (1, 8), (512, 32, 1)),
                ("cube", 8, (1, 8), (512, 32, 64)),
                ("vector", 8, (1, 8), (512, 32, 1)),
            ),
            "candidate_2": (("mixed", 48, (1, 32), (128, 32, 256)),),
        }
        assert (
            tuple(
                (
                    s["kind"],
                    s["launch"]["cores"],
                    tuple(s["launch"]["parts"]),
                    tuple(s["launch"]["tile"]),
                )
                for s in candidate.schedule
            )
            == launches[candidate_id]
        )
        if candidate_id in {"candidate_0", "candidate_1"}:
            cube_steps = [s for s in candidate.schedule if s["kind"] == "cube"]
            assert len(cube_steps) == 1
            assert tuple(cube_steps[0]["launch"]["tile"]) == (512, 32, 64)
            expected = 3 if candidate_id == "candidate_0" else 5
            assert candidate.execution.submissions == expected
        else:
            assert len(candidate.schedule) == 1
            step = candidate.schedule[0]
            assert step["kind"] == "mixed"
            assert tuple(step["launch"]["tile"]) == (128, 32, 256)
        region = region_for_source_candidate(base, candidate)
    return module, args, graph, region


def seeded_inputs(
    name: str, module: nn.Module, args: tuple[torch.Tensor, ...], seed: int
) -> tuple[nn.Module, tuple[torch.Tensor, ...], torch.Tensor]:
    """Generate nonuniform ABI-matching inputs and a non-vacuous reference."""

    generator = torch.Generator().manual_seed(seed)
    real = tuple(
        torch.randn(tuple(a.shape), dtype=a.dtype, generator=generator)
        if a.dtype.is_floating_point
        else torch.randint(
            -127, 128, tuple(a.shape), dtype=a.dtype, generator=generator
        )
        for a in args
    )
    if name == CASE_NAMES[0]:
        module = module.to_empty(device="cpu")
        weight = module.get_parameter("w")
        with torch.no_grad():
            weight.copy_(
                torch.randint(
                    -127, 128, weight.shape, dtype=torch.int8, generator=generator
                )
            )
        reference = real[0].double() @ weight.double().t()
    else:
        with torch.no_grad():
            reference = module(*real)
    assert reference.shape == (32, 4096)
    assert torch.isfinite(reference).all() and torch.count_nonzero(reference) > 0
    return module, real, reference
