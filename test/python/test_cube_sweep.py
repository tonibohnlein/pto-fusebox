from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from pto_fusebox import (
    NormalizedGraph,
    RegionSolveResult,
    ScheduleContractError,
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


class SplitCubeChain(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        intermediate = torch.mm(lhs, middle)
        return torch.mm(intermediate, rhs, out_dtype=torch.float32)


class SplitCubeChainWithTwoOutputs(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        intermediate = torch.mm(lhs, middle)
        output = torch.mm(intermediate, rhs, out_dtype=torch.float32)
        return intermediate, output


class SplitCubeChainWithResidentRhs(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        shared_rhs: torch.Tensor,
        third_rhs: torch.Tensor,
        sink_rhs: torch.Tensor,
    ) -> torch.Tensor:
        first = torch.mm(lhs, shared_rhs)
        second = torch.mm(first, shared_rhs)
        third = torch.mm(second, third_rhs)
        return torch.mm(third, sink_rhs, out_dtype=torch.float32)


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


def _lowered_split_chain(
    *, m: int = 16, inner: int = 2048, n: int = 16
) -> tuple[NormalizedGraph, RegionSolveResult]:
    graph = export_and_normalize(
        SplitCubeChain(),
        (
            torch.empty(m, 64, dtype=torch.bfloat16, device="meta"),
            torch.empty(64, inner, dtype=torch.bfloat16, device="meta"),
            torch.empty(inner, n, dtype=torch.bfloat16, device="meta"),
        ),
    )
    regions = extract_solver_regions(graph)
    assert len(regions) == 1
    lowered = regions[0].lower(graph)
    return graph, RegionSolveResult(
        region=regions[0],
        status="lowered",
        problem=lowered.problem,
        solution=None,
        solver_op_to_graph=lowered.solver_op_to_graph,
        solver_tensor_to_value=lowered.solver_tensor_to_value,
        diagnostics=regions[0].diagnostics,
    )


@pytest.mark.parametrize(
    ("m", "k", "n", "source_replay_expected"),
    (
        (16, 64, 32, True),
        (256, 256, 256, True),
        (64, 272, 80, True),
        (32, 736, 64, True),
        (64, 512, 256, True),
        # The selected plan is split-K and is now a source-ready replay.
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
    assert bool(source_ready) is source_replay_expected


def test_deep_k_surface_carries_split_and_no_split_candidates() -> None:
    graph, region = _lowered_region(128, 8192, 128)
    sweep = enumerate_cube_plans(region, sweep_binary=_sweep_binary())

    assert sweep.selected.grid.split_k == 16
    split_factors = {candidate.grid.split_k for candidate in sweep.candidates}
    assert {1, 2, 4, 8, 16}.issubset(split_factors)
    for candidate in sweep.candidates:
        step = candidate.solution["steps"][0]
        if candidate.grid.split_k == 1:
            assert step["plan"]["split_merge_policy"] == "none"
        else:
            policy = step["plan"]["split_merge_policy"]
            assert policy in {
                "first_partial_then_atomic",
                "aiv_zero_seed_then_atomic",
            }
            assert step["plan"]["first_partial_then_atomic"]["present"] is (
                policy == "first_partial_then_atomic"
            )
            assert step["plan"]["aiv_zero_seed_then_atomic"]["present"] is (
                policy == "aiv_zero_seed_then_atomic"
            )

    no_split = next(
        candidate for candidate in sweep.candidates if candidate.id == "p1_q1_s1"
    )
    forced = region_for_cube_candidate(region, no_split)
    typed = scheduled_region(forced).steps[0]
    plan = typed.plan
    assert isinstance(plan, CubeKernelPlan)
    assert typed.launch.tile_k == plan.matmuls[0].k_loop.l1_window_k == 512
    assert can_emit_region(graph, forced)
    source = emit_pypto_region(graph, forced, program_name="deep_k_no_split").source
    assert "pl.spmd(1," in source
    assert "atomic=pl.AtomicType.Add" not in source

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


def test_split_cube_dag_replays_upstream_then_unique_atomic_sink() -> None:
    graph, region = _lowered_split_chain()
    sweep = enumerate_cube_plans(region, sweep_binary=_sweep_binary())
    candidate = next(item for item in sweep.candidates if item.id == "p1_q1_s2")
    forced = region_for_cube_candidate(region, candidate)
    plan = scheduled_region(forced).steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert plan.execution_order == (0, 1)
    assert [matmul.is_sink for matmul in plan.matmuls] == [False, True]
    assert plan.matmuls[0].effective_contraction == plan.matmuls[0].contraction
    assert (
        plan.matmuls[1].effective_contraction * plan.split_k
        == plan.matmuls[1].contraction
    )
    assert can_emit_region(graph, forced)

    source = emit_pypto_region(graph, forced, program_name="split_cube_chain").source

    assert source.count("pl.spmd(") == 2
    assert source.count("pl.create_l1(") == 1
    assert source.index("matmul_0_tile_0_accumulator") < source.index(
        "matmul_1_tile_0_accumulator"
    )
    assert "[0, split_index * 1024]" in source
    assert "[split_index * 1024, 0]" in source
    assert source.count("atomic=pl.AtomicType.Add") == 1
    assert "pl.assemble(matmul_0_l1" in source

    assert forced.solution is not None
    solution = copy.deepcopy(forced.solution)
    descriptor = solution["steps"][0]["plan"]
    descriptor["split_merge_policy"] = "first_partial_then_atomic"
    descriptor["first_partial_then_atomic"] = {
        "present": True,
        "first_work_units": 1,
        "atomic_work_units": 1,
        "synchronization_cycles": descriptor["aiv_zero_seed_then_atomic"][
            "synchronization_cycles"
        ],
    }
    descriptor["aiv_zero_seed_then_atomic"] = {
        "present": False,
        "seed_work_units": 0,
        "atomic_work_units": 0,
        "seed_bytes": 0,
        "synchronization_cycles": 0.0,
    }
    first_partial = emit_pypto_region(
        graph,
        replace(forced, solution=solution),
        program_name="split_cube_chain_first_partial",
    ).source
    assert first_partial.count("pl.spmd(") == 2
    assert first_partial.count("pl.create_l1(") == 2
    assert first_partial.count("atomic=pl.AtomicType.Add") == 1
    assert "pl.full(" not in first_partial


def test_split_cube_dag_retains_a_boundary_panel_inside_each_share() -> None:
    graph, region = _lowered_split_chain(m=512, n=512)
    sweep = enumerate_cube_plans(region, sweep_binary=_sweep_binary())
    candidate = next(item for item in sweep.candidates if item.id == "p1_q1_s8")
    forced = region_for_cube_candidate(region, candidate)
    plan = scheduled_region(forced).steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert plan.matmuls[1].retained_panels.rhs
    assert can_emit_region(graph, forced)

    source = emit_pypto_region(graph, forced, program_name="split_retained_rhs").source

    retained = "matmul_1_rhs_retained = pl.slice("
    first_output_tile = "matmul_1_tile_0_rhs_init_0 = pl.slice("
    assert source.count(retained) == 1
    assert source.index(retained) < source.index(first_output_tile)
    assert "split_index * 256" in source


def test_split_cube_dag_materializes_a_resident_operand_once_per_share() -> None:
    arguments = tuple(
        torch.empty(64, 64, dtype=torch.bfloat16, device="meta") for _ in range(4)
    )
    graph = export_and_normalize(SplitCubeChainWithResidentRhs(), arguments)
    regions = extract_solver_regions(graph)
    assert len(regions) == 1
    lowered = regions[0].lower(graph)
    region = RegionSolveResult(
        region=regions[0],
        status="lowered",
        problem=lowered.problem,
        solution=None,
        solver_op_to_graph=lowered.solver_op_to_graph,
        solver_tensor_to_value=lowered.solver_tensor_to_value,
        diagnostics=regions[0].diagnostics,
    )
    sweep = enumerate_cube_plans(region, sweep_binary=_sweep_binary())
    candidate = next(item for item in sweep.candidates if item.id == "p1_q1_s2")
    forced = region_for_cube_candidate(region, candidate)
    plan = scheduled_region(forced).steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert len(plan.resident_boundaries) == 1
    assert plan.resident_boundaries[0].use_count == 2
    assert can_emit_region(graph, forced)

    source = emit_pypto_region(graph, forced, program_name="split_resident_rhs").source

    resident = "resident_0_rhs = pl.slice("
    assert source.count(resident) == 1
    assert "matmul_0_tile_0_rhs_init = pl.slice(resident_0_rhs" in source
    assert "matmul_1_tile_0_rhs_init = pl.slice(resident_0_rhs" in source
    assert source.count("atomic=pl.AtomicType.Add") == 1


def test_split_cube_dag_rejects_a_second_split_accumulator() -> None:
    graph, region = _lowered_split_chain()
    sweep = enumerate_cube_plans(region, sweep_binary=_sweep_binary())
    candidate = next(item for item in sweep.candidates if item.id == "p1_q1_s2")
    forced = region_for_cube_candidate(region, candidate)
    assert forced.solution is not None
    solution = copy.deepcopy(forced.solution)
    matmuls = solution["steps"][0]["plan"]["matmuls"]
    duplicate = copy.deepcopy(matmuls[1])
    duplicate["instance"] = 1
    duplicate["is_sink"] = False
    duplicate["final_drain"]["target_l1"] = True
    duplicate["final_drain"]["atomic"] = False
    matmuls[1]["instance"] = 2
    matmuls.insert(1, duplicate)
    solution["steps"][0]["plan"]["execution_order"] = [0, 1, 1]

    with pytest.raises(SourceEmissionError, match="exactly one split accumulator"):
        emit_pypto_region(graph, replace(forced, solution=solution))


def test_cube_sweep_rejects_a_multi_output_split_group() -> None:
    graph = export_and_normalize(
        SplitCubeChainWithTwoOutputs(),
        (
            torch.empty(16, 64, dtype=torch.bfloat16, device="meta"),
            torch.empty(64, 2048, dtype=torch.bfloat16, device="meta"),
            torch.empty(2048, 16, dtype=torch.bfloat16, device="meta"),
        ),
    )
    regions = extract_solver_regions(graph)
    assert len(regions) == 1
    lowered = regions[0].lower(graph)
    region = RegionSolveResult(
        region=regions[0],
        status="lowered",
        problem=lowered.problem,
        solution=None,
        solver_op_to_graph=lowered.solver_op_to_graph,
        solver_tensor_to_value=lowered.solver_tensor_to_value,
        diagnostics=regions[0].diagnostics,
    )

    with pytest.raises(RuntimeError, match="homogeneous MatMul DAG"):
        enumerate_cube_plans(region, sweep_binary=_sweep_binary())


def test_deep_k_split_source_uses_one_dependency_between_parallel_phases() -> None:
    graph, region = _lowered_region(128, 8192, 128)
    sweep = enumerate_cube_plans(region, sweep_binary=_sweep_binary())
    selected = region_for_cube_candidate(region, sweep.selected)

    source = emit_pypto_region(graph, selected, program_name="deep_k_zero_seed").source
    assert "with pl.spmd(1," in source
    assert "as zero_seed_task:" in source
    assert "with pl.spmd(16," in source
    assert "deps=[zero_seed_task]" in source
    assert "split_index = split_work_index % 16" in source
    assert "pl.full([128, 128], dtype=pl.FP32, value=0.0)" in source
    assert source.count("atomic=pl.AtomicType.Add") == 1
    assert "split_index * 512" in source
    assert "while " not in source and "pl.system" not in source

    # Exercise the alternative policy against the identical typed child plan.
    assert selected.solution is not None
    solution = copy.deepcopy(selected.solution)
    plan = solution["steps"][0]["plan"]
    spatial = plan["spatial_tiles"]
    split = plan["split_k"]
    plan["split_merge_policy"] = "first_partial_then_atomic"
    plan["first_partial_then_atomic"] = {
        "present": True,
        "first_work_units": spatial,
        "atomic_work_units": spatial * (split - 1),
        "synchronization_cycles": plan["aiv_zero_seed_then_atomic"][
            "synchronization_cycles"
        ],
    }
    plan["aiv_zero_seed_then_atomic"] = {
        "present": False,
        "seed_work_units": 0,
        "atomic_work_units": 0,
        "seed_bytes": 0,
        "synchronization_cycles": 0.0,
    }
    first_partial = replace(selected, solution=solution)
    first_source = emit_pypto_region(
        graph, first_partial, program_name="deep_k_first_partial"
    ).source
    assert "as first_partial_task:" in first_source
    assert "deps=[first_partial_task]" in first_source
    assert "split_index = split_work_index % 15 + 1" in first_source
    assert "pl.full(" not in first_source
    assert first_source.count("atomic=pl.AtomicType.Add") == 1


def test_deep_k_split_contract_rejects_malformed_policy_descriptors() -> None:
    _, region = _lowered_region(128, 8192, 128)
    sweep = enumerate_cube_plans(region, sweep_binary=_sweep_binary())
    selected = region_for_cube_candidate(region, sweep.selected)
    assert selected.solution is not None

    wrong_count = copy.deepcopy(selected.solution)
    wrong_count["steps"][0]["plan"]["aiv_zero_seed_then_atomic"][
        "atomic_work_units"
    ] += 1

    both_present = copy.deepcopy(selected.solution)
    both_present["steps"][0]["plan"]["first_partial_then_atomic"]["present"] = True

    wrong_policy = copy.deepcopy(selected.solution)
    wrong_policy["steps"][0]["plan"]["split_merge_policy"] = "first_partial_then_atomic"

    for malformed in (wrong_count, both_present, wrong_policy):
        with pytest.raises(ScheduleContractError, match="inconsistent"):
            scheduled_region(replace(selected, solution=malformed))


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
