"""Opt-in end-to-end checks against an independently installed PyPTO/PTOAS.

The standalone source backend intentionally does not depend on PyPTO.  Set
``PTO_FUSEBOX_PYPTO_INTEGRATION=1`` in an environment whose ``pypto`` import and
``PTOAS_ROOT`` point at the versions under validation to exercise this gate.
"""

from __future__ import annotations

import copy
import importlib
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.pr2335_vector import (
    build_examples as build_pr2335_examples,
)
from pto_fusebox import (
    RegionSolveResult,
    emit_pypto_region,
    enumerate_cube_plans,
    export_and_normalize,
    extract_solver_regions,
    region_for_cube_candidate,
    scheduled_region,
    solve_graph,
)
from pto_fusebox.schedule.schema import (
    CubeKernelPlan,
    MixedKernelPlan,
    VectorKernelPlan,
)
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


class _DeepKMatmul(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.mm(lhs, rhs)


class _SplitCubeChain(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        intermediate = torch.mm(lhs, middle)
        return torch.mm(intermediate, rhs, out_dtype=torch.float32)


class _ChainedMatmul(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(torch.mm(lhs, middle), rhs)


class _DiamondMatmul(nn.Module):
    def forward(
        self,
        shared: torch.Tensor,
        lhs_weight: torch.Tensor,
        rhs_weight: torch.Tensor,
    ) -> torch.Tensor:
        lhs = torch.mm(shared, lhs_weight)
        rhs = torch.mm(shared, rhs_weight)
        return torch.mm(lhs, rhs)


class _FanoutMatmul(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        first_rhs: torch.Tensor,
        second_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.mm(lhs, middle)
        return torch.mm(shared, first_rhs), torch.mm(shared, second_rhs)


class _WideSoftmax(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.softmax(value, dim=-1)


class _Silu(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.reciprocal(torch.exp(value * -1.0) + 1.0)


class _LayerNorm(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        centered = value - value.mean(dim=-1, keepdim=True)
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(variance + 1.0e-5)
        return normalized * gamma + beta


class _StreamedReduction(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sum(value, dim=-1, keepdim=True)


class _StreamedNormalize(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value / torch.sum(value, dim=-1, keepdim=True)


class _NamingCollision(nn.Module):
    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        return torch.exp(pl)


class _C2VEpilogue(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(value, weight) + bias


class _AttentionCore(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.mm(query, key.t())
        probabilities = torch.softmax(scores, dim=-1)
        return torch.mm(probabilities, value)


class _DenseSwiGlu(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.mm(value, gate_weight, out_dtype=torch.float32)
        up = torch.mm(value, up_weight, out_dtype=torch.float32)
        activation = (gate * torch.reciprocal(torch.exp(-gate) + 1.0) * up).to(
            torch.bfloat16
        )
        return torch.mm(activation, down_weight, out_dtype=torch.float32)


def _solver() -> Path:
    configured = os.environ.get("PTO_FUSEBOX_TEST_SOLVER")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "build" / "mlsys_mixed"


def _assert_static_vector_frames(pto: str) -> None:
    assert re.search(r"partition_tensor_view<[^>]*\?", pto) is None
    assert re.search(r"valid_(?:row|col) = %arg[0-9]+", pto) is None


def _assert_single_spmd_orchestration(source: str, work_units: int) -> None:
    submits = re.findall(r"\brt_submit_ai[cv]_task\(", source)
    assert len(submits) == 1
    assert source.count("launch_spec.set_block_num(") == 1
    assert f"launch_spec.set_block_num({work_units});" in source
    assert "region_index" not in source


def _compile_source(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    skip_ptoas: bool,
) -> tuple[str, int]:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")

    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(module, args)
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    assert solved.regions_solved == 1
    assert len(solved.regions) == 1
    plan = scheduled_region(solved.regions[0]).steps[0].plan
    assert isinstance(plan, (CubeKernelPlan, VectorKernelPlan))

    source = emit_pypto_region(graph, solved.regions[0], program_name=name).source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / name),
        dump_passes=False,
        skip_ptoas=skip_ptoas,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    assert len(pto_files) == 1
    generated_cpp = list((compiled.output_dir / "ptoas").glob("*.cpp"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(orchestration_files) == 1
    _assert_single_spmd_orchestration(
        orchestration_files[0].read_text(encoding="utf-8"),
        plan.work_units,
    )
    return pto_files[0].read_text(encoding="utf-8"), len(generated_cpp)


def _compile_mixed_source(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, str, MixedKernelPlan]:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")

    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(module, args)
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    assert solved.regions_solved == 1
    assert len(solved.regions) == 1
    plan = scheduled_region(solved.regions[0]).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)

    source = emit_pypto_region(graph, solved.regions[0], program_name=name).source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / name),
        dump_passes=False,
        skip_ptoas=True,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 1
    assert len(orchestration_files) == 1
    orchestration = orchestration_files[0].read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_task(") == 1
    assert orchestration.count("launch_spec.set_block_num(") == 1
    assert f"launch_spec.set_block_num({plan.active_groups});" in orchestration
    assert "for (region_index" not in orchestration
    return source, pto_files[0].read_text(encoding="utf-8"), plan


@pytest.mark.parametrize(
    ("name", "module", "args", "dir_mask", "slot_size", "slot_count"),
    [
        (
            "mixed_c2v",
            _C2VEpilogue(),
            (
                torch.zeros(32, 64),
                torch.zeros(64, 32),
                torch.zeros(1, 32),
            ),
            1,
            4096,
            8,
        ),
        (
            "mixed_attention",
            _AttentionCore(),
            (
                torch.zeros(96, 64),
                torch.zeros(64, 64),
                torch.zeros(64, 128),
            ),
            3,
            24576,
            4,
        ),
        (
            "mixed_dense_swiglu",
            _DenseSwiGlu(),
            (
                torch.zeros(128, 64, dtype=torch.bfloat16),
                torch.zeros(64, 128, dtype=torch.bfloat16),
                torch.zeros(64, 128, dtype=torch.bfloat16),
                torch.zeros(128, 64, dtype=torch.bfloat16),
            ),
            3,
            4096,
            4,
        ),
    ],
)
def test_mixed_source_lowers_through_the_pypto_split_pipeline(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    dir_mask: int,
    slot_size: int,
    slot_count: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, pto, _ = _compile_mixed_source(name, module, args, tmp_path, monkeypatch)

    assert "pl.split(pl.SplitMode.UP_DOWN" in source
    assert pto.count("pto.kernel_kind = #pto.kernel_kind<cube>") == 1
    assert pto.count("pto.kernel_kind = #pto.kernel_kind<vector>") == 1
    pipe = (
        f"{{dir_mask = {dir_mask}, slot_size = {slot_size}, slot_num = {slot_count}}}"
    )
    assert f"pto.aic_initialize_pipe {pipe}" in pto
    assert f"pto.aiv_initialize_pipe {pipe}" in pto
    assert "pto.tpush_to_aiv" in pto
    assert "pto.tpop_from_aic" in pto
    assert "pto.tfree_from_aic" in pto
    if dir_mask == 3:
        assert "pto.tpush_to_aic" in pto
        assert "pto.tpop_from_aiv" in pto
        assert "pto.tfree_from_aiv" in pto
    else:
        assert "pto.tpush_to_aic" not in pto
        assert "pto.tpop_from_aiv" not in pto


@pytest.mark.parametrize(
    ("name", "module", "args", "expected_pto_op", "static_vector_frames"),
    [
        (
            "pointwise_chain",
            _PointwiseChain(),
            (torch.zeros(96, 320), torch.ones(96, 320)),
            "pto.texp",
            True,
        ),
        (
            "sum_of_squares",
            _SumOfSquares(),
            (torch.ones(128, 1024),),
            "pto.trowsum",
            True,
        ),
        (
            "matmul_with_tail",
            _MatmulWithTail(),
            (torch.zeros(64, 272), torch.zeros(272, 80)),
            "pto.tmatmul.acc",
            False,
        ),
        (
            "chained_matmul",
            _ChainedMatmul(),
            (
                torch.zeros(64, 128, dtype=torch.bfloat16),
                torch.zeros(128, 96, dtype=torch.bfloat16),
                torch.zeros(96, 80, dtype=torch.bfloat16),
            ),
            "pto.tinsert",
            False,
        ),
        (
            "wide_softmax",
            _WideSoftmax(),
            (torch.zeros(32, 8192),),
            "pto.trowmax",
            True,
        ),
        (
            "silu",
            _Silu(),
            (torch.zeros(512, 256),),
            "pto.trecip",
            True,
        ),
        (
            "layer_norm",
            _LayerNorm(),
            (
                torch.zeros(512, 256),
                torch.ones(1, 256),
                torch.zeros(1, 256),
            ),
            "pto.trowsum",
            True,
        ),
        (
            "naming_collision",
            _NamingCollision(),
            (torch.zeros(64, 128),),
            "pto.texp",
            True,
        ),
        (
            "streamed_reduction",
            _StreamedReduction(),
            (torch.ones(5, 32771),),
            "pto.trowsum",
            True,
        ),
        (
            "streamed_normalize",
            _StreamedNormalize(),
            (torch.ones(5, 32771),),
            "pto.trowsum",
            True,
        ),
    ],
)
def test_generated_source_compiles_through_pypto_and_ptoas(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    expected_pto_op: str,
    static_vector_frames: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pto, generated_cpp = _compile_source(
        name,
        module,
        args,
        tmp_path,
        monkeypatch,
        skip_ptoas=False,
    )
    assert generated_cpp == 1
    assert expected_pto_op in pto
    if static_vector_frames:
        _assert_static_vector_frames(pto)


@pytest.mark.parametrize(
    "name",
    [
        "pr2335_softmax_512x256",
        "pr2335_softmax_256x512",
        "pr2335_softmax_128x1024",
        "pr2335_softmax_32x8192",
        "pr2335_rms_norm",
        "pr2335_layer_norm",
        "pr2335_silu",
    ],
)
def test_pr2335_vector_surface_lowers_static_frames(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = build_pr2335_examples()[name]
    pto, _ = _compile_source(
        name,
        module,
        args,
        tmp_path,
        monkeypatch,
        skip_ptoas=True,
    )
    _assert_static_vector_frames(pto)


@pytest.mark.parametrize(
    "policy", ["aiv_zero_seed_then_atomic", "first_partial_then_atomic"]
)
def test_deep_k_split_protocol_lowers_as_two_dependency_linked_tasks(
    policy: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _DeepKMatmul(),
        (torch.zeros(128, 8192), torch.zeros(8192, 128)),
    )
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    region = solved.regions[0]
    assert region.solution is not None
    solution = copy.deepcopy(region.solution)
    plan = solution["steps"][0]["plan"]
    assert plan["split_k"] > 1
    spatial = plan["spatial_tiles"]
    split = plan["split_k"]
    synchronization = max(
        plan["first_partial_then_atomic"]["synchronization_cycles"],
        plan["aiv_zero_seed_then_atomic"]["synchronization_cycles"],
    )
    if policy == "first_partial_then_atomic":
        plan["split_merge_policy"] = policy
        plan["first_partial_then_atomic"] = {
            "present": True,
            "first_work_units": spatial,
            "atomic_work_units": spatial * (split - 1),
            "synchronization_cycles": synchronization,
        }
        plan["aiv_zero_seed_then_atomic"] = {
            "present": False,
            "seed_work_units": 0,
            "atomic_work_units": 0,
            "seed_bytes": 0,
            "synchronization_cycles": 0.0,
        }
    else:
        sink = next(matmul for matmul in plan["matmuls"] if matmul["is_sink"])
        plan["split_merge_policy"] = policy
        plan["first_partial_then_atomic"] = {
            "present": False,
            "first_work_units": 0,
            "atomic_work_units": 0,
            "synchronization_cycles": 0.0,
        }
        plan["aiv_zero_seed_then_atomic"] = {
            "present": True,
            "seed_work_units": spatial,
            "atomic_work_units": spatial * split,
            "seed_bytes": spatial * sink["final_drain"]["bytes"],
            "synchronization_cycles": synchronization,
        }
    region = replace(region, solution=solution)
    typed = scheduled_region(region).steps[0]
    assert isinstance(typed.plan, CubeKernelPlan)
    source = emit_pypto_region(graph, region, program_name=f"deep_k_{policy}").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / policy),
        dump_passes=False,
        skip_ptoas=True,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 2
    assert len(orchestration_files) == 1
    orchestration = orchestration_files[0].read_text(encoding="utf-8")
    submissions = re.findall(r"\brt_submit_(ai[cv])_task\(", orchestration)
    block_counts = [
        int(value)
        for value in re.findall(
            r"launch_spec\.set_block_num\(([0-9]+)\)", orchestration
        )
    ]
    assert orchestration.count("set_dependencies(") == 1
    dependency_position = orchestration.index("set_dependencies(")
    second_submit_position = orchestration.find("rt_submit_", dependency_position)
    assert second_submit_position > dependency_position
    pto_by_name = {path.stem: path.read_text(encoding="utf-8") for path in pto_files}
    effective_k = plan["matmuls"][0]["effective_contraction"]
    assert effective_k * split == 8192
    assert {share * effective_k for share in range(split)} == set(
        range(0, 8192, effective_k)
    )
    assert f"split_index * {effective_k}" in source
    if policy == "aiv_zero_seed_then_atomic":
        assert typed.plan.aiv_zero_seed_then_atomic.present
        assert "deps=[zero_seed_task]" in source
        assert submissions == ["aiv", "aic"]
        assert block_counts == [spatial, spatial * split]
        seed_pto = pto_by_name["region0000_cube_zero_seed"]
        atomic_pto = pto_by_name["region0000_cube_atomic_all"]
        assert "pto.texpands" in seed_pto and "pto.tstore" in seed_pto
        assert "atomic_add" not in seed_pto
        assert "atomicType = #pto<atomic_type atomic_add>" in atomic_pto
    else:
        assert typed.plan.first_partial_then_atomic.present
        assert "deps=[first_partial_task]" in source
        assert submissions == ["aic", "aic"]
        assert block_counts == [spatial, spatial * (split - 1)]
        first_pto = pto_by_name["region0000_cube_first"]
        atomic_pto = pto_by_name["region0000_cube_atomic_rest"]
        assert "pto.tstore" in first_pto and "atomic_add" not in first_pto
        assert "atomicType = #pto<atomic_type atomic_add>" in atomic_pto
    assert "for (region_index" not in orchestration


def test_deep_k_no_split_candidate_compiles_through_pypto_and_ptoas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _DeepKMatmul(),
        (torch.zeros(128, 8192), torch.zeros(8192, 128)),
    )
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
        sweep_binary=_solver().parent / "cube_plan_sweep",
    )
    no_split = next(
        candidate for candidate in sweep.candidates if candidate.id == "p1_q1_s1"
    )
    region = region_for_cube_candidate(unsolved, no_split)
    typed = scheduled_region(region).steps[0]
    assert isinstance(typed.plan, CubeKernelPlan)
    assert typed.launch.tile_k == typed.plan.matmuls[0].k_loop.l1_window_k == 512

    source = emit_pypto_region(graph, region, program_name="deep_k_no_split").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "deep_k_no_split"),
        dump_passes=False,
        skip_ptoas=False,
    )

    pto_files = list(compiled.output_dir.rglob("*.pto"))
    generated_cpp = list((compiled.output_dir / "ptoas").glob("*.cpp"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == len(generated_cpp) == len(orchestration_files) == 1
    pto = pto_files[0].read_text(encoding="utf-8")
    orchestration = orchestration_files[0].read_text(encoding="utf-8")
    assert "atomic_add" not in pto
    assert orchestration.count("rt_submit_aic_task(") == 1
    assert "launch_spec.set_block_num(1);" in orchestration
    assert "set_dependencies(" not in orchestration


def test_split_cube_dag_lowers_upstream_and_sink_into_each_atomic_share(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _SplitCubeChain(),
        (
            torch.empty(16, 64, dtype=torch.bfloat16, device="meta"),
            torch.empty(64, 2048, dtype=torch.bfloat16, device="meta"),
            torch.empty(2048, 16, dtype=torch.bfloat16, device="meta"),
        ),
    )
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
        sweep_binary=_solver().parent / "cube_plan_sweep",
    )
    candidate = next(item for item in sweep.candidates if item.id == "p1_q1_s2")
    region = region_for_cube_candidate(unsolved, candidate)
    source = emit_pypto_region(graph, region, program_name="split_cube_chain").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "split_cube_chain"),
        dump_passes=False,
        skip_ptoas=True,
    )

    pto_files = list(compiled.output_dir.rglob("*.pto"))
    assert len(pto_files) == 2
    pto_by_name = {path.stem: path.read_text(encoding="utf-8") for path in pto_files}
    atomic = pto_by_name["region0000_cube_atomic_all"]
    assert atomic.count("pto.tmatmul") >= 2
    assert "atomicType = #pto<atomic_type atomic_add>" in atomic
    orchestration = next(
        (compiled.output_dir / "orchestration").glob("*.cpp")
    ).read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_aiv_task(") == 1
    assert orchestration.count("rt_submit_aic_task(") == 1
    assert orchestration.count("set_dependencies(") == 1
    assert "launch_spec.set_block_num(1);" in orchestration
    assert "launch_spec.set_block_num(2);" in orchestration


def test_cut_fp32_chain_compiles_as_two_dependency_linked_spmd_kernels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    module = _ChainedMatmul()
    args = (
        torch.zeros(64, 128),
        torch.zeros(128, 96),
        torch.zeros(96, 80),
    )
    graph = export_and_normalize(module, args)
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    region = solved.regions[0]
    schedule = scheduled_region(region)
    assert len(schedule.steps) == 2

    source = emit_pypto_region(graph, region, program_name="cut_fp32_chain").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "cut_fp32_chain"),
        dump_passes=False,
        skip_ptoas=True,
    )

    assert len(list(compiled.output_dir.rglob("*.pto"))) == 2
    orchestration = next(
        (compiled.output_dir / "orchestration").glob("*.cpp")
    ).read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_aic_task(") == 2
    assert "launch_spec.set_block_num(12);" in orchestration
    assert "launch_spec.set_block_num(4);" in orchestration
    assert "add_output(intermediate_tensor_2)" in orchestration
    assert "add_input(intermediate_tensor_2)" in orchestration
    assert "add_output(ext_output)" in orchestration
    assert "add_inout(intermediate_tensor_2)" not in orchestration
    assert "add_inout(ext_output)" not in orchestration


def test_cut_fp32_fanout_compiles_with_two_outputs_and_one_shared_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _FanoutMatmul(), tuple(torch.zeros(64, 64) for _ in range(4))
    )
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    region = solved.regions[0]
    schedule = scheduled_region(region)
    assert [step.solver_ops for step in schedule.steps] == [(0,), (1, 2)]

    source = emit_pypto_region(graph, region, program_name="cut_fp32_fanout").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "cut_fp32_fanout"),
        dump_passes=False,
        skip_ptoas=True,
    )

    assert len(list(compiled.output_dir.rglob("*.pto"))) == 2
    orchestration = next(
        (compiled.output_dir / "orchestration").glob("*.cpp")
    ).read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_aic_task(") == 2
    assert "add_output(ext_output_0)" in orchestration
    assert "add_output(ext_output_1)" in orchestration
    assert "add_output(intermediate_tensor_2)" in orchestration
    assert "add_input(intermediate_tensor_2)" in orchestration
    assert "add_inout(intermediate_tensor_2)" not in orchestration
    assert "add_inout(ext_output_0)" not in orchestration
    assert "add_inout(ext_output_1)" not in orchestration


@pytest.mark.parametrize(
    ("name", "module", "args", "expected_matmuls"),
    [
        (
            "retained_panel_matmul",
            _MatmulWithTail(),
            (
                torch.zeros(512, 64, dtype=torch.bfloat16),
                torch.zeros(64, 2048, dtype=torch.bfloat16),
            ),
            2,
        ),
        (
            "diamond_matmul",
            _DiamondMatmul(),
            tuple(torch.zeros(32, 32, dtype=torch.bfloat16) for _ in range(3)),
            3,
        ),
    ],
)
def test_non_split_cube_dag_source_compiles_through_pypto(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    expected_matmuls: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pto, _ = _compile_source(
        name,
        module,
        args,
        tmp_path,
        monkeypatch,
        skip_ptoas=True,
    )
    assert pto.count("pto.tmatmul ") == expected_matmuls
