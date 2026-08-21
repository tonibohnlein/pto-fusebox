from __future__ import annotations

import ast
import copy
import os
from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.basic import build_examples
from examples.torch_frontend.pr2335_vector import (
    build_examples as build_pr2335_examples,
)
from pto_fusebox import (
    KernelKind,
    ScheduleContractError,
    SourceEmissionError,
    can_emit_region,
    emit_pypto_region,
    export_and_normalize,
    scheduled_region,
    solve_graph,
)
from pto_fusebox.ir import normalized_graph_sha256
from pto_fusebox.schedule.schema import (
    AxisPartition,
    CubeKernelPlan,
    MixedAlgorithm,
    MixedCrossCoreProtocol,
    MixedKernelPlan,
    VectorKernelPlan,
    VectorReplayPhase,
    VectorSpatialPolicy,
    VectorStreamKind,
)
from pto_fusebox.source.api import _append_spmd_statement
from pto_fusebox.source.common import SourceWriter
from torch import nn


def _solver() -> Path:
    configured = os.environ.get("PTO_FUSEBOX_TEST_SOLVER")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "build" / "mlsys_mixed"


@cache
def _solved(name: str):
    module, args = build_examples()[name]
    graph = export_and_normalize(module, args)
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    assert solved.regions_solved
    assert len(solved.regions) == 1
    return graph, solved.regions[0]


def _solve_module(module: nn.Module, args: tuple[torch.Tensor, ...]):
    graph = export_and_normalize(module, args)
    solved = solve_graph(graph, solver_binary=_solver(), solver_workers=2)
    assert solved.regions_solved == 1
    assert len(solved.regions) == 1
    return graph, solved.regions[0]


class _C2VEpilogue(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(value, weight) + bias


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


class _V2COnly(nn.Module):
    def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.exp(value), weight)


@cache
def _pr2335_solved(name: str):
    module, args = build_pr2335_examples()[name]
    return _solve_module(module, args)


pytestmark = pytest.mark.skipif(
    not _solver().is_file(), reason="built solver unavailable"
)


def _assert_single_spmd_grid(source: str, work_units: int) -> None:
    assert source.count("pl.spmd(") == 1
    assert f"for region_index in pl.spmd({work_units}," in source
    assert "pl.manual_scope(" not in source
    assert "pl.parallel(" not in source
    assert "pl.at(" not in source


def test_multi_step_composer_rejects_unmodeled_step_prologue() -> None:
    source = """
@pl.program
class Step:
    def main(self, output):
        scratch = pl.create_tensor([8, 8], dtype=pl.FP32)
        for region_index in pl.spmd(1):
            output = scratch
        return output
"""

    with pytest.raises(SourceEmissionError, match="only one SPMD loop"):
        _append_spmd_statement(SourceWriter(), source, 0, protected={"output"})


def test_multi_step_composer_rejects_split_task_bundle() -> None:
    source = """
@pl.program
class Step:
    def main(self, output):
        with pl.spmd(1) as first_task:
            output = output
        with pl.spmd(15, deps=[first_task]) as atomic_task:
            output = output
        return output
"""

    with pytest.raises(SourceEmissionError, match="only one SPMD loop"):
        _append_spmd_statement(SourceWriter(), source, 0, protected={"output"})


def test_vector_solution_is_a_complete_typed_emission_contract() -> None:
    graph, result = _solved("softmax")
    schedule = scheduled_region(result)
    assert can_emit_region(graph, result)

    assert schedule.tensor_values == result.solver_tensor_to_value
    assert len(schedule.steps) == 1
    step = schedule.steps[0]
    assert step.kind is KernelKind.VECTOR
    assert step.op_order == (0, 1, 2, 3, 4)
    assert isinstance(step.plan, VectorKernelPlan)
    body = step.plan.phase(VectorReplayPhase.BODY)
    assert body.input_lifetimes[0].use_count == 2
    assert body.ops == (0, 1, 2, 3, 4)
    assert step.plan.physical_frame.align_rows
    assert step.plan.physical_frame.element_granule == 8
    assert step.plan.physical_frame.iteration_cols == 1024
    assert step.plan.physical_frame.iteration_rows == 128
    assert step.plan.physical_frame.reduced_axis == 1

    source = emit_pypto_region(graph, result, program_name="softmax_fused").source
    ast.parse(source)
    _assert_single_spmd_grid(source, 16)
    assert "pl.pipeline(" not in source
    assert "region_rows" not in source
    assert "pl.load(arg_value," in source
    assert "[8, 1024], [8, 1024], target_memory=pl.Mem.Vec" in source
    assert source.count("pl.load(") == 1
    assert source.count("pl.store(") == 1
    assert "pl.row_max" in source
    assert "pl.row_sum" in source
    assert "pl.row_expand_div" in source
    assert "auto_fuse" not in source and "auto_tile" not in source
    assert (
        source == emit_pypto_region(graph, result, program_name="softmax_fused").source
    )


def test_cube_solution_emits_exact_spatial_and_k_window_schedule() -> None:
    graph, result = _solved("matmul")
    schedule = scheduled_region(result)
    assert can_emit_region(graph, result)

    step = schedule.steps[0]
    assert step.kind is KernelKind.CUBE
    assert isinstance(step.plan, CubeKernelPlan)
    assert step.plan.m_partition.big == 32
    assert step.plan.m_partition.small == 32
    assert step.plan.m_partition.num_big == 0
    assert step.plan.m_partition.parts == 4
    assert step.plan.n_partition.big == 64
    assert step.plan.n_partition.small == 64
    assert step.plan.n_partition.num_big == 0
    assert step.plan.n_partition.parts == 3
    assert step.plan.execution_order == (0,)
    assert step.plan.resident_boundaries == ()

    source = emit_pypto_region(graph, result, program_name="matmul_fused").source
    ast.parse(source)
    _assert_single_spmd_grid(source, 12)
    assert "region_row = m_index * 32" in source
    assert "region_col = n_index * 64" in source
    assert "for k_window in pl.pipeline(1, 3, stage=2):" in source
    assert source.count("pl.tile.matmul(") == 1
    assert source.count("pl.tile.matmul_acc(") == 2
    assert "[region_row, 240], [32, 16]" in source
    assert "[240, region_col], [16, 64]" in source
    assert source.count("pl.store(") == 1
    assert "auto_fuse" not in source and "auto_tile" not in source


def test_vector_emission_is_generic_over_pointwise_operation_order() -> None:
    class PointwiseChain(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.maximum(torch.exp(lhs * 0.5) + rhs, rhs)

    graph, result = _solve_module(
        PointwiseChain(), (torch.zeros(96, 320), torch.ones(96, 320))
    )
    source = emit_pypto_region(graph, result, program_name="pointwise_chain").source

    assert [op.kind for op in graph.ops] == ["mul", "exp", "add", "maximum"]
    assert source.index("pl.mul") < source.index("pl.exp")
    assert source.index("pl.exp") < source.index("pl.add")
    assert source.index("pl.add") < source.index("pl.maximum")
    assert source.count("pl.load(") == 2
    assert source.count("pl.store(") == 1


def test_vector_reduction_uses_pinned_iteration_frame_not_output_extent() -> None:
    class SumOfSquares(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.sum(value * value, dim=-1, keepdim=True)

    graph, result = _solve_module(SumOfSquares(), (torch.ones(128, 1024),))
    source = emit_pypto_region(graph, result, program_name="sum_of_squares").source

    assert [op.kind for op in graph.ops] == ["mul", "sum"]
    assert "[8, 1024], [8, 1024], target_memory=pl.Mem.Vec" in source
    assert "valid_cols" not in source
    assert "pl.row_sum" in source
    assert source.count("pl.load(") == 1
    assert source.count("pl.store(") == 1


def test_narrow_row_reduction_serializes_lowered_scratch_frame() -> None:
    class NarrowRowReduction(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.sum(value, dim=-1, keepdim=True)

    graph, result = _solve_module(NarrowRowReduction(), (torch.ones(16384, 16),))
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, VectorKernelPlan)
    body = plan.phase(VectorReplayPhase.BODY)
    assert len(body.workspaces) == 1
    assert body.workspaces[0].logical[1] == 16
    assert body.workspaces[0].physical[1] == 128

    source = emit_pypto_region(
        graph, result, program_name="narrow_row_reduction"
    ).source
    _assert_single_spmd_grid(source, 96)
    assert "pl.tile.create([" in source
    assert ", 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)" in source


def test_large_bare_reduction_replays_folded_stats_and_tail() -> None:
    class BareReduction(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.sum(value, dim=-1, keepdim=True)

    graph, result = _solve_module(BareReduction(), (torch.ones(5, 32771),))
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, VectorKernelPlan)
    assert plan.kind is VectorStreamKind.REDUCTION_FOLDED

    source = emit_pypto_region(graph, result, program_name="bare_reduction").source
    ast.parse(source)
    _assert_single_spmd_grid(source, plan.work_units)
    assert "for stats_chunk, (stats_state,) in pl.pipeline(" in source
    assert "stats_next_state = pl.add(stats_state," in source
    assert "stats_tail_state = pl.add(stats_result," in source
    assert source.count("pl.store(") == 1
    assert "for apply_chunk" not in source


def test_large_normalization_replays_stats_then_spanning_apply() -> None:
    class Normalize(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value / torch.sum(value, dim=-1, keepdim=True)

    graph, result = _solve_module(Normalize(), (torch.ones(5, 32771),))
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, VectorKernelPlan)
    assert plan.kind is VectorStreamKind.REDUCTION_SPANNING

    source = emit_pypto_region(graph, result, program_name="normalize").source
    ast.parse(source)
    _assert_single_spmd_grid(source, plan.work_units)
    assert "for stats_chunk, (stats_state,) in pl.pipeline(" in source
    assert "for apply_chunk in pl.pipeline(" in source
    assert "pl.row_expand_div(" in source
    assert source.count("output = pl.store(") == 2
    assert "apply_tail" in source


def test_cube_emission_is_generic_over_shape_and_k_tail() -> None:
    class MatmulWithTail(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs)

    graph, result = _solve_module(
        MatmulWithTail(), (torch.zeros(64, 272), torch.zeros(272, 80))
    )
    source = emit_pypto_region(graph, result, program_name="matmul_with_tail").source

    assert "[region_row, 240], [16, 32]" in source
    assert "[240, 0], [32, 80]" in source
    assert "region_col =" not in source
    assert "pl.tile.matmul_acc" in source
    assert source.count("pl.store(") == 1


def test_cube_singleton_partition_axis_uses_literal_zero_coordinate() -> None:
    class MatmulWithSingletonMPartition(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs)

    graph, result = _solve_module(
        MatmulWithSingletonMPartition(),
        (torch.zeros(16, 272), torch.zeros(272, 32)),
    )
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert plan.m_partition.parts == 1
    assert plan.n_partition.parts == 2

    source = emit_pypto_region(graph, result, program_name="matmul_singleton_m").source

    ast.parse(source)
    assert "region_row =" not in source
    assert "pl.tile.load(arg_lhs, [0, 0]" in source
    assert "pl.store(accumulator, [0, region_col], output)" in source


def test_single_bf16_matmul_uses_fp32_accumulator_and_bf16_drain() -> None:
    class Bf16Matmul(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs)

    graph, result = _solve_module(
        Bf16Matmul(),
        (
            torch.zeros(64, 64, dtype=torch.bfloat16),
            torch.ones(64, 64, dtype=torch.bfloat16),
        ),
    )
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert plan.matmuls[0].storage_dtype == "bf16"

    source = emit_pypto_region(graph, result, program_name="bf16_matmul").source

    ast.parse(source)
    _assert_single_spmd_grid(source, plan.work_units)
    assert "out_dtype=pl.FP32" in source
    assert "pl.assemble(output, matmul_0_tile_0_accumulator" in source
    assert "pl.cast(" not in source


def test_cube_chain_replays_outer_k_and_l1_intermediate_in_plan_order() -> None:
    class ChainedMatmul(nn.Module):
        def forward(
            self,
            lhs: torch.Tensor,
            middle: torch.Tensor,
            rhs: torch.Tensor,
        ) -> torch.Tensor:
            return torch.mm(torch.mm(lhs, middle), rhs)

    graph, result = _solve_module(
        ChainedMatmul(),
        (
            torch.zeros(64, 128, dtype=torch.bfloat16),
            torch.zeros(128, 96, dtype=torch.bfloat16),
            torch.zeros(96, 80, dtype=torch.bfloat16),
        ),
    )
    schedule = scheduled_region(result)
    assert len(schedule.steps) == 1
    plan = schedule.steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert plan.execution_order == (0, 1)
    assert [matmul.k_loop.full_chunks for matmul in plan.matmuls] == [4, 3]

    source = emit_pypto_region(graph, result, program_name="chained_matmul").source

    ast.parse(source)
    _assert_single_spmd_grid(source, 4)
    assert source.count("pl.create_l1(") == 1
    assert source.count("pl.matmul(") == 2
    assert source.count("pl.matmul_acc(") == 2
    assert source.index("matmul_0_tile_0_accumulator") < source.index(
        "matmul_1_tile_0_accumulator"
    )
    assert "pl.cast(" not in source
    assert "pl.assemble(matmul_0_l1" in source
    assert "pl.assemble(output, matmul_1_tile_0_accumulator" in source
    assert source.count("pl.store(") == 0


def test_cube_retained_panel_is_sliced_once_outside_output_tile_replay() -> None:
    class WideMatmul(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs)

    graph, result = _solve_module(
        WideMatmul(),
        (
            torch.zeros(512, 64, dtype=torch.bfloat16),
            torch.zeros(64, 2048, dtype=torch.bfloat16),
        ),
    )
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert plan.matmuls[0].output_grid == (2, 1)
    assert plan.matmuls[0].retained_panels.rhs

    source = emit_pypto_region(graph, result, program_name="retained_rhs").source

    ast.parse(source)
    _assert_single_spmd_grid(source, plan.work_units)
    assert source.count("matmul_0_rhs_retained = pl.slice(") == 1
    assert source.index("matmul_0_rhs_retained = pl.slice(") < source.index(
        "matmul_0_tile_0_rhs_init = pl.slice(matmul_0_rhs_retained"
    )
    assert source.count("pl.matmul(") == 2
    assert source.count("pl.matmul_acc(") == 2
    assert source.count("pl.assemble(output,") == 2


def test_general_cube_path_rejects_lowered_l0_capacity_overflow() -> None:
    class WideMatmul(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs)

    graph, result = _solve_module(
        WideMatmul(),
        (
            torch.zeros(512, 64, dtype=torch.bfloat16),
            torch.zeros(64, 2048, dtype=torch.bfloat16),
        ),
    )
    assert result.problem is not None
    problem = copy.deepcopy(result.problem)
    problem["l0_matmul_config"]["l0a_bytes"] = 1

    with pytest.raises(SourceEmissionError, match="lowered L0 operand capacity"):
        emit_pypto_region(graph, replace(result, problem=problem))


def test_cube_diamond_replays_three_requests_in_selected_topological_order() -> None:
    class DiamondMatmul(nn.Module):
        def forward(
            self,
            shared: torch.Tensor,
            lhs_weight: torch.Tensor,
            rhs_weight: torch.Tensor,
        ) -> torch.Tensor:
            lhs = torch.mm(shared, lhs_weight)
            rhs = torch.mm(shared, rhs_weight)
            return torch.mm(lhs, rhs)

    graph, result = _solve_module(
        DiamondMatmul(),
        tuple(torch.zeros(32, 32, dtype=torch.bfloat16) for _ in range(3)),
    )
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert plan.execution_order == (0, 1, 2)

    source = emit_pypto_region(graph, result, program_name="diamond_matmul").source

    ast.parse(source)
    assert source.count("pl.create_l1(") == 2
    assert source.count("pl.matmul(") == 3
    assert (
        source.index("matmul_0_tile_0_accumulator")
        < source.index("matmul_1_tile_0_accumulator")
        < source.index("matmul_2_tile_0_accumulator")
    )
    assert "pl.slice(matmul_0_l1_0" in source
    assert "pl.slice(matmul_1_l1_0" in source
    assert source.count("pl.assemble(output,") == 1


def test_two_cube_steps_materialize_one_dependency_linked_intermediate() -> None:
    class Fp32ChainedMatmul(nn.Module):
        def forward(
            self,
            lhs: torch.Tensor,
            middle: torch.Tensor,
            rhs: torch.Tensor,
        ) -> torch.Tensor:
            return torch.mm(torch.mm(lhs, middle), rhs)

    graph, result = _solve_module(
        Fp32ChainedMatmul(),
        (
            torch.zeros(64, 128),
            torch.zeros(128, 96),
            torch.zeros(96, 80),
        ),
    )
    schedule = scheduled_region(result)
    assert len(schedule.steps) == 2
    assert [step.kind for step in schedule.steps] == [
        KernelKind.CUBE,
        KernelKind.CUBE,
    ]

    emitted = emit_pypto_region(graph, result, program_name="cut_chained_matmul")
    source = emitted.source

    ast.parse(source)
    assert emitted.kinds == (KernelKind.CUBE, KernelKind.CUBE)
    assert emitted.kind is KernelKind.CUBE
    assert source.count("pl.spmd(") == 2
    assert "intermediate_tensor_2 = pl.create_tensor([64, 96], dtype=pl.FP32)" in source
    assert "pl.store(step_0_accumulator" in source
    assert "pl.tile.load(intermediate_tensor_2" in source
    assert source.index("for step_0_region_index") < source.index(
        "for step_1_region_index"
    )
    assert "auto_fuse" not in source and "auto_tile" not in source


def test_multi_step_source_rejects_a_consumer_launch_before_its_producer() -> None:
    class Fp32ChainedMatmul(nn.Module):
        def forward(
            self,
            lhs: torch.Tensor,
            middle: torch.Tensor,
            rhs: torch.Tensor,
        ) -> torch.Tensor:
            return torch.mm(torch.mm(lhs, middle), rhs)

    graph, result = _solve_module(
        Fp32ChainedMatmul(),
        (
            torch.zeros(64, 128),
            torch.zeros(128, 96),
            torch.zeros(96, 80),
        ),
    )
    assert result.solution is not None
    reordered = copy.deepcopy(result.solution)
    reordered["steps"].reverse()

    with pytest.raises(SourceEmissionError, match="consumed before its producer"):
        emit_pypto_region(graph, replace(result, solution=reordered))


def test_fanout_cube_step_uses_op_order_for_sparse_ids_and_two_outputs() -> None:
    class Fp32Fanout(nn.Module):
        def forward(
            self,
            lhs: torch.Tensor,
            middle: torch.Tensor,
            first_rhs: torch.Tensor,
            second_rhs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            shared = torch.mm(lhs, middle)
            return torch.mm(shared, first_rhs), torch.mm(shared, second_rhs)

    graph, result = _solve_module(
        Fp32Fanout(),
        tuple(torch.zeros(64, 64) for _ in range(4)),
    )
    schedule = scheduled_region(result)
    assert [step.solver_ops for step in schedule.steps] == [(0,), (1, 2)]
    assert schedule.steps[1].op_order == (1, 2)

    emitted = emit_pypto_region(graph, result, program_name="fanout_cube")
    source = emitted.source

    ast.parse(source)
    assert emitted.kinds == (KernelKind.CUBE, KernelKind.CUBE)
    assert source.count("pl.spmd(") == 2
    assert source.count("pl.create_tensor(") == 1
    assert "step_1_resident_0_lhs = pl.slice(intermediate_tensor_2" in source
    assert source.count("pl.matmul(") == 2
    assert "output_0 = pl.assemble(" in source
    assert "output_1 = pl.assemble(" in source
    assert "return output_0, output_1" in source


def test_source_backend_rejects_degenerate_cube_resident_lifetime() -> None:
    graph, result = _solved("matmul")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    matmul = solution["steps"][0]["plan"]["matmuls"][0]
    lhs_region = copy.deepcopy(matmul["lhs"])
    matmul["lhs_resident_boundary"] = 0
    solution["steps"][0]["plan"]["resident_boundaries"] = [
        {
            "id": 0,
            "region": lhs_region,
            "role": "lhs",
            "first_use": 0,
            "last_use": 0,
            "use_count": 1,
            "bytes": lhs_region["height"] * lhs_region["width"] * 4,
        }
    ]

    with pytest.raises(SourceEmissionError, match="resident-boundary"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_source_backend_rejects_cube_execution_order_drift() -> None:
    graph, result = _solved("matmul")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["execution_order"] = []

    with pytest.raises(SourceEmissionError, match="execution order"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_schedule_contract_rejects_an_incomplete_operation_order() -> None:
    _, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["op_order"] = solution["steps"][0]["op_order"][:-1]

    with pytest.raises(ScheduleContractError, match="not a permutation"):
        scheduled_region(replace(result, solution=solution))


def test_schedule_contract_rejects_legacy_or_dropped_step_fields() -> None:
    _, result = _solved("softmax")
    assert result.solution is not None

    legacy = dict(copy.deepcopy(result.solution))
    legacy["schema_version"] = "pto_fusebox.solution.v3"
    with pytest.raises(ScheduleContractError, match="solution schema"):
        scheduled_region(replace(result, solution=legacy))

    for field in ("sequential_tiles", "op_order"):
        incomplete = copy.deepcopy(result.solution)
        incomplete["steps"][0].pop(field)
        with pytest.raises(ScheduleContractError, match=f"omits fields.*{field}"):
            scheduled_region(replace(result, solution=incomplete))


def test_schedule_contract_rejects_vector_phase_order_or_frame_drift() -> None:
    _, result = _solved("softmax")
    assert result.solution is not None

    reordered = copy.deepcopy(result.solution)
    body = reordered["steps"][0]["plan"]["phases"][0]
    body["ops"][0], body["ops"][1] = body["ops"][1], body["ops"][0]
    with pytest.raises(ScheduleContractError, match="preserve.*operation order"):
        scheduled_region(replace(result, solution=reordered))

    stale_frame = copy.deepcopy(result.solution)
    stale_frame["steps"][0]["plan"]["phases"][0]["tensor_frames"].pop()
    with pytest.raises(ScheduleContractError, match="tensor_frames do not cover"):
        scheduled_region(replace(result, solution=stale_frame))

    stale_workspace = copy.deepcopy(result.solution)
    workspace = stale_workspace["steps"][0]["plan"]["phases"][0]["workspaces"][0]
    workspace["source_tensor"] += 1
    with pytest.raises(ScheduleContractError, match="wrong source tensor"):
        scheduled_region(replace(result, solution=stale_workspace))


def test_schedule_contract_requires_stream_phase_boundary_lifetime_coverage() -> None:
    class Normalize(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value / torch.sum(value, dim=-1, keepdim=True)

    _, result = _solve_module(Normalize(), (torch.ones(5, 32771),))
    assert result.solution is not None
    missing = copy.deepcopy(result.solution)
    stats = missing["steps"][0]["plan"]["phases"][1]
    assert stats["name"] == "stats" and stats["input_lifetimes"]
    stats["input_lifetimes"] = []

    with pytest.raises(ScheduleContractError, match="boundary-tensor uses"):
        scheduled_region(replace(result, solution=missing))


def test_schedule_contract_accepts_sparse_cube_resident_identity() -> None:
    _, result = _solved("matmul")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    matmul = solution["steps"][0]["plan"]["matmuls"][0]
    lhs_region = copy.deepcopy(matmul["lhs"])
    matmul["lhs_resident_boundary"] = 0
    solution["steps"][0]["plan"]["resident_boundaries"] = [
        {
            "id": 19,
            "region": lhs_region,
            "role": "lhs",
            "first_use": 0,
            "last_use": 0,
            "use_count": 1,
            "bytes": lhs_region["height"] * lhs_region["width"] * 4,
        }
    ]

    schedule = scheduled_region(replace(result, solution=solution))
    plan = schedule.steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert plan.resident_boundaries[0].id == 19
    assert plan.matmuls[0].lhs_resident_boundary == 0


def test_schedule_contract_rejects_cube_k_or_residency_drift() -> None:
    _, result = _solved("matmul")
    assert result.solution is not None

    stale_k = copy.deepcopy(result.solution)
    stale_k["steps"][0]["plan"]["matmuls"][0]["k_loop"]["tail"] += 1
    with pytest.raises(ScheduleContractError, match="k_loop does not cover K"):
        scheduled_region(replace(result, solution=stale_k))

    stale_variant = copy.deepcopy(result.solution)
    stale_variant["steps"][0]["plan"]["matmuls"][0]["output_variants"][0]["count"] += 1
    with pytest.raises(ScheduleContractError, match="output_variants"):
        scheduled_region(replace(result, solution=stale_variant))

    shortened_k = copy.deepcopy(result.solution)
    matmul = shortened_k["steps"][0]["plan"]["matmuls"][0]
    matmul["contraction"] = 240
    matmul["effective_contraction"] = 240
    matmul["k_loop"]["tail"] = 0
    for variant in matmul["output_variants"]:
        variant["l0_tail"] = None
    with pytest.raises(ScheduleContractError, match="contraction differs"):
        scheduled_region(replace(result, solution=shortened_k))

    stale_region = copy.deepcopy(result.solution)
    stale_region["steps"][0]["plan"]["matmuls"][0]["lhs"]["height"] += 1
    with pytest.raises(ScheduleContractError, match="axis bindings"):
        scheduled_region(replace(result, solution=stale_region))


def test_schedule_contract_rejects_common_sequential_drift() -> None:
    _, vector = _solved("softmax")
    assert vector.solution is not None
    stale_vector = copy.deepcopy(vector.solution)
    stale_vector["steps"][0]["sequential_tiles"][0] = 1
    with pytest.raises(ScheduleContractError, match="nonzero sequential tiles"):
        scheduled_region(replace(vector, solution=stale_vector))

    _, cube = _solved("matmul")
    assert cube.solution is not None
    stale_sequential = copy.deepcopy(cube.solution)
    stale_sequential["steps"][0]["sequential_tiles"][0] -= 16
    with pytest.raises(ScheduleContractError, match="common sequential tile"):
        scheduled_region(replace(cube, solution=stale_sequential))


def test_schedule_contract_rejects_unknown_p4_recipe_version() -> None:
    class Softmax(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.softmax(value, dim=-1)

    _, result = _solve_module(Softmax(), (torch.ones(16, 4096),))
    assert result.solution is not None
    assert result.solution["steps"][0]["plan"]["kind"] == "softmax_flash"
    assert result.solution["steps"][0]["plan"]["p4_recipe"]["apply_substitutions"] == [
        {"op": 0, "value": "running_max"},
        {"op": 3, "value": "running_sum"},
    ]
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["p4_recipe"]["version"] = "softmax_flash.v2"

    with pytest.raises(ScheduleContractError, match="version is unsupported"):
        scheduled_region(replace(result, solution=solution))


def test_schedule_contract_rejects_p4_semantic_role_drift() -> None:
    class Softmax(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.softmax(value, dim=-1)

    _, result = _solve_module(Softmax(), (torch.ones(16, 4096),))
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    bindings = solution["steps"][0]["plan"]["p4_recipe"]["apply_substitutions"]
    bindings[0]["value"], bindings[1]["value"] = (
        bindings[1]["value"],
        bindings[0]["value"],
    )

    with pytest.raises(ScheduleContractError, match="versioned state contract"):
        scheduled_region(replace(result, solution=solution))


def test_materialized_schedule_rejects_an_inactive_phase_replay() -> None:
    _, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["phases"][1]["ops"] = [0]

    with pytest.raises(ScheduleContractError, match="must be empty"):
        scheduled_region(replace(result, solution=solution))


def test_source_backend_rejects_vector_lifetime_drift() -> None:
    graph, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["phases"][0]["input_lifetimes"] = []

    with pytest.raises(SourceEmissionError, match="input lifetimes"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_source_backend_rejects_partition_that_does_not_cover_output() -> None:
    graph, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["m_partition"]["big"] += 1

    with pytest.raises(SourceEmissionError, match="iteration frame"):
        emit_pypto_region(graph, replace(result, solution=solution))


def test_source_backend_rejects_unpriced_vector_spatial_policy() -> None:
    graph, result = _solved("softmax")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["spatial_policy"] = "exact_balanced"
    mutated = replace(result, solution=solution)

    assert not can_emit_region(graph, mutated)
    with pytest.raises(SourceEmissionError, match="clamped-overlap spatial policy"):
        emit_pypto_region(graph, mutated)


def test_source_backend_rejects_non_last_axis_reduction() -> None:
    graph, result = _solved("softmax")
    first = replace(graph.ops[0], attributes={"axis": 0, "keepdim": True})
    invalid_graph = replace(graph, ops=(first, *graph.ops[1:]))
    assert result.problem is not None
    problem = copy.deepcopy(dict(result.problem))
    frontend_mapping = dict(problem["frontend_mapping"])
    frontend_mapping["normalized_graph_sha256"] = normalized_graph_sha256(invalid_graph)
    problem["frontend_mapping"] = frontend_mapping

    with pytest.raises(SourceEmissionError, match="last-axis keepdim"):
        emit_pypto_region(invalid_graph, replace(result, problem=problem))


def test_source_backend_binds_the_solution_to_the_exact_normalized_graph() -> None:
    class PointwiseChain(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.exp(value * 0.5)

    graph, result = _solve_module(PointwiseChain(), (torch.randn(8, 32),))
    first = graph.ops[0]
    assert first.kind == "mul"
    scalar_attributes = dict(first.attributes)
    assert scalar_attributes["scalars"] == [{"position": 1, "value": 0.5}]
    scalar_attributes["scalars"] = [{"position": 1, "value": 0.75}]
    changed_graphs = (
        replace(graph, ops=(replace(first, kind="add"), *graph.ops[1:])),
        replace(
            graph,
            ops=(replace(first, attributes=scalar_attributes), *graph.ops[1:]),
        ),
    )

    for changed_graph in changed_graphs:
        assert not can_emit_region(changed_graph, result)
        with pytest.raises(SourceEmissionError, match="does not match the graph"):
            emit_pypto_region(changed_graph, result)


def test_source_backend_names_cannot_shadow_the_dsl_or_generated_locals() -> None:
    class AdversarialNames(nn.Module):
        def forward(
            self,
            pl: torch.Tensor,
            region_row: torch.Tensor,
            tensor_0: torch.Tensor,
            accumulator: torch.Tensor,
            stats_running_max: torch.Tensor,
            auto_fuse: torch.Tensor,
            auto_tile: torch.Tensor,
        ) -> torch.Tensor:
            return (
                pl
                + region_row
                + tensor_0
                + accumulator
                + stats_running_max
                + auto_fuse
                + auto_tile
            )

    inputs = tuple(torch.randn(8, 32) for _ in range(7))
    graph, result = _solve_module(AdversarialNames(), inputs)

    assert can_emit_region(graph, result)
    source = emit_pypto_region(graph, result).source
    ast.parse(source)
    for name in (
        "pl",
        "region_row",
        "tensor_0",
        "accumulator",
        "stats_running_max",
        "auto_fuse",
        "auto_tile",
    ):
        assert f"arg_{name}: pl.Tensor" in source
    assert "def main(\n        self,\n        pl:" not in source


def test_vector_source_emits_multiple_outputs_in_region_abi_order() -> None:
    class TwoOutputs(nn.Module):
        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            first = torch.exp(value)
            return first, first + 1.0

    graph, result = _solve_module(TwoOutputs(), (torch.ones(64, 128),))

    assert can_emit_region(graph, result)
    source = emit_pypto_region(graph, result, program_name="two_outputs").source

    ast.parse(source)
    assert "output_0: pl.Out[" in source
    assert "output_1: pl.Out[" in source
    assert source.count("pl.store(") == 2
    assert source.index("output_0 = pl.store") < source.index("output_1 = pl.store")
    assert "return output_0, output_1" in source


def test_source_readiness_rejects_transposed_matmul() -> None:
    class TransposedMatmul(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs.t())

    graph, result = _solve_module(
        TransposedMatmul(),
        (torch.ones(64, 96), torch.ones(128, 96)),
    )

    assert not can_emit_region(graph, result)
    with pytest.raises(SourceEmissionError, match="non-transposed matmul"):
        emit_pypto_region(graph, result)


def test_generic_round_trip_emits_the_solver_owned_mixed_pipeline() -> None:
    graph, result = _solved("attention_core")
    step = scheduled_region(result).steps[0]

    assert step.kind is KernelKind.MIXED
    assert isinstance(step.plan, MixedKernelPlan)
    assert step.plan.algorithm is MixedAlgorithm.GENERIC
    assert step.plan.protocol is MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE
    assert can_emit_region(graph, result)

    source = emit_pypto_region(graph, result, program_name="attention_mixed").source
    ast.parse(source)
    _assert_single_spmd_grid(source, step.plan.active_groups)
    assert "pl.split(pl.SplitMode.UP_DOWN, slot_num=4)" in source
    assert "pl.pipeline(1, stage=3" in source
    assert source.count("pl.tensor.matmul(") == 2
    assert "b_trans=True" in source
    assert "pl.tensor.row_max(" in source
    assert "pl.tensor.row_sum(" in source
    assert "pl.tensor.assemble(" in source
    assert "auto_fuse" not in source and "auto_tile" not in source


def test_mixed_typed_contract_rejects_stale_fifo_geometry() -> None:
    _, result = _solved("attention_core")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["fifos"][0]["slot_bytes"] += 4

    with pytest.raises(
        ScheduleContractError, match="differs from its transfer geometry"
    ):
        scheduled_region(replace(result, solution=solution))


def test_mixed_typed_contract_rejects_stale_protocol_bundle() -> None:
    _, result = _solved("attention_core")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["protocol_producer_bundle"] = [1]

    with pytest.raises(
        ScheduleContractError, match="inconsistent single-round-trip protocol"
    ):
        scheduled_region(replace(result, solution=solution))


def test_mixed_source_rejects_shared_fifo_over_capacity() -> None:
    graph, result = _solved("attention_core")
    assert result.problem is not None
    problem = dict(result.problem)
    problem["vec_capacity"] = 130_000
    stale = replace(result, problem=problem)

    assert not can_emit_region(graph, stale)
    with pytest.raises(
        SourceEmissionError, match="shared FIFO and vector stage exceed Vec capacity"
    ):
        emit_pypto_region(graph, stale)


def test_one_way_c2v_emits_matmul_and_generic_vector_epilogue() -> None:
    graph, result = _solve_module(
        _C2VEpilogue(),
        (
            torch.zeros(32, 64),
            torch.zeros(64, 32),
            torch.zeros(1, 32),
        ),
    )
    step = scheduled_region(result).steps[0]

    assert step.kind is KernelKind.MIXED
    assert isinstance(step.plan, MixedKernelPlan)
    assert step.plan.protocol is MixedCrossCoreProtocol.ONE_WAY
    assert can_emit_region(graph, result)

    source = emit_pypto_region(graph, result, program_name="c2v_epilogue").source
    ast.parse(source)
    _assert_single_spmd_grid(source, step.plan.active_groups)
    assert "pl.split(pl.SplitMode.UP_DOWN, slot_num=8)" in source
    assert "pl.range(1, init_values=(output,))" in source
    assert source.count("pl.tensor.matmul(") == 1
    assert "pl.tensor.add(" in source
    assert source.count("pl.tensor.assemble(") == 1


def test_dense_swiglu_emits_two_producers_vector_dag_and_down_accumulator() -> None:
    graph, result = _solve_module(
        _DenseSwiGlu(),
        (
            torch.zeros(128, 64, dtype=torch.bfloat16),
            torch.zeros(64, 128, dtype=torch.bfloat16),
            torch.zeros(64, 128, dtype=torch.bfloat16),
            torch.zeros(128, 64, dtype=torch.bfloat16),
        ),
    )
    step = scheduled_region(result).steps[0]

    assert step.kind is KernelKind.MIXED
    assert isinstance(step.plan, MixedKernelPlan)
    assert step.plan.algorithm is MixedAlgorithm.DENSE_SWIGLU_MLP
    assert step.op_order[:2] == (1, 0)
    assert can_emit_region(graph, result)

    source = emit_pypto_region(graph, result, program_name="dense_swiglu").source
    ast.parse(source)
    _assert_single_spmd_grid(source, step.plan.active_groups)
    assert "pl.split(pl.SplitMode.UP_DOWN, slot_num=4)" in source
    assert "pl.pipeline(0, 128, 64, stage=3" in source
    assert source.count("pl.tensor.matmul(") == 3
    assert source.count("pl.tensor.matmul_acc(") == 1
    assert "pl.tensor.recip(" in source
    assert 'target_type=pl.BF16, mode="round"' in source


def test_source_backend_rejects_unimplemented_one_way_v2c() -> None:
    graph, result = _solve_module(
        _V2COnly(),
        (torch.zeros(32, 64), torch.zeros(64, 32)),
    )

    assert not can_emit_region(graph, result)
    with pytest.raises(SourceEmissionError, match="one-way C->V topology"):
        emit_pypto_region(graph, result)


@pytest.mark.parametrize(
    (
        "name",
        "kind",
        "work_units",
        "tile",
        "strip",
        "body_trips",
        "body_stages",
        "chunk",
        "tail",
        "full_peak_ub_bytes",
        "chunk_peak_ub_bytes",
        "latency_cycles",
    ),
    [
        (
            "pr2335_softmax_512x256",
            "materialized",
            16,
            (32, 256),
            (32, 256),
            1,
            1,
            0,
            0,
            65_664,
            65_664,
            15_121.378472222223,
        ),
        (
            "pr2335_softmax_256x512",
            "materialized",
            16,
            (16, 512),
            (16, 512),
            1,
            1,
            0,
            0,
            65_600,
            65_600,
            15_217.378472222223,
        ),
        (
            "pr2335_softmax_128x1024",
            "materialized",
            16,
            (8, 1024),
            (8, 1024),
            1,
            1,
            0,
            0,
            65_568,
            65_568,
            15_633.378472222223,
        ),
        (
            "pr2335_softmax_32x8192",
            "softmax_flash",
            32,
            (1, 8192),
            (0, 0),
            0,
            1,
            480,
            32,
            524_320,
            184_576,
            26_604.218098958336,
        ),
        (
            "pr2335_rms_norm",
            "materialized",
            24,
            (22, 512),
            (22, 512),
            1,
            1,
            0,
            0,
            147_552,
            147_552,
            17_580.31396484375,
        ),
        (
            "pr2335_layer_norm",
            "materialized",
            24,
            (22, 256),
            (22, 256),
            1,
            1,
            0,
            0,
            73_824,
            73_824,
            15_732.204915364582,
        ),
        (
            "pr2335_silu",
            "pointwise",
            8,
            (256, 64),
            (64, 64),
            4,
            2,
            64,
            0,
            196_608,
            98_304,
            12_664.0,
        ),
    ],
)
def test_pr2335_vector_surface_is_source_ready(
    name: str,
    kind: str,
    work_units: int,
    tile: tuple[int, int],
    strip: tuple[int, int],
    body_trips: int,
    body_stages: int,
    chunk: int,
    tail: int,
    full_peak_ub_bytes: int,
    chunk_peak_ub_bytes: int,
    latency_cycles: float,
) -> None:
    graph, result = _pr2335_solved(name)
    schedule = scheduled_region(result)

    assert can_emit_region(graph, result)
    assert len(schedule.steps) == 1
    plan = schedule.steps[0].plan
    assert isinstance(plan, VectorKernelPlan)
    assert plan.kind.value == kind
    assert plan.spatial_policy is VectorSpatialPolicy.CLAMPED_OVERLAP
    assert plan.work_units == work_units
    assert plan.tile == tile
    assert plan.strip == strip
    assert plan.chunk == chunk
    assert plan.tail == tail
    assert plan.full_peak_ub_bytes == full_peak_ub_bytes
    assert plan.chunk_peak_ub_bytes == chunk_peak_ub_bytes
    assert schedule.steps[0].latency == pytest.approx(latency_cycles)
    body = plan.phase(VectorReplayPhase.BODY)
    assert body.loop is not None
    assert body.loop.trip_count == body_trips
    assert body.loop.pipeline_stages == body_stages

    source = emit_pypto_region(graph, result, program_name=name).source
    ast.parse(source)
    _assert_single_spmd_grid(source, work_units)
    assert "auto_fuse" not in source and "auto_tile" not in source
    assert "valid_rows = pl." not in source
    assert "valid_cols = pl." not in source
    assert "region_rows = pl." not in source
    assert "region_cols = pl." not in source
    assert "clamp=" not in source
    if plan.m_partition.parts == 1:
        assert "region_row =" not in source
    if plan.n_partition.parts == 1:
        assert "region_col =" not in source
    if kind == "materialized":
        assert "pl.pipeline(" not in source
        assert "strip_row =" not in source
        assert "strip_col =" not in source
    if kind == "pointwise":
        assert "pl.pipeline(4, stage=2)" in source
        assert "strip_row =" in source
        assert "strip_col =" not in source


def test_pr2335_wide_softmax_replays_the_typed_online_phases() -> None:
    graph, result = _pr2335_solved("pr2335_softmax_32x8192")
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, VectorKernelPlan)
    assert plan.kind is VectorStreamKind.SOFTMAX_FLASH
    assert plan.p4_recipe is not None
    assert plan.p4_recipe.version == "softmax_flash.v1"

    stats = plan.phase(VectorReplayPhase.STATS)
    apply = plan.phase(VectorReplayPhase.APPLY)
    assert stats.loop is not None and apply.loop is not None
    source = emit_pypto_region(graph, result, program_name="renamed_program").source
    assert (
        f"pl.pipeline({stats.loop.first_chunk}, "
        f"{stats.loop.first_chunk + stats.loop.trip_count}, stage=2," in source
    )
    assert (
        f"pl.pipeline({apply.loop.first_chunk}, "
        f"{apply.loop.first_chunk + apply.loop.trip_count}, stage=2)" in source
    )
    assert f"[{plan.free_tile_alloc}, {plan.chunk}]" in source
    assert source.count("pl.load(") == 5
    assert source.count("pl.store(") == 2
    assert "init_values=(initial_local_max, initial_local_sum,)" in source
    assert "stats_result_max, stats_result_sum = pl.yield_(" in source
    assert "pl.maximum" in source
    assert "pl.row_expand_sub" in source
    assert "pl.row_expand_div" in source
    assert "softmax" not in source.lower()


def test_pr2335_silu_scalar_left_division_lowers_to_reciprocal() -> None:
    graph, result = _pr2335_solved("pr2335_silu")
    source = emit_pypto_region(graph, result, program_name="generic_pointwise").source

    assert [op.kind for op in graph.ops] == ["mul", "exp", "add", "div", "mul"]
    assert result.problem is not None
    assert result.problem["vector_primitive_families"] == [
        "scalar_mul",
        "exp",
        "scalar_add",
        "recip",
        "mul",
    ]
    assert source.count("pl.recip(") == 1
    assert "silu" not in source.lower()


def test_pr2335_ragged_layernorm_emits_static_clamped_regions() -> None:
    graph, result = _pr2335_solved("pr2335_layer_norm")
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, VectorKernelPlan)
    assert plan.m_partition == AxisPartition(big=22, small=21, num_big=8, parts=24)

    source = emit_pypto_region(graph, result, program_name="layer_norm").source

    assert "m_big_before = pl.min(region_index, 8)" in source
    assert "region_row = pl.min(region_index * 21 + m_big_before * 1, 490)" in source
    assert "region_col =" not in source
    assert "strip_row =" not in source
    assert "strip_col =" not in source
    assert "[22, 256], target_memory=pl.Mem.Vec" in source
    assert source.count("[1, 256], target_memory=pl.Mem.Vec") == 2
    assert "region_rows" not in source
    assert "valid_rows" not in source


def test_ragged_pointwise_clamps_region_and_strip_origins_not_shapes() -> None:
    class RaggedPointwise(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.exp(value * 0.5) + value

    graph, result = _solve_module(RaggedPointwise(), (torch.zeros(257, 65),))
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, VectorKernelPlan)
    assert plan.kind is VectorStreamKind.POINTWISE
    assert plan.m_partition == AxisPartition(big=17, small=16, num_big=1, parts=16)
    assert plan.tile == (17, 65)
    assert plan.strip == (9, 65)
    assert plan.strip_grid == (2, 1)

    source = emit_pypto_region(graph, result, program_name="ragged_pointwise").source

    assert "region_row = pl.min(region_index * 16 + m_big_before * 1, 240)" in source
    assert "strip_row = pl.min(strip_index * 9, 8)" in source
    assert "region_col =" not in source
    assert "strip_col =" not in source
    assert "[9, 65], target_memory=pl.Mem.Vec" in source
    assert "valid_rows" not in source
    assert "valid_cols" not in source


def test_reduction_result_cast_is_analytic_but_not_source_ready() -> None:
    class ReductionResultCast(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.sum(dim=-1, keepdim=True).to(torch.int8)

    graph, result = _solve_module(ReductionResultCast(), (torch.randn(16, 512),))

    assert result.status == "solved"
    assert not can_emit_region(graph, result)
    with pytest.raises(
        SourceEmissionError, match="cast chain rooted in a reduction result"
    ):
        emit_pypto_region(graph, result)


def test_float_to_int8_is_analytic_but_not_source_ready() -> None:
    class FloatToInt8(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.to(torch.int8)

    values = torch.tensor([[1.6, -1.6, 63.99, -63.99]], dtype=torch.float32)
    assert torch.equal(
        values.to(torch.int8), torch.tensor([[1, -1, 63, -63]], dtype=torch.int8)
    )
    assert torch.equal(
        values.to(torch.float16).to(torch.int8),
        torch.tensor([[1, -1, 64, -64]], dtype=torch.int8),
    )
    graph, result = _solve_module(FloatToInt8(), (values,))

    assert result.status == "solved"
    assert result.problem is not None
    assert result.problem["dtypes"] == ["FP32", "FP16", "INT8"]
    assert not can_emit_region(graph, result)
    with pytest.raises(SourceEmissionError, match="Torch float-to-INT8 truncation"):
        emit_pypto_region(graph, result)


def test_cast_chain_alignment_is_local_to_its_physical_shape_class() -> None:
    class CastThenBroadcast(torch.nn.Module):
        def forward(self, value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            restored = value.to(torch.float16).to(torch.float32)
            return restored + bias

    graph, result = _solve_module(
        CastThenBroadcast(),
        (torch.randn(3, 512), torch.randn(3, 1)),
    )
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, VectorKernelPlan)
    bias_tensor = result.solver_tensor_to_value.index(graph.inputs[1])
    body_frames = {
        frame.tensor: frame
        for frame in plan.phase(VectorReplayPhase.BODY).tensor_frames
    }

    assert body_frames[bias_tensor].logical[1] == 1
    assert body_frames[bias_tensor].physical[0] == 8
    assert body_frames[bias_tensor].physical[1] == 1
    source = emit_pypto_region(graph, result).source
    assert "[8, 1], [3, 1], target_memory=pl.Mem.Vec" in source
    assert "[32, 1], [3, 1], target_memory=pl.Mem.Vec" not in source


def test_softmax_source_rejects_loop_or_generated_work_drift() -> None:
    graph, result = _pr2335_solved("pr2335_softmax_32x8192")
    assert result.solution is not None

    stale_loop = copy.deepcopy(result.solution)
    stale_loop["steps"][0]["plan"]["phases"][1]["loop"]["trip_count"] -= 1
    stale_result = replace(result, solution=stale_loop)
    assert not can_emit_region(graph, stale_result)
    with pytest.raises(SourceEmissionError, match="stats loop is inconsistent"):
        emit_pypto_region(graph, stale_result)

    stale_work = copy.deepcopy(result.solution)
    stale_work["steps"][0]["plan"]["p4_work"]["stats_update"]["primitives"][0][
        "thin"
    ] += 1
    stale_result = replace(result, solution=stale_work)
    assert not can_emit_region(graph, stale_result)
    with pytest.raises(SourceEmissionError, match="generated work differs"):
        emit_pypto_region(graph, stale_result)

    stale_frame = copy.deepcopy(result.solution)
    stats_phase = stale_frame["steps"][0]["plan"]["phases"][1]
    frame = stats_phase["tensor_frames"][0]
    frame["physical"][1] += 8
    for workspace in stats_phase["workspaces"]:
        if workspace["source_tensor"] == frame["tensor"]:
            workspace["physical"][1] += 8
    stale_result = replace(result, solution=stale_frame)
    assert not can_emit_region(graph, stale_result)
    with pytest.raises(SourceEmissionError, match="frame differs from its recipe role"):
        emit_pypto_region(graph, stale_result)
