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
from examples.torch_frontend.static_mixed import (
    StaticAttentionCore,
    build_examples as build_static_mixed_examples,
)
from pto_fusebox import (
    PyPTORuntimeValidShapeArgument,
    RuntimeValidShapeSpec,
    bind_emitted_inputs,
    emit_pypto_callable,
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
    MixedEngine,
    MixedKernelPlan,
    MixedPipelineMode,
    MixedTransferDirection,
    VectorKernelPlan,
    VectorReplayPhase,
    VectorSpatialPolicy,
    VectorStreamKind,
)
from pto_fusebox.source.api import (
    _append_spmd_statement,
    _inline_local_name,
    _program_as_inline_callable,
)
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
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved
    assert len(solved.regions) == 1
    return graph, solved.regions[0]


def _solve_module(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    require_source_codegen: bool = True,
):
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=require_source_codegen,
    )
    assert solved.regions_solved == 1
    assert len(solved.regions) == 1
    return graph, solved.regions[0]


def _assert_pypto_main_mixed_scope(source: str, plan: MixedKernelPlan) -> None:
    assert "pl.split(pl.SplitMode.UP_DOWN)" in source
    assert "pl.cross_core_slot(" not in source
    assert source.count("pl.cross_core_pipe(") == len(plan.fifos)
    for fifo in plan.fifos:
        direction = (
            "pl.CrossCoreDirection.CUBE_TO_VECTOR"
            if fifo.direction is MixedTransferDirection.CUBE_TO_VECTOR
            else "pl.CrossCoreDirection.VECTOR_TO_CUBE"
        )
        assert (
            "pl.cross_core_pipe("
            f"tensor_id={fifo.tensor}, direction={direction}, "
            f"valid_shape=[{fifo.valid_rows}, {fifo.valid_cols}], "
            f"slot_size_bytes={fifo.slot_bytes}, slot_num={fifo.slot_count}, "
            f"pipe_id={fifo.pipe_id}, bundle={fifo.bundle})"
        ) in source


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


class _V2COnlyRhs(nn.Module):
    def forward(
        self, lhs: torch.Tensor, value: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.mm(lhs, torch.exp(value + bias))


class _V2CSharedLhs(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.exp(value), value)


class _V2CSharedRhs(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.mm(value, torch.exp(value))


class _StreamingSoftmaxPv(nn.Module):
    def forward(self, scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.softmax(scores, dim=-1), value)


class _V2CDualRole(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        produced = torch.exp(value)
        return torch.mm(produced, produced)


class _AttentionResidual(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(torch.mm(query, key.t()), dim=-1)
        return torch.mm(probabilities, value) + residual


class _RhsRoundTripPointwise(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        first_lhs: torch.Tensor,
        first_rhs: torch.Tensor,
    ) -> torch.Tensor:
        reply = torch.exp(torch.mm(first_lhs, first_rhs))
        return torch.exp(torch.mm(lhs, reply))


class _AttentionRowReduction(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(torch.mm(query, key.t()), dim=-1)
        return torch.sum(torch.mm(probabilities, value), dim=-1, keepdim=True)


class _ColumnReductionRhs(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        first_lhs: torch.Tensor,
        first_rhs: torch.Tensor,
    ) -> torch.Tensor:
        reply = torch.sum(torch.mm(first_lhs, first_rhs), dim=0, keepdim=True)
        return torch.exp(torch.mm(lhs, reply))


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


def test_cube_retained_transposed_rhs_uses_physical_owner_coordinates() -> None:
    class WideTransposedMatmul(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs.t())

    graph, result = _solve_module(
        WideTransposedMatmul(),
        (
            torch.zeros(512, 64, dtype=torch.bfloat16),
            torch.zeros(2048, 64, dtype=torch.bfloat16),
        ),
    )
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, CubeKernelPlan)
    assert plan.matmuls[0].retained_panels.rhs

    source = emit_pypto_region(
        graph, result, program_name="retained_transposed_rhs"
    ).source

    ast.parse(source)
    assert (
        "matmul_0_rhs_retained = pl.slice(arg_rhs, [256, 64], [region_col, 0 * 64])"
    ) in source
    assert (
        "matmul_0_tile_0_rhs_init = pl.slice(matmul_0_rhs_retained, [256, 16], [0, 0])"
    ) in source
    assert "a_trans=False, b_trans=True" in source
    assert "pl.slice(arg_rhs, [64, 256], [0 * 64, region_col])" not in source


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

    for schema in ("pto_fusebox.solution.v3", "pto_fusebox.solution.v5"):
        legacy = dict(copy.deepcopy(result.solution))
        legacy["schema_version"] = schema
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


def test_source_readiness_emits_a_zero_copy_transposed_matmul_operand() -> None:
    class TransposedMatmul(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs.t())

    graph, result = _solve_module(
        TransposedMatmul(),
        (torch.ones(64, 96), torch.ones(128, 96)),
        require_source_codegen=False,
    )

    assert can_emit_region(graph, result)
    source = emit_pypto_region(graph, result, program_name="transposed_matmul").source
    assert "pl.tile.move(pl.tile.transpose_view(rhs_mat_natural" in source
    assert "rhs_mat = pl.tile.transpose_view(" not in source
    assert "pl.create_tensor" not in source


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
    _assert_pypto_main_mixed_scope(source, step.plan)
    assert step.plan.max_trips_per_group == 1
    assert step.plan.pipeline_stages == 1
    assert not step.plan.overlap_implementable
    assert "pl.range(1, init_values=" in source
    assert source.count("pl.tensor.matmul(") == 2
    assert "b_trans=True" in source
    assert "pl.tensor.row_max(" in source
    assert "pl.tensor.row_sum(" in source
    assert "pl.tensor.assemble(" in source
    assert "auto_fuse" not in source and "auto_tile" not in source


def test_emitted_abi_reorders_torch_inputs_by_normalized_value_id() -> None:
    module, args = build_examples()["attention_core"]
    graph, result = _solve_module(module, args)
    emitted = emit_pypto_region(graph, result, program_name="attention_abi")

    names = graph.value_map()
    assert tuple(names[value_id].name for value_id in emitted.input_value_ids) == (
        "key",
        "query",
        "value",
    )
    bound = bind_emitted_inputs(module, graph, emitted, args)
    assert all(
        actual is expected
        for actual, expected in zip(bound, (args[1], args[0], args[2]), strict=True)
    )


def test_callable_source_exposes_the_stable_named_region_abi() -> None:
    module, args = build_examples()["attention_core"]
    graph, result = _solve_module(module, args)
    emitted = emit_pypto_callable(
        graph, result, function_name="generated_attention_core"
    )

    assert emitted.function_name == "generated_attention_core"
    assert emitted.kind is KernelKind.MIXED
    assert tuple(argument.name for argument in emitted.input_arguments) == (
        "arg_key",
        "arg_query",
        "arg_value",
    )
    assert tuple(argument.name for argument in emitted.output_arguments) == ("output",)
    assert tuple(
        graph.value_map()[value_id].name for value_id in emitted.input_value_ids
    ) == ("key", "query", "value")
    assert all(
        actual is expected
        for actual, expected in zip(
            bind_emitted_inputs(module, graph, emitted, args),
            (args[1], args[0], args[2]),
            strict=True,
        )
    )

    tree = ast.parse(emitted.source)
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) == 1
    function = functions[0]
    assert function.name == emitted.function_name
    assert [argument.arg for argument in function.args.args] == [
        "arg_key",
        "arg_query",
        "arg_value",
        "output",
    ]
    assert ast.unparse(function.decorator_list[0]) == "pl.inline"
    assert "pl.spmd(" in emitted.source
    assert "pl.split(pl.SplitMode.UP_DOWN)" in emitted.source
    assert "@pl.program" not in emitted.source
    assert "auto_fuse" not in emitted.source and "auto_tile" not in emitted.source

    standalone_tree = ast.parse(
        emit_pypto_region(graph, result, program_name="generated_attention_core").source
    )
    standalone_class = next(
        node for node in standalone_tree.body if isinstance(node, ast.ClassDef)
    )
    standalone_main = next(
        node for node in standalone_class.body if isinstance(node, ast.FunctionDef)
    )
    standalone_parameters = {argument.arg for argument in standalone_main.args.args}
    standalone_locals = {
        node.id
        for statement in standalone_main.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id not in standalone_parameters
    }
    callable_locals = {
        node.id
        for statement in function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id not in {argument.arg for argument in function.args.args}
    }
    assert callable_locals == {
        _inline_local_name(function.name, name) for name in standalone_locals
    }

    reverse_locals = {
        _inline_local_name(function.name, name): name for name in standalone_locals
    }

    class RestoreStandaloneLocals(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802 - AST API.
            restored = reverse_locals.get(node.id)
            if restored is None:
                return node
            return ast.copy_location(ast.Name(id=restored, ctx=node.ctx), node)

    restored_body = [
        RestoreStandaloneLocals().visit(statement)
        for statement in copy.deepcopy(function.body)
    ]
    assert ast.dump(ast.Module(body=restored_body, type_ignores=[])) == ast.dump(
        ast.Module(body=standalone_main.body, type_ignores=[])
    )


def test_separately_emitted_callables_have_disjoint_local_namespaces() -> None:
    module, args = build_examples()["attention_core"]
    graph, result = _solve_module(module, args)
    hidden = emit_pypto_callable(graph, result, function_name="hidden_projection")
    history = emit_pypto_callable(graph, result, function_name="history_projection")

    def local_bindings(emitted_source: str) -> set[str]:
        function = next(
            node
            for node in ast.parse(emitted_source).body
            if isinstance(node, ast.FunctionDef)
        )
        parameters = {argument.arg for argument in function.args.args}
        return {
            node.id
            for statement in function.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id not in parameters
        }

    hidden_locals = local_bindings(hidden.source)
    history_locals = local_bindings(history.source)
    assert hidden_locals
    assert history_locals
    assert hidden_locals.isdisjoint(history_locals)
    assert tuple(argument.name for argument in hidden.input_arguments) == tuple(
        argument.name for argument in history.input_arguments
    )


def test_callable_source_preserves_a_multi_step_task_graph() -> None:
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
    emitted = emit_pypto_callable(
        graph, result, function_name="generated_chained_matmul"
    )

    assert emitted.kinds == (KernelKind.CUBE, KernelKind.CUBE)
    assert emitted.source.count("pl.spmd(") == 2
    intermediate = _inline_local_name(emitted.function_name, "intermediate_tensor_2")
    step_0_index = _inline_local_name(emitted.function_name, "step_0_region_index")
    step_1_index = _inline_local_name(emitted.function_name, "step_1_region_index")
    assert f"{intermediate} = pl.create_tensor" in emitted.source
    assert emitted.source.index(f"for {step_0_index}") < emitted.source.index(
        f"for {step_1_index}"
    )


def test_callable_extraction_rejects_unexpected_program_members() -> None:
    source = """\
import pypto.language as pl

@pl.program
class UnexpectedMember:
    marker = 1

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, value: pl.Tensor[[8, 32], pl.FP32]) -> pl.Tensor[[8, 32], pl.FP32]:
        return value
"""

    with pytest.raises(
        SourceEmissionError, match="program class contains unexpected members: Assign"
    ):
        _program_as_inline_callable(source, "generated")


def test_callable_runtime_valid_shape_keeps_physical_frame_static() -> None:
    module, args = build_pr2335_examples()["pr2335_rms_norm"]
    graph, result = _solve_module(module, args)
    emitted = emit_pypto_callable(
        graph,
        result,
        function_name="generated_rms_norm_chunk",
        runtime_valid_shape=RuntimeValidShapeSpec(),
    )

    assert emitted.runtime_valid_shapes == (
        PyPTORuntimeValidShapeArgument(name="valid_rows", axis=0, physical_extent=512),
    )
    tree = ast.parse(emitted.source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    assert [argument.arg for argument in function.args.args] == [
        "arg_value",
        "arg_gamma",
        "valid_rows",
        "output",
    ]
    runtime_annotation = function.args.args[2].annotation
    assert runtime_annotation is not None
    assert ast.unparse(runtime_annotation) == "pl.Scalar[pl.INDEX]"
    assert "pl.Tensor[[512, 512], pl.FP32]" in emitted.source
    assert "pl.Out[pl.Tensor[[512, 512], pl.FP32]]" in emitted.source

    loads = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pl"
        and node.func.attr == "load"
    ]
    wide_loads = [
        node
        for node in loads
        if isinstance(node.args[0], ast.Name) and node.args[0].id == "arg_value"
    ]
    broadcast_loads = [
        node
        for node in loads
        if isinstance(node.args[0], ast.Name) and node.args[0].id == "arg_gamma"
    ]
    assert wide_loads and broadcast_loads
    assert all(
        ast.unparse(node.args[3].elts[0]).startswith("pl.max(pl.min(valid_rows - ")
        for node in wide_loads
        if isinstance(node.args[3], ast.List)
    )
    assert all(
        ast.unparse(node.args[3].elts[0]) == "1"
        for node in broadcast_loads
        if isinstance(node.args[3], ast.List)
    )


def test_callable_runtime_valid_shape_rejects_schedule_defining_variation() -> None:
    vector_module, vector_args = build_pr2335_examples()["pr2335_rms_norm"]
    vector_graph, vector_result = _solve_module(vector_module, vector_args)
    with pytest.raises(SourceEmissionError, match="only outer/free axis 0"):
        emit_pypto_callable(
            vector_graph,
            vector_result,
            runtime_valid_shape=RuntimeValidShapeSpec(axis=1),
        )
    namespaced = emit_pypto_callable(
        vector_graph,
        vector_result,
        runtime_valid_shape=RuntimeValidShapeSpec(argument_name="region_index"),
    )
    namespaced_function = next(
        node
        for node in ast.parse(namespaced.source).body
        if isinstance(node, ast.FunctionDef)
    )
    assert "region_index" in {
        argument.arg for argument in namespaced_function.args.args
    }
    assert "region_index" not in {
        node.id
        for statement in namespaced_function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    mixed_module, mixed_args = build_examples()["attention_core"]
    mixed_graph, mixed_result = _solve_module(mixed_module, mixed_args)
    with pytest.raises(SourceEmissionError, match="one homogeneous vector step"):
        emit_pypto_callable(
            mixed_graph,
            mixed_result,
            runtime_valid_shape=RuntimeValidShapeSpec(),
        )


def test_emitted_abi_binds_lifted_parameters_by_normalized_value_id() -> None:
    module, args = build_static_mixed_examples()["pypto_lib_static_dense_swiglu"]
    graph, result = _solve_module(module, args)
    emitted = emit_pypto_region(graph, result, program_name="dense_swiglu_abi")
    bound = bind_emitted_inputs(module, graph, emitted, args)

    expected_by_target = {
        "gate_weight": module.gate_weight,
        "up_weight": module.up_weight,
        "down_weight": module.down_weight,
    }
    values = graph.value_map()
    for value_id, tensor in zip(emitted.input_value_ids, bound, strict=True):
        value = values[value_id]
        if value.role == "user_input":
            assert tensor is args[0]
        else:
            assert value.target is not None
            assert tensor is expected_by_target[value.target]


def test_mixed_typed_contract_rejects_stale_fifo_geometry() -> None:
    _, result = _solved("attention_core")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["fifos"][0]["slot_bytes"] += 4

    with pytest.raises(
        ScheduleContractError, match="differs from its transfer geometry"
    ):
        scheduled_region(replace(result, solution=solution))


def test_mixed_typed_contract_rejects_missing_cube_l1_peak() -> None:
    _, result = _solved("attention_core")
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["plan"]["cube_stage_peak_l1_bytes"] = 0

    with pytest.raises(
        ScheduleContractError, match="cube_stage_peak_l1_bytes must be positive"
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


def test_mixed_source_rejects_planned_fifo_rings_over_capacity() -> None:
    graph, result = _solved("attention_core")
    assert result.problem is not None
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    c2v_fifo_bytes = sum(
        fifo.reserved_bytes
        for fifo in plan.fifos
        if fifo.direction is MixedTransferDirection.CUBE_TO_VECTOR
    )
    required_vec_bytes = plan.vector_stage_peak_ub_bytes + c2v_fifo_bytes
    assert required_vec_bytes > 0
    problem = dict(result.problem)
    problem["vec_capacity"] = required_vec_bytes - 1
    stale = replace(result, problem=problem)

    assert not can_emit_region(graph, stale)
    with pytest.raises(
        SourceEmissionError, match="C2V FIFO rings and vector stage exceed Vec capacity"
    ):
        emit_pypto_region(graph, stale)


def test_mixed_source_rejects_cube_peak_plus_v2c_rings_over_l1_capacity() -> None:
    graph, result = _solved("attention_core")
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    v2c_fifo_bytes = sum(
        fifo.reserved_bytes
        for fifo in plan.fifos
        if fifo.direction is MixedTransferDirection.VECTOR_TO_CUBE
    )
    assert plan.cube_stage_peak_l1_bytes > 0
    assert v2c_fifo_bytes > 0
    assert result.problem is not None
    problem = dict(result.problem)
    problem["l1_capacity"] = plan.cube_stage_peak_l1_bytes + v2c_fifo_bytes - 1
    stale = replace(result, problem=problem)

    assert not can_emit_region(graph, stale)
    with pytest.raises(
        SourceEmissionError, match="cube stage and V2C FIFO rings exceed L1 capacity"
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
    _assert_pypto_main_mixed_scope(source, step.plan)
    assert "pl.range(1, init_values=(output,))" in source
    assert source.count("pl.tensor.matmul(") == 1
    assert "pl.tensor.col_expand_add(" in source
    assert "pl.tensor.add(" not in source
    assert source.count("pl.tensor.assemble(") == 1


def test_one_way_c2v_can_stream_successor_items_on_fewer_mixed_groups() -> None:
    graph = export_and_normalize(
        _C2VEpilogue(),
        (
            torch.zeros(384, 64),
            torch.zeros(64, 256),
            torch.zeros(1, 256),
        ),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved == 1
    result = solved.regions[0]
    step = scheduled_region(result).steps[0]

    assert step.kind is KernelKind.MIXED
    assert isinstance(step.plan, MixedKernelPlan)
    assert step.plan.protocol is MixedCrossCoreProtocol.ONE_WAY
    assert step.plan.active_groups < min(
        step.plan.spatial_tiles, step.plan.group_capacity
    )
    assert step.plan.max_trips_per_group >= 2
    assert step.plan.pipeline_stages == 2
    assert step.plan.requested_skew_depth == 1
    assert step.plan.model_overlap_granted
    assert step.plan.overlap_implementable

    source = emit_pypto_region(graph, result, program_name="c2v_streamed").source
    ast.parse(source)
    _assert_single_spmd_grid(source, step.plan.active_groups)
    assert (
        f"pl.pipeline({step.plan.max_trips_per_group}, stage=2, init_values=(output,))"
    ) in source


@pytest.mark.parametrize(
    ("active_groups", "trips", "pipeline_stages"),
    [(24, 1, 1), (6, 4, 2)],
)
def test_one_way_c2v_replays_frozen_group_count_controls(
    active_groups: int,
    trips: int,
    pipeline_stages: int,
) -> None:
    graph = export_and_normalize(
        _C2VEpilogue(),
        (
            torch.zeros(384, 64),
            torch.zeros(64, 256),
            torch.zeros(1, 256),
        ),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    result = solved.regions[0]
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    step = solution["steps"][0]
    plan = step["plan"]
    assert plan["spatial_tiles"] == 24
    assert plan["protocol"] == "one_way"
    plan["active_groups"] = active_groups
    plan["min_trips_per_group"] = trips
    plan["max_trips_per_group"] = trips
    plan["pipeline_stages"] = pipeline_stages
    plan["requested_skew_depth"] = pipeline_stages - 1
    plan["model_overlap_granted"] = trips >= 2
    plan["overlap_implementable"] = trips >= 2
    step["launch"]["cores"] = active_groups * 3

    forced = replace(result, solution=solution)
    forced_plan = scheduled_region(forced).steps[0].plan
    assert isinstance(forced_plan, MixedKernelPlan)
    assert can_emit_region(graph, forced)
    source = emit_pypto_region(
        graph,
        forced,
        program_name=f"c2v_groups_{active_groups}",
    ).source
    _assert_single_spmd_grid(source, active_groups)
    if pipeline_stages == 2:
        assert f"pl.pipeline({trips}, stage=2" in source
    else:
        assert "pl.pipeline(" not in source
        assert f"pl.range({trips}, init_values=(output,))" in source

    inconsistent = copy.deepcopy(solution)
    inconsistent_plan = inconsistent["steps"][0]["plan"]
    inconsistent_overlap = trips < 2
    inconsistent_plan["model_overlap_granted"] = inconsistent_overlap
    inconsistent_plan["overlap_implementable"] = inconsistent_overlap
    inconsistent_plan["pipeline_stages"] = 2 if inconsistent_overlap else 1
    inconsistent_plan["requested_skew_depth"] = 1 if inconsistent_overlap else 0
    with pytest.raises(
        ScheduleContractError,
        match="one-way pipeline depth differs from its successor loop",
    ):
        scheduled_region(replace(result, solution=inconsistent))


@pytest.mark.parametrize(("active_groups", "trips"), [(12, 1), (6, 2)])
def test_cvc_replays_frozen_group_count_controls(
    active_groups: int,
    trips: int,
) -> None:
    graph = export_and_normalize(
        StaticAttentionCore(),
        (
            torch.zeros(768, 64),
            torch.zeros(64, 64),
            torch.zeros(64, 128),
        ),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    result = solved.regions[0]
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    step = solution["steps"][0]
    plan = step["plan"]
    assert plan["spatial_tiles"] == 12
    assert plan["protocol"] == "single_round_trip_bundle"
    assert result.problem is not None
    assert result.problem["require_source_codegen"] is True
    assert plan["m_partition"] == {
        "parts": 12,
        "small": 64,
        "big": 64,
        "num_big": 0,
    }
    c2v_ring_bytes = sum(
        fifo["reserved_bytes"]
        for fifo in plan["fifos"]
        if fifo["direction"] == "cube_to_vector"
    )
    assert plan["vector_stage_peak_ub_bytes"] == 49280
    assert c2v_ring_bytes == 65536
    assert plan["vector_stage_peak_ub_bytes"] + c2v_ring_bytes == 114816
    plan["active_groups"] = active_groups
    plan["min_trips_per_group"] = trips
    plan["max_trips_per_group"] = trips
    plan["model_overlap_granted"] = trips >= 2
    plan["overlap_implementable"] = trips >= 2
    plan["pipeline_fill_absorbed"] = trips >= 2
    plan["pipeline_stages"] = 3 if trips >= 2 else 1
    plan["requested_skew_depth"] = 2 if trips >= 2 else 0
    step["launch"]["cores"] = active_groups * 3

    forced = replace(result, solution=solution)
    forced_plan = scheduled_region(forced).steps[0].plan
    assert isinstance(forced_plan, MixedKernelPlan)
    assert can_emit_region(graph, forced)
    source = emit_pypto_region(
        graph,
        forced,
        program_name=f"cvc_groups_{active_groups}",
    ).source
    _assert_single_spmd_grid(source, active_groups)
    if trips >= 2:
        assert f"pl.pipeline({trips}, stage=3" in source
    else:
        assert f"pl.range({trips}, init_values=" in source
        assert "stage=3" not in source

    inconsistent = copy.deepcopy(solution)
    inconsistent_plan = inconsistent["steps"][0]["plan"]
    inconsistent_overlap = trips < 2
    inconsistent_plan["model_overlap_granted"] = inconsistent_overlap
    inconsistent_plan["overlap_implementable"] = inconsistent_overlap
    inconsistent_plan["pipeline_fill_absorbed"] = inconsistent_overlap
    with pytest.raises(
        ScheduleContractError,
        match="round-trip pipeline differs from its successor loop",
    ):
        scheduled_region(replace(result, solution=inconsistent))


def test_mixed_plan_rejects_launch_participation_drift() -> None:
    graph = export_and_normalize(
        _C2VEpilogue(),
        (
            torch.zeros(32, 64),
            torch.zeros(64, 32),
            torch.zeros(1, 32),
        ),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    result = solved.regions[0]
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    solution["steps"][0]["launch"]["cores"] -= 1
    with pytest.raises(
        ScheduleContractError,
        match="launch cores differ from its mixed group participation",
    ):
        scheduled_region(replace(result, solution=solution))


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
    _assert_pypto_main_mixed_scope(source, step.plan)
    assert "pl.pipeline(0, 128, 64, stage=3" in source
    assert source.count("pl.tensor.matmul(") == 3
    assert source.count("pl.tensor.matmul_acc(") == 1
    assert "pl.tensor.recip(" in source
    assert 'target_type=pl.BF16, mode="round"' in source


def test_multi_round_trip_attention_epilogue_emits_one_ordered_generic_loop() -> None:
    graph, result = _solve_module(
        _AttentionResidual(),
        (
            torch.zeros(96, 64),
            torch.zeros(64, 64),
            torch.zeros(64, 128),
            torch.zeros(96, 128),
        ),
    )
    step = scheduled_region(result).steps[0]

    assert step.kind is KernelKind.MIXED
    assert isinstance(step.plan, MixedKernelPlan)
    assert step.plan.algorithm is MixedAlgorithm.GENERIC
    assert step.plan.protocol is MixedCrossCoreProtocol.MULTI_ROUND_TRIP_SEQUENTIAL
    assert step.plan.mode is MixedPipelineMode.MULTI_ROUND_TRIP_SEQUENTIAL
    assert not step.plan.model_overlap_granted
    assert not step.plan.overlap_implementable
    assert can_emit_region(graph, result)

    source = emit_pypto_region(graph, result, program_name="attention_residual").source
    ast.parse(source)
    _assert_single_spmd_grid(source, step.plan.active_groups)
    _assert_pypto_main_mixed_scope(source, step.plan)
    assert "for mixed_trip, (output_iter,) in pl.range(" in source
    assert "pl.pipeline(" not in source
    assert source.count("pl.tensor.matmul(") == 2
    assert "pl.tensor.row_max(" in source
    assert "pl.tensor.row_sum(" in source
    assert "pl.tensor.add(" in source
    assert source.index("first_cube_acc_first") < source.index("vector_3")
    assert source.index("vector_7") < source.index("second_cube_acc_first")
    assert source.index("second_cube_acc_first") < source.index("vector_11")


@pytest.mark.parametrize(
    ("module", "args", "expected_axes"),
    [
        (
            _RhsRoundTripPointwise(),
            (
                torch.zeros(96, 64),
                torch.zeros(64, 64),
                torch.zeros(64, 32),
            ),
            ((False, True), (False, True), (True, True)),
        ),
    ],
)
def test_multi_round_trip_replays_transfer_specific_axes(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    expected_axes: tuple[tuple[bool, bool], ...],
) -> None:
    graph, result = _solve_module(module, args)
    step = scheduled_region(result).steps[0]
    assert isinstance(step.plan, MixedKernelPlan)
    assert tuple((fifo.spatial_m, fifo.spatial_n) for fifo in step.plan.fifos) == (
        expected_axes
    )
    assert can_emit_region(graph, result)
    source = emit_pypto_region(graph, result, program_name="axis_replay").source
    ast.parse(source)
    _assert_pypto_main_mixed_scope(source, step.plan)


def test_multi_round_trip_final_row_reduction_emits_as_a_following_vector_step() -> (
    None
):
    module = _AttentionRowReduction()
    args = (
        torch.zeros(96, 64),
        torch.zeros(64, 64),
        torch.zeros(64, 128),
    )
    graph, analytic = _solve_module(module, args, require_source_codegen=False)
    analytic_schedule = scheduled_region(analytic)
    assert [step.kind for step in analytic_schedule.steps] == [
        KernelKind.MIXED,
        KernelKind.VECTOR,
    ]
    assert can_emit_region(graph, analytic)
    source = emit_pypto_region(
        graph, analytic, program_name="attention_row_reduction"
    ).source
    ast.parse(source)

    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved
    assert solved.regions[0].solution is not None
    assert can_emit_region(graph, solved.regions[0])


def test_column_reduction_round_trip_stops_at_the_static_frontend_boundary() -> None:
    graph = export_and_normalize(
        _ColumnReductionRhs(),
        (
            torch.zeros(96, 1),
            torch.zeros(64, 64),
            torch.zeros(64, 32),
        ),
    )
    assert any(
        not op.supported
        and op.opaque_reason
        == "only last-axis reductions with keepdim=True are supported"
        for op in graph.ops
    )


@pytest.mark.parametrize(
    ("module", "args", "expected_axes"),
    [
        (
            _V2COnly(),
            (torch.zeros(96, 64), torch.zeros(64, 128)),
            (True, False),
        ),
        (
            _V2COnlyRhs(),
            (
                torch.zeros(96, 64),
                torch.zeros(64, 128),
                torch.zeros(1, 128),
            ),
            (False, True),
        ),
    ],
)
def test_one_way_v2c_emits_vector_producer_and_matmul_consumer(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    expected_axes: tuple[bool, bool],
) -> None:
    graph, result = _solve_module(module, args)
    step = scheduled_region(result).steps[0]

    assert step.kind is KernelKind.MIXED
    assert isinstance(step.plan, MixedKernelPlan)
    assert step.plan.protocol is MixedCrossCoreProtocol.ONE_WAY
    assert tuple(stage.engine for stage in step.plan.stages) == (
        MixedEngine.VECTOR,
        MixedEngine.CUBE,
    )
    assert len(step.plan.fifos) == 1
    fifo = step.plan.fifos[0]
    assert fifo.direction is MixedTransferDirection.VECTOR_TO_CUBE
    assert (fifo.spatial_m, fifo.spatial_n) == expected_axes
    assert can_emit_region(graph, result)

    source = emit_pypto_region(graph, result, program_name="v2c_one_way").source
    ast.parse(source)
    _assert_single_spmd_grid(source, step.plan.active_groups)
    _assert_pypto_main_mixed_scope(source, step.plan)
    assert "pl.tensor.exp(" in source
    if not expected_axes[0]:
        assert "pl.tensor.col_expand_add(" in source
    assert source.count("pl.tensor.matmul(") == 1
    assert source.index("pl.tensor.exp(") < source.index("pl.tensor.matmul(")
    assert source.count("pl.tensor.assemble(") == 1


def test_one_way_v2c_replays_a_frozen_stage_two_group_loop() -> None:
    graph, result = _solve_module(
        _V2COnly(),
        (torch.zeros(96, 64), torch.zeros(64, 128)),
    )
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    step = solution["steps"][0]
    plan = step["plan"]
    assert plan["spatial_tiles"] == 2
    plan["active_groups"] = 1
    plan["min_trips_per_group"] = 2
    plan["max_trips_per_group"] = 2
    plan["pipeline_stages"] = 2
    plan["requested_skew_depth"] = 1
    plan["model_overlap_granted"] = True
    plan["overlap_implementable"] = True
    step["launch"]["cores"] = 3

    forced = replace(result, solution=solution)
    forced_plan = scheduled_region(forced).steps[0].plan
    assert isinstance(forced_plan, MixedKernelPlan)
    assert can_emit_region(graph, forced)
    source = emit_pypto_region(
        graph,
        forced,
        program_name="v2c_stage_two",
    ).source
    _assert_single_spmd_grid(source, 1)
    assert "pl.pipeline(2, stage=2" in source


@pytest.mark.parametrize(
    ("module", "external_role"),
    [(_V2CSharedLhs(), "rhs"), (_V2CSharedRhs(), "lhs")],
)
def test_one_way_v2c_reloads_shared_boundary_operand_from_gm(
    module: nn.Module,
    external_role: str,
) -> None:
    graph, result = _solve_module(module, (torch.zeros(64, 64),))
    step = scheduled_region(result).steps[0]
    assert isinstance(step.plan, MixedKernelPlan)
    assert len(step.plan.fifos) == 1

    source = emit_pypto_region(graph, result, program_name="v2c_shared").source
    _assert_pypto_main_mixed_scope(source, step.plan)
    assert f"sink_{external_role}_first_tile = pl.tensor.slice(arg_value" in source


def test_streaming_softmax_to_pv_replays_one_typed_publication_loop() -> None:
    graph, result = _solve_module(
        _StreamingSoftmaxPv(),
        (torch.zeros(16, 4096), torch.zeros(4096, 64)),
    )
    step = scheduled_region(result).steps[0]
    assert isinstance(step.plan, MixedKernelPlan)
    plan = step.plan
    stream = plan.stages[0].vector_stream
    assert stream is not None
    assert stream.kind is VectorStreamKind.SOFTMAX_FLASH
    assert plan.source_codegen_ready
    assert plan.fifos[0].valid_cols == stream.chunk
    assert plan.stages[1].cube_window_k == (stream.chunk,)

    source = emit_pypto_region(
        graph, result, program_name="streaming_softmax_pv"
    ).source
    ast.parse(source)
    _assert_single_spmd_grid(source, plan.active_groups)
    _assert_pypto_main_mixed_scope(source, plan)
    assert "for stats_chunk" in source
    assert "for apply_chunk" in source
    apply_phase = stream.phase(VectorReplayPhase.APPLY)
    assert apply_phase.loop is not None
    assert apply_phase.tail is not None
    assert (
        f"for apply_chunk, (sink_acc,) in pl.pipeline("
        f"{apply_phase.loop.first_chunk}, "
        f"{apply_phase.loop.first_chunk + apply_phase.loop.trip_count}, "
        f"stage={apply_phase.loop.pipeline_stages}"
    ) in source
    assert source.count("pl.tensor.matmul(") == 1
    assert source.count("pl.tensor.matmul_acc(") == 2
    assert "apply_tail" in source
    assert f"valid_shape=[16, {apply_phase.tail.extent}]" in source
    assert "stats_result_max = stats_tail_next_max" not in source
    assert "stats_result_sum = stats_tail_next_sum" not in source
    assert "pl.tensor.row_expand_sub(apply_input, stats_tail_next_max)" in source
    assert "pl.tensor.row_expand_div(apply_tensor_3, stats_tail_next_sum)" in source


def test_streaming_softmax_to_pv_keeps_phase_local_pipeline_separate() -> None:
    # Freeze the mixed emitter contract directly. Source-oriented whole-region
    # costing may legitimately cut this graph into homogeneous kernels.
    graph, result = _solve_module(
        _StreamingSoftmaxPv(),
        (torch.zeros(384, 4096), torch.zeros(4096, 64)),
        require_source_codegen=False,
    )
    plan = scheduled_region(result).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    assert plan.spatial_tiles == 24
    assert plan.active_groups == 24
    assert plan.max_trips_per_group == 1
    assert plan.pipeline_stages == 1
    assert not plan.model_overlap_granted
    assert not plan.overlap_implementable

    source = emit_pypto_region(
        graph,
        result,
        program_name="streaming_softmax_pv_phase_local",
    ).source
    assert "pl.range(1, init_values=(output,))" in source
    assert "for apply_chunk, (sink_acc,) in pl.pipeline(" in source


def test_one_way_v2c_dual_role_uses_one_complete_fifo_panel() -> None:
    graph, result = _solve_module(
        _V2CDualRole(), (torch.zeros(64, 64),), require_source_codegen=True
    )
    step = scheduled_region(result).steps[0]
    assert isinstance(step.plan, MixedKernelPlan)
    plan = step.plan
    assert plan.m_partition.parts == 1
    assert plan.n_partition.parts == 1
    assert len(plan.fifos) == 1
    assert plan.fifos[0].spatial_m
    assert plan.fifos[0].spatial_n

    source = emit_pypto_region(graph, result, program_name="v2c_dual_role").source
    ast.parse(source)
    _assert_pypto_main_mixed_scope(source, plan)
    assert "pl.tensor.matmul(vector_1, vector_1" in source


def test_one_way_v2c_dual_role_rejects_partitioned_source_contract() -> None:
    graph, result = _solve_module(
        _V2CDualRole(), (torch.zeros(64, 64),), require_source_codegen=True
    )
    assert result.solution is not None
    solution = copy.deepcopy(result.solution)
    step = solution["steps"][0]
    plan = step["plan"]

    partition = {"big": 32, "small": 32, "num_big": 0, "parts": 2}
    plan["m_partition"] = partition
    plan["n_partition"] = copy.deepcopy(partition)
    plan["cube_stage_peak_l1_bytes"] = 8192
    plan["spatial_tiles"] = 4
    plan["work_units"] = 4
    plan["active_groups"] = 4
    plan["pipeline_extent"] = 4
    step["launch"]["parts"] = [2, 2]
    step["launch"]["tile"] = [32, 32, 64]
    step["launch"]["cores"] = 12
    fifo = plan["fifos"][0]
    fifo["valid_rows"] = 32
    fifo["valid_cols"] = 32
    fifo["slot_bytes"] = 4096
    fifo["reserved_bytes"] = 32768

    mutated = replace(result, solution=solution)
    mutated_plan = scheduled_region(mutated).steps[0].plan
    assert isinstance(mutated_plan, MixedKernelPlan)
    assert mutated_plan.m_partition.parts == 2
    assert mutated_plan.n_partition.parts == 2
    assert not can_emit_region(graph, mutated)
    with pytest.raises(
        SourceEmissionError,
        match="dual-role FIFO requires one complete square spatial region",
    ):
        emit_pypto_region(graph, mutated)


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

    graph, result = _solve_module(
        ReductionResultCast(),
        (torch.randn(16, 512),),
        require_source_codegen=False,
    )

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
    graph, result = _solve_module(
        FloatToInt8(), (values,), require_source_codegen=False
    )

    assert result.status == "solved"
    assert result.problem is not None
    assert result.problem["dtypes"] == ["FP32", "FP16", "INT8"]
    assert not can_emit_region(graph, result)
    with pytest.raises(SourceEmissionError, match="Torch float-to-INT8 truncation"):
        emit_pypto_region(graph, result)


def test_cast_chain_broadcast_alignment_matches_non_singleton_axis() -> None:
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
    assert body_frames[bias_tensor].physical[0] == 16
    assert body_frames[bias_tensor].physical[1] == 1
    source = emit_pypto_region(graph, result).source
    assert "[16, 1], [3, 1], target_memory=pl.Mem.Vec" in source
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
