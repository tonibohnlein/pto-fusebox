"""Mechanical replay of homogeneous cube schedules as PyPTO DSL."""

from __future__ import annotations

from ..schedule import CubeKernelPlan
from ..schedule.schema import (
    CubeMatmulPlan,
    CubeSpatialPolicy,
    CubeSplitMergePolicy,
)
from .common import (
    EmissionContext,
    SourceEmissionError,
    SourceWriter,
    emit_partition_indices,
    literal,
    program_header,
    solver_tensor_for_value,
    static_shape,
    validate_grid,
    validate_partition_extent,
)


def emit_cube(
    context: EmissionContext,
    program_name: str,
) -> str:
    """Emit the installed single-matmul cube schedule subset."""

    graph = context.graph
    lowered = context.lowered
    step = context.step
    io = context.interface
    plan = step.plan
    if not isinstance(plan, CubeKernelPlan):
        raise SourceEmissionError("cube step does not carry a cube plan")
    if step.retained_tensors:
        raise SourceEmissionError(
            "single-step cube source cannot carry inter-kernel retained tensors"
        )
    if not plan.emit_compatible:
        raise SourceEmissionError("cube plan is not marked emit-compatible")
    if plan.spatial_policy is not CubeSpatialPolicy.UNIFORM:
        raise SourceEmissionError(
            "cube source v1 supports only uniform spatial partitions"
        )
    if (
        step.split != 1
        or plan.split_k != 1
        or plan.split_merge_policy is not CubeSplitMergePolicy.NONE
    ):
        raise SourceEmissionError(
            "cube split-K emission is not implemented in source v1"
        )
    if plan.resident_boundaries:
        raise SourceEmissionError(
            "cube resident-boundary emission is not implemented in source v1"
        )
    matmuls = plan.matmuls
    if len(matmuls) != 1 or len(step.solver_ops) != 1:
        raise SourceEmissionError("cube source v1 supports exactly one matmul")
    matmul = matmuls[0]
    solver_op = step.solver_ops[0]
    if plan.execution_order != (solver_op,):
        raise SourceEmissionError(
            "cube execution order does not match the selected one-matmul step"
        )
    if matmul.instance != 0 or matmul.op != solver_op:
        raise SourceEmissionError("cube request identity does not match its solver op")
    if matmul.retained_panels.lhs or matmul.retained_panels.rhs:
        raise SourceEmissionError(
            "cube retained-panel emission is not implemented in source v1"
        )
    if matmul.lhs_producer != -1 or matmul.rhs_producer != -1:
        raise SourceEmissionError("cube source v1 requires external matmul operands")
    if matmul.output_grid != (1, 1):
        raise SourceEmissionError(
            "cube source v1 requires one L0 output tile per region"
        )
    if matmul.accumulator_dtype != "fp32" or matmul.storage_dtype != "fp32":
        raise SourceEmissionError(
            "cube source v1 currently supports FP32 accumulation and storage"
        )

    m_partition = plan.m_partition
    n_partition = plan.n_partition
    validate_grid(step, plan.work_units, m_partition, n_partition)
    if m_partition.big != m_partition.small or n_partition.big != n_partition.small:
        raise SourceEmissionError(
            "cube source v1 does not yet emit ragged spatial regions"
        )

    graph_op = graph.op_map()[lowered.operation(solver_op).graph_op_id]
    if (
        graph_op.kind != "matmul"
        or graph_op.attributes.get("lhs_transposed")
        or graph_op.attributes.get("rhs_transposed")
    ):
        raise SourceEmissionError("cube source v1 requires a non-transposed matmul")
    lowered_op = lowered.operation(solver_op)
    op_inputs = list(lowered_op.inputs)
    op_outputs = list(lowered_op.outputs)
    if len(op_inputs) != 2 or len(op_outputs) != 1:
        raise SourceEmissionError("cube matmul must have two inputs and one output")
    output_rows, output_cols = static_shape(
        graph.value_map()[io.output_value], field="cube output"
    )
    validate_partition_extent(m_partition, output_rows, "cube.m_partition")
    validate_partition_extent(n_partition, output_cols, "cube.n_partition")
    lhs_value = lowered.tensor(op_inputs[0]).value_id
    rhs_value = lowered.tensor(op_inputs[1]).value_id
    if lhs_value not in io.input_arguments or rhs_value not in io.input_arguments:
        raise SourceEmissionError("cube operands must be direct region inputs")
    if solver_tensor_for_value(lowered, io.output_allocation_owner) != op_outputs[0]:
        raise SourceEmissionError("cube matmul result must be the region output")

    k_loop = matmul.k_loop
    chunk = k_loop.chunk
    full_chunks = k_loop.full_chunks
    tail = k_loop.tail
    stages = k_loop.pipeline_stages
    if (
        step.sequential_tiles != (k_loop.l1_window_k,)
        or step.launch.tile_k != k_loop.l1_window_k
    ):
        raise SourceEmissionError(
            "cube launch and sequential tile differ from its L1 K window"
        )
    contraction = matmul.contraction
    if full_chunks * chunk + tail != contraction or full_chunks == 0:
        raise SourceEmissionError(
            "cube K-window descriptor does not cover the contraction"
        )
    if stages not in {1, 2} or (full_chunks == 1 and stages != 1):
        raise SourceEmissionError("cube K-window pipeline stage count is unsupported")
    if tail >= chunk:
        raise SourceEmissionError("cube K-window tail must be smaller than its chunk")
    output_tile = list(matmul.output_tile)
    if output_tile != [m_partition.big, n_partition.big]:
        raise SourceEmissionError(
            "cube output tile does not match its spatial partition"
        )
    _validate_l0_variant(matmul, output_tile, chunk, tail, full_chunks)

    writer = program_header(
        program_name, io, graph, m_partition.parts * n_partition.parts
    )
    indent = 4
    emit_partition_indices(writer, indent, m_partition, n_partition, emit_extents=False)
    writer.line(
        indent,
        f"with pl.at(level=pl.Level.CORE_GROUP, name_hint={literal(context.region_id + '_cube')}):",
    )
    indent += 1
    lhs_arg = io.input_arguments[lhs_value]
    rhs_arg = io.input_arguments[rhs_value]
    _emit_cube_window(
        writer, indent, lhs_arg, rhs_arg, output_tile, chunk, "0", first=True
    )
    if full_chunks > 1:
        loop = "pl.pipeline" if stages > 1 else "pl.range"
        stage = f", stage={stages}" if stages > 1 else ""
        writer.line(indent, f"for k_window in {loop}(1, {full_chunks}{stage}):")
        _emit_cube_window(
            writer,
            indent + 1,
            lhs_arg,
            rhs_arg,
            output_tile,
            chunk,
            f"k_window * {chunk}",
            first=False,
        )
    if tail:
        _emit_cube_window(
            writer,
            indent,
            lhs_arg,
            rhs_arg,
            output_tile,
            tail,
            str(full_chunks * chunk),
            first=False,
            suffix="_tail",
        )
    writer.line(
        indent,
        f"{io.output_argument} = pl.store(accumulator, [region_row, region_col], "
        f"{io.output_argument})",
    )
    writer.line(2, f"return {io.output_argument}")
    return writer.render()


def _emit_cube_window(
    writer: SourceWriter,
    indent: int,
    lhs: str,
    rhs: str,
    output_tile: list[int],
    k_extent: int,
    k_offset: str,
    *,
    first: bool,
    suffix: str = "",
) -> None:
    m_extent, n_extent = output_tile
    writer.line(
        indent,
        f"lhs_mat{suffix} = pl.tile.load({lhs}, [region_row, {k_offset}], "
        f"[{m_extent}, {k_extent}], target_memory=pl.Mem.Mat)",
    )
    writer.line(
        indent,
        f"rhs_mat{suffix} = pl.tile.load({rhs}, [{k_offset}, region_col], "
        f"[{k_extent}, {n_extent}], target_memory=pl.Mem.Mat)",
    )
    writer.line(
        indent,
        f"lhs_left{suffix} = pl.tile.move(lhs_mat{suffix}, target_memory=pl.Mem.Left)",
    )
    writer.line(
        indent,
        f"rhs_right{suffix} = pl.tile.move(rhs_mat{suffix}, target_memory=pl.Mem.Right)",
    )
    if first:
        writer.line(
            indent,
            f"accumulator = pl.tile.matmul(lhs_left{suffix}, rhs_right{suffix})",
        )
    else:
        writer.line(
            indent,
            f"accumulator = pl.tile.matmul_acc(accumulator, lhs_left{suffix}, rhs_right{suffix})",
        )


def _validate_l0_variant(
    matmul: CubeMatmulPlan,
    output_tile: list[int],
    chunk: int,
    tail: int,
    full_chunks: int,
) -> None:
    variants = matmul.output_variants
    if len(variants) != 1:
        raise SourceEmissionError("cube source v1 requires one output-shape variant")
    variant = variants[0]
    if variant.count != 1 or variant.shape != tuple(output_tile):
        raise SourceEmissionError("cube output variant does not match one region tile")
    expected = (output_tile[0], output_tile[1], chunk)
    if variant.l0_init.tile != expected:
        raise SourceEmissionError(
            "cube initial L0 tile differs from the emitted K window"
        )
    if full_chunks > 1:
        if variant.l0_rolled is None or variant.l0_rolled.tile != expected:
            raise SourceEmissionError(
                "cube rolled L0 tile differs from the emitted K window"
            )
    if tail:
        expected_tail = (output_tile[0], output_tile[1], tail)
        if variant.l0_tail is None or variant.l0_tail.tile != expected_tail:
            raise SourceEmissionError("cube tail L0 tile differs from the emitted tail")
