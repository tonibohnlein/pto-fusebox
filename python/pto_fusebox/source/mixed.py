"""Mechanical replay of supported mixed cube/vector schedules as PyPTO DSL."""

from __future__ import annotations

from collections.abc import Mapping

from ..ir import NormalizedOp
from ..schedule import MixedKernelPlan
from ..schedule.schema import (
    MixedAlgorithm,
    MixedCrossCoreProtocol,
    MixedEngine,
    MixedPipelineAxis,
    MixedPipelineMode,
    MixedStagePlan,
    MixedTransferDirection,
    MixedVectorSplit,
    VectorStreamKind,
)
from .common import (
    EmissionContext,
    SourceEmissionError,
    SourceWriter,
    broadcast_operands,
    emit_return,
    literal,
    program_preamble,
    pypto_dtype,
    scalar_operand,
    solver_tensor_for_value,
)


def emit_mixed(context: EmissionContext, program_name: str) -> str:
    """Emit one solver-owned mixed schedule through PyPTO's split pipeline."""

    plan = context.step.plan
    if not isinstance(plan, MixedKernelPlan):
        raise SourceEmissionError("mixed step does not carry a mixed plan")
    if len(context.schedule.steps) != 1:
        raise SourceEmissionError(
            "mixed source currently requires the region's only selected step"
        )
    _validate_common(context, plan)
    if plan.algorithm is MixedAlgorithm.DENSE_SWIGLU_MLP:
        return _emit_dense_swiglu(context, program_name, plan)
    if plan.protocol is MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE:
        return _emit_single_round_trip(context, program_name, plan)
    if plan.protocol is MixedCrossCoreProtocol.ONE_WAY:
        return _emit_one_way_c2v(context, program_name, plan)
    raise SourceEmissionError(
        f"mixed source does not implement protocol {plan.protocol.value!r}"
    )


def _validate_common(context: EmissionContext, plan: MixedKernelPlan) -> None:
    if not plan.emit_compatible or not plan.source_codegen_ready:
        raise SourceEmissionError("mixed plan is not source-codegen ready")
    if plan.split_k != 1 or context.step.split != 1:
        raise SourceEmissionError("mixed source currently requires split_k=1")
    if len(context.interface.output_values) != 1:
        raise SourceEmissionError("mixed source currently requires one region output")
    if (
        plan.m_partition.big != plan.m_partition.small
        or plan.n_partition.big != plan.n_partition.small
        or plan.m_partition.num_big != 0
        or plan.n_partition.num_big != 0
    ):
        raise SourceEmissionError("mixed source requires a uniform spatial grid")
    if (
        plan.vector_split is not MixedVectorSplit.ROWS
        or plan.vector_lanes != 2
        or plan.m_partition.big % plan.vector_lanes != 0
    ):
        raise SourceEmissionError(
            "mixed source requires the supported two-lane row split"
        )
    if plan.model_overlap_granted != plan.overlap_implementable:
        raise SourceEmissionError("mixed model and source overlap contracts disagree")
    if plan.active_groups * plan.max_trips_per_group != plan.pipeline_work_items:
        raise SourceEmissionError(
            "mixed group loop does not cover its logical work items"
        )
    if not plan.fifos:
        raise SourceEmissionError("mixed source requires at least one FIFO")
    slot_counts = {fifo.slot_count for fifo in plan.fifos}
    if len(slot_counts) != 1:
        raise SourceEmissionError("mixed FIFOs require one shared slot count")
    vec_capacity = context.problem.get("vec_capacity")
    if (
        not isinstance(vec_capacity, int)
        or isinstance(vec_capacity, bool)
        or vec_capacity <= 0
    ):
        raise SourceEmissionError("mixed source requires a positive Vec capacity")
    physical_fifo_bytes = max(fifo.reserved_bytes for fifo in plan.fifos)
    required_vec_bytes = physical_fifo_bytes + plan.vector_stage_peak_ub_bytes
    if required_vec_bytes > vec_capacity:
        raise SourceEmissionError(
            "mixed shared FIFO and vector stage exceed Vec capacity: "
            f"{required_vec_bytes} > {vec_capacity} bytes"
        )


def _mixed_header(
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
) -> SourceWriter:
    slot_count = plan.fifos[0].slot_count
    writer = program_preamble(program_name, context.interface, context.graph)
    writer.line(
        2,
        f"for region_index in pl.spmd({plan.active_groups}, "
        f"name_hint={literal(context.region_id + '_mixed')}, "
        "optimizations=[pl.split(pl.SplitMode.UP_DOWN, "
        f"slot_num={slot_count})]):",
    )
    return writer


def _emit_one_way_c2v(
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
) -> str:
    if (
        plan.algorithm is not MixedAlgorithm.GENERIC
        or plan.mode is not MixedPipelineMode.ONE_WAY
        or plan.pipeline_axis is not MixedPipelineAxis.SPATIAL_REGION
        or plan.pipeline_stages != 1
        or len(plan.stages) != 2
        or tuple(stage.engine for stage in plan.stages)
        != (MixedEngine.CUBE, MixedEngine.VECTOR)
        or len(plan.stages[0].ops) != 1
        or len(plan.transfers) != 1
        or len(plan.fifos) != 1
        or plan.fifos[0].direction is not MixedTransferDirection.CUBE_TO_VECTOR
        or plan.fifos[0].slot_count != 8
    ):
        raise SourceEmissionError(
            "mixed plan is not the supported one-way C->V topology"
        )
    vector_stage = plan.stages[1]
    _require_materialized_vector_stage(vector_stage)
    crossing = plan.transfers[0].tensor
    cube_op = plan.stages[0].ops[0]
    if context.lowered.operation(cube_op).outputs != (crossing,):
        raise SourceEmissionError("C->V transfer is not the cube-stage result")

    output_tensor = solver_tensor_for_value(
        context.lowered, context.interface.output_allocation_owner
    )
    if context.lowered.operation(vector_stage.ops[-1]).outputs != (output_tensor,):
        raise SourceEmissionError(
            "C->V vector stage does not produce the region output"
        )

    writer = _mixed_header(context, program_name, plan)
    output = context.interface.output_argument
    trips = plan.max_trips_per_group
    writer.line(
        3,
        f"for mixed_trip, (output_iter,) in pl.range({trips}, "
        f"init_values=({output},)):",
    )
    row, col = _emit_spatial_coordinates(writer, 4, plan, "mixed_trip")
    local = {
        crossing: _emit_matmul_tile(
            writer,
            4,
            context,
            cube_op,
            plan.stages[0].cube_window_k[0],
            rows=plan.m_partition.big,
            cols=plan.n_partition.big,
            row_offset=row,
            col_offset=col,
            local={},
            prefix="producer",
        )
    }
    result = _emit_vector_stage(
        writer,
        4,
        context,
        vector_stage,
        local,
        frame_rows=plan.m_partition.big,
        frame_cols=plan.n_partition.big,
        row_offset=row,
        col_offset=col,
        allow_external=True,
    )
    writer.line(
        4,
        f"next_output = pl.tensor.assemble(output_iter, {result}, [{row}, {col}])",
    )
    writer.line(4, f"{output} = pl.yield_(next_output)")
    emit_return(writer, context.interface)
    return writer.render()


def _emit_single_round_trip(
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
) -> str:
    if (
        plan.algorithm is not MixedAlgorithm.GENERIC
        or plan.mode is not MixedPipelineMode.SINGLE_ROUND_TRIP_SKEW
        or plan.pipeline_axis is not MixedPipelineAxis.SPATIAL_REGION
        or plan.pipeline_stages != 3
        or plan.requested_skew_depth != 2
        or len(plan.stages) != 3
        or tuple(stage.engine for stage in plan.stages)
        != (MixedEngine.CUBE, MixedEngine.VECTOR, MixedEngine.CUBE)
        or len(plan.stages[0].ops) != 1
        or len(plan.stages[2].ops) != 1
        or len(plan.transfers) != 2
        or tuple(fifo.direction for fifo in plan.fifos)
        != (
            MixedTransferDirection.CUBE_TO_VECTOR,
            MixedTransferDirection.VECTOR_TO_CUBE,
        )
        or any(fifo.slot_count != 4 for fifo in plan.fifos)
    ):
        raise SourceEmissionError(
            "mixed plan is not the supported generic C->V->C topology"
        )
    vector_stage = plan.stages[1]
    _require_materialized_vector_stage(vector_stage)
    first_op = plan.stages[0].ops[0]
    sink_op = plan.stages[2].ops[0]
    first_crossing = plan.transfers[0].tensor
    reply_crossing = plan.transfers[1].tensor
    if (
        context.lowered.operation(first_op).outputs != (first_crossing,)
        or context.lowered.operation(sink_op).inputs[0] != reply_crossing
    ):
        raise SourceEmissionError("round-trip transfers do not connect the cube stages")
    crossing_rows = plan.fifos[0].valid_rows
    crossing_cols = plan.fifos[0].valid_cols
    if (crossing_rows, crossing_cols) != (
        plan.fifos[1].valid_rows,
        plan.fifos[1].valid_cols,
    ) or crossing_rows != plan.m_partition.big:
        raise SourceEmissionError("round-trip FIFO frames disagree")

    writer = _mixed_header(context, program_name, plan)
    output = context.interface.output_argument
    trips = plan.max_trips_per_group
    writer.line(
        3,
        f"for mixed_trip, (output_iter,) in pl.pipeline({trips}, stage=3, "
        f"init_values=({output},)):",
    )
    row, col = _emit_spatial_coordinates(writer, 4, plan, "mixed_trip")
    local = {
        first_crossing: _emit_matmul_tile(
            writer,
            4,
            context,
            first_op,
            plan.stages[0].cube_window_k[0],
            rows=crossing_rows,
            cols=crossing_cols,
            row_offset=row,
            col_offset="0",
            local={},
            prefix="producer",
        )
    }
    local[reply_crossing] = _emit_vector_stage(
        writer,
        4,
        context,
        vector_stage,
        local,
        frame_rows=crossing_rows,
        frame_cols=crossing_cols,
        row_offset=row,
        col_offset="0",
        allow_external=True,
    )
    result = _emit_matmul_tile(
        writer,
        4,
        context,
        sink_op,
        plan.stages[2].cube_window_k[0],
        rows=plan.m_partition.big,
        cols=plan.n_partition.big,
        row_offset=row,
        col_offset=col,
        local=local,
        prefix="sink",
    )
    writer.line(
        4,
        f"next_output = pl.tensor.assemble(output_iter, {result}, [{row}, {col}])",
    )
    writer.line(4, f"{output} = pl.yield_(next_output)")
    emit_return(writer, context.interface)
    return writer.render()


def _emit_dense_swiglu(
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
) -> str:
    dense = plan.dense_mlp
    if (
        dense is None
        or plan.protocol is not MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE
        or plan.mode is not MixedPipelineMode.SINGLE_ROUND_TRIP_SKEW
        or plan.pipeline_axis is not MixedPipelineAxis.INTERMEDIATE_FEATURE_CHUNK
        or len(plan.stages) != 4
        or tuple(stage.engine for stage in plan.stages)
        != (
            MixedEngine.CUBE,
            MixedEngine.CUBE,
            MixedEngine.VECTOR,
            MixedEngine.CUBE,
        )
        or any(len(plan.stages[index].ops) != 1 for index in (0, 1, 3))
        or len(plan.fifos) != 3
        or tuple(fifo.direction for fifo in plan.fifos)
        != (
            MixedTransferDirection.CUBE_TO_VECTOR,
            MixedTransferDirection.CUBE_TO_VECTOR,
            MixedTransferDirection.VECTOR_TO_CUBE,
        )
        or any(fifo.slot_count != 4 for fifo in plan.fifos)
    ):
        raise SourceEmissionError(
            "mixed plan is not the supported dense SwiGLU topology"
        )
    gate_op = plan.stages[0].ops[0]
    up_op = plan.stages[1].ops[0]
    vector_stage = plan.stages[2]
    down_op = plan.stages[3].ops[0]
    _require_materialized_vector_stage(vector_stage)
    gate = context.lowered.operation(gate_op)
    up = context.lowered.operation(up_op)
    down = context.lowered.operation(down_op)
    if gate.inputs[0] != up.inputs[0] or down.inputs[0] != plan.transfers[2].tensor:
        raise SourceEmissionError(
            "dense SwiGLU projection wiring differs from its plan"
        )
    if (plan.transfers[0].tensor, plan.transfers[1].tensor) != (
        gate.outputs[0],
        up.outputs[0],
    ) or tuple(plan.stages[2].ops) == ():
        raise SourceEmissionError("dense SwiGLU producer bundle is incomplete")
    rows = plan.m_partition.big
    cols = plan.n_partition.big
    if dense.persistent_accumulator_bytes != rows * cols * 4:
        raise SourceEmissionError("dense SwiGLU persistent accumulator size is stale")

    writer = _mixed_header(context, program_name, plan)
    row, col = _emit_spatial_coordinates(writer, 3, plan, None)
    output_dtype = pypto_dtype(context.lowered.tensor(down.outputs[0]).dtype)
    writer.line(
        3,
        f"down_acc_init = pl.tensor.create([{rows}, {cols}], "
        f"dtype={output_dtype}, layout=pl.TensorLayout.ND)",
    )
    writer.line(
        3,
        "for feature, (down_acc,) in pl.pipeline("
        f"0, {dense.intermediate_extent}, {dense.intermediate_chunk}, "
        f"stage={plan.pipeline_stages}, init_values=(down_acc_init,)):",
    )
    local = {
        gate.outputs[0]: _emit_matmul_tile(
            writer,
            4,
            context,
            gate_op,
            dense.gate_window_k,
            rows=rows,
            cols=dense.intermediate_chunk,
            row_offset=row,
            col_offset="feature",
            local={},
            prefix="gate",
        ),
        up.outputs[0]: _emit_matmul_tile(
            writer,
            4,
            context,
            up_op,
            dense.up_window_k,
            rows=rows,
            cols=dense.intermediate_chunk,
            row_offset=row,
            col_offset="feature",
            local={},
            prefix="up",
        ),
    }
    activation = _emit_vector_stage(
        writer,
        4,
        context,
        vector_stage,
        local,
        frame_rows=rows,
        frame_cols=dense.intermediate_chunk,
        row_offset=row,
        col_offset="feature",
        allow_external=False,
    )
    if local.get(down.inputs[0]) != activation:
        raise SourceEmissionError(
            "dense SwiGLU activation does not feed the down projection"
        )
    down_rhs = _emit_matrix_operand(
        writer,
        4,
        context,
        down.inputs[1],
        role="rhs",
        rows=dense.intermediate_chunk,
        cols=cols,
        row_offset="feature",
        col_offset=col,
        prefix="down_rhs",
        transposed=False,
        local=local,
    )
    writer.line(4, "if feature == 0:")
    writer.line(
        5,
        f"down_first = pl.tensor.matmul({activation}, {down_rhs}, "
        f"a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype={output_dtype})",
    )
    writer.line(5, "down_next = pl.yield_(down_first)")
    writer.line(4, "else:")
    writer.line(
        5,
        f"down_later = pl.tensor.matmul_acc(down_acc, {activation}, {down_rhs}, "
        "a_trans=False, b_trans=False)",
    )
    writer.line(5, "down_next = pl.yield_(down_later)")
    writer.line(4, "down_tile = pl.yield_(down_next)")
    output = context.interface.output_argument
    writer.line(
        3,
        f"{output} = pl.tensor.assemble({output}, down_tile, [{row}, {col}])",
    )
    emit_return(writer, context.interface)
    return writer.render()


def _require_materialized_vector_stage(stage: MixedStagePlan) -> None:
    if (
        stage.engine is not MixedEngine.VECTOR
        or stage.vector_stream is None
        or stage.vector_stream.kind is not VectorStreamKind.MATERIALIZED
    ):
        raise SourceEmissionError("mixed vector stage is not materialized")


def _emit_spatial_coordinates(
    writer: SourceWriter,
    indent: int,
    plan: MixedKernelPlan,
    trip: str | None,
) -> tuple[str, str]:
    item = "region_index"
    if trip is not None and (plan.m_partition.parts > 1 or plan.n_partition.parts > 1):
        writer.line(
            indent,
            f"mixed_item = region_index * {plan.max_trips_per_group} + {trip}",
        )
        item = "mixed_item"
    if plan.m_partition.parts > 1:
        writer.line(indent, f"mixed_m = {item} // {plan.n_partition.parts}")
        row = f"mixed_m * {plan.m_partition.big}"
    else:
        row = "0"
    if plan.n_partition.parts > 1:
        writer.line(indent, f"mixed_n = {item} % {plan.n_partition.parts}")
        col = f"mixed_n * {plan.n_partition.big}"
    else:
        col = "0"
    return row, col


def _emit_matmul_tile(  # noqa: PLR0913
    writer: SourceWriter,
    indent: int,
    context: EmissionContext,
    solver_op: int,
    k_window: int,
    *,
    rows: int,
    cols: int,
    row_offset: str,
    col_offset: str,
    local: Mapping[int, str],
    prefix: str,
) -> str:
    operation = context.lowered.operation(solver_op)
    graph_op = context.graph.op_map()[operation.graph_op_id]
    if (
        graph_op.kind != "matmul"
        or len(operation.inputs) != 2
        or len(operation.outputs) != 1
    ):
        raise SourceEmissionError(f"mixed cube operation {solver_op} is not a matmul")
    lhs_tensor, rhs_tensor = operation.inputs
    contraction = context.lowered.tensor(lhs_tensor).width
    if contraction != context.lowered.tensor(rhs_tensor).height or k_window <= 0:
        raise SourceEmissionError(f"mixed matmul {solver_op} has invalid K geometry")
    lhs_transposed = graph_op.attributes.get("lhs_transposed") is True
    rhs_transposed = graph_op.attributes.get("rhs_transposed") is True
    output_dtype = pypto_dtype(context.lowered.tensor(operation.outputs[0]).dtype)

    first_extent = min(k_window, contraction)
    lhs = _emit_matrix_operand(
        writer,
        indent,
        context,
        lhs_tensor,
        role="lhs",
        rows=rows,
        cols=first_extent,
        row_offset=row_offset,
        col_offset="0",
        prefix=f"{prefix}_lhs_first",
        transposed=lhs_transposed,
        local=local,
    )
    rhs = _emit_matrix_operand(
        writer,
        indent,
        context,
        rhs_tensor,
        role="rhs",
        rows=first_extent,
        cols=cols,
        row_offset="0",
        col_offset=col_offset,
        prefix=f"{prefix}_rhs_first",
        transposed=rhs_transposed,
        local=local,
    )
    writer.line(
        indent,
        f"{prefix}_acc_first = pl.tensor.matmul({lhs}, {rhs}, "
        f"a_trans={lhs_transposed}, b_trans={rhs_transposed}, "
        f"c_matrix_nz=False, out_dtype={output_dtype})",
    )
    if k_window >= contraction:
        return f"{prefix}_acc_first"
    full_chunks, tail = divmod(contraction, k_window)
    accumulator = f"{prefix}_acc_first"
    if full_chunks > 1:
        loop = "pl.pipeline" if full_chunks >= 3 else "pl.range"
        stage = ", stage=2" if loop == "pl.pipeline" else ""
        writer.line(
            indent,
            f"for {prefix}_k, ({prefix}_acc_iter,) in {loop}(1, {full_chunks}{stage}, "
            f"init_values=({accumulator},)):",
        )
        lhs = _emit_matrix_operand(
            writer,
            indent + 1,
            context,
            lhs_tensor,
            role="lhs",
            rows=rows,
            cols=k_window,
            row_offset=row_offset,
            col_offset=f"{prefix}_k * {k_window}",
            prefix=f"{prefix}_lhs",
            transposed=lhs_transposed,
            local=local,
        )
        rhs = _emit_matrix_operand(
            writer,
            indent + 1,
            context,
            rhs_tensor,
            role="rhs",
            rows=k_window,
            cols=cols,
            row_offset=f"{prefix}_k * {k_window}",
            col_offset=col_offset,
            prefix=f"{prefix}_rhs",
            transposed=rhs_transposed,
            local=local,
        )
        writer.line(
            indent + 1,
            f"{prefix}_acc_next = pl.tensor.matmul_acc({prefix}_acc_iter, {lhs}, {rhs}, "
            f"a_trans={lhs_transposed}, b_trans={rhs_transposed})",
        )
        writer.line(indent + 1, f"{prefix}_acc = pl.yield_({prefix}_acc_next)")
        accumulator = f"{prefix}_acc"
    if tail:
        k_offset = full_chunks * k_window
        lhs = _emit_matrix_operand(
            writer,
            indent,
            context,
            lhs_tensor,
            role="lhs",
            rows=rows,
            cols=tail,
            row_offset=row_offset,
            col_offset=str(k_offset),
            prefix=f"{prefix}_lhs_tail",
            transposed=lhs_transposed,
            local=local,
        )
        rhs = _emit_matrix_operand(
            writer,
            indent,
            context,
            rhs_tensor,
            role="rhs",
            rows=tail,
            cols=cols,
            row_offset=str(k_offset),
            col_offset=col_offset,
            prefix=f"{prefix}_rhs_tail",
            transposed=rhs_transposed,
            local=local,
        )
        writer.line(
            indent,
            f"{prefix}_acc_tail = pl.tensor.matmul_acc({accumulator}, {lhs}, {rhs}, "
            f"a_trans={lhs_transposed}, b_trans={rhs_transposed})",
        )
        accumulator = f"{prefix}_acc_tail"
    return accumulator


def _emit_matrix_operand(  # noqa: PLR0913
    writer: SourceWriter,
    indent: int,
    context: EmissionContext,
    tensor: int,
    *,
    role: str,
    rows: int,
    cols: int,
    row_offset: str,
    col_offset: str,
    prefix: str,
    transposed: bool,
    local: Mapping[int, str],
) -> str:
    if tensor in local:
        if transposed:
            raise SourceEmissionError(
                "mixed local matmul operands cannot be transposed"
            )
        return local[tensor]
    argument = _argument_for_tensor(context, tensor)
    slice_rows, slice_cols = rows, cols
    slice_row, slice_col = row_offset, col_offset
    if transposed:
        slice_rows, slice_cols = cols, rows
        slice_row, slice_col = col_offset, row_offset
    name = f"{prefix}_tile"
    writer.line(
        indent,
        f"{name} = pl.tensor.slice({argument}, [{slice_rows}, {slice_cols}], "
        f"[{slice_row}, {slice_col}])",
    )
    if role not in {"lhs", "rhs"}:
        raise SourceEmissionError(f"unknown matmul operand role {role!r}")
    return name


def _emit_vector_stage(  # noqa: PLR0913
    writer: SourceWriter,
    indent: int,
    context: EmissionContext,
    stage: MixedStagePlan,
    local: dict[int, str],
    *,
    frame_rows: int,
    frame_cols: int,
    row_offset: str,
    col_offset: str,
    allow_external: bool,
) -> str:
    graph_ops = context.graph.op_map()
    for solver_op in stage.ops:
        operation = context.lowered.operation(solver_op)
        if len(operation.outputs) != 1:
            raise SourceEmissionError(
                f"mixed vector op {solver_op} must have one output"
            )
        operands: list[str] = []
        for tensor in operation.inputs:
            if tensor not in local:
                if not allow_external:
                    raise SourceEmissionError(
                        f"dense mixed vector op {solver_op} has an external tensor operand"
                    )
                local[tensor] = _emit_vector_slice(
                    writer,
                    indent,
                    context,
                    tensor,
                    frame_rows=frame_rows,
                    frame_cols=frame_cols,
                    row_offset=row_offset,
                    col_offset=col_offset,
                )
            operands.append(local[tensor])
        output = operation.outputs[0]
        graph_op = graph_ops[operation.graph_op_id]
        expression = _tensor_vector_expression(
            graph_op,
            operands,
            input_shapes=[
                (
                    context.lowered.tensor(tensor).height,
                    context.lowered.tensor(tensor).width,
                )
                for tensor in operation.inputs
            ],
            output_dtype=context.lowered.tensor(output).dtype,
        )
        name = f"vector_{output}"
        writer.line(indent, f"{name} = {expression}")
        local[output] = name
    if not stage.ops:
        raise SourceEmissionError("mixed vector stage is empty")
    output = context.lowered.operation(stage.ops[-1]).outputs[0]
    return local[output]


def _emit_vector_slice(  # noqa: PLR0913
    writer: SourceWriter,
    indent: int,
    context: EmissionContext,
    tensor: int,
    *,
    frame_rows: int,
    frame_cols: int,
    row_offset: str,
    col_offset: str,
) -> str:
    descriptor = context.lowered.tensor(tensor)
    if descriptor.height not in {1, frame_rows} and descriptor.height < frame_rows:
        raise SourceEmissionError(
            f"mixed external tensor {tensor} cannot cover the propagated row frame"
        )
    if descriptor.width not in {1, frame_cols} and descriptor.width < frame_cols:
        raise SourceEmissionError(
            f"mixed external tensor {tensor} cannot cover the propagated column frame"
        )
    rows = 1 if descriptor.height == 1 else frame_rows
    cols = 1 if descriptor.width == 1 else frame_cols
    row = "0" if descriptor.height == 1 else row_offset
    col = "0" if descriptor.width == 1 else col_offset
    name = f"vector_input_{tensor}"
    writer.line(
        indent,
        f"{name} = pl.tensor.slice({_argument_for_tensor(context, tensor)}, "
        f"[{rows}, {cols}], [{row}, {col}])",
    )
    return name


def _tensor_vector_expression(
    op: NormalizedOp,
    operands: list[str],
    *,
    input_shapes: list[tuple[int, int]],
    output_dtype: str,
) -> str:
    unary = {"exp", "log", "abs", "sqrt", "rsqrt", "neg"}
    if op.kind in unary and len(operands) == 1:
        return f"pl.tensor.{op.kind}({operands[0]})"
    if op.kind == "cast" and len(operands) == 1:
        return (
            f"pl.tensor.cast({operands[0]}, target_type={pypto_dtype(output_dtype)}, "
            'mode="round")'
        )
    if op.kind in {"sum", "max"} and len(operands) == 1:
        if op.attributes.get("axis") != -1 or op.attributes.get("keepdim") is not True:
            raise SourceEmissionError(
                f"mixed vector reduction {op.id} is not last-axis keepdim"
            )
        reduction = "row_sum" if op.kind == "sum" else "row_max"
        return f"pl.tensor.{reduction}({operands[0]})"
    binary = {
        "add": "add",
        "sub": "sub",
        "mul": "mul",
        "div": "div",
        "maximum": "maximum",
        "minimum": "minimum",
    }
    if op.kind not in binary:
        raise SourceEmissionError(f"mixed vector source does not implement {op.kind!r}")
    if len(operands) == 2:
        if input_shapes[0] == input_shapes[1]:
            return f"pl.tensor.{binary[op.kind]}({operands[0]}, {operands[1]})"
        wide_index, thin_index, geometry = broadcast_operands(input_shapes)
        if op.kind in {"sub", "div"} and wide_index != 0:
            raise SourceEmissionError(
                f"mixed vector op {op.id} uses unsupported reverse broadcast {op.kind}"
            )
        operation = {
            "add": "add",
            "sub": "sub",
            "mul": "mul",
            "div": "div",
            "maximum": "max",
            "minimum": "min",
        }[op.kind]
        return (
            f"pl.tensor.{geometry}_expand_{operation}"
            f"({operands[wide_index]}, {operands[thin_index]})"
        )
    if len(operands) != 1:
        raise SourceEmissionError(f"mixed vector op {op.id} has unsupported arity")
    position, scalar = scalar_operand(op)
    if op.kind == "div" and position == 0 and scalar == 1:
        return f"pl.tensor.recip({operands[0]})"
    if position != 1:
        raise SourceEmissionError(
            f"mixed vector op {op.id} has an unsupported scalar position"
        )
    scalar_op = {"add": "adds", "sub": "subs", "mul": "muls", "div": "divs"}.get(
        op.kind
    )
    if scalar_op is None:
        raise SourceEmissionError(f"mixed vector op {op.id} has no scalar form")
    return f"pl.tensor.{scalar_op}({operands[0]}, {literal(scalar)})"


def _argument_for_tensor(context: EmissionContext, tensor: int) -> str:
    descriptor = context.lowered.tensor(tensor)
    value_id = (
        descriptor.alias_of if descriptor.alias_of is not None else descriptor.value_id
    )
    try:
        return context.interface.input_arguments[value_id]
    except KeyError as error:
        raise SourceEmissionError(
            f"mixed external tensor {tensor} ({value_id}) is not a region input"
        ) from error
