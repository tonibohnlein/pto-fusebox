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
    VectorReplayPhase,
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
from .vector import (
    _softmax_semantic_ops,
    _validate_softmax_frames,
    _validate_softmax_generated_work,
    _validate_softmax_loops,
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
    if plan.protocol is MixedCrossCoreProtocol.MULTI_ROUND_TRIP_SEQUENTIAL:
        return _emit_multi_round_trip_sequential(context, program_name, plan)
    if plan.protocol is MixedCrossCoreProtocol.ONE_WAY:
        return _emit_one_way(context, program_name, plan)
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
    vec_capacity = context.problem.get("vec_capacity")
    if (
        not isinstance(vec_capacity, int)
        or isinstance(vec_capacity, bool)
        or vec_capacity <= 0
    ):
        raise SourceEmissionError("mixed source requires a positive Vec capacity")
    c2v_fifo_bytes = sum(
        fifo.reserved_bytes
        for fifo in plan.fifos
        if fifo.direction is MixedTransferDirection.CUBE_TO_VECTOR
    )
    required_vec_bytes = c2v_fifo_bytes + plan.vector_stage_peak_ub_bytes
    if required_vec_bytes > vec_capacity:
        raise SourceEmissionError(
            "mixed C2V FIFO rings and vector stage exceed Vec capacity: "
            f"{required_vec_bytes} > {vec_capacity} bytes"
        )
    l1_capacity = context.problem.get("l1_capacity")
    if (
        not isinstance(l1_capacity, int)
        or isinstance(l1_capacity, bool)
        or l1_capacity <= 0
    ):
        raise SourceEmissionError("mixed source requires a positive L1 capacity")
    v2c_fifo_bytes = sum(
        fifo.reserved_bytes
        for fifo in plan.fifos
        if fifo.direction is MixedTransferDirection.VECTOR_TO_CUBE
    )
    required_l1_bytes = plan.cube_stage_peak_l1_bytes + v2c_fifo_bytes
    if required_l1_bytes > l1_capacity:
        raise SourceEmissionError(
            "mixed cube stage and V2C FIFO rings exceed L1 capacity: "
            f"{required_l1_bytes} > {l1_capacity} bytes"
        )


def _mixed_header(
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
) -> SourceWriter:
    slot_counts = {fifo.slot_count for fifo in plan.fifos}
    if len(slot_counts) != 1:
        raise SourceEmissionError(
            "PyPTO supports one cross-core slot count per mixed scope, "
            f"but the plan requires {sorted(slot_counts)}"
        )
    slot_count = next(iter(slot_counts))
    optimizations = [
        "pl.split(pl.SplitMode.UP_DOWN)",
        f"pl.cross_core_slot(slot_num={slot_count})",
    ]
    writer = program_preamble(program_name, context.interface, context.graph)
    writer.line(
        2,
        f"for region_index in pl.spmd({plan.active_groups}, "
        f"name_hint={literal(context.region_id + '_mixed')}, "
        f"optimizations=[{', '.join(optimizations)}]):",
    )
    return writer


def _emit_one_way(
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
) -> str:
    engines = tuple(stage.engine for stage in plan.stages)
    if engines == (MixedEngine.CUBE, MixedEngine.VECTOR):
        return _emit_one_way_c2v(context, program_name, plan)
    if engines == (MixedEngine.VECTOR, MixedEngine.CUBE):
        return _emit_one_way_v2c(context, program_name, plan)
    raise SourceEmissionError(
        "mixed plan is not a supported directional one-way topology"
    )


def _emit_one_way_c2v(
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
) -> str:
    if (
        plan.algorithm is not MixedAlgorithm.GENERIC
        or plan.mode is not MixedPipelineMode.ONE_WAY
        or plan.pipeline_axis is not MixedPipelineAxis.SPATIAL_REGION
        or plan.pipeline_stages not in {1, 2}
        or (plan.pipeline_stages == 2) is not plan.overlap_implementable
        or plan.requested_skew_depth != plan.pipeline_stages - 1
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
    _require_in_memory_vector_stage(vector_stage)
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
    trip_loop = "pl.pipeline" if plan.pipeline_stages == 2 else "pl.range"
    trip_stage = ", stage=2" if plan.pipeline_stages == 2 else ""
    writer.line(
        3,
        f"for mixed_trip, (output_iter,) in {trip_loop}({trips}{trip_stage}, "
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


def _emit_one_way_v2c(
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
) -> str:
    if (
        plan.algorithm is not MixedAlgorithm.GENERIC
        or plan.mode is not MixedPipelineMode.ONE_WAY
        or plan.pipeline_axis is not MixedPipelineAxis.SPATIAL_REGION
        or plan.pipeline_stages not in {1, 2}
        or (plan.pipeline_stages == 2) is not plan.overlap_implementable
        or plan.requested_skew_depth != plan.pipeline_stages - 1
        or len(plan.stages) != 2
        or tuple(stage.engine for stage in plan.stages)
        != (MixedEngine.VECTOR, MixedEngine.CUBE)
        or len(plan.stages[1].ops) != 1
        or len(plan.transfers) != 1
        or len(plan.fifos) != 1
        or plan.fifos[0].direction is not MixedTransferDirection.VECTOR_TO_CUBE
        or plan.fifos[0].slot_count != 8
    ):
        raise SourceEmissionError(
            "mixed plan is not the supported one-way V->C topology"
        )
    vector_stage = plan.stages[0]
    cube_stage = plan.stages[1]
    stream = vector_stage.vector_stream
    if stream is None:
        raise SourceEmissionError("V->C vector stage omits its stream plan")
    crossing = plan.transfers[0].tensor
    sink_op = cube_stage.ops[0]
    sink = context.lowered.operation(sink_op)
    if context.lowered.operation(vector_stage.ops[-1]).outputs != (
        crossing,
    ) or sink.inputs.count(crossing) not in {1, 2}:
        raise SourceEmissionError("V->C transfer is not a sink matmul operand")
    output_tensor = solver_tensor_for_value(
        context.lowered, context.interface.output_allocation_owner
    )
    if sink.outputs != (output_tensor,):
        raise SourceEmissionError("V->C cube stage does not produce the region output")
    fifo = plan.fifos[0]
    if stream.kind is VectorStreamKind.SOFTMAX_FLASH:
        return _emit_streaming_softmax_v2c(
            context, program_name, plan, vector_stage, cube_stage, crossing
        )
    _require_in_memory_vector_stage(vector_stage)
    crossing_roles = sink.inputs.count(crossing)
    if crossing_roles == 1 and fifo.spatial_m == fifo.spatial_n:
        raise SourceEmissionError("V->C single-role FIFO has ambiguous spatial axes")
    if crossing_roles == 2 and (
        not fifo.spatial_m
        or not fifo.spatial_n
        or plan.m_partition.parts != 1
        or plan.n_partition.parts != 1
    ):
        raise SourceEmissionError(
            "V->C dual-role FIFO requires one complete square spatial region"
        )
    if (
        vector_stage.valid_rows * plan.vector_lanes != fifo.valid_rows
        or vector_stage.valid_cols != fifo.valid_cols
    ):
        raise SourceEmissionError("V->C vector stage and FIFO frames disagree")

    writer = _mixed_header(context, program_name, plan)
    output = context.interface.output_argument
    trip_loop = "pl.pipeline" if plan.pipeline_stages == 2 else "pl.range"
    trip_stage = ", stage=2" if plan.pipeline_stages == 2 else ""
    writer.line(
        3,
        f"for mixed_trip, (output_iter,) in {trip_loop}"
        f"({plan.max_trips_per_group}{trip_stage}, "
        f"init_values=({output},)):",
    )
    row, col = _emit_spatial_coordinates(writer, 4, plan, "mixed_trip")
    vector_row = row if fifo.spatial_m else "0"
    vector_col = col if fifo.spatial_n else "0"
    vector_local: dict[int, str] = {}
    crossing_value = _emit_vector_stage(
        writer,
        4,
        context,
        vector_stage,
        vector_local,
        frame_rows=fifo.valid_rows,
        frame_cols=fifo.valid_cols,
        row_offset=vector_row,
        col_offset=vector_col,
        allow_external=True,
    )
    # Only the declared crossing is local to the cube stage. External values
    # loaded while replaying the vector DAG remain GM operands unless the typed
    # plan gives them their own cross-core FIFO.
    cube_local = {crossing: crossing_value}
    result = _emit_matmul_tile(
        writer,
        4,
        context,
        sink_op,
        cube_stage.cube_window_k[0],
        rows=plan.m_partition.big,
        cols=plan.n_partition.big,
        row_offset=row,
        col_offset=col,
        local=cube_local,
        prefix="sink",
    )
    writer.line(
        4,
        f"next_output = pl.tensor.assemble(output_iter, {result}, [{row}, {col}])",
    )
    writer.line(4, f"{output} = pl.yield_(next_output)")
    emit_return(writer, context.interface)
    return writer.render()


def _emit_streaming_softmax_v2c(  # noqa: PLR0915 -- typed phase replay.
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
    vector_stage: MixedStagePlan,
    cube_stage: MixedStagePlan,
    crossing: int,
) -> str:
    """Publish online-softmax apply chunks directly into one sink matmul.

    The homogeneous vector plan remains authoritative for both passes. The
    mixed contract only replaces the APPLY store with one V->C publication per
    chunk and accumulates those disjoint K contributions on AIC.
    """

    stream = vector_stage.vector_stream
    if stream is None or stream.kind is not VectorStreamKind.SOFTMAX_FLASH:
        raise SourceEmissionError("streaming V->C requires a softmax_flash plan")
    recipe = stream.p4_recipe
    if recipe is None or recipe.version != "softmax_flash.v1":
        raise SourceEmissionError("streaming V->C requires softmax_flash recipe v1")
    if recipe.state != ("running_max", "running_sum"):
        raise SourceEmissionError("streaming V->C has an unknown softmax state")
    if stream.axis != 1 or stream.stream_passes != 2:
        raise SourceEmissionError("streaming V->C requires a two-pass last-axis stream")
    if stream.full_chunks * stream.chunk + stream.tail != stream.extent:
        raise SourceEmissionError("streaming V->C chunks do not cover the contraction")
    _validate_softmax_loops(
        stream,
        stream.phase(VectorReplayPhase.STATS),
        stream.phase(VectorReplayPhase.APPLY),
    )
    _validate_softmax_generated_work(stream)
    _validate_softmax_frames(
        stream,
        context.lowered,
        stream.phase(VectorReplayPhase.STATS),
        stream.phase(VectorReplayPhase.APPLY),
    )
    max_op, sum_op = _softmax_semantic_ops(context, stream)
    sink_op = cube_stage.ops[0]
    sink = context.lowered.operation(sink_op)
    fifo = plan.fifos[0]
    if (
        sink.inputs[0] != crossing
        or sink.inputs.count(crossing) != 1
        or len(cube_stage.cube_window_k) != 1
        or cube_stage.cube_window_k[0] != stream.chunk
        or fifo.valid_rows != vector_stage.valid_rows * plan.vector_lanes
        or fifo.valid_cols != stream.chunk
        or fifo.spatial_m is not True
        or fifo.spatial_n is not False
    ):
        raise SourceEmissionError(
            "streaming V->C softmax and sink K-publication contracts disagree"
        )
    input_tensor = recipe.input_tensor
    input_argument = _argument_for_tensor(context, input_tensor)
    max_tensor = context.lowered.operation(max_op).outputs[0]
    sum_tensor = context.lowered.operation(sum_op).outputs[0]
    stats = stream.phase(VectorReplayPhase.STATS)
    apply = stream.phase(VectorReplayPhase.APPLY)
    if stats.init is None or stats.loop is None or apply.loop is None:
        raise SourceEmissionError("streaming V->C softmax phase loops are incomplete")

    writer = _mixed_header(context, program_name, plan)
    output = context.interface.output_argument
    writer.line(
        3,
        f"for mixed_trip, (output_iter,) in pl.range({plan.max_trips_per_group}, "
        f"init_values=({output},)):",
    )
    row, col = _emit_spatial_coordinates(writer, 4, plan, "mixed_trip")
    running_max, running_sum = _emit_tensor_softmax_stats_chunk(
        writer,
        4,
        input_argument=input_argument,
        prefix="initial",
        rows=fifo.valid_rows,
        cols=stream.chunk,
        row_offset=row,
        col_offset="0",
        valid_cols=stream.chunk,
        old_max=None,
        old_sum=None,
    )
    if stats.loop.trip_count:
        stats_loop = "pl.pipeline" if stats.loop.pipeline_stages > 1 else "pl.range"
        stats_stage = (
            f", stage={stats.loop.pipeline_stages}"
            if stats.loop.pipeline_stages > 1
            else ""
        )
        stop = stats.loop.first_chunk + stats.loop.trip_count
        writer.line(
            4,
            "for stats_chunk, (stats_max, stats_sum) in "
            f"{stats_loop}({stats.loop.first_chunk}, {stop}{stats_stage}, "
            f"init_values=({running_max}, {running_sum},)):",
        )
        writer.line(5, f"stats_col = stats_chunk * {stream.chunk}")
        next_max, next_sum = _emit_tensor_softmax_stats_chunk(
            writer,
            5,
            input_argument=input_argument,
            prefix="stats",
            rows=fifo.valid_rows,
            cols=stream.chunk,
            row_offset=row,
            col_offset="stats_col",
            valid_cols=stream.chunk,
            old_max="stats_max",
            old_sum="stats_sum",
        )
        writer.line(
            5,
            f"stats_result_max, stats_result_sum = pl.yield_({next_max}, {next_sum})",
        )
        running_max, running_sum = "stats_result_max", "stats_result_sum"
    if stream.tail:
        if stats.tail is None or not stats.tail.present:
            raise SourceEmissionError("streaming V->C softmax omits its stats tail")
        tail_max, tail_sum = _emit_tensor_softmax_stats_chunk(
            writer,
            4,
            input_argument=input_argument,
            prefix="stats_tail",
            rows=fifo.valid_rows,
            cols=stream.chunk,
            row_offset=row,
            col_offset=str(stream.full_chunks * stream.chunk),
            valid_cols=stream.tail,
            old_max=running_max,
            old_sum=running_sum,
        )
        writer.line(4, f"{running_max} = {tail_max}")
        writer.line(4, f"{running_sum} = {tail_sum}")

    output_dtype = pypto_dtype(context.lowered.tensor(sink.outputs[0]).dtype)
    writer.line(
        4,
        f"sink_acc_init = pl.tensor.create([{plan.m_partition.big}, "
        f"{plan.n_partition.big}], dtype={output_dtype}, layout=pl.TensorLayout.ND)",
    )
    apply_loop = "pl.pipeline" if apply.loop.pipeline_stages > 1 else "pl.range"
    apply_stage = (
        f", stage={apply.loop.pipeline_stages}"
        if apply.loop.pipeline_stages > 1
        else ""
    )
    apply_stop = apply.loop.first_chunk + apply.loop.trip_count
    writer.line(
        4,
        "for apply_chunk, (sink_acc,) in "
        f"{apply_loop}({apply.loop.first_chunk}, {apply_stop}{apply_stage}, "
        "init_values=(sink_acc_init,)):",
    )
    writer.line(5, f"apply_col = apply_chunk * {stream.chunk}")
    apply_valid_cols: str | int = stream.chunk
    probability = _emit_tensor_softmax_apply_chunk(
        writer,
        5,
        context,
        apply,
        input_tensor=input_tensor,
        input_argument=input_argument,
        max_tensor=max_tensor,
        sum_tensor=sum_tensor,
        output_tensor=crossing,
        running_max=running_max,
        running_sum=running_sum,
        prefix="apply",
        row_offset=row,
        col_offset="apply_col",
        rows=fifo.valid_rows,
        cols=stream.chunk,
        valid_cols=apply_valid_cols,
    )
    rhs = _emit_streaming_sink_rhs(
        writer,
        5,
        context,
        sink.inputs[1],
        rows=stream.chunk,
        valid_rows=apply_valid_cols,
        cols=plan.n_partition.big,
        row_offset="apply_col",
        col_offset=col,
        prefix="sink_rhs",
    )
    writer.line(5, "if apply_chunk == 0:")
    writer.line(
        6,
        f"sink_first = pl.tensor.matmul({probability}, {rhs}, a_trans=False, "
        f"b_trans=False, c_matrix_nz=False, out_dtype={output_dtype})",
    )
    writer.line(6, "sink_next = pl.yield_(sink_first)")
    writer.line(5, "else:")
    writer.line(
        6,
        f"sink_later = pl.tensor.matmul_acc(sink_acc, {probability}, {rhs}, "
        "a_trans=False, b_trans=False)",
    )
    writer.line(6, "sink_next = pl.yield_(sink_later)")
    writer.line(5, "sink_acc_next = pl.yield_(sink_next)")
    accumulator = "sink_acc_next"
    if stream.tail:
        if apply.tail is None or not apply.tail.present:
            raise SourceEmissionError("streaming V->C softmax omits its apply tail")
        tail_col = apply.tail.chunk_index * stream.chunk
        probability = _emit_tensor_softmax_apply_chunk(
            writer,
            4,
            context,
            apply,
            input_tensor=input_tensor,
            input_argument=input_argument,
            max_tensor=max_tensor,
            sum_tensor=sum_tensor,
            output_tensor=crossing,
            running_max=running_max,
            running_sum=running_sum,
            prefix="apply_tail",
            row_offset=row,
            col_offset=str(tail_col),
            rows=fifo.valid_rows,
            cols=stream.chunk,
            valid_cols=apply.tail.extent,
        )
        rhs = _emit_streaming_sink_rhs(
            writer,
            4,
            context,
            sink.inputs[1],
            rows=stream.chunk,
            valid_rows=apply.tail.extent,
            cols=plan.n_partition.big,
            row_offset=str(tail_col),
            col_offset=col,
            prefix="sink_rhs_tail",
        )
        writer.line(
            4,
            f"sink_tail = pl.tensor.matmul_acc({accumulator}, {probability}, {rhs}, "
            "a_trans=False, b_trans=False)",
        )
        accumulator = "sink_tail"
    writer.line(
        4,
        f"next_output = pl.tensor.assemble(output_iter, {accumulator}, [{row}, {col}])",
    )
    writer.line(4, f"{output} = pl.yield_(next_output)")
    emit_return(writer, context.interface)
    return writer.render()


def _emit_tensor_softmax_stats_chunk(  # noqa: PLR0913
    writer: SourceWriter,
    indent: int,
    *,
    input_argument: str,
    prefix: str,
    rows: int,
    cols: int,
    row_offset: str,
    col_offset: str,
    valid_cols: str | int,
    old_max: str | None,
    old_sum: str | None,
) -> tuple[str, str]:
    tile = f"{prefix}_input"
    writer.line(
        indent,
        f"{tile} = pl.tensor.slice({input_argument}, [{rows}, {cols}], "
        f"[{row_offset}, {col_offset}], valid_shape=[{rows}, {valid_cols}], "
        "pad_value=pl.PadValue.min)",
    )
    local_max = f"{prefix}_local_max"
    writer.line(indent, f"{local_max} = pl.tensor.row_max({tile})")
    next_max = local_max
    if old_max is not None:
        next_max = f"{prefix}_next_max"
        writer.line(indent, f"{next_max} = pl.tensor.maximum({old_max}, {local_max})")
    shifted = f"{prefix}_shifted"
    exponent = f"{prefix}_exponent"
    writer.line(indent, f"{shifted} = pl.tensor.row_expand_sub({tile}, {next_max})")
    writer.line(indent, f"{exponent} = pl.tensor.exp({shifted})")
    local_sum = f"{prefix}_local_sum"
    writer.line(indent, f"{local_sum} = pl.tensor.row_sum({exponent})")
    next_sum = local_sum
    if old_sum is not None:
        if old_max is None:
            raise SourceEmissionError("streaming softmax update omits old maximum")
        delta = f"{prefix}_delta"
        correction = f"{prefix}_correction"
        scaled_sum = f"{prefix}_scaled_sum"
        next_sum = f"{prefix}_next_sum"
        writer.line(indent, f"{delta} = pl.tensor.sub({old_max}, {next_max})")
        writer.line(indent, f"{correction} = pl.tensor.exp({delta})")
        writer.line(indent, f"{scaled_sum} = pl.tensor.mul({old_sum}, {correction})")
        writer.line(indent, f"{next_sum} = pl.tensor.add({scaled_sum}, {local_sum})")
    return next_max, next_sum


def _emit_tensor_softmax_apply_chunk(  # noqa: PLR0913
    writer: SourceWriter,
    indent: int,
    context: EmissionContext,
    phase,
    *,
    input_tensor: int,
    input_argument: str,
    max_tensor: int,
    sum_tensor: int,
    output_tensor: int,
    running_max: str,
    running_sum: str,
    prefix: str,
    row_offset: str,
    col_offset: str,
    rows: int,
    cols: int,
    valid_cols: str | int,
) -> str:
    if len(phase.ops) != 3:
        raise SourceEmissionError("streaming softmax apply phase is incomplete")
    tile = f"{prefix}_input"
    writer.line(
        indent,
        f"{tile} = pl.tensor.slice({input_argument}, [{rows}, {cols}], "
        f"[{row_offset}, {col_offset}], valid_shape=[{rows}, {valid_cols}], "
        "pad_value=pl.PadValue.min)",
    )
    local: dict[int, str] = {
        input_tensor: tile,
        max_tensor: running_max,
        sum_tensor: running_sum,
    }
    graph_ops = context.graph.op_map()
    for solver_op in phase.ops:
        operation = context.lowered.operation(solver_op)
        try:
            operands = [local[tensor] for tensor in operation.inputs]
        except KeyError as error:
            raise SourceEmissionError(
                "streaming softmax apply uses an unavailable tensor"
            ) from error
        output = operation.outputs[0]
        expression = _tensor_vector_expression(
            graph_ops[operation.graph_op_id],
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
        name = f"{prefix}_tensor_{output}"
        writer.line(indent, f"{name} = {expression}")
        local[output] = name
    try:
        return local[output_tensor]
    except KeyError as error:
        raise SourceEmissionError(
            "streaming softmax apply does not produce the FIFO tensor"
        ) from error


def _emit_streaming_sink_rhs(
    writer: SourceWriter,
    indent: int,
    context: EmissionContext,
    tensor: int,
    *,
    rows: int,
    valid_rows: str | int,
    cols: int,
    row_offset: str,
    col_offset: str,
    prefix: str,
) -> str:
    argument = _argument_for_tensor(context, tensor)
    writer.line(
        indent,
        f"{prefix} = pl.tensor.slice({argument}, [{rows}, {cols}], "
        f"[{row_offset}, {col_offset}], valid_shape=[{valid_rows}, {cols}], "
        "pad_value=pl.PadValue.zero)",
    )
    return prefix


def _emit_single_round_trip(
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
) -> str:
    if (
        plan.algorithm is not MixedAlgorithm.GENERIC
        or plan.mode is not MixedPipelineMode.SINGLE_ROUND_TRIP_SKEW
        or plan.pipeline_axis is not MixedPipelineAxis.SPATIAL_REGION
        or plan.pipeline_stages not in {1, 3}
        or (plan.pipeline_stages == 3) is not plan.overlap_implementable
        or plan.requested_skew_depth != (2 if plan.pipeline_stages == 3 else 0)
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
    trip_loop = "pl.pipeline" if plan.pipeline_stages == 3 else "pl.range"
    trip_stage = ", stage=3" if plan.pipeline_stages == 3 else ""
    writer.line(
        3,
        f"for mixed_trip, (output_iter,) in {trip_loop}({trips}{trip_stage}, "
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


def _emit_multi_round_trip_sequential(
    context: EmissionContext,
    program_name: str,
    plan: MixedKernelPlan,
) -> str:
    """Replay one linear C->V->C->V plan without claiming skew overlap."""

    if (
        plan.algorithm is not MixedAlgorithm.GENERIC
        or plan.mode is not MixedPipelineMode.MULTI_ROUND_TRIP_SEQUENTIAL
        or plan.pipeline_axis is not MixedPipelineAxis.SPATIAL_REGION
        or plan.pipeline_stages != 1
        or plan.requested_skew_depth != 0
        or plan.model_overlap_granted
        or plan.overlap_implementable
        or len(plan.stages) != 4
        or tuple(stage.engine for stage in plan.stages)
        != (
            MixedEngine.CUBE,
            MixedEngine.VECTOR,
            MixedEngine.CUBE,
            MixedEngine.VECTOR,
        )
        or any(len(plan.stages[index].ops) != 1 for index in (0, 2))
        or len(plan.transfers) != 3
        or any(
            transfer.producer_stage != index or transfer.consumer_stage != index + 1
            for index, transfer in enumerate(plan.transfers)
        )
        or tuple(fifo.direction for fifo in plan.fifos)
        != (
            MixedTransferDirection.CUBE_TO_VECTOR,
            MixedTransferDirection.VECTOR_TO_CUBE,
            MixedTransferDirection.CUBE_TO_VECTOR,
        )
        or any(fifo.slot_count != 4 for fifo in plan.fifos)
    ):
        raise SourceEmissionError(
            "mixed plan is not the supported sequential C->V->C->V topology"
        )
    first_vector = plan.stages[1]
    final_vector = plan.stages[3]
    _require_in_memory_vector_stage(first_vector)
    _require_in_memory_vector_stage(final_vector)
    first_cube = plan.stages[0].ops[0]
    second_cube = plan.stages[2].ops[0]
    first_crossing, reply_crossing, final_crossing = (
        transfer.tensor for transfer in plan.transfers
    )
    if (
        context.lowered.operation(first_cube).outputs != (first_crossing,)
        or reply_crossing not in context.lowered.operation(second_cube).inputs
        or context.lowered.operation(second_cube).outputs != (final_crossing,)
    ):
        raise SourceEmissionError(
            "sequential multi-round-trip transfers do not connect their stages"
        )
    output_tensor = solver_tensor_for_value(
        context.lowered, context.interface.output_allocation_owner
    )
    if context.lowered.operation(final_vector.ops[-1]).outputs != (output_tensor,):
        raise SourceEmissionError(
            "sequential multi-round-trip vector tail does not produce the output"
        )

    writer = _mixed_header(context, program_name, plan)
    output = context.interface.output_argument
    writer.line(
        3,
        f"for mixed_trip, (output_iter,) in pl.range({plan.max_trips_per_group}, "
        f"init_values=({output},)):",
    )
    row, col = _emit_spatial_coordinates(writer, 4, plan, "mixed_trip")
    first_rows = plan.fifos[0].valid_rows
    first_cols = plan.fifos[0].valid_cols
    first_row = row if plan.fifos[0].spatial_m else "0"
    first_col = col if plan.fifos[0].spatial_n else "0"
    local = {
        first_crossing: _emit_matmul_tile(
            writer,
            4,
            context,
            first_cube,
            plan.stages[0].cube_window_k[0],
            rows=first_rows,
            cols=first_cols,
            row_offset=first_row,
            col_offset=first_col,
            local={},
            prefix="first_cube",
        )
    }
    local[reply_crossing] = _emit_vector_stage(
        writer,
        4,
        context,
        first_vector,
        local,
        frame_rows=first_rows,
        frame_cols=first_cols,
        row_offset=first_row,
        col_offset=first_col,
        allow_external=True,
    )
    final_rows = plan.fifos[2].valid_rows
    final_cols = plan.fifos[2].valid_cols
    final_row = row if plan.fifos[2].spatial_m else "0"
    final_col = col if plan.fifos[2].spatial_n else "0"
    local[final_crossing] = _emit_matmul_tile(
        writer,
        4,
        context,
        second_cube,
        plan.stages[2].cube_window_k[0],
        rows=final_rows,
        cols=final_cols,
        row_offset=final_row,
        col_offset=final_col,
        local=local,
        prefix="second_cube",
    )
    result = _emit_vector_stage(
        writer,
        4,
        context,
        final_vector,
        local,
        frame_rows=final_rows,
        frame_cols=final_cols,
        row_offset=final_row,
        col_offset=final_col,
        allow_external=True,
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


def _require_in_memory_vector_stage(stage: MixedStagePlan) -> None:
    if (
        stage.engine is not MixedEngine.VECTOR
        or stage.vector_stream is None
        or stage.vector_stream.kind
        not in {VectorStreamKind.MATERIALIZED, VectorStreamKind.POINTWISE}
    ):
        raise SourceEmissionError(
            "mixed vector stage is not an in-memory materialized/pointwise replay"
        )


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
    if role not in {"lhs", "rhs"}:
        raise SourceEmissionError(f"unknown matmul operand role {role!r}")
    if tensor in local:
        if transposed:
            raise SourceEmissionError(
                "mixed local matmul operands cannot be transposed"
            )
        descriptor = context.lowered.tensor(tensor)
        complete_contraction = (
            role == "lhs" and cols == descriptor.width and col_offset == "0"
        ) or (role == "rhs" and rows == descriptor.height and row_offset == "0")
        if complete_contraction:
            return local[tensor]
        local_row = "0" if role == "lhs" else row_offset
        local_col = col_offset if role == "lhs" else "0"
        name = f"{prefix}_tile"
        writer.line(
            indent,
            f"{name} = pl.tensor.slice({local[tensor]}, [{rows}, {cols}], "
            f"[{local_row}, {local_col}])",
        )
        return name
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
