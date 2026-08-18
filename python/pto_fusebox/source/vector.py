"""Mechanical replay of homogeneous vector schedules as PyPTO DSL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..ir import NormalizedGraph, NormalizedOp
from ..lowered import LoweredRegion
from ..schedule import VectorKernelPlan
from ..schedule.schema import (
    VectorCoordinateTransform,
    VectorPhasePlan,
    VectorPhysicalFramePlan,
    VectorReductionSplitKind,
    VectorReplayPhase,
    VectorSpatialPolicy,
    VectorStreamKind,
    VectorWorkspaceFramePlan,
)
from .common import (
    EmissionContext,
    Interface,
    SourceEmissionError,
    SourceWriter,
    ceil_div,
    emit_partition_indices,
    literal,
    program_header,
    pypto_dtype,
    solver_tensor_for_value,
    static_shape,
    validate_grid,
    validate_partition_extent,
)


def emit_vector(
    context: EmissionContext,
    program_name: str,
) -> str:
    """Emit one installed homogeneous vector schedule."""

    graph = context.graph
    lowered = context.lowered
    step = context.step
    io = context.interface
    plan = step.plan
    if not isinstance(plan, VectorKernelPlan):
        raise SourceEmissionError("vector step does not carry a vector plan")
    if step.retained_tensors:
        raise SourceEmissionError(
            "single-step vector source cannot carry inter-kernel retained tensors"
        )
    if plan.coordinate_transform is not VectorCoordinateTransform.NONE:
        raise SourceEmissionError(
            "vector coordinate transforms are not implemented in source v1"
        )
    if plan.spatial_policy is not VectorSpatialPolicy.CLAMPED_OVERLAP:
        raise SourceEmissionError(
            "vector source requires the solver-priced clamped-overlap spatial policy"
        )
    _validate_cast_roots(graph, lowered)
    _validate_cast_semantics(graph, lowered)
    if (
        plan.reduction_split.kind is not VectorReductionSplitKind.NONE
        or plan.reduction_split.factor != 1
    ):
        raise SourceEmissionError(
            "vector reduction split emission is not implemented in source v1"
        )
    if plan.kind is VectorStreamKind.SOFTMAX_FLASH:
        return _emit_softmax_flash(context, program_name, plan)
    if plan.kind not in {
        VectorStreamKind.MATERIALIZED,
        VectorStreamKind.POINTWISE,
    }:
        raise SourceEmissionError(
            "vector source v1 supports materialized or pointwise replay, got "
            f"{plan.kind.value!r}"
        )
    if step.sequential_tiles is None or any(step.sequential_tiles):
        raise SourceEmissionError(
            "materialized/pointwise vector source requires zero sequential tiles"
        )
    if plan.kind is VectorStreamKind.MATERIALIZED and plan.axis != 0:
        raise SourceEmissionError(
            "materialized vector replay cannot carry a stream axis"
        )

    m_partition = plan.m_partition
    n_partition = plan.n_partition
    validate_grid(step, plan.work_units, m_partition, n_partition)
    output_rows, output_cols = static_shape(
        graph.value_map()[io.output_value], field="vector output"
    )
    validate_partition_extent(m_partition, output_rows, "vector.m_partition")
    validate_partition_extent(n_partition, output_cols, "vector.n_partition")
    tile = list(plan.tile)
    strip = list(plan.strip)
    strip_grid = list(plan.strip_grid)
    frame = plan.physical_frame
    body_phase = plan.phase(VectorReplayPhase.BODY)
    body = body_phase.loop
    if body is None:
        raise SourceEmissionError("vector body phase omits its loop descriptor")
    trips = body.trip_count
    stages = body.pipeline_stages
    if body.first_chunk != 0 or trips != strip_grid[0] * strip_grid[1]:
        raise SourceEmissionError("vector body loop does not match its strip grid")
    if strip_grid != [
        ceil_div(tile[0], strip[0]),
        ceil_div(tile[1], strip[1]),
    ]:
        raise SourceEmissionError("vector strip grid does not cover its region tile")
    if stages not in {1, 2} or (trips == 1 and stages != 1):
        raise SourceEmissionError("vector body pipeline stage count is unsupported")
    expected_tile = [
        frame.iteration_rows if frame.reduced_axis == 2 else m_partition.big,
        frame.iteration_cols if frame.reduced_axis == 1 else n_partition.big,
    ]
    if tile != expected_tile:
        raise SourceEmissionError(
            "vector tile does not match the solver-owned iteration frame"
        )

    expected_rows = max(tensor.height for tensor in lowered.tensors)
    expected_cols = max(tensor.width for tensor in lowered.tensors)
    if (frame.iteration_rows, frame.iteration_cols) != (
        expected_rows,
        expected_cols,
    ):
        raise SourceEmissionError(
            "vector physical frame does not match the lowered iteration geometry"
        )
    _validate_vector_body_lifetimes(plan, body_phase.ops, lowered)
    writer = program_header(
        program_name, io, graph, m_partition.parts * n_partition.parts
    )
    indent = 4
    coordinates = emit_partition_indices(
        writer,
        indent,
        m_partition,
        n_partition,
        clamped_overlap_extents=(output_rows, output_cols),
    )
    writer.line(
        indent,
        f"with pl.at(level=pl.Level.CORE_GROUP, name_hint={literal(context.region_id + '_vector')}):",
    )
    indent += 1

    strip_row = "0"
    strip_col = "0"
    if trips > 1:
        loop = "pl.pipeline" if stages > 1 else "pl.range"
        stage = f", stage={stages}" if stages > 1 else ""
        writer.line(indent, f"for strip_index in {loop}({trips}{stage}):")
        indent += 1
        row_index = (
            "strip_index" if strip_grid[1] == 1 else f"strip_index // {strip_grid[1]}"
        )
        col_index = "0" if strip_grid[1] == 1 else f"strip_index % {strip_grid[1]}"
        row_offset = f"{row_index} * {strip[0]}"
        col_offset = f"{col_index} * {strip[1]}"
        if strip_grid[0] * strip[0] > tile[0]:
            row_offset = f"pl.min({row_offset}, {tile[0] - strip[0]})"
        if strip_grid[1] * strip[1] > tile[1]:
            col_offset = f"pl.min({col_offset}, {tile[1] - strip[1]})"
        if strip_grid[0] > 1:
            writer.line(indent, f"strip_row = {row_offset}")
            strip_row = "strip_row"
        if strip_grid[1] > 1:
            writer.line(indent, f"strip_col = {col_offset}")
            strip_col = "strip_col"
    _emit_vector_body(
        writer,
        indent,
        graph,
        lowered,
        body_phase.ops,
        io,
        frame,
        strip_rows=strip[0],
        strip_cols=strip[1],
        row_offset=_sum_offsets(coordinates.row, strip_row),
        col_offset=_sum_offsets(coordinates.col, strip_col),
        plan=plan,
    )
    writer.line(2, f"return {io.output_argument}")
    return writer.render()


def _emit_softmax_flash(
    context: EmissionContext,
    program_name: str,
    plan: VectorKernelPlan,
) -> str:
    """Replay the versioned online-softmax schedule carried by ``plan``."""

    graph = context.graph
    lowered = context.lowered
    step = context.step
    io = context.interface
    recipe = plan.p4_recipe
    if recipe is None or recipe.version != "softmax_flash.v1":
        raise SourceEmissionError("softmax_flash source requires recipe v1")
    if recipe.state != ("running_max", "running_sum"):
        raise SourceEmissionError("softmax_flash recipe has an unknown state contract")
    if plan.axis != 1 or plan.physical_frame.reduced_axis != 1:
        raise SourceEmissionError(
            "softmax_flash source currently supports last-axis replay only"
        )
    if plan.stream_passes != 2 or plan.chunk <= 0 or plan.full_chunks <= 0:
        raise SourceEmissionError("softmax_flash stream geometry is incomplete")
    if plan.full_chunks * plan.chunk + plan.tail != plan.extent:
        raise SourceEmissionError(
            "softmax_flash chunks do not cover the reduced extent"
        )
    if step.sequential_tiles is None or tuple(step.sequential_tiles) != (
        plan.chunk,
    ) * len(step.solver_ops):
        raise SourceEmissionError(
            "softmax_flash sequential tiles differ from the streamed chunk"
        )

    m_partition = plan.m_partition
    n_partition = plan.n_partition
    validate_grid(step, plan.work_units, m_partition, n_partition)
    output_rows, output_cols = static_shape(
        graph.value_map()[io.output_value], field="softmax_flash output"
    )
    validate_partition_extent(m_partition, output_rows, "vector.m_partition")
    validate_partition_extent(n_partition, output_cols, "vector.n_partition")
    if n_partition.parts != 1 or plan.extent != output_cols:
        raise SourceEmissionError(
            "softmax_flash requires one complete reduced-axis partition"
        )
    if plan.free_tile != m_partition.big or plan.tile != (
        m_partition.big,
        n_partition.big,
    ):
        raise SourceEmissionError("softmax_flash tile differs from its partitions")
    if (
        plan.physical_frame.iteration_rows,
        plan.physical_frame.iteration_cols,
    ) != (output_rows, output_cols):
        raise SourceEmissionError(
            "softmax_flash frame differs from the lowered iteration geometry"
        )

    stats = plan.phase(VectorReplayPhase.STATS)
    apply = plan.phase(VectorReplayPhase.APPLY)
    _validate_softmax_loops(plan, stats, apply)
    _validate_softmax_generated_work(plan)
    stats_init = stats.init
    if stats_init is None:
        raise SourceEmissionError("softmax_flash stats phase omits initialization")
    max_op, sum_op = _softmax_semantic_ops(context, plan)
    input_tensor = recipe.input_tensor
    _validate_softmax_frames(plan, lowered, stats, apply)
    input_value = lowered.tensor(input_tensor).value_id
    try:
        input_argument = io.input_arguments[input_value]
    except KeyError as error:
        raise SourceEmissionError(
            "softmax_flash input is not a direct region input"
        ) from error
    max_tensor = lowered.operation(max_op).outputs[0]
    sum_tensor = lowered.operation(sum_op).outputs[0]
    output_tensor = solver_tensor_for_value(lowered, io.output_allocation_owner)

    writer = program_header(program_name, io, graph, plan.work_units)
    indent = 4
    coordinates = emit_partition_indices(
        writer,
        indent,
        m_partition,
        n_partition,
        clamped_overlap_extents=(output_rows, output_cols),
    )
    writer.line(
        indent,
        f"with pl.at(level=pl.Level.CORE_GROUP, name_hint={literal(context.region_id + '_vector')}):",
    )
    indent += 1

    running_max, running_sum = _emit_softmax_stats_chunk(
        writer,
        indent,
        lowered,
        stats,
        input_tensor=input_tensor,
        input_argument=input_argument,
        prefix="initial",
        row_offset=coordinates.row,
        col_offset="0",
        valid_rows=str(plan.free_tile),
        valid_cols=str(stats_init.extent),
        old_max=None,
        old_sum=None,
    )
    if stats.loop is None:
        raise SourceEmissionError("softmax_flash stats phase omits its loop")
    if stats.loop.trip_count:
        stats_iter_max = "stats_running_max"
        stats_iter_sum = "stats_running_sum"
        stats_result_max = "stats_result_max"
        stats_result_sum = "stats_result_sum"
        _emit_loop_header(
            writer,
            indent,
            "stats_chunk",
            stats.loop,
            iter_values=(stats_iter_max, stats_iter_sum),
            init_values=(running_max, running_sum),
        )
        loop_indent = indent + 1
        writer.line(loop_indent, f"stats_col = stats_chunk * {plan.chunk}")
        next_max, next_sum = _emit_softmax_stats_chunk(
            writer,
            loop_indent,
            lowered,
            stats,
            input_tensor=input_tensor,
            input_argument=input_argument,
            prefix="stats",
            row_offset=coordinates.row,
            col_offset="stats_col",
            valid_rows=str(plan.free_tile),
            valid_cols=str(plan.chunk),
            old_max=stats_iter_max,
            old_sum=stats_iter_sum,
        )
        writer.line(
            loop_indent,
            f"{stats_result_max}, {stats_result_sum} = "
            f"pl.yield_({next_max}, {next_sum})",
        )
        running_max = stats_result_max
        running_sum = stats_result_sum
    if plan.tail:
        if stats.tail is None or not stats.tail.present:
            raise SourceEmissionError("softmax_flash stats phase omits its tail")
        tail_max, tail_sum = _emit_softmax_stats_chunk(
            writer,
            indent,
            lowered,
            stats,
            input_tensor=input_tensor,
            input_argument=input_argument,
            prefix="stats_tail",
            row_offset=coordinates.row,
            col_offset=str(stats.tail.chunk_index * plan.chunk),
            valid_rows=str(plan.free_tile),
            valid_cols=str(stats.tail.extent),
            old_max=running_max,
            old_sum=running_sum,
        )
        writer.line(indent, f"{running_max} = {tail_max}")
        writer.line(indent, f"{running_sum} = {tail_sum}")

    if apply.loop is None:
        raise SourceEmissionError("softmax_flash apply phase omits its loop")
    if apply.loop.trip_count:
        _emit_loop_header(writer, indent, "apply_chunk", apply.loop)
        loop_indent = indent + 1
        writer.line(loop_indent, f"apply_col = apply_chunk * {plan.chunk}")
        _emit_softmax_apply_chunk(
            writer,
            loop_indent,
            context,
            apply,
            input_tensor=input_tensor,
            input_argument=input_argument,
            max_tensor=max_tensor,
            sum_tensor=sum_tensor,
            output_tensor=output_tensor,
            running_max=running_max,
            running_sum=running_sum,
            prefix="apply",
            row_offset=coordinates.row,
            col_offset="apply_col",
            valid_rows=str(plan.free_tile),
            valid_cols=str(plan.chunk),
        )
    if plan.tail:
        if apply.tail is None or not apply.tail.present:
            raise SourceEmissionError("softmax_flash apply phase omits its tail")
        _emit_softmax_apply_chunk(
            writer,
            indent,
            context,
            apply,
            input_tensor=input_tensor,
            input_argument=input_argument,
            max_tensor=max_tensor,
            sum_tensor=sum_tensor,
            output_tensor=output_tensor,
            running_max=running_max,
            running_sum=running_sum,
            prefix="apply_tail",
            row_offset=coordinates.row,
            col_offset=str(apply.tail.chunk_index * plan.chunk),
            valid_rows=str(plan.free_tile),
            valid_cols=str(apply.tail.extent),
        )
    writer.line(2, f"return {io.output_argument}")
    return writer.render()


def _validate_softmax_loops(
    plan: VectorKernelPlan,
    stats: VectorPhasePlan,
    apply: VectorPhasePlan,
) -> None:
    if stats.init is None or not stats.init.present:
        raise SourceEmissionError("softmax_flash stats phase omits initialization")
    if stats.init.chunk_index != 0 or stats.init.extent != plan.chunk:
        raise SourceEmissionError("softmax_flash initialization differs from its chunk")
    if stats.loop is None or (
        stats.loop.first_chunk,
        stats.loop.trip_count,
    ) != (1, plan.full_chunks - 1):
        raise SourceEmissionError("softmax_flash stats loop is inconsistent")
    if apply.loop is None or (
        apply.loop.first_chunk,
        apply.loop.trip_count,
    ) != (0, plan.full_chunks):
        raise SourceEmissionError("softmax_flash apply loop is inconsistent")
    for name, loop in (("stats", stats.loop), ("apply", apply.loop)):
        expected_stages = 2 if loop.trip_count >= 2 else 1
        if loop.pipeline_stages != expected_stages:
            raise SourceEmissionError(
                f"softmax_flash {name} pipeline depth is inconsistent"
            )
    for name, phase in (("stats", stats), ("apply", apply)):
        tail = phase.tail
        if plan.tail:
            if (
                tail is None
                or not tail.present
                or tail.chunk_index != plan.full_chunks
                or tail.extent != plan.tail
            ):
                raise SourceEmissionError(f"softmax_flash {name} tail is inconsistent")
        elif tail is not None and tail.present:
            raise SourceEmissionError(f"softmax_flash {name} has a spurious tail")


def _validate_softmax_generated_work(plan: VectorKernelPlan) -> None:
    def work(phase) -> dict[str, tuple[int, int, int]]:
        result = {
            primitive.kind: (
                primitive.wide,
                primitive.thin,
                primitive.stream_starts,
            )
            for primitive in phase.primitives
        }
        if len(result) != len(phase.primitives):
            raise SourceEmissionError(
                "softmax_flash generated work contains duplicate primitives"
            )
        return result

    expected_init = {
        "exp": (1, 0, 0),
        "row_expand_sub": (1, 0, 1),
        "row_sum": (1, 0, 0),
        "row_max": (1, 0, 0),
    }
    expected_update = {
        "add": (0, 3, 2),
        "mul": (0, 1, 0),
        "exp": (1, 1, 0),
        "row_expand_sub": (1, 0, 1),
        "row_sum": (1, 0, 0),
        "row_max": (1, 0, 0),
    }
    if (
        not plan.p4_work.generated
        or work(plan.p4_work.stats_init) != expected_init
        or work(plan.p4_work.stats_update) != expected_update
        or plan.p4_work.finalize.generated
        or plan.p4_work.finalize.primitives
    ):
        raise SourceEmissionError("softmax_flash generated work differs from recipe v1")


def _validate_softmax_frames(
    plan: VectorKernelPlan,
    lowered: LoweredRegion,
    stats: VectorPhasePlan,
    apply: VectorPhasePlan,
) -> None:
    """Check the recipe's wide data and thin state against solver-owned frames."""

    wide_logical = (plan.free_tile, plan.chunk)
    wide_physical = (plan.free_tile_alloc, plan.chunk)
    thin_logical = (plan.free_tile, 1)
    thin_physical = (plan.free_tile_alloc, 1)
    for phase in (stats, apply):
        for frame in phase.tensor_frames:
            tensor = lowered.tensor(frame.tensor)
            if tensor.height != plan.physical_frame.iteration_rows:
                raise SourceEmissionError(
                    "softmax_flash tensor height differs from its iteration frame"
                )
            if tensor.width == plan.extent:
                expected_logical = wide_logical
                expected_physical = wide_physical
            elif tensor.width == 1:
                expected_logical = thin_logical
                expected_physical = thin_physical
            else:
                raise SourceEmissionError(
                    "softmax_flash tensor is neither wide data nor thin state"
                )
            if frame.logical != expected_logical or frame.physical != expected_physical:
                raise SourceEmissionError(
                    "softmax_flash tensor frame differs from its recipe role"
                )


def _softmax_semantic_ops(
    context: EmissionContext,
    plan: VectorKernelPlan,
) -> tuple[int, int]:
    recipe = plan.p4_recipe
    if recipe is None:
        raise SourceEmissionError("softmax_flash recipe is absent")
    bindings = {binding.value: binding.op for binding in recipe.apply_substitutions}
    if set(bindings) != {"running_max", "running_sum"}:
        raise SourceEmissionError("softmax_flash substitutions are incomplete")
    max_op = bindings["running_max"]
    sum_op = bindings["running_sum"]
    lowered = context.lowered
    graph_ops = context.graph.op_map()
    stats = plan.phase(VectorReplayPhase.STATS)
    apply = plan.phase(VectorReplayPhase.APPLY)
    if tuple(graph_ops[lowered.operation(op).graph_op_id].kind for op in stats.ops) != (
        "max",
        "sub",
        "exp",
        "sum",
    ):
        raise SourceEmissionError("softmax_flash stats semantics are unsupported")
    if tuple(graph_ops[lowered.operation(op).graph_op_id].kind for op in apply.ops) != (
        "sub",
        "exp",
        "div",
    ):
        raise SourceEmissionError("softmax_flash apply semantics are unsupported")
    max_desc = lowered.operation(max_op)
    sum_desc = lowered.operation(sum_op)
    sub_desc = lowered.operation(stats.ops[1])
    exp_desc = lowered.operation(stats.ops[2])
    div_desc = lowered.operation(apply.ops[2])
    if (
        max_desc.inputs != (recipe.input_tensor,)
        or sub_desc.inputs != (recipe.input_tensor, max_desc.outputs[0])
        or exp_desc.inputs != (sub_desc.outputs[0],)
        or sum_desc.inputs != (exp_desc.outputs[0],)
        or div_desc.inputs != (exp_desc.outputs[0], sum_desc.outputs[0])
    ):
        raise SourceEmissionError("softmax_flash tensor wiring is unsupported")
    return max_op, sum_op


def _emit_loop_header(
    writer: SourceWriter,
    indent: int,
    index: str,
    loop,
    *,
    iter_values: tuple[str, ...] = (),
    init_values: tuple[str, ...] = (),
) -> None:
    if bool(iter_values) != bool(init_values) or len(iter_values) != len(init_values):
        raise SourceEmissionError("loop-carried names and initial values disagree")
    function = "pl.pipeline" if loop.pipeline_stages > 1 else "pl.range"
    stage = f", stage={loop.pipeline_stages}" if loop.pipeline_stages > 1 else ""
    carried = ""
    init = ""
    if iter_values:
        carried = f", ({', '.join(iter_values)})"
        init = f", init_values=({', '.join(init_values)},)"
    stop = loop.first_chunk + loop.trip_count
    writer.line(
        indent,
        f"for {index}{carried} in {function}({loop.first_chunk}, {stop}{stage}{init}):",
    )


def _emit_softmax_stats_chunk(  # noqa: PLR0913 -- explicit typed contract fields.
    writer: SourceWriter,
    indent: int,
    lowered: LoweredRegion,
    phase: VectorPhasePlan,
    *,
    input_tensor: int,
    input_argument: str,
    prefix: str,
    row_offset: str,
    col_offset: str,
    valid_rows: str,
    valid_cols: str,
    old_max: str | None,
    old_sum: str | None,
) -> tuple[str, str]:
    frames = {frame.tensor: frame for frame in phase.tensor_frames}
    workspaces = {workspace.op: workspace for workspace in phase.workspaces}
    try:
        input_frame = frames[input_tensor]
    except KeyError as error:
        raise SourceEmissionError("softmax_flash stats input has no frame") from error
    max_op, sum_op = phase.ops[0], phase.ops[3]
    try:
        max_workspace = workspaces[max_op]
        sum_workspace = workspaces[sum_op]
    except KeyError as error:
        raise SourceEmissionError(
            "softmax_flash reduction workspace is absent"
        ) from error
    dtype = pypto_dtype(lowered.tensor(input_tensor).dtype)
    physical_rows, physical_cols = input_frame.physical
    tile = f"{prefix}_input"
    writer.line(
        indent,
        f"{tile} = "
        f"{_static_vector_load(input_argument, row_offset, col_offset, (physical_rows, physical_cols), (valid_rows, valid_cols))}",
    )
    max_scratch = f"{prefix}_max_scratch"
    writer.line(
        indent,
        f"{max_scratch} = pl.tile.create([{max_workspace.physical[0]}, "
        f"{max_workspace.physical[1]}], dtype={dtype}, target_memory=pl.Mem.Vec)",
    )
    local_max = f"{prefix}_local_max"
    writer.line(indent, f"{local_max} = pl.row_max({tile}, {max_scratch})")
    new_max = local_max
    if old_max is not None:
        new_max = f"{prefix}_next_max"
        writer.line(indent, f"{new_max} = pl.maximum({old_max}, {local_max})")
    shifted = f"{prefix}_shifted"
    exponent = f"{prefix}_exponent"
    writer.line(indent, f"{shifted} = pl.row_expand_sub({tile}, {new_max})")
    writer.line(indent, f"{exponent} = pl.exp({shifted})")
    sum_scratch = f"{prefix}_sum_scratch"
    writer.line(
        indent,
        f"{sum_scratch} = pl.tile.create([{sum_workspace.physical[0]}, "
        f"{sum_workspace.physical[1]}], dtype={dtype}, target_memory=pl.Mem.Vec)",
    )
    local_sum = f"{prefix}_local_sum"
    writer.line(indent, f"{local_sum} = pl.row_sum({exponent}, {sum_scratch})")
    new_sum = local_sum
    if old_sum is not None:
        if old_max is None:
            raise SourceEmissionError("softmax_flash update omits its old maximum")
        delta = f"{prefix}_delta"
        correction = f"{prefix}_correction"
        scaled_sum = f"{prefix}_scaled_sum"
        new_sum = f"{prefix}_next_sum"
        writer.line(indent, f"{delta} = pl.sub({old_max}, {new_max})")
        writer.line(indent, f"{correction} = pl.exp({delta})")
        writer.line(indent, f"{scaled_sum} = pl.mul({old_sum}, {correction})")
        writer.line(indent, f"{new_sum} = pl.add({scaled_sum}, {local_sum})")
    return new_max, new_sum


def _emit_softmax_apply_chunk(  # noqa: PLR0913 -- explicit typed contract fields.
    writer: SourceWriter,
    indent: int,
    context: EmissionContext,
    phase: VectorPhasePlan,
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
    valid_rows: str,
    valid_cols: str,
) -> None:
    frames = {frame.tensor: frame for frame in phase.tensor_frames}
    try:
        input_frame = frames[input_tensor]
    except KeyError as error:
        raise SourceEmissionError("softmax_flash apply input has no frame") from error
    physical_rows, physical_cols = input_frame.physical
    tile = f"{prefix}_input"
    writer.line(
        indent,
        f"{tile} = "
        f"{_static_vector_load(input_argument, row_offset, col_offset, (physical_rows, physical_cols), (valid_rows, valid_cols))}",
    )
    local = {
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
                "softmax_flash apply phase uses an unavailable tensor"
            ) from error
        graph_op = graph_ops[operation.graph_op_id]
        output = operation.outputs[0]
        name = f"{prefix}_tensor_{output}"
        expression = _vector_expression(
            writer,
            indent,
            graph_op,
            operands,
            list(operation.inputs),
            output,
            context.lowered,
            physical_rows,
            physical_cols,
            solver_op,
            None,
        )
        writer.line(indent, f"{name} = {expression}")
        local[output] = name
    try:
        result = local[output_tensor]
    except KeyError as error:
        raise SourceEmissionError(
            "softmax_flash apply phase does not produce the region output"
        ) from error
    writer.line(
        indent,
        f"{context.interface.output_argument} = pl.store({result}, "
        f"[{row_offset}, {col_offset}], {context.interface.output_argument})",
    )


def _emit_vector_body(
    writer: SourceWriter,
    indent: int,
    graph: NormalizedGraph,
    lowered: LoweredRegion,
    op_order: tuple[int, ...],
    io: Interface,
    frame: VectorPhysicalFramePlan,
    *,
    strip_rows: int,
    strip_cols: int,
    row_offset: str,
    col_offset: str,
    plan: VectorKernelPlan,
) -> None:
    graph_ops = graph.op_map()
    producers = _tensor_producers(lowered)
    phase = plan.phase(VectorReplayPhase.BODY)
    frames = {item.tensor: item for item in phase.tensor_frames}
    workspaces = {item.op: item for item in phase.workspaces}
    local: dict[int, str] = {}

    def tensor_shape(tensor: int) -> tuple[int, int]:
        descriptor = lowered.tensor(tensor)
        return descriptor.height, descriptor.width

    def ensure_loaded(tensor: int) -> str:
        if tensor in local:
            return local[tensor]
        if producers[tensor] is not None:
            raise SourceEmissionError(
                f"solver tensor {tensor} is used before its producer"
            )
        value_id = lowered.tensor(tensor).value_id
        argument = io.input_arguments.get(value_id)
        if argument is None:
            raise SourceEmissionError(
                f"external solver tensor {tensor} ({value_id}) is not a direct region input"
            )
        rows, cols = tensor_shape(tensor)
        logical_rows = 1 if rows == 1 else min(strip_rows, rows)
        logical_cols = 1 if cols == 1 else min(strip_cols, cols)
        try:
            tensor_frame = frames[tensor]
        except KeyError as error:
            raise SourceEmissionError(
                f"vector body omits the physical frame for tensor {tensor}"
            ) from error
        if tensor_frame.logical != (logical_rows, logical_cols):
            raise SourceEmissionError(
                f"vector body frame for tensor {tensor} differs from its logical slice"
            )
        physical_rows, physical_cols = tensor_frame.physical
        input_row_offset = "0" if rows == 1 else row_offset
        input_col_offset = "0" if cols == 1 else col_offset
        valid_rows = "1" if rows == 1 else str(logical_rows)
        valid_cols = "1" if cols == 1 else str(logical_cols)
        name = f"tensor_{tensor}"
        writer.line(
            indent,
            f"{name} = "
            f"{_static_vector_load(argument, input_row_offset, input_col_offset, (physical_rows, physical_cols), (valid_rows, valid_cols))}",
        )
        local[tensor] = name
        return name

    for solver_op in op_order:
        lowered_op = lowered.operation(solver_op)
        op_inputs = list(lowered_op.inputs)
        op_outputs = list(lowered_op.outputs)
        if len(op_outputs) != 1:
            raise SourceEmissionError(f"solver op {solver_op} must have one output")
        operands = [ensure_loaded(tensor) for tensor in op_inputs]
        output = op_outputs[0]
        graph_op_id = lowered_op.graph_op_id
        try:
            graph_op = graph_ops[graph_op_id]
        except KeyError as error:
            raise SourceEmissionError(
                f"solver op {solver_op} maps to unknown graph op"
            ) from error
        expression = _vector_expression(
            writer,
            indent,
            graph_op,
            operands,
            op_inputs,
            output,
            lowered,
            strip_rows,
            strip_cols,
            solver_op,
            workspaces.get(solver_op),
        )
        local[output] = f"tensor_{output}"
        writer.line(indent, f"{local[output]} = {expression}")

    output_tensor = solver_tensor_for_value(lowered, io.output_allocation_owner)
    try:
        output_tile = local[output_tensor]
    except KeyError as error:
        raise SourceEmissionError(
            "region output is not produced by the selected step"
        ) from error
    writer.line(
        indent,
        f"{io.output_argument} = pl.store({output_tile}, "
        f"[{row_offset}, {col_offset}], {io.output_argument})",
    )


def _sum_offsets(*offsets: str) -> str:
    """Combine coordinate expressions without materializing zero scalars."""

    nonzero = [offset for offset in offsets if offset != "0"]
    return " + ".join(nonzero) if nonzero else "0"


def _static_vector_load(
    argument: str,
    row_offset: str,
    col_offset: str,
    physical: tuple[int, int],
    valid: tuple[str, str],
) -> str:
    """Render a plan-bounded load without re-symbolizing its valid frame.

    The spatial policy already keeps the logical valid frame in bounds.
    Requesting ``clamp=True`` would make PyPTO derive runtime valid extents and
    would therefore violate the solver's static replay contract.
    """

    physical_rows, physical_cols = physical
    valid_rows, valid_cols = valid
    return (
        f"pl.load({argument}, [{row_offset}, {col_offset}], "
        f"[{physical_rows}, {physical_cols}], valid_shape=[{valid_rows}, {valid_cols}], "
        "target_memory=pl.Mem.Vec)"
    )


def _vector_expression(  # noqa: PLR0913 -- arguments are explicit contract fields.
    writer: SourceWriter,
    indent: int,
    op: NormalizedOp,
    operands: list[str],
    input_tensors: list[int],
    output_tensor: int,
    lowered: LoweredRegion,
    strip_rows: int,
    strip_cols: int,
    solver_op: int,
    workspace: VectorWorkspaceFramePlan | None,
) -> str:
    unary = {
        "exp": "exp",
        "log": "log",
        "abs": "abs",
        "sqrt": "sqrt",
        "rsqrt": "rsqrt",
        "neg": "neg",
    }
    if op.kind in unary and len(operands) == 1:
        return f"pl.{unary[op.kind]}({operands[0]})"
    if op.kind == "cast" and len(operands) == 1:
        dtype = pypto_dtype(lowered.tensor(output_tensor).dtype)
        return f"pl.cast({operands[0]}, target_type={dtype})"
    if op.kind in {"sum", "max"} and len(operands) == 1:
        if op.attributes.get("axis") != -1 or op.attributes.get("keepdim") is not True:
            raise SourceEmissionError(
                f"vector source v1 supports only last-axis keepdim reductions, got {op.id}"
            )
        source_tensor = lowered.tensor(input_tensors[0])
        rows = source_tensor.height
        cols = source_tensor.width
        logical_rows = 1 if rows == 1 else min(strip_rows, rows)
        logical_cols = 1 if cols == 1 else min(strip_cols, cols)
        if workspace is None or workspace.source_tensor != input_tensors[0]:
            raise SourceEmissionError(
                f"vector reduction {solver_op} omits its workspace frame"
            )
        if workspace.logical != (logical_rows, logical_cols):
            raise SourceEmissionError(
                f"vector reduction {solver_op} workspace has stale logical shape"
            )
        physical_rows, physical_cols = workspace.physical
        scratch = f"scratch_{solver_op}"
        writer.line(
            indent,
            f"{scratch} = pl.tile.create([{physical_rows}, {physical_cols}], "
            f"dtype={pypto_dtype(source_tensor.dtype)}, target_memory=pl.Mem.Vec)",
        )
        reduction = "row_sum" if op.kind == "sum" else "row_max"
        return f"pl.{reduction}({operands[0]}, {scratch})"

    if op.kind not in {"add", "sub", "mul", "div", "maximum", "minimum"}:
        raise SourceEmissionError(f"vector source v1 does not implement {op.kind!r}")
    if len(operands) == 1:
        position, scalar = _single_scalar(op)
        if op.kind == "div" and position == 0 and scalar == 1:
            return f"pl.recip({operands[0]})"
        if position != 1:
            raise SourceEmissionError(
                f"{op.id} uses an unsupported scalar operand position"
            )
        return f"pl.{op.kind}({operands[0]}, {literal(scalar)})"
    if len(operands) != 2:
        raise SourceEmissionError(
            f"{op.kind} requires one tensor plus a scalar or two tensors"
        )

    shapes = [
        (lowered.tensor(tensor).height, lowered.tensor(tensor).width)
        for tensor in input_tensors
    ]
    if shapes[0] == shapes[1]:
        return f"pl.{op.kind}({operands[0]}, {operands[1]})"
    wide_index, thin_index, geometry = _broadcast_operands(shapes)
    if op.kind in {"sub", "div"} and wide_index != 0:
        raise SourceEmissionError(f"reverse broadcast {op.kind} is not implemented")
    operation = {
        "add": "add",
        "sub": "sub",
        "mul": "mul",
        "div": "div",
        "maximum": "max",
        "minimum": "min",
    }[op.kind]
    return f"pl.{geometry}_expand_{operation}({operands[wide_index]}, {operands[thin_index]})"


def _broadcast_operands(shapes: list[tuple[int, int]]) -> tuple[int, int, str]:
    for wide in (0, 1):
        thin = 1 - wide
        wide_rows, wide_cols = shapes[wide]
        thin_rows, thin_cols = shapes[thin]
        if thin_rows == wide_rows and thin_cols == 1 and wide_cols > 1:
            return wide, thin, "row"
        if thin_rows == 1 and thin_cols == wide_cols and wide_rows > 1:
            return wide, thin, "col"
    raise SourceEmissionError(f"unsupported vector broadcast geometry {shapes}")


def _single_scalar(op: NormalizedOp) -> tuple[int, int | float]:
    scalars = op.attributes.get("scalars")
    if not isinstance(scalars, Sequence) or len(scalars) != 1:
        raise SourceEmissionError(f"{op.id} does not carry exactly one scalar operand")
    scalar = scalars[0]
    if not isinstance(scalar, Mapping):
        raise SourceEmissionError(f"{op.id} has a malformed scalar operand")
    position = scalar.get("position")
    if not isinstance(position, int) or isinstance(position, bool):
        raise SourceEmissionError(f"{op.id} has an invalid scalar operand position")
    value = scalar.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SourceEmissionError(f"{op.id} has a non-numeric scalar")
    return position, value


def _tensor_producers(lowered: LoweredRegion) -> list[int | None]:
    producers: list[int | None] = [None] * len(lowered.tensors)
    for operation in lowered.operations:
        for tensor in operation.outputs:
            if producers[tensor] is not None:
                raise SourceEmissionError(
                    f"solver tensor {tensor} has multiple producers"
                )
            producers[tensor] = operation.index
    return producers


def _validate_cast_roots(graph: NormalizedGraph, lowered: LoweredRegion) -> None:
    """Reject cast chains whose source physical box cannot yet be widened."""

    graph_ops = graph.op_map()
    producers = _tensor_producers(lowered)
    for operation in lowered.operations:
        graph_op = graph_ops.get(operation.graph_op_id)
        if graph_op is None or graph_op.kind != "cast" or len(operation.inputs) != 1:
            continue
        producer = producers[operation.inputs[0]]
        if producer is None:
            continue
        source_op = graph_ops.get(lowered.operation(producer).graph_op_id)
        if source_op is not None and source_op.kind in {"sum", "max"}:
            raise SourceEmissionError(
                "vector source does not support a native cast chain rooted in a "
                "reduction result"
            )


def _validate_cast_semantics(graph: NormalizedGraph, lowered: LoweredRegion) -> None:
    """Reject captured casts whose Torch semantics cannot be replayed exactly."""

    graph_ops = graph.op_map()
    values = graph.value_map()
    selected_graph_ops = {operation.graph_op_id for operation in lowered.operations}
    for graph_op_id in selected_graph_ops:
        graph_op = graph_ops.get(graph_op_id)
        if graph_op is None or graph_op.kind != "cast" or len(graph_op.outputs) != 1:
            continue
        if values[graph_op.outputs[0]].dtype == "int8":
            raise SourceEmissionError(
                "vector source cannot preserve Torch float-to-INT8 truncation "
                "through the Ascend910B native FP16 conversion path"
            )


def _validate_vector_body_lifetimes(
    plan: VectorKernelPlan,
    op_order: tuple[int, ...],
    lowered: LoweredRegion,
) -> None:
    tensor_count = len(lowered.tensors)
    producers = _tensor_producers(lowered)
    expected: dict[int, list[tuple[int, int]]] = {}
    position = {solver_op: index for index, solver_op in enumerate(op_order)}
    for solver_op in op_order:
        for argument, tensor in enumerate(lowered.operation(solver_op).inputs):
            if producers[tensor] is None:
                expected.setdefault(tensor, []).append((solver_op, argument))

    body = plan.phase(VectorReplayPhase.BODY)
    actual: dict[int, list[tuple[int, int]]] = {}
    for lifetime in body.input_lifetimes:
        tensor = lifetime.tensor
        if not 0 <= tensor < tensor_count:
            raise SourceEmissionError(
                f"vector input lifetime contains out-of-range tensor {tensor}"
            )
        if tensor in actual:
            raise SourceEmissionError(
                f"vector body has duplicate lifetime descriptors for tensor {tensor}"
            )
        uses = [(use.op, use.arg) for use in lifetime.uses]
        if any(op not in position for op, _ in uses):
            raise SourceEmissionError(
                "vector input lifetime references an unselected op"
            )
        if lifetime.use_count != len(uses):
            raise SourceEmissionError("vector input lifetime use count is inconsistent")
        first = min(position[op] for op, _ in uses)
        last = max(position[op] for op, _ in uses)
        if lifetime.first_use_step != first or lifetime.last_use_step != last:
            raise SourceEmissionError("vector input lifetime bounds are inconsistent")
        actual[tensor] = sorted(uses)

    normalized_expected = {tensor: sorted(uses) for tensor, uses in expected.items()}
    if actual != normalized_expected:
        raise SourceEmissionError(
            "vector body input lifetimes do not match boundary-tensor uses"
        )
