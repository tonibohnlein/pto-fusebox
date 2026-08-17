"""Mechanical replay of homogeneous vector schedules as PyPTO DSL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..ir import NormalizedGraph, NormalizedOp
from ..lowered import LoweredRegion
from ..schedule import VectorKernelPlan
from ..schedule.schema import (
    VectorCoordinateTransform,
    VectorPhysicalFramePlan,
    VectorReductionSplitKind,
    VectorReplayPhase,
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
    """Emit the installed materialized/pointwise vector schedule subset."""

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
    if step.sequential_tiles is None or any(step.sequential_tiles):
        raise SourceEmissionError(
            "materialized/pointwise vector source requires zero sequential tiles"
        )
    if plan.kind not in {
        VectorStreamKind.MATERIALIZED,
        VectorStreamKind.POINTWISE,
    }:
        raise SourceEmissionError(
            "vector source v1 supports materialized or pointwise replay, got "
            f"{plan.kind.value!r}"
        )
    if plan.coordinate_transform is not VectorCoordinateTransform.NONE:
        raise SourceEmissionError(
            "vector coordinate transforms are not implemented in source v1"
        )
    if (
        plan.reduction_split.kind is not VectorReductionSplitKind.NONE
        or plan.reduction_split.factor != 1
    ):
        raise SourceEmissionError(
            "vector reduction split emission is not implemented in source v1"
        )
    if plan.axis != 0:
        raise SourceEmissionError(
            "streamed vector phase emission is not implemented in source v1"
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
    emit_partition_indices(writer, indent, m_partition, n_partition, emit_extents=True)
    iteration_rows = (
        str(frame.iteration_rows) if frame.reduced_axis == 2 else "region_rows"
    )
    iteration_cols = (
        str(frame.iteration_cols) if frame.reduced_axis == 1 else "region_cols"
    )
    writer.line(
        indent,
        f"with pl.at(level=pl.Level.CORE_GROUP, name_hint={literal(context.region_id + '_vector')}):",
    )
    indent += 1

    if trips > 1:
        loop = "pl.pipeline" if stages > 1 else "pl.range"
        stage = f", stage={stages}" if stages > 1 else ""
        writer.line(indent, f"for strip_index in {loop}({trips}{stage}):")
        indent += 1
        row_index = (
            "strip_index" if strip_grid[1] == 1 else f"strip_index // {strip_grid[1]}"
        )
        col_index = "0" if strip_grid[1] == 1 else f"strip_index % {strip_grid[1]}"
        writer.line(indent, f"strip_row = {row_index} * {strip[0]}")
        writer.line(indent, f"strip_col = {col_index} * {strip[1]}")
    else:
        writer.line(indent, "strip_row = 0")
        writer.line(indent, "strip_col = 0")
    writer.line(
        indent,
        f"valid_rows = pl.max(pl.min({iteration_rows} - strip_row, {strip[0]}), 0)",
    )
    writer.line(
        indent,
        f"valid_cols = pl.max(pl.min({iteration_cols} - strip_col, {strip[1]}), 0)",
    )

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
        plan=plan,
    )
    writer.line(2, f"return {io.output_argument}")
    return writer.render()


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
        row_offset = "0" if rows == 1 else "region_row + strip_row"
        col_offset = "0" if cols == 1 else "region_col + strip_col"
        valid_rows = "1" if rows == 1 else "valid_rows"
        valid_cols = "1" if cols == 1 else "valid_cols"
        name = f"tensor_{tensor}"
        writer.line(
            indent,
            f"{name} = pl.load({argument}, [{row_offset}, {col_offset}], "
            f"[{physical_rows}, {physical_cols}], valid_shape=[{valid_rows}, {valid_cols}], "
            "target_memory=pl.Mem.Vec, clamp=True)",
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
        f"[region_row + strip_row, region_col + strip_col], {io.output_argument})",
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
        scalar = _single_scalar(op)
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


def _single_scalar(op: NormalizedOp) -> int | float:
    scalars = op.attributes.get("scalars")
    if not isinstance(scalars, Sequence) or len(scalars) != 1:
        raise SourceEmissionError(f"{op.id} does not carry exactly one scalar operand")
    scalar = scalars[0]
    if not isinstance(scalar, Mapping) or scalar.get("position") != 1:
        raise SourceEmissionError(
            f"{op.id} uses an unsupported scalar operand position"
        )
    value = scalar.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SourceEmissionError(f"{op.id} has a non-numeric scalar")
    return value


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
