"""Fail-closed PyPTO DSL source emission from a solved Fusebox region.

The backend consumes solver-owned schedules.  It deliberately supports a
small, structurally complete subset first; unsupported schedules raise instead
of being approximated or sent through a second planner in PyPTO.
"""

from __future__ import annotations

import ast
import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .ir import NormalizedGraph, NormalizedOp, NormalizedValue, ShapeDimension
from .schedule import (
    AxisPartition,
    KernelKind,
    KernelStep,
    ScheduleContractError,
    scheduled_region,
)
from .solver import RegionSolveResult


class SourceEmissionError(ValueError):
    """Raised when a valid analytic schedule is outside the source backend."""


@dataclass(frozen=True)
class EmittedPyPTOSource:
    """One deterministic PyPTO program and the schedule step it implements."""

    program_name: str
    region_id: str
    kind: KernelKind
    source: str


@dataclass(frozen=True)
class _PhysicalFrameContract:
    element_granule: int
    iteration_rows: int
    iteration_cols: int
    reduced_axis: int
    align_rows: bool


class _SourceWriter:
    def __init__(self) -> None:
        self._lines: list[str] = []

    def line(self, indent: int = 0, text: str = "") -> None:
        self._lines.append(f"{'    ' * indent}{text}" if text else "")

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"


def emit_pypto_region(
    graph: NormalizedGraph,
    result: RegionSolveResult,
    *,
    program_name: str | None = None,
) -> EmittedPyPTOSource:
    """Emit one explicitly scheduled region as ordinary PyPTO DSL.

    V1 emits a single homogeneous solver step.  Materialized vector replay and
    a single, spatial, output-stationary matmul are supported.  Streamed vector
    phases, split-K, retained panels, multi-matmul and mixed schedules fail
    closed until their complete replay contracts have source implementations.
    """

    try:
        schedule = scheduled_region(result)
    except ScheduleContractError as error:
        raise SourceEmissionError(str(error)) from error
    if len(schedule.steps) != 1:
        raise SourceEmissionError(
            "source v1 requires exactly one selected kernel step; "
            f"the solver selected {len(schedule.steps)}"
        )
    if result.problem is None:
        raise SourceEmissionError("solved region has no lowered problem")

    step = schedule.steps[0]
    chosen_name = _class_name(program_name or f"fused_{result.region.id}")
    if step.kind is KernelKind.VECTOR:
        source = _emit_vector(graph, result, step, chosen_name)
    elif step.kind is KernelKind.CUBE:
        source = _emit_cube(graph, result, step, chosen_name)
    else:
        raise SourceEmissionError("mixed PyPTO source emission is not implemented yet")

    # This catches backend bugs before source leaves Fusebox.  It is not a
    # substitute for parsing/lowering with the target PyPTO checkout.
    ast.parse(source)
    if "auto_fuse" in source or "auto_tile" in source:
        raise AssertionError("generated source must encode the plan directly")
    return EmittedPyPTOSource(
        program_name=chosen_name,
        region_id=result.region.id,
        kind=step.kind,
        source=source,
    )


def _emit_vector(
    graph: NormalizedGraph,
    result: RegionSolveResult,
    step: KernelStep,
    program_name: str,
) -> str:
    problem = _problem(result)
    plan = step.plan
    if plan.get("kind") not in {"materialized", "pointwise"}:
        raise SourceEmissionError(
            "vector source v1 supports materialized or pointwise replay, got "
            f"{plan.get('kind')!r}"
        )
    if plan.get("coordinate_transform") != "none":
        raise SourceEmissionError(
            "vector coordinate transforms are not implemented in source v1"
        )
    split = _mapping(plan.get("reduction_split"), "vector.reduction_split")
    if split.get("kind") != "none" or split.get("factor") != 1:
        raise SourceEmissionError(
            "vector reduction split emission is not implemented in source v1"
        )
    if _positive_int(plan.get("axis"), "vector.axis", allow_zero=True) != 0:
        raise SourceEmissionError(
            "streamed vector phase emission is not implemented in source v1"
        )

    m_partition = AxisPartition.from_json(
        plan.get("m_partition"), field="vector.m_partition"
    )
    n_partition = AxisPartition.from_json(
        plan.get("n_partition"), field="vector.n_partition"
    )
    _validate_grid(step, plan, m_partition, n_partition)
    interface = _interface(graph, result)
    output_rows, output_cols = _static_shape(
        graph.value_map()[interface.output_value], field="vector output"
    )
    _validate_partition_extent(m_partition, output_rows, "vector.m_partition")
    _validate_partition_extent(n_partition, output_cols, "vector.n_partition")
    tile = _int_list(plan.get("tile"), 2, "vector.tile")
    strip = _int_list(plan.get("strip"), 2, "vector.strip")
    strip_grid = _int_list(plan.get("strip_grid"), 2, "vector.strip_grid")
    frame = _physical_frame_contract(plan)
    body = _mapping(plan.get("body"), "vector.body")
    trips = _positive_int(body.get("trip_count"), "vector.body.trip_count")
    stages = _positive_int(body.get("pipeline_stages"), "vector.body.pipeline_stages")
    if body.get("first_chunk") != 0 or trips != strip_grid[0] * strip_grid[1]:
        raise SourceEmissionError("vector body loop does not match its strip grid")
    if strip_grid != [
        _ceil_div(tile[0], strip[0]),
        _ceil_div(tile[1], strip[1]),
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

    problem = _problem(result)
    expected_rows = max(
        _positive_int(item, "vector problem height")
        for item in _array(problem, "heights")
    )
    expected_cols = max(
        _positive_int(item, "vector problem width")
        for item in _array(problem, "widths")
    )
    if (frame.iteration_rows, frame.iteration_cols) != (
        expected_rows,
        expected_cols,
    ):
        raise SourceEmissionError(
            "vector physical frame does not match the lowered iteration geometry"
        )
    _validate_vector_body_lifetimes(plan, step, problem)
    writer = _program_header(
        program_name, interface, graph, m_partition.parts * n_partition.parts
    )
    indent = 4
    _emit_partition_indices(writer, indent, m_partition, n_partition, emit_extents=True)
    iteration_rows = (
        str(frame.iteration_rows) if frame.reduced_axis == 2 else "region_rows"
    )
    iteration_cols = (
        str(frame.iteration_cols) if frame.reduced_axis == 1 else "region_cols"
    )
    writer.line(
        indent,
        f"with pl.at(level=pl.Level.CORE_GROUP, name_hint={_literal(result.region.id + '_vector')}):",
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
        result,
        step,
        problem,
        interface,
        frame,
        strip_rows=strip[0],
        strip_cols=strip[1],
    )
    writer.line(2, f"return {interface.output_argument}")
    return writer.render()


def _emit_vector_body(
    writer: _SourceWriter,
    indent: int,
    graph: NormalizedGraph,
    result: RegionSolveResult,
    step: KernelStep,
    problem: Mapping[str, Any],
    interface: _Interface,
    frame: _PhysicalFrameContract,
    *,
    strip_rows: int,
    strip_cols: int,
) -> None:
    inputs = _array(problem, "inputs")
    outputs = _array(problem, "outputs")
    dtypes = _array(problem, "dtypes")
    heights = _array(problem, "heights")
    widths = _array(problem, "widths")
    graph_ops = graph.op_map()
    producers = _tensor_producers(outputs, len(dtypes))
    local: dict[int, str] = {}

    def tensor_shape(tensor: int) -> tuple[int, int]:
        return (
            _positive_int(heights[tensor], f"heights[{tensor}]"),
            _positive_int(widths[tensor], f"widths[{tensor}]"),
        )

    def ensure_loaded(tensor: int) -> str:
        if tensor in local:
            return local[tensor]
        if producers[tensor] is not None:
            raise SourceEmissionError(
                f"solver tensor {tensor} is used before its producer"
            )
        value_id = result.solver_tensor_to_value[tensor]
        argument = interface.input_arguments.get(value_id)
        if argument is None:
            raise SourceEmissionError(
                f"external solver tensor {tensor} ({value_id}) is not a direct region input"
            )
        rows, cols = tensor_shape(tensor)
        logical_rows = 1 if rows == 1 else min(strip_rows, rows)
        logical_cols = 1 if cols == 1 else min(strip_cols, cols)
        physical_rows, physical_cols = _allocated_frame(
            rows,
            cols,
            logical_rows,
            logical_cols,
            frame,
        )
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

    for solver_op in step.op_order:
        op_inputs = [
            _index(item, len(dtypes), f"inputs[{solver_op}]")
            for item in _sequence(inputs[solver_op])
        ]
        op_outputs = [
            _index(item, len(dtypes), f"outputs[{solver_op}]")
            for item in _sequence(outputs[solver_op])
        ]
        if len(op_outputs) != 1:
            raise SourceEmissionError(f"solver op {solver_op} must have one output")
        operands = [ensure_loaded(tensor) for tensor in op_inputs]
        output = op_outputs[0]
        graph_op_id = result.solver_op_to_graph[solver_op]
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
            dtypes,
            heights,
            widths,
            frame,
            strip_rows,
            strip_cols,
            solver_op,
        )
        local[output] = f"tensor_{output}"
        writer.line(indent, f"{local[output]} = {expression}")

    output_tensor = _solver_tensor_for_value(result, interface.output_value)
    try:
        output_tile = local[output_tensor]
    except KeyError as error:
        raise SourceEmissionError(
            "region output is not produced by the selected step"
        ) from error
    writer.line(
        indent,
        f"{interface.output_argument} = pl.store({output_tile}, "
        f"[region_row + strip_row, region_col + strip_col], {interface.output_argument})",
    )


def _vector_expression(  # noqa: PLR0913 -- every argument is one explicit emit contract.
    writer: _SourceWriter,
    indent: int,
    op: NormalizedOp,
    operands: list[str],
    input_tensors: list[int],
    output_tensor: int,
    dtypes: Sequence[Any],
    heights: Sequence[Any],
    widths: Sequence[Any],
    frame: _PhysicalFrameContract,
    strip_rows: int,
    strip_cols: int,
    solver_op: int,
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
        dtype = _pypto_dtype(str(dtypes[output_tensor]))
        return f"pl.cast({operands[0]}, target_type={dtype})"
    if op.kind in {"sum", "max"} and len(operands) == 1:
        if op.attributes.get("axis") != -1 or op.attributes.get("keepdim") is not True:
            raise SourceEmissionError(
                f"vector source v1 supports only last-axis keepdim reductions, got {op.id}"
            )
        rows = _positive_int(heights[input_tensors[0]], "reduction input height")
        cols = _positive_int(widths[input_tensors[0]], "reduction input width")
        logical_rows = 1 if rows == 1 else min(strip_rows, rows)
        logical_cols = 1 if cols == 1 else min(strip_cols, cols)
        physical_rows, physical_cols = _allocated_frame(
            rows,
            cols,
            logical_rows,
            logical_cols,
            frame,
        )
        scratch = f"scratch_{solver_op}"
        writer.line(
            indent,
            f"{scratch} = pl.tile.create([{physical_rows}, {physical_cols}], "
            f"dtype={_pypto_dtype(str(dtypes[input_tensors[0]]))}, target_memory=pl.Mem.Vec)",
        )
        reduction = "row_sum" if op.kind == "sum" else "row_max"
        return f"pl.{reduction}({operands[0]}, {scratch})"

    if op.kind not in {"add", "sub", "mul", "div", "maximum", "minimum"}:
        raise SourceEmissionError(f"vector source v1 does not implement {op.kind!r}")
    if len(operands) == 1:
        scalar = _single_scalar(op)
        return f"pl.{op.kind}({operands[0]}, {_literal(scalar)})"
    if len(operands) != 2:
        raise SourceEmissionError(
            f"{op.kind} requires one tensor plus a scalar or two tensors"
        )

    shapes = [
        (
            _positive_int(heights[tensor], f"heights[{tensor}]"),
            _positive_int(widths[tensor], f"widths[{tensor}]"),
        )
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


def _emit_cube(
    graph: NormalizedGraph,
    result: RegionSolveResult,
    step: KernelStep,
    program_name: str,
) -> str:
    problem = _problem(result)
    plan = step.plan
    if plan.get("emit_compatible") is not True:
        raise SourceEmissionError("cube plan is not marked emit-compatible")
    if plan.get("spatial_policy") != "uniform":
        raise SourceEmissionError(
            "cube source v1 supports only uniform spatial partitions"
        )
    if (
        step.split != 1
        or plan.get("split_k") != 1
        or plan.get("split_merge_policy") != "none"
    ):
        raise SourceEmissionError(
            "cube split-K emission is not implemented in source v1"
        )
    if _sequence(plan.get("resident_boundaries")):
        raise SourceEmissionError(
            "cube resident-boundary emission is not implemented in source v1"
        )
    matmuls = _sequence(plan.get("matmuls"))
    if len(matmuls) != 1 or len(step.solver_ops) != 1:
        raise SourceEmissionError("cube source v1 supports exactly one matmul")
    matmul = _mapping(matmuls[0], "cube.matmul")
    solver_op = step.solver_ops[0]
    execution_order = [
        _index(item, len(result.solver_op_to_graph), "cube.execution_order")
        for item in _sequence(plan.get("execution_order"))
    ]
    if execution_order != [solver_op]:
        raise SourceEmissionError(
            "cube execution order does not match the selected one-matmul step"
        )
    if (
        _nonnegative_int(matmul.get("instance"), "cube.matmul.instance") != 0
        or _index(matmul.get("op"), len(result.solver_op_to_graph), "cube.matmul.op")
        != solver_op
    ):
        raise SourceEmissionError("cube request identity does not match its solver op")
    retained = _mapping(matmul.get("retained_panels"), "cube.retained_panels")
    if retained.get("lhs") or retained.get("rhs"):
        raise SourceEmissionError(
            "cube retained-panel emission is not implemented in source v1"
        )
    if matmul.get("lhs_producer") != -1 or matmul.get("rhs_producer") != -1:
        raise SourceEmissionError("cube source v1 requires external matmul operands")
    if matmul.get("output_grid") != [1, 1]:
        raise SourceEmissionError(
            "cube source v1 requires one L0 output tile per region"
        )
    if (
        matmul.get("accumulator_dtype") != "fp32"
        or matmul.get("storage_dtype") != "fp32"
    ):
        raise SourceEmissionError(
            "cube source v1 currently supports FP32 accumulation and storage"
        )

    m_partition = AxisPartition.from_json(
        plan.get("m_partition"), field="cube.m_partition"
    )
    n_partition = AxisPartition.from_json(
        plan.get("n_partition"), field="cube.n_partition"
    )
    _validate_grid(step, plan, m_partition, n_partition)
    if m_partition.big != m_partition.small or n_partition.big != n_partition.small:
        raise SourceEmissionError(
            "cube source v1 does not yet emit ragged spatial regions"
        )

    graph_op = graph.op_map()[result.solver_op_to_graph[solver_op]]
    if (
        graph_op.kind != "matmul"
        or graph_op.attributes.get("lhs_transposed")
        or graph_op.attributes.get("rhs_transposed")
    ):
        raise SourceEmissionError("cube source v1 requires a non-transposed matmul")
    inputs = _array(problem, "inputs")
    outputs = _array(problem, "outputs")
    op_inputs = [
        _index(item, len(result.solver_tensor_to_value), "matmul input")
        for item in inputs[solver_op]
    ]
    op_outputs = [
        _index(item, len(result.solver_tensor_to_value), "matmul output")
        for item in outputs[solver_op]
    ]
    if len(op_inputs) != 2 or len(op_outputs) != 1:
        raise SourceEmissionError("cube matmul must have two inputs and one output")
    interface = _interface(graph, result)
    output_rows, output_cols = _static_shape(
        graph.value_map()[interface.output_value], field="cube output"
    )
    _validate_partition_extent(m_partition, output_rows, "cube.m_partition")
    _validate_partition_extent(n_partition, output_cols, "cube.n_partition")
    lhs_value = result.solver_tensor_to_value[op_inputs[0]]
    rhs_value = result.solver_tensor_to_value[op_inputs[1]]
    if (
        lhs_value not in interface.input_arguments
        or rhs_value not in interface.input_arguments
    ):
        raise SourceEmissionError("cube operands must be direct region inputs")
    if _solver_tensor_for_value(result, interface.output_value) != op_outputs[0]:
        raise SourceEmissionError("cube matmul result must be the region output")

    k_loop = _mapping(matmul.get("k_loop"), "cube.matmul.k_loop")
    chunk = _positive_int(k_loop.get("chunk"), "cube.matmul.k_loop.chunk")
    full_chunks = _nonnegative_int(
        k_loop.get("full_chunks"), "cube.matmul.k_loop.full_chunks"
    )
    tail = _nonnegative_int(k_loop.get("tail"), "cube.matmul.k_loop.tail")
    stages = _positive_int(
        k_loop.get("pipeline_stages"), "cube.matmul.k_loop.pipeline_stages"
    )
    contraction = _positive_int(matmul.get("contraction"), "cube.matmul.contraction")
    if full_chunks * chunk + tail != contraction or full_chunks == 0:
        raise SourceEmissionError(
            "cube K-window descriptor does not cover the contraction"
        )
    if stages not in {1, 2} or (full_chunks == 1 and stages != 1):
        raise SourceEmissionError("cube K-window pipeline stage count is unsupported")
    if tail >= chunk:
        raise SourceEmissionError("cube K-window tail must be smaller than its chunk")
    output_tile = _int_list(matmul.get("output_tile"), 2, "cube.matmul.output_tile")
    if output_tile != [m_partition.big, n_partition.big]:
        raise SourceEmissionError(
            "cube output tile does not match its spatial partition"
        )
    _validate_l0_variant(matmul, output_tile, chunk, tail, full_chunks)

    writer = _program_header(
        program_name, interface, graph, m_partition.parts * n_partition.parts
    )
    indent = 4
    _emit_partition_indices(
        writer, indent, m_partition, n_partition, emit_extents=False
    )
    writer.line(
        indent,
        f"with pl.at(level=pl.Level.CORE_GROUP, name_hint={_literal(result.region.id + '_cube')}):",
    )
    indent += 1
    lhs_arg = interface.input_arguments[lhs_value]
    rhs_arg = interface.input_arguments[rhs_value]
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
        f"{interface.output_argument} = pl.store(accumulator, [region_row, region_col], "
        f"{interface.output_argument})",
    )
    writer.line(2, f"return {interface.output_argument}")
    return writer.render()


def _emit_cube_window(
    writer: _SourceWriter,
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
            indent, f"accumulator = pl.tile.matmul(lhs_left{suffix}, rhs_right{suffix})"
        )
    else:
        writer.line(
            indent,
            f"accumulator = pl.tile.matmul_acc(accumulator, lhs_left{suffix}, rhs_right{suffix})",
        )


def _validate_l0_variant(
    matmul: Mapping[str, Any],
    output_tile: list[int],
    chunk: int,
    tail: int,
    full_chunks: int,
) -> None:
    variants = _sequence(matmul.get("output_variants"))
    if len(variants) != 1:
        raise SourceEmissionError("cube source v1 requires one output-shape variant")
    variant = _mapping(variants[0], "cube.output_variant")
    if variant.get("count") != 1 or variant.get("shape") != output_tile:
        raise SourceEmissionError("cube output variant does not match one region tile")
    expected = [output_tile[0], output_tile[1], chunk]
    init = _mapping(variant.get("l0_init"), "cube.l0_init")
    if init.get("tile") != expected:
        raise SourceEmissionError(
            "cube initial L0 tile differs from the emitted K window"
        )
    if full_chunks > 1:
        rolled = _mapping(variant.get("l0_rolled"), "cube.l0_rolled")
        if rolled.get("tile") != expected:
            raise SourceEmissionError(
                "cube rolled L0 tile differs from the emitted K window"
            )
    if tail:
        tail_plan = _mapping(variant.get("l0_tail"), "cube.l0_tail")
        if tail_plan.get("tile") != [output_tile[0], output_tile[1], tail]:
            raise SourceEmissionError("cube tail L0 tile differs from the emitted tail")


@dataclass(frozen=True)
class _Interface:
    input_arguments: Mapping[str, str]
    output_value: str
    output_argument: str


def _interface(graph: NormalizedGraph, result: RegionSolveResult) -> _Interface:
    if len(result.region.output_values) != 1:
        raise SourceEmissionError("source v1 supports exactly one region output")
    values = graph.value_map()
    input_arguments: dict[str, str] = {}
    used: set[str] = {"self", "region_index"}
    for index, value_id in enumerate(result.region.input_values):
        value = values[value_id]
        if value.alias_of is not None:
            raise SourceEmissionError("source v1 does not emit aliased region inputs")
        name = _unique_name(_identifier(value.name or f"input_{index}"), used)
        input_arguments[value_id] = name
    output_value = result.region.output_values[0]
    output_argument = _unique_name("output", used)
    return _Interface(input_arguments, output_value, output_argument)


def _program_header(
    program_name: str,
    interface: _Interface,
    graph: NormalizedGraph,
    work_units: int,
) -> _SourceWriter:
    values = graph.value_map()
    output = values[interface.output_value]
    writer = _SourceWriter()
    writer.line(0, '"""Generated by PTO-Fusebox from a solver-owned schedule."""')
    writer.line()
    writer.line(0, "import pypto.language as pl")
    writer.line()
    writer.line()
    writer.line(0, "@pl.program")
    writer.line(0, f"class {program_name}:")
    writer.line(1, "@pl.function(type=pl.FunctionType.Orchestration)")
    writer.line(1, "def main(")
    writer.line(2, "self,")
    for value_id, argument in interface.input_arguments.items():
        writer.line(2, f"{argument}: {_tensor_type(values[value_id])},")
    writer.line(2, f"{interface.output_argument}: pl.Out[{_tensor_type(output)}],")
    writer.line(1, f") -> {_tensor_type(output)}:")
    writer.line(2, "with pl.manual_scope():")
    writer.line(3, f"for region_index in pl.parallel({work_units}):")
    return writer


def _emit_partition_indices(
    writer: _SourceWriter,
    indent: int,
    m_partition: AxisPartition,
    n_partition: AxisPartition,
    *,
    emit_extents: bool,
) -> None:
    writer.line(indent, f"m_index = region_index // {n_partition.parts}")
    writer.line(indent, f"n_index = region_index % {n_partition.parts}")
    _emit_axis_partition(writer, indent, "m", m_partition, emit_extent=emit_extents)
    _emit_axis_partition(writer, indent, "n", n_partition, emit_extent=emit_extents)


def _emit_axis_partition(
    writer: _SourceWriter,
    indent: int,
    prefix: str,
    partition: AxisPartition,
    *,
    emit_extent: bool,
) -> None:
    if partition.big == partition.small:
        writer.line(
            indent,
            f"region_{'row' if prefix == 'm' else 'col'} = {prefix}_index * {partition.big}",
        )
        if emit_extent:
            writer.line(
                indent,
                f"region_{'rows' if prefix == 'm' else 'cols'} = {partition.big}",
            )
        return
    writer.line(
        indent, f"{prefix}_big_before = pl.min({prefix}_index, {partition.num_big})"
    )
    writer.line(
        indent,
        f"region_{'row' if prefix == 'm' else 'col'} = "
        f"{prefix}_index * {partition.small} + {prefix}_big_before * "
        f"{partition.big - partition.small}",
    )
    if emit_extent:
        writer.line(
            indent,
            f"region_{'rows' if prefix == 'm' else 'cols'} = {partition.small} + "
            f"(pl.min({prefix}_index + 1, {partition.num_big}) - {prefix}_big_before) * "
            f"{partition.big - partition.small}",
        )


def _physical_frame_contract(plan: Mapping[str, Any]) -> _PhysicalFrameContract:
    item = _mapping(plan.get("physical_frame"), "vector.physical_frame")
    reduced_axis = _positive_int(
        item.get("reduced_axis"), "physical_frame.reduced_axis", allow_zero=True
    )
    if reduced_axis not in {0, 1, 2}:
        raise SourceEmissionError("physical_frame.reduced_axis must be 0, 1, or 2")
    align_rows = item.get("align_rows")
    if not isinstance(align_rows, bool):
        raise SourceEmissionError("physical_frame.align_rows must be boolean")
    return _PhysicalFrameContract(
        element_granule=_positive_int(
            item.get("element_granule"), "physical_frame.element_granule"
        ),
        iteration_rows=_positive_int(
            item.get("iteration_rows"), "physical_frame.iteration_rows"
        ),
        iteration_cols=_positive_int(
            item.get("iteration_cols"), "physical_frame.iteration_cols"
        ),
        reduced_axis=reduced_axis,
        align_rows=align_rows,
    )


def _allocated_frame(
    tensor_rows: int,
    tensor_cols: int,
    logical_rows: int,
    logical_cols: int,
    contract: _PhysicalFrameContract,
) -> tuple[int, int]:
    row_broadcast = tensor_rows == 1 and contract.iteration_rows > 1
    col_broadcast = tensor_cols == 1 and contract.iteration_cols > 1
    thin_reduction_row = tensor_rows == 1 and contract.reduced_axis == 2
    thin_reduction_col = tensor_cols == 1 and contract.reduced_axis == 1
    rows, cols = logical_rows, logical_cols
    if contract.align_rows and not row_broadcast and not thin_reduction_row:
        rows = _align_up(rows, contract.element_granule)
    if not col_broadcast and not thin_reduction_col:
        cols = _align_up(cols, contract.element_granule)
    return rows, cols


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


def _validate_grid(
    step: KernelStep,
    plan: Mapping[str, Any],
    m_partition: AxisPartition,
    n_partition: AxisPartition,
) -> None:
    work_units = _positive_int(plan.get("work_units"), "plan.work_units")
    if (
        step.parts_m != m_partition.parts
        or step.parts_n != n_partition.parts
        or work_units != m_partition.parts * n_partition.parts
    ):
        raise SourceEmissionError(
            "selected grid, partition, and work-unit counts disagree"
        )


def _validate_partition_extent(
    partition: AxisPartition, extent: int, field: str
) -> None:
    covered = (
        partition.num_big * partition.big
        + (partition.parts - partition.num_big) * partition.small
    )
    if covered != extent:
        raise SourceEmissionError(
            f"{field} covers {covered} elements but its output axis has extent {extent}"
        )


def _tensor_producers(outputs: Sequence[Any], tensor_count: int) -> list[int | None]:
    producers: list[int | None] = [None] * tensor_count
    for op, encoded in enumerate(outputs):
        for item in _sequence(encoded):
            tensor = _index(item, tensor_count, f"outputs[{op}]")
            if producers[tensor] is not None:
                raise SourceEmissionError(
                    f"solver tensor {tensor} has multiple producers"
                )
            producers[tensor] = op
    return producers


def _validate_vector_body_lifetimes(
    plan: Mapping[str, Any],
    step: KernelStep,
    problem: Mapping[str, Any],
) -> None:
    inputs = _array(problem, "inputs")
    outputs = _array(problem, "outputs")
    tensor_count = len(_array(problem, "dtypes"))
    producers = _tensor_producers(outputs, tensor_count)
    expected: dict[int, list[tuple[int, int]]] = {}
    position = {solver_op: index for index, solver_op in enumerate(step.op_order)}
    for solver_op in step.op_order:
        for argument, item in enumerate(_sequence(inputs[solver_op])):
            tensor = _index(item, tensor_count, f"inputs[{solver_op}]")
            if producers[tensor] is None:
                expected.setdefault(tensor, []).append((solver_op, argument))

    lifetimes = _mapping(plan.get("input_lifetimes"), "vector.input_lifetimes")
    encoded = _sequence(lifetimes.get("body"))
    actual: dict[int, list[tuple[int, int]]] = {}
    for index, item in enumerate(encoded):
        lifetime = _mapping(item, f"vector.input_lifetimes.body[{index}]")
        tensor = _index(
            lifetime.get("tensor"), tensor_count, f"vector input lifetime {index}"
        )
        if tensor in actual:
            raise SourceEmissionError(
                f"vector body has duplicate lifetime descriptors for tensor {tensor}"
            )
        uses = [
            (
                _index(
                    _mapping(use, "vector input use").get("op"),
                    len(inputs),
                    "vector input use op",
                ),
                _nonnegative_int(
                    _mapping(use, "vector input use").get("arg"),
                    "vector input use arg",
                ),
            )
            for use in _sequence(lifetime.get("uses"))
        ]
        if any(op not in position for op, _ in uses):
            raise SourceEmissionError(
                "vector input lifetime references an unselected op"
            )
        if _positive_int(lifetime.get("use_count"), "vector input use_count") != len(
            uses
        ):
            raise SourceEmissionError("vector input lifetime use count is inconsistent")
        first = min(position[op] for op, _ in uses)
        last = max(position[op] for op, _ in uses)
        if (
            _nonnegative_int(
                lifetime.get("first_use_step"), "vector input first_use_step"
            )
            != first
            or _nonnegative_int(
                lifetime.get("last_use_step"), "vector input last_use_step"
            )
            != last
        ):
            raise SourceEmissionError("vector input lifetime bounds are inconsistent")
        actual[tensor] = sorted(uses)

    normalized_expected = {tensor: sorted(uses) for tensor, uses in expected.items()}
    if actual != normalized_expected:
        raise SourceEmissionError(
            "vector body input lifetimes do not match boundary-tensor uses"
        )


def _solver_tensor_for_value(result: RegionSolveResult, value_id: str) -> int:
    matches = [
        index
        for index, item in enumerate(result.solver_tensor_to_value)
        if item == value_id
    ]
    if len(matches) != 1:
        raise SourceEmissionError(
            f"value {value_id} does not map to exactly one solver tensor"
        )
    return matches[0]


def _tensor_type(value: NormalizedValue) -> str:
    rows, cols = _static_shape(value, field=value.id)
    return f"pl.Tensor[[{rows}, {cols}], {_pypto_dtype(value.dtype)}]"


def _static_shape(value: NormalizedValue, *, field: str) -> tuple[int, int]:
    if len(value.shape) != 2 or any(
        isinstance(dim, ShapeDimension) for dim in value.shape
    ):
        raise SourceEmissionError(
            f"source v1 requires a static rank-2 value, got {field}"
        )
    rows, cols = value.shape
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise SourceEmissionError(f"source v1 requires static dimensions for {field}")
    return rows, cols


def _pypto_dtype(dtype: str) -> str:
    names = {
        "FP32": "pl.FP32",
        "FP16": "pl.FP16",
        "BF16": "pl.BF16",
        "INT32": "pl.INT32",
        "INT16": "pl.INT16",
        "INT8": "pl.INT8",
        "BOOL": "pl.BOOL",
        "float32": "pl.FP32",
        "float16": "pl.FP16",
        "bfloat16": "pl.BF16",
        "int32": "pl.INT32",
        "int16": "pl.INT16",
        "int8": "pl.INT8",
        "bool": "pl.BOOL",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise SourceEmissionError(f"unsupported PyPTO dtype {dtype!r}") from error


def _class_name(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value)
    result = "".join(word[:1].upper() + word[1:] for word in words) or "FusedRegion"
    if result[0].isdigit():
        result = f"Fused{result}"
    return result


def _identifier(value: str) -> str:
    result = re.sub(r"\W", "_", value).strip("_") or "value"
    if result[0].isdigit() or keyword.iskeyword(result):
        result = f"value_{result}"
    return result


def _unique_name(base: str, used: set[str]) -> str:
    result = base
    suffix = 1
    while result in used:
        result = f"{base}_{suffix}"
        suffix += 1
    used.add(result)
    return result


def _problem(result: RegionSolveResult) -> Mapping[str, Any]:
    if result.problem is None:
        raise SourceEmissionError("solved region has no problem")
    return result.problem


def _array(value: Mapping[str, Any], field: str) -> Sequence[Any]:
    return _sequence(value.get(field))


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceEmissionError(f"{field} must be an object")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SourceEmissionError("expected an array in the solver contract")
    return value


def _int_list(value: Any, size: int, field: str) -> list[int]:
    items = _sequence(value)
    if len(items) != size:
        raise SourceEmissionError(f"{field} must contain {size} integers")
    return [_positive_int(item, field) for item in items]


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SourceEmissionError(f"{field} must be a {qualifier} integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    return _positive_int(value, field, allow_zero=True)


def _index(value: Any, bound: int, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result >= bound:
        raise SourceEmissionError(f"{field} contains out-of-range index {result}")
    return result


def _align_up(value: int, granule: int) -> int:
    return ((value + granule - 1) // granule) * granule


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _literal(value: Any) -> str:
    return repr(value)
