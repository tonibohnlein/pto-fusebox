"""Shared naming, interface, and partition mechanics for source backends."""

from __future__ import annotations

import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..ir import NormalizedGraph, NormalizedOp, NormalizedValue, ShapeDimension
from ..lowered import LoweredRegion
from ..schedule import AxisPartition, KernelStep, ScheduledRegion


class SourceEmissionError(ValueError):
    """Raised when a valid analytic schedule is outside the source backend."""


def broadcast_operands(shapes: Sequence[tuple[int, int]]) -> tuple[int, int, str]:
    """Classify one supported rank-two singleton broadcast.

    Returns the wide operand index, thin operand index, and PyPTO expansion
    geometry (``row`` for ``[M, 1]`` and ``col`` for ``[1, N]``).  Keeping this
    classifier shared makes homogeneous and mixed vector replay choose the same
    operation for the same tensor shapes.
    """

    if len(shapes) != 2:
        raise SourceEmissionError(
            f"vector broadcast requires two operands, got {len(shapes)}"
        )
    for wide in (0, 1):
        thin = 1 - wide
        wide_rows, wide_cols = shapes[wide]
        thin_rows, thin_cols = shapes[thin]
        if thin_rows == wide_rows and thin_cols == 1 and wide_cols > 1:
            return wide, thin, "row"
        if thin_rows == 1 and thin_cols == wide_cols and wide_rows > 1:
            return wide, thin, "col"
    raise SourceEmissionError(f"unsupported vector broadcast geometry {shapes}")


class SourceWriter:
    """Small deterministic indentation-aware source builder."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def line(self, indent: int = 0, text: str = "") -> None:
        """Append one line at ``indent`` levels."""

        self._lines.append(f"{'    ' * indent}{text}" if text else "")

    def render(self) -> str:
        """Return source with one canonical trailing newline."""

        return "\n".join(self._lines) + "\n"


@dataclass(frozen=True)
class Interface:
    """PyPTO orchestration signature for one scheduled region."""

    input_arguments: Mapping[str, str]
    output_arguments: Mapping[str, str]
    output_allocation_owners: Mapping[str, str]

    @property
    def output_values(self) -> tuple[str, ...]:
        """Return outputs in their region ABI order."""

        return tuple(self.output_arguments)

    @property
    def output_value(self) -> str:
        """Return the unique output expected by a one-output renderer."""

        if len(self.output_arguments) != 1:
            raise SourceEmissionError(
                "this schedule renderer requires exactly one region output"
            )
        return self.output_values[0]

    @property
    def output_argument(self) -> str:
        """Return the unique output argument expected by a one-output renderer."""

        return self.output_arguments[self.output_value]

    @property
    def output_allocation_owner(self) -> str:
        """Return the unique output allocation owner for compatibility."""

        return self.output_allocation_owners[self.output_value]


@dataclass(frozen=True)
class EmissionContext:
    """Validated graph, problem, schedule, and ABI for one source backend."""

    graph: NormalizedGraph
    problem: Mapping[str, Any]
    lowered: LoweredRegion
    schedule: ScheduledRegion
    step: KernelStep
    interface: Interface

    @property
    def region_id(self) -> str:
        return self.lowered.region_id


@dataclass(frozen=True)
class PartitionCoordinates:
    """Source expressions for one region's logical origin."""

    row: str
    col: str


def interface(graph: NormalizedGraph, lowered: LoweredRegion) -> Interface:
    """Derive deterministic names for the region ABI."""

    values = graph.value_map()
    input_arguments: dict[str, str] = {}
    used: set[str] = {"self", "pl", "region_index"}
    for index, value_id in enumerate(lowered.region_inputs):
        try:
            value = values[value_id]
        except KeyError as error:
            raise SourceEmissionError(
                f"region input {value_id!r} is absent from the normalized graph"
            ) from error
        if value.alias_of is not None:
            raise SourceEmissionError("source v1 does not emit aliased region inputs")
        name = unique_name(f"arg_{identifier(value.name or f'input_{index}')}", used)
        input_arguments[value_id] = name
    output_arguments: dict[str, str] = {}
    output_owners: dict[str, str] = {}
    multiple_outputs = len(lowered.region_outputs) != 1
    for index, (output_value, output_owner) in enumerate(
        zip(
            lowered.region_outputs,
            lowered.output_allocation_owners,
            strict=True,
        )
    ):
        if output_value not in values:
            raise SourceEmissionError(
                f"region output {output_value!r} is absent from the normalized graph"
            )
        name = "output" if not multiple_outputs else f"output_{index}"
        output_arguments[output_value] = unique_name(name, used)
        output_owners[output_value] = output_owner
    return Interface(input_arguments, output_arguments, output_owners)


def program_header(
    program_name: str,
    io: Interface,
    graph: NormalizedGraph,
    work_units: int,
    *,
    kernel_name_hint: str,
) -> SourceWriter:
    """Emit one homogeneous kernel as a single SPMD grid dispatch."""

    writer = program_preamble(program_name, io, graph)
    writer.line(
        2,
        f"for region_index in pl.spmd({work_units}, "
        f"name_hint={literal(kernel_name_hint)}):",
    )
    return writer


def program_preamble(
    program_name: str,
    io: Interface,
    graph: NormalizedGraph,
) -> SourceWriter:
    """Emit a program and orchestration signature without scheduling its body."""

    values = graph.value_map()
    writer = SourceWriter()
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
    for value_id, argument in io.input_arguments.items():
        writer.line(2, f"{argument}: {tensor_type(values[value_id])},")
    for output_value, output_argument in io.output_arguments.items():
        writer.line(
            2,
            f"{output_argument}: pl.Out[{tensor_type(values[output_value])}],",
        )
    return_types = [tensor_type(values[value]) for value in io.output_values]
    return_type = (
        return_types[0]
        if len(return_types) == 1
        else f"tuple[{', '.join(return_types)}]"
    )
    writer.line(1, f") -> {return_type}:")
    return writer


def emit_return(writer: SourceWriter, io: Interface) -> None:
    """Emit the region return in ABI order."""

    arguments = tuple(io.output_arguments.values())
    if len(arguments) == 1:
        writer.line(2, f"return {arguments[0]}")
    else:
        writer.line(2, f"return {', '.join(arguments)}")


def emit_partition_indices(
    writer: SourceWriter,
    indent: int,
    m_partition: AxisPartition,
    n_partition: AxisPartition,
    *,
    clamped_overlap_extents: tuple[int, int] | None = None,
) -> PartitionCoordinates:
    """Emit deterministic coordinates for two solver-owned axis partitions.

    ``clamped_overlap_extents`` turns the balanced logical ownership into a
    static physical replay: every work unit uses the partition's maximum tile,
    and a ragged final origin is clamped backwards.  The resulting overlap is
    intentional and must already be priced by the selected schedule.
    """

    if m_partition.parts > 1:
        m_index = "region_index" if n_partition.parts == 1 else "m_index"
        if m_index != "region_index":
            writer.line(indent, f"m_index = region_index // {n_partition.parts}")
    else:
        m_index = "0"
    if n_partition.parts > 1:
        n_index = "region_index" if m_partition.parts == 1 else "n_index"
        if n_index != "region_index":
            writer.line(indent, f"n_index = region_index % {n_partition.parts}")
    else:
        n_index = "0"
    m_extent = None if clamped_overlap_extents is None else clamped_overlap_extents[0]
    n_extent = None if clamped_overlap_extents is None else clamped_overlap_extents[1]
    row = _emit_axis_partition(
        writer,
        indent,
        "m",
        m_index,
        m_partition,
        clamped_overlap_extent=m_extent,
    )
    col = _emit_axis_partition(
        writer,
        indent,
        "n",
        n_index,
        n_partition,
        clamped_overlap_extent=n_extent,
    )
    return PartitionCoordinates(row=row, col=col)


def _emit_axis_partition(
    writer: SourceWriter,
    indent: int,
    prefix: str,
    index: str,
    partition: AxisPartition,
    *,
    clamped_overlap_extent: int | None,
) -> str:
    coordinate = "row" if prefix == "m" else "col"
    if partition.parts == 1:
        return "0"
    if partition.big == partition.small:
        writer.line(
            indent,
            f"region_{coordinate} = {index} * {partition.big}",
        )
        return f"region_{coordinate}"
    writer.line(
        indent,
        f"{prefix}_big_before = pl.min({index}, {partition.num_big})",
    )
    offset = (
        f"{index} * {partition.small} + {prefix}_big_before * "
        f"{partition.big - partition.small}"
    )
    if clamped_overlap_extent is not None:
        maximum_offset = clamped_overlap_extent - partition.big
        if maximum_offset < 0:
            raise SourceEmissionError(
                f"clamped {prefix} partition tile exceeds its logical extent"
            )
        offset = f"pl.min({offset}, {maximum_offset})"
    writer.line(
        indent,
        f"region_{coordinate} = {offset}",
    )
    return f"region_{coordinate}"


def validate_grid(
    step: KernelStep,
    work_units: int,
    m_partition: AxisPartition,
    n_partition: AxisPartition,
) -> None:
    """Check that launch and partition fields describe one grid."""

    if (
        step.parts_m != m_partition.parts
        or step.parts_n != n_partition.parts
        or work_units != m_partition.parts * n_partition.parts
    ):
        raise SourceEmissionError(
            "selected grid, partition, and work-unit counts disagree"
        )


def validate_partition_extent(
    partition: AxisPartition, extent: int, field: str
) -> None:
    """Check exact coverage of an axis partition."""

    covered = (
        partition.num_big * partition.big
        + (partition.parts - partition.num_big) * partition.small
    )
    if covered != extent:
        raise SourceEmissionError(
            f"{field} covers {covered} elements but its output axis has extent {extent}"
        )


def solver_tensor_for_value(lowered: LoweredRegion, value_id: str) -> int:
    """Return the unique lowered tensor for a normalized value."""

    matches = [
        tensor.index for tensor in lowered.tensors if tensor.value_id == value_id
    ]
    if len(matches) != 1:
        raise SourceEmissionError(
            f"value {value_id} does not map to exactly one solver tensor"
        )
    return matches[0]


def tensor_type(value: NormalizedValue) -> str:
    """Render one static PyPTO tensor type in solver-owned rank-two geometry."""

    rows, cols = static_shape(value, field=value.id)
    return f"pl.Tensor[[{rows}, {cols}], {pypto_dtype(value.dtype)}]"


def static_shape(value: NormalizedValue, *, field: str) -> tuple[int, int]:
    """Return the static rank-two geometry consumed by the source backend.

    The analytic solver represents a rank-one ``[N]`` value as the row vector
    ``[1, N]``.  Preserve that same geometry in the generated PyPTO ABI so a
    rank-one row-broadcast operand does not become source-infeasible after it
    has already been admitted and priced correctly.
    """

    if len(value.shape) not in {1, 2} or any(
        isinstance(dim, ShapeDimension) for dim in value.shape
    ):
        raise SourceEmissionError(
            f"source v1 requires a static rank-1 or rank-2 value, got {field}"
        )
    if not all(isinstance(dim, int) for dim in value.shape):
        raise SourceEmissionError(f"source v1 requires static dimensions for {field}")
    if len(value.shape) == 1:
        (cols,) = value.shape
        assert isinstance(cols, int)
        return 1, cols
    rows, cols = value.shape
    assert isinstance(rows, int) and isinstance(cols, int)
    return rows, cols


def pypto_dtype(dtype: str) -> str:
    """Map one normalized/lowered dtype to its PyPTO spelling."""

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
        "fp32": "pl.FP32",
        "fp16": "pl.FP16",
        "bf16": "pl.BF16",
        "int32": "pl.INT32",
        "int16": "pl.INT16",
        "int8": "pl.INT8",
        "bool": "pl.BOOL",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise SourceEmissionError(f"unsupported PyPTO dtype {dtype!r}") from error


def class_name(value: str) -> str:
    """Convert an arbitrary program label into a deterministic class name."""

    words = re.findall(r"[A-Za-z0-9]+", value)
    result = "".join(word[:1].upper() + word[1:] for word in words) or "FusedRegion"
    if result[0].isdigit():
        result = f"Fused{result}"
    return result


def identifier(value: str) -> str:
    """Convert an arbitrary value name into a Python identifier."""

    result = re.sub(r"\W", "_", value).strip("_") or "value"
    if result[0].isdigit() or keyword.iskeyword(result):
        result = f"value_{result}"
    return result


def unique_name(base: str, used: set[str]) -> str:
    """Reserve a deterministic unique identifier."""

    result = base
    suffix = 1
    while result in used:
        result = f"{base}_{suffix}"
        suffix += 1
    used.add(result)
    return result


def ceil_div(value: int, divisor: int) -> int:
    """Return the integer ceiling of ``value / divisor``."""

    return (value + divisor - 1) // divisor


def scalar_operand(op: NormalizedOp) -> tuple[int, int | float]:
    """Return the unique normalized scalar operand position and value."""

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


def literal(value: Any) -> str:
    """Render a deterministic Python literal."""

    return repr(value)
