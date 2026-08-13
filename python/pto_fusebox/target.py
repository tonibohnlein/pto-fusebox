"""Extensible target admission and machine profiles for frontend lowering."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from .ir import PROBLEM_SCHEMA, NormalizedOp, NormalizedValue

_VECTOR_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
    "minimum",
    "exp",
    "log",
    "abs",
    "sqrt",
    "rsqrt",
    "neg",
    "sum",
    "max",
}


@dataclass(frozen=True)
class TargetProfile:
    name: str

    def admission_reason(self, op: NormalizedOp, values: Mapping[str, NormalizedValue]) -> str | None:
        """Return ``None`` when ``op`` is supported, otherwise a stable reason."""

        raise NotImplementedError

    def problem_fields(self) -> dict[str, object]:
        raise NotImplementedError

    def native_cast_path(self, source: str, target: str) -> tuple[str, ...] | None:
        """Return native destination dtypes for each conversion hop."""

        return (target,) if source != target else ()


@dataclass(frozen=True)
class Ascend910BTarget(TargetProfile):
    name: str = "ascend910b"

    def admission_reason(self, op: NormalizedOp, values: Mapping[str, NormalizedValue]) -> str | None:
        if not op.supported:
            return op.opaque_reason or "operator is opaque"
        if op.metadata_only:
            if op.kind == "transpose_view":
                source = values[op.inputs[0]]
                if source.producer is not None:
                    return "transpose of a region-produced value requires an explicit solver layout edge"
            return None
        operands = [values[value_id] for value_id in (*op.inputs, *op.outputs)]
        if any(value.storage_offset != 0 for value in operands):
            return "nonzero or unknown tensor storage offset is not represented by the v1 solver problem"
        if any(
            not _has_supported_dense_layout(value)
            for value in operands
            if not _is_explicit_transpose_alias(value, op, values)
        ):
            return "non-dense tensor layout is not represented by the v1 solver problem"
        dtypes = {value.dtype for value in operands}
        if "unknown" in dtypes:
            return "tensor dtype is unavailable"
        if op.kind == "matmul":
            return _matmul_admission_reason(op, values, operands)
        if op.kind == "cast":
            return _cast_admission_reason(op, values, dtypes, self)
        if op.kind in _VECTOR_OPS:
            return _vector_admission_reason(op, values, operands)
        return f"normalized operator {op.kind!r} has no Ascend910B lowering"

    def problem_fields(self) -> dict[str, object]:
        # These values mirror the checked-in production adapter. They are
        # descriptor data rather than hidden frontend constants, so another
        # target can supply a different profile without changing normalization.
        return {
            "schema_version": PROBLEM_SCHEMA,
            "fast_memory_capacity": 1 << 30,
            "num_cube_cores": 24,
            "num_vector_cores": 48,
            "cube_capacity": 128 * 1024,
            "l1_capacity": 512 * 1024,
            # PyPTO's Ascend910B profile reserves 8 KiB from the physical
            # 192-KiB Vec buffer. Use the same allocatable capacity so an
            # external schedule remains buildable by the generated DSL.
            "vec_capacity": 184 * 1024,
            "cube_compute_cost": 1,
            "kernel_fill_cost": 10000,
            "per_task_overhead_cycles": 64,
            "cube_split_sync_cycles": 0,
            "cube_freq_hz": 1.85e9,
            "bw_gm_l1": 135.0,
            "bw_l0c_gm": 70.0,
            "bw_l1_l0a": 441.0,
            "bw_l1_l0b": 220.5,
            "bw_gm_ub": 100.9,
            "bw_ub_gm": 188.46,
            "hbm_aggregate_gibps": 900.0,
            "l0_tile_m": 128,
            "l0_tile_n": 256,
            "l0_matmul_config": {
                "l0a_bytes": 64 * 1024,
                "l0b_bytes": 64 * 1024,
                "l0c_bytes": 128 * 1024,
                "min_m": 16,
                "min_n": 16,
                "min_k": 16,
                "align_m": 16,
                "align_n": 16,
                "align_k": 16,
                "allow_a_stationary": True,
                "allow_b_stationary": True,
                "allow_double_buffer_c": False,
                "allow_padding": False,
                "allow_k_boundary": True,
                "bw_l0a": 129.7,
                "bw_l0b": 85.4,
                "bw_drain": 118.0,
                "bw_l0c_l1": 128.0 * (1 << 30) / 1.85e9,
                "drain_fixed_cycles": 164.0,
                "drain_row_cycles": 4.45,
                "drain_penalty_cycles": 2.6,
                "drain_c0_bytes": 32,
                "mad_fp32_passes": 2,
                "mad_head_cycles": 21,
                "mad_k_fractal_bytes": 32,
            },
            "vec_reg_bytes": 256,
            "vec_dma_align_bytes": 32,
            "vec_op_head": 14.0,
            "vec_op_tail": 18.0,
            "vec_slope_pw": 2.0,
            "vec_slope_reduce": 14.0,
            "allow_model_ahead_split_k": True,
            "allow_model_ahead_multi_reduction_stream": True,
            "fuse_cube_vector": True,
            # PTO-Fusebox is the planner in the standalone source-to-source
            # architecture.  Its search space is therefore the analytic model,
            # not the narrower set of schedules replayable by the historical
            # in-compiler AutoFuse emitter.
            "require_buildable_mixed": False,
            "allow_model_ahead_mixed_multi_roundtrip": True,
            "require_uniform_cube_dag_grid": False,
            "use_hierarchical_cube_cost": True,
        }

    def native_cast_path(self, source: str, target: str) -> tuple[str, ...] | None:
        edges = {
            "float32": ("float16", "bfloat16", "int16", "int32"),
            "float16": ("float32", "int32", "int16", "int8"),
            "bfloat16": ("float32", "int32"),
            "int16": ("float16", "float32"),
            "int32": ("float32", "int16", "float16"),
            "int8": ("float16",),
        }
        if source == target:
            return ()
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, ())])
        visited = {source}
        while queue:
            current, path = queue.popleft()
            for destination in edges.get(current, ()):
                if destination in visited:
                    continue
                next_path = (*path, destination)
                if destination == target:
                    return next_path
                visited.add(destination)
                queue.append((destination, next_path))
        return None


def resolve_target(target: str | TargetProfile) -> TargetProfile:
    if isinstance(target, TargetProfile):
        return target
    if target.lower() in {"ascend910b", "910b", "a2a3"}:
        return Ascend910BTarget()
    raise ValueError(f"unknown PTO-Fusebox target {target!r}")


def _matmul_admission_reason(
    op: NormalizedOp,
    values: Mapping[str, NormalizedValue],
    operands: list[NormalizedValue],
) -> str | None:
    if len(op.inputs) != 2 or len(op.outputs) != 1:
        return "matmul must have two inputs and one output"
    flattened_linear = bool(op.attributes.get("flatten_leading_dimensions"))
    ranks_supported = (
        len(values[op.inputs[0]].shape) >= 2
        and len(values[op.inputs[1]].shape) == 2
        and len(values[op.outputs[0]].shape) >= 2
        and (all(len(value.shape) == 2 for value in operands) or flattened_linear)
    )
    if not ranks_supported:
        return "Ascend910B cube scheduling requires rank-2 normalized matmul tensors"
    lhs_geometry = _solver_geometry(values[op.inputs[0]])
    rhs_geometry = _solver_geometry(values[op.inputs[1]])
    output_geometry = _solver_geometry(values[op.outputs[0]])
    if lhs_geometry is not None and rhs_geometry is not None and output_geometry is not None:
        lhs_width, lhs_height = lhs_geometry
        rhs_width, rhs_height = rhs_geometry
        output_width, output_height = output_geometry
        if min(lhs_width, lhs_height, rhs_width, rhs_height, output_width, output_height) <= 0:
            return "matmul dimensions must be positive"
        if lhs_width != rhs_height:
            return (
                "matmul contraction dimensions do not match: "
                f"lhs K={lhs_width}, rhs K={rhs_height}"
            )
        if output_height != lhs_height or output_width != rhs_width:
            return (
                "matmul output geometry does not match its inputs: "
                f"expected [{lhs_height},{rhs_width}], got [{output_height},{output_width}]"
            )
    if any(values[value_id].dtype not in {"float16", "bfloat16", "float32"} for value_id in op.inputs):
        return "unsupported Ascend910B cube operand dtype"
    return None


def _cast_admission_reason(
    op: NormalizedOp,
    values: Mapping[str, NormalizedValue],
    dtypes: set[str],
    target: TargetProfile,
) -> str | None:
    if dtypes - {"float16", "bfloat16", "float32", "int8"}:
        return "unsupported Ascend910B cast endpoint dtype"
    source = values[op.inputs[0]].dtype
    destination = values[op.outputs[0]].dtype
    if target.native_cast_path(source, destination) is None:
        return f"Ascend910B has no native cast path from {source} to {destination}"
    return None


def _vector_admission_reason(
    op: NormalizedOp,
    values: Mapping[str, NormalizedValue],
    operands: list[NormalizedValue],
) -> str | None:
    # BF16 is a legal tensor storage/cube/cast endpoint, but PTOAS A2/A3
    # does not implement ordinary BF16 vector arithmetic.
    if any(value.dtype not in {"float16", "float32"} for value in operands):
        return "Ascend910B vector arithmetic supports FP16 and FP32 only"
    output = values[op.outputs[0]]
    if any(values[value_id].dtype != output.dtype for value_id in op.inputs):
        return "Ascend910B vector arithmetic requires one explicit tensor dtype"
    if op.kind in {"sum", "max"}:
        return None
    output_geometry = _solver_geometry(output)
    if output_geometry is None:
        return None
    input_geometries = [_solver_geometry(values[value_id]) for value_id in op.inputs]
    if any(geometry is None for geometry in input_geometries):
        return None
    if any(
        not _supported_broadcast(geometry, output_geometry)
        for geometry in input_geometries
        if geometry is not None
    ):
        return (
            "vector broadcasting must be equal, scalar, row-singleton, or column-singleton in solver geometry"
        )
    return None


def _solver_geometry(value: NormalizedValue) -> tuple[int, int] | None:
    dims: list[int] = []
    for dim in value.shape:
        if not isinstance(dim, int):
            return None
        dims.append(dim)
    if not dims:
        return (1, 1)
    if len(dims) == 1:
        return (dims[0], 1)
    height = 1
    for dim in dims[:-1]:
        height *= dim
    return (dims[-1], height)


def _has_supported_dense_layout(value: NormalizedValue) -> bool:
    if any(not isinstance(dim, int) for dim in value.shape):
        return True
    if value.strides is None:
        return False
    expected = _dense_strides(value.shape)
    return expected is not None and value.strides == expected


def _dense_strides(shape: tuple[object, ...]) -> tuple[int, ...] | None:
    if any(not isinstance(dim, int) for dim in shape):
        return None
    stride = 1
    result: list[int] = []
    for dim in reversed(shape):
        if not isinstance(dim, int):
            return None
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


def _supported_broadcast(input_geometry: tuple[int, int], output_geometry: tuple[int, int]) -> bool:
    width, height = input_geometry
    output_width, output_height = output_geometry
    return (
        input_geometry in (output_geometry, (1, 1))
        or (width == 1 and height == output_height)
        or (height == 1 and width == output_width)
    )


def _is_explicit_transpose_alias(
    value: NormalizedValue,
    op: NormalizedOp,
    values: Mapping[str, NormalizedValue],
) -> bool:
    if value.role != "alias" or value.id not in op.inputs or value.alias_of is None:
        return False
    source = values[value.alias_of]
    return (
        op.kind == "matmul"
        and len(value.shape) == 2
        and value.shape == tuple(reversed(source.shape))
        and value.strides == tuple(reversed(source.strides or ()))
        and _has_supported_dense_layout(source)
    )
