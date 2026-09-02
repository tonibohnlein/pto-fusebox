"""Adapters that place generated static callables inside native PyPTO-lib code.

Fusebox does not recreate PyPTO orchestration. These adapters consume one
normalized graph and its complete solve result, then patch one import in a
copy of the native entry point. Dynamic control flow, communication, paged
access and routing remain owned by PyPTO-lib.
"""

from __future__ import annotations

import ast
import keyword
from dataclasses import dataclass

from .ir import NormalizedGraph, NormalizedOp
from .solver import SolveResult
from .source import (
    EmittedPyPTOCallable,
    SourceEmissionError,
    emit_pypto_static_bundle,
)


@dataclass(frozen=True)
class FlashMtpProjectionOverlay:
    """Generated projection module plus the minimally patched decode source."""

    module_name: str
    source: str
    decode_source: str
    static_callables: tuple[EmittedPyPTOCallable, ...]
    native_op_ids: tuple[str, ...]


def emit_flash_mtp_decode_projection_overlay(
    graph: NormalizedGraph,
    result: SolveResult,
    *,
    native_decode_source: str,
    module_name: str = "fusebox_mtp_projection",
) -> FlashMtpProjectionOverlay:
    """Emit production INT8 projection and wire it into ``decode_mtp``.

    ``graph`` is the complete static projection DAG, not two caller-selected
    branch graphs. Fusebox extracts and solves its maximal supported regions;
    this adapter only realizes the remaining native shape operations and
    patches the decode import. No MTP operator or recognizer is introduced.
    """

    if not module_name or any(
        not part.isidentifier() or keyword.iskeyword(part)
        for part in module_name.split(".")
    ):
        raise SourceEmissionError(
            f"Flash-MTP overlay module name is invalid: {module_name!r}"
        )

    bundle = emit_pypto_static_bundle(
        graph,
        result,
        function_prefix="fusebox_mtp",
    )
    hidden, history = _projection_callables(graph, bundle.callables)
    _validate_projection_callable(
        graph,
        hidden,
        value_name="hidden_padded",
        rows=16,
        value_dtype="bfloat16",
    )
    _validate_projection_callable(
        graph,
        history,
        value_name="history_flat",
        rows=32,
        value_dtype="float32",
    )
    _validate_native_projection_ops(
        graph,
        bundle.native_op_ids,
        hidden=hidden,
        history=history,
    )

    hidden_bindings = _projection_bindings(
        graph,
        hidden,
        semantic_bindings={
            "hidden_padded": "hidden_padded",
            "enorm_weight": "enorm_2d",
            "e_smooth": "e_smooth_2d",
            "e_projection_weight": "e_proj_w",
            "e_projection_scale": "e_scale_2d",
        },
        output="hidden_projected",
    )
    history_bindings = _projection_bindings(
        graph,
        history,
        semantic_bindings={
            "history_flat": "history_flat",
            "hnorm_weight": "hnorm_2d",
            "h_smooth": "h_smooth_2d",
            "h_projection_weight": "h_proj_w",
            "h_projection_scale": "h_scale_2d",
        },
        output="history_projected",
    )

    functions = [
        _callable_function(hidden),
        _callable_function(history),
        ast.parse(
            _flash_mtp_wrapper_source(
                hidden.function_name,
                hidden_bindings,
                history.function_name,
                history_bindings,
            )
        ).body[0],
    ]
    module = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="pypto.language", asname="pl")]),
            *functions,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    source = ast.unparse(module) + "\n"
    if "auto_tile" in source or "auto_fuse" in source:
        raise SourceEmissionError(
            "Flash-MTP overlay must encode static schedules directly"
        )
    decode_source = _patch_decode_import(native_decode_source, module_name)
    return FlashMtpProjectionOverlay(
        module_name=module_name,
        source=source,
        decode_source=decode_source,
        static_callables=bundle.callables,
        native_op_ids=bundle.native_op_ids,
    )


def _projection_callables(
    graph: NormalizedGraph,
    callables: tuple[EmittedPyPTOCallable, ...],
) -> tuple[EmittedPyPTOCallable, EmittedPyPTOCallable]:
    """Identify the solver-selected regions by stable input lineage."""

    values = graph.value_map()
    by_value_input: dict[str, EmittedPyPTOCallable] = {}
    for emitted in callables:
        input_names = {values[value_id].name for value_id in emitted.input_value_ids}
        for name in ("hidden_padded", "history_flat"):
            if name not in input_names:
                continue
            if name in by_value_input:
                raise SourceEmissionError(
                    f"production projection has multiple regions consuming {name!r}"
                )
            by_value_input[name] = emitted
    if set(by_value_input) != {"hidden_padded", "history_flat"}:
        raise SourceEmissionError(
            "complete production projection must expose exactly one maximal "
            "hidden region and one maximal history region"
        )
    if len(callables) != 2:
        raise SourceEmissionError(
            f"complete production projection emitted {len(callables)} static "
            "regions; expected 2"
        )
    return by_value_input["hidden_padded"], by_value_input["history_flat"]


def _validate_projection_callable(
    graph: NormalizedGraph,
    emitted: EmittedPyPTOCallable,
    *,
    value_name: str,
    rows: int,
    value_dtype: str,
) -> None:
    values = graph.value_map()
    input_values = [values[value_id] for value_id in emitted.input_value_ids]
    prefix = "e" if value_name == "hidden_padded" else "h"
    norm_name = "enorm_weight" if prefix == "e" else "hnorm_weight"
    expected_inputs = {
        value_name: ((rows, 4096), value_dtype),
        norm_name: ((1, 4096), "float32"),
        f"{prefix}_smooth": ((1, 4096), "float32"),
        f"{prefix}_projection_weight": ((4096, 4096), "int8"),
        f"{prefix}_projection_scale": ((1, 4096), "float32"),
    }
    if len(input_values) != 5 or len({value.name for value in input_values}) != 5:
        raise SourceEmissionError(
            f"production projection region for {value_name!r} requires five "
            "uniquely named inputs"
        )
    actual_inputs = {
        value.name: (tuple(value.shape), value.dtype) for value in input_values
    }
    if actual_inputs != expected_inputs:
        raise SourceEmissionError(
            f"production projection inputs for {value_name!r} are stale: "
            f"{actual_inputs!r}"
        )
    value_inputs = [value for value in input_values if value.name == value_name]
    if len(value_inputs) != 1:
        raise SourceEmissionError(
            f"production projection region requires input {value_name!r}"
        )
    value = value_inputs[0]
    if tuple(value.shape) != (rows, 4096) or value.dtype != value_dtype:
        raise SourceEmissionError(
            f"production projection input {value_name!r} is stale: "
            f"shape={value.shape!r}, dtype={value.dtype!r}"
        )
    if len(emitted.output_value_ids) != 1:
        raise SourceEmissionError("production projection region requires one output")
    output = values[emitted.output_value_ids[0]]
    output_shape = tuple(output.shape)
    if output_shape != (rows, 4096):
        raise SourceEmissionError(
            f"production projection output shape is stale: {output_shape!r}"
        )
    if output.dtype != "float32":
        raise SourceEmissionError("production projection output must remain FP32")


def _validate_native_projection_ops(
    graph: NormalizedGraph,
    native_op_ids: tuple[str, ...],
    *,
    hidden: EmittedPyPTOCallable,
    history: EmittedPyPTOCallable,
) -> None:
    """Prove that the hand-written native tail matches the normalized DAG."""

    operations = graph.op_map()
    values = graph.value_map()
    native = [operations[op_id] for op_id in native_op_ids]
    if [op.kind for op in native] != ["opaque", "view", "opaque", "add"]:
        raise SourceEmissionError(
            "production projection native boundary must contain only the "
            "hidden prefix, metadata view, history reshape, and grouped add"
        )
    reasons = tuple(op.opaque_reason for op in native if not op.supported)
    expected_reasons = len(reasons) == 2 and all(
        reason is not None for reason in reasons
    )
    if expected_reasons:
        expected_reasons = "aten.slice.Tensor" in (
            reasons[0] or ""
        ) and "aten.reshape.default" in (reasons[1] or "")
    if not expected_reasons:
        raise SourceEmissionError(
            f"production projection native shape boundaries are stale: {reasons!r}"
        )

    hidden_output = _single_callable_output(hidden, "hidden")
    history_output = _single_callable_output(history, "history")
    hidden_slice, hidden_view, history_reshape, grouped_add = native
    if hidden_slice.inputs != (hidden_output,) or history_reshape.inputs != (
        history_output,
    ):
        raise SourceEmissionError(
            "production projection native tail does not consume the two "
            "solver-selected branch outputs"
        )
    if hidden_slice.attributes.get("literal_args") != [
        {"position": 1, "value": 0},
        {"position": 2, "value": 0},
        {"position": 3, "value": 8},
    ]:
        raise SourceEmissionError(
            "production projection hidden slice must select rows [0:8] on axis 0"
        )
    if history_reshape.attributes.get("literal_args") != [
        {"position": 1, "value": [8, 4, 4096]},
    ]:
        raise SourceEmissionError(
            "production projection history reshape must produce [8, 4, 4096]"
        )
    slice_output = _single_op_output(hidden_slice, "hidden slice")
    view_output = _single_op_output(hidden_view, "hidden singleton view")
    reshape_output = _single_op_output(history_reshape, "history reshape")
    add_output = _single_op_output(grouped_add, "grouped add")
    if hidden_view.inputs != (slice_output,) or grouped_add.inputs != (
        view_output,
        reshape_output,
    ):
        raise SourceEmissionError(
            "production projection native tail value lineage is stale"
        )
    expected_values = {
        slice_output: ((8, 4096), "float32"),
        view_output: ((8, 1, 4096), "float32"),
        reshape_output: ((8, 4, 4096), "float32"),
        add_output: ((8, 4, 4096), "float32"),
    }
    actual_values = {
        value_id: (tuple(values[value_id].shape), values[value_id].dtype)
        for value_id in expected_values
    }
    if actual_values != expected_values:
        raise SourceEmissionError(
            f"production projection native tail shapes or dtypes are stale: "
            f"{actual_values!r}"
        )
    if values[view_output].alias_of != slice_output:
        raise SourceEmissionError(
            "production projection hidden singleton view must alias the slice"
        )
    if graph.outputs != (add_output,):
        raise SourceEmissionError(
            "production projection grouped add must be the sole graph output"
        )


def _single_callable_output(
    emitted: EmittedPyPTOCallable,
    label: str,
) -> str:
    if len(emitted.output_value_ids) != 1:
        raise SourceEmissionError(
            f"production projection {label} callable requires one output"
        )
    return emitted.output_value_ids[0]


def _single_op_output(op: NormalizedOp, label: str) -> str:
    if len(op.outputs) != 1:
        raise SourceEmissionError(
            f"production projection {label} requires one normalized output"
        )
    return op.outputs[0]


def _projection_bindings(
    graph: NormalizedGraph,
    emitted: EmittedPyPTOCallable,
    *,
    semantic_bindings: dict[str, str],
    output: str,
) -> tuple[str, ...]:
    values = graph.value_map()
    try:
        arguments = [
            semantic_bindings[values[argument.value_id].name]
            for argument in emitted.input_arguments
        ]
    except KeyError as error:
        raise SourceEmissionError(
            f"production projection binding is missing semantic input {error.args[0]!r}"
        ) from error
    if len(emitted.output_arguments) != 1:
        raise SourceEmissionError("production projection binding requires one output")
    arguments.extend(output for _argument in emitted.output_arguments)
    return tuple(arguments)


def _callable_function(emitted: EmittedPyPTOCallable) -> ast.FunctionDef:
    tree = ast.parse(emitted.source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise SourceEmissionError("generated callable module must define one function")
    return functions[0]


def _flash_mtp_wrapper_source(
    hidden_name: str,
    hidden_arguments: tuple[str, ...],
    history_name: str,
    history_arguments: tuple[str, ...],
) -> str:
    hidden_call = ", ".join(hidden_arguments)
    history_call = ", ".join(history_arguments)
    return f"""@pl.inline
def mtp_projection(
    hidden_states: pl.Tensor[[8, 4096], pl.BF16],
    prev_hidden_states: pl.Tensor[[8, 4, 4096], pl.FP32],
    enorm_w: pl.Tensor[[4096], pl.FP32],
    hnorm_w: pl.Tensor[[4096], pl.FP32],
    e_proj_w: pl.Tensor[[4096, 4096], pl.INT8],
    e_proj_w_scale: pl.Tensor[[4096], pl.FP32],
    e_proj_smooth: pl.Tensor[[4096], pl.FP32],
    h_proj_w: pl.Tensor[[4096, 4096], pl.INT8],
    h_proj_w_scale: pl.Tensor[[4096], pl.FP32],
    h_proj_smooth: pl.Tensor[[4096], pl.FP32],
    hidden_states_out: pl.Tensor[[8, 4, 4096], pl.FP32],
):
    hidden_padded = pl.create_tensor([16, 4096], dtype=pl.BF16)
    for hidden_row in pl.spmd(8, name_hint="fusebox_mtp_hidden_copy"):
        hidden_copy_tile = hidden_states[hidden_row:hidden_row + 1, 0:4096]
        hidden_padded[hidden_row:hidden_row + 1, 0:4096] = hidden_copy_tile
    for hidden_zero_row in pl.spmd(8, name_hint="fusebox_mtp_hidden_zero"):
        hidden_zero_tile = pl.full([1, 4096], dtype=pl.BF16, value=0.0)
        hidden_padded[hidden_zero_row + 8:hidden_zero_row + 9, 0:4096] = hidden_zero_tile

    history_flat = pl.reshape(prev_hidden_states, [32, 4096])
    enorm_2d = pl.reshape(enorm_w, [1, 4096])
    hnorm_2d = pl.reshape(hnorm_w, [1, 4096])
    e_smooth_2d = pl.reshape(e_proj_smooth, [1, 4096])
    h_smooth_2d = pl.reshape(h_proj_smooth, [1, 4096])
    e_scale_2d = pl.reshape(e_proj_w_scale, [1, 4096])
    h_scale_2d = pl.reshape(h_proj_w_scale, [1, 4096])
    hidden_projected = pl.create_tensor([16, 4096], dtype=pl.FP32)
    history_projected = pl.create_tensor([32, 4096], dtype=pl.FP32)
    hidden_projected = {hidden_name}({hidden_call})
    history_projected = {history_name}({history_call})

    output_flat = pl.reshape(hidden_states_out, [8, 16384])
    for combine_index in pl.spmd(128, name_hint="fusebox_mtp_combine"):
        history_row = combine_index // 4
        output_chunk = combine_index - history_row * 4
        token = history_row // 4
        hyperconnection = history_row - token * 4
        col = output_chunk * 1024
        hidden_projection_tile = hidden_projected[token:token + 1, col:col + 1024]
        history_tile = history_projected[history_row:history_row + 1, col:col + 1024]
        combined = pl.add(hidden_projection_tile, history_tile)
        output_col = hyperconnection * 4096 + col
        output_flat[token:token + 1, output_col:output_col + 1024] = combined
    return pl.reshape(output_flat, [8, 4, 4096])
"""


def _patch_decode_import(source: str, module_name: str) -> str:
    expected = "from mtp_projection import golden_mtp_projection, mtp_projection"
    if source.count(expected) != 1:
        raise SourceEmissionError(
            "native decode source does not contain the expected MTP projection import"
        )
    replacement = (
        "from mtp_projection import golden_mtp_projection\n"
        f"from {module_name} import mtp_projection"
    )
    return source.replace(expected, replacement)
