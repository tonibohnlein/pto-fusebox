"""Adapters that place generated static callables inside native PyPTO-lib code.

Fusebox does not recreate PyPTO orchestration. These adapters compose already
solved static callables and patch one import in a copy of the native entry
point. Dynamic control flow, communication, paged access and routing remain
owned by PyPTO-lib.
"""

from __future__ import annotations

import ast
import keyword
from dataclasses import dataclass

from .ir import NormalizedGraph
from .solver import RegionSolveResult
from .source import EmittedPyPTOCallable, SourceEmissionError, emit_pypto_callable


@dataclass(frozen=True)
class FlashMtpProjectionOverlay:
    """Generated projection module plus the minimally patched decode source."""

    module_name: str
    source: str
    decode_source: str
    hidden_callable: EmittedPyPTOCallable
    history_callable: EmittedPyPTOCallable


def emit_flash_mtp_decode_projection_overlay(
    hidden_graph: NormalizedGraph,
    hidden_result: RegionSolveResult,
    history_graph: NormalizedGraph,
    history_result: RegionSolveResult,
    *,
    native_decode_source: str,
    module_name: str = "fusebox_mtp_projection",
) -> FlashMtpProjectionOverlay:
    """Emit production INT8 projection and wire it into ``decode_mtp``.

    The two branches are ordinary normalized RMSNorm/quantize/INT8-matmul/
    dequant DAGs. This function adds only the native MTP composition: it pads
    the eight-token hidden branch to its sixteen-row physical frame, flattens
    the 8×4 history, and adds the shared hidden projection to each history
    projection. No MTP operator or schedule recognizer is introduced.
    """

    if not module_name or any(
        not part.isidentifier() or keyword.iskeyword(part)
        for part in module_name.split(".")
    ):
        raise SourceEmissionError(
            f"Flash-MTP overlay module name is invalid: {module_name!r}"
        )

    hidden = emit_pypto_callable(
        hidden_graph,
        hidden_result,
        function_name="_fusebox_hidden_projection",
    )
    history = emit_pypto_callable(
        history_graph,
        history_result,
        function_name="_fusebox_history_projection",
    )
    _validate_projection_callable(hidden_graph, hidden, rows=16, value_dtype="bfloat16")
    _validate_projection_callable(
        history_graph, history, rows=32, value_dtype="float32"
    )

    hidden_bindings = _projection_bindings(
        hidden_graph,
        hidden,
        value="hidden_padded",
        norm="enorm_2d",
        smooth="e_smooth_2d",
        weight="e_proj_w",
        scale="e_scale_2d",
        output="hidden_projected",
    )
    history_bindings = _projection_bindings(
        history_graph,
        history,
        value="history_flat",
        norm="hnorm_2d",
        smooth="h_smooth_2d",
        weight="h_proj_w",
        scale="h_scale_2d",
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
        hidden_callable=hidden,
        history_callable=history,
    )


def _validate_projection_callable(
    graph: NormalizedGraph,
    emitted: EmittedPyPTOCallable,
    *,
    rows: int,
    value_dtype: str,
) -> None:
    values = graph.value_map()
    if len(graph.inputs) != 5 or len(graph.outputs) != 1:
        raise SourceEmissionError(
            "production projection branch requires five inputs and one output"
        )
    expected_names = {
        "value",
        "norm_weight",
        "smooth",
        "projection_weight",
        "projection_scale",
    }
    actual_names = {values[value].name for value in graph.inputs}
    if actual_names != expected_names:
        raise SourceEmissionError(
            f"production projection input names are stale: {actual_names!r}"
        )
    expected_shapes = (
        (rows, 4096),
        (1, 4096),
        (1, 4096),
        (4096, 4096),
        (1, 4096),
    )
    actual_shapes = tuple(tuple(values[value].shape) for value in graph.inputs)
    if actual_shapes != expected_shapes:
        raise SourceEmissionError(
            f"production projection branch shapes are stale: {actual_shapes!r}"
        )
    expected_dtypes = (
        value_dtype,
        "float32",
        "float32",
        "int8",
        "float32",
    )
    actual_dtypes = tuple(values[value].dtype for value in graph.inputs)
    if actual_dtypes != expected_dtypes:
        raise SourceEmissionError(
            f"production projection branch dtypes are stale: {actual_dtypes!r}"
        )
    output_shape = tuple(values[graph.outputs[0]].shape)
    if output_shape != (rows, 4096):
        raise SourceEmissionError(
            f"production projection output shape is stale: {output_shape!r}"
        )
    if values[graph.outputs[0]].dtype != "float32":
        raise SourceEmissionError("production projection output must remain FP32")
    if set(emitted.input_value_ids) != set(graph.inputs):
        raise SourceEmissionError("projection callable input lineage is incomplete")
    if emitted.output_value_ids != graph.outputs:
        raise SourceEmissionError("projection callable output lineage is stale")


def _projection_bindings(
    graph: NormalizedGraph,
    emitted: EmittedPyPTOCallable,
    *,
    value: str,
    norm: str,
    smooth: str,
    weight: str,
    scale: str,
    output: str,
) -> tuple[str, ...]:
    values = graph.value_map()
    semantic = {
        "value": value,
        "norm_weight": norm,
        "smooth": smooth,
        "projection_weight": weight,
        "projection_scale": scale,
    }
    arguments = [
        semantic[values[argument.value_id].name] for argument in emitted.input_arguments
    ]
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
