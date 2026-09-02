"""Generate static Qwen callables for a native PyPTO orchestration.

This example demonstrates the hybrid authoring contract. ``qwen3.py`` holds
ordinary Torch definitions of two static tensor regions. Fusebox replaces
those definitions with scheduled PyPTO callables, while the companion
``hybrid_qwen_orchestration.py.in`` remains hand-authored PyPTO orchestration.
Torch is not imported or executed by the generated program.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

from pto_fusebox import (
    EmittedPyPTOCallable,
    NormalizedGraph,
    SourceEmissionError,
    emit_pypto_callable,
    export_and_normalize,
    solve_graph,
)

from ._runner import Example
from .qwen3 import build_examples

_RMS_MODULE = "generated_qwen_rms_norm"
_RMS_FUNCTION = "generated_qwen_rms_norm"
_LM_HEAD_MODULE = "generated_qwen_lm_head"
_LM_HEAD_FUNCTION = "generated_qwen_lm_head"
_ORCHESTRATION_FILE = "native_qwen_output_head.py"
_TEMPLATE = Path(__file__).with_name("hybrid_qwen_orchestration.py.in")


@dataclass(frozen=True)
class HybridQwenOutputHeadSources:
    """The two generated modules and their native orchestration caller."""

    rms_norm: EmittedPyPTOCallable
    lm_head: EmittedPyPTOCallable
    orchestration_source: str

    def files(self) -> dict[str, str]:
        """Return the complete source tree keyed by output filename."""

        return {
            f"{_RMS_MODULE}.py": self.rms_norm.source,
            f"{_LM_HEAD_MODULE}.py": self.lm_head.source,
            _ORCHESTRATION_FILE: self.orchestration_source,
        }


def emit_hybrid_qwen_output_head(
    solver_binary: str | Path,
    *,
    solver_workers: int = 2,
) -> HybridQwenOutputHeadSources:
    """Solve the Torch regions and link them into native PyPTO source."""

    examples = build_examples()
    rms_graph, rms_norm = _emit_static_callable(
        examples["qwen3_rms_norm_chunk"],
        solver_binary,
        solver_workers=solver_workers,
        function_name=_RMS_FUNCTION,
    )
    lm_graph, lm_head = _emit_static_callable(
        examples["qwen3_lm_head_chunk"],
        solver_binary,
        solver_workers=solver_workers,
        function_name=_LM_HEAD_FUNCTION,
    )
    rms_arguments = _ordered_call_arguments(
        rms_graph,
        rms_norm,
        {
            "hidden_states": "hidden_states",
            "norm_weight": "norm_weight",
        },
        output="normalized",
    )
    lm_arguments = _ordered_call_arguments(
        lm_graph,
        lm_head,
        {
            "normalized": "normalized",
            "lm_head_weight": "lm_head_weight",
        },
        output="output",
    )
    orchestration_source = _render_orchestration(
        rms_arguments=rms_arguments,
        lm_arguments=lm_arguments,
    )
    return HybridQwenOutputHeadSources(
        rms_norm=rms_norm,
        lm_head=lm_head,
        orchestration_source=orchestration_source,
    )


def _emit_static_callable(
    example: Example,
    solver_binary: str | Path,
    *,
    solver_workers: int,
    function_name: str,
) -> tuple[NormalizedGraph, EmittedPyPTOCallable]:
    module, example_args = example
    graph = export_and_normalize(module.eval(), example_args)
    solved = solve_graph(
        graph,
        solver_binary=solver_binary,
        solver_workers=solver_workers,
        require_source_codegen=True,
    )
    if not solved.successful or not solved.whole_graph_codegen_ready:
        diagnostics = [*solved.graph_diagnostics]
        diagnostics.extend(
            diagnostic for region in solved.regions for diagnostic in region.diagnostics
        )
        raise SourceEmissionError(
            f"static region {function_name!r} is not source-ready: "
            f"{tuple(diagnostics)!r}"
        )
    if len(solved.regions) != 1:
        raise SourceEmissionError(
            f"static region {function_name!r} must solve as one callable"
        )
    return graph, emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name=function_name,
    )


def _ordered_call_arguments(
    graph: NormalizedGraph,
    emitted: EmittedPyPTOCallable,
    semantic_bindings: dict[str, str],
    *,
    output: str,
) -> str:
    """Bind native names through the emitted normalized-value ABI."""

    values = graph.value_map()
    try:
        arguments = [
            semantic_bindings[
                values[argument.value_id].target or values[argument.value_id].name
            ]
            for argument in emitted.input_arguments
        ]
    except KeyError as error:
        raise SourceEmissionError(
            f"native Qwen binding is missing semantic input {error.args[0]!r}"
        ) from error
    if len(emitted.output_arguments) != 1:
        raise SourceEmissionError("native Qwen callable requires one output")
    return ", ".join([*arguments, output])


def _render_orchestration(*, rms_arguments: str, lm_arguments: str) -> str:
    source = _TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__RMS_MODULE__": _RMS_MODULE,
        "__RMS_FUNCTION__": _RMS_FUNCTION,
        "__RMS_ARGUMENTS__": rms_arguments,
        "__LM_HEAD_MODULE__": _LM_HEAD_MODULE,
        "__LM_HEAD_FUNCTION__": _LM_HEAD_FUNCTION,
        "__LM_HEAD_ARGUMENTS__": lm_arguments,
    }
    for placeholder, replacement in replacements.items():
        if placeholder not in source:
            raise SourceEmissionError(
                f"native Qwen orchestration omits placeholder {placeholder}"
            )
        source = source.replace(placeholder, replacement)
    if "__" in source:
        raise SourceEmissionError("native Qwen orchestration has stale placeholders")
    ast.parse(source)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solver-workers", type=int, default=2)
    args = parser.parse_args()

    sources = emit_hybrid_qwen_output_head(
        args.solver,
        solver_workers=args.solver_workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sources.files().items():
        destination = args.output_dir / name
        destination.write_text(source, encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
