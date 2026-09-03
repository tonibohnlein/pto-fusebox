"""Generate static Qwen callables for a native PyPTO orchestration.

This example demonstrates the hybrid authoring contract. ``qwen3.py`` holds
one ordinary Torch definition of the complete static output-head DAG. Fusebox
chooses fusion and cut boundaries for that DAG and emits one scheduled PyPTO
callable, while the companion
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
    QwenOutputHeadOverlay,
    SolveResult,
    SourceEmissionError,
    emit_pypto_callable,
    emit_qwen_output_head_overlay,
    export_and_normalize,
    solve_graph,
)

from ._runner import Example
from .qwen3 import build_examples, build_production_qwen_output_head

_OUTPUT_HEAD_MODULE = "generated_qwen_output_head"
_OUTPUT_HEAD_FUNCTION = "generated_qwen_output_head"
_ORCHESTRATION_FILE = "native_qwen_output_head.py"
_TEMPLATE = Path(__file__).with_name("hybrid_qwen_orchestration.py.in")


@dataclass(frozen=True)
class HybridQwenOutputHeadSources:
    """The generated maximal static callable and its native caller."""

    output_head: EmittedPyPTOCallable
    orchestration_source: str

    def files(self) -> dict[str, str]:
        """Return the complete source tree keyed by output filename."""

        return {
            f"{_OUTPUT_HEAD_MODULE}.py": self.output_head.source,
            _ORCHESTRATION_FILE: self.orchestration_source,
        }


def emit_hybrid_qwen_output_head(
    solver_binary: str | Path,
    *,
    solver_workers: int = 2,
) -> HybridQwenOutputHeadSources:
    """Solve the complete static Torch DAG and link it into native PyPTO."""

    examples = build_examples()
    graph, output_head = _emit_static_callable(
        examples["qwen3_rms_lm_head"],
        solver_binary,
        solver_workers=solver_workers,
        function_name=_OUTPUT_HEAD_FUNCTION,
    )
    output_head_arguments = _ordered_call_arguments(
        graph,
        output_head,
        {
            "hidden_states": "hidden_states",
            "norm_weight": "norm_weight",
            "lm_head_weight": "lm_head_weight",
        },
        output="output",
    )
    orchestration_source = _render_orchestration(
        output_head_arguments=output_head_arguments,
    )
    return HybridQwenOutputHeadSources(
        output_head=output_head,
        orchestration_source=orchestration_source,
    )


def emit_production_qwen_output_head_overlay(
    solver_binary: str | Path,
    *,
    native_decode_source: str,
    solver_workers: int = 2,
    module_name: str = "fusebox_qwen_output_head",
) -> QwenOutputHeadOverlay:
    """Replace the real Qwen decode output-head import with one solved DAG."""

    graph, solved = _solve_static_graph(
        build_production_qwen_output_head(),
        solver_binary,
        solver_workers=solver_workers,
        function_name="production_qwen_output_head",
    )
    return emit_qwen_output_head_overlay(
        graph,
        solved,
        native_decode_source=native_decode_source,
        module_name=module_name,
    )


def _emit_static_callable(
    example: Example,
    solver_binary: str | Path,
    *,
    solver_workers: int,
    function_name: str,
) -> tuple[NormalizedGraph, EmittedPyPTOCallable]:
    graph, solved = _solve_static_graph(
        example,
        solver_binary,
        solver_workers=solver_workers,
        function_name=function_name,
    )
    return graph, emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name=function_name,
    )


def _solve_static_graph(
    example: Example,
    solver_binary: str | Path,
    *,
    solver_workers: int,
    function_name: str,
) -> tuple[NormalizedGraph, SolveResult]:
    """Capture and solve one complete static graph without caller partitioning."""

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
    return graph, solved


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


def _render_orchestration(*, output_head_arguments: str) -> str:
    source = _TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__OUTPUT_HEAD_MODULE__": _OUTPUT_HEAD_MODULE,
        "__OUTPUT_HEAD_FUNCTION__": _OUTPUT_HEAD_FUNCTION,
        "__OUTPUT_HEAD_ARGUMENTS__": output_head_arguments,
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
