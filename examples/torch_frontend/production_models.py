"""Emit the first maximal static regions for four production PyPTO-lib models.

Model names select native wiring only. Every tensor graph still goes through
the ordinary Torch export, normalization, Fusebox solve, and generic PyPTO DSL
emitter; this module contains no scheduling recognizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pto_fusebox import (
    EmittedPyPTOCallable,
    NormalizedGraph,
    SolveResult,
    SourceEmissionError,
    emit_deepseek_mtp_projection_overlay,
    emit_pypto_callable,
    emit_qwen_output_head_overlay,
    export_and_normalize,
    pypto_lib_model_manifest,
    solve_graph,
    validate_pypto_lib_model,
)

from ._runner import Example
from .deepseek_v4 import (
    DEEPSEEK_V4_FLASH_MTP_GEOMETRY,
    DEEPSEEK_V4_PRO_MTP_GEOMETRY,
    DeepSeekV4MtpGeometry,
    build_production_dspark_projection,
    build_production_mtp_decode_projection,
)
from .qwen3 import build_production_qwen_output_head


@dataclass(frozen=True)
class GeneratedModelSource:
    """One generated or minimally patched source file."""

    relative_path: str
    source: str


@dataclass(frozen=True)
class ProductionModelIntegration:
    """Generated source ownership for one production PyPTO-lib model."""

    model_name: str
    implemented_static_regions: tuple[str, ...]
    generated_sources: tuple[GeneratedModelSource, ...]
    patched_native_sources: tuple[GeneratedModelSource, ...]
    callables: tuple[EmittedPyPTOCallable, ...]

    def files(self) -> dict[str, str]:
        """Return generated and patched source files by relative path."""

        files = {
            source.relative_path: source.source
            for source in (*self.generated_sources, *self.patched_native_sources)
        }
        expected = len(self.generated_sources) + len(self.patched_native_sources)
        if len(files) != expected:
            raise SourceEmissionError(
                f"model integration {self.model_name!r} repeats a source path"
            )
        return files


def emit_production_model_integration(
    model_name: str,
    pypto_lib_root: str | Path,
    solver_binary: str | Path,
    *,
    solver_workers: int = 2,
) -> ProductionModelIntegration:
    """Emit the implemented maximal static regions for one production model."""

    manifest = pypto_lib_model_manifest(model_name)
    validated = validate_pypto_lib_model(pypto_lib_root, manifest)
    if model_name == "deepseek_v4_flash_dspark":
        return _emit_dspark(
            model_name,
            solver_binary,
            solver_workers=solver_workers,
        )
    if model_name in {"deepseek_v4_flash_mtp", "deepseek_v4_pro"}:
        return _emit_mtp(
            model_name,
            validated.model_dir,
            solver_binary,
            solver_workers=solver_workers,
        )
    if model_name == "qwen3_14b":
        return _emit_qwen(
            model_name,
            validated.model_dir,
            solver_binary,
            solver_workers=solver_workers,
        )
    raise AssertionError(f"manifest has no production integration: {model_name}")


def _emit_dspark(
    model_name: str,
    solver_binary: str | Path,
    *,
    solver_workers: int,
) -> ProductionModelIntegration:
    graph, solved = _solve_complete_graph(
        build_production_dspark_projection(),
        solver_binary,
        solver_workers=solver_workers,
        label="DSpark projection",
    )
    emitted = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name="fusebox_dspark_projection_decode",
        expose_completion_task=True,
    )
    return ProductionModelIntegration(
        model_name=model_name,
        implemented_static_regions=("dspark_projection",),
        generated_sources=(
            GeneratedModelSource("fusebox_dspark_projection.py", emitted.source),
        ),
        patched_native_sources=(),
        callables=(emitted,),
    )


def _emit_mtp(
    model_name: str,
    model_dir: Path,
    solver_binary: str | Path,
    *,
    solver_workers: int,
) -> ProductionModelIntegration:
    geometry_by_model: dict[str, DeepSeekV4MtpGeometry] = {
        "deepseek_v4_flash_mtp": DEEPSEEK_V4_FLASH_MTP_GEOMETRY,
        "deepseek_v4_pro": DEEPSEEK_V4_PRO_MTP_GEOMETRY,
    }
    try:
        geometry = geometry_by_model[model_name]
    except KeyError as error:
        raise SourceEmissionError(
            f"DeepSeek MTP integration has no static geometry for {model_name!r}"
        ) from error
    graph, solved = _solve_complete_projection(
        solver_binary,
        geometry=geometry,
        solver_workers=solver_workers,
    )
    module_name = f"fusebox_{model_name}_mtp_projection"
    overlay = emit_deepseek_mtp_projection_overlay(
        graph,
        solved,
        native_source=(model_dir / "decode_mtp.py").read_text(encoding="utf-8"),
        module_name=module_name,
    )
    return ProductionModelIntegration(
        model_name=model_name,
        implemented_static_regions=("mtp_projection",),
        generated_sources=(GeneratedModelSource(f"{module_name}.py", overlay.source),),
        patched_native_sources=(
            GeneratedModelSource("decode_mtp.py", overlay.decode_source),
        ),
        callables=overlay.static_callables,
    )


def _emit_qwen(
    model_name: str,
    model_dir: Path,
    solver_binary: str | Path,
    *,
    solver_workers: int,
) -> ProductionModelIntegration:
    graph, solved = _solve_complete_graph(
        build_production_qwen_output_head(),
        solver_binary,
        solver_workers=solver_workers,
        label="Qwen output head",
    )
    module_name = "fusebox_qwen_output_head"
    overlay = emit_qwen_output_head_overlay(
        graph,
        solved,
        native_decode_source=(model_dir / "decode_fwd.py").read_text(encoding="utf-8"),
        module_name=module_name,
    )
    return ProductionModelIntegration(
        model_name=model_name,
        implemented_static_regions=("output_head",),
        generated_sources=(GeneratedModelSource(f"{module_name}.py", overlay.source),),
        patched_native_sources=(
            GeneratedModelSource("decode_fwd.py", overlay.decode_source),
        ),
        callables=(overlay.output_head,),
    )


def _solve_complete_projection(
    solver_binary: str | Path,
    *,
    geometry: DeepSeekV4MtpGeometry,
    solver_workers: int,
) -> tuple[NormalizedGraph, SolveResult]:
    """Solve the full INT8 projection graph, retaining native shape boundaries."""

    module, args = build_production_mtp_decode_projection(geometry)
    graph = export_and_normalize(module.eval(), args)
    solved = solve_graph(
        graph,
        solver_binary=solver_binary,
        solver_workers=solver_workers,
        require_source_codegen=True,
    )
    if not solved.successful or not solved.regions_solved or len(solved.regions) != 2:
        raise SourceEmissionError(
            "DeepSeek MTP projection must expose two solved maximal regions"
        )
    return graph, solved


def _solve_complete_graph(
    example: Example,
    solver_binary: str | Path,
    *,
    solver_workers: int,
    label: str,
) -> tuple[NormalizedGraph, SolveResult]:
    """Solve one fully static Torch graph without caller-selected cuts."""

    module, args = example
    graph = export_and_normalize(module.eval(), args)
    solved = solve_graph(
        graph,
        solver_binary=solver_binary,
        solver_workers=solver_workers,
        require_source_codegen=True,
    )
    if not solved.whole_graph_codegen_ready or len(solved.regions) != 1:
        raise SourceEmissionError(f"{label} must solve as one source-ready region")
    return graph, solved
