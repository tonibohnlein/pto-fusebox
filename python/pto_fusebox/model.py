"""Production PyPTO-lib model integration contracts.

Fusebox owns maximal static tensor regions.  The surrounding PyPTO-lib model
continues to own dynamic shapes, communication, indirect access, routing,
sampling, and runtime control flow.  These manifests name both sides of that
boundary against real model entry points so integration cannot silently drift
back to disconnected microbenchmarks.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path


class StaticRegionOwnership(str, Enum):
    """How a native symbol contributes work to one Fusebox integration."""

    WHOLE_CALLABLE = "whole_callable"
    STATIC_SUBREGIONS = "static_subregions"


class NativeBoundaryKind(str, Enum):
    """Operations deliberately retained by native PyPTO orchestration."""

    COMMUNICATION = "communication"
    DYNAMIC_SHAPE = "dynamic_shape"
    INDIRECT_ACCESS = "indirect_access"
    METADATA = "metadata"
    ROUTING = "routing"
    SAMPLING = "sampling"


@dataclass(frozen=True)
class PyPTOLibStaticRegion:
    """One native callable or callable body containing schedulable regions."""

    name: str
    module: str
    symbols: tuple[str, ...]
    ownership: StaticRegionOwnership
    topologies: tuple[str, ...]


@dataclass(frozen=True)
class PyPTOLibNativeBoundary:
    """One named native module that remains outside static Fusebox planning."""

    name: str
    modules: tuple[str, ...]
    kind: NativeBoundaryKind


@dataclass(frozen=True)
class PyPTOLibModelManifest:
    """Exact static/native ownership contract for one production model tree."""

    name: str
    entry_points: tuple[str, ...]
    static_regions: tuple[PyPTOLibStaticRegion, ...]
    native_boundaries: tuple[PyPTOLibNativeBoundary, ...]


@dataclass(frozen=True)
class ValidatedPyPTOLibModel:
    """A model manifest proven against one concrete PyPTO-lib checkout."""

    manifest: PyPTOLibModelManifest
    model_dir: Path
    files: tuple[Path, ...]


def pypto_lib_model_manifests() -> tuple[PyPTOLibModelManifest, ...]:
    """Return the four production integration targets in stable order."""

    deepseek_common = (
        PyPTOLibStaticRegion(
            "rmsnorm",
            "rmsnorm.py",
            ("rms_norm",),
            StaticRegionOwnership.WHOLE_CALLABLE,
            ("vector",),
        ),
        PyPTOLibStaticRegion(
            "qkv_projection",
            "qkv_proj_rope.py",
            ("qkv_proj_rope",),
            StaticRegionOwnership.STATIC_SUBREGIONS,
            ("vector", "cube", "mixed"),
        ),
        PyPTOLibStaticRegion(
            "shared_expert",
            "expert_shared.py",
            ("expert_shared",),
            StaticRegionOwnership.WHOLE_CALLABLE,
            ("vector", "cube", "mixed"),
        ),
        PyPTOLibStaticRegion(
            "routed_expert",
            "expert_routed.py",
            ("expert_routed",),
            StaticRegionOwnership.STATIC_SUBREGIONS,
            ("vector", "cube", "mixed"),
        ),
        PyPTOLibStaticRegion(
            "lm_head",
            "lm_head.py",
            ("lm_head",),
            StaticRegionOwnership.STATIC_SUBREGIONS,
            ("cube", "vector"),
        ),
    )
    flash_common = tuple(
        replace(region, symbols=("expert_routed_tile",))
        if region.name == "routed_expert"
        else region
        for region in deepseek_common
    )
    return (
        PyPTOLibModelManifest(
            name="deepseek_v4_flash_dspark",
            entry_points=("decode_fwd.py", "prefill_fwd.py"),
            static_regions=(
                *flash_common,
                PyPTOLibStaticRegion(
                    "dspark_projection",
                    "dspark_drafter.py",
                    ("dspark_proj",),
                    StaticRegionOwnership.WHOLE_CALLABLE,
                    ("cube", "vector", "mixed"),
                ),
                PyPTOLibStaticRegion(
                    "output_projection",
                    "decode_o_proj.py",
                    ("decode_o_proj_tp1",),
                    StaticRegionOwnership.STATIC_SUBREGIONS,
                    ("cube", "vector"),
                ),
            ),
            native_boundaries=(
                PyPTOLibNativeBoundary(
                    "sparse_attention",
                    (
                        "decode_csa.py",
                        "decode_hca.py",
                        "decode_swa.py",
                        "prefill_sparse_attn.py",
                    ),
                    NativeBoundaryKind.INDIRECT_ACCESS,
                ),
                PyPTOLibNativeBoundary(
                    "moe_routing",
                    ("moe.py", "gate.py"),
                    NativeBoundaryKind.ROUTING,
                ),
                PyPTOLibNativeBoundary(
                    "context_parallel_exchange",
                    (
                        "decode_cp_allgather.py",
                        "prefill_cp_token_allgather.py",
                    ),
                    NativeBoundaryKind.COMMUNICATION,
                ),
                PyPTOLibNativeBoundary(
                    "decode_metadata",
                    ("decode_prepare.py", "prefill_metadata.py"),
                    NativeBoundaryKind.METADATA,
                ),
            ),
        ),
        PyPTOLibModelManifest(
            name="deepseek_v4_flash_mtp",
            entry_points=("decode_fwd_mtp.py", "decode_mtp.py", "prefill_mtp.py"),
            static_regions=(
                PyPTOLibStaticRegion(
                    "mtp_projection",
                    "mtp_projection.py",
                    ("mtp_projection",),
                    StaticRegionOwnership.WHOLE_CALLABLE,
                    ("vector", "cube", "mixed"),
                ),
                *flash_common,
            ),
            native_boundaries=(
                PyPTOLibNativeBoundary(
                    "sparse_attention",
                    (
                        "decode_sparse_attn_csa.py",
                        "decode_sparse_attn_hca.py",
                        "decode_sparse_attn_swa.py",
                        "prefill_sparse_attn.py",
                    ),
                    NativeBoundaryKind.INDIRECT_ACCESS,
                ),
                PyPTOLibNativeBoundary(
                    "moe_routing",
                    ("decode_moe.py", "prefill_moe.py", "gate.py"),
                    NativeBoundaryKind.ROUTING,
                ),
                PyPTOLibNativeBoundary(
                    "distributed_prefill",
                    (
                        "prefill_cp_exchange.py",
                        "prefill_cp_zigzag.py",
                    ),
                    NativeBoundaryKind.COMMUNICATION,
                ),
                PyPTOLibNativeBoundary(
                    "sampling",
                    ("sample.py",),
                    NativeBoundaryKind.SAMPLING,
                ),
            ),
        ),
        PyPTOLibModelManifest(
            name="deepseek_v4_pro",
            entry_points=(
                "decode_fwd.py",
                "decode_mtp.py",
                "prefill_fwd.py",
                "prefill_mtp.py",
            ),
            static_regions=(
                PyPTOLibStaticRegion(
                    "mtp_projection",
                    "mtp_projection.py",
                    ("mtp_projection",),
                    StaticRegionOwnership.WHOLE_CALLABLE,
                    ("vector", "cube", "mixed"),
                ),
                *deepseek_common,
            ),
            native_boundaries=(
                PyPTOLibNativeBoundary(
                    "sparse_attention",
                    ("decode_sparse_attn.py", "prefill_sparse_attn.py"),
                    NativeBoundaryKind.INDIRECT_ACCESS,
                ),
                PyPTOLibNativeBoundary(
                    "moe_routing",
                    ("moe.py", "gate.py"),
                    NativeBoundaryKind.ROUTING,
                ),
                PyPTOLibNativeBoundary(
                    "indexer_topk",
                    ("decode_indexer.py", "prefill_indexer.py"),
                    NativeBoundaryKind.INDIRECT_ACCESS,
                ),
                PyPTOLibNativeBoundary(
                    "dynamic_input_pack",
                    ("input_pack.py",),
                    NativeBoundaryKind.DYNAMIC_SHAPE,
                ),
            ),
        ),
        PyPTOLibModelManifest(
            name="qwen3_14b",
            entry_points=("decode_fwd.py", "prefill_fwd.py"),
            static_regions=(
                PyPTOLibStaticRegion(
                    "output_head",
                    "rms_lm_head.py",
                    ("rms_lm_head", "rms_lm_head_fp32"),
                    StaticRegionOwnership.WHOLE_CALLABLE,
                    ("vector", "cube"),
                ),
                PyPTOLibStaticRegion(
                    "decode_layer",
                    "decode_fwd.py",
                    ("_decode_layer",),
                    StaticRegionOwnership.STATIC_SUBREGIONS,
                    ("vector", "cube", "mixed"),
                ),
                PyPTOLibStaticRegion(
                    "prefill_layer",
                    "prefill_fwd.py",
                    ("prefill_layer",),
                    StaticRegionOwnership.STATIC_SUBREGIONS,
                    ("vector", "cube", "mixed"),
                ),
            ),
            native_boundaries=(
                PyPTOLibNativeBoundary(
                    "paged_attention",
                    ("paged_attention_pypto.py", "paged_attention_cce.py"),
                    NativeBoundaryKind.INDIRECT_ACCESS,
                ),
                PyPTOLibNativeBoundary(
                    "sampling",
                    ("topk_select.py", "greedy_sample.py"),
                    NativeBoundaryKind.SAMPLING,
                ),
                PyPTOLibNativeBoundary(
                    "host_orchestration",
                    ("decode_fwd.py", "prefill_fwd.py"),
                    NativeBoundaryKind.DYNAMIC_SHAPE,
                ),
            ),
        ),
    )


def pypto_lib_model_manifest(name: str) -> PyPTOLibModelManifest:
    """Return one manifest by exact production model directory name."""

    matches = [
        manifest for manifest in pypto_lib_model_manifests() if manifest.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown PyPTO-lib model {name!r}")
    return matches[0]


def validate_pypto_lib_model(
    pypto_lib_root: str | Path,
    manifest: PyPTOLibModelManifest,
) -> ValidatedPyPTOLibModel:
    """Fail closed if a production model tree drifts from its manifest."""

    root = Path(pypto_lib_root).resolve()
    model_dir = root / "models" / manifest.name
    if not model_dir.is_dir():
        raise ValueError(f"PyPTO-lib model directory is missing: {model_dir}")

    required_modules = set(manifest.entry_points)
    required_modules.update(region.module for region in manifest.static_regions)
    required_modules.update(
        module for boundary in manifest.native_boundaries for module in boundary.modules
    )
    files: list[Path] = []
    for relative in sorted(required_modules):
        path = model_dir / relative
        if not path.is_file():
            raise ValueError(
                f"PyPTO-lib model {manifest.name!r} is missing {relative!r}"
            )
        files.append(path)

    for region in manifest.static_regions:
        path = model_dir / region.module
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions = native_callable_definitions(tree)
        missing = set(region.symbols) - definitions.keys()
        if missing:
            raise ValueError(
                f"PyPTO-lib model {manifest.name!r} static region {region.name!r} "
                f"is missing symbols {tuple(sorted(missing))!r}"
            )
    return ValidatedPyPTOLibModel(
        manifest=manifest,
        model_dir=model_dir,
        files=tuple(files),
    )


def native_callable_definitions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Resolve local function bodies and explicit ``pl.jit`` exports, without import.

    Native modules may export a shared body as both ``pl.jit.inline(body)``
    and ``pl.jit(options)(body)``. Arbitrary assignments and external bodies
    are not evidence that a callable exists.
    """

    definitions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            body = None
            call = node.value
            if (
                isinstance(call, ast.Call)
                and len(call.args) == 1
                and isinstance(call.args[0], ast.Name)
            ):
                factory = call.func
                if isinstance(factory, ast.Call) and not factory.args:
                    factory = factory.func
                if ast.unparse(factory) in {"pl.jit", "pl.jit.inline"}:
                    body = definitions.get(call.args[0].id)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    if body is None:
                        definitions.pop(target.id, None)
                    else:
                        definitions[target.id] = body
    return definitions


def native_tensor_layouts(source: str, symbol: str) -> dict[str, str]:
    """Read physical tensor layouts from a native callable's named ABI."""

    function = native_callable_definitions(ast.parse(source)).get(symbol)
    if function is None:
        raise ValueError(f"native callable {symbol!r} has no local function body")
    layouts: dict[str, str] = {}
    for argument in function.args.args:
        annotation = argument.annotation
        if (
            isinstance(annotation, ast.Subscript)
            and ast.unparse(annotation.value) == "pl.Out"
        ):
            annotation = annotation.slice
        if (
            not isinstance(annotation, ast.Subscript)
            or ast.unparse(annotation.value) != "pl.Tensor"
        ):
            continue
        fields = annotation.slice
        if not isinstance(fields, ast.Tuple) or len(fields.elts) not in {2, 3}:
            raise ValueError(
                f"native tensor {argument.arg!r} has an unsupported annotation"
            )
        layout = "ND" if len(fields.elts) == 2 else ast.unparse(fields.elts[2])
        if layout in {"pl.ND", "pl.TensorLayout.ND"}:
            layout = "ND"
        elif layout in {"pl.NZ", "pl.TensorLayout.NZ"}:
            layout = "NZ"
        if layout not in {"ND", "NZ"}:
            raise ValueError(
                f"native tensor {argument.arg!r} has unknown layout {layout!r}"
            )
        layouts[argument.arg] = layout
    return layouts
