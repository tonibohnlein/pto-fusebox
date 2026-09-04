from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from examples.torch_frontend.basic import build_examples as build_basic_examples
from examples.torch_frontend.deepseek_v4 import (
    DEEPSEEK_V4_DECODE_TOKENS,
    DEEPSEEK_V4_DSPARK_TARGET_LAYERS,
    DEEPSEEK_V4_FLASH_MTP_GEOMETRY,
    DEEPSEEK_V4_HC_MULT,
    DEEPSEEK_V4_HIDDEN,
    DEEPSEEK_V4_LINEAR_TOKENS,
    DEEPSEEK_V4_PREFILL_TOKENS,
    DEEPSEEK_V4_PRO_MTP_GEOMETRY,
    DeepSeekV4MtpGeometry,
    build_examples as build_deepseek_examples,
    build_production_dspark_projection,
    build_production_mtp_decode_projection,
    build_production_mtp_history_projection_branch,
    build_production_mtp_projection_branch,
    build_production_mtp_prefill_projection_branch,
)
from examples.torch_frontend.hybrid_qwen import (
    emit_hybrid_qwen_output_head,
    emit_production_qwen_output_head_overlay,
)
from examples.torch_frontend.orchestration_boundaries import (
    build_examples as build_boundary_examples,
)
from examples.torch_frontend.qwen3 import build_examples as build_qwen_examples
from examples.torch_frontend.qwen3 import (
    QWEN_BATCH_TILE,
    QWEN_LM_HEAD_K_CHUNK,
    QWEN_PRODUCTION_HIDDEN,
    QWEN_PRODUCTION_VOCAB,
    QWEN_REFERENCE_RMS_K_CHUNK,
    QWEN_VOCAB_CHUNK,
    build_production_qwen_output_head,
)
from examples.torch_frontend.static_mixed import (
    build_examples as build_static_mixed_examples,
)
from examples.torch_frontend.pr2335_vector import (
    build_examples as build_pr2335_examples,
)
from examples.torch_frontend.production_models import (
    emit_production_model_integration,
)
from pto_fusebox import (
    KernelKind,
    RuntimeValidShapeSpec,
    SourceEmissionError,
    can_emit_region,
    emit_deepseek_mtp_projection_overlay,
    emit_pypto_callable,
    emit_flash_mtp_decode_projection_overlay,
    emit_pypto_region,
    emit_pypto_static_bundle,
    export_and_normalize,
    extract_solver_regions,
    pypto_lib_model_manifest,
    pypto_lib_model_manifests,
    scheduled_region,
    solve_graph,
    validate_pypto_lib_model,
)
from pto_fusebox.schedule.schema import CubeKernelPlan, MixedKernelPlan
from pto_fusebox.ir import normalized_graph_sha256
from torch import nn

Example = tuple[nn.Module, tuple[torch.Tensor, ...]]


class _MetadataViewIntoNativeBoundary(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        value: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        static = torch.mm(lhs, rhs)
        viewed = value.view(64, 64)
        gathered = torch.index_select(viewed, 0, index)
        return static + gathered


def _all_examples() -> dict[str, Example]:
    return {
        **build_basic_examples(),
        **build_deepseek_examples(),
        **build_qwen_examples(),
        **build_pr2335_examples(),
        **build_static_mixed_examples(),
    }


def _test_solver() -> Path:
    configured = os.environ.get("PTO_FUSEBOX_TEST_SOLVER")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "build" / "mlsys_mixed"


def test_production_model_manifests_name_all_four_integration_targets() -> None:
    manifests = pypto_lib_model_manifests()

    assert tuple(manifest.name for manifest in manifests) == (
        "deepseek_v4_flash_dspark",
        "deepseek_v4_flash_mtp",
        "deepseek_v4_pro",
        "qwen3_14b",
    )
    for manifest in manifests:
        assert manifest.entry_points
        assert manifest.static_regions
        assert manifest.native_boundaries
        assert len({region.name for region in manifest.static_regions}) == len(
            manifest.static_regions
        )
        assert len({boundary.name for boundary in manifest.native_boundaries}) == len(
            manifest.native_boundaries
        )
        assert pypto_lib_model_manifest(manifest.name) == manifest

    with pytest.raises(ValueError, match="unknown PyPTO-lib model"):
        pypto_lib_model_manifest("deepseek_v4_unknown")


def test_production_model_manifests_match_configured_pypto_lib_checkout() -> None:
    pypto_lib_root = os.environ.get("PTO_FUSEBOX_PYPTO_LIB_ROOT")
    if pypto_lib_root is None:
        pytest.skip("set PTO_FUSEBOX_PYPTO_LIB_ROOT to a pypto-lib checkout")

    validated = tuple(
        validate_pypto_lib_model(pypto_lib_root, manifest)
        for manifest in pypto_lib_model_manifests()
    )
    assert tuple(model.manifest.name for model in validated) == (
        "deepseek_v4_flash_dspark",
        "deepseek_v4_flash_mtp",
        "deepseek_v4_pro",
        "qwen3_14b",
    )
    assert all(model.files for model in validated)


@pytest.mark.parametrize(
    ("name", "expected_kinds"),
    [
        ("softmax", ["max", "sub", "exp", "sum", "div"]),
        ("matmul", ["matmul"]),
        (
            "attention_core",
            ["transpose_view", "matmul", "max", "sub", "exp", "sum", "div", "matmul"],
        ),
    ],
)
def test_basic_examples_export_as_expected(
    name: str, expected_kinds: list[str]
) -> None:
    module, args = build_basic_examples()[name]
    graph = export_and_normalize(module, args)

    assert [op.kind for op in graph.ops] == expected_kinds
    assert len(extract_solver_regions(graph)) == 1


@pytest.mark.parametrize(
    ("builder", "name", "expected_kinds"),
    [
        (
            build_deepseek_examples,
            "deepseek_v4_rmsnorm",
            ["cast", "mul", "sum", "mul", "add", "rsqrt", "mul", "mul", "cast"],
        ),
        (
            build_deepseek_examples,
            "deepseek_v4_mtp_projection",
            [
                "cast",
                "mul",
                "sum",
                "mul",
                "add",
                "rsqrt",
                "mul",
                "mul",
                "transpose_view",
                "matmul",
                "view",
                "mul",
                "sum",
                "mul",
                "add",
                "rsqrt",
                "mul",
                "mul",
                "transpose_view",
                "matmul",
                "add",
            ],
        ),
        (
            build_qwen_examples,
            "qwen3_rms_norm_chunk",
            ["cast", "mul", "sum", "mul", "add", "rsqrt", "mul", "mul", "cast"],
        ),
        (
            build_qwen_examples,
            "qwen3_lm_head_chunk",
            ["transpose_view", "matmul"],
        ),
        (
            build_qwen_examples,
            "qwen3_rms_lm_head",
            [
                "cast",
                "mul",
                "sum",
                "mul",
                "add",
                "rsqrt",
                "mul",
                "mul",
                "cast",
                "transpose_view",
                "matmul",
            ],
        ),
    ],
)
def test_model_examples_form_one_supported_region(
    builder, name: str, expected_kinds: list[str]
) -> None:
    module, args = builder()[name]
    graph = export_and_normalize(module, args)

    assert [op.kind for op in graph.ops] == expected_kinds
    assert all(op.supported for op in graph.ops)
    regions = extract_solver_regions(graph)
    assert len(regions) == 1
    assert regions[0].op_ids == tuple(op.id for op in graph.ops)


def test_deepseek_int8_projection_is_ordinary_supported_algebra() -> None:
    module, args = build_deepseek_examples()["deepseek_v4_int8_mtp_branch"]
    graph = export_and_normalize(module, args)

    assert [op.kind for op in graph.ops] == [
        "cast",
        "mul",
        "sum",
        "mul",
        "add",
        "rsqrt",
        "mul",
        "mul",
        "mul",
        "abs",
        "max",
        "maximum",
        "div",
        "mul",
        "mul",
        "cast",
        "cast",
        "cast",
        "div",
        "transpose_view",
        "matmul",
        "cast",
        "mul",
        "mul",
    ]
    assert all(op.supported for op in graph.ops)
    assert [
        op.attributes.get("mode", "none") for op in graph.ops if op.kind == "cast"
    ] == ["none", "rint", "round", "trunc", "none"]
    assert graph.ops[20].attributes["source_operator"] == "aten._int_mm.default"
    assert len(extract_solver_regions(graph)) == 1


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_hybrid_qwen_example_links_torch_regions_into_native_pypto() -> None:
    sources = emit_hybrid_qwen_output_head(_test_solver(), solver_workers=2)
    files = sources.files()

    assert list(files) == [
        "generated_qwen_output_head.py",
        "native_qwen_output_head.py",
    ]
    for source in files.values():
        ast.parse(source)
        assert "auto_tile" not in source and "auto_fuse" not in source
    assert "@pl.inline" in sources.output_head.source
    assert sources.output_head.kind is KernelKind.MIXED
    assert "torch" not in sources.orchestration_source.lower()
    assert "from generated_qwen_output_head import generated_qwen_output_head" in (
        sources.orchestration_source
    )
    assert (
        "generated_qwen_output_head(hidden_states, norm_weight, "
        "lm_head_weight, output)" in sources.orchestration_source
    )
    assert "pl.create_tensor" not in sources.orchestration_source
    assert "__OUTPUT_HEAD_" not in sources.orchestration_source


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_production_dspark_projection_is_one_completion_aware_static_region() -> None:
    module, args = build_production_dspark_projection()
    graph = export_and_normalize(module, args)
    assert graph.value_map()[graph.inputs[-1]].shape == (
        DEEPSEEK_V4_LINEAR_TOKENS,
        DEEPSEEK_V4_HIDDEN * DEEPSEEK_V4_DSPARK_TARGET_LAYERS,
    )
    solved = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.whole_graph_codegen_ready
    assert len(solved.regions) == 1
    schedule = scheduled_region(solved.regions[0])
    # A chunked BF16 matmul carries FP32 until its logical BF16 storage
    # boundary. The former mixed choice sized that crossing as BF16 and was
    # therefore not source-feasible; the corrected model keeps the complete
    # static graph but selects an explicit cube-to-vector GM cut.
    assert [step.kind for step in schedule.steps] == [
        KernelKind.CUBE,
        KernelKind.VECTOR,
    ]
    cube_plan = schedule.steps[0].plan
    assert isinstance(cube_plan, CubeKernelPlan)
    assert cube_plan.matmuls[0].accumulator_dtype == "fp32"
    assert cube_plan.matmuls[0].storage_dtype == "bf16"
    assert cube_plan.matmuls[0].k_loop.full_chunks > 1
    emitted = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name="fusebox_dspark_projection",
        expose_completion_task=True,
    )
    assert emitted.completion_task is not None
    assert "pl.Scalar[pl.TASK_ID]" in emitted.source
    assert "auto_tile" not in emitted.source and "auto_fuse" not in emitted.source


@pytest.mark.parametrize(
    "geometry",
    (DEEPSEEK_V4_FLASH_MTP_GEOMETRY, DEEPSEEK_V4_PRO_MTP_GEOMETRY),
)
def test_production_mtp_graph_uses_explicit_static_geometry(
    geometry: DeepSeekV4MtpGeometry,
) -> None:
    """Model selection supplies shapes; the normalized DAG remains generic."""

    module, args = build_production_mtp_decode_projection(geometry)
    graph = export_and_normalize(module, args)
    values = graph.value_map()
    by_name = {values[value_id].name: values[value_id] for value_id in graph.inputs}

    assert tuple(by_name["hidden_padded"].shape) == (
        geometry.linear_tokens,
        geometry.hidden_size,
    )
    assert tuple(by_name["history_flat"].shape) == (
        geometry.decode_tokens * geometry.hyperconnections,
        geometry.hidden_size,
    )
    assert tuple(values[graph.outputs[0]].shape) == (
        geometry.decode_tokens,
        geometry.hyperconnections,
        geometry.hidden_size,
    )


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_production_qwen_output_head_overlay_preserves_native_window_abi() -> None:
    module, args = build_production_qwen_output_head()
    graph = export_and_normalize(module, args)
    assert [graph.value_map()[value_id].shape for value_id in graph.inputs] == [
        (1, QWEN_PRODUCTION_HIDDEN),
        (QWEN_PRODUCTION_VOCAB, QWEN_PRODUCTION_HIDDEN),
        (QWEN_BATCH_TILE, QWEN_PRODUCTION_HIDDEN),
    ]
    native = "from rms_lm_head import rms_lm_head, rms_lm_head_fp32\n"
    overlay = emit_production_qwen_output_head_overlay(
        _test_solver(),
        native_decode_source=native,
        solver_workers=2,
    )

    assert "pl.Tensor[[16, 5120], pl.FP32]" in overlay.source
    assert "pl.Tensor[[152064, 5120], pl.BF16]" in overlay.source
    assert "static_output = pl.create_tensor([16, 152064]" in overlay.source
    assert overlay.source.count("valid_shape=[valid_rows, 192]") == 1
    assert "pl.store(output_tile, [row_offset, output_col], out)" in overlay.source
    # Only this logical prefix belongs to the generated callable. Padded rows
    # outside valid_rows need not match a native implementation that chooses
    # to compute its complete physical frame.
    assert "[0, output_col], [16, 192], valid_shape=[valid_rows, 192]" in overlay.source
    assert overlay.decode_source == (
        "from rms_lm_head import rms_lm_head\n"
        "from fusebox_qwen_output_head import rms_lm_head_fp32\n"
    )
    assert "auto_tile" not in overlay.source and "auto_fuse" not in overlay.source


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_production_qwen_lm_head_accumulator_and_traffic_parity() -> None:
    """Compare loop-weighted traffic, not static PTO statement counts.

    The native implementation has one syntactic drain inside a 33-trip
    grid-stride loop, while Fusebox expands nine output tiles per spatial
    owner.  Counting the statement bodies makes the generated drain look
    larger even though both schedules write the complete output exactly once.
    """

    module, args = build_production_qwen_output_head()
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    schedule = scheduled_region(solved.regions[0])
    assert [step.kind for step in schedule.steps] == [
        KernelKind.VECTOR,
        KernelKind.CUBE,
    ]
    cube_plan = schedule.steps[1].plan
    assert isinstance(cube_plan, CubeKernelPlan)
    assert cube_plan.spatial_tiles == 24
    matmul = cube_plan.matmuls[0]

    # The existing source candidate already retains one FP32 accumulator
    # through every K contribution for each output tile.  A second
    # "persistent accumulator" mechanism would therefore be a duplicate.
    assert matmul.accumulator_dtype == "fp32"
    assert matmul.storage_dtype == "fp32"
    assert matmul.k_loop.full_chunks > 1
    assert matmul.final_drain.required
    assert not matmul.final_drain.atomic
    assert matmul.final_drain.tile_count == math.prod(matmul.output_grid)

    output_bytes = QWEN_BATCH_TILE * QWEN_PRODUCTION_VOCAB * 4
    generated_drain_bytes = matmul.final_drain.bytes * cube_plan.spatial_tiles
    native_vocab_trips = QWEN_PRODUCTION_VOCAB // QWEN_VOCAB_CHUNK
    native_drain_bytes = QWEN_BATCH_TILE * QWEN_VOCAB_CHUNK * 4 * native_vocab_trips
    assert generated_drain_bytes == native_drain_bytes == output_bytes

    full_weight_bytes = QWEN_PRODUCTION_VOCAB * QWEN_PRODUCTION_HIDDEN * 2
    generated_weight_bytes = (
        matmul.rhs.height * matmul.rhs.width * 2 * cube_plan.spatial_tiles
    )
    native_k_trips = QWEN_PRODUCTION_HIDDEN // QWEN_LM_HEAD_K_CHUNK
    native_weight_bytes = (
        QWEN_VOCAB_CHUNK
        * QWEN_LM_HEAD_K_CHUNK
        * 2
        * native_vocab_trips
        * native_k_trips
    )
    assert generated_weight_bytes == native_weight_bytes == full_weight_bytes

    # Fusebox retains each spatial owner's activation panel.  The native
    # grid-stride loop reloads the activation for every vocabulary tile.
    generated_activation_bytes = (
        matmul.retained_panels.lhs_bytes * cube_plan.spatial_tiles
    )
    native_activation_bytes = (
        QWEN_BATCH_TILE * QWEN_LM_HEAD_K_CHUNK * 2 * native_vocab_trips * native_k_trips
    )
    assert generated_activation_bytes == (
        QWEN_BATCH_TILE * QWEN_PRODUCTION_HIDDEN * 2 * cube_plan.spatial_tiles
    )
    assert generated_activation_bytes < native_activation_bytes


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
@pytest.mark.parametrize(
    ("model_name", "region", "callable_count", "patched_file", "mtp_geometry"),
    (
        ("deepseek_v4_flash_dspark", "dspark_projection", 1, None, None),
        (
            "deepseek_v4_flash_mtp",
            "mtp_projection",
            2,
            "decode_mtp.py",
            DEEPSEEK_V4_FLASH_MTP_GEOMETRY,
        ),
        (
            "deepseek_v4_pro",
            "mtp_projection",
            2,
            "decode_mtp.py",
            DEEPSEEK_V4_PRO_MTP_GEOMETRY,
        ),
        ("qwen3_14b", "output_head", 1, "decode_fwd.py", None),
    ),
)
def test_production_model_integration_emits_first_maximal_region(
    model_name: str,
    region: str,
    callable_count: int,
    patched_file: str | None,
    mtp_geometry: DeepSeekV4MtpGeometry | None,
) -> None:
    pypto_lib_root = os.environ.get("PTO_FUSEBOX_PYPTO_LIB_ROOT")
    if pypto_lib_root is None:
        pytest.skip("set PTO_FUSEBOX_PYPTO_LIB_ROOT to a pypto-lib checkout")
    integration = emit_production_model_integration(
        model_name,
        pypto_lib_root,
        _test_solver(),
        solver_workers=2,
    )

    assert integration.model_name == model_name
    assert integration.implemented_static_regions == (region,)
    assert len(integration.callables) == callable_count
    assert all(
        "auto_tile" not in source.source for source in integration.generated_sources
    )
    assert all(
        "auto_fuse" not in source.source for source in integration.generated_sources
    )
    if patched_file is None:
        assert integration.patched_native_sources == ()
        assert integration.callables[0].completion_task is not None
    else:
        assert tuple(
            source.relative_path for source in integration.patched_native_sources
        ) == (patched_file,)
    if mtp_geometry is not None:
        source = integration.generated_sources[0].source
        hidden = mtp_geometry.hidden_size
        assert f"pl.Tensor[[{hidden}, {hidden}], pl.INT8]" in source
        assert (
            f"pl.Tensor[[{mtp_geometry.decode_tokens}, "
            f"{mtp_geometry.hyperconnections}, {hidden}], pl.FP32]" in source
        )
        combine_cols = math.gcd(hidden, 1024)
        combine_work = (
            mtp_geometry.decode_tokens
            * mtp_geometry.hyperconnections
            * (hidden // combine_cols)
        )
        assert f"pl.spmd({combine_work}, name_hint='fusebox_mtp_combine')" in source
    assert len(integration.files()) == len(integration.generated_sources) + len(
        integration.patched_native_sources
    )


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_production_flash_mtp_branches_emit_static_decode_callables() -> None:
    analytic_module, analytic_args = build_production_mtp_projection_branch()
    analytic_graph = export_and_normalize(analytic_module, analytic_args)
    analytic = solve_graph(
        analytic_graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=False,
    )
    assert analytic.successful and not analytic.whole_graph_codegen_ready
    assert analytic.regions[0].solution is not None

    solved = []
    for builder, rows in (
        (build_production_mtp_projection_branch, DEEPSEEK_V4_LINEAR_TOKENS),
        (
            build_production_mtp_history_projection_branch,
            DEEPSEEK_V4_DECODE_TOKENS * DEEPSEEK_V4_HC_MULT,
        ),
    ):
        module, args = builder()
        graph = export_and_normalize(module, args)
        result = solve_graph(
            graph,
            solver_binary=_test_solver(),
            solver_workers=2,
            require_source_codegen=True,
        )
        assert result.successful and result.whole_graph_codegen_ready
        assert tuple(graph.value_map()[graph.inputs[0]].shape) == (
            rows,
            DEEPSEEK_V4_HIDDEN,
        )
        schedule = scheduled_region(result.regions[0])
        source_ops = [op for step in schedule.steps for op in step.op_order]
        compute_op_count = sum(not op.metadata_only for op in graph.ops)
        assert len(source_ops) == len(set(source_ops)) == compute_op_count
        assert [step.kind for step in schedule.steps] == [
            KernelKind.VECTOR,
            KernelKind.VECTOR,
            KernelKind.VECTOR,
            KernelKind.CUBE,
            KernelKind.VECTOR,
        ]
        source = emit_pypto_callable(
            graph,
            result.regions[0],
            function_name=f"projection_{rows}",
        ).source
        assert "mode='rint'" in source
        assert "mode='round'" in source
        assert "mode='trunc'" in source
        assert "b_trans=True" in source
        assert "out_dtype=pl.INT32" in source
        solved.append((graph, result.regions[0]))

    full_module, full_args = build_production_mtp_decode_projection()
    full_graph = export_and_normalize(full_module, full_args)
    full_result = solve_graph(
        full_graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert full_result.regions_solved
    assert not full_result.whole_graph_supported
    assert len(full_result.regions) == 2
    assert tuple(len(region.region.op_ids) for region in full_result.regions) == (
        24,
        24,
    )
    expected_branch_steps = [
        KernelKind.VECTOR,
        KernelKind.VECTOR,
        KernelKind.VECTOR,
        KernelKind.CUBE,
        KernelKind.VECTOR,
    ]
    assert [
        [step.kind for step in scheduled_region(region).steps]
        for region in full_result.regions
    ] == [expected_branch_steps, expected_branch_steps]

    native = "from mtp_projection import golden_mtp_projection, mtp_projection\n"
    overlay = emit_flash_mtp_decode_projection_overlay(
        full_graph,
        full_result,
        native_decode_source=native,
    )
    compile(overlay.source, "<fusebox_mtp_projection>", "exec")
    assert overlay.source.count("@pl.inline") == 3
    assert "pl.create_tensor([16, 4096], dtype=pl.BF16)" in overlay.source
    assert "pl.reshape(prev_hidden_states, [32, 4096])" in overlay.source
    assert "pl.spmd(128, name_hint='fusebox_mtp_combine')" in overlay.source
    assert len(overlay.static_callables) == 2
    assert overlay.native_op_ids == ("op0048", "op0049", "op0050", "op0051")
    assert "auto_tile" not in overlay.source and "auto_fuse" not in overlay.source
    assert overlay.decode_source == (
        "from mtp_projection import golden_mtp_projection\n"
        "from fusebox_mtp_projection import mtp_projection\n"
    )
    pro_native = """from mtp_projection import (
    _quantize_weight_per_out,
    golden_mtp_projection,
    mtp_projection,
)
"""
    pro_overlay = emit_deepseek_mtp_projection_overlay(
        full_graph,
        full_result,
        native_source=pro_native,
        module_name="fusebox_pro_mtp_projection",
    )
    assert pro_overlay.decode_source == (
        "from mtp_projection import _quantize_weight_per_out, "
        "golden_mtp_projection\n"
        "from fusebox_pro_mtp_projection import mtp_projection\n"
    )
    with pytest.raises(SourceEmissionError, match="exactly once; found 0"):
        emit_deepseek_mtp_projection_overlay(
            full_graph,
            full_result,
            native_source="from mtp_projection import golden_mtp_projection\n",
        )
    with pytest.raises(SourceEmissionError, match="must be unaliased"):
        emit_deepseek_mtp_projection_overlay(
            full_graph,
            full_result,
            native_source=(
                "from mtp_projection import mtp_projection as native_projection\n"
            ),
        )
    with pytest.raises(SourceEmissionError, match="exactly once; found 2"):
        emit_deepseek_mtp_projection_overlay(
            full_graph,
            full_result,
            native_source=(
                "from mtp_projection import mtp_projection\n"
                "from mtp_projection import mtp_projection\n"
            ),
        )
    with pytest.raises(SourceEmissionError, match="module name is invalid"):
        emit_flash_mtp_decode_projection_overlay(
            full_graph,
            full_result,
            native_decode_source=native,
            module_name="fusebox-mtp-projection",
        )


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_flash_mtp_native_tail_fails_closed_on_semantic_drift() -> None:
    module, args = build_production_mtp_decode_projection()
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    native = "from mtp_projection import golden_mtp_projection, mtp_projection\n"

    def mutate_op(op_id: str, **changes):
        mutated = replace(
            graph,
            ops=tuple(
                replace(op, **changes) if op.id == op_id else op for op in graph.ops
            ),
        )
        assert normalized_graph_sha256(mutated) != normalized_graph_sha256(graph)
        regions = []
        for region in solved.regions:
            assert region.problem is not None
            problem = dict(region.problem)
            frontend = dict(problem["frontend_mapping"])
            frontend["normalized_graph_sha256"] = normalized_graph_sha256(mutated)
            problem["frontend_mapping"] = frontend
            regions.append(replace(region, problem=problem))
        return mutated, replace(solved, graph=mutated, regions=tuple(regions))

    slice_attrs = dict(graph.op_map()["op0048"].attributes)
    for position, value in ((2, 1), (2, 8), (3, 7), (3, 16)):
        literal_args = slice_attrs["literal_args"]
        assert isinstance(literal_args, list)
        literals = []
        for item in literal_args:
            assert isinstance(item, dict)
            literals.append(dict(item))
        next(item for item in literals if item["position"] == position)["value"] = value
        attributes = {**slice_attrs, "literal_args": literals}
        mutated, mutated_result = mutate_op("op0048", attributes=attributes)
        with pytest.raises(SourceEmissionError, match="slice must select rows"):
            emit_flash_mtp_decode_projection_overlay(
                mutated,
                mutated_result,
                native_decode_source=native,
            )

    same_shape_args = slice_attrs["literal_args"]
    assert isinstance(same_shape_args, list)
    same_shape_literals = []
    for item in same_shape_args:
        assert isinstance(item, dict)
        same_shape_literals.append(dict(item))
    next(item for item in same_shape_literals if item["position"] == 2)["value"] = 8
    next(item for item in same_shape_literals if item["position"] == 3)["value"] = 16
    mutated, mutated_result = mutate_op(
        "op0048",
        attributes={**slice_attrs, "literal_args": same_shape_literals},
    )
    with pytest.raises(SourceEmissionError, match="slice must select rows"):
        emit_flash_mtp_decode_projection_overlay(
            mutated,
            mutated_result,
            native_decode_source=native,
        )

    reshape = graph.op_map()["op0050"]
    reshape_attrs = dict(reshape.attributes)
    reshape_literals = reshape_attrs["literal_args"]
    assert isinstance(reshape_literals, list)
    mutated_reshape_literals = []
    for item in reshape_literals:
        assert isinstance(item, dict)
        mutated_reshape_literals.append(dict(item))
    mutated_reshape_literals[0]["value"] = [16, 8192]
    mutated, mutated_result = mutate_op(
        "op0050",
        attributes={
            **reshape_attrs,
            "literal_args": mutated_reshape_literals,
        },
    )
    with pytest.raises(SourceEmissionError, match="reshape must produce"):
        emit_flash_mtp_decode_projection_overlay(
            mutated,
            mutated_result,
            native_decode_source=native,
        )

    hidden_output = graph.op_map()["op0048"].inputs[0]
    mutated, mutated_result = mutate_op("op0050", inputs=(hidden_output,))
    with pytest.raises(SourceEmissionError, match="branch outputs"):
        emit_flash_mtp_decode_projection_overlay(
            mutated,
            mutated_result,
            native_decode_source=native,
        )

    add = graph.op_map()["op0051"]
    for inputs in (tuple(reversed(add.inputs)), (add.inputs[1], add.inputs[1])):
        mutated, mutated_result = mutate_op("op0051", inputs=inputs)
        with pytest.raises(SourceEmissionError, match="value lineage is stale"):
            emit_flash_mtp_decode_projection_overlay(
                mutated,
                mutated_result,
                native_decode_source=native,
            )


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_rank_one_rmsnorm_weight_lifts_to_a_row_broadcast_abi() -> None:
    module, args = build_deepseek_examples()["deepseek_v4_rmsnorm"]
    graph = export_and_normalize(module, args)
    weight = next(value for value in graph.values if value.name == "p_weight")
    assert weight.shape == (1024,)

    solved = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.whole_graph_codegen_ready
    assert len(solved.regions) == 1
    emitted = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name="deepseek_v4_rmsnorm",
    )
    assert "arg_p_weight: pl.Tensor[[1, 1024], pl.FP32]" in emitted.source
    assert "pl.col_expand_mul" in emitted.source


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_production_flash_mtp_prefill_branch_is_source_ready() -> None:
    module, args = build_production_mtp_prefill_projection_branch()
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert solved.successful and solved.whole_graph_codegen_ready
    assert graph.value_map()[graph.inputs[0]].shape == (
        DEEPSEEK_V4_PREFILL_TOKENS,
        DEEPSEEK_V4_HIDDEN,
    )
    schedule = scheduled_region(solved.regions[0])
    assert [step.kind for step in schedule.steps] == [
        KernelKind.VECTOR,
        KernelKind.VECTOR,
        KernelKind.VECTOR,
        KernelKind.CUBE,
        KernelKind.VECTOR,
    ]
    source = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name="fusebox_mtp_prefill_projection",
    ).source
    ast.parse(source)
    assert "b_trans=True" in source
    assert "out_dtype=pl.INT32" in source


@pytest.mark.parametrize(
    ("name", "expected_kinds"),
    [
        ("mixed_c2v_single_item", ["matmul", "add"]),
        ("mixed_c2v_streamed_groups", ["matmul", "add"]),
        (
            "mixed_cvc_streamed_groups",
            ["transpose_view", "matmul", "max", "sub", "exp", "sum", "div", "matmul"],
        ),
        (
            "pypto_lib_static_attention",
            ["transpose_view", "matmul", "max", "sub", "exp", "sum", "div", "matmul"],
        ),
        (
            "pypto_lib_static_dense_swiglu",
            [
                "matmul",
                "matmul",
                "neg",
                "exp",
                "add",
                "div",
                "mul",
                "mul",
                "cast",
                "matmul",
            ],
        ),
        (
            "pypto_lib_static_attention_residual",
            [
                "transpose_view",
                "matmul",
                "max",
                "sub",
                "exp",
                "sum",
                "div",
                "matmul",
                "add",
            ],
        ),
    ],
)
def test_static_mixed_examples_export_one_coherent_dag(
    name: str, expected_kinds: list[str]
) -> None:
    module, args = build_static_mixed_examples()[name]
    graph = export_and_normalize(module, args)

    assert [op.kind for op in graph.ops] == expected_kinds
    assert all(op.supported for op in graph.ops)
    assert len(extract_solver_regions(graph)) == 1


@pytest.mark.parametrize(
    ("name", "expected_regions", "expected_boundaries"),
    [
        (
            "paged_attention_static_regions",
            [
                ["transpose_view", "matmul", "max", "sub", "exp", "sum", "div"],
                ["matmul"],
            ],
            ["aten.index_select.default"],
        ),
        (
            "moe_static_regions",
            [
                ["matmul"],
                [
                    "matmul",
                    "matmul",
                    "neg",
                    "exp",
                    "add",
                    "div",
                    "mul",
                    "mul",
                    "matmul",
                ],
            ],
            ["aten.topk.default", "aten.reshape.default", "aten.index_select.default"],
        ),
    ],
)
def test_orchestration_examples_preserve_explicit_opaque_boundaries(
    name: str,
    expected_regions: list[list[str]],
    expected_boundaries: list[str],
) -> None:
    module, args = build_boundary_examples()[name]
    graph = export_and_normalize(module, args)
    regions = extract_solver_regions(graph)

    op_map = graph.op_map()
    assert [[op_map[op].kind for op in region.op_ids] for region in regions] == (
        expected_regions
    )
    assert [
        str(op.attributes["source_operator"]) for op in graph.ops if not op.supported
    ] == expected_boundaries


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
@pytest.mark.parametrize(
    ("name", "boundary_count"),
    [("paged_attention_static_regions", 1), ("moe_static_regions", 3)],
)
def test_static_bundle_wires_solved_regions_around_native_boundaries(
    name: str,
    boundary_count: int,
) -> None:
    module, args = build_boundary_examples()[name]
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.successful
    assert not result.whole_graph_supported
    bundle = emit_pypto_static_bundle(
        graph, result, function_prefix=f"generated_{name}"
    )
    repeated = emit_pypto_static_bundle(
        graph, result, function_prefix=f"generated_{name}"
    )
    assert bundle == repeated
    assert len(bundle.callables) == 2
    assert bundle.graph == graph
    assert len(bundle.native_operations) == boundary_count
    assert [callable.region_id for callable in bundle.callables] == [
        "region0000",
        "region0001",
    ]
    assert [callable.function_name for callable in bundle.callables] == [
        f"generated_{name}_region0000",
        f"generated_{name}_region0001",
    ]
    for callable in bundle.callables:
        assert "@pl.inline" in callable.source
        assert "auto_fuse" not in callable.source
        assert "auto_tile" not in callable.source

    first, second = bundle.callables
    boundaries = bundle.native_operations
    assert set(first.output_value_ids) & (
        set(second.input_value_ids) | set(boundaries[0].inputs)
    )
    assert set(boundaries[-1].outputs) & set(second.input_value_ids)

    static_op_ids = {
        op_id for region in result.regions for op_id in region.region.op_ids
    }
    native_op_ids = set(bundle.native_op_ids)
    assert static_op_ids.isdisjoint(native_op_ids)
    assert static_op_ids | native_op_ids == {op.id for op in graph.ops}
    represented = static_op_ids | native_op_ids
    assert all(
        value.producer is None or value.producer in represented
        for value in graph.values
    )


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_static_bundle_keeps_admitted_metadata_on_the_native_side() -> None:
    module = _MetadataViewIntoNativeBoundary()
    args = (
        torch.zeros(64, 64),
        torch.zeros(64, 64),
        torch.zeros(64, 64),
        torch.arange(64),
    )
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    bundle = emit_pypto_static_bundle(graph, result)

    assert [op.kind for op in bundle.native_operations] == ["view", "opaque"]
    view, gather = bundle.native_operations
    assert view.metadata_only and view.supported
    assert gather.inputs[0] == view.outputs[0]
    assert graph.value_map()[gather.inputs[0]].producer == view.id


@pytest.mark.parametrize("name", sorted(_all_examples()))
def test_example_matmuls_are_semantically_coherent_and_cube_sized(name: str) -> None:
    module, args = _all_examples()[name]
    graph = export_and_normalize(module, args)
    values = graph.value_map()

    for op in graph.ops:
        if op.kind != "matmul":
            continue
        lhs = values[op.inputs[0]]
        rhs = values[op.inputs[1]]
        output = values[op.outputs[0]]
        lhs_shape = tuple(
            dimension for dimension in lhs.shape if isinstance(dimension, int)
        )
        rhs_shape = tuple(
            dimension for dimension in rhs.shape if isinstance(dimension, int)
        )
        output_shape = tuple(
            dimension for dimension in output.shape if isinstance(dimension, int)
        )
        assert len(lhs_shape) == len(lhs.shape)
        assert len(rhs_shape) == len(rhs.shape)
        assert len(output_shape) == len(output.shape)
        m = math.prod(lhs_shape[:-1])
        lhs_k = lhs_shape[-1]
        rhs_k, n = rhs_shape
        output_m = math.prod(output_shape[:-1])
        output_n = output_shape[-1]
        assert (lhs_k, output_m, output_n) == (rhs_k, m, n)
        assert min(m, n, lhs_k) >= 16


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_all_examples_solve_as_complete_supported_regions() -> None:
    solver = _test_solver()
    for name, (module, args) in _all_examples().items():
        graph = export_and_normalize(module, args)
        result = solve_graph(graph, solver_binary=solver, solver_workers=2)
        assert result.successful, {
            "example": name,
            "statuses": [region.status for region in result.regions],
            "diagnostics": [region.diagnostics for region in result.regions],
        }


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_attention_solver_selects_complete_cube_vector_cube_group() -> None:
    module, args = build_basic_examples()["attention_core"]
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.successful
    assert result.regions_solved
    assert result.whole_graph_codegen_ready
    region = result.regions[0]
    assert can_emit_region(graph, region)
    assert region.solution is not None
    assert region.solution["steps"][0]["ops"] == list(
        range(len(region.solver_op_to_graph))
    )
    schedule = region.solution["steps"][0]["plan"]
    assert schedule["source_codegen_ready"] is True
    vector_stages = [
        stage for stage in schedule["stages"] if stage["engine"] == "vector"
    ]
    assert len(vector_stages) == 1
    assert vector_stages[0]["vector_stream"]["kind"] == "materialized"


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
@pytest.mark.parametrize(
    "name",
    [
        "mixed_c2v_single_item",
        "mixed_c2v_streamed_groups",
        "mixed_cvc_streamed_groups",
        "pypto_lib_static_attention",
        "pypto_lib_static_dense_swiglu",
        "pypto_lib_static_attention_residual",
    ],
)
def test_static_mixed_examples_solve_and_emit_generic_pypto_source(
    name: str,
) -> None:
    module, args = build_static_mixed_examples()[name]
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.successful
    assert result.whole_graph_codegen_ready
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.problem is not None
    assert region.problem["require_source_codegen"] is True
    assert can_emit_region(graph, region)
    plan = scheduled_region(region).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    slot_counts = {fifo.slot_count for fifo in plan.fifos}
    assert len(slot_counts) == 1
    source = emit_pypto_region(graph, region, program_name=name).source
    assert "pl.cross_core_slot(" not in source
    assert source.count("pl.cross_core_pipe(") == len(plan.fifos)
    for fifo in plan.fifos:
        direction = (
            "pl.CrossCoreDirection.CUBE_TO_VECTOR"
            if fifo.direction.value == "cube_to_vector"
            else "pl.CrossCoreDirection.VECTOR_TO_CUBE"
        )
        assert (
            "pl.cross_core_pipe("
            f"tensor_id={fifo.tensor}, direction={direction}, "
            f"valid_shape=[{fifo.valid_rows}, {fifo.valid_cols}], "
            f"slot_size_bytes={fifo.slot_bytes}, slot_num={fifo.slot_count}, "
            f"pipe_id={fifo.pipe_id}, bundle={fifo.bundle})"
        ) in source
    assert "auto_fuse" not in source and "auto_tile" not in source


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_mixed_examples_discriminate_single_item_from_cross_core_streaming() -> None:
    plans: dict[str, MixedKernelPlan] = {}
    sources: dict[str, str] = {}
    for name in ("mixed_c2v_single_item", "mixed_c2v_streamed_groups"):
        module, args = build_static_mixed_examples()[name]
        graph = export_and_normalize(module, args)
        result = solve_graph(
            graph,
            solver_binary=_test_solver(),
            solver_workers=2,
            require_source_codegen=True,
        )
        assert result.successful
        region = result.regions[0]
        plan = scheduled_region(region).steps[0].plan
        assert isinstance(plan, MixedKernelPlan)
        plans[name] = plan
        sources[name] = emit_pypto_region(graph, region, program_name=name).source

    single = plans["mixed_c2v_single_item"]
    assert single.spatial_tiles == 1
    assert single.active_groups == 1
    assert single.max_trips_per_group == 1
    assert single.pipeline_stages == 1
    assert not single.overlap_implementable
    assert "pl.pipeline(" not in sources["mixed_c2v_single_item"]

    streamed = plans["mixed_c2v_streamed_groups"]
    assert streamed.spatial_tiles == 24
    assert streamed.active_groups == 6
    assert streamed.max_trips_per_group == 4
    assert streamed.pipeline_stages == 2
    assert streamed.requested_skew_depth == 1
    assert streamed.overlap_implementable
    assert "pl.pipeline(4, stage=2" in sources["mixed_c2v_streamed_groups"]

    module, args = build_static_mixed_examples()["mixed_cvc_streamed_groups"]
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert result.successful
    region = result.regions[0]
    round_trip = scheduled_region(region).steps[0].plan
    assert isinstance(round_trip, MixedKernelPlan)
    assert round_trip.spatial_tiles == 4
    assert round_trip.active_groups == 4
    assert round_trip.max_trips_per_group == 1
    assert round_trip.pipeline_stages == 1
    assert round_trip.requested_skew_depth == 0
    assert not round_trip.overlap_implementable
    round_trip_source = emit_pypto_region(
        graph, region, program_name="mixed_cvc_streamed_groups"
    ).source
    assert "pl.range(1, init_values=" in round_trip_source


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
@pytest.mark.parametrize(
    ("builder", "name", "expected_kind", "runtime_rows"),
    [
        (build_pr2335_examples, "pr2335_rms_norm", KernelKind.VECTOR, True),
        (build_qwen_examples, "qwen3_rms_norm_chunk", KernelKind.VECTOR, True),
        (build_qwen_examples, "qwen3_lm_head_chunk", KernelKind.CUBE, False),
        (
            build_static_mixed_examples,
            "pypto_lib_static_attention",
            KernelKind.MIXED,
            False,
        ),
        (
            build_static_mixed_examples,
            "pypto_lib_static_dense_swiglu",
            KernelKind.MIXED,
            False,
        ),
    ],
)
def test_pypto_lib_comparison_regions_emit_stable_callable_abis(
    builder,
    name: str,
    expected_kind: KernelKind,
    runtime_rows: bool,
) -> None:
    module, args = builder()[name]
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.successful
    assert result.whole_graph_codegen_ready
    assert len(result.regions) == 1
    emitted = emit_pypto_callable(
        graph,
        result.regions[0],
        function_name=f"generated_{name}",
        runtime_valid_shape=RuntimeValidShapeSpec() if runtime_rows else None,
    )
    assert emitted.kind is expected_kind
    assert emitted.input_value_ids
    assert emitted.output_value_ids == graph.outputs
    assert "@pl.inline" in emitted.source
    assert "auto_fuse" not in emitted.source and "auto_tile" not in emitted.source
    if runtime_rows:
        assert len(emitted.runtime_valid_shapes) == 1
        assert emitted.runtime_valid_shapes[0].axis == 0
    else:
        assert emitted.runtime_valid_shapes == ()


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_qwen_lm_head_replays_the_transposed_weight_without_copying_it() -> None:
    module, args = build_qwen_examples()["qwen3_lm_head_chunk"]
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    emitted = emit_pypto_callable(
        graph, result.regions[0], function_name="generated_qwen_lm_head"
    )

    assert emitted.kind is KernelKind.CUBE
    assert "pl.tile.transpose_view(" in emitted.source
    assert "pl.create_tensor" not in emitted.source


def test_qwen_components_preserve_the_native_static_chunk_contract() -> None:
    examples = build_qwen_examples()
    rms_module, rms_args = examples["qwen3_rms_norm_chunk"]
    lm_module, lm_args = examples["qwen3_lm_head_chunk"]
    connected_module, connected_args = examples["qwen3_rms_lm_head"]
    rms_weight = rms_module.get_parameter("norm_weight")
    lm_weight = lm_module.get_parameter("lm_head_weight")

    assert QWEN_BATCH_TILE == 16
    assert QWEN_REFERENCE_RMS_K_CHUNK == 128
    assert QWEN_LM_HEAD_K_CHUNK == 512
    assert QWEN_VOCAB_CHUNK == 192
    assert rms_args[0].shape == (QWEN_BATCH_TILE, QWEN_LM_HEAD_K_CHUNK)
    assert rms_args[0].dtype is torch.bfloat16
    assert tuple(rms_weight.shape) == (1, QWEN_LM_HEAD_K_CHUNK)
    assert rms_weight.dtype is torch.float32
    assert lm_args[0].shape == (QWEN_BATCH_TILE, QWEN_LM_HEAD_K_CHUNK)
    assert lm_args[0].dtype is torch.bfloat16
    assert tuple(lm_weight.shape) == (
        QWEN_VOCAB_CHUNK,
        QWEN_LM_HEAD_K_CHUNK,
    )
    assert lm_weight.dtype is torch.bfloat16
    assert connected_args[0].dtype is torch.bfloat16

    rms_graph = export_and_normalize(rms_module, rms_args)
    lm_graph = export_and_normalize(lm_module, lm_args)
    connected_graph = export_and_normalize(connected_module, connected_args)
    assert rms_graph.value_map()[rms_graph.outputs[0]].shape == (
        QWEN_BATCH_TILE,
        QWEN_LM_HEAD_K_CHUNK,
    )
    assert rms_graph.value_map()[rms_graph.outputs[0]].dtype == "bfloat16"
    for graph in (lm_graph, connected_graph):
        output = graph.value_map()[graph.outputs[0]]
        assert output.shape == (QWEN_BATCH_TILE, QWEN_VOCAB_CHUNK)
        assert output.dtype == "float32"


def test_model_examples_expose_maximal_static_graphs_before_solving() -> None:
    """Integration examples must not pre-partition supported tensor DAGs."""

    examples = {
        "attention": build_static_mixed_examples()["pypto_lib_static_attention"],
        "dense_swiglu": build_static_mixed_examples()["pypto_lib_static_dense_swiglu"],
        "qwen_output_head": build_qwen_examples()["qwen3_rms_lm_head"],
        "deepseek_rmsnorm": build_deepseek_examples()["deepseek_v4_rmsnorm"],
        "reduced_mtp": build_deepseek_examples()["deepseek_v4_mtp_projection"],
    }
    for name, (module, args) in examples.items():
        graph = export_and_normalize(module, args)
        regions = extract_solver_regions(graph)
        assert len(regions) == 1, name
        assert regions[0].op_ids == tuple(op.id for op in graph.ops), name

    module, args = build_production_mtp_decode_projection()
    graph = export_and_normalize(module, args)
    regions = extract_solver_regions(graph)
    static_op_ids = {op_id for region in regions for op_id in region.op_ids}
    assert len(regions) == 2
    assert tuple(len(region.op_ids) for region in regions) == (24, 24)
    assert tuple(op.kind for op in graph.ops if op.id not in static_op_ids) == (
        "opaque",
        "view",
        "opaque",
        "add",
    )


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_connected_qwen_rms_norm_lm_head_is_one_exact_v2c_region() -> None:
    module, args = build_qwen_examples()["qwen3_rms_lm_head"]
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.successful
    assert result.whole_graph_codegen_ready
    assert len(result.regions) == 1
    schedule = scheduled_region(result.regions[0])
    assert len(schedule.steps) == 1
    step = schedule.steps[0]
    assert step.kind is KernelKind.MIXED
    assert step.solver_ops == tuple(range(10))
    plan = step.plan
    assert isinstance(plan, MixedKernelPlan)
    assert plan.protocol.value == "one_way"
    assert [stage.engine.value for stage in plan.stages] == ["vector", "cube"]
    assert plan.active_groups == 3
    assert plan.spatial_tiles == 3
    assert plan.pipeline_stages == 1
    assert len(plan.fifos) == 1
    fifo = plan.fifos[0]
    assert fifo.direction.value == "vector_to_cube"
    assert (fifo.valid_rows, fifo.valid_cols) == (
        QWEN_BATCH_TILE,
        QWEN_LM_HEAD_K_CHUNK,
    )
    assert fifo.slot_bytes == 16 * 512 * 2
    assert fifo.reserved_bytes == fifo.slot_bytes * fifo.slot_count

    emitted = emit_pypto_callable(
        graph,
        result.regions[0],
        function_name="generated_qwen_rms_lm_head",
    )
    assert emitted.kind is KernelKind.MIXED
    assert emitted.source.count("pl.spmd(") == 1
    assert "pl.CrossCoreDirection.VECTOR_TO_CUBE" in emitted.source
    assert "b_trans=True" in emitted.source
    assert "pl.create_tensor" not in emitted.source
    assert "auto_fuse" not in emitted.source and "auto_tile" not in emitted.source


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_deepseek_mtp_projection_emits_one_generic_branched_region() -> None:
    module, args = build_deepseek_examples()["deepseek_v4_mtp_projection"]
    graph = export_and_normalize(module, args)
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.successful
    assert result.whole_graph_supported
    assert result.whole_graph_codegen_ready
    assert len(result.regions) == 1
    schedule = scheduled_region(result.regions[0])
    assert [step.kind for step in schedule.steps] == [KernelKind.MIXED]
    plan = schedule.steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    assert plan.protocol.value == "branched_round_trip_bundle"
    assert [stage.engine.value for stage in plan.stages] == [
        "vector",
        "cube",
        "vector",
        "cube",
        "vector",
    ]
    region = result.regions[0]
    assert region.solver_op_to_graph == tuple(
        op.id for op in graph.ops if not op.metadata_only
    )
    assert tuple(op for step in schedule.steps for op in step.solver_ops) == tuple(
        range(len(region.solver_op_to_graph))
    )
    emitted = emit_pypto_callable(
        graph,
        result.regions[0],
        function_name="generated_deepseek_mtp_projection",
    )
    assert emitted.source.count("pl.spmd(") == 1
    assert emitted.source.count("pl.create_tensor(") == 0
    assert emitted.source.count("pl.cross_core_pipe(") == 4
    assert emitted.source.count("pl.CrossCoreDirection.VECTOR_TO_CUBE") == 2
    assert emitted.source.count("pl.CrossCoreDirection.CUBE_TO_VECTOR") == 2
    pipe_specs = re.findall(r"pl\.cross_core_pipe\(([^)]*)\)", emitted.source)
    assert [
        int(match.group(1))
        for spec in pipe_specs
        if (match := re.search(r"\bpipe_id=([0-9]+)", spec)) is not None
    ] == [0, 2, 1, 3]
    assert [
        match.group(1)
        for spec in pipe_specs
        if (
            match := re.search(
                r"CrossCoreDirection\.(VECTOR_TO_CUBE|CUBE_TO_VECTOR)", spec
            )
        )
        is not None
    ] == ["VECTOR_TO_CUBE", "VECTOR_TO_CUBE", "CUBE_TO_VECTOR", "CUBE_TO_VECTOR"]
    assert "auto_fuse" not in emitted.source and "auto_tile" not in emitted.source


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_vector_to_cube_pipeline_is_source_ready() -> None:
    class VectorToCube(nn.Module):
        def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.exp(value), weight)

    graph = export_and_normalize(
        VectorToCube(),
        (torch.zeros(96, 64), torch.zeros(64, 128)),
    )
    assert [op.kind for op in graph.ops] == ["exp", "matmul"]

    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert result.successful
    region = result.regions[0]
    assert region.solution is not None
    assert region.solution["steps"][0]["ops"] == [0, 1]
    schedule = region.solution["steps"][0]["plan"]
    assert schedule["source_codegen_ready"] is True
    assert schedule["split_k"] == 1
    assert schedule["work_units"] == schedule["spatial_tiles"]
    assert schedule["pipeline_extent"] == schedule["spatial_tiles"]
    assert [stage["engine"] for stage in schedule["stages"]] == ["vector", "cube"]
    assert (
        schedule["stages"][0]["valid_rows"] * schedule["vector_lanes"]
        == schedule["fifos"][0]["valid_rows"]
    )
    assert schedule["stages"][0]["valid_cols"] == 64
    assert schedule["stages"][0]["vector_stream"]["kind"] == "pointwise"
    assert schedule["mode"] == "one_way"
    assert schedule["transfers"] == [
        {
            "tensor": 1,
            "producer_stage": 0,
            "consumer_stage": 1,
            "producer_engine": "vector",
            "consumer_engine": "cube",
        }
    ]
    assert len(schedule["fifos"]) == 1
    assert schedule["fifos"][0]["direction"] == "vector_to_cube"
    assert schedule["fifos"][0]["spatial_m"] is True
    assert schedule["fifos"][0]["spatial_n"] is False
    assert result.whole_graph_codegen_ready
    assert can_emit_region(graph, region)


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_vector_to_cube_rhs_pipeline_has_source_ready_k_by_n_geometry() -> None:
    class VectorToCubeRhs(nn.Module):
        def forward(
            self, lhs: torch.Tensor, value: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            return torch.mm(lhs, torch.exp(value + bias))

    graph = export_and_normalize(
        VectorToCubeRhs(),
        (
            torch.zeros(96, 64),
            torch.zeros(64, 128),
            torch.zeros(1, 128),
        ),
    )
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.successful
    region = result.regions[0]
    assert region.solution is not None
    assert region.solution["steps"][0]["ops"] == [0, 1, 2]
    schedule = region.solution["steps"][0]["plan"]
    assert schedule["source_codegen_ready"] is True
    assert schedule["split_k"] == 1
    assert [stage["engine"] for stage in schedule["stages"]] == ["vector", "cube"]
    vector_stage = schedule["stages"][0]
    fifo = schedule["fifos"][0]
    assert vector_stage["valid_rows"] * schedule["vector_lanes"] == 64
    assert vector_stage["valid_cols"] == schedule["n_partition"]["big"]
    assert fifo["valid_rows"] == 64
    assert fifo["valid_cols"] == schedule["n_partition"]["big"]
    assert fifo["direction"] == "vector_to_cube"
    assert fifo["spatial_m"] is False
    assert fifo["spatial_n"] is True
    assert result.whole_graph_codegen_ready
    assert can_emit_region(graph, region)


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_softmax_to_pv_serializes_the_complete_flash_stream() -> None:
    class SoftmaxPv(nn.Module):
        def forward(self, scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.softmax(scores, dim=-1), value)

    graph = export_and_normalize(
        SoftmaxPv(),
        (torch.zeros(16, 4096), torch.zeros(4096, 64)),
    )
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.regions_solved
    assert result.whole_graph_codegen_ready
    region = result.regions[0]
    assert can_emit_region(graph, region)
    assert region.solution is not None
    assert region.solution["steps"][0]["ops"] == list(range(6))
    schedule = region.solution["steps"][0]["plan"]
    vector_stage = schedule["stages"][0]
    stream = vector_stage["vector_stream"]
    assert stream["kind"] == "softmax_flash"
    assert stream["extent"] == 4096
    assert stream["chunk"] < stream["extent"]
    assert stream["full_chunks"] > 1
    assert stream["tail"] > 0
    phases = {phase["name"]: phase for phase in stream["phases"]}
    assert phases["stats"]["loop"]["trip_count"] > 0
    assert phases["apply"]["loop"]["trip_count"] > 0
    fifo = schedule["fifos"][0]
    assert fifo["valid_cols"] == stream["chunk"]
    assert schedule["stages"][1]["cube_window_k"] == [stream["chunk"]]


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_multi_role_vector_to_cube_uses_one_complete_fifo_panel() -> None:
    class MultiRole(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            produced = torch.exp(value)
            return torch.mm(produced, produced)

    graph = export_and_normalize(MultiRole(), (torch.zeros(64, 64),))
    result = solve_graph(
        graph,
        solver_binary=_test_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )

    assert result.regions_solved
    assert result.whole_graph_codegen_ready
    region = result.regions[0]
    assert can_emit_region(graph, region)
    assert region.solution is not None
    schedule = region.solution["steps"][0]["plan"]
    assert schedule["m_partition"]["parts"] == 1
    assert schedule["n_partition"]["parts"] == 1
    assert schedule["fifos"][0]["spatial_m"] is True
    assert schedule["fifos"][0]["spatial_n"] is True


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_internal_transpose_boundary_is_not_whole_graph_codegen_ready() -> None:
    class InternalTranspose(nn.Module):
        def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.exp(value).t(), weight)

    graph = export_and_normalize(
        InternalTranspose(),
        (torch.zeros(8, 16), torch.zeros(8, 24)),
    )
    result = solve_graph(graph, solver_binary=_test_solver(), solver_workers=2)

    assert not result.whole_graph_supported
    assert not result.whole_graph_codegen_ready


@pytest.mark.skipif(
    not _test_solver().is_file(), reason="built mlsys_mixed solver is unavailable"
)
def test_shape_changing_cube_to_vector_uses_the_crossing_frame() -> None:
    class MatmulReduce(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.sum(torch.mm(lhs, rhs), dim=-1, keepdim=True)

    graph = export_and_normalize(
        MatmulReduce(),
        (torch.zeros(96, 64), torch.zeros(64, 128)),
    )
    result = solve_graph(graph, solver_binary=_test_solver(), solver_workers=2)

    assert result.regions_solved
    assert not result.whole_graph_codegen_ready
    region = result.regions[0]
    assert not can_emit_region(graph, region)
    assert region.solution is not None
    assert region.solution["steps"][0]["ops"] == [0, 1]
    schedule = region.solution["steps"][0]["plan"]
    fifo = schedule["fifos"][0]
    vector_stage = schedule["stages"][1]
    assert fifo["direction"] == "cube_to_vector"
    assert fifo["valid_cols"] == 128
    assert vector_stage["valid_cols"] == 128
    assert vector_stage["vector_stream"]["tile"][1] == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
