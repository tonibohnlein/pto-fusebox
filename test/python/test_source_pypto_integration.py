"""Opt-in end-to-end checks against an independently installed PyPTO/PTOAS.

The standalone source backend intentionally does not depend on PyPTO.  Set
``PTO_FUSEBOX_PYPTO_INTEGRATION=1`` in an environment whose ``pypto`` import and
``PTOAS_ROOT`` point at the versions under validation to exercise this gate.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import os
import re
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from examples.torch_frontend._runner import Example
from examples.torch_frontend.deepseek_v4 import (
    build_examples as build_deepseek_examples,
    build_production_mtp_decode_projection,
    build_production_mtp_history_projection_branch,
    build_production_mtp_prefill_projection_branch,
    build_production_mtp_projection_branch,
)
from examples.torch_frontend.hybrid_qwen import (
    emit_hybrid_qwen_output_head,
    emit_production_qwen_output_head_overlay,
)
from examples.torch_frontend.orchestration_boundaries import (
    build_examples as build_boundary_examples,
)
from examples.torch_frontend.pr2335_vector import (
    build_examples as build_pr2335_examples,
)
from examples.torch_frontend.qwen3 import build_examples as build_qwen_examples
from pto_fusebox import (
    EmittedPyPTOCallable,
    KernelKind,
    RegionSolveResult,
    RuntimeValidShapeSpec,
    emit_pypto_callable,
    emit_flash_mtp_decode_projection_overlay,
    emit_pypto_region,
    emit_pypto_static_bundle,
    enumerate_cube_plans,
    export_and_normalize,
    extract_solver_regions,
    region_for_cube_candidate,
    scheduled_region,
    solve_graph,
)
from pto_fusebox.schedule.schema import (
    CubeKernelPlan,
    MixedCrossCoreProtocol,
    MixedEngine,
    MixedKernelPlan,
    MixedTransferDirection,
    VectorKernelPlan,
    VectorStreamKind,
)
from pto_fusebox.target import Ascend910BTarget
from torch import nn


pytestmark = pytest.mark.skipif(
    os.environ.get("PTO_FUSEBOX_PYPTO_INTEGRATION") != "1",
    reason="set PTO_FUSEBOX_PYPTO_INTEGRATION=1 with PyPTO and PTOAS configured",
)


class _PointwiseChain(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.exp(lhs * 0.5) + rhs, rhs)


def test_callable_region_expands_inside_native_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generated callable contributes its exact SPMD task to a native program."""

    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    module = _PointwiseChain()
    args = (torch.zeros(96, 320), torch.ones(96, 320))
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved == 1
    emitted = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name="generated_pointwise",
    )
    module_name = re.sub(r"\W", "_", f"generated_{tmp_path.name}")
    (tmp_path / f"{module_name}.py").write_text(emitted.source, encoding="utf-8")
    caller = (
        f"from {module_name} import generated_pointwise\n"
        + "import pypto.language as pl\n\n\n"
        + "@pl.program\n"
        + "class NativeOrchestration:\n"
        + "    @pl.function(type=pl.FunctionType.Orchestration)\n"
        + "    def main(\n"
        + "        self,\n"
        + "        arg_lhs: pl.Tensor[[96, 320], pl.FP32],\n"
        + "        arg_rhs: pl.Tensor[[96, 320], pl.FP32],\n"
        + "        output: pl.Out[pl.Tensor[[96, 320], pl.FP32]],\n"
        + "    ) -> pl.Tensor[[96, 320], pl.FP32]:\n"
        + "        output = generated_pointwise(arg_lhs, arg_rhs, output)\n"
        + "        return output\n"
    )
    caller_path = tmp_path / "native_orchestration.py"
    caller_path.write_text(caller, encoding="utf-8")

    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    compiled = ir.compile(
        pl.loads(str(caller_path)),
        output_dir=str(tmp_path / "callable_pointwise"),
        dump_passes=False,
        skip_ptoas=True,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 1
    assert len(orchestration_files) == 1
    _assert_single_spmd_orchestration(
        orchestration_files[0].read_text(encoding="utf-8"), 8
    )


def test_flash_mtp_overlay_imports_inside_real_decode_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated static callables replace only the native projection import."""

    pypto_lib_root = os.environ.get("PTO_FUSEBOX_PYPTO_LIB_ROOT")
    if pypto_lib_root is None:
        pytest.skip("set PTO_FUSEBOX_PYPTO_LIB_ROOT to a pypto-lib checkout")
    assert pypto_lib_root is not None
    model_dir = Path(pypto_lib_root) / "models" / "deepseek_v4_flash_mtp"
    native_decode = model_dir / "decode_mtp.py"
    if not native_decode.is_file():
        pytest.skip("current pypto-lib checkout has no Flash-MTP decode entry point")

    module, args = build_production_mtp_decode_projection()
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved
    assert not solved.whole_graph_supported
    assert len(solved.regions) == 2

    overlay = emit_flash_mtp_decode_projection_overlay(
        graph,
        solved,
        native_decode_source=native_decode.read_text(encoding="utf-8"),
    )
    overlay_path = tmp_path / f"{overlay.module_name}.py"
    decode_path = tmp_path / "decode_mtp_fusebox.py"
    overlay_path.write_text(overlay.source, encoding="utf-8")
    decode_path.write_text(overlay.decode_source, encoding="utf-8")

    monkeypatch.syspath_prepend(str(model_dir))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    caller_path = tmp_path / "flash_mtp_overlay_probe.py"
    caller_path.write_text(
        f"""from {overlay.module_name} import mtp_projection
import pypto.language as pl


@pl.program
class FlashMtpOverlayProbe:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
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
        hidden_states_out: pl.Out[pl.Tensor[[8, 4, 4096], pl.FP32]],
    ) -> pl.Tensor[[8, 4, 4096], pl.FP32]:
        hidden_states_out = mtp_projection(
            hidden_states,
            prev_hidden_states,
            enorm_w,
            hnorm_w,
            e_proj_w,
            e_proj_w_scale,
            e_proj_smooth,
            h_proj_w,
            h_proj_w_scale,
            h_proj_smooth,
            hidden_states_out,
        )
        return hidden_states_out
""",
        encoding="utf-8",
    )
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    compiled = ir.compile(
        pl.loads(str(caller_path)),
        output_dir=str(tmp_path / "flash_mtp_overlay_probe"),
        dump_passes=False,
        skip_ptoas=True,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 13
    assert len(orchestration_files) == 1
    assert orchestration_files[0].read_text(encoding="utf-8").count("rt_submit_") == 13

    spec = importlib.util.spec_from_file_location("decode_mtp_fusebox", decode_path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert callable(loaded.decode_mtp)
    assert loaded.mtp_projection is not None
    assert "auto_tile" not in overlay.source and "auto_fuse" not in overlay.source


@pytest.mark.parametrize(
    ("builder", "name"),
    (
        (build_production_mtp_projection_branch, "mtp_hidden_decode"),
        (build_production_mtp_history_projection_branch, "mtp_history_decode"),
        (build_production_mtp_prefill_projection_branch, "mtp_hidden_prefill"),
    ),
)
def test_production_mtp_physical_broadcast_frames_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    builder: Callable[[], Example],
    name: str,
) -> None:
    """Cross-layer guard for the quantize row-scale physical frame."""

    module, args = builder()
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.successful and solved.whole_graph_codegen_ready
    emitted = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name=f"generated_{name}",
    )

    pto, orchestration = _compile_callable_in_native_orchestration(
        emitted,
        tmp_path,
        monkeypatch,
        name=name,
    )
    assert len(pto) == 5
    assert orchestration.count("rt_submit_") == 5


def _compile_callable_in_native_orchestration(
    emitted: EmittedPyPTOCallable,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
) -> tuple[list[str], str]:
    """Import a generated callable from an independent native PyPTO program."""

    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    tree = ast.parse(emitted.source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) == 1
    function = functions[0]
    assert all(argument.annotation is not None for argument in function.args.args)
    assert function.returns is not None
    assert len(emitted.output_arguments) == 1

    module_name = re.sub(r"\W", "_", f"generated_{name}_{tmp_path.name}")
    (tmp_path / f"{module_name}.py").write_text(emitted.source, encoding="utf-8")
    parameter_lines: list[str] = []
    for argument in function.args.args:
        assert argument.annotation is not None
        parameter_lines.append(
            f"        {argument.arg}: {ast.unparse(argument.annotation)},"
        )
    parameters = "\n".join(parameter_lines)
    argument_names = ", ".join(argument.arg for argument in function.args.args)
    output_name = emitted.output_arguments[0].name
    caller = (
        f"from {module_name} import {emitted.function_name}\n"
        + "import pypto.language as pl\n\n\n"
        + "@pl.program\n"
        + f"class Native{name.title().replace('_', '')}:\n"
        + "    @pl.function(type=pl.FunctionType.Orchestration)\n"
        + "    def main(\n"
        + "        self,\n"
        + parameters
        + "\n"
        + f"    ) -> {ast.unparse(function.returns)}:\n"
        + f"        {output_name} = {emitted.function_name}({argument_names})\n"
        + f"        return {output_name}\n"
    )
    caller_path = tmp_path / f"native_{name}.py"
    caller_path.write_text(caller, encoding="utf-8")

    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    compiled = ir.compile(
        pl.loads(str(caller_path)),
        output_dir=str(tmp_path / f"compiled_{name}"),
        dump_passes=False,
        skip_ptoas=True,
    )
    pto = [
        path.read_text(encoding="utf-8") for path in compiled.output_dir.rglob("*.pto")
    ]
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(orchestration_files) == 1
    return pto, orchestration_files[0].read_text(encoding="utf-8")


def test_callable_multi_step_cube_preserves_ordered_gm_cut_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = export_and_normalize(
        _ChainedMatmul(),
        (
            torch.zeros(64, 128),
            torch.zeros(128, 96),
            torch.zeros(96, 80),
        ),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    emitted = emit_pypto_callable(
        graph, solved.regions[0], function_name="generated_cube_chain"
    )

    pto, orchestration = _compile_callable_in_native_orchestration(
        emitted, tmp_path, monkeypatch, name="cube_chain"
    )
    assert len(pto) == 2
    assert orchestration.count("rt_submit_aic_task(") == 2
    produced = set(re.findall(r"params_t0\.add_output\(([^)]+)\);", orchestration))
    consumed = set(re.findall(r"params_t1\.add_input\(([^)]+)\);", orchestration))
    assert len(produced & consumed) == 1
    assert orchestration.index("rt_submit_aic_task(0") < orchestration.index(
        "rt_submit_aic_task(1"
    )


def test_completion_aware_callable_feeds_native_task_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ChainedMatmul()
    args = (
        torch.zeros(64, 128),
        torch.zeros(128, 96),
        torch.zeros(96, 80),
    )
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    emitted = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name="fusebox_completion_probe",
        expose_completion_task=True,
    )
    assert emitted.completion_task is not None
    tree = ast.parse(emitted.source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    assert isinstance(function.returns, ast.Subscript)
    assert isinstance(function.returns.slice, ast.Tuple)
    tensor_return = function.returns.slice.elts[0]
    module_name = "generated_completion_probe"
    (tmp_path / f"{module_name}.py").write_text(emitted.source, encoding="utf-8")
    parameters = "\n".join(
        f"        {argument.arg}: {ast.unparse(argument.annotation)},"
        for argument in function.args.args
        if argument.annotation is not None
    )
    arguments = ", ".join(argument.arg for argument in function.args.args)
    output = emitted.output_arguments[0].name
    caller = f"""from {module_name} import {emitted.function_name}
import pypto.language as pl


@pl.program
class NativeCompletionProbe:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
{parameters}
    ) -> {ast.unparse(tensor_return)}:
        {output}, projection_done = {emitted.function_name}({arguments})
        completion_fence = pl.system.task_dummy(deps=[projection_done])
        return {output}
"""
    caller_path = tmp_path / "native_completion_probe.py"
    caller_path.write_text(caller, encoding="utf-8")

    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    compiled = ir.compile(
        pl.loads(str(caller_path)),
        output_dir=str(tmp_path / "compiled_completion_probe"),
        dump_passes=False,
        skip_ptoas=True,
    )
    orchestration = next(
        (compiled.output_dir / "orchestration").glob("*.cpp")
    ).read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_") >= 3
    assert "set_dependencies(" in orchestration


def test_callable_split_k_preserves_two_phase_dependency_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = export_and_normalize(
        _DeepKMatmul(),
        (torch.zeros(128, 8192), torch.zeros(8192, 128)),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    region = solved.regions[0]
    typed = scheduled_region(region).steps[0]
    assert isinstance(typed.plan, CubeKernelPlan)
    assert typed.plan.split_k > 1
    emitted = emit_pypto_callable(
        graph, region, function_name="generated_split_k_matmul"
    )

    pto, orchestration = _compile_callable_in_native_orchestration(
        emitted, tmp_path, monkeypatch, name="split_k"
    )
    assert len(pto) == 2
    assert orchestration.count("rt_submit_") == 2
    assert orchestration.count("set_dependencies(") == 1
    assert any("atomic_add" in source for source in pto)


def test_callable_mixed_attention_preserves_one_split_spmd_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = export_and_normalize(
        _AttentionCore(),
        (
            torch.zeros(96, 64),
            torch.zeros(64, 64),
            torch.zeros(64, 128),
        ),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    region = solved.regions[0]
    plan = scheduled_region(region).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    emitted = emit_pypto_callable(graph, region, function_name="generated_attention")

    pto, orchestration = _compile_callable_in_native_orchestration(
        emitted, tmp_path, monkeypatch, name="mixed_attention"
    )
    assert len(pto) == 1
    assert "pto.kernel_kind = #pto.kernel_kind<cube>" in pto[0]
    assert "pto.kernel_kind = #pto.kernel_kind<vector>" in pto[0]
    assert orchestration.count("rt_submit_task(") == 1
    assert f"launch_spec.set_block_num({plan.active_groups});" in orchestration


def test_callable_dense_swiglu_preserves_the_generic_mixed_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = export_and_normalize(
        _DenseSwiGlu(),
        (
            torch.zeros(128, 64, dtype=torch.bfloat16),
            torch.zeros(64, 128, dtype=torch.bfloat16),
            torch.zeros(64, 128, dtype=torch.bfloat16),
            torch.zeros(128, 64, dtype=torch.bfloat16),
        ),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    region = solved.regions[0]
    plan = scheduled_region(region).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    emitted = emit_pypto_callable(graph, region, function_name="generated_dense_swiglu")

    try:
        pto, orchestration = _compile_callable_in_native_orchestration(
            emitted, tmp_path, monkeypatch, name="dense_swiglu"
        )
    except RuntimeError as error:
        if "MemoryReuse cannot reconcile divergent L0C accumulator buffers" in str(
            error
        ):
            pytest.xfail(
                "recent PyPTO main still lacks the nested-accumulator join repair"
            )
        raise
    assert len(pto) == 1
    assert pto[0].count("pto.tpush_to_aiv") >= 2
    assert "pto.tpush_to_aic" in pto[0]
    assert orchestration.count("rt_submit_task(") == 1
    assert f"launch_spec.set_block_num({plan.active_groups});" in orchestration


def test_callable_runtime_valid_shape_lowers_without_dynamic_physical_tiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = build_pr2335_examples()["pr2335_rms_norm"]
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    emitted = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name="generated_rms_norm_chunk",
        runtime_valid_shape=RuntimeValidShapeSpec(),
    )

    pto, orchestration = _compile_callable_in_native_orchestration(
        emitted, tmp_path, monkeypatch, name="runtime_valid_rms"
    )
    assert len(pto) == 1
    assert re.search(r"valid_row = %\d+", pto[0])
    assert "v_row=?" in pto[0]
    assert "rows=?" not in pto[0]
    assert "outs(%output" in pto[0]
    assert "!pto.partition_tensor_view<?x512xf32>" in pto[0]
    assert orchestration.count("rt_submit_aiv_task(") == 1


@pytest.mark.parametrize(
    ("name", "runtime_rows", "expected_submit"),
    [
        ("qwen3_rms_norm_chunk", True, "rt_submit_aiv_task("),
        ("qwen3_lm_head_chunk", False, "rt_submit_aic_task("),
    ],
)
def test_callable_qwen_static_components_lower_inside_native_orchestration(
    name: str,
    runtime_rows: bool,
    expected_submit: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = build_qwen_examples()[name]
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    emitted = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name=f"generated_{name}",
        runtime_valid_shape=RuntimeValidShapeSpec() if runtime_rows else None,
    )

    pto, orchestration = _compile_callable_in_native_orchestration(
        emitted, tmp_path, monkeypatch, name=name
    )
    assert len(pto) == 1
    assert orchestration.count(expected_submit) == 1
    if name == "qwen3_lm_head_chunk":
        assert "pto.tmatmul" in pto[0]
        assert re.search(r"loc=right, dtype=bf16, rows=160, cols=32", pto[0])
        assert re.search(r"loc=right, dtype=bf16, rows=32, cols=32", pto[0])


def test_callable_connected_qwen_v2c_lowers_as_one_mixed_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = build_qwen_examples()["qwen3_rms_lm_head"]
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.whole_graph_codegen_ready
    plan = scheduled_region(solved.regions[0]).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    assert len(plan.fifos) == 1
    assert plan.fifos[0].direction.value == "vector_to_cube"
    emitted = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name="generated_qwen_rms_lm_head",
    )

    pto, orchestration = _compile_callable_in_native_orchestration(
        emitted, tmp_path, monkeypatch, name="qwen_rms_lm_head"
    )
    assert len(pto) == 1
    assert "pto.kernel_kind = #pto.kernel_kind<cube>" in pto[0]
    assert "pto.kernel_kind = #pto.kernel_kind<vector>" in pto[0]
    assert "pto.tpush_to_aic" in pto[0]
    assert "pto.tpop_from_aiv" in pto[0]
    assert not [
        line
        for line in pto[0].splitlines()
        if "pto.tmov" in line and line.count("loc=mat") >= 2
    ], "a V2C pop already lands in Mat; a redundant Mat -> Mat move is illegal"
    assert orchestration.count("rt_submit_task(") == 1
    assert f"launch_spec.set_block_num({plan.active_groups});" in orchestration


def test_maximal_qwen_callable_composes_in_native_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fusebox sees the full static output head before choosing its schedule."""

    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    sources = emit_hybrid_qwen_output_head(_solver(), solver_workers=2)
    for name, source in sources.files().items():
        (tmp_path / name).write_text(source, encoding="utf-8")
    caller_path = tmp_path / "native_qwen_rms_lm_head.py"
    caller_path.write_text(sources.orchestration_source, encoding="utf-8")

    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    compiled = ir.compile(
        pl.loads(str(caller_path)),
        output_dir=str(tmp_path / "native_qwen_rms_lm_head"),
        dump_passes=False,
        skip_ptoas=True,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 1
    assert len(orchestration_files) == 1
    orchestration = orchestration_files[0].read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_task(") == 1
    assert "intermediate_tensor" not in orchestration
    assert "add_output(ext_output)" in orchestration
    assert "add_inout(" not in orchestration


def test_production_qwen_output_head_preserves_dynamic_native_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pypto_lib_root = os.environ.get("PTO_FUSEBOX_PYPTO_LIB_ROOT")
    if pypto_lib_root is None:
        pytest.skip("set PTO_FUSEBOX_PYPTO_LIB_ROOT to a pypto-lib checkout")
    model_dir = Path(pypto_lib_root) / "models" / "qwen3_14b"
    overlay = emit_production_qwen_output_head_overlay(
        _solver(),
        native_decode_source=(model_dir / "decode_fwd.py").read_text(encoding="utf-8"),
        solver_workers=2,
    )
    (tmp_path / f"{overlay.module_name}.py").write_text(
        overlay.source,
        encoding="utf-8",
    )
    caller = f"""from {overlay.module_name} import rms_lm_head_fp32
from config import QWEN3_14B_DIMS as D
import pypto.language as pl


@pl.program
class ProductionQwenOutputHead:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        hidden_states: pl.Tensor[[16, 5120], pl.FP32],
        final_norm_weight: pl.Tensor[[1, 5120], pl.FP32],
        lm_head_weight: pl.Tensor[[152064, 5120], pl.BF16],
        row_offset: pl.Scalar[pl.INDEX],
        valid_rows: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[D.batch, 152064], pl.FP32]],
    ) -> pl.Tensor[[D.batch, 152064], pl.FP32]:
        out = rms_lm_head_fp32(
            hidden_states,
            final_norm_weight,
            lm_head_weight,
            out,
            row_offset,
            valid_rows,
        )
        return out
"""
    caller_path = tmp_path / "production_qwen_output_head.py"
    caller_path.write_text(caller, encoding="utf-8")

    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    monkeypatch.delitem(sys.modules, "config", raising=False)
    monkeypatch.syspath_prepend(str(model_dir))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    compiled = ir.compile(
        pl.loads(str(caller_path)),
        output_dir=str(tmp_path / "compiled_production_qwen_output_head"),
        dump_passes=False,
        skip_ptoas=True,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration = next(
        (compiled.output_dir / "orchestration").glob("*.cpp")
    ).read_text(encoding="utf-8")
    assert len(pto_files) == 3
    assert orchestration.count("rt_submit_aiv_task") == 2
    assert orchestration.count("rt_submit_aic_task") == 1
    assert "gm_pipe_buffer_" not in orchestration


def test_callable_deepseek_mtp_projection_compiles_as_one_branched_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = build_deepseek_examples()["deepseek_v4_mtp_projection"]
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.whole_graph_codegen_ready
    schedule = scheduled_region(solved.regions[0])
    assert [step.kind for step in schedule.steps] == [KernelKind.MIXED]
    plan = schedule.steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    assert plan.protocol is MixedCrossCoreProtocol.BRANCHED_ROUND_TRIP_BUNDLE
    assert tuple(stage.engine for stage in plan.stages) == (
        MixedEngine.VECTOR,
        MixedEngine.CUBE,
        MixedEngine.VECTOR,
        MixedEngine.CUBE,
        MixedEngine.VECTOR,
    )
    assert tuple(fifo.direction for fifo in plan.fifos) == (
        MixedTransferDirection.VECTOR_TO_CUBE,
        MixedTransferDirection.VECTOR_TO_CUBE,
        MixedTransferDirection.CUBE_TO_VECTOR,
        MixedTransferDirection.CUBE_TO_VECTOR,
    )
    emitted = emit_pypto_callable(
        graph,
        solved.regions[0],
        function_name="generated_deepseek_mtp_projection",
    )

    pto, orchestration = _compile_callable_in_native_orchestration(
        emitted, tmp_path, monkeypatch, name="deepseek_mtp_projection"
    )
    assert len(pto) == 1
    assert not [
        line
        for program in pto
        for line in program.splitlines()
        if "pto.tmov" in line and line.count("loc=mat") >= 2
    ], "a V2C pop already lands in Mat; a redundant Mat -> Mat move is illegal"
    _assert_single_spmd_orchestration(orchestration, plan.active_groups)
    produced = {
        tensor: int(task)
        for task, tensor in re.findall(
            r"params_t(\d+)\.add_output\(([^)]+)\);",
            orchestration,
        )
        if not tensor.startswith("ext_output")
    }
    consumed = {
        tensor: int(task)
        for task, tensor in re.findall(
            r"params_t(\d+)\.add_input\(([^)]+)\);",
            orchestration,
        )
    }
    solver_cuts = set(produced) & set(consumed)
    compiler_pipe_buffers = {
        tensor for tensor in produced if tensor.startswith("gm_pipe_buffer_")
    }
    external_inputs = set(consumed) - set(produced)
    assert set(produced) == solver_cuts | compiler_pipe_buffers
    assert not compiler_pipe_buffers & set(consumed)
    assert all(tensor.startswith("ext_arg_") for tensor in external_inputs)
    assert not solver_cuts


@pytest.mark.parametrize(
    ("name", "expected_pto_count"),
    [("paged_attention_static_regions", 2), ("moe_static_regions", 3)],
)
def test_static_bundle_callables_lower_independently_around_native_boundaries(
    name: str,
    expected_pto_count: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = build_boundary_examples()[name]
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved
    assert not solved.whole_graph_supported
    bundle = emit_pypto_static_bundle(
        graph, solved, function_prefix=f"generated_{name}"
    )

    pto_count = 0
    for index, emitted in enumerate(bundle.callables):
        pto, orchestration = _compile_callable_in_native_orchestration(
            emitted,
            tmp_path,
            monkeypatch,
            name=f"{name}_region_{index}",
        )
        pto_count += len(pto)
        assert orchestration.count("rt_submit_") >= 1
    assert pto_count == expected_pto_count


class _SumOfSquares(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sum(value * value, dim=-1, keepdim=True)


class _MatmulWithTail(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.mm(lhs, rhs)


class _DeepKMatmul(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.mm(lhs, rhs)


class _SplitCubeChain(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        intermediate = torch.mm(lhs, middle)
        return torch.mm(intermediate, rhs, out_dtype=torch.float32)


class _ChainedMatmul(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(torch.mm(lhs, middle), rhs)


class _DiamondMatmul(nn.Module):
    def forward(
        self,
        shared: torch.Tensor,
        lhs_weight: torch.Tensor,
        rhs_weight: torch.Tensor,
    ) -> torch.Tensor:
        lhs = torch.mm(shared, lhs_weight)
        rhs = torch.mm(shared, rhs_weight)
        return torch.mm(lhs, rhs)


class _FanoutMatmul(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        first_rhs: torch.Tensor,
        second_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.mm(lhs, middle)
        return torch.mm(shared, first_rhs), torch.mm(shared, second_rhs)


class _WideSoftmax(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.softmax(value, dim=-1)


class _Silu(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.reciprocal(torch.exp(value * -1.0) + 1.0)


class _LayerNorm(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        centered = value - value.mean(dim=-1, keepdim=True)
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(variance + 1.0e-5)
        return normalized * gamma + beta


class _StreamedReduction(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sum(value, dim=-1, keepdim=True)


class _StreamedNormalize(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value / torch.sum(value, dim=-1, keepdim=True)


class _NamingCollision(nn.Module):
    def forward(self, pl: torch.Tensor) -> torch.Tensor:
        return torch.exp(pl)


class _C2VEpilogue(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(value, weight) + bias


class _V2COnly(nn.Module):
    def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.exp(value), weight)


class _V2COnlyRhs(nn.Module):
    def forward(
        self, lhs: torch.Tensor, value: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.mm(lhs, torch.exp(value + bias))


class _V2CSharedLhs(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.exp(value), value)


class _V2CSharedRhs(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.mm(value, torch.exp(value))


class _StreamingSoftmaxPv(nn.Module):
    def forward(self, scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.mm(torch.softmax(scores, dim=-1), value)


class _V2CDualRole(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        produced = torch.exp(value)
        return torch.mm(produced, produced)


class _AttentionCore(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.mm(query, key.t())
        probabilities = torch.softmax(scores, dim=-1)
        return torch.mm(probabilities, value)


class _AttentionResidual(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(torch.mm(query, key.t()), dim=-1)
        return torch.mm(probabilities, value) + residual


class _RhsRoundTripPointwise(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        first_lhs: torch.Tensor,
        first_rhs: torch.Tensor,
    ) -> torch.Tensor:
        reply = torch.exp(torch.mm(first_lhs, first_rhs))
        return torch.exp(torch.mm(lhs, reply))


class _DenseSwiGlu(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.mm(value, gate_weight, out_dtype=torch.float32)
        up = torch.mm(value, up_weight, out_dtype=torch.float32)
        activation = (gate * torch.reciprocal(torch.exp(-gate) + 1.0) * up).to(
            torch.bfloat16
        )
        return torch.mm(activation, down_weight, out_dtype=torch.float32)


class _FeatureBlend(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        first_weight: torch.Tensor,
        second_weight: torch.Tensor,
        sink_weight: torch.Tensor,
    ) -> torch.Tensor:
        first = torch.mm(lhs, first_weight, out_dtype=torch.float32)
        second = torch.mm(lhs, second_weight, out_dtype=torch.float32)
        blended = (first + second).to(torch.bfloat16)
        return torch.mm(blended, sink_weight, out_dtype=torch.float32)


def _solver() -> Path:
    configured = os.environ.get("PTO_FUSEBOX_TEST_SOLVER")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "build" / "mlsys_mixed"


def _assert_static_vector_frames(pto: str) -> None:
    assert re.search(r"partition_tensor_view<[^>]*\?", pto) is None
    assert re.search(r"valid_(?:row|col) = %arg[0-9]+", pto) is None


def _assert_single_spmd_orchestration(source: str, work_units: int) -> None:
    submits = re.findall(r"\brt_submit_ai[cv]_task\(", source)
    assert len(submits) == 1
    assert source.count("launch_spec.set_block_num(") == 1
    assert f"launch_spec.set_block_num({work_units});" in source
    assert "region_index" not in source


def _pto_pipe_ids(pto: str, op_name: str) -> set[int]:
    lines = [line for line in pto.splitlines() if f"pto.{op_name}" in line]
    ids: set[int] = set()
    for line in lines:
        match = re.search(r"\bid = (-?[0-9]+)", line)
        # PyPTO's automatic pipe is the implicit default channel and therefore
        # carries no public id= attribute. Explicit low-level pipes still do.
        ids.add(int(match.group(1)) if match is not None else 0)
    return ids


def _compile_source(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    skip_ptoas: bool,
) -> tuple[str, int]:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")

    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved == 1
    assert len(solved.regions) == 1
    plan = scheduled_region(solved.regions[0]).steps[0].plan
    assert isinstance(plan, (CubeKernelPlan, VectorKernelPlan))

    source = emit_pypto_region(graph, solved.regions[0], program_name=name).source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / name),
        dump_passes=False,
        skip_ptoas=skip_ptoas,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    assert len(pto_files) == 1
    generated_cpp = list((compiled.output_dir / "ptoas").glob("*.cpp"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(orchestration_files) == 1
    _assert_single_spmd_orchestration(
        orchestration_files[0].read_text(encoding="utf-8"),
        plan.work_units,
    )
    return pto_files[0].read_text(encoding="utf-8"), len(generated_cpp)


def _compile_mixed_source(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    forced_active_groups: int | None = None,
) -> tuple[str, str, MixedKernelPlan]:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")

    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved == 1
    assert len(solved.regions) == 1
    region = solved.regions[0]
    if forced_active_groups is not None:
        assert region.solution is not None
        solution = copy.deepcopy(region.solution)
        step = solution["steps"][0]
        descriptor = step["plan"]
        protocol = descriptor["protocol"]
        assert protocol in {"one_way", "single_round_trip_bundle"}
        spatial_tiles = descriptor["spatial_tiles"]
        assert spatial_tiles % forced_active_groups == 0
        trips = spatial_tiles // forced_active_groups
        descriptor["active_groups"] = forced_active_groups
        descriptor["min_trips_per_group"] = trips
        descriptor["max_trips_per_group"] = trips
        overlap = trips >= 2
        descriptor["pipeline_stages"] = (
            3
            if protocol == "single_round_trip_bundle" and overlap
            else 2
            if overlap
            else 1
        )
        descriptor["requested_skew_depth"] = (
            2
            if protocol == "single_round_trip_bundle" and overlap
            else 1
            if overlap
            else 0
        )
        descriptor["model_overlap_granted"] = overlap
        descriptor["overlap_implementable"] = overlap
        descriptor["pipeline_fill_absorbed"] = (
            protocol == "single_round_trip_bundle" and overlap
        )
        step["launch"]["cores"] = forced_active_groups * (
            1 + descriptor["vector_lanes"]
        )
        region = replace(region, solution=solution)
    plan = scheduled_region(region).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)

    source = emit_pypto_region(graph, region, program_name=name).source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / name),
        dump_passes=False,
        skip_ptoas=True,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 1
    assert len(orchestration_files) == 1
    orchestration = orchestration_files[0].read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_task(") == 1
    assert orchestration.count("launch_spec.set_block_num(") == 1
    assert f"launch_spec.set_block_num({plan.active_groups});" in orchestration
    assert "for (region_index" not in orchestration
    return source, pto_files[0].read_text(encoding="utf-8"), plan


def test_nonstreaming_v2c_stage_two_lowers_through_pypto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, pto, plan = _compile_mixed_source(
        "mixed_v2c_stage_two",
        _V2COnly(),
        (torch.zeros(96, 64), torch.zeros(64, 128)),
        tmp_path,
        monkeypatch,
        forced_active_groups=1,
    )
    assert plan.spatial_tiles == 2
    assert plan.active_groups == 1
    assert plan.max_trips_per_group == 2
    assert plan.pipeline_stages == 2
    assert "pl.pipeline(2, stage=2" in source
    assert "pto.tpush_to_aic" in pto
    assert "pto.tpop_from_aiv" in pto
    assert "pto.tfree_from_aiv" in pto


def test_one_trip_cvc_uses_a_serial_loop_and_fits_vec_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, plan = _compile_mixed_source(
        "mixed_attention_one_trip",
        _AttentionCore(),
        (
            torch.zeros(768, 64),
            torch.zeros(64, 64),
            torch.zeros(64, 128),
        ),
        tmp_path,
        monkeypatch,
        forced_active_groups=12,
    )
    assert plan.protocol is MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE
    assert plan.spatial_tiles == plan.active_groups == 12
    assert plan.min_trips_per_group == plan.max_trips_per_group == 1
    assert plan.pipeline_stages == 1
    assert plan.requested_skew_depth == 0
    assert not plan.overlap_implementable
    assert "pl.range(1, init_values=" in source
    assert "stage=3" not in source
    fifo_bytes = sum(
        fifo.reserved_bytes
        for fifo in plan.fifos
        if fifo.direction.value == "cube_to_vector"
    )
    assert plan.vector_stage_peak_ub_bytes + fifo_bytes == 114816
    assert plan.vector_stage_peak_ub_bytes + fifo_bytes <= 188416


@pytest.mark.parametrize(
    ("name", "module", "args"),
    [
        (
            "mixed_c2v",
            _C2VEpilogue(),
            (
                torch.zeros(32, 64),
                torch.zeros(64, 32),
                torch.zeros(1, 32),
            ),
        ),
        (
            "mixed_c2v_streamed_groups",
            _C2VEpilogue(),
            (
                torch.zeros(384, 64),
                torch.zeros(64, 256),
                torch.zeros(1, 256),
            ),
        ),
        (
            "mixed_v2c_lhs",
            _V2COnly(),
            (torch.zeros(96, 64), torch.zeros(64, 128)),
        ),
        (
            "mixed_v2c_rhs",
            _V2COnlyRhs(),
            (
                torch.zeros(96, 64),
                torch.zeros(64, 128),
                torch.zeros(1, 128),
            ),
        ),
        (
            "mixed_v2c_shared_lhs",
            _V2CSharedLhs(),
            (torch.zeros(64, 64),),
        ),
        (
            "mixed_v2c_shared_rhs",
            _V2CSharedRhs(),
            (torch.zeros(64, 64),),
        ),
        (
            "mixed_streaming_softmax_pv",
            _StreamingSoftmaxPv(),
            (torch.zeros(16, 4096), torch.zeros(4096, 64)),
        ),
        (
            "mixed_v2c_dual_role",
            _V2CDualRole(),
            (torch.zeros(64, 64),),
        ),
        (
            "mixed_attention",
            _AttentionCore(),
            (
                torch.zeros(96, 64),
                torch.zeros(64, 64),
                torch.zeros(64, 128),
            ),
        ),
        (
            "mixed_attention_streamed_groups",
            _AttentionCore(),
            (
                torch.zeros(384, 64),
                torch.zeros(64, 64),
                torch.zeros(64, 128),
            ),
        ),
        (
            "mixed_dense_swiglu",
            _DenseSwiGlu(),
            (
                torch.zeros(128, 64, dtype=torch.bfloat16),
                torch.zeros(64, 128, dtype=torch.bfloat16),
                torch.zeros(64, 128, dtype=torch.bfloat16),
                torch.zeros(128, 64, dtype=torch.bfloat16),
            ),
        ),
        (
            "mixed_generic_feature_blend",
            _FeatureBlend(),
            (
                torch.zeros(128, 64, dtype=torch.bfloat16),
                torch.zeros(64, 128, dtype=torch.bfloat16),
                torch.zeros(64, 128, dtype=torch.bfloat16),
                torch.zeros(128, 64, dtype=torch.bfloat16),
            ),
        ),
        (
            "mixed_attention_residual",
            _AttentionResidual(),
            (
                torch.zeros(96, 64),
                torch.zeros(64, 64),
                torch.zeros(64, 128),
                torch.zeros(96, 128),
            ),
        ),
        (
            "mixed_rhs_round_trip_pointwise",
            _RhsRoundTripPointwise(),
            (
                torch.zeros(96, 64),
                torch.zeros(64, 64),
                torch.zeros(64, 32),
            ),
        ),
    ],
)
def test_mixed_source_lowers_through_the_pypto_split_pipeline(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        source, pto, plan = _compile_mixed_source(
            name, module, args, tmp_path, monkeypatch
        )
    except RuntimeError as error:
        if name == "mixed_dense_swiglu" and (
            "MemoryReuse cannot reconcile divergent L0C accumulator buffers"
            in str(error)
        ):
            pytest.xfail(
                "recent PyPTO main still lacks the nested-accumulator join repair"
            )
        raise

    assert "pl.split(pl.SplitMode.UP_DOWN" in source
    assert "pl.cross_core_slot(" not in source
    assert source.count("pl.cross_core_pipe(") == len(plan.fifos)
    assert pto.count("pto.kernel_kind = #pto.kernel_kind<cube>") == 1
    assert pto.count("pto.kernel_kind = #pto.kernel_kind<vector>") == 1
    assert pto.count("pto.aic_initialize_pipe") == len(plan.fifos)
    assert pto.count("pto.aiv_initialize_pipe") == len(plan.fifos)
    for fifo in plan.fifos:
        direction = 1 if fifo.direction.value == "cube_to_vector" else 2
        assert f"dir_mask = {direction}" in pto
        assert f"slot_num = {fifo.slot_count}" in pto
        assert f"slot_size = {fifo.slot_bytes}" in pto
    expected_c2v_ids = {
        fifo.pipe_id for fifo in plan.fifos if fifo.direction.value == "cube_to_vector"
    }
    expected_v2c_ids = {
        fifo.pipe_id for fifo in plan.fifos if fifo.direction.value == "vector_to_cube"
    }
    if any(fifo.direction.value == "cube_to_vector" for fifo in plan.fifos):
        assert _pto_pipe_ids(pto, "tpush_to_aiv")
        assert _pto_pipe_ids(pto, "tpush_to_aiv") == _pto_pipe_ids(pto, "tpop_from_aic")
        assert _pto_pipe_ids(pto, "tpop_from_aic") == _pto_pipe_ids(
            pto, "tfree_from_aic"
        )
        assert _pto_pipe_ids(pto, "tpush_to_aiv") == expected_c2v_ids
    if any(fifo.direction.value == "vector_to_cube" for fifo in plan.fifos):
        assert _pto_pipe_ids(pto, "tpush_to_aic")
        assert _pto_pipe_ids(pto, "tpush_to_aic") == _pto_pipe_ids(pto, "tpop_from_aiv")
        assert _pto_pipe_ids(pto, "tpop_from_aiv") == _pto_pipe_ids(
            pto, "tfree_from_aiv"
        )
        assert _pto_pipe_ids(pto, "tpush_to_aic") == expected_v2c_ids
    if name == "mixed_c2v":
        assert "pto.tcolexpandadd" in pto
        assert "pto.tadd" not in pto
    if name == "mixed_c2v_streamed_groups":
        assert plan.spatial_tiles == 24
        assert plan.active_groups == 6
        assert plan.max_trips_per_group == 4
        assert plan.pipeline_stages == 2
        assert plan.requested_skew_depth == 1
        assert plan.overlap_implementable
        assert "pl.pipeline(4, stage=2" in source
    if name == "mixed_attention_streamed_groups":
        assert plan.spatial_tiles == 4
        assert plan.active_groups == 4
        assert plan.max_trips_per_group == 1
        assert plan.pipeline_stages == 1
        assert plan.requested_skew_depth == 0
        assert not plan.overlap_implementable
        assert "pl.range(1, init_values=" in source


def test_multi_round_trip_source_lowers_to_ordered_two_trip_loops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, pto, plan = _compile_mixed_source(
        "mixed_attention_residual_two_trips",
        _AttentionResidual(),
        (
            torch.zeros(3072, 64),
            torch.zeros(32, 64),
            torch.zeros(32, 64),
            torch.zeros(3072, 64),
        ),
        tmp_path,
        monkeypatch,
    )

    assert plan.protocol.value == "multi_round_trip_sequential"
    assert plan.max_trips_per_group == 2
    assert "pl.range(2, init_values=" in source
    assert "pl.pipeline(" not in source
    # PyPTO outlines one ordered loop per core kind.  It must not convert this
    # unsupported second round trip into a skewed cross-core pipeline.
    assert pto.count("scf.for") == 2
    assert "scf.pipeline" not in pto
    assert _pto_pipe_ids(pto, "tpush_to_aiv") == _pto_pipe_ids(pto, "tpop_from_aic")
    assert _pto_pipe_ids(pto, "tpush_to_aic") == _pto_pipe_ids(pto, "tpop_from_aiv")


def test_mixed_source_rejects_duplicate_pypto_pipe_id() -> None:
    graph = export_and_normalize(
        _C2VEpilogue(),
        (
            torch.zeros(32, 64),
            torch.zeros(64, 32),
            torch.zeros(1, 32),
        ),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved == 1
    source = emit_pypto_region(
        graph, solved.regions[0], program_name="mixed_duplicate_slot"
    ).source
    parser_diagnostics = importlib.import_module("pypto.language.parser.diagnostics")
    pl = importlib.import_module("pypto.language")
    pipe = re.search(r"pl\.cross_core_pipe\([^)]*\)", source)
    assert pipe is not None
    duplicated = source.replace(pipe.group(0), f"{pipe.group(0)}, {pipe.group(0)}", 1)
    assert duplicated != source
    with pytest.raises(
        parser_diagnostics.ParserSyntaxError,
        match="Duplicate pl.cross_core_pipe pipe_id=0",
    ):
        pl.parse_program(duplicated)


@pytest.mark.parametrize(
    ("name", "module", "args", "expected_pto_op", "static_vector_frames"),
    [
        (
            "pointwise_chain",
            _PointwiseChain(),
            (torch.zeros(96, 320), torch.ones(96, 320)),
            "pto.texp",
            True,
        ),
        (
            "sum_of_squares",
            _SumOfSquares(),
            (torch.ones(128, 1024),),
            "pto.trowsum",
            True,
        ),
        (
            "matmul_with_tail",
            _MatmulWithTail(),
            (torch.zeros(64, 272), torch.zeros(272, 80)),
            "pto.tmatmul.acc",
            False,
        ),
        (
            "chained_matmul",
            _ChainedMatmul(),
            (
                torch.zeros(64, 128, dtype=torch.bfloat16),
                torch.zeros(128, 96, dtype=torch.bfloat16),
                torch.zeros(96, 80, dtype=torch.bfloat16),
            ),
            "pto.tinsert",
            False,
        ),
        (
            "wide_softmax",
            _WideSoftmax(),
            (torch.zeros(32, 8192),),
            "pto.trowmax",
            True,
        ),
        (
            "silu",
            _Silu(),
            (torch.zeros(512, 256),),
            "pto.trecip",
            True,
        ),
        (
            "layer_norm",
            _LayerNorm(),
            (
                torch.zeros(512, 256),
                torch.ones(1, 256),
                torch.zeros(1, 256),
            ),
            "pto.trowsum",
            True,
        ),
        (
            "naming_collision",
            _NamingCollision(),
            (torch.zeros(64, 128),),
            "pto.texp",
            True,
        ),
        (
            "streamed_reduction",
            _StreamedReduction(),
            (torch.ones(5, 32771),),
            "pto.trowsum",
            True,
        ),
        (
            "streamed_normalize",
            _StreamedNormalize(),
            (torch.ones(5, 32771),),
            "pto.trowsum",
            True,
        ),
    ],
)
def test_generated_source_compiles_through_pypto_and_ptoas(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    expected_pto_op: str,
    static_vector_frames: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pto, generated_cpp = _compile_source(
        name,
        module,
        args,
        tmp_path,
        monkeypatch,
        skip_ptoas=False,
    )
    assert generated_cpp == 1
    assert expected_pto_op in pto
    if static_vector_frames:
        _assert_static_vector_frames(pto)


class _MultiPassVectorTarget(Ascend910BTarget):
    """Keep the complete reduction chain selected for source integration."""

    def problem_fields(self) -> dict[str, object]:
        fields = super().problem_fields()
        fields["kernel_fill_cost"] = 100_000_000
        return fields


class _MultiPassReductionChain(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(
            torch.sum(value * value, dim=-1, keepdim=True) / value.shape[-1] + 1e-6
        )
        normalized = value * inverse_rms
        inverse_amax = torch.reciprocal(
            torch.amax(torch.abs(normalized), dim=-1, keepdim=True)
        )
        return normalized * inverse_amax


class _Int8ProjectionBranch(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        quantized = value.to(torch.float16).to(torch.int8)
        accumulator = torch.ops.aten._int_mm.default(quantized, weight.t())
        return accumulator.float() * scale


class _BranchedInt8Projection(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        lhs_weight: torch.Tensor,
        rhs_weight: torch.Tensor,
        lhs_scale: torch.Tensor,
        rhs_scale: torch.Tensor,
    ) -> torch.Tensor:
        lhs_quantized = lhs.to(torch.float16).to(torch.int8)
        rhs_quantized = rhs.to(torch.float16).to(torch.int8)
        lhs_accumulator = torch.ops.aten._int_mm.default(lhs_quantized, lhs_weight.t())
        rhs_accumulator = torch.ops.aten._int_mm.default(rhs_quantized, rhs_weight.t())
        return lhs_accumulator.float() * lhs_scale + rhs_accumulator.float() * rhs_scale


def test_general_multi_pass_vector_source_compiles_through_pypto_and_ptoas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(_MultiPassReductionChain(), (torch.ones(16, 32768),))
    solved = solve_graph(
        graph,
        target=_MultiPassVectorTarget(),
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved
    region = solved.regions[0]
    schedule = scheduled_region(region)
    assert len(schedule.steps) == 1
    plan = schedule.steps[0].plan
    assert isinstance(plan, VectorKernelPlan)
    assert plan.kind is VectorStreamKind.MULTI_PASS

    source = emit_pypto_region(
        graph, region, program_name="general_multi_pass_vector"
    ).source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "general_multi_pass_vector"),
        dump_passes=False,
        skip_ptoas=False,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 1
    assert len(orchestration_files) == 1
    pto = pto_files[0].read_text(encoding="utf-8")
    assert pto.count("pto.trowsum") >= 3
    assert pto.count("pto.trowmax") >= 3
    _assert_static_vector_frames(pto)
    _assert_single_spmd_orchestration(
        orchestration_files[0].read_text(encoding="utf-8"), plan.work_units
    )


def test_generic_int8_projection_round_trip_compiles_through_pypto_and_ptoas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _Int8ProjectionBranch(),
        (
            torch.ones(64, 256),
            torch.ones(128, 256, dtype=torch.int8),
            torch.ones(64, 1),
        ),
    )
    solved = solve_graph(
        graph,
        target=_MultiPassVectorTarget(),
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved
    region = solved.regions[0]
    schedule = scheduled_region(region)
    assert len(schedule.steps) == 1
    plan = schedule.steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    assert plan.protocol is MixedCrossCoreProtocol.SINGLE_ROUND_TRIP_BUNDLE
    assert tuple(stage.engine for stage in plan.stages) == (
        MixedEngine.VECTOR,
        MixedEngine.CUBE,
        MixedEngine.VECTOR,
    )
    assert tuple(fifo.direction for fifo in plan.fifos) == (
        MixedTransferDirection.VECTOR_TO_CUBE,
        MixedTransferDirection.CUBE_TO_VECTOR,
    )

    source = emit_pypto_region(
        graph, region, program_name="generic_int8_projection_round_trip"
    ).source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "generic_int8_projection_round_trip"),
        dump_passes=False,
        skip_ptoas=False,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 1
    assert len(orchestration_files) == 1
    pto = pto_files[0].read_text(encoding="utf-8")
    assert "pto.tmatmul" in pto
    assert "pto.tpush_to_aic" in pto
    assert "pto.tpop_from_aiv" in pto
    assert "pto.tpush_to_aiv" in pto
    assert "pto.tpop_from_aic" in pto
    _assert_single_spmd_orchestration(
        orchestration_files[0].read_text(encoding="utf-8"), plan.active_groups
    )


def test_generic_branched_int8_round_trip_compiles_through_pypto_and_ptoas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _BranchedInt8Projection(),
        (
            torch.ones(64, 256),
            torch.ones(64, 256),
            torch.ones(128, 256, dtype=torch.int8),
            torch.ones(128, 256, dtype=torch.int8),
            torch.ones(64, 1),
            torch.ones(64, 1),
        ),
    )
    solved = solve_graph(
        graph,
        target=_MultiPassVectorTarget(),
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved
    region = solved.regions[0]
    schedule = scheduled_region(region)
    assert len(schedule.steps) == 1
    plan = schedule.steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    assert plan.protocol is MixedCrossCoreProtocol.BRANCHED_ROUND_TRIP_BUNDLE
    assert tuple(stage.engine for stage in plan.stages) == (
        MixedEngine.VECTOR,
        MixedEngine.VECTOR,
        MixedEngine.CUBE,
        MixedEngine.CUBE,
        MixedEngine.VECTOR,
    )
    assert tuple(fifo.direction for fifo in plan.fifos) == (
        MixedTransferDirection.VECTOR_TO_CUBE,
        MixedTransferDirection.VECTOR_TO_CUBE,
        MixedTransferDirection.CUBE_TO_VECTOR,
        MixedTransferDirection.CUBE_TO_VECTOR,
    )

    source = emit_pypto_region(
        graph, region, program_name="generic_branched_int8_round_trip"
    ).source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "generic_branched_int8_round_trip"),
        dump_passes=False,
        skip_ptoas=False,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 1
    assert len(orchestration_files) == 1
    pto = pto_files[0].read_text(encoding="utf-8")
    assert pto.count("pto.tmatmul") == 2
    assert pto.count("pto.tpush_to_aic") >= 2
    assert pto.count("pto.tpop_from_aiv") >= 2
    assert pto.count("pto.tpush_to_aiv") >= 2
    assert pto.count("pto.tpop_from_aic") >= 2
    _assert_single_spmd_orchestration(
        orchestration_files[0].read_text(encoding="utf-8"), plan.active_groups
    )


def test_generic_int8_round_trip_static_tail_compiles_through_pypto_and_ptoas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _Int8ProjectionBranch(),
        (
            torch.ones(50, 256),
            torch.ones(128, 256, dtype=torch.int8),
            torch.ones(50, 1),
        ),
    )
    solved = solve_graph(
        graph,
        target=_MultiPassVectorTarget(),
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    assert solved.regions_solved
    region = solved.regions[0]
    plan = scheduled_region(region).steps[0].plan
    assert isinstance(plan, MixedKernelPlan)
    assert plan.m_partition.big * plan.m_partition.parts > 50

    source = emit_pypto_region(
        graph, region, program_name="generic_int8_round_trip_static_tail"
    ).source
    assert "region_row = pl.min(m_index * 32, 18)" in source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "generic_int8_round_trip_static_tail"),
        dump_passes=False,
        skip_ptoas=False,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 1
    assert len(orchestration_files) == 1
    assert "pto.tmatmul" in pto_files[0].read_text(encoding="utf-8")
    _assert_single_spmd_orchestration(
        orchestration_files[0].read_text(encoding="utf-8"), plan.active_groups
    )


@pytest.mark.parametrize(
    "name",
    [
        "pr2335_softmax_512x256",
        "pr2335_softmax_256x512",
        "pr2335_softmax_128x1024",
        "pr2335_softmax_32x8192",
        "pr2335_rms_norm",
        "pr2335_layer_norm",
        "pr2335_silu",
    ],
)
def test_pr2335_vector_surface_lowers_static_frames(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, args = build_pr2335_examples()[name]
    pto, _ = _compile_source(
        name,
        module,
        args,
        tmp_path,
        monkeypatch,
        skip_ptoas=True,
    )
    _assert_static_vector_frames(pto)


@pytest.mark.parametrize(
    "policy", ["aiv_zero_seed_then_atomic", "first_partial_then_atomic"]
)
def test_deep_k_split_protocol_lowers_as_two_dependency_linked_tasks(
    policy: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _DeepKMatmul(),
        (torch.zeros(128, 8192), torch.zeros(8192, 128)),
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    region = solved.regions[0]
    assert region.solution is not None
    solution = copy.deepcopy(region.solution)
    plan = solution["steps"][0]["plan"]
    assert plan["split_k"] > 1
    spatial = plan["spatial_tiles"]
    split = plan["split_k"]
    synchronization = max(
        plan["first_partial_then_atomic"]["synchronization_cycles"],
        plan["aiv_zero_seed_then_atomic"]["synchronization_cycles"],
    )
    if policy == "first_partial_then_atomic":
        plan["split_merge_policy"] = policy
        plan["first_partial_then_atomic"] = {
            "present": True,
            "first_work_units": spatial,
            "atomic_work_units": spatial * (split - 1),
            "synchronization_cycles": synchronization,
        }
        plan["aiv_zero_seed_then_atomic"] = {
            "present": False,
            "seed_work_units": 0,
            "atomic_work_units": 0,
            "seed_bytes": 0,
            "synchronization_cycles": 0.0,
        }
    else:
        sink = next(matmul for matmul in plan["matmuls"] if matmul["is_sink"])
        plan["split_merge_policy"] = policy
        plan["first_partial_then_atomic"] = {
            "present": False,
            "first_work_units": 0,
            "atomic_work_units": 0,
            "synchronization_cycles": 0.0,
        }
        plan["aiv_zero_seed_then_atomic"] = {
            "present": True,
            "seed_work_units": spatial,
            "atomic_work_units": spatial * split,
            "seed_bytes": spatial * sink["final_drain"]["bytes"],
            "synchronization_cycles": synchronization,
        }
    region = replace(region, solution=solution)
    typed = scheduled_region(region).steps[0]
    assert isinstance(typed.plan, CubeKernelPlan)
    source = emit_pypto_region(graph, region, program_name=f"deep_k_{policy}").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / policy),
        dump_passes=False,
        skip_ptoas=True,
    )
    pto_files = list(compiled.output_dir.rglob("*.pto"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == 2
    assert len(orchestration_files) == 1
    orchestration = orchestration_files[0].read_text(encoding="utf-8")
    submissions = re.findall(r"\brt_submit_(ai[cv])_task\(", orchestration)
    block_counts = [
        int(value)
        for value in re.findall(
            r"launch_spec\.set_block_num\(([0-9]+)\)", orchestration
        )
    ]
    assert orchestration.count("set_dependencies(") == 1
    dependency_position = orchestration.index("set_dependencies(")
    second_submit_position = orchestration.find("rt_submit_", dependency_position)
    assert second_submit_position > dependency_position
    pto_by_name = {path.stem: path.read_text(encoding="utf-8") for path in pto_files}
    effective_k = plan["matmuls"][0]["effective_contraction"]
    assert effective_k * split == 8192
    assert {share * effective_k for share in range(split)} == set(
        range(0, 8192, effective_k)
    )
    assert f"split_index * {effective_k}" in source
    if policy == "aiv_zero_seed_then_atomic":
        assert typed.plan.aiv_zero_seed_then_atomic.present
        assert "deps=[zero_seed_task]" in source
        assert submissions == ["aiv", "aic"]
        assert block_counts == [spatial, spatial * split]
        seed_pto = pto_by_name["region0000_cube_zero_seed"]
        atomic_pto = pto_by_name["region0000_cube_atomic_all"]
        assert "pto.texpands" in seed_pto and "pto.tstore" in seed_pto
        assert "atomic_add" not in seed_pto
        assert "atomicType = #pto<atomic_type atomic_add>" in atomic_pto
    else:
        assert typed.plan.first_partial_then_atomic.present
        assert "deps=[first_partial_task]" in source
        assert submissions == ["aic", "aic"]
        assert block_counts == [spatial, spatial * (split - 1)]
        first_pto = pto_by_name["region0000_cube_first"]
        atomic_pto = pto_by_name["region0000_cube_atomic_rest"]
        assert "pto.tstore" in first_pto and "atomic_add" not in first_pto
        assert "atomicType = #pto<atomic_type atomic_add>" in atomic_pto
    assert "for (region_index" not in orchestration


def test_deep_k_no_split_candidate_compiles_through_pypto_and_ptoas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _DeepKMatmul(),
        (torch.zeros(128, 8192), torch.zeros(8192, 128)),
    )
    regions = extract_solver_regions(graph)
    assert len(regions) == 1
    lowered = regions[0].lower(graph)
    unsolved = RegionSolveResult(
        region=regions[0],
        status="lowered",
        problem=lowered.problem,
        solution=None,
        solver_op_to_graph=lowered.solver_op_to_graph,
        solver_tensor_to_value=lowered.solver_tensor_to_value,
        diagnostics=regions[0].diagnostics,
    )
    sweep = enumerate_cube_plans(
        unsolved,
        sweep_binary=_solver().parent / "cube_plan_sweep",
    )
    no_split = next(
        candidate for candidate in sweep.candidates if candidate.id == "p1_q1_s1"
    )
    region = region_for_cube_candidate(unsolved, no_split)
    typed = scheduled_region(region).steps[0]
    assert isinstance(typed.plan, CubeKernelPlan)
    assert typed.launch.tile_k == typed.plan.matmuls[0].k_loop.l1_window_k == 512

    source = emit_pypto_region(graph, region, program_name="deep_k_no_split").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "deep_k_no_split"),
        dump_passes=False,
        skip_ptoas=False,
    )

    pto_files = list(compiled.output_dir.rglob("*.pto"))
    generated_cpp = list((compiled.output_dir / "ptoas").glob("*.cpp"))
    orchestration_files = list((compiled.output_dir / "orchestration").glob("*.cpp"))
    assert len(pto_files) == len(generated_cpp) == len(orchestration_files) == 1
    pto = pto_files[0].read_text(encoding="utf-8")
    orchestration = orchestration_files[0].read_text(encoding="utf-8")
    assert "atomic_add" not in pto
    assert orchestration.count("rt_submit_aic_task(") == 1
    assert "launch_spec.set_block_num(1);" in orchestration
    assert "set_dependencies(" not in orchestration


def test_split_cube_dag_lowers_upstream_and_sink_into_each_atomic_share(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _SplitCubeChain(),
        (
            torch.empty(16, 64, dtype=torch.bfloat16, device="meta"),
            torch.empty(64, 2048, dtype=torch.bfloat16, device="meta"),
            torch.empty(2048, 16, dtype=torch.bfloat16, device="meta"),
        ),
    )
    regions = extract_solver_regions(graph)
    assert len(regions) == 1
    lowered = regions[0].lower(graph)
    unsolved = RegionSolveResult(
        region=regions[0],
        status="lowered",
        problem=lowered.problem,
        solution=None,
        solver_op_to_graph=lowered.solver_op_to_graph,
        solver_tensor_to_value=lowered.solver_tensor_to_value,
        diagnostics=regions[0].diagnostics,
    )
    sweep = enumerate_cube_plans(
        unsolved,
        sweep_binary=_solver().parent / "cube_plan_sweep",
    )
    candidate = next(item for item in sweep.candidates if item.id == "p1_q1_s2")
    region = region_for_cube_candidate(unsolved, candidate)
    source = emit_pypto_region(graph, region, program_name="split_cube_chain").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "split_cube_chain"),
        dump_passes=False,
        skip_ptoas=True,
    )

    pto_files = list(compiled.output_dir.rglob("*.pto"))
    assert len(pto_files) == 2
    pto_by_name = {path.stem: path.read_text(encoding="utf-8") for path in pto_files}
    atomic = pto_by_name["region0000_cube_atomic_all"]
    assert atomic.count("pto.tmatmul") >= 2
    assert "atomicType = #pto<atomic_type atomic_add>" in atomic
    orchestration = next(
        (compiled.output_dir / "orchestration").glob("*.cpp")
    ).read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_aiv_task(") == 1
    assert orchestration.count("rt_submit_aic_task(") == 1
    assert orchestration.count("set_dependencies(") == 1
    assert "launch_spec.set_block_num(1);" in orchestration
    assert "launch_spec.set_block_num(2);" in orchestration


def test_cut_fp32_chain_compiles_as_two_dependency_linked_spmd_kernels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    module = _ChainedMatmul()
    args = (
        torch.zeros(64, 128),
        torch.zeros(128, 96),
        torch.zeros(96, 80),
    )
    graph = export_and_normalize(module, args)
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    region = solved.regions[0]
    schedule = scheduled_region(region)
    assert len(schedule.steps) == 2

    source = emit_pypto_region(graph, region, program_name="cut_fp32_chain").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "cut_fp32_chain"),
        dump_passes=False,
        skip_ptoas=True,
    )

    assert len(list(compiled.output_dir.rglob("*.pto"))) == 2
    orchestration = next(
        (compiled.output_dir / "orchestration").glob("*.cpp")
    ).read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_aic_task(") == 2
    assert "launch_spec.set_block_num(12);" in orchestration
    assert "launch_spec.set_block_num(4);" in orchestration
    assert "add_output(intermediate_tensor_2)" in orchestration
    assert "add_input(intermediate_tensor_2)" in orchestration
    assert "add_output(ext_output)" in orchestration
    assert "add_inout(intermediate_tensor_2)" not in orchestration
    assert "add_inout(ext_output)" not in orchestration


def test_cut_fp32_fanout_compiles_with_two_outputs_and_one_shared_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ir = importlib.import_module("pypto.ir")
    pl = importlib.import_module("pypto.language")
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    graph = export_and_normalize(
        _FanoutMatmul(), tuple(torch.zeros(64, 64) for _ in range(4))
    )
    solved = solve_graph(
        graph,
        solver_binary=_solver(),
        solver_workers=2,
        require_source_codegen=True,
    )
    region = solved.regions[0]
    schedule = scheduled_region(region)
    assert [step.solver_ops for step in schedule.steps] == [(0,), (1, 2)]

    source = emit_pypto_region(graph, region, program_name="cut_fp32_fanout").source
    compiled = ir.compile(
        pl.parse_program(source),
        output_dir=str(tmp_path / "cut_fp32_fanout"),
        dump_passes=False,
        skip_ptoas=True,
    )

    assert len(list(compiled.output_dir.rglob("*.pto"))) == 2
    orchestration = next(
        (compiled.output_dir / "orchestration").glob("*.cpp")
    ).read_text(encoding="utf-8")
    assert orchestration.count("rt_submit_aic_task(") == 2
    assert "add_output(ext_output_0)" in orchestration
    assert "add_output(ext_output_1)" in orchestration
    assert "add_output(intermediate_tensor_2)" in orchestration
    assert "add_input(intermediate_tensor_2)" in orchestration
    assert "add_inout(intermediate_tensor_2)" not in orchestration
    assert "add_inout(ext_output_0)" not in orchestration
    assert "add_inout(ext_output_1)" not in orchestration


@pytest.mark.parametrize(
    ("name", "module", "args", "expected_matmuls"),
    [
        (
            "retained_panel_matmul",
            _MatmulWithTail(),
            (
                torch.zeros(512, 64, dtype=torch.bfloat16),
                torch.zeros(64, 2048, dtype=torch.bfloat16),
            ),
            2,
        ),
        (
            "diamond_matmul",
            _DiamondMatmul(),
            tuple(torch.zeros(32, 32, dtype=torch.bfloat16) for _ in range(3)),
            3,
        ),
    ],
)
def test_non_split_cube_dag_source_compiles_through_pypto(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    expected_matmuls: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pto, _ = _compile_source(
        name,
        module,
        args,
        tmp_path,
        monkeypatch,
        skip_ptoas=True,
    )
    assert pto.count("pto.tmatmul ") == expected_matmuls
