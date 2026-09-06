"""Thin entry programs for genuine production-geometry PyPTO-lib controls.

These controls import and call the checked-out PyPTO-lib implementation.  They
contain no copied tensor algebra, schedule, or reduced geometry.  A device
campaign must put exactly the selected model directory on ``sys.path`` before
loading the returned source, so the unqualified imports used by PyPTO-lib keep
their native meaning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeControl:
    """One actual PyPTO-lib static callable and its minimal public wrapper."""

    model_name: str
    module_name: str
    callable_name: str
    program_name: str
    source: str
    tensor_specs_name: str | None
    golden_name: str | None
    expected_lowerable: bool = True
    expected_failure_substring: str | None = None


def _dspark() -> NativeControl:
    return NativeControl(
        model_name="deepseek_v4_flash_dspark",
        module_name="dspark_proj",
        callable_name="dspark_proj",
        program_name="native_dspark_projection",
        tensor_specs_name="build_tensor_specs",
        golden_name="golden_dspark_proj",
        source="""from dspark_proj import dspark_proj
import pypto.language as pl


@pl.jit
def native_dspark_projection(
    main_hidden: pl.Tensor[[16, 12288], pl.BF16],
    main_proj_w: pl.Tensor[[4096, 12288], pl.BF16],
    main_norm_w: pl.Tensor[[4096], pl.BF16],
    main_x: pl.Out[pl.Tensor[[16, 4096], pl.BF16]],
) -> pl.Tensor[[16, 4096], pl.BF16]:
    return dspark_proj(main_hidden, main_proj_w, main_norm_w, main_x)
""",
    )


def _mtp(model_name: str, hidden: int) -> NativeControl:
    class_suffix = "FlashMtp" if hidden == 4096 else "Pro"
    program_name = f"native_{class_suffix.lower()}_projection"
    return NativeControl(
        model_name=model_name,
        module_name="mtp_projection",
        callable_name="mtp_projection",
        program_name=program_name,
        tensor_specs_name="build_tensor_specs",
        golden_name="golden_mtp_projection",
        expected_lowerable=hidden != 7168,
        expected_failure_substring=(
            None if hidden != 7168 else "Vec buffer usage (230464 bytes)"
        ),
        source=f"""from mtp_projection import mtp_projection
import pypto.language as pl


@pl.jit
def {program_name}(
    hidden_states: pl.Tensor[[8, {hidden}], pl.BF16],
    prev_hidden_states: pl.Tensor[[8, 4, {hidden}], pl.FP32],
    enorm_w: pl.Tensor[[{hidden}], pl.FP32],
    hnorm_w: pl.Tensor[[{hidden}], pl.FP32],
    e_proj_w: pl.Tensor[[{hidden}, {hidden}], pl.INT8],
    e_proj_w_scale: pl.Tensor[[{hidden}], pl.FP32],
    e_proj_smooth: pl.Tensor[[{hidden}], pl.FP32],
    h_proj_w: pl.Tensor[[{hidden}, {hidden}], pl.INT8],
    h_proj_w_scale: pl.Tensor[[{hidden}], pl.FP32],
    h_proj_smooth: pl.Tensor[[{hidden}], pl.FP32],
    hidden_states_out: pl.Out[pl.Tensor[[8, 4, {hidden}], pl.FP32]],
) -> pl.Tensor[[8, 4, {hidden}], pl.FP32]:
    return mtp_projection(
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
""",
    )


def _qwen() -> NativeControl:
    return NativeControl(
        model_name="qwen3_14b",
        module_name="rms_lm_head",
        callable_name="rms_lm_head_fp32",
        program_name="native_qwen_output_head",
        tensor_specs_name=None,
        golden_name=None,
        source="""from config import QWEN3_14B_DIMS as D
from rms_lm_head import rms_lm_head_fp32
import pypto.language as pl


@pl.jit
def native_qwen_output_head(
    hidden_states: pl.Tensor[[16, 5120], pl.FP32],
    final_norm_weight: pl.Tensor[[1, 5120], pl.FP32],
    lm_head_weight: pl.Tensor[[152064, 5120], pl.BF16],
    row_offset: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    out: pl.Out[pl.Tensor[[D.batch, 152064], pl.FP32]],
) -> pl.Tensor[[D.batch, 152064], pl.FP32]:
    out.bind_dynamic(0, D.batch)
    return rms_lm_head_fp32(
        hidden_states,
        final_norm_weight,
        lm_head_weight,
        out,
        row_offset,
        valid_rows,
    )
""",
    )


NATIVE_CONTROLS = (
    _dspark(),
    _mtp("deepseek_v4_flash_mtp", 4096),
    _mtp("deepseek_v4_pro", 7168),
    _qwen(),
)


def native_control(model_name: str) -> NativeControl:
    """Return the unique production control for ``model_name``."""

    matches = [
        control for control in NATIVE_CONTROLS if control.model_name == model_name
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown production PyPTO-lib control {model_name!r}")
    return matches[0]
