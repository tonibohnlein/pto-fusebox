"""Torch forms of supported DeepSeek V4 static tensor DAGs.

The algebra follows the Torch goldens in pypto-lib's RMSNorm and MTP projection
implementations.  Both the original unquantized composition and the production
row-quantized INT8 projection branch are retained as distinct examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._runner import Example, run_examples

DEEPSEEK_V4_HIDDEN = 4096
DEEPSEEK_V4_HC_MULT = 4
DEEPSEEK_V4_DECODE_TOKENS = 8
DEEPSEEK_V4_LINEAR_TOKENS = 16
DEEPSEEK_V4_PREFILL_TOKENS = 128
DEEPSEEK_V4_DSPARK_TARGET_LAYERS = 3


@dataclass(frozen=True)
class DeepSeekV4MtpGeometry:
    """Static physical frame supplied by native MTP orchestration."""

    hidden_size: int
    hyperconnections: int = DEEPSEEK_V4_HC_MULT
    decode_tokens: int = DEEPSEEK_V4_DECODE_TOKENS
    linear_tokens: int = DEEPSEEK_V4_LINEAR_TOKENS
    prefill_tokens: int = DEEPSEEK_V4_PREFILL_TOKENS

    def __post_init__(self) -> None:
        if (
            self.hidden_size <= 0
            or self.hyperconnections <= 0
            or self.decode_tokens <= 0
            or self.linear_tokens < self.decode_tokens
            or self.prefill_tokens <= 0
        ):
            raise ValueError("DeepSeek MTP geometry has invalid positive extents")


DEEPSEEK_V4_FLASH_MTP_GEOMETRY = DeepSeekV4MtpGeometry(hidden_size=DEEPSEEK_V4_HIDDEN)
DEEPSEEK_V4_PRO_MTP_GEOMETRY = DeepSeekV4MtpGeometry(hidden_size=7168)


class DeepSeekV4RmsNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        wide = value.float()
        inverse_rms = torch.rsqrt((wide * wide).mean(dim=-1, keepdim=True) + self.eps)
        return (wide * inverse_rms * self.weight).to(value.dtype)


class DeepSeekV4DsparkProjection(nn.Module):
    """Production-shape DSpark target projection followed by RMSNorm."""

    def __init__(self, hidden_size: int, target_layers: int, eps: float = 1e-6) -> None:
        super().__init__()
        input_size = hidden_size * target_layers
        self.main_projection = nn.Parameter(
            torch.empty(hidden_size, input_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.norm_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.eps = eps

    def forward(self, main_hidden: torch.Tensor) -> torch.Tensor:
        projected = torch.mm(main_hidden, self.main_projection.t()).to(torch.bfloat16)
        wide = projected.float()
        inverse_rms = torch.rsqrt((wide * wide).mean(dim=-1, keepdim=True) + self.eps)
        return (wide * inverse_rms * self.norm_weight.float()).to(torch.bfloat16)


class DeepSeekV4MtpProjection(nn.Module):
    """Unquantized ``e_proj(enorm(x)) + h_proj(hnorm(previous))`` graph."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        # Rank-two singleton rows match the native PyPTO broadcast ABI and
        # remain ordinary external tensors in generated source.
        self.enorm_weight = nn.Parameter(torch.ones(1, hidden_size))
        self.hnorm_weight = nn.Parameter(torch.ones(1, hidden_size))
        self.e_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.eps = eps

    def _rms_norm(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        wide = value.float()
        inverse_rms = torch.rsqrt((wide * wide).mean(dim=-1, keepdim=True) + self.eps)
        return wide * inverse_rms * weight

    def forward(self, hidden: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        embedded = self.e_proj(self._rms_norm(hidden, self.enorm_weight))
        projected_history = self.h_proj(self._rms_norm(previous, self.hnorm_weight))
        return embedded + projected_history


class DeepSeekV4Int8ProjectionBranch(nn.Module):
    """Generic RMSNorm/smooth/row-quantize/INT8-matmul/dequant branch.

    This is the ordinary tensor algebra used by both ``e_proj`` and ``h_proj``
    in the production Flash-MTP projection.  The HC replication and the final
    per-stream addition are orchestration/composition concerns, not a special
    quantization operator or an MTP recognizer.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(
        self,
        value: torch.Tensor,
        norm_weight: torch.Tensor,
        smooth: torch.Tensor,
        projection_weight: torch.Tensor,
        projection_scale: torch.Tensor,
    ) -> torch.Tensor:
        wide = value.float()
        inverse_rms = torch.rsqrt((wide * wide).mean(dim=-1, keepdim=True) + self.eps)
        normalized = wide * inverse_rms * norm_weight * smooth
        amax = torch.clamp_min(
            torch.amax(torch.abs(normalized), dim=-1, keepdim=True),
            1e-12,
        )
        quant_scale = torch.reciprocal(amax) * 127.0
        quantized = (
            torch.round(normalized * quant_scale)
            .to(torch.int32)
            .to(torch.float16)
            .to(torch.int8)
        )
        dequant_scale = torch.reciprocal(quant_scale)
        accumulator = torch.ops.aten._int_mm.default(quantized, projection_weight.t())
        return accumulator.float() * dequant_scale * projection_scale


def build_production_dspark_projection() -> Example:
    """Return the full fixed decode DSpark projection-to-RMSNorm DAG."""

    input_size = DEEPSEEK_V4_HIDDEN * DEEPSEEK_V4_DSPARK_TARGET_LAYERS
    with torch.device("meta"):
        module = DeepSeekV4DsparkProjection(
            DEEPSEEK_V4_HIDDEN,
            DEEPSEEK_V4_DSPARK_TARGET_LAYERS,
        )
        main_hidden = torch.empty(
            DEEPSEEK_V4_LINEAR_TOKENS,
            input_size,
            dtype=torch.bfloat16,
        )
    return module, (main_hidden,)


class DeepSeekV4Int8MtpProjection(nn.Module):
    """Complete static Flash-MTP projection tensor DAG.

    The physical hidden frame and flattened history frame are the stable ABI
    supplied by native PyPTO orchestration.  Both projection branches and the
    final logical composition are captured together; Fusebox, rather than the
    caller, decides where the currently supported source regions are cut.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        decode_tokens: int = DEEPSEEK_V4_DECODE_TOKENS,
        hyperconnections: int = DEEPSEEK_V4_HC_MULT,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.decode_tokens = decode_tokens
        self.hyperconnections = hyperconnections
        self.branch = DeepSeekV4Int8ProjectionBranch(hidden_size, eps)

    def forward(
        self,
        hidden_padded: torch.Tensor,
        history_flat: torch.Tensor,
        enorm_weight: torch.Tensor,
        hnorm_weight: torch.Tensor,
        e_smooth: torch.Tensor,
        h_smooth: torch.Tensor,
        e_projection_weight: torch.Tensor,
        h_projection_weight: torch.Tensor,
        e_projection_scale: torch.Tensor,
        h_projection_scale: torch.Tensor,
    ) -> torch.Tensor:
        hidden_projected = self.branch(
            hidden_padded,
            enorm_weight,
            e_smooth,
            e_projection_weight,
            e_projection_scale,
        )
        history_projected = self.branch(
            history_flat,
            hnorm_weight,
            h_smooth,
            h_projection_weight,
            h_projection_scale,
        )
        hidden_logical = hidden_projected[: self.decode_tokens].unsqueeze(1)
        history_logical = history_projected.reshape(
            self.decode_tokens,
            self.hyperconnections,
            self.hidden_size,
        )
        return hidden_logical + history_logical


def build_production_mtp_projection_branch(
    geometry: DeepSeekV4MtpGeometry = DEEPSEEK_V4_FLASH_MTP_GEOMETRY,
) -> Example:
    """Return one production MTP branch at its configured physical frame.

    Native decode orchestration supplies a logical token extent inside a
    possibly larger INT8 matmul frame. Fusebox plans that static physical
    compilation unit; orchestration owns the logical prefix and padded-frame
    construction.
    """

    tokens = geometry.linear_tokens
    hidden = geometry.hidden_size
    return (
        DeepSeekV4Int8ProjectionBranch(hidden),
        (
            torch.empty(tokens, hidden, dtype=torch.bfloat16, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(hidden, hidden, dtype=torch.int8, device="meta"),
            torch.empty(1, hidden, device="meta"),
        ),
    )


def build_production_mtp_history_projection_branch(
    geometry: DeepSeekV4MtpGeometry = DEEPSEEK_V4_FLASH_MTP_GEOMETRY,
) -> Example:
    """Return the configured token × hyperconnection history branch."""

    rows = geometry.decode_tokens * geometry.hyperconnections
    hidden = geometry.hidden_size
    return (
        DeepSeekV4Int8ProjectionBranch(hidden),
        (
            torch.empty(rows, hidden, dtype=torch.float32, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(hidden, hidden, dtype=torch.int8, device="meta"),
            torch.empty(1, hidden, device="meta"),
        ),
    )


def build_production_mtp_prefill_projection_branch(
    geometry: DeepSeekV4MtpGeometry = DEEPSEEK_V4_FLASH_MTP_GEOMETRY,
) -> Example:
    """Return one configured production prefill frame.

    Prefill orchestration repeats this fixed physical frame and owns any final
    logical tail.  It deliberately reuses the same ordinary branch algebra as
    decode; the model name and execution phase never enter the solver.
    """

    rows = geometry.prefill_tokens
    hidden = geometry.hidden_size
    return (
        DeepSeekV4Int8ProjectionBranch(hidden),
        (
            torch.empty(rows, hidden, dtype=torch.bfloat16, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(hidden, hidden, dtype=torch.int8, device="meta"),
            torch.empty(1, hidden, device="meta"),
        ),
    )


def build_production_mtp_decode_projection(
    geometry: DeepSeekV4MtpGeometry = DEEPSEEK_V4_FLASH_MTP_GEOMETRY,
) -> Example:
    """Return the complete production decode projection as one Torch graph.

    Native orchestration owns construction of the physical hidden frame and
    flattening of the history input. It does not pre-partition the two static
    projection branches before Fusebox receives their shared output DAG.
    """

    hidden = geometry.hidden_size
    history_rows = geometry.decode_tokens * geometry.hyperconnections
    return (
        DeepSeekV4Int8MtpProjection(
            hidden,
            decode_tokens=geometry.decode_tokens,
            hyperconnections=geometry.hyperconnections,
        ),
        (
            torch.empty(
                geometry.linear_tokens,
                hidden,
                dtype=torch.bfloat16,
                device="meta",
            ),
            torch.empty(history_rows, hidden, dtype=torch.float32, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(hidden, hidden, dtype=torch.int8, device="meta"),
            torch.empty(hidden, hidden, dtype=torch.int8, device="meta"),
            torch.empty(1, hidden, device="meta"),
            torch.empty(1, hidden, device="meta"),
        ),
    )


def build_examples() -> dict[str, Example]:
    """Return reduced but 910B-buildable model inputs.

    The token and hidden extents are intentionally smaller than a production
    checkpoint, but remain at least one legal cube tile in every matmul axis.
    """

    torch.manual_seed(0)
    return {
        "deepseek_v4_rmsnorm": (
            DeepSeekV4RmsNorm(1024),
            (torch.randn(128, 1024, dtype=torch.bfloat16),),
        ),
        "deepseek_v4_mtp_projection": (
            DeepSeekV4MtpProjection(256),
            (
                torch.randn(64, 256, dtype=torch.bfloat16),
                torch.randn(64, 256, dtype=torch.float32),
            ),
        ),
        "deepseek_v4_int8_mtp_branch": (
            DeepSeekV4Int8ProjectionBranch(256),
            (
                torch.randn(16, 256, dtype=torch.bfloat16),
                torch.ones(1, 256),
                torch.ones(1, 256),
                torch.randint(-8, 9, (256, 256), dtype=torch.int8),
                torch.ones(1, 256),
            ),
        ),
    }


if __name__ == "__main__":
    run_examples(build_examples())
