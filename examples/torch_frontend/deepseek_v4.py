"""Torch forms of supported DeepSeek V4 static tensor DAGs.

The algebra follows the Torch goldens in pypto-lib's RMSNorm and MTP projection
implementations.  Both the original unquantized composition and the production
row-quantized INT8 projection branch are retained as distinct examples.
"""

from __future__ import annotations

import torch
from torch import nn

from ._runner import Example, run_examples

DEEPSEEK_V4_HIDDEN = 4096
DEEPSEEK_V4_HC_MULT = 4
DEEPSEEK_V4_DECODE_TOKENS = 8
DEEPSEEK_V4_LINEAR_TOKENS = 16
DEEPSEEK_V4_PREFILL_TOKENS = 128


class DeepSeekV4RmsNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        wide = value.float()
        inverse_rms = torch.rsqrt((wide * wide).mean(dim=-1, keepdim=True) + self.eps)
        return (wide * inverse_rms * self.weight).to(value.dtype)


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
        accumulator = torch._int_mm(quantized, projection_weight.t())
        return accumulator.float() * dequant_scale * projection_scale


def build_production_mtp_projection_branch() -> Example:
    """Return one production Flash-MTP branch at its physical linear frame.

    Native decode orchestration has eight logical tokens, but deliberately
    rounds the INT8 matmul frame up to sixteen rows. Fusebox plans that static
    physical compilation unit; orchestration owns the logical eight-row
    prefix and padded-frame construction.
    """

    tokens = DEEPSEEK_V4_LINEAR_TOKENS
    hidden = DEEPSEEK_V4_HIDDEN
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


def build_production_mtp_history_projection_branch() -> Example:
    """Return the 8-token × 4-hyperconnection production history branch."""

    rows = DEEPSEEK_V4_DECODE_TOKENS * DEEPSEEK_V4_HC_MULT
    hidden = DEEPSEEK_V4_HIDDEN
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


def build_production_mtp_prefill_projection_branch() -> Example:
    """Return one 128-token production prefill branch.

    Prefill orchestration repeats this fixed physical frame and owns any final
    logical tail.  It deliberately reuses the same ordinary branch algebra as
    decode; the model name and execution phase never enter the solver.
    """

    rows = DEEPSEEK_V4_PREFILL_TOKENS
    hidden = DEEPSEEK_V4_HIDDEN
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
