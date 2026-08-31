"""Shape-reduced Torch forms of supported DeepSeek V4-Pro tensor DAGs.

The algebra follows the Torch goldens in pypto-lib's
``models/deepseek_v4_pro/rmsnorm.py`` and ``mtp_projection.py``. The projection
example intentionally omits the production INT8 quantization stages so it
isolates the pointwise/reduction/matmul graph currently understood by Fusebox.
"""

from __future__ import annotations

import torch
from torch import nn

from ._runner import Example, run_examples


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
    }


if __name__ == "__main__":
    run_examples(build_examples())
