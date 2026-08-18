"""Torch programs from the vector AutoTile PR #2335 performance surface.

The original comparison paired hand-tiled PyPTO kernels with tensor-level
functions tagged for the in-compiler AutoTile prototype.  These modules retain
the same formulas, shapes, and dtypes for the standalone path:

``Torch -> PTO-Fusebox -> explicit PyPTO DSL``.
"""

from __future__ import annotations

import torch
from torch import nn

from ._runner import Example, run_examples

RMS_EPS = 1.0e-6
LAYER_EPS = 1.0e-5


class Softmax(nn.Module):
    """Stable last-axis softmax at one fixed runtime shape."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.softmax(value, dim=-1)


class RmsNorm(nn.Module):
    """FP32 RMSNorm with a row-broadcast scale."""

    def forward(self, value: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        mean_square = (value * value).mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(mean_square + RMS_EPS) * gamma


class LayerNorm(nn.Module):
    """FP32 LayerNorm with row-broadcast affine parameters."""

    def forward(
        self,
        value: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        # Preserve the exact PR #2335 tensor DAG: scale the wide value before
        # reducing it, rather than scaling the thin reduction result.  The two
        # forms are mathematically equivalent, but they have different vector
        # primitive geometry and therefore different grounded costs.
        mean = (value * (1.0 / 256.0)).sum(dim=-1, keepdim=True)
        centered = value - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        return centered / torch.sqrt(variance + LAYER_EPS) * gamma + beta


class Silu(nn.Module):
    """SiLU decomposed into ordinary pointwise primitives."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        denominator = torch.exp(value * -1.0) + 1.0
        return value * torch.reciprocal(denominator)


def build_examples() -> dict[str, Example]:
    """Return the exact seven shapes compared by closed PyPTO PR #2335."""

    return {
        "pr2335_softmax_512x256": (Softmax(), (torch.zeros(512, 256),)),
        "pr2335_softmax_256x512": (Softmax(), (torch.zeros(256, 512),)),
        "pr2335_softmax_128x1024": (Softmax(), (torch.zeros(128, 1024),)),
        "pr2335_softmax_32x8192": (Softmax(), (torch.zeros(32, 8192),)),
        "pr2335_rms_norm": (
            RmsNorm(),
            (torch.zeros(512, 512), torch.ones(1, 512)),
        ),
        "pr2335_layer_norm": (
            LayerNorm(),
            (
                torch.zeros(512, 256),
                torch.ones(1, 256),
                torch.zeros(1, 256),
            ),
        ),
        "pr2335_silu": (Silu(), (torch.zeros(512, 256),)),
    }


if __name__ == "__main__":
    run_examples(build_examples())
