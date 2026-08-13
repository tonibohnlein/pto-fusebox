"""Shape-reduced Torch form of the Qwen3 final RMSNorm and LM-head DAG.

This is the tensor algebra from pypto-lib's
``models/qwen3_14b/rms_lm_head.py`` with demonstration dimensions and FP32
weights. It is an exporter example, not a replacement checkpoint definition.
"""

from __future__ import annotations

import torch
from torch import nn

from ._runner import Example, run_examples


class Qwen3RmsLmHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        wide = hidden_states.float()
        inverse_rms = torch.rsqrt((wide * wide).mean(dim=-1, keepdim=True) + self.eps)
        normalized = wide * inverse_rms * self.norm_weight
        return self.lm_head(normalized)


def build_examples() -> dict[str, Example]:
    """Return a reduced but 910B-buildable Qwen3 output-head example."""

    torch.manual_seed(0)
    return {
        "qwen3_rms_lm_head": (
            Qwen3RmsLmHead(hidden_size=256, vocab_size=512),
            (torch.randn(64, 256, dtype=torch.bfloat16),),
        )
    }


if __name__ == "__main__":
    run_examples(build_examples())
