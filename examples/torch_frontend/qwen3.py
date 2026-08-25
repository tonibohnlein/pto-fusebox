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


class Qwen3RmsNormChunk(nn.Module):
    """One fixed physical RMSNorm chunk from the native Qwen orchestration."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(1, hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(
            (hidden_states * hidden_states).mean(dim=-1, keepdim=True) + self.eps
        )
        return (hidden_states * inverse_rms * self.norm_weight).to(torch.bfloat16)


class Qwen3LmHeadChunk(nn.Module):
    """One fixed physical LM-head projection with production weight layout."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.lm_head_weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        return torch.mm(
            normalized,
            self.lm_head_weight.t(),
            out_dtype=torch.float32,
        )


def build_examples() -> dict[str, Example]:
    """Return a reduced but 910B-buildable Qwen3 output-head example."""

    torch.manual_seed(0)
    rms_norm = Qwen3RmsNormChunk(hidden_size=256)
    lm_head = Qwen3LmHeadChunk(hidden_size=256, vocab_size=512)
    with torch.no_grad():
        lm_head.lm_head_weight.normal_(std=0.02)
    return {
        "qwen3_rms_norm_chunk": (
            rms_norm,
            (torch.randn(64, 256),),
        ),
        "qwen3_lm_head_chunk": (
            lm_head,
            (torch.randn(64, 256, dtype=torch.bfloat16),),
        ),
        "qwen3_rms_lm_head": (
            Qwen3RmsLmHead(hidden_size=256, vocab_size=512),
            (torch.randn(64, 256, dtype=torch.bfloat16),),
        ),
    }


if __name__ == "__main__":
    run_examples(build_examples())
