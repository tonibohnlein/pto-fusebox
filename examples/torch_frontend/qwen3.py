"""Shape-reduced Torch form of the Qwen3 final RMSNorm and LM-head DAG.

The fixtures mirror the static chunk contract in pypto-lib's
``models/qwen3_14b/rms_lm_head.py``: 16 rows, 128-wide RMSNorm accumulation
chunks, a 512-wide LM-head K window, a 192-wide vocabulary tile, BF16 stored
activations and weights, and an FP32 projection result.  They are reduced
kernel comparisons, not replacement checkpoint definitions.
"""

from __future__ import annotations

import torch
from torch import nn

from ._runner import Example, run_examples

QWEN_BATCH_TILE = 16
# The native PyPTO-lib control accumulates RMS statistics in 128-wide chunks.
# Fusebox remains free to select a different source-ready schedule.
QWEN_REFERENCE_RMS_K_CHUNK = 128
QWEN_LM_HEAD_K_CHUNK = 512
QWEN_VOCAB_CHUNK = 192
QWEN_PRODUCTION_HIDDEN = 5120
QWEN_PRODUCTION_VOCAB = 152064


class Qwen3RmsLmHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(1, hidden_size))
        self.lm_head_weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        wide = hidden_states.float()
        inverse_rms = torch.rsqrt(
            torch.sum(wide * wide, dim=-1, keepdim=True) * (1.0 / wide.shape[-1])
            + self.eps
        )
        normalized = wide * inverse_rms * self.norm_weight
        return torch.mm(
            normalized.to(torch.bfloat16),
            self.lm_head_weight.t(),
            out_dtype=torch.float32,
        )


class Qwen3RmsNormChunk(nn.Module):
    """One fixed physical RMSNorm chunk from the native Qwen orchestration."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(1, hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        wide = hidden_states.float()
        inverse_rms = torch.rsqrt(
            torch.sum(wide * wide, dim=-1, keepdim=True) * (1.0 / wide.shape[-1])
            + self.eps
        )
        return (wide * inverse_rms * self.norm_weight).to(torch.bfloat16)


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


def build_production_qwen_output_head() -> Example:
    """Return the full static Qwen3-14B decode output-head DAG.

    Meta tensors keep the production vocabulary weight out of host memory.
    Fusebox still receives the exact ``16 x 5120`` activation frame and
    ``152064 x 5120`` BF16 weight shape used by PyPTO-lib.
    """

    with torch.device("meta"):
        module = Qwen3RmsLmHead(
            hidden_size=QWEN_PRODUCTION_HIDDEN,
            vocab_size=QWEN_PRODUCTION_VOCAB,
        )
        hidden_states = torch.empty(
            QWEN_BATCH_TILE,
            QWEN_PRODUCTION_HIDDEN,
            dtype=torch.float32,
        )
    return module, (hidden_states,)


def build_examples() -> dict[str, Example]:
    """Return chunk-exact, reduced Qwen3 output-head comparisons."""

    torch.manual_seed(0)
    rms_norm = Qwen3RmsNormChunk(hidden_size=QWEN_LM_HEAD_K_CHUNK)
    lm_head = Qwen3LmHeadChunk(
        hidden_size=QWEN_LM_HEAD_K_CHUNK,
        vocab_size=QWEN_VOCAB_CHUNK,
    )
    connected = Qwen3RmsLmHead(
        hidden_size=QWEN_LM_HEAD_K_CHUNK,
        vocab_size=QWEN_VOCAB_CHUNK,
    )
    with torch.no_grad():
        lm_head.lm_head_weight.normal_(std=0.02)
        connected.lm_head_weight.copy_(lm_head.lm_head_weight)
    return {
        "qwen3_rms_norm_chunk": (
            rms_norm,
            (
                torch.randn(
                    QWEN_BATCH_TILE,
                    QWEN_LM_HEAD_K_CHUNK,
                    dtype=torch.bfloat16,
                ),
            ),
        ),
        "qwen3_lm_head_chunk": (
            lm_head,
            (
                torch.randn(
                    QWEN_BATCH_TILE,
                    QWEN_LM_HEAD_K_CHUNK,
                    dtype=torch.bfloat16,
                ),
            ),
        ),
        "qwen3_rms_lm_head": (
            connected,
            (
                torch.randn(
                    QWEN_BATCH_TILE,
                    QWEN_LM_HEAD_K_CHUNK,
                    dtype=torch.bfloat16,
                ),
            ),
        ),
    }


if __name__ == "__main__":
    run_examples(build_examples())
