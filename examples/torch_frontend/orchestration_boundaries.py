"""Static Fusebox regions around native data-dependent orchestration boundaries.

These reduced examples model the ownership boundary used by pypto-lib. Fusebox
plans only the affine, statically shaped tensor regions. Paged gathers, TopK,
and token routing remain explicit operations for native PyPTO orchestration.
"""

from __future__ import annotations

import torch
from torch import nn

from ._runner import Example, run_examples


class PagedAttentionStaticRegions(nn.Module):
    """QK/softmax and PV separated by one opaque paged-value gather."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(torch.mm(query, key.t()), dim=-1)
        gathered_value = torch.index_select(value_cache, 0, block_table)
        return torch.mm(probabilities, gathered_value)


class MoeStaticRegions(nn.Module):
    """Router projection and expert SwiGLU around opaque token routing."""

    def forward(
        self,
        tokens: torch.Tensor,
        router_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        router_scores = torch.mm(tokens, router_weight)
        _, expert_indices = torch.topk(router_scores, k=1, dim=-1)
        routed_tokens = torch.index_select(tokens, 0, expert_indices.reshape(-1))
        gate = torch.mm(routed_tokens, gate_weight)
        up = torch.mm(routed_tokens, up_weight)
        silu = gate * torch.reciprocal(torch.exp(-gate) + 1.0)
        return torch.mm(silu * up, down_weight)


def build_examples() -> dict[str, Example]:
    """Return deterministic static-region/opaque-boundary examples."""

    return {
        "paged_attention_static_regions": (
            PagedAttentionStaticRegions(),
            (
                torch.zeros(64, 64),
                torch.zeros(64, 64),
                torch.zeros(128, 64),
                torch.arange(64),
            ),
        ),
        "moe_static_regions": (
            MoeStaticRegions(),
            (
                torch.zeros(64, 64),
                torch.zeros(64, 16),
                torch.zeros(64, 128),
                torch.zeros(64, 128),
                torch.zeros(128, 64),
            ),
        ),
    }


if __name__ == "__main__":
    run_examples(build_examples())
