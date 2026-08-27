"""Static mixed kernels derived from current PyPTO-lib model code.

The attention algebra follows the dense QK/softmax/PV core used by the Qwen
decode implementations.  The MLP follows the gate/up/SiLU/down ordering in
Qwen and DeepSeek dense or expert MLPs.  Shapes are reduced for repeatable
local capture while retaining legal 910B cube dimensions, dtypes, and operation
order.  Neither graph carries a model-name or algorithm recognizer hint.

The two C2V epilogues deliberately use the same two-op DAG at different
shapes.  The small case has one mixed item.  The larger case has 24 spatial
items but uses six 1-AIC + 2-AIV groups, giving each group four successor items
for the cross-core stage-2 pipeline.
"""

from __future__ import annotations

import torch
from torch import nn

from ._runner import Example, run_examples


class StaticAttentionCore(nn.Module):
    """Dense QK -> stable softmax -> PV with static sequence dimensions."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.mm(query, key.t())
        probabilities = torch.softmax(scores, dim=-1)
        return torch.mm(probabilities, value)


class StaticC2VEpilogue(nn.Module):
    """One Cube matmul followed by a broadcast Vector epilogue."""

    def forward(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(value, weight) + bias


class StaticAttentionResidual(nn.Module):
    """Static attention core followed by a generic vector residual epilogue."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.mm(query, key.t())
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.mm(probabilities, value)
        return context + residual


class StaticDenseSwiGlu(nn.Module):
    """BF16 gate/up projections, FP32 SiLU, and BF16 down projection."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_weight = nn.Parameter(
            torch.empty(hidden_size, intermediate_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.up_weight = nn.Parameter(
            torch.empty(hidden_size, intermediate_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.down_weight = nn.Parameter(
            torch.empty(intermediate_size, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate = torch.mm(value, self.gate_weight, out_dtype=torch.float32)
        up = torch.mm(value, self.up_weight, out_dtype=torch.float32)
        activation = (gate * torch.reciprocal(torch.exp(-gate) + 1.0) * up).to(
            torch.bfloat16
        )
        return torch.mm(activation, self.down_weight, out_dtype=torch.float32)


def build_examples() -> dict[str, Example]:
    """Return deterministic static attention and dense SwiGLU examples."""

    torch.manual_seed(0)
    swiglu = StaticDenseSwiGlu(hidden_size=64, intermediate_size=128)
    with torch.no_grad():
        swiglu.gate_weight.normal_(std=0.25)
        swiglu.up_weight.normal_(std=0.25)
        swiglu.down_weight.normal_(std=0.25)
    return {
        "mixed_c2v_single_item": (
            StaticC2VEpilogue(),
            (
                torch.randn(32, 64) * 0.1,
                torch.randn(64, 32) * 0.1,
                torch.randn(1, 32) * 0.1,
            ),
        ),
        "mixed_c2v_streamed_groups": (
            StaticC2VEpilogue(),
            (
                torch.randn(384, 64) * 0.1,
                torch.randn(64, 256) * 0.1,
                torch.randn(1, 256) * 0.1,
            ),
        ),
        "mixed_cvc_streamed_groups": (
            StaticAttentionCore(),
            (
                torch.randn(384, 64) * 0.1,
                torch.randn(64, 64) * 0.1,
                torch.randn(64, 128) * 0.1,
            ),
        ),
        "pypto_lib_static_attention": (
            StaticAttentionCore(),
            (
                torch.randn(96, 64) * 0.1,
                torch.randn(64, 64) * 0.1,
                torch.randn(64, 128) * 0.1,
            ),
        ),
        "pypto_lib_static_attention_residual": (
            StaticAttentionResidual(),
            (
                torch.randn(96, 64) * 0.1,
                torch.randn(64, 64) * 0.1,
                torch.randn(64, 128) * 0.1,
                torch.randn(96, 128) * 0.1,
            ),
        ),
        "pypto_lib_static_dense_swiglu": (
            swiglu,
            (torch.randn(128, 64, dtype=torch.bfloat16) * 0.25,),
        ),
    }


if __name__ == "__main__":
    run_examples(build_examples())
