"""Small positive Torch capture examples used by the frontend contract tests."""

from __future__ import annotations

import torch
from torch import nn

from ._runner import Example, run_examples


class Softmax(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.softmax(value, dim=-1)


class Matmul(nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.mm(lhs, rhs)


class AttentionCore(nn.Module):
    """Generic QK -> softmax -> PV tensor DAG, without an attention recognizer."""

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scores = torch.mm(query, key.t())
        probabilities = torch.softmax(scores, dim=-1)
        return torch.mm(probabilities, value)


def build_examples() -> dict[str, Example]:
    """Return deterministic, 910B-buildable representative inputs."""

    return {
        "softmax": (Softmax(), (torch.zeros(128, 1024),)),
        "matmul": (Matmul(), (torch.zeros(128, 256), torch.zeros(256, 192))),
        "attention_core": (
            AttentionCore(),
            (
                torch.zeros(96, 64),
                torch.zeros(64, 64),
                torch.zeros(64, 128),
            ),
        ),
    }


if __name__ == "__main__":
    run_examples(build_examples())
