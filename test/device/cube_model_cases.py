"""Predeclared shape surface for cube model-versus-silicon validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CubeModelCase:
    """One matmul shape and the mechanism it isolates."""

    name: str
    m: int
    k: int
    n: int
    purpose: str
    split_k_control: bool = False
    source_replay_expected: bool = True


CUBE_MODEL_CASES = (
    CubeModelCase(
        "tiny_underfill",
        16,
        64,
        32,
        "launch and underfill control",
    ),
    CubeModelCase(
        "balanced_square",
        256,
        256,
        256,
        "compute-bound spatial tiling and split-K negative control",
        split_k_control=True,
    ),
    CubeModelCase(
        "ragged_tail",
        64,
        272,
        80,
        "K tail, unusual N, GM-to-L1 feed, and FIXPIPE drain",
    ),
    CubeModelCase(
        "deep_k_thin",
        32,
        736,
        64,
        "serial outer-K windows and persistent L0C accumulation",
    ),
    CubeModelCase(
        "rectangular_reuse",
        64,
        512,
        256,
        "M/N partitioning, operand reuse, and L1 pressure",
    ),
    CubeModelCase(
        "split_k_positive",
        128,
        8192,
        128,
        "grid-level split-K positive with insufficient spatial work",
        split_k_control=True,
        source_replay_expected=False,
    ),
)
