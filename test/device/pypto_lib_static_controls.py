"""Independent reduced PyPTO-lib controls for paired device comparisons.

The controls retain the scheduling idioms used by Qwen's tensor-level PyPTO
implementations: native 16-row work units, one split SPMD grid for
QK/softmax/PV, and a stage-3 feature pipeline for gate/up/SiLU/down. They
deliberately use PyPTO's automatic split contract instead of Fusebox's emitted
explicit pipe descriptors. The controls use fixed 16-row tiles, independently
of the current Fusebox plans (48 rows for attention and 32 rows for dense
SwiGLU at these shapes).
"""

from __future__ import annotations


def attention_source() -> str:
    """Return a fixed [96,64] x [64,64] x [64,128] attention control."""

    return """
import pypto.language as pl


@pl.program
class PyPTOLibStaticAttentionControl:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        query: pl.Tensor[[96, 64], pl.FP32],
        key: pl.Tensor[[64, 64], pl.FP32],
        value: pl.Tensor[[64, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[96, 128], pl.FP32]],
    ) -> pl.Tensor[[96, 128], pl.FP32]:
        for row_block in pl.spmd(
            6,
            name_hint="pypto_lib_attention",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            row = row_block * 16
            query_tile = pl.tensor.slice(query, [16, 64], [row, 0])
            key_tile = pl.tensor.slice(key, [64, 64], [0, 0])
            scores = pl.tensor.matmul(
                query_tile,
                key_tile,
                b_trans=True,
                out_dtype=pl.FP32,
            )
            row_max = pl.tensor.row_max(scores)
            shifted = pl.tensor.row_expand_sub(scores, row_max)
            exponentials = pl.tensor.exp(shifted)
            row_sum = pl.tensor.row_sum(exponentials)
            probabilities = pl.tensor.row_expand_div(exponentials, row_sum)
            value_tile = pl.tensor.slice(value, [64, 128], [0, 0])
            context = pl.tensor.matmul(
                probabilities,
                value_tile,
                out_dtype=pl.FP32,
            )
            output = pl.tensor.assemble(output, context, [row, 0])
        return output
""".lstrip()


def dense_swiglu_source() -> str:
    """Return a fixed BF16 128x64 -> 128x128 -> 128x64 MLP control."""

    return """
import pypto.language as pl


@pl.program
class PyPTOLibStaticDenseSwiGluControl:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        value: pl.Tensor[[128, 64], pl.BF16],
        gate_weight: pl.Tensor[[64, 128], pl.BF16],
        up_weight: pl.Tensor[[64, 128], pl.BF16],
        down_weight: pl.Tensor[[128, 64], pl.BF16],
        output: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
    ) -> pl.Tensor[[128, 64], pl.FP32]:
        for row_block in pl.spmd(
            8,
            name_hint="pypto_lib_dense_swiglu",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            row = row_block * 16
            down_seed = pl.tensor.create(
                [16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
            )
            for feature, (down_acc,) in pl.pipeline(
                0, 128, 64, stage=3, init_values=(down_seed,)
            ):
                input_tile = pl.tensor.slice(value, [16, 64], [row, 0])
                gate_tile = pl.tensor.slice(gate_weight, [64, 64], [0, feature])
                up_tile = pl.tensor.slice(up_weight, [64, 64], [0, feature])
                gate = pl.tensor.matmul(
                    input_tile, gate_tile, out_dtype=pl.FP32
                )
                up = pl.tensor.matmul(input_tile, up_tile, out_dtype=pl.FP32)
                sigmoid = pl.tensor.recip(
                    pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(gate)), 1.0)
                )
                activation = pl.tensor.cast(
                    pl.tensor.mul(pl.tensor.mul(gate, sigmoid), up),
                    target_type=pl.BF16,
                    mode="round",
                )
                down_tile = pl.tensor.slice(down_weight, [64, 64], [feature, 0])
                if feature == 0:
                    next_down = pl.tensor.matmul(
                        activation, down_tile, out_dtype=pl.FP32
                    )
                else:
                    next_down = pl.tensor.matmul_acc(
                        down_acc, activation, down_tile
                    )
                down_acc = pl.yield_(next_down)
            output = pl.tensor.assemble(output, down_acc, [row, 0])
        return output
""".lstrip()
