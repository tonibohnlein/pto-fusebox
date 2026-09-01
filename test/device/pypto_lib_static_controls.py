"""Independent static PyPTO controls for paired device comparisons.

The attention, dense-SwiGLU and Qwen controls retain the scheduling idioms used
by the corresponding tensor-level PyPTO-lib implementations, reduced only to
the fixture dimensions. They deliberately use ordinary PyPTO scheduling
constructs instead of Fusebox's emitted explicit pipe descriptors. The
unquantized DeepSeek MTP fixture has no production-equivalent PyPTO-lib kernel,
so its explicitly named control is an independent unfused baseline for the
same reduced algebra.
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


def qwen_rms_norm_source() -> str:
    """Return the reduced native two-pass Qwen RMSNorm control."""

    return """
import pypto.language as pl


@pl.program
class PyPTOLibQwenRmsNormControl:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        hidden_states: pl.Tensor[[16, 512], pl.BF16],
        final_norm_weight: pl.Tensor[[1, 512], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 512], pl.BF16]],
    ) -> pl.Tensor[[16, 512], pl.BF16]:
        for row in pl.parallel(0, 16, 16):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="qwen_rms_norm",
                allow_early_resolve=True,
            ):
                square_sum = pl.full([1, 16], dtype=pl.FP32, value=0.0)
                for chunk_index in pl.range(4):
                    column = chunk_index * 128
                    chunk = pl.cast(
                        pl.slice(hidden_states, [16, 128], [row, column]),
                        target_type=pl.FP32,
                    )
                    square_sum = pl.add(
                        square_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(chunk, chunk)),
                            [1, 16],
                        ),
                    )
                inverse_rms = pl.reshape(
                    pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / 512.0), 1.0e-6)),
                    [16, 1],
                )
                for chunk_index in pl.range(4):
                    column = chunk_index * 128
                    chunk = pl.cast(
                        pl.slice(hidden_states, [16, 128], [row, column]),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(final_norm_weight, [1, 128], [0, column])
                    normalized = pl.col_expand_mul(
                        pl.row_expand_mul(chunk, inverse_rms), gamma
                    )
                    output = pl.assemble(
                        output,
                        pl.cast(normalized, target_type=pl.BF16),
                        [row, column],
                    )
        return output
""".lstrip()


def qwen_lm_head_source() -> str:
    """Return the reduced native 512-wide Qwen LM-head control."""

    return """
import pypto.language as pl


@pl.program
class PyPTOLibQwenLmHeadControl:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        normalized: pl.Tensor[[16, 512], pl.BF16],
        lm_head_weight: pl.Tensor[[192, 512], pl.BF16],
        output: pl.Out[pl.Tensor[[16, 192], pl.FP32]],
    ) -> pl.Tensor[[16, 192], pl.FP32]:
        for block in pl.spmd(1, name_hint="qwen_lm_head"):
            hidden = pl.slice(normalized, [16, 512], [0, 0])
            weight = pl.slice(lm_head_weight, [192, 512], [0, 0])
            result = pl.matmul(
                hidden,
                weight,
                b_trans=True,
                out_dtype=pl.FP32,
            )
            output = pl.assemble(output, result, [0, block * 192])
        return output
""".lstrip()


def qwen_rms_lm_head_source() -> str:
    """Return native reduced Qwen RMSNorm and LM-head with a GM cut."""

    return """
import pypto.language as pl


@pl.program
class PyPTOLibQwenRmsLmHeadControl:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        hidden_states: pl.Tensor[[16, 512], pl.BF16],
        final_norm_weight: pl.Tensor[[1, 512], pl.FP32],
        lm_head_weight: pl.Tensor[[192, 512], pl.BF16],
        output: pl.Out[pl.Tensor[[16, 192], pl.FP32]],
    ) -> pl.Tensor[[16, 192], pl.FP32]:
        final_normed = pl.create_tensor([16, 512], dtype=pl.BF16)
        for row in pl.parallel(0, 16, 16):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="qwen_rms_norm",
                allow_early_resolve=True,
            ):
                square_sum = pl.full([1, 16], dtype=pl.FP32, value=0.0)
                for chunk_index in pl.range(4):
                    column = chunk_index * 128
                    chunk = pl.cast(
                        pl.slice(hidden_states, [16, 128], [row, column]),
                        target_type=pl.FP32,
                    )
                    square_sum = pl.add(
                        square_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(chunk, chunk)),
                            [1, 16],
                        ),
                    )
                inverse_rms = pl.reshape(
                    pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / 512.0), 1.0e-6)),
                    [16, 1],
                )
                for chunk_index in pl.range(4):
                    column = chunk_index * 128
                    chunk = pl.cast(
                        pl.slice(hidden_states, [16, 128], [row, column]),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(final_norm_weight, [1, 128], [0, column])
                    normalized = pl.col_expand_mul(
                        pl.row_expand_mul(chunk, inverse_rms), gamma
                    )
                    final_normed = pl.assemble(
                        final_normed,
                        pl.cast(normalized, target_type=pl.BF16),
                        [row, column],
                    )
        for block in pl.spmd(1, name_hint="qwen_lm_head"):
            hidden = pl.slice(final_normed, [16, 512], [0, 0])
            weight = pl.slice(lm_head_weight, [192, 512], [0, 0])
            result = pl.matmul(
                hidden,
                weight,
                b_trans=True,
                out_dtype=pl.FP32,
            )
            output = pl.assemble(output, result, [0, block * 192])
        return output
""".lstrip()


def deepseek_mtp_unfused_source() -> str:
    """Return an independent unfused control for the reduced unquantized MTP DAG.

    The production PyPTO-lib MTP kernel includes INT8 activation quantization,
    which the current Torch fixture intentionally omits.  This control is
    therefore a static PyPTO baseline for the same reduced algebra, not a claim
    to reproduce the production kernel's schedule.
    """

    return """
import pypto.language as pl


@pl.program
class IndependentDeepSeekMtpUnfusedControl:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        hidden: pl.Tensor[[64, 256], pl.BF16],
        enorm_weight: pl.Tensor[[1, 256], pl.FP32],
        e_proj_weight: pl.Tensor[[256, 256], pl.FP32],
        previous: pl.Tensor[[64, 256], pl.FP32],
        hnorm_weight: pl.Tensor[[1, 256], pl.FP32],
        h_proj_weight: pl.Tensor[[256, 256], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
    ) -> pl.Tensor[[64, 256], pl.FP32]:
        embedded_normalized = pl.create_tensor([64, 256], dtype=pl.FP32)
        for block in pl.spmd(2, name_hint="mtp_enorm"):
            row = block * 32
            hidden_tile = pl.tensor.cast(
                pl.tensor.slice(hidden, [32, 256], [row, 0]),
                target_type=pl.FP32,
                mode="round",
            )
            square = pl.tensor.mul(hidden_tile, hidden_tile)
            square_sum = pl.tensor.row_sum(square)
            inverse_rms = pl.tensor.rsqrt(
                pl.tensor.adds(pl.tensor.muls(square_sum, 1.0 / 256.0), 1.0e-6)
            )
            normalized = pl.tensor.col_expand_mul(
                pl.tensor.row_expand_mul(hidden_tile, inverse_rms),
                pl.tensor.slice(enorm_weight, [1, 256], [0, 0]),
            )
            embedded_normalized = pl.tensor.assemble(
                embedded_normalized, normalized, [row, 0]
            )

        embedded = pl.create_tensor([64, 256], dtype=pl.FP32)
        for block in pl.spmd(4, name_hint="mtp_e_proj"):
            row_part = block // 2
            column_part = block % 2
            lhs = pl.tensor.slice(
                embedded_normalized, [32, 256], [row_part * 32, 0]
            )
            rhs = pl.tensor.slice(
                e_proj_weight, [128, 256], [column_part * 128, 0]
            )
            projected = pl.tensor.matmul(
                lhs, rhs, b_trans=True, out_dtype=pl.FP32
            )
            embedded = pl.tensor.assemble(
                embedded, projected, [row_part * 32, column_part * 128]
            )

        history_normalized = pl.create_tensor([64, 256], dtype=pl.FP32)
        for block in pl.spmd(8, name_hint="mtp_hnorm"):
            row = block * 8
            previous_tile = pl.tensor.slice(previous, [8, 256], [row, 0])
            history_square = pl.tensor.mul(previous_tile, previous_tile)
            history_square_sum = pl.tensor.row_sum(history_square)
            history_inverse_rms = pl.tensor.rsqrt(
                pl.tensor.adds(
                    pl.tensor.muls(history_square_sum, 1.0 / 256.0), 1.0e-6
                )
            )
            history_normalized_tile = pl.tensor.col_expand_mul(
                pl.tensor.row_expand_mul(previous_tile, history_inverse_rms),
                pl.tensor.slice(hnorm_weight, [1, 256], [0, 0]),
            )
            history_normalized = pl.tensor.assemble(
                history_normalized, history_normalized_tile, [row, 0]
            )

        history = pl.create_tensor([64, 256], dtype=pl.FP32)
        for block in pl.spmd(4, name_hint="mtp_h_proj"):
            column = block * 64
            history_lhs = pl.tensor.slice(
                history_normalized, [64, 256], [0, 0]
            )
            history_rhs = pl.tensor.slice(
                h_proj_weight, [64, 256], [column, 0]
            )
            history_projected = pl.tensor.matmul(
                history_lhs,
                history_rhs,
                b_trans=True,
                out_dtype=pl.FP32,
            )
            history = pl.tensor.assemble(
                history, history_projected, [0, column]
            )

        for block in pl.spmd(8, name_hint="mtp_add"):
            row = block * 8
            embedded_tile = pl.tensor.slice(embedded, [8, 256], [row, 0])
            history_tile = pl.tensor.slice(history, [8, 256], [row, 0])
            output = pl.tensor.assemble(
                output,
                pl.tensor.add(embedded_tile, history_tile),
                [row, 0],
            )
        return output
""".lstrip()
