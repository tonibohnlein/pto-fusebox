"""Independent PyPTO controls for ragged streaming-softmax diagnostics."""

from __future__ import annotations

from textwrap import indent


def independent_three_pass_softmax_pv_source(
    *, rows: int = 16, extent: int = 4096, columns: int = 64, chunk: int = 160
) -> str:
    """Return a non-online softmax→PV program with an explicit ragged tail.

    Fusebox's mixed arm updates online max/sum state while reading each chunk.
    This control deliberately uses three complete sweeps: global maximum,
    global exponential sum, then normalized matmul accumulation.  It therefore
    shares neither the online recurrence nor its reduction order. Its tail
    intentionally uses the same fixed-frame fill/valid-shape publication
    contract so comparisons isolate arithmetic from that transport rule.
    """

    if min(rows, extent, columns, chunk) <= 0:
        raise ValueError("softmax control extents must be positive")
    full_chunks, tail = divmod(extent, chunk)
    if full_chunks < 2 or tail <= 0:
        raise ValueError("softmax control requires at least two chunks and a tail")
    if tail % 16 != 0 or chunk % 16 != 0:
        raise ValueError("softmax control chunk and tail must be fractal-aligned")

    source = f'''"""Independent three-pass ragged softmax→PV control."""

import pypto.language as pl


@pl.jit
def independent_softmax_pv(
    scores: pl.Tensor[[{rows}, {extent}], pl.FP32],
    value: pl.Tensor[[{extent}, {columns}], pl.FP32],
    output: pl.Out[pl.Tensor[[{rows}, {columns}], pl.FP32]],
) -> pl.Tensor[[{rows}, {columns}], pl.FP32]:
    first = pl.tensor.slice(
        scores, [{rows}, {chunk}], [0, 0],
        valid_shape=[{rows}, {chunk}], pad_value=pl.PadValue.min)
    running_max = pl.tensor.row_max(first)
    for max_chunk, (current_max,) in pl.range(
        1, {full_chunks}, init_values=(running_max,)):
        block = pl.tensor.slice(
            scores, [{rows}, {chunk}], [0, max_chunk * {chunk}],
            valid_shape=[{rows}, {chunk}], pad_value=pl.PadValue.min)
        next_max = pl.tensor.maximum(current_max, pl.tensor.row_max(block))
        yielded_max = pl.yield_(next_max)
    tail_scores = pl.tensor.slice(
        scores, [{rows}, {chunk}], [0, {full_chunks * chunk}],
        valid_shape=[{rows}, {tail}], pad_value=pl.PadValue.min)
    running_max = pl.tensor.maximum(yielded_max, pl.tensor.row_max(tail_scores))

    first_exp = pl.tensor.exp(pl.tensor.row_expand_sub(first, running_max))
    running_sum = pl.tensor.row_sum(first_exp)
    for sum_chunk, (current_sum,) in pl.range(
        1, {full_chunks}, init_values=(running_sum,)):
        block = pl.tensor.slice(
            scores, [{rows}, {chunk}], [0, sum_chunk * {chunk}],
            valid_shape=[{rows}, {chunk}], pad_value=pl.PadValue.min)
        block_exp = pl.tensor.exp(pl.tensor.row_expand_sub(block, running_max))
        next_sum = pl.tensor.add(current_sum, pl.tensor.row_sum(block_exp))
        yielded_sum = pl.yield_(next_sum)
    tail_exp = pl.tensor.exp(
        pl.tensor.row_expand_sub(tail_scores, running_max))
    running_sum = pl.tensor.add(yielded_sum, pl.tensor.row_sum(tail_exp))

    first_probability = pl.tensor.row_expand_div(first_exp, running_sum)
    first_value = pl.tensor.slice(
        value, [{chunk}, {columns}], [0, 0],
        valid_shape=[{chunk}, {columns}], pad_value=pl.PadValue.zero)
    accumulator = pl.tensor.matmul(
        first_probability, first_value,
        a_trans=False, b_trans=False, out_dtype=pl.FP32)
    for apply_chunk, (current_accumulator,) in pl.range(
        1, {full_chunks}, init_values=(accumulator,)):
        block = pl.tensor.slice(
            scores, [{rows}, {chunk}], [0, apply_chunk * {chunk}],
            valid_shape=[{rows}, {chunk}], pad_value=pl.PadValue.min)
        probability = pl.tensor.row_expand_div(
            pl.tensor.exp(pl.tensor.row_expand_sub(block, running_max)),
            running_sum)
        value_block = pl.tensor.slice(
            value, [{chunk}, {columns}], [apply_chunk * {chunk}, 0],
            valid_shape=[{chunk}, {columns}], pad_value=pl.PadValue.zero)
        next_accumulator = pl.tensor.matmul_acc(
            current_accumulator, probability, value_block,
            a_trans=False, b_trans=False)
        yielded_accumulator = pl.yield_(next_accumulator)
    tail_probability = pl.tensor.row_expand_div(tail_exp, running_sum)
    tail_probability = pl.tensor.fillpad(
        tail_probability, pad_value=pl.PadValue.zero)
    tail_probability = pl.tensor.set_validshape(
        tail_probability, {rows}, {tail})
    tail_value = pl.tensor.slice(
        value, [{chunk}, {columns}], [{full_chunks * chunk}, 0],
        valid_shape=[{tail}, {columns}], pad_value=pl.PadValue.zero)
    accumulator = pl.tensor.matmul_acc(
        yielded_accumulator, tail_probability, tail_value,
        a_trans=False, b_trans=False)
    output = pl.tensor.assemble(output, accumulator, [0, 0])
    return output
'''
    signature_end = f") -> pl.Tensor[[{rows}, {columns}], pl.FP32]:\n"
    header, operations = source.split(signature_end, maxsplit=1)
    return_line = "    return output\n"
    if not operations.endswith(return_line):
        raise AssertionError("independent control body lost its return")
    operations = operations[: -len(return_line)]
    return (
        header
        + signature_end
        + "    with pl.at(level=pl.Level.CORE_GROUP):\n"
        + indent(operations, "    ")
        + return_line
    )
