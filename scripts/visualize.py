"""Render PTO Fusebox problem, partition, and per-kernel algorithm diagrams."""

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

KERNEL_COLORS = [
    "#ffb3ba",
    "#ffdfba",
    "#ffffba",
    "#baffc9",
    "#bae1ff",
    "#e8baff",
    "#ffbaff",
    "#c2c2f0",
    "#ffb3e6",
]

EVENT_COLORS = {
    "LOAD": "#d9ecff",
    "LOOP": "#eee6ff",
    "PIPELINE": "#e2d5ff",
    "COMPUTE": "#dcf5df",
    "CARRY": "#fff1bf",
    "DRAIN": "#ffe1bd",
    "STORE": "#ffd8c2",
    "RELEASE": "#eeeeee",
}


@dataclass(frozen=True)
class AlgorithmEvent:
    """One ordered action and the on-chip state immediately after it."""

    phase: str
    action: str
    details: tuple[str, ...]
    memory: tuple[str, ...]


def load_json(filepath: str | Path) -> JsonObject:
    """Load one problem or solution JSON document."""
    with Path(filepath).open(encoding="utf-8") as file:
        return json.load(file)


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _html(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _at(data: JsonObject, key: str, index: int, default: Any) -> Any:
    values = data.get(key, [])
    return values[index] if index < len(values) else default


def _shape(problem: JsonObject, tensor: int) -> str:
    height = problem["heights"][tensor]
    width = problem["widths"][tensor]
    dtype = _at(problem, "dtypes", tensor, "?")
    return f"T{tensor} [{height}×{width}] {dtype}"


def _op_name(problem: JsonObject, op: int) -> str:
    primitive = _at(problem, "vector_primitive_families", op, "generic")
    op_type = problem["op_types"][op]
    if op_type != "MatMul" and primitive != "generic":
        return primitive
    return op_type


def hw_legend(problem: JsonObject) -> str:
    """Return the Ascend 910B grounded machine summary."""

    def kb(key: str) -> str:
        return f"{problem.get(key, 0) // 1024}KB"

    frequency = (problem.get("cube_freq_hz", 0) or 0) / 1e9
    bandwidth = (
        f"grounded (pto-isa cycles, {frequency:.2f}GHz): "
        f"GM→L1 {problem.get('bw_gm_l1')}  L0C→GM {problem.get('bw_l0c_gm')}  "
        f"GM↔UB {problem.get('bw_gm_ub')}/{problem.get('bw_ub_gm')} GiB/s"
    )
    lines = [
        "Ascend 910B (1 die)",
        f"Cube/AIC: {problem.get('num_cube_cores')} cores   "
        f"Vector/AIV: {problem.get('num_vector_cores')} cores",
        f"L1/Mat: {kb('l1_capacity')}   L0C/Acc: {kb('cube_capacity')}   UB: {kb('vec_capacity')}",
        bandwidth,
        "Tile: cube 16×16 fractal  |  vector sub-16 rows + 32B DMA-block width",
    ]
    return "Instance Info\n" + "\n".join(lines)


def _graph_prelude(name: str, rankdir: str) -> list[str]:
    return [
        f"digraph {name} {{",
        f"    graph [rankdir={rankdir}, splines=ortho, outputorder=edgesfirst, "
        "overlap=false, concentrate=false, newrank=true, nodesep=0.5, ranksep=0.7, pad=0.2, "
        'bgcolor="white"];',
        '    node [fontname="Helvetica,Arial,sans-serif", fontsize=10, color="#404040"];',
        '    edge [fontname="Helvetica,Arial,sans-serif", fontsize=9, color="#586069", '
        "arrowsize=0.7, penwidth=1.1];",
        "",
    ]


def build_instance_dot(problem: JsonObject) -> str:
    """Build DOT for the raw operation/tensor DAG."""
    lines = _graph_prelude("Instance", "TB")
    lines.append(
        f'    InstanceInfo [label="{_dot_escape(hw_legend(problem))}", shape=note, '
        'style=filled, fillcolor="#fffde1", margin=0.2];'
    )
    for tensor, (width, height) in enumerate(zip(problem["widths"], problem["heights"])):
        label = f"Tensor {tensor}\n{height}×{width}"
        lines.append(
            f'    T{tensor} [label="{label}", shape=box, style="rounded,filled", '
            'fillcolor="#cae8f2", margin=0.12];'
        )
    for op in range(len(problem["op_types"])):
        label = f"Op {op}\n{_op_name(problem, op)}"
        lines.append(
            f'    Op{op} [label="{label}", shape=ellipse, style=filled, fillcolor="#d9f2d9", ordering=out];'
        )
    for op, inputs in enumerate(problem["inputs"]):
        for tensor in inputs:
            lines.append(f"    T{tensor}:s -> Op{op}:n;")
        for tensor in problem["outputs"][op]:
            lines.append(f"    Op{op}:s -> T{tensor}:n;")
    lines.append("}")
    return "\n".join(lines)


def _tensor_roles(
    problem: JsonObject, solution: JsonObject
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    tensor_count = len(problem["widths"])
    producer: dict[int, int] = {}
    consumers: dict[int, list[int]] = {tensor: [] for tensor in range(tensor_count)}
    for op, inputs in enumerate(problem["inputs"]):
        for tensor in inputs:
            consumers[tensor].append(op)
        for tensor in problem["outputs"][op]:
            producer[tensor] = op

    op_steps: dict[int, list[int]] = {}
    for step, ops in enumerate(solution["subgraphs"]):
        for op in ops:
            op_steps.setdefault(op, []).append(step)

    retained: dict[int, list[int]] = {}
    for step, tensors in enumerate(solution.get("tensors_to_retain", [])):
        for tensor in tensors:
            retained.setdefault(tensor, []).append(step)

    roles: dict[int, list[str]] = {}
    required = set(problem.get("required_outputs", []))
    for tensor in range(tensor_count):
        tensor_roles: list[str] = []
        if tensor in retained:
            tensor_roles.append(f"retained {','.join(map(str, retained[tensor]))}")
        prod = producer.get(tensor)
        prod_steps = set(op_steps.get(prod, []))
        consumer_steps = {step for op in consumers[tensor] for step in op_steps.get(op, [])}
        if prod is None or tensor in required or not consumers[tensor] or consumer_steps - prod_steps:
            tensor_roles.append("GM")
        if any(step in consumer_steps for step in prod_steps):
            tensor_roles.append("on-chip")
        if not tensor_roles:
            tensor_roles.append("GM")
        roles[tensor] = tensor_roles
    return roles, op_steps


def _kernel_kind(problem: JsonObject, solution: JsonObject, step: int) -> str:
    ops = solution["subgraphs"][step]
    has_cube = any(problem["op_types"][op] == "MatMul" for op in ops)
    has_vector = any(problem["op_types"][op] in ("Pointwise", "Reduction") for op in ops)
    if has_cube and has_vector:
        return "mixed cube/vector"
    return "cube / AIC" if has_cube else "vector / AIV"


def _kernel_summary(problem: JsonObject, solution: JsonObject, step: int) -> str:
    width, height, contraction = _at(solution, "granularities", step, [0, 0, 0])
    parts_m, parts_n = _at(solution, "parts", step, [0, 0])
    split = _at(solution, "splits", step, 1)
    cores = _at(solution, "cores", step, "?")
    latency = _at(solution, "subgraph_latencies", step, 0.0)
    if parts_m and parts_n:
        grid = f"{parts_m}×{parts_n} regions"
    else:
        grid = "uniform tile grid"
    tile = f"max region {height}×{width}"
    if contraction:
        tile += f"×K{contraction}"
    split_text = f", split ×{split}" if split > 1 else ""
    return (
        f"Kernel {step} · {_kernel_kind(problem, solution, step)} · {tile} · "
        f"{grid}{split_text} · {cores} cores · latency {latency:.1f}"
    )


def _kernel_legend(problem: JsonObject, solution: JsonObject) -> str:
    rows = [
        '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">',
        '<TR><TD COLSPAN="2" BGCOLOR="#f7f7f7"><B>Chosen kernels</B></TD></TR>',
    ]
    for step in range(len(solution["subgraphs"])):
        color = KERNEL_COLORS[step % len(KERNEL_COLORS)]
        rows.append(
            f'<TR><TD BGCOLOR="{color}" WIDTH="18"></TD>'
            f'<TD ALIGN="LEFT">{_html(_kernel_summary(problem, solution, step))}</TD></TR>'
        )
    rows.append("</TABLE>")
    return "".join(rows)


def build_solution_dot(problem: JsonObject, solution: JsonObject) -> str:
    """Build the partition DAG using color only, without fused-group cluster boxes."""
    lines = _graph_prelude("Solution", "TB")
    lines.append(
        f'    InstanceInfo [label="{_dot_escape(hw_legend(problem))}", shape=note, '
        'style=filled, fillcolor="#fffde1", margin=0.2];'
    )
    lines.append(f"    KernelLegend [shape=plain, label=<{_kernel_legend(problem, solution)}>];")
    lines.append("    InstanceInfo:s -> KernelLegend:n [style=invis, weight=20];")

    roles, op_steps = _tensor_roles(problem, solution)
    capacity = problem.get("fast_memory_capacity", 0)
    dtype_bytes = {"FP32": 4, "INT32": 4, "FP16": 2, "BF16": 2, "INT16": 2, "INT8": 1, "BOOL": 1}
    for tensor, (width, height) in enumerate(zip(problem["widths"], problem["heights"])):
        dtype = _at(problem, "dtypes", tensor, "FP32")
        size = width * height * dtype_bytes.get(dtype, 4)
        tensor_roles = roles[tensor]
        fills = [
            "#ffe16b" if role.startswith("retained") else "#c7c7c7" if role == "on-chip" else "#cae8f2"
            for role in tensor_roles
        ]
        style = "rounded,filled,wedged" if len(fills) > 1 else "rounded,filled"
        border = "#c62828" if capacity and size > capacity else "#404040"
        penwidth = 2.4 if capacity and size > capacity else 1.0
        label = f"Tensor {tensor}\n{height}×{width} {dtype}\n[{' | '.join(tensor_roles)}]"
        lines.append(
            f'    T{tensor} [label="{_dot_escape(label)}", shape=box, style="{style}", '
            f'fillcolor="{":".join(fills)}", color="{border}", penwidth={penwidth}, margin=0.12];'
        )

    seq_k = solution.get("seq_k", [])
    op_order = solution.get("op_order", [])
    for op in range(len(problem["op_types"])):
        steps = op_steps.get(op, [])
        if not steps:
            lines.append(
                f'    Op{op} [label="Op {op}\\n{_dot_escape(_op_name(problem, op))}\\nunscheduled", '
                'shape=ellipse, style=filled, fillcolor="#e0e0e0", ordering=out];'
            )
            continue
        extras: list[str] = []
        first = steps[0]
        if (
            first < len(seq_k)
            and seq_k[first] is not None
            and first < len(op_order)
            and op in op_order[first]
        ):
            stream = seq_k[first][op_order[first].index(op)]
            if stream:
                extras.append(f"{'seq-K' if problem['op_types'][op] == 'MatMul' else 'stream'} {stream}")
        if len(steps) > 1:
            extras.append(f"recomputed in {','.join(map(str, steps))}")
        label = f"Op {op}\n{_op_name(problem, op)}"
        if extras:
            label += "\n" + " · ".join(extras)
        fill = ":".join(KERNEL_COLORS[step % len(KERNEL_COLORS)] for step in steps)
        style = "filled,wedged" if len(steps) > 1 else "filled"
        lines.append(
            f'    Op{op} [label="{_dot_escape(label)}", shape=ellipse, style="{style}", '
            f'fillcolor="{fill}", ordering=out];'
        )

    for op, inputs in enumerate(problem["inputs"]):
        for tensor in inputs:
            lines.append(f"    T{tensor}:s -> Op{op}:n;")
        for tensor in problem["outputs"][op]:
            lines.append(f"    Op{op}:s -> T{tensor}:n;")
    producer, _ = _tensor_graph(problem)
    for tensor in range(len(problem["widths"])):
        if tensor not in producer:
            lines.append(f"    KernelLegend:s -> T{tensor}:n [style=invis, weight=10];")
    lines.append("}")
    return "\n".join(lines)


def _tensor_graph(problem: JsonObject) -> tuple[dict[int, int], dict[int, list[int]]]:
    producer: dict[int, int] = {}
    consumers = {tensor: [] for tensor in range(len(problem["widths"]))}
    for op, inputs in enumerate(problem["inputs"]):
        for tensor in inputs:
            consumers[tensor].append(op)
        for tensor in problem["outputs"][op]:
            producer[tensor] = op
    return producer, consumers


def _group_tensors(
    problem: JsonObject, ops: list[int]
) -> tuple[list[int], list[int], dict[int, int], dict[int, list[int]]]:
    producer, consumers = _tensor_graph(problem)
    op_set = set(ops)
    boundary_inputs = sorted(
        {tensor for op in ops for tensor in problem["inputs"][op] if producer.get(tensor) not in op_set}
    )
    required = set(problem.get("required_outputs", []))
    boundary_outputs = sorted(
        {
            tensor
            for op in ops
            for tensor in problem["outputs"][op]
            if tensor in required
            or not consumers[tensor]
            or any(user not in op_set for user in consumers[tensor])
        }
    )
    return boundary_inputs, boundary_outputs, producer, consumers


def _vector_frame(problem: JsonObject, plan: JsonObject, tensor: int) -> str:
    height = problem["heights"][tensor]
    width = problem["widths"][tensor]
    if plan.get("axis") == 1:
        tile_h = 1 if height == 1 else plan.get("free_tile", height)
        tile_w = 1 if width == 1 else plan.get("chunk", width)
    elif plan.get("axis") == 2:
        tile_h = 1 if height == 1 else plan.get("chunk", height)
        tile_w = 1 if width == 1 else plan.get("free_tile", width)
    else:
        strip = plan.get("strip", plan.get("tile", [height, width]))
        tile_h = 1 if height == 1 else min(height, strip[0])
        tile_w = 1 if width == 1 else min(width, strip[1])
    return f"T{tensor} [{tile_h}×{tile_w}]"


def _source_order(solution: JsonObject, step: int) -> list[int]:
    order = _at(solution, "op_order", step, None)
    return list(order) if order is not None else list(solution["subgraphs"][step])


def _append_source_replay(
    events: list[AlgorithmEvent],
    problem: JsonObject,
    solution: JsonObject,
    step: int,
    plan: JsonObject,
    phase: str,
    substitute_reductions: bool,
    carry: tuple[str, ...] = (),
) -> None:
    ops = _source_order(solution, step)
    boundary_inputs, boundary_outputs, _, consumers = _group_tensors(problem, ops)
    positions = {op: position for position, op in enumerate(ops)}
    last_use = {
        tensor: max((positions[user] for user in consumers[tensor] if user in positions), default=-1)
        for tensor in consumers
    }
    boundary_input_set = set(boundary_inputs)
    live: set[int] = set()

    for position, op in enumerate(ops):
        op_type = problem["op_types"][op]
        outputs = problem["outputs"][op]
        substitutes_reduction = substitute_reductions and op_type == "Reduction"
        missing_inputs = sorted(
            tensor
            for tensor in problem["inputs"][op]
            if tensor in boundary_input_set and tensor not in live and not substitutes_reduction
        )
        if missing_inputs:
            live.update(missing_inputs)
            events.append(
                AlgorithmEvent(
                    phase,
                    "LOAD",
                    tuple(f"GM → UB  {_vector_frame(problem, plan, tensor)}" for tensor in missing_inputs),
                    tuple([f"UB: T{tensor}" for tensor in sorted(live)] + list(carry)),
                )
            )
        for tensor in outputs:
            live.add(tensor)
        if substitutes_reduction:
            action = "CARRY"
            details = (f"Op {op} · {_op_name(problem, op)} is supplied by finalized online statistics",)
        else:
            action = "COMPUTE"
            inputs = ", ".join(f"T{tensor}" for tensor in problem["inputs"][op]) or "∅"
            produced = ", ".join(f"T{tensor}" for tensor in outputs)
            details = (f"Op {op} · {_op_name(problem, op)}: {inputs} → {produced}",)
        state = tuple([f"UB: T{tensor}" for tensor in sorted(live)] + list(carry))
        events.append(AlgorithmEvent(phase, action, details, state))

        releasable = sorted(
            tensor
            for tensor in live
            if tensor not in boundary_outputs and last_use.get(tensor, -1) == position
        )
        if releasable:
            for tensor in releasable:
                live.remove(tensor)
            events.append(
                AlgorithmEvent(
                    phase,
                    "RELEASE",
                    tuple(f"release T{tensor} after its last topological use" for tensor in releasable),
                    tuple([f"UB: T{tensor}" for tensor in sorted(live)] + list(carry)),
                )
            )

    if boundary_outputs:
        events.append(
            AlgorithmEvent(
                phase,
                "STORE",
                tuple(f"UB → GM  {_vector_frame(problem, plan, tensor)}" for tensor in boundary_outputs),
                tuple(carry),
            )
        )


def _primitive_work(phase: JsonObject) -> str:
    work: list[str] = []
    for primitive in phase.get("primitives", []):
        counts = []
        if primitive.get("wide"):
            counts.append(f"{primitive['wide']} wide")
        if primitive.get("thin"):
            counts.append(f"{primitive['thin']} thin")
        if primitive.get("stream_starts"):
            counts.append(f"{primitive['stream_starts']} stream start")
        work.append(f"{primitive['kind']} ({', '.join(counts)})")
    return "; ".join(work) if work else "source-DAG statistics cone"


def _stats_ops(problem: JsonObject, solution: JsonObject, step: int) -> list[int]:
    order = _source_order(solution, step)
    reductions = [op for op in order if problem["op_types"][op] == "Reduction"]
    if not reductions:
        return []
    producer, _ = _tensor_graph(problem)
    group = set(order)
    needed = set(reductions)
    pending = list(reductions)
    while pending:
        op = pending.pop()
        for tensor in problem["inputs"][op]:
            parent = producer.get(tensor)
            if parent in group and parent not in needed:
                needed.add(parent)
                pending.append(parent)
    return [op for op in order if op in needed]


def _stats_order(problem: JsonObject, solution: JsonObject, step: int) -> str:
    stats_ops = _stats_ops(problem, solution, step)
    if not stats_ops:
        return "source-DAG statistics cone"
    return "topological statistics: " + " → ".join(f"Op {op} {_op_name(problem, op)}" for op in stats_ops)


def _vector_events(
    problem: JsonObject, solution: JsonObject, step: int, plan: JsonObject
) -> list[AlgorithmEvent]:
    events: list[AlgorithmEvent] = []
    kind = plan["kind"]
    split = plan.get("reduction_split", {})
    seed = split.get("seed", {})
    if seed.get("present"):
        events.append(
            AlgorithmEvent(
                "cross-core prologue",
                "STORE",
                (
                    f"seed {seed['work_units']} spatial output regions with zero",
                    f"non-atomic UB → GM [{seed['valid_rows']}×{seed['valid_cols']}]",
                ),
                (),
            )
        )

    if kind in ("materialized", "pointwise"):
        body = plan.get("body", {})
        strip_h, strip_w = plan.get("strip", plan.get("tile", [0, 0]))
        row_strips, width_strips = plan.get("strip_grid", [1, 1])
        events.append(
            AlgorithmEvent(
                "strip driver",
                "LOOP",
                (
                    f"one logical region → {row_strips}×{width_strips} UB strips",
                    f"representative strip [{strip_h}×{strip_w}], repeat {body.get('trip_count', 1)} times",
                    f"stage {body.get('pipeline_stages', 1)}; "
                    f"physical UB peak {plan.get('chunk_peak_ub_bytes', 0)} B",
                ),
                (),
            )
        )
        _append_source_replay(events, problem, solution, step, plan, "strip body", False)
        return events

    serial = plan.get("serial_phases", {})
    p4_work = plan.get("p4_work", {})
    carry_name = {
        "softmax_flash": "UB carry: (row max m, normalizer l)",
        "layernorm_welford": "UB carry: (mean, M2, count)",
    }.get(kind, "UB carry: reduction accumulator")
    stats_ops = _stats_ops(problem, solution, step)
    boundary_inputs, _, _, _ = _group_tensors(problem, stats_ops or _source_order(solution, step))
    init = serial.get("stats_init", {})
    if init.get("present"):
        events.append(
            AlgorithmEvent(
                "statistics init · serial",
                "LOAD",
                tuple(
                    f"GM → UB  {_vector_frame(problem, plan, tensor)} at chunk {init.get('chunk_index', 0)}"
                    for tensor in boundary_inputs
                ),
                tuple(f"UB: T{tensor}" for tensor in boundary_inputs),
            )
        )
        init_work = (
            _primitive_work(p4_work.get("stats_init", {}))
            if p4_work.get("generated")
            else _stats_order(problem, solution, step)
        )
        events.append(
            AlgorithmEvent(
                "statistics init · serial",
                "COMPUTE",
                (f"extent {init.get('extent', plan.get('chunk'))}", init_work),
                (carry_name,),
            )
        )

    stats = plan.get("stats", {})
    if stats.get("trip_count", 0):
        update_work = (
            _primitive_work(p4_work.get("stats_update", {}))
            if p4_work.get("generated")
            else _stats_order(problem, solution, step)
        )
        events.append(
            AlgorithmEvent(
                "statistics rolled",
                "PIPELINE" if stats.get("pipeline_stages", 1) > 1 else "LOOP",
                (
                    f"chunks {stats.get('first_chunk')}… × {stats.get('trip_count')} iterations",
                    f"stage {stats.get('pipeline_stages', 1)}: load chunk k+1 overlaps statistics(k)",
                    update_work,
                ),
                (carry_name, "UB ping/pong: streamed input"),
            )
        )

    stats_tail = serial.get("stats_tail", {})
    if stats_tail.get("present"):
        events.append(
            AlgorithmEvent(
                "statistics tail · serial",
                "COMPUTE",
                (
                    f"load and reduce ragged chunk {stats_tail.get('chunk_index')}",
                    f"valid extent {stats_tail.get('extent')}",
                ),
                (carry_name,),
            )
        )

    finalize = serial.get("finalize", {})
    if finalize.get("present"):
        finalize_work = (
            _primitive_work(p4_work.get("finalize", {}))
            if p4_work.get("generated")
            else "finalize reduction accumulator"
        )
        events.append(
            AlgorithmEvent(
                "statistics finalize · serial",
                "CARRY",
                (finalize_work,),
                (carry_name.replace("carry", "final statistics"),),
            )
        )

    apply = plan.get("apply", {})
    if apply.get("trip_count", 0) or serial.get("apply_tail", {}).get("present"):
        events.append(
            AlgorithmEvent(
                "spanning apply",
                "PIPELINE" if apply.get("pipeline_stages", 1) > 1 else "LOOP",
                (
                    f"re-read input; chunks × {apply.get('trip_count', 0)} "
                    f"at stage {apply.get('pipeline_stages', 1)}",
                    "source reductions are substituted by finalized statistics",
                ),
                (carry_name.replace("carry", "final statistics"),),
            )
        )
        _append_source_replay(
            events,
            problem,
            solution,
            step,
            plan,
            "apply body · topological order",
            True,
            (carry_name.replace("carry", "final statistics"),),
        )
        apply_tail = serial.get("apply_tail", {})
        if apply_tail.get("present"):
            events.append(
                AlgorithmEvent(
                    "apply tail · serial",
                    "STORE",
                    (
                        f"replay and store ragged chunk {apply_tail.get('chunk_index')}",
                        f"valid extent {apply_tail.get('extent')}",
                    ),
                    (),
                )
            )
    else:
        _, boundary_outputs, _, _ = _group_tensors(problem, _source_order(solution, step))
        events.append(
            AlgorithmEvent(
                "reduction result",
                "STORE",
                tuple(f"UB accumulator → GM  {_shape(problem, tensor)}" for tensor in boundary_outputs),
                (),
            )
        )
    return events


def _region(region: JsonObject) -> str:
    return f"T{region['tensor']} [{region['height']}×{region['width']}]"


def _l0_summary(plan: JsonObject | None) -> str:
    if not plan:
        return "no child L0 phase"
    tile = plan["tile"]
    buffers = plan["buffer_depths"]
    loop = plan["k_loop"]
    return (
        f"{tile[0]}×{tile[1]}×{tile[2]}, {plan['stationarity']}-stationary, "
        f"A/B/C buffers {buffers[0]}/{buffers[1]}/{buffers[2]}, "
        f"child stage {loop['pipeline_stages']}"
    )


def _cube_operand_source(
    mm: JsonObject,
    side: str,
    request_index: int,
    last_use: dict[int, int],
) -> str:
    producer = mm.get(f"{side}_producer", -1)
    if producer >= 0:
        release = " · release after load" if last_use.get(producer) == request_index else ""
        return f"{side.upper()}: L1 result from request {producer}{release}"
    retained = mm.get("retained_panels", {}).get(side, False)
    if retained:
        retained_bytes = mm.get("retained_panels", {}).get(f"{side}_bytes", 0)
        return f"{side.upper()}: GM {_region(mm[side])} → retained L1 once ({retained_bytes} B)"
    return f"{side.upper()}: GM {_region(mm[side])}"


def _cube_last_uses(plan: JsonObject) -> dict[int, int]:
    last_use: dict[int, int] = {}
    for index, mm in enumerate(plan.get("matmuls", [])):
        for producer in (mm.get("lhs_producer", -1), mm.get("rhs_producer", -1)):
            if producer >= 0:
                last_use[producer] = index
    return last_use


def _flow_node(
    node_id: str,
    label: str,
    fillcolor: str,
    *,
    shape: str = "box",
    penwidth: float = 1.2,
) -> str:
    return (
        f'        {node_id} [label="{_dot_escape(label)}", shape={shape}, '
        f'style="rounded,filled", fillcolor="{fillcolor}", penwidth={penwidth}, margin=0.14];'
    )


def _append_cube_request_flow(
    lines: list[str],
    mm: JsonObject,
    index: int,
    last_use: dict[int, int],
    result_nodes: dict[int, str],
) -> tuple[str, str]:
    variants = mm.get("output_variants", [])
    representative = max(variants, key=lambda variant: variant["shape"][0] * variant["shape"][1])
    variant_text = ", ".join(
        f"{variant['shape'][0]}×{variant['shape'][1]} ×{variant['count']}" for variant in variants
    )
    k_loop = mm["k_loop"]
    full_chunks = k_loop["full_chunks"]
    stage_two = k_loop["pipeline_stages"] >= 2 and full_chunks >= 2
    steady = max(0, full_chunks - 2)
    chunk = k_loop["chunk"]
    drain = mm["final_drain"]
    target = "L1 Mat" if drain["target_l1"] else "GM"
    atomic = " atomic-add" if drain["atomic"] else ""
    retained = mm.get("retained_panels", {})
    streamed_sides = [
        side.upper()
        for side in ("lhs", "rhs")
        if mm.get(f"{side}_producer", -1) < 0 and not retained.get(side, False)
    ]
    k_feed = (
        " + ".join(streamed_sides) + " K panel: GM → L1"
        if streamed_sides
        else "K panels: retained/local L1 slices"
    )
    init_l0 = _l0_summary(representative.get("l0_init"))
    rolled_l0 = _l0_summary(representative.get("l0_rolled"))
    l0_summary = f"L0 {init_l0}"
    if rolled_l0 != init_l0:
        l0_summary += f"; rolled {rolled_l0}"
    request_label = (
        f"MATMUL REQUEST {index} · Op {mm['op']}\n"
        f"OUTPUT-TILE LOOP ×{drain['tile_count']} "
        f"({mm['output_grid'][0]}×{mm['output_grid'][1]} grid)\n"
        f"one iteration shown: C [{representative['shape'][0]}×{representative['shape'][1]}] · "
        f"variants {variant_text}\n"
        f"{l0_summary}"
    )
    prefix = f"R{index}"
    input_node = f"{prefix}_Input"
    lines.extend(
        [
            f"    subgraph cluster_{prefix} {{",
            f'        label="{_dot_escape(request_label)}";',
            '        labelloc="t";',
            '        labeljust="l";',
            '        color="#b9a8dc";',
            '        bgcolor="#fcfbff";',
            '        style="rounded";',
            "        penwidth=1.3;",
            '        fontname="Helvetica,Arial,sans-serif";',
            "        fontsize=11;",
            _flow_node(
                input_node,
                (
                    "Panels available to this C tile\n"
                    f"{_cube_operand_source(mm, 'lhs', index, last_use)}\n"
                    f"{_cube_operand_source(mm, 'rhs', index, last_use)}\n"
                    f"Aₖ [{representative['shape'][0]}×{chunk}]  +  "
                    f"Bₖ [{chunk}×{representative['shape'][1]}]"
                ),
                "#d9ecff",
            ),
        ]
    )

    stages: list[tuple[str, str, str, str, str | None]] = []
    if stage_two:
        stages.append(
            (
                f"{prefix}_Fill",
                f"K0 · {k_feed}\nslot 0 / local slice\nfill",
                "#d9ecff",
                "C tile in L0C\nempty",
                None,
            )
        )
        stages.append(
            (
                f"{prefix}_First",
                f"K0: L1 → L0 → Matrix\nK1: {k_feed}\noverlap",
                "#e2d5ff",
                "C tile in L0C\nΣ K0",
                "TMATMUL",
            )
        )
        if steady:
            stages.append(
                (
                    f"{prefix}_Steady",
                    f"Kᵢ: L1 → L0 → Matrix\nKᵢ₊₁: {k_feed}\nrepeat ×{steady}",
                    "#e2d5ff",
                    f"C tile in L0C\nΣ K0…K{full_chunks - 2}",
                    f"TMATMUL_ACC ×{steady}",
                )
            )
        stages.append(
            (
                f"{prefix}_Drain",
                f"K{full_chunks - 1}: L1 → L0 → Matrix\nno next prefetch\ndrain",
                "#dcf5df",
                f"C tile in L0C\nΣ K0…K{full_chunks - 1} · complete",
                "TMATMUL_ACC",
            )
        )
    else:
        remaining = max(0, full_chunks - 1)
        stages.append(
            (
                f"{prefix}_Load0",
                f"K0 · {k_feed}\nfeed",
                "#d9ecff",
                "C tile in L0C\nempty",
                None,
            )
        )
        stages.append(
            (
                f"{prefix}_Work0",
                "K0 → L0 / Matrix\nafter feed completes",
                "#dcf5df",
                "C tile in L0C\nΣ K0",
                "TMATMUL",
            )
        )
        if remaining:
            stages.append(
                (
                    f"{prefix}_Serial",
                    f"K1…K{full_chunks - 1}\nfeed, then Matrix\nserial ×{remaining}",
                    "#dcf5df",
                    f"C tile in L0C\nΣ K0…K{full_chunks - 1} · complete",
                    f"TMATMUL_ACC ×{remaining}",
                )
            )

    if k_loop.get("tail", 0):
        stages.append(
            (
                f"{prefix}_Tail",
                f"ragged K tail {k_loop['tail']}\nGM → L1 → L0 / Matrix\nserial",
                "#dcf5df",
                "C tile in L0C\nfull K complete",
                "TMATMUL_ACC tail",
            )
        )

    pipeline_nodes = [stage[0] for stage in stages]
    accumulator_nodes = [f"{stage[0]}_C" for stage in stages]
    for node_id, label, fillcolor, accumulator, operation in stages:
        lines.append(_flow_node(node_id, label, fillcolor))
        lines.append(_flow_node(f"{node_id}_C", accumulator, "#fff1bf", penwidth=1.5))
        if operation is not None:
            lines.append(
                f'        {node_id}:s -> {node_id}_C:n [style=dashed, color="#7aa874", '
                f'xlabel="{_dot_escape(operation)}", fontsize=8];'
            )
    lines.append(f"        {{ rank=same; {'; '.join(pipeline_nodes)}; }}")
    lines.append(f"        {{ rank=same; {'; '.join(accumulator_nodes)}; }}")
    lines.append(f"        {{ rank=same; {input_node}; {'; '.join(pipeline_nodes)}; }}")
    lines.append(f"        {input_node}:e -> {pipeline_nodes[0]}:w;")
    for lhs, rhs in zip(pipeline_nodes, pipeline_nodes[1:]):
        lines.append(f"        {lhs}:e -> {rhs}:w;")
    for lhs, rhs in zip(accumulator_nodes, accumulator_nodes[1:]):
        lines.append(f'        {lhs}:e -> {rhs}:w [color="#b99100", penwidth=2.2];')

    result_node = f"{prefix}_Result"
    result_label = (
        f"FIXPIPE\nC tile → {target}{atomic}\nonce per output tile\n"
        f"[{drain['valid_rows']}×{drain['valid_cols']}]"
    )
    lines.append(_flow_node(result_node, result_label, "#ffd8c2", penwidth=1.5))
    lines.append(f'        {accumulator_nodes[-1]}:s -> {result_node}:n [color="#b99100", penwidth=2.2];')
    lines.append("    }")

    producers = sorted(
        {producer for producer in (mm.get("lhs_producer", -1), mm.get("rhs_producer", -1)) if producer >= 0}
    )
    for producer in producers:
        lines.append(
            f'    {result_nodes[producer]}:s -> {input_node}:w [color="#8f6bc3", penwidth=1.5, minlen=2];'
        )
    return input_node, result_node


def _algorithm_title(problem: JsonObject, solution: JsonObject, step: int, plan: JsonObject) -> str:
    summary = _kernel_summary(problem, solution, step)
    if _at(solution, "vector_stream", step, None) is not None:
        detail = (
            f"VectorStreamPlan: {plan['kind']} · {plan.get('work_units', '?')} logical work units · "
            f"one diagrammed region · UB peak {plan.get('chunk_peak_ub_bytes', 0)} B"
        )
    else:
        detail = (
            f"CubeSchedulePlan: {plan['spatial_policy']} · {plan['work_units']} work units · "
            f"one diagrammed region · L1 peak {plan['peak_l1_bytes']} B"
        )
    return summary + "\n" + detail


def _build_cube_algorithm_dot(
    problem: JsonObject,
    solution: JsonObject,
    step: int,
    plan: JsonObject,
) -> str:
    """Build a tile-centric flow for one cube work unit."""
    lines = _graph_prelude("CubeKernelSchedule", "TB")
    lines.append("    graph [compound=true];")
    title = _algorithm_title(problem, solution, step, plan)
    color = KERNEL_COLORS[step % len(KERNEL_COLORS)]
    lines.append(
        f'    Title [label="{_dot_escape(title)}", shape=box, style="rounded,filled", '
        f'fillcolor="{color}", penwidth=1.6, margin=0.2];'
    )
    previous = "Title"
    seed = plan.get("seed", {})
    if seed.get("present"):
        seed_label = (
            "separate AIV split-K seed\n"
            f"{seed['work_units']} UB-safe zero stores\n"
            "not part of the cube tile pipeline"
        )
        lines.append(_flow_node("Seed", seed_label, "#ffd8c2", penwidth=1.4))
        lines.append("    Title:s -> Seed:n;")
        previous = "Seed"

    last_use = _cube_last_uses(plan)
    result_nodes: dict[int, str] = {}
    for index, mm in enumerate(plan.get("matmuls", [])):
        input_node, result_node = _append_cube_request_flow(lines, mm, index, last_use, result_nodes)
        producers = {
            producer for producer in (mm.get("lhs_producer", -1), mm.get("rhs_producer", -1)) if producer >= 0
        }
        if index == 0:
            lines.append(f"    {previous}:s -> {input_node}:w [minlen=2];")
        elif index - 1 not in producers:
            lines.append(f"    {previous}:s -> {input_node}:n [style=invis, weight=8];")
        result_nodes[index] = result_node
        previous = result_node
    lines.append("}")
    return "\n".join(lines)


def _event_label(event: AlgorithmEvent, index: int) -> str:
    color = EVENT_COLORS[event.action]
    details = '<BR ALIGN="LEFT"/>'.join(_html(line) for line in event.details) or "&nbsp;"
    return (
        '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="7">'
        f'<TR><TD BGCOLOR="{color}" ALIGN="LEFT"><B>{index + 1:02d} · {_html(event.action)}</B>'
        f'<BR/><FONT POINT-SIZE="9">{_html(event.phase)}</FONT></TD></TR>'
        f'<TR><TD ALIGN="LEFT">{details}</TD></TR></TABLE>>'
    )


def _memory_label(event: AlgorithmEvent) -> str:
    rows = '<BR ALIGN="LEFT"/>'.join(_html(line) for line in event.memory)
    if not rows:
        rows = "no live on-chip value"
    return (
        '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">'
        '<TR><TD BGCOLOR="#f7f7f7"><B>Live after step</B></TD></TR>'
        f'<TR><TD ALIGN="LEFT">{rows}</TD></TR></TABLE>>'
    )


def build_algorithm_dot(problem: JsonObject, solution: JsonObject, step: int) -> str:
    """Build a one-work-unit algorithm/liveness timeline for one chosen kernel."""
    if step < 0 or step >= len(solution["subgraphs"]):
        raise ValueError(f"kernel step must be in [0, {len(solution['subgraphs'])}), got {step}")
    vector_plan = _at(solution, "vector_stream", step, None)
    cube_plan = _at(solution, "cube_schedule", step, None)
    if vector_plan is not None:
        plan = vector_plan
        events = _vector_events(problem, solution, step, plan)
    elif cube_plan is not None:
        return _build_cube_algorithm_dot(problem, solution, step, cube_plan)
    else:
        raise ValueError(
            f"kernel {step} has no vector or cube schedule descriptor; regenerate the solution "
            "with the current PTO Fusebox serializer"
        )

    lines = _graph_prelude("KernelAlgorithm", "TB")
    title = _algorithm_title(problem, solution, step, plan)
    color = KERNEL_COLORS[step % len(KERNEL_COLORS)]
    lines.append(
        f'    Title [label="{_dot_escape(title)}", shape=box, style="rounded,filled", '
        f'fillcolor="{color}", penwidth=1.6, margin=0.2];'
    )
    for index, event in enumerate(events):
        lines.append(f"    E{index} [shape=plain, label={_event_label(event, index)}];")
        lines.append(f"    M{index} [shape=plain, label={_memory_label(event)}];")
        lines.append(f"    {{ rank=same; E{index}; M{index}; }}")
        lines.append(f'    E{index} -> M{index} [dir=none, style=dashed, color="#b0b0b0", constraint=false];')
        if index == 0:
            lines.append("    Title:s -> E0:n;")
        else:
            lines.append(f"    E{index - 1}:s -> E{index}:n;")
            lines.append(f"    M{index - 1}:s -> M{index}:n [style=invis, weight=2];")
    lines.append("}")
    return "\n".join(lines)


def _write_dot(dot: str, out_filepath: str | Path, description: str) -> None:
    path = Path(out_filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dot + "\n", encoding="utf-8")
    print(f"{description} DOT written to {path}")


def generate_instance_dot(problem: JsonObject, out_filepath: str | Path) -> None:
    """Write the raw problem diagram."""
    _write_dot(build_instance_dot(problem), out_filepath, "Instance")


def generate_solution_dot(problem: JsonObject, solution: JsonObject, out_filepath: str | Path) -> None:
    """Write the colored fusion-partition diagram."""
    _write_dot(build_solution_dot(problem, solution), out_filepath, "Solution")


def generate_algorithm_dot(
    problem: JsonObject, solution: JsonObject, step: int, out_filepath: str | Path
) -> None:
    """Write one selected kernel's algorithm timeline."""
    _write_dot(build_algorithm_dot(problem, solution, step), out_filepath, f"Kernel {step} algorithm")


def generate_algorithm_dots(
    problem: JsonObject, solution: JsonObject, output_prefix: str | Path
) -> list[Path]:
    """Write one algorithm DOT per homogeneous vector/cube kernel."""
    prefix = Path(output_prefix)
    paths: list[Path] = []
    for step in range(len(solution["subgraphs"])):
        if (
            _at(solution, "vector_stream", step, None) is None
            and _at(solution, "cube_schedule", step, None) is None
        ):
            continue
        path = prefix.parent / f"{prefix.name}-kernel-{step}.dot"
        generate_algorithm_dot(problem, solution, step, path)
        paths.append(path)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    instance = subparsers.add_parser("instance", help="render the raw tensor/op DAG")
    instance.add_argument("input_json")
    instance.add_argument("out_dot")

    solution = subparsers.add_parser("solution", help="render the colored fusion partition")
    solution.add_argument("input_json")
    solution.add_argument("solution_json")
    solution.add_argument("out_dot")

    algorithm = subparsers.add_parser("algorithm", help="render one kernel algorithm")
    algorithm.add_argument("input_json")
    algorithm.add_argument("solution_json")
    algorithm.add_argument("step", type=int)
    algorithm.add_argument("out_dot")

    algorithms = subparsers.add_parser("algorithms", help="render every vector/cube kernel algorithm")
    algorithms.add_argument("input_json")
    algorithms.add_argument("solution_json")
    algorithms.add_argument("output_prefix")
    return parser.parse_args()


def main() -> None:
    """Run the visualization CLI."""
    args = _parse_args()
    problem = load_json(args.input_json)
    if args.mode == "instance":
        generate_instance_dot(problem, args.out_dot)
        return

    solution = load_json(args.solution_json)
    if args.mode == "solution":
        generate_solution_dot(problem, solution, args.out_dot)
    elif args.mode == "algorithm":
        generate_algorithm_dot(problem, solution, args.step, args.out_dot)
    else:
        generate_algorithm_dots(problem, solution, args.output_prefix)


if __name__ == "__main__":
    main()
