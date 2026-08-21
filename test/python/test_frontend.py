from __future__ import annotations

import json
import stat
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from pto_fusebox import (
    Ascend910BTarget,
    NormalizedGraph,
    export_and_normalize,
    extract_solver_regions,
    normalize_exported,
    solve_graph,
)
from torch import nn


class RmsNorm(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        wide = value.float()
        variance = (wide * wide).mean(dim=-1, keepdim=True)
        return (wide * torch.rsqrt(variance + 1e-5) * self.weight).to(value.dtype)


class Bf16ChainWithFp32Sink(nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        middle: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        intermediate = torch.mm(lhs, middle)
        return torch.mm(intermediate, rhs, out_dtype=torch.float32)


def _capture_documented_example(name: str) -> NormalizedGraph:
    class Softmax(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.softmax(value, dim=-1)

    class Matmul(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs)

    class AttentionCore(nn.Module):
        def forward(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
        ) -> torch.Tensor:
            return torch.mm(torch.softmax(torch.mm(q, k.t()), dim=-1), v)

    class TopKBoundary(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            before = torch.exp(value)
            selected, _ = torch.topk(before, 8, dim=-1)
            return torch.abs(selected)

    examples: dict[str, tuple[nn.Module, tuple[torch.Tensor, ...]]] = {
        "rmsnorm": (RmsNorm(64), (torch.zeros(8, 64, dtype=torch.float16),)),
        "softmax": (Softmax(), (torch.zeros(32, 128),)),
        "linear_rank3": (nn.Linear(16, 24), (torch.zeros(2, 5, 16),)),
        "matmul": (Matmul(), (torch.zeros(16, 32), torch.zeros(32, 24))),
        "qk_softmax_pv": (
            AttentionCore(),
            (torch.zeros(16, 32), torch.zeros(24, 32), torch.zeros(24, 40)),
        ),
        "topk_boundary": (TopKBoundary(), (torch.zeros(4, 32),)),
    }
    module, args = examples[name]
    return export_and_normalize(module, args)


def _render_solver_contract(graph: NormalizedGraph) -> str:
    """Render only semantics consumed by region extraction and the solver."""

    lines = [
        f"inputs={','.join(graph.inputs)}",
        f"outputs={','.join(graph.outputs)}",
    ]
    for value in graph.values:
        shape = "x".join(str(dim) for dim in value.shape) or "scalar"
        lines.append(
            f"value {value.id} {value.role} {value.dtype}[{shape}] "
            f"producer={value.producer or '-'} alias={value.alias_of or '-'} "
            f"target={value.target or '-'}"
        )
    for op in graph.ops:
        attributes = json.dumps(op.attributes, sort_keys=True, separators=(",", ":"))
        lines.append(
            f"op {op.id} {op.kind}({','.join(op.inputs)})->{','.join(op.outputs)} "
            f"supported={int(op.supported)} metadata={int(op.metadata_only)} "
            f"reason={op.opaque_reason or '-'} attrs={attributes}"
        )
    for pattern in graph.patterns:
        lines.append(
            f"pattern {pattern.kind} ops={','.join(pattern.ops)} "
            f"substitutions={','.join(pattern.apply_substitutions)}"
        )
    for region in extract_solver_regions(graph):
        lowered = region.lower(graph)
        problem = lowered.problem
        lines.append(
            f"region {region.id} ops={','.join(region.op_ids)} "
            f"inputs={','.join(region.input_values)} outputs={','.join(region.output_values)} "
            f"diagnostics={'|'.join(region.diagnostics) or '-'}"
        )
        aliases = problem["frontend_mapping"]["solver_tensor_alias_of"]
        synthetic = problem["frontend_mapping"]["solver_tensor_synthetic"]
        for index, value_id in enumerate(lowered.solver_tensor_to_value):
            lines.append(
                f"  tensor {index}={value_id} {problem['dtypes'][index]}"
                f"[{problem['heights'][index]}x{problem['widths'][index]}] "
                f"alias={aliases[index] or '-'} synthetic={int(synthetic[index])}"
            )
        for index, graph_op in enumerate(lowered.solver_op_to_graph):
            inputs = ",".join(str(item) for item in problem["inputs"][index])
            outputs = ",".join(str(item) for item in problem["outputs"][index])
            lines.append(
                f"  solver_op {index} graph={graph_op} {problem['op_types'][index]} "
                f"primitive={problem['vector_primitive_families'][index]} "
                f"geometry={problem['vector_op_geometries'][index]} "
                f"capability={problem['vector_op_capabilities'][index]} "
                f"mixed={problem['mixed_vector_semantics'][index]} "
                f"({inputs})->({outputs})"
            )
        lines.append(
            f"  required={','.join(str(item) for item in problem['required_outputs']) or '-'}"
        )
        lines.append(
            "  p4="
            + json.dumps(problem["p4_patterns"], sort_keys=True, separators=(",", ":"))
        )
    return "\n".join(lines)


@pytest.mark.parametrize(
    "name",
    ["rmsnorm", "softmax", "linear_rank3", "matmul", "qk_softmax_pv", "topk_boundary"],
)
def test_documented_examples_have_exact_solver_dag(name: str) -> None:
    graph = _capture_documented_example(name)
    expected_path = Path(__file__).with_name("frontend_contracts") / f"{name}.txt"
    assert _render_solver_contract(graph) == expected_path.read_text(
        encoding="utf-8"
    ).rstrip("\n")


def test_rmsnorm_keeps_parameter_external_and_lowers_mean() -> None:
    graph = export_and_normalize(
        RmsNorm(64), (torch.randn(8, 64, dtype=torch.float16),)
    )

    assert [op.kind for op in graph.ops] == [
        "cast",
        "mul",
        "sum",
        "mul",
        "add",
        "rsqrt",
        "mul",
        "mul",
        "cast",
    ]
    weight = next(value for value in graph.values if value.target == "weight")
    assert weight.role == "parameter"
    assert weight.id in graph.inputs
    assert "1e-05" not in graph.to_json(indent=None) or "scalars" in graph.to_json(
        indent=None
    )
    assert (
        extract_solver_regions(graph)[0]
        .lower(graph)
        .problem["op_types"]
        .count("Reduction")
        == 1
    )


def test_exported_program_is_the_authoritative_core_api() -> None:
    program = torch.export.export(RmsNorm(16), (torch.randn(2, 16),))
    direct = normalize_exported(program)
    convenience = export_and_normalize(RmsNorm(16), (torch.randn(2, 16),))
    assert direct.to_json() == convenience.to_json()


def test_matmul_out_dtype_normalizes_as_an_ordinary_cube_operation() -> None:
    graph = export_and_normalize(
        Bf16ChainWithFp32Sink(),
        (
            torch.empty(16, 64, dtype=torch.bfloat16, device="meta"),
            torch.empty(64, 2048, dtype=torch.bfloat16, device="meta"),
            torch.empty(2048, 16, dtype=torch.bfloat16, device="meta"),
        ),
    )

    assert [op.kind for op in graph.ops] == ["matmul", "matmul"]
    assert graph.ops[1].attributes["source_operator"] == "aten.mm.dtype"
    assert graph.value_map()[graph.ops[0].outputs[0]].dtype == "bfloat16"
    assert graph.value_map()[graph.ops[1].outputs[0]].dtype == "float32"


def test_buffer_is_named_external_state_without_embedding_payload() -> None:
    class Buffered(nn.Module):
        offset: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("offset", torch.ones(16))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.offset

    graph = export_and_normalize(Buffered(), (torch.randn(4, 16),))
    offset = next(value for value in graph.values if value.target == "offset")
    assert offset.role == "buffer"
    assert offset.id in graph.inputs
    assert "tensor(" not in graph.to_json(indent=None)


def test_softmax_is_generic_dag_with_exact_p4_descriptor() -> None:
    class Softmax(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.softmax(value, dim=-1)

    graph = export_and_normalize(Softmax(), (torch.randn(32, 128),))

    assert [op.kind for op in graph.ops] == ["max", "sub", "exp", "sum", "div"]
    assert [pattern.kind for pattern in graph.patterns] == ["softmax_flash"]
    assert [
        (binding.op, binding.value) for binding in graph.patterns[0].apply_bindings
    ] == [("op0000", "running_max"), ("op0003", "running_sum")]
    decoded = NormalizedGraph.from_json(graph.to_json())
    assert decoded.patterns[0].apply_bindings == graph.patterns[0].apply_bindings
    lowered = extract_solver_regions(graph)[0].lower(graph).problem
    assert lowered["p4_patterns"] == [
        {
            "kind": "softmax_flash",
            "ops": [0, 1, 2, 3, 4],
            "apply_substitutions": [
                {"op": 0, "value": "running_max"},
                {"op": 3, "value": "running_sum"},
            ],
        }
    ]


@pytest.mark.parametrize("rank", [2, 3])
def test_linear_lowers_to_matmul_bias_and_preserves_transposed_weight(
    rank: int,
) -> None:
    module = nn.Linear(16, 24, bias=True)
    shape = (5, 16) if rank == 2 else (2, 5, 16)
    graph = export_and_normalize(module, (torch.randn(*shape),))

    assert [op.kind for op in graph.ops] == ["transpose_view", "matmul", "add"]
    matmul = graph.ops[1]
    assert matmul.attributes["rhs_transposed"] is True
    weight_view = next(value for value in graph.values if value.id == matmul.inputs[1])
    assert weight_view.alias_of
    assert weight_view.strides == (1, 16)
    problem = extract_solver_regions(graph)[0].lower(graph).problem
    assert problem["op_types"] == ["MatMul", "Pointwise"]
    assert problem["vector_op_geometries"][1] == "col_expand"


def test_duplicate_matmul_operands_remain_ordered_and_distinct_roles() -> None:
    class SelfMatmul(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.mm(value, value)

    graph = export_and_normalize(SelfMatmul(), (torch.randn(16, 16),))
    matmul = next(op for op in graph.ops if op.kind == "matmul")
    assert matmul.inputs == (matmul.inputs[0], matmul.inputs[0])
    lowered = extract_solver_regions(graph)[0].lower(graph).problem
    assert lowered["inputs"][0][0] == lowered["inputs"][0][1]


def test_target_rejects_incoherent_matmul_geometry() -> None:
    class Matmul(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.mm(lhs, rhs)

    graph = export_and_normalize(
        Matmul(),
        (torch.randn(16, 32), torch.randn(32, 24)),
    )
    op = next(op for op in graph.ops if op.kind == "matmul")
    values = graph.value_map()
    target = Ascend910BTarget()

    bad_rhs = dict(values)
    bad_rhs[op.inputs[1]] = replace(
        bad_rhs[op.inputs[1]],
        shape=(31, 24),
        strides=(24, 1),
    )
    assert target.admission_reason(op, bad_rhs) == (
        "matmul contraction dimensions do not match: lhs K=32, rhs K=31"
    )

    bad_output = dict(values)
    bad_output[op.outputs[0]] = replace(
        bad_output[op.outputs[0]],
        shape=(16, 25),
        strides=(25, 1),
    )
    assert target.admission_reason(op, bad_output) == (
        "matmul output geometry does not match its inputs: expected [16,24], got [16,25]"
    )


def test_qk_softmax_pv_is_one_generic_cube_vector_cube_region() -> None:
    class AttentionCore(nn.Module):
        def forward(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
        ) -> torch.Tensor:
            logits = torch.mm(q, k.t())
            probabilities = torch.softmax(logits, dim=-1)
            return torch.mm(probabilities, v)

    graph = export_and_normalize(
        AttentionCore(),
        (torch.randn(16, 32), torch.randn(24, 32), torch.randn(24, 40)),
    )

    assert [op.kind for op in graph.ops] == [
        "transpose_view",
        "matmul",
        "max",
        "sub",
        "exp",
        "sum",
        "div",
        "matmul",
    ]
    regions = extract_solver_regions(graph)
    assert len(regions) == 1
    lowered = regions[0].lower(graph).problem
    assert lowered["op_types"] == [
        "MatMul",
        "Reduction",
        "Pointwise",
        "Pointwise",
        "Reduction",
        "Pointwise",
        "MatMul",
    ]
    assert lowered["fuse_cube_vector"] is True
    assert lowered["require_buildable_mixed"] is False
    assert lowered["allow_model_ahead_split_k"] is True
    assert lowered["allow_model_ahead_multi_reduction_stream"] is True
    assert lowered["allow_model_ahead_mixed_multi_roundtrip"] is True
    assert lowered["require_uniform_cube_dag_grid"] is False


def test_topk_is_opaque_and_splits_supported_regions() -> None:
    class TopKBoundary(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            before = torch.exp(value)
            selected, _ = torch.topk(before, 8, dim=-1)
            return torch.abs(selected)

    graph = export_and_normalize(TopKBoundary(), (torch.randn(4, 32),))
    opaque = [op for op in graph.ops if not op.supported]
    assert len(opaque) == 1
    assert "topk" in str(opaque[0].attributes["source_operator"])
    regions = extract_solver_regions(graph)
    assert [
        [graph.op_map()[op_id].kind for op_id in region.op_ids] for region in regions
    ] == [
        ["exp"],
        ["abs"],
    ]


def test_opaque_bypass_diamond_cannot_rejoin_across_boundary() -> None:
    class OpaqueBypassDiamond(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            shared = torch.exp(value)
            opaque = torch.sin(shared)
            bypass = torch.abs(shared)
            return opaque + bypass

    graph = export_and_normalize(OpaqueBypassDiamond(), (torch.randn(4, 32),))
    assert [op.kind for op in graph.ops] == ["exp", "opaque", "abs", "add"]
    regions = extract_solver_regions(graph)
    assert [region.op_ids for region in regions] == [("op0000", "op0002"), ("op0003",)]
    assert regions[1].diagnostics == (
        "boundary op0001: unsupported operator aten.sin.default",
    )


def test_unsafe_view_and_non_matmul_transpose_are_opaque() -> None:
    class UnsafeViews(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            changed = value.reshape(4, 16)
            return torch.exp(changed.t())

    graph = export_and_normalize(UnsafeViews(), (torch.randn(8, 8),))
    reasons = [op.opaque_reason for op in graph.ops if not op.supported]
    assert reasons == [
        "aten.reshape.default may copy storage and is not a metadata-only alias",
        "only an immediately-consumed rank-2 matmul transpose is supported",
    ]


def test_nondefault_operator_semantics_decline_instead_of_being_dropped() -> None:
    class NonDefaultSemantics(nn.Module):
        def forward(
            self, lhs: torch.Tensor, rhs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            scaled_add = torch.add(lhs, rhs, alpha=2)
            widened_softmax = torch.softmax(
                lhs.to(torch.float16), dim=-1, dtype=torch.float32
            )
            return scaled_add, widened_softmax

    graph = export_and_normalize(
        NonDefaultSemantics(),
        (torch.randn(4, 16), torch.randn(4, 16)),
    )
    reasons = [op.opaque_reason for op in graph.ops if not op.supported]
    assert reasons == [
        "aten.add.Tensor has unsupported keyword arguments: alpha",
        "dtype-changing softmax must be represented as explicit casts",
    ]


def test_nonfinite_scalar_is_rejected_before_publication() -> None:
    class NonfiniteScalar(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + float("inf")

    with pytest.raises(ValueError, match="cannot contain a non-finite float"):
        export_and_normalize(NonfiniteScalar(), (torch.randn(4, 16),))


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_dtype_changing_reductions_decline(reduction: str) -> None:
    class DtypeChangingReduction(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            if reduction == "sum":
                return torch.sum(value, dim=-1, keepdim=True, dtype=torch.float32)
            return torch.mean(value, dim=-1, keepdim=True, dtype=torch.float32)

    graph = export_and_normalize(
        DtypeChangingReduction(),
        (torch.randn(4, 16, dtype=torch.float16),),
    )
    assert extract_solver_regions(graph) == []


@pytest.mark.parametrize("operation", ["reshape", "contiguous"])
def test_copy_capable_views_are_opaque(operation: str) -> None:
    class CopyCapableView(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            transposed = value.t()
            if operation == "reshape":
                return transposed.reshape(2, 6)
            return transposed.contiguous()

    graph = export_and_normalize(CopyCapableView(), (torch.randn(3, 4),))
    reasons = [op.opaque_reason for op in graph.ops if not op.supported]
    assert reasons[-1] == (
        f"aten.{operation}.default may copy storage and is not a metadata-only alias"
    )


def test_duplicate_structured_outputs_preserve_order_and_pytree() -> None:
    class StructuredOutputs(nn.Module):
        def forward(self, value: torch.Tensor) -> dict[str, object]:
            result = torch.exp(value)
            return {"pair": (result, result), "original": [value]}

    graph = export_and_normalize(StructuredOutputs(), (torch.randn(4, 16),))
    assert graph.outputs == ("v0001", "v0001", "v0000")
    assert graph.output_tree_spec == [
        1,
        {
            "type": "builtins.dict",
            "context": '["pair", "original"]',
            "children_spec": [
                {
                    "type": "builtins.tuple",
                    "context": "null",
                    "children_spec": [
                        {"type": None, "context": None, "children_spec": []},
                        {"type": None, "context": None, "children_spec": []},
                    ],
                },
                {
                    "type": "builtins.list",
                    "context": "null",
                    "children_spec": [
                        {"type": None, "context": None, "children_spec": []},
                    ],
                },
            ],
        },
    ]
    lowered = extract_solver_regions(graph)[0].lower(graph).problem
    assert lowered["required_outputs"] == [1]


def test_metadata_only_graph_output_maps_to_allocation_owner() -> None:
    class AliasedOutput(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.exp(value).view(2, 8)

    graph = export_and_normalize(AliasedOutput(), (torch.randn(2, 8),))
    region = extract_solver_regions(graph)[0]
    lowered = region.lower(graph).problem
    assert region.output_values == ("v0002",)
    assert lowered["frontend_mapping"]["region_outputs"] == ["v0002"]
    assert lowered["frontend_mapping"]["region_output_allocation_owners"] == ["v0001"]
    assert lowered["required_outputs"] == [1]


def test_metadata_alias_between_compute_ops_preserves_solver_dependency() -> None:
    class AliasedIntermediate(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            intermediate = torch.exp(value).view(2, 8)
            return intermediate + 1.0

    graph = export_and_normalize(AliasedIntermediate(), (torch.randn(2, 8),))
    lowered = extract_solver_regions(graph)[0].lower(graph).problem
    assert lowered["inputs"] == [[0], [1]]
    assert lowered["outputs"] == [[1], [2]]


def test_internal_transpose_is_an_explicit_region_boundary() -> None:
    class InternalTranspose(nn.Module):
        def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.exp(value).t(), weight)

    graph = export_and_normalize(
        InternalTranspose(),
        (torch.randn(8, 16), torch.randn(8, 24)),
    )
    regions = extract_solver_regions(graph)
    assert [region.op_ids for region in regions] == [("op0000",), ("op0002",)]
    assert regions[1].diagnostics == (
        "boundary op0001: transpose of a region-produced value requires an explicit solver layout edge",
    )


def test_non_dense_external_input_declines() -> None:
    class Pointwise(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.exp(value)

    source = torch.randn(8, 4).t()
    graph = export_and_normalize(Pointwise(), (source,))
    assert graph.values[0].strides == (1, 4)
    assert extract_solver_regions(graph) == []


def test_nonzero_external_storage_offset_declines() -> None:
    class Pointwise(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.exp(value)

    source = torch.randn(9, 8)[1:]
    graph = export_and_normalize(Pointwise(), (source,))
    assert graph.values[0].strides == (8, 1)
    assert graph.values[0].storage_offset == 8
    assert extract_solver_regions(graph) == []


def test_geometry_preserving_alias_cannot_hide_nondense_storage() -> None:
    class ViewedMatmul(nn.Module):
        def forward(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.mm(value.view(4, 3), weight)

    source = torch.randn(3, 4).t()
    graph = export_and_normalize(ViewedMatmul(), (source, torch.randn(3, 8)))
    assert graph.values[0].strides == (1, 4)
    assert extract_solver_regions(graph) == []


def test_multiple_outputs_and_returned_consumed_value_are_required_outputs() -> None:
    class TwoOutputs(nn.Module):
        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            first = value * 2.0
            second = first + 3.0
            return first, second

    graph = export_and_normalize(TwoOutputs(), (torch.randn(4, 16),))
    assert len(graph.outputs) == 2
    problem = extract_solver_regions(graph)[0].lower(graph).problem
    assert len(problem["required_outputs"]) == 2


def test_dynamic_outer_dimension_is_preserved_and_static_lowering_declines(
    tmp_path: Path,
) -> None:
    class DynamicPointwise(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.exp(value)

    batch = torch.export.Dim("batch", min=1, max=128)
    graph = export_and_normalize(
        DynamicPointwise(),
        (torch.randn(8, 32),),
        dynamic_shapes={"value": {0: batch}},
    )
    assert graph.constraints["s0"] == {"minimum": 1, "maximum": 128}
    regions = extract_solver_regions(graph)
    with pytest.raises(ValueError, match="schedule-defining symbolic"):
        regions[0].lower(graph)
    result = solve_graph(graph, solver_binary=tmp_path / "not-needed")
    assert [region.status for region in result.regions] == ["declined"]
    assert result.solver_binary == ""


def test_shape_derived_symbolic_scalar_is_an_explicit_boundary() -> None:
    class ShapeScale(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * value.shape[0]

    static_graph = export_and_normalize(ShapeScale(), (torch.randn(4, 8),))
    assert static_graph.ops[0].supported
    assert static_graph.ops[0].attributes["scalars"] == [{"position": 1, "value": 4}]

    batch = torch.export.Dim("batch", min=1, max=8)
    dynamic_graph = export_and_normalize(
        ShapeScale(),
        (torch.randn(4, 8),),
        dynamic_shapes={"value": {0: batch}},
    )
    assert len(dynamic_graph.ops) == 1
    op = dynamic_graph.ops[0]
    assert not op.supported
    assert op.inputs == (dynamic_graph.inputs[0],)
    assert op.opaque_reason is not None
    assert "depends on unrepresented symbolic or metadata value" in op.opaque_reason
    assert extract_solver_regions(dynamic_graph) == []


def test_normalized_json_is_deterministic_and_rejects_unknown_schema() -> None:
    first = export_and_normalize(RmsNorm(16), (torch.randn(2, 16),))
    second = export_and_normalize(RmsNorm(16), (torch.randn(2, 16),))
    assert first.to_json() == second.to_json()
    decoded = NormalizedGraph.from_json(first.to_json())
    assert decoded.to_json() == first.to_json()
    assert decoded.output_tree_spec == first.output_tree_spec
    assert decoded.values[0].strides == first.values[0].strides
    assert decoded.values[0].storage_offset == first.values[0].storage_offset
    encoded = first.to_dict()
    encoded["schema_version"] = "pto_fusebox.normalized_graph.v2"
    with pytest.raises(ValueError, match="unsupported normalized graph schema"):
        NormalizedGraph.from_dict(encoded)


def test_normalized_schema_rejects_malformed_dependencies_and_alias_cycles() -> None:
    class TwoOps(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.abs(torch.exp(value))

    graph = export_and_normalize(TwoOps(), (torch.randn(4, 16),))
    reversed_ops = json.loads(graph.to_json())
    reversed_ops["ops"] = list(reversed(reversed_ops["ops"]))
    with pytest.raises(ValueError, match="not topologically ordered"):
        NormalizedGraph.from_dict(reversed_ops)

    alias_cycle = json.loads(graph.to_json())
    alias_cycle["values"][0]["alias_of"] = alias_cycle["values"][1]["id"]
    alias_cycle["values"][1]["alias_of"] = alias_cycle["values"][0]["id"]
    with pytest.raises(ValueError, match="alias cycle"):
        NormalizedGraph.from_dict(alias_cycle)

    missing_input = json.loads(graph.to_json())
    missing_input["inputs"] = []
    with pytest.raises(ValueError, match="producerless value .* graph input"):
        NormalizedGraph.from_dict(missing_input)

    malformed_arity = json.loads(graph.to_json())
    malformed_arity["ops"][1]["inputs"] = []
    with pytest.raises(ValueError, match=r"op0001 \(abs\) expects 1 tensor input"):
        NormalizedGraph.from_dict(malformed_arity)


def test_normalized_schema_rejects_invalid_pattern_substitution() -> None:
    class Softmax(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.softmax(value, dim=-1)

    encoded = json.loads(
        export_and_normalize(Softmax(), (torch.randn(4, 16),)).to_json()
    )
    encoded["patterns"][0]["ops"] = encoded["patterns"][0]["ops"][:-1]
    encoded["patterns"][0]["apply_substitutions"] = ["op0004"]
    with pytest.raises(ValueError, match="substitutions must belong"):
        NormalizedGraph.from_dict(encoded)


def test_mixed_scalar_semantics_require_exact_constants() -> None:
    class ScalarNearMisses(nn.Module):
        def forward(
            self, value: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            exponent = torch.exp(-value)
            return exponent + 1, exponent + 2, 1 / (value + 3), 3 / (value + 4)

    graph = export_and_normalize(ScalarNearMisses(), (torch.randn(2, 8),))
    semantics = [
        semantic
        for region in extract_solver_regions(graph)
        for semantic in region.lower(graph).problem["mixed_vector_semantics"]
    ]
    assert semantics == [
        "neg",
        "exp",
        "scalar_add",
        "none",
        "none",
        "recip",
        "mul",
        "none",
        "recip",
        "mul",
    ]


def test_scalar_add_and_sub_use_grounded_scalar_primitive() -> None:
    class ScalarArithmetic(nn.Module):
        def forward(
            self, lhs: torch.Tensor, rhs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return lhs + 2, lhs - 2, lhs + rhs

    graph = export_and_normalize(
        ScalarArithmetic(),
        (torch.randn(2, 8), torch.randn(2, 8)),
    )
    primitives = [
        primitive
        for region in extract_solver_regions(graph)
        for primitive in region.lower(graph).problem["vector_primitive_families"]
    ]
    assert primitives == ["scalar_add", "scalar_add", "add"]


def test_reciprocal_and_division_keep_distinct_grounded_costs() -> None:
    class ReciprocalAndDivision(nn.Module):
        def forward(
            self, lhs: torch.Tensor, rhs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return 1 / (lhs + 3), lhs / rhs

    graph = export_and_normalize(
        ReciprocalAndDivision(),
        (torch.randn(2, 8), torch.randn(2, 8)),
    )
    problems = [region.lower(graph).problem for region in extract_solver_regions(graph)]
    primitive_costs = [
        (primitive, slope, fixed)
        for problem in problems
        for primitive, slope, fixed in zip(
            problem["vector_primitive_families"],
            problem["vec_slopes"],
            problem["vec_fixed_costs"],
            strict=True,
        )
    ]

    assert ("recip", 2.0, 30.0) in primitive_costs
    assert ("div", 4.0, 30.0) in primitive_costs


def test_serialized_vector_costs_match_the_grounded_primitive_table() -> None:
    class GroundedChain(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            logged = torch.log(value)
            absolute = torch.abs(logged)
            rooted = torch.sqrt(absolute)
            negated = -rooted
            shifted = negated + 1.0
            return shifted * 2.0

    graph = export_and_normalize(GroundedChain(), (torch.rand(4, 64) + 1.0,))
    problem = extract_solver_regions(graph)[0].lower(graph).problem

    assert problem["vector_primitive_families"] == [
        "log",
        "abs",
        "sqrt",
        "scalar_mul",
        "scalar_add",
        "scalar_mul",
    ]
    assert list(
        zip(
            problem["vec_slopes"],
            problem["vec_fixed_costs"],
            strict=True,
        )
    ) == [
        (2.0, 33.0),
        (1.0, 29.0),
        (2.0, 39.0),
        (1.0, 26.0),
        (1.0, 31.0),
        (1.0, 26.0),
    ]


def test_bf16_vector_arithmetic_is_boundary_but_cast_is_admitted() -> None:
    class Bf16Arithmetic(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            wide = value.float()
            return (value + value).float() + wide

    graph = export_and_normalize(
        Bf16Arithmetic(), (torch.randn(4, 32, dtype=torch.bfloat16),)
    )
    regions = extract_solver_regions(graph)
    kinds = [
        [graph.op_map()[op_id].kind for op_id in region.op_ids] for region in regions
    ]
    assert kinds == [["cast"], ["cast", "add"]]


def test_mixed_dtype_vector_arithmetic_declines_without_implicit_promotion() -> None:
    class MixedDtypeAdd(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return lhs + rhs

    graph = export_and_normalize(
        MixedDtypeAdd(),
        (torch.randn(4, 16, dtype=torch.float16), torch.randn(4, 16)),
    )
    assert extract_solver_regions(graph) == []


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape"),
    [
        ((2, 1, 4), (1, 3, 4)),
        ((2, 3, 4), (3, 4)),
    ],
)
def test_unrepresentable_torch_broadcast_declines(
    lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]
) -> None:
    class BroadcastAdd(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return lhs + rhs

    graph = export_and_normalize(
        BroadcastAdd(),
        (torch.randn(lhs_shape), torch.randn(rhs_shape)),
    )
    assert extract_solver_regions(graph) == []


@pytest.mark.parametrize("rhs_shape", [(2, 3, 1), (4,)])
def test_representable_single_axis_broadcast_is_admitted(
    rhs_shape: tuple[int, ...],
) -> None:
    class BroadcastAdd(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return lhs + rhs

    graph = export_and_normalize(
        BroadcastAdd(),
        (torch.randn(2, 3, 4), torch.randn(rhs_shape)),
    )
    assert len(extract_solver_regions(graph)) == 1


def test_ambiguous_scalar_tensor_broadcast_declines() -> None:
    class BroadcastAdd(nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return lhs + rhs

    graph = export_and_normalize(
        BroadcastAdd(),
        (torch.randn(4, 16), torch.randn(1, 1)),
    )
    assert extract_solver_regions(graph) == []


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype", "solver_dtypes"),
    [
        (torch.float16, torch.bfloat16, ["FP16", "FP32", "BF16"]),
        (torch.bfloat16, torch.float16, ["BF16", "FP32", "FP16"]),
        (torch.float32, torch.int8, ["FP32", "FP16", "INT8"]),
    ],
)
def test_casts_expand_to_native_910b_chains(
    source_dtype: torch.dtype,
    target_dtype: torch.dtype,
    solver_dtypes: list[str],
) -> None:
    class Cast(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.to(target_dtype)

    graph = export_and_normalize(Cast(), (torch.randn(4, 16).to(source_dtype),))
    assert [op.kind for op in graph.ops] == ["cast"]
    lowered = extract_solver_regions(graph)[0].lower(graph)
    assert lowered.problem["dtypes"] == solver_dtypes
    assert lowered.problem["op_types"] == ["Pointwise", "Pointwise"]
    assert lowered.problem["vector_primitive_families"] == ["cast", "cast"]
    assert lowered.solver_op_to_graph == (graph.ops[0].id, graph.ops[0].id)
    assert lowered.problem["frontend_mapping"]["solver_tensor_synthetic"] == [
        False,
        True,
        False,
    ]


def test_int8_is_limited_to_an_unconsumed_returned_cast_result() -> None:
    class Int8Intermediate(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.to(torch.int8).to(torch.float32)

    graph = export_and_normalize(Int8Intermediate(), (torch.randn(4, 16),))
    region = extract_solver_regions(graph)[0]

    with pytest.raises(
        ValueError, match="INT8 only as an unconsumed returned cast result"
    ):
        region.lower(graph)


def test_same_dtype_to_is_alias_unless_copy_is_requested() -> None:
    class SameDtypeTo(nn.Module):
        def __init__(self, *, copy: bool) -> None:
            super().__init__()
            self.copy = copy

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.to(torch.float32, copy=self.copy)

    alias_graph = export_and_normalize(SameDtypeTo(copy=False), (torch.randn(2, 8),))
    assert [op.kind for op in alias_graph.ops] == ["view"]
    assert alias_graph.ops[0].metadata_only
    assert extract_solver_regions(alias_graph) == []

    copy_graph = export_and_normalize(SameDtypeTo(copy=True), (torch.randn(2, 8),))
    assert not copy_graph.ops[0].supported
    assert copy_graph.ops[0].opaque_reason == (
        "same-dtype to(copy=True) requires an explicit copy lowering"
    )
    assert extract_solver_regions(copy_graph) == []


def test_same_dtype_to_copy_operator_is_not_a_metadata_alias() -> None:
    class SameDtypeToCopy(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten._to_copy.default(value, dtype=torch.float32)

    graph = export_and_normalize(SameDtypeToCopy(), (torch.randn(2, 8),))
    assert not graph.ops[0].supported
    assert graph.ops[0].opaque_reason == (
        "same-dtype to(copy=True) requires an explicit copy lowering"
    )
    assert extract_solver_regions(graph) == []


def test_solve_graph_uses_versioned_json_and_preserves_mappings(tmp_path: Path) -> None:
    class Pointwise(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.exp(value)

    graph = export_and_normalize(Pointwise(), (torch.randn(4, 16),))
    solver = tmp_path / "solver.py"
    solver.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "assert sys.argv[1:3]==['--threads','2']\n"
        "problem=json.loads(pathlib.Path(sys.argv[3]).read_text())\n"
        "assert problem['schema_version']=='pto_fusebox.problem.v1'\n"
        "pathlib.Path(sys.argv[4]).write_text(json.dumps({"
        "'schema_version':'pto_fusebox.solution.v4','steps':[{"
        "'kind':'vector','ops':[0],'op_order':[0],"
        "'launch':{'tile':[16,4,1],'parts':[1,1],'split':1,'cores':1},"
        "'latency_cycles':1.0,'plan':{}}]}))\n",
        encoding="utf-8",
    )
    solver.chmod(solver.stat().st_mode | stat.S_IXUSR)

    result = solve_graph(graph, solver_binary=solver, solver_workers=2)

    assert result.successful
    assert result.regions[0].solver_op_to_graph == (graph.ops[0].id,)
    problem = result.regions[0].problem
    assert problem is not None
    assert problem["frontend_mapping"]["region_id"] == "region0000"


def test_solve_graph_rejects_nonpositive_worker_count(tmp_path: Path) -> None:
    class Pointwise(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.exp(value)

    graph = export_and_normalize(Pointwise(), (torch.randn(4, 16),))
    with pytest.raises(ValueError, match="solver_workers must be a positive integer"):
        solve_graph(graph, solver_binary=tmp_path / "not-used", solver_workers=0)


def test_invalid_solver_subgraph_is_not_reported_as_solved(tmp_path: Path) -> None:
    class Pointwise(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.exp(value)

    graph = export_and_normalize(Pointwise(), (torch.randn(4, 16),))
    solver = tmp_path / "invalid_solver.py"
    solver.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({"
        "'schema_version':'pto_fusebox.solution.v4','steps':[{"
        "'kind':'vector','ops':[999],'op_order':[999],"
        "'launch':{'tile':[16,4,1],'parts':[1,1],'split':1,'cores':1},"
        "'latency_cycles':1.0,'plan':{}}]}))\n",
        encoding="utf-8",
    )
    solver.chmod(solver.stat().st_mode | stat.S_IXUSR)

    result = solve_graph(graph, solver_binary=solver)
    assert [region.status for region in result.regions] == ["infeasible"]
    assert (
        result.regions[0].diagnostics[-1]
        == "solver step 0 references an invalid subgraph"
    )


def test_missing_solver_does_not_trigger_a_build(tmp_path: Path) -> None:
    class Pointwise(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.exp(value)

    graph = export_and_normalize(Pointwise(), (torch.randn(2, 4),))
    with pytest.raises(FileNotFoundError, match="build it explicitly"):
        solve_graph(graph, solver_binary=tmp_path / "missing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
