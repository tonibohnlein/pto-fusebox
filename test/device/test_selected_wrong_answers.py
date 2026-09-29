"""Execute the original incorrect candidates; never use timings as correctness.

Run serially on each allocated device with PTO_FUSEBOX_RUN_DEVICE_TESTS=1 and
PTO_FUSEBOX_DEVICE_ID set. Tests perform the lazy kernel build on first launch.
The history reference retains the original dense-weight algebra and 2e-2
max-relative-error contract, not the new NZ native projection ABI.
"""

from __future__ import annotations

import hashlib
import os
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from pto_fusebox import bind_emitted_call, emit_pypto_region


_SPEC = importlib.util.spec_from_file_location(
    "_fusebox_wrong_answer_cases", Path(__file__).with_name("wrong_answer_cases.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_CASES = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CASES
_SPEC.loader.exec_module(_CASES)
pytestmark = pytest.mark.skipif(
    os.environ.get("PTO_FUSEBOX_RUN_DEVICE_TESTS") != "1",
    reason="requires an explicitly allocated Ascend device",
)


@pytest.mark.parametrize("case", _CASES.CASE_NAMES)
def test_selected_wrong_answer_candidate(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypto import ir, runtime
    from pypto.backend import BackendType
    import pypto.language as pl

    assert "PTO_FUSEBOX_DEVICE_ID" in os.environ, "allocate a device explicitly"
    assert os.environ.get("PTO_FUSEBOX_PLATFORM", "a2a3") == "a2a3", (
        "these frozen candidates target Ascend910B silicon"
    )
    solver = Path(
        os.environ.get(
            "PTO_FUSEBOX_TEST_SOLVER",
            str(Path(__file__).parents[2] / "build" / "mlsys_mixed"),
        )
    )
    module, args, graph, region = _CASES.frozen_case(case, solver)
    emitted = emit_pypto_region(graph, region, program_name=case)
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    config = runtime.RunConfig(
        arch=BackendType.Ascend910B,
        execution_mode=runtime.ExecutionMode.ONBOARD,
        device_id=int(os.environ["PTO_FUSEBOX_DEVICE_ID"]),
        save_kernels=True,
        save_kernels_dir=str(tmp_path / case),
    )
    compiled = ir.compile(pl.parse_program(emitted.source), **config.compile_kwargs())
    integer = case == "int8_m32_n4096_k4096"
    repeats = int(os.environ.get("PTO_FUSEBOX_DEVICE_REPEATS", "50"))
    assert repeats > 0
    for seed in _CASES.SEEDS:
        module, real, reference = _CASES.seeded_inputs(case, module, args, seed)
        dtype = torch.int32 if integer else torch.float32
        sentinel = -(2**31) if integer else torch.nan
        signatures: set[str] = set()
        for _ in range(repeats):
            output = torch.full(reference.shape, sentinel, dtype=dtype)
            compiled(
                *bind_emitted_call(module, graph, emitted, real, (output,)),
                config=config,
            )
            assert torch.isfinite(output).all(), (
                f"{case}, seed={seed}: nonfinite/unwritten output"
            )
            if integer:
                assert not (output == sentinel).any(), "unwritten INT32 output"
                torch.testing.assert_close(
                    output.double(), reference.double(), rtol=0, atol=0
                )
            else:
                error = (output.double() - reference.double()).abs().max()
                scale = reference.double().abs().max()
                assert error / scale <= 2e-2, (
                    f"{case}, seed={seed}: relative error={error / scale}"
                )
            signature = hashlib.sha256(output.numpy().tobytes()).hexdigest()
            signatures.add(signature)
        assert len(signatures) == 1, f"{case}, seed={seed}: unstable output"
        (tmp_path / f"{case}-seed{seed}.sha256").write_text(
            next(iter(signatures)) + "\n"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
