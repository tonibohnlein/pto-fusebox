"""Compile-only checks for the pinned silicon regressions; not numeric closure."""

from __future__ import annotations

import os
import importlib.util
import sys
from pathlib import Path

import pytest
from pto_fusebox import emit_pypto_region


_SPEC = importlib.util.spec_from_file_location(
    "_fusebox_wrong_answer_cases",
    Path(__file__).parents[1] / "device" / "wrong_answer_cases.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_CASES = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CASES  # Torch export resolves the module's globals by name.
_SPEC.loader.exec_module(_CASES)


@pytest.mark.skipif(
    os.environ.get("PTO_FUSEBOX_PYPTO_INTEGRATION") != "1",
    reason="set PTO_FUSEBOX_PYPTO_INTEGRATION=1 with the refreshed PyPTO/PTOAS lane",
)
@pytest.mark.parametrize("case", _CASES.CASE_NAMES)
def test_exact_wrong_answer_candidate_compiles(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypto import ir
    import pypto.language as pl

    solver = Path(
        os.environ.get(
            "PTO_FUSEBOX_TEST_SOLVER",
            str(Path(__file__).parents[2] / "build" / "mlsys_mixed"),
        )
    )
    _, _, graph, region = _CASES.frozen_case(case, solver)
    emitted = emit_pypto_region(graph, region, program_name=case)
    monkeypatch.setenv("PYPTO_CODEGEN_MAX_WORKERS", "2")
    compiled = ir.compile(
        pl.parse_program(emitted.source),
        output_dir=str(tmp_path / case),
        skip_ptoas=False,
    )
    assert tuple(compiled.output_dir.rglob("*.pto"))
    assert tuple((compiled.output_dir / "ptoas").rglob("*.cpp"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
