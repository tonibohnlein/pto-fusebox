#!/usr/bin/env python3
"""
Benchmark regression test for mlsys-solver.

Runs all (or selected) benchmarks and compares against a stored baseline.
Exits non-zero if any solution is invalid or if regressions exceed the
tolerance threshold.

Usage:
    python scripts/regression.py                  # run all, compare
    python scripts/regression.py bert-block       # run one benchmark
    python scripts/regression.py --update         # run all, promote improvements
    python scripts/regression.py --budget 5       # use 5s budget for every benchmark
    python scripts/regression.py --quick          # 2s budget (smoke test)
    python scripts/regression.py --fail-on-regress # exit 1 if any regression > tolerance
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent.parent
BENCH_DIR    = REPO_ROOT / "benchmarks"
BASELINE     = REPO_ROOT / "scripts" / "baseline.json"
BUILD_DIR    = REPO_ROOT / "build"

# How much worse than baseline is still "same" (fraction, e.g. 0.03 = 3%)
REGRESS_TOL  = 0.03

# ── Time budget (mirrors main.cpp logic) ──────────────────────────────────────

def default_budget(name: str, num_ops: int) -> float:
    """Return the normal solver budget for this benchmark (seconds)."""
    try:
        idx = int(name.split("mlsys-2026-")[1])
        if idx <= 4:   return 2.0
        if idx <= 8:   return 5.0
        if idx <= 12:  return 15.0
        if idx <= 16:  return 30.0
        if idx <= 20:  return 60.0
        return 120.0
    except (IndexError, ValueError):
        pass
    if num_ops <= 10:  return 2.0
    if num_ops <= 25:  return 5.0
    if num_ops <= 50:  return 15.0
    if num_ops <= 100: return 30.0
    if num_ops <= 200: return 60.0
    return 120.0

def count_ops(bench_path: Path) -> int:
    try:
        with open(bench_path) as f:
            d = json.load(f)
        return len(d.get("op_types", d.get("ops", [])))
    except Exception:
        return 50  # fallback

# ── Run one benchmark ──────────────────────────────────────────────────────────

def run_benchmark(bench_path: Path, budget: float | None, solver: Path,
                   extra_args: list[str] = []) -> dict:
    """
    Run the solver on bench_path with a time limit of `budget` seconds.
    Returns a dict with keys: latency (float), valid (bool), wall (float), output (str).
    """
    name = bench_path.stem
    num_ops = count_ops(bench_path)
    if budget is None:
        budget = default_budget(name, num_ops)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_path = tf.name

    try:
        t0 = time.monotonic()
        result = subprocess.run(
            [str(solver)] + extra_args + [str(bench_path), out_path],
            capture_output=True, text=True,
            timeout=budget * 3 + 10  # generous wall timeout
        )
        wall = time.monotonic() - t0
        output = result.stdout + result.stderr

        # Parse total latency from stderr ("Total: XXXX.X")
        latency = float("inf")
        for line in output.splitlines():
            if line.strip().startswith("Total:"):
                try:
                    latency = float(line.split()[1])
                except (IndexError, ValueError):
                    pass

        # Parse summary block (newer format: "  Final: XXXX.X")
        for line in output.splitlines():
            s = line.strip()
            if s.startswith("Final:"):
                try:
                    latency = float(s.split()[1])
                except (IndexError, ValueError):
                    pass

        # Validity: check the solution JSON was written and passes basic sanity
        valid = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        if latency == float("inf") or latency <= 0:
            valid = False

        return {"latency": latency, "valid": valid, "wall": wall, "output": output}
    except subprocess.TimeoutExpired:
        return {"latency": float("inf"), "valid": False, "wall": budget * 3,
                "output": "TIMEOUT"}
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass

# ── Baseline helpers ───────────────────────────────────────────────────────────

def load_baseline() -> dict:
    if not BASELINE.exists():
        return {}
    with open(BASELINE) as f:
        return json.load(f)

def save_baseline(data: dict) -> None:
    with open(BASELINE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_lat(v):
    if v == float("inf"): return "      inf"
    return f"{v:>9.1f}"

def delta_str(new_lat, base_lat):
    if base_lat is None:   return "    new"
    if base_lat == 0:      return "      -"
    pct = 100.0 * (new_lat - base_lat) / base_lat
    return f"{pct:>+7.1f}%"

def status_str(new_lat, base_lat, valid):
    if not valid:          return "INVALID"
    if base_lat is None:   return "new"
    if new_lat == float("inf"): return "inf"
    pct = (new_lat - base_lat) / max(base_lat, 1e-9)
    if pct < -REGRESS_TOL: return "IMPROVED"
    if pct >  REGRESS_TOL: return "REGRESSED"
    return "same"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmarks", nargs="*",
        help="Benchmark names to run (default: all). Omit .json extension.")
    ap.add_argument("--update", action="store_true",
        help="Update baseline with results that beat the stored best.")
    ap.add_argument("--force-update", action="store_true",
        help="Update baseline with ALL results (even regressions).")
    ap.add_argument("--budget", type=float, default=None,
        help="Time budget in seconds for every benchmark (overrides per-instance budget).")
    ap.add_argument("--quick", action="store_true",
        help="Use 2s budget for all benchmarks (fast smoke test).")
    ap.add_argument("--fail-on-regress", action="store_true",
        help="Exit with code 1 if any benchmark regresses beyond tolerance.")
    ap.add_argument("--strict", action="store_true",
        help="Exit with code 1 on any invalid solution (default: only warn).")
    ap.add_argument("--build", default=str(BUILD_DIR),
        help="Path to build directory containing the mlsys binary.")
    ap.add_argument("--solver-args", default="",
        help="Extra arguments to pass to the solver (e.g. '--v2').")
    args = ap.parse_args()

    solver = Path(args.build) / "mlsys"

    if not solver.exists():
        print(f"Error: solver not found at {solver}")
        print("Build first: cd build && cmake --build . -j$(nproc)")
        sys.exit(1)

    budget = args.budget
    if args.quick:
        budget = 2.0

    # Resolve benchmark list
    if args.benchmarks:
        bench_files = []
        for b in args.benchmarks:
            p = BENCH_DIR / (b if b.endswith(".json") else b + ".json")
            if not p.exists():
                print(f"Warning: benchmark not found: {p}")
            else:
                bench_files.append(p)
    else:
        bench_files = sorted(f for f in BENCH_DIR.glob("*.json"))

    if not bench_files:
        print("No benchmarks to run.")
        sys.exit(0)

    baseline  = load_baseline()
    commit    = git_commit()
    today     = str(date.today())

    print(f"Solver : {solver}")
    print(f"Commit : {commit}")
    print(f"Budget : {'per-instance' if budget is None else f'{budget}s'}")
    print(f"Benchmarks: {len(bench_files)}")
    print()

    hdr = f"{'Instance':<30}  {'Baseline':>9}  {'Result':>9}  {'Delta':>7}  {'Wall':>5}  Status"
    print(hdr)
    print("-" * len(hdr))

    n_improved  = 0
    n_same      = 0
    n_regressed = 0
    n_invalid   = 0
    n_new       = 0
    updated     = {}

    for bench_path in bench_files:
        name = bench_path.stem
        base_entry = baseline.get(name)
        base_lat   = base_entry["latency"] if base_entry else None

        extra = args.solver_args.split() if args.solver_args else []
        run = run_benchmark(bench_path, budget, solver, extra)
        lat   = run["latency"]
        valid = run["valid"]

        status = status_str(lat, base_lat, valid)
        d_str  = delta_str(lat, base_lat)

        if   status == "INVALID":   n_invalid  += 1
        elif status == "new":       n_new      += 1
        elif status == "IMPROVED":  n_improved += 1
        elif status == "REGRESSED": n_regressed+= 1
        else:                       n_same     += 1

        # Decide on update
        if valid and lat < float("inf"):
            if args.force_update:
                updated[name] = {"latency": lat, "commit": commit, "date": today}
            elif args.update and (base_lat is None or lat < base_lat):
                updated[name] = {"latency": lat, "commit": commit, "date": today}

        tag = ""
        if   status == "IMPROVED":  tag = " <--"
        elif status == "REGRESSED": tag = " !!!"
        elif status == "INVALID":   tag = " XXX"

        print(f"{name:<30}  {fmt_lat(base_lat) if base_lat else '      ---'}  "
              f"{fmt_lat(lat)}  {d_str}  {run['wall']:>4.0f}s  {status}{tag}")

    print("-" * len(hdr))
    print(f"  improved={n_improved}  same={n_same}  regressed={n_regressed}"
          f"  invalid={n_invalid}  new={n_new}")

    if updated:
        baseline.update(updated)
        save_baseline(baseline)
        print(f"\nBaseline updated: {len(updated)} entries written to {BASELINE}")

    rc = 0
    if n_invalid > 0:
        msg = f"{n_invalid} invalid solution(s) (may be expected with --quick budget)"
        if args.strict:
            print(f"\nFAIL: {msg}")
            rc = 1
        else:
            print(f"\nWARN: {msg}")
    if args.fail_on_regress and n_regressed > 0:
        print(f"\nFAIL: {n_regressed} regression(s) beyond {REGRESS_TOL*100:.0f}% tolerance")
        rc = 1

    sys.exit(rc)

if __name__ == "__main__":
    main()
