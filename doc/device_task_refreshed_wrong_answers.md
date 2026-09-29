# Device task: frozen wrong-answer candidates on the refreshed compiler lane

## Objective and scope

Determine whether the two silent wrong-answer findings from the
`41bac5ac` campaign persist on the refreshed PyPTO/PTOAS lane. Execute the
original INT8 cube candidate and both affected Flash-MTP history candidates,
with the historically correct history candidate as a control.

This is a focused static-region correctness campaign, not four-model closure,
an NZ-layout implementation, or cost-model calibration. No product edits,
coefficient fitting, tolerance relaxation, replacement solver picks, or PRs.
Campaign-local diagnostic harnesses and independent controls are allowed;
keep their source and document every correction. Do not edit the pinned trees.

## 0. Immutable publication pins and environment

| Component | Repository / branch | Revision |
| --- | --- | --- |
| Fusebox code under test | `https://github.com/tonibohnlein/pto-fusebox.git`, `main` | `74de905bb0b064f3f851d9336a7897889d580927` |
| PyPTO integration | `https://github.com/tonibohnlein/pypto.git`, `fix/fusebox-integration-cast-fragment` | `338eeaf667963c7a0f6066b5ca89bdf6f01b20f2` |
| PyPTO upstream base | `https://github.com/hw-native-sys/pypto.git` | `54521824f2777fcba180322b7148ce335d254709` |
| PyPTO runtime | Integration gitlink and checkout | `6e383fc57c12a8bd4dce045289b8b4f5d55a15f1` |
| PyPTO-lib | `https://github.com/hw-native-sys/pypto-lib.git` | `e9d0f1ceebf4cf57c5041ac4b5e197686057c3a3` |
| PTOAS | Integration `toolchain/versions.env` | `v0.65` |

Check out the integration SHA directly. No remote rebase, merge, cherry-pick,
or manual conflict resolution is needed or authorized. Verify the upstream
base is an ancestor, submodules equal their gitlinks, and tracked trees are
clean. Record full revisions, tree hashes, Python/Torch versions, imports,
loaded extension paths, compiler versions, and actual build commands.

Build PyPTO and its runtime inside the campaign root, using the checkout's
dependency/build constraints (including nanobind). Do not reuse foreign `.so`
files or allow another runtime to shadow the campaign build. Install the
checkout-pinned PTOAS and verify it through PyPTO's own resolver. Explicitly
set `PTOAS_ROOT`; changing PATH does not override a foreign PTOAS_ROOT.
The pinned x86-64 wheel SHA-256 is
`28ba50ddc684b3b011262cd26677a2509428beaf5e267cb6bee5b4c1f776712b`.
If using a different official payload, record its own checksum, not the wheel's.

Select two Ascend 910B devices through the host's allocation/queue mechanism.
Guard every device runner against executing outside its allocation. Run device
jobs sequentially unless the allocator guarantees isolation. Do not run beside
a foreign all-device sweep. Use disk-backed campaign directories and separate
artifact roots per device and case. Choose and record explicit build/test
worker counts appropriate for the remote machine.

### Host gates

Set `PTO_FUSEBOX_PYPTO_LIB_ROOT`, `PTO_FUSEBOX_TEST_SOLVER`, and the two sweep
binary variables to these checkouts. Build both repositories. Require:

1. Full PyPTO unit suite: zero unexpected failures. The local Python 3.14 run
   gave 15,573 passed, 78 skipped, 1 xfailed; version-dependent collection or
   skips must be explained, not silently rebaselined.
2. Fusebox CTest: 6/6; default `test/python` suite: zero failures, with collected
   IDs and skip reasons retained. Unset the integration flag for this run.
3. With `PTO_FUSEBOX_PYPTO_INTEGRATION=1`, run:

   ```bash
   python -m pytest test/python/test_source_pypto_integration.py \
       test/python/test_wrong_answer_reproducers.py -q --tb=short
   ```

   Local result on these pins: **84 passed, zero skipped, zero failed** with
   the real PTOAS v0.65. All four frozen regression rows must actually run.
4. Changed-surface Ruff/format, Pyright, and `git diff --check` pass. Preserve
   complete logs; do not hide failing IDs behind truncated output.
5. Collect the four tests in `test/device/test_selected_wrong_answers.py`.
   Run PyPTO `tests/st/runtime/ops/test_abs.py` on each allocated device before
   and after device work; require 4/4 on each idle device.

Stop device work on provenance, build, or unexpected host-gate failure. A
different pass/skip count alone requires an ID/reason diff, not an automatic
product-defect classification. Never fix a product test inside this campaign.

## 1. Freeze the exact subjects, not the new solver selection

Use the checked-in `test/device/wrong_answer_cases.py` helpers. Load the module
with a registered importlib module (as both checked-in test files do); do not
use an unregistered `runpy` module that Torch export cannot resolve.

| Test case | Frozen identity | Expected structure |
| --- | --- | --- |
| `int8_m32_n4096_k4096` | `p1_q8_s1_k64_i64` | INT8 inputs, exact INT32 output; physical L0 tile `[32,512,64]` |
| `flashmtp_history_candidate_0` | `candidate_0` | Three submissions; cube launch tile `[512,32,64]` |
| `flashmtp_history_candidate_1` | `candidate_1` | Five submissions; same cube launch tile |
| `flashmtp_history_candidate_2_control` | `candidate_2` | One mixed step; launch tile `[128,32,256]` |

The fixture checks each history partition and every launch's kind, parts,
cores, and tile. Tile tuple conventions differ between cube physical geometry
and schedule launch fields: preserve their field names rather than comparing
unlabelled tuples. An absent or changed candidate is a failed reproduction,
not permission to substitute the current argmin.

Freeze graph, problem, solution, full typed schedule, named ABI, physical
strides/layout, transpose, modeled costs, candidate ID, and emitted source in
two fresh processes with distinct hash seeds. Add a dump-on compilation of
the same source and compare final PTO and orchestration, normalizing only
documented embedded paths. Keep structural comparisons separate from timing.

All four subjects retain **dense ND weights physically `[N,K]` with a
transposed matmul consumer**. The latest native Flash-MTP uses NZ weights;
it is not an interchangeable control for this historical dense regression.
Do not relabel dense tensors as NZ, silently pack/unpack weights, or change
the graph to match the newer native implementation.

## 2. Execute on both devices with five seeds

Run the checked-in file once per allocated device, using its explicit device
environment. From the Fusebox root, after supplying `DEVICE_ID` and `CAMPAIGN`:

```bash
PTO_FUSEBOX_RUN_DEVICE_TESTS=1 \
PTO_FUSEBOX_PLATFORM=a2a3 \
PTO_FUSEBOX_DEVICE_ID="$DEVICE_ID" \
PTO_FUSEBOX_DEVICE_REPEATS=50 \
python -m pytest test/device/test_selected_wrong_answers.py -v --tb=long \
    --basetemp="$CAMPAIGN/device-$DEVICE_ID/pytest"
```

Do not use fail-fast across independent cases. First execution must complete
the lazy incore build and actual launch: `ir.compile`, `.pto`, or generated
`.cpp` alone is not execution evidence. Retain kernel build/load diagnostics.

Use the committed seeds **11, 23, 37, 53, 71**, nonuniform inputs, named
`bind_emitted_call` bindings, and the declared input/output dtypes and storage.
Validate bindings before launch; never infer output direction from a matching
shape/annotation or bind FP32 storage to a BF16 parameter. Reuse identical
seed inputs across the history alternatives and both devices.

Required checks:

- INT8 cube: exact equality to the FP64 product of integer inputs, with an
  INT32 output and no `-2^31` unwritten sentinels. The bound on these inputs
  cannot overflow INT32. Do not accept BF16/FP32-style relative tolerance.
- History: the unchanged fixture/reference algebra, with
  `max(abs(actual-reference)) / max(abs(reference)) <= 2e-2`. Assert a finite,
  nonzero reference; retain max absolute error and mismatch maps as well.
- Full output coverage and zero nonfinite values on every successful launch.
  Sentinel checks supplement reference comparisons; finite output is not
  correctness. Do not compare unspecified allocation padding.
- One signature per case/seed over 50 launches, first launch included.
  Compare the saved per-seed hashes across devices explicitly. A fully passing
  matrix contains 4 cases x 5 seeds x 50 launches x 2 devices = 2,000 launches.

A failed assertion stops that pytest case early. Record actual coverage; do
not claim 2,000 launches merely because 50 was configured. A campaign-local
diagnostic runner may collect the remaining seeds and failures using the same
fixtures, bindings, and thresholds. It must preserve the original failed test
result and record each seed independently.

## 3. If a wrong answer persists, localize it without fitting

Continue unaffected cases. For each failing case, record the first divergent
intermediate and a small single-variable discriminator before attributing the
bug to Fusebox, PyPTO, PTOAS, or runtime. The shared `[32,512,64]` physical tile
is a clue, not a proven cause.

For history, distinguish quantized activation, INT32 matmul accumulator, and
dequantized result. Compare each stage against its own math applied to the
device's preceding intermediate. Mechanically compare instrumented source
against the frozen source so observation does not silently change the plan.
Use an independently written minimal PyPTO control where necessary, outside
the product trees. Control arithmetic must preserve rounding/cast semantics.

If useful for attribution, run the minimal control on the pinned plain
upstream base as a separate arm. Do not require explicit-pipe Fusebox programs
to compile on a base missing that DSL surface. A passing refreshed lane only
proves closure on that lane; naming the fixing upstream commit requires a
separate controlled comparison or bisection.

No failed candidate may enter ranking or calibration. This focused task does
not request a new performance matrix or native/generated ratios; those follow
after correctness and the shared layout-aware schedule work.

## 4. Native adapter compatibility is a separate host verdict

Verify all four current manifests and actual native symbols at the pinned
PyPTO-lib checkout. DSpark's projection is in `dspark_drafter.py`; its
whole-drafter fixtures are not projection-only controls. Flash-MTP exports a
shared JIT body and declares NZ projection weights; Pro remains ND.

The generated Flash-MTP integration must reject that NZ ABI before emitting a
bundle. The passing refusal test is **not generated NZ support or native
overlay closure**. Report Pro's expected native-lowering limitation separately.
None of these adapter results substitutes for the dense device regressions.

## 5. Report and archive

Provide separate verdicts for:

1. Refreshed-lane provenance and host gates.
2. Frozen candidate identity/layout contract.
3. Exact INT8 cube correctness, two devices.
4. History candidate_0 correctness, two devices.
5. History candidate_1 correctness, two devices.
6. History candidate_2 positive control, two devices.
7. Cross-device and first-launch stability.
8. Native adapter compatibility and explicit NZ refusal (host only).

State failures, blocked stages, and missing seed/device coverage explicitly.
Archive the brief, regeneration commands, complete logs, frozen artifacts,
input/reference/output hashes, error maps, per-seed results, final PTO/C++, and
all diagnostic harnesses. Record binaries' hashes but exclude build trees,
checkouts, environments, wheels, and large binaries. Generate a payload manifest
before packaging; verify hashes and the file set in both directions from a
fresh extraction, then publish the archive SHA-256.

Finish with tracked-clean product trees, released device allocations, and no
campaign processes. Report the archive's exact location; do not claim it was
copied to Downloads unless that copy was actually made and verified.
