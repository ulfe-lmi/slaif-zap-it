# OAP Coding-Agent Report — 000-a

## Work order
- Identifier/order/objective/PR mode: `000-a` / `oap/orders/000-a-professional-baseline-and-ci.md` / Objective 000 professional baseline and CI / `CREATE_NEW_PR`.

## Status
COMPLETE

## Executive summary

Brought the repository to an honest professional baseline without touching GPU
runtime or the service roadmap: added setuptools packaging with a CPU-only dev
extra; repaired both previously-red CPU tests against their documented root
causes; added real-PyYAML grammar coverage, package/import smoke tests and a
suite-level offline/CUDA-free guard; applied Ruff format/lint with minimal
behavior-preserving fixes; measured coverage and set a defensible ratchet;
added least-privilege GitHub Actions CI plus CodeQL; refreshed documentation,
contributing guidance and third-party provenance; executed the legacy-artifact
audit decisions. All required local verification layers PASSED and all five PR
checks are green on the implementation head.

## Authoritative GitHub state
- Repository: `ulfe-lmi/slaif-zap-it`
- PR: https://github.com/ulfe-lmi/slaif-zap-it/pull/44 (OPEN, not merged by coding)
- Base: `main` @ `12257a6c31ad654d1b7114e50cf9679fcc2fb260`
- Head branch: `oap/000-a-professional-baseline-and-ci`
- Start SHA: `12257a6c31ad654d1b7114e50cf9679fcc2fb260`
- Implementation head SHA: `ac1e1921670f229adf4d6958bb59a6fc2b32440d`
- Report publication commit: SELF
- New PR: yes (#44, exactly one); amended existing: no; coding merge: NO.

## Changes/files

Added: `pyproject.toml`, `.github/workflows/ci.yml`,
`.github/workflows/codeql.yml`, `CONTRIBUTING.md`, `THIRD_PARTY_NOTICES.md`,
`docs/BASELINE.md`, `tests/test_real_yaml_config.py`,
`tests/test_package_imports.py`, `tests/test_environment_guard.py`,
`oap/active`, `oap/orders/000-a-professional-baseline-and-ci.md`.
Renamed: `ALGORITMS.md` -> `ALGORITHMS-DETAILED.md` (`git mv`, content
verbatim). Untracked from index: `everything.txt`, six `last_results/*.jpg`
(local copies preserved on disk). Modified: `.gitignore`, `README.md`,
`INSTALL.md`, `huggingface_downloader.py`, `modules/classifier/clip.py`,
`modules/visualizer.py`, all product sources reformatted by Ruff where needed,
`tests/conftest.py`, `tests/test_src_exports.py`,
`tests/test_huggingface_downloader.py`,
`tests/test_batch_entrypoints.py`, plus Ruff reformat of remaining
product/test files. Diff: 45 files changed, ~611 insertions / ~8487 deletions
(deletions dominated by untracked binaries and generated concatenation).

## Acceptance evidence

1. Clean CPU/dev install on supported Python — PASSED (see verification).
2. One canonical CPU command runs the complete suite — PASSED: 95 passed,
   0 failed, 0 skipped, ~3.5 s wall, no network/GPU/model dependency.
3. Format/lint/wheel/import smoke — PASSED (details below).
4. Coverage measured with explained exclusions and enforced gate — PASSED:
   66.65% total branch coverage; `fail_under = 64` enforced via pyproject.
5. CI workflows on PR+main least-privilege; CodeQL enabled and green — PASSED.
6. Existing CLI/config behavior covered/documented; compatibility changes none
   beyond documented test repairs — PASSED.
7. Docs consistent; no API/GPU readiness claims — PASSED.
8. No large generated/model/cache/private artifact or secret in the diff — PASSED.
9. No host/GPU/service mutation; before/after snapshots identical — PASSED.
10. Correct branch, exactly one PR from base SHA, checks green, no merge — PASSED.

## Verification
- `python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'`: PASSED — clean
  install of package + dev tooling only; heavy stack absent by design.
- `.venv/bin/python -m build --wheel`: PASSED — `zap_it-0.1.0-py3-none-any.whl`
  contains `src`, all eight `modules.*` packages, LICENSE metadata.
- Wheel import smoke: PASSED — extracted wheel to `/tmp/opencode/wheelcheck`,
  imported `src`, `src.config`, `src.postprocessing`, all module packages from
  the wheel path (asserted `src.__file__` resolves inside extracted site dir),
  using only lightweight CPU stubs for GPU libraries.
- CLI entry-point smoke: PASSED within the canonical suite —
  `tests/test_zap_it_batch_cli.py` exercises the real argparse/dispatch path of
  `zap-it-batch.py` under the documented stub harness. Direct
  `.venv/bin/python zap-it-batch.py --help` requires torch/cv2 (pre-existing,
  unchanged behavior; full runtime is the conda env per INSTALL.md) — recorded
  as NOT RUN outside conda intentionally, not as pass.
- `.venv/bin/ruff format --check .`: PASSED (70→92 files formatted during
  development; final state clean).
- `.venv/bin/ruff check .`: PASSED (11 initial findings fixed: unused
  imports/locals removed behaviorally-inertly).
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — **95 passed, 0 failed, 0 skipped**, 1 pre-existing intentional
  DeprecationWarning from the `zap_it_postseg_processing` shim import, 3.46 s;
  TOTAL branch coverage 66.65%, gate 64% reached.
- No-network/no-CUDA assertion: PASSED — session-scoped socket block in
  `tests/conftest.py` (opt-out `ZAP_IT_TESTS_ALLOW_SOCKETS=1`) plus
  `tests/test_environment_guard.py` proving sockets raise and CUDA stays
  unavailable/stubbed.
- Docs/config schema examples validated against real YAML loader: PASSED —
  `tests/test_real_yaml_config.py` parses every shipped `configs/*.yaml` with
  real `yaml.safe_load` through `load_config` plus behavior cases.
- Secret/large-artifact scan: PASSED — regex scan of full staged diff
  (`hf_…`, `ghp_…`, `github_pat_…`, AKIA, PEM blocks, credential-assignment
  patterns): 0 hits; largest added file 14 KB (order transcript); added-file
  sizes reviewed manually.
- Baseline reproduction before repair: PASSED — isolated venv reproduced the
  order-documented red state exactly (`1 failed, 63 passed` + collection error
  on `test_huggingface_downloader.py`; 0.32 s).

## CI/checks
All checks on PR head `ac1e1921670f229adf4d6958bb59a6fc2b32440d`:
- `static (format, lint, build)` — PASS (~20 s)
- `tests (py3.10)` — PASS (~19 s)
- `tests (py3.11)` — PASS (~15 s)
- `tests (py3.12)` — PASS (~18 s)
- `Analyze (python)` (CodeQL workflow job) — PASS (~58 s)
- `CodeQL` — PASS
None pending, failed or missing at report time.

## GPU/service/resource evidence
- Before snapshot: physical GPU0 UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182` 2161 MiB used by unrelated PID
  66522 (`/opt/venv/bin/python`); physical GPU1 UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8` 6 MiB, idle.
- After snapshot: byte-identical readings (same UUIDs, same memory values, same
  single compute app). Zero allocations attributable to this objective; no CUDA
  context was ever created (suite guard proves it locally as well).
- Visible logical device mapping: not applicable — no GPU code executed; no
  `CUDA_VISIBLE_DEVICES` manipulation performed.
- Host: no listeners opened, no ports bound, no services touched, no `/dev/shm`
  usage; scratch limited to `/tmp/opencode` venvs/wheel extraction.

## Documentation/provenance
- `docs/BASELINE.md`: stub-vs-real inventory per test layer; harness reality;
  repaired red items; packaging/coverage baseline.
- README: dev/testing section + honest "no HTTP service yet" statement;
  INSTALL: new section 0 CPU-only dev environment while conda GPU flow kept
  as-is for later objectives.
- `CONTRIBUTING.md`: canonical commands, hostile-input/security pointer,
  no-secrets/no-weights rules, PR discipline.
- `THIRD_PARTY_NOTICES.md`: MIT preservation; license pointers for CPU/dev and
  GPU/runtime stacks; model provenance table flagging that BLIP-3
  (`Salesforce/xgen-mm-phi3-mini-instruct-r-v1`) has carried non-commercial
  research terms historically and must be license-re-verified before any
  production/commercial deployment; model revisions noted unpinned pending
  later objective.
- SECURITY.md reviewed: consistent with current reality; left byte-unchanged to
  avoid mutating orchestration-truth content.

## Deferred human adjudication
- Critical register action: NONE
- No CRITICAL.md append was ordered by `000-a`; none performed. No candidate
  meeting the five-condition threshold was discovered; the BLIP-3 license note
  above is ordinary provenance documentation routed through normal OAP review,
  not a register entry.

## Safety/scope confirmations
- No API/service/FastAPI/Docker work; no in-memory core refactor beyond test
  seams; no algorithm/threshold/default changes; the only runtime-code edits
  are lint-motivated removals of provably-unused bindings (`torch = self._torch`
  in `clip.py:filter_masks`, unused `import cv2` in `visualizer.py`, unused
  locals in downloader/tests) — each behavior-preserving and covered by green
  tests.
- No GPU allocation, no driver/system/firewall/port/global-config mutation; no
  sudo use; protected GPU0 workload untouched (snapshot evidence).
- Physical GPU1 untouched as ordered for this objective.
- `oap/bin/*` governance tooling restored byte-identical to HEAD after an
  accidental formatter touch was detected and reverted; verified via
  `git diff HEAD --exit-code -- oap/bin/`.
- No secrets, raw images, credentials, model weights or private configuration
  entered logs, OAP artifacts or the diff.

## Limitations/blockers
- The CPU suite still runs most pipeline logic under the documented stub
  harness; true-library behavior (PIL/torch/detectron2/transformers semantics,
  inference quality) remains uncovered until de-stubbing objectives. Inventory
  is explicit in docs/BASELINE.md; de-stubbing beyond PyYAML was explicitly out
  of scope.
- Direct CLI invocation requires the operator conda GPU environment; console-
  script entry points were deliberately not fabricated for scripts lacking a
  `main()` refactor (would exceed minimal seams).
- Model revisions remain unpinned (planned later objective); THIRD_PARTY_NOTICES
  records the review requirement instead of pretending completion.
- Coverage gate is an honest ratchet (66.65% measured vs 64% floor), not a
  high-water claim; weak spots (`modules/verifier/blip3.py` 20%,
  `modules/classifier/clip.py` 32%, `modules/output/yolo.py` 47%) are visible
  in the coverage report for future ratchets.

## Factual strategic follow-up
- Branch protection on `main` reportedly absent (order evidence); CI now exists
  but nothing enforces it on main pushes — strategic may want protection rules
  before later objectives rely on them.
- BLIP-3 checkpoint license needs human/operator confirmation before any
  commercial/redistribution decision (flagged in THIRD_PARTY_NOTICES.md).
- Coverage ratchet points available for objectives that add engine tests.
