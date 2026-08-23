# OAP Coding-Agent Report — 006-b

## Work order

- Identifier/order/objective/PR mode: `006-b`, close release artifact/install
  gaps, amend existing PR.
- Repository: `ulfe-lmi/slaif-zap-it`
- Branch: `oap/006-a-release-and-integration`
- PR: [#50](https://github.com/ulfe-lmi/slaif-zap-it/pull/50), existing PR only

## Status

COMPLETE

## Executive summary

Closed the ordered release-engineering gaps without changing API/model/schema or
scientific behavior. The sdist now carries the documented install/audit inputs;
direct and sdist-built wheels have equal member content; all three supported
Python CI jobs perform isolated built-wheel service smokes; tracked-tree secret
findings are compared fail-closed; dependencies and project URLs are bounded;
the installed unit and public examples are checkout-independent; archive
denylist/version tests were extended; and the local academic harness now runs
goat crop A/B/A using both supplied images.

## Authoritative GitHub state

- PR state: `OPEN`, non-draft, `MERGEABLE`, `CLEAN`; title unchanged:
  `Objective 006-a: package and qualify the 0.1.0 release candidate`.
- Base: `main` at `1758c3989454a000c71c2fc986db505bb70f3a5b`.
- Starting remote/report head: `1230c92377bc4c7f16d7817025626584d63ed638`.
- Implementation head SHA: `d9e8aeb52783ea44763afd3a4d6860bee97e503b`.
- Report publication commit: SELF
- New PR: NO; amended existing PR #50: YES; coding merge: NO.

## Changes/files

- Packaging/metadata: `MANIFEST.in`, `pyproject.toml`.
- CI/install verification: `.github/workflows/ci.yml`,
  `scripts/smoke_installed_package.py`.
- Secret enforcement: `scripts/scan_release_artifacts.py`, `SECURITY.md`.
- Artifact policy: `scripts/verify_release_artifacts.py` and generated-data
  tests in `tests/test_release_candidate.py`.
- Operator/docs accuracy: `deploy/zap-it-local.service`, `README.md`,
  `INSTALL.md`, `docs/RUNBOOK.md`.
- Academic A/B/A: `scripts/smoke_local_goats.py` and generated CPU tests.
- Provenance assertions: `tests/test_service_api.py`.
- Exact round transcript: `oap/active` and
  `oap/orders/006-b-release-artifact-and-install-closure.md`.

## Acceptance evidence

1. **PR/topology — PASSED.** PR #50 remains the sole open PR on the original
   branch/base/title. One bounded implementation commit is remote before this
   report-only child.
2. **Protected OAP/critical state — PASSED.** `CRITICAL.md` remained unchanged
   at SHA-256
   `d639ebf52f5bfb6b49cc05838f63b359268ffde0829c1aca73f639e5c2c961c7`.
   Prior reports were not modified. The four goat paths are absent from Git
   and present as ignored local operator files.
3. **Release artifacts — PASSED.** Final-tree evidence:
   - direct wheel: 63 members, SHA-256
     `343d9e9ae4c8fbf4784417796374c99d3a35bb011c08042bac384c37c7bf15b4`;
   - sdist: 141 members, SHA-256
     `b90e1a9e7fafe44c7ba476f20d467574a0b17fc83a57cc43914631e65b7e644d`;
   - sdist-built wheel: 63 members, SHA-256
     `eb1bf81441e2bcd326d261cf3f60df91a51409a6b7d5b57879c534fc560c7374`;
   - direct/derived wheel member-manifest SHA-256: both
     `7643cecbca0a23b48644fdb97a08536a709cc6c8a586ce880c1ba98191b9a48c`.
   Required support files, package modules, public notices, entrypoint and
   denylist checks passed; no goat/media/weight/output/private-env/OAP payload
   entered an artifact.
4. **Installed matrix — PASSED.** Final direct and sdist-derived wheels were
   installed in fresh isolated environments outside the checkout. Each proved
   `src.__version__ == "0.1.0"`, site-packages resolution of
   `src.runtime.live_service`, the `zap-it-service` executable, and fake-engine
   JSON/ZIP `service.package_version == "0.1.0"`.
5. **Secret baseline — PASSED.** Enforced tracked-tree comparison found exactly
   the five existing reviewed findings. Unpacked direct wheel, sdist and
   sdist-derived wheel scans found zero unexpected findings. The five exact
   exceptions are documented in `SECURITY.md`; generated tests cover an
   explained finding, an unexpected finding, malformed baseline and scanner
   failure.
6. **Metadata/unit/docs — PASSED.** Base dependency ranges and
   `python-multipart` are bounded as ordered, repository/documentation/issues
   URLs are present, public commands use `configs/tomato.yaml`, the installed
   unit has no checkout `WorkingDirectory` or file URI, and `systemd-analyze
   verify deploy/zap-it-local.service` passes without installing the unit.
7. **Archive/version tests — PASSED.** Generated CPU tests cover traversal,
   absolute paths, symlink/hardlink, oversize, goat/media/weight payloads,
   output directories with the legitimate `modules/output/*.py` exception,
   private env/config names with the exact public env template, missing sdist
   members, source version, and JSON/ZIP provenance.
8. **Focused local academic run — PASSED.** The installed candidate ran nine
   successful cases: aliases `a1`, `b`, `a2`, each at L2 JSON, L3 JSON and L3
   ZIP. Both inputs were independently decoded at `[5568, 4176]` and centrally
   cropped to `[2784, 2088]`; A/B image and crop digests were distinct; all
   cases returned HTTP 200; zero-persistence was true and the shared-memory
   workspace had zero files/bytes afterward.
9. **Canonical verification — PASSED.** Local CPU, formatting, lint,
   compilation, shell syntax, artifact build/install/scan, Twine and unit
   validation passed. All current GitHub checks passed on the implementation
   head.
10. **Immutable report — PASSED.** This file is the only path in the final SELF
    child commit, with the implementation head as its first parent; remote
    parent/bytes are verified before signaling.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 359 passed, 1 honest GPU-marked skip, 77.40% total coverage.
- `.venv/bin/pytest -q tests/test_release_candidate.py`: PASSED — 20 passed.
- `.venv/bin/pytest -q tests/test_service_api.py`: PASSED — 49 passed.
- `.venv/bin/ruff format --check .`: PASSED.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q scripts src tests`: PASSED.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist-from-sdist/*.whl`:
  PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py --compare-wheels
  dist/zap_it-0.1.0-py3-none-any.whl dist-from-sdist/zap_it-0.1.0-py3-none-any.whl`:
  PASSED — equal member names, sizes and content hashes.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz
  dist-from-sdist/*.whl --baseline .secrets.baseline`:
  PASSED — three archives, zero unexpected findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline
  .secrets.baseline`: PASSED — five baseline findings.
- `.venv/bin/twine check dist/* dist-from-sdist/*`: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- `git diff --check`: PASSED.

## CI/checks

All checks below are SUCCESS/PASSED on implementation SHA
`d9e8aeb52783ea44763afd3a4d6860bee97e503b`:

- `static (format, lint, build)`: PASSED.
- `release (artifact audit)`: PASSED.
- `tests (py3.10)`: PASSED.
- `tests (py3.11)`: PASSED.
- `tests (py3.12)`: PASSED.
- `Analyze (python)`: PASSED.
- `CodeQL`: PASSED.

The matrix preserves editable CPU coverage and additionally installs/build-smokes
the wheel outside the checkout in every supported Python job. The release job
builds direct and sdist-derived wheels, compares them, scans them, validates
the unit, and runs the no-checkout installed smoke.

## GPU/service/resource evidence

- Before/after the focused run, physical GPU1 was NVIDIA GeForce RTX 2080 Ti,
  11264 MiB, PCI `00000000:00:0C.0`, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`.
- Launch used `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=1`; the application reported one visible device as
  logical `cuda:0` with the expected UUID.
- Physical GPU0 remained protected: its UUID and unrelated compute process /
  2492 MiB observation were unchanged across the run. No GPU0 allocation,
  reset, kill or reconfiguration was performed.
- Focused service: loopback `127.0.0.1:23654`, one installed candidate process,
  stopped after the nine-case run. The port is free afterward.
- `/dev/shm/slaif-zap-it` is empty afterward. No request image, YAML, crop,
  response or bearer key was persisted or reported. Peak GPU memory was not
  re-profiled in this focused round; accepted 006-a resource evidence is
  reused as ordered.

## Documentation/provenance

The sdist allowlist now includes `INSTALL.md`, the qualified GPU lock and
`.secrets.baseline`, while the wheel continues to contain package modules,
entrypoint, license and public notices without operator templates. Project
URLs, dependency bounds, the installed-unit public documentation link, and the
local-only two-image harness contract are documented in the same change.

## Deferred human adjudication

- Critical register action: NONE
- No CRITICAL append, edit or disposition was performed.

## Safety/scope confirmations

- No merge, tag, release, package publication, new PR, branch creation,
  history/visibility/settings change, Docker/gateway/LAN change, systemd
  installation, model download or production-data operation.
- No raw goat bytes/crops/config/responses, credentials, private environment
  values or model weights entered Git/OAP artifacts.
- Existing accepted rights mitigation, `CRIT-0001`, model pins, API/schema and
  scientific behavior were preserved.

## Limitations/blockers

`CRIT-0001` and the other pre-existing release/legal gates remain open as
documented; this round does not authorize final public release or merge. The
single GPU-marked CPU-suite test remains an honest skip outside the focused
live run; no public CI GPU runner was introduced.

## Factual strategic follow-up

PR #50 is ready for strategic review at the verified implementation head. The
coding agent does not merge, accept, close, release, or choose a next order.
