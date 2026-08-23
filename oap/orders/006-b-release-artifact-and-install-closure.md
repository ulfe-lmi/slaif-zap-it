# OAP Work Order — 006-b — Close release artifact, install and verification gaps

## Objective

Amend existing Objective-006 PR #50 and close strategic-review-proven gaps in
the otherwise accepted `006-a` release-candidate work. Make the sdist actually
self-contained for the documented install path, prove built-wheel installation
on every supported Python, make the tracked-tree secret baseline enforceable,
bound lightweight dependencies, remove checkout-only/broken operator examples,
and make the local academic A/B/A harness use both goat images as intended.

Preserve the accepted `006-a` rights mitigation, exact `CRIT-0001` append,
packaging design, gateway/container decisions, live GPU evidence and every
external/release prohibition. This is a narrow continuation on the same PR, not
a new release or architecture objective.

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`; remote base `main` remains
  `1758c3989454a000c71c2fc986db505bb70f3a5b`.
- Numeric objective / round: `006 / 006-b`; mode `AMEND_EXISTING_PR`.
- Sole branch: `oap/006-a-release-and-integration`.
- Sole PR: #50, `OPEN`, non-draft, mergeable/clean, title unchanged:
  `Objective 006-a: package and qualify the 0.1.0 release candidate`.
- Current local/remote/PR head:
  `1230c92377bc4c7f16d7817025626584d63ed638`, immutable report-only
  `006-a` commit. Its first parent is implementation
  `3fe090d08e122b33d0d68d1eb4569a38fdf9cb09`; it changes only
  `oap/reports/006-a-report.md`.
- All report-head checks are SUCCESS: static, release artifact audit, Python
  3.10/3.11/3.12, Analyze and CodeQL. No check is pending/missing.
- Required new report: `oap/reports/006-b-report.md`. Every prior order/report
  and every prior `CRITICAL.md` byte is immutable.
- `CRITICAL.md` must remain byte-identical at SHA-256
  `d639ebf52f5bfb6b49cc05838f63b359268ffde0829c1aca73f639e5c2c961c7`;
  do not append a mitigation update or human disposition in this round.

## Strategic review findings and exact corrections

### 1. Self-contained source distribution

The accepted manifest excludes media/OAP correctly, but the built sdist omits
`INSTALL.md`, `requirements-gpu-cu124.lock`, and `.secrets.baseline` even though
its docs and audit tools depend on them. Thus the source artifact cannot perform
the documented qualified-GPU installation or reproduce its own secret audit.

- Add those three files to the explicit sdist allowlist.
- Keep all four goat paths, all media/demo payloads, model weights/caches,
  generated outputs, private env files and OAP transcript files excluded.
- Make `verify_release_artifacts.py` require in the sdist: `INSTALL.md`, the GPU
  lock, `.secrets.baseline`, LICENSE/notices/README/changelog/release notes,
  service env example, uninstalled unit, artifact verifier/scanner, installed
  service modules and supported config examples.
- Require the wheel's package modules, console entrypoint, license and public
  notices as before; do not put the GPU lock or operator templates into wheel
  package imports merely to satisfy a test.
- Build a wheel directly and build a second wheel from the sdist. Require equal
  wheel member manifests/content and an installed smoke for the sdist-built
  wheel. Record final-tree hashes only.

### 2. Built-artifact matrix, not editable-install inference

The Python 3.10/3.11/3.12 jobs currently install the checkout editable. Only a
Python-3.12 local smoke installed a built wheel. Green editable tests do not prove
the released wheel imports or exposes its entrypoint on every declared Python.

- Preserve the full editable CPU suite/coverage matrix.
- In **each** Python 3.10, 3.11 and 3.12 CI job, also build or consume the wheel,
  create a fresh isolated venv outside the checkout, install the wheel with
  service dependencies, `cd` outside the checkout, and prove:
  - `import src` and `src.__version__ == "0.1.0"`;
  - `src.runtime.live_service` resolves under that venv's site-packages;
  - the `zap-it-service` console script exists;
  - a fake-engine app can be created and one generated multipart response has
    `service.package_version == "0.1.0"` for JSON and the ZIP manifest.
- No CUDA, model download, listener or real fixture is allowed in public CI.
- The release audit job must build wheel+sdist, verify both, build/verify the
  sdist-derived wheel and run its no-checkout installed smoke.

### 3. Secret baseline must fail closed

The current CI commands run `detect-secrets` and parse its JSON but never compare
new tracked-tree findings with the baseline; `detect-secrets scan` normally exits
zero even when findings exist. Artifact extraction uses an enforcing comparator,
but the current tracked tree does not.

- Extend one repository helper (prefer the existing scanner) with an explicit
  tracked-tree mode that compares `(path, detector type, hashed secret)` to the
  committed baseline and exits nonzero for any addition, removal/path mismatch
  that requires review, malformed baseline, or scanner failure.
- CI must call this enforcing mode. Remove the cosmetic JSON-only commands.
- Add CPU tests with a generated tree/baseline proving one explained finding
  passes and one unexpected synthetic secret fails; do not add a real credential.
- Document every current baseline exception exactly:
  - three pinned model-revision hex strings in `src/runtime/models.py`;
  - the environment-variable name in `src/service/settings.py`;
  - the synthetic test key in `tests/test_service_units.py`.
- Continue enforcing unpacked wheel/sdist scans. Reports list counts/path classes,
  never candidate secret text.

### 4. Dependency and project metadata bounds

The build backend and dev tools are pinned, but the three base dependencies and
`python-multipart` remain open-ended contrary to the finalized order.

Use these explicit compatibility ranges and test their resolution in CI:

```toml
numpy>=1.26.4,<3
pillow>=10.4,<13
pyyaml>=6.0.2,<7
python-multipart>=0.0.9,<1
```

Keep the qualified GPU lock exact and authoritative for the physical target.
Add normal project URLs for repository, documentation and issue reporting. Do
not change version `0.1.0`, API model `zap-it-1`, schema `zap-it.v1`, scientific
dependencies, model revisions or supported profiles.

### 5. Installed-service and documentation accuracy

- The Type=simple installed unit must not require the source checkout as its
  `WorkingDirectory` or file documentation root. Remove `WorkingDirectory` or
  use a neutral existing operator home; point `Documentation=` to a stable public
  repository document. Preserve private `EnvironmentFile`, loopback policy,
  explicit installed-venv placeholder and uninstalled status.
- `systemd-analyze verify` must pass without creating the placeholder executable
  and without installing/enabling/reloading the unit.
- Replace every `configs/example.yaml` and tracked `configs/goats.yaml` command
  with an existing redistributable tracked example such as
  `configs/tomato.yaml`. Keep the ignored academic harness in its clearly local-
  only section.
- Remove the invalid installed `zap-it-service --help` smoke unless real safe CLI
  help is implemented and tested. Prefer import/metadata/entrypoint existence
  checks that do not require GPU env or open a listener.
- Add link/path tests so public docs cannot reference missing tracked release
  inputs accidentally.

### 6. Artifact-deny completeness and version tests

- Treat a top-level or unexpected `output/` member as forbidden while allowing
  the legitimate Python package `modules/output/*.py`.
- Reject `.env`, secret/private env basenames and equivalent private config
  members while continuing to allow only the exact public
  `deploy/service.env.example` template.
- Add generated archive tests for traversal, absolute paths, symlink/hardlink,
  oversize members, goat basenames, media/weights, unexpected output directories,
  private env files, and missing required release members. Include positive tests
  for the legitimate `modules/output` package and exact env example.
- Add direct tests for source/editable version exposure and JSON/ZIP
  `service.package_version`; installed matrix evidence covers wheels.

### 7. Two-image local academic A/B/A fidelity

The 006-a harness can accept an arbitrary `--image`, but its default and live run
used goat image 2 as A, a generated image as B, then goat image 2 again. The
human supplied **both** goat images for this academic regression, and Objective
005's accepted A/B/A used image 1, image 2, image 1 with `goats2.yaml`.

- Change the opt-in interface to accept explicit/default `image_a` and `image_b`
  paths: goat image 1 and goat image 2. Use `goats2.yaml` as the one academic
  config unless an explicit local config path is supplied.
- Independently path-guard, decode and central-50%-crop both images in memory.
  Run A/B/A as crop 1, crop 2, crop 1; each gets L2 JSON, L3 JSON and L3 ZIP.
- Emit only aliases, dimensions, digests, status/latency/size/count evidence and
  zero-persistence facts. Never emit semantic labels/prompts/raw bodies.
- Add generated CPU tests proving two distinct images and exact A/B/A ordering.
- Run one focused real local two-image A/B/A on the installed candidate after
  corrections. Reuse the accepted 006-a auth/synthetic/restart/resource evidence;
  do not repeat the full live matrix. Final service must again be stopped, GPU1
  idle, port free, shared-memory root empty and GPU0 unchanged.

## Expected scope and non-goals

Expected changes: manifest/packaging metadata, CI release/install steps, secret
scanner and tests, artifact verifier/tests, unit template/docs/link fixes,
version tests, two-image local harness, exact order/active transcript. Preserve
the four tracked deletions, ignored host files, `CRIT-0001`, gateway contract,
license inventory and accepted live implementation.

Non-goals:

- no new PR/branch/title, no merge/tag/release/package publication;
- no CRITICAL append/edit/disposition and no history/visibility/settings change;
- no Docker, gateway implementation, LAN/public bind, systemd installation,
  model download or production data;
- no API/model/schema/scientific behavior change;
- no raw goat bytes/crops/config/responses committed or reported;
- no repeat of the full 006-a GPU rehearsal beyond the focused two-image harness
  and final cleanup proof.

## Acceptance and verification

1. PR #50 remains unique, same base/branch/title and contains one new bounded
   implementation commit before the `006-b` report-only commit.
2. `CRITICAL.md` and prior OAP artifacts remain byte-identical; the four goat
   paths remain absent from Git and present ignored locally.
3. Final wheel/sdist and sdist-built wheel contain every required support file,
   no denied content, pass metadata/member/secret verification, and report hashes
   from the final implementation tree.
4. Python 3.10/3.11/3.12 CI proves both full CPU behavior and no-checkout built-
   wheel install/import/entrypoint/fake JSON+ZIP provenance.
5. Tracked-tree detect-secrets comparison is fail-closed and regression-tested;
   the five existing false positives are documented and no new one is hidden.
6. Dependency bounds/project URLs and installed unit/docs are coherent; no
   tracked documentation command references a nonexistent config or checkout-
   dependent installed service.
7. Artifact negative tests and version JSON/ZIP tests cover the review findings.
8. Real local goat image1/image2/image1 central-crop harness passes nine cases
   with zero persistence and sanitized evidence; final GPU/host cleanup passes.
9. Full canonical CPU suite/coverage, Ruff format/lint, compile, shell syntax,
   unit validation, docs/path checks, artifact builds/installs/scans and every
   current GitHub CI/CodeQL check are SUCCESS on implementation and report heads.
10. Coding publishes immutable `oap/reports/006-b-report.md` with exact command
   states, artifact/matrix/secret/harness evidence, SELF parent topology and
   release-gate table; coding never merges.

## Deferred human adjudication

- Decision: NONE

`CRIT-0001` already records the one underlying public-history dilemma. These are
ordinary release-engineering correctness gaps and must not create a duplicate
entry or mitigation update.

## Publication and response

Commit/push all non-report changes first and capture the literal implementation
SHA. Then create exactly `oap/reports/006-b-report.md`; its final SELF commit
changes only that path and has the implementation SHA as first parent. Verify
remote bytes/topology and all checks, send exact FIFO `OK`, and exit. Coding never
merges or begins another objective.
