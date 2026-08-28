# OAP Coding-Agent Report — 015-b

## Work order

- Identifier/order/objective/PR mode: `015-b` — close SAM2 contract test and documentation gaps; amend the existing Objective-015 PR.
- Repository: `ulfe-lmi/slaif-zap-it`
- PR: [#71](https://github.com/ulfe-lmi/slaif-zap-it/pull/71)
- Base: `main` at `1c6e42c28e3a4c29fff4c16be8311176ba07621a`
- Starting head SHA: `01760808c0f4a5549a313fb2b422e447da7ce674`

## Status

COMPLETE

## Executive summary

Amended PR #71 with decisive CPU/API contract evidence for request-local SAM2
configuration, corrected the unsafe `dtype` classification, and corrected the
14-scalar documentation wording. The expanded tests demonstrate exact
constructor forwarding, counted one-model A/B/A isolation, strict validation
and admission boundaries, pre-engine structured errors, deterministic
capabilities, all-verbosity manifests, JSON/ZIP parity, and raw-versus-remapped
candidate counts.

The expanded 015-b tests supply evidence that the immutable 015-a report claimed
too broadly. The immutable 015-a report was not edited or deleted.

## Authoritative GitHub state

- PR #71: OPEN, non-draft, MERGEABLE, based on `main`.
- Branch: `oap/015-a-request-local-sam2-configuration`.
- Implementation head SHA: `8eb9e4f16070795ae54d4fae9a7807cb6ad67660`
- Report publication commit: SELF
- New PR: no; amended existing PR #71: yes; coding merge: NO.

## Changes/files

Implementation commit `8eb9e4f16070795ae54d4fae9a7807cb6ad67660` contains:

- Expanded `tests/test_sam2_configuration.py` from the prior 16-test evidence
  to 180 focused tests and added the controlled core/API empty-mask fixture.
- Added `dtype` to the API hostile-key denylist so it is classified as
  `unsafe_config`, matching the other operator-owned runtime controls.
- Corrected 14-total-scalar wording in `TESTING.md`, `docs/API.md`,
  `docs/CONFIG.md`, and `docs/SERVICE-DATASHEET.md`.
- Included the exact `oap/active` and immutable 015-b order transcript.

## Acceptance evidence

1. **Constructor/lifecycle contract — PASSED.** The literal 14-field tuple is
   asserted in order. Factory-spy coverage proves only the safe scalars are
   forwarded, `point_grids=None` and `output_mode="binary_mask"` are fixed, and
   profile/debug/model/path/device/dtype/cache/arbitrary controls are excluded.
   The counted A/B/A fake constructs three distinct generators around one model,
   returns different crop-0/crop-1 proposal masks, never writes a generator into
   the resident holder, restores A values, and records no request-time model
   load/reload/download, `.to()`, `.half()`, or dtype activity.

2. **Validation/profile/admission matrix — PASSED.** Focused parameterization
   covers every numeric lower/upper boundary and intrinsic failure, both boolean
   values, strict boolean/integer/number/string/null/non-finite cases, unknown
   profile/key and unsafe/fixed controls, deepest-layer rejection, every profile,
   explicit/profile/default precedence including equal inherited values, crop
   prompt formulas, multimask multipliers, equality/above-cap behavior, all six
   SAM2 cap environment variables, startup cap bounds, and deterministic 80%
   warnings without clamping.

3. **Public API rejection path — PASSED.** Invalid intrinsic configuration is
   HTTP 400 `invalid_config`; capacity excess is HTTP 413 `resource_limit`.
   A gate-acquisition spy, engine spy, and readiness spy prove both requests are
   rejected before inference, and response bodies contain only the sanitized
   error envelope.

4. **Capabilities — PASSED.** Two authenticated responses are byte-equivalent
   and expose exact field types/ranges, defaults, profiles, operator maxima,
   source precedence, formulas, and fixed-control exclusions. Missing and wrong
   credentials return 401. The readiness provider, gate, engine, and mutable
   request state are not consulted; no credential, path, GPU UUID, or process
   value is disclosed. The explicit `CapabilitiesResponse` schema is present in
   enabled OpenAPI; docs/OpenAPI remain disabled when the app is configured that
   way for the private-LAN policy.

5. **All verbosity levels and manifests — PASSED.** Verbosity 0, 1, 2, and 3
   each expose a complete typed `service.sam2` with all 14 effective/source
   entries, normalized request, selected profile, exact estimates, raw count,
   three-decimal nonnegative timing, and warnings. Monotonic artifact/object/L3
   fields remain gated. JSON and ZIP manifest metadata match at every level,
   with timing included in semantic parity comparisons and excluded only from
   byte-determinism claims.

6. **Raw candidate semantics — PASSED.** The controlled core fake returns one
   non-empty and one empty raw candidate, then exercises area filtering, CLIP,
   and BLIP3 fakes. L3 reports `actual_candidate_count=2` while
   `candidate_counts.sam2_candidates=1`; later stages do not alter the raw
   count.

## Verification

- `.venv/bin/pytest -q tests/test_sam2_configuration.py`: PASSED — 180 passed.
- `.venv/bin/pytest -q tests/test_sam2_configuration.py tests/test_segmenter_sam2.py tests/test_live_runtime.py tests/test_live_service_units.py tests/test_service_units.py tests/test_service_api.py tests/test_core_config.py tests/test_core_engine.py`: PASSED — 373 passed.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 646 passed, 79.59% total coverage, required 64.0% reached.
- GPU integration test: SKIPPED — explicit live GPU opt-in was not enabled; this order forbids repeating 015-a GPU inference evidence.
- `.venv/bin/ruff format --check .`: PASSED — 144 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current documents.
- `git diff --check`: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`: PASSED — wheel and sdist audited.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`: PASSED — no unexpected built-artifact findings.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`: PASSED — exact seven reviewed baseline findings.
- `.venv/bin/python -m twine check dist/*`: PASSED.
- Read-only pinned GPU signature probe with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python`: PASSED — all 14 public scalars exist; upstream variadic kwargs are present but wrapper forwarding excludes them; `point_grids` and `output_mode` are fixed to `None` and `binary_mask`.
- Preliminary over-strict signature assertion: FAILED — the installed upstream class exposes variadic kwargs; the final probe correctly treats that as an upstream capability while proving the request wrapper excludes arbitrary kwargs.
- Systemd/shell syntax: NOT RUN — no unit or shell file was changed by this order.

## CI/checks

All checks below are current-head SUCCESS for implementation SHA
`8eb9e4f16070795ae54d4fae9a7807cb6ad67660`:

- `static (format, lint, build)`: SUCCESS — CI run `33216919152`, job `99002516355`.
- `tests (py3.10)`: SUCCESS — CI run `33216919152`, job `99002516459`.
- `tests (py3.11)`: SUCCESS — CI run `33216919152`, job `99002516444`.
- `tests (py3.12)`: SUCCESS — CI run `33216919152`, job `99002516494`.
- `release (artifact audit)`: SUCCESS — CI run `33216919152`, job `99002516451`.
- `Analyze (python)`: SUCCESS — CodeQL run `33216919146`, job `99002516654`.
- `CodeQL`: SUCCESS — check run `99002709301`.

## GPU/service/resource evidence

- No GPU inference, restart, model load, or service mutation was performed in
  015-b.
- The order-assigned target remains physical GPU index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, NVIDIA GeForce RTX 3090,
  24,576 MiB, PCI `00000000:0B:00.0`, driver `610.43.02`; the application
  mapping is logical `cuda:0`.
- User unit `zap-it-lan.service` is enabled and active/running, PID `426972`,
  `NRestarts=0`, with exactly one listener at `10.8.132.76:17891`; readiness
  returned HTTP success. The assigned-GPU query showed only PID `426972`, with
  current used memory `13,950 MiB`.
- The mode-0600 operator environment remained unchanged with digest
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
- The request workspace contained zero entries after verification. No request
  image/config/result, credential, or raw input was written to the repository,
  persistent disk, or OAP evidence.

## Documentation/provenance

The API, configuration, testing, and service datasheet now say **14 total safe
generator scalars, including `use_m2m`**, while retaining the authoritative field
list. No model identity, revision, checkpoint, device, dtype, cache, residency,
network, authentication, artifact, or response policy was changed.

## Deferred human adjudication

- Critical register action: NONE

## Safety/scope confirmations

- Exact active order `015-b` was executed; no adjacent order was selected.
- Existing PR #71 was amended; no new PR was created, and no merge or auto-merge
  was performed.
- Unassigned GPUs, unrelated processes/services, firewall/VPN/network, global
  credentials, and the live service were not modified.
- The immutable active order and exact order bytes were committed with the
  implementation before the implementation SHA was captured.

## Limitations/blockers

- The GPU integration test remains SKIPPED by explicit policy and was not needed
  because 015-a already supplied the ordered live A/B/A qualification evidence.
- The installed upstream constructor accepts variadic kwargs; the wrapper is
  fail-closed and excludes them, as proved by the final signature probe and CPU
  factory-spy test.

## Factual strategic follow-up

PR #71 is current at implementation head
`8eb9e4f16070795ae54d4fae9a7807cb6ad67660`, open and mergeable, with all required
current-head CI/CodeQL checks successful. Strategic review/acceptance and merge
remain outside coding authority.
