# OAP Coding-Agent Report — 017-b

## Work order

- Identifier/order/objective/PR mode: `017-b`, close candidate-view contract and
  live proof, amend existing PR.
- Active selector: `017-b`.
- Immutable order SHA in implementation commit:
  `425b1697a776f4d928f2a75e0a5de12c3ef2fb559102a30f013c03dc9b7cbbd3`.

## Status

COMPLETE

## Executive summary

Implemented the bounded 017-b corrections on Objective-017 PR #73. Candidate
view artifacts now use fixed tokenized names, candidate views are one declared
top-level capability, both exported BLIP3 compositor names use the safe
mask-aware path, resized BLIP3 target pixels are restored exactly, dilation is
an exact local-window squared Euclidean distance transform, and debug resource
admission is split before CLIP and before BLIP3 QA.

The CPU/fake evidence, package/static gates, current-head CI/CodeQL, and the
authorized live qualification passed. The corrected service was restarted
exactly once, then remained on PID `528963`, enabled, active, ready, and clean
through the real L3 ZIP and A/B/A checks.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`.
- PR: [#73 — Objective 017: mask-isolated candidate views](https://github.com/ulfe-lmi/slaif-zap-it/pull/73).
- PR state: open, non-draft, mergeable, clean; base `main`.
- Required branch: `oap/017-a-mask-isolated-candidate-views`.
- Verified base SHA: `645c8604f9c189e1367e6e27a4ce8298c109482a`.
- Starting PR/report head SHA: `c7e530de18e659633048d9d317bb5e7cd3eca0d8`.
- Starting 017-a implementation parent: `d0ad70e7b978b7c314db596245d061cb42e6c390`.
- Implementation head SHA: `e08eb3cccc6395ee7bcf8aeb3994b4638afec0d7`.
- Implementation parent: `c7e530de18e659633048d9d317bb5e7cd3eca0d8`.
- Report publication commit: SELF.
- New PR: no. Existing PR amended: yes. Coding merge/acceptance: NO.
- Remote branch was verified at the implementation SHA before report
  publication.

## Changes/files

The implementation commit contains the exact 017-b selector and order plus a
bounded 22-path diff: `RELEASE_NOTES.md`, `TESTING.md`,
`docs/API.md`, `docs/CONFIG.md`, `docs/CORE.md`, `docs/OUTPUT-PARITY.md`,
`docs/SERVICE-DATASHEET.md`, `modules/classifier/clip.py`,
`modules/verifier/blip3.py`, `oap/active`,
`oap/orders/017-b-close-candidate-view-contract-and-live-proof.md`,
`src/core/engine.py`, `src/core/mask_views.py`,
`src/service/capabilities.py`, `src/service/schemas.py`,
`src/service/yaml_input.py`, `tests/test_candidate_view_api.py`,
`tests/test_core_engine.py`, `tests/test_mask_views.py`,
`tests/test_sam2_configuration.py`, `tests/test_service_units.py`, and
`tests/test_verifier_blip3.py`.

Diff size from the starting report head: 1,303 insertions and 263 deletions.
No model, dependency, lockfile, service unit, environment file, credential,
protected GPU, unrelated process, firewall, route, or persistent request-data
path was changed.

## Acceptance evidence

1. **Fixed names and identity — PASSED.** Runtime CLIP and BLIP3 sinks,
   records, capabilities, schemas, tests, docs, JSON, and ZIP evidence use
   `clip-candidate-view-CANDIDATE-####.png` and
   `blip3-verification-CANDIDATE-####-QUESTION-####.png`. Typed input records
   reject missing tokens, unsafe segments, stage mismatches, and numeric ID
   mismatches. Live records and members were one-to-one with five records from
   each stage.

2. **Capabilities/OpenAPI — PASSED.** `CapabilitiesResponse` has one required
   top-level `candidate_views: CandidateViewsCapability`, forbids dynamic
   top-level extras, and no longer nests candidate views under raw SAM2. Local
   runtime JSON, Pydantic schema, and application OpenAPI properties matched.
   The authenticated live capability response had the same top-level policy,
   exact templates/defaults/ranges/formula, no nested duplicate, and no
   credential/path/GPU-topology disclosure.

3. **Safe BLIP3 composition and resize identity — PASSED.** The old
   untouched rectangular/minimum-128 compositor and obsolete darkening/padding
   path were removed. `compose_verification_image` delegates to the same safe
   shared builder as `compose_blip3_verification_image`. High-contrast
   diagonal/tiny-mask tests and live exact PNG reconstruction proved left
   target-only pixels, zero outside support, exact right-target equality, and
   contour exclusion from target/support violations.

4. **Exact bounded dilation — PASSED.** The implementation uses a two-pass
   lower-envelope squared Euclidean distance transform over only the target
   bbox expanded by the effective radius. It has no radius-sized image cache
   and preserves holes, components, borders, clipping, and radius zero. The
   independent random brute-force oracle passed. The ordered 1672x941,
   effective-radius-512 subprocess regression completed in about 0.55 seconds
   with 56,056 KiB maximum RSS and valid exact output, below the 30-second and
   512-MiB limits.

5. **Two-phase admission — PASSED.** CLIP debug capacity is checked before
   the CLIP processor/model seam. Actual post-CLIP labels and scores are used
   for BLIP3 debug capacity immediately before the BLIP3 seam. Label-specific
   and negative-threshold `any` tests forced count, per-item, and total-byte
   failures; all proved zero forbidden BLIP3 QA calls and no partial stage
   artifacts. CLIP overflow likewise proved zero CLIP calls.

6. **Configuration, identity, API, and state evidence — PASSED.** Offline
   tests cover omitted/top/child defaults, inclusive endpoints, nulls,
   mappings, unknown fields/stages, bool-as-number, nonfinite values, all
   ranges, `min > max`, unsupported mode/fill, `clip.padding`, strict
   `clip.debug`, lower-verbosity stripping, persistent filtered indices,
   final object identity, JSON/ZIP manifests, input-record validation, and
   resident CLIP A/B/A debug isolation. The injected API fixture returned
   final objects plus exact CLIP and BLIP3 PNGs with matching media types,
   sizes, hashes, and ZIP members.

7. **CPU/static/package/CI gates — PASSED.** The canonical suite, all named
   static/package checks, and all seven current-head GitHub checks passed.

8. **Corrected-head live qualification — PASSED.** On one stable corrected
   PID, three authenticated L3 ZIP A/B/A requests returned HTTP 200 with five
   objects, five CLIP records, five BLIP3 records, exact names, exact input
   PNGs, exact RLE reconstruction, JSON/ZIP descriptor parity, and all pixel
   invariants. Both stages changed radii and input hashes for B and restored A.

9. **Final service state — PASSED.** The corrected newest service remained
   enabled, active, ready, on one listener and one assigned-GPU process with
   `NRestarts=0`, unchanged environment digest, sanitized journal, and empty
   request workspace.

## Verification

- `.venv/bin/pytest -q tests/test_candidate_view_api.py tests/test_mask_views.py tests/test_verifier_blip3.py tests/test_core_engine.py tests/test_sam2_configuration.py tests/test_service_units.py`: **PASSED** — 368 passed, 1 warning.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: **PASSED** — 783 passed, 1 honest GPU test skipped, 81.21% total coverage.
- `.venv/bin/ruff format --check .`: **PASSED**.
- `.venv/bin/ruff check .`: **PASSED**.
- `.venv/bin/python -m compileall -q src modules scripts tests`: **PASSED**.
- `.venv/bin/python scripts/check_documentation.py`: **PASSED** — 27 current documents.
- `git diff --check`: **PASSED**.
- `.venv/bin/python -m build --wheel --sdist`: **PASSED**.
- `scripts/verify_release_artifacts.py` on wheel and sdist: **PASSED** — 67
  wheel members and 160 sdist members.
- sdist-built wheel verification and wheel comparison: **PASSED** — identical
  member manifest digest `6cacab676f4cd7ef6f3d56a0bd17e928a3f994c5c5f3dd9e6a32a54550305c5a`.
- Release archive secret scan: **PASSED** — no unexpected findings.
- Tracked-tree secret baseline scan: **PASSED** — exactly the existing seven
  reviewed findings, no additions/removals/path changes.
- `python -m twine check` on wheel, sdist, and rebuilt wheel: **PASSED**.
- `systemd-analyze verify deploy/zap-it-local.service`: **PASSED**.
- Current-document stale search for numeric-only candidate/question names,
  minimum-128/untouched-context compositor semantics, and nested candidate
  capability claims: **PASSED** — no matches.
- Radius-512 independent oracle/resource tests: **PASSED**.
- Candidate-view JSON/ZIP, strict schema, name, identity, A/B/A, and two-phase
  admission tests: **PASSED**.
- Live endpoint/auth/capability check: **PASSED** — health/readiness 200,
  missing/wrong POST credentials 401, authenticated capabilities/metrics 200,
  private-LAN docs/OpenAPI 404.
- Live debug-false request: **PASSED** — HTTP 200, five final objects, zero
  candidate debug records/artifacts.
- Live corrected A/B/A ZIP proof: **PASSED** — all three HTTP 200 and all
  exact-media/pixel/name/identity checks passed.

## CI/checks

All seven checks were successful on implementation SHA
`e08eb3cccc6395ee7bcf8aeb3994b4638afec0d7`:

- `static (format, lint, build)`: **PASSED**, CI run `33280980325`, job
  `99175987388`.
- `release (artifact audit)`: **PASSED**, CI run `33280980325`, job
  `99175987295`.
- `tests (py3.10)`: **PASSED**, CI run `33280980325`, job `99175987387`.
- `tests (py3.11)`: **PASSED**, CI run `33280980325`, job `99175987417`.
- `tests (py3.12)`: **PASSED**, CI run `33280980325`, job `99175987385`.
- `Analyze (python)`: **PASSED**, CodeQL run `33280980331`, job
  `99175987312`.
- `CodeQL`: **PASSED**, check run `99176076590`.

## GPU/service/resource evidence

The order-assigned device was reverified before the GPU phase and remained the
only visible device: physical index `0`, UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
GeForce RTX 3090, 24,576 MiB, driver `610.43.02`, PyTorch `2.5.1+cu124`, CUDA
12.4, application logical `cuda:0`. The process environment was confirmed as
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0`. No unassigned
device or unrelated process was touched.

Before restart: enabled/active/ready user unit, PID `513853`, one listener at
`10.8.132.76:17891`, `NRestarts=0`, process RSS 4,433,552 KiB, GPU used/free
10,821/13,303 MiB, and one assigned-GPU compute process.

The order-authorized restart was performed exactly once. PID `528963` became
ready after normal resident checkpoint loading; it stayed active throughout
all live checks. Final state: enabled/active/running, PID `528963`, one
listener, one assigned-GPU process, `NRestarts=0`, process RSS 3,493,048 KiB,
GPU used/free 11,201/12,923 MiB, and readiness HTTP 200. Metrics recorded
Torch peak allocated/reserved bytes of 11,122,675,200/11,381,243,904, GPU
free bytes 13,550,485,504, and host high-water RSS 13,538,369,536 bytes.
The peak reserved value was below the ordered 90% physical-memory ceiling.

The mode-0600 operator environment digest remained
`bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
`/dev/shm/slaif-zap-it` remained mode 0700 and empty after each request and at
final verification. Journal scanning found no credentials, prompts, answers,
raw request data, traceback, OOM, or host/cache path disclosure.

### Live candidate-view evidence

The authorized ignored fixture was cropped in RAM from 5568x4176 to 2784x2088.
The corrected A/B/A ZIP requests all returned five final objects, 11 artifact
descriptors, 13 ZIP members including manifest/YAML-free result entries, and
five CLIP plus five BLIP3 input records. The recorded summaries were:

| Request | HTTP | latency ms | response bytes | CLIP/BLIP records | radii for both stages |
|---|---:|---:|---:|---:|---|
| A1 | 200 | 3754.5 | 467,476 | 5/5 | 6, 10, 14, 19, 20 |
| B | 200 | 3403.4 | 559,476 | 5/5 | 16, 28, 41, 55, 60 |
| A2 | 200 | 3199.2 | 467,481 | 5/5 | 6, 10, 14, 19, 20 |

All descriptor/member media types were `image/png`, sizes and SHA-256 values
matched, input records validated, source IDs were one-based, filtered indices
were persistent zero-based values, question IDs were one-based, and final
objects were present. Reconstructed object RLEs reproduced every recorded
CLIP/BLIP3 model-input PNG byte-for-byte. CLIP inputs were zero outside `D`;
BLIP3 left inputs were zero outside `M`, right inputs were zero outside `D`,
right target pixels equaled left target pixels, and contours were outside `M`
and inside `D` only.

For both stages, A1 and A2 artifact hash-prefix sets were identical and B was
different. The A radii/hash sets restored exactly after B. The service PID,
listener, model initialization, and residency transition count stayed stable;
the live metrics transition count was zero.

The first live harness attempt reset before an HTTP response because it passed
no key due to a harness environment-variable bug. A corrected-key attempt was
rejected with 400 `unsupported_field` because the in-memory legacy derivation
retained `clip.padding`; it ran no inference. A diagnostic 200 request showed
the original fixture rules produced five objects and CLIP records but no
BLIP3 debug records. These failures/retries are disclosed. The final bounded
`any,1.0` in-memory rule made five BLIP3 questions applicable; the complete
corrected A/B/A sequence then passed without OOM, timeout, failed HTTP status,
second process, or second restart.

## Documentation/provenance

Current docs now describe the tokenized names, independent top-level
capability, safe helper semantics, right-target equality after resize,
local-window radius-512 dilation, two-phase admission, pixel-isolation scope,
and bounded local qualification behavior. The independent stale-current-doc
search was clean. No historical qualification record was rewritten.

The report deliberately records only sanitized hashes, dimensions, counts,
timings, statuses, and resource facts; it contains no source pixels, uploaded
YAML, prompts, answers, credentials, model weights, or client filenames.

## Deferred human adjudication

- Critical register action: NONE.
- No CRITICAL entry was appended, edited, reordered, or closed.

## Safety/scope confirmations

- Only the exact order-assigned physical GPU 0/UUID was used; it was exposed as
  logical `cuda:0`.
- No service other than the explicitly authorized `zap-it-lan.service` was
  restarted or changed.
- No merge, auto-merge, release, tag, publish, firewall, network, driver,
  system CUDA, unrelated process, device, dependency, model identity, or
  credential mutation occurred.
- Request bytes/results remained in RAM or the configured empty RAM-backed
  workspace; no repository output or persistent request data was used.
- PR #73 remains open; coding did not accept or merge it.

## Limitations/blockers

Semantic label/answer accuracy was not an acceptance gate and no accuracy or
recall claim is made. The live proof is bounded local research evidence on the
assigned RTX 3090 and does not authorize public/WAN exposure, production data,
release, or merge. The disclosed pre-inference harness failures were corrected
without a second restart; the final requested sequence itself had only HTTP
200 responses.

## Factual strategic follow-up

PR #73 is ready for strategic review at implementation head
`e08eb3cccc6395ee7bcf8aeb3994b4638afec0d7`; it was not merged or accepted by
this coding round. The final child of this implementation commit is this
immutable report-only SELF commit.
