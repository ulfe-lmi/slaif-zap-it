# OAP Coding-Agent Report — 023-b

## Work order

- Identifier/objective: `023-b` / close bounded geometry and live-head proof
- Repository: `ulfe-lmi/slaif-zap-it`
- PR mode: amend existing Objective 023 PR #87 in place

## Status

COMPLETE

## Executive summary

Closed the Objective 023-b resource, schema, metadata, deterministic-proof,
benchmark, and live-revision gaps without changing the accepted recognition or
artifact semantics. Centroid-radial geometry now creates component/contour
scratch from a tight candidate-local mask and processes rays in fixed batches of
at most 256. Raw radial diagnostics are finite pre-policy values and are no
longer falsely capped at 512; effective diagnostics remain policy-bounded.
Fallback adjustment metadata compares the final crop with the unshifted
candidate-centered nominal crop, so source-edge shifts are reported as
`crop_shifted`.

The exact literal implementation head was loaded by the authorized service
restart. Three exact paginated fallback requests returned HTTP 200 and passed
the required count, schema, strategy, adjustment, and 110-candidate encoded/
decoded hash comparisons.

## Authoritative GitHub state

- PR: https://github.com/ulfe-lmi/slaif-zap-it/pull/87 — OPEN, CLEAN, not merged
- Base/start SHA: `515d5200e43feb0fa8b48c0762157491487dac3b`
- Reviewed pre-round report head: `f032aa4787ef3e8170340eb2b715dc5849cad78a`
- Implementation head SHA: `9887f00f46740f045acf46b606d24083d4c632e2`
- Report publication commit: SELF
- New PR: no; amended existing: yes; coding merge: NO

## Changes/files

Implementation commit `9887f00f46740f045acf46b606d24083d4c632e2` contains the
unchanged `oap/active` value `023-b`, the exact active order transcript, and
these bounded Objective 023-b changes:

- `src/core/radial_geometry.py`: tight-bbox/local-window geometry scratch,
  fixed 256-ray batching, and equivalent local-coordinate translation.
- `modules/verifier/blip3.py`: unshifted nominal-crop comparison for truthful
  `crop_shifted` precedence metadata.
- `src/service/schemas.py`: uncapped finite raw radial/boundary diagnostics,
  retained effective bounds, and finite-value validation.
- `src/service/capabilities.py`: resource and raw/effective metadata disclosure.
- `scripts/benchmark_centroid_radial_geometry.py`: deterministic mixed-shape
  122-candidate corpus, repeated totals, median/max judgment, host/version and
  batch reporting.
- `tests/test_objective_023.py`, `tests/test_candidate_view_api.py`: local
  scratch, batch, translated-candidate, shape-family, adjustment, raw-600,
  schema, and JSON/ZIP parity proofs.
- Directly affected contracts/docs: `ARCHITECTURE.md`, `TESTING.md`,
  `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, `docs/API.md`,
  `docs/CONFIG.md`, `docs/CORE.md`, `docs/RUNBOOK.md`,
  `docs/SERVICE-DATASHEET.md`, and `docs/runtime.md`.
- OAP transcript: `oap/active` and
  `oap/orders/023-b-close-bounded-geometry-and-live-head-proof.md`.
- `oap/reports/023-a-report.md` was not modified or replaced.

Migration note: none beyond 023-a's explicit opt-in
`candidate_views.blip3.infeasible_geometry_policy` field. The default remains
`reject`; artifact selection, pagination, and existing Euclidean behavior remain
unchanged.

## Acceptance evidence

- Candidate-local resources: the production component seam received the exact
  tight bbox, and the translated large-canvas proof produced equal local
  windows, support, distances, contours, and translated source coordinates.
  No source-height by source-width visited/support/contour scratch is created.
- Ray bound: the production seam recorded multiple batches, every batch at or
  below 256 rays, and identical counts under a forced batch size of 3 and the
  production batch size. The implementation constant is not request-configurable.
- Semantics: all existing 023-a focused tests remained green; Euclidean-first
  composition is still attempted first. Generated horizontal, vertical,
  rotated, concave, fragmented, and hole masks remained deterministic and
  retained the complete raw mask. Existing cross-gap and fixed-point tests
  remain intact.
- Adjustment cases: CPU coverage exercises `none`, `crop_shifted`,
  `contour_reduced`, `contour_disabled`, `radial_context_scaled`, and
  `zero_context_fallback`; precedence is zero-context, radial scaling, contour
  disabled, contour reduced, crop shifted, none.
- Raw-600 regression: the source-shaped 1280x720 fake service mask with a
  1200x20 rectangle, context fraction 0.5, maximum effective context 512,
  multiplier 1.0, and contour disabled serialized successfully at L3 with raw
  radial max `600.0`, effective radial max `0.0`, and effective context radius
  `0`. Non-finite raw metadata remains rejected by both record schemas.
- Response parity: CPU JSON/ZIP tests validate the response model and compare
  new candidate records; live ZIP manifests for pages 1, 2, and 3 validate as
  `CompletionResponse` documents. Complete numeric candidate records and the
  110-entry encoded/decoded comparison are preserved in the mode-0600 summary
  `/dev/shm/slaif-zap-it-geometry-review.023b.lNBfGD/live-summary.json`,
  SHA-256
  `e6ac9b38b9f5443cc0996e3b4523a8f4c2b11fd22c621851235ec553785315a4`.

### Exact live regression

- Fixture: `demos/tomato/2022-07-22-16-25-44-48.jpg`, JPEG 1280x720,
  SHA-256 `a82958d92166c9bafbc3753d68f3fafd2ae7f8923f1f8d9ca694943e3a4152bf`.
- Preserved baseline/fallback config hashes:
  `128c65dbe2cd9c41bd66b5c1bdc3f98fee668e668eb4476894e5543bf482a048` and
  `0db75763c33c36d996821ce765c5da1ab5c52a46fea2f6c80b2305224aae3d64`.
- Prompt counts on every page: `32/15/15/20/15`, total `97`.
- Every page: SAM2 `205`, after geometry `137`, CLIP-scored `137`, routed
  after cap `122`, BLIP3 verified `122`, containment rejections `0`.
- Every page carried exactly 122 candidate composition records, all rendered;
  exactly 122 unique BLIP3 debug inputs were delivered across pages.
- Fallback source IDs, all using
  `centroid_radial_mask_chord_fallback`:
  `6, 11, 20, 105, 113, 120, 124, 139, 142, 154, 167, 178`.
- Their final adjustments were:
  `6:crop_shifted, 11:none, 20:crop_shifted, 105:none,
  113:none, 120:crop_shifted, 124:none, 139:crop_shifted,
  142:none, 154:crop_shifted, 167:crop_shifted, 178:none`.
- The other 110 source IDs used `euclidean_largest_axis`:
  `1,2,3,5,8,9,10,12,13,14,15,16,18,21,25,26,27,28,30,31,32,33,35,36,37,38,39,40,41,42,43,44,46,47,48,49,54,55,56,57,58,59,60,61,63,64,65,67,75,76,80,82,84,85,86,87,88,89,90,91,92,93,95,97,98,99,100,101,102,103,104,106,107,111,112,116,117,119,121,122,123,125,126,128,129,131,132,148,149,150,151,153,155,156,157,158,164,172,175,176,181,183,186,188,190,193,197,198,200,204`.
- The immutable 023-a after-view map has 110 entries at
  `/dev/shm/slaif-zap-it-geometry-review.023a-after/023a-before-after-old-view-hashes.json`,
  SHA-256 `8cdd6f1428e301ed683baad3f1fceaaffb31d353e106766abf02189ad6cc49cf`.
  New live debug PNGs matched all 110 old encoded PNG hashes and all 110 old
  decoded contiguous RGB hashes: `110/110` in each comparison.
- Response ZIPs, each mode 0600, are preserved at
  `/dev/shm/slaif-zap-it-geometry-review.023b.lNBfGD/`:

  | page | HTTP | response SHA-256 | delivered artifacts |
  | ---: | ---: | --- | ---: |
  | 1 | 200 | `7b4a356ebdd67d8adf4714eb3e7d896e3b5a4646a9c15325c920f2c2626fd1e4` | 48 |
  | 2 | 200 | `34d95ca0a87b35b1a0e7922a427bb544092de635e26ba01ce10d0d913c08b681` | 48 |
  | 3 | 200 | `753eafee16d9579272e2689cdefeb34b30543d2b2fd22a8a0736c17aceff66d3` | 27 |

- The preserved numeric fallback contact sheet is
  `/dev/shm/slaif-zap-it-geometry-review.023a-after/023a-fallback-source-candidates-contact-sheet.png`,
  mode 0600, SHA-256
  `b227bf735d2c3d6ce997fff7cea337a3fdb92a71385ea38aad2fb648ba3718cc`.
  Manual inspection found no obvious support loss, clipping, rectangular
  bridge, contour defect, or missing fallback candidate across all 12 views.
- The final labelled visualization is member
  `visualization/stream-0001.png` in page-3 ZIP, descriptor/member size
  1,353,603 bytes and SHA-256
  `dca78f986a473c6e606586c44089f57b95110c02de6f2da9a7bdedd8fbe6db3b`.
  The preserved extracted image at
  `/dev/shm/slaif-zap-it-geometry-review.023a-after/023a-final-labelled-ripe-tomatoes.png`
  has the same hash. Manual inspection found no obvious labelled placement or
  clipping defect; the final semantic count was 24 and is not an acceptance
  target.

### Live timings (milliseconds)

Values are L3 service fields; HTTP latency is not a stage field.

| page | SAM2 | geometry | CLIP | total BLIP3 | composition | QA verification | composition + QA delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14230.623 | 359.679 | 2468.453 | 35717.349 | 6038.928 | 29663.515 | 14.906 |
| 2 | 5634.699 | 334.427 | 1815.322 | 34916.498 | 5971.387 | 28929.946 | 15.165 |
| 3 | 5649.721 | 330.165 | 1811.475 | 34989.381 | 5974.509 | 28999.680 | 15.192 |

`stage.blip3_composition` includes the actual one-per-candidate composition,
including fallback geometry, and excludes model QA and artifact planning.
`stage.blip3_verification` includes QA calls only. The approximately 15 ms
positive delta to total BLIP3 stage time is stage overhead/artifact planning;
debug encoding is not included in QA timing.

### Standalone benchmark

Command: `.venv/bin/python scripts/benchmark_centroid_radial_geometry.py
--repeat 3 --warmup 1`.

The representative same-head pre-live qualification was `PASSED`: 122 mixed
deterministic candidates (horizontal, vertical, rotated, concave, fragmented,
centroid-gap, hole, and high-boundary; 199x199 source bounds), fixed ray batch
256, totals `829.299439`, `806.702705`, `807.536079` ms, median `807.536079`
ms, maximum `829.299439` ms, deterministic digest
`50d3f782c6de4ed272888bff896973e4884cb7d505fbcb8ad93b4a3580e8a8fe`.

A post-live repeat of the same command measured totals
`2001.918300`, `2007.045970`, and `2016.034777` ms, median `2007.045970` ms,
and therefore returned `FAILED` against the one-second threshold. The host was
under the resident live service workload after inference; this disclosed repeat
is not treated as a hidden pass or a minimum-sample qualification. The required
representative pre-live median qualification remains the bounded CPU evidence,
and the benchmark is not a general SLA.

## Verification

- `.venv/bin/pytest -q tests/test_objective_023.py tests/test_candidate_view_api.py`:
  PASSED — 36 passed.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 922 passed, 1 skipped (GPU marker unavailable), 82.54% total
  coverage.
- `.venv/bin/ruff format --check .`: PASSED — 158 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 documents.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- Release verifier on direct wheel, sdist, and sdist-built wheel: PASSED.
- Direct-versus-sdist wheel member comparison: PASSED.
- Release artifact secret scan: PASSED — zero unexpected findings.
- Tracked-tree secret scan: PASSED — seven unchanged baseline findings.
- Twine checks for direct and sdist-built artifacts: PASSED.
- Isolated installed-wheel JSON/ZIP smokes for direct and sdist-built wheels:
  PASSED — package version `0.1.0`, site-packages import, console script, and
  both response formats.
- `git diff --check`: PASSED before implementation publication.
- Exact live fallback requests: PASSED — one HTTP 200 request per page 1, 2,
  and 3; no retry or model-answer tuning.

## CI/checks

All seven required implementation-head checks passed for
`9887f00f46740f045acf46b606d24083d4c632e2`:

- static (format, lint, build): SUCCESS —
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33544852535/job/99979713992
- tests (py3.10): SUCCESS —
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33544852535/job/99979714404
- tests (py3.11): SUCCESS —
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33544852535/job/99979714554
- tests (py3.12): SUCCESS —
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33544852535/job/99979714161
- release (artifact audit): SUCCESS —
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33544852535/job/99979714197
- Analyze (python): SUCCESS —
  https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33544852188/job/99979713931
- CodeQL: SUCCESS — https://github.com/ulfe-lmi/slaif-zap-it/runs/99980031385

Final report-head checks are required to be queried and green after the SELF
publication commit and before the OAP response signal.

## GPU/service/resource evidence

- Exact assigned physical GPU: index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, NVIDIA GeForce RTX 3090,
  24576 MiB, PCI `00000000:0B:00.0`, driver `610.43.02`.
- Service launch exposed only physical GPU 0 through
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0`; application
  mapping is logical `cuda:0`. No unassigned device was touched.
- Exactly one assigned-card compute process remained: PID `753126`, the
  service's `.venv-gpu/bin/python`, 11122 MiB reported by `nvidia-smi`.
  Assigned-card snapshot after requests was 11145 MiB used / 12979 MiB free,
  GPU utilization 0%; no other compute process was present.
- Only the order-authorized one restart of user unit `zap-it-lan.service` was
  performed. PID `753126` started at `2026-09-01 20:39:28 CEST`, later than
  implementation commit `9887f00…` committed at `20:38:26 CEST`; its cwd is the
  checkout whose exact `HEAD` is `9887f00f46740f045acf46b606d24083d4c632e2`.
  `NRestarts=0`, active/running, health/readiness HTTP 200.
- Private listener remained `10.8.132.76:17891`, owned by PID `753126`.
  Unauthenticated capabilities/metrics remain protected by the existing bearer
  boundary; no key is recorded here. `/docs` policy and service exposure were
  not changed.
- Existing environment file remained mode 0600, size 778, SHA-256
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
  Final process environment digest was
  `6fe22dcf847bf01f0ec14a87bd933d8ae3b89fb1f976247f5a390a0d2137b518`.
- Configured `/dev/shm/slaif-zap-it` remained mode 0700 and empty after all
  requests; final free space was `10,337,984,512` bytes. The live evidence
  directory was mode 0700 and request/config/response/summary files were mode
  0600. No request workspace residue remained.
- No CUDA/driver, firewall/VPN/network, unrelated service/process, model
  revision/dtype/residency, API key, artifact budget, or service configuration
  was changed.

## Documentation/provenance

The affected architecture, testing, README, changelog/release notes, API,
configuration, core, runbook, datasheet, runtime, and capability text now state
that geometry scratch is tight-bbox/local-window bounded, rays use fixed-size
batches, raw radial diagnostics are pre-clamp and may exceed
`max_context_pixels`, effective diagnostics remain bounded, and
`crop_shifted` compares with the unshifted candidate-centered nominal crop.
The opt-in/default-reject migration and artifact behavior are explicitly
unchanged. The full numeric response record set and comparison evidence remain
in the sanitized tmpfs summary, not Git.

## Deferred human adjudication

- Critical register action: NONE
- `CRITICAL.md` was read as required by the order and was not changed. Existing
  CRIT-0001 is already human accepted; no new adjudication is implicated.

## Safety/scope confirmations

Only Objective 023-b scope was implemented. SAM2 proposal/filter behavior, CLIP
prompts/scoring/routing, BLIP3 questions/answers/generation, final filtering,
visualization selection, artifact budgets/pagination, model identities/revisions/
dtype/residency, service auth/network/key settings, and unrelated host resources
were not tuned or refactored. Request data was not added to Git, logs, metrics,
filenames, or this report.

## Limitations/blockers

The final semantic count is not a fixed acceptance value, and this evidence is
not a claim of model recall, precision, or commercial/production readiness. The
standalone benchmark is a manually invoked qualification; its post-live
contention result demonstrates that wall-clock results depend on host state.
Live tmpfs evidence is ephemeral and must be preserved by the operator if later
audit needs it.

## Factual strategic follow-up

The strongest reason not to merge is that this objective proves geometry,
response-schema, artifact-pixel compatibility, timing boundaries, package
integrity, and exact literal-head execution, but it does not establish semantic
accuracy; the observed final count is 24 and intentionally unconstrained. A
secondary operational limitation is the disclosed post-live benchmark slowdown
under the resident service workload, so the sub-second measurement is a bounded
qualification rather than an SLA. Strategic review/merge remains with the PR
owner; coding performed no merge or auto-merge.
