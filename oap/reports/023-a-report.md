# OAP Coding-Agent Report — 023-a

## Work order

- Identifier/objective: `023-a` / centroid-radial mask-chord fallback
- Repository: `ulfe-lmi/slaif-zap-it`
- PR mode: new PR, exactly one PR for Objective 023

## Status

COMPLETE

## Executive summary

Implemented the explicitly opt-in `centroid_radial_mask_chord` fallback for
one-image BLIP3 `single_dilated_blur` composition. The existing Euclidean path
is still attempted first; omission and explicit `reject` preserve its behavior,
and successful Euclidean views retain their pixels and existing values. The
fallback is pure, deterministic, mask-only, request-local, and reports bounded
geometry/timing metadata.

## Authoritative GitHub state

- PR: https://github.com/ulfe-lmi/slaif-zap-it/pull/87 — OPEN, CLEAN, not merged
- Base/start SHA: `515d5200e43feb0fa8b48c0762157491487dac3b`
- Implementation commits: `507550f62b9157d3b00cdadf5d34c674cfb154a5`, then
  `6d63de7a2fce65b94e8212543e34a7bab25b79a4`
- Implementation head SHA: `6d63de7a2fce65b94e8212543e34a7bab25b79a4`
- Report publication commit: SELF
- New PR: yes; amended existing: no; coding merge: NO

## Changes/files

- Added `src/core/radial_geometry.py` and exported the production geometry.
- Added strict request-local policy/config handling in `src/core/mask_views.py`
  and `src/service/yaml_input.py`.
- Added Euclidean-first fallback routing, contour/crop adjustment order,
  fixed-point scaling, source-pixel composition, and separate composition/QA
  timers in `modules/verifier/blip3.py`.
- Added bounded L3 fields to the candidate-view records and runtime provenance
  schema, plus capability/OpenAPI disclosure.
- Added `tests/test_objective_023.py`, API assertions, and
  `scripts/benchmark_centroid_radial_geometry.py`.
- Updated the maintained README, architecture, API/config/core/output-parity,
  runtime/runbook/datasheet/testing, changelog, and release notes.
- Published unchanged active/order transcript paths `oap/active` and
  `oap/orders/023-a-centroid-radial-mask-chord-fallback.md`.

## Acceptance evidence

- Policy: only `reject` and `centroid_radial_mask_chord` are accepted; default
  is `reject`; invalid/non-string values return sanitized `invalid_config`.
- Geometry: whole-mask float64 centroid, 8-connected components, external
  contour walks with deterministic ordering, inclusive all-octant rasterized
  chords/spokes, cross-gap positive counts, exact Euclidean exterior contour,
  full nominal crop shifting, contour reduction/disable order, common
  millionth fixed-point scale, and zero-context fallback are covered by CPU
  tests.
- Compatibility: feasible fallback-enabled views were pixel/crop/support/
  contour/resize and existing-metadata identical to the Euclidean path.
- Benchmark: `python scripts/benchmark_centroid_radial_geometry.py --repeat 3
  --warmup 1` — PASSED; 122 masks, 199x199 bounds, one warmup, three repeats,
  total `312.345..327.396 ms`, deterministic digest
  `ec3aecc72f99dab51160caade55b2fc60be4cd20d910cf1cb2942ae6be6cf4ba`.

### Exact live regression

The required image SHA-256 was
`a82958d92166c9bafbc3753d68f3fafd2ae7f8923f1f8d9ca694943e3a4152bf`.
The strategic baseline config/ZIP/fallback config hashes were respectively
`128c65dbe2cd9c41bd66b5c1bdc3f98fee668e668eb4476894e5543bf482a048`,
`ce35534fccd36a0ed05b759d7aec40d872932fefe89b228267663d6000f20d3a`, and
`0db75763c33c36d996821ce765c5da1ab5c52a46fea2f6c80b2305224aae3d64`.

Before restart, baseline pages 1/2/3 each reproduced:

```text
SAM2 205; after geometry 137; CLIP scored 137; initially routed 122;
routed after cap 122; BLIP3 verified 110; containment rejections 12;
rejected source IDs 6,11,20,105,113,120,124,139,142,154,167,178;
prompt counts 32/15/15/20/15; total prompts 97.
```

Baseline page-2/page-3 response ZIP hashes were
`5544107a6cf1ecfbe9c6949361b31c7bc5cd17e20760218d8942c3a55be64edd` and
`a462ce8566dc5d497705bc6ba298ae6315dd48511b6957d889ca8ac898cda600`.

After the one authorized restart, fallback pages 1/2/3 each returned HTTP 200
and reproduced SAM2 `205`, after geometry `137`, CLIP scored `137`, routed
`122/122`, BLIP3 verified `122`, and zero containment rejections. Exactly the
prior 12 IDs used `centroid_radial_mask_chord_fallback`; the other 110 used
`euclidean_largest_axis`. The 110 old rendered candidates had all encoded PNG
SHA-256 values and all decoded contiguous RGB SHA-256 values equal before and
after. The complete bounded comparison map is preserved at
`/dev/shm/slaif-zap-it-geometry-review.023a-after/023a-before-after-old-view-hashes.json`
(mode 0600, SHA-256
`8cdd6f1428e301ed683baad3f1fceaaffb31d353e106766abf02189ad6cc49cf`).

There were exactly 122 unique delivered fallback-page debug inputs, one for
each verified candidate, and each fallback candidate was sent to BLIP3 once.
The numeric-only contact sheet is
`/dev/shm/slaif-zap-it-geometry-review.023a-after/023a-fallback-source-candidates-contact-sheet.png`
(mode 0600, SHA-256
`b227bf735d2c3d6ce997fff7cea337a3fdb92a71385ea38aad2fb648ba3718cc`). The
final labelled image is
`/dev/shm/slaif-zap-it-geometry-review.023a-after/023a-final-labelled-ripe-tomatoes.png`
(mode 0600, SHA-256
`dca78f986a473c6e606586c44089f57b95110c02de6f2da9a7bdedd8fbe6db3b`); its
fixed ZIP member was `visualization/stream-0001.png`, with logical
`visualization_id=final-labelled-ripe-tomatoes` and descriptor hash/size parity.
The post-restart page-1/page-2/page-3 response ZIP hashes were
`bb7982911dde6456739728e1542ebf8868d2d49805182af4ca68080c322a678c`,
`f8fc9c8bb5ddaed8585d13970a977e5ca161a74757715c07803c37ccabd1c101`, and
`85c7fbe62c7da0a1997234db65fefb95c76601985c2541f9e6c2b936ef3b4373`.

Visual inspection of all 12 views via the contact sheet and of the final
labelled image found no obvious support loss, clipping, rectangular bridge,
contour defect, or unsent candidate. Disconnected shapes remained visibly
disconnected where present. Final count was 24 and is not treated as a fixed
semantic acceptance value.

### Live timings (milliseconds)

Values are the exact L3 timing fields; HTTP latency is the end-to-end request
measurement. Page 1 is the post-restart warm-up/cold-effect page; pages 2 and 3
are warm repeats.

| page | HTTP latency | SAM2 | geometry | CLIP | total BLIP3 | composition | QA verification |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45063.6 | 12191.046 | 253.850 | 1285.758 | 28916.639 | 3265.796 | 25641.097 |
| 2 | 35255.7 | 4409.942 | 253.092 | 830.623 | 28246.555 | 3217.010 | 25019.437 |
| 3 | 35082.7 | 4412.095 | 253.459 | 828.031 | 28240.406 | 3202.553 | 25027.645 |

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 909 passed, 1 skipped (GPU marker unavailable), 82.42% total
  coverage.
- `.venv/bin/pytest -q tests/test_objective_023.py tests/test_mask_views.py tests/test_candidate_view_api.py tests/test_objective_021.py`:
  PASSED — 97 passed.
- `.venv/bin/ruff format --check .` and `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 documents.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- Release verifier on direct and sdist-built archives: PASSED; direct wheel
  and sdist member comparison: PASSED; Twine checks: PASSED.
- Release secret scan: PASSED — zero archive findings; tracked-tree scan:
  7 unchanged baseline findings.
- Isolated sdist-built wheel JSON/ZIP service smoke: PASSED; systemd unit
  verification: PASSED; JSON/ZIP response schema and artifact hash/size
  parity: PASSED.

## CI/checks

All checks passed on literal implementation head
`6d63de7a2fce65b94e8212543e34a7bab25b79a4`:

- CI run https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33540643877:
  static `99965739551`, tests py3.10 `99965739765`, tests py3.11
  `99965739442`, tests py3.12 `99965739521`, release audit `99965739207` —
  SUCCESS.
- CodeQL run https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33540643775,
  check `99966073789` — SUCCESS.

## GPU/service/resource evidence

- Exact assigned physical GPU: index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, NVIDIA GeForce RTX 3090,
  24576 MiB, PCI `00000000:0B:00.0`, driver `610.43.02`; process environment
  exposed only `CUDA_VISIBLE_DEVICES=0` as logical `cuda:0`.
- Before/after assigned-card process: PID `708466` / 10870 MiB, then PID
  `737533` / 11122 MiB. One controlled restart of only user unit
  `zap-it-lan.service` was performed; final `NRestarts=0`, active/running,
  health/readiness HTTP 200, private listener `10.8.132.76:17891` owned by
  PID `737533`.
- Private boundaries: unauthenticated capabilities HTTP 401, docs HTTP 404,
  metrics HTTP 401. Environment file was mode 0600, size 778, SHA-256
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`;
  bearer was never written to evidence or report.
- `/dev/shm/slaif-zap-it` remained mode 0700 and empty after requests, with
  10,354,151,424 bytes free at final check. Evidence directories were mode
  0700 and all evidence files mode 0600. No request workspace residue remained.
- No unassigned GPU, unrelated process, CUDA/driver, firewall, network,
  service unit, model cache, port, artifact budget, or residency configuration
  was changed.

## Documentation/provenance

The effective configuration, capability disclosure, OpenAPI models, L3
composition/input records, scalar-radius compatibility semantics, fixed-point
rounding, contour ordering, degenerate centroid convention, cross-gap behavior,
adjustment precedence, timing boundaries, default compatibility, and benchmark
procedure are documented in the maintained files listed above.

## Deferred human adjudication

- Critical register action: NONE
- `CRITICAL.md` was read as required by the order and was not changed.

## Safety/scope confirmations

Only Objective 023 scope was implemented. SAM2, CLIP prompts/scoring/routing,
BLIP3 questions/answers/generation, final filtering, visualization execution,
artifact pagination/budgets, model identities/revisions/dtype/residency, API
exposure, and unrelated host processes were not tuned or refactored.

## Limitations/blockers

The live run is exact geometry/provenance regression evidence, not a claim of
semantic recall, precision, or production/release readiness. Final semantic
count is not fixed by the order. The benchmark is a manually invoked bounded
CPU workload and does not establish a general SLA. Existing CRIT-0001 is
already human accepted and no new adjudication is implicated.

## Factual strategic follow-up

The strongest reason not to merge is that no deterministic geometry test or
live parity check establishes model accuracy; that limitation is explicitly
outside this objective, while all ordered geometry, compatibility, schema,
artifact, timing, package, CI, and exact live-regression criteria passed.
Strategic review/merge decision remains with the PR owner; coding performed no
merge or auto-merge.
