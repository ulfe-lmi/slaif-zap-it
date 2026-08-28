# OAP Coding-Agent Report — 012-b

## Work order

- Identifier/order/objective/PR mode: `012-b`, labelled-renderer evidence
  hardening, amend the existing Objective-012 PR.

## Status

COMPLETE

## Executive summary

Amended Objective-012 with focused, test-only evidence for deterministic
labelled rendering and bounded label layout. The tests now compare repeated
rendered arrays and shared-service PNG bytes directly, independently vary the
final label and instance ID, inspect the renderer's actual Pillow rectangles and
text bounds at image edges, prove dynamic fitting retains the ID and finite
confidence suffix, cover physically impossible tiny canvases accurately, and
repeat nearby-object placement. No production code or live service state was
changed.

## Authoritative GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`; PR [#68](https://github.com/ulfe-lmi/slaif-zap-it/pull/68)
  is `OPEN`, base `main`, branch `oap/012-a-labelled-api-visualization`, and
  `MERGEABLE`/`CLEAN` at verification.
- Remote base SHA: `ce41b0becfb53cfe96ac11570a1af23b2d963311`.
- Starting SHA: `f09ed52ef3b7a03461b9f98c20b6443506c2f6b4` (012-a SELF report).
- Implementation head SHA: `c10c2518b9c77dd564c776f26fd9a2cbaa6be45b`.
- Report publication commit: SELF
- New PR: no; amended existing PR #68: yes; coding merge: NO.

## Changes/files

Implementation commit `c10c2518b9c77dd564c776f26fd9a2cbaa6be45b` contains only:

- `tests/test_labelled_visualization.py`: direct array/PNG hash, pixel
  sensitivity, Pillow geometry, fitting, tiny-canvas, and repeated-placement
  evidence.
- `oap/active`: exact wrapper-selected `012-b` selector.
- `oap/orders/012-b-labelled-renderer-evidence-hardening.md`: exact active
  order transcript.

No production renderer, core, service, dependency, or documentation file was
changed.

## Acceptance evidence

- `test_labelled_renderer_repeats_exact_bytes_and_is_sensitive_to_final_label_and_id`:
  repeated calls use identical image/mask/object/alpha/confidence inputs and
  assert exact array equality, equal array-byte SHA-256, byte-equal PNGs from
  the service's shared `_encode_png` path, and equal PNG hashes. Independent
  `dataclasses.replace` variants change only the final structured label or
  only `instance_id`; each changes pixels and PNG hash. The original object,
  metadata, and mask remain unchanged.
- `test_labelled_edge_layout_records_bounded_boxes_and_pillow_text_bounds` is
  table-driven for top, bottom, left, right, top-left corner-adjacent, and
  bottom-right corner-adjacent masks on a `160x120` image. A delegating
  recorder observes the actual renderer `rectangle` and `text` calls, verifies
  each selected background rectangle has positive extent within the image, and
  verifies each actual Pillow `textbbox` is inside both the background and
  image after independently establishing that the mandatory suffix fits.
- `test_labelled_layout_shortens_long_label_and_preserves_finite_suffix` uses a
  long safe label on a `112x96` image, asserts measured Pillow text width is no
  greater than the available `108` pixels, and asserts actual visible text is
  shortened while preserving the exact `23   CLIP 0.88` suffix.
- `test_border_and_tiny_masks_are_bounded` retains `1x1`, `2x3`, and `8x8`
  cases, records bounded rectangles, bounds visible text length, and repeats
  output/placement. Its comment accurately treats an unfittable bitmap suffix
  as deterministic Pillow clipping rather than claiming impossible readability.
- `test_nearby_label_boxes_are_not_completely_overlapped_when_candidates_allow`
  now repeats output and directly compares the recorded background rectangles
  and text coordinates, in addition to the existing non-complete-overlap
  assertion.

## Verification

- `.venv/bin/pytest -q tests/test_labelled_visualization.py tests/test_visualizer.py tests/test_core_engine.py tests/test_core_renderers.py tests/test_service_api.py tests/test_parity_hardening.py`:
  PASSED — 138 tests.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 447 tests passed, 1 opt-in GPU test skipped because
  `ZAP_IT_RUN_GPU=1` was not enabled; total coverage `77.82%`.
- `.venv/bin/ruff format --check .`: PASSED — 142 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current
  documents.
- `.venv/bin/python -m pip check`: PASSED — no broken requirements.
- `.venv/bin/python -m build --wheel --sdist`: PASSED — wheel and sdist built;
  existing setuptools license deprecation warnings were non-fatal.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  PASSED — wheel and sdist member audits.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`:
  PASSED — exactly 7 reviewed baseline findings.
- `.venv/bin/python scripts/scan_release_artifacts.py dist/*.whl dist/*.tar.gz --baseline .secrets.baseline`:
  PASSED — no unexpected archive findings.
- `.venv/bin/python -m twine check dist/*`: PASSED.
- `git diff --check`: PASSED.
- No model download, CUDA inference, request persistence, or production
  renderer adjustment was needed in this test/comment-only continuation.

## CI/checks

All current implementation-head checks completed `SUCCESS` for
`c10c2518b9c77dd564c776f26fd9a2cbaa6be45b` on PR #68:

- `static (format, lint, build)` — PASSED; CI run `33194575340`, job
  `98928320866`.
- `tests (py3.10)` — PASSED; CI run `33194575340`, job `98928320319`.
- `tests (py3.11)` — PASSED; CI run `33194575340`, job `98928320475`.
- `tests (py3.12)` — PASSED; CI run `33194575340`, job `98928320121`.
- `release (artifact audit)` — PASSED; CI run `33194575340`, job
  `98928320061`.
- `Analyze (python)` — PASSED; CodeQL workflow run `33194575297`, job
  `98928319847`.
- `CodeQL` — PASSED; run `98928559354`.

## GPU/service/resource evidence

- Service: hinton2 user unit `zap-it-lan.service` remained enabled and active;
  readiness returned HTTP 200 with the existing ready strategy.
- Listener/process: exactly one `10.8.132.76:17891` listener, MainPID `388703`,
  `NRestarts=0`; no restart was performed during this continuation.
- Assigned GPU only: physical index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, `24576 MiB`, driver `610.43.02`; the service sees it as
  logical `cuda:0`. The compute snapshot contained only service PID `388703`
  on that assigned UUID, using `13524 MiB` of the observed `13547 MiB`.
  No unassigned device or unrelated process was touched.
- Final GPU sample: `13547 MiB` used and `10577 MiB` free on the assigned card.
- `/dev/shm/slaif-zap-it` was empty; its root remained mode `0700`.
- The operator environment remained mode `0600`; the inference-key raw-value
  SHA-256 remained
  `cd7fb7f4189d1e5b0d759d09f718b309058724cb20294a40c296ce1fbb45cc51`.

## Documentation/provenance

No documentation change or runtime dependency change was necessary. The
implementation is confined to mechanized CPU evidence and the immutable OAP
transcript. The prior 012-a live service/GPU evidence remains applicable
because production behavior and service state were not changed.

## Deferred human adjudication

- Critical register action: NONE

## Safety/scope confirmations

- Only Objective-012 PR #68 was amended; no adjacent Objective 013/014 work was
  started, no new PR was created, and no merge or auto-merge was performed.
- API renderer names/policy, final-stage authority, sanitization, confidence
  format, palette, artifacts, limits, authentication, binding, model/runtime
  policy, and legacy `annotated` behavior were not changed.
- No model download, model lifecycle change, second process, key rotation,
  network/firewall/TLS change, public exposure, persistent request data, or
  protected-resource mutation occurred. Raw image/config contents and bearer
  credentials were not placed in the report or OAP evidence.

## Limitations/blockers

- The canonical CPU suite honestly skipped its opt-in live GPU test; no live
  inference phase was required because the production renderer was unchanged.
- Build emitted existing setuptools deprecation warnings for the project's
  license metadata; the build and all artifact audits passed.

## Factual strategic follow-up

PR #68 is pushed at implementation head plus this report-only SELF child and
remains open and mergeable. Strategic review/acceptance and any merge decision
remain outside coding authority.
