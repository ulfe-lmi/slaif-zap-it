# Testing and verification

Use exact evidence statuses: `PASSED`, `FAILED`, `SKIPPED`, `NOT RUN`,
`BLOCKED`, `PENDING`, or `MISSING`. A skipped or unavailable check is not a
pass. Tests must exercise the behavior they claim rather than replacing it with
an unrelated mock.

## Canonical CPU checks

```bash
.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m compileall -q src modules scripts tests
.venv/bin/python -m build --wheel --sdist
```

## Tiers

- T0 static/package: build/import, Ruff format/check, typing as adopted, docs,
  OpenAPI/schema, secret/license checks.
- T1 CPU unit: config, preprocess/postprocess, geometry, ordering, YOLO, identity
  PNG, serializers and existing regressions; no CUDA/network/model download.
- T2 CPU API: fake engine, multipart, levels, JSON/ZIP bytes, errors, auth,
  limits, concurrency/cancel, `/dev/shm` cleanup.
- T2 model control: explicit/none settings, separate credentials, repository
  index schemas, fixed-name/body policy, lifecycle idempotency/retry, gate
  pause/drain, cancellation/timeout rollback, fake-holder cleanup, repeated
  cycles, metrics labels and OpenAPI routes.
- T3 GPU integration: explicit opt-in; physical GPU1 only; pinned models; one
  test at a time; before/after UUID/process/VRAM evidence; redistributable image.
- T4 local deployment: verified free scoped port, health/readiness, all levels,
  repeated calls, authentication and docs policy, no residue on the assigned
  GPU, rollback/restart.

GitHub-hosted CI runs T0–T2 only. GPU tests must be markers and skip honestly
without required host/model assets. A self-hosted GPU check is not required until
an explicit operational order creates and secures it.

## Required semantic checks

- exact YOLO class/order/normalized coordinates and empty output;
- uint16 PNG dimensions/dtype/0 background/IDs/object bijection;
- disconnected components share one object ID;
- overlap winner policy and full overlap-preserving masks;
- L3 uncompressed column-major RLE round-trip, ID association, overlap and run limits;
- ROI/resize maps to original image;
- absent optional stages do not fabricate fields;
- lower verbosity does not execute extra expensive stages;
- YAML rejects unsafe/path/device/model fields and resource attacks;
- SAM2 request-local generator construction forwards only the 14 total safe
  generator scalars, including `use_m2m`,
  uses fixed binary-mask controls, and proves model identity/state isolation;
- SAM2 strict types/ranges, profile precedence, exact crop prompt estimates,
  operator caps, equality-at-cap acceptance and deterministic 80% warnings;
- authenticated `/v1/capabilities` is static and gate/readiness independent;
- `service.sam2` is present at L0-L3 and JSON/ZIP raw candidate/timing metadata
  agrees;
- raw-SAM2 renderer determinism, source-indexed IDs, exact score labels,
  independent contact-sheet pagination, border/disconnected masks, all-candidate
  union/overlap/uncovered accounting, bounded histograms, nearest-neighbor
  diagnostic downscaling and source/destination dimensions;
- raw-SAM2 fixed names, typed L3 manifest arithmetic, JSON/ZIP hash parity,
  authenticated static capability policy, legacy rectangular JPEG compatibility,
  and pre-readiness rejection for fixed artifact count/per-item/total/response
  budget insufficiency;
- one-image-only and config-only multipart cardinality;
- no persistent files after success/error/cancel;
- model/request state isolation across repeated/concurrent calls;
- fixed-model `UNAVAILABLE|LOADING|READY|UNLOADING` transitions, load/unload
  idempotency, readiness/admission race, active drain and cold-memory proof;
- selected physical GPU visibility and no protected-GPU allocation in live tier;
- bounded artifact count/per-item/total/base64/ZIP budget and no-truncation checks;
- post-filter precedence, area-first segmentation short-circuiting with
  not-evaluated `0/0` maxsize bbox dimensions, inclusive later bbox dimensions,
  aggregate reconciliation, source-indexed numeric-only rejection records,
  256-record truncation and the programmatic two-wide-candidate roof regression;
- L3-only post-filter diagnostics, closed reason schema, L0-L2 omission and
  JSON/ZIP diagnostic parity;
- BLIP3 source-space crop coordinates, exact Euclidean support, disconnected
  component/hole preservation, exact source-pixel identity, exterior-only
  contour, Pillow Gaussian blur, fixed instruction ordering, same-candidate
  image reuse and candidate-local containment rejection;
- shared candidate-view builder exact Euclidean dilation, zero-fill isolation,
  hole/disconnected/border/tiny-mask behavior, deterministic floor intensity,
  bilinear RGB plus nearest-neighbor support reapplication, immutable inputs,
  CLIP processor byte identity, BLIP3 sole-image byte identity, fixed one-based
  source/question names, zero-based filtered indices, strict request-local
  configuration, capability disclosure and pre-model debug admission;
- BLIP3 L3 fixed `blip3-verification-CANDIDATE-####-QUESTION-####.png` names, exact QA/debug
  image parity, JSON/ZIP hash parity, and nested debug stripping at L0-L2;
- radius-512 local-window dilation resource regression with an independent
  brute-force oracle, and two-phase CLIP/BLIP3 debug admission with zero-call
  negative cases;
- visualization execution policy, final-object labelled pixel/placement/
  sanitization/confidence checks, geometry pre-inference rejection, metrics
  privacy/cardinality and A/B/A state-isolation checks;

The explicit lifecycle test tier remains CPU/fake-only unless an order grants a
live qualification. A live qualification must use the exact order-assigned
physical GPU, prove PID/listener continuity across cold-load-infer-drain-
unload-reload-infer-unload, and record Torch allocated/reserved cold memory
separately from the small persistent CUDA context.

## Objective 019 single-image BLIP3 acceptance evidence

The current generated-array evidence includes one exact 512x512 RGB uint8
high-contrast striped distractor inside a nonrectangular mask bbox. It asserts
source-space `M`/Euclidean `D` support, byte identity under `D`, Gaussian-blurred
scene context, clipped borders/corners, holes, disconnected components, exact
radius/contour formulas and deterministic PNG hashes. Independent generated
arrays derive source geometry and Pillow blur; no external photographs, model
downloads or CUDA are used. Accepted and containment-rejected crops are tested.

The semantic seams are literal CPU/fake captures: the real CLIP
`classify_single` path is exercised while a bounded processor records its
`images=` argument, and the BLIP3 QA holder receives one shared image;
lossless debug PNG decoding is compared with the exact model input. A focused core
flow carries one-based source candidate IDs and zero-based post-SAM2 filtered
indices across removal, CLIP, BLIP3, final area ordering, labelled
visualization, composition records, debug records, JSON objects and ZIP manifest
objects.

The service matrix validates effective candidate-view policy at verbosity L0,
L1, L2 and L3, including `applied` false/true behavior. One injected service
instance performs generated-image A/B/A requests varying both CLIP and BLIP3
context fractions; exact resident fake-holder identities remain stable while
the actual CLIP `initialize` and BLIP3 `_Blip3QA` construction seams are
guarded to record any forbidden reinitialization attempt. The observed attempt
list remains empty, while model inputs and effective metadata change for B and
restore exactly for A2. These are boundary/provenance tests only; they do not
measure semantic-model accuracy, recall or precision.

Every OAP order names exact commands and broad/focused scope. “All tests passed”
means the entire named set ran and passed. Local success never substitutes for a
required GitHub check.

## Governance mechanics

When OAP governance helpers change, test that finalized orders must contain
exactly one deferred-adjudication decision; `NONE` creates no register mutation;
`APPEND CRIT-NNNN` requires complete threshold/adversarial-reasoning fields;
`append_critical.py` rejects duplicate/malformed/unfinished entries and preserves
all existing bytes; `check_state.py` reports duplicate IDs and latest human
adjudication without treating `DEFERRED`, `REJECTED`, or `CHANGE REQUIRED` as
accepted. These tests are CPU-only and must not become an excuse to add routine
critical entries.
