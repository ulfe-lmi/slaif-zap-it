# OAP Work Order — 016-a — Bounded raw SAM2 visualizations

## Objective

Make verbosity-3 SAM2 debugging truthful and auditable through bounded,
deterministic API artifacts. Replace the service's misleading per-candidate
rectangular crop debug view with paginated candidate contact sheets that show
each exact raw mask separately and label it with a stable candidate ID,
predicted IoU and stability score. Add union coverage, overlap-count heatmap and
uncovered-pixel images, plus typed manifest facts that explain representation,
coverage, overlap and truncation.

This is a new Objective-016 branch and PR. It consumes the request-local SAM2
configuration foundation merged in Objective 015. Do not add or claim the
aerial-solar fixture, reference polygons, quality-profile accuracy thresholds or
panel recall/precision acceptance reserved for Objective 017.

## Verified starting state

- Remote `main` is merge commit
  `8081152403657f5e737ab0b491e0b89f587209e1`, whose second parent is exact
  Objective-015 report head `1fff2ce908a32b37bc9cb0c09104ec93b94e96cc`.
  Objective-015 PR #71 is merged. Post-merge CI run `33220971751` and CodeQL
  run `33220971762` are successful. GitHub has no open PR.
- Create branch `oap/016-a-bounded-raw-sam2-visualizations` from exact remote
  main and exactly one PR titled `Objective 016: bounded raw SAM2
  visualizations` against `main`. Preserve the atomically published 016 order
  and active selector while branching; do not amend/replay PR #71.
- Objective 015 now keeps the approved SAM2 model resident and constructs a
  fresh request-local generator. `service.sam2.actual_candidate_count` is the
  raw generator count, while `candidate_counts.sam2_candidates` is the
  non-empty post-remap count. The live A/B/A qualification proved 8/7/8 raw
  candidates with one model initialization, and crop-0/crop-1 produced 25/62.
- In `src/core/engine.py`, non-empty raw candidates are remapped to original
  coordinates as `all_masks_pre`, retaining `predicted_iou`,
  `stability_score`, and zero-based `_source_index`. Today API-safe
  `mask_generator.debug: true` loops over these candidates and emits bounding
  rectangle image crops named from `frame_id`; it does not show the exact mask,
  candidate ID, score, union, overlap count or uncovered pixels. Overlapping
  masks cannot be understood from the ordinary combined stage overlay.
- The existing service validator accepts only boolean `mask_generator.debug`;
  it forces it false below verbosity 3. The request cannot select paths or
  artifact destinations. `BoundedMemoryArtifactSink`, response artifact limits,
  raw-byte limits, final encoded-size checks and fixed API-safe naming already
  exist. The trusted legacy CLI uses a filesystem sink and its historical
  per-patch filenames/output must remain compatible.
- CPU environment currently has Pillow 12.3.0 and NumPy 2.5.2; live GPU
  environment has Pillow 10.4.0 and NumPy 1.26.4. Determinism claims therefore
  apply to identical inputs in the same pinned environment, not byte identity
  across arbitrary encoder/library versions.
- Live private-LAN service `zap-it-lan.service` is enabled, active, ready at
  exact `10.8.132.76:17891`, MainPID `449821`, `NRestarts=0`, with one listener,
  one visible CUDA device and an empty request workspace. The mode-0600
  environment digest is
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`;
  never print, log, commit or report either credential value.
- Host `hinton2`; explicitly assigned physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`. PyTorch is 2.5.1+cu124,
  CUDA build 12.4, and the process sees only logical `cuda:0`. PID `449821` is
  the sole compute process, using approximately 10,084 MiB. `/dev/shm` is a
  12-GiB tmpfs with approximately 12 GiB free; `/dev/shm/slaif-zap-it` is mode
  0700 and empty.
- `CRITICAL.md` is unchanged at the merged SHA. CRIT-0001 has a human
  `ACCEPTED` disposition. This bounded visualization design presents no new
  five-condition deferred-adjudication dilemma.

## Required architecture

### 1. Pure deterministic raw-candidate renderer

Add a small pure NumPy/Pillow rendering component under `src/core` and invoke it
only from the API-safe SAM2 debug path. It receives the original RGB image and
the complete non-empty `all_masks_pre` sequence in generator/source order. It
must not import SAM2, Detectron2, CLIP or BLIP3, mutate masks/image/candidates,
perform model work, access files/network/environment, or receive any prompt,
label, filename or client path.

For public display, define `candidate_id = _source_index + 1`. IDs are therefore
one-based, stable generator-order identifiers and may contain gaps only when a
raw candidate was empty and omitted before remapping. Never renumber the
remaining candidates densely. Validate mask/image shape agreement and accept
disconnected/border-touching masks.

Candidate colors must be selected by a fixed arithmetic palette keyed only by
`candidate_id`. Rendering order, crop padding, resizing, layout, text formatting
and encoding inputs must be deterministic. Equal renderer inputs in one pinned
environment must produce equal arrays and PNG hashes. Timing/request IDs are not
part of this determinism claim.

### 2. Paginated separate-candidate contact sheets

Choose contact sheets instead of one artifact per candidate. This avoids
overwriting overlap evidence and keeps artifact count bounded. Fix and expose
these versioned rendering constants:

```text
columns = 3
rows = 4
candidates_per_sheet = 12
maximum_contact_sheets = 8
maximum_represented_candidates = 96
tile_content_width = 320
tile_content_height = 240
tile_label_height = 28
mask_alpha = 0.45
```

For every represented candidate, make one independent tile:

1. compute the exact mask bounding box;
2. add deterministic context padding equal to 10 percent of the larger bbox
   dimension, with a minimum of 4 source pixels, clamped to image boundaries;
3. crop RGB and mask with identical coordinates;
4. resize RGB bilinearly and mask with nearest-neighbor into a letterboxed
   320x240 content area without changing aspect ratio;
5. alpha-overlay only the exact mask using its deterministic color; and
6. draw one bounded label bar wholly inside the tile.

The label format is exactly:

```text
C0001  IoU 0.843  stability 0.912
```

Use three decimals for finite numeric scores and `n/a` for absent or non-finite
scores. The text is constructed solely from the numeric candidate ID and quality
values; no user-controlled text is rendered. Labels must stay within the tile,
be clipped/truncated defensively if needed, and never overlap another tile's
label. Empty page cells use a fixed neutral fill.

Represent candidates in ascending source index. Emit only these fixed API names:

```text
sam2-candidates-page-0001.png
...
sam2-candidates-page-0008.png
```

Never include `frame_id`, input basename, profile, prompt, CLIP/BLIP label,
question or request-controlled identifier in an API artifact name. If more than
96 non-empty candidates exist, render the first 96 in generator order, record
the exact truncation count and add one fixed aggregate warning. Do not silently
drop candidates or create a ninth page.

### 3. Coverage, overlap and uncovered diagnostics

Compute coverage statistics from every non-empty original-resolution raw mask,
not merely the first 96 represented in sheets. Use bounded accumulation that
cannot overflow at the Objective-015 maximum estimated prediction count (an
unsigned 32-bit count canvas is acceptable). Produce:

- `sam2-union-coverage.png`: black for uncovered pixels and white for pixels
  covered by at least one candidate;
- `sam2-overlap-heatmap.png`: black at zero coverage and a documented fixed
  deterministic color ramp for positive overlap counts; scaling may use the
  request's observed maximum, which is reported numerically; and
- `sam2-uncovered-pixels.png`: white for uncovered pixels and black for covered
  pixels, the exact binary inverse of the union visualization before resizing.

All three names are fixed. Preserve original aspect ratio, never upscale, and
downscale with nearest-neighbor only when needed to stay at or below 2,000,000
output pixels. Report original image dimensions and diagnostic output dimensions
so the visualization does not masquerade as an exact-size mask when downscaled.
Exact full-resolution numeric counts must be computed before resizing:
`covered_pixel_count`, `uncovered_pixel_count`, `max_overlap_count`, and an
overlap histogram. Bound that histogram exactly: include string keys for every
count from zero through `min(max_overlap_count, 255)`, add
`overlap_histogram_overflow_pixel_count` as the exact number of pixels whose
count is 256 or greater, and add `overlap_histogram_truncated` equal to whether
`max_overlap_count > 255`. Thus ordinary outputs retain a complete distribution
while adversarially deep overlaps cannot make manifest cardinality unbounded.

The union and uncovered arrays must be logical inverses at source resolution;
covered plus uncovered must equal `image_width * image_height`. The overlap
canvas must equal the per-pixel sum of all non-empty masks, including candidates
not shown in contact sheets.

### 4. Typed manifest and capabilities contract

At verbosity 3 with `mask_generator.debug: true`, add a typed optional
`raw_visualization` child to `service.sam2` in JSON and ZIP `manifest.json` with
at least:

```json
{
  "enabled": true,
  "candidate_id_base": 1,
  "raw_candidate_count": 123,
  "visualizable_candidate_count": 121,
  "omitted_empty_candidate_count": 2,
  "represented_candidate_count": 96,
  "represented_candidate_ids": [1, 2],
  "truncated_candidate_count": 25,
  "contact_sheet_count": 8,
  "covered_pixel_count": 1000,
  "uncovered_pixel_count": 500,
  "max_overlap_count": 7,
  "overlap_histogram": {"0": 500, "1": 700},
  "overlap_histogram_overflow_pixel_count": 0,
  "overlap_histogram_truncated": false,
  "source_dimensions": {"width": 50, "height": 30},
  "diagnostic_dimensions": {"width": 50, "height": 30}
}
```

The real represented-ID list contains every rendered ID and is bounded at 96.
Require all counts to be non-negative and cross-consistent. The raw candidate
count must agree with the existing `actual_candidate_count`; visualizable plus
omitted-empty must equal raw count; represented plus truncated must equal
visualizable; sheet count must equal ceiling(represented/12); histogram plus
overflow pixels, covered/uncovered and source area must reconcile.

The child must be absent when debug is false or verbosity is below 3. Preserve
the existing `service.sam2` configuration/provenance fields at all levels.
JSON artifact descriptors and ZIP members must carry exact names, media types,
sizes and SHA-256 values. No new schema version is required because this is an
optional additive L3 diagnostic under `zap-it.v1`.

Extend authenticated `/v1/capabilities` with a typed static raw-SAM2-debug
policy describing the trigger, fixed artifact names, candidate ID base, layout,
96-candidate/8-sheet/2,000,000-pixel limits, deterministic crop/score semantics
and truncation behavior. It must remain static, authenticated, path/secret/GPU-
topology-free and must not acquire readiness or the inference gate.

### 5. API/legacy routing and bounded resource admission

Only the service-safe path (`service_safe_artifact_names=True`) uses these new
fixed PNG diagnostics. Preserve trusted legacy CLI/batch
`mask_generator.debug` behavior and its historical per-candidate rectangular
patch names/format unless a compatibility test proves an existing behavior was
already different. Do not route API requests through filesystem sinks.

Continue stripping debug below L3. At L3, `debug: true` is the sole request
switch; add no client-controlled page, size, color, font, path or destination
field.

Before readiness, inference-gate acquisition or GPU work, preflight the fixed
worst-case raw bytes and artifact count for these diagnostics using decoded
image dimensions, the constants above, configured response/debug artifact
limits, configured per-artifact limit, configured total raw-artifact budget,
the identity artifact and configured visualization streams. A SAM2-debug-only
request that cannot hold the maximum 8 sheets plus 3 diagnostics must fail with
structured `response_too_large`; it must not run SAM2 and must not silently
reduce the documented 96-candidate capacity because of operator budgets.

The generated artifacts must still pass the existing sink checks, PNG encoded
per-artifact/total checks, JSON/base64/ZIP response-size checks and deadline.
Map sink/render/encoding budget failures to existing sanitized service errors;
never leak shapes, paths, YAML, content, model internals or exception strings.
The fixed renderer maximum is 11 new artifacts and 42,698,880 raw RGB bytes:
eight `960 x 1072 x 3` sheets plus three diagnostics of at most
`2,000,000 x 3` bytes each. Preflight and tests must use the exact implemented
formula and must not rely on a rounded estimate.

Do not weaken `max_response_artifacts=64`, `max_debug_artifacts=48`,
`max_single_artifact_bytes=32 MiB`, `max_total_raw_artifact_bytes=128 MiB` or
`max_response_bytes=256 MiB` defaults. Do not change Objective-015 SAM2 request
caps or model residency.

## Required CPU/API tests

1. Pure-renderer tests with small generated images/masks prove deterministic
   colors, source-index-derived one-based IDs, exact crop alignment, border and
   disconnected masks, no input mutation, and repeated equal arrays/PNG hashes.
2. Instrumented drawing tests prove the exact label text for finite, absent and
   non-finite scores; labels use no user-controlled text and remain inside each
   tile. Two overlapping raw candidates must occupy distinct tile regions so
   neither overwrites the other.
3. Pagination tests cover 0, 1, 12, 13, 96 and more-than-96 non-empty candidates,
   exact page/name/ID ordering, empty cells, 8-page ceiling, represented and
   truncated counts, and one aggregate truncation warning.
4. Exact small-mask tests prove union, source overlap counts, maximum overlap,
   complete histogram, uncovered inverse, full pixel accounting, and that masks
   beyond candidate 96 still affect coverage/overlap diagnostics.
5. Large-aspect and >2,000,000-pixel tests prove deterministic nearest-neighbor
   downscaling, no upscaling, aspect preservation and accurate source/diagnostic
   dimension disclosure without allocating unbounded intermediates beyond the
   required source overlap canvas.
6. Engine/sink compatibility tests prove API-safe fixed PNG names and new
   summaries while the legacy CLI path retains its existing rectangular JPEG
   patch names. Frame names, labels, prompts and questions must not affect API
   artifact paths.
7. API tests prove the child/artifacts appear only for L3 plus debug true;
   lower levels strip debug; debug false is unchanged; JSON and ZIP manifest
   facts/names/hash/size/media type match encoded bytes; schemas/OpenAPI and
   authenticated capabilities describe the exact static policy.
8. Resource tests prove insufficient artifact count, per-artifact bytes, total
   raw bytes and final encoded response limits return structured bounded errors.
   The predictable SAM2 raw-debug insufficiency must be rejected before
   readiness/gate/engine calls. Existing limits must continue to govern other
   artifacts.
9. Preserve/run existing SAM2 request-local A/B/A, capabilities, labelled
   renderer, BLIP3 mask verification, post-filter diagnostics, hostile YAML,
   lifecycle, timeout/cancel, artifact, auth, metrics, packaging, legacy CLI and
   model-residency tests.

Run and report the canonical CPU suite with coverage, focused raw-renderer/core/
service/API/schema/resource/legacy tests, Ruff format/check, compileall,
documentation checker, affected shell/systemd validation, wheel/sdist build,
release-artifact and tracked-tree secret scans, `twine check`, and `git diff
--check`. Public CI must remain CPU/offline and download no models. Every
required current-head CI and CodeQL check must be present and successful.

## Bounded live private-LAN qualification

Keep the existing service enabled, active and ready during ordinary
implementation. After the implementation head and CPU/static checks pass, one
controlled restart of only `zap-it-lan.service` is authorized to activate the
new service code. Before restart recheck all GPU index/UUID/PCI/name/VRAM/process
facts, driver/CUDA/PyTorch, `/dev/shm`, listener/unit ownership, port, environment
mode/digest and free capacity without printing or reporting a key. Start no
second model process; touch no driver, firewall, route, VPN, unrelated unit or
unassigned device.

After readiness returns:

1. Require health/readiness 200, missing/wrong inference credentials 401,
   authenticated capabilities/metrics 200, and docs/OpenAPI 404. Capabilities
   must match the ordered raw-debug constants without disclosing host facts.
2. Use one already-authorized ignored local fixture, cropped/resized in memory
   to a bounded input. Do not copy its bytes/YAML/prompts into Git, OAP or logs.
   Use verbosity 3, ZIP response, `mask_generator.debug: true`, a safe accepted
   SAM2 configuration, and no CLIP, BLIP3, stage visualization or other debug
   flag so the artifact evidence is isolated.
3. Require successful fixed-name contact sheet(s), union, overlap and uncovered
   PNG members. Verify manifest/member media type, byte size and SHA-256; decode
   every PNG; require dimensions within constants, a non-empty represented-ID
   list, score-label/contact-sheet content that is not a blank canvas, and exact
   manifest arithmetic. Do not print or preserve image content.
4. Repeat the exact request on the stable process and require identical raw
   visualization metadata except timing and identical PNG hashes/bytes. The
   service PID, one registry initialization, one listener and sole assigned-GPU
   process must remain stable. If SAM2 itself produces a different raw proposal
   set, disclose it as a failed determinism qualification rather than weakening
   acceptance.
5. Send a deliberately resource-insufficient request using an isolated
   CPU/FastAPI test configuration, not by weakening the live operator service
   environment. The live unit's operator limits/env file must remain unchanged.
6. Record bounded request latency, ZIP size, candidate/page/artifact counts,
   coverage/overlap aggregates, host RSS and assigned-GPU peak/free memory.
   Require `NRestarts=0`, empty request workspace, sanitized journal, unchanged
   mode-0600 environment digest, one listener, and leave the unit enabled,
   active and ready at `10.8.132.76:17891`.

Disclose any failed request, readiness delay, timeout, nondeterministic proposal
or artifact, score/ID mismatch, budget failure, residue, unexpected restart or
GPU/process change. A combined overlay alone, silent candidate truncation,
manifest/hash mismatch, use of user text in artifact names, missing uncovered/
overlap evidence, or service instability is not acceptance.

## Documentation and provenance

Update README, API, CONFIG, CORE, ALGORITHMS, OUTPUT-PARITY, TESTING, runtime,
runbook, service datasheet and capability/schema documentation as applicable.
State exactly:

- why a combined overlap overlay is insufficient;
- L3 plus `mask_generator.debug: true` trigger;
- one-based source-order candidate IDs and score format;
- contact-sheet layout/crop/pagination/truncation limits;
- fixed safe artifact names;
- union/overlap/uncovered color and downscale semantics;
- exact full-resolution numeric accounting and optional manifest child;
- deterministic scope and encoder-version limitation;
- preflight and final response budgets; and
- unchanged lower levels and legacy CLI behavior.

Do not claim that contact sheets are segmentation quality validation or that the
quality profile covers every solar array. Do not change model identities,
revisions, licenses, network/auth/key policy, GPU selection, cache/offline mode,
dtype, residency, CLIP/BLIP3 behavior, response limit defaults or accepted
CRIT-0001 disposition.

## Non-goals

- no aerial solar image, reference polygon or accuracy benchmark;
- no change to SAM2 generation/filtering, profiles, request scalars, operator
  prompt caps, candidate ordering, CLIP, BLIP3, post-filtering, final labels,
  labelled renderer, YOLO or identity-mask semantics;
- no request-selected raw-debug layout, size, palette, text, path, destination,
  model, device or artifact name;
- no persistent request data, filesystem API output, new worker/process,
  public/WAN bind, TLS/gateway, firewall/VPN/network mutation, key rotation or
  disclosure, release/tag/upload or unrelated cleanup;
- no cross-Pillow-version byte-determinism claim.

## Acceptance and report contract

Acceptance requires every requirement above: separate bounded candidate tiles
with stable IDs and scores, exact all-candidate coverage/overlap/uncovered facts,
fixed API-safe names, honest pagination/truncation, proactive resource
admission, typed JSON/ZIP/capability contracts, unchanged lower levels/legacy
CLI, green CPU/current-head CI and CodeQL, and a successful repeated live L3
qualification on the exact assigned RTX 3090.

The strongest reason not to accept is that generating many full-image mask
visualizations could turn a diagnostic request into an unbounded memory/
response amplification path while still hiding overlap through overwrite.
Answer it with separate cropped tiles, a fixed 96-candidate/8-page ceiling,
three bounded diagnostic images, pre-inference worst-case admission, existing
sink/encoded-response limits, explicit truncation, and exact numeric accounting
over all candidates including those not shown in sheets.

Push all implementation and exact active/order bytes before reporting. Record a
literal 40-hex implementation SHA. Then create exactly
`oap/reports/016-a-report.md`, commit only that report as the final report-only
SELF child, push, verify remote parent/one-path topology and bytes, send exactly
one response FIFO `OK`, perform no later mutation, and exit. Coding never
merges.

## Deferred human adjudication

- Decision: NONE
