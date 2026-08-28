# OAP Work Order — 013-a — Mask-aware BLIP3 verification

## Objective

Replace BLIP3's rectangle-only candidate crop with a deterministic bounded
paired verification image: an unmodified context view on the left and an
aligned mask spotlight on the right. BLIP3 must judge the pixels selected by
the exact SAM2 mask, not merely whether the requested object appears somewhere
inside the bounding crop.

The right view must preserve every selected RGB sample, dim only the exterior,
and mark the exact exterior mask boundary with a thin yellow contour. Combine
the bounded client target question with a fixed final region-specific
instruction. At service verbosity 3, an existing per-rule `debug: true` option
must return the exact image passed to BLIP3 under a fixed safe lossless artifact
name.

This is one new Objective-013 branch and PR. Do not begin post-filter rejection
diagnostics; those remain Objective 014.

## Verified starting state

- Remote `main` is `43fcfe99b47545b218a70338f02c01f69b35a29e`, the merge of
  Objective-012 PR #68. Its post-merge CI run `33195032350` and CodeQL run
  `33195032325` are successful. GitHub has no open PR.
- Required branch: `oap/013-a-mask-aware-blip3-verification`; create exactly one
  PR titled `Objective 013: mask-aware BLIP3 verification` against `main`.
- Coding checkout is on the merged Objective-012 branch with only the atomically
  published next order/active transcript uncommitted. Fetch and branch from
  exact remote main while preserving those bytes; do not replay or amend PR #68.
- `modules/verifier/blip3.py` currently computes the candidate mask bbox,
  expands dimensions to at least 128 pixels, and passes an ordinary RGB crop to
  `qa.answer`. The segmentation mask is not cropped or shown. This permits a
  target elsewhere in the rectangle to cause a false approval.
- Existing service policy permits bounded BLIP3 rules with `question`,
  `trueresult`, `falseresult`, `newcategory`, and boolean `debug`; it fixes at
  most 32 executed questions and 32 generated tokens per question. YAML scalars
  remain bounded to 16,384 characters. Uploaded YAML cannot select models,
  devices, paths, filenames, fonts, network or code.
- The current engine does not pass its bounded artifact sink into BLIP3 even
  when a BLIP3 rule requests debug. Nested BLIP3 debug flags also are not
  stripped below verbosity 3. The legacy BLIP3 helper's debug names contain a
  label/rule fragment and write a JPEG crop plus answer text; this is not the
  required exact, fixed-name audit image.
- Objective 012 added final-object `annotated-labelled` output and is preserved.
  The new verifier acts earlier, inside BLIP3, and must not infer final instance
  IDs or change visualization semantics.
- Persistent `zap-it-lan.service` is enabled, active and ready at exact
  `10.8.132.76:17891`, MainPID `388703`, `NRestarts=0`, with one listener. Its
  mode-0600 operator environment contains the unchanged inference key; never
  print, rotate, commit, log or report the key.
- Host `hinton2`; assigned physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`. The service is the only
  compute process and used 13,524 MiB in the strategic snapshot. The application
  sees only logical `cuda:0`.
- `/dev/shm` is an approximately 12-GiB tmpfs with the private service root
  empty. No second model process or new listener is needed. The existing local,
  ignored goat fixture/config is operator-authorized for real qualification and
  must remain outside Git/OAP/chat; no operator roof/panel photograph was found
  in the repository or strategic workspace, so the exact roof-with-panels
  hard-negative is a programmatic CPU regression and must not be misreported as
  a real-photograph model benchmark.

## Requirements

### 1. Deterministic crop and scale transform

1. Add a focused reusable composer in `modules/verifier/blip3.py` (or one narrow
   adjacent module) which accepts one RGB `uint8` image and the exact boolean
   segmentation mask. Reject shape/type mismatches and empty masks explicitly;
   do not silently switch to bbox-only input.
2. Derive the inclusive mask bbox over all connected components. Add symmetric
   context padding of `max(16, ceil(0.125 * max(mask_bbox_width,
   mask_bbox_height)))` pixels, and make each desired crop dimension at least
   128 pixels. Center, clamp and back-shift the crop so image and mask use the
   exact same `[y0:y1, x0:x1]` coordinates and the desired extent is retained
   whenever the source boundary permits it.
3. Preserve all disconnected components inside the single context crop. Handle
   masks/crops touching every source boundary and a mask spanning the source
   without negative, wrapped or off-by-one coordinates.
4. Bound the scaled context view to a maximum long side of 768 pixels. For
   smaller crops, scale uniformly toward a 256-pixel short side unless the
   768-pixel long-side cap intervenes. Do not distort aspect ratio. Use one
   explicit deterministic nearest-neighbor mapping for both RGB and mask so
   every displayed RGB sample is an exact source value and the mask remains
   aligned. Document the unavoidable possibility that extreme downsampling can
   coalesce subpixel structure; do not claim interpolation-free one-to-one
   source resolution after scaling.
5. Produce positive nonzero output dimensions deterministically and expose
   enough private transform metadata for tests to verify crop coordinates,
   scaled size and left/right alignment without OCR or visual guesswork.

### 2. Paired spotlight image

1. Build one RGB array with the scaled untouched context on the left, a fixed
   four-pixel dark divider, and the aligned spotlight view on the right. The
   paired dimensions are exactly `(scaled_height, 2 * scaled_width + 4, 3)`.
2. In the right view, every pixel selected by the scaled mask must be byte-for-
   byte identical to the corresponding left-view RGB pixel.
3. For right-view pixels outside the selected mask and outside the contour,
   apply exactly 60% darkening with a deterministic integer rule (retain 40% of
   each channel). Do not alter the left view. Desaturation is optional and is
   omitted unless a test-backed need is demonstrated.
4. Compute the contour from the exact scaled mask, not its bbox. Use a four-
   pixel exterior dilation ring and fixed yellow RGB `(255, 224, 0)`. Paint the
   contour only on exterior pixels so no opaque color overwrites the selected
   object. Every disconnected component must be outlined. At a source/image
   boundary, preserve selected pixels rather than drawing an interior contour.
5. Use bounded NumPy/Pillow operations with no per-source-pixel Python object
   graph, filesystem dependency, random state, remote font/model resource, or
   request-controlled transform parameter. Repeated equal inputs must produce
   identical paired arrays and encoded PNG bytes.

### 3. Fixed verification instruction and answer behavior

1. Define one module constant for the fixed instruction. The composed query
   must identify the left view as untouched context and the right view as the
   exact candidate, include the client-supplied target question verbatim inside
   a clearly delimited target section, and end with this semantic instruction:

   `In the right-hand image, is the region inside the yellow outline itself the requested object? Ignore everything outside the outline. Answer exactly Yes or No.`

2. Put the fixed instruction after the client target so client text cannot
   accidentally displace the region-specific task. Do not log, persist or use
   the composed query in any filename, metric label or artifact ID.
3. Pass the exact paired `PIL.Image` and composed query to every executed
   `qa.answer` call, including both `any,<threshold>` and label-specific rules.
   Build/cache the paired image once per candidate and reuse it if more than one
   rule is actually asked.
4. Preserve existing deterministic generation, `trueresult`/`falseresult`/
   `newcategory` interpretation, structured `blip3_answer`, request-local rules,
   fixed holder, model identity/revision/dtype/device, maximum 32 executed
   questions and maximum 32 generated tokens. The constant instruction adds
   only a fixed bounded overhead to the existing bounded client scalar.

### 4. Exact verbosity-3 audit artifact and safe naming

1. Wire the core's existing bounded artifact sink into BLIP3 when and only when
   an effective BLIP3 rule has `debug: true`. Below verbosity 3, strip all nested
   BLIP3 debug flags to `false` and add one bounded aggregate warning without
   echoing a rule name, question or label.
2. For each executed service question whose rule has effective `debug: true`,
   store the exact paired array passed to `qa.answer` as a lossless PNG image
   artifact. Use the fixed service logical name
   `blip3-verification-{candidate_index:04d}-{question_index:04d}.png`, where
   both indices are deterministic zero-based numeric ordinals. Do not include
   request filenames, prompts, labels, rule names, answers or arbitrary text.
3. Do not create a duplicate answer-text debug artifact in the service: the
   bounded answer is already structured metadata, and image-only debug keeps 32
   questions within the 48-artifact count ceiling before other requested debug
   streams are considered. Existing total/per-artifact/response/deadline checks
   still fail closed when combined debug output exceeds its budget.
4. Preserve trusted CLI debug capability. If legacy filesystem answer files are
   retained, remove prompt/label/rule text from their basenames and use only the
   trusted frame stem plus numeric candidate/question ordinals. Never let a
   label, question or answer become a path component in either API or CLI mode.
5. The JSON base64 artifact and ZIP member must carry the same PNG bytes, size
   and SHA-256 descriptor. Debug arrays remain request-scoped in RAM or the
   existing CLI sink; no service request data persists and `/dev/shm` cleanup
   remains exact.

### 5. CPU and integration tests

Add focused tests plus the complete canonical suite. Required mechanized cases:

1. **Correct-mask positive:** a programmatic roof scene has photovoltaic-panel
   pixels selected by the candidate mask while roof/vegetation remain outside.
   An injected QA that judges only the selected spotlight region receives the
   exact paired image and fixed query, returns `Yes`, and the final label remains
   `solar_panel`.
2. **Same-crop hard negative:** use the same scene/context with panels visibly
   elsewhere inside the crop but a candidate mask selecting roof tiles. The
   injected QA receives the spotlight image, returns `No`, and the candidate's
   final label becomes `negative`. Prove the test would distinguish this from a
   bbox-level presence check rather than merely returning a preprogrammed
   sequence.
3. Exact crop-coordinate and mask alignment tests for ordinary, top/bottom/
   left/right-border and corner masks; scale-up and scale-down mappings; a mask
   spanning the image; and multiple disconnected components.
4. Pixel tests prove left pixels are exact nearest-neighbor source samples,
   right selected pixels exactly equal the left, non-contour exterior pixels
   follow the 40% channel rule, the yellow contour is exactly the exterior
   four-pixel dilation ring rather than a bbox, and no selected pixel is painted.
5. Repeated composer/filter runs and service PNG encoding produce identical
   arrays, bytes and SHA-256. The QA object records the exact image/query it
   received; the debug artifact bytes decode to that same array.
6. Safe fixed debug names for multiple candidates/questions; labels, questions,
   answers and hostile path separators cannot affect names. Debug is absent at
   L0-L2, accepted at L3, bounded by sink count/single/total limits, and leaves
   no residue on failure.
7. `any,<threshold>` and label rules both use paired input and composed queries;
   true/false/newcategory behavior remains compatible. Existing 32-question
   preflight and 32-token generation cap remain enforced, including failure
   before QA on 33 planned questions.
8. Existing no-BLIP3, dry-run, adaptive residency, cancellation/restoration,
   labelled visualization, JSON/ZIP, hostile-YAML and CLI tests remain green.
9. Update assertions/docs that currently describe an ordinary crop, JPEG debug
   patch or service BLIP3 answer-text artifact; do not leave contradictory
   current-truth claims.

Run and report the canonical CPU suite with coverage, focused BLIP3/composer/
core/API/artifact tests, Ruff format/check, compileall, documentation checker,
shell syntax where changed, wheel/sdist build, artifact audit, tracked-tree and
built-artifact secret scans, `twine check`, and `git diff --check`. Public CI
must not use CUDA or download models. All required current-head CI and CodeQL
checks must be present and successful.

### 6. Bounded live private-LAN qualification

Keep the current service enabled and active during ordinary implementation.
After the implementation head is committed and CPU/static checks pass, one
controlled restart of only `zap-it-lan.service` is authorized so the running
process uses Objective-013 code. Before restart independently re-verify exact
assigned physical index/UUID/PCI/name/VRAM/process ownership, unit/listener,
`/dev/shm`, and key-file digest without reading or reporting the key. Do not
start a second model process or touch drivers, firewall, routes, VPN or any
unrelated unit.

After readiness returns:

1. Prove missing/wrong inference keys still return 401, authenticated readiness/
   metrics work as before, and docs/OpenAPI remain 404.
2. Use the already-authorized local ignored goat fixture/config through one
   sanitized in-memory request mapping: disable unrelated debug streams, enable
   BLIP3 debug for one bounded rule, retain a supported BLIP3 profile, and
   optionally request Objective-012 `annotated-labelled`. Do not copy raw
   image/config/labels/questions/answers into Git, OAP, logs or chat.
3. Send real verbosity-3 JSON and ZIP requests plus one repeated JSON request.
   Require BLIP3 executed with at least one bounded answer and at least one
   `blip3-verification-####-####.png` artifact. Prove each audit artifact is RGB,
   has the exact paired-width/divider relationship, contains an unchanged left
   view and a materially altered right view with yellow contour pixels, has no
   user-derived path fragment, and agrees byte/hash/size between JSON and ZIP.
   Repeated equal requests must produce the same ordered verification-artifact
   names and PNG digests.
4. CPU/injected tests are the authoritative exact roof/panel positive and
   same-crop hard-negative regression. The live goat request qualifies the real
   pinned model/instruction/artifact integration; do not claim it is a solar-
   panel accuracy benchmark or that model accuracy improved beyond tested
   cases.
5. Prove one unchanged post-restart PID/listener across requests, only the
   assigned GPU process, bounded response/resource metrics, sanitized journal,
   empty request workspace, preserved mode-0600 environment and unchanged key
   digest. Leave the unit enabled, active, ready on `10.8.132.76:17891`.

Disclose every failed live attempt and corrective action. A skipped paired-image
live request, missing debug artifact, unsafe name, hash mismatch, service/GPU
drift or request residue is not acceptance.

## Documentation and provenance

Update `README.md`, `ARCHITECTURE.md`, `TESTING.md`, `docs/API.md`,
`docs/CONFIG.md`, `docs/ALGORITHMS.md`, `docs/CORE.md`,
`docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`, `docs/SERVICE-DATASHEET.md`, and any
other current document whose ordinary-crop/debug claims change. State the exact
crop/context/nearest-scale/spotlight/contour/instruction behavior, service
verbosity/debug naming and limits, the distinction between mask-level
verification and guaranteed semantic accuracy, and the fact that structured
answers/labels remain independent of debug artifacts.

Do not change approved model identity/revision/license/dtype/residency claims.
No new runtime dependency or remote asset is expected; if one becomes necessary,
stop and issue factual evidence rather than silently expanding supply-chain
scope.

## Non-goals

- no Objective 014 post-filter reason counts or schema fields;
- no labelled-renderer redesign, Detectron2/panoptic, geometry, video or dataset
  export;
- no claim of universal BLIP3 accuracy, no prompt training/fine-tuning, model
  change/download, sampling, token increase or request-selected transform;
- no second process/service, GPU sharing/MPS/MIG, public/WAN bind, TLS/gateway/
  firewall/VPN/network change, key rotation/disclosure, release/tag/upload or
  persistent service request data;
- no rewrite of prior immutable orders/reports and no merge by coding.

## Acceptance and report contract

Acceptance requires all requirements above, including direct proof that a roof
mask is rejected when panel pixels are elsewhere in the same context and that a
correct panel mask is retained; exact selected-pixel preservation; exterior
dimming; component-following contour; deterministic paired array/PNG; safe
fixed-name L3 audit artifacts; unchanged question/token/resource/security law;
green CPU/CI/CodeQL; and satisfactory real private-LAN pinned-model integration.

The strongest reason not to accept is that a side-by-side image could still
encourage crop-level presence answers, or its contour/resampling could silently
alter or misalign the selected object. Answer it with the fixed instruction
last, a right-side exterior-only contour, exact left/right selected-pixel
identity, transform metadata tests, the same-crop hard negative, and the exact
real image passed to the pinned QA exposed as a bounded audit artifact.

Push all implementation and exact active/order bytes before reporting. Record a
literal 40-hex implementation SHA. Then create exactly
`oap/reports/013-a-report.md`, commit only that report as the final `SELF` child,
push, verify remote parent/one-path topology and bytes, send exactly one response
FIFO `OK`, perform no later mutation, and exit. Coding never merges.

## Deferred human adjudication

- Decision: NONE
