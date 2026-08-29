# OAP Work Order — 017-a — Mask-isolated candidate views for CLIP and BLIP3

## Objective

Eliminate rectangular semantic leakage from every CLIP and BLIP3 model input.
Add one pure, deterministic, request-local mask-view builder shared by both
stages. The builder must decide which original RGB pixels remain visible from
the exact SAM2 binary mask and its bounded morphological dilation, never from a
bounding rectangle. Wire strict `candidate_views.clip` and
`candidate_views.blip3` API configuration, public effective configuration and
candidate identity, lossless exact-model-input debug artifacts, authenticated
capabilities, self-contained pixel-isolation tests and a detailed documentation
refresh.

This is one cohesive Objective-017 branch and PR. A partial merge that fixes
only CLIP or only BLIP3 is not acceptable: it would publish an API whose shared
security boundary is only half true. Keep the existing private-LAN service
enabled, active and ready on its current code throughout ordinary development,
CPU tests, review preparation and CI. Only after the implementation head and
all local CPU/static gates are green is one controlled restart authorized for
live qualification. Leave the newest accepted service enabled, active and ready
after testing.

## GitHub state

- Numeric objective / round: `017-a`.
- Mode: `CREATE_NEW_PR`.
- Repository/default base and verified SHA:
  `ulfe-lmi/slaif-zap-it`, `main`,
  `645c8604f9c189e1367e6e27a4ce8298c109482a`.
- Required branch: `oap/017-a-mask-isolated-candidate-views`, created from that
  exact remote-main SHA after fetching.
- Required PR: exactly one non-draft PR titled
  `Objective 017: mask-isolated candidate views` against `main`.
- Existing PR/URL/current head: N/A. GitHub has no open PR. Objective-016 PR
  #72 is merged. Post-merge CI run `33226121554` and CodeQL run `33226121552`
  are successful on the exact base SHA.
- The local checkout currently points at the old Objective-016 report branch,
  but its tree is byte-identical to remote `main`. Do not branch from its commit
  identity: fetch and branch from the exact base SHA above. Do not amend or
  replay PR #72.

## Verified current state

- `modules/classifier/clip.py` currently computes the tight mask bbox, expands
  it by `clip.padding` (default 20), and sends the untouched rectangular RGB
  crop to `CLIPProcessor`. A distractor outside the mask but inside that
  rectangle therefore remains visible. Its service debug artifact is the same
  unmasked patch, uses a filtered-list position, and is JPEG.
- `modules/verifier/blip3.py` currently constructs a minimum-128-pixel padded
  rectangular crop. The left side of the pair is the untouched crop; the right
  side merely darkens pixels outside the candidate mask and outlines the mask.
  The fixed instruction explicitly calls the left side untouched context. This
  still lets BLIP3 classify an unrelated object outside the selected mask.
- The service has no `candidate_views` top-level configuration. The hostile
  validator currently admits `clip.padding` as a scalar unless it resembles a
  path, and it has no strict nested CLIP view policy. The response and
  capabilities expose no effective candidate-view values.
- The engine assigns zero-based `_source_index` before post-SAM2 filtering and
  preserves it internally through CLIP, BLIP3, final filtering and
  visualization. Public raw-SAM2 diagnostics already define the stable
  one-based `candidate_id = _source_index + 1`. Final `ObjectResult` retains
  `source_index`, but the API `ObjectRecord` omits it and no stable post-SAM2
  filtered-list index is retained. BLIP3 service debug names use the current
  list position rather than source identity.
- The current bounded artifact sink, 48-debug-artifact default, 64-response-
  artifact default, 32-MiB per-artifact default, 128-MiB total-raw-artifact
  default, 256-MiB response default, exact hash/size descriptors, deterministic
  ZIP manifest, L3-only debug stripping and 32-question BLIP3 limit remain the
  governing resource boundaries.
- The CPU package intentionally does not require OpenCV; the locked GPU
  environment does. The new shared core boundary must remain CPU/offline and
  testable with the declared base dependencies. Do not add SciPy, OpenCV or a
  model dependency merely to implement binary dilation. A bounded pure
  NumPy/Pillow implementation or another solution using already-declared base
  dependencies is required.
- Current live host `hinton2`: user unit `zap-it-lan.service` is enabled,
  active/running and ready, PID `498617`, `NRestarts=0`, one listener at exact
  `10.8.132.76:17891`, health/readiness 200, working directory the coding repo.
  Its mode-0600 environment file digest is
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`;
  never print, log, commit or report either credential value.
- The operator-assigned device is physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI
  `00000000:0B:00.0`, NVIDIA GeForce RTX 3090, 24,576 MiB, driver
  `610.43.02`. The process is the sole compute process and currently uses
  approximately 13,436 MiB. PyTorch is `2.5.1+cu124`, CUDA build 12.4, and
  only logical `cuda:0` is permitted. `/dev/shm` has approximately 11,453 MiB
  free; `/dev/shm/slaif-zap-it` is mode 0700 and empty.
- `CRITICAL.md` is unchanged on current main. CRIT-0001 has a human `ACCEPTED`
  disposition. This reversible in-memory isolation design introduces no new
  five-condition deferred-human-adjudication dilemma.

## Required architecture and bounded scope

### 1. Shared pure mask-view builder

Add one small component under `src/core` used by both CLIP and BLIP3. It must be
pure and request-local: no model imports, files, network, environment access,
global mutable cache or retained per-request arrays. It receives an RGB uint8
`H x W x 3` source image, a boolean `H x W` SAM2 mask, the stable source
candidate ID and an already validated stage view configuration. Reject empty,
non-boolean, mis-shaped or otherwise invalid inputs with bounded sanitized core
errors.

Return an immutable typed result containing at least:

- target-only RGB image;
- dilated-context RGB image;
- target mask `M` in the returned crop;
- dilated support mask `D` in the returned crop;
- half-open source-coordinate target bbox and context bbox;
- effective integer dilation radius;
- source candidate ID; and
- deterministic dimensions/configuration metadata sufficient to audit later
  resize/letterbox transforms without retaining source image data.

The returned RGB arrays are cropped only after masking and must own their memory
or otherwise be immutable from the caller's perspective. Do not mutate the
source image, source mask or candidate dictionary.

### 2. Exact dilation and visibility law

Let `M` be the full-source boolean mask. Compute its tight inclusive pixel bbox
and let `L = max(bbox_width, bbox_height)`. Compute:

```text
raw_radius = ceil(context_fraction * L)
effective_radius = min(max(raw_radius, min_context_pixels), max_context_pixels)
```

Construct `D` by Euclidean/circular or elliptical morphological dilation of
`M` by exactly `effective_radius` source pixels, clipped at source boundaries.
`effective_radius == 0` means `D == M`. Dilation operates on the exact mask,
preserves disconnected components and holes until the actual dilation reaches
them, and must never fill the mask bbox or introduce a rectangular bridge.
Calculate the half-open context crop from the tight bbox of `D`, only after `D`
exists.

For the target-only image, source RGB is retained exactly where `M` is true and
every pixel outside `M` is the configured fill. Strict zero mode means byte
value `[0,0,0]` in every channel. No contour or interpolation may modify an
inside-`M` source pixel.

For the context image, source RGB is retained byte-identically where `M` is
true. In `D minus M`, RGB is multiplied channel-wise by
`context_intensity`; choose and document one integer rounding rule and test it.
Every pixel outside `D` is the configured fill. An optional fixed yellow or cyan
contour may be drawn only in `D minus M`; it must never overwrite `M` and may
not create non-fill pixels outside `D`.

The crop rectangle is storage only. It never decides which source pixels remain
visible. These invariants are binding for every target and context output:

```text
target-only[outside M] == fill
context[outside D] == fill
target-only[inside M] == source[inside M]
context[inside M] == source[inside M]
```

### 3. Resizing and small candidates

Never enlarge a source-space bbox/crop to satisfy a model minimum. The only
valid sequence is: construct `M`, construct `D`, neutralize prohibited pixels,
crop tightly around `D`, then resize or letterbox those already-masked arrays.
Use nearest-neighbor for masks, bilinear or bicubic for RGB and the configured
fill for letterboxing. Reapply target/support masks after RGB resampling where
needed so interpolation cannot create non-fill pixels outside the resized
support or allow contour/context pixels to bleed into the resized target.

BLIP3 may retain its bounded target-short-side 256 / maximum-long-side 768
model-input policy, but the old minimum 128 source-space crop and untouched
context crop must disappear. CLIP may rely on its pinned processor's normal
resize after receiving the already-masked input. Tests must capture the exact
preprocessor argument, not a post-processor tensor approximation.

### 4. CLIP integration

Replace the unmasked padded patch with the builder's dilated-context view.
Target pixels remain full intensity, `D minus M` supplies limited dimmed local
context and all pixels outside `D` are zero. Pass that exact uint8 RGB array to
`CLIPProcessor(images=...)`.

The optional two-encoding weighted target/context enhancement is a non-goal for
this objective. Do not delay or complicate the isolation fix with new scoring,
prompt-vector weighting or model changes. Existing prompt similarity, winning
label, score and resident model behavior remain unchanged apart from the image
pixels supplied.

At service verbosity 3 with `clip.debug: true`, store the exact lossless RGB
array passed as the processor's image argument, using only:

```text
clip-candidate-view-CANDIDATE-0008.png
```

where the number is the one-based source candidate ID, zero-padded to four
digits. Do not use the current filtered position, frame ID, input name, prompt,
label or any client text. The PNG bytes decoded from JSON/ZIP must equal the
preprocessor input array. Do not emit a JPEG substitute.

### 5. BLIP3 integration

Replace the old context/spotlight composition with one pair made from the same
shared builder result:

- left: target-only view, original target pixels and fill outside `M`;
- right: dilated-context view, original target pixels, limited dimmed context
  only in `D minus M`, fill outside `D`, and optional contour outside `M`.

No untouched rectangular crop may appear anywhere in the pair. Use a fixed
divider and deterministic after-mask scaling/letterboxing. The exact paired PIL
image passed to `_Blip3QA.answer` must be the same array retained by debug.
Build it once per candidate and reuse it for every applicable rule; do not
retain it across requests.

Replace the fixed instruction with this semantic content while keeping the
bounded client target question in the existing delimited section:

```text
Judge only the selected target shown in isolation on the left. The right side
provides limited local context. Do not classify objects visible only in the
context ring. Answer exactly Yes or No.
```

At L3 when an effective BLIP3 rule has `debug: true`, use only:

```text
blip3-verification-CANDIDATE-0008-QUESTION-0003.png
```

Candidate and question IDs are public one-based IDs. Filenames contain no
filtered position, frame/input name, rule name, target question, answer, label,
prompt or other client text. Preserve the existing fixed question-count,
question-length/token and generation-token limits. Debug must not create an
answer-text duplicate.

### 6. Request configuration and strict validation

Add `candidate_views` as a dedicated algorithmic top-level section, separate
from BLIP3 rule names:

```yaml
candidate_views:
  clip:
    mode: mask_dilated
    context_fraction: 0.10
    min_context_pixels: 0
    max_context_pixels: 64
    outside_fill: zero
    context_intensity: 0.35
  blip3:
    mode: mask_dilated
    context_fraction: 0.10
    min_context_pixels: 0
    max_context_pixels: 64
    outside_fill: zero
    context_intensity: 0.35
    contour_width: 2
```

Omission of the section or either child resolves to these safe defaults. The
same effective defaults apply to the service and trusted legacy/core calls, so
no configured CLIP/BLIP3 stage falls back to an unmasked rectangle.

The service validator must reject null/non-mapping top and child blocks,
unknown children, unknown fields, booleans masquerading as numbers/integers,
non-finite values, invalid types and values outside these exact limits:

| field | accepted value/range |
|---|---|
| `mode` | string `mask_dilated` only |
| `context_fraction` | finite number 0.0 through 0.5 |
| `min_context_pixels` | integer 0 through 256 |
| `max_context_pixels` | integer 0 through 512 |
| `outside_fill` | string `zero` only |
| `context_intensity` | finite number 0.0 through 1.0 |
| `contour_width` | BLIP3-only integer 0 through 16 |

Require `min_context_pixels <= max_context_pixels`. Reject `contour_width`
under the CLIP child. Unsupported fields/modes use `unsupported_field`; bad
types, nulls, nonfinite/range/cross-field values use `invalid_config`. Never
clamp, ignore or replace an explicit request value.

Strict zero is the sole public fill mode in this first implementation. Neutral
and blurred modes are explicitly future work, not claimed capabilities. They
must be rejected rather than silently approximated.

`clip.padding` may no longer influence HTTP model pixels: the hostile service
validator must reject it as `unsupported_field` with documentation directing
clients to `candidate_views.clip`. For trusted legacy CLI configs only, either
remove it with an explicit bounded deprecation warning or translate a valid
nonnegative integer to an equivalent fixed-radius mask dilation; it must never
restore an unmasked rectangular crop and must never be silently preferred over
explicit `candidate_views.clip`. Preserve legacy CLI execution and filesystem
adapter compatibility otherwise.

### 7. Request-locality, identity and typed manifest

Candidate-view configuration and all generated arrays are request-local. Never
mutate resident CLIP/BLIP3 model holders or reuse view configuration/arrays from
an earlier request. Two A/B/A requests with different context fractions must
produce A/B/A effective manifests and view pixels while model initialization
counts and service PID remain stable.

Keep internal `_source_index` zero-based. Immediately after post-SAM2 area/bbox
filtering, assign each retained candidate a stable zero-based
`_filtered_index` in that retained source-order list. Preserve both through
CLIP, BLIP3, final-label filtering, final deterministic object ordering,
visualization, debug metadata and serialization. Do not renumber after a later
candidate is rejected.

Publicly define:

```text
source_candidate_id = _source_index + 1  # one-based
filtered_index = _filtered_index         # zero-based post-SAM2-filter list
question_id = question_index + 1         # one-based
```

Add `source_candidate_id` and `filtered_index` to every L2/L3 object record.
Keep `instance_id` independent: it is assigned only after final filtering and
deterministic ordering. Add a default-compatible optional filtered-index field
or property to `ObjectResult` so existing direct constructors remain source
compatible.

Every response, including L0/L1, must include typed
`service.candidate_views.clip` and `.blip3` effective configuration plus whether
that stage was applied. Include bounded requested/effective provenance if it can
be done without ambiguity; effective values and application status are
mandatory. At L3 add bounded model-input records for emitted debug artifacts,
each containing stage, source candidate ID, filtered index, optional one-based
question ID, exact fixed artifact name, target/context bboxes, effective radius,
and source/crop/model-input dimensions. Records must correspond one-for-one to
candidate-view debug artifacts and contain no image pixels or client text.

Use explicit Pydantic schemas and OpenAPI snapshots. This is an additive service
manifest extension under `zap-it.v1`; do not bump the schema version. JSON and
ZIP `manifest.json` must agree exactly. Artifact descriptors/member bytes must
agree on fixed name, media type `image/png`, SHA-256 and size.

### 8. Capabilities and resource admission

Extend authenticated static `/v1/capabilities` with typed candidate-view
defaults, field types/ranges, allowed values, stage-specific fields, dilation
formula, public identity bases, exact debug triggers/name templates and the
statement that bbox is storage-only after masking. It must remain static,
authenticated, path/secret/GPU-topology-free and must not acquire readiness or
the inference gate.

Preserve all existing artifact and response limits. Do not silently truncate
candidate-view debug artifacts. Before CLIP/BLIP3 model work, once candidate and
applicable-question counts/shapes are known, reject a predictably impossible
debug request with the existing structured `response_too_large`/resource error
rather than running that stage and partially emitting artifacts. Dynamic sink,
per-artifact, raw-total, encoded JSON/ZIP and deadline enforcement remains a
second line of defense. Do not weaken raw-SAM2 debug admission or visualization
reservations, and include candidate-view debug artifacts in combined count/byte
accounting.

Process candidates sequentially and release transient builder/model-input arrays
promptly. Never accumulate all candidate views except the already bounded
requested lossless debug arrays in the service memory sink. No request image,
mask or model input may be written outside the in-memory sink or `/dev/shm`
request workspace, and the workspace must be empty after every request.

## Non-goals

- No SAM2 model/configuration/profile change, accuracy benchmark, aerial-solar
  fixture, reference polygons or panel recall/precision claim.
- No change to CLIP prompt-vector scoring, two-view weighted embeddings,
  BLIP3 decision parsing, model identity/revision, residency, device, dtype,
  cache, weights or generation limits.
- No public neutral/blurred fill mode in this objective; only strict zero.
- No Detectron2, new ML model, SciPy/OpenCV base dependency, uploaded point grid,
  client path, filename, artifact destination or arbitrary renderer.
- No change to SAM2 raw visualization semantics, labelled final renderer,
  post-filter reason precedence, YOLO, identity PNG, geometry or legacy dataset
  export beyond compatibility updates required by the new safe view boundary.
- No API schema-version bump, release/tag/publish, production claim, network,
  firewall, VPN, driver, system CUDA or unrelated-service mutation.

## Required self-contained CPU tests

1. **Rectangular leakage.** Generate a 512x512 RGB array, a nonrectangular mask
   and a high-contrast striped distractor outside `M` but inside `bbox(M)`.
   Prove it is absent from target-only, absent from context when outside `D`,
   all target exterior pixels equal zero and all context pixels outside `D`
   equal zero.
2. **Dilation boundary/formula.** Put one marker within and one outside the
   configured Euclidean dilation radius. Prove only the inner marker can appear
   in context, neither can appear in target-only, and the exact ceil/min/max
   radius formula is reported. Cover fraction zero and min/max overrides.
3. **BBox hole.** Use U/ring masks and a distractor inside the empty bbox hole.
   Prove it stays zero until actual morphological dilation reaches it; no bbox
   fill occurs.
4. **Disconnected masks.** Prove all components remain visible, intervening
   space stays zero unless reached by real dilation and no rectangular bridge
   is introduced.
5. **Borders.** Cover masks touching each edge and every corner, including a
   disconnected border case. Prove no negative coordinates, wraparound,
   additional source pixels or invalid output dimensions.
6. **Small masks/resizing.** Use masks below model minimums. Prove neutralization
   precedes resize/letterbox, the source-space crop is not enlarged, masks use
   nearest-neighbor, RGB uses the documented interpolation and prohibited
   pixels remain zero in model input.
7. **Contour.** Prove contour pixels are only `D minus M`, target RGB remains
   byte-identical, zero outside `D` remains zero and width/color/placement are
   deterministic including width zero.
8. **Determinism and immutability.** Equal image/mask/config/source ID produces
   equal arrays, masks, bboxes, radius, metadata and PNG hashes; inputs are not
   mutated. Different config changes only the documented pixels/metadata.
9. **Exact CLIP seam.** Capture the exact array supplied to the mocked
   `CLIPProcessor(images=...)`; require byte equality with the builder-derived
   dilated-context model input and prove the old unmasked bbox path cannot run.
   Debug PNG decode must equal that exact array.
10. **Exact BLIP3 seam.** Capture the exact PIL/array supplied to mocked
    `_Blip3QA.answer`; require byte equality with the deterministic left-target /
    right-context pair and its debug PNG. Prove no untouched rectangular context
    appears, composition is built once per candidate/reused across questions,
    and the fixed query preserves the bounded target question plus new
    instruction.
11. **Identity/names.** Exercise post-filter removals and final reordering.
    Prove source ID and filtered index survive every stage/object/debug record,
    filenames use one-based source/question IDs, and prompts, labels, rule names,
    answers, frame names and path-like client text cannot affect names. Prove
    artifact/record one-to-one correspondence.
12. **Validation.** Exhaust valid endpoints/defaults and invalid null, mapping,
    bool-as-number, nonfinite, range, min>max, unknown stage/field, unsupported
    fill/mode/CLIP contour and service `clip.padding` cases. Prove no explicit
    value is clamped/ignored. Preserve question/scalar/node/depth/token bounds.
13. **Manifest/capabilities.** Prove effective values at L0-L3, applied false/true,
    L3-only bounded input records, L2 source/filter identity, schemas/OpenAPI,
    authenticated static capabilities and JSON/ZIP name/hash/size/manifest
    parity. Capabilities must disclose no operator paths, key or physical GPU.
14. **Request isolation.** Run A/B/A fake/service requests with different CLIP
    and BLIP3 fractions and capture exact model inputs. Prove A/B/A effective
    manifests/pixels, no shared config/array mutation and stable resident-holder
    initialization counts.
15. **Resource bounds.** Prove combined CLIP/BLIP3/raw-SAM2/visualization debug
    artifact count, single bytes, total raw bytes, final JSON/ZIP and deadline
    limits. Predictable insufficiency must fail before the affected CLIP/BLIP3
    model call, emit no partial candidate-view set and not leak internal values.
16. **Legacy compatibility.** Trusted CLI still runs, never supplies unmasked
    rectangles, handles/deprecates `clip.padding` explicitly, retains safe
    filesystem adapter behavior and does not route API inputs to filesystem.
    Existing annotated rendering and all unrelated outputs remain compatible.

Use generated arrays for every deterministic acceptance gate; do not depend on
an external photograph or semantic model answer. Preserve and run the existing
labelled-renderer, BLIP3 verification/query/resource, CLIP, post-filter,
request-local SAM2 A/B/A, raw-SAM2 visualization, hostile YAML, schema,
artifact, auth, lifecycle, timeout/cancel, metrics, packaging, legacy and
residency tests.

## Verification

### CPU/static before any service restart

- Add focused builder, CLIP, BLIP3, core, validator, envelope/schema,
  capabilities, resource and legacy tests for every criterion above.
- Run the canonical full CPU suite with coverage. Public CI stays CPU/offline,
  downloads no models and requires no key/GPU/network.
- Run Ruff format/check, compileall, `scripts/check_documentation.py`, affected
  shell syntax and systemd validation, wheel/sdist build,
  `verify_release_artifacts.py`, archive and tracked-tree secret scans against
  the existing baseline, `twine check`, and `git diff --check`.
- Do not restart, stop, reload or reconfigure `zap-it-lan.service` until all of
  the above is green on the implementation head and enough execution time
  remains to complete the full restart/readiness/live/cleanup evidence. If that
  gate cannot be met, leave the old service running and report INCOMPLETE; do
  not perform a speculative last-minute restart.

### Bounded live private-LAN qualification

After the CPU/static gate, one controlled restart of only
`zap-it-lan.service` is authorized. Immediately beforehand recheck every GPU
index/UUID/PCI/name/VRAM/process, driver/CUDA/PyTorch fact, `/dev/shm`, listener,
unit ownership, port, environment mode/digest and free capacity without printing
or reporting a key. Start no second model process and touch no driver, firewall,
route, VPN, unrelated unit or device.

After readiness:

1. Require health/readiness 200; missing/wrong inference key 401; authenticated
   capabilities and metrics 200; docs/OpenAPI 404. Capabilities must contain the
   exact candidate-view contract and no host/path/secret disclosure.
2. Use one already-authorized ignored local fixture only in RAM or the mode-0700
   `/dev/shm` request workspace. Do not copy image/YAML/prompt bytes into Git,
   OAP, journal or report. Send one bounded L3 ZIP request that runs SAM2, CLIP
   and at least one BLIP3 question; enables `clip.debug` and one BLIP3 rule debug;
   keeps every tested candidate serializable; and requests explicit zero-fill
   candidate-view values. Return final objects plus the exact CLIP and BLIP3
   model-input PNGs in that one response.
3. Verify fixed safe names, one-based source/question IDs and zero-based filtered
   indices; object/debug-record association; JSON/ZIP manifest agreement;
   member media type/size/SHA; decodable lossless PNGs; and byte/pixel isolation
   against decoded L3 masks and builder metadata. Require every CLIP pixel
   outside reconstructed `D` to be zero and both BLIP sides to obey their target
   and support masks. Do not print or retain image content.
4. Run bounded A/B/A requests with different CLIP and BLIP3 context fractions.
   Require A/B/A effective manifests and radius/input changes without state
   leakage, stable service PID, one registry/model initialization, one listener
   and one assigned-GPU process. Semantic answers need not be identical and are
   not an accuracy acceptance gate; disclose them only as bounded counts/status.
5. Record bounded request latencies, response/artifact counts and sizes, candidate
   counts, host RSS and assigned-GPU peak/free memory. Require no failed request,
   OOM, deadline violation, restart, second process, credential/path leakage or
   leftover workspace entry.
6. Re-run readiness/health/auth/capabilities/metrics and verify `NRestarts=0`
   for the new PID, exactly one listener, sanitized journal, unchanged mode-0600
   environment digest and an empty request workspace. Leave the unit enabled,
   active and ready at `10.8.132.76:17891` on the Objective-017 implementation.

Live semantic correctness is not judged by a particular CLIP/BLIP3 label. The
deterministic acceptance gate is the exact pixel boundary supplied to each
processor. Disclose any failed request, delayed readiness, incomplete pixel
reconstruction, second restart, state leakage or cleanup failure; do not retry
silently or weaken acceptance.

### CI/checks

Push the implementation/control commit and wait for every required current-head
CI and CodeQL check. Each expected Python version, static/build, release audit,
CodeQL workflow/check must be present and successful with none pending, failed,
cancelled or missing before a COMPLETE report. The final report-only SELF child
must also have all required checks successful.

## Acceptance criteria

1. Every CLIP image argument is an exact-mask-derived dilated-context view;
   source pixels outside `D` are zero and the old untouched rectangle is
   unreachable.
2. Every BLIP3 image is target-only left and bounded dilated-context right;
   source pixels outside `M`/`D` are zero as applicable and no untouched crop is
   visible.
3. Dilation, cropping, contour, resize/letterbox and metadata are deterministic,
   boundary-safe and correct for holes, disconnected, border and tiny masks.
4. Strict request-local configuration is accepted at documented limits,
   defaults safely, rejects invalid/unsupported values without clamping or
   ignoring, and does not leak between A/B/A requests.
5. One L3 API request returns final output and exact lossless CLIP/BLIP3
   model-input artifacts with fixed safe source-ID names and exact ZIP manifest
   hashes/sizes.
6. One-based source candidate ID and zero-based filtered index survive filtering,
   classification, verification, final ordering, visualization, debug records
   and public objects; filenames never depend on client text.
7. Effective values and limits are truthful in every response/capabilities;
   docs and live API agree and contain no stale untouched-crop/padding filename
   claim.
8. Existing response/artifact/deadline/question/token limits, SAM2 residency,
   labelled rendering, raw diagnostics, post-filter diagnostics and legacy CLI
   remain green.
9. CPU/static, current-head CI/CodeQL and bounded live qualification are fully
   green. The newest code is left enabled, active and ready on the assigned
   private-LAN endpoint with one process and an empty request workspace.

The strongest reason not to merge is that a convincing debug renderer could
still differ from the actual array passed to a model, or interpolation could
reintroduce forbidden pixels after a correct source-space mask. Answer this
with byte-identity processor-seam tests, after-resize support-mask assertions,
one-to-one manifest/debug records and bounded live reconstruction—not visual
inspection alone.

## Documentation/provenance

Perform a deliberate consistency refresh of `README.md`, `ARCHITECTURE.md`,
`docs/API.md`, `docs/CONFIG.md`, `docs/ALGORITHMS.md`, `docs/CORE.md`,
`docs/OUTPUT-PARITY.md`, `docs/SERVICE-DATASHEET.md`, `docs/RUNBOOK.md`,
`TESTING.md`, `CHANGELOG.md` and `RELEASE_NOTES.md` where applicable. Search the
entire current documentation for stale `untouched context`, rectangular CLIP
crop/padding, spotlight-only, old BLIP filename/index and omitted candidate-view
configuration/capability claims; correct every current-document occurrence.
Historical documents may remain historical only when unmistakably labelled.

Document exact formulas/defaults/limits, zero-only fill support, contour and
resize order, public ID bases, filename templates, response levels, debug
triggers, artifact limits, schema compatibility, legacy `clip.padding`
migration/deprecation, deterministic scope and the difference between pixel
isolation and semantic model accuracy. Update the documentation checker index
if needed. No model/revision/license/provenance or dependency claim may change
without exact evidence.

## Security/resource/protected-host constraints

- Uploaded YAML controls only the enumerated scalar candidate-view values. It
  never controls paths, filenames, models, revisions, weights, cache, device,
  dtype, network, code, point grids, artifact destinations or residency.
- Never include prompts, labels, questions, rules, answers, input names or
  request IDs in candidate-view artifact paths. Use only fixed ASCII templates
  and bounded numeric source/question IDs.
- Keep one process, one worker and one active inference. Expose only assigned
  physical GPU0 through `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=0`; application code uses logical `cuda:0`. Fail closed
  on UUID/visibility mismatch. No fallback or free-GPU heuristic.
- Do not expose credential values, raw YAML/image/mask/model-input bytes,
  operator paths, exception strings or host topology in logs, errors, Git,
  reports or capabilities. Local live inspection must use the existing private
  key without printing it.
- Preserve `/dev/shm` only, mode 0700, no persistent request data, bounded
  in-memory artifacts and cleanup on success/failure/timeout/cancellation.
- No system package/driver/CUDA/firewall/route/VPN mutation; no unrelated
  process/service/device mutation; no release/tag/publish/merge/auto-merge by
  coding.

## Deferred human adjudication

- Decision: NONE

## GitHub publication

Commit the exact activated `oap/active` and immutable
`oap/orders/017-a-mask-isolated-candidate-views.md` bytes with the bounded
implementation. Keep the diff limited to this feature, its tests,
documentation, OAP transcript and directly required compatibility changes.
Push all non-report work before reporting and record a literal 40-hex
implementation SHA.

Create exactly `oap/reports/017-a-report.md` as the final report-only SELF child.
That final commit must change only the report, have the implementation SHA as
its sole parent, be pushed, have all required checks green, and be verified
against the remote before sending exactly one response FIFO `OK`. Make no later
mutation. Coding does not merge or enable auto-merge.

## Required immutable report evidence

- Exact order/branch/PR/base/head/implementation SHA/report SELF SHA and
  one-path parent topology; bounded changed-file and commit inventory.
- Criterion-by-criterion evidence for builder invariants, model-input byte
  identity, config validation, identity propagation, debug names,
  manifest/capabilities, A/B/A isolation, resources and legacy behavior.
- Exact focused and canonical CPU test commands/counts, coverage, static/build/
  docs/package/secret-scan results and every current-head CI/CodeQL check URL,
  run/job/status.
- Before/after live service PID, unit/readiness/auth/listener/model-init/GPU/RSS/
  `/dev/shm`/environment-digest facts; bounded request latency/count/size and
  pixel-reconstruction results without content or credentials; explicit count
  of restarts and any failed attempt.
- Documentation files reviewed/changed and a stale-claim search result.
- Deferred human adjudication action `NONE`; confirmation that CRITICAL.md,
  credentials, model pins, unassigned resources and unrelated state were not
  changed.
- Strongest reason not to merge and the exact evidence that answers it.
- Honest limitations/blockers. COMPLETE is allowed only when all ordered CPU,
  CI and live gates pass and the newest service is left enabled/active/ready.
