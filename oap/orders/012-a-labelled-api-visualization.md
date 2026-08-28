# OAP Work Order — 012-a — Labelled API visualization

## Objective

Add a lightweight, deterministic, Detectron2-free `annotated-labelled`
visualization renderer to the supported HTTP API. At verbosity 3 it must return
a PNG containing the existing segmentation overlay plus each final object's
sanitized class label and exact manifest instance number. An explicit bounded
option may also display the final finite CLIP confidence.

This is one new Objective-012 branch and PR. Preserve the merged authenticated
private-LAN service and its existing inference key. Do not begin mask-aware
BLIP3 input construction or post-filter diagnostics in this PR; those are
Objectives 013 and 014.

## Verified starting state

- Remote `main`: `ce41b0becfb53cfe96ac11570a1af23b2d963311`, the merged
  Objective-011 authenticated private-LAN service. Its post-merge CI and CodeQL
  runs are successful. GitHub has no open PR.
- Required branch: `oap/012-a-labelled-api-visualization`; create exactly one
  PR titled `Objective 012: labelled API visualization` against `main`.
- Coding checkout is clean on the historical merged Objective-011 branch.
  Fetch and branch from exact remote main while preserving the atomically
  published 012 order/active transcript; do not replay 011.
- Persistent user unit `zap-it-lan.service` is enabled, active, and ready at
  exact `10.8.132.76:17891`, with one owned PID `373595` at the strategic
  snapshot. Its mode-0600 operator environment contains the fixed inference
  key; never print, replace, commit, log, or report that key.
- Host `hinton2`; assigned physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24,576 MiB, driver `610.43.02`. The live service is the only
  compute process and owned its measured residency. The human has assigned this
  GPU exclusively to ZAP-IT work, but exact index+UUID fail-closed law remains.
- `/dev/shm` is an 11.2-GiB tmpfs with the service root private and no request
  residue. Loopback ports 17892, 17893, and 23654 were available but are not
  reservations. Do not start a second model service in this objective.
- Current API allowlists `annotated|alpha-overlay`; both draw only deterministic
  colored masks. `panoptic` uses optional unqualified Detectron2 and is rejected
  before inference. Current visualization artifacts are RGB PNGs named only
  from already-bounded safe visualization IDs and are limited by stream count,
  per-artifact bytes, total raw bytes, response bytes, deadline, and artifact
  count.
- The core currently renders visualizations before final `ObjectResult`
  construction. Final object ordering assigns request-local instance IDs only
  after CLIP, optional BLIP3, and label filtering. The new renderer therefore
  cannot correctly infer manifest IDs from raw mask-list position.

## Requirements

### 1. New API renderer and final-object authority

1. Add canonical renderer name `annotated-labelled`. Do not alias it to
   `panoptic` and do not import or require Detectron2.
2. The HTTP YAML policy accepts `annotated-labelled` only within a
   `visualization.blip3` entry. Reject it under `sam2` or `clip` before model
   execution because those stages do not represent final ordered objects.
3. Refactor the core/render call boundary as narrowly as necessary so this
   renderer consumes the exact final ordered `ObjectResult` sequence. Visible
   instance numbers must be `ObjectResult.instance_id`, and visible class text
   must be `ObjectResult.label` after CLIP, optional BLIP3 mutation, final label
   filtering, and deterministic ordering.
4. Structured labels, scores, instance IDs, objects, YOLO, identity PNG, and
   JSON/ZIP manifests remain independent of whether visualization is requested.
   Visualization must not mutate objects, masks, metadata, ordering, or class
   mapping.
5. Existing `annotated` and `alpha-overlay` behavior remains backward
   compatible, including deterministic palette/output for existing inputs.
   `panoptic`, unknown names, unsafe IDs, and malformed rules remain rejected.

### 2. Rendering contract

1. Begin with the existing deterministic alpha mask overlay on the original
   RGB image. `alpha` keeps the existing validated finite range and semantics.
2. Draw one visible label for every final object with a non-empty mask. Default
   display format is exactly `<sanitized-label> <instance_id>`, for example
   `solar_panel 3`.
3. Add optional entry field `show_confidence`, which must be a strict boolean
   and defaults to `false`. When true and `ObjectResult.clip_score` is finite,
   append exactly `   CLIP <score-to-two-decimals>`. Omit the confidence suffix
   for absent or non-finite values; never render `nan` or `inf`.
4. Sanitize visible labels deterministically before drawing:
   - normalize to Unicode NFKC;
   - collapse whitespace and strip edges;
   - permit only ASCII letters/digits, space, `_`, `-`, `.`, and `+` in the
     visible form; replace other/control/path-separator characters with a
     single safe replacement and collapse repeats;
   - use `unknown` when the result is empty;
   - limit the label portion to at most 48 visible characters before adding the
     instance/confidence suffix, and shorten further when required to fit.
   The original structured label remains unchanged in metadata.
5. Use a repository/runtime-stable Pillow bitmap/default font without adding a
   remote font dependency. Render a high-contrast bounded background/foreground
   combination so text remains legible on the mask and image.
6. Compute each primary anchor deterministically from the exact mask bbox and
   centroid. Clamp every label background/text rectangle within image bounds.
   Handle masks and anchors touching every border, small images, empty sanitized
   labels, and long strings without exceptions or writes beyond the image.
7. For nearby objects, evaluate a fixed deterministic sequence of candidate
   positions (for example above, below, inside, lateral) and greedily choose a
   zero-overlap position when one fits. Otherwise choose the candidate with the
   least occupied-label intersection using deterministic tie-breaking. Labels
   need not solve global packing, but distinct labels must not completely
   overlap when a non-overlapping in-bounds placement exists.
8. Colors and placement derive only from final deterministic object order and
   geometry. Repeated renders of identical RGB bytes, masks, objects, entry
   settings, and dependency versions must produce identical PNG bytes/hash.
9. Prompt text, raw BLIP questions/answers, user filenames, and labels never
   become artifact IDs, filenames, paths, metric labels, or log fragments.
   Artifact name remains `visualization/<safe-id>.png`.

### 3. Limits, schemas, and documentation

1. Treat `annotated-labelled` as one ordinary uint8 RGB visualization stream.
   Preserve the existing exact `height * width * 3` pre-inference raw reservation,
   maximum eight streams, per-artifact/total raw/response size, deadline, and
   artifact-count checks. Text layout must not create an unbounded canvas,
   font/cache, string, or per-pixel Python structure.
2. Add `show_confidence` to the strict visualization-entry allowlist only;
   reject non-boolean values and all unknown fields before inference.
3. Update schemas/examples as applicable and update `docs/API.md`,
   `docs/CONFIG.md`, `docs/OUTPUT-PARITY.md`, `docs/CORE.md`, root README,
   architecture/datasheet/testing/runbook material where claims occur. State
   clearly:
   - `annotated` remains mask-only;
   - `annotated-labelled` is final-stage, labelled, L3-only, deterministic and
     Detectron2-free;
   - `panoptic` remains unsupported;
   - labels always remain available structurally independent of rendering.
4. Include this supported request shape in API/config documentation:

   ```yaml
   visualization:
     blip3:
       - id: labelled-result
         renderer: annotated-labelled
         alpha: 0.55
         show_confidence: true
   ```

### 4. CPU and artifact verification

Add focused tests plus the complete canonical suite. Required mechanized cases:

1. Actual PNG pixels prove text/background is rendered, not merely that an
   `ImageDraw` method was called. The correct sanitized final class label and
   manifest instance number drive the rendered result.
2. BLIP3-mutated final label wins over an earlier CLIP-only label, and final
   object order/IDs exactly match the JSON/ZIP manifest.
3. Labels remain within all four image boundaries; tiny/border masks do not
   crash; dynamic shortening keeps the visible box in bounds.
4. Two or more nearby masks use distinct non-completely-overlapping label boxes
   whenever the fixed candidate set provides such a placement.
5. NFKC/whitespace/control/path-separator sanitization, empty fallback, 48-char
   cap, and dynamic truncation are exact and deterministic. The structured
   original label is not changed.
6. `show_confidence=false` default, true finite two-decimal display, absent
   score, NaN, infinities, and strict non-boolean rejection.
7. Repeated array and encoded PNG bytes have identical SHA-256. Different final
   label/instance inputs materially change the labelled image.
8. JSON base64 artifact bytes and ZIP member bytes have the same hash and size
   recorded in each manifest. Labels cannot affect member paths.
9. Existing `annotated` golden/compatibility behavior is unchanged.
10. HTTP config accepts the exact new final-stage rule, rejects it at earlier
    stages, and continues rejecting `panoptic`, unknown renderers, unsafe IDs,
    unknown entry fields, too many streams, and invalid alpha/budget cases.
11. L0-L2 skip visualization execution and raw reservation exactly as before;
    L3 labelled streams honor existing single/total/artifact/response/deadline
    boundaries and cleanup.

Run and report the canonical CPU suite with coverage, focused visualization/API
tests, Ruff format/check, compileall, documentation checker, shell syntax where
changed, wheel/sdist build, artifact audit, tracked-tree and built-artifact
secret scans, `twine check`, and `git diff --check`. No model download or CUDA
may occur in public CI. All required current-head CI and CodeQL checks must be
present and successful.

### 5. Bounded live private-LAN evidence

The service must remain enabled and active during ordinary implementation.
After the implementation head is committed and CPU/static checks pass, one
controlled restart of the owned `zap-it-lan.service` is authorized so the
running process uses the Objective-012 code. Do not stop/disable it for final
cleanup; leave it enabled, active, ready, and serving the existing key.

Before restart re-verify exact assigned index+UUID/process ownership, unit,
listener, `/dev/shm`, and key-file digest without reading/reporting the key.
Use only physical index 0 / assigned UUID, exposed as logical `cuda:0`. Do not
start a second model process, touch drivers/network/firewall, or signal anything
except the exact owned user unit.

After readiness returns:

1. Prove missing/wrong inference keys still return 401 and docs/OpenAPI remain
   404. Use the on-disk key only inside a non-echoing local command.
2. Send one real verbosity-3 BLIP3-final request using `annotated-labelled`,
   `alpha: 0.55`, and `show_confidence: true` as both JSON and ZIP. Use an
   already-authorized local fixture/config without copying raw content into
   Git/OAP/logs/chat.
3. Decode the PNG, prove RGB shape equals original image dimensions, the
   labelled artifact differs from the corresponding mask-only artifact, every
   final object has a manifest ID/label, JSON and ZIP artifact hashes/sizes
   agree, and repeat execution produces the identical labelled PNG digest.
   CPU tests provide exact text/placement pixel semantics; do not use OCR as a
   security oracle.
4. Prove one unchanged post-restart PID/listener during all requests, only the
   assigned GPU process, bounded response/resource metrics, sanitized logs,
   empty request workspace, preserved mode-0600 environment and unchanged
   inference-key digest.
5. Leave the persistent private-LAN service enabled, active, ready, with exactly
   one `10.8.132.76:17891` listener. Record only sanitized hashes/counts/sizes,
   never labels/prompts/raw images/config/keys.

Disclose every failed live attempt and corrective change. A skipped, partial, or
non-deterministic live labelled-artifact result is not acceptance.

## Non-goals

- no Detectron2/panoptic activation or dependency;
- no mask-aware BLIP3 spotlight/crop/instruction/debug-artifact work (Objective
  013);
- no post-filter rejection diagnostics (Objective 014);
- no model identity/revision/residency/lifecycle change, multi-process GPU
  sharing, MPS, MIG, or new model download;
- no public/WAN bind, TLS/gateway/firewall/VPN/network change, second service,
  new system unit, key rotation/disclosure, or management-endpoint exposure;
- no request-selected fonts/files/paths, persistent request artifacts, prompt or
  label filenames, geometry, video, dataset export, release/tag/package upload;
- no rewrite of prior immutable orders/reports and no merge by coding.

## Acceptance and report contract

All requirements and tests above pass; diff remains bounded to renderer/core/API
policy/schema/tests/docs plus exact OAP transcript; existing output contracts and
private-LAN behavior remain green. The strongest reason not to accept is that a
rendered label could disagree with the structured final object or escape bounds
while superficial artifact tests still pass. Answer it with direct final
`ObjectResult` authority, pixel-level placement/sanitization tests, JSON/ZIP
hash parity, and repeated real output evidence.

Push all implementation and exact active/order bytes before reporting. Record a
literal 40-hex implementation SHA. Then create exactly
`oap/reports/012-a-report.md`, commit only that report as the final `SELF` child,
push, verify remote parent/one-path topology and bytes, send exactly one response
FIFO `OK`, perform no later mutation, and exit. Coding never merges.

## Deferred human adjudication

- Decision: NONE
