# OAP Work Order — 012-b — Labelled-renderer evidence hardening

## Objective

Amend Objective-012 PR #68 to close the strategic review gaps in the focused
`annotated-labelled` test evidence. Add direct deterministic array/PNG proof,
prove that the rendered instance number materially affects pixels, and prove
computed label/text placement and dynamic fitting remain bounded at image
edges. Make the smallest production adjustment only if these stronger tests
expose a real defect.

This is a same-PR continuation. Do not begin Objective 013 mask-aware BLIP3
verification or Objective 014 post-filter diagnostics.

## Verified starting state

- Remote `main` remains `ce41b0becfb53cfe96ac11570a1af23b2d963311`.
- The unique Objective-012 PR is #68, `Objective 012: labelled API
  visualization`, branch `oap/012-a-labelled-api-visualization`, base `main`.
- PR implementation commit is
  `05abf3795658e0b0ad0e5ebefb47affc415bf834`; immutable 012-a SELF report
  commit is `f09ed52ef3b7a03461b9f98c20b6443506c2f6b4`, whose literal parent is the
  implementation commit and whose only changed path is
  `oap/reports/012-a-report.md`.
- All seven current-head PR checks are successful and the PR is cleanly
  mergeable.
- Strategic review accepts the implementation seam, API policy, documentation,
  JSON/ZIP parity, real private-LAN request evidence, and service/GPU/security
  invariants. The continuation is required because the focused suite does not
  itself directly test all ordered determinism and layout claims.
- Specifically, `tests/test_labelled_visualization.py` currently has no
  repeated `render_annotated_labelled` array equality plus repeated encoded-PNG
  equality assertion; it varies a final label but not an instance ID to prove
  pixel sensitivity; and its tiny/border test asserts only shape/dtype, not the
  computed label/text rectangle, actual dynamic shortening, or edge placement
  invariants.
- Persistent `zap-it-lan.service` is enabled, active and ready at exact
  `10.8.132.76:17891`, MainPID `388703`, with one listener. It is the only
  compute process on assigned physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, RTX 3090 24,576 MiB. The
  mode-0600 inference-key file and its digest are unchanged. Strategic
  authenticated checks returned readiness 200, unauthenticated metrics 401 and
  authenticated metrics 200. `/dev/shm/slaif-zap-it` is empty.

## Requirements

### 1. Direct deterministic output evidence

1. Add a focused test which invokes `render_annotated_labelled` at least twice
   with identical RGB bytes, masks, final objects, alpha and confidence option,
   and asserts exact array equality and equal array-byte SHA-256.
2. Encode both repeated arrays through the same deterministic PNG path used by
   service visualization artifacts, or through the exact shared encoder that
   feeds that path. Assert byte-for-byte PNG equality and equal SHA-256. Do not
   merely compare manifest descriptors derived from one byte sequence.
3. Independently vary only the final structured label and only the final
   `instance_id`, keeping image/mask/settings fixed, and prove each change
   materially changes rendered pixels and encoded PNG hash. Continue proving
   the original structured object/mask is not mutated.

### 2. Direct layout and fitting evidence

1. Add table-driven cases for masks touching the top, bottom, left and right
   edges plus corner-adjacent masks on ordinary service-meaningful image sizes.
   Directly verify every computed selected label background rectangle is
   within `[0,width] x [0,height]` and has positive bounded extent.
2. Directly verify the actual Pillow text bounds for each selected label are
   contained by its background rectangle and the image whenever the required
   instance suffix can physically fit with the fixed bitmap font. Do not use
   output shape alone as containment evidence and do not use OCR.
3. Use a deliberately long safe label and a narrow-but-sufficient image to
   prove dynamic fitting actually shortens the visible label, preserves the
   exact instance suffix (and finite confidence suffix when enabled), and
   yields a measured text width no greater than the available width.
4. Retain explicit degenerate tiny-image coverage. Where the fixed font makes
   even the mandatory suffix physically impossible to fit, require
   deterministic bounded behavior with no exception, unbounded allocation or
   out-of-canvas write; state this physical-impossibility behavior accurately
   in a code comment/test rather than claiming that a readable full suffix fits
   a one-pixel canvas.
5. Add a direct repeated placement assertion for the nearby-object case, not
   only a single-run non-complete-overlap assertion.
6. Prefer testing a small pure layout helper returning the chosen text and box
   geometry if that makes the invariants directly observable. Keep it private
   or narrowly scoped; do not add a broad public API. If the current renderer
   cannot expose or satisfy these invariants cleanly, make the minimum bounded
   refactor and preserve its visible contract.

### 3. Compatibility, scope and verification

1. Do not change API renderer names, final-stage authority, sanitization
   allowlist/cap, confidence format, palette, artifact naming, response limits,
   structured results, authentication, service binding, model/runtime policy,
   or legacy `annotated` output unless a stronger test proves a production bug
   requiring a narrowly documented correction.
2. Run the amended focused labelled/visualizer/core/API tests and the complete
   canonical CPU suite with coverage. Run Ruff format/check, compileall,
   documentation checker, build/audit/secret scans, twine and `git diff
   --check`. Documentation changes are not expected unless a production
   behavior/comment correction makes an existing claim inaccurate.
3. Push the amendment to the existing Objective-012 branch/PR only. Require all
   current-head CI and CodeQL checks present and successful.
4. Keep the live service running during tests. If this continuation changes
   tests/comments only, do not restart it and independently prove the PID,
   listener, readiness, GPU UUID/process, key digest and empty request workspace
   stayed unchanged. If and only if production renderer/core/service code must
   change, one controlled restart of only `zap-it-lan.service` is authorized
   after CPU/CI evidence; then repeat the bounded real labelled JSON/ZIP/repeat
   proof from 012-a and leave the unit enabled, active and ready. Never start a
   second model process or expose the key.

## Non-goals

- no Objective 013 spotlight image/instruction/debug-artifact work;
- no Objective 014 rejection diagnostics;
- no new renderer, font or dependency; no Detectron2/panoptic activation;
- no model download, model identity/residency/lifecycle change, second service,
  GPU sharing, public/WAN/network/firewall/TLS change, key rotation/disclosure,
  release/tag/upload, or persistent request data;
- no rewrite of immutable 012-a order/report and no merge by coding.

## Acceptance and report contract

Acceptance requires direct mechanized proof of repeated array bytes, repeated
encoded PNG bytes/hashes, independent label and instance pixel sensitivity,
measured edge/background/text containment, actual dynamic truncation, and
repeated nearby-label placement. The full suite and all current-head GitHub
checks must be green, the diff must remain narrowly within the same PR, and the
private-LAN service/GPU/key invariants must remain intact.

The strongest reason not to merge remains that Pillow clipping could make the
tests pass by preserving only canvas shape while the intended label or instance
text lies outside the image. Answer it with directly observable layout
geometry, actual pixel/hash sensitivity and repeated encoded output—not with a
shape-only assertion or a method-call mock.

Record a new literal 40-hex amended implementation SHA. Then create exactly
`oap/reports/012-b-report.md`, commit only that report as the final `SELF`
child, push, verify its remote one-path topology/parent/bytes, send exactly one
response FIFO `OK`, and perform no later mutation. Coding never merges.

## Deferred human adjudication

- Decision: NONE
