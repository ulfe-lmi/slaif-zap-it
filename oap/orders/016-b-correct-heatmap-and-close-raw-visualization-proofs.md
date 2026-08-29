# OAP Work Order — 016-b — Correct heatmap zero coverage and close raw-visualization proofs

## Objective

Amend Objective-016 PR #72 to correct a real renderer contract violation and
close the specific proof gaps found during strategic review. Uncovered pixels
must be black in the overlap heatmap, tiny candidate crops must be enlarged into
useful contact-sheet tiles, label drawing and lower-level omission must be
tested at the actual rendering/API seams, and manifest cross-field invariants
must be validated rather than merely produced by convention.

This is a same-PR remediation round. Do not add Objective-017 solar fixtures,
polygons or semantic-quality acceptance.

## Verified starting state

- Remote `main` remains
  `8081152403657f5e737ab0b491e0b89f587209e1` with successful post-merge CI
  and CodeQL. PR #72 is open, non-draft and mergeable on branch
  `oap/016-a-bounded-raw-sam2-visualizations`.
- PR #72 head is report-only SELF
  `a0f6fabbf74d30a8d4a0fe0c99966cd3d2d10d21`; its parent is implementation
  SHA `7f393710d53966941acc3adf7bde2f194180fb7e`. SELF changes only
  `oap/reports/016-a-report.md`. All seven report-head CI/CodeQL checks pass.
- The implementation scope and live evidence are otherwise promising: 706 CPU
  tests passed; the live service produced 28 source candidates, three contact
  sheets and the three diagnostic images twice with identical raw artifact
  bytes/metadata. Preserve those bounded architecture and safety decisions.
- Concrete defect: `src/core/raw_visualizations.py::_heatmap()` initializes an
  RGB zero canvas but, whenever any covered pixel exists, writes blue as
  `255 * (1 - fraction)` for every pixel. Therefore overlap count zero becomes
  `[0, 0, 255]`, not required black `[0, 0, 0]`. The focused test checks only
  that a covered pixel is non-black and does not inspect a zero-coverage pixel.
  Documentation and the 016-a report claim the intended black-zero behavior,
  so code/evidence currently disagree.
- Contact-tile scaling includes a hard `1.0` ceiling. A small padded crop is
  centered at native size inside a 320x240 tile instead of being enlarged,
  making small rooftop-style masks needlessly difficult to see. The order said
  to resize the crop into the tile and reserved the no-upscale rule only for the
  three full-image diagnostics.
- The score-format test calls `_candidate_label()` directly; it does not prove
  the exact string is drawn, that the final text bounding box stays inside the
  28-pixel label bar, or that a maximum admitted public candidate ID fits. The
  pagination test checks sheet shape but not fixed neutral empty cells or
  disjoint tile placement.
- L0-L2 tests check only omission of the `raw_visualization` child; they do not
  prove absence of fixed `sam2-*` raw artifacts or that debug reached the engine
  as false. No focused test proves changing frame/user metadata cannot change
  API artifact names, although the implementation currently appears fixed-name.
- The Pydantic schema applies independent non-negative/max-length constraints,
  but does not validate the ordered cross-field arithmetic. Runtime construction
  currently produces consistent values; make this an explicit checked
  invariant so future refactors cannot publish contradictory manifests.
- Live service after the authorized 016-a activation is enabled, active and
  ready at `10.8.132.76:17891`, PID `465389`, `NRestarts=0`, one listener and
  one compute process. It uses only assigned physical GPU0 UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, RTX 3090
  24,576 MiB, driver 610.43.02, exposed as logical `cuda:0`. The request
  workspace is empty and the mode-0600 environment digest remains
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`.
  Never print or report a credential value.

## Required remediation

### 1. Correct black-zero heatmap semantics

Change the deterministic heatmap so overlap count zero remains exactly RGB
black even when the observed maximum is positive. Apply the color ramp only to
`overlap > 0` pixels. Preserve deterministic observed-maximum scaling and make
positive counts visible; when counts 1 and 2 coexist they must map to distinct
non-black colors. Do not change union, uncovered, count canvas, histogram,
downscaling, filenames or manifest fields.

Add exact tests for:

- an all-zero overlap image -> every heatmap pixel black;
- a mixed 0/1/2 canvas -> zero black, positive non-black, 1 and 2 distinct;
- before diagnostic resizing, the heatmap's black-pixel mask equals the union
  image's uncovered/black mask; and
- after nearest-neighbor resizing, zero-source regions selected by the resized
  union remain black in the resized heatmap.

The 016-b report must explicitly supersede 016-a report acceptance item 3 only
to the extent that the first implementation did not actually keep zero coverage
black. Do not rewrite the immutable 016-a report.

### 2. Enlarge small candidate crops and prove mask visibility

Remove the tile-only no-upscale ceiling. Scale each padded crop by the smaller
of `320 / crop_width` and `240 / crop_height`, whether below or above one, then
letterbox. Continue bilinear RGB and nearest-neighbor mask resizing, exact mask
alpha, aspect preservation, deterministic centering and boundary clamping.
The three full-image diagnostics must still never upscale.

Add a generated one-pixel candidate with the four-pixel minimum context. Prove
that its source crop is enlarged, the selected mask occupies more than one
rendered pixel, the overlay color is visible only on the resized exact mask,
and no padding/border escape occurs. Cover a border-touching one-pixel mask and
a disconnected mask. Expose only the minimum pure helper/test seam needed; do
not make layout request-configurable.

Update capabilities/docs wherever necessary to state that candidate crops may
be enlarged for readability while diagnostic images never upscale.

### 3. Prove actual label drawing, tile isolation and fixed cells

Instrument or factor the actual label-drawing seam so tests record the string,
coordinates and final text bounding box passed to Pillow. Require:

- exact finite text `C0001  IoU 0.843  stability 0.912`;
- exact `n/a` behavior for absent/non-finite scores;
- the largest admitted ID under Objective-015 prediction capacity (at least
  `C24576`) fits without losing the ID or either score field;
- every drawn bounding box lies wholly within its own 320x28 label bar; and
- no client label, prompt, question, frame name or path-like metadata is drawn.

For 13 candidates, inspect the second sheet: its first tile contains candidate
13, the other eleven content/label cells remain the fixed neutral/background
fills, and no tile writes into another tile's rectangle. Prove two overlapping
source masks remain independently visible in separate tile rectangles.

### 4. Validate manifest arithmetic explicitly

Add a pure bounded validation/finalization seam invoked after raw generator
count and omitted-empty count are known and before artifacts/summary are
published. It must reject inconsistent internal facts without echoing content.
Validate at least:

```text
raw_candidate_count = visualizable_candidate_count + omitted_empty_candidate_count
visualizable_candidate_count = represented_candidate_count + truncated_candidate_count
len(represented_candidate_ids) = represented_candidate_count
represented IDs are strictly increasing, unique, one-based and <= raw count
contact_sheet_count = ceil(represented_candidate_count / 12)
contact_sheet_count <= 8 and represented_candidate_count <= 96
artifact_names = exact expected page sequence followed by exact three diagnostics
covered_pixel_count + uncovered_pixel_count = source width * source height
sum(overlap_histogram.values) + overflow_pixel_count = source area
overlap_histogram["0"] = uncovered_pixel_count
max_overlap_count <= visualizable_candidate_count
overflow/truncated histogram flags agree with max_overlap_count > 255
diagnostic dimensions are positive, no larger than source, aspect-preserving
within deterministic integer rounding and <= 2,000,000 pixels
```

Normal internal inconsistency is an `inference_failure`/typed core failure with
a fixed sanitized message, not a client `invalid_config`, traceback or raw
mapping echo. Unit tests must tamper representative counts, IDs, page names,
pixel totals and histogram facts and prove rejection. Remove the duplicated
`raw_candidate_count == 3` assertion in the existing focused test.

### 5. Close API naming/level and legacy proofs

At verbosity 0, 1 and 2 with uploaded `mask_generator.debug: true`, prove:

- the effective core config reaching the fake engine has debug false;
- `service.sam2.raw_visualization` is absent; and
- no artifact descriptor/ZIP member begins with `sam2-` (the ordinary L1+
  identity artifact remains unchanged).

Run the service-safe engine twice with different frame IDs and candidate
metadata containing labels/prompts/questions/path separators. Require the exact
same fixed raw artifact-name tuple and prove no supplied string fragment appears
in a name or drawn label. Separately retain/run the existing trusted legacy
engine test requiring historical `<frame>_sam2-patch0000.jpg`; do not change its
behavior.

Require JSON/ZIP hash/size/media parity again after the renderer correction.
Keep proactive count/raw/response admission and every Objective-016 constant
unchanged.

## Required verification

Run focused raw renderer/API/resource/schema/core/legacy tests, then the full CPU
suite with coverage. Run Ruff format/check, compileall, documentation checker,
affected shell/systemd checks, build wheel+sdist, release artifact verification,
built/tracked secret scans, `twine check`, and `git diff --check`. Public CI
remains CPU/offline with no model downloads. Require all seven current-head
CI/CodeQL checks green on the 016-b implementation and final report heads.

## Bounded live requalification

Keep PID `465389` serving while implementing. After the 016-b implementation
head and CPU/static gates pass, one controlled restart of only
`zap-it-lan.service` is authorized to activate the corrected renderer. Before
restart recheck the exact assigned GPU facts/process, driver/CUDA/PyTorch,
listener/unit, `/dev/shm`, environment mode/digest and free capacity. Start no
second process and touch no driver, firewall, VPN, route, unrelated unit or key.

After readiness:

1. Recheck health/readiness 200, missing/wrong auth 401, authenticated
   capabilities/metrics 200 and docs/OpenAPI 404.
2. Repeat the same bounded, authorized in-memory L3 ZIP request twice with only
   SAM2 debug enabled. Require identical raw metadata and artifact bytes/hashes,
   stable PID/one registry initialization/one GPU process and no residue.
3. The selected live fixture previously had 1,661 uncovered pixels and no
   diagnostic downscale. Decode union, heatmap and uncovered artifacts without
   printing/persisting content. Require the union-black pixel mask, uncovered-
   white pixel mask and heatmap-all-channels-zero mask to be identical; require
   every covered heatmap pixel non-black. Reconcile black/white counts to the
   manifest. If the regenerated proposal set changes, apply the same relational
   assertions to the new exact counts and disclose the change.
4. Decode a contact sheet and require label/content regions nonblank; the
   unit-level enlarged-one-pixel proof remains authoritative for tiny masks.
5. Record bounded latency, ZIP size, candidates/pages/artifacts, coverage/max
   overlap, RSS and assigned-GPU peak/free memory. Require sanitized journal,
   unchanged environment digest, `NRestarts=0`, empty workspace and leave the
   service enabled, active and ready on `10.8.132.76:17891`.

No new resource-limit rejection is needed live; its CPU pre-engine proof remains
authoritative. Disclose every failed harness attempt or request. Never print a
key, response body, fixture name/content, prompt or image bytes.

## Non-goals and safety

- No new PR, numeric objective, solar fixture/polygon/accuracy scope, SAM2
  configuration/profile/cap change or model reload per request.
- No change to response limit defaults, candidate ID/page constants, palette,
  union/uncovered semantics, CLIP, BLIP3, filters, labelled renderer, YOLO,
  identity PNG, network/auth/device/cache/residency policy or CRIT-0001.
- No public exposure, persistent request data, filesystem API output, extra
  worker/process, release/tag/upload, key rotation/disclosure, history rewrite
  or unrelated cleanup.

The strongest reason not to merge remains a diagnostic that claims exact
coverage while its pixels or manifest disagree with that claim. Acceptance
requires black-zero heatmap pixels, explicit arithmetic validation, actual
draw-seam and level/name tests, and repeated corrected live artifact evidence.

## Publication/report contract

Amend exact PR #72 and branch
`oap/016-a-bounded-raw-sam2-visualizations`. Commit the exact active selector and
016-b order with the bounded remediation, capture the literal implementation
SHA, then create only `oap/reports/016-b-report.md` as its report-only SELF
child. Verify remote parent, one-path topology, exact report bytes and all final
checks before FIFO response. Coding never merges.

## Deferred human adjudication

- Decision: NONE
