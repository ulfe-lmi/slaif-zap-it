# In-memory single-image core

This document describes the canonical core entry point, typed results,
deterministic ordering and renderer semantics, artifact-sink boundary, and
compatibility guarantees for the batch and service adapters.

## Canonical entry point

```python
from src.core import CoreConfig, run_single_image

outcome = run_single_image(
    image_rgb,  # decoded RGB numpy array (H, W, 3)
    config,  # normalized CoreConfig (see below)
    frame_id="frame-0001",
    segmenter_state=None,  # reusable model-holder state dicts
    clip_state=None,
    blip3_state=None,
    dryrun=False,
    verbosity=1,
    device=None,
    artifact_sink=sink,  # required when any debug flag is configured
    class_labels=("cat",),  # effective class mapping order
    render_visualizations=True,  # service enables this only for L3
)
outcome.result  # typed PipelineResult
outcome.segmenter_state  # model states to thread forward
```

The engine performs no filesystem access of its own. Stage callables are
injectable via :class:`src.core.StageFunctions` for testing with fakes.

## Configuration boundary

`CoreConfig.from_mapping(mapping)` normalizes exactly the algorithmic fields
(`alpha`, `preprocessing`, `mask_generator`, `postsam2processing`, `clip`,
`blip3`, `visualization`) with the legacy semantics (`visualization.alpha`
hoisted, `roi: false -> None`). `classify_config_fields` partitions top-level
keys into *algorithm*, *batch/deployment-only* (`images`, `video`,
`export_yolo_det`) and *unrecognized*; the core never reads batch-only keys.
This is the normalized algorithm boundary. The service performs its hostile
upload policy validation before constructing `CoreConfig`; the trusted CLI may
pass its broader legacy configuration through a separate adapter.

`config_digest(config)` returns a stable SHA-256 over the normalized values
(provenance hook; excludes wall-clock time).

## Pipeline stages

The chain matches the historical behavior:

```text
ROI crop -> optional resize -> SAM2 masks -> area/bbox filter
        -> optional CLIP labeling -> optional BLIP3 verification
        -> keep-label filter -> deterministic final-object ordering + id assignment
        -> optional visualization arrays
```

Stage statuses, candidate counts, post-filter diagnostics and wall-clock timings
are recorded on the result (`stage_statuses`, `candidate_counts`,
`post_filter_diagnostics`, `timings`). Timings are
observability data and intentionally excluded from all deterministic payload
serialization.

The result also carries a typed SAM2 service manifest. It records requested
values, the explicit/profile/default source of every effective scalar, the
exact crop/grid prompt and mask-prediction estimates, the raw generator count,
resource warnings and measured `stage.sam2` duration. The live adapter creates
one fresh generator around its resident model for each request; it does not
thread that generator through reusable model state.

The post-SAM2 filter and its diagnostic sidecar share one evaluator. It applies
the terminal precedence `maxsize` -> `empty_mask` -> `max_w` -> `max_h`, using
strict `>` rejection and inclusive threshold retention. The area comparison is
performed before segmentation access; a `maxsize` record retains the exact area
and uses `0/0` bbox dimensions because they were not evaluated. Empty masks also
use `0/0` for their distinct reason. Only later outcomes measure inclusive pixel
bbox extents on the exact remapped mask. Counts reconcile evaluated with retained
plus the four removal counters. L3 serializes only numeric rejection records in
input order, at most 256, and reports `rejections_truncated`; lower levels do not
expose this field. Source indices are the core's pre-filter SAM2 ordinals, with a
trusted legacy ordinal fallback.

## Mask remapping fix (regression evidence)

The previous implementation projected resized-space mask pixels forward with
``Yg = y + int(rpos * scaleY)``. With downscaled inference images this leaves
destination rows unreachable: an ROI of height 11 resized to 5 rows reaches at
most ``y + int(4 * 11/5) = y + 8``, so original ROI rows 9-10 could never be
part of any mask, silently shrinking masks, areas and bboxes near ROI edges.

The core now uses the exact inverse nearest-neighbor mapping: each destination
pixel samples ``floor(local * resized/roi_extent)`` (clamped), guaranteeing
full coverage, monotonic coordinates and in-bounds writes. The regression is
pinned by
`tests/test_core_engine.py::test_inverse_remap_regression_downscale_full_coverage`.

## Final-object ordering (single definition)

1. descending mask area;
2. ascending centroid row;
3. ascending centroid column;
4. ascending original candidate index among the filtered candidates.

Instance IDs `1..N` are assigned after final filtering. Object records, YOLO
line order and identity-mask values all derive from this one ordering
(`src/core/ordering.py`).

## Typed results

`ObjectResult` carries per object: `instance_id`, `source_index`, the complete
boolean source `mask` (original-image coordinates), scalar stage metadata
(`clip_label`, `clip_score`, `predicted_iou`, `stability_score`,
`blip3_answer`, ...), assigned `class_id` (+`class_id_source`), per-object
warnings, and computed geometry accessors (pixel bbox, normalized bbox, area,
centroid). A `geometry()` hook exists but the canonical path never executes
the geometry stage, so no geometry fields are ever fabricated.

`PipelineResult` carries image dimensions, ROI box, resize info, ordered
objects, stage statuses, candidate counts, post-filter diagnostics, rendered
visualization arrays, warnings, timings and provenance. `serialized_metadata()` produces the
deterministic JSON-friendly view (arrays skipped, NumPy scalars converted);
byte-stability across repeated identical runs is tested.

## Renderers

**YOLO** (`render_yolo(objects, image_width=…, image_height=…)`): one
five-field line per object, fixed six-decimal precision, coordinates
normalized to ORIGINAL image dimensions, lines terminated with `\n`; empty
detections produce an empty string. Class ids come from the engine-assigned
effective mapping (label list position); labels absent from the mapping fall
back to class id `0` and record a per-object warning — objects are never
silently dropped.

**Identity PNG** (`render_identity_png(objects, width=…, height=…)`)
returns lossless 16-bit grayscale PNG bytes: dimensions exactly equal the
original image, background `0`, object IDs `1..N`; disconnected components of
one object share that object's ID; contested pixels are won by the
larger-area object (ties by smaller instance ID). More than 65535 objects
raise `IdentityMaskOverflowError` before any pixel allocation. The
single-valued projection deliberately loses overlap facts; overlap truth
remains available through each object's retained source mask.

The service calls the renderer with `ensure_all_ids=True`. If the baseline
winner would fully occlude an object, this mode seeds each already-visible ID
with a row-major baseline representative and completes the missing IDs with a
deterministic augmenting-path assignment. Candidate masks are traversed in
baseline-visible then row-major order through fixed-size chunks, so matching
auxiliary memory is bounded by object count plus one documented scan chunk;
the algorithm does not claim a globally minimum number of raster overrides.
If distinct pixels cannot be reserved, rendering fails closed.

Both renderers consume the ordered object tuple, so bijective agreement
between object list, YOLO lines and PNG IDs holds by construction.

The pure `src.core.raw_visualizations` renderer is a separate API-safe seam.
It accepts only an RGB array and source-indexed raw mask records, makes no
model/filesystem/network/environment calls, and returns bounded RGB arrays plus
typed coverage facts. It uses one-based source IDs, deterministic arithmetic
colors, independent 3x4 contact-sheet tiles (320x240 content plus a 28-pixel
label bar), fixed score labels and at most 96 represented candidates. Candidate
tiles may enlarge padded crops for readability; its full-image diagnostics never
upscale. Its union,
overlap heatmap and uncovered images and their exact numeric accounting cover
all non-empty candidates, including those not represented in a page. The API
engine stores these arrays as the fixed PNG names only when the service-safe
L3 debug switch is active; the legacy path does not call this renderer.

## Artifact sinks

Sinks accept logical names only (relative, no traversal, no absolute paths):

- `MemoryArtifactSink` — stores bytes/text/image-arrays/records in RAM in
  insertion order; trusted CLI compatibility uses this boundary.
- `BoundedMemoryArtifactSink` — service-only bounded sink that refuses a new
  artifact before retention when count, per-item or total raw-byte limits would
  be exceeded.
- `FilesystemArtifactSink(root_dir)` — compatibility adapter for the legacy
  CLI: writes artifacts under the operator-selected output directory with
  atomic writes and mode `0600`.

When any debug flag (`preprocessing.debug`, `mask_generator.debug`,
`postsam2processing.debug`, CLIP debug, BLIP3 rule debug) is configured, a
sink MUST be supplied; otherwise the engine raises `CoreError` instead of
silently skipping or writing to arbitrary locations. CLIP and BLIP3 debug store
only the exact RGB arrays passed to their processors as lossless PNGs. Service
names are `clip-candidate-view-CANDIDATE-####.png` and
`blip3-verification-CANDIDATE-####-QUESTION-####.png`; trusted CLI names prefix
these tokenized names with a sanitized frame stem and never include prompts,
labels, rule names or answers. No answer-text artifact is generated.

BLIP3 composes one source-space image once per applicable candidate and reuses
the same final image for all questions: exact Euclidean support pixels are
restored from source bytes, the exterior contour is painted outside support,
and all other local scene pixels are Pillow-Gaussian-blurred. The centered nominal
crop uses inclusive raw/support bboxes and a half-open array-slice bbox. A crop
that cannot contain support plus contour is rejected for that candidate before
QA/debug work. The service validator strips nested
BLIP3 debug flags below L3 with one aggregate warning. L3 input records retain
source candidate ID, filtered index, status, bboxes, radii, widths, sigma and
source/model dimensions; `blip3_candidate_views` is separate from one-record-
per-debug-question artifact inputs.

## Compatibility notes

- `run_frame_pipeline(...)` keeps its exact signature and return shape; it now
  binds `out_dir` as a `FilesystemArtifactSink`, delegates computation to the
  core and additionally exposes `core_result` on `FramePipelineResult`.
- Legacy monkeypatch targets (`src.batch.run_sam2`, `apply_roi`,
  `save_roi_debug`, ...) remain importable; stage resolution reads module
  globals at call time.
- Per-frame JSON output key order changes slightly (the internal
  `segmentation` array was never serialized anyway); content semantics are
  unchanged.
- Final objects are now emitted in the deterministic order defined above
  rather than raw arrival order.
- Service annotated overlays use a stable bounded palette. The service's
  `annotated` stream is mask-only; `annotated-labelled` is an L3-only
  final-object renderer that uses sanitized labels and exact instance IDs
  without changing structured results. Legacy composite and writer behavior
  remains a trusted CLI concern.

## Geometry drift resolution

`modules/geometry` implements Canny/Hough line extraction but is NOT invoked
anywhere in the canonical frame path; the CLI docstring previously claimed
otherwise and has been corrected. No geometry fields are produced or
fabricated by the core; the typed hook stays reserved for a later objective.

## Verification

See [`TESTING.md`](../TESTING.md). Core coverage lives in
`tests/test_core_*.py` plus the extended `tests/test_run_frame_pipeline.py`.
All tests run CPU-only with no CUDA, network or model downloads.

The 020 execution order is preprocessing, SAM2 proposals, optional geometry,
raw CLIP crop construction and complete scoring, permissive routing, separate
BLIP3 composition/verification, final filtering, ordering, visualization, and
serialization. Source candidate IDs are one-based and stable; post-filter
indices are zero-based. Deterministic tests do not establish semantic accuracy
or recall.
