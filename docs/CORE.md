# ZAP-IT in-memory single-image core (objective 001-a)

Status: implemented in objective `001-a`. This document describes the
canonical single-image core entry point, its typed results, the deterministic
ordering and renderer semantics, the artifact-sink boundary and the
compatibility guarantees for the legacy CLI/batch path. It does **not** claim
any HTTP/API or live-GPU readiness; those are objectives 002+.

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
This is a trusted-CLI boundary only — it is not yet the hostile-upload policy
validator planned for objective 002.

`config_digest(config)` returns a stable SHA-256 over the normalized values
(provenance hook; excludes wall-clock time).

## Pipeline stages

The chain matches the historical behavior:

```text
ROI crop -> optional resize -> SAM2 masks -> area/bbox filter
        -> optional CLIP labeling -> optional BLIP3 verification
        -> keep-label filter -> visualization arrays
        -> deterministic final-object ordering + id assignment
```

Stage statuses, candidate counts and wall-clock timings are recorded on the
result (`stage_statuses`, `candidate_counts`, `timings`). Timings are
observability data and intentionally excluded from all deterministic payload
serialization.

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
objects, stage statuses, candidate counts, rendered visualization arrays,
warnings, timings and provenance. `serialized_metadata()` produces the
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

Both renderers consume the ordered object tuple, so bijective agreement
between object list, YOLO lines and PNG IDs holds by construction.

## Artifact sinks

Sinks accept logical names only (relative, no traversal, no absolute paths):

- `MemoryArtifactSink` — stores bytes/text/image-arrays/records in RAM in
  insertion order; the future service path uses this and never touches disk.
- `FilesystemArtifactSink(root_dir)` — compatibility adapter for the legacy
  CLI: writes artifacts under the operator-selected output directory with
  atomic writes and mode `0600`.

When any debug flag (`preprocessing.debug`, `mask_generator.debug`,
`postsam2processing.debug`, CLIP debug, BLIP3 rule debug) is configured, a
sink MUST be supplied; otherwise the engine raises `CoreError` instead of
silently skipping or writing to arbitrary locations. Debug artifact names
match the historical filenames (`<frame>-roi01.jpg`,
`<frame>_sam2-patch0000.jpg`, `<frame>_sam2-filtered-patch0000.jpg`,
CLIP/BLIP3 patch JPEGs and BLIP3 `.txt` answer files), so CLI output remains
equivalent when configured.

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
- Visualization composites still use random colors today; they are explicitly
  out of scope for the deterministic renderers above and remain unchanged.

## Geometry drift resolution

`modules/geometry` implements Canny/Hough line extraction but is NOT invoked
anywhere in the canonical frame path; the CLI docstring previously claimed
otherwise and has been corrected. No geometry fields are produced or
fabricated by the core; the typed hook stays reserved for a later objective.

## Verification

See `TESTING.md`. Objective-specific coverage lives in
`tests/test_core_*.py` plus the extended `tests/test_run_frame_pipeline.py`.
All tests run CPU-only with no CUDA, network or model downloads.
