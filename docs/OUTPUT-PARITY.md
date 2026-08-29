# ZAP-IT output parity matrix

> **Local academic regression policy.** The repository owner confirmed
> redistribution rights for the goat image/config assets. They remain ignored
> operator inputs and excluded from packages as defense in depth. They are not
> accuracy goldens or public CI fixtures.

This matrix is the authoritative inventory for the current YAML pipeline. A
module or helper is not evidence that the live service produces its output.

| Source or output | Current behavior | Service classification |
|---|---|---|
| Image decode, RGB conversion, dimensions | JPEG/PNG/WebP, bounded header and pixel decode | Public L0-L3 metadata and input guard |
| ROI crop and resize | Core preprocessing; masks are remapped to original coordinates | Public L3 stage status/provenance; object coordinates at L2 |
| SAM2 candidate masks and quality fields | Candidate masks are filtered and ordered | Public L2 fields when produced; bounded L3 counts/status |
| Request-local SAM2 generator policy | Resident model is reused; a fresh generator is built from strict effective scalars per request | `service.sam2` at every level and in ZIP manifest; raw candidate count is separate from post-remap L3 count |
| Post-SAM2 area/bbox filter | Removes candidates before optional classification; one deterministic short-circuit reason and bounded numeric records are retained | Public L3 candidate counts/status and `post_filter_diagnostics` |
| CLIP labels, scores and class map | Request labels refresh resident CLIP prompts | Public L0 class mapping, L2 label/score, L3 provenance |
| Candidate views for CLIP/BLIP3 | One request-local pure builder computes exact Euclidean dilation, zero-fill isolation and deterministic floor-rounded context; CLIP receives the context view, BLIP3 receives target-only left plus dilated-context right | Effective policy at every response level; L2 public source/filter identity; L3 exact lossless processor/QA PNGs and bounded numeric input records; this is not an accuracy guarantee |
| BLIP3 verification | Pinned FP16 holder; each QA call reuses one deterministic mask-isolated pair with exact selected-pixel preservation, exterior dimming and configured contour; residency remains capacity-selected | Public L2/L3 fields when executed; L3 debug returns the exact paired PNG under the tokenized CANDIDATE/QUESTION name; this is not an accuracy guarantee |
| Geometry Canny/Hough helpers | Helpers and tests exist, but canonical core does not call them; helpers may write TSV/debug files | Not supported by the service; legacy compatibility only where explicitly wired |
| `annotated`/`alpha-overlay` in-memory streams | Bounded RGB mask-only overlays; service executes them only at L3 | Bounded operator/service diagnostic |
| `annotated-labelled` in-memory stream | Final-object RGB overlay with sanitized label and exact instance ID; deterministic, L3-only and Detectron2-free | Bounded operator/service diagnostic |
| Panoptic/Detectron2 renderer | Detectron2 is absent and renderer is not a live capability | Unsupported for service; legacy helper only |
| Stage statuses, candidate counts, timings | Produced by the core | Bounded L3 service metadata |
| Object bbox, normalized bbox, area, centroid | Derived from source masks | Public L2 metadata |
| SAM quality, CLIP score, BLIP answer, geometry fields | Included only when an executed stage produced the field | Public L2 fields; geometry is currently absent |
| YOLO text | Five-field normalized final-object lines | Public L0-L3 completion text and ZIP entry |
| Identity PNG | Deterministic uint16 projection, 0 background, IDs 1..N, larger-area overlap winner with deterministic representatives | Public L1-L3 artifact |
| Raw SAM2 candidate diagnostics | Source-order one-based IDs, separate paginated mask tiles, all-candidate union/overlap/uncovered accounting and bounded typed facts | Public L3 only when `mask_generator.debug: true`; fixed PNG names; never a quality benchmark |
| Per-object source masks | Exact boolean masks retained in request result | Public L3 uncompressed column-major RLE; overlap is preserved |
| JSON envelope and artifact descriptors | Level-gated, base64 binary artifacts with hashes and sizes | Public L0-L3 |
| ZIP manifest and raw artifacts | Data-free manifest plus the same raw artifacts and YOLO text | Public L0-L3 `zip` response |
| ROI/SAM2/post-filter/CLIP debug patches | In-memory only at L3, opaque names, bounded sink | Bounded operator/service diagnostic |
| BLIP3 debug images | Bounded in-memory lossless PNGs only at L3, named `blip3-verification-CANDIDATE-####-QUESTION-####.png`; the exact paired image passed to QA is retained and no answer-text duplicate is made | Public L3 artifact; structured answers/labels remain independent |
| Legacy image writer | Writes configured image sequences through the trusted batch adapter | Legacy CLI-only |
| Legacy video/MJPEG writer | Writes configured video streams through the trusted batch adapter | Legacy CLI-only |
| `images` and `video` batch output mappings | API parser ignores them with a warning | Legacy CLI-only; unsafe/inappropriate for service |
| `export_yolo_det` dataset export | Trusted batch adapter writes dataset annotations | Legacy CLI-only |
| Raw request/config/result persistence | Not used by the service | Unsafe/inappropriate for service |
| Goat academic fixture E2E | Local central-50% crop only; sanitized aliases/digests and bounded resource evidence | Rights confirmed; ignored operator inputs, excluded from packages as defense in depth |

## Post-filter diagnostic policy

The canonical filter evaluates `maxsize`, empty mask, `max_w`, and `max_h` in
that exact precedence. The area comparison is terminal and occurs before
segmentation access; `maxsize` records retain the exact area and carry `0/0` bbox
dimensions because bbox dimensions were not evaluated. Empty masks also carry
`0/0` for their distinct reason. Other width and height values are inclusive
pixel extents over the remapped segmentation. Rejection comparisons are strict;
threshold equality is retained, and aggregate counts satisfy the
evaluated/retained invariant. L3 records contain only numeric source index,
reason, area and bbox dimensions, are ordered by candidate input, and are capped
at 256 with explicit truncation.
The field is absent at L0-L2 and is copied unchanged into JSON and ZIP
manifest responses. The roof/panel regression is generated CPU filter evidence,
not a real-image or SAM2-recall claim.

## Renderer policy

The service accepts only bounded in-memory annotated streams under
`visualization.sam2`, `visualization.clip`, or `visualization.blip3`, with safe
logical IDs and a maximum of eight streams. `annotated-labelled` is accepted
only under `visualization.blip3`; `panoptic`, unknown renderers,
composite/file/video rules, unsafe IDs and malformed entries are rejected before
inference. Before L3 inference, the service reserves exactly
`stream_count * height * width * 3` raw RGB bytes for annotated streams, rejects
single-stream or combined-budget overflow, and subtracts that reservation from
the debug sink budget. The CLI retains its existing trusted configuration and
writer behavior.

RLE, artifact preparation, base64 expansion and ZIP entry assembly share the
absolute request deadline. RLE transition detection is vectorized in fixed-size
column-major chunks and never retains a second full-size flattened mask.

## Raw SAM2 diagnostic policy

The combined ordinary overlay is insufficient for overlapping raw proposals: it
can show only the last composited color and cannot identify the source mask or
distinguish uncovered pixels from overwritten pixels. L3 plus
`mask_generator.debug: true` therefore emits independent contact-sheet tiles
with `C<source_id>` labels and three fixed diagnostics. Pages are 3x4 with
320x240 content and a 28-pixel label bar, capped at eight pages/96 candidates.
The crop uses clamped `ceil(10%)` context with a four-pixel minimum, RGB
bilinear and mask nearest-neighbor resizing, and 0.45 exact-mask alpha.
Padded candidate crops may be enlarged into their 320x240 tiles for
readability; the three full-image diagnostics never upscale.

`sam2-union-coverage.png` is black/white uncovered/covered,
`sam2-overlap-heatmap.png` is black at zero with a fixed observed-maximum
scaled ramp, and `sam2-uncovered-pixels.png` is the exact inverse of union at
source resolution before nearest-neighbor downscale to at most 2,000,000
pixels. Exact source counts and a bounded 0..255 histogram plus overflow are in
the optional `service.sam2.raw_visualization` child. The fixed names are
`sam2-candidates-page-0001.png`..`-0008.png` and the three diagnostic names;
truncation is explicit. API preflight reserves the exact 11-artifact RGB
formula before readiness/gate/engine work; encoded response checks remain
authoritative. Legacy CLI rectangular JPEG patches and all lower levels remain
unchanged.

## Geometry status

Geometry is intentionally not activated in the service. The API rejects a
top-level `geometry` field with `unsupported_field` before inference. Future
activation requires a separately governed scientific-stage order and an
in-memory refactor of the current file-writing helper.
