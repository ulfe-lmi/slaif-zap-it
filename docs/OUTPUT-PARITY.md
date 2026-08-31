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
| CLIP labels, complete scores and class map | Each safe identifier owns one indivisible scalar prompt or an ordered array of independent prompts; every item refreshes resident CLIP text embeddings, class scores use the maximum prompt similarity with lowest-index ties, and routing sees only semantic classes | Public L0 class mapping, L2 winner/score, L3 complete vectors, prompt accounting and routing provenance |
| Candidate views for CLIP/BLIP3 | CLIP uses an untouched rectangular raw source crop; trusted CLI may explicitly retain its exact-mask compatibility view; BLIP3 uses one source-space image with exact Euclidean support, exterior contour and Gaussian-blurred scene context | Effective policy at every response level; L3 composition records and exact lossless processor/QA PNGs; this is pixel-boundary evidence, not an accuracy guarantee |
| BLIP3 verification | Pinned FP16 holder; each candidate is composed once and every applicable QA call reuses the same one-image input; residency remains capacity-selected | Public L2/L3 fields when executed; candidate-local containment rejection is bounded and non-mutating; L3 debug returns the exact sole QA PNG under the tokenized CANDIDATE/QUESTION name |
| BLIP3 question admission | Operator-owned startup limit defaults to 256 and accepts 1..256 questions/request; canonical routing plans at most one question per routed candidate and legacy rules share the total cap | Authenticated capability/runtime metadata; planned excess is structured `resource_limit` 413 before BLIP3 generation, distinct from `response_too_large` assembly overflow |
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
| BLIP3 debug images | Bounded in-memory lossless PNGs only at L3, named `blip3-verification-CANDIDATE-####-QUESTION-####.png`; the exact sole image passed to QA is retained and no answer-text duplicate is made | Public L3 artifact; structured answers/labels remain independent |
| Legacy image writer | Writes configured image sequences through the trusted batch adapter | Legacy CLI-only |
| Legacy video/MJPEG writer | Writes configured video streams through the trusted batch adapter | Legacy CLI-only |
| `images` and `video` batch output mappings | API parser ignores them with a warning | Legacy CLI-only; unsafe/inappropriate for service |
| `export_yolo_det` dataset export | Trusted batch adapter writes dataset annotations | Legacy CLI-only |
| Raw request/config/result persistence | Not used by the service | Unsafe/inappropriate for service |
| Goat academic fixture E2E | Local central-50% crop only; sanitized aliases/digests and bounded resource evidence | Rights confirmed; ignored operator inputs, excluded from packages as defense in depth |

## Post-filter diagnostic policy

The canonical filter evaluates optional area, inclusive bbox, aspect-ratio, and
border rules for every SAM2 candidate, including empty masks. Fixed precedence,
strict rejection, and equality retention are recorded in L3. Each bounded
rejection includes source candidate ID, nullable bbox, area, dimensions,
configured limit and reason; aggregate counts reconcile exactly and records are
capped at 256 with explicit truncation.
The field is absent at L0-L2 and is copied unchanged into JSON and ZIP
manifest responses. The roof/panel regression is generated CPU filter evidence,
not a real-image or SAM2-recall claim.

## Renderer policy

The service accepts only bounded in-memory annotated streams under
`visualization.sam2`, `visualization.clip`, or `visualization.blip3`, with safe
logical IDs and a maximum of eight streams. `annotated-labelled` is accepted
only under `visualization.blip3`; `panoptic`, unknown renderers,
composite/file/video rules, unsafe IDs and malformed entries are rejected before
inference. Optional streams are admitted after rendering through the same
request-local ledger as raw-SAM2 and candidate-view artifacts. A count,
per-item, aggregate-raw or response-byte miss records a typed omission and
does not replace successful inference with HTTP 413. The CLI retains its
existing trusted configuration and writer behavior.

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
truncation is explicit. The service may omit these optional bytes under its
ledger budgets while preserving the raw manifest and stage results; only an
essential response that cannot fit remains `response_too_large`. Legacy CLI
rectangular JPEG patches and all lower levels remain unchanged.

## Geometry status

The service accepts only optional canonical `postsam2processing` geometry
impossibility rules and reports every evaluated rejection at L3. The unrelated
top-level `geometry` batch section remains unsupported. Legacy `maxsize`,
`max_w`, and `max_h` aliases are explicit compatibility inputs with migration
warnings.

Objective 020 keeps CLIP and BLIP3 image builders independent: CLIP receives an
untouched rectangular `raw_bbox_crop` and BLIP3 receives its separately
composed one-image contextual view. Complete CLIP vectors, routing reasons,
and answer mappings are preserved in L3; `clip_scored` is the count captured
before routing, while L2 objects contain relevant semantic evidence. Semantic
accuracy is not inferred from parity tests.
