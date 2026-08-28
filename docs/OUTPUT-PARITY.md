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
| Post-SAM2 area/bbox filter | Removes candidates before optional classification | Public L3 candidate counts/status |
| CLIP labels, scores and class map | Request labels refresh resident CLIP prompts | Public L0 class mapping, L2 label/score, L3 provenance |
| BLIP3 verification | Pinned FP16 holder; each QA call receives one deterministic mask-aware context/spotlight pair with exact selected-pixel preservation, exterior dimming and contour; residency remains capacity-selected | Public L2/L3 fields when executed; L3 debug returns the exact paired PNG under a fixed numeric name; this is not an accuracy guarantee |
| Geometry Canny/Hough helpers | Helpers and tests exist, but canonical core does not call them; helpers may write TSV/debug files | Not supported by the service; legacy compatibility only where explicitly wired |
| `annotated`/`alpha-overlay` in-memory streams | Bounded RGB mask-only overlays; service executes them only at L3 | Bounded operator/service diagnostic |
| `annotated-labelled` in-memory stream | Final-object RGB overlay with sanitized label and exact instance ID; deterministic, L3-only and Detectron2-free | Bounded operator/service diagnostic |
| Panoptic/Detectron2 renderer | Detectron2 is absent and renderer is not a live capability | Unsupported for service; legacy helper only |
| Stage statuses, candidate counts, timings | Produced by the core | Bounded L3 service metadata |
| Object bbox, normalized bbox, area, centroid | Derived from source masks | Public L2 metadata |
| SAM quality, CLIP score, BLIP answer, geometry fields | Included only when an executed stage produced the field | Public L2 fields; geometry is currently absent |
| YOLO text | Five-field normalized final-object lines | Public L0-L3 completion text and ZIP entry |
| Identity PNG | Deterministic uint16 projection, 0 background, IDs 1..N, larger-area overlap winner with deterministic representatives | Public L1-L3 artifact |
| Per-object source masks | Exact boolean masks retained in request result | Public L3 uncompressed column-major RLE; overlap is preserved |
| JSON envelope and artifact descriptors | Level-gated, base64 binary artifacts with hashes and sizes | Public L0-L3 |
| ZIP manifest and raw artifacts | Data-free manifest plus the same raw artifacts and YOLO text | Public L0-L3 `zip` response |
| ROI/SAM2/post-filter/CLIP debug patches | In-memory only at L3, opaque names, bounded sink | Bounded operator/service diagnostic |
| BLIP3 debug images | Bounded in-memory lossless PNGs only at L3, named `blip3-verification-####-####.png`; the exact paired image passed to QA is retained and no answer-text duplicate is made | Public L3 artifact; structured answers/labels remain independent |
| Legacy image writer | Writes configured image sequences through the trusted batch adapter | Legacy CLI-only |
| Legacy video/MJPEG writer | Writes configured video streams through the trusted batch adapter | Legacy CLI-only |
| `images` and `video` batch output mappings | API parser ignores them with a warning | Legacy CLI-only; unsafe/inappropriate for service |
| `export_yolo_det` dataset export | Trusted batch adapter writes dataset annotations | Legacy CLI-only |
| Raw request/config/result persistence | Not used by the service | Unsafe/inappropriate for service |
| Goat academic fixture E2E | Local central-50% crop only; sanitized aliases/digests and bounded resource evidence | Rights confirmed; ignored operator inputs, excluded from packages as defense in depth |

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

## Geometry status

Geometry is intentionally not activated in the service. The API rejects a
top-level `geometry` field with `unsupported_field` before inference. Future
activation requires a separately governed scientific-stage order and an
in-memory refactor of the current file-writing helper.
