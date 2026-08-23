# API target

> Status (objective 005-a): the v1 contract below is implemented, bounded and
> CPU-tested with a fake engine; see
> [docs/API.md](API.md) for the binding wire documentation. This file remains
> as the original target statement.

```text
POST /v1/completions
Content-Type: multipart/form-data
```

Fields: one `image`, one UTF-8 YAML `config`, `verbosity=0..3`, optional
`response_format=json|zip`, fixed/optional `model`, `stream=false` only.

This is a ZAP-IT image-pipeline contract on the requested path, not generic
OpenAI text-completions compatibility.

## Levels

| Level | Adds |
|---|---|
| 0 | YOLO 5-field normalized lines in `choices[0].text` |
| 1 | uint16 identity PNG: 0 background, 1..N request-local objects |
| 2 | per-object bbox/area/SAM/CLIP/BLIP/geometry fields actually produced |
| 3 | bounded stage/full metadata, overlap-preserving RLE masks, overlays, timings, provenance, warnings |

Levels are monotonic but do not force disabled expensive stages. Same object
ordering drives YOLO, object list and PNG. Disconnected components may share ID.
Overlaps use a documented deterministic raster winner; full output preserves
per-object masks as uncompressed column-major COCO-style RLE.

JSON artifacts use `{name,media_type,encoding:"base64",sha256,size,data}`. ZIP
contains stable `manifest.json`, `detections.yolo.txt`, `identity-mask.png` and
level-gated artifacts. `usage` is null unless a meaningful non-token resource
schema is versioned later.

## Errors

Stable sanitized envelope and codes for malformed multipart, invalid/oversized
image, unsafe/invalid YAML, unsupported field/level/format/model, busy, not ready,
timeout/cancel, inference failure, insufficient RAM/shared memory and response
too large. No raw input, stack, secret or host path.
