# SLAIF ZAP-IT — normative architecture for agents

## Product

Professionalize the existing YAML-driven ZAP-IT pipeline and expose one local,
stateless, single-image service endpoint:

```text
POST /v1/completions
image file + YAML config + verbosity(0..3) + response_format(json|zip)
```

This is a ZAP-IT-specific multimodal contract using the conventional path, not
generic OpenAI text completion compatibility.

## Existing capability chain

```text
preprocess ROI/resize
 -> SAM2 automatic masks
 -> area/bbox filter
 -> CLIP label+score
 -> optional BLIP3 question/answer relabel
 -> label filtering
 -> optional Canny/Hough lines+intersections
 -> stage visualizations / serialized fields / YOLO export
```

Preserve CLI/config behavior while extracting a pure in-memory core. Do not claim
outputs absent from actual modules.

## Target components

```text
FastAPI app
  request guard (size/media/auth/concurrency)
  YAML safe parser + API policy validator
  device guard (physical GPU1 -> visible cuda:0)
  persistent model registry/state
  in-memory pipeline engine
  deterministic object/result model
  artifact renderer (YOLO, uint16 identity PNG, overlays, JSON/ZIP)
  cleanup/metrics/error mapping
```

API path must not call legacy folder/batch writers directly. Introduce adapters
around reusable algorithms; preserve old CLI through compatibility wrappers.

## Request

`multipart/form-data`:

- `image`: exactly one JPEG/PNG/WebP (allowed types finalized by order);
- `config`: exactly one UTF-8 `.yaml|.yml` file;
- `verbosity`: `0|1|2|3` (aliases may be accepted but canonical response is int);
- `response_format`: `json|zip`; default JSON;
- `model`: optional fixed service ID; no arbitrary model selection;
- `stream`: unsupported/false initially.

`yaml.safe_load` only. Reject aliases/structures that exceed limits. API policy
forbids paths, output destinations, URLs, commands, imports, devices, secrets,
arbitrary model IDs/revisions, and batch input/output controls. Normalize,
validate and hash effective config; never echo secrets/path data.

## Response and levels

OpenAI-like completion envelope carries `id`, `object=text_completion`, `created`,
`model`, one `choice`, finish reason, service metadata, and level-gated artifacts.
`choices[0].text` at every level contains normalized YOLO lines:

```text
<class_id> <cx> <cy> <width> <height>
```

Coordinates are normalized to original image dimensions; class mapping is
returned/defined from effective config. No confidence sixth field unless a later
versioned contract explicitly adds it.

- L0: envelope + YOLO text + minimal image/config/service identifiers.
- L1: L0 + `identity-mask.png`, 16-bit grayscale; 0 background; IDs 1..N.
- L2: L1 + objects with instance ID, class/label, bbox pixel+normalized, area,
  SAM quality fields, CLIP score, BLIP answer, geometry only when produced.
- L3: L2 + bounded raw/stage metadata, per-object overlap-preserving mask encoding,
  overlays, geometry tables, timings, warnings, provenance, effective-config
  digest/content policy, optional debug artifacts.

Lower response levels do not execute extra optional modules. JSON binary artifacts
are base64 objects; ZIP contains `manifest.json`, `detections.yolo.txt`, PNGs and
other files. Build in memory or `/dev/shm`; enforce maximum response size.

## Identity mask

- `uint16` PNG, dimensions exactly original input image;
- 0=background; 1..N=request-local instance IDs;
- one object may have multiple disconnected components with same ID;
- overlapping source masks require a documented deterministic winner order;
- full output preserves original per-object masks/RLE so rasterization does not
  silently erase overlap facts;
- object list and PNG IDs must bijectively agree; reject overflow >65535.

## Memory and state

Request bytes, decoded image, intermediate arrays and response live in RAM.
Filesystem-only legacy/module calls receive unique mode-0700 workspaces under
`SLAIF_ZAP_IT_TMP_ROOT` (default `/dev/shm/slaif-zap-it`) with mode-0600 files and
`finally` cleanup. Never write request data into repo/cwd/legacy output folders.
No persistent raw inputs/outputs by default. Model weights/caches are persistent
operator assets outside repo; model objects may stay loaded. Request state must
be new per call.

## GPU and process

Physical target is GPU index 1. Human/operator preflight on 2026-08-23 observed:

```text
GPU1: NVIDIA GeForce RTX 2080 Ti, 11264 MiB
UUID: GPU-c457dbaf-991c-dc23-c781-0dc030776dd8
GPU0: separate RTX 2080 Ti with an unrelated workload at that time
```

Re-verify index/UUID/PCI/name/VRAM/processes before every live GPU objective; this
snapshot is planning evidence, not a permanent guarantee. GPU0 remains protected
even if later idle. Launch with:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
```

Inside process only `cuda:0` exists. Never use physical GPU0. Startup verifies
visible count/device and expected UUID when configured. One service process,
one GPU request, bounded queue. No multiple Uvicorn workers; no fork after CUDA.
With ~11 GB GPU1 VRAM, never assume SAM2+CLIP+BLIP3 co-residency. Objective 003
must measure individual/combined profiles and select an explicit safe strategy;
unsupported configurations are rejected honestly rather than spilling to GPU0.

## Service/network

Bind `127.0.0.1` on a port proved unused by live `ss`/process inspection. Port is
runtime config, never assumed by constitution. `/healthz` reports process health;
`/readyz` reports model readiness/device. Optional API-key auth precedes any LAN
exposure. Do not change firewall/network in normal development orders.

## Error contract

Stable JSON error envelope with request ID and code; no stack, raw YAML/image,
paths, credentials or model internals. Distinguish invalid image/config, unsupported
field/level/format, too large, busy, timeout/cancel, not ready, inference failure,
response too large, and insufficient `/dev/shm`.

## HWP-preloaded modernization sequence

Detailed draft `NNN-a` work orders for all seven numeric objectives live under
`oap/strategic-instructions/initial-orders/`. They are human-preloaded intent,
not active orders. Strategic reconciles each against live GitHub/host evidence,
removes every draft/VERIFY marker, and only then publishes it.

1. **000 Professional baseline:** audit/current tests, packaging, CPU CI+CodeQL,
   lint, documentation/security/provenance; no API/GPU mutation.
2. **001 In-memory core:** typed single-image engine, artifact sinks, deterministic
   object order, pure YOLO and uint16 identity mask; preserve CLI/video.
3. **002 API contract:** multipart `/v1/completions`, hostile-input policy,
   JSON/ZIP L0–L3, limits/errors/auth/health/concurrency with fake engine; no live
   GPU service.
4. **003 GPU1 qualification:** exact environment/model/license/revision audit and
   bounded live tests; measure what fits safely on the actual ~11 GB GPU1.
5. **004 Loopback activation:** one real service worker/request on freshly verified
   loopback port and pinned GPU1 UUID; E2E, cleanup, restart and rollback.
6. **005 Full parity/hardening:** overlap-preserving masks, safe geometry/
   visualization parity, limits, failure/cancel/state-isolation, metrics and
   service datasheet.
7. **006 Packaging/integration/release readiness:** build/install artifacts,
   operator packaging, optional tested container, SLAIF gateway/auth integration,
   license/supply-chain review and release candidate; external deployment/final
   release only after every applicable human adjudication gate is cleared.

One numeric objective=one PR; `b..z` amend the same PR. Strategic may refine or
split based on verified reality but must not casually replace the preloaded
outcome or skip dependency gates.

## Security

Treat images/YAML as hostile. Bound dimensions, bytes, CPU/GPU time, object count,
artifacts, YAML depth/aliases, queue and logs. Pin/review remote-code models. No
request-selected model/download/network/filesystem. Never log/persist raw content
or secrets. `/dev/shm` is ephemeral, not an authorization boundary.

## HWP, HJP and critical adjudication

Human Work Preloading places intent/architecture/roadmap/detailed objective drafts
before OAP. Strategic must resolve ordinary ambiguity and make consequential
provisional decisions; it may not stop merely from reluctance to choose.

`CRITICAL.md` is a rare append-only Deferred Human Adjudication Register. Append
only if all five conditions in that file hold: unresolved by existing law,
material alternatives, serious possible impact, safe provisional continuation,
and plausible human reversal before deployment. Not for bugs, failed tests,
TODOs, normal tradeoffs, limitations, style, dependencies or low-risk reversible
choices. Strategic authors exact entry; coding appends only by order. Agents never
close/edit prior entries. Open entries allow safe development but block their
stated production/public/destructive/release gate pending latest human
`ACCEPTED` disposition.

## Tests

CPU CI must run without CUDA/model downloads. Use dependency injection/fakes and
small redistributable fixtures. Live tests are explicit `gpu` markers, force
physical GPU1 visibility, snapshot all GPU processes before/after, serialize, and
skip honestly when unavailable. CI green is necessary, not sufficient.
