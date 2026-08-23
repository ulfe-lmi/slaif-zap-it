# OAP Work Order — 005-a — Full-output parity, hardening, metrics and datasheet

Objective `005-a`. Advance the accepted loopback service from the Objective-004
MVP to a bounded, evidence-rich local release candidate: account for every
legacy/current output, add overlap-preserving full masks, harden artifact and
resource budgets before allocation/encoding, prove request/model-state
isolation under repeated and failure traffic, add content-safe local metrics,
and publish an exact service datasheet. Preserve the measured SAM2+CLIP
resident profile, legacy CLI, physical-GPU1 isolation and loopback-only scope.

## Authoritative prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`.
- Numeric objective / round: `005 / 005-a`.
- Mode: `CREATE_NEW_PR`.
- Objective 004 accepted as PR #48 and squash-merged to remote `main` at
  `22e827eaab15a5eb3299a6b5bfd156eb96c68946`.
- Post-merge CI and CodeQL on `22e827e...` are both SUCCESS.
- No PR or remote branch exists for Objective 005.
- Required branch: `oap/005-a-full-output-parity-hardening-and-evidence`, created
  from remote `main` at `22e827e...`.
- Required PR title: `Objective 005-a: full-output parity, resource hardening,
  metrics and service datasheet`.
- `oap/active` on merged main contains historical `004-d`; do not replay it.
  Commit the new exact order transcript and `005-a` selector on the new branch.
- Predecessor evidence: immutable `oap/reports/004-d-report.md`, with final
  implementation/report topology accepted and no open critical entry.

## Verified current product/runtime baseline

### Supported live profile and host

- Qualified strategy: `sam2_clip_resident_blip3_rejected`; supported profiles
  `sam2`, `clip`, `sam2_clip`; live service runs resident SAM2+CLIP and rejects
  BLIP3-dependent configuration before load.
- Physical GPU1 target: RTX 2080 Ti, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI
  `00000000:00:0C.0`, 11264 MiB; strategic pre-order snapshot was 6 MiB used /
  10815 MiB free with no compute process.
- Physical GPU0: separate RTX 2080 Ti, UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI
  `00000000:00:08.0`, protected unrelated PID 66522; its memory is workload-
  owned and variable (2501 MiB at strategic snapshot). Never touch it.
- Driver 580.178.04; qualified Python 3.12 / Torch 2.5.1+cu124 runtime;
  FastAPI 0.141.1, Uvicorn 0.52.4, python-multipart 0.0.32,
  opencv-python-headless 4.10.0.84. Detectron2 and prometheus-client are absent.
- Host: ~52.7 GiB RAM with ~47.4 GiB available at snapshot; `/dev/shm` is a
  27-GiB tmpfs and empty. Ports 17891 and 23654 were free. Re-verify every fact
  before live work; these observations are not reservations.

### Current response/output behavior

- L0: completion envelope and normalized five-field YOLO.
- L1: L0 plus deterministic uint16 identity PNG with bounded complete
  object-ID projection.
- L2: L1 plus object records (ID/class/label, bbox, area, centroid, SAM quality,
  CLIP score and produced BLIP/geometry hooks).
- L3: L2 plus stage statuses, candidate counts, timings, warnings, provenance,
  configured visualization PNGs and memory-sink debug artifacts.
- `ObjectResult.mask` retains exact per-object overlap truth internally, but no
  public L3 overlap-preserving mask representation exists.
- Current response assembly base64-encodes artifacts and only then checks JSON
  size; ZIP is built after JSON/base64 construction. Artifact count is silently
  truncated to 64 after artifacts have already been produced/encoded. This is
  not acceptable release-candidate resource behavior.

### Current scientific/output audit facts

- The canonical core executes ROI/resize, SAM2, post-filtering, optional CLIP,
  optional BLIP3, label filtering, deterministic ordering and configured
  visualization arrays.
- The live profile supports SAM2+CLIP only; BLIP3 stays unsupported.
- `modules/geometry` contains Canny/Hough helpers but neither the canonical
  legacy frame path nor the in-memory core calls them. Its helper also writes
  TSV/debug files directly. Geometry is therefore preserved-but-unwired/dead
  behavior, not a currently produced output.
- Qualified in-memory visualization renderer: `annotated`. The optional
  `panoptic` renderer requires absent/unqualified Detectron2 and is not a live
  service capability.
- Current debug paths can produce ROI images, SAM2 candidate patches,
  post-filter patches and CLIP patches. CLIP debug logical names currently may
  derive from prompt text and must become opaque/content-free. BLIP3 debug is
  legacy-only because BLIP3 is rejected in the live profile.
- No `/metrics` endpoint or request/resource metrics registry exists.

### Current limits/evidence baseline

- Existing defaults: image upload 20 MiB; config 256 KiB; decoded pixels
  64,000,000; response 256 MiB; request deadline 120 s; queue depth 0;
  Retry-After 5 s; YAML depth 16, nodes 10,000, collection items 512 and scalar
  chars 16,384.
- Accepted Objective-004 CPU baseline: 319 passed, one intentional GPU-marker
  skip, 75.99% coverage; Ruff/package/CI/CodeQL green.
- Accepted live baseline on the 128×128 synthetic fixture: steady requests
  approximately 434–437 ms, first request approximately 0.9 s, process RSS
  approximately 1,984,604 KiB, GPU1 approximately 1849 MiB ready and 5749 MiB
  during inference, returning to 6 MiB after stop. These are bounded fixture
  measurements, not an SLA or soak.

## Binding output/parity decisions

### Parity matrix

Create `docs/OUTPUT-PARITY.md`. Account for preprocessing, SAM2, post-SAM2,
CLIP, BLIP3, geometry, visualization, serialized metadata, debug paths, YOLO,
identity mask, legacy image/video writers and dataset export. Classify every
output as exactly one of:

1. public service L0–L3;
2. bounded operator/service diagnostic;
3. legacy CLI-only;
4. unsupported or currently dead/unwired; or
5. unsafe/inappropriate for the service.

No output may be claimed merely because a helper exists.

### Geometry and visualization

- Do NOT activate geometry in this objective. Keep the module/tests and legacy
  compatibility intact, keep API `geometry` rejected with a stable pre-
  inference status, and correct README/CONFIG/API text that still implies the
  canonical pipeline executes it. State precisely that future activation would
  require a governed scientific-stage order and in-memory refactor.
- Support only bounded in-memory `annotated` visualization streams in the live
  service at verbosity 3. Reject `panoptic`, unknown renderers, malformed rules,
  unsafe IDs and too many streams before inference. Detectron2 remains absent.
- Maximum service visualization streams: 8. Logical IDs must match a bounded
  safe pattern and may not contain path separators or user filenames.
- Lower verbosity must not render visualization/debug outputs solely for
  response enrichment. Add a service/core execution flag or equivalent that
  preserves legacy CLI behavior while making L0–L2 skip render-only work.

### Exact overlap representation

At L3 add a deterministic per-object uncompressed COCO-style RLE record:

```text
mask_rle = {
  encoding: "coco_rle_uncompressed",
  size: [height, width],
  order: "column-major",
  counts: [background_run, foreground_run, ...]
}
```

- Round-trip must reproduce every source mask bit exactly, including overlap,
  disconnected components, empty-edge runs and deterministic bytes/JSON.
- `instance_id` associates each RLE with the L2/L3 object, YOLO line and
  identity PNG value. ZIP manifest must carry equivalent structured data; do
  not duplicate a second unbounded mask payload without need.
- Encode through bounded streaming/chunked runs. Defaults: at most 250,000 RLE
  runs per object and 1,000,000 total runs per response. Exceeding either limit
  fails before runaway list/base64/ZIP growth with stable
  `response_too_large`; never silently omit overlap truth.

## Binding resource and artifact policy

Preserve existing upload/config/pixel/deadline/queue/response limits and add:

```text
max_image_width                 8192
max_image_height                8192
max_objects                     256
max_visualization_streams       8
max_response_artifacts          64
max_debug_artifacts             48
max_single_artifact_bytes       33554432   # 32 MiB
max_total_raw_artifact_bytes    134217728  # 128 MiB
max_mask_rle_runs_per_object    250000
max_mask_rle_runs_total         1000000
min_host_available_bytes        2147483648 # 2 GiB
min_shm_free_bytes              67108864   # 64 MiB
```

All are operator settings with validated positive values and documented env
names following `SLAIF_ZAP_IT_*`. Request YAML cannot change them.

- Enforce width/height immediately after image header decode and before full
  array allocation where Pillow permits.
- Enforce object/RLE limits before full response construction.
- Replace the unbounded `MemoryArtifactSink` service use with a bounded sink or
  budget tracker. Refuse new debug artifacts once count/per-artifact/total raw
  byte limits would be exceeded; stable error, no silent truncation.
- Remove post-hoc artifact slicing. Output parity is all-or-error.
- Prepare raw artifacts once. For JSON, account for exact/upper-bound base64
  expansion before encoding. For ZIP, write raw prepared bytes directly and a
  data-free manifest rather than constructing and decoding a duplicate base64
  JSON payload first.
- Check host-available RAM and `/dev/shm` at request admission and readiness.
  Use sanitized 507 errors (`insufficient_memory`, `insufficient_shm`) without
  paths or host internals. No persistent-disk fallback.
- The 120-s deadline remains honest drain-before-gate-release semantics for
  synchronous CUDA; do not claim hard CUDA cancellation.

## Metrics and logging policy

Add `prometheus-client` to the service extra and pinned GPU lock, using a custom
registry without default process collectors. Expose loopback-only `/metrics`
as Prometheus text. Metrics are process-local and reset on restart.

Allowed metrics, with only finite labels:

- request counts by stable outcome/error code;
- successful completion counts by verbosity and `json|zip`;
- busy/not-ready/timeout/cancel/response-limit counters;
- active-inference gauge and readiness gauge;
- request/inference/serialization duration histograms with fixed buckets;
- response-byte, object-count and artifact-count histograms;
- current Torch GPU allocated/reserved byte gauges for logical `cuda:0` when
  available, without physical-host paths or process identifiers.

Never use request ID, config digest, filenames, raw labels/prompts/answers,
model-cache paths, arbitrary error text or user values as metric labels. Keep
runtime logs to safe startup/access/status facts; no raw bodies/content or host
filesystem paths. Add exact privacy/cardinality tests and live scrape evidence.

## Required implementation scope

1. Implement and document the parity matrix and exact L0–L3 artifact catalog.
2. Add bounded overlap RLE and schema/OpenAPI/JSON/ZIP parity.
3. Enforce typed visualization policy and L3-only render execution for service
   while preserving CLI behavior.
4. Make debug names opaque and content-free; bound debug generation at the
   sink, not only at serialization.
5. Add all operator resource settings and admission/readiness checks above.
6. Refactor JSON/ZIP preparation for early exact/bounded size failure and zero
   partial response/workspace residue.
7. Add safe custom-registry Prometheus metrics and `/metrics` documentation.
8. Harden inference, serialization, shutdown, low-memory, low-shm, timeout and
   cancellation recovery so the resident registry remains usable.
9. Add alternating-state tests for different label maps, visualization/debug
   policies, images and formats; resident weights may persist, request-derived
   prompts/masks/artifacts/warnings may not.
10. Publish `docs/SERVICE-DATASHEET.md` with purpose, non-goals, exact API
    levels, supported/unsupported stages, hardware/software matrix, limits,
    measured latency/RSS/VRAM/response data, metrics, privacy/persistence,
    evidence, limitations and deployment prerequisites.
11. Update README, API, CORE, CONFIG, SECURITY, TESTING and RUNBOOK only where
    necessary for the implemented contract. Preserve legacy CLI/video behavior.

## Required CPU/static verification

- Complete canonical CPU suite with exact pass/skip/warning/coverage counts.
- Focused RLE round-trip/property tests: empty edge runs, all-zero/all-one,
  disjoint/disconnected, overlap, checkerboard run-limit failure, dimensions,
  ID association, deterministic JSON/ZIP and fuzzed bounded small masks.
- Visualization policy/execution-spy tests proving L0–L2 skip rendering and L3
  supports only bounded `annotated` streams.
- Geometry pre-inference rejection and documentation consistency tests.
- Artifact-budget boundary tests at count, per-item, total raw, base64 and ZIP
  sizes; prove no post-hoc truncation and no partial sink residue.
- Image dimension/object/RLE/host-memory/shm settings and boundary tests.
- Failure/cancel/serialization/shutdown tests followed by successful recovery.
- Alternating A/B/A request-state isolation tests covering resident CLIP label
  refresh, masks, artifacts, warnings and config digests.
- Metrics values/cardinality/privacy tests; scrape must contain only the finite
  label domains above.
- Malformed image/YAML/multipart, decompression, path/model/device/renderer,
  unsafe artifact-name and serialization property tests with bounded fixtures.
- Ruff format/lint, shell syntax, compile, wheel build/import, OpenAPI/schema,
  docs-link/claim, secret/large-artifact and exact-diff inspection.
- GitHub CI matrix and CodeQL: every expected check present and SUCCESS on
  implementation and final report heads.

## Required live GPU1/service evidence

Use only synthetic/redistributable fixtures and the measured resident profile.
Freshly re-verify GPUs/processes, UUID mapping, ports, RAM and `/dev/shm` before
each activation. Prefer verified-free 17891, then 23654. Capture sanitized
before/loading/ready/load/failure/restart/final snapshots.

1. Clean start: one IPv4 loopback listener, one process/worker/inference,
   physical GPU1 visible only as logical `cuda:0`, genuine 503→200 readiness.
2. Release-candidate E2E: L0–L3 JSON/ZIP, L3 annotated visualization, exact RLE
   decode/ID/area/dimension checks, metrics scrape and optional bearer-auth
   unauthorized/authorized behavior. BLIP3/panoptic/geometry reject before
   inference.
3. Bounded sequential load: 32 requests alternating two synthetic images, two
   materially different label maps and JSON/ZIP/L0–L3 combinations. Record
   per-request latency/response bytes/object/artifact counts; report p50/p95/
   max and state hashes without raw labels/config.
4. State isolation: explicit A/B/A sequences prove class maps, YOLO, masks,
   visualizations, warnings, RLE and prompt embeddings return to A with no B
   residue.
5. Controlled overlap: five rounds using the operator-only delay hook; exactly
   one active inference, deterministic follower 503+Retry-After, no rejected-
   request allocation and correct busy metrics.
6. Failure matrix: invalid input, injected inference/serialization failure,
   deadline, client disconnect, response/artifact/RLE limit, low-host-memory and
   low-shm simulations; every case followed by a successful request and zero
   request residue. Do not try to exhaust real host RAM/shm.
7. Resource sampling: baseline, ready, every fourth request, peak observed and
   post-stop process RSS; GPU1 used plus in-process allocated/reserved gauges;
   `/dev/shm` children; metrics counters. Evidence is a bounded test, not soak.
8. Stop/restart: graceful stop, port/GPU cleanup, restart, one L3 JSON+ZIP
   request, final stop. End with no listener/process, GPU1 near idle, only the
   protected GPU0 process and empty service shared-memory root.
9. Inspect actual logs and metrics for raw images/YAML, filenames, labels,
   prompts/answers, request IDs as labels, credentials, headers, filesystem/
   cache paths, tracebacks and unbounded cardinality.

## Acceptance criteria

1. `docs/OUTPUT-PARITY.md` accounts honestly for every source/docs output and
   agrees with implementation/tests.
2. L3 overlap RLE round-trips exact per-object masks and agrees bijectively with
   object, YOLO and identity IDs in JSON and ZIP.
3. Geometry and panoptic remain explicit pre-inference unsupported capabilities;
   supported annotated visualization is bounded and L3-only in service mode.
4. Artifact/RLE/image/object/RAM/shm/response limits fail before runaway work,
   never silently truncate, and leave no residue/corrupted registry state.
5. Metrics/logs are content-safe and low-cardinality with exact tests and live
   inspection.
6. Alternating and repeated requests prove request-state isolation and no
   obvious unbounded RSS/VRAM growth in the documented 32-request window.
7. Failure, timeout, cancel, busy, serialization and resource-rejection paths
   recover to a successful request with one inference maximum.
8. Service datasheet and runbook exactly match measured hardware, software,
   limits, performance, supported stages and non-production limitations.
9. Physical GPU0/unrelated workloads remain untouched; final service state is
   stopped and clean.
10. Full CPU/static/package and all six GitHub checks are SUCCESS on the final
    report head.
11. One Objective-005 branch/PR and report-only SELF contract hold; coding never
    merges.

## Non-goals and protected boundaries

- no geometry activation or new scientific stage;
- no Detectron2/panoptic or BLIP3 live enablement;
- no LAN/public exposure, TLS, proxy, gateway, firewall/VPN or customer data;
- no multi-worker/GPU, async job queue, persistence/history, training or model
  substitution;
- no system driver/CUDA/systemd/global credential or unrelated service change;
- no physical GPU0 use or mutation;
- no Docker/release/gateway integration (Objective 006);
- no claim of production soak, SLA, leak-proof operation or final release.

## Documentation and provenance

Add `docs/OUTPUT-PARITY.md` and `docs/SERVICE-DATASHEET.md`; update navigation,
API artifact/error/metrics catalog, exact limits, security/privacy, runbook and
tested hardware. Retain approved model/revision/license references without
weights, credentials or private cache paths. Correct every remaining claim that
geometry currently executes.

## Deferred human adjudication

- Decision: `NONE`

Geometry non-activation, annotated-only visualization, RLE format, resource
budgets and metrics are bounded reversible engineering decisions resolved by
the architecture and current evidence. No critical-register threshold is met.
External deployment, customer data, commercial/model licensing and final
release remain outside this objective and behind Objective-006 human gates.

## GitHub publication and report

- Create the required branch from remote main and exactly one Objective-005 PR
  with the required title. Coding never merges.
- Commit all implementation, dependencies, tests, docs/datasheet, exact order
  transcript and `oap/active` before the report.
- Push, create the PR, and require all six current-head checks SUCCESS.
- Capture literal implementation SHA.
- Create exactly one immutable `oap/reports/005-a-report.md` containing that SHA
  and `Report publication commit: SELF`.
- Final report commit changes only that report; first parent equals the literal
  implementation SHA. Push and verify remote head, parent, path, exact bytes and
  all report-head checks before response signaling.
- Report parity classification, exact limits, CPU/live commands and statuses,
  32-request table/aggregates, resource samples, state-isolation hashes,
  metrics/log inspection, GPU isolation, final stopped state, limitations and
  critical action `NONE`.

## Coding response

Send exact FIFO `OK` only after the remote PR, complete fresh evidence,
immutable SELF report and final stopped host state are verified. A truthful
partial/blocked report also signals. Coding never merges.
