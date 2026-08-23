# SLAIF ZAP-IT modernization and service architecture

## 1. Purpose

ZAP-IT (“Zero-shot Anything Pipeline for Image Tasks”) is already a meaningful
prototype rather than an empty repository. Its current code composes automatic
segmentation, open-vocabulary classification, optional visual-language
verification, geometric analysis, visualization, and YOLO export under YAML
configuration. The goal is to retain that research flexibility while making the
repository professionally maintainable and exposing the pipeline through a
small local API suitable for controlled SLAIF deployment.

The target is not a generic image-generation service and not a language model.
It is a deterministic orchestration service around existing computer-vision
models. It accepts one image and one ZAP-IT YAML configuration, runs the enabled
pipeline, and returns only evidence that the enabled modules actually produced.

This document is the human-readable architecture. `ARCHITECTURE-for-agents.md`
is the compact normative execution version. If they conflict, the active OAP
order must resolve the conflict; an agent must not silently select the weaker
interpretation.

## 2. Audited starting point

The repository currently contains:

- a command-line batch driver and reusable frame pipeline;
- preprocessing with optional region of interest and resizing;
- SAM2 automatic mask generation using a configurable generator;
- mask post-processing by area and bounding dimensions;
- CLIP zero-shot labeling with configured class prompts;
- optional BLIP3/XGen-MM question-answer verification and relabeling;
- optional Canny/Hough line and intersection extraction;
- stage visualization and image/video output;
- YOLO detection dataset export;
- dry-run paths and a substantial set of mocked unit tests;
- Conda-oriented Python 3.10 / PyTorch 2.3 / CUDA 12.1 environment guidance.

The current frame result already distinguishes a rendered image, final masks,
and a serialized mask view. However, several legacy modules assume filesystem
output directories, some debug/geometry paths write files directly, model and
service packaging are not yet separated, CI is absent, and public-facing
contracts are incomplete. Modernization must build on existing tests and
behavior; replacing the system wholesale would throw away useful work and make
regression detection harder.

## 3. Product boundary

### 3.1 In scope

The modernized repository owns:

- the ZAP-IT Python package and backward-compatible CLI;
- typed configuration and in-memory single-image pipeline APIs;
- model loading and lifecycle for SAM2, CLIP and optional BLIP3;
- post-processing, geometry, visualization and artifact rendering;
- deterministic object IDs, YOLO records and identity-mask PNGs;
- the local FastAPI service and its `/v1/completions` contract;
- CPU test suite, optional GPU integration suite, CI and CodeQL;
- local deployment on a verified unused loopback port;
- documentation, provenance, security, installation and service operation.

### 3.2 Out of scope until separately ordered

- changing the scientific meaning of SAM2/CLIP/BLIP3 outputs;
- training or fine-tuning models;
- uploading or redistributing model weights;
- public internet exposure, TLS, multi-tenant identity, billing or quotas;
- a hosted object store or persistent result history;
- asynchronous job queues or distributed inference;
- automatic use of physical GPU 0;
- arbitrary user-selected model repositories or `trust_remote_code` execution;
- claiming OpenAI `/v1/completions` wire compatibility beyond the documented
  ZAP-IT-specific endpoint path and envelope;
- video API processing in the first service release (existing CLI video support
  remains preserved).

## 4. Architectural principles

1. **Preserve scientific behavior before improving it.** Baseline tests and
   sanitized golden fixtures must characterize the current pipeline before
   invasive refactors.
2. **Separate core computation from I/O.** The service calls an in-memory core,
   not the batch CLI and not directory-oriented writers.
3. **Configuration is data, not authority.** Uploaded YAML may tune allowed
   pipeline parameters; it may not control host filesystem, devices, downloads,
   imports, commands, credentials or deployment.
4. **Physical GPU selection is an operational invariant.** This multi-GPU host
   dedicates physical GPU index 1 to the service. Visibility masking makes it
   logical `cuda:0` inside the process.
5. **Results are monotonic and honest.** Higher verbosity adds available detail;
   lower verbosity never invents fields or triggers an expensive optional model
   solely to fill a response.
6. **Raw request data is ephemeral.** CPU memory is preferred; `/dev/shm` is a
   controlled compatibility workspace, not persistent storage.
7. **One process, one GPU task initially.** CUDA models are expensive and not
   safe to duplicate through Uvicorn workers.
8. **CI is independent of scarce models and GPUs.** Model/GPU behavior has a
   separate explicit test tier.
9. **GitHub and OAP evidence govern change.** Every objective is reviewable,
   bounded, tested and merged only by the strategic agent.
10. **Human judgment is front-loaded.** Human Work Preloading (HWP) encodes intent,
    architecture, constraints, sequencing and acceptance before the autonomous
    loop so routine progress does not depend on repeated human supervision.
11. **Residual judgment is preserved, not used to stop.** Human Judgment
    Postloading (HJP) requires the strategic agent to make rare consequential
    provisional decisions, record only genuinely material dilemmas in the
    append-only `CRITICAL.md`, and defer authoritative human adjudication to the
    stated pre-deployment or release gate.

## 5. Target system context

```text
Local client
  |
  | multipart/form-data
  | image + config YAML + verbosity + response_format
  v
127.0.0.1:<verified-unused-port>
  SLAIF ZAP-IT FastAPI service
    request guard / optional API key
    safe YAML parser + policy validator
    concurrency and deadline control
    in-memory pipeline engine
      persistent model registry
      preprocess -> SAM2 -> postprocess -> CLIP -> optional BLIP3 -> geometry
    deterministic result builder
    artifact renderer
      YOLO text
      uint16 identity PNG
      object/geometry metadata
      overlays and full artifacts
    JSON or ZIP serializer
  |
  | CUDA_VISIBLE_DEVICES=1
  v
Physical GPU index 1 only
```

The OpenCode strategic and coding agents are development/control-plane clients;
they are not part of the runtime request path.

## 6. Package decomposition

The exact module names may be adjusted by an OAP order after repository audit,
but responsibilities should converge on the following separation.

### 6.1 Configuration layer

Responsibilities:

- parse trusted operator configuration and untrusted per-request YAML through
  separate entry points;
- use `yaml.safe_load` only;
- validate a typed schema with bounds and defaults;
- produce an immutable/effectively immutable normalized configuration;
- calculate a stable digest for provenance and cache/model-state decisions;
- distinguish API-safe algorithm fields from batch-only/path/deployment fields;
- reject unknown dangerous fields rather than silently honoring them.

The API allowlist should cover parameters already understood by preprocessing,
mask generation, mask post-processing, CLIP labels, BLIP verification, geometry,
and visualization. It must not permit input folders, output roots, debug paths,
video paths, URLs, devices, environment variables, model downloads, Python
symbols, commands or service settings.

### 6.2 Device guard

Responsibilities:

- inspect the visible CUDA environment before model initialization;
- require one visible accelerator when GPU mode is configured;
- report sanitized device name, physical UUID/PCI mapping and VRAM;
- compare the visible device with an operator-pinned expected GPU UUID when
  available;
- fail readiness rather than silently falling back to physical GPU 0;
- expose a CPU mode only for tests or explicitly supported degraded operation.

The service process is launched with:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
```

Inside the process, PyTorch sees the selected physical card as `cuda:0`. Calling
`cuda:1` after masking is a bug. Every model and subprocess must receive the same
visibility environment.

### 6.3 Persistent model registry

SAM2, CLIP and BLIP3 initialization is substantially more expensive than one
request. A process-local registry should hold model objects and immutable model
metadata across requests while keeping request-specific masks, embeddings,
answers and artifacts isolated.

The registry should:

- pin approved model identifiers and revisions in operator configuration;
- separate model weights/cache location from repository and request workspace;
- load lazily or eagerly under one well-defined lifecycle;
- reuse SAM2/CLIP models across label/config changes where safe;
- rebuild only request/config-specific prompt embeddings as needed;
- bound any configuration-derived cache (initially one or a very small LRU);
- expose readiness and load errors without raw paths/secrets;
- never allow uploaded YAML to select a new remote repository;
- avoid duplicate model loading through multiple server workers.

BLIP3 currently uses remote model code and several compatibility patches. That
path must be pinned, reviewed and tested. If the complete SAM2+CLIP+BLIP3 profile
does not fit reliably in the verified GPU-1 VRAM, the service must either use a
safe explicit load/unload policy or reject BLIP3-enabled configurations. It must
not spill to GPU 0 or pretend the stage ran.

### 6.4 In-memory pipeline engine

The engine accepts a decoded RGB image array and validated configuration and
returns a typed `PipelineResult`. It should adapt existing module functions
rather than duplicate algorithms.

A result should preserve:

- original and effective dimensions, ROI and scale mapping;
- final object masks and stable request-local instance IDs;
- per-object area, bbox and available SAM quality fields;
- available CLIP label/score and BLIP answer/relabel evidence;
- available geometry lines/intersections;
- normalized warnings and stage timings;
- model/config/service provenance;
- optional stage renderings and full-detail mask representations.

No core method should require a caller-controlled output directory. Legacy
filesystem-oriented functions should receive an artifact sink abstraction or a
service-owned ephemeral workspace. Existing CLI adapters may continue writing to
folders using a filesystem sink.

### 6.5 Artifact sinks

Three implementations are useful:

1. `MemoryArtifactSink`: bytes/arrays/structured records held in RAM.
2. `ShmArtifactSink`: unique per-request directory under `/dev/shm` for libraries
   that require paths or for bounded large ZIP assembly.
3. `FilesystemArtifactSink`: legacy CLI/batch behavior only; never selected by an
   uploaded API config.

All sink methods use logical artifact names, not user paths. The service chooses
the sink. A `finally` block removes every shared-memory workspace after response
serialization, cancellation or error.

### 6.6 Result renderer

The renderer is pure or nearly pure: it converts `PipelineResult` into level-
gated public artifacts. It must not rerun models.

Outputs include:

- normalized YOLO detection lines;
- a 16-bit identity PNG;
- per-object JSON records;
- overlap-preserving per-object mask encodings at full level;
- stage overlays/visualizations already produced or renderable from results;
- geometry tables and metadata;
- a manifest with hashes, sizes, media types and provenance.

### 6.7 FastAPI application

The API layer owns transport, validation, authentication, limits, concurrency,
error mapping and lifecycle. It delegates inference to the engine in a bounded
thread/executor because PyTorch execution is synchronous.

The initial server uses one Uvicorn process/worker. An `asyncio.Semaphore(1)` or
single-thread inference executor serializes GPU execution. A small bounded queue
may be added only with explicit behavior; uncontrolled waiting is not acceptable.

## 7. `/v1/completions` request contract

### 7.1 Why this path

The requested path is `/v1/completions`. The service may use an OpenAI-like
completion envelope to ease gateway/client integration, but the request is
multipart and image-specific. Documentation must call it a ZAP-IT completion,
not a compatible text-completion implementation.

### 7.2 Multipart fields

```text
image            required file; exactly one JPEG/PNG/WebP initially
config           required UTF-8 YAML file; exactly one
verbosity        required/default 0; integer 0..3
response_format  optional json|zip; default json
model            optional fixed identifier; unknown values rejected
stream           omitted/false only in v1
```

A later contract may accept inline base64 JSON, but supporting two request forms
in the first version adds ambiguity without product value.

### 7.3 Limits

Operator settings define at least:

- maximum encoded image bytes;
- maximum decoded width, height and total pixels;
- maximum YAML bytes, depth, aliases/collection size and string lengths;
- maximum generated objects;
- maximum per-stage and total execution time;
- maximum JSON/ZIP response bytes;
- maximum artifacts and debug artifacts;
- maximum queued/running requests;
- minimum free `/dev/shm` and host memory thresholds.

The application reads at most `limit+1` bytes and rejects early. Decompression
bomb protection is required before allocating unbounded image arrays.

## 8. Completion response

### 8.1 Stable envelope

A JSON response should resemble:

```json
{
  "id": "cmpl_zapit_...",
  "object": "text_completion",
  "created": 0,
  "model": "slaif-zap-it",
  "choices": [
    {
      "index": 0,
      "text": "0 0.500000 0.500000 0.250000 0.250000\n",
      "finish_reason": "stop"
    }
  ],
  "usage": null,
  "zap_it": {
    "schema_version": "1",
    "verbosity": 0,
    "image": {"width": 0, "height": 0},
    "class_names": ["..."],
    "config_sha256": "...",
    "artifacts": []
  }
}
```

Exact fields are frozen by a dedicated API objective and OpenAPI tests. `usage`
may be `null` because this is not token inference; inventing token counts would
be misleading.

For `response_format=zip`, return `application/zip` assembled in RAM or shared
memory. It contains a stable `manifest.json` and the same logical content.

### 8.2 Verbosity 0 — YOLO

The completion text contains one line per final object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All four coordinates are normalized to the original input image dimensions and
written with fixed precision. Class IDs are deterministic and map to returned or
documented class names derived from the effective config. Empty detection is an
empty string, not a fabricated object. The first contract should not append a
confidence sixth column because ordinary YOLO detection labels use five fields;
quality scores are available at higher levels.

### 8.3 Verbosity 1 — identity mask

Adds `identity-mask.png`:

- grayscale 16-bit PNG;
- dimensions exactly equal the original image;
- pixel 0 means background;
- values 1..N identify final response objects;
- disconnected components belonging to one object retain the same value;
- object IDs are request-local and have no cross-request identity meaning.

SAM masks may overlap, while one PNG pixel has one value. Therefore the renderer
must define a deterministic winner policy. Recommended design: rank final objects
by an explicit stable key and assign each contested pixel to the highest-ranked
object. The exact key is frozen in the API contract and emitted in provenance.
Full verbosity also carries overlap-preserving per-object masks so the raster
projection is not misrepresented as lossless.

### 8.4 Verbosity 2 — object records

Adds one record per final object, limited to produced fields:

- `instance_id` matching PNG value;
- class ID/name and label source;
- pixel and normalized bbox;
- area in pixels and optional fraction;
- SAM `predicted_iou`, `stability_score` and other retained quality fields;
- CLIP label/score where CLIP ran;
- BLIP3 answer/relabel evidence where BLIP3 ran;
- geometry lines/intersections where enabled;
- warnings about omitted/unsupported fields.

Missing stages produce absent/null fields plus stage status; they do not receive
plausible defaults.

### 8.5 Verbosity 3 — full

Adds every bounded, safe output the enabled pipeline can provide:

- normalized effective config or a policy-redacted representation plus digest;
- full per-stage statuses and timings;
- final and selected intermediate object metadata;
- overlap-preserving per-object masks (for example COCO RLE or individual PNGs);
- stage visualizations and final overlay;
- geometry TSV-equivalent structured data;
- available BLIP answers and CLIP prompt/class evidence;
- model identifiers/revisions, package/service version and device metadata;
- warnings, applied limits and deterministic-order information;
- optional debug artifacts explicitly allowed by server policy.

“Full” remains bounded. It never exposes host paths, environment, secrets, model
weights/cache, arbitrary debug dumps or raw stack traces.

## 9. Identity, ordering and overlap

Instance IDs must be deterministic for the same code/model/config/input within
the determinism guarantees of upstream models. A candidate ordering key should
be based only on public result fields and original candidate index, e.g. label
class order, selected score/quality, descending area, bbox coordinates and final
stable tie-breaker. The exact order needs characterization before freezing.

The object list, YOLO lines and identity mask must use the same ordering. Tests
must cover:

- no objects;
- one object;
- multiple disjoint objects;
- one object with disconnected components;
- overlapping masks;
- equal-score/tie cases;
- object count near PNG limit;
- resize/ROI mapping back to original dimensions;
- deterministic bytes across repeated fake-engine runs.

## 10. Memory and `/dev/shm`

### 10.1 Normal path

The request body is bounded, decoded into CPU memory and passed as a NumPy/PIL
object. Intermediate masks/metadata remain in memory. PNG and JSON are generated
into `BytesIO`. The service does not persist images, YAML or results.

### 10.2 Compatibility path

Some current geometry/debug/output code writes by path. During migration, the
service may create:

```text
/dev/shm/slaif-zap-it/<opaque-request-id>/
```

The root is operator-created mode 0700. Per-request directory is 0700 and files
0600. The ID is random and contains no filename/user data. The service checks
available shared memory before work, never follows client symlinks/paths, and
removes the directory recursively in `finally`.

If `/dev/shm` is missing/too small, the service returns a stable error. It must
not silently fall back to persistent `/tmp` or the repository unless the
operator explicitly configures another ephemeral RAM-backed mount.

### 10.3 Resource release

Request references are dropped after serialization. Inference uses
`torch.inference_mode()`. Selective `torch.cuda.empty_cache()` is not a routine
per-request solution and should be used only when profiling proves value; it can
hurt performance and does not free live tensors. Memory leaks are tested through
repeated-request RSS/VRAM observation.

## 11. GPU-1 isolation

The machine is multi-GPU. “GPU1” means the physical device currently reported by
`nvidia-smi` index 1, not logical PyTorch index after visibility masking. The
strategic first order must capture:

- hostname and OS;
- `nvidia-smi` index, UUID, PCI bus, exact model and VRAM for every GPU;
- existing compute/graphics processes on every GPU;
- driver and CUDA runtime;
- PyTorch/CUDA versions in the intended environment;
- whether the stated 22/24 GB GPU-1 memory is real;
- whether `/dev/shm` and a candidate port are available.

Service environment:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

The process then uses `torch.device("cuda:0")`. Startup should refuse readiness
if more than one GPU is visible in strict mode or if the visible UUID differs
from `SLAIF_ZAP_IT_EXPECTED_GPU_UUID`.

No order may terminate another process, reset a GPU, modify persistence/power
mode, install/change a system driver, or use GPU0 unless a human-approved isolated
order explicitly authorizes it. GPU integration tests snapshot processes and
memory before and after. They run serially and clean only processes they started.

## 12. Concurrency and lifecycle

Initial service constraints:

- one process;
- one Uvicorn worker;
- one loaded model registry;
- one active inference request;
- bounded queue of zero or a very small configured number;
- deterministic busy response (`429` or `503`, frozen by contract);
- explicit readiness while models load;
- graceful shutdown waits/cancels safely and removes shared-memory workspaces.

Threading can improve transport responsiveness but not bypass the GPU semaphore.
Multiple worker processes would each load weights and are forbidden until
separate measured architecture work.

## 13. Security model

### 13.1 Threat inputs

Both uploaded image bytes and YAML are hostile. Threats include decompression
bombs, malformed codecs, YAML resource attacks, path traversal, arbitrary model
or code loading, oversized results, GPU/CPU memory exhaustion, long-running
inference, debug artifact leakage and log injection.

### 13.2 Controls

- bounded streaming reads and decoded-pixel checks;
- safe YAML loader, structural limits and typed allowlist;
- no request paths/URLs/model revisions/devices/commands;
- model/cache/network downloads are operator-controlled startup operations;
- local loopback binding by default;
- optional constant-time API-key validation before LAN exposure;
- stable sanitized errors and structured logs without raw content;
- opaque request IDs and bounded metric labels;
- shared-memory permissions and guaranteed cleanup;
- dependency/license/provenance review, especially remote model code;
- single-request resource gate and deadlines;
- no debug mode from client unless separately allowlisted and bounded.

## 14. Observability

Safe logs include request ID, route, status, verbosity, response format, image
size/dimensions, config digest, stage names/status, object/artifact counts,
timings, queue/busy state and sanitized device/service versions. They exclude raw
image/config, labels/prompts if customer-sensitive, answers, filenames, paths,
headers and credentials.

Health endpoints:

- `/healthz`: process/event-loop health; no model load guarantee;
- `/readyz`: configuration valid, expected GPU visible, model registry ready,
  shared-memory root usable;
- optionally `/metrics`: private/local Prometheus metrics with bounded labels.

## 15. Professional repository target

The modernization should move toward the standards demonstrated by other SLAIF
services without copying domain-specific claims:

- `pyproject.toml` with package metadata and separated base/model/dev extras;
- importable `src/` package and stable CLI entry point;
- reproducible environment/lock strategy compatible with GPU stack;
- `.github/workflows/ci.yml` and CodeQL;
- Ruff format/lint, pytest, coverage baseline/ratchet and optional typing;
- CPU tests requiring no model downloads;
- API, auth, configuration, service and artifact tests;
- Docker/Compose or systemd only after host-native GPU service is proven;
- README, installation, configuration, API contract, service datasheet,
  provenance, security, contributing and third-party notices;
- pinned model revisions/checksums outside Git where licensing permits;
- explicit limitations and tested hardware matrix.

Modernization is staged to avoid mixing packaging, scientific refactor, API and
live GPU deployment into one unreviewable PR.

## 16. Test architecture

### 16.1 Tier 0 — static/package

Syntax/import/package build, Ruff format/check, metadata, docs links, secret scan,
OpenAPI schema and configuration examples.

### 16.2 Tier 1 — CPU unit

Pure preprocess/postprocess/geometry/config/result/YOLO/identity-mask functions,
using fakes for model modules. No network, model download or CUDA.

### 16.3 Tier 2 — CPU service contract

FastAPI TestClient/HTTPX against fake engine: multipart validation, levels,
JSON/ZIP bytes, identity PNG pixel values, errors, auth, limits, concurrency,
cancellation and `/dev/shm` cleanup.

### 16.4 Tier 3 — local GPU integration

Explicit opt-in marker/environment. Force physical GPU1 visibility. Verify
SAM2/CLIP and optional BLIP3 separately, then a full allowed config on a small
redistributable image. Record exact device UUID, peak VRAM, latency and stage
outputs. Never run in public GitHub-hosted CI.

### 16.5 Tier 4 — deployment smoke

Start local service on verified unused loopback port; call health/readiness and
all verbosity levels; inspect no persistent residue, no GPU0 process, stable
repeated behavior, graceful restart and bounded response sizes.

GitHub CI runs tiers 0–2. A required GPU check needs a deliberately managed
self-hosted runner and must not be enabled casually on a shared workstation.

## 17. Migration roadmap

### Objective 000 — professional baseline

Inventory and characterize current behavior; repair/run existing CPU tests;
establish package metadata, lint/format, CI/CodeQL, coverage baseline,
documentation/security/provenance. Do not add API or alter GPU/services.

### Objective 001 — in-memory core and result contract

Extract typed in-memory single-image engine, artifact sink abstraction, stable
object result, YOLO renderer and uint16 identity mask with exhaustive CPU tests.
Preserve legacy CLI.

### Objective 002 — API surface with fake engine

Implement `/v1/completions`, schemas, limits, errors, optional auth, JSON/ZIP
levels, health/readiness and CPU contract tests. Bind only in test process; no
live GPU deployment.

### Objective 003 — GPU1 runtime qualification

Build/verify repo-local environment on the actual host; pin GPU1 UUID; verify
model revisions/licenses; profile SAM2/CLIP/BLIP3 memory; run bounded live tests;
select and document a free loopback port. Do not touch GPU0.

### Objective 004 — local service activation

Create operator config and service launcher/systemd unit if warranted; start one
worker on verified port and GPU1; run repeated E2E and cleanup/resource tests;
prove rollback. No public exposure.

### Objective 005 — full artifact parity and hardening

Complete full verbosity, overlap-preserving masks, geometry/visualization/debug
parity, resource limits, metrics, cancellation, failure injection and service
datasheet.

### Objective 006 — packaging/gateway/release

Only after local acceptance and before crossing any applicable open
`CRITICAL.md` gate: prepare container/Compose if useful, SLAIF gateway route,
key/auth policy, release/versioning, installation skill and distribution/license
review. Actual external deployment or final release requires latest human `ACCEPTED`
disposition for all applicable entries.

The strategic agent may split or reorder objectives based on verified facts, but
must preserve dependencies and one coherent PR per numeric objective.

## 18. OAP and OpenCode execution

Two OpenCode clients run in separate directories. The coding shell wrapper blocks
on `control.fifo` and starts a fresh foreground `opencode run` for each order.
The strategic client is one persistent interactive TUI. Coding model/variant
selectors are passed to `opencode run`. Strategic model/variant selectors are
materialized into a private mode-0600 runtime agent configuration and selected by
name, because the TUI does not accept a variant flag. Session sharing is disabled.
Exact two-byte `OK` messages synchronize; GitHub and versioned OAP files carry
truth.

`NNN-a` creates one branch/PR. Lettered continuations amend that PR. Coding
publishes a report-only final commit and never merges. Strategic independently
verifies GitHub, CI, security, GPU evidence and acceptance; only strategic may
merge. A crash is recovered from active order, GitHub and immutable reports, not
from remembered conversation.

## 19. Human Work Preloading and Human Judgment Postloading

### 19.1 Why the human is outside the routine loop

This OAP design deliberately concentrates human work before execution. The
constitution, this architecture, the compact agent architecture, roadmap, work-
order templates, security law and acceptance criteria are not merely context;
they are preloaded human engineering judgment. They remove the need for a human
to act as the normal scheduler, architectural memory, reviewer and tie-breaker in
every agent turn.

```text
Human Work Preloading
  -> durable intent / architecture / constraints / roadmap / acceptance
  -> persistent strategic agent
  -> bounded coding agent rounds
  -> PR + CI + evidence
  -> strategic correction or merge
```

The strategic agent therefore has a **decision duty**. It must not stop merely
because a security boundary, trust-model choice, deletion policy, authorization
model or deployment design is consequential and it would prefer the human to
choose. It must inspect the available law and evidence, compare alternatives,
choose the best provisional design, prefer least privilege and reversibility,
require tests and mitigation, and continue the roadmap.

### 19.2 The narrow postloading mechanism

Some decisions remain legitimately worthy of later human authority. HJP preserves
those decisions through `CRITICAL.md`, formally the **Deferred Human Adjudication
Register**. The register is an honest ledger of consequential autonomous
judgments, not a generic vulnerability list and not a dumping ground for
uncertainty.

A new entry is permitted only when all five register conditions hold:

1. existing human instructions, constitution, architecture, order and evidence do
   not resolve the issue;
2. materially different alternatives remain and one must be selected;
3. a wrong choice could materially affect security, authorization, privacy, data
   integrity/loss, trust, public exposure, deployment or release safety;
4. one provisional choice can be implemented and tested safely without crossing
   a non-delegable external or production boundary; and
5. a competent human could plausibly reject or materially change it before
   deployment.

This high threshold is intentional. The following do **not** justify entries:
normal refactors, implementation bugs, test failures, TODOs, known limitations,
style, ordinary dependency choices, speculative concerns, low-impact reversible
tradeoffs, or repeated references to a dilemma already registered. Those remain
ordinary strategic and coding work.

### 19.3 Decision, append and merge flow

```text
material unresolved dilemma
  -> strategic investigates and decides provisionally
  -> strategic states strongest argument that its decision is wrong
  -> strategic defines assumptions, blast radius, mitigations, rollback and gate
  -> exact CRIT-NNNN entry is included in current/same-PR continuation order
  -> coding appends exact bytes with append_critical.py
  -> strategic verifies append-only integrity, implementation and CI
  -> strategic may merge for continued development
  -> human adjudicates before the stated deployment/release boundary
```

Strategic authors the entry; coding only appends exact ordered bytes. Agents may
never edit, delete, reorder, weaken, close or mark a prior entry human-approved.
Later autonomous mitigation is appended as a separate update and does not close
the human gate. One underlying dilemma receives one entry rather than one entry
per round.

An open entry changes the semantics of merge:

```text
strategic merge
  = best available bounded provisional engineering decision,
    safe for continued development and supported by current evidence

strategic merge
  != human approval for production deployment across the registered gate
```

Thus green CI and strategic acceptance may allow the autonomous project to reach
an MVP or release candidate without interrupting the human repeatedly. Before
production deployment, public exposure, real customer data, irreversible
production mutation, security-policy relaxation at an external boundary, or
final release, every applicable entry must have a latest human disposition of
`ACCEPTED`; deferred, rejected or change-required entries remain blocking.

### 19.4 Non-delegable actions remain non-delegable

“Forced to decide” does not mean “authorized to perform every action.” Strategic
must still stop before a boundary that requires human-exclusive facts or legal/
organizational authority, or before an irreversible external action not already
pre-authorized. It may choose and implement the recommended design in a local,
reversible, testable environment and record the dilemma; it may not use a
`CRITICAL.md` entry as a waiver to expose an unauthenticated public service,
process real customer data, destroy production data, disable mandatory controls,
expand external privileges, or release contrary to the stated human gate.

## 20. Open decisions requiring live evidence

The constitution deliberately does not guess:

- exact GPU-1 model/VRAM/UUID and current workload;
- host OS/driver/CUDA/PyTorch compatibility;
- exact OpenCode provider/model/variant for each role;
- current test pass/fail/coverage state;
- target Python/PyTorch/SAM2/Transformers version set;
- whether BLIP3 co-resides within GPU1 memory safely;
- exact unused loopback port;
- actual `/dev/shm` capacity and request/result limits;
- final overlap winner key and public schema details;
- whether Docker or host systemd is operationally preferable;
- whether API-key auth is required for loopback-only initial deployment.

The strategic agent resolves these in bounded orders with evidence. If a choice
meets all five `CRITICAL.md` conditions, it decides provisionally and records the
rare deferred adjudication rather than stopping. Silence is not permission to
assume, and the register is not permission to avoid ordinary analysis.
