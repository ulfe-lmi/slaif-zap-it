# Local service runbook

This runbook operates the tested local ZAP-IT service from one host process.
It defaults to `127.0.0.1` and may use one human-authorized explicit RFC1918
address with mandatory bearer authentication. It exposes one operator-selected
physical GPU as logical `cuda:0`. Below 24,576 MiB the historical 11 GB
RTX 2080 Ti uses the live-qualified sequential stage-boundary lifecycle; at or
above 24,576 MiB the assigned RTX 3090 uses the live-qualified all-resident
lifecycle. The Objective 009 matrix covers all four supported profiles. This is
bounded local research evidence, not a public/WAN, customer-data,
production-release, SLA, accuracy, or commercial-license runbook. Geometry,
panoptic, deployment and release gates remain separate.

The native `/v1/completions` route documented below is the private operator,
research, and debugging surface. It is not OpenAI Completions compatibility,
not gateway-facing, and not the general-public SLAIF contract. The separate
`/v1/responses` route is the future gateway/public compatibility facade; the
gateway repository and public deployment remain later work.

## Before every activation

Use the repo-owned `.venv-gpu` environment and verify the host without changing
any other process:

```bash
cd "$HOME/opencode-work/slaif-zap-it"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
ss -H -ltn
df -h /dev/shm
```

Confirm that the explicitly assigned physical index has the expected UUID,
PCI/model/VRAM facts and no compute process; all other devices' unrelated
processes are unchanged. The candidate port is not a
reservation: `scripts/serve_local.sh start` re-verifies it with `ss` and a
transient bind immediately before the service opens it.

Prepare the operator environment outside Git. The example contains no secret:

```bash
install -d -m 700 "$HOME/.config/slaif-zap-it"
install -m 600 deploy/service.env.example \
  "$HOME/.config/slaif-zap-it/service.env"
${EDITOR:-vi} "$HOME/.config/slaif-zap-it/service.env"
set -a
. "$HOME/.config/slaif-zap-it/service.env"
set +a
```

Set the explicit physical index and matching fresh UUID, then set
`SLAIF_ZAP_IT_MODEL_CACHE_ROOT` to the operator cache containing the pinned
SAM2, CLIP and BLIP3 snapshots. Do not add model IDs, revisions, devices,
URLs, paths, dtypes, or credentials to request YAML; those are startup policy. Keep the
port to a freshly verified free value when the commands below need to reference
`$SLAIF_ZAP_IT_PORT`. If automatic selection is used instead, capture the port
reported by `start` and export it before issuing curl/smoke commands.

## Start, inspect, and stop

```bash
scripts/serve_local.sh start
scripts/serve_local.sh status
scripts/serve_local.sh logs
curl --fail-with-body http://127.0.0.1:"$SLAIF_ZAP_IT_PORT"/healthz
curl --fail-with-body http://127.0.0.1:"$SLAIF_ZAP_IT_PORT"/readyz
scripts/serve_local.sh stop
```

`start` creates only a mode-0700 runtime directory under
`/dev/shm/slaif-zap-it`, with mode-0600 pid/log files. It returns after
`/healthz` is live; `/readyz` remains `503 not_ready` while the resident model
registry loads and becomes `200 ready` only after the registry, device guard,
and shared-memory checks pass. The launcher uses one process, one Uvicorn
worker, and one active inference slot. `stop` signals only the PID whose command
line matches this checkout, waits for graceful shutdown, removes the pid/log
files, and leaves the service stopped. `restart` is a controlled stop followed
by a fresh port/device preflight.

## Authenticated private-LAN activation

This mode requires explicit human authorization. Verify the host's real RFC1918
interface and subnet, then install the private operator files without printing
the generated key:

```bash
.venv/bin/python scripts/install_private_lan_service.py \
  --host 10.8.132.76 --cidr 10.8.132.0/24 --port 17891 \
  --physical-gpu-index 0 \
  --expected-gpu-uuid GPU-a91444df-4e87-011e-3347-9b3a4b9f9575
systemctl --user daemon-reload
systemctl --user enable --now zap-it-lan.service
```

The installer creates or updates
`~/.config/slaif-zap-it/service.env` as mode 0600 and preserves an existing
valid `SLAIF_ZAP_IT_API_KEY`; it never prints the key. It installs only the
user-level `zap-it-lan.service` unit. Inspect status with
`systemctl --user status zap-it-lan.service` and verify `ss` shows the exact
host address, never `0.0.0.0`. The persistent LAN profile uses model-control
mode `none`, so clients cannot load or unload models.

LAN clients receive the fixed key through an appropriately protected operator
channel and send `Authorization: Bearer <key>`. Requests without it receive
401. Interactive docs and OpenAPI return 404 on this listener. Health/readiness
remain content-free and unauthenticated; metrics require the bearer. This mode
does not supply TLS, WAN/public exposure, per-user identities or firewall policy.

Rollback is `systemctl --user disable --now zap-it-lan.service`; remove only the
repo-owned user unit/config if the operator also intends to discard the fixed
key. No system service, firewall, route, GPU driver or shared model cache is
changed.

For a cold explicit lifecycle activation, set
`SLAIF_ZAP_IT_MODEL_CONTROL_MODE=explicit` and a private
`SLAIF_ZAP_IT_MODEL_CONTROL_API_KEY` distinct from the inference key before
starting. Verify the cold state with an authenticated
`POST /v2/repository/index` (expect `UNAVAILABLE`), then load with an empty
body at `/v2/repository/models/zap-it-1/load`. Query the index while loading;
`/healthz` stays 200 and `/readyz` stays 503 until the index is `READY`.
Unload with an empty body at the matching `unload` path. A successful response
has an empty body and is returned only after admission pause, active drain,
holder release, and the measured Torch memory proof. Repeat load/infer/unload
once before treating lifecycle repeatability as qualified. The control key is
never placed in request YAML, logs, metrics or evidence.

With the service launched using a small operator-only
`SLAIF_ZAP_IT_TEST_DELAY_SECONDS` value for drain observation, the bounded
operator helper performs the complete qualification:

```bash
.venv-gpu/bin/python scripts/smoke_model_control.py \
  --port "$SLAIF_ZAP_IT_PORT" --timeout 900
```

It mechanically checks separate control/inference credentials, cold readiness,
invalid-operation no-allocation behavior, concurrent `LOADING`/`UNLOADING`
visibility, two real combined SAM2+CLIP+BLIP3 L3 inferences, active drain and
new-request rejection, idempotency, Torch/physical-GPU/RSS release bounds,
PID/listener continuity, metrics and final stop/port/GPU/shared-memory cleanup.
Its output contains only statuses, bounded counts/timings, state names,
content-free semantic digests and resource facts; it never prints response
bodies, request content, prompts, answers, credentials or cache paths.

Never use `killall`, GPU reset, `systemctl` on unrelated units, firewall
commands, or a wildcard/public bind. The optional `deploy/zap-it-local.service`
file is shipped uninstalled; installing or enabling it is a deliberate operator
choice and is not required by tests.

## Request smoke

### Responses facade qualification

The future compatibility surface is a separate, stateless JSON endpoint over
the same one-slot inference authority:

```text
POST /v1/responses
model=zap-it-1
input=[one user message: one inline image data URL + one inline YAML data URL]
tools=[] or [{"type":"image_generation"}]
```

Use the official pinned client after health, readiness, authenticated
capabilities and the exact private-LAN listener have been checked:

```bash
.venv/bin/python scripts/qualify_responses.py \
  --host "$SLAIF_ZAP_IT_HOST" --port "$SLAIF_ZAP_IT_PORT" \
  --evidence-root "$SLAIF_ZAP_IT_TMP_ROOT"
```

The qualification validates the typed SDK `Response`, `output_text`, public
projection, exactly one completed `image_generation_call`, strict base64 and
PNG dimensions. It retains only bounded mode-0600 summaries/hashes below a
mode-0700 `/dev/shm` directory and prints no key, YAML, prompt, answer or
response body. A no-tool request is covered by the CPU/fake contract unless a
protocol-only live uncertainty requires another inference. The facade never
exposes native completion identity masks, RLE, candidate views, contact sheets,
debug artifacts or ZIP members, and `image_generation` invokes no generative
model. Do not claim `slaif-api-gateway` or public/WAN qualification; that path
is unchanged and remains future work.

The endpoint is the ZAP-IT-specific multipart contract, not generic OpenAI text
completion compatibility:

```text
POST /v1/completions
image=<one JPEG/PNG/WebP>  config=<one UTF-8 YAML>  verbosity=0|1|2|3
response_format=json|zip  model=zap-it-1 (optional)  stream=false (optional)
```

Inspect the authenticated static policy before sending requests:

```bash
curl --fail-with-body \
  -H "Authorization: Bearer $SLAIF_ZAP_IT_API_KEY" \
  http://127.0.0.1:"$SLAIF_ZAP_IT_PORT"/v1/capabilities
```

The capabilities response documents the exact request-local SAM2 defaults,
canonical CLIP prompt union/limits (scalar or independent array items),
per-class maximum aggregation and deterministic prompt-index ties, and
case-sensitive `fast`, `balanced`, and `quality` profiles, strict intrinsic
ranges, startup operator caps and prompt/prediction estimation formulas. It
also describes fixed model/revision, logical device, dtype, residency,
cache/checkpoint/config and artifact-destination policy without disclosing
credentials, sensitive paths, GPU topology or process state. The route is
authenticated but does not require readiness or consume the inference slot.

Use a small redistributable image and safe algorithm-only YAML. A representative
configuration is:

```yaml
alpha: 0.6
clip:
  labels:
    red: "a red object"
    green: "a green object"
```

At each verbosity, verify the normalized five-field YOLO lines. L1 contains a
16-bit `identity-mask.png` with original dimensions, background `0`, and IDs
`1..N`; L2 contains only produced object/SAM/CLIP fields; L3 adds bounded stage
status, timings, pinned model/device provenance, warnings, annotated/debug
artifacts, and exact per-object column-major RLE. `annotated` remains mask-only;
the L3-only `annotated-labelled` stream uses final structured labels and exact
instance IDs with deterministic bounded placement. ZIP contains `manifest.json`,
`detections.yolo.txt`, and the same level-gated artifact names. A request
containing bounded nested BLIP3 verification rules is supported, with at most 32
uploaded rule definitions. The service fixes BLIP3 to FP16 and uses a separate
operator-owned 1..256 planned questions/request capacity (default 256), plus
32 generated tokens per question; model IDs, revisions,
dtype, paths and runtime controls remain rejected. Set
`SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS` only in the operator environment; it is never
accepted from uploaded YAML. Planned excess returns typed `resource_limit` 413
before BLIP3 generation, while response assembly overflow remains
`response_too_large`. L3 visualization members use fixed ordinal names such as
`visualization/stream-0001.png`; `visualization_id` preserves the validated
configured stream ID as logical metadata only, never as a path or member name.
Geometry and panoptic visualization remain unsupported.

The `mask_generator` section may select only the documented safe scalars. The
service resolves each field as explicit value, then profile override, then
server default, and records that source in `service.sam2`. It constructs one
fresh generator around the resident model for each accepted request; weights
are not reloaded when these scalars change. Intrinsic/type failures return
`invalid_config` 400. Operator field or estimated-work violations return the
non-retryable `resource_limit` 413 before generator construction and inference.
The manifest's `actual_candidate_count` is the raw generator count, while the
existing L3 `candidate_counts.sam2_candidates` remains the post-remap,
non-empty count. Capacity rejection details also list sanitized estimates,
causing values, public limits and admissible request-safe alternatives.

For L3 post-filter evidence, inspect `service.post_filter_diagnostics` beside
`candidate_counts`. It reports mutually exclusive `maxsize`, `empty_mask`,
`max_w`, `max_h`, or retained outcomes in that precedence. The `maxsize` branch
is decided before segmentation access and its record has the exact area with
zero bbox dimensions because they were not evaluated; empty masks also have zero
dimensions for their distinct reason. Other bbox extents are inclusive, and
threshold rejection is strict. Verify the aggregate equality, the candidate-count
cross-checks, numeric-only source-indexed records and the 256-record cap/truncation.
Repeat JSON and ZIP requests and compare the
diagnostic values in the response and `manifest.json`. The two-wide-candidate
roof scenario is a programmatic CPU regression and must not be reported as a
real-image accuracy result.

For a raw-SAM2 L3 audit, set only `mask_generator.debug: true`. The service
returns fixed `sam2-candidates-page-0001.png`..`-0008.png` pages and
`sam2-union-coverage.png`, `sam2-overlap-heatmap.png`, and
`sam2-uncovered-pixels.png`; pages are 3x4 with 320x240 content and a 28-pixel
label bar, capped at 96 represented candidates. IDs are one-based source-order
IDs, with gaps for empty proposals, and labels show IoU/stability to three
decimals or `n/a`. The three diagnostics account for every non-empty raw mask,
including truncated candidates: union is black/white uncovered/covered,
overlap is a fixed ramp scaled by its observed maximum, and uncovered is the
exact inverse before nearest-neighbor downscale to at most 2,000,000 pixels.
Candidate crops may be enlarged into their fixed 320x240 tiles; the diagnostics
never upscale.
Check the typed `service.sam2.raw_visualization` arithmetic, histogram overflow,
source/diagnostic dimensions and fixed artifact hashes in both JSON and ZIP.
The request continues through readiness, gate and engine when optional raw or
visualization bytes cannot fit. Inspect `service.artifact_delivery` for the
typed omission reason and `truncated: true`; only an essential response that
cannot fit returns `response_too_large`.

For a bounded L3 BLIP3 audit, set `debug: true` on one rule. The verifier
composes one image per applicable candidate: source pixels under exact
Euclidean support are restored, a thin exterior contour is painted, and all
other crop pixels are Gaussian-blurred scene context. The centered source crop
uses inclusive raw/support bboxes and a half-open slice bbox; a crop that cannot
contain support plus contour is rejected for that candidate without QA or
debug work. The final image uses bilinear resizing with a 256-pixel short side
and 768-pixel long-side cap. The service returns the exact lossless image as
`blip3-verification-CANDIDATE-####-QUESTION-####.png` at L3 only; it never puts
the question, label, answer or filename in that logical name. This qualifies
the pixel/instruction/artifact integration, not universal BLIP3 accuracy.

For a candidate whose nominal crop cannot contain the requested support, the
optional request field `candidate_views.blip3.infeasible_geometry_policy:
centroid_radial_mask_chord` enables the deterministic fallback. The default
`reject` behavior and all successful Euclidean views remain compatible. The
fallback uses the complete-mask centroid and external contour walk in a
tight-bbox/local-window scratch region, and processes rays in fixed-size
batches. It may shift the full nominal crop; `crop_shifted` compares against
the unshifted centered nominal crop. It reduces or disables the contour before
applying one common millionth radial scale, and records the highest-precedence
adjustment at L3. Raw radial diagnostics are pre-clamp and may exceed
`max_context_pixels`; effective values cannot. It is mask-only geometry:
zero-valued gaps may be crossed while measuring a whole-mask chord, but no
rectangular bridge is added. The standalone benchmark uses a deterministic
122-candidate mix of elongated, rotated, concave, fragmented, centroid-gap,
hole, and high-boundary masks up to the 199-by-199 reference bound; its median
and maximum are reported outside normal CI wall-clock assertions.

The operator-only `scripts/profile_matrix.py` harness sends the exact sanitized
sequence `sam2, sam2_clip, sam2_blip3, sam2_clip_blip3, sam2_clip_blip3,
sam2_blip3, sam2_clip, sam2` as authenticated L3 JSON requests. It requires
the all-resident strategy, logical `cuda:0`, all three pinned model identities,
zero transitions, stage-specific semantics, bounded BLIP3 answers, repeatable
content-free shape digests, the 90% physical-memory ceiling, and no request
residue. It uses only a generated 128x128 RGB fixture and in-memory API-safe
YAML and never prints response bodies or answer text.

Scrape `GET /metrics` during the bounded run. It is process-local and contains
only finite labels; it must not contain filenames, labels, prompts, answers,
request IDs, paths, credentials or raw content. With an API key configured,
scraping requires the same bearer key as completions.

The smoke helper also has explicit operator-only checks for hostile input and
recovery paths. Run them only against the matching test activation:

```bash
python scripts/smoke_local_service.py --port "$SLAIF_ZAP_IT_PORT" \
  --levels 0 1 2 3 --formats json zip --busy --repeat 3
python scripts/smoke_local_service.py --port "$SLAIF_ZAP_IT_PORT" --invalid
python scripts/smoke_local_service.py --port "$SLAIF_ZAP_IT_PORT" --failure
python scripts/smoke_local_service.py --port "$SLAIF_ZAP_IT_PORT" --timeout
python scripts/smoke_local_service.py --port "$SLAIF_ZAP_IT_PORT" --cancel
```

Failure, timeout/cancel and response-size checks require a fresh restart with
the corresponding operator environment set before the service process starts;
they are never selectable through request YAML. For response-size rejection,
use a deliberately small operator cap such as
`SLAIF_ZAP_IT_MAX_RESPONSE_BYTES=1` and run
`--response-too-large`, then restore the normal cap before the final E2E.
For a bounded serialization-deadline recovery check, an operator may combine a
short `SLAIF_ZAP_IT_REQUEST_DEADLINE_SECONDS` with the private
`SLAIF_ZAP_IT_TEST_SERIALIZATION_DELAY_SECONDS` hook; both are process-start
settings and are unset for normal operation.

The service keeps request bytes, decoded arrays, results, and artifacts in
memory. If a future compatibility stage needs a path, it must use a unique
mode-0700 workspace below the configured `/dev/shm` root and mode-0600 files;
the current resident path does not write request data there.

## Evidence and bounded checks

During a live round capture sanitized snapshots at these points: before start,
after start/not-ready, ready, during E2E, during overlapping requests, after
stop, and after restart. Include:

```bash
ss -H -ltn
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
find /dev/shm/slaif-zap-it -mindepth 1 -maxdepth 2 -printf '%M %p\n'
```

Confirm one service PID owns all three resident models and no wildcard listener
exists; in loopback mode no LAN listener may exist. Every unassigned device's
process/memory lines are byte-stable, and the root is empty
after stop. For concurrency, overlap two real requests: one runs, the other
must receive deterministic `503 service_busy` with `Retry-After` when queue
depth is zero. Repeat a bounded number of calls and record latency, response
size, assigned-GPU allocated/reserved memory, process RSS, and residue; this is not a
soak test or a leak-proof claim.

Exercise invalid image/YAML, hostile model-control rejection, injected operator-only
failure (`SLAIF_ZAP_IT_TEST_INJECT=failure`), and timeout/cancel behavior with
the test-only delay hook (`SLAIF_ZAP_IT_TEST_INJECT=timeout` plus
`SLAIF_ZAP_IT_TEST_DELAY_SECONDS` and a short operator deadline). Use a small
operator response limit to prove `response_too_large`. These hooks are not
available to request YAML and must be unset for normal operation.

Inspect logs for identifiers, counts, statuses, and timings only. They must not
contain image/YAML bodies, prompts/answers, filenames, headers, keys, model
cache paths, stack traces, or request-derived persistence. Run the CPU suite,
Ruff, package build, and remote CI separately; a local live smoke does not
replace those checks.

## Rollback

Rollback is reversible and host-local:

```bash
scripts/serve_local.sh stop || true
ss -H -ltn
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
find /dev/shm/slaif-zap-it -mindepth 1 -maxdepth 2 -print
```

Remove only this checkout's optional user configuration and, if explicitly
desired, its repo-owned launcher/unit files. Do not remove shared
model-cache entries, stop unrelated GPU work, alter NVIDIA/CUDA, firewall/VPN,
global credentials, or delete persistent user data. The end-of-round state is a
free scoped port, no ZAP-IT process, the assigned GPU near its pre-round
baseline, other devices unchanged, and an empty `/dev/shm/slaif-zap-it` root.

## Installed candidate and local academic regression

The unpublished 0.1.0 wheel exposes the foreground zap-it-service console
entrypoint. Install it in a clean venv outside the checkout, source a private
mode-0600 EnvironmentFile, and use the same assigned-GPU UUID, logical cuda:0 and
scope-specific checks documented above. The optional user-systemd template is
Type=simple, uses that EnvironmentFile and an explicit installed-venv
placeholder; it is shipped uninstalled. Upgrade is a stopped venv replacement
followed by the same readiness check. Rollback restores the prior venv. A
deliberate uninstall stops the service and removes only the unit, private
config and candidate venv; it never changes system CUDA, shared caches or
unrelated services.

After building/installing the candidate, the optional local academic regression
is explicit and restricted to the selected GPU:

~~~bash
.venv-gpu/bin/python scripts/smoke_local_goats.py \
  --port "$SLAIF_ZAP_IT_PORT" \
  --image-a demos/goats/goats1.jpg \
  --image-b demos/goats/goats2.jpg \
  --config configs/goats2.yaml \
  --api-key "$SLAIF_ZAP_IT_API_KEY"
~~~

The repository owner has confirmed redistribution rights for these four
fixture/config paths. They nevertheless remain ignored operator inputs and are
excluded from packages and release artifacts as defense in depth. The harness
refuses missing, symlinked or out-of-root files,
safe-loads and allowlists the legacy YAML including nested BLIP3 rules, strips
operator/model controls, independently crops both goat images to exactly the
middle 50 percent in memory, and emits only sanitized aliases/digests/
dimensions/statuses/timings/counts. Its `--benchmark` mode sends exactly ten
BLIP3-enabled L3 JSON requests in A,B,A,B,A,B,A,B,A,B order and reports
stage execution, transition/restore timings, GPU peak/free memory, host RSS,
repeatability and first/minimum/median/nearest-rank-p95/maximum latency statistics.
It never
prints or persists the source YAML, crop, prompts, labels, answers, response
bodies or bearer key.

For Objective 020 investigations, use only sanitized L3 metadata: raw SAM2
counts, geometry rejection facts, complete CLIP vectors, the pre-routing
`clip_scored` count, routing reasons/cap outcomes including clear negatives,
BLIP3 mapping records, final counts, and stage timings. CLIP and
BLIP3 debug PNGs must equal their literal model inputs; the two builders are
never interchangeable.
