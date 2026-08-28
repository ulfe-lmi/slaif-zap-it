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

The capabilities response documents the exact request-local SAM2 defaults and
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
containing bounded nested BLIP3 verification rules is supported. The service
fixes BLIP3 to FP16, 32 questions/request and 32 generated tokens per question;
model IDs, revisions, dtype, paths and runtime controls remain rejected.
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
non-empty count.

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

For a bounded L3 BLIP3 audit, set `debug: true` on one rule. The verifier passes
each executed question a paired image: untouched context on the left and an
exact mask spotlight on the right, separated by four dark pixels and outlined
only on the exterior by a four-pixel yellow contour. Its dimensions use a
128-pixel minimum, uniform nearest-neighbor scaling toward a 256-pixel short
side, and a 768-pixel long-side cap. The service returns the exact lossless
image as `blip3-verification-####-####.png` at L3 only; it never puts the
question, label, answer or filename in that logical name. This qualifies the
image/instruction/artifact integration, not universal BLIP3 accuracy.

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
