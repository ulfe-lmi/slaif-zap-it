# OAP Work Order — 005-b — Local goats E2E and final resource gaps

Objective `005-b`, continuation of numeric Objective 005 on existing PR #49.
Preserve the accepted `005-a` parity/RLE/metrics design while closing four
strategic-review gaps: local-only nonredistributable goats semantic E2E,
serialization deadline enforcement and vectorized RLE throughput, raw
visualization-allocation preflight, and missing live auth/resource-rejection
recovery evidence. Do not rewrite any activated order or report.

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`; numeric objective / round `005 / 005-b`.
- Mode: `AMEND_EXISTING_PR`; do not create a branch or PR.
- Existing branch: `oap/005-a-full-output-parity-hardening-and-evidence`.
- Sole Objective-005 PR: #49, title `Objective 005-a: full-output parity,
  resource hardening, metrics and service datasheet`, base `main`.
- Remote/base main remains accepted Objective-004 squash
  `22e827eaab15a5eb3299a6b5bfd156eb96c68946`.
- Current local/remote PR head:
  `17f67bd6eedc527c239e93326bc9afc4b1d43daa`, immutable report-only
  `005-a` commit; its sole parent is implementation
  `080583988a3ee9ca11ff3169770f9e8eeaaf3f49` and it changes only
  `oap/reports/005-a-report.md`.
- All six report-head checks are SUCCESS; PR is open, mergeable and clean.
- Required new report: `oap/reports/005-b-report.md`. Earlier reports/orders
  remain byte-for-byte immutable.

## Review findings and binding corrections

### 1. Nonredistributable goats semantic E2E

Human clarification after `005-a` activation establishes:

- `demos/goats/goats1.jpg`, `demos/goats/goats2.jpg` and `configs/goats2.yaml`
  are local academic-test assets and are **not redistributable**;
- both images were historically tested with `goats2.yaml`;
- service E2E must use the middle 50% in both image dimensions to avoid
  excessive GPU load.

Use them only in explicit local GPU evidence:

1. Open each tracked 5568×4176 image and crop in memory to the central 50%
   box `(1392, 1044, 4176, 3132)`, producing exactly 2784×2088.
2. Never write/commit the crop, raw upload, response artifacts or temporary
   sanitized YAML to persistent disk. If a temporary path is unavoidable, use
   a private `/dev/shm` workspace and prove cleanup; in-memory bytes are
   preferred.
3. Read `configs/goats2.yaml` locally and construct an API-safe mapping in
   memory. Retain relevant goat/negative CLIP prompt mappings, post-SAM2
   filtering, visualization labels/alpha and one supported annotated stream.
   Remove request-forbidden model IDs/revisions, all `mask_generator` controls,
   BLIP3, panoptic, paths, images/video output and YOLO export controls. Use no
   raw caller/operator paths in the resulting upload.
4. Do not copy prompt text, image bytes, crops, labels, filenames, raw YAML or
   raw response content into logs, metrics, OAP reports, PR body or generated
   evidence files. Report only sanitized fixture aliases (`goats-A/B`), crop
   dimensions, config/image digests, status, counts, sizes, timing and resource
   aggregates.
5. Run both cropped images through at least L2 JSON and L3 JSON+ZIP. Validate
   five-field YOLO, object/identity/RLE ID agreement, exact RLE dimensions and
   areas, annotated artifact presence, content-safe metrics/logs and no residue.
6. Run local A/B/A (`goats1` / `goats2` / `goats1`) and prove A semantic hashes
   restore exactly with no B request-state leakage.
7. Update parity matrix/datasheet with a prominent `NONREDISTRIBUTABLE — local
   academic E2E only; excluded from packages/release fixtures` entry. Objective
   006 must explicitly review the existing repository assets and exclude them
   from distributable artifacts unless human rights clearance changes.

Do not add the goats run to public/GitHub CI. Synthetic fixtures remain the
redistributable CI and adversarial evidence.

### 2. Whole-request deadline and RLE throughput

`encode_mask_rle()` currently uses nested Python loops over every pixel. RLE,
artifact preparation, base64 and ZIP assembly occur after the inference gate
and outside `_run_engine_bounded`; the documented 120-second whole-request
deadline is therefore not enforced during serialization.

- Replace per-pixel Python iteration with deterministic column-major NumPy
  chunk transition detection or an equivalently vectorized bounded encoder.
  Do not allocate a second full-size flattened mask. Use a fixed documented
  chunk maximum and preserve exact current RLE bytes/semantics.
- Add an absolute monotonic serialization/request deadline to response context
  or equivalent. Check it during every RLE chunk/object, artifact preparation,
  JSON base64 expansion and ZIP entry loop, plus before returning the response.
- If the budget expires, return stable `504 timeout`; do not partially return,
  leak sink state, corrupt the resident registry or release a second inference
  early.
- Add large uniform and checkerboard mask tests proving fixed auxiliary memory,
  vectorized chunk bounds, exact round-trip/run limits and deterministic output.
- Add a CPU serialization-timeout injection followed by success. Add a safe
  live operator-only serialization/RLE-limit rejection and recovery; do not
  attempt a real 64-MP timeout or exhaust host memory.

### 3. Visualization raw-allocation preflight

The service permits eight 8192×8192 annotated streams. Current engine rendering
can allocate/retain every RGB array before `_collect_raw_artifacts()` encodes
and rejects the 128-MiB raw-artifact budget.

- Before inference/rendering, calculate exact supported annotated output raw
  bytes as `stream_count * height * width * 3` (or the exact dtype/shape
  equivalent).
- Reject `response_too_large` before model execution if one stream exceeds the
  single-artifact raw limit or all streams plus the configured debug sink raw
  budget exceed the total raw limit.
- Subtract reserved visualization raw bytes from the service debug sink budget,
  so the combined raw artifact allocation cannot exceed the operator total.
- Add engine-spy tests proving rejection occurs before inference and boundary
  acceptance works. Preserve L0–L2 no-render behavior and legacy CLI defaults.

### 4. Missing live auth/resource evidence

`005-a` has CPU resource tests and live synthetic success/failure evidence, but
its report does not prove the ordered live optional bearer-auth flow or live
simulated low-host-memory, low-shm, RLE/artifact and serialization rejection
with subsequent registry recovery.

- Restart once with a private temporary API key supplied only through process
  environment; prove `/v1/completions` and `/metrics` return 401 without/wrong
  bearer and succeed with the correct bearer. Never print/store the key.
- Use operator settings/injection to simulate host/shm floors above observed
  capacity without consuming resources; verify sanitized 507 codes before
  inference, then normal restart/recovery.
- Exercise RLE-run and combined artifact budget rejection safely, then normal
  success from the same registry when possible or after controlled restart.
- Exercise serialization deadline rejection and recovery as above.
- Metrics/log scans must reflect finite counters but contain no key, auth
  header, raw fixture/config/labels/prompts, request-ID label or host path.

## Current host and safety law

At strategic review the service was stopped, ports 17891/23654 free, GPU1 at
6 MiB with no compute process, `/dev/shm/slaif-zap-it` empty, and GPU0 held only
protected PID 66522 at independently varying memory. Re-verify live before every
activation.

Use only physical GPU1 UUID
`GPU-c457dbaf-991c-dc23-c781-0dc030776dd8` through
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`, one
process/worker/inference and IPv4 loopback. Never touch GPU0/PID 66522, system
CUDA/driver, firewall/VPN, unrelated services or global credentials. Final
state is stopped, GPU1 idle, ports free and shared-memory root empty.

## Required verification

### CPU/static

- Retain the complete `005-a` suite and add focused tests for all corrections.
- RLE small-property round trips plus large uniform/checkerboard chunk/memory/
  timeout cases; exact prior encoding compatibility.
- Visualization raw-preflight boundaries with inference/render spies and
  combined debug budget accounting.
- Auth/resource/RLE/artifact/serialization rejection and recovery tests.
- Goats helper logic may be tested using synthetic stand-ins, but must not make
  nonredistributable assets part of CI or package data.
- Complete canonical pytest/coverage, Ruff format/lint, shell syntax, compile,
  wheel build/import, OpenAPI/docs, secret/large-artifact and exact-diff checks.
- All six GitHub checks SUCCESS on implementation and report heads.

### Live local

1. Fresh host/GPU/process/port/RAM/shm snapshot and clean start/readiness.
2. Exact central-crop goats1/goats2 L2 JSON and L3 JSON+ZIP matrix using the
   in-memory safe derivative of goats2.yaml; sanitized results only.
3. Goats A/B/A state-isolation and latency/response/object/artifact/RSS/VRAM/
   metrics observations. This is local academic regression evidence, not an
   accuracy benchmark or redistributable golden test.
4. Bearer auth matrix for completions and metrics.
5. Simulated low-RAM, low-shm, RLE/artifact and serialization timeout rejection
   plus successful recovery.
6. Re-run one synthetic L3 JSON+ZIP case to prove prior deterministic contract.
7. Log/metrics/content scan, graceful stop/restart once, one post-restart goats
   or synthetic L3 request, and final stopped/clean host proof.

## Acceptance criteria

1. Both central-50% goats images and a safe in-memory goats2-derived config are
   exercised locally with complete sanitized evidence and zero persistence.
2. Nonredistributable status is unmistakable in parity/datasheet/report and is
   carried as an Objective-006 package/release constraint.
3. RLE encoding has no per-pixel Python loop/full copy, is chunk-bounded and
   obeys the whole-request deadline through final serialization.
4. Visualization/debug raw allocation is rejected before inference when the
   combined budget cannot fit.
5. Live auth and every missing resource/serialization rejection recovers safely
   with content-safe metrics/logs.
6. Existing `005-a` parity, metrics, RLE, state-isolation, CLI, API and GPU1
   behavior remain green; no new scientific capability is activated.
7. PR #49 remains the sole Objective-005 PR; final `005-b` report-only SELF
   commit has the literal implementation SHA parent and all six checks green.
8. Final host is stopped/clean and GPU0/unrelated workloads are untouched.

## Non-goals

- no redistribution, copying, packaging or publication of goats image/YAML
  content or derived crops/artifacts;
- no accuracy claim/golden output based on goats;
- no geometry, panoptic or BLIP3 activation;
- no LAN/public exposure, gateway, Docker, systemd installation, release,
  customer data, persistence, multi-worker/GPU or model change;
- no rewrite of `005-a` history and no new branch/PR;
- coding never merges.

## Deferred human adjudication

- Decision: `NONE`

The human has explicitly resolved fixture rights: local testing is allowed and
redistribution is not. The technical corrections are ordinary bounded bugs.
No critical-register threshold is met. Objective 006 must honor the explicit
nonredistribution constraint before packaging/release.

## GitHub/OAP publication

- Amend PR #49 only on the existing branch; preserve its title/base.
- Commit correction code/tests/docs plus exact `005-b` order transcript and
  `oap/active` before report; do not commit goats derivatives/evidence payloads.
- Push and require all six implementation-head checks SUCCESS.
- Capture literal implementation SHA and create exactly one immutable
  `oap/reports/005-b-report.md` with `Report publication commit: SELF`.
- Final commit changes only that report; verify parent/path/remote bytes/PR head
  and all report-head checks before response signaling.
- Report sanitized goats crop/config digests, request/status/count/size/latency/
  resource aggregates, nonredistribution, auth/resource/timeout recovery,
  inherited versus fresh evidence, final host state and critical action `NONE`.

## Coding response

Send exact FIFO `OK` only after PR #49, complete local/CI evidence, immutable
report and final stopped host are verified. Coding never merges.
