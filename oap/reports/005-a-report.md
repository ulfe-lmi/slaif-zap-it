# OAP Coding-Agent Report — 005-a

## Work order

- Identifier/order/objective/PR mode: `005-a` / Objective 005 / `CREATE_NEW_PR`
- Repository: `ulfe-lmi/slaif-zap-it`
- Required branch: `oap/005-a-full-output-parity-hardening-and-evidence`
- Required PR: [#49](https://github.com/ulfe-lmi/slaif-zap-it/pull/49)

## Status

COMPLETE

## Executive summary

Objective 005-a is implemented and published as PR #49. The service now has an
honest output parity catalog, L3 overlap-preserving column-major uncompressed
RLE, bounded visualization/debug/artifact/resource budgets, early JSON/ZIP size
guards, content-safe custom metrics, client-disconnect recovery, and a service
datasheet. Geometry, panoptic and BLIP3 remain explicitly unsupported in the
qualified live profile. Legacy CLI adapters and trusted filesystem writers are
preserved.

## Authoritative GitHub state

- Base: remote `main` at starting SHA `22e827eaab15a5eb3299a6b5bfd156eb96c68946`
- Starting checkout SHA: `22e827eaab15a5eb3299a6b5bfd156eb96c68946`
- Implementation head SHA: `080583988a3ee9ca11ff3169770f9e8eeaaf3f49`
- PR #49: OPEN, exact required title, base `main`, head at implementation SHA
- New PR: yes; amended existing: no; coding merge: NO
- Report publication commit: SELF

## Changes/files

Implementation commit `0805839…` contains the exact active `005-a` order
transcript and selector plus:

- `src/service/rle.py`, bounded `BoundedMemoryArtifactSink`, operator settings,
  image/resource admission, JSON/ZIP response preparation, and metrics;
- core L3-only rendering control, deterministic annotated palette, and opaque
  service debug names while retaining legacy CLI naming behavior;
- visualization policy validation, schema/OpenAPI updates, error handling and
  disconnect recovery;
- `docs/OUTPUT-PARITY.md`, `docs/SERVICE-DATASHEET.md`, API/config/core/security/
  runbook/testing documentation, and operator env examples;
- focused Objective-005 CPU tests in `tests/test_parity_hardening.py`;
- `prometheus-client==0.21.1` in the GPU lock and service extra.

## Acceptance evidence

1. **Parity catalog:** PASSED — `docs/OUTPUT-PARITY.md` classifies preprocessing,
   SAM2, post-filtering, CLIP, BLIP3, geometry, visualization, metadata, debug,
   YOLO, identity PNG, legacy image/video writers and dataset export.
2. **RLE and ID agreement:** PASSED — CPU and live tests round-tripped empty,
   all-zero/all-one, disconnected, checkerboard and overlapping masks; live L3
   JSON/ZIP object RLE records matched by instance ID and area.
3. **Geometry/panoptic policy:** PASSED — geometry, panoptic and unknown
   visualization rules reject before inference; only bounded annotated streams
   execute at L3; lower levels skip render-only execution.
4. **Budgets and cleanup:** PASSED — image dimensions, object count, stream,
   artifact count, per-item/total raw bytes, RLE runs, base64 JSON and ZIP size
   boundaries are covered; no post-hoc artifact slicing remains; low-memory and
   low-shm admission simulations fail before the engine and leave no sink state.
5. **Metrics/privacy:** PASSED — custom registry has no default process
   collectors, only finite stable labels, unlabeled size/count/timing histograms,
   readiness/active/GPU gauges, and no raw content, filenames, prompts, labels,
   answers, paths, credentials or request IDs in metrics/log inspection.
6. **State isolation:** PASSED — 32 alternating live requests succeeded; A/B/A
   state hashes restored A exactly (`78ec669d31fc0888` / B
   `f41c35322bc337e3` / A `78ec669d31fc0888`) for YOLO, class mapping, objects
   and config digest.
7. **Failure/recovery:** PASSED — targeted live failure, timeout, response
   limit, invalid input, client disconnect and busy probes were followed by
   successful normal requests or clean restart recovery. Client disconnect no
   longer emitted a traceback.
8. **Datasheet/docs:** PASSED — exact levels, unsupported stages, limits,
   hardware/software profile, bounded measurements, metrics, privacy and
   prerequisites are in `docs/SERVICE-DATASHEET.md` and linked documentation.
9. **GPU isolation/final state:** PASSED — only physical GPU1 was visible to
   the service; GPU0 PID 66522 remained the unrelated protected process; final
   service state is stopped and clean.
10. **Verification/CI:** PASSED — full CPU/static checks and all six required
    current-head GitHub checks are SUCCESS.
11. **OAP topology:** PASSED — one Objective-005 PR, implementation head first,
    report-only SELF child planned; coding did not merge.

### Bounded 32-request live table

| Request set | Count | Successful | Formats/levels |
|---|---:|---:|---|
| Alternating A/B synthetic images and label maps | 32 | 32 | every L0/L1/L2/L3 × JSON/ZIP combination, four each |

Aggregate latency was p50 `450.4 ms`, p95 `502.5 ms`, maximum `723.6 ms`;
response sizes ranged from `746` to `3716` bytes. There were two distinct
variant/config state hashes and no rejected request allocation. A separate
same-shape A/B/A sequence restored A exactly as recorded above.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — `332 passed, 1 skipped`, 76.44% total coverage; the skip is the
  explicit opt-in GPU marker.
- `.venv/bin/pytest -q tests/test_service_api.py tests/test_parity_hardening.py`:
  PASSED — 62 tests.
- `.venv/bin/ruff format --check .`:
  PASSED.
- `.venv/bin/ruff check .`:
  PASSED.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh`:
  PASSED.
- `.venv/bin/python -m compileall -q src modules`:
  PASSED.
- `.venv/bin/python -m build --wheel`:
  PASSED — wheel built successfully.
- Focused RLE/visualization/budget/resource/metrics tests:
  PASSED — included in the CPU suite.
- Live all-level smoke after final code state:
  PASSED — 10 cases: L0-L3 JSON/ZIP, BLIP3 rejection and repeat stability.
- Live annotated/RLE/JSON/ZIP/metrics probe:
  PASSED — 200 JSON and ZIP, eight RLE round trips, annotated artifact, two
  unsupported pre-inference cases, content-safe scrape.
- Live controlled overlap with `SLAIF_ZAP_IT_TEST_INJECT=delay`:
  PASSED — follower 503 `service_busy`, `Retry-After: 5`, slow request 200.
- Live failure/timeout/response-limit smoke probes:
  PASSED for each targeted expected-error subcase — injected failure 500
  `inference_failure`, injected timeout 504 `timeout`, and configured cap 413
  `response_too_large`; each was followed by normal restart/recovery. The
  helper's companion baseline level is intentionally reported as failed when
  the whole process is configured to reject every inference.
- Live invalid input and client-cancel recovery:
  PASSED — stable invalid image/YAML errors and subsequent 200 recovery; final
  cancel run emitted no `Traceback` or `ClientDisconnect` log text.
- PR current-head CI/CodeQL:
  PASSED — all six checks SUCCESS at implementation SHA.

## CI/checks

At implementation SHA `080583988a3ee9ca11ff3169770f9e8eeaaf3f49`:

| Check | State |
|---|---|
| `static (format, lint, build)` | SUCCESS |
| `tests (py3.10)` | SUCCESS |
| `tests (py3.11)` | SUCCESS |
| `tests (py3.12)` | SUCCESS |
| `Analyze (python)` | SUCCESS |
| `CodeQL` | SUCCESS |

## GPU/service/resource evidence

- Fresh preflight: physical GPU1 was NVIDIA GeForce RTX 2080 Ti, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI `00000000:00:0C.0`,
  11264 MiB, driver `580.178.04`, 6 MiB used / 10815 MiB free. The service
  process saw exactly one visible GPU as logical `cuda:0` under
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=1`.
- Protected GPU0 was UUID `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI
  `00000000:00:08.0`, with unrelated PID 66522 and 2501 MiB used at the
  preflight/final snapshots. No ZAP-IT process appeared on GPU0.
- Verified loopback endpoint: `127.0.0.1:17891`; one Uvicorn process/worker and
  one inference executor. Peak observed live request snapshot was approximately
  5749 MiB GPU1 used, 5740 MiB process GPU memory and 1,589,556 KiB RSS.
- Host preflight: 50,843,197,440 bytes available RAM and 28,273,922,048 bytes
  available `/dev/shm` capacity. No request data was written to persistent disk.
- Final stop snapshot: port free, no `scripts/serve_local.py` process or runtime
  pidfile, GPU1 returned to 6 MiB used, protected GPU0 remained present, and
  `/dev/shm/slaif-zap-it` had no children.

## Documentation/provenance

Updated README, API/API-target, CONFIG, CORE, ALGORITHMS, SECURITY, TESTING and
RUNBOOK; added the parity matrix and service datasheet. Model identities and
runtime pins remain the accepted Objective-003 references; no weights,
credentials, private cache paths, images, YAML bodies or customer data were
committed or placed in OAP evidence.

## Deferred human adjudication

- Critical register action: NONE
- The finalized order explicitly resolved `NONE`; no critical-register read,
  append, edit, or disposition was performed.

## Safety/scope confirmations

- Geometry was not activated; panoptic and BLIP3 live enablement were not added.
- No LAN/public exposure, gateway, firewall/VPN, TLS, Docker, systemd,
  multi-worker, persistence, training, model substitution or customer data.
- Physical GPU0, its process, system CUDA/driver, unrelated services, global
  credentials and unrelated ports were not modified.
- The service was stopped after live evidence and no auto-merge or merge action
  was performed.

## Limitations/blockers

Evidence is a bounded synthetic/redistributable fixture window, not a soak,
SLA, production-readiness or leak-proof claim. The live profile remains
resident SAM2+CLIP only, loopback-only, one process/request at a time. External
deployment, gateway integration, commercial/model licensing and final release
remain Objective-006/human-gated work.

## Factual strategic follow-up

Strategic review/acceptance and any merge decision remain outside coding scope.
The next action is governed by the existing OAP process; coding does not choose
or activate another order.

Implementation head SHA: 080583988a3ee9ca11ff3169770f9e8eeaaf3f49
Report publication commit: SELF
