# OAP Coding-Agent Report — 004-c

## Work order

- Identifier: `004-c`; numeric objective `004`; mode `AMEND_EXISTING_PR`
- Repository: `ulfe-lmi/slaif-zap-it`
- Existing PR: #48, `Objective 004: loopback service activation on physical GPU1 with recovered live E2E evidence`

## Status

COMPLETE

## Executive summary

Closed the three independent review findings without changing accepted 004-b
behavior: service identity-mask projection now uses a deterministic complete
source-pixel assignment, startup logs no longer print host filesystem paths,
and configured ephemeral roots are canonicalized and required to be strict
descendants of `/dev/shm`. The corrected service passed real GPU1 L0–L3 JSON/ZIP
and repeat evidence, including BLIP3 rejection, and the final host state is
stopped and clean.

The immutable 004-b report's earlier no-host-path statement was overbroad: the
004-b runtime log did contain its `shm_root` field. This report does not rewrite
that history. The corrected log emits `shm_ready=true` and bounded free
capacity, and the exact corrected scan found no host paths or injected
secret-looking operator paths.

## Authoritative GitHub state

- PR: [#48](https://github.com/ulfe-lmi/slaif-zap-it/pull/48), OPEN, mergeable,
  clean, base `main`, existing branch
  `oap/004-a-loopback-service-activation`
- PR title unchanged; no new PR; existing PR amended; coding merge: NO
- Remote `main` / base SHA: `1a4272d60c52cc045f57f2842652485efdb7a55c`
- Starting branch/report head SHA: `d297833b1d1d542574ed6959d7efd903a1d13909`
- Implementation head SHA: `7076c3053ce4ad05bddaf7bbb8847b2ddc708bfe`
- Implementation parent SHA: `d297833b1d1d542574ed6959d7efd903a1d13909`
- Report publication commit: SELF

## Changes/files

Implementation commit `7076c305…` contains only the ordered remediation,
tests/docs, `oap/active`, and the immutable 004-c order transcript:

- `docs/runtime.md`
- `oap/active`
- `oap/orders/004-c-close-identity-log-and-shm-review-gaps.md`
- `scripts/serve_local.sh`
- `src/__init__.py`
- `src/core/__init__.py`
- `src/core/errors.py`
- `src/core/renderers.py`
- `src/runtime/live_service.py`
- `src/runtime/shm.py`
- `src/service/envelope.py`
- `tests/test_core_renderers.py`
- `tests/test_live_service_units.py`
- `tests/test_runtime_units.py`
- `tests/test_service_units.py`
- `tests/test_src_exports.py`

## Acceptance evidence

1. Identity projection: complete deterministic bipartite assignment with
   minimum representative-pixel overrides; focused tests cover occlusion,
   `{p,q}`/`{p}` adversarial reassignment, three-way augmenting behavior,
   impossible matching, deterministic bytes, and legacy winner behavior.
2. Legacy rendering: `ensure_all_ids=False` path and larger-area/tie winner
   tests remain green; source masks are never modified.
3. Impossible projection: typed `IdentityMaskProjectionError` maps to stable
   sanitized `inference_failure` with an identity-representation message.
4. Runtime logs: corrected startup line contains `shm_ready=true`; exact scan
   found no repository/cache/shared-memory paths, raw inputs, filenames,
   credentials, headers, prompts/answers, traceback text, or injected
   secret-looking operator path strings.
5. Shared-memory containment: canonical valid descendants pass; normalized
   descendants pass; `/dev/shm` itself, `..` escapes, intermediate symlink
   escapes, final symlinks, insecure roots, and ordinary persistent paths fail
   before listener/model startup. The shell launcher also rejects invalid roots
   without a traceback.
6. Live GPU1 service: corrected-head real smoke passed 10 cases covering L0–L3
   JSON/ZIP, three repeats, identity PNG IDs, and BLIP3 rejection. After one
   stop/restart, a post-restart six-case JSON smoke including L0–L3, one repeat,
   and BLIP3 rejection also passed.
7. CPU/static/package/CI: all local required gates passed; all six required
   GitHub checks are SUCCESS on the implementation head and were re-verified
   after report publication.
8. PR/publication: PR #48 remains the sole Objective-004 PR. This report is
   the only path in the final SELF child of the literal implementation SHA.
9. Final host state: no ZAP-IT listener/process, GPU1 at its 6 MiB idle
   baseline, protected GPU0 workload unchanged, and the service shared-memory
   root empty.

## Verification

- `git fetch --prune origin`: PASSED — remote branch/base reconciled before
  mutation.
- `git diff --check` and staged diff check: PASSED.
- `.venv/bin/pytest -q tests/test_core_renderers.py tests/test_runtime_units.py tests/test_live_runtime.py tests/test_live_service_units.py tests/test_service_units.py tests/test_src_exports.py`: PASSED — 140 focused tests before the final launcher test.
- `.venv/bin/pytest -q tests/test_live_service_units.py::test_launcher_shm_rejection_is_sanitized_before_service_start`: PASSED — sanitized exit code 2 and no traceback.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — `315 passed, 1 skipped`, two known deprecation warnings, total coverage `75.88%`; the single skip was the explicit opt-in GPU integration marker.
- `.venv/bin/ruff format --check .`: PASSED — 116 files already formatted.
- `.venv/bin/ruff check .`: PASSED.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh`: PASSED.
- `.venv/bin/python -m compileall -q src scripts/serve_local.py`: PASSED.
- `.venv/bin/python -m build --wheel`: PASSED — wheel built; only upstream setuptools license-metadata deprecation warnings.
- Wheel import check from `zap_it-0.1.0-py3-none-any.whl`: PASSED — package import and `IdentityMaskProjectionError` import resolved.
- Bounded random matching property check: PASSED — 500 cases agreed with
  brute-force injective-assignment existence.
- Negative canonical-root launch through `scripts/serve_local.sh start`:
  PASSED — rejected with exit code 2 before model/listener work; no traceback;
  no GPU1 compute process was created.
- Fresh physical GPU/port/shared-memory snapshots before live work: PASSED —
  target GPU1 was idle at 6 MiB, protected GPU0 was separately occupied, and
  port `127.0.0.1:17891` was free.
- `scripts/serve_local.sh start`, `/healthz`, `/readyz`: PASSED — one
  loopback worker reached genuine readiness on the pinned visible GPU1.
- `timeout 900 .venv-gpu/bin/python scripts/smoke_local_service.py --port 17891 --levels 0 1 2 3 --formats json zip --repeat 3`: PASSED — 10 cases.
- `timeout 300 .venv-gpu/bin/python scripts/smoke_local_service.py --port 17891 --levels 0 1 2 3 --formats json --repeat 1`: PASSED after the required stop/restart — 6 cases.
- Runtime-log safety scan using `rg -n -i '(/dev/shm|/synology|/tmp/|/home/|traceback|filename|authorization|bearer|prompt|answer|secret|password|api[_-]?key)'`: PASSED — no matches.
- Final `ss`, service-process, `nvidia-smi`, compute-process, and
  `find /dev/shm/slaif-zap-it -mindepth 1 -maxdepth 3`: PASSED — stopped,
  clean, GPU1 at 6 MiB, only protected GPU0 PID 66522 remained, and no
  shared-memory children remained.
- Fully occluded live fixture: SKIPPED — the bounded live fixture produced
  disjoint masks; the adversarial CPU matching tests are the authoritative
  evidence for that combinatorial case and no live occlusion claim is made.
- Previously accepted busy/failure/deadline/cancellation/response-limit
  behavior: NOT RUN — inherited from 004-b as authorized; no touched path
  regressed in the required corrected-head smoke.

## CI/checks

Implementation head `7076c305…` and the final report head each had all six
required checks present and SUCCESS:

- `Analyze (python)`: SUCCESS
- `CodeQL`: SUCCESS
- `static (format, lint, build)`: SUCCESS
- `tests (py3.10)`: SUCCESS
- `tests (py3.11)`: SUCCESS
- `tests (py3.12)`: SUCCESS

## GPU/service/resource evidence

- Physical GPU0: NVIDIA GeForce RTX 2080 Ti, UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI `00000000:00:08.0`,
  11264 MiB; protected PID 66522 was the only GPU0 compute process in all
  snapshots.
- Physical GPU1: NVIDIA GeForce RTX 2080 Ti, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI `00000000:00:0C.0`,
  11264 MiB; 6 MiB idle baseline, 5749 MiB during the corrected live run,
  and 6 MiB after stop.
- Service launch used `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=1`; application provenance reported one visible
  logical device `cuda:0`, the pinned UUID, and strategy
  `sam2_clip_resident_blip3_rejected`. BLIP3 remained rejected before load.
- Service binding was loopback-only at `127.0.0.1:17891`; one process, one
  worker, and one active inference slot were used.
- Runtime artifacts were mode 0700/0600 as applicable; request data stayed in
  memory. The canonical service shared-memory root was empty after stop.

## Documentation/provenance

`docs/runtime.md` documents canonical `/dev/shm` containment, symlink/traversal
rejection, sanitized startup logging, and cleanup behavior. No model weights,
credentials, customer data, raw request content, or generated request artifacts
entered the repository or OAP evidence.

## Deferred human adjudication

- Critical register action: NONE
- The order explicitly decided `NONE`; `CRITICAL.md` was not read or modified.

## Safety/scope confirmations

- No new branch, PR, merge, auto-merge, release, LAN exposure, firewall/VPN,
  systemd, Docker, CUDA/driver, GPU0, unrelated process, or unrelated service
  mutation.
- No model capability, threshold, class-order, BLIP3 enablement, worker/queue,
  legacy CLI, or persistent request-data behavior changed.
- Existing 004-a and 004-b orders/reports were not edited, deleted, renamed,
  or replaced.

## Limitations/blockers

No blockers. A live fully occluded fixture was not produced; the ordered CPU
matching adversarial tests provide the authoritative proof for that case.

## Factual strategic follow-up

PR #48 remains open for strategic review/acceptance. Coding has not merged or
selected a next order.
