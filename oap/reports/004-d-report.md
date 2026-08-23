# OAP Coding-Agent Report — 004-d

## Work order

- Identifier: `004-d`; numeric objective `004`; mode `AMEND_EXISTING_PR`
- Repository: `ulfe-lmi/slaif-zap-it`
- Existing PR: #48, `Objective 004: loopback service activation on physical GPU1 with recovered live E2E evidence`

## Status

COMPLETE

## Executive summary

Replaced the structurally unbounded 004-c identity representative matcher with
a deterministic complete augmenting-path assignment. Source-mask candidates are
scanned in row-major fixed-size NumPy chunks; Python state is limited to object
count, assignment/path state and one candidate chunk. Already-bijective baseline
canvases use a no-matching fast path. Existing winner behavior, legacy rendering,
typed impossible-projection errors, overlap/source-mask retention and service
contracts remain intact.

Focused CPU evidence includes the retained adversarial cases, a multi-object
augmenting chain, 500 brute-force existence comparisons, fast-path instrumentation,
and a 700x700 broad-mask allocation/chunk-bound regression. The full CPU suite,
package/static checks, all six implementation-head GitHub checks, and the ordered
fresh physical-GPU1 loopback smoke/restart evidence passed. Final host state is
stopped and clean.

## Authoritative GitHub state

- PR: [#48](https://github.com/ulfe-lmi/slaif-zap-it/pull/48), OPEN, mergeable,
  clean; base `main`; existing branch
  `oap/004-a-loopback-service-activation`
- PR title unchanged; no new PR; existing PR amended; coding merge: NO
- Remote `main` / base SHA: `1a4272d60c52cc045f57f2842652485efdb7a55c`
- Starting branch/report head SHA: `bfee4d31371306d922f0a80c53093b96225af48c`
- Implementation head SHA: `76b6e5407c1739b9513c6f89a951d480d6b3eae2`
- Implementation parent SHA: `bfee4d31371306d922f0a80c53093b96225af48c`
- Report publication commit: SELF

## Changes/files

Implementation commit `76b6e5407c1739b9513c6f89a951d480d6b3eae2` contains only:

- `src/core/renderers.py` — bounded chunked matching, baseline fast path and
  deterministic iterative augmenting paths; legacy raster path unchanged
- `tests/test_core_renderers.py` — chain, fast-path, bounded-memory and 500-case
  existence coverage
- `docs/CORE.md` — accurate bounded matching and non-minimality contract
- `oap/active` — exact active selector `004-d`
- `oap/orders/004-d-bound-identity-matching-memory.md` — immutable order transcript

No earlier order or report was rewritten.

## Acceptance evidence

1. No auxiliary per-mask-pixel Python graph, tuple, set or edge collection is
   present. The matcher retains object-sized NumPy assignment/cursor arrays,
   an object-index stack and one candidate array bounded by
   `IDENTITY_CANDIDATE_CHUNK_SIZE = 65536`.
2. Complete deterministic matching passed all retained 004-c adversarial,
   impossible, determinism and legacy tests, the explicit multi-representative
   augmenting-chain case, and 500 deterministic brute-force existence cases.
3. The 700x700 broad-mask regression exercised a 490,000-pixel mask that would
   have created hundreds of thousands of old Python graph elements. Direct
   instrumentation observed candidate arrays no larger than 65,536 and the
   test passed its conservative 16 MiB Python-allocation peak bound.
4. The fast-path test proved `_candidate_chunk` is not entered when every ID is
   already visible; baseline output is therefore preserved byte-for-byte.
5. `ensure_all_ids=False`, retained source masks, uint16 dimensions/IDs,
   deterministic winner behavior and the stable typed impossible-projection
   error remained green. Fresh real service smoke preserved eight YOLO/object/
   PNG IDs at L1-L3.
6. PR #48 remains the sole Objective-004 PR. The implementation was pushed
   before this report; all six required checks were SUCCESS on its head.

## Verification

- `git fetch --prune origin`: PASSED — remote branch, base and PR were
  reconciled before mutation.
- `.venv/bin/pytest -q tests/test_core_renderers.py`: PASSED — 22 tests.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  PASSED — 319 passed, 1 explicit opt-in GPU module skip, 75.99% total
  coverage against the 64% gate.
- `.venv/bin/ruff format --check .`: PASSED — 116 files formatted.
- `.venv/bin/ruff check .`: PASSED.
- `bash -n scripts/serve_local.sh scripts/serve_local_stop.sh`: PASSED.
- `.venv/bin/python -m compileall -q src scripts/serve_local.py`: PASSED.
- `.venv/bin/python -m build --wheel`: PASSED — produced the project wheel;
  only existing setuptools license deprecation warnings were emitted.
- Wheel extraction/import check for `zap_it-0.1.0-py3-none-any.whl`: PASSED —
  imported `src` and verified the bounded renderer constant.
- `git diff --check` and staged diff check: PASSED.
- Changed-file secret-pattern scan: PASSED — no private-key, credential or
  secret assignment pattern matched. Changed files remained small source/docs/
  transcript files; no weights, cache, image or generated large artifact was
  added.
- `gh pr checks 48 --watch --interval 10`: PASSED on implementation head —
  `Analyze (python)`, `CodeQL`, `static (format, lint, build)`, `tests (py3.10)`,
  `tests (py3.11)` and `tests (py3.12)` all passed.
- Fresh GPU/process/port/shared-memory preflight: PASSED — physical GPU1 was
  idle, port `127.0.0.1:17891` was free, and the canonical shared-memory root
  had no residue.
- Loopback start, `/healthz` and `/readyz`: PASSED — one real worker reached
  readiness on pinned GPU1.
- `timeout 900 .venv-gpu/bin/python scripts/smoke_local_service.py --port 17891 --levels 0 1 2 3 --formats json zip --repeat 3`:
  PASSED — 10 cases; L0-L3 JSON/ZIP, three stable repeats, eight YOLO lines/
  objects/PNG IDs and BLIP3 rejection.
- Required stop/restart and post-restart readiness: PASSED — one controlled
  stop returned GPU1 to idle before restart.
- `timeout 300 .venv-gpu/bin/python scripts/smoke_local_service.py --port 17891 --levels 0 1 2 3 --formats json --repeat 1`:
  PASSED after restart — 6 cases including L1-L3 identity artifacts and
  BLIP3 rejection.
- Final stop plus `ss`, process, `nvidia-smi` and shared-memory snapshots:
  PASSED — no ZAP-IT listener/process or shared-memory child remained.
- Previously accepted failure/busy/deadline/cancel/response-limit and log-
  safety evidence: INHERITED — order 004-d authorizes inheritance because the
  changed paths are identity rendering/tests/docs only; fresh normal smoke
  showed no regression.
- Live fully-occluded model-generated fixture: SKIPPED — the existing fixture
  remains disjoint; the ordered CPU adversarial/existence tests are the
  authoritative matching evidence and no live occlusion claim is made.

## CI/checks

Implementation head `76b6e5407c1739b9513c6f89a951d480d6b3eae2` had all six
required checks SUCCESS:

- `Analyze (python)`: SUCCESS
- `CodeQL`: SUCCESS
- `static (format, lint, build)`: SUCCESS
- `tests (py3.10)`: SUCCESS
- `tests (py3.11)`: SUCCESS
- `tests (py3.12)`: SUCCESS

## GPU/service/resource evidence

- Physical GPU0: NVIDIA GeForce RTX 2080 Ti, UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI `00000000:00:08.0`,
  11264 MiB; protected PID 66522 remained the only compute process at 2492 MiB
  in pre-run and final snapshots.
- Physical GPU1: NVIDIA GeForce RTX 2080 Ti, UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI `00000000:00:0C.0`,
  11264 MiB; 6 MiB before live work, 5749 MiB during service inference and
  6 MiB after final stop.
- Launch used `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES=1`; application provenance reported one visible
  logical device `cuda:0` and the pinned GPU1 UUID. No GPU0 allocation or
  process mutation occurred.
- Service binding was IPv4 loopback-only at `127.0.0.1:17891`, with one
  process, one worker and one active inference slot. The service was started,
  stopped, restarted once, smoke-tested, and finally stopped.
- `/dev/shm` was a 27 GiB tmpfs. Runtime artifacts were mode 0700 for the
  private runtime directory and mode 0600 for pid/log files; the root was
  empty after final stop. Request data stayed in memory.

## Documentation/provenance

`docs/CORE.md` now documents baseline seeding, deterministic augmenting paths,
fixed candidate chunks and the absence of a global minimum-override claim.
The implementation uses the already qualified Objective-003 resident model
profile; no model identity, threshold, class order, BLIP3 capability or legacy
CLI behavior changed. No model weights, credentials, customer data, raw
requests, cache paths or generated request artifacts entered the repository or
OAP evidence.

## Deferred human adjudication

- Critical register action: NONE
- The active order explicitly decided `NONE`; `CRITICAL.md` was not read or
  modified.

## Safety/scope confirmations

- No new branch, PR, merge, auto-merge, release, LAN exposure, firewall/VPN,
  systemd, Docker, CUDA/driver, GPU0, unrelated process or unrelated service
  mutation occurred.
- No API limit weakening, object dropping, scientific model/threshold/order
  change, BLIP3 enablement, runtime refactor or persistent request-data change
  occurred.
- Existing 004-a, 004-b and 004-c orders/reports remain immutable.

## Limitations/blockers

No blockers. The live fixture did not produce overlapping/fully occluded masks;
the CPU chain, impossible and 500-case existence tests provide the ordered
proof for those cases. Prior accepted failure/recovery evidence is inherited as
authorized above.

## Factual strategic follow-up

PR #48 remains open for strategic review/acceptance. Coding has not merged,
accepted, released or selected a next order.
