# OAP Work Order — 004-c — Close identity, log-safety and shared-memory review gaps

Objective `004-c`, continuation of numeric Objective 004 on the existing PR.
Correct three concrete defects found by independent strategic review of the
otherwise complete `004-b` implementation: incomplete identity-mask bijection
allocation, a false no-host-path runtime-log claim, and non-canonical
`/dev/shm` containment validation. Preserve all accepted `004-b` behavior and
evidence outside this narrow remediation. Do not rewrite any activated order or
existing report.

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Numeric objective / round: `004 / 004-c`.
- Mode: `AMEND_EXISTING_PR`.
- Remote `main`: `1a4272d60c52cc045f57f2842652485efdb7a55c`, accepted
  Objective-003 merge.
- Existing branch: `oap/004-a-loopback-service-activation`.
- Existing sole Objective-004 PR: #48, title `Objective 004: loopback service
  activation on physical GPU1 with recovered live E2E evidence`, base `main`,
  mergeable/clean at review.
- Current remote/local branch and PR head:
  `d297833b1d1d542574ed6959d7efd903a1d13909`.
- `d297833...` is the immutable report-only `004-b` commit. Its first parent is
  implementation SHA `fea4319c9bfd4845c347a89ac46ef48fcd43fa97`, and it changes
  only `oap/reports/004-b-report.md`.
- All six report-head CI/CodeQL checks were SUCCESS before this order. New
  commits must trigger and pass the same complete check set.
- Required new report: `oap/reports/004-c-report.md`. Never edit, delete,
  rename or replace `004-a-report.md` or `004-b-report.md`.

## Strategic review findings — binding corrections

### 1. Complete identity-mask bijection

The `004-b` service calls `render_identity_png(..., ensure_all_ids=True)`. Its
current greedy row-major reservation can falsely fail even when an injective
object-to-source-pixel assignment exists. Example: missing object A can use
pixels `{p,q}`, missing object B can use only `{p}`; greedy A→p makes B fail,
while A→q and B→p is valid. The single added test covers only one missing
object and does not prove the documented “fails only when distinct pixels
cannot be reserved” claim.

Replace the greedy repair with a deterministic, bounded, complete assignment
for the contract it claims:

- each response object must receive one distinct representative pixel that is
  true in that object's retained source mask;
- preserve the existing larger-area/tie winner canvas everywhere except the
  minimum deterministic representative-pixel overrides needed for bijection;
- prefer already visible winner pixels where possible, then deterministic
  row-major candidates with stable instance-ID ordering;
- succeed whenever an injective assignment exists within the bounded object/
  pixel inputs; fail with a typed core error only when it is mathematically
  impossible, not because of greedy ordering;
- keep legacy callers unchanged when `ensure_all_ids=False`;
- retain source masks/overlap truth unchanged;
- map an impossible projection to the existing sanitized stable
  `inference_failure` response with an accurate identity-representation
  message, not the inaccurate current “object limit” message.

Add focused tests for: one occluded object; the adversarial `{p,q}` / `{p}`
case; three-way reassignment/augmenting behavior; an actually impossible case
(two objects whose only source pixel is the same single pixel); deterministic
bytes; no changes to the default legacy winner policy; and stable sanitized
service error mapping.

### 2. Runtime log safety and truthful evidence

`004-b` reports that a final service-log scan found no host paths, but
`src/runtime/live_service.py` explicitly writes
`shm_root=/dev/shm/slaif-zap-it` to the runtime log. This contradicts both the
report claim and the active order's log-safety requirement.

- Remove absolute shared-memory, repository, model-cache and other host paths
  from service runtime-log content. Emit only a safe logical fact such as
  `shm_ready=true` plus bounded free capacity.
- Preserve useful PID, loopback endpoint, device identity, strategy, status and
  timing facts that are allowed by the order.
- Add a focused log-line/unit assertion that uses deliberately secret-looking
  operator paths and proves they do not appear.
- In the `004-c` report explicitly acknowledge that the immutable `004-b`
  no-host-path statement was overbroad/incorrect and record the corrected exact
  scan. Do not rewrite `004-b` history.

### 3. Canonical ephemeral-root containment

The entrypoint currently checks only
`str(Path(tmp_root)).startswith("/dev/shm/")`. Text such as
`/dev/shm/../../tmp/zap-it` passes that test while resolving outside the
RAM-backed boundary, and intermediate symlinks can also escape.

- Resolve/canonicalize the configured root before creation/use and require it
  to be a strict descendant of canonical `/dev/shm`.
- Reject lexical traversal, an intermediate symlink escape, `/dev/shm` itself,
  and any resolved target outside `/dev/shm` before model loading or listener
  creation.
- Continue rejecting a symlink final root and insecure permissions.
- Do not broaden to `/tmp` or persistent-disk fallback.
- Add CPU tests for the default valid root, normalized valid descendants,
  `..` escape, intermediate-symlink escape, `/dev/shm` itself and an ordinary
  persistent path. Error text must be sanitized.

## Current host constraints

At strategic review, physical GPU1 remained idle at 6 MiB with no compute
process and both candidate ports were free. Physical GPU0 retained only the
unrelated protected PID 66522; its memory had independently changed with that
workload and must be snapshotted fresh rather than assumed byte-constant from
this observation. Re-verify all GPU UUID/PCI/name/VRAM/process facts, ports and
`/dev/shm` immediately before live work.

- Physical GPU1 target UUID remains
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`.
- Launch only with `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`, and the pinned UUID.
- Never allocate on, stop, reset or otherwise touch physical GPU0 or PID 66522.
- One process, one worker, one active inference; loopback only.
- Final state STOPPED: no listener/process, GPU1 idle, and the service
  shared-memory root empty.

## Required verification

### CPU/static/package

- focused renderer/envelope/runtime/log-safety tests including every adversarial
  case above;
- complete canonical CPU suite with exact pass/skip/coverage counts;
- Ruff format and lint, shell syntax, compile, wheel build/import checks and
  `git diff --check`;
- inspect the exact diff against `d297833...` and keep it limited to these
  corrections, their tests/docs, the immutable `004-c` order transcript and
  `oap/active`.

### Live GPU1/service

- negative canonical-root traversal/escape launch proof: fail before model
  load, listener or GPU1 allocation;
- normal start with fresh port/GPU evidence and genuine readiness;
- rerun real L0–L3 JSON and ZIP plus BLIP3 rejection and repeat cases on the
  corrected current head, proving YOLO/object/PNG IDs remain bijective;
- capture a real fully occluded case if the bounded fixture produces one;
  otherwise the adversarial CPU matching tests are the authoritative proof for
  that combinatorial case and must not be represented as live evidence;
- inspect the actual runtime log for raw inputs, filenames, credentials,
  headers, prompts/answers, traceback text, repository/cache/shared-memory host
  paths, and secret-looking injected operator path strings;
- stop/restart once, perform one post-restart E2E request, then final stop;
- final scans: no ZAP-IT listener/process, only the unrelated GPU0 process,
  GPU1 at idle baseline and no child under `/dev/shm/slaif-zap-it`.

Previously accepted busy/failure/deadline/cancellation/response-limit behavior
need not be repeated unless these corrections touch their paths or a regression
appears. Report that this evidence is inherited from `004-b`, not freshly run.

## Non-goals and protected boundaries

- no new branch, PR, numeric objective or change to the PR title;
- no rewrite of `004-a`/`004-b` orders or reports;
- no scientific model, threshold, class-order or legacy CLI behavior change;
- no LAN/public exposure, TLS, proxy, firewall/VPN, Docker or installed systemd;
- no BLIP3 enablement, multi-worker/GPU, queue expansion or persistent request
  data;
- no broad renderer/API/runtime refactor unrelated to the three findings;
- no GPU0 or unrelated service/process mutation;
- coding never merges.

## Acceptance criteria

1. Identity projection is deterministic and complete whenever an injective
   source-pixel assignment exists, and fails safely only when none exists.
2. Default legacy identity rendering is byte-compatible outside the explicit
   service mode; source masks remain unchanged.
3. Impossible identity projection produces a stable sanitized response whose
   message accurately describes identity representation.
4. Runtime logs contain no filesystem host path or injected secret-looking
   operator path, and the new report corrects the prior overclaim honestly.
5. Canonical temp-root validation cannot escape `/dev/shm` through `..`,
   symlinks or textual-prefix tricks and fails before CUDA/listener mutation.
6. Real supported L0–L3 JSON/ZIP behavior remains green on physical GPU1 with
   exact object/YOLO/PNG ID bijection.
7. Full CPU/static/package gates and every expected GitHub CI/CodeQL check are
   SUCCESS on the final report head.
8. PR #48 remains the sole Objective-004 PR, and the `004-c` final commit is a
   one-path report-only SELF child of the literal remediation implementation
   SHA.
9. Final host state is stopped/clean and physical GPU0/unrelated workloads were
   untouched.

## GitHub/OAP publication

- Amend PR #48 on `oap/004-a-loopback-service-activation`; do not create any PR.
- Commit/push all correction code, tests, docs, exact order transcript and
  `oap/active` before the report.
- Require all expected current-head checks present and SUCCESS; pending,
  skipped, missing or failed is not green.
- Capture the literal remediation implementation SHA.
- Create exactly one immutable `oap/reports/004-c-report.md` with that SHA and
  `Report publication commit: SELF`.
- The final report commit changes only that report path and has the remediation
  implementation SHA as first parent. Push and verify remote head, parent,
  path, exact bytes and all report-head checks before response signaling.

## Deferred human adjudication

- Decision: `NONE`

These findings are ordinary correctness, evidence and containment bugs with
clear least-privilege fixes. They do not meet the `CRITICAL.md` threshold.

## Coding response

Send exact FIFO `OK` only after PR #48, the complete corrected evidence, final
stopped host, and immutable `004-c` report-head CI are verified. Coding never
merges.
