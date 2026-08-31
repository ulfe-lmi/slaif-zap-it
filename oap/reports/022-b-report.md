# OAP Coding-Agent Report — 022-b

## Work order

- Identifier/order/objective/PR mode: `022-b` / close runtime boundaries and
  exact live proof / Objective 022 / amend existing PR.
- Repository: `ulfe-lmi/slaif-zap-it`.
- Pull request: [#78](https://github.com/ulfe-lmi/slaif-zap-it/pull/78), open,
  base `main`, branch `oap/022-a-canonical-clip-multiprompt`.

## Status

PARTIAL

## Executive summary

The runtime boundary corrections, operator-owned BLIP3 question capacity,
exact configuration fixture, tests, schemas, package checks, and documentation
are implemented and pushed. CPU, package, security, and all seven required
implementation-head checks passed. One authorized restart loaded the corrected
service on the assigned GPU; the authenticated negative returned HTTP 400
`invalid_config` with measured 80/77 tokens, and the one exact tomato request
returned HTTP 200 with valid ZIP/hash/count/stage evidence and a labelled PNG.

Completion is partial because the service-safe ZIP member for the configured
`final-labelled-ripe-tomatoes` visualization was emitted as
`visualization/stream-0001.png`. The PNG content is the validated final
annotated-labelled image, but the exact configured safe-ID member-name proof
required by this order is missing. No second restart or positive request was
performed.

## Authoritative GitHub state

- Repository/PR URL/state/base/head: `ulfe-lmi/slaif-zap-it`, PR #78 open,
  base `main`, head `d3e5cb29768c964f378ede462182c6808ead6b78`.
- Accepted base SHA: `d341a3c4ba47b71d10d70682771b315041dcbcb8`.
- Starting SHA: `b2d105bd507bdc2bfb06bcb068be057576301e8a`.
- Implementation head SHA:
  `d3e5cb29768c964f378ede462182c6808ead6b78`.
- Report publication commit: SELF
- New PR: no; amended existing PR #78: yes; coding merge: NO.
- The implementation commit has parent `b2d105bd…`; the 022-a report commit
  was not rewritten, amended, squashed, deleted, or replaced.

## Changes/files

One implementation commit (`d3e5cb2…`, 27 paths) contains:

- runtime/settings: `src/service/settings.py`, `src/runtime/live_service.py`,
  `modules/verifier/blip3.py`, `src/service/schemas.py`,
  `src/service/capabilities.py`, `src/service/yaml_input.py`;
- tests/fixture: `tests/test_objective_022.py`,
  `tests/test_real_yaml_config.py`, `tests/test_sam2_configuration.py`,
  `tests/test_service_units.py`,
  `tests/fixtures/configs/ripe-tomato-multiprompt.yaml`;
- maintained contract/config: `README.md`, `INSTALL.md`, `ARCHITECTURE.md`,
  `TESTING.md`, `MANIFEST.in`, `deploy/service.env.example`,
  `docs/ALGORITHMS.md`, `docs/API.md`, `docs/CONFIG.md`, `docs/CORE.md`,
  `docs/OUTPUT-PARITY.md`, `docs/RUNBOOK.md`, `docs/SERVICE-DATASHEET.md`,
  `docs/runtime.md`;
- unchanged orchestration transcript: `oap/active` and
  `oap/orders/022-b-close-runtime-boundaries-and-exact-live-proof.md`.

Migration notes: existing operator environments without
`SLAIF_ZAP_IT_BLIP3_MAX_QUESTIONS` use the safe default 256. The optional
operator setting accepts only 1..256 and is never read from uploaded YAML.
`max_new_tokens=32` remains fixed. No dependency was added and no model,
device, cache, network, deadline, artifact budget, or residency policy was
changed.

## Acceptance evidence

1. Typed prompt boundary: PASSED. The resident adapter translates
   `ClipPromptValidationError` to sanitized HTTP 400 `invalid_config`, retains
   the exception cause, and leaves unrelated runtime exceptions mapped to the
   existing HTTP 500 `inference_failure`. Focused service tests cover both.
   The live negative request returned HTTP 400, reason `token_limit`, class
   `ripe_tomato`, prompt index 0, measured token count 80, allowed limit 77,
   in 0.481788 seconds, before engine inference. No raw synthetic prompt is
   recorded here.

2. Operator BLIP3 capacity: PASSED for implementation and runtime plumbing.
   `ServiceSettings` is frozen, defaults to 256, accepts 32 and 256, and
   rejects empty, zero, negative, non-integer, and over-256 operator values
   before loading/listening. The resident loader copies the exact setting and
   retains fixed `max_new_tokens=32`. The CPU no-model plan accepts 256
   canonical routed candidates and rejects 257 before composition with typed
   `resource_limit` details. The live capability and L3 runtime metadata
   disclose 256 questions/request, operator-only control, and the BLIP3
   planning stage. No candidate is truncated or silently skipped.

3. Exact fixture: PASSED. The committed fixture is byte-for-byte the YAML
   appendix (4,090 bytes, SHA-256
   `e89a149d2f3530d1b7a4cb919b3641f230fb51fbd6a0ee59805587d4db89cd3d`). The
   public parser and fake/API path prove ordered classes and counts
   `ripe_tomato=32`, `foliage=15`, `stem_or_vine=15`,
   `greenhouse_structure=20`, `background=15`, total 97; five semantic score
   keys; route target `ripe_tomato`; and accepted canonical `falsecategory`.

4. CPU/package/CI gates: PASSED. The full CPU suite was 888 passed and 1
   explicit GPU skip with 81.98% total coverage. Focused Objective 022,
   runtime, API, YAML, schema, capability, and adjacent tests were 389 passed.
   Wheel/sdist checks, archive/member comparison, tracked-tree baseline,
   Twine, systemd template, and isolated wheel/sdist-built-wheel JSON/ZIP
   smokes passed. The public tracked secret scan remained exactly 7 reviewed
   findings.

5. Exact live result: PARTIAL. The one request returned HTTP 200 in 44.788440
   seconds and produced a 1,401,276-byte ZIP. ZIP validation passed fixed safe
   member checks, no traversal/symlink/duplicate members, artifact budgets,
   manifest hash/size parity, five-class vectors, route target, 30 final
   `ripe_tomato` objects, inclusive bbox dimensions from 12 through 113 pixels,
   and zero artifact omissions. Counts were raw SAM2 205, geometry evaluated
   205 with 172 retained and 33 rejected, CLIP scored 172, initially routed
   156, routed after cap 156, BLIP3 candidate views rendered 136 and rejected
   20 for containment, BLIP3 verified 136, and final retained 30. The final
   image member was present as `visualization/stream-0001.png`, not the
   required configured safe-ID name.

6. Live artifact evidence: PASSED except the member-name criterion. Preserved
   files are under the mode-0700 directory
   `/dev/shm/slaif-zap-it-022b.LIzzBL`:

   - ZIP: `result.zip`, 1,401,276 bytes, SHA-256
     `bfe91b43233bd8e8def9654c1f5af3f6e041ec9dd1ed68b0fcf2a196550e4a84`;
   - labelled review copy extracted from that ZIP member:
     `final-labelled-ripe-tomatoes.png`, 1,344,837 bytes, SHA-256
     `df74674dca77c563da589962461d3558eeeec1cf70349979f6593486cd3bf783`,
     RGB PNG, 1280x720, mode 0600.

   The ZIP also contained `manifest.json`, `detections.yolo.txt`, and
   `identity-mask.png` (9,401 bytes, SHA-256
   `74a3f198ccc7bac1bf98d8017efd596393bc5ec19fb4038625758174ccaebc86`).
   No prompt text appeared in any member name.

7. Visual observations: the source contains approximately 18 prominent ripe
   fruits by bounded manual inspection, with many additional small background
   fruits. The labelled output contains 30 final masks, all labelled
   `ripe_tomato`. At least 8 obvious small/background or partially occluded
   fruits appear missed. At least 4 final masks are obvious false positives on
   stems, structure, or background, including a lower-left mask. The central
   foreground and upper clusters are visibly fragmented into overlapping
   masks; at least 2 masks visibly span adjacent fruit regions. These are
   visual observations, not recall, precision, or semantic-perfection claims.

## Verification

- `git fetch origin --prune`: PASSED.
- `.venv/bin/pytest -q tests/test_objective_022.py tests/test_service_units.py tests/test_sam2_configuration.py tests/test_real_yaml_config.py tests/test_live_service_units.py`: PASSED — 389 passed.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — 888 passed, 1 explicit GPU skip, 81.98% total coverage.
- `.venv/bin/ruff format --check .`: PASSED.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current documents.
- `git diff --check` and staged `git diff --cached --check`: PASSED.
- `.venv/bin/python -m build --wheel --sdist`: PASSED.
- `scripts/verify_release_artifacts.py` on wheel/sdist and sdist-built wheel:
  PASSED.
- `scripts/verify_release_artifacts.py --compare-wheels`: PASSED.
- `scripts/scan_release_artifacts.py` archive scan: PASSED — 0 unexpected
  findings.
- `scripts/scan_release_artifacts.py --tracked-tree`: PASSED — 7 reviewed
  findings, no additions/removals.
- `.venv/bin/python -m twine check` on wheel/sdist artifacts: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- Isolated direct-wheel and sdist-built-wheel `smoke_installed_package.py`:
  PASSED — JSON/ZIP package-version parity and site-packages import.
- Exact fixture public parse/API count and route assertions: PASSED.
- No-model capacity 32/256/invalid/257 planning tests: PASSED.
- Live request ZIP safety, manifest arithmetic, hash/size parity and semantic
  count validation: PASSED; exact safe-ID visualization member name: MISSING.

## CI/checks

All checks below are on implementation SHA
`d3e5cb29768c964f378ede462182c6808ead6b78` and are `PASSED`:

- `static (format, lint, build)` — [CI job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33428694306/job/99608469798).
- `tests (py3.10)` — [CI job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33428694306/job/99608470125).
- `tests (py3.11)` — [CI job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33428694306/job/99608470121).
- `tests (py3.12)` — [CI job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33428694306/job/99608470131).
- `release (artifact audit)` — [CI job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33428694306/job/99608470172).
- `Analyze (python)` — [CodeQL workflow job](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33428694285/job/99608469752).
- `CodeQL` — [CodeQL check](https://github.com/ulfe-lmi/slaif-zap-it/runs/99608747157).

## GPU/service/resource evidence

- One controlled restart of only user unit `zap-it-lan.service`: PASSED. Old
  PID 685637 changed once to PID 697088; `NRestarts=0`; unit remained enabled,
  active, and running.
- Listener: exactly `10.8.132.76:17891`; health/readiness HTTP 200; wrong
  bearer HTTP 401; private `/docs` HTTP 404; authenticated capabilities and
  metrics HTTP 200.
- Physical GPU: index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
  GeForce RTX 3090, 24576 MiB, driver `610.43.02`; process visibility exposed
  only this assigned card as logical `cuda:0`.
- Sole compute process after live proof: PID 697088, service Python process;
  Xorg remained the only non-compute GPU row. Current device usage was 10,891
  MiB with 13,233 MiB free; Torch peak was 10,361.5 MiB allocated and 10,544.0
  MiB reserved; maximum RSS was 12,805.5 MiB.
- `/dev/shm/slaif-zap-it`: mode 0700, empty after request cleanup. The retained
  evidence directory is separate, mode 0700; retained ZIP/PNG files are mode
  0600. Negative request/config intermediates were removed.
- Environment file remained mode 0600, unchanged SHA-256
  `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`; the
  new capacity used the code default because the environment key was absent.
  The bearer was read ephemerally, never printed, copied, logged, placed in
  process arguments, committed, or included in this report.

## Documentation/provenance

Maintained README, installation, architecture, testing, API, config, algorithm,
core, parity, runbook, runtime, and service-datasheet material now describes
the canonical `string | ordered array[string]` prompt contract, independent
embedding/max aggregation/ties, scalar comma/newline behavior, typed adapter
errors, canonical `falsecategory`, operator-only BLIP3 capacity, planning-stage
`resource_limit`, and distinct response assembly `response_too_large` behavior.
The exact public synthetic fixture is included in the sdist through the explicit
manifest allowlist; no model weights, customer data, credentials, or live image
bytes were committed.

## Deferred human adjudication

- Critical register action: NONE.
- `CRITICAL.md` was read because the active order required the current-register
  refresh. No entry was appended or changed.

## Safety/scope confirmations

- Only the active `022-b` order was executed; no adjacent order was selected.
- The existing PR #78 was amended; no new PR was created and no merge or
  auto-merge was attempted.
- No unassigned GPU, unrelated process, system CUDA/driver, firewall/VPN,
  unrelated service, port configuration, environment credential, model cache,
  or persistent request-data location was mutated.
- Request YAML and response data used RAM-backed tmpfs only; raw request content
  and answers are not in this report or GitHub metadata. The committed fixture
  is the deliberate public synthetic prompt-text exception specified by the
  order.
- Existing CLI/trusted legacy prompt behavior remains intact; API capacity is
  startup-owned and request state is not reused.

## Limitations/blockers

The live response proved the corrected validation/capacity path and all required
semantic counts, but the service-safe envelope still names the configured final
visualization `stream-0001`. Therefore the exact configured safe-ID member-name
criterion is `MISSING`, and this report cannot claim `COMPLETE`. The source
visualization image itself is preserved for strategic review. No second
restart, hot reload, request retry, YAML mutation, or response fabrication was
performed.

## Factual strategic follow-up

Keep PR #78 open and adjudicate a same-PR correction for service-safe
visualization naming so the configured validated ID is emitted as the fixed
member name (without deriving names from prompt text). A future governed live
qualification must reverify the assigned GPU/service state and obtain fresh
exact-name evidence; this round's one-restart/one-positive-request allowance
was exhausted. The strongest reason not to merge is the mismatch between the
required safe-ID artifact contract and the observed generic member name. Its
answer is the explicit CPU/ZIP evidence and preserved labelled output above,
but only a source correction plus fresh governed proof can close the blocker.
