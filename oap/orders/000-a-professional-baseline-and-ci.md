# OAP Work Order — 000-a — Professional baseline and CI

Objective `000-a`. Strategic-activated final order. Bring the existing ZAP-IT
repository to an honest professional baseline: preserve and characterize current
behavior; establish reproducible package/dev setup; audit, repair and extend CPU
tests; enable CI and CodeQL; refresh core documentation/security/provenance.
Do not implement the service API and do not mutate GPU runtime in this objective.

## GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `000 / 000-a`
- Mode: `CREATE_NEW_PR`
- Verified default branch and 40-hex SHA: `main` @
  `12257a6c31ad654d1b7114e50cf9679fcc2fb260` (verified live via `gh` and local
  clone on 2026-08-23; local tree clean and synced, nothing to commit)
- Required new branch name: `oap/000-a-professional-baseline-and-ci`
- Existing objective PR: none. `gh pr list --state all` shows no open PRs;
  latest merged PR is #43; no objective PR exists.
- Required PR title: `Objective 000-a: professional baseline, honest CPU test
  repair, packaging and CI`

## Verified current state (all evidence gathered live 2026-08-23)

Repository/test/package/CI/docs state:

- Layout: pipeline code in `modules/` (input, segmenter, classifier, verifier,
  geometry, output, visualizer) plus `src/` (batch.py, config.py,
  postprocessing.py); CLI scripts `zap-it-batch.py`, `zap_it_config.py`,
  `zap_it_postseg_processing.py` compatibility shim, `huggingface_downloader.py`.
- Packaging: none. No `pyproject.toml`, `setup.py`, or `setup.cfg` exists.
  `environment.yml` targets conda env `zap-it` with python=3.10, torch 2.3.x,
  torchvision/torchaudio 0.18/2.3, pytorch-cuda=12.1, and pip installs of
  detectron2, SAM2, open-clip-torch, transformers>=4.41, opencv-python-headless.
  These heavy deps are NOT installed host-wide.
- Tests: 21 files under `tests/`. `tests/conftest.py` injects fake `torch`,
  `PIL`, `detectron2`, `huggingface_hub`, `cv2`, and even `yaml` modules when
  real ones are absent. Consequence: the suite runs CPU-only in well under a
  second with only numpy+pytest installed, but real-library behavior (true YAML
  parsing, true PIL imaging) is currently untested. This must be inventoried
  honestly, not silently accepted and not silently deleted.
- Exact existing CPU test command/result/count/failures: with CPython 3.12.3
  and a venv containing only numpy+pytest, `python3 -m pytest -q` fails during
  collection: `ERROR tests/test_huggingface_downloader.py — ImportError:
  cannot import name 'resolve_output_dir' from 'huggingface_downloader'`.
  Excluding that file (`--ignore=tests/test_huggingface_downloader.py`):
  `1 failed, 63 passed` where the failure is
  `tests/test_src_exports.py::test_src_re_exports_match_batch`; total wall time
  ~0.3 s; no network, no GPU access.
- Docs: README.md, INSTALL.md, TESTING.md, SECURITY.md, ALGORITMS.md (typo
  filename), docs/, configs/, demos/ exist. Tracked generated/legacy artifacts
  confirmed by `git ls-files`: `everything.txt` (~300 KB generated concatenation
  at repo root), `ALGORITMS.md`, and binary result images under
  `last_results/*.jpg`. LICENSE is MIT (Janez Pers / jpers1).
- Current GitHub Actions/CodeQL/branch-protection facts: no `.github/workflows`
  directory exists; `gh run list --branch main` returns empty (CI has never
  run); branch protection API returns 404 ("Branch not protected").
- Local dirty/unrelated work preservation plan: working tree is clean; create
  the objective branch from `origin/main` at the verified SHA above; never
  reset, clean, rebase unrelated refs, or discard anything.
- All GPUs/processes summarized read-only (no GPU used by this objective):
  physical GPU0 = UUID `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI
  `00000000:00:08.0`, NVIDIA GeForce RTX 2080 Ti, 11264 MiB total, ~2161 MiB
  used by unrelated process PID 66522 (`/opt/venv/bin/python`) — PROTECTED,
  absolutely untouched by this objective. Physical GPU1 = UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI `00000000:00:0C.0`,
  NVIDIA GeForce RTX 2080 Ti, 11264 MiB total, idle (~6 MiB). Objective 000
  performs zero GPU allocation on either device; snapshot before/after must
  prove it. Host context read-only facts: `/dev/shm` is tmpfs with 27G free;
  loopback listeners currently include 127.0.0.1:32995 and 631 (cups) plus LAN
  listener :8000; none are relevant or mutable here.

## Scope

1. Inventory current public CLI, config grammar, modules, outputs, tests and
   docs; record a concise baseline characterization (e.g. docs/BASELINE.md or a
   dedicated TESTING.md section) that explicitly lists which test layers run
   against stubbed modules versus real libraries.
2. Add modern `pyproject.toml` package metadata (setuptools backend acceptable)
   with a `[dev]` extra containing pytest, ruff, coverage tooling and build
   support, without breaking existing entry points. The dev install MUST NOT
   pull torch/SAM2/detectron2/transformers or any model download.
3. Make existing CPU/mocked tests deterministic and runnable without CUDA,
   model download, network, private data or repository output mutation.
   Specifically repair the two live red items honestly:
   a. fix collection failure in `tests/test_huggingface_downloader.py` by
      aligning it with the actual exports of `huggingface_downloader.py`, or
      restoring the intended function if commit history shows module drift —
      document which and why;
   b. repair `tests/test_src_exports.py::test_src_re_exports_match_batch`
      against the actual current `src/` export surface, preserving the test's
      intent rather than deleting assertions.
4. Fix genuine in-scope defects exposed by tests; do not replace behavior with
   mocks merely to pass. Every intentional behavior change gets a one-line
   rationale in the PR body or report.
5. Add focused missing tests proportionate to baseline gaps: package/import
   smoke, entrypoint/config loading exercised through REAL `yaml.safe_load`
   (at least minimal real-YAML config grammar coverage, since the current suite
   tests YAML only through the fake), and current pure pipeline boundaries.
6. Add Ruff format/lint policy and pytest config inside `pyproject.toml`;
   measure coverage for `src/` and `modules/`; set a defensible initial gate
   (measured value minus a small documented ratchet). If immediate enforcement
   is technically unsafe, document a precise staged ratchet instead — never an
   invented high number.
7. Add least-privilege GitHub Actions CI running on PRs and main pushes:
   static/format job, CPU test matrix over Python 3.10/3.11/3.12 on
   ubuntu-latest executing the canonical suite, and a CodeQL workflow (python
   queries, `security-events: write`, other jobs `contents: read`, pinned major
   action versions). No secrets, no GPU, no model downloads in CI.
8. Refresh README, installation, configuration navigation and testing
   instructions so they match verified commands exactly and make no API/GPU
   service readiness claims. Keep the conda GPU environment documented as-is
   for later objectives.
9. Add/refresh CONTRIBUTING.md, SECURITY.md review, THIRD_PARTY_NOTICES.md and
   provenance/dependency/model notes (SAM2/detectron2/open-clip/transformers
   provenance and license pointers), preserving the existing MIT license.
10. Audit tracked generated/legacy artifacts — `everything.txt`,
    `last_results/*.jpg`, `ALGORITMS.md` naming typo, any stray caches. For each
    either remove from tracking with a compatibility/migration explanation and
    .gitignore hygiene, or retain with explicit justification. Renames must not
    break existing references without an update.

### Provisional strategic decisions (binding for this objective)

- Supported Python range: `>=3.10,<3.13`; CI matrix 3.10/3.11/3.12. Rationale:
  environment.yml pins 3.10 while host system CPython is 3.12.3 and the current
  suite already passes under 3.12.
- Dev environment contract: plain `python3 -m venv .venv` +
  `pip install -e '.[dev]'`. The conda `environment.yml` remains untouched as
  the future full-GPU stack reference.
- De-stubbing the entire conftest is NOT required in objective 000; mandatory
  deliverable is the honest inventory plus the minimal real-YAML coverage in
  item 5. Larger de-stubbing belongs to later objectives.
- Artifact audit outcomes in item 10 are coding-executed but each decision must
  be stated and justified in the PR description.

## Non-goals

- no `/v1/completions`, FastAPI, service schemas, Docker/systemd or API key;
- no in-memory core refactor beyond minimal package/test seams;
- no model download, real inference or GPU use of any kind;
- no CUDA/driver/environment/system package/firewall/port/service changes;
- no physical GPU0 or GPU1 allocation;
- no scientific algorithm/threshold/default change unless required to fix a
  clearly demonstrated bug and explicitly documented/tested;
- no model weights/results corpus committed;
- no adjacent roadmap implementation (objectives 001+ content).

## Acceptance criteria

1. A clean clone can install the documented CPU/dev test environment using
   supported Python (3.10–3.12) via the canonical command without installing or
   downloading GPU models.
2. One canonical CPU command runs the complete existing+new CPU suite; all
   tests pass; count and duration reported in the report; no network/GPU/model
   dependency.
3. `ruff format --check .` and `ruff check .` pass; wheel builds; package
   imports and preserved CLI entry points smoke-test successfully.
4. Coverage is measured with explained exclusions and an initial non-regressive
   gate enforced, or a precise staged ratchet documented if immediate
   enforcement is technically unsafe.
5. GitHub CI workflows run on PR and main with least privileges and execute the
   canonical static/package/CPU suite; CodeQL enabled and green, or the exact
   repository-policy blocker reported verbatim in the report.
6. Existing supported CLI/config behavior remains covered and documented; any
   intentional compatibility change is explicit.
7. README/install/config/testing/security/contributing/provenance/third-party
   docs are consistent with actual code and do not claim API or GPU readiness.
8. No large generated/model/cache/private artifact or secret enters the diff.
9. No GPU process, listener, system service, firewall, global OpenCode config
   or unrelated host state is changed; before/after GPU snapshots show zero
   allocations attributable to this objective.
10. Correct branch and exactly one PR exist from `main` @
    `12257a6c31ad654d1b7114e50cf9679fcc2fb260`; required checks pass; coding
    agent never merges; immutable report-only SELF child commit is the remote
    PR head before the response signal.

## Required verification (exact commands)

- Clean CPU/dev install: `python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'`
- Package build/import/CLI smoke: `.venv/bin/python -m build --wheel`, import
  of packaged modules, and `--help` smoke of preserved CLI entry points
- Ruff: `.venv/bin/ruff format --check . && .venv/bin/ruff check .`
- Full CPU pytest+coverage: `.venv/bin/pytest -q --cov=src --cov=modules
  --cov-report=term-missing` (all green; counts + duration reported)
- No-network/no-CUDA assertion: suite-level guard or equivalent test proving
  no torch.cuda/network/model-download path executes; CI runners have no GPU
- Docs/config/schema examples: rendered examples validated against the real
  YAML loader in tests
- Secret/large-artifact scan: state method (e.g. grep-based scan plus diff size
  review) and result
- GitHub CI and CodeQL check names: list exact required check names observed on
  the PR head; all present and successful, none pending/failed/missing
- Read-only before/after GPU process snapshot proving no objective allocation:
  `nvidia-smi --query-gpu=index,uuid,memory.used --format=csv` plus compute-apps
  query captured at start and end of execution, included in the report

## Security/resource constraints

Treat the host as shared. Routine repo-local environment setup (venvs, pip into
`.venv`) is coding-agent work. Never use GPU, download model weights, modify
global Conda/OpenCode configuration, system drivers/CUDA, other processes,
ports, services, firewall/VPN, or sudo system state. Preserve unrelated
working-tree files and untracked local content. Do not print credentials or
private provider configuration. Report prose and green CI are evidence, not
acceptance.

## Deferred human adjudication

- Decision: `NONE`

No dilemma in this objective satisfies all five CRITICAL.md threshold
conditions. Packaging backend choice, Python support window, CI matrix, Ruff
policy, coverage ratchet and legacy-artifact dispositions are ordinary
reversible engineering decisions resolved by this order, the constitutions and
the architecture; none can materially affect a security boundary, privacy,
trust model, deployment safety or release acceptability at this stage. Do not
create a CRITICAL entry merely because modernization choices require judgment.
If implementation unexpectedly exposes a genuinely material dilemma meeting all
five conditions, report it as a candidate in the report and continue all
unambiguous safe scope; strategic decides next round. Coding may not invent the
entry.

## GitHub publication and report

Create exactly one new objective branch `oap/000-a-professional-baseline-and-ci`
from remote `main` at the verified base SHA and exactly one PR titled as
specified above. Push all non-report work including the exact activated order
and active pointer transcript under `oap/`. Inspect and, if in-scope, fix CI on
the current head. Capture the literal implementation SHA. Then publish exactly
one immutable report-only final commit (child of the implementation SHA) whose
message/body states the literal implementation SHA and `Report publication
commit: SELF`; push it, verify remote PR head equals the report commit and its
parent equals the implementation SHA. Mutate nothing further. Send response
signal `OK` only after that verification. The report must state exact tests
(passed/failed/skipped counts), CI/CodeQL status per check name, coverage
numbers, dependency changes, files touched, artifact-audit decisions, safety
evidence (GPU snapshots), skips/failures/limitations, and any critical-entry
candidate. Never merge; strategic merges.
