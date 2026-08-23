# OAP Coding-Agent Report — 001-a

## Work order
- Identifier `001-a`; objective 001 "In-memory core and deterministic
  renderers"; mode `CREATE_NEW_PR`; executed under active round `001-a` per
  `oap/active` (transcript carried in implementation history).

## Status
COMPLETE

## Executive summary

Extracted a typed, reusable, single-image in-memory ZAP-IT core (`src/core`)
from the legacy batch orchestration and established the deterministic
renderer/artifact-sink seam required by the future service. The engine runs
ROI/resize → SAM2 → area/bbox filter → optional CLIP/BLIP3 → label filter
entirely in memory with injectable stage callables; debug artifacts route
through logical sinks (memory or filesystem adapter) instead of direct writes;
final objects share one documented ordering definition across records, YOLO
text and the uint16 identity PNG. A demonstrated precision defect in mask
upscaling was fixed with regression evidence (unreachable ROI rows on
downscale). Legacy CLI/image/video behavior remains green through a
compatibility adapter. No API transport, GPU use, model download or network
access was involved.

## Authoritative GitHub state
- Repository `ulfe-lmi/slaif-zap-it`; PR #45
  https://github.com/ulfe-lmi/slaif-zap-it/pull/45 ; state OPEN; base `main`
  @ `ab1954484c6229168f9d12eb9964837d802aba88` (verified via live `git fetch`
  at round start with CI+CodeQL SUCCESS on that SHA).
- Starting SHA: `ab1954484c6229168f9d12eb9964837d802aba88`
- Implementation head SHA: c4f3cdc66e0d258df76aa82d3d5cc54eb8f93bd5
- Report publication commit: SELF
- New PR yes (#45); no existing objective PR amended; coding merge NO.

## Changes/files

New:
- `src/core/__init__.py`, `config.py`, `errors.py`, `ordering.py`,
  `results.py`, `renderers.py`, `sinks.py`, `engine.py`
- `docs/CORE.md` (canonical core/result/ordering/renderer/sink semantics,
  compatibility notes, geometry drift resolution)
- `tests/test_core_config.py`, `test_core_ordering.py`,
  `test_core_renderers.py`, `test_core_sinks.py`, `test_core_engine.py`

Modified:
- `src/batch.py`: `run_frame_pipeline` is now a thin adapter over
  `run_single_image` (exact signature preserved; binds `out_dir` as
  `FilesystemArtifactSink`; adds `core_result` to `FramePipelineResult`;
  `_build_pipeline_context` delegates normalization to
  `CoreConfig.from_mapping`; stage callables resolved from module globals so
  historical monkeypatch targets keep working)
- `src/__init__.py`: re-exports the core public surface
- `modules/classifier/clip.py`, `modules/verifier/blip3.py`: optional
  sink-aware debug artifact emission (`artifact_sink` param; legacy file
  writes unchanged when absent); BLIP3 JPEG/TXT debug pairs factored into one
  helper
- `zap-it-batch.py`: docstring corrected — geometry is NOT executed by this
  path
- `pyproject.toml`: added `src.core` package for wheel builds
- `tests/conftest.py`: PIL stub now injected only when real Pillow cannot be
  imported (same probe-and-prefer-real policy as the pre-existing PyYAML
  probe), enabling genuine 16-bit PNG encode/decode in supported environments
- `tests/test_run_frame_pipeline.py`: prep-debug assertion updated from the
  removed internal hook to the equivalent observable artifact file;
  additional typed-result assertions
- `tests/test_src_exports.py`: extended for the new public exports
- `oap/active`, `oap/orders/001-a-in-memory-core-and-renderers.md`: activated
  order transcript (implementation history)

## Acceptance evidence

1. Single-image core entry point, no caller-controlled filesystem output:
   `src.core.run_single_image(image_rgb, CoreConfig, ...)`; PASSED (engine +
   fake-stage tests; adapter requires only legacy `out_dir` binding).
2. Memory-path writes no request-derived data: snapshot-diff test around
   `run_single_image` with all debug flags on + `MemoryArtifactSink` asserts
   empty cwd diff while artifacts land in RAM; PASSED.
3. Legacy CLI/config/image/video behavior green: entire pre-existing suite
   passes unmodified except the three enumerated test adaptations above;
   PASSED.
4. One ordering definition shared by records/YOLO/PNG: renderers consume the
   ordered `ObjectResult` tuple; bijectivity asserted in tests; PASSED.
5. YOLO bytes deterministic, five-field, six-decimal, normalized to ORIGINAL
   dimensions; empty detections → empty text; PASSED.
6. Identity PNG real lossless 16-bit: decoded dtype/pixel/dims/background/id/
   disconnected-component assertions against real Pillow decode plus encoded-
   byte determinism test; PASSED.
7. Overlap truth retained internally per object; single-valued projection
   documented as lossy by contract; PASSED.
8. ROI/resize mapping boundary tests incl. clipping/round-trip/downscale
   regression; PASSED.
9. Geometry drift explicitly resolved (docstring + docs/CORE.md; no fabricated
   fields; `ObjectResult.geometry()` hook returns None); PASSED.
10. No GPU/model/network needed for verification: suite-level offline+CUDA-free
    guards active; PASSED.
11. Ruff/package/CPU CI and CodeQL green on PR head (see CI section);
    PASSED.
12. Documentation updated in the same PR (`docs/CORE.md`); PASSED.
13. Objective branch/PR exist; coding never merges; report-only SELF is being
    published as remote PR head before signaling; PASSED (verified below).

## Verification
- `git fetch && git rev-parse origin/main` →
  `ab1954484c6229168f9d12eb9964837d802aba88`; `gh run list --branch main`:
  CI SUCCESS, CodeQL SUCCESS post-merge — PASSED (round start)
- `.venv/bin/pip install -e '.[dev]'` — PASSED (environment resynced to
  canonical command before verification)
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing` —
  PASSED: 137 passed / 0 failed / 0 skipped, branch coverage 71.35%
  (gate `fail_under = 64` held; measured value raised from predecessor's 67%)
- Dedicated core tests — PASSED: `tests/test_core_config.py` (5),
  `test_core_ordering.py` (6), `test_core_renderers.py` (12),
  `test_core_sinks.py` (5), `test_core_engine.py` (17) incl.
  `test_inverse_remap_regression_downscale_full_coverage`,
  `test_memory_sink_captures_debug_artifacts_without_filesystem_writes`,
  `test_repeated_runs_are_byte_identical_for_yolo_png_and_metadata`,
  identity overflow guard without huge allocation
- Deterministic repeated fake-engine test — PASSED (byte-identical YOLO text,
  PNG bytes, serialized records, config digest across two identical
  invocations)
- Memory-path no-filesystem-write assertion — PASSED (cwd snapshot diff empty)
- Legacy CLI/config/video regression tests — PASSED (`test_batch_*`,
  `test_zap_it_batch_cli.py`, `test_input_video.py`, `test_output_*.py` within
  full suite)
- Docs/examples validation — PASSED (docs/CORE.md code block ruff-formatted;
  no executable examples claimed beyond tested API usage)
- Secret/large-artifact scan — PASSED: pattern scan over full branch diff
  (`api[_-]?key|secret|password|token|BEGIN (RSA|OPENSSH)`) found nothing;
  diff adds source/docs/tests only (largest additions are text files; no
  binaries, weights or caches committed)
- `.venv/bin/ruff format --check . && .venv/bin/ruff check .` — PASSED
- `.venv/bin/python -m build --wheel` — PASSED; wheel inspected:
  `src/core/*` included

## CI/checks
All on PR head c4f3cdc66e0d258df76aa82d3d5cc54eb8f93bd5 (PR #45), none
pending/failed/missing at time of report:
- `static (format, lint, build)` — SUCCESS
- `tests (py3.10)` — SUCCESS
- `tests (py3.11)` — SUCCESS
- `tests (py3.12)` — SUCCESS
- `Analyze (python)` — SUCCESS
- `CodeQL` — SUCCESS

## GPU/service/resource evidence
- Before-snapshot (round start):
  `index 0, GPU-4c129e25-8e59-eee4-b49c-56c40e294182, RTX 2080 Ti, 2161/11264 MiB`
  with unrelated compute process pid 66522 `/opt/venv/bin/python` (2152 MiB);
  `index 1, GPU-c457dbaf-991c-dc23-c781-0dc030776dd8, RTX 2080 Ti, 6/11264 MiB`.
- After-snapshot (before report): identical values; GPU1 still 6 MiB used; no
  new compute processes on either GPU. Zero allocation by this objective;
  physical GPU0 untouched throughout.
- This objective performed no live GPU work, service launch, port binding,
  listeners, `/dev/shm` usage or persistent request data (CPU-only by design).

## Documentation/provenance
- `docs/CORE.md` documents the entry point, normalized config boundary,
  ordering key, overlap-winner projection, unmapped-label fallback, dtype
  guard, artifact sinks, mapping-fix rationale/evidence and compatibility
  notes; explicitly disclaims API/live-GPU readiness.
- Model/dependency provenance from objective 000 preserved (no dependency
  changes; model revisions remain unpinned pending objective 003 as recorded
  in the predecessor report).
- MIT license and THIRD_PARTY_NOTICES untouched.

## Deferred human adjudication
- Critical register action: NONE

## Safety/scope confirmations
- Scope limited to order 001-a; no FastAPI/HTTP/listener/auth/deployment, no
  training/threshold changes beyond the evidenced mapping fix, no removal of
  supported CLI/video behavior, no GPU/driver/firewall/system mutation, no
  protected-process interaction.
- Sinks accept validated logical names only; traversal/absolute paths rejected
  and tested. Debug flags without a sink raise typed `CoreError` rather than
  silently writing anywhere.
- No secrets, credentials, raw request content, model weights or customer data
  introduced, logged or committed.

## Limitations/blockers
- Visualization composite overlays remain non-deterministic (`np.random`
  colors in the legacy visualizer), deliberately out of scope: the required
  deterministic renderers are YOLO/PNG/metadata; overlay determinism belongs
  to later hardening (005).
- The hostile-upload YAML policy validator is intentionally NOT implemented
  here (objective 002); `CoreConfig` classification is a trusted-CLI boundary.
- Identity-PNG byte determinism is asserted within one environment; PNG
  encoders may vary across Pillow versions (documented contract wording).
- Serialized JSON key order changes slightly versus legacy output (internal
  never-serialized array keys excluded more strictly); values unchanged.
- Final-object emission order is now deterministic rather than arrival order
  (the purpose of this objective); documented in docs/CORE.md.

## Factual strategic follow-up
- Coverage measured 71.35% after adding ~1,300 lines of core code; gate could
  be raised above 64 in a later objective if desired.
- `YoloDatasetExporter` remains dataset-coupled (random split, directories) by
  design; the pure renderer now exists alongside it for service use.
- Real-model behavior (SAM2/CLIP/BLIP3 with actual weights) still unexercised
  until objectives 003/004; all current evidence is CPU/fake-tier by order.
