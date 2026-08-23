# OAP Work Order — 001-a — In-memory core and deterministic renderers

Objective `001-a`. Extract a typed, reusable, single-image in-memory ZAP-IT core
from the existing batch orchestration while preserving supported CLI/config
behavior. Establish the stable internal result model, artifact-sink boundary,
deterministic final-object ordering, pure YOLO rendering, and uint16
identity-mask rendering required by the future service. This objective creates
the computational/service seam; it does not add FastAPI, activate a service,
download models, or use a GPU.

## Prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `001 / 001-a`
- Mode: `CREATE_NEW_PR`
- Objective 000 merged on remote `main`: squash merge commit
  `ab1954484c6229168f9d12eb9964837d802aba88` (PR #44); post-merge runs on main:
  `CI` SUCCESS, `CodeQL` SUCCESS — prerequisite satisfied.
- Verified current default branch and 40-hex base SHA: `main` @
  `ab1954484c6229168f9d12eb9964837d802aba88` (verified live via `git fetch`
  immediately before publication).
- Required new branch name: `oap/001-a-in-memory-core-and-renderers`, created
  from remote `main` at the base SHA above. Note: the shared local clone was
  left checked out on the finished `oap/000-a-professional-baseline-and-ci`
  branch; create the new branch explicitly from `origin/main` without resetting,
  cleaning or discarding any existing local state.
- Existing objective-001 PR: none (`gh pr list --state open` shows no PRs after
  #44 merged).
- Required PR title: `Objective 001-a: typed in-memory single-image core,
  deterministic renderers, artifact sinks`

## Verified current state (post-000 tree, evidence gathered live 2026-08-23)

- Package/module layout and canonical CPU commands: setuptools package `zap-it`
  0.1.0 (`pyproject.toml`, requires-python >=3.10,<3.13); pipeline code in
  `src/batch.py` (1279 lines: `PipelineContext` dataclass at :41,
  `FramePipelineResult` at :60, `run_frame_pipeline` at :248, `process_folder`),
  `src/config.py` (`load_config` using real `yaml.safe_load`; normalization
  limited to hoisting `visualization.alpha` to top level and `roi: false -> null`),
  `modules/{input,segmenter,classifier,verifier,geometry,output,visualizer.py}`.
  Canonical CPU commands: `python3 -m venv .venv && .venv/bin/pip install -e
  '.[dev]'`, then `.venv/bin/pytest -q --cov=src --cov=modules`,
  `.venv/bin/ruff format --check . && .venv/bin/ruff check .`,
  `.venv/bin/python -m build --wheel`. Suite currently 95 passed / 0 failed /
  0 skipped under CPython 3.12 with numpy+PIL+PyYAML+pytest only; branch
  coverage gate `fail_under = 64` enforced in pyproject.
- Exact `run_frame_pipeline` signature/result/state behavior: accepts
  `(frame_id: str, orig_np: np.ndarray, *, context: PipelineContext,
  segmenter_state, clip_state, blip3_state, out_dir: str, dryrun: bool,
  verbosity: int, device=None, yolo_exporter=None)` and returns
  `(FramePipelineResult(rendered: Mapping[str, np.ndarray], final_masks:
  List[dict], serialized: List[dict]), segmenter_state, clip_state,
  blip3_state)`. It already accepts a decoded RGB array (bootstrap hypothesis
  confirmed) but still requires `out_dir` and writes debug artifacts through it.
  Mask upscaling back to original coordinates uses a per-pixel loop with
  `int(rpos * scaleY)` nearest-neighbor mapping inside ROI/scale factors.
- All remaining direct filesystem writes reachable from single-image execution:
  (1) ROI debug JPEG via `save_roi_debug` when `preprocessing.debug` and ROI set;
  (2) raw SAM2 patch JPEGs when `mask_generator.debug`;
  (3) filtered-patch JPEGs when `postsam2processing.debug`;
  (4) `out_dir`+`fname_stem` forwarded into `run_clip`/`run_blip3`, whose
  filters write debug JPEGs (and `.txt` answer files for BLIP3;
  `modules/verifier/blip3.py:328-368`) when their configs enable `debug`;
  (5) optional `yolo_exporter.process_image(...)` which persists dataset images/
  labels. `generate_visualizations` returns arrays without writing files; the
  batch writers persist them outside the core.
- Current config parsing/normalization and batch-only fields: `load_config`
  performs safe_load plus minimal normalization; configs carry deployment/path
  fields (output roots, `export_yolo_det` dataset settings, debug flags,
  `images`/`video` writer sections) that are algorithm-independent and must be
  classified as such at the boundary.
- Current YOLO exporter semantics: `YoloDatasetExporter.process_image` derives
  each bbox from mask pixel extents, normalizes cx/cy/w/h against the sample
  frame (ROI crop when `sample_roi` else full image), emits exactly five fields
  `f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"`, filters by
  `export_yolo_det.labels`, maps class id by position in that comma-separated
  label list, skips masks whose cropped segmentation is empty — but splits
  train/val with `random.random()` and creates directories/dataset YAMLs at
  init: correct line format, non-reusable dataset coupling.
- Current visualization/debug output behavior: `render_annotated` assigns
  overlay colors from `np.random.randint` (non-deterministic overlays);
  composites are computed in memory and returned.
- Current geometry implementation vs canonical path: `modules/geometry/
  geometry.py` (199 lines, 2 passing unit tests) is NOT invoked anywhere in
  `src/batch.py` or the frame pipeline; only `zap-it-batch.py`'s docstring
  claims optional geometry execution. Documentation/code drift confirmed.
- Existing CPU tests relevant to this objective:
  `tests/test_run_frame_pipeline.py` (1 test), `test_output_yolo.py` (2),
  `test_geometry.py` (2), `test_input_images*.py` (~8 total), all running under
  the stub harness inventoried in `docs/BASELINE.md`.
- Predecessor report/known compatibility promises: report `oap/reports/
  000-a-report.md` promises preserved CLI/config behavior, honest stub-vs-real
  inventory, CPU-only CI, no API/GPU readiness claims, MIT license preserved,
  model revisions unpinned until Objective 003, BLIP-3 license re-verification
  recorded in THIRD_PARTY_NOTICES.md before any production decision.

## Required design

Converge on the responsibilities in ARCHITECTURE-for-agents without forcing
exact class or module names where the post-000 tree suggests a cleaner
implementation:

```text
decoded RGB image + validated/normalized pipeline configuration
    -> in-memory single-image engine
    -> typed PipelineResult / ObjectResult / stage status
    -> pure deterministic renderers
         YOLO text
         uint16 identity PNG
         structured per-object metadata
    -> optional logical artifacts through an artifact sink
```

The CLI/batch layer remains an adapter around the same core and may use a
filesystem sink. The future API must be able to use a memory/shared-memory sink
without caller-controlled paths.

### Provisional strategic decisions (binding unless evidence forces documented deviation)

- Final-object ordering key (single definition driving IDs, YOLO order and PNG
  values): sort by descending mask area, tie-broken by ascending centroid
  (row, then column), then by ascending original candidate index. Assign IDs
  1..N after final filtering. Document and test, including exact-tie cases.
- Identity-PNG overlap winner policy: the larger-area object wins contested
  pixels (consistent with the existing visualizer z-order); every object keeps
  its complete source mask in the result records so overlap truth is never lost.
  Document and test.
- YOLO numeric semantics: reuse exporter math (pixel-extent bboxes, six-decimal
  fixed precision, normalization to ORIGINAL image dimensions in the core
  renderer) but drop split randomness and filesystem coupling entirely.
- Unmapped-label policy: every final object receives a YOLO line (bijective
  agreement with object list/PNG ids is architectural law). If an object's
  label is absent from the effective class mapping, assign class id 0 and
  record a per-object warning in structured metadata rather than dropping the
  object silently.
- Identity dtype guard: reject/guard final object counts > 65535 with a clear
  typed error; test the guard directly without allocating huge structures.

## Scope

1. **Separate core computation from batch/file orchestration.** Refactor
   existing reusable functions rather than duplicating SAM2/CLIP/BLIP
   algorithms. A single-image core call accepts already-decoded image data and
   normalized configuration/state and returns structured results without
   requiring a batch input directory or caller-selected output directory.
2. **Define typed result contracts.** Maintainable typed structures for
   image/effective dimensions, ROI/scale mapping, final objects, retained SAM
   quality (`predicted_iou`, `stability_score`), CLIP/BLIP evidence when
   present, stage statuses/warnings/timings as appropriate, rendered/stage
   artifacts and provenance hooks. NumPy arrays may remain internal fields;
   public metadata serialization must be explicit and deterministic.
3. **Establish a configuration boundary suitable for later API validation.**
   Keep legacy trusted CLI configuration compatible; expose a normalized
   configuration representation and clear classification of algorithm fields
   versus batch/path/deployment/debug fields. Do not claim the complete
   hostile-upload policy yet.
4. **Introduce logical artifact sinks.** At minimum a memory sink for
   bytes/arrays/records plus a preserved filesystem sink/adapter for legacy CLI
   behavior. Add shared-memory abstraction only if useful at this layer; the API
   objective finishes lifecycle integration. Sink APIs accept logical artifact
   names, never arbitrary request paths.
5. **Remove service-blocking filesystem coupling from the core.** Convert ROI
   debug, SAM patch debug, postsam2 debug, CLIP/BLIP3 debug and visualization-
   adjacent writes reachable from the single-image core into returned artifacts
   or sink operations gated behind their existing config flags. Preserve
   equivalent CLI output when configured. Do not delete a supported artifact
   because it is inconvenient for the service.
6. **Define one final-object identity and ordering model** exactly as decided
   above; object records, YOLO lines and identity-mask values share it.
7. **Create a pure YOLO renderer** per the decisions above; empty detections
   produce empty text.
8. **Create the uint16 identity-mask renderer**: lossless grayscale 16-bit PNG,
   exactly original dimensions, background 0, IDs 1..N, disconnected components
   sharing one ID, deterministic winner policy, source masks retained, count
   guarded.
9. **Preserve ROI/resize mapping semantics.** Bboxes, masks, YOLO coordinates
   and identity PNG expressed in original-image coordinates. Focused tests for
   edges, clipping, resize/ROI round-trips. The current per-pixel
   `int(rpos*scale)` remap is a suspected precision-loss point: audit it; fix
   with regression evidence if a demonstrated bug exists, otherwise freeze
   current behavior deliberately with a characterization test.
10. **Resolve geometry drift explicitly.** Correct the misleading CLI docstring
    and record in docs that the canonical frame path does not execute geometry;
    add typed result hooks/pure adapter seams only. Do not fabricate geometry
    output or enable the stage.
11. **Keep model lifecycle state reusable but request state isolated.** Existing
    segmenter/CLIP/BLIP state objects pass through the core; image masks,
    answers, artifacts and mutable config state must not leak across calls.
12. **Adapt legacy CLI/batch/video callers** through compatibility adapters;
    image AND video CLI behavior stays green.
13. **Add exhaustive CPU/fake tests**: no objects, one, multiple disjoint,
    disconnected components, overlap winners, equal-score/tie ordering,
    near-limit ID guard without huge allocations, ROI/resize mapping edges,
    class mapping incl. unmapped-label warning, byte-stable repeated outputs,
    no-filesystem-writes on the memory path, legacy caller compatibility. No
    CUDA/model/network dependency.

## Non-goals

- no FastAPI/HTTP route or `/v1/completions` transport;
- no live listener, systemd unit, Docker deployment or API key;
- no model download, CUDA allocation or real SAM2/CLIP/BLIP inference;
- no GPU/driver/environment/service/firewall/network mutation;
- no public wire-schema freeze beyond deterministic renderer semantics needed
  by Objective 002;
- no training/fine-tuning or scientific threshold/default changes without a
  demonstrated bug and explicit regression evidence;
- no removal of supported CLI/video behavior;
- no implementation of Objective 002+ merely because the new core makes it easy.

## Acceptance criteria

1. One documented Python-level single-image core entry point accepts a decoded
   RGB image plus normalized configuration/state and returns a typed result
   without requiring caller-controlled filesystem output.
2. Memory-path execution writes no request-derived data to repository/cwd/legacy
   output directories; any compatibility path is explicit and tested.
3. Legacy supported CLI/config/image/video behavior remains green under the
   canonical CPU/mocked suite; compatibility changes are explicit/documented.
4. Final object IDs and ordering are defined once and shared by object list,
   YOLO renderer and identity-mask renderer.
5. YOLO bytes are deterministic, five-field, fixed precision, normalized to
   original-image dimensions; empty detections produce empty text.
6. Identity PNG is a real 16-bit lossless PNG with exact original dimensions,
   background 0, IDs 1..N, disconnected-component preservation and deterministic
   overlap handling; tests inspect decoded dtype/pixel values and encoded-byte
   determinism where the encoder permits.
7. Overlap truth remains available internally/per-object and is not
   misrepresented as lossless by the single-valued projection.
8. ROI/resize mapping and bbox/area calculations have focused boundary tests.
9. Geometry/documentation drift is explicitly resolved or recorded as a bounded
   later-stage limitation; no fabricated fields.
10. No GPU/model/network access is needed for complete objective verification.
11. Ruff/package/CPU CI and CodeQL remain green on the current PR head.
12. Documentation describing core/result/ordering/artifact semantics is updated
    in the same PR.
13. Correct objective branch/PR exists, coding never merges, and the immutable
    report-only SELF child is the remote PR head before response signal.

## Required verification (exact commands/states)

- Predecessor remote-main/CI state: `main` @ `ab195448…` with CI+CodeQL SUCCESS
  (re-confirm at round start; report observed values)
- Canonical package/Ruff/static: `.venv/bin/ruff format --check . &&
  .venv/bin/ruff check .` and `.venv/bin/python -m build --wheel` — PASSED
- Full CPU suite and coverage: `.venv/bin/pytest -q --cov=src --cov=modules
  --cov-report=term-missing` — all green, counts/duration reported, 64% gate
  held or honestly raised with measured value
- Dedicated core/result/YOLO/identity-mask tests — named, PASSED
- Deterministic repeated fake-engine test: two identical invocations produce
  byte-identical YOLO text, PNG bytes and serialized metadata — PASSED
- Memory-path no-filesystem-write assertion (cwd/repo/output snapshot diff) —
  PASSED
- Legacy CLI/config/video regression tests — PASSED
- Docs/examples validation — PASSED (validated in tests where applicable)
- Secret/large-artifact scan — method and result stated
- GitHub required checks: `static (format, lint, build)`, `tests (py3.10)`,
  `tests (py3.11)`, `tests (py3.12)`, `Analyze (python)`, `CodeQL` — all
  present and SUCCESS, none pending/failed/missing on PR head
- Read-only GPU before/after snapshot proving zero allocation by this objective
  (`nvidia-smi --query-gpu=index,uuid,memory.used --format=csv` + compute-apps)

## Documentation and provenance

Update architecture-facing package docs to identify the canonical core entry
point, typed result semantics, object ordering, overlap projection, artifact
sinks and compatibility guarantees. Do not claim API or live GPU readiness.
Preserve model/dependency provenance established in Objective 000.

## Security/resource constraints

Treat the host as shared. This objective is CPU-only and must not allocate
either GPU. No model downloads, system package/CUDA/driver changes, listeners,
services, firewall/VPN changes or unrelated process mutation. Never let
normalized config or artifact sinks turn a YAML path/string into host
authority. Preserve unrelated working-tree state and never print credentials or
provider configuration.

## Deferred human adjudication

- Decision: `NONE`

Ordering-key details, overlap winner policy, unmapped-label fallback, sink
shape and compatibility adapters are ordinary reversible engineering decisions
resolved by this order, the constitutions and the architecture law; none can
materially affect a security boundary, privacy, trust model, deployment safety
or release acceptability at this stage. Do not create a CRITICAL entry merely
because these choices require judgment. If implementation exposes a genuinely
material dilemma meeting all five register conditions, report it as a candidate
in the report and continue all unambiguous safe scope; strategic decides next
round. Coding may not invent the entry.

## GitHub publication and report

Create exactly one new objective branch `oap/001-a-in-memory-core-and-renderers`
from remote `main` @ `ab1954484c6229168f9d12eb9964837d802aba88` and exactly one
PR titled as specified. Carry the exact activated order and `oap/active`
transcript in implementation history. Push all non-report work, inspect/fix
in-scope current-head CI, never merge. After all implementation evidence is
remote, capture the literal implementation SHA and publish exactly one final
report-only commit with literal SHA and `Report publication commit: SELF`;
verify parent/path/bytes/current head. Send response `OK` only after that
verification. Report exact compatibility behavior, tests/coverage,
deterministic-output semantics, files, skips/failures/limitations and safety
evidence including both GPU snapshots.
