# OAP Work Order — 001-a — In-memory core and deterministic renderers

> DRAFT UNTIL Objective 000 is merged and strategic live/GitHub reconciliation is complete. Do not publish as-is.
>
> **HWP status:** this draft is preloaded human engineering intent. Strategic may refine exact implementation details, split an unreviewable scope, or change a criterion only when verified evidence requires it. It must not casually replace the intended outcome with a new plan.

## Objective

Extract a typed, reusable, single-image in-memory ZAP-IT core from the existing
batch orchestration while preserving supported CLI/config behavior. Establish the
stable internal result model, artifact-sink boundary, deterministic final-object
ordering, pure YOLO rendering, and uint16 identity-mask rendering required by the
future service. This objective creates the computational/service seam; it does
not add FastAPI, activate a service, download models, or use a GPU.

## Prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `001 / 001-a`
- Mode: `CREATE_NEW_PR`
- Objective 000 merged on remote `main`, merge SHA and checks: VERIFY:
- Verified current default branch and 40-hex base SHA: VERIFY:
- Required new branch name: VERIFY:
- Existing objective-001 PR: N/A after strategic confirms none: VERIFY:
- Required PR title: VERIFY:

Do not activate while Objective 000 remains open or its baseline CI is not green.

## Verified current state

Strategic must replace these placeholders using the post-000 repository, not the
bootstrap snapshot:

- current package/module layout and canonical CPU commands: VERIFY:
- exact `run_frame_pipeline` signature/result/state behavior: VERIFY:
- all remaining direct filesystem writes reachable from single-image execution: VERIFY:
- current config parsing/normalization and batch-only path/debug fields: VERIFY:
- current YOLO exporter semantics, label mapping, ROI behavior and tests: VERIFY:
- current visualization/debug output behavior: VERIFY:
- current geometry implementation and whether the canonical frame pipeline actually executes it: VERIFY:
- existing CPU tests relevant to ROI/resize/masks/CLIP/BLIP/visualization/YOLO: VERIFY:
- predecessor report/known compatibility promises: VERIFY:

Bootstrap audit found a useful reusable seam in `src/batch.py`: `run_frame_pipeline`
accepts an image array and returns rendered outputs, final masks and serialized
non-array metadata, but it still receives `out_dir` and several enabled debug/model
paths write by filename. Treat that as a hypothesis to re-check after Objective
000 rather than as immutable truth.

## Required design

Converge on the responsibilities in `ARCHITECTURE.md`, without forcing exact class
or module names where the post-000 tree suggests a cleaner implementation:

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

## Scope

1. **Separate core computation from batch/file orchestration.** Refactor existing
   reusable functions rather than duplicating SAM2/CLIP/BLIP algorithms. A
   single-image core call must accept already-decoded image data and normalized
   configuration/state and return structured results without requiring a batch
   input directory or caller-selected output directory.
2. **Define typed result contracts.** Introduce maintainable typed structures for
   image/effective dimensions, ROI/scale mapping, final objects, retained SAM
   quality, CLIP/BLIP evidence when present, stage statuses/warnings/timings as
   appropriate, rendered/stage artifacts and provenance hooks. NumPy arrays may
   remain internal fields but public metadata serialization must be explicit and
   deterministic.
3. **Establish a configuration boundary suitable for later API validation.** Keep
   legacy trusted CLI configuration compatible, but expose a normalized
   configuration representation and clear classification of algorithm fields
   versus batch/path/deployment/debug fields. Do not yet claim the complete
   hostile-upload policy unless implemented and tested.
4. **Introduce logical artifact sinks where required.** At minimum support a
   memory sink for bytes/arrays/records and preserve a filesystem sink/adapter for
   legacy CLI behavior. Add a shared-memory sink abstraction only if it is useful
   at this layer; the API objective may finish lifecycle integration. Sink APIs
   accept logical artifact names, never arbitrary request paths.
5. **Remove service-blocking filesystem coupling from the core.** Convert ROI
   debug, SAM patch debug, postprocessing debug, visualization and similar writes
   reachable from the single-image core to returned artifacts or sink operations.
   Preserve equivalent CLI output when configured. Do not silently delete an
   existing supported artifact just because it is inconvenient for the service.
6. **Define one final-object identity and ordering model.** Assign request-local
   instance IDs 1..N after final filtering. The same ordering must drive object
   records, YOLO lines and identity-mask values. The key must be explicit,
   deterministic for equal fake-engine inputs, based on stable result fields plus
   an unambiguous tie-breaker, and documented/tested.
7. **Create a pure YOLO renderer.** Render exactly five fields per final object:
   `<class_id> <cx> <cy> <width> <height>`, normalized to original input image
   dimensions with fixed precision. Reuse semantics from the dataset exporter
   where correct, but do not make service rendering depend on train/val split,
   image writes, random state or dataset directory creation.
8. **Create the uint16 identity-mask renderer.** Produce a lossless grayscale
   16-bit PNG at exactly original image dimensions: 0 background, 1..N final
   instance IDs. One object may contain multiple disconnected components with the
   same value. Define and test a deterministic overlap winner policy; retain the
   source per-object masks so a future full response can preserve overlap truth.
   Reject/guard object counts that cannot fit the public uint16 identity space.
9. **Preserve ROI/resize mapping semantics.** Bboxes, masks, YOLO coordinates and
   identity PNG must be expressed in original-image coordinates. Add focused tests
   around edges, clipping and resize/ROI round-trips. If current pixel remapping
   contains a demonstrated bug, fix it with regression evidence rather than
   silently freezing the bug into the service contract.
10. **Audit geometry honestly.** If current documentation claims geometry that
    the canonical frame path does not execute, resolve the documentation/code
    drift explicitly. This objective may add typed result hooks and pure geometry
    adapters, but must not invent geometry output or trigger a previously disabled
    scientific stage merely to satisfy the future API.
11. **Keep model lifecycle state reusable but request state isolated.** Existing
    segmenter/CLIP/BLIP state objects may pass through the core; image masks,
    answers, artifacts and mutable config state must not leak across calls.
12. **Adapt legacy CLI/batch/video callers.** Existing supported image and video
    CLI behavior must continue through compatibility adapters. Do not remove video
    CLI support just because the first API is single-image only.
13. **Add exhaustive CPU/fake tests.** Cover no objects, one object, multiple
    disjoint objects, disconnected components, overlap, equal-score/tie cases,
    near-limit IDs without huge memory allocation, ROI/resize mapping, class
    mapping, stable byte output, absence of filesystem writes for the memory path,
    and compatibility of legacy callers. No CUDA/model/network dependency.

## Non-goals

- no FastAPI/HTTP route or `/v1/completions` transport;
- no live listener, systemd unit, Docker deployment or API key;
- no model download, CUDA allocation or real SAM2/CLIP/BLIP inference;
- no GPU/driver/environment/service/firewall/network mutation;
- no public wire-schema freeze beyond deterministic renderer semantics needed by
  Objective 002;
- no training/fine-tuning or scientific threshold/default changes without a
  demonstrated bug and explicit regression evidence;
- no removal of supported CLI/video behavior;
- no implementation of Objective 002+ merely because the new core makes it easy.

## Acceptance criteria

1. One documented Python-level single-image core entry point accepts a decoded RGB
   image plus normalized configuration/state and returns a typed result without
   requiring caller-controlled filesystem output.
2. Memory-path execution writes no request-derived data to repository/cwd/legacy
   output directories; any compatibility path is explicit and tested.
3. Legacy supported CLI/config/image/video behavior remains green under the
   canonical CPU/mocked suite; compatibility changes are explicit and documented.
4. Final object IDs and ordering are defined once and shared by object list, YOLO
   renderer and identity-mask renderer.
5. YOLO bytes are deterministic, five-field, fixed precision and normalized to
   original-image dimensions; empty detections produce empty text.
6. Identity PNG is a real 16-bit lossless PNG with exact original dimensions,
   background 0, IDs 1..N, disconnected-component preservation and deterministic
   overlap handling. Tests inspect decoded dtype/pixel values and deterministic
   encoded bytes where the chosen encoder permits stable bytes.
7. Overlap truth remains available internally/per-object and is not misrepresented
   as lossless by the single-valued identity projection.
8. ROI/resize mapping and bbox/area calculations have focused boundary tests.
9. Geometry/documentation drift is explicitly resolved or recorded as a bounded
   later-stage limitation; no fabricated fields.
10. No GPU/model/network access is needed for the complete objective verification.
11. Ruff/package/CPU CI and CodeQL remain green on the current PR head.
12. Documentation describing core/result/ordering/artifact semantics is updated in
    the same PR.
13. Correct objective branch/PR exists, coding never merges, and the immutable
    report-only SELF child is the remote PR head before response signal.

## Required verification

Strategic must replace with exact post-implementation commands/results:

- predecessor remote-main/CI state: VERIFY:
- canonical package/Ruff/static checks: VERIFY:
- full CPU suite and coverage: VERIFY:
- dedicated core/result/YOLO/identity-mask tests: VERIFY:
- deterministic repeated fake-engine test: VERIFY:
- memory-path no-filesystem-write assertion: VERIFY:
- legacy CLI/config/video regression tests: VERIFY:
- docs/examples validation: VERIFY:
- secret/large-artifact scan: VERIFY:
- GitHub required checks/CodeQL: VERIFY:
- read-only GPU before/after snapshot proving zero allocation by this objective: VERIFY:

## Documentation and provenance

Update architecture-facing package docs as needed to identify the canonical core
entry point, typed result semantics, object ordering, overlap projection,
artifact sinks and compatibility guarantees. Do not claim API or live GPU
readiness. Preserve model/dependency provenance established in Objective 000.

## Security/resource constraints

Treat the host as shared. This objective is CPU-only and must not allocate either
GPU. No model downloads, system package/CUDA/driver changes, listeners, services,
firewall/VPN changes or unrelated process mutation. Never let normalized config or
artifact sinks turn a YAML path/string into host authority. Preserve unrelated
working-tree state and never print credentials/provider configuration.

## Deferred human adjudication

- Decision: `NONE`
- Ordinary choices such as dataclass/module naming, stable ordering key details,
  sink shape and compatibility adapters are strategic engineering decisions, not
  reasons to create `CRITICAL.md` entries.
- If verified evidence exposes a genuinely material dilemma satisfying all five
  register conditions, strategic must make the provisional decision and replace
  this section with exact `APPEND CRIT-NNNN` bytes before activation.

## GitHub publication and report

Create exactly one new objective branch/PR from verified remote `main`. Carry the
exact activated order and `oap/active` transcript in implementation history.
Push all non-report work, inspect/fix in-scope current-head CI, and never merge.
After all implementation evidence is remote, capture the literal implementation
SHA and publish exactly one final report-only commit with `Report publication
commit: SELF`; verify its parent/path/bytes and that it is the current PR head.
Report exact compatibility behavior, tests/coverage, deterministic-output
semantics, files, skips/failures/limitations and safety evidence.