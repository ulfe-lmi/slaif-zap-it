# OAP Work Order — 004-d — Bound identity-matching memory

Objective `004-d`, continuation of numeric Objective 004 on existing PR #48.
Replace the correct but structurally unbounded `004-c` identity representative
matcher with a deterministic complete algorithm whose auxiliary memory is
bounded independently of the total number of true pixels across object masks.
Preserve all accepted correctness, log-safety, canonical shared-memory,
service, GPU and governance behavior. Do not rewrite any prior order/report.

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Mode: `AMEND_EXISTING_PR`; numeric objective / round `004 / 004-d`.
- Sole Objective-004 PR: #48, existing branch
  `oap/004-a-loopback-service-activation`, base remote `main` at
  `1a4272d60c52cc045f57f2842652485efdb7a55c`.
- Current PR/branch head:
  `bfee4d31371306d922f0a80c53093b96225af48c`, immutable report-only
  `004-c` commit.
- `bfee4d3...` first parent is remediation implementation
  `7076c3053ce4ad05bddaf7bbb8847b2ddc708bfe`; its only changed path is
  `oap/reports/004-c-report.md`.
- All six required checks are SUCCESS and PR #48 is open, mergeable and clean.
- Required report: `oap/reports/004-d-report.md`; all earlier orders/reports
  remain immutable.

## Strategic review finding

The `004-c` matching logic is mathematically correct on its tested small cases,
but `_representative_assignment()` materializes:

- a Python `(row, col)` tuple for every true pixel in every object mask;
- a union-pixel `set`, sorted `list`, and pixel-index `dict`;
- a graph node/list for every unique source pixel; and
- a residual edge pair for every object-to-pixel membership.

The public service permits up to 64,000,000 decoded pixels. One broad valid
mask can therefore create tens of millions of Python objects and graph nodes;
multiple overlapping masks multiply the edge count. The 500 random property
cases in `004-c` prove small-case existence correctness, not resource
boundedness. This violates the order's bounded matching requirement and the
service resource law. Green CI and the 128×128 live fixture do not cover it.

## Binding correction

Implement a deterministic complete injective object-to-source-pixel matching
without a per-true-pixel Python graph or collection.

Required properties:

1. Auxiliary Python state is `O(number_of_objects + fixed_scan_chunk)` beyond
   the already-existing masks and output canvas. It must not scale with the
   union area or sum of mask areas through Python tuples, sets, dict entries,
   graph nodes or edges.
2. Candidate pixels are traversed deterministically in row-major order through
   bounded NumPy chunks or an equivalently bounded representation. Any
   temporary candidate array has a fixed documented maximum independent of
   image dimensions.
3. Use a complete augmenting-path assignment (or an equally complete bounded
   algorithm): succeed whenever an injective assignment exists; fail with the
   existing typed `IdentityMaskProjectionError` only when none exists.
4. Prefer the baseline canvas's already visible representative pixels and do
   not alter the baseline raster when every object ID is already visible.
   When repair is needed, preserve baseline pixels except deterministic
   representative overrides required by the bounded augmenting result.
5. Preserve the `ensure_all_ids=False` legacy bytes, retained source masks,
   stable sanitized impossible-projection error, uint16/dimension/object-ID
   contract and deterministic output.
6. Remove obsolete min-cost/full-graph code and any documentation claim that
   depends on globally minimizing override count if the bounded algorithm does
   not prove that stronger property. The required product properties are
   deterministic winner behavior, object/YOLO/PNG bijection and safe bounded
   execution.

## Required focused verification

- Retain and pass all `004-c` adversarial/existence/determinism/impossible/
  legacy tests and the 500-case brute-force agreement check.
- Add an augmenting-chain case that requires moving more than one existing
  representative.
- Add a large broad-mask regression that would have forced the old algorithm
  to create hundreds of thousands or more Python graph elements, but now
  completes under a documented conservative time and auxiliary-memory bound.
  Prefer direct instrumentation of the bounded candidate iterator/chunk size
  and Python allocation peak over a flaky wall-clock-only assertion.
- Add a fast-path test proving no matching scan/materialization occurs when all
  IDs are already visible.
- Add a structural regression that fails if a per-pixel Python graph/set/dict
  representation is reintroduced; do not rely solely on source-text matching.
- Run the complete canonical CPU suite, Ruff format/lint, shell syntax,
  compile, wheel build/import, diff and secret/large-artifact checks.

## Live verification and host law

The changed code is response rendering, so rerun on physical GPU1:

- fresh GPU/process/port/shared-memory snapshot;
- normal loopback start and readiness with exact pinned GPU1 UUID;
- current-head real L0–L3 JSON and ZIP, BLIP3 rejection and repeat smoke,
  proving eight YOLO/object/PNG IDs remain bijective;
- one stop/restart and post-restart L1 or higher request;
- final stop with no listener/process/shared-memory child and GPU1 back at idle.

At strategic review GPU1 was idle at 6 MiB and no ZAP-IT service listened.
GPU0 had only protected PID 66522 with independently varying memory. Recheck
live; never allocate on, stop, reset or otherwise touch GPU0 or that workload.
Use `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`,
one process/worker/inference and IPv4 loopback only.

Previously accepted failure/busy/deadline/cancel/response-limit, log-safety and
canonical-root evidence need not be rerun unless touched or regressed. Report
inherited versus fresh evidence exactly.

## Scope and non-goals

- Expected behavior files: identity renderer/helper plus only necessary error,
  tests and accurate identity documentation.
- Commit exact immutable `004-d` order transcript and `oap/active` with the
  implementation.
- No new PR/branch/title/numeric objective; no earlier artifact rewrite.
- No API limit weakening, smaller max-pixel substitution, object dropping,
  scientific model/threshold/order change or legacy renderer change.
- No unrelated runtime/launcher/service refactor, BLIP3 enablement, LAN/public
  exposure, systemd installation, Docker, GPU0 use or persistent request data.
- Coding never merges.

## Acceptance criteria

1. No auxiliary per-mask-pixel Python graph/tuple/set/dict collection exists.
2. Matching remains complete and deterministic on exhaustive/random/adversarial
   small cases, including augmenting chains and impossible input.
3. Large broad masks exercise a fixed candidate-memory bound and do not create
   size-proportional Python object graphs.
4. Already-bijective masks take a fast path and preserve baseline bytes.
5. Real GPU1 L0–L3 JSON/ZIP remains bijective and final host state is clean.
6. Full CPU/static/package and all six GitHub checks are SUCCESS on both
   implementation and report heads.
7. PR #48 remains the sole Objective-004 PR; final `004-d` SELF commit changes
   only its report and has the literal implementation SHA as first parent.

## GitHub/OAP publication

- Amend PR #48 only.
- Push all non-report work first and capture its literal SHA.
- Create one immutable `oap/reports/004-d-report.md` with
  `Report publication commit: SELF`.
- Final report commit changes only that path; verify parent, remote bytes, PR
  head and every report-head check before FIFO response.

## Deferred human adjudication

- Decision: `NONE`

This is an ordinary resource-correctness bug with a bounded reversible fix and
does not meet the critical-register threshold.

## Coding response

Send exact FIFO `OK` only after corrected PR/report/CI evidence and final stopped
host verification. Coding never merges.
