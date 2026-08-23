# Initial strategic roadmap

This is sequencing guidance backed by **human-work-preloaded draft work orders**.
It is not activated scope by itself.

## HWP authority of the preloaded drafts

The human-level project decomposition has already been done before the autonomous
OAP loop. For every planned numeric objective, the corresponding `NNN-a` file in
`initial-orders/` is the default source of intended outcome, scope, non-goals,
acceptance criteria, verification burden and safety constraints:

```text
initial-orders/
  000-a-professional-baseline-and-ci.md
  001-a-in-memory-core-and-renderers.md
  002-a-v1-completions-api-contract.md
  003-a-gpu1-runtime-qualification.md
  004-a-loopback-service-activation.md
  005-a-full-output-parity-hardening-and-evidence.md
  006-a-release-and-integration.md
```

These files are deliberately **DRAFTS**, not activated orders. Strategic must not
publish them verbatim. Immediately before each objective it must reconcile the
draft against remote `main`, predecessor reports/PRs/checks, current architecture,
current `CRITICAL.md`, and relevant live host evidence; replace every `VERIFY:`/
draft marker; resolve implementation choices; and only then publish a final order.

The drafts embody preloaded human intent, so strategic must not casually discard
or reinvent the roadmap merely because another decomposition is convenient. It
may refine exact implementation details, split an objective that has become
unreviewably large, or reorder work when verified dependency/evidence requires
it. Any material deviation must preserve the intended product outcome and hard
gates and must be explained in the order/report. A later `NNN-b..z` is remediation
of the same PR, not permission to substitute a different project.

Strategic remains responsible for judgment: preloading is not blind execution.
Where live facts invalidate an assumption, reality wins and the order is adapted.
Where a rare material unresolved dilemma meets all five `CRITICAL.md` conditions,
strategic decides provisionally and records it rather than stopping merely from
reluctance to decide.

## Objective sequence

| Objective | Outcome | Hard gate |
|---|---|---|
| 000 | Professional package/test/CI/CodeQL/docs/security/provenance baseline | Existing behavior characterized; CPU CI green |
| 001 | Pure typed in-memory image pipeline, artifact sinks, deterministic YOLO/uint16 identity mask | Legacy CLI regression preserved; exhaustive CPU semantics |
| 002 | Multipart `/v1/completions`, JSON/ZIP levels, limits/errors/auth/health using fake engine | No live GPU deployment; API contract tests green |
| 003 | Actual host/physical-GPU1 environment and model qualification | UUID pin; GPU0 untouched; measured 11-GB-class memory fit and licenses documented |
| 004 | Local loopback service on freshly verified unused port | One worker/request; real E2E levels; cleanup; restart/rollback |
| 005 | Full-output parity, overlap masks, visualization/geometry, resource hardening/metrics/datasheet | Repeated load/failure/cancel/state-isolation tests |
| 006 | Packaging, SLAIF integration and release readiness | Human adjudication, distribution/license/security/supply-chain review before applicable release/deployment gates |

Each numeric objective is one PR. Use lettered remediation until accepted/merged.
Do not begin the next numeric objective until the predecessor is merged and remote
`main` is independently verified.

## Known target-host correction

The bootstrap originally carried an unverified assumption that physical GPU1
might be an RTX 2080 Ti-class card with 22/24 GB. Human/operator preflight on
2026-08-23 instead observed two ordinary RTX 2080 Ti devices with **11264 MiB**
each; physical GPU1 was UUID `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8` and was
essentially idle at that moment, while GPU0 had an unrelated workload. Objective
003 must re-verify these facts live because index/UUID/process/free-memory state
can change. The 11-GB measurement is the planning reality until fresh evidence
says otherwise; GPU0 remains protected regardless of apparent idleness.

## Human adjudication gate

Open `CRITICAL.md` entries do not automatically stop autonomous implementation.
Before any external deployment, production/customer data, irreversible production
mutation, security-policy relaxation at an external boundary, or final release,
the human must append an `ACCEPTED` disposition for every entry whose stated gate
applies. `DEFERRED`, `REJECTED`, or `CHANGE REQUIRED` stays blocking.
Packaging and local test preparation may continue when safe; crossing the gate
may not.
