# Strategic initialization — SLAIF ZAP-IT

You supervise an existing research prototype, not a greenfield rewrite. Inspect
current GitHub and local clone before planning. Existing pipeline/tests may have
changed after this bootstrap.

## Required first reconnaissance

- `gh repo view`, default branch/current SHA/open PRs/checks/branch protection;
- repository tree, current tests, last commit, dirty/unrelated local work;
- run existing test suite in a safe CPU environment without model downloads;
- identify packaging/docs/CI gaps versus the reference SLAIF service;
- `opencode --version`, authenticated provider/model choices, `gh auth status`;
- all GPU index/UUID/PCI/name/VRAM/process state; verify the active-order
  operator-assigned index+UUID claim;
- intended Python/PyTorch/CUDA/SAM2/CLIP/BLIP3 environment and model caches;
- `/dev/shm` type/capacity/permissions;
- loopback listeners and an unused candidate port; do not reserve/change yet.

Human/operator preflight on 2026-08-23 observed physical GPU1 as an RTX 2080 Ti,
11264 MiB, UUID `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, with GPU0 carrying an
unrelated workload. Objective 008 separately recorded an explicit hinton2
assignment of physical GPU0 to an RTX 3090, UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`. These are host-specific evidence.
Re-verify the exact index/UUID assigned by the active order before live use; do
not infer selection from idleness, and fail closed on a mismatch. All unassigned
devices and unrelated workloads remain protected.

## Human-work-preloaded objective sequence

The intended program is already decomposed into detailed draft `NNN-a` work
orders under `initial-orders/`:

```text
000-a-professional-baseline-and-ci.md
001-a-in-memory-core-and-renderers.md
002-a-v1-completions-api-contract.md
003-a-gpu1-runtime-qualification.md
004-a-loopback-service-activation.md
005-a-full-output-parity-hardening-and-evidence.md
006-a-release-and-integration.md
```

These drafts are human-preloaded intent, **not activated orders**. Do not invent
the roadmap from scratch and do not publish a draft verbatim. Before each numeric
objective, read its draft completely, reconcile it against the newly merged
remote main, predecessor report/PR/checks, architecture, current `CRITICAL.md` and
relevant host facts, then replace every `VERIFY:` and `DRAFT UNTIL` statement with
exact evidence and bounded decisions. Preserve intended outcome, non-goals,
acceptance and dependency gates unless verified reality requires a documented
material deviation.

Work in strategic `drafts/`; publish only after coherent PR-sized scope. Human
work has been preloaded into these artifacts: do not stop merely because a
consequential choice lacks an explicit human answer. Decide provisionally. Use
repository `CRITICAL.md` only when every strict threshold condition holds;
otherwise keep ordinary decisions in the order/architecture/evidence.

## Product sequence

1. Professional baseline and honest current-test/CI/docs/provenance state.
2. In-memory typed single-image core, artifact sink, YOLO and identity PNG.
3. `/v1/completions` transport and fake-engine CPU contract.
4. Operator-assigned-GPU qualification and dependency/model revision/license
   audit.
5. Loopback service activation on verified unused port with rollback/E2E.
6. Full artifact parity/resource hardening/metrics/datasheet.
7. Packaging/gateway/release readiness only after local evidence and applicable
   human adjudication gates.

Preserve dependencies: do not deploy API before pure core; do not activate live
service before fake/API and GPU qualification; do not expose LAN before auth/
policy. Split objectives if a PR becomes unreviewable.

## OAP operations

Final order publish:

```bash
python "$REPO_ROOT/oap/bin/publish_order.py" \
  --repo-root "$REPO_ROOT" --source "$FINAL" --id NNN-L
python "$REPO_ROOT/oap/bin/oap_fifo.py" send --fifo "$CONTROL_FIFO"
python "$REPO_ROOT/oap/bin/oap_fifo.py" wait --fifo "$RESPONSE_FIFO"
```

Then independently inspect GitHub/report and merge only on full satisfaction and
all required green checks. Verify any ordered CRITICAL append is exact and
append-only. Continue after a recorded provisional decision; stop only before a
non-delegable human gate. Maintain `workorders/EXECUTION_TIMINGS.md` locally and
use `critical-drafts/` for strategic-authored entry sources.
