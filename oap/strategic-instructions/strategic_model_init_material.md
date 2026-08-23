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
- all GPU index/UUID/PCI/name/VRAM/process state; verify physical GPU1 claim;
- intended Python/PyTorch/CUDA/SAM2/CLIP/BLIP3 environment and model caches;
- `/dev/shm` type/capacity/permissions;
- loopback listeners and an unused candidate port; do not reserve/change yet.

Do not treat seed order as final. Work in strategic `drafts/`; replace every
`VERIFY:` and `DRAFT UNTIL` statement with exact evidence and bounded decisions.
Publish only after coherent PR-sized scope. Human work has been preloaded into
these artifacts: do not stop merely because a consequential choice lacks an
explicit human answer. Decide provisionally. Use repository `CRITICAL.md` only
when every strict threshold condition holds; otherwise keep ordinary decisions
in the order/architecture/evidence.

## Product sequence

1. Professional baseline and honest current-test/CI/docs/provenance state.
2. In-memory typed single-image core, artifact sink, YOLO and identity PNG.
3. `/v1/completions` transport and fake-engine CPU contract.
4. Physical GPU1 qualification and dependency/model revision/license audit.
5. Loopback service activation on verified unused port with rollback/E2E.
6. Full artifact parity/resource hardening/metrics/datasheet.
7. Packaging/gateway/release only after local evidence.

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
