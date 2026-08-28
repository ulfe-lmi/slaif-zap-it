# OAP Work Order — 015-e — Reconcile the static tokenizer journal record

## Objective

Correct one factual contradiction in the immutable 015-d report without changing
product code or live state. Record that the new boot journal did contain the
allowed fixed “Special tokens…” vocabulary message, explain why it is not a
credential/token/path/request leak, and preserve the otherwise complete
Objective-015 evidence. This is a report-reconciliation round on existing PR
#71, not an implementation change.

## Verified starting state

- Remote `main` remains
  `1c6e42c28e3a4c29fff4c16be8311176ba07621a`.
- PR #71 is open and cleanly mergeable on branch
  `oap/015-a-request-local-sam2-configuration`, base `main`, at report-only SELF
  head `bb0b55a44b917ce0a10b8780855f2470667460c5`. Its parent is 015-d
  implementation SHA `792dfad5ef14320fee6fac72bbeb24d1da3478a7`; SELF changes
  only `oap/reports/015-d-report.md`. All current-head CI/CodeQL checks pass.
- The 015-d code and tests are satisfactory: the exact TIMM path-bearing
  `FutureWarning` is absent, while the same message from another module and an
  unrelated warning remain visible. Do not modify them.
- The live service is enabled, active, ready at `10.8.132.76:17891`, PID
  `449821`, `NRestarts=0`, with one listener and the sole compute process on
  assigned GPU0 UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`.
- The actual new-boot journal contains the static line:
  `Special tokens have been added in the vocabulary, make sure the associated
  word embeddings are fine-tuned or trained.` It contains no absolute path,
  bearer/authorization value, environment variable name, request/image/YAML,
  user label/prompt, cache/checkpoint/repository location or customer content.
  It is a model-vocabulary notice; the English word “tokens” is not evidence of
  an API credential.
- The 015-d order explicitly allowed such a fixed path-free dependency message.
  However, 015-d report acceptance item 7 incorrectly says the journal scan
  found “zero ... fixed tokenizer records.” That clause contradicts the actual
  journal and must not remain the final factual record.

## Required action

1. Make no source, test, documentation, dependency, configuration, service,
   model, GPU, auth, network or artifact change.
2. Commit only exact `oap/active` plus this immutable 015-e order as the
   non-report implementation/control commit.
3. In `oap/reports/015-e-report.md`, explicitly supersede only the erroneous
   015-d “zero fixed tokenizer records” claim:
   - state that one fixed tokenizer vocabulary notice was present;
   - state that it was allowed by 015-d because it was static and contained no
     path, secret, environment name or request-controlled material;
   - retain the true result that the TIMM warning, TIMM filename,
     `FutureWarning`, path-bearing records, auth/key material, request content,
     errors and tracebacks were absent; and
   - state that 015-d's code/live acceptance remains complete.
4. Do not edit prior reports. Do not invent a new sanitizer count by rerunning a
   differently scoped scan; reconcile against the already observed boot record.

## Verification and live handling

- Verify the non-report commit changes only `oap/active` and this order.
- Run `git diff --check` and the documentation checker only as necessary for the
  OAP Markdown addition; no product test/build rerun is required because no
  product or test bytes change. GitHub's normal current-head checks must still
  all pass on implementation/control and report-only SELF heads.
- Read-only verify final PID `449821`, ready state, `NRestarts=0`, one listener,
  assigned-GPU process, empty request workspace and unchanged environment
  mode/digest. Do not restart, infer, or send completion requests. Never read or
  print a key value.

## Non-goals/safety

- No new PR, merge, release, tag or next objective.
- No log suppression/filter change, service restart, GPU inference, request
  mutation or credential operation.
- No rewrite of prior order/report bytes.

The strongest reason not to merge is an evidence trail that contradicts the
host journal even though the underlying behavior is safe. Acceptance requires a
precise append-only correction, not another code change or suppression of a
benign model notice.

## Publication/report contract

Amend exact PR #71 and branch `oap/015-a-request-local-sam2-configuration`.
Capture the literal control-implementation SHA, then create only
`oap/reports/015-e-report.md` as its report-only SELF child. Verify parent,
one-path topology, remote bytes and all current-head checks before FIFO response.
Coding never merges.

## Deferred human adjudication

- Decision: NONE
