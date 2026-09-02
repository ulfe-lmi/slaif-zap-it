# OAP Work Order 026-a — fix Responses warning sanitization

## Objective

Create one minimal Objective 026 PR that fixes character-spaced warning strings
in the deterministic `zap-it.public.v1` projection returned by
`POST /v1/responses`.

The exact implementation correction is confined to
`src/service/responses.py::_bounded_warning`: replace the current space
separator in the character generator join with an empty separator. Preserve
the helper's complete existing policy:

1. call `str(value)`;
2. replace every character whose code point is below 32 with one ordinary
   space;
3. preserve every other character exactly and in order; and
4. truncate the sanitized result to at most 256 characters.

The helper is shared by public top-level `warnings` and
`sam2.resource_warnings`; both paths must receive the corrected result. Do not
change where warnings originate or how either list is bounded.

Add the requested focused unit and HTTP regression coverage, run the full
suite, and leave the corrected private-LAN service healthy, ready, and running.
Do not refactor warning handling or alter any other endpoint behavior.

## Deferred human adjudication

- Decision: NONE

This is a verified one-line correctness bug with an unambiguous, reversible
fix and no material architecture or trust-boundary choice.

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Remote default branch: `main` at
  `90c4b4923e4924dcffed185a0bf54ffeea5f7eb4`.
- Objective 025 PR #89 is MERGED at that commit. Its remote report-only SELF
  head is `0de349e36ba81af1780d74f69403fc0fffd582cc`; implementation parent is
  `febc88d0494d747a28324c8230057eac527b6661`.
- Create exactly one new branch from current `origin/main`, preferably
  `oap/026-a-fix-responses-warning-sanitization`, and exactly one new PR titled
  `Objective 026: fix Responses warning sanitization`.
- The local worktree is clean but still names the merged Objective 025 branch.
  Fetch first and create the new branch at the exact remote-main commit. Do not
  build on the old branch tip or rewrite history.
- Existing unrelated Dependabot PRs #79–#86 are out of scope.
- Do not merge, enable auto-merge, force-push, amend published commits, publish
  a release/tag/package, or touch another repository.

Read the coding constitution, communication contract, architecture/security/
testing material, exact active order, merged Objective 025 report, current
Responses adapter/tests, and `CRITICAL.md` before mutation.

## Independently verified root cause

Merged `src/service/responses.py` contains:

```python
def _bounded_warning(value: Any) -> str:
    text = str(value)
    text = " ".join(character if ord(character) >= 32 else " " for character in text)
    return text[:256]
```

The generator correctly maps each code point below 32 to a space, but
`" ".join(...)` additionally inserts a space between every generated
character. That transforms ordinary warning text into character-spaced output.
The correct expression is `"".join(...)`.

The same helper is called in exactly two public-projection locations:

- every bounded `metadata["resource_warnings"]` entry under
  `sam2.resource_warnings`; and
- every result/config warning under top-level `warnings`.

Responses inference always validates the inline YAML at verbosity 2, so the
existing YAML validator emits these exact safe top-level warnings when relevant:

```text
debug flag postsam2processing.debug ignored at verbosity below 3
diagnostic_artifacts selection is valid but not applied below verbosity 3
```

No evidence points to a warning-generation, YAML, schema, or serialization bug.
Do not broaden the fix.

## Required implementation and tests

### 1. One-line helper correction

Change only the join separator from one space to the empty string. Retain
`str(value)`, the exact `ord(character) >= 32` boundary, one-space replacement,
and final `[:256]` truncation. Do not introduce Unicode normalization, `isprintable`,
regular expressions, stripping, whitespace collapsing, or a replacement helper.

### 2. Focused sanitizer regression

Extend the existing Responses contract test module; do not create a parallel
warning subsystem or broad refactor. Prove all of the following exactly:

1. A normal printable warning is byte-for-byte unchanged.
2. Newline, tab, NUL, and at least one other code point below 32 each become a
   single ordinary space. Printable characters that were adjacent in the input
   remain adjacent—no separator spaces are added between them.
3. A printable input longer than 256 characters yields exactly its first 256
   characters and has length 256.
4. Projection construction applies the same corrected sanitizer to both a
   top-level warning and a SAM2 resource warning. Do not expose or expand
   unbounded metadata to prove this.

It is acceptable for this narrow test to import the private helper directly,
because regression of this exact helper is the defect being fixed. Keep the
test deterministic and CPU-only.

### 3. HTTP Responses regression

Using the existing fake-engine `/v1/responses` helpers and an inline YAML config,
prove the decoded assistant `response.output_text` projection contains the
complete exact string:

```text
debug flag postsam2processing.debug ignored at verbosity below 3
```

Add a separate or combined request containing a valid `diagnostic_artifacts`
selection and prove the exact complete string:

```text
diagnostic_artifacts selection is valid but not applied below verbosity 3
```

These assertions must compare complete list entries, not substrings, so
character-spacing cannot regress unnoticed. Confirm no character-spaced version
is present.

Use a valid minimal YAML configuration and retain the standard Responses input
shape. Do not change the validator or warnings merely to make the test pass.

### 4. Preservation proof

Retain and run the existing Objective 024/025 tests that prove:

- no-tool output remains message-only with `tools: []`, `tool_choice: "none"`,
  and `parallel_tool_calls: false`;
- the optional image tool still returns the echoed typed tool,
  `tool_choice: "auto"`, and exactly one valid deterministic
  `image_generation_call` PNG;
- public object records, projection structure/determinism, schemas, bounds,
  errors, and private `/v1/completions` behavior remain unchanged.

Do not update expected output by weakening any of these assertions.

## Strict scope and non-goals

Expected product/test paths are only:

- `src/service/responses.py` — one separator change; and
- `tests/test_objective_024.py` — focused regression additions.

Plus the required OAP active/order/report transcript. Any other path requires a
specific necessity explained in the report and should ordinarily be removed
from the diff.

Specifically forbidden:

- no inference, SAM2, geometry, CLIP, routing, BLIP3, rendering, artifact,
  object-record, model, residency, or configuration behavior changes;
- no YAML validator, warning text/generation/order/list-limit, response schema,
  public projection field, serializer, endpoint, authentication, resource bound,
  error, or capabilities change;
- no `/v1/completions` change;
- no OpenAI Responses metadata/image-generation behavior change;
- no documentation rewrite for a self-evident output bug whose documented
  contract is already correct;
- no dependency, deployment, gateway, device, network, credential, or logging
  change; and
- no modification of prior immutable OAP orders/reports or `CRITICAL.md`.

## Required verification

Run and report exact commands/statuses for:

1. `.venv/bin/pytest -q tests/test_objective_024.py`;
2. the complete CPU/fake suite with coverage at or above the maintained gate;
3. Ruff format/check and compileall;
4. the maintained documentation check (documents must remain unchanged);
5. wheel/sdist build, release-member verification, archive/tracked secret scans,
   sdist-to-wheel comparison, and Twine checks;
6. `git diff --check`; and
7. every required GitHub CI and CodeQL check on both implementation and final
   report-only SELF heads.

The focused report must name the added tests and their exact pass count; the
full report must distinguish the expected explicit GPU-marker skip from a pass.

## Authorized live-service qualification

After implementation-head local tests and required GitHub checks pass, perform
one controlled restart so the fixed Responses behavior is actually running.
The independently refreshed pre-order facts are:

- current service PID `837516`, started `Wed Sep 2 12:23:59 2026`;
- exact listener `10.8.132.76:17891`;
- health/readiness `200/200`; unauthenticated Responses `401`;
- assigned physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI
  `00000000:0B:00.0`, NVIDIA GeForce RTX 3090, 24576 MiB, driver
  `610.43.02`;
- the service is the sole compute process on that card at about 10870 MiB;
- process visibility is `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES=0`, logical `cuda:0`, with the expected UUID pinned;
  and
- `/dev/shm` is a 12 GiB tmpfs with about 9.7 GiB free.

Immediately before restart, re-verify every assigned GPU fact/process,
environment, listener owner, service state, and `/dev/shm`. Protect every
unassigned device/workload. Use only the repository launcher and existing
operator environment. Do not print or persist the bearer. Make no firewall,
route, VPN, driver, CUDA, model-cache, system-wide, gateway, or credential
change.

After readiness, send one bounded authenticated Responses request built from a
small synthetic in-memory image and safe inline YAML that exercises both warning
sources above. Retain/report only content-free status plus equality booleans or
the two already-public expected warning constants; do not retain raw image,
YAML, response body, bearer, prompt, or model output. Prove:

- HTTP 200 and valid `zap-it.public.v1` JSON output text;
- the two exact complete warning entries are present with no character spacing;
- existing no-tool metadata remains `none`/empty;
- a separate optional image-tool fake/SDK regression already covers image
  output, so do not perform expensive duplicate live inference unless needed;
- health/readiness remain 200, unauthorized Responses remains 401, listener and
  assigned GPU ownership remain exact; and
- one bounded native `/v1/completions` smoke remains 200 and structurally
  unchanged.

Leave the corrected service running. If it cannot become healthy/ready or the
live warning proof fails, roll back to merged main `90c4b49…`, leave that service
healthy, and report failure instead of abandoning the listener.

## Acceptance criteria

1. Ordinary warning characters are no longer separated by inserted spaces.
2. Code points below 32 become spaces; all other characters remain exact; final
   length is at most 256.
3. Both public top-level and SAM2 resource warnings use the corrected helper.
4. The debug and diagnostic-artifact HTTP warnings match the two required full
   strings exactly.
5. Existing Responses metadata and optional image-generation behavior remain
   unchanged.
6. No inference, validator, warning-generation, schema, object, renderer,
   endpoint, private completion, bound, auth, or gateway behavior changes.
7. Focused/full tests, package/security gates, and all CI/CodeQL checks pass.
8. The corrected service is healthy, ready, live-qualified on the exact assigned
   GPU/private-LAN listener, and left running.

## Report and SELF contract

Write immutable `oap/reports/026-a-report.md`. Identify PR/base/head,
implementation SHA, exact changed files/diff, before/after helper expression,
both projection call paths, focused/full test commands/results, exact warning
equality evidence, Responses metadata/image preservation, packaging/security/
CI results, live restart or rollback facts, final PID/listener/GPU/health/
readiness, private completion preservation, strongest reason not to merge and
its answer, and `Deferred human adjudication: NONE`.

After implementation, verification, push, and report content are complete,
commit only `oap/reports/026-a-report.md` as final SELF. Its first parent must be
the reviewed implementation head. Push it, wait for every required final-SELF
check to complete successfully, then signal the response FIFO with exact `OK`.
Coding must not merge.
