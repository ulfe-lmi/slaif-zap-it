# OAP Work Order 024-c — synchronize gateway and API documentation

## Objective

Amend Objective 024 PR #88 in place with a documentation-only product
correction. Preserve all accepted 024-a/024-b runtime behavior and evidence,
while removing independently verified contradictions that still direct a future
SLAIF gateway to the private native endpoint and misdescribe the service's
inference/authentication surfaces.

The final documentation must state one architecture consistently:

- `POST /v1/completions` is the native/private multipart operator, research,
  and debugging API. It is intentionally not OpenAI Completions compatibility,
  is not the slaif-api-gateway backend contract, and is not the general-public
  SLAIF surface.
- `POST /v1/responses` is the narrow stateless/non-streaming OpenAI
  Responses-compatible facade intended for future gateway/public routing. It
  returns canonical public JSON and optionally one standard
  `image_generation_call` annotated PNG.
- The gateway repository remains unchanged and currently lacks its canonical
  Responses multimodal/image-generation path. Full SDK -> gateway -> ZAP-IT
  qualification remains later cross-repository work.

Do not alter runtime code, schemas, capabilities output, API behavior, models,
pipeline semantics, authentication implementation, deployment topology, or the
gateway repository.

## Deferred human adjudication

- Decision: NONE

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Remote `main` remains
  `32812032781c5d7daf54d5b7586b3c01d3270c48`.
- Amend only open PR #88, branch
  `oap/024-a-openai-responses-compatible-facade`, titled
  `Objective 024: OpenAI Responses-compatible facade`.
- The reviewed remote head is immutable 024-b report-only SELF commit
  `dbbd087b009646e533f10bcdec889900296137fb`; its first parent is accepted
  corrective implementation commit
  `639a319041cfa7f72f8fa5d645d43f062d24bcb7`.
- All seven CI/CodeQL checks on that head are green; GitHub reports
  MERGEABLE/CLEAN and the worktree is clean. The code and live behavior satisfy
  the reviewed 024-b correction, but documentation contradictions below block
  merge.
- Continue on the same branch/PR. Commit the published 024-c order/active
  transcript and docs/test correction after the immutable 024-b report. Do not
  modify any prior order or report.
- Do not merge, enable auto-merge, rewrite history, amend commits, or touch
  another PR.

Refresh current GitHub/local state, `CRITICAL.md`, docs checker, capabilities
metadata, and running service facts before mutation. The service is currently
healthy/ready as PID 815951 on exact authorized listener
`10.8.132.76:17891`, using only assigned physical GPU0 UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`. Because this round must not change
runtime source, do not restart or rerun expensive inference. Leave it running.

## Independent review evidence

### Finding 1 — gateway document selects the forbidden surface

`docs/GATEWAY-INTEGRATION.md` currently says the future backend request is
multipart `POST /v1/completions`, forwards native `verbosity` and
`response_format`, and asks the gateway to map JSON/ZIP artifacts. This is the
superseded pre-Objective-024 design. It directly contradicts the human
requirement that `/v1/completions` not pass through slaif-api-gateway and that
the new standard Responses facade be the gateway-facing compatibility surface.

### Finding 2 — API document contains false/exclusive and incomplete auth claims

`docs/API.md` lists both endpoints but later says `/v1/completions` “remains the
only inference contract.” It also describes configured/private-LAN bearer
enforcement only for `/v1/completions` and `/metrics`, omitting the new
`/v1/responses` surface. Those statements are inconsistent with the implemented
route and authenticated capabilities metadata.

The document also does not make all three required `/v1/completions` non-goals
as explicit as the dedicated Responses document: private research/debug,
not gateway-facing, and not general-public.

## Required documentation correction

### 1. Rewrite the proposed gateway mapping

Update `docs/GATEWAY-INTEGRATION.md` so it is an accurate future dependency
contract, not a claim of completed integration:

- The future gateway backend call is JSON `POST /v1/responses` using fixed
  model `zap-it-1`.
- It uses the normal Responses input-item/content-part structure with exactly
  one inline base64 `input_image` and one safe inline base64 YAML `input_file`.
- It is stateless/non-streaming with `store=false`; unsupported state and tools
  remain explicit errors.
- The optional canonical declaration is
  `tools: [{"type":"image_generation"}]`. The output is assistant
  `output_text` containing the deterministic public projection plus exactly one
  standard `image_generation_call.result` PNG when requested.
- The gateway must not route public/general requests to `/v1/completions`,
  proxy native verbosity, JSON/ZIP/debug artifacts, or invent a ZAP-specific
  output type/bypass.
- Keep the separate high-entropy backend bearer boundary and bounded timeout.
  Do not invent per-user auth, quota, rate, billing, TLS, or gateway functions
  inside ZAP-IT.
- State clearly that current slaif-api-gateway does not yet support this
  Responses multimodal/image-generation path and was not changed by this PR.
- Define the later cross-repository official-SDK qualification and truthful
  non-token accounting without claiming it already passes.

Remove every obsolete statement that prescribes multipart
`/v1/completions`, native verbosity/response_format, JSON/ZIP forwarding, or
OpenAI text-completion mapping for the future gateway.

### 2. Correct the API and architecture wording

Update `docs/API.md` so:

- its opening and endpoint descriptions explicitly label `/v1/completions` as
  native/private operator/research/debug, not OpenAI Completions, not intended
  for slaif-api-gateway, and not the general-public SLAIF contract;
- `/v1/responses` is explicitly the bounded future gateway/public
  compatibility surface, without claiming completed gateway or public/WAN
  deployment;
- the KServe/Triton model-management paragraph says there is no V2 tensor
  inference endpoint, rather than falsely declaring `/v1/completions` the only
  inference contract;
- configured-key and private-LAN authentication descriptions include
  `/v1/responses` consistently with the actual route. Preserve the existing
  health/capabilities/model-control distinctions and do not invent a new auth
  policy.

Review `ARCHITECTURE.md`, `README.md`, `docs/SERVICE-DATASHEET.md`, and
`docs/RUNBOOK.md` only for directly related wording. Make the private/native
versus future public/facade division explicit where needed, but avoid unrelated
editorial churn. In particular, do not change historical documentation under
`docs/history/`.

### 3. Add durable documentation regression proof

Extend the maintained documentation checker or an Objective 024 contract test
with deterministic assertions that current non-historical docs cannot regress
to the obsolete mapping. At minimum prove:

- the gateway integration document selects `/v1/responses` and its standard
  `image_generation_call` path;
- it explicitly excludes `/v1/completions` as the gateway/public contract;
- it contains no instruction that the backend request is multipart
  `/v1/completions` or that native JSON/ZIP debug artifacts are gateway output;
- API documentation identifies both supported inference surfaces without the
  false phrase that completions is the only inference contract; and
- the API authentication text includes Responses under the implemented fixed
  bearer/private-LAN policy.

Keep assertions resilient to harmless prose changes: test required architectural
facts and forbidden obsolete claims, not entire paragraphs.

## Non-goals and preservation

- No source under `src/` or `modules/`, schema, capabilities, configuration,
  renderer, scripts, package dependencies, deployment files, or gateway code.
- No service restart, model load, GPU inference, new qualification artifact,
  network/auth change, or public deployment.
- Do not modify the activated 024-a/024-b orders or immutable reports. A
  PR-range `git diff --check` currently reports one orchestration-only trailing
  blank line in the immutable 024-a order. Do not rewrite that activated
  transcript to make the command silent; report the known non-product warning
  honestly and require no new whitespace errors in 024-c files.
- Preserve the live 024-b official SDK/native qualification because this round
  makes no runtime-source change. Prove that the 024-c implementation diff from
  `639a319…` contains no `src/`, `modules/`, deployment, dependency, schema, or
  capability path.

## Required verification

Run and report:

1. focused documentation/Objective 024 regression checks;
2. the maintained documentation checker;
3. the complete CPU/fake suite;
4. formatting, lint, build, release-artifact, twine, secret, banned-file, and
   tracked-sensitive-name gates in the established environments; and
5. all seven required GitHub CI/CodeQL checks on implementation and final SELF
   heads.

Recheck only content-free live facts after the docs-only commit: PID/start time,
health/readiness, exact listener, assigned GPU UUID/process, unauthenticated
Responses rejection, and absence of runtime-source changes since the live
024-b product commit. Do not read/print the bearer and do not repeat model
inference solely for documentation.

## Acceptance criteria

1. No current non-historical documentation directs slaif-api-gateway to
   `/v1/completions`.
2. The proposed future gateway mapping uses canonical `/v1/responses` input,
   public JSON output, and optional standard image-generation output.
3. `/v1/completions` is unambiguously documented as native/private/debug,
   non-OpenAI-Completions, non-gateway, and non-general-public.
4. `/v1/responses` is unambiguously documented as the future gateway/public
   compatibility surface without claiming completed gateway/public deployment.
5. API authentication wording matches implemented fixed-key/private-LAN
   behavior for Responses and preserves all other auth distinctions.
6. Durable tests reject the obsolete gateway and “only inference contract”
   wording.
7. No runtime/product source or behavior changes; all existing tests/checks
   remain green.
8. The already qualified corrected service remains healthy, ready, and running
   on the authorized private-LAN/GPU assignment.

## Report and SELF contract

Write immutable `oap/reports/024-c-report.md`. Identify PR #88, base, 024-c
implementation commit, all commits added this round, exact changed files/diff,
the corrected/forbidden documentation claims, focused/full tests, all checks,
proof of no runtime-source change, retained live service facts, the known
immutable 024-a diff-check warning, security/scope review, strongest reason not
to merge and answer, and `Deferred human adjudication: NONE`.

After docs/tests, verification, push, and report content are complete, commit
only `oap/reports/024-c-report.md` as final SELF. Its first parent must be the
reviewed 024-c implementation head. Push, wait for every required SELF check,
then signal the response FIFO with exact `OK`. Coding must not merge.
