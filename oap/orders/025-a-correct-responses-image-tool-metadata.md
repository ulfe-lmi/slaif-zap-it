# OAP Work Order 025-a — correct Responses image-tool metadata

## Objective

Create one narrowly scoped Objective 025 PR that corrects the successful
`POST /v1/responses` envelope when the accepted request contains
`tools: [{"type":"image_generation"}]`.

The merged Objective 024 implementation currently emits one completed
`image_generation_call` but simultaneously reports `tool_choice: "none"` and
`tools: []`. That is internally and OpenAI-semantically inconsistent: `none`
means that no tool may be called, while the response contains a tool call and
does not disclose the tool made available by the request.

Use this exact effective metadata policy:

- with accepted `tools: [{"type":"image_generation"}]`, return
  `tools: [{"type":"image_generation"}]`, `tool_choice: "auto"`, and retain
  `parallel_tool_calls: false`;
- without the tool declaration, preserve the existing response metadata
  `tools: []`, `tool_choice: "none"`, and `parallel_tool_calls: false`.

Do not change the accepted request subset. In particular, do not add a public
`tool_choice` request field in this objective. The request omits it and the
current official Responses default is `auto`; the service's narrow adapter then
selects its only declared tool and emits exactly one canonical annotated PNG.

Preserve every Objective 024 inference, public projection, renderer, error,
bound, authentication, private `/v1/completions`, and gateway-boundary
behavior. This is a protocol-metadata correction, not a facade redesign.

## Deferred human adjudication

- Decision: NONE

This is an ordinary, reversible protocol conformance bug whose correct outcome
is resolved by the official schema and examples. It does not meet the
`CRITICAL.md` threshold.

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`.
- Remote default branch: `main` at
  `fae1397bac15792bd6c064ee943c2f0f615aea9d`.
- Objective 024 PR #88 is actually MERGED at that exact merge commit; merged at
  `2026-09-02T09:20:49Z`. Its remote head is immutable report-only SELF commit
  `89079f6d5257b36b983c99589ef96f1035304b7c`.
- Create exactly one new branch from current `origin/main`, preferably
  `oap/025-a-correct-responses-image-tool-metadata`, and exactly one new PR
  titled `Objective 025: correct Responses image-tool metadata`.
- The local worktree is clean but still names the merged Objective 024 branch.
  Fetch first and create the new branch from the exact remote-main SHA; do not
  build on the old branch tip or rewrite history.
- Existing unrelated Dependabot PRs #79–#86 are out of scope.
- Do not merge, enable auto-merge, force-push, amend published commits, publish a
  release/tag/package, or modify another repository.

Read the coding constitution, communication contract, full architecture/API
material applicable to Responses, security/testing law, active order, merged
Objective 024 orders/reports, and current `CRITICAL.md` before mutation.

## Independent strategic evidence and canonical decision

Current merged `src/service/responses.py::build_responses_response` always sets:

```json
{"parallel_tool_calls":false,"tool_choice":"none","tools":[]}
```

It does so even after appending an output item with
`type: "image_generation_call"`. The merged `ResponsesResponse` Pydantic model
then hard-codes `tool_choice` to `Literal["none"]` and constrains `tools` to an
empty `List[Any]`, so the locally maintained schema enforces the contradiction.
The Objective 024 SDK test proves only that the SDK can deserialize the output;
it does not assert semantic tool metadata.

Current official OpenAI evidence reviewed on 2026-09-02 establishes:

1. The image-generation guide declares
   `tools: [{"type":"image_generation"}]` and obtains the base64 result from
   an `image_generation_call` output item:
   <https://developers.openai.com/api/docs/guides/tools-image-generation>.
2. The current Responses create reference says omitted `tool_choice` defaults
   to `auto`, defines `tools` as the array of tools available while generating
   the response, and defines `tool_choice` as the selection policy:
   <https://developers.openai.com/api/reference/cli/resources/responses/methods/create>.
3. Current official successful-response examples echo the supplied tool in the
   response `tools` array and report `tool_choice: "auto"` when the tool is
   selected. The same schema defines `none` as no tool call and `auto` as a
   choice between a message and one or more tool calls.
4. The repository-pinned official `openai==3.7.0` SDK types accept
   `Response.tool_choice` values `none|auto|required|...` and a typed
   `Response.tools` array containing `ImageGeneration`.

Therefore the required correction is `auto` plus the declared image-generation
tool for the tool-bearing success response. Do not substitute `required`, a
forced-tool object, a ZAP-specific item, or a new request control. Retaining
`parallel_tool_calls: false` is valid and truthful because this facade executes
at most one declared tool and one call.

## Required implementation

### 1. Build conditionally truthful response metadata

In the existing response adapter, derive successful response metadata from the
already validated `image_generation` boolean:

- false: preserve `tool_choice: "none"` and `tools: []` exactly;
- true: emit `tool_choice: "auto"` and exactly
  `tools: [{"type":"image_generation"}]`.

Continue to emit one assistant message in all successes and exactly one
completed `image_generation_call` only when the tool is present. Keep output
ordering, IDs, timestamps, deterministic public JSON, base64 PNG bytes,
renderer settings, and all response-size calculations unchanged except for
including the corrected metadata bytes naturally in the existing final size
check.

Do not infer tool metadata from output after the fact or maintain two unrelated
truth sources. Use the existing validated request decision that already controls
canonical rendering.

### 2. Correct the maintained response schema

Replace the misleading empty `List[Any]` response declaration with the existing
typed `ResponsesTool` shape, bounded to zero or one element. Allow only the two
effective `tool_choice` string values this service can actually return:
`"none"` and `"auto"`.

Add a small cross-field invariant in the response model or equivalently focused
builder/schema proof so impossible local success envelopes cannot regress:

- an `image_generation_call` requires exactly the image-generation tool and
  `tool_choice: "auto"`;
- the declared image-generation tool requires exactly one such output call;
- the no-tool envelope has no image call and uses `tool_choice: "none"`.

Do not broaden response tools or output-item types.

### 3. Strengthen SDK qualification and contract tests

Extend the existing Objective 024 tests instead of creating a parallel facade
test framework. At minimum prove:

1. A no-tool response remains message-only with `tools: []`,
   `tool_choice: "none"`, and `parallel_tool_calls: false`.
2. A request with `tools: [{"type":"image_generation"}]` returns one message,
   one image call, `tools: [{"type":"image_generation"}]`,
   `tool_choice: "auto"`, and `parallel_tool_calls: false`.
3. The maintained Pydantic response schema rejects mismatched combinations such
   as an image call with `tool_choice: "none"`/empty tools and a declared image
   tool without its call.
4. The generated OpenAPI response schema advertises the typed bounded tool array
   and both actual tool-choice values rather than an always-empty untyped list.
5. The current official `openai==3.7.0` SDK deserializes the live-shaped
   response, exposes `response.tool_choice == "auto"`, exposes exactly one
   typed response tool whose `type == "image_generation"`, still yields the
   canonical JSON through `response.output_text`, and still yields exactly one
   valid PNG through the typed output item.
6. The no-tool and tool-bearing public projection text are unchanged/equal for
   the same fake inference, and canonical PNG equality with the shared existing
   renderer remains byte-for-byte proven.
7. Existing unsupported request fields/tools, auth, capacity, body/cardinality,
   errors, and private completions regressions remain unchanged and green.

Update `scripts/qualify_responses.py` so its official-SDK qualification fails if
the semantic response metadata is wrong. Its bounded content-free summary may
add the response tool count/type and effective `tool_choice`; it must not record
or print the bearer, input content, projection content, prompts, answers, image
bytes, or any other request data.

### 4. Synchronize only directly affected documentation

Update `docs/RESPONSES-FACADE.md` and any directly applicable generated/current
API description so the successful output contract explicitly states the
conditional metadata above. Preserve the established distinction:

- `/v1/completions` is native/private research/debug and behaviorally
  unchanged;
- `/v1/responses` is the bounded future gateway/public compatibility facade;
- current `slaif-api-gateway` still lacks this canonical multimodal/image-output
  path and is not changed or newly qualified here.

Do not claim public/WAN deployment or completed gateway integration.

## Non-goals and preservation

- No inference-engine, SAM2, geometry, CLIP, routing, BLIP3, final-label,
  candidate-view, renderer, PNG-encoder, public-projection, or artifact changes.
- No request-schema expansion, new state, streaming, conversations, persistence,
  background mode, additional tools, arbitrary URLs/files, usage/token
  fabrication, or gateway code.
- No `/v1/completions` request, response, error, byte fixture, native artifact,
  or documentation-boundary regression.
- No model/device/path/network/auth/concurrency/timeout/operator-limit change.
- No credential output, raw image/YAML/result retention, or request-content
  logging.
- Do not modify historical docs, earlier immutable OAP orders/reports,
  `CRITICAL.md`, unrelated dependencies, or Dependabot branches.

Keep the diff PR-sized and centered on the response builder/schema, existing
Objective 024 tests, official SDK qualifier, and directly affected facade docs.

## Required verification

Run and report exact commands/statuses for:

1. focused Responses facade, schema/OpenAPI, SDK, qualification-helper, and
   documentation regressions;
2. the complete CPU/fake suite with coverage at or above the maintained gate;
3. Ruff format/check and compileall;
4. wheel/sdist build, release-member verification, archive/tracked secret
   scans, sdist-to-wheel comparison, and Twine checks;
5. deployment/systemd verification because the service will be restarted; and
6. all required GitHub CI and CodeQL checks on the implementation head and final
   SELF head.

Do not use GPU inference as a substitute for the deterministic fake/SDK contract
proof.

## Authorized live-service qualification

After implementation-head tests and checks pass, perform one controlled restart
of the existing private-LAN service so the corrected product code is running and
can be qualified. The currently authorized facts, independently refreshed on
2026-09-02, are:

- service PID before work: `815951`, started `Wed Sep 2 10:32:15 2026`;
- listener: exactly `10.8.132.76:17891`;
- assigned physical GPU index `0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI
  `00000000:0B:00.0`, NVIDIA GeForce RTX 3090, 24576 MiB, driver
  `610.43.02`;
- that service is the sole compute process reported on the assigned card and
  uses about 10572 MiB; application visibility is
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`, logical `cuda:0`,
  with the expected UUID set;
- `/dev/shm` is a 12 GiB tmpfs with about 9.7 GiB available;
- the service is healthy/ready and authenticated under the existing fixed
  private-LAN bearer policy.

Re-verify every GPU/index/UUID/PCI/name/VRAM/process, environment, `/dev/shm`,
listener ownership, and service state immediately before restart. Protect every
unassigned device/workload. Use only the repository launcher and one worker;
make no firewall, route, VPN, driver, CUDA, model-cache, system-wide, gateway, or
credential changes. Do not print/read the bearer into evidence; let the existing
private operator environment supply it.

The live qualification must use the official SDK script with its synthetic
in-memory 32x24 image/YAML and prove, without exposing content:

- authenticated HTTP success and official `SDKResponse` parsing;
- `response.tool_choice == "auto"`;
- exactly one typed response tool of type `image_generation`;
- exactly one completed `image_generation_call` and valid deterministic PNG;
- the expected public projection schema; and
- content-free size/hash/timing evidence only.

Also perform bounded content-free probes for health/readiness, unauthenticated
Responses rejection, assigned GPU/process ownership, and listener ownership.
Run focused native `/v1/completions` smoke/regression evidence sufficient to
show it was not altered. Leave the corrected service running. If the new service
cannot become healthy/ready or the live SDK qualification fails, roll back to
the already merged `fae1397…` product state and leave that healthy service
running; report the failure honestly rather than abandoning the service.

## Acceptance criteria

1. Tool-bearing successful Responses no longer claim `tool_choice: "none"` or
   hide the declared tool.
2. Their metadata is exactly `tool_choice: "auto"`,
   `tools: [{"type":"image_generation"}]`, and
   `parallel_tool_calls: false`.
3. No-tool Responses retain exactly the current no-tool metadata and output.
4. The local response schema mechanically rejects inconsistent tool/output
   combinations and OpenAPI advertises only implemented shapes.
5. The official OpenAI Python SDK obtains and types both the echoed tool metadata
   and image-generation output normally.
6. Public projection text, final annotated PNG bytes, inference, and private
   `/v1/completions` behavior are unchanged.
7. Bounds and error behavior remain enforced, including the final response-size
   check over the corrected envelope.
8. Current docs state the corrected conditional metadata and preserve the
   private-native/future-facade/gateway-dependency boundary.
9. Full tests, packaging/security gates, and every required CI/CodeQL check pass.
10. The corrected service is healthy, ready, qualified on the exact authorized
    GPU/private-LAN assignment, and left running.

## Report and SELF contract

Write immutable `oap/reports/025-a-report.md`. Identify the new PR, base/head,
implementation SHA, exact changed files/diff, official-semantic decision,
before/after response metadata, focused/full/SDK/OpenAPI/docs/package/security
evidence, CI URLs/status, live restart/rollback facts, post-restart PID/listener/
GPU/health/readiness, bounded qualification summary, native preservation,
security/scope review, strongest reason not to merge and its answer, and
`Deferred human adjudication: NONE`.

After implementation, verification, push, and report content are complete,
commit only `oap/reports/025-a-report.md` as final SELF. Its first parent must be
the reviewed implementation head. Push it, wait for every required final-SELF
check to complete successfully, then signal the response FIFO with exact `OK`.
Coding must not merge.
