# OAP Work Order — 005-a — Full-output parity, hardening, metrics and datasheet

> DRAFT UNTIL Objective 004 is merged and strategic live/GitHub reconciliation is complete. Do not publish as-is.
>
> **HWP status:** preloaded human engineering intent. Strategic must use the proven live service and measured model profile as reality, preserving bounded outputs and honest stage support rather than inflating “full” into an unbounded debug dump.

## Objective

Complete the service from “working local MVP” to a hardened release candidate for
controlled deployment. Reach honest parity with every safe output the enabled
legacy pipeline can actually produce, preserve overlap truth, integrate geometry
and visualizations where genuinely executed, harden request/result/resource
limits, add safe metrics and failure/cancellation behavior, run repeated live
stress/failure evidence on physical GPU1, and publish a service datasheet with
measured capabilities and limitations. Keep the service loopback-only.

## Prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `005 / 005-a`
- Mode: `CREATE_NEW_PR`
- Objective 004 merged on remote main, merge SHA/checks: VERIFY:
- Verified default branch/base SHA: VERIFY:
- Required branch/PR title: VERIFY:
- Existing objective-005 PR: N/A after strategic confirms none: VERIFY:
- Current live service port/profile/runbook from 004: VERIFY:

## Verified current state

- exact supported live model profiles/stages from 003/004: VERIFY:
- current L0–L3 content and missing legacy-safe outputs: VERIFY:
- current visualization stages/artifact names and semantics: VERIFY:
- current geometry functions, whether/where they execute, and expected outputs: VERIFY:
- current overlap representation and identity raster policy: VERIFY:
- current limits/timeouts/queue/response-size behavior: VERIFY:
- current logging/metrics instrumentation: VERIFY:
- current repeated-run VRAM/RSS/latency baseline: VERIFY:
- current API/security/test/coverage gaps: VERIFY:

## Scope

1. **Audit “everything it can produce.”** Map every current safe pipeline output
   from preprocessing, SAM2, postprocessing, CLIP, optional BLIP3, geometry,
   visualization, serialized metadata and YOLO into one of: public L0–L3 artifact,
   operator-only diagnostic, legacy CLI-only output, unsupported/dead behavior, or
   unsafe/not appropriate for service. Document the classification.
2. **Complete L3 bounded parity.** Return every bounded safe artifact actually
   produced by the enabled request configuration: final and selected stage
   metadata, rendered overlays, available CLIP/BLIP evidence, geometry outputs,
   warnings, timings, model/config/service provenance and applied limits. Do not
   expose host paths, environment, secrets, weights/cache details or raw stacks.
3. **Overlap-preserving masks.** In addition to the single-valued uint16 identity
   projection, provide a full-level per-object mask representation (e.g. bounded
   individual PNGs or COCO-style RLE) that preserves overlaps exactly. Verify
   object IDs align bijectively across object records, YOLO, identity PNG and
   overlap-preserving masks.
4. **Geometry integration.** If geometry is a real supported algorithm stage,
   wire it into the canonical core/service path with typed structured lines/
   intersections and tests. If legacy docs overstated execution or the feature is
   not safe/maintained, correct docs and expose a clear unsupported status instead
   of fabricating parity.
5. **Visualization parity.** Ensure stage/final visualizations use the same object
   data and deterministic ordering as structured outputs. Convert legacy path
   writers to logical artifacts without changing CLI behavior unexpectedly.
6. **Debug policy.** Define the finite set of debug artifacts that may be enabled
   by API-safe config/operator policy. Bound count/size and prevent caller paths.
   Full verbosity is not permission to dump arbitrary internal state.
7. **Resource limits.** Calibrate and enforce encoded image/config bytes, decoded
   pixels/dimensions, YAML complexity, max objects, artifact count/size, total
   response bytes, queue depth, request/stage deadline, minimum host RAM/shared
   memory and any measured GPU-profile restrictions. Limits must be documented,
   tested at boundaries and fail with stable errors.
8. **Response-size preflight.** Estimate/track generated artifact bytes and abort
   safely before runaway base64/ZIP assembly. Ensure partial artifacts/workspaces
   clean up.
9. **Cancellation/failure hardening.** Test client disconnect/cancel where the
   stack permits, inference exceptions, serialization failure, full `/dev/shm`
   simulation, insufficient memory gates and service shutdown during bounded
   work. Never leave request state or corrupt persistent model registry state.
10. **Repeated load/stress.** On physical GPU1, execute a documented bounded load
    sequence across supported configs, including repeated sequential requests and
    controlled overlapping clients. Record latency distributions, object/artifact
    counts, response sizes, peak/end VRAM and host RSS. Do not exceed the one-GPU
    concurrency model or call this a production-scale benchmark.
11. **State-isolation testing.** Alternate materially different YAML configs and
    images through repeated calls; prove labels, masks, artifacts, prompt/config
    caches and warnings from request A do not leak into request B.
12. **Metrics.** Add safe local metrics if useful: requests/status, busy/rejected,
    stage duration, total latency, object/artifact counts, response bytes, model
    readiness and bounded resource gauges. Labels must be low-cardinality and must
    not include filenames, raw labels/prompts, config text, secrets or request IDs.
13. **Logging review.** Confirm structured logs remain content-safe under errors
    and L3. Include config digest rather than raw YAML and sanitized model/service
    identifiers rather than private paths.
14. **Security/property/fuzz tests.** Add targeted malformed image/YAML/multipart,
    path/model/device attempts, decompression and serialization edge cases using
    bounded local fixtures. Do not introduce flaky internet fuzzing into CI.
15. **Service datasheet.** Publish a concise but evidence-rich document covering
    purpose, non-goals, API levels, supported model stages/configurations,
    measured target hardware, resource limits, latency/memory observations,
    security/privacy behavior, persistence, known limitations, tests/evidence and
    deployment prerequisites.
16. **Release-candidate E2E.** Re-run local loopback service from a clean operator
    start and exercise all supported response levels/formats, health/readiness,
    auth policy, busy behavior, failure cleanup and restart using documented
    commands.

## Non-goals

- no LAN/public exposure, TLS/reverse proxy, multi-tenant identity or billing;
- no multi-GPU/multi-worker scale-out;
- no persistence/history/job queue;
- no training/fine-tuning;
- no unsupported BLIP3 activation merely for parity;
- no arbitrary raw debug/internal dumps;
- no SLAIF gateway integration or final release yet;
- no production/customer data.

## Acceptance criteria

1. A documented parity matrix accounts for every legacy/current output and clearly
   distinguishes public, operator-only, CLI-only, unsupported/dead and unsafe.
2. L3 exposes all bounded safe outputs of enabled supported stages without
   inventing disabled-stage data.
3. Per-object full masks preserve overlap truth and agree with object IDs; identity
   PNG remains deterministic documented projection.
4. Geometry/visualization behavior is either genuinely integrated/tested or
   documentation is corrected to an honest unsupported/deferred state.
5. All major input/output/resource limits have boundary tests and stable sanitized
   error behavior.
6. Response-size, `/dev/shm`, timeout/cancel and injected failure paths clean up
   completely and leave the model registry usable for subsequent requests.
7. Repeated/alternating requests show no cross-request state leakage and no obvious
   unbounded VRAM/RSS growth in the documented bounded test.
8. Safe metrics/logs contain no raw request content, secrets or high-cardinality
   customer-derived labels.
9. Local release-candidate E2E succeeds on physical GPU1 for all claimed supported
   profiles; GPU0 remains untouched.
10. Service datasheet exactly reflects measured hardware, limits, supported stages,
    evidence and limitations.
11. CPU CI/coverage/CodeQL and opt-in GPU/live tests relevant to this objective are
    green or explicitly BLOCKED with no false pass claim.
12. Correct one-PR/report-only SELF contract is satisfied; coding never merges.

## Required verification

- predecessor remote-main/CI/live state: VERIFY:
- parity matrix review against source/docs: VERIFY:
- full CPU suite/coverage/Ruff/package/CodeQL: VERIFY:
- overlap mask/object/YOLO/identity consistency tests: VERIFY:
- geometry/visualization parity tests: VERIFY:
- boundary/resource/property tests: VERIFY:
- cancellation/failure/cleanup tests: VERIFY:
- alternating-state isolation tests: VERIFY:
- live sequential/overlap load run with latency/VRAM/RSS/response-size evidence: VERIFY:
- metrics/log privacy inspection: VERIFY:
- release-candidate clean-start E2E and restart: VERIFY:
- GPU0 no-allocation evidence: VERIFY:
- datasheet/docs consistency check: VERIFY:

## Documentation and provenance

Update API artifact catalog, limits/errors, observability, tested-hardware matrix,
security/privacy notes, operator runbook and service datasheet. Record exact model
and dependency revisions already approved, but never secrets or private cache
paths. Any unsupported stage must be stated explicitly.

## Security/resource constraints

Live GPU1/service tests are authorized only within the measured Objective-003/004
profile and loopback boundary. Do not touch GPU0, firewall/VPN, unrelated
processes/services, system NVIDIA/CUDA or production data. Keep request content
memory/`/dev/shm` only and clean it. Metrics/logs must not become a persistence
backchannel.

## Deferred human adjudication

- Decision: `NONE`
- Exact resource limits, metrics set and safe artifact classifications are strategic
  engineering decisions driven by evidence.
- If parity exposes a material trust/privacy/security dilemma meeting all five
  strict conditions, strategic decides provisionally and replaces this section
  with exact `APPEND CRIT-NNNN` bytes before activation.

## GitHub publication and report

Create one objective-005 branch/PR. Push code/tests/docs/datasheet before the final
report-only SELF commit. Report the parity matrix, exact limits, test/load tables,
latency/VRAM/RSS/response sizes, metrics/log review, GPU isolation, E2E evidence,
known unsupported behavior and any critical-register action. Never merge.