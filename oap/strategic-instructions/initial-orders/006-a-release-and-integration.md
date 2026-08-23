# OAP Work Order — 006-a — Packaging, SLAIF integration and release readiness

> DRAFT UNTIL Objective 005 is merged, the local release-candidate evidence is complete, and strategic live/GitHub reconciliation is complete. Do not publish as-is.
>
> **HWP status:** preloaded human engineering intent. This objective prepares integration/release artifacts; it does not grant authority to cross an unresolved human deployment/release gate in `CRITICAL.md`.

## Objective

Convert the proven loopback release candidate into a professionally distributable
and integrable SLAIF service package. Finalize versioning/release metadata,
operator installation and optional container/systemd packaging, define the SLAIF
gateway integration contract and authentication policy, complete distribution/
license/supply-chain review, and produce a release candidate whose deployment is
mechanically reproducible. Actual external deployment or final release across an
applicable open `CRITICAL.md` gate remains human-controlled.

## Prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `006 / 006-a`
- Mode: `CREATE_NEW_PR`
- Objective 005 merged on remote `main`, merge SHA/checks: VERIFY:
- Verified default branch/base SHA: VERIFY:
- Required branch/PR title: VERIFY:
- Existing objective-006 PR: N/A after strategic confirms none: VERIFY:
- Current release-candidate service version/profile/port policy: VERIFY:
- Current `CRITICAL.md` open entries and applicable gates: VERIFY:

Before activation, strategic must inspect `CRITICAL.md` completely and distinguish
entries that merely require later human adjudication from any gate that blocks
this objective's actual external release/deployment action.

## Verified current state

- package/build metadata and current versioning strategy: VERIFY:
- tested native installation/launch/runbook from 004/005: VERIFY:
- whether a user-level systemd unit already exists and is preferred: VERIFY:
- whether Docker/Compose adds real deployment value on the target GPU stack: VERIFY:
- current SLAIF gateway expectations/repository contract if integration is in scope: VERIFY:
- loopback auth policy and requirements before gateway/LAN exposure: VERIFY:
- complete dependency/model/license/third-party inventory: VERIFY:
- CI/CodeQL/coverage/security scan status: VERIFY:
- release/tag/package publication permissions and human policy: VERIFY:
- every open deferred-human-adjudication gate: VERIFY:

## Scope

1. **Versioning and release metadata.** Establish/confirm semantic versioning or a
   clearly documented project version scheme; make package/service version
   available in metadata and safe API provenance. Produce changelog/release notes
   from the modernization objectives without overstating scientific capability.
2. **Clean installation path.** Prove a clean supported host can install the
   package/service dependencies using documented commands. Separate CPU/dev,
   service and GPU/model dependencies where practical; preserve a reproducible
   path for the tested target host.
3. **Operator packaging.** Finalize native launcher/user-systemd packaging if it
   is the proven operational path. Add Dockerfile/Compose only if GPU passthrough,
   model cache, `/dev/shm`, UID/permissions and one-worker behavior can be tested
   honestly; do not create an untested decorative container artifact.
4. **Configuration/secrets packaging.** Provide templates/examples for operator
   settings with no secrets committed. Document restrictive permissions and how
   expected GPU UUID, port, model cache/revisions, API key and limits are supplied.
5. **Gateway integration contract.** If SLAIF gateway integration is currently
   available/desired, define the route/service registration needed to forward a
   one-image+YAML request to ZAP-IT without weakening the API's cardinality,
   limits or auth. Keep the ZAP-IT backend on loopback/private topology unless a
   separate architecture explicitly requires otherwise.
6. **Authentication boundary.** Before any non-loopback exposure, require a real
   authenticated caller boundary. Decide whether gateway-only trust, a backend API
   key, mTLS or another existing SLAIF mechanism is the least-complex adequate
   control. Test the chosen mechanism and never log tokens. An auth relaxation
   that materially changes the trust boundary may require deferred human
   adjudication if all five conditions hold.
7. **Network/deployment documentation.** Clearly distinguish proven loopback MVP,
   optional gateway/private deployment and unimplemented public-internet concerns
   such as TLS/reverse proxy/rate limiting. Do not silently bind `0.0.0.0`.
8. **License/distribution review.** Reconcile Python dependencies, model code,
   model weights/revisions and optional remote-code components with LICENSE,
   `THIRD_PARTY_NOTICES.md` and distribution plans. Do not redistribute weights or
   code where terms do not allow it. Record download-at-install/runtime behavior.
9. **Supply-chain hardening.** Pin/lock meaningful dependencies/revisions, use
   least-privilege GitHub Actions, artifact/package provenance where available,
   dependency/security scanning, and no long-lived release secret if trusted
   publishing is available. Exact publishing mechanism is strategic and must be
   tested before release.
10. **Build artifacts.** Produce and inspect source/wheel artifacts; test install
    from built artifact in an isolated environment. Ensure no model cache, test
    output, credentials, CRITICAL private notes beyond intended public register,
    generated request data or host-specific runtime file is packaged accidentally.
11. **Release-candidate deployment rehearsal.** From documented artifacts/config,
    recreate the local/private service, verify GPU1 isolation, health/readiness,
    representative L0–L3 calls, auth/gateway path if enabled, cleanup and rollback.
12. **Operational handoff.** Finalize README/runbook/API/service datasheet,
    installation, upgrade/rollback, backup/non-persistence expectations,
    monitoring, known limitations and incident/security reporting.
13. **Human gate audit.** Generate an explicit release-gate table from
    `CRITICAL.md`: entry ID, latest human disposition, stated gate, applicability to
    planned deployment/release and blocking status. Agents may not append human
    `ACCEPTED` decisions themselves.
14. **Release preparation, not unauthorized release.** It is acceptable to build a
    release candidate, draft release notes, create a PR and prepare commands while
    a human gate remains open. Do not publish a final release/tag/package or make
    an external deployment if the constitution or an applicable critical entry
    reserves that boundary for human authority.
15. **If all human gates are already cleared and release authority is explicitly
    delegated by current human/project policy**, strategic may execute the
    documented release step only if the active order is finalized to authorize it
    and all checks are green. Otherwise stop at a fully reproducible release
    candidate and report the exact human action required.

## Non-goals

- no public internet deployment by default;
- no multi-tenant billing/quota/user-management platform;
- no multi-GPU/multi-worker scaling;
- no scientific model training/fine-tuning;
- no persistent result store or job queue;
- no bypass of open `CRITICAL.md` deployment/release gates;
- no redistribution of model weights without verified permission;
- no untested Docker/container claim;
- no weakening of request validation/resource limits for gateway convenience.

## Acceptance criteria

1. Package/service versioning, build metadata, changelog and release notes are
   coherent and accurately describe capabilities/limitations.
2. A clean isolated install from built distribution artifacts succeeds using the
   documented supported path and passes package/API smoke tests.
3. Native operator packaging is reproducible; any container artifact included is
   actually exercised with correct GPU1, `/dev/shm`, cache and one-worker behavior.
4. Operator configuration/secrets are externalized with safe templates and no
   credential/runtime-host leakage in Git or package artifacts.
5. Any gateway integration preserves exactly-one-image/config semantics, limits,
   error handling and an authenticated trust boundary.
6. License/third-party/model provenance is complete enough to determine what is
   and is not redistributed.
7. Release/build workflows are least-privilege and pass current security/CodeQL/
   dependency checks without hiding warnings.
8. A release-candidate deployment rehearsal from documented artifacts succeeds
   and proves GPU1-only inference, health/readiness, auth path, cleanup and rollback.
9. Artifact inspection shows no weights/cache/request data/secrets/unwanted host
   files.
10. The service datasheet/runbook/install/upgrade/rollback/security docs are
    consistent with actual tested behavior.
11. A complete human gate audit identifies every applicable `CRITICAL.md` entry
    and does not falsely treat autonomous mitigation as human acceptance.
12. No final external deployment/release is crossed while an applicable human gate
    remains unresolved.
13. CPU CI/coverage/CodeQL and required live/private rehearsal evidence are green
    or honestly reported BLOCKED.
14. Correct one-PR/report-only SELF contract is satisfied; coding never merges.

## Required verification

- predecessor remote-main/CI/release-candidate state: VERIFY:
- open `CRITICAL.md` entries/latest human dispositions/gate table: VERIFY:
- clean build and isolated wheel/sdist install: VERIFY:
- package-content inspection/secret/large-artifact scan: VERIFY:
- native service install/start/stop/rollback rehearsal: VERIFY:
- container/Compose test if included, otherwise explicit NOT INCLUDED rationale: VERIFY:
- gateway/auth integration E2E if included: VERIFY:
- GPU1 UUID/process isolation and GPU0 no-allocation: VERIFY:
- API representative L0–L3/JSON/ZIP smoke: VERIFY:
- license/third-party/model provenance audit: VERIFY:
- GitHub Actions/CodeQL/dependency/supply-chain checks: VERIFY:
- release artifact hashes/provenance: VERIFY:
- exact remaining human action before release/deployment, if any: VERIFY:

## Documentation and provenance

Finalize public/operator documentation around installation, service operation,
API, auth/topology, models/licenses, tested hardware, resource limits, privacy,
non-persistence, observability, upgrade/rollback and release process. Keep the
Deferred Human Adjudication Register visible and honest; do not rewrite history to
make the release appear cleaner.

## Security/resource constraints

Any live rehearsal remains within the proven GPU1 and private/loopback topology.
Do not touch GPU0, unrelated processes/services, system driver/CUDA or production
customer data. Network/gateway mutation must be explicitly in the finalized order
and reversible. Credentials remain outside Git and logs. Human-exclusive release/
deployment authority and every applicable critical gate remain binding.

## Deferred human adjudication

- Decision: `NONE`
- Distribution mechanism, native-vs-container packaging and gateway plumbing are
  ordinary strategic decisions unless they expose a genuinely material unresolved
  trust/security/license dilemma satisfying all five strict conditions.
- Before activation strategic must inspect existing entries; do not duplicate an
  already-registered dilemma. If a new qualifying issue exists, decide
  provisionally and replace this section with exact `APPEND CRIT-NNNN` bytes.

## GitHub publication and report

Create one objective-006 branch/PR from verified remote main. Push package,
workflows, docs, integration code and release-candidate evidence before the final
report-only SELF commit. Report build/install artifacts, hashes, deployment
rehearsal, auth/gateway behavior, license/supply-chain findings, GPU isolation,
CI/security checks and the exact human-gate table. Coding never merges and never
publishes the final release. Strategic may merge development only under the OAP
merge law; final deployment/release remains subject to the explicit human gates.