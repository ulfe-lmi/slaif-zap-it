# OAP Work Order — 006-a — Package and qualify the 0.1.0 release candidate

## Objective

Create the sole Objective-006 PR and turn merged Objective 005 into an
**unpublished, mechanically reproducible ZAP-IT 0.1.0 release candidate** for
the proven native loopback/GPU1 deployment. Close packaging, installed-service,
version/provenance, distribution-content, license, supply-chain and operational
handoff gaps. Preserve the local academic goats regression while removing its
nonredistributable inputs from the tracked public tip and every build/release
artifact. Define, but do not implement or deploy, the separate SLAIF gateway
adapter contract.

This order prepares and qualifies development state. It does **not** authorize a
tag, GitHub/PyPI package or release, history rewrite, ref deletion, repository
visibility change, external/gateway deployment, LAN/public listener, production
data, or system service installation/enabling.

## Authoritative state and PR mode

- Repository: `ulfe-lmi/slaif-zap-it`; visibility is `PUBLIC`; default branch
  `main`.
- Numeric objective / round: `006 / 006-a`; mode `CREATE_NEW_PR`.
- Verified base: remote `main`
  `1758c3989454a000c71c2fc986db505bb70f3a5b`, the Objective-005 squash merge
  from PR #49.
- Post-merge GitHub runs on that exact SHA are SUCCESS: CI
  `32667342256` and CodeQL `32667342278` (static, Python 3.10/3.11/3.12,
  Analyze and CodeQL).
- Required branch: `oap/006-a-release-and-integration`.
- Required PR title: `Objective 006-a: package and qualify the 0.1.0 release candidate`.
- No Objective-006 branch or PR, open or closed, and no repository tags or
  GitHub releases existed at activation.
- Create exactly one new PR from the verified base. Coding never merges,
  tags, publishes, enables auto-merge, changes repository settings, or writes
  another repository.
- Current package metadata is `zap-it 0.1.0`; API model ID remains `zap-it-1`
  and schema remains `zap-it.v1`. There is no published version to supersede.
- Current `CRITICAL.md` contains no entries and has pre-append SHA-256
  `edacb6f86ebac60da896714c4c10ef65d1c0b8dc792441ac46ffa9680d3ba7ca`.
  This order requires exact append `CRIT-0001` below.

## Reconciled facts and frozen decisions

### 1. Fixture rights and public-history boundary

The human states the goats academic inputs are not redistributable. Current
public `main` tracks these exact paths:

```text
configs/goats.yaml
configs/goats2.yaml
demos/goats/goats1.jpg
demos/goats/goats2.jpg
```

The two JPEGs are each approximately 5.1 MB and have existed in public history
since the initial 2025 commit. The current wheel does not contain them, but the
public tree/history and GitHub source archives do. The safe provisional action is
to remove all four from the current tracked tip while leaving the authorized
local working copies intact and ignored. History/visibility remediation is
human-gated by `CRIT-0001`.

Use `git rm --cached` or an equivalent index-only operation. **Do not delete,
overwrite, resize, rename or move the host copies.** Add exact root-anchored
ignore rules for the two YAML paths and `demos/goats/`. No raw fixture/crop,
prompt, label, YAML mapping, response or derivative may enter a commit, build
artifact, CI artifact, log, report or OAP evidence.

### 2. Local academic E2E remains real and reusable

Add a clearly named opt-in local harness plus synthetic CPU tests for the
harness. It must:

- accept the four local ignored paths (with `goats2.yaml` as the academic config)
  only from operator/local defaults or explicit CLI arguments;
- refuse absent/non-file/symlink/out-of-root surprises safely and never search
  the network or Git history for fixtures;
- decode each image, crop exactly the middle 50% in both directions in memory
  (`left=width/4`, `top=height/4`, `right=3*width/4`,
  `bottom=3*height/4`, with deterministic integer handling), and never persist
  the crop;
- use `yaml.safe_load`, derive a new API-safe allowlisted mapping in memory, and
  strip/reject model/revision, BLIP3, panoptic, device, path/input/output/export,
  cache, URL/network, code/import and service-resource controls; never upload the
  raw legacy YAML;
- exercise A/B/A state isolation against the loopback service with L2 JSON, L3
  JSON and L3 ZIP, validate status/schema/artifact/RLE/identity/YOLO consistency,
  and bound request/response sizes and time;
- send a configured bearer token without printing it when auth is enabled;
- emit only sanitized aliases, original/crop dimensions, fixture/config digests,
  statuses, byte counts, latencies and object/artifact counts; no semantic labels,
  prompts, raw bodies or content snippets.

CPU CI uses generated images and YAML only. The real goat run is explicit,
GPU1-only and local. For the release-candidate rehearsal in this order, run one
real A/B/A sequence after building/installing the candidate; the verified local
files exist now, so absence is `BLOCKED`, not a fabricated pass.

### 3. Release/version policy

- Retain `0.1.0` as the intended first MVP version; call it an **unpublished
  0.1.0 release candidate**, not a released package.
- Establish one importable version source using installed distribution metadata
  with an honest source-tree fallback. Expose the package version in safe service
  provenance and test the source/editable/wheel-installed cases. Do not change
  `zap-it-1` or `zap-it.v1`.
- Add an accurate changelog and release-candidate notes covering Objectives
  000–006, limitations, tested hardware/profile and blocked capabilities. Do not
  claim accuracy, SLA, production, BLIP3, geometry, panoptic, gateway or public
  readiness.
- No `v0.1.0` tag, GitHub release, PyPI upload, publication workflow execution or
  release-secret creation in this objective.

### 4. Distribution contents and clean install

Make wheel and sdist first-class, auditable artifacts:

- use an explicit source-distribution allowlist/manifest; include package code,
  supported legacy entrypoints/config examples, tests using generated or
  redistributable fixtures, operator templates, license/notices and required
  public docs;
- exclude all `demos/`, `last_results/`, local academic inputs, generated
  outputs, caches, model weights, image/video payloads, OAP transcript material,
  credentials, private environment files and host runtime files from wheel and
  sdist;
- add a deterministic artifact verifier that inspects both archive member names
  and extracted content safely: reject absolute/traversal/symlink surprises,
  goat paths/basenames, images/video/model/cache extensions, secrets/private env,
  outputs and unexpected large members; verify required metadata, LICENSE,
  notices, console entrypoint and package modules;
- build wheel and sdist from a clean source snapshot twice with a fixed
  `SOURCE_DATE_EPOCH`; compare member path sets and member-byte digests, record
  raw artifact SHA-256/size, and explain any harmless container-metadata-only
  difference rather than claiming bit reproducibility falsely;
- run metadata validation (`twine check` or equivalent), wheel-content checks,
  and isolated Python 3.10, 3.11 and 3.12 installs from the built wheel; at least
  one isolated install must start the installed service/application against the
  fake engine or equivalent no-GPU package smoke without importing from the
  checkout;
- prove an sdist-built wheel has the same accepted member/content contract and
  passes the installed smoke.

Pin the build backend to the qualified compatible version rather than an
unbounded `setuptools>=68`. Keep lightweight dependency ranges bounded and keep
the exact CPython-3.12/cu124 GPU lock and pinned SAM2 revision as the target-host
authority. Do not add model downloads to builds or CPU CI.

### 5. Native operator packaging; no container

The native host path is proven. Docker 29.1.3 is present, but no ZAP-IT container
or GPU-passthrough/cache/UID/`/dev/shm` qualification exists. **Do not add a
Dockerfile or Compose file.** Record this as an intentional non-inclusion, not a
missing test.

Add an installed foreground console entrypoint for the service. Preserve the
repo-owned `scripts/serve_local.sh` operator path. Refine the shipped, uninstalled
user-systemd template to a `Type=simple` foreground service using a private
operator `EnvironmentFile`, an explicit installed-venv executable placeholder,
loopback-only settings and normal systemd stop semantics. Validate the unit
syntax, but do not copy it to the user's systemd tree, run `daemon-reload`, enable
or start it. Document clean install, config permissions, start/health/readiness,
upgrade, rollback and uninstall for both the installed foreground/systemd
candidate and the repo launcher.

### 6. SLAIF gateway contract; no cross-repository implementation

Live reconciliation of public `ulfe-lmi/slaif-api-gateway` main
`ce0cf95685796477685a3aab6edacb39def6c27b` found a native-module foundation but
only the separately reviewed `facial_scoring` adapter. It has no ZAP-IT multipart
module adapter. This PR must not modify that repository or pretend integration is
complete.

Add a precise integration contract/follow-up document for a future separate
gateway PR:

- a fixed native module ID/model mapping to backend `zap-it-1` and multipart
  `POST /v1/completions`;
- exactly one bounded supported image plus one bounded API-safe YAML/config,
  `verbosity=0..3`, `response_format=json|zip`, `stream=false`, no URLs/file IDs,
  no arbitrary backend/model/path/device fields, no retries and bounded timeout;
- client gateway authorization terminates at the gateway. A distinct
  gateway-owned high-entropy backend bearer secret is sent through
  `SLAIF_ZAP_IT_API_KEY`; client Authorization is never forwarded, persisted or
  logged;
- co-located loopback backend only for the first integration. Any cross-host,
  LAN, TLS/mTLS or public topology is separate architecture and authorization;
- response/error mapping, zero token usage/non-token accounting, privacy/logging,
  artifact-size handling and exact tests required in that future gateway repo.

No gateway E2E is required here because no gateway adapter is implemented.
Report it `NOT INCLUDED — separate repository/order required`, not PASS/SKIP.

### 7. License, provenance and supply chain

- Update `THIRD_PARTY_NOTICES.md`, installation/runtime docs and a release gate
  inventory so Python/service/GPU dependencies, SAM2 code, model identities,
  revisions, licenses and download-at-operator-runtime behavior are explicit.
- Model weights are never packaged. BLIP3/XGen-MM remains unsupported on this
  host and CC-BY-NC/research-only; CLIP's pinned card lacks an SPDX deployment
  license. State that commercial/deployed model use and any weight redistribution
  require human/legal clearance; do not convert model qualification into license
  approval.
- Inventory every tracked media path against documented rights. For the release
  artifacts, use an allowlist and exclude all demo/media payloads, even if a
  subset has attribution. Do not infer rights for unlisted media. Report remaining
  public-repository media/history facts as release gates, without creating another
  CRITICAL entry for the same fixture issue.
- Pin GitHub Actions to these live-resolved immutable commits with version
  comments: `actions/checkout@11d5960a326750d5838078e36cf38b85af677262`
  (v4), `actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065`
  (v5), and
  `github/codeql-action@6d786de4d6f3531a740e445b53a42b622bbbace8`
  (v3).
- Preserve least `contents: read`; CodeQL alone may retain
  `security-events: write`. Do not add write/package/release permissions or
  secrets.
- Add pinned `detect-secrets==1.5.0` (or a repository-local equivalently pinned
  invocation of that exact version), a reviewed baseline with only explained
  false positives, and CI verification of the tracked current tree and unpacked
  artifacts. No raw secret candidate may be copied into reports.
- Add Dependabot configuration for `pip` and `github-actions` version updates.
  Repository settings currently have Dependabot security updates and secret
  scanning disabled and `main` is unprotected; document these as exact human
  repository-setting recommendations, but do not mutate settings or call them
  enabled.

### 8. Release-candidate deployment rehearsal

After CPU/package verification and before report publication:

1. Re-verify all GPU index/UUID/PCI/name/VRAM/process state, driver/CUDA/Torch,
   `/dev/shm`, service processes and a free loopback port. Physical GPU1 must
   still be UUID `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8` or the run blocks.
2. Preserve GPU0 and unrelated PID 66522 (or its live successor if the process
   changed); never allocate, signal or inspect private content from it.
3. Use the installed release-candidate code with
   `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`, one
   worker/request and the already qualified operator model cache. Bind only a
   freshly verified unused `127.0.0.1` port.
4. Enable a fresh process-local bearer key supplied outside Git/logs and prove
   missing/wrong 401 and correct 200 for completions and metrics.
5. Run synthetic L0–L3 JSON plus L3 ZIP, health/readiness, unsupported BLIP3,
   repeated restart and graceful rollback. Then run the local goat A/B/A harness
   required above using central crops only.
6. Record sanitized latency/size/count/resource facts, process/GPU snapshots and
   candidate artifact SHA-256s. Do not record fixture content.
7. Stop the service. Require port free, GPU1 idle/no ZAP-IT process,
   `/dev/shm/slaif-zap-it` empty, GPU0 unchanged and no systemd/container/network
   mutation.

## Expected implementation scope

Expected paths include packaging metadata/manifest, version/provenance code,
installed entrypoint, release/artifact and local-goat harness scripts, generated-
fixture tests, CI/Dependabot/security scan configuration, native unit template,
changelog/release/gateway/license/install/runbook/datasheet docs, exact
`CRITICAL.md` append, and exact OAP transcript. Removing the four tracked goat
paths is required. Preserve legacy CLI behavior and existing supported API/model
behavior.

If a necessary change falls outside this bounded outcome, report it rather than
silently expanding into another repository, public deployment, history rewrite
or final release.

## Non-goals

- no final tag, release, PyPI/GitHub package publication or signing-key creation;
- no history rewrite, force push, ref deletion or repository visibility/settings
  mutation;
- no Docker/Compose artifact;
- no SLAIF gateway code or live gateway integration;
- no LAN/public bind, firewall/VPN/TLS/reverse-proxy mutation or production data;
- no installation/enabling of systemd units;
- no GPU0 use, multi-worker/multi-GPU scaling or persistent request store;
- no BLIP3, geometry or panoptic activation and no scientific model change;
- no raw/local academic fixture or derivative committed or packaged;
- no release-rights claim for models, historical fixtures or unverified media.

## Acceptance criteria

1. The branch/PR is uniquely correct and based on literal remote-main SHA
   `1758c3989454a000c71c2fc986db505bb70f3a5b`.
2. `CRIT-0001` is appended byte-for-byte with the helper; every prior
   `CRITICAL.md` byte remains identical and the entry is the sole terminal
   append.
3. The four goat paths are absent from the tracked PR tip and present only as
   ignored local files on the host; no raw/derived bytes entered Git/OAP/builds.
4. The opt-in harness proves central-50% in-memory crop, safe config derivation,
   A/B/A L2/L3 JSON/ZIP behavior and zero persistence; CPU tests use generated
   fixtures only.
5. Wheel and sdist pass strict content/secret/metadata/license checks, isolated
   installs and installed no-checkout smoke across supported Python versions;
   artifact hashes/member digests are recorded.
6. Version/provenance/changelog/release notes consistently describe unpublished
   0.1.0 and do not overclaim capability or authorization.
7. The foreground installed service and uninstalled Type=simple unit template
   are documented/validated; legacy launcher remains supported; no host systemd
   state changed.
8. Gateway contract is exact and honest, backend bearer boundary is least
   privilege, and actual gateway work is clearly separate/not implemented.
9. License/model/media inventory distinguishes code/package distribution from
   operator-downloaded weights and unresolved deployment/commercial rights.
10. Actions are immutable-pinned, workflow permissions remain least privilege,
    detect-secrets/artifact scans and Dependabot config are present, and no
    release credential/permission is added.
11. Full canonical CPU suite/coverage, Ruff format/lint, compile, shell syntax,
    artifact builds/checks/isolated installs and all six GitHub checks are green
    on implementation and report heads.
12. Installed-candidate GPU1 rehearsal, bearer auth, synthetic levels/formats,
    local goat A/B/A, restart/rollback and cleanup pass with GPU0 untouched.
13. Final report contains a release-gate table with `CRIT-0001` OPEN/BLOCKING,
    public history not remediated, repository security settings still factual,
    gateway/container NOT INCLUDED, and exact human actions before release.
14. No prohibited external, destructive, deployment or release boundary is
    crossed; coding never merges.

## Required verification and evidence

- Git/GitHub base, unique PR, commits/files, no tags/releases and repository
  visibility/settings checked independently before mutation and before report.
- Pre-append `CRITICAL.md` SHA-256, exact helper command, dry run, appended diff
  and proof the preimage is an exact prefix.
- `git ls-files`/archive-member scans proving all four goat paths absent;
  ignored-file checks proving local copies remain without printing content.
- Central-crop unit/property tests, hostile config stripping tests, no-write tests
  and required local real A/B/A run.
- Full CPU suite with coverage at or above the existing 64% gate; no GPU/model
  downloads in CI.
- Ruff format/check, compileall, `bash -n`, unit-file validation and docs/link
  checks where available.
- Two clean wheel+sdist builds, metadata/wheel checks, safe archive inspection,
  member digest comparison, SHA-256/size, unpacked detect-secrets scan and
  isolated install/smoke on Python 3.10–3.12.
- Full changed-tree/commit secret and large-file scan; no generated artifact,
  cache, weight, raw response, host env or local fixture staged.
- Fresh live host/GPU facts and installed-candidate synthetic/auth/goat/restart/
  rollback evidence as specified; final stopped/empty/idle state.
- All required GitHub CI and CodeQL checks SUCCESS on implementation head and
  final report-only head; no missing/pending/skipped check called pass.

## Deferred human adjudication — APPEND CRIT-0001

- Decision: APPEND CRIT-0001

All five register conditions are satisfied. Human instructions resolve that the
files are not redistributable, but not which externally disruptive remedy is
authorized for already-public history. The provisional tip/artifact exclusion is
safe and reversible; the competing history/visibility actions materially affect
public exposure and shared Git integrity and may be rejected by the owner.

Coding must place the following exact bytes in a temporary source file and run:

```bash
.venv/bin/python oap/bin/append_critical.py \
  --repo-root . --source <exact-source-file> --id CRIT-0001
```

Exact entry bytes:

```markdown
## CRIT-0001 — Remediation of nonredistributable goat fixtures already present in public history
- Status: OPEN — HUMAN ADJUDICATION REQUIRED
- Introduced: 2026-08-23
- OAP objective/round: 006 / 006-a
- Pull request: N/A at decision time; appended by the Objective 006-a PR
- Domain: release / public exposure
- Priority: P1
- Human adjudication required before: any final tag, package/source release, claim that the public repository is rights-cleared, destructive history rewrite, or repository-visibility change
- Threshold attestation: ALL FIVE CRITICAL-ENTRY CONDITIONS SATISFIED

### Dilemma

The human has stated that `demos/goats/goats1.jpg`,
`demos/goats/goats2.jpg`, and the associated goat YAML test material are not
redistributable and may be used only for local academic testing. Live
reconciliation found four applicable tracked paths — those two images plus
`configs/goats.yaml` and `configs/goats2.yaml` — in a public GitHub repository,
including commits dating to 2025. Removing them from the current tip and release
artifacts is safe and necessary, but it does not remove historical blobs, forks,
clones, caches, or existing commit references. A complete Git history purge or
repository-visibility change would be externally disruptive and cannot guarantee
recall of copies already obtained.

### Provisional decision

Objective 006-a will remove all four paths from the tracked current tip while
preserving the authorized local host copies as ignored, operator-supplied test
inputs. It will add an opt-in local E2E harness that crops the central 50% in
both dimensions in memory, derives only an API-safe allowlisted mapping from
`goats2.yaml`, writes no derivative, and emits only sanitized aggregate evidence.
Wheel, sdist, CI artifacts, release manifests, documentation examples, and future
source-release inputs will have explicit deny checks for the four paths and for
unlicensed local fixtures generally. No agent will rewrite history, force-push,
delete refs, change repository visibility, publish a tag/package/release, or
claim historical remediation.

### Why this decision

It immediately stops carrying the known files in the current distributable tip,
preserves the academic regression value the human requested, is reviewable in an
ordinary PR, and avoids an irreversible shared-history operation without owner
authorization. It also creates mechanical artifact checks so a future package
cannot silently reintroduce the files. Development can continue to a bounded
unpublished release candidate while final release remains blocked.

### Strongest case that this decision is wrong

Tip deletion may create a misleading sense that the exposure is fixed while the
same bytes remain downloadable from public history. The correct immediate action
could instead be to make the repository private and perform a coordinated purge
of every affected ref before any further public work. Conversely, a history
rewrite may be disproportionate because it breaks the commit SHAs used by OAP,
invalidates downstream clones, cannot revoke existing copies, and may erase
useful provenance without delivering actual recall. A competent repository owner
with the relevant rights and downstream-consumer facts could reasonably choose a
different remedy.

### Alternatives considered

1. Leave the files tracked but exclude only wheel/sdist members — rejected
   provisionally because GitHub source archives and the current public tree would
   still redistribute known prohibited material.
2. Delete from the current tip and preserve ignored local copies — selected as
   the least-destructive immediate mitigation.
3. Rewrite all Git history and force-update refs — deferred because it is
   destructive, externally coordinated, and cannot guarantee recall.
4. Make the repository private temporarily or permanently — deferred because it
   changes an external access boundary and may affect unknown users/integrations.
5. Keep/re-add the files after obtaining written rights clearance — allowed only
   after explicit human adjudication establishing that clearance.

### Assumptions

- The human statement applies to both goat images and both tracked goat YAML
  files.
- Local academic use remains authorized; redistribution is not.
- No GitHub releases or tags currently exist, but the public repository/history
  itself has already exposed the tracked blobs.
- The strategic/coding agents do not possess legal authority or complete facts
  about forks, clones, third-party caches, or repository consumers.
- Other media will be separately inventoried for release packaging; absence of
  evidence is not treated as rights clearance.

### Failure mode and blast radius

Without mitigation, current-tree clones, GitHub source archives, and a future
tag could continue distributing the four files. With only the provisional
mitigation, historical commits remain accessible and an uninformed maintainer
could restore the ignored paths. A destructive purge could break every recorded
commit identifier, open downstream work, OAP evidence links, forks, and local
clones while still failing to revoke prior downloads.

### Mitigations and evidence

- Remove the four paths from tracking without deleting the authorized local
  working copies; add exact root-anchored ignore rules and explanatory rights
  documentation.
- Make release verification reject the paths, goat fixture basenames, image/model
  payload extensions outside an explicit allowlist, caches, outputs, secrets,
  and traversal/absolute archive members in both wheel and sdist.
- Exercise the local-only fixtures only through an explicit opt-in harness with
  in-memory central-50% crops and sanitized A/B/A aggregate results.
- Keep synthetic redistributable fixtures as the only CI/live evidence embedded
  in the repository or artifacts.
- Record that history is not remediated and block final release/clearance claims
  pending human disposition.

### Reversibility and rollback

The tip deletion, ignore rules, harness, and artifact checks are ordinary Git
changes and can be reverted after documented rights clearance. The local fixture
copies are preserved and are not rewritten. No history rewrite, ref deletion,
visibility mutation, tag, package upload, or release is performed autonomously.

### Exact question for the human adjudicator

Given that the four nonredistributable goat fixture files have already existed in
the public repository history, which authoritative remedy do you approve: (A)
keep the repository public, accept that historical copies cannot be recalled,
and retain the current-tip/artifact exclusion with an explicit incident record;
(B) temporarily or permanently make the repository private; (C) authorize a
coordinated history purge/force-update with downstream notification; or (D)
provide documented redistribution clearance? Please also identify any affected
forks/releases/caches or organizational/legal notification requirement known to
you.

### Autonomous follow-up allowed before adjudication

Agents may implement and merge the current-tip removal, local-only cropped E2E
harness, artifact deny checks, rights/provenance documentation, packaging,
loopback release rehearsal, and an unpublished 0.1.0 release candidate. Agents
may not rewrite public history, force-push/delete refs, change visibility, publish
a final tag/package/release, deploy externally, or state that the historical
exposure or rights issue is closed.
```

The implementation commit must contain the append. Before commit, prove the
original register bytes are an exact prefix and the appended normalized entry is
byte-identical to the block above. Agents never append a human disposition.

## Publication and coding response

- Commit all implementation, required removals, exact active/order transcript and
  `CRIT-0001` append before the report. Push and create the unique PR.
- Record literal implementation SHA. Then create exactly
  `oap/reports/006-a-report.md`; final SELF commit changes only that report and
  has the implementation SHA as first parent.
- Report every criterion and exact `PASSED|FAILED|SKIPPED|NOT RUN|BLOCKED|PENDING|MISSING`
  state, artifact hashes/member scans, local-only fixture handling, critical
  prefix proof, PR/CI topology, host/GPU/service cleanup and the exact human gate
  table.
- Verify remote report bytes/parent/one-path commit and all six report-head checks
  SUCCESS. Then send exact response FIFO `OK` and exit. `OK` is synchronization,
  never acceptance. Coding never merges or starts another objective.
