# OAP Work Order — 003-a — Physical GPU1 runtime qualification

Objective `003-a`. Qualify the actual target host, physical GPU index 1 and the
real ZAP-IT model stack for safe local service use. Build a repo-owned runtime
environment, pin exact GPU identity and model/dependency revisions, audit
licenses/provenance, measure SAM2/CLIP/BLIP3 memory and latency separately and in
meaningful combinations, determine what can reliably run on the available GPU1,
and record an operator-safe resource strategy. Select but do not permanently
activate the loopback service port. Do not touch GPU0 or unrelated workloads.

## Prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `003 / 003-a`
- Mode: `CREATE_NEW_PR`
- Objective 002 merged on remote `main`: squash merge commit
  `b3bf252ca7a37c00a75276c4d5bac176316655e0` (PR #46); post-merge CI+CodeQL
  green — prerequisite satisfied.
- Verified current default branch and 40-hex base SHA: `main` @
  `b3bf252ca7a37c00a75276c4d5bac176316655e0` (verify live again immediately
  before branch creation).
- Required branch name: `oap/003-a-gpu1-runtime-qualification`
- Required PR title: `Objective 003-a: physical GPU1 runtime qualification,
  model revision/license pinning, measured resource strategy`
- Existing objective-003 PR: none expected (confirm none open at round start).

## Verified host state (strategic live inspection 2026-08-23; re-verify at execution)

- hostname/OS/kernel: `maelstrom1`, Ubuntu 24.04.4 LTS, kernel 6.8.0-138-generic.
- All GPUs (read-only snapshot): index 0 UUID
  `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI 00000000:00:08.0, NVIDIA
  GeForce RTX 2080 Ti, 11264 MiB total, 2161 MiB used — PROTECTED, never
  allocate/reset/kill; index 1 UUID
  `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI 00000000:00:0C.0, NVIDIA
  GeForce RTX 2080 Ti, 11264 MiB total, 6 MiB used, 10815 MiB free — TARGET.
- Compute processes: single unrelated process PID 66522 (/opt/venv/bin/python,
  2152 MiB) on GPU0 only. Protected evidence, never cleanup target.
- Driver/toolkit: NVIDIA driver 580.178.04; system nvcc 13.3 exists but is
  irrelevant to wheel runtimes (PyTorch wheels bundle their CUDA runtime;
  driver 580 runs cu12x wheels).
- Repo-supported Python/PyTorch/CUDA stack today: requires-python >=3.10,<3.13;
  system CPython 3.12.3; NO conda/mamba/micromamba anywhere; no usable PyTorch
  installed; environment.yml remains historical reference only and is NOT
  executable on this host as-is.
- Model identifiers/revisions/cache state: pipeline references
  facebook/sam2-hiera-large, openai/clip-vit-base-patch32,
  Salesforce/xgen-mm-phi3-mini-instruct-r-v1. User HF cache (~1.7 GB) holds
  only unrelated models — SAM2/CLIP/BLIP3 downloads ARE required this
  objective into the operator-controlled default cache; weights never committed.
- trust_remote_code: BLIP3/XGen-MM requires it; audit exactly what executes,
  pin the revision, keep client YAML powerless over it.
- Licenses/redistribution: SAM2 Apache-2.0; OpenAI CLIP checkpoint terms
  non-commercial research; XGen-MM (BLIP-3) research/non-commercial terms —
  already flagged in THIRD_PARTY_NOTICES.md (Objective 000). Local research
  use is authorized scope; commercial/redistribution stays behind the human
  release gate. Verify each claim against actual model cards during download;
  record exact URLs/revisions/hashes where provided.
- `/dev/shm`: tmpfs 27G total, empty, rw,nosuid,nodev — adequate for bounded
  request artifacts; weights do NOT go there.
- Host RAM/disk: 52 GB RAM (~46 available); ample disk in $HOME.
- Loopback listeners today: 127.0.0.1:32995 (unknown/unrelated), 631 cups,
  resolved DNS sockets; LAN listener 10.8.132.72:8000 unrelated workload.
  Candidate port preference for Objective 004 frozen below.

## GPU isolation law

Physical target is index 1. Every live ZAP-IT process in this objective must
inherit:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
```

Inside Python, the selected physical card must appear as logical cuda:0 with
visible device count exactly one in strict GPU mode; record and pin physical
UUID `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`. Never use logical cuda:1 after
masking. Never kill/reset/reconfigure/suspend/migrate or allocate on physical
GPU0. Never terminate another user's/process's GPU work. If GPU1 becomes
occupied by an unrelated process or cannot be safely used, mark that test
BLOCKED and wait/reschedule; do not steal memory.

## Frozen strategic decisions (binding unless live evidence forces documented deviation)

- Runtime environment mechanism: repo-owned pip venv (e.g. `.venv-gpu` under
  $HOME project space or repo .venv, gitignored) using CPython 3.12 with
  pinned PyTorch cu12x wheels (exact minor chosen by coding from what resolves
  coherently with SAM2/transformers on this driver); NO conda attempt, NO
  system package changes. Record every exact version in a committed
  requirements lock file plus docs/runtime.md table.
- Dependency drift resolution includes making `modules.visualizer`'s
  detectron2 import LAZY/optional so the GPU service runtime does not need a
  detectron2 build unless legacy visualization composites are explicitly used;
  behavior preserved when detectron2 is present. This is an allowed bounded
  refactor under Objective 001 seams law.
- Approved model set pinned by revision commit SHA at download time:
  facebook/sam2-hiera-large, openai/clip-vit-base-patch32,
  Salesforce/xgen-mm-phi3-mini-instruct-r-v1 (+its required processor deps).
  Any deviation to a smaller/different revision purely for VRAM fit must be
  surfaced as a scientific-model-identity question in the report and defaults
  to NOT changing the approved identity without strategic review next round.
- Conservative measurement progression with hard stop rule: before each stage
  load, compute predicted allocation vs current free VRAM; if predicted peak
  would exceed 90% of the 11264 MiB budget, skip that combination and record
  BLOCKED/rejected instead of attempting. Never intentionally OOM either GPU.
- Resource strategy selection is evidence-driven among: SAM2+CLIP resident
  with BLIP3 rejected per-request; staged BLIP3 load/unload; CPU placement of
  CLIP if latency acceptable; explicit profile restriction in readiness. The
  chosen strategy must be implemented as operator configuration honored by
  readiness/config validation, not request YAML.
- Loopback candidate port for Objective 004: prefer 127.0.0.1:17891; fallbacks
  23654 then any verified-unused port in 20000–40000 chosen live at activation
  time; record choice + verification method in docs; reserve nothing.
- Device guard must fail closed: strict GPU mode with wrong UUID/count refuses
  readiness and never silently falls back to CPU/GPU0; unit tests inject fake
  torch metadata for mismatch paths.

## Scope

1. Create/verify the repo-owned runtime environment per frozen mechanism; pin
   coherent exact versions (torch/torchvision, SAM2, transformers, accelerate,
   pillow/numpy already present); document import smoke of the full real stack.
2. Resolve dependency drift incl. the lazy detectron2 import and any SAM2/
   transformers compatibility patches (documented/tested/provenanced, minimal).
3. Pin model identities/revisions per frozen set; record source URLs, revision
   SHAs, license names, approximate cache sizes; verify downloads land in the
   operator cache; never commit weights.
4. Audit remote-code risk for XGen-MM: list executed files from the pinned
   revision, state why acceptable locally, ensure client YAML cannot alter it.
5. Qualify the device guard end-to-end: masked probe proves exactly one visible
   device whose UUID/name match expectations; fail-closed tests for wrong
   UUID/count and safe CPU-only behavior outside strict mode.
6. Measure stages separately on a small redistributable fixture image/config:
   load time, inference latency, baseline/peak/resident VRAM, host RAM where
   material, post-inference cleanup. Full GPU snapshots before/after each class.
7. Measure intended combinations conservatively (SAM2->CLIP always; add BLIP3
   only if individual evidence permits) under the hard stop rule.
8. Choose and implement the explicit resource strategy per evidence; encode it
   in operator config/readiness; document supported and unsupported profiles.
9. Bounded repeated-inference stability runs per supported profile (enough to
   expose obvious leaks/stale state; no production-soak claims). Record
   peak/end memory and confirm outputs stay deterministic-shaped.
10. Prove zero GPU0 impact with before/after all-GPU snapshots and compute-app
    listings around every live class.
11. Qualify /dev/shm/slaif-zap-it creation with mode-0700 dirs / 0600 files,
    capacity check, cleanup helpers, no residue.
12. Verify candidate loopback ports unused live (`ss`) and record the 004 port
    decision without binding anything persistently.
13. Add opt-in GPU integration tests (`gpu` marker, auto-skip honestly when
    prerequisites absent, serialized, forced mask env inside the test process)
    isolated from public CPU CI; CI stays CUDA-free.
14. Produce tested-hardware/runtime record: docs/runtime.md with host/GPU1/
    driver/Python/torch/model revisions, measured tables, supported/unsupported
    profiles, reproduction commands, rollback/cleanup notes. Remove stale
    22/24 GB implications wherever they remain.

## Non-goals

- no physical GPU0 use, even if idle;
- no system NVIDIA driver/CUDA upgrade/downgrade;
- no firewall/VPN/network exposure;
- no long-running API service/systemd activation yet;
- no arbitrary model replacement solely to make memory fit;
- no model training/fine-tuning;
- no public CI GPU runner;
- no claim that BLIP3/full profile is supported unless measured stable;
- no containerization/gateway/release work.

## Acceptance criteria

1. Live evidence records exact GPU1 UUID/PCI/name/VRAM and proves the mask
   exposes only physical GPU1 as logical cuda:0.
2. GPU0 untouched throughout; before/after snapshots prove no ZAP-IT process or
   allocation there.
3. Reproducible documented runtime environment imports the required real stack.
4. Approved model identities/revisions/licenses/provenance and remote-code
   audit documented honestly.
5. Each measured stage has load/inference/peak-VRAM evidence on GPU1 with
   redistributable fixtures.
6. Every claimed-supported profile has a measured resource strategy and
   repeated live success; unsafe combinations are rejected by configuration/
   readiness, never by spilling to GPU0 or blind crash.
7. Device guard fails closed on wrong UUID/visibility; no silent fallback.
8. /dev/shm root/permissions/capacity/cleanup verified.
9. Candidate loopback port selected/recorded for 004; nothing left listening.
10. Opt-in GPU tests isolated from normal CPU CI with explicit prerequisites.
11. CPU CI/CodeQL green; live GPU evidence reported separately, never
    misrepresented as GitHub-hosted CI.
12. Correct objective PR/report-only SELF contract satisfied; coding never merges.

## Required verification

- live nvidia-smi all-device snapshot before/after each test class — reported
- compute-process snapshot before/after — reported
- masked PyTorch device-count/name/UUID mapping probe output — reported
- Python/PyTorch/CUDA/library version capture — reported
- model revision/license/provenance audit table — reported
- SAM2-only / CLIP-only / BLIP3-only metrics or honest BLOCKED evidence — reported
- combined supported-profile metrics + repeated-run results — reported
- wrong-UUID/wrong-visibility fail-closed tests — PASSED
- /dev/shm permissions/capacity/cleanup checks — PASSED
- listener scan + candidate-port selection evidence — reported
- canonical CPU suite/Ruff/build/CI/CodeQL regression — PASSED (all six checks
  SUCCESS on head; suite green; gate held)
- read-only GPU before/after snapshot proving zero net allocation beyond
  declared measurement windows — reported

## Documentation and provenance

Update docs/runtime.md, THIRD_PARTY_NOTICES.md (exact revisions/licenses), 
docs/BASELINE.md pointers and README navigation. State supported/unsupported
profiles and the resource strategy. Do not publish credentials, private cache
paths or unrelated process details beyond sanitized isolation evidence.

## Security/resource constraints

First objective authorized to allocate physical GPU1, strictly limited to
bounded ZAP-IT tests and repo-owned runtime setup. No GPU0, no other-process
mutation, no driver/CUDA system changes, no firewall/VPN, no persistent
service. Model downloads only from approved sources/revisions into the
operator cache with recorded provenance; avoid unnecessary re-downloads.
Never print tokens/credentials.

## Deferred human adjudication

- Decision: `NONE`

Memory-fit strategy, supported-profile selection and exact pinned dependency
revisions are engineering decisions made from measurements under the frozen
conservative-progression law. A change that materially alters the service's
scientific model identity or trust boundary may qualify for CRITICAL.md only
if all five strict conditions hold; if live evidence forces e.g. replacing an
approved checkpoint outright, surface it in the report as a candidate instead
of deciding silently, and continue all unambiguous safe scope. Strategic
decides next round; coding may not invent the entry.

## GitHub publication and report

Create exactly one branch `oap/003-a-gpu1-runtime-qualification` from remote
`main` @ `b3bf252ca7a37c00a75276c4d5bac176316655e0` and exactly one PR titled
as specified. Keep live host changes bounded/reproducible; version operator-
facing configuration/docs needed for later activation (never secrets/weights).
Push implementation/evidence code before the final report-only SELF commit
(literal implementation SHA parent, single report path, `Report publication
commit: SELF`); exercise/fix in-scope CI; never merge. Send response OK only
after remote head/parent/bytes verification. Report exact GPU/process
snapshots, versions, model provenance, memory/latency tables, supported and
unsupported profiles, selected port, shared-memory results, skips/blockers,
rollback/cleanup actions and both GPU snapshots proving GPU0 untouched.
