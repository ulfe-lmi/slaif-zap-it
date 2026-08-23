# OAP Work Order — 003-a — Physical GPU1 runtime qualification

> DRAFT UNTIL Objective 002 is merged and strategic live/GitHub reconciliation is complete. Do not publish as-is.
>
> **HWP status:** this draft is preloaded human engineering intent. Strategic owns exact environment/model decisions after live evidence, but must preserve physical-GPU1 isolation, GPU0 protection, bounded experiments and honest capability conclusions.

## Objective

Qualify the actual target host, physical GPU index 1 and the real ZAP-IT model
stack for safe local service use. Build or verify a repo-owned runtime environment,
pin exact GPU identity and model/dependency revisions, audit licenses/provenance,
measure SAM2/CLIP/BLIP3 memory and latency separately and in meaningful
combinations, determine what can reliably run on the available GPU1, and record an
operator-safe resource strategy. Select but do not yet permanently activate the
loopback service port. Do not touch GPU0 or unrelated workloads.

## Prerequisite and GitHub state

- Repository: `ulfe-lmi/slaif-zap-it`
- Numeric objective / round: `003 / 003-a`
- Mode: `CREATE_NEW_PR`
- Objective 002 merged on remote `main`, merge SHA and checks: VERIFY:
- Verified current default branch and 40-hex base SHA: VERIFY:
- Required branch name / PR title: VERIFY:
- Existing objective-003 PR: N/A after strategic confirms none: VERIFY:

## Known host evidence to re-verify

On 2026-08-23 the human/operator preflight observed:

```text
GPU 0: NVIDIA GeForce RTX 2080 Ti
       UUID GPU-4c129e25-8e59-eee4-b49c-56c40e294182
       11264 MiB total, ~2161 MiB in use by unrelated Python process

GPU 1: NVIDIA GeForce RTX 2080 Ti
       UUID GPU-c457dbaf-991c-dc23-c781-0dc030776dd8
       11264 MiB total, ~6 MiB in use

/dev/shm: 27G total, effectively empty
```

This corrects the earlier 22/24 GB hypothesis in bootstrap documentation. These
facts are **not timeless**: strategic must re-run read-only GPU/process/driver/
shared-memory/listener inspection immediately before activation and record exact
current evidence. GPU0 remains protected even if it later appears idle.

## Verified current state

Replace with exact live evidence:

- hostname/OS/kernel: VERIFY:
- all GPU indices, UUIDs, PCI buses, names, total/used memory: VERIFY:
- all GPU compute/graphics processes and ownership where visible: VERIFY:
- NVIDIA driver and exposed CUDA version: VERIFY:
- repo-supported Python/PyTorch/CUDA stack after Objectives 000–002: VERIFY:
- current model identifiers/revisions/cache state for SAM2, CLIP and BLIP3: VERIFY:
- `trust_remote_code` use and compatibility patches: VERIFY:
- exact model licenses/redistribution constraints and source URLs: VERIFY:
- `/dev/shm` capacity/mount/options/permissions: VERIFY:
- candidate unused loopback ports and existing listeners: VERIFY:
- available host RAM/swap and disk/cache constraints: VERIFY:

## GPU isolation law

Physical target is index **1**. Every live ZAP-IT process in this objective must
inherit:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
```

Inside Python, the selected physical card must appear as logical `cuda:0` and the
visible device count must be exactly one in strict GPU mode. Record and later pin
the physical UUID. Never use logical `cuda:1` after masking.

Never kill, reset, reconfigure, suspend, migrate, or allocate on physical GPU0.
Never terminate another user's/process's GPU work. If GPU1 becomes occupied by an
unrelated process or cannot be safely used, mark the live test BLOCKED and wait or
reschedule; do not steal memory.

## Scope

1. **Create/verify a repo-owned runtime environment.** Use the project's supported
   environment mechanism under the user's/project space. Pin a coherent Python,
   PyTorch, CUDA-runtime-package and model-library set. Do not replace the system
   NVIDIA driver/CUDA installation merely for convenience.
2. **Resolve dependency drift.** Establish exact versions for SAM2, torchvision,
   OpenCLIP/CLIP dependencies, Transformers and BLIP3/XGen-MM dependencies that
   work together on this host. Minimize ad-hoc monkeypatches; where upstream
   compatibility patches are unavoidable, document/test/provenance them.
3. **Pin model identities/revisions.** Operator configuration, not request YAML,
   owns approved model IDs/revisions. Record source, revision/commit where
   possible, license, approximate cache size and checksum/provenance mechanism.
   Never commit weights.
4. **Audit remote-code risk.** BLIP3 or any model requiring `trust_remote_code`
   must be pinned and reviewed. Document exactly what code executes and why it is
   acceptable for this controlled local service. Do not let client YAML alter it.
5. **Device guard qualification.** Implement/finish startup checks that compare the
   one visible CUDA device with expected UUID/device metadata and fail readiness
   on mismatch. Exercise mismatch/CPU-only behavior safely with injected tests;
   live success must prove physical GPU1 maps to logical cuda:0.
6. **Measure stages separately.** On a small redistributable test image/config,
   profile SAM2, CLIP and BLIP3 independently: load time, inference latency,
   baseline/peak VRAM, retained/resident VRAM after inference, host RAM where
   material, and cleanup behavior. Snapshot all GPUs/processes before/after.
7. **Measure meaningful combinations.** Test the intended SAM2→CLIP path and,
   only if individual evidence permits, SAM2→CLIP→BLIP3. Do not intentionally OOM
   a shared workstation. Use conservative progression and stop before a predicted
   unsafe allocation.
8. **Choose an explicit resource strategy from evidence.** With ~11 GB GPU1, do
   not assume all models co-reside. Select one of, or a measured combination of:
   resident SAM2+CLIP with BLIP3 rejected; staged load/unload; CPU placement for a
   bounded component if scientifically/operationally acceptable; alternate
   approved smaller revision; or explicit configuration restrictions. Any change
   in scientific model identity versus the existing prototype must be surfaced
   and justified, not smuggled in as an optimization.
9. **Test repeated inference stability.** For every supported live profile, run a
   bounded repeated sequence sufficient to expose obvious VRAM/RSS leaks and
   stale request state. Record peak/end memory and stage outputs; do not claim
   production soak from a short test.
10. **Confirm no GPU0 impact.** Before/after evidence must show no ZAP-IT process
    or new allocation on GPU0. Existing unrelated GPU0 PID/memory is protected
    evidence, not noise to clean up.
11. **Shared-memory qualification.** Verify `/dev/shm/slaif-zap-it` can be safely
    created/owned with correct permissions, has adequate capacity for bounded
    request artifacts, and cleanup helpers operate without persistent residue.
    Do not put model weights there.
12. **Select a candidate loopback port.** Inspect listeners and choose an unused
    `127.0.0.1` port for Objective 004; record it in operator/private runtime
    configuration and docs. Do not rely on the number being free forever and do
    not leave the service running merely to reserve it.
13. **Add opt-in GPU integration tests.** Mark and guard them so public CPU CI does
    not accidentally download models or touch CUDA. Tests must fail/skip honestly
    on wrong/missing device and must serialize GPU use.
14. **Produce a tested-hardware/runtime record.** Document exact host/GPU1/driver/
    Python/PyTorch/model revisions, supported pipeline profiles, measured memory/
    latency, known unsupported combinations and reproduction commands.

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

1. Current live evidence records exact GPU1 UUID/PCI/name/VRAM and proves the
   service mask exposes only physical GPU1 as logical cuda:0.
2. GPU0 is untouched throughout; before/after process/memory snapshots prove no
   ZAP-IT allocation or process there.
3. A reproducible repo/operator runtime environment with exact versions is
   documented and can import the required stack.
4. Approved SAM2/CLIP/BLIP3 model identities/revisions/licenses/provenance and any
   remote-code execution are documented honestly.
5. Each relevant model stage has measured load/inference/peak VRAM evidence on
   GPU1, with bounded redistributable fixtures/configs.
6. Every pipeline profile claimed supported has a measured resource strategy and
   repeated live success. Unsupported/unsafe combinations are explicitly rejected
   by configuration/readiness rather than spilling to GPU0 or crashing blindly.
7. Device guard fails closed on wrong UUID/visibility and does not silently select
   GPU0 or CPU in strict GPU service mode.
8. `/dev/shm` root/permissions/capacity and cleanup are verified.
9. A currently unused loopback candidate port is selected and recorded for 004,
   but no persistent service remains after this objective.
10. Opt-in GPU tests are isolated from normal CPU CI and have explicit prerequisites.
11. CPU CI/CodeQL remains green; live GPU evidence is reported separately and not
    misrepresented as GitHub-hosted CI.
12. Correct objective PR/report-only SELF contract is satisfied; coding never
    merges.

## Required verification

- live `nvidia-smi` all-device snapshot before/after each test class: VERIFY:
- compute-process snapshot before/after: VERIFY:
- masked PyTorch device-count/name/UUID mapping probe: VERIFY:
- Python/PyTorch/CUDA/library version capture: VERIFY:
- model revision/license/provenance audit: VERIFY:
- SAM2-only profile metrics: VERIFY:
- CLIP-only profile metrics: VERIFY:
- BLIP3-only profile metrics or safe BLOCKED/unsupported evidence: VERIFY:
- combined supported-profile metrics and repeated-run test: VERIFY:
- wrong-UUID/wrong-visibility fail-closed tests: VERIFY:
- `/dev/shm` permissions/capacity/cleanup: VERIFY:
- listener scan and candidate-port selection: VERIFY:
- CPU/static/CI/CodeQL regression: VERIFY:

## Documentation and provenance

Update tested-hardware/runtime docs, model/dependency provenance, exact GPU1 UUID
operator guidance, supported/unsupported configurations, memory strategy and live
test procedure. Remove every stale implication that the target 2080 Ti has
22/24 GB; current measured hardware is 11264 MiB unless fresh evidence changes.
Do not publish credentials, private cache paths or unrelated process details
beyond sanitized evidence necessary to prove isolation.

## Security/resource constraints

This is the first objective authorized to allocate physical GPU1. Authorization is
strictly limited to bounded ZAP-IT tests and repo-owned runtime setup. No GPU0,
other process mutation, driver/CUDA system changes, firewall/VPN changes or
persistent service. Model downloads, if required, must be approved upstream
sources/revisions and stored in operator-controlled cache; record provenance and
avoid re-downloading unnecessarily.

## Deferred human adjudication

- Decision: `NONE`
- Memory-fit strategy, supported profile selection and exact pinned compatible
  dependency revisions are strategic engineering decisions to make from evidence.
- A change that materially alters the service's scientific model identity or
  trust boundary may qualify for `CRITICAL.md` only if all five strict conditions
  hold. Strategic must decide provisionally rather than simply stop.

## GitHub publication and report

Create one objective-003 branch/PR from verified remote main. Keep live host
changes bounded/reproducible and version operator-facing configuration/docs needed
for later service activation. Push implementation/evidence code before the final
report-only SELF commit; report exact GPU/process snapshots, versions, model
provenance, memory/latency tables, supported/unsupported profiles, selected port,
shared-memory results, skips/blockers and rollback/cleanup. Never merge.