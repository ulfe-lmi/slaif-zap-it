# OAP Coding-Agent Report — 006-a

## Work order
- Identifier/order/objective/PR mode: 006-a / 006-a-release-and-integration.md / Objective 006 / CREATE_NEW_PR
- Repository: ulfe-lmi/slaif-zap-it
- Branch: oap/006-a-release-and-integration
- Pull request: https://github.com/ulfe-lmi/slaif-zap-it/pull/50
- Required title: Objective 006-a: package and qualify the 0.1.0 release candidate

## Status
COMPLETE

## Executive summary

Objective 006-a produced an unpublished, reproducible 0.1.0 release candidate for the native loopback service. It adds audited wheel/sdist contents, version provenance, an installed foreground entrypoint, a Type=simple uninstalled user-systemd template, release/security scans, operator documentation, and a separate future gateway contract. The four nonredistributable goat inputs were removed from the tracked tip and remain present only as ignored local operator files.

The bounded candidate and live rehearsal passed. Final tag/package/source publication, public-history remediation, repository visibility changes, gateway work, containers and systemd installation remain outside scope. CRIT-0001 is OPEN/BLOCKING for the stated human release gate.

## Authoritative GitHub state

- Remote main/base SHA: 1758c3989454a000c71c2fc986db505bb70f3a5b
- Starting SHA: 1758c3989454a000c71c2fc986db505bb70f3a5b
- Implementation head SHA: 3fe090d08e122b33d0d68d1eb4569a38fdf9cb09
- Report publication commit: SELF
- PR #50 is OPEN, non-draft, based on the required main SHA, mergeable/clean, not merged, and has no auto-merge.
- Repository visibility is PUBLIC. There are zero GitHub releases and zero tags. The main protection endpoint has no required status checks, review requirements or admin enforcement; owner review is recommended.
- GitHub software truth and the remote branch both resolve to implementation SHA 3fe090d08e122b33d0d68d1eb4569a38fdf9cb09 before this report.

## Changes/files

- Packaging: pinned setuptools build backend, bounded dev tooling including detect-secrets 1.5.0 and Twine, importable version fallback, installed zap-it-service entrypoint, data-file notices, explicit MANIFEST.in allowlist, and artifact verifier.
- Distribution safety: wheel/sdist deny media, demos, caches, outputs, model payloads, OAP material, credentials, local goat names, unsafe archive paths, symlinks and oversized members. Unpacked archive secret scans use the reviewed baseline.
- Local academic path: scripts/smoke_local_goats.py accepts ignored local inputs only, rejects symlinks/out-of-root files, safe-loads and derives an API-safe mapping, strips resident SAM2 tuning, crops the central 50 percent in memory, performs A/B/A L2 JSON/L3 JSON/L3 ZIP, and emits sanitized evidence only.
- Operator handoff: installed foreground documentation, private EnvironmentFile guidance, Type=simple unit template, runbook, changelog, release notes and service provenance.
- Integration/provenance: future gateway contract, model/license/media inventory, Dependabot configuration, immutable GitHub Actions pins, and CRIT-0001.
- Required tracked removals: configs/goats.yaml, configs/goats2.yaml, demos/goats/goats1.jpg, demos/goats/goats2.jpg. The local files remain on the host, root-anchored ignored, and were not read into Git/OAP evidence.

## Acceptance evidence

| Criterion | State | Evidence |
| --- | --- | --- |
| 1. Correct branch/PR/base | PASSED | PR #50, required title, base 1758c3989454a000c71c2fc986db505bb70f3a5b, implementation head 3fe090d08e122b33d0d68d1eb4569a38fdf9cb09. |
| 2. Exact CRIT-0001 append | PASSED | Helper dry-run and append completed; source bytes matched the order block; pre-append prefix SHA-256 edacb6f86ebac60da896714c4c10ef65d1c0b8dc792441ac46ffa9680d3ba7ca; entry SHA-256 a4a5d0d989c60a6c99d27db7c37474faa2c76b196dfa60759442c7431b4f883f; register SHA-256 d639ebf52f5bfb6b49cc05838f63b359268ffde0829c1aca73f639e5c2c961c7. |
| 3. Goat tip removal/ignored local copies | PASSED | Four exact paths are absent from the staged/committed index, remain regular local files, and each is covered by a root-anchored ignore rule. No goat bytes or derivatives are in the implementation commit or artifacts. |
| 4. Opt-in harness and local A/B/A | PASSED | Generated CPU tests pass; live installed-candidate run passed 9 cases, crop dimensions 2784x2088, and zero persistence. |
| 5. Wheel/sdist and clean installs | PASSED | Two clean builds, archive/member verifier, unpacked detect-secrets scan, Twine, isolated no-checkout fake-engine smoke, sdist-built wheel verification, and member digest comparison passed. |
| 6. Version/docs honesty | PASSED | 0.1.0 is consistently described as unpublished; API zap-it-1 and zap-it.v1 remain unchanged; limitations and release notes are explicit. |
| 7. Native service packaging | PASSED | Installed foreground entrypoint imports from site-packages; Type=simple unit syntax passed systemd-analyze verify; private EnvironmentFile and clean install/upgrade/rollback/uninstall are documented; no systemd tree mutation occurred. |
| 8. Gateway boundary | PASSED | Exact future contract and bearer boundary are documented. Gateway E2E is NOT INCLUDED — separate repository/order required. |
| 9. License/provenance/media | PASSED | THIRD_PARTY_NOTICES.md and docs/RELEASE-GATE-INVENTORY.md inventory model, package, media and unresolved rights gates. |
| 10. Supply chain/security | PASSED | Actions are pinned to required commits, permissions remain least privilege, detect-secrets 1.5.0 baseline and artifact scans pass, Dependabot config is present, and no release credentials/settings were added. |
| 11. CPU/package/CI verification | PASSED | 343 CPU tests passed, 1 GPU-marked test skipped honestly, coverage 77.40%, Ruff/compile/shell/systemd/package checks passed, and all implementation-head checks passed. |
| 12. Installed GPU1 rehearsal | PASSED | Fresh GPU1 UUID/process/port preflight; auth, synthetic L0-L3 JSON, L3 ZIP, BLIP3 rejection, goat A/B/A, restart, cleanup and GPU0 invariance passed. |
| 13. Final report/release gates | PASSED | This report contains CRIT-0001 OPEN/BLOCKING, public history not remediated, factual repository security settings, gateway/container exclusion and required human actions. |
| 14. Prohibited-boundary safety | PASSED | No merge, tag, publication, history rewrite, force push, ref deletion, visibility change, gateway edit, container, LAN bind, systemd install or GPU0 mutation occurred. |

## Verification

- git fetch origin main and git rev-parse origin/main: PASSED — required remote base.
- git rm --cached -- configs/goats.yaml configs/goats2.yaml demos/goats/goats1.jpg demos/goats/goats2.jpg: PASSED — index-only removal; local files preserved.
- git check-ignore -v on all four goat paths: PASSED — exact root-anchored rules.
- python3 oap/bin/append_critical.py --repo-root . --source oap/CRIT-0001-source.tmp.md --id CRIT-0001 --dry-run: PASSED.
- python3 oap/bin/append_critical.py --repo-root . --source oap/CRIT-0001-source.tmp.md --id CRIT-0001: PASSED — source was temporary and is absent from the commit.
- Critical prefix/terminal-entry byte comparison: PASSED.
- .venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing: PASSED — 343 passed, 1 skipped, 77.40% coverage.
- .venv/bin/pytest -q tests/test_release_candidate.py tests/test_live_service_units.py tests/test_real_yaml_config.py: PASSED — 57 passed.
- .venv/bin/ruff format --check .: PASSED.
- .venv/bin/ruff check .: PASSED.
- python3 -m compileall -q src modules scripts tests: PASSED.
- bash -n scripts/serve_local.sh scripts/serve_local_stop.sh: PASSED.
- systemd-analyze verify deploy/zap-it-local.service: PASSED.
- .venv/bin/detect-secrets scan --baseline .secrets.baseline --no-verify: PASSED.
- .venv/bin/python scripts/verify_release_artifacts.py /dev/shm/slaif-zap-it-final.2T4t2f/*.whl /dev/shm/slaif-zap-it-final.2T4t2f/*.tar.gz: PASSED.
- .venv/bin/python scripts/scan_release_artifacts.py /dev/shm/slaif-zap-it-final.2T4t2f/*.whl /dev/shm/slaif-zap-it-final.2T4t2f/*.tar.gz --baseline .secrets.baseline: PASSED — 0 unexpected findings.
- .venv/bin/python -m twine check /dev/shm/slaif-zap-it-final.2T4t2f/*.whl /dev/shm/slaif-zap-it-final.2T4t2f/*.tar.gz: PASSED.
- Two fixed SOURCE_DATE_EPOCH builds: PASSED — wheel bytes and wheel/sdist member manifests reproduced; sdist gzip container hashes differed while all 137 member paths/bytes matched.
- sdist-built wheel verifier: PASSED — wheel SHA-256 ba986bc3e26e62aa22852d35d985408a0528323ed4e9ec7e9a14fbc0cc0faa77.
- isolated installed no-checkout fake-engine smoke: PASSED — package version 0.1.0 and service app import/create passed.
- live GPU1 service rehearsal: PASSED — installed site-packages candidate, one process/request, loopback-only.

## Artifact evidence

| Artifact | Size | SHA-256 | Member count | Member-manifest SHA-256 |
| --- | ---: | --- | ---: | --- |
| zap_it-0.1.0-py3-none-any.whl | 120726 | ba986bc3e26e62aa22852d35d985408a0528323ed4e9ec7e9a14fbc0cc0faa77 | 63 | a265cb86d44602f073f934b44fd15137c80267e05705341e976a08d29568affa |
| zap_it-0.1.0.tar.gz | 213973 | c78b256a68db4ffb287631c195a86a7c1f30ff9d962fda65fa93ae77c9a7aa76 | 137 | c1902907297c16f8b81c882f64926fa27841908da46b3d66bf2a10db7527be32 |

The independent second sdist gzip hash was 845669d8396f13b90a5912f8d2846dd6ff7180b291952ab0bdaa41fa6f69838b; its member count and member-manifest SHA-256 were identical. This is reported as container-metadata-only variance, not false bit-level reproducibility.

## CI/checks

Implementation-head PR #50 checks at SHA 3fe090d08e122b33d0d68d1eb4569a38fdf9cb09:

- Analyze (python): PASSED.
- CodeQL: PASSED.
- static (format, lint, build): PASSED.
- release (artifact audit): PASSED.
- tests (py3.10): PASSED.
- tests (py3.11): PASSED.
- tests (py3.12): PASSED.

No check was pending, skipped or missing at report preparation.

## GPU/service/resource evidence

- Physical GPU1: index 1, UUID GPU-c457dbaf-991c-dc23-c781-0dc030776dd8, PCI 00000000:00:0C.0, NVIDIA GeForce RTX 2080 Ti, 11264 MiB, driver 580.178.04.
- Launch mask: CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=1; application logical device cuda:0. Torch 2.5.1+cu124 with CUDA 12.4; torch.cuda.is_available() was true in the qualified environment.
- GPU0 protection: physical GPU0 UUID GPU-4c129e25-8e59-eee4-b49c-56c40e294182 and unrelated PID 66522 remained unchanged. The sanitized compute-evidence hash before/after was b2909915a74f5536013fdfd090a364757c1293d5f6f7df1617f8ae09d90344fd.
- Selected loopback port: 17891, freshly unused before activation and free after stop. No LAN/public bind or unrelated service mutation.
- GPU1 memory: ready 1849 MiB used / 8973 MiB free; during live E2E 6655 MiB used / 4167 MiB free; post-stop 6 MiB used / 10815 MiB free. Candidate process RSS was approximately 1332 MiB ready and 2010 MiB during live E2E.
- Service process model: one installed foreground process, one worker, one inference request at a time. Restart health passed.
- Resource cleanup: no ZAP-IT compute process, no listener on the selected port, zero files under /dev/shm/slaif-zap-it, and ephemeral launch log removed.

## Documentation/provenance

Updated README, INSTALL, RUNBOOK, SERVICE-DATASHEET, CHANGELOG, RELEASE_NOTES, THIRD_PARTY_NOTICES, RELEASE-GATE-INVENTORY and GATEWAY-INTEGRATION. Package provenance reports 0.1.0 safely from installed distribution metadata with a source-tree fallback. Model weights remain operator-downloaded and unpackaged.

## Deferred human adjudication

- Critical register action: APPENDED CRIT-0001
- Exact strategic entry appended byte-for-byte with append-only prefix proof; no human disposition was written.
- CRIT-0001 status: OPEN — HUMAN ADJUDICATION REQUIRED.
- Human gate: before any final tag, package/source release, rights-cleared claim, destructive history rewrite or repository-visibility change, the owner must choose the authoritative public-history remedy and review known forks/releases/caches/notifications.

## Safety/scope confirmations

- The implementation commit contains no raw goat image/YAML bytes, crops, prompts, labels, raw responses, credentials, model weights, cache files, generated artifacts or private environment files.
- The local academic files are still present as operator-held ignored files and were never deleted, resized, renamed, moved or persisted as derivatives.
- No other repository was modified. The SLAIF gateway repository was only reconciled at its recorded commit; no gateway adapter or E2E was implemented.
- No Docker/Compose artifact, LAN/public listener, final tag, package upload, GitHub release, systemd installation, firewall/VPN/network change, history rewrite, force push or merge occurred.
- GPU0 and unrelated services/processes were protected.

## Limitations/blockers

- CRIT-0001 remains OPEN/BLOCKING. Current-tip removal does not remediate public Git history, forks, clones, caches or prior source archives.
- Model commercial/deployment rights are not approved. CLIP lacks an SPDX deployment license in its pinned card; BLIP3/XGen-MM is unsupported on this host and CC-BY-NC/research-only.
- Remaining repository media history is inventoried but not rights-cleared. All demo/assets media is excluded from wheel/sdist.
- This is an unpublished release candidate, not a final release, production deployment, accuracy benchmark, SLA, gateway integration or container qualification.

## Factual strategic follow-up

- Human adjudicate CRIT-0001 before final release/public rights claims and record the disposition append-only.
- Human/legal review model and media rights and decide repository security settings, including Dependabot security updates, secret scanning/push protection and main branch protection.
- A future gateway PR must implement and independently test docs/GATEWAY-INTEGRATION.md; a container/public topology requires a separate order.

## Final publication

- Implementation commit parent: 1758c3989454a000c71c2fc986db505bb70f3a5b.
- Implementation commit: 3fe090d08e122b33d0d68d1eb4569a38fdf9cb09.
- Final report-only commit must change only this file and have implementation SHA 3fe090d08e122b33d0d68d1eb4569a38fdf9cb09 as its first parent.
- Coding merge: NO.
