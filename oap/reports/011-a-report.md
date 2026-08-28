# OAP Coding-Agent Report — 011-a

## Work order

- Identifier: `011-a`
- Objective: authenticated private-LAN ZAP-IT service on hinton2 at
  `10.8.132.76:17891`, with a fixed operator bearer and the assigned physical
  GPU only.
- PR mode: one exact objective PR; amended existing PR #67 after preflight
  found its already-pushed implementation branch.
- Deferred human adjudication decision in the order: `NONE`.

## Status

PASSED

## Executive summary

Implemented and activated the authenticated `private_lan` service. Loopback
remains the default; private-LAN startup accepts only an explicit RFC1918
IPv4 address/CIDR, rejects wildcard/public/hostname/link-local/multicast and
Docker-default bridge addresses, disables interactive API documentation, and
requires the fixed bearer for inference and metrics. The operator installer
creates or preserves the fixed key without printing it, writes the protected
environment file, and installs the hardened user unit.

The persistent unit is enabled and active at exactly `10.8.132.76:17891`.
Authenticated real L0-L3 requests and L3 ZIP rendering passed, including a
post-restart L3 request. The restart preserved the key and changed the owned
service PID. No merge or auto-merge was performed.

## Authoritative GitHub state

- Repository: `https://github.com/ulfe-lmi/slaif-zap-it`
- PR: `https://github.com/ulfe-lmi/slaif-zap-it/pull/67`
- PR state: `OPEN`, merge state `CLEAN`; base `main` at
  `a4f02b79be888c769f811625dcc0ad939b35a098`.
- Branch: `oap/011-a-authenticated-private-lan-service`.
- Starting checkout SHA: `94cc4db30bba577cf03d56a8f883bfdaac97f291`.
- Implementation head SHA: `ca00da5279da8f792173da1403d8567aa5566fad`.
- Report publication commit: `SELF`.
- New PR: no in this execution; the exact existing PR #67 was amended.
- Coding merge/auto-merge: `NO`.

## Changes/files

- Added immutable `loopback|private_lan` launch scope, RFC1918/CIDR
  validation, Docker bridge rejection, LAN bearer enforcement, and LAN docs
  disabling in `src/runtime/live_service.py` and `src/service/app.py`.
- Updated the scoped launcher and port selection in `scripts/serve_local.py`,
  `scripts/serve_local.sh`, and `src/runtime/ports.py`.
- Added `scripts/install_private_lan_service.py`, which atomically writes the
  mode-0600 operator environment and a mode-0644 user unit, preserves a valid
  existing key, and never prints its value.
- Added/updated private-LAN tests in `tests/test_live_service_units.py`,
  `tests/test_model_control.py`, and `tests/test_private_lan_installer.py`.
- Updated `ARCHITECTURE.md`, `INSTALL.md`, `README.md`, `SECURITY.md`,
  `TESTING.md`, `deploy/service.env.example`, `docs/API.md`, `docs/RUNBOOK.md`,
  and `docs/SERVICE-DATASHEET.md`.
- Carried the immutable active transcript and order in `oap/active` and
  `oap/orders/011-a-authenticated-private-lan-service.md`.

Implementation commits on the exact PR branch are `0d8fa0be46c14ab60d52781166d0d2640d0b0e47`,
`94cc4db30bba577cf03d56a8f883bfdaac97f291`, and
`ca00da5279da8f792173da1403d8567aa5566fad`. The final child changes only
this report.

## Acceptance evidence

1. **Network scope and binding — PASSED.** `loopback` remains the default;
   `private_lan` requires an explicit matching RFC1918 IPv4 host/CIDR. Negative
   tests cover wildcard, hostname, public, loopback, scope mismatch and
   `172.17.0.0/16` Docker bridge addresses. Live `ss` showed one listener at
   `10.8.132.76:17891`, with zero wildcard and zero loopback listeners.
2. **Authentication and documentation — PASSED.** Missing and wrong bearer
   POST requests returned `401 unauthorized`; authenticated health/readiness
   returned `200`, authenticated metrics returned `200`, and missing/wrong
   metrics credentials returned `401`. `/docs`, `/redoc`, and `/openapi.json`
   each returned `404` on the LAN listener.
3. **Installer and persistence — PASSED.** The operator environment is at the
   required `%h/.config/slaif-zap-it/service.env` with mode `0600`; the key
   length is 64 and its file digest was unchanged by restart. The user unit is
   installed at `%h/.config/systemd/user/zap-it-lan.service`, mode `0644`,
   enabled and active. The unit uses exact repository paths, `UMask=0077`,
   `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=strict`,
   `ProtectHome=read-only`, `ReadWritePaths=/dev/shm/slaif-zap-it`, restricted
   SUID/SGID creation, locked personality, and control-group teardown.
4. **GPU/process policy — PASSED.** Fresh host evidence was hinton2,
   `enp1s0=10.8.132.76/24`, and the assigned physical device was index `0`,
   UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`,
   NVIDIA GeForce RTX 3090, 24,576 MiB, driver `610.43.02`. Launch used
   `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0`; the process
   reported one visible device and logical `cuda:0` with the same UUID. The
   final compute snapshot contained only the owned service process. No other
   GPU, process, firewall, route, VPN, driver, or system unit was changed.
5. **Real service behavior — PASSED.** Authenticated real inference passed at
   L0, L1, L2 and L3 JSON; the L3 result contained 8 objects and 8 stage
   statuses. Authenticated L3 ZIP also passed with 3 members. After restart,
   authenticated L3 JSON again returned `200` with the expected bounded
   semantics.
6. **Restart/resource cleanup — PASSED.** PID `372647` was replaced by owned
   PID `373595`; the unit remained enabled/active and readiness returned `200`.
   Final `ss` listener count was exactly one. The service cgroup reported
   `MemoryCurrent=15179296768` and `MemoryPeak=21780410368` bytes after model
   load/inference; assigned-GPU usage was 12,259 MiB of 24,576 MiB. The
   canonical shared-memory root was mode `0700` and empty after requests.
   The key was absent from journal output and all tracked files; request
   config/image markers were absent from logs.

The first automatic unit attempt safely exited after detecting a transient
competing compute process on the assigned card. It did not kill or alter that
process; the configured user-unit restart produced the final owned process and
all final checks passed.

## Verification

- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`:
  `PASSED` — 424 passed, 1 opt-in GPU test skipped honestly, 77.31% total
  coverage.
- `.venv/bin/pytest -q tests/test_live_service_units.py tests/test_private_lan_installer.py`:
  `PASSED` — 51 passed.
- `.venv/bin/ruff format --check .`: `PASSED` — 141 files already formatted.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED`.
- `git diff --check`: `PASSED`.
- `.venv/bin/python scripts/check_documentation.py`: `PASSED` — 27 current
  documents.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`:
  `PASSED` — exactly 7 reviewed baseline findings.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`:
  `PASSED` — wheel and sdist member audits.
- `.venv/bin/python -m twine check dist/*`: `PASSED`.
- `systemd-analyze verify %h/.config/systemd/user/zap-it-lan.service`:
  `PASSED`.
- `systemd-analyze security --user zap-it-lan.service`: `PASSED` command
  execution; the advisory analyzer score was `8.8 EXPOSED` because optional
  syscall and IP-address filters are not set, while the required user-unit
  filesystem, privilege and lifecycle controls were verified above.
- `systemctl --user enable --now zap-it-lan.service`: `PASSED` — enabled and
  active.
- `systemctl --user restart zap-it-lan.service`: `PASSED` — new owned PID,
  unchanged key, readiness restored.
- Live authenticated service checks against `10.8.132.76:17891`: `PASSED` —
  auth negatives, health/readiness, metrics, docs/OpenAPI negatives, real
  L0-L3 JSON, real L3 ZIP, post-restart L3, listener, GPU and residue checks.

## CI/checks

All checks below completed `SUCCESS` for implementation head
`ca00da5279da8f792173da1403d8567aa5566fad` on PR #67:

- `static (format, lint, build)` — `PASSED`.
- `tests (py3.10)` — `PASSED`.
- `tests (py3.11)` — `PASSED`.
- `tests (py3.12)` — `PASSED`.
- `release (artifact audit)` — `PASSED`.
- `Analyze (python)` — `PASSED`.
- `CodeQL` — `PASSED`.

## GPU/service/resource evidence

- Persistent user unit: enabled/active, one Uvicorn worker and one inference
  request slot; final MainPID `373595`.
- Listener: exactly `10.8.132.76:17891`; no `0.0.0.0:17891` or
  `127.0.0.1:17891` listener.
- Visible device: physical index `0` mapped to logical `cuda:0`, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`; one final compute row owned by
  PID `373595`.
- Final GPU query: 24,576 MiB total, 12,259 MiB used, 11,865 MiB free; no
  non-service compute row.
- Shared memory: `/dev/shm/slaif-zap-it` mode `0700`, zero entries after
  requests.
- Files: operator environment mode `0600`; generated user unit mode `0644`;
  no key in Git, journal, unit body, report, or chat.

## Documentation/provenance

The network/auth contract, installer procedure, user-unit operation and
rollback, LAN error/docs behavior, fixed model-control `none` profile, GPU
mapping, and security boundary are documented in the changed root, API,
runbook, datasheet, install, deployment-template and security documents. No
runtime dependency was added; the installer uses the Python standard library.
Pinned model assets remain operator-managed outside Git.

## Deferred human adjudication

- Critical register action: `NONE`.
- No `CRITICAL.md` mutation was ordered or performed.

## Safety/scope confirmations

- No merge, auto-merge, release, package upload, public/WAN bind, TLS claim,
  firewall/route/VPN change, driver/CUDA change, unrelated systemd mutation,
  second service, multi-process lease, model-control exposure, or unassigned
  GPU operation was performed.
- The only external activation was the explicitly ordered user unit on the
  assigned hinton2 address/GPU. Model-control mode is `none`; LAN clients
  cannot mutate model residency.
- Request image/config/result data stayed in memory in the service. No raw
  request content, credentials, cache contents, or customer data entered the
  report, logs, Git history or chat.

## Limitations/blockers

The private-LAN mode is intentionally RFC1918-only and bearer-authenticated;
it does not provide TLS, WAN/public exposure or per-user authorization. The
service is a single process/worker with one active request as ordered. No
blocker remains for this order.

## Factual strategic follow-up

Review PR #67 and its live evidence. The service remains enabled and active at
the ordered private address; rollback, if selected by the operator, is the
documented user-unit disable/stop path. Coding does not merge, accept, release,
or select a subsequent OAP order.
