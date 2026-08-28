# OAP Work Order — 011-a — Authenticated private-LAN service

## Objective

Implement and activate one persistent ZAP-IT service on hinton2's explicit
RFC1918 address `10.8.132.76`, protected by one generated fixed inference API
key stored only in a mode-0600 operator environment file. Preserve loopback as
the default and require an explicit `private_lan` network scope for LAN binding.

This objective is directly authorized by the human on 2026-08-28. It
reprioritizes the previously drafted cooperative cross-process lease/handoff
work to a future objective; this service remains one process and one assigned
GPU and makes no multi-process ownership claim.

## Verified state

- Base: remote `main` at `a4f02b79be888c769f811625dcc0ad939b35a098`,
  Objective 010 merged with all post-report checks green.
- Host: `hinton2`; private interface `enp1s0` is `10.8.132.76/24`, default
  gateway `10.8.132.1`; Docker's down `172.17.0.1` is not a deployment target.
- Assigned GPU: physical index 0, UUID
  `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, RTX 3090
  24,576 MiB; fresh baseline 15 MiB and no compute process.
- `/dev/shm`: 12-GiB tmpfs; no prior service config/unit/process exists.
- Required branch/PR: `oap/011-a-authenticated-private-lan-service`, one new PR.

## Requirements

1. Add immutable network scope `loopback|private_lan`, default `loopback`.
2. Loopback retains current behavior. `private_lan` accepts only an explicit
   non-loopback RFC1918 IPv4 address. Reject hostnames, wildcard/unspecified,
   multicast, link-local, Docker address, public address, or scope/host mismatch.
3. LAN startup fails closed unless the inference API key is at least 32
   characters. Uploaded requests cannot select network scope/address/key.
4. Bind only `10.8.132.76`, never `0.0.0.0`; retain one worker/request and exact
   assigned index+UUID guard. No firewall, route, VPN, TLS, driver, or system
   service outside the user's unit is modified.
5. Disable `/docs`, `/redoc`, and `/openapi.json` on the LAN listener. Keep
   bounded `/healthz` and `/readyz`; require the inference bearer for
   `/v1/completions` and `/metrics`. Persistent deployment uses model-control
   mode `none`, so remote model mutation is disabled and the fixed profile loads
   at startup.
6. Update launcher, environment template, architecture/security/install/runbook/
   API/datasheet docs and tests. Add an operator installer that creates/reuses a
   random key without printing it, writes only a mode-0600 environment file,
   and installs a user-systemd unit with exact repository paths.
7. Activate the user service on port 17891. Verify file permissions, unit
   hardening, exact LAN listener, missing/wrong key 401, correct key health/
   readiness/metrics and real L3 inference, restart persistence, sanitized logs,
   GPU/device/process limits and no request residue.

## Non-goals

No public/WAN bind, DNS, TLS termination, gateway, firewall change, arbitrary
client allowlist, multi-user authorization, rate-limiter beyond existing one
active request/zero queue, cross-process lease, second service, GPU sharing,
container/Kubernetes, model-control exposure, release or package upload.

## Acceptance

- CPU/static/package/secret/CodeQL checks green with explicit negative LAN tests.
- One fixed key exists only in `%h/.config/slaif-zap-it/service.env`, mode 0600;
  it is absent from Git, commands, logs, reports and chat.
- `ss` shows exactly `10.8.132.76:17891`, not wildcard/loopback; the persistent
  user unit is active and enabled.
- Unauthenticated/wrong-key inference and metrics return 401; authenticated real
  inference succeeds; health/readiness are honest; docs/OpenAPI are 404.
- Restart retains the same on-disk key, obtains a new owned PID, reaches ready,
  and leaves no duplicate process/listener or shared-memory residue.

Deferred human adjudication: NONE. The human explicitly authorized the private
LAN exposure and fixed on-disk key. The implementation remains least-privilege,
RFC1918-only, authenticated, reversible and host-local.
