# Security law

## Inputs

Treat uploaded image and YAML as hostile. Enforce encoded-byte, decoded-pixel,
dimension, YAML-byte/depth/alias/collection/string, object/artifact, response,
time, queue, host-memory, `/dev/shm` and GPU-memory limits.

Use `yaml.safe_load` plus typed allowlist. API YAML cannot control paths, output
directories, URLs, downloads, imports, commands, devices, environment, secrets,
service settings, arbitrary models/revisions, debug destinations or persistence.
No request-triggered network access or remote-code load.

## Data lifecycle

Raw image/config/result persistence and raw-body logging are off by default.
Prefer RAM. If paths are required, use opaque mode-0700 per-request directories
under configured RAM-backed `/dev/shm`; files 0600; no symlinks/client names;
unconditional cleanup. No silent persistent-disk fallback.

Logs/errors/metrics exclude images, YAML, prompts/labels if sensitive, BLIP
answers, filenames, headers, keys, host paths, stack traces and high-cardinality
content. The custom `/metrics` registry uses only finite stable outcome,
verbosity and format labels plus unlabeled sizes/counts/timings. Use opaque
request IDs, hashes/counts/status/timings in safe diagnostics.

## GPU and host

Physical GPU1 only through `CUDA_VISIBLE_DEVICES=1`; logical application device
`cuda:0`. Pin expected GPU UUID for deployment. Never touch GPU0 processes,
reset GPUs, modify system drivers/CUDA, firewall/VPN/network, unrelated systemd,
ports or global OpenCode/provider credentials without exact human-approved order.

One server process/worker and one GPU inference initially. Model weights/caches
are operator assets outside Git. Pin/review model revisions and licenses,
especially `trust_remote_code`; uploaded config cannot change them.

## API

Bind loopback by default. Before LAN/gateway exposure require protected API key or
trusted gateway identity, constant-time comparison, request limits and sanitized
errors. Never accept caller-asserted trusted identity headers. Do not expose
debug/docs/metrics publicly without policy.

## Supply chain

Preserve MIT project license and audit every model/library license/notice. Do not
commit weights, credentials, private data, generated debug corpus or caches. Pin
runtime dependencies/revisions; CI uses least permissions and no untrusted secret
execution. Report suspected exposure immediately; do not “clean up” before
preserving safe evidence.

## Deferred security judgment

A consequential unresolved security/trust decision is not automatically a reason
to stop development. Strategic must investigate, choose the least-dangerous
reversible provisional design, require mitigations/tests/rollback, and continue.
Only if every `CRITICAL.md` threshold condition holds may it order a rare
append-only entry for later human adjudication.

The register is not a waiver. An open entry cannot authorize public exposure,
production/customer data, destructive production mutation, disabling mandatory
controls, external privilege expansion, or release across its stated human gate.
Normal vulnerabilities, bugs and test failures are fixed or reported through the
ordinary OAP loop; they are not converted into CRITICAL entries to avoid work.
