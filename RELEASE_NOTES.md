# 0.1.0 release-candidate notes

Status: unpublished, untagged and not approved for package/source release.

This candidate is the reproducible development handoff for the ZAP-IT native
loopback service. It is tested as a bounded local service on the qualified
physical GPU1 profile (NVIDIA GeForce RTX 2080 Ti, 11264 MiB; application
device cuda:0 after visibility masking) and as a CPU/fake-engine package.
Those facts are evidence for this candidate, not an SLA, accuracy claim,
production approval or rights clearance.

The candidate contains the legacy YAML-driven CLI, typed in-memory core,
deterministic YOLO/identity-mask/RLE renderers, and the fixed zap-it-1 /
zap-it.v1 API contract. It does not activate BLIP3, geometry or panoptic
stages on the qualified host. Model weights are downloaded by an operator at
runtime and are never packaged.

Before any final tag, package/source release or rights-cleared claim, a human
must resolve the open CRIT-0001 public-history gate, review model and media
rights, and confirm repository security settings. Gateway integration and
container deployment are separate work and are not included.
