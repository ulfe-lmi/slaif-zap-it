# 0.1.0 release-candidate notes

Status: unpublished and untagged. This is a qualified local research candidate,
not a production or commercial-model-use approval.

This candidate is the reproducible development handoff for the ZAP-IT native
loopback service. It is tested as a bounded local service on the qualified
physical GPU1 profile (NVIDIA GeForce RTX 2080 Ti, 11264 MiB; application
device cuda:0 after visibility masking) and as a CPU/fake-engine package.
Those facts are evidence for this candidate, not an SLA, accuracy claim,
production approval or rights clearance.

The candidate contains the legacy YAML-driven CLI, typed in-memory core,
deterministic YOLO/identity-mask/RLE renderers, and the fixed `zap-it-1` /
`zap-it.v1` API contract. SAM2, CLIP, and sequential BLIP3 are qualified on the
11 GB host. Geometry and panoptic rendering remain outside the service. Model
weights are downloaded by an operator and are never packaged.

CRIT-0001 is accepted: the repository owner confirmed redistribution rights for
the goat image/YAML fixtures and did not require a history rewrite. Before a
final release, reviewers must still assess model/commercial-use terms, remaining
media inventory, repository security settings, and intended deployment scope.
Gateway integration and container deployment remain separate work.
