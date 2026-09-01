# 0.1.0 release-candidate notes

Objective 023 adds the opt-in BLIP3 `centroid_radial_mask_chord` fallback after
the existing containment rejection. The default `reject` path and feasible
Euclidean compositions remain compatible; fallback geometry and adjustment
evidence are exposed at L3.

Objective 017 adds request-local mask-isolated candidate views for CLIP and
BLIP3. The service exposes effective view settings at every response level and
fixed tokenized PNG artifacts plus bounded L3 model-input records. This remains
bounded local research evidence and does not claim semantic accuracy or release
readiness.

Status: unpublished and untagged. This is a qualified local research candidate,
not a production or commercial-model-use approval.

This candidate is the reproducible development handoff for the ZAP-IT native
loopback service. It is tested as a bounded local service on the historical
sequential physical GPU1 profile (NVIDIA GeForce RTX 2080 Ti, 11264 MiB) and on
the assigned all-resident physical GPU0 profile (NVIDIA GeForce RTX 3090,
24576 MiB), with each masked process using only logical `cuda:0`; it is also
tested as a CPU/fake-engine package. Those facts are evidence for this
candidate, not an SLA, accuracy claim, production approval or rights clearance.

The candidate contains the legacy YAML-driven CLI, typed in-memory core,
deterministic YOLO/identity-mask/RLE renderers, and the fixed `zap-it-1` /
`zap-it.v1` API contract. The 11 GB host uses the live-qualified sequential
stage-boundary lifecycle; the 24,576-MiB host uses the live-qualified
all-resident lifecycle. Objectives 007–009 include real all-resident evidence
for all four supported profiles. Geometry/panoptic, gateway/deployment,
licensing/media review and final release remain separate gates. Model weights
are downloaded by an operator and are never packaged.

CRIT-0001 is accepted: the repository owner confirmed redistribution rights for
the goat image/YAML fixtures and did not require a history rewrite. Before a
final release, reviewers must still assess model/commercial-use terms, remaining
media inventory, repository security settings, and intended deployment scope.
Gateway integration and container deployment remain separate work.
