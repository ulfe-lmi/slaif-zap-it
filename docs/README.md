# ZAP-IT documentation

This index separates current product/operator documentation from dated
qualification evidence and historical modernization records. Objectives 007–009
close the GPU-memory deferrals: below 24,576 MiB the historical 11 GB card uses
the live-qualified sequential lifecycle, while at or above 24,576 MiB the
assigned RTX 3090 uses the live-qualified all-resident lifecycle. The real
Objective 009 matrix covers all four supported profiles. Geometry/panoptic,
deployment, licensing, media and final-release gates remain separate and are
not represented as memory-blocked work.

## Start here

| Audience | Document | Purpose |
| --- | --- | --- |
| New users | [Project README](../README.md) | Capabilities, status, and quickstarts |
| Installers | [Installation](../INSTALL.md) | CPU tooling, GPU runtime, models, and service install |
| Batch users | [Configuration reference](CONFIG.md) | YAML fields for CLI and service-safe requests |
| API clients | [HTTP API](API.md) | Multipart request, response levels, limits, and errors |
| Operators | [Runbook](RUNBOOK.md) | Preflight, start, health, smoke, stop, and rollback |
| Reviewers | [Service datasheet](SERVICE-DATASHEET.md) | Supported scope, measured evidence, and limitations |

## Design and behavior

- [Architecture](../ARCHITECTURE.md) — current component and trust boundaries.
- [Core library](CORE.md) — typed single-image pipeline and deterministic outputs.
- [Algorithms](ALGORITHMS.md) — SAM2, CLIP, BLIP3, filtering, rendering, and YOLO.
- [Output parity](OUTPUT-PARITY.md) — current service, CLI-only, and unsupported outputs.
- [GPU runtime](runtime.md) — pinned environment, model provenance, residency, and measurements.
- [Gateway integration](GATEWAY-INTEGRATION.md) — proposed future SLAIF gateway adapter contract.

## Project governance

- [Security](../SECURITY.md)
- [Testing](../TESTING.md)
- [Contributing](../CONTRIBUTING.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)
- [Changelog](../CHANGELOG.md)
- [Release notes](../RELEASE_NOTES.md)
- [Release-gate inventory](RELEASE-GATE-INVENTORY.md)

## Historical records

Documents in [`history/`](history/) describe the starting repository, original
targets, and bootstrap context at specific points in time. They are preserved
for provenance and are not current product instructions.

The immutable OAP orders and reports under [`../oap/`](../oap/) are development
evidence. They intentionally retain the facts and decisions that were true when
each round ran; use current documentation for present behavior.
