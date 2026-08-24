# Historical record: bootstrap sources and adaptation

> Prepared before modernization began. Repository and tool descriptions below
> are historical and must not be used as current operational instructions.

Prepared 2026-08-22 from:

- `ulfe-lmi/slaif-zap-it` current `main`: existing SAM2/CLIP/BLIP3/geometry/
  visualization/YOLO pipeline, configs and tests;
- `ulfe-lmi/slaif-service-facial-manipulation-scoring`: SLAIF packaging, CI,
  CodeQL, service-contract, API, security/provenance/documentation patterns;
- `ulfe-lmi/slaif-agent-site` OAP transcript/role separation and the existing
  `CRITICAL.md` autonomous post-merge human-review queue pattern;
- `ulfe-lmi/slaif-local-coding` exact FIFO, atomic order publication, SELF-report
  and two-agent wrapper implementation;
- SLAIF OAP manual;
- official OpenCode CLI, rules, agents, permissions and model/variant docs.

No implementation code or model weights were copied into this overlay. The OAP
mechanics are adapted for OpenCode (`opencode run` coding rounds, persistent TUI
strategy, `--auto`, `AGENTS.md`, `opencode.json`). The pack additionally formalizes
Human Work Preloading, Human Judgment Postloading, a strict five-condition
critical-entry threshold, append-only autonomous decisions, and human adjudication
before applicable deployment/release gates. Verify all live facts before
activation.
