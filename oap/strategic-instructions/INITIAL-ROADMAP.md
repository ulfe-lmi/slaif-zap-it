# Initial strategic roadmap

This is sequencing guidance, not activated scope.

| Objective | Outcome | Hard gate |
|---|---|---|
| 000 | Professional package/test/CI/CodeQL/docs/security/provenance baseline | Existing behavior characterized; CPU CI green |
| 001 | Pure typed in-memory image pipeline, artifact sinks, deterministic YOLO/uint16 identity mask | Legacy CLI regression preserved; exhaustive CPU semantics |
| 002 | Multipart `/v1/completions`, JSON/ZIP levels, limits/errors/auth/health using fake engine | No live GPU deployment; API contract tests green |
| 003 | Actual host/GPU1 environment and model qualification | UUID pin; GPU0 untouched; memory fit and licenses documented |
| 004 | Local loopback service on verified unused port | One worker/request; E2E levels; cleanup; rollback |
| 005 | Full-output parity, overlap masks, visualization/geometry, resource hardening/metrics/datasheet | Repeated load/failure/cancel tests |
| 006 | Optional container/systemd polish, SLAIF gateway route and release | Human approval, distribution/license/security review |

Each numeric objective is one PR. Use lettered remediation until accepted/merged.
Strategic may split/reorder after evidence but may not skip dependency gates.

## Human adjudication gate

Open `CRITICAL.md` entries do not automatically stop autonomous implementation.
Before any external deployment, production/customer data, irreversible production
mutation, or final release, the human must append an `ACCEPTED` disposition for every entry whose stated
gate applies. `DEFERRED`, `REJECTED`, or `CHANGE REQUIRED` stays blocking.
Packaging and local test preparation may continue when safe; crossing the gate
may not.
