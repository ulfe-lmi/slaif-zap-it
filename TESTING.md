# Verification law

Exact statuses only: `PASSED|FAILED|SKIPPED|NOT RUN|BLOCKED|PENDING|MISSING`.
Skipped/pending/unavailable is never pass. Never weaken/delete tests or over-mock
the behavior under test to finish an order.

## Tiers

- T0 static/package: build/import, Ruff format/check, typing as adopted, docs,
  OpenAPI/schema, secret/license checks.
- T1 CPU unit: config, preprocess/postprocess, geometry, ordering, YOLO, identity
  PNG, serializers and existing regressions; no CUDA/network/model download.
- T2 CPU API: fake engine, multipart, levels, JSON/ZIP bytes, errors, auth,
  limits, concurrency/cancel, `/dev/shm` cleanup.
- T3 GPU integration: explicit opt-in; physical GPU1 only; pinned models; one
  test at a time; before/after UUID/process/VRAM evidence; redistributable image.
- T4 local deployment: verified free loopback port, health/readiness, all levels,
  repeated calls, no residue/GPU0 process, rollback/restart.

GitHub-hosted CI runs T0–T2 only. GPU tests must be markers and skip honestly
without required host/model assets. A self-hosted GPU check is not required until
an explicit operational order creates and secures it.

## Required semantic checks

- exact YOLO class/order/normalized coordinates and empty output;
- uint16 PNG dimensions/dtype/0 background/IDs/object bijection;
- disconnected components share one object ID;
- overlap winner policy and full overlap-preserving masks;
- ROI/resize maps to original image;
- absent optional stages do not fabricate fields;
- lower verbosity does not execute extra expensive stages;
- YAML rejects unsafe/path/device/model fields and resource attacks;
- one-image-only and config-only multipart cardinality;
- no persistent files after success/error/cancel;
- model/request state isolation across repeated/concurrent calls;
- physical GPU1 visibility and no GPU0 allocation in live tier.

Every OAP order names exact commands and broad/focused scope. “All tests passed”
means the entire named set ran and passed. Local success never substitutes for a
required GitHub check.

## Governance mechanics

When OAP governance helpers change, test that finalized orders must contain
exactly one deferred-adjudication decision; `NONE` creates no register mutation;
`APPEND CRIT-NNNN` requires complete threshold/adversarial-reasoning fields;
`append_critical.py` rejects duplicate/malformed/unfinished entries and preserves
all existing bytes; `check_state.py` reports duplicate IDs and latest human
adjudication without treating `DEFERRED`, `REJECTED`, or `CHANGE REQUIRED` as
accepted. These tests are CPU-only and must not become an excuse to add routine
critical entries.
