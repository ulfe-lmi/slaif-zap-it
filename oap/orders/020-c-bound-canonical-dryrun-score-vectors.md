# OAP Work Order 020-c — Bound canonical dry-run score vectors

## Authority and exact state

- Continue Objective 020 on existing PR #76 and branch
  `oap/020-a-domain-neutral-clip-routing-pipeline`; create no new PR.
- Start from published report head
  `510af456da522b2268b20cac6e0800e6d4dc4e50`, whose correction parent is
  `701397b17a14b85be0a7a07e8de3c6aec4b1bea6`; remote `main` remains
  `cc325d5d97acefe7624aecfe9fa157dbf37ce600` unless fresh reconciliation shows
  an external change, in which case report it and stop before mutation.
- Preserve all prior orders/reports. Publish this exact order, set `oap/active`
  to `020-c`, make one minimal implementation/test commit, and then one
  report-only `OAP report 020-c (SELF)` commit changing only
  `oap/reports/020-c-report.md`.
- No merge/auto-merge, service/GPU/model/cache/network/key/host mutation, live
  inference, deployment, dependency change, or `CRITICAL.md` action.
- Critical-register action: NONE.

## Defect

`_DryRunClipFilter` currently computes each canonical simulated score as
`0.80 - 0.03 * label_index - 0.01 * candidate_index`. Although finite, values
for sufficiently late candidates fall below `-1.0`. Public
`ClipRoutingDiagnostic` correctly constrains complete CLIP score vectors to
finite cosine-like values in `[-1, 1]`. A maximum-size canonical dry-run can
therefore generate internally contradictory response evidence even though the
configuration is valid.

## Required correction

- Make every canonical dry-run per-label score a deterministic finite float in
  the closed interval `[-1.0, 1.0]` for every admitted candidate ordinal and all
  1..32 configured labels.
- Keep configured label order, deterministic winner/tie-breaking, complete
  vectors, configured winning prompt, stable source identity, request-local
  label refresh, and legacy noncanonical dry-run behavior unchanged.
- Prefer a transparent bounded construction over relying on downstream schema
  rejection. Do not silently omit a label or candidate.
- Add a direct CPU boundary test using 256 candidates and 32 canonical labels.
  Assert every vector has all 32 keys in configuration order, all values are
  finite and within `[-1, 1]`, winner/score/prompt agree with the vector, and
  `ClipRoutingDiagnostic` can validate representative first and last candidates.
- Retain the existing canonical dry-run pipeline test. Add no model imports or
  GPU requirements.
- If the public docs describe the simulated score range, reconcile them;
  otherwise do not churn documentation.

## Preservation and verification

- Do not change real CLIP scoring, raw crop pixels, routing policy, BLIP3,
  geometry, SAM2, artifact behavior, schemas, presets, or service policy.
- Run the focused dry-run/routing/schema tests, full default and coverage suites,
  formatting, lint, compile, documentation, diff check, package/release parity,
  twine, systemd, secret scans, and isolated direct/sdist install smokes.
- Require all seven checks successful on the exact implementation SHA, then all
  seven on the report-only SELF head before FIFO signaling.
- Report exact files, commands/results, SHAs, PR state, and strongest remaining
  reason not to merge. Do not claim live model accuracy.

