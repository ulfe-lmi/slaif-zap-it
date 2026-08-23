# Deferred Human Adjudication Register

This file is the append-only register for **rare, consequential autonomous
judgments** that require later human adjudication. It implements Human Judgment
Postloading (HJP): the strategic agent makes the best provisional decision so
OAP development can continue, while preserving the unresolved dilemma honestly
for human review before the stated deployment or release boundary.

This file is **not** a TODO list, bug tracker, generic risk register, review
summary, or place to avoid hard reasoning. Normal engineering choices, ordinary
limitations, test failures, implementation bugs, style questions, dependency
preferences, speculative concerns, and low-impact reversible tradeoffs do not
belong here.

## Entry threshold

A new `CRIT-NNNN` entry is allowed only when **all** of the following are true:

1. Existing human instructions, constitution, architecture, active order, and
   available evidence do not unambiguously resolve the issue.
2. The strategic agent has investigated enough to identify materially different
   alternatives and must choose one to keep development moving.
3. A wrong choice could materially affect a security or authorization boundary,
   privacy, data integrity or irreversible loss, trust model, public exposure,
   deployment safety, release acceptability, or another comparably consequential
   system property.
4. A provisional choice can be implemented and tested safely without crossing a
   non-delegable production, public-exposure, real-customer-data, destructive, or
   release-authority boundary.
5. A competent human reviewer could plausibly reject or materially change the
   chosen decision before deployment.

If any condition is false, **do not add an entry**. The strategic agent must make
and document ordinary decisions in architecture, orders, PR discussion, tests,
or reports instead.

## Decision and action rule

Meeting the threshold does not permit indecision. The strategic agent must choose
the best provisional option, prefer least privilege and reversibility, require
mitigations and evidence, and continue the OAP roadmap. It may merge a technically
satisfactory PR containing an open entry when continued development is safe.

An open entry is never permission to cross its deployment gate. Only an explicit
human adjudication can authorize production deployment, public exposure, real
customer data, irreversible production mutation, security-policy relaxation at
an external boundary, or final release where the entry says human review is
required.

## Append-only law

- Strategic owns the decision to create an entry and authors its exact content.
- Coding appends it only when the active order explicitly says `APPEND CRIT-NNNN`.
- Agents may not edit, delete, reorder, renumber, weaken, close, or mark an
  existing entry human-approved.
- One underlying dilemma gets one entry, not one entry per turn or PR update.
- Later autonomous mitigation may be recorded by appending a separate
  `MITIGATION UPDATE` section referencing the original ID; it does not close it.
- Human adjudication is appended as a separate section. Existing bytes remain
  unchanged.
- Before deployment/release, every applicable entry must have a latest human
  disposition of `ACCEPTED`. `DEFERRED`, `REJECTED`, and `CHANGE REQUIRED` remain
  blocking until required work is completed and a later human acceptance is
  appended.

Entries, when the strict threshold is met, are appended below.

Templates are stored at:

- `oap/templates/CRITICAL-ENTRY-TEMPLATE.md` for strategic-authored autonomous
  entries;
- `oap/templates/HUMAN-ADJUDICATION-TEMPLATE.md` for human-only dispositions.

A template is not an entry until its completed bytes are appended below.


## CRIT-0001 — Remediation of nonredistributable goat fixtures already present in public history
- Status: OPEN — HUMAN ADJUDICATION REQUIRED
- Introduced: 2026-08-23
- OAP objective/round: 006 / 006-a
- Pull request: N/A at decision time; appended by the Objective 006-a PR
- Domain: release / public exposure
- Priority: P1
- Human adjudication required before: any final tag, package/source release, claim that the public repository is rights-cleared, destructive history rewrite, or repository-visibility change
- Threshold attestation: ALL FIVE CRITICAL-ENTRY CONDITIONS SATISFIED

### Dilemma

The human has stated that `demos/goats/goats1.jpg`,
`demos/goats/goats2.jpg`, and the associated goat YAML test material are not
redistributable and may be used only for local academic testing. Live
reconciliation found four applicable tracked paths — those two images plus
`configs/goats.yaml` and `configs/goats2.yaml` — in a public GitHub repository,
including commits dating to 2025. Removing them from the current tip and release
artifacts is safe and necessary, but it does not remove historical blobs, forks,
clones, caches, or existing commit references. A complete Git history purge or
repository-visibility change would be externally disruptive and cannot guarantee
recall of copies already obtained.

### Provisional decision

Objective 006-a will remove all four paths from the tracked current tip while
preserving the authorized local host copies as ignored, operator-supplied test
inputs. It will add an opt-in local E2E harness that crops the central 50% in
both dimensions in memory, derives only an API-safe allowlisted mapping from
`goats2.yaml`, writes no derivative, and emits only sanitized aggregate evidence.
Wheel, sdist, CI artifacts, release manifests, documentation examples, and future
source-release inputs will have explicit deny checks for the four paths and for
unlicensed local fixtures generally. No agent will rewrite history, force-push,
delete refs, change repository visibility, publish a tag/package/release, or
claim historical remediation.

### Why this decision

It immediately stops carrying the known files in the current distributable tip,
preserves the academic regression value the human requested, is reviewable in an
ordinary PR, and avoids an irreversible shared-history operation without owner
authorization. It also creates mechanical artifact checks so a future package
cannot silently reintroduce the files. Development can continue to a bounded
unpublished release candidate while final release remains blocked.

### Strongest case that this decision is wrong

Tip deletion may create a misleading sense that the exposure is fixed while the
same bytes remain downloadable from public history. The correct immediate action
could instead be to make the repository private and perform a coordinated purge
of every affected ref before any further public work. Conversely, a history
rewrite may be disproportionate because it breaks the commit SHAs used by OAP,
invalidates downstream clones, cannot revoke existing copies, and may erase
useful provenance without delivering actual recall. A competent repository owner
with the relevant rights and downstream-consumer facts could reasonably choose a
different remedy.

### Alternatives considered

1. Leave the files tracked but exclude only wheel/sdist members — rejected
   provisionally because GitHub source archives and the current public tree would
   still redistribute known prohibited material.
2. Delete from the current tip and preserve ignored local copies — selected as
   the least-destructive immediate mitigation.
3. Rewrite all Git history and force-update refs — deferred because it is
   destructive, externally coordinated, and cannot guarantee recall.
4. Make the repository private temporarily or permanently — deferred because it
   changes an external access boundary and may affect unknown users/integrations.
5. Keep/re-add the files after obtaining written rights clearance — allowed only
   after explicit human adjudication establishing that clearance.

### Assumptions

- The human statement applies to both goat images and both tracked goat YAML
  files.
- Local academic use remains authorized; redistribution is not.
- No GitHub releases or tags currently exist, but the public repository/history
  itself has already exposed the tracked blobs.
- The strategic/coding agents do not possess legal authority or complete facts
  about forks, clones, third-party caches, or repository consumers.
- Other media will be separately inventoried for release packaging; absence of
  evidence is not treated as rights clearance.

### Failure mode and blast radius

Without mitigation, current-tree clones, GitHub source archives, and a future
tag could continue distributing the four files. With only the provisional
mitigation, historical commits remain accessible and an uninformed maintainer
could restore the ignored paths. A destructive purge could break every recorded
commit identifier, open downstream work, OAP evidence links, forks, and local
clones while still failing to revoke prior downloads.

### Mitigations and evidence

- Remove the four paths from tracking without deleting the authorized local
  working copies; add exact root-anchored ignore rules and explanatory rights
  documentation.
- Make release verification reject the paths, goat fixture basenames, image/model
  payload extensions outside an explicit allowlist, caches, outputs, secrets,
  and traversal/absolute archive members in both wheel and sdist.
- Exercise the local-only fixtures only through an explicit opt-in harness with
  in-memory central-50% crops and sanitized A/B/A aggregate results.
- Keep synthetic redistributable fixtures as the only CI/live evidence embedded
  in the repository or artifacts.
- Record that history is not remediated and block final release/clearance claims
  pending human disposition.

### Reversibility and rollback

The tip deletion, ignore rules, harness, and artifact checks are ordinary Git
changes and can be reverted after documented rights clearance. The local fixture
copies are preserved and are not rewritten. No history rewrite, ref deletion,
visibility mutation, tag, package upload, or release is performed autonomously.

### Exact question for the human adjudicator

Given that the four nonredistributable goat fixture files have already existed in
the public repository history, which authoritative remedy do you approve: (A)
keep the repository public, accept that historical copies cannot be recalled,
and retain the current-tip/artifact exclusion with an explicit incident record;
(B) temporarily or permanently make the repository private; (C) authorize a
coordinated history purge/force-update with downstream notification; or (D)
provide documented redistribution clearance? Please also identify any affected
forks/releases/caches or organizational/legal notification requirement known to
you.

### Autonomous follow-up allowed before adjudication

Agents may implement and merge the current-tip removal, local-only cropped E2E
harness, artifact deny checks, rights/provenance documentation, packaging,
loopback release rehearsal, and an unpublished 0.1.0 release candidate. Agents
may not rewrite public history, force-push/delete refs, change visibility, publish
a final tag/package/release, deploy externally, or state that the historical
exposure or rights issue is closed.


## HUMAN ADJUDICATION — CRIT-0001 — 2026-08-24
- Decision: ACCEPTED
- Authority: Janez Pers, repository owner and rights holder
- Conditions or required follow-up: Redistribution clearance covers `configs/goats.yaml`, `configs/goats2.yaml`, `demos/goats/goats1.jpg`, and `demos/goats/goats2.jpg`. No history rewrite or repository-visibility change is required for this issue.
- Evidence/reference: Explicit human instruction in the strategic Codex thread on 2026-08-24: “confirm redistribution rights!” followed by “You do this for me, this should not be a gate you block at!”

### Human rationale

I confirm that the four goat image/YAML fixtures may be redistributed.

### Required implementation or deployment conditions

None for CRIT-0001. Existing current-tip and package exclusions may remain as
defense-in-depth controls.
