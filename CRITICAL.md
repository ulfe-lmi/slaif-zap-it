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
