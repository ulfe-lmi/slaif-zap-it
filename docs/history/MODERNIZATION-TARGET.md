# Historical record: professional baseline target

> This modernization target has been implemented through the current package,
> API, runtime, release tooling, and documentation. It remains only as planning
> provenance; see the [current documentation index](../README.md).

Preserve current algorithms/CLI while converging toward SLAIF service quality:

- `pyproject.toml`, importable package, stable CLI, separated dev/model/service
  dependencies and reproducible environment;
- CPU-only public CI, Ruff format/lint, pytest, coverage baseline/ratchet,
  optional typing, CodeQL and least GitHub permissions;
- audited/repaired existing tests plus new config/result/API tests;
- README, installation, configuration, algorithms, API contract, service
  datasheet, provenance, security, contributing and third-party notices;
- pinned model revisions/checksums outside Git; no weights/caches/results corpus;
- in-memory core before FastAPI; local loopback before packaging/gateway;
- tested-hardware/limitations claims only;
- HWP/HJP governance: rare material autonomous decisions are append-only in
  `CRITICAL.md`; applicable open entries require human disposition before
  deployment/release.

Do not combine baseline, core refactor, API and live deployment into one PR.
`everything.txt`, legacy shims, result artifacts and typo-named docs are audited
before removal/renaming; preserve compatibility or provide explicit migration.
