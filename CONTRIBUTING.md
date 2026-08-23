# Contributing to ZAP-IT

Thanks for considering a contribution. ZAP-IT is being modernized into a
professionally tested package; please keep changes consistent with that goal.

## Development setup

CPU-only tooling (no torch/GPU downloads required):

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The heavy GPU stack lives in the operator-managed pip environment described in
[INSTALL.md](INSTALL.md); do not add model/runtime weights to Git.

## Ground rules

- Run the canonical checks before every push:

  ```bash
  .venv/bin/ruff format --check .
  .venv/bin/ruff check .
  .venv/bin/python -m build --wheel
  .venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing
  ```

- Treat uploaded images/YAML as hostile input in anything user-facing; see
  [SECURITY.md](SECURITY.md).
- Never commit secrets, credentials, model weights, datasets or generated
  result artifacts.
- Keep documentation truthful: do not claim API/GPU service readiness that the
  code does not yet have.
- Tests must not require network access or CUDA. If you need heavier fixtures,
  mark them clearly and keep them skippable honestly (a skipped test never
  counts as a pass).

## Pull requests

- One logical change per PR; update docs/tests in the same PR as behavior.
- Explain intentional behavior changes with one-line rationales in the PR body.
- CI must be green on the PR head before review; the maintainer merges.
