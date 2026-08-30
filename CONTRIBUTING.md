# Contributing

Thanks for considering a contribution. This is a small portfolio-style project;
the bar is: keep it clean, keep it tested, keep it honest.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows CMD
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install          # optional but recommended, runs the gate on every commit
```

## Before opening a PR

Run the same checks CI runs:

```bash
ruff check .
black --check .
mypy src data
pytest -v
```

All four must pass. If you're not able to verify something locally (e.g. no
GPU), say so in the PR description rather than skipping it silently.

## Guidelines

- Match existing patterns (typing style, bare-name imports in `src`/`data`,
  test structure) rather than introducing a new convention for one PR.
- Prefer the smallest change that achieves the goal; call out any trade-offs.
- Add or update tests for behavior you change. `tests/test_train_eval_smoke.py`
  is the reference for end-to-end changes; unit tests belong next to the
  module they cover.
- For anything touching training/evaluation numerics, show that behavior is
  unchanged (or intentionally changed) rather than asserting it.

## Reporting issues

Open a GitHub issue with steps to reproduce, the command you ran, and the
full error output. For dependency/security issues, mention the specific
package and version.
