# Contributing

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
```

This installs the package in editable mode plus all dev and benchmark dependencies.

## Quality gates

Every PR must pass all five gates. CI runs the same commands:

```bash
uv lock --check          # lockfile in sync with pyproject.toml
uv run ruff check .      # lint
uv run ruff format --check .
uv run mypy src          # strict typing on the library
uv run pytest -q         # test suite
```

## Claim discipline

This repo makes narrow, evidence-backed claims and marks everything else as a
gap or roadmap item. Before touching README, docs, or docstrings, read
[docs/rfcs/RFC-009-public-positioning-and-claim-discipline.md](docs/rfcs/RFC-009-public-positioning-and-claim-discipline.md).
In short: the shipped judge is a heuristic + observation-space latent proxy,
not a learned JEPA model — prose must never suggest otherwise, and every
reported number carries its `judge_mode` provenance.

## PR expectations

- Behavior changes require tests. Scoring math, labeling rules, metric values,
  and JSON/JSONL schemas are contracts — changing them needs an RFC, not just a diff.
- Do not regenerate files under `artifacts/` unless the PR includes a
  reproduction note: exact commands, seed, and why the regeneration was needed.
- Keep prose direct and free of hype. No "state of the art", no promises about
  future work stated as fact.
