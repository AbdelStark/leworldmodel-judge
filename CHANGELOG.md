# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- RFC-011 cloud benchmark runs on Hugging Face Jobs: `jobs/run_benchmark.py`
  (remote payload with `smoke` / `synthetic-benchmark` / `metaworld-benchmark`
  presets, cloud `provenance.json`, publication to the
  `abdelstark/leworldmodel-judge-runs` dataset repo), `jobs/launch.py`
  (preflight, launch, contract verify gate, dataset card generator),
  scoped `ml-intern` operator/reviewer steps (`jobs/intern_ops.py` +
  versioned prompts), and a manual-dispatch `hf-jobs-benchmark` workflow.
  The library itself is unchanged and stays stdlib-only.
- `artifacts/hard-family-real-fresh-capture-2026-07-13/`: the first Meta-World
  capture collected after the 0.2.0 label-rule freeze (seed 1013, 60 episodes,
  90 held-out evaluation prefixes vs 18), executed and published on Hugging
  Face Jobs. Composite judge holds hit rate 0.949 / FPR 0.137 / pairwise 0.978
  on the fresh slice and the calibrated threshold transfers across captures
  (0.29768 vs 0.298006). Results in `docs/benchmark.md`, manifest entry in
  `artifacts/README.md`.

## [0.2.0] - 2026-07-10

### Added

- Single `lewm-judge` CLI (also `python -m leworldmodel_judge`) with subcommands
  `collect`, `prefixes`, `latents`, `baselines`, `judge`, `evaluate`, `report`,
  and `demo`, replacing the 8 top-level scripts in `scripts/`.
- MIT `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, and GitHub Actions CI
  (lockfile check, ruff lint + format, mypy strict on `src`, pytest).
- Strict typing across the library, shipped with a `py.typed` marker.
- Prefix-only composite judge artifacts for the held-out run
  (`judge-composite.jsonl`, `summary-composite.json`, threshold 0.298006):
  identical headline metrics to the hybrid run on the same split, now the
  cutoff-time source of the headline table and the checked-in
  heuristic-vs-hybrid comparison. The demo bundle gained an explicit
  `judge_mode` provenance line.

### Changed

- Library modules renamed and split: `data.py` into `schema.py`, `prefixes.py`,
  and `labels.py`; `evaluate.py` into `metrics.py`; script logic moved into
  `collect.py`, `plotting.py`, `report.py`, `demo.py`, and `cli.py`.
- Core package is now stdlib-only; matplotlib moved to the optional `viz`
  extra and metaworld mirrored as the optional `metaworld` extra.
- Docs consolidated to five documents (`vision.md`, `method.md`,
  `benchmark.md`, `contracts.md`, `roadmap.md`) plus the RFC decision log
  (RFC-001..010).
- Checked-in artifacts curated to the two canonical runs:
  `hard-family-real-held-out-2026-04-28` and
  `hard-family-synthetic-benchmark-2026-04-23-v2`, with a manifest README.
- Refreshed the stale synthetic benchmark artifact: the checked-in labels
  predated the push-v3 label hardening shipped in 0.1.0, and the labeling
  rules did not change in 0.2.0. Regeneration under the unchanged rules
  (rollouts byte-identical at seed 7) flipped two push-v3 doomed-family
  recoverability labels (`at_risk` to `recoverable` at the 0.25 cutoff), left
  headline metrics unchanged, and added the deepened evaluator keys (AUROC/AP,
  calibration provenance). The two held-out PNGs were re-rendered for
  matplotlib drift (all data files unchanged).

### Removed

- `scripts/` directory (replaced by the CLI), `docs/spec/` working notes,
  superseded artifact runs, and the empty `demo/` and `results/` placeholders.

Scoring math, labeling rules, metric values, and JSON/JSONL schemas are
unchanged from 0.1.0; the kept artifacts remain valid outputs of the new code.

## [0.1.0] - 2026-04-28

### Added

- Initial benchmark pipeline: synthetic and Meta-World rollout collection,
  prefix building, failure labeling, baselines, composite heuristic judge,
  and prefix-level evaluation with cutoff metrics.
- Hybrid latent judge: observation-space latent cache with a
  linear-extrapolation predictor blended into the composite score.
- Held-out family calibration artifacts with family-split threshold
  provenance and replay reporting.
