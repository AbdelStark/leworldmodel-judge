# Contracts

Every record schema and the CLI surface, in one place. These shapes are pinned by the test suite
and by the checked-in artifacts: a schema change here is a benchmark-contract change and must ship
with regenerated artifacts. Field names are exact.

## Storage stance

File-based storage only — no database layer:

- JSONL for rollouts, prefixes, baselines, judge rows, latent cache
- JSON for summaries
- CSV for benchmark and demo tables
- Markdown + PNG/SVG for reports and demo artifacts

Every result file must make it possible to reproduce the benchmark claim later. If a threshold
was chosen on the same benchmark slice, the result artifact says so plainly; in-slice tuning is
never presented as held-out calibration.

## Rollout step (`rollouts.jsonl`)

One JSON object per environment step:

| Field | Type | Notes |
|---|---|---|
| `episode_id` | str | e.g. `push-v3-doomed-ep-0` |
| `task_id` | str | one of the locked tasks |
| `timestep` | int | |
| `episode_horizon` | int | |
| `observation` | list[float] | flat vector; v1 does not store images |
| `action` | list[float] | |
| `reward` | float | |
| `done` | bool | |
| `success_label` | bool | episode-level ground truth, backfilled onto all steps |
| `info` | dict | optional; see sub-keys below |

`info` sub-keys consumed downstream: `obj_to_target`, `in_place_reward`, `near_object`,
`grasp_success`, `grasp_reward`, `success`, `unscaled_reward`, `source`, `policy_family`.

## Prefix record (`prefixes.jsonl`)

Produced by `lewm-judge prefixes`. One row per (episode, cutoff):

| Field | Type | Notes |
|---|---|---|
| `episode_id`, `task_id` | str | |
| `prefix_index` | int | number of steps in the prefix |
| `prefix_fraction` | float | cutoff fraction (0.25 / 0.50 / 0.75 by default) |
| `final_success_label` | bool | |
| `prefix_failure_label` | bool | effectively doomed at this cutoff |
| `prefix_recoverability_label` | str | `recoverable` / `at_risk` / `doomed` |
| `sparse_reward_prefix` | float | count of success events in the prefix, from `info.success` — not the dense reward sum |
| `policy_family` | str \| null | from `info.policy_family` |
| `progress_proxy` | float | |
| `distance_progress` | float \| null | |
| `target_distance_start` / `target_distance_last` / `target_distance_best` | float \| null | |
| `in_place_score`, `near_object_score` | float \| null | |
| `grasp_signal_peak`, `success_signal_peak` | float \| null | |
| `reward_density` | float \| null | |
| `stall_score` | float \| null | |

Missing Meta-World info produces `None` for the metric fields, never `0.0`. Zero is a measured
value; absence stays absent.

The canonical join key across prefixes, baselines, judge rows, and latent cache rows is
`(task_id, episode_id, prefix_fraction)` (`schema.prefix_key`).

## Baseline row (`baselines.jsonl`)

Produced by `lewm-judge baselines`:

`episode_id`, `task_id`, `policy_family`, `prefix_fraction`, `sparse_reward_score`,
`terminal_success_score`, `progress_proxy_score`.

## Judge rows

Produced by `lewm-judge judge`. All modes share the core keys; `judge_mode` names the
implementation, so provenance travels with every row.

### Composite (`--mode heuristic_surprise` → `"judge_mode": "composite_prefix_judge"`)

`episode_id`, `task_id`, `policy_family`, `prefix_fraction`, `on_track_score`, `failure_score`,
`implausibility_score`, `uncertainty_score`, `progress_evidence`, `distance_progress_evidence`,
`in_place_evidence`, `near_object_evidence`, `grasp_evidence`, `reward_evidence`,
`stall_evidence`, `judge_mode`.

### Hybrid (`--mode hybrid_surprise` → `"judge_mode": "hybrid_prefix_latent_judge"`)

All composite keys, plus: `latent_mismatch_score`, `latent_alignment_score`,
`context_latent_norm`, `predicted_future_latent_norm`, `actual_future_latent_norm`. Requires
`--latent-cache`; prefixes without a cache row degrade to the composite score (the CLI warns).
The latent fields derive from post-cutoff observations, so hybrid rows are replay-time, not
cutoff-time — see [method.md](method.md#cutoff-time-vs-replay-time-judging).

### Dummy (`--mode dummy` → `"judge_mode": "dummy"`)

`episode_id`, `task_id`, `prefix_fraction`, `on_track_score: 0.0`, `failure_score: 0.0`,
`implausibility_score: 0.0`, `uncertainty_score: 1.0`, `judge_mode`. A null judge for
sanity-checking the evaluation harness.

## Latent cache row (`latent-cache.jsonl`)

Produced by `lewm-judge latents`:

`episode_id`, `task_id`, `policy_family`, `prefix_fraction`, `prefix_index`,
`latent_cache_version` (currently `"v0.1"`), `context_latent`, `predicted_future_latent`,
`actual_future_latent` (float lists; predicted and actual have equal length),
`context_latent_norm`, `predicted_future_latent_norm`, `actual_future_latent_norm`,
`latent_alignment_score`, `latent_mismatch_score`.

These are observation-space proxies (mean-pooled windows + linear extrapolation), not learned
latents — see [method.md](method.md#hybrid-latent-judge-judge_mode-hybrid_prefix_latent_judge).

## Summary JSON (`summary.json`)

Produced by `lewm-judge evaluate`.

- `thresholds`: `judge_failure_threshold`, `progress_failure_threshold`
- `calibration.judge`: `recommended_threshold`, `hit_rate`, `false_positive_rate`,
  `balanced_accuracy`, `mode` (`"held_out_family_split"` or `"in_slice_balanced_accuracy"`),
  `cohort_stats` (`count`, `failure_labels`, `non_failure_labels`, `pairwise_accuracy`, `auroc`,
  `average_precision`), and for held-out runs `evaluation_stats` and `evaluation_cohort`
- `calibration.progress`: the fixed progress-baseline threshold block
  (`mode: "fixed_progress_baseline"`)
- `calibration.provenance`: `calibration_families` and `evaluation_families` (sorted lists),
  `family_overlap` (bool), `calibration_count`, `evaluation_count`,
  `calibration_failure_labels`, `calibration_non_failure_labels`, `evaluation_failure_labels`,
  `evaluation_non_failure_labels`
- `overall`, `families.<family>`, `tasks.<task>`: `count`, `failure_labels`,
  `non_failure_labels`, `failure_label_coverage`, and for each of judge /
  sparse-absence baseline / progress baseline: hits, false positives, hit rate, false positive
  rate, pairwise accuracy, AUROC, average precision (key pattern `judge_*`,
  `baseline_sparse_absence_*`, `baseline_progress_*`)

The `*_auroc` keys are definitional aliases of the corresponding `*_pairwise_accuracy` keys:
`judge_auroc` and `judge_pairwise_accuracy` (and the baseline counterparts) carry the same
tie-handled Mann–Whitney statistic, computed once. One metric, serialized under two names.

Degenerate slices (one label class, empty slice) report `null` metrics, never fabricated numbers.

**Held-out validity rule:** `held_out_family_split` is only emitted when calibration and
evaluation families are disjoint. If they overlap, the evaluator falls back to in-slice semantics
and `family_overlap: true` says so.

## Demo artifact bundle

Produced by `lewm-judge demo --output <stem>.md`. Sibling filenames are derived from the output
stem and are contract:

| File | Content |
|---|---|
| `<stem>.md` | markdown artifact; section headings are contract (below) |
| `<stem>-comparison.csv` | joined prefix/baseline/judge rows |
| `<stem>-timeline.png` (or `.svg` without matplotlib) | mean score timeline per cutoff |
| `<stem>-push-v3-hard-disagreement-pack.csv` | family-diverse push-v3 disagreement rows |
| `<stem>-score-replay.csv` | one row per episode, per-cutoff score replay |

Markdown section headings (pinned by tests): `Biggest baseline-vs-judge disagreements`,
`Push-v3 hard-family disagreement pack`, `Score-over-time replays`,
`Evidence decomposition example`.

Comparison CSV columns (exact header order):

```text
task_id, episode_id, policy_family, prefix_cutoff, prefix_index, baseline_metric, judge_metric,
sparse_reward_signal, judge_signal, success_label, prefix_failure_label,
prefix_recoverability_label, judge_on_track_score, judge_implausibility_score,
judge_uncertainty_score, baseline_vs_judge_gap, latent_mismatch_score, latent_alignment_score,
progress_proxy, distance_progress, target_distance_last, target_distance_best, in_place_score,
grasp_signal_peak, success_signal_peak, reward_density, stall_score
```

`baseline_metric` is `1 - progress_proxy_score` (a failure-direction baseline so the gap column
compares like with like); `judge_metric` is the judge `failure_score`. The latent columns are
always in the header and hold the empty string for non-hybrid judge rows. The disagreement pack
selects from the same joined rows, so it uses the same columns; it prefers family diversity over
raw gap magnitude.

Score-replay CSV: one row per episode with `task_id`, `episode_id`, `policy_family`,
`success_label`, `final_prefix_failure_label`, `final_prefix_recoverability_label`,
`max_abs_gap`, `max_judge_metric`, `max_baseline_metric`, and per-cutoff fields
`cutoff_<c>_{judge_metric,baseline_metric,sparse_reward_signal,gap,judge_delta,baseline_delta}`
where `<c>` is the cutoff with `.` replaced by `p` (e.g. `cutoff_0p75_judge_metric`).

Demo obligations (what the bundle must show): prefix state as a compact markdown replay, judge
score over time across cutoffs, final success vs failure label, baseline vs judge timing
difference, at least one disagreement example where sparse reward is late or blind, and one
evidence decomposition example. A viewer should understand the claim in under 30 seconds. The
artifact must say explicitly that this is a verifier-style judging surface, not yet a faithful
JEPA world model implementation.

## Family report bundle

Produced by `lewm-judge report --output-dir <dir>`: `family-report.md` (threshold provenance,
honesty notes, per-family metric table) and `family-report.png` (or `.svg` without matplotlib).

## CLI surface

One console entry point: `lewm-judge` (equivalently `python -m leworldmodel_judge`). Comma
lists are plain CSV strings. All paths come from flags; nothing is hardcoded.

| Subcommand | Flags (default) |
|---|---|
| `collect` | `--source {synthetic,metaworld}` (synthetic), `--task` (required; task id, comma list, or `all`), `--episodes` (5), `--output` (required), `--max-steps` (none; metaworld only), `--seed` (7), `--policy-family` (random; comma list from `random`, `expert`, `weak`, `doomed`, `misleading`) |
| `prefixes` | `--input` (required), `--output` (required), `--fractions` (`0.25,0.50,0.75`) |
| `latents` | `--rollouts` (required), `--prefixes` (required), `--output` (required) |
| `baselines` | `--input` (required), `--output` (required) |
| `judge` | `--input` (required), `--output` (required), `--mode {heuristic_surprise,hybrid_surprise,dummy}` (heuristic_surprise), `--latent-cache` (none) |
| `evaluate` | `--prefixes`, `--baselines`, `--judge`, `--output` (all required), `--calibration-families` (none), `--evaluation-families` (none) |
| `report` | `--summary` (required), `--output-dir` (required) |
| `demo` | `--prefixes`, `--baselines`, `--judge`, `--output` (all required), `--families` (none; comma list filter) |

`--source metaworld` requires Meta-World — repo users: `uv sync --group benchmark`; pip/wheel
users: the `metaworld` extra. Synthetic mode is stdlib-only.

## Synthetic fixture stance

The synthetic generator exists so the pipeline and CI can run schema-valid end-to-end without
Meta-World installed, and so failure families can be constructed with guaranteed semantics. The
synthetic hard-family benchmark is a controlled proof surface; it is not the real benchmark, and
synthetic separation is never claimed as real embodied value.

## Related docs

- [method.md](method.md) — how these records are computed
- [benchmark.md](benchmark.md) — the metrics and results built on these schemas
- [rfcs/RFC-008-result-and-artifact-contracts.md](rfcs/RFC-008-result-and-artifact-contracts.md) — why everything is file-based
