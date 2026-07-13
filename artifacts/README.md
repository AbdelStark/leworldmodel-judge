# Artifacts

Three curated benchmark runs. Everything the repo claims numerically traces back to a
`summary.json` (or `summary-composite.json`) in one of these directories.

Layout rule: `rollouts.jsonl` is the checked-in input of record for each run. Every
other file (prefixes, latent cache, baseline rows, judge rows, summary, report, demo
bundle) is derived from it and can be regenerated with the commands below. Regenerated
JSONL/JSON/CSV/markdown outputs are deterministic; the PNG plots require matplotlib
(`uv sync --extra viz`) and without it the pipeline falls back to SVG files with
different extensions.

Provenance rule: every judge row carries a `judge_mode` field. The real run scores with
`hybrid_prefix_latent_judge` (CLI mode `hybrid_surprise`), the synthetic run with
`composite_prefix_judge` (CLI mode `heuristic_surprise`). Quote numbers together with
their mode.

Earlier runs (the 2026-04-23 real and synthetic v1 smokes, the in-slice real v2, the
multitask smoke, `demo-pass-4`) were removed in the 0.2.0 restructure: they either
duplicated these rollouts byte-for-byte under superseded thresholds or carried no
discriminative content. Git history retains them.

---

## `hard-family-real-held-out-2026-04-28/` — canonical real run

**What it is.** Real Meta-World rollouts (reach-v3, push-v3, pick-place-v3; scripted
expert policies degraded per policy family: expert, weak, doomed, misleading; 1 episode
per task per family = 12 episodes, max 75 steps, 900 step rows, 36 prefixes at cutoffs
0.25/0.50/0.75), scored with the hybrid latent judge and evaluated with a
family-held-out threshold: the threshold is calibrated only on the `doomed`+`weak`
prefixes and evaluated only on the `expert`+`misleading` prefixes, with the split
recorded in `summary.json` under `.calibration.provenance`.

**Dates.** Rollouts captured 2026-04-23 (real Meta-World smoke capture). Prefix
labeling, hybrid judge scoring, held-out evaluation, report, and demo bundle produced
2026-04-28. The two PNG plots (`report/family-report.png`, `demo-artifact-timeline.png`)
were re-rendered 2026-07-10 so the checked-in bytes match the current matplotlib
(rendering drift only; every JSON/JSONL/CSV/markdown file is unchanged from
2026-04-28).

**Claim it supports.** At an equal failure hit rate of 1.0 for all three signals, the
judge separates from both baselines on false-positive rate and ranking under a
threshold whose calibration families are disjoint from the evaluation families.

**Headline numbers** (evaluation slice, n=18; all paths are jq paths into
`summary.json`):

| Metric | Judge (hybrid) | Sparse-reward baseline | Progress baseline |
|---|---:|---:|---:|
| Failure hit rate | 1.0 | 1.0 | 1.0 |
| False positive rate | 0.1 | 1.0 | 1.0 |
| Pairwise ranking accuracy | 1.0 | 0.5 | 0.55625 |
| AUROC | 1.0 | 0.5 | 0.55625 |
| Average precision | 1.0 | 0.444444 | 0.485243 |

- Judge: `.overall.judge_failure_hit_rate`, `.overall.judge_false_positive_rate`,
  `.overall.judge_pairwise_accuracy`, `.overall.judge_auroc`,
  `.overall.judge_average_precision`
- Sparse baseline: `.overall.baseline_sparse_absence_hit_rate`,
  `.overall.baseline_sparse_absence_false_positive_rate`,
  `.overall.baseline_sparse_absence_pairwise_accuracy`,
  `.overall.baseline_sparse_absence_auroc`,
  `.overall.baseline_sparse_absence_average_precision`
- Progress baseline: `.overall.baseline_progress_hit_rate`,
  `.overall.baseline_progress_false_positive_rate`,
  `.overall.baseline_progress_pairwise_accuracy`, `.overall.baseline_progress_auroc`,
  `.overall.baseline_progress_average_precision`
- Threshold: 0.311141 at `.thresholds.judge_failure_threshold`, mode
  `held_out_family_split` at `.calibration.judge.mode`
- Split provenance: `.calibration.provenance.calibration_families` =
  `["doomed","weak"]`, `.calibration.provenance.evaluation_families` =
  `["expert","misleading"]`, `.calibration.provenance.family_overlap` = `false`;
  cohorts 18/18 prefixes (9 fail / 9 non-fail calibration, 8 fail / 10 non-fail
  evaluation)
- Per-family judge FPR: expert 0.2 (`.families.expert.judge_false_positive_rate`),
  misleading 0.0 (`.families.misleading.judge_false_positive_rate`)
- Per-task judge FPR: reach-v3 0.0, push-v3 0.0, pick-place-v3 0.25 — the single false
  positive (`.tasks."pick-place-v3".judge_false_positive_rate`)

**Caveats — read before quoting these numbers.**

- n=18 evaluation prefixes from 12 episodes. This is a small slice.
- All three signals hit 1.0 failure hit rate, so the judge's win is entirely in
  false-positive rate and ranking (pairwise/AUROC/AP), not in detection.
- The rollouts here are byte-identical to the 2026-04-23 real smoke capture.
  "Held-out" refers to the family split and threshold provenance, not to new episodes.
- The 1.0 AUROC/AP values are perfect-separation results on a tiny slice. This
  artifact proves the wiring and the provenance story, not broad generalization.

**Contents.** `rollouts.jsonl` (input of record), `prefixes.jsonl`,
`latent-cache.jsonl`, `baselines.jsonl`, `judge-hybrid.jsonl`, `summary.json`,
`report/family-report.md`, `report/family-report.png`, `demo-artifact.md`,
`demo-artifact-comparison.csv`, `demo-artifact-push-v3-hard-disagreement-pack.csv`,
`demo-artifact-score-replay.csv`, `demo-artifact-timeline.png`.

**Regeneration.** From the repo root, after `uv sync --extra viz`:

```bash
RUN=artifacts/hard-family-real-held-out-2026-04-28

uv run lewm-judge prefixes  --input $RUN/rollouts.jsonl --output $RUN/prefixes.jsonl \
  --fractions 0.25,0.50,0.75
uv run lewm-judge latents   --rollouts $RUN/rollouts.jsonl --prefixes $RUN/prefixes.jsonl \
  --output $RUN/latent-cache.jsonl
uv run lewm-judge baselines --input $RUN/prefixes.jsonl --output $RUN/baselines.jsonl
uv run lewm-judge judge     --input $RUN/prefixes.jsonl --latent-cache $RUN/latent-cache.jsonl \
  --mode hybrid_surprise --output $RUN/judge-hybrid.jsonl
uv run lewm-judge evaluate  --prefixes $RUN/prefixes.jsonl --baselines $RUN/baselines.jsonl \
  --judge $RUN/judge-hybrid.jsonl \
  --calibration-families doomed,weak --evaluation-families expert,misleading \
  --output $RUN/summary.json
uv run lewm-judge report    --summary $RUN/summary.json --output-dir $RUN/report
uv run lewm-judge demo      --prefixes $RUN/prefixes.jsonl --baselines $RUN/baselines.jsonl \
  --judge $RUN/judge-hybrid.jsonl --families expert,misleading \
  --output $RUN/demo-artifact.md
```

The demo command derives its sibling files (`-comparison.csv`, `-timeline.png`,
`-push-v3-hard-disagreement-pack.csv`, `-score-replay.csv`) from the `--output` stem
automatically.

`rollouts.jsonl` itself came from the real collector (recorded command shape of the
2026-04-23 capture, default seed 7):

```bash
uv run lewm-judge collect --source metaworld --task all --episodes 1 --max-steps 75 \
  --policy-family expert,weak,doomed,misleading --output $RUN/rollouts.jsonl
```

Re-running it requires the `metaworld` extra (pulls MuJoCo) and produces a new physics
capture, not this file. That is why the rollouts are checked in.

---

## `hard-family-real-fresh-capture-2026-07-13/` — fresh capture at 5×, collected on HF Jobs

**What it is.** Real Meta-World rollouts collected **after** the 0.2.0 label-rule freeze
(base seed `1013` — new episodes, not the 2026-04-23 capture): 3 tasks × 4 policy families ×
5 episodes = 60 episodes, max 75 steps, 180 prefixes at cutoffs 0.25/0.50/0.75. Scored with
**both** judges on the same prefixes (`judge-composite.jsonl`, `judge-hybrid.jsonl`) and
evaluated with the family-held-out threshold (calibrated on `doomed`+`weak`, evaluated on
`expert`+`misleading`). The run was executed on Hugging Face Jobs
([RFC-011](../docs/rfcs/RFC-011-hf-jobs-pipeline.md)) with the package pinned to commit
`666670e`; this directory is a verbatim copy of the published, contract-verified run
[`runs/metaworld-benchmark-20260713-080743-g666670e`](https://huggingface.co/datasets/abdelstark/leworldmodel-judge-runs/tree/main/runs/metaworld-benchmark-20260713-080743-g666670e),
whose `agent/` folder also holds the `ml-intern` operator/review transcripts (not mirrored
here).

**Dates.** Collected, scored, evaluated, and published 2026-07-13 in a single cloud run.

**Claim it supports.** The held-out threshold story survives a fresh capture at 5× the
evaluation cohort: the calibrated operating point transfers (threshold `0.29768` here vs
`0.298006` on the 2026-04-28 capture), and the judge keeps a large ranking gap over both
baselines (pairwise 0.978 vs 0.50/0.554) while perfect separation honestly degrades
(hit rate 0.949, FPR 0.137 at n=90).

**Headline numbers** (evaluation slice, n=90; jq paths into `summary-composite.json`, judge
rows `judge-composite.jsonl`, `judge_mode: composite_prefix_judge`):

| Metric | Judge (composite) | Sparse-reward baseline | Progress baseline |
|---|---:|---:|---:|
| Failure hit rate | 0.948718 | 1.0 | 1.0 |
| False positive rate | 0.137255 | 1.0 | 1.0 |
| Pairwise ranking accuracy / AUROC | 0.977878 | 0.5 | 0.554299 |
| Average precision | 0.970748 | 0.433333 | 0.463337 |

- JSON paths mirror the canonical real run above (`.overall.judge_*`,
  `.overall.baseline_sparse_absence_*`, `.overall.baseline_progress_*`)
- Threshold: `0.29768` at `.thresholds.judge_failure_threshold`, mode
  `held_out_family_split`, `family_overlap: false`; cohorts 90/90 prefixes
  (39 failure / 51 non-failure in each)
- Hybrid replay-time variant (`summary.json`, `judge-hybrid.jsonl`,
  `judge_mode: hybrid_prefix_latent_judge`): threshold `0.313437`, hit `0.923077`,
  FPR `0.137255`, pairwise `0.97637`, AP `0.970098`
- Misses concentrate in `push-v3` (`.tasks."push-v3".judge_failure_hit_rate` = `0.833333`);
  false positives concentrate in `pick-place-v3`
  (`.tasks."pick-place-v3".judge_false_positive_rate` = `0.26087`)

**Caveats.**

- 90 evaluation prefixes from 60 episodes is bigger than n=18, still modest.
- Both baselines hit 1.0 detection; the judge's win remains false positives and ranking.
- Labels and judge share the evidence family (see
  [method.md](../docs/method.md#label-circularity)); the capture is fresh but the
  circularity caveat is unchanged.

**Contents.** The RFC-008 file surface (rollouts of record, prefixes, latent cache,
baselines, both judge files, both summaries, report, demo bundle) plus `provenance.json` —
cloud run provenance with the pinned git sha, installed package version, per-stage argv and
wall-clock, and sha256 of every file.

**Regeneration.** `uv run jobs/launch.py launch --preset metaworld-benchmark` reruns the
protocol on Hugging Face Jobs (new physics capture, new run id — Meta-World is not
byte-reproducible across simulator builds). Everything downstream of the checked-in
`rollouts.jsonl` regenerates deterministically with the same command shapes as the canonical
real run, substituting `RUN=artifacts/hard-family-real-fresh-capture-2026-07-13`, judge mode
`heuristic_surprise` → `judge-composite.jsonl` / `summary-composite.json` alongside the
hybrid pair, and `--calibration-families weak,doomed`.

---

## `hard-family-synthetic-benchmark-2026-04-23-v2/` — canonical synthetic run

**What it is.** Synthetic rollouts from the built-in family generator (3 tasks x 4
policy families x 2 episodes = 24 episodes, horizon 20, 480 step rows, 72 prefixes at
cutoffs 0.25/0.50/0.75), scored with the composite heuristic judge and evaluated with
an in-slice calibrated threshold. This is the controlled proof surface: failure
families are constructed, so the run isolates judge behavior from real-capture noise.

**Dates.** Rollouts captured 2026-04-23; they are identical to that capture (the synthetic
collector is deterministic at the default seed 7). Labels and all downstream files
(`prefixes.jsonl`, `baselines.jsonl`, `judge.jsonl`, `summary.json`, report) were regenerated
2026-07-10 under the 0.2.0 hardened labeling — the push-v3 hardening reclassified two `doomed`
0.25-cutoff recoverability labels (`at_risk` → `recoverable`; no failure label changed) and the
current evaluator schema added AUROC / average-precision fields and `.calibration.provenance`.
Headline metrics are unchanged; the original 2026-04-23 files are in git history.

**Claim it supports.** On constructed hard families, the composite judge keeps a 1.0
hit rate at a 0.029 false-positive rate and 0.985 pairwise accuracy, while the
sparse-reward baseline is maximally blunt (FPR 1.0, pairwise 0.5) and the progress
baseline ranks these families badly (pairwise 0.147) and detects nothing (hit rate
0.0).

**Headline numbers** (n=72; jq paths into `summary.json`):

| Metric | Judge (composite) | Sparse-reward baseline | Progress baseline |
|---|---:|---:|---:|
| Failure hit rate | 1.0 | 1.0 | 0.0 |
| False positive rate | 0.029412 | 1.0 | 0.176471 |
| Pairwise ranking accuracy | 0.985294 | 0.5 | 0.147059 |
| AUROC | 0.985294 | 0.5 | 0.147059 |
| Average precision | 0.666667 | 0.055556 | 0.060606 |

- Judge: `.overall.judge_failure_hit_rate`, `.overall.judge_false_positive_rate`,
  `.overall.judge_pairwise_accuracy`, `.overall.judge_auroc`,
  `.overall.judge_average_precision`
- Sparse baseline: `.overall.baseline_sparse_absence_hit_rate`,
  `.overall.baseline_sparse_absence_false_positive_rate`,
  `.overall.baseline_sparse_absence_pairwise_accuracy`,
  `.overall.baseline_sparse_absence_auroc`,
  `.overall.baseline_sparse_absence_average_precision`
- Progress baseline: `.overall.baseline_progress_hit_rate`,
  `.overall.baseline_progress_false_positive_rate`,
  `.overall.baseline_progress_pairwise_accuracy`, `.overall.baseline_progress_auroc`,
  `.overall.baseline_progress_average_precision`
- Threshold: 0.360053 at `.thresholds.judge_failure_threshold`, mode
  `in_slice_balanced_accuracy` at `.calibration.judge.mode`
- Split provenance: `.calibration.provenance.calibration_families` =
  `.calibration.provenance.evaluation_families` = `["all"]`,
  `.calibration.provenance.family_overlap` = `true` — the file itself records that
  this is an in-slice threshold, not a held-out one

**Caveats.**

- Failure-label coverage is 0.055556 (`.overall.failure_label_coverage`): only 4 of 72
  prefixes carry a failure label, so hit rate — and average precision — rest on 4
  positives.
- The threshold is in-slice (calibrated on the same slice it is evaluated on,
  `family_overlap: true`). It is a debugging operating point, not a held-out one; the
  held-out story lives in the real run above.
- Synthetic dynamics are the repo's own generator. This run demonstrates judge
  behavior under controlled failure modes, not transfer to real robotics data.

**Contents.** `rollouts.jsonl` (input of record), `prefixes.jsonl`, `baselines.jsonl`,
`judge.jsonl`, `summary.json`, `report/family-report.md`, `report/family-report.png`.

**Regeneration.** From the repo root, after `uv sync --extra viz`. Synthetic collection
is deterministic with the default seed (7), so even the rollouts are regenerable here;
the checked-in `rollouts.jsonl` remains the input of record.

```bash
RUN=artifacts/hard-family-synthetic-benchmark-2026-04-23-v2

uv run lewm-judge collect   --source synthetic --task all --episodes 2 \
  --policy-family expert,weak,doomed,misleading --output $RUN/rollouts.jsonl
uv run lewm-judge prefixes  --input $RUN/rollouts.jsonl --output $RUN/prefixes.jsonl \
  --fractions 0.25,0.50,0.75
uv run lewm-judge baselines --input $RUN/prefixes.jsonl --output $RUN/baselines.jsonl
uv run lewm-judge judge     --input $RUN/prefixes.jsonl --mode heuristic_surprise \
  --output $RUN/judge.jsonl
uv run lewm-judge evaluate  --prefixes $RUN/prefixes.jsonl --baselines $RUN/baselines.jsonl \
  --judge $RUN/judge.jsonl --output $RUN/summary.json
uv run lewm-judge report    --summary $RUN/summary.json --output-dir $RUN/report
```

`evaluate` is run without `--calibration-families`/`--evaluation-families` on purpose:
that is what makes this an in-slice threshold.
