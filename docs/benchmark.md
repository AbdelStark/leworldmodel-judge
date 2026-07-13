# Benchmark

The benchmark contract, the checked-in results with full provenance, and the exact commands that
regenerate them. The contract is the product surface: the judge is only as credible as the
evaluation protocol around it.

## Contract

### Locked v1 slice

Environment family: **Meta-World**. Locked tasks:

- `reach-v3` — simplest contact-light manipulation task; good for debugging prefix scoring and
  progress proxies
- `push-v3` — longer-horizon object interaction and recoverability failure; useful for showing
  that sparse reward often reveals failure too late
- `pick-place-v3` — richest of the trio; most likely to expose partial progress vs eventual
  failure mismatch

This trio is small enough to ship and diverse enough to prevent the artifact from looking like a
one-task toy.

### Task

Given a partial manipulation rollout prefix, predict success likelihood, failure likelihood,
recoverability, implausibility, and optional uncertainty.

### Prefix cutoffs

Prefixes are evaluated at `0.25`, `0.50`, and `0.75` — fractions of the episode horizon.

### Labels

Each prefix carries `final_success_label`, `prefix_failure_label` (binary: trajectory already
effectively doomed at this prefix), and `prefix_recoverability_label` (`recoverable` / `at_risk` /
`doomed`). Labels come from environment-truth signals where possible, plus documented task
heuristics (see [method.md](method.md#labeling-rules)).

### Policy families

Trajectory families are first-class benchmark design, not an implementation detail:

- `expert` — scripted expert policy
- `weak` — slower but ultimately successful (degraded expert)
- `doomed` — strong early progress, then regression / unrecoverability
- `misleading` — shaping-heavy partial progress without success
- `random` — random actions

The synthetic generator guarantees these failure modes by construction; the Meta-World collector
approximates them by corrupting a scripted expert policy per family.

### Required baselines

- terminal success only
- cumulative sparse reward
- heuristic progress proxy

### Primary metrics

- early failure detection accuracy (hit rate at the chosen threshold)
- pairwise ranking accuracy (= AUROC; one statistic serialized under two keys, see
  [contracts.md](contracts.md#summary-json-summaryjson)) for failure-labeled vs
  non-failure-labeled prefixes — non-failure includes `at_risk`, not just `recoverable`
- average precision for failure detection under sparse positives
- false positive rate at the chosen failure threshold
- threshold provenance (fixed, in-slice, or held-out)
- calibration notes if uncertainty is emitted

Two scoping notes on "early". All metrics pool the 0.25/0.50/0.75 cutoffs; `summary.json` slices
by task and family but not yet by cutoff, so earliness is currently structural — a verdict exists
at prefix time, where sparse reward is silent by definition — and is not measured as detection
lead time. That per-cutoff slice is a known gap. And the labeling rules cannot emit `doomed`
before fraction 0.5 (`labels.py` gates most push/pick-place cases at 0.75), which bounds any
earliness claim from below.

### Calibration provenance rule

Every summary states whether its threshold is **fixed**, **in-slice**, or **held-out**. Held-out
runs must name the calibration families, the evaluation families, and the failure / non-failure
counts of each cohort. See [method.md](method.md#calibration-protocol).

### Non-negotiables

A serious result from this repo must have all five:

1. **prefix-level labels** that are inspectable and task-aware
2. **baseline vs judge comparison** on the same records
3. **family-aware slices** instead of a single aggregate headline
4. **calibration provenance** stating whether thresholds are fixed, in-slice, or held-out
5. **artifact files** that let someone inspect rollouts, prefixes, judge outputs, and summary
   metrics without rerunning the full stack

The judge must be compared against sparse reward. No judge-only victory lap. Credibility comes
from showing where the judge wins, where it loses, and why the evidence decomposition makes those
outcomes believable — not from one topline number.

### Maturity ladder

- **Debug-grade** — synthetic hard-family benchmark, in-slice threshold recommendation
- **Showcase-grade** — real hard-family run with family report, honest writeup of weak tasks,
  replay-visible score trajectories
- **Publishable-grade** — held-out calibration, wider failure taxonomy, task-level ablations, and
  a defensible claim that the operating point generalizes beyond the slice used to choose it

The current headline artifact reaches showcase-grade with a real held-out threshold story; it is
not publishable-grade (see the caveats below).

## Results

### Headline: held-out family split, real Meta-World

Source: `artifacts/hard-family-real-held-out-2026-04-28/summary-composite.json` — the
prefix-only composite judge, which reads nothing past the cutoff. Judge rows
(`judge-composite.jsonl`) carry `"judge_mode": "composite_prefix_judge"` (CLI
`--mode heuristic_surprise`). Threshold `0.298006` (`.thresholds.judge_failure_threshold`),
calibration mode `held_out_family_split` (`.calibration.judge.mode`), calibrated **only** on the
`weak` + `doomed` families and evaluated **only** on `expert` + `misleading`
(`.calibration.provenance.{calibration_families,evaluation_families}`, `family_overlap: false`).
Calibration cohort: 18 prefixes, 9 failure / 9 non-failure. Evaluation cohort: 18 prefixes,
8 failure / 10 non-failure (`.calibration.provenance`).

| Metric (evaluation slice, n=18) | Judge (composite) | Sparse-reward baseline | Progress baseline |
|---|---:|---:|---:|
| Failure hit rate | **1.00** | 1.00 | 1.00 |
| False positive rate | **0.10** | 1.00 | 1.00 |
| Pairwise ranking accuracy (= AUROC) | **1.00** | 0.50 | 0.556 |
| Average precision | **1.00** | 0.444 | 0.485 |

Calibration asymmetry: only the judge threshold is calibrated. The progress baseline keeps its
fixed default (`progress_failure_threshold` `0.2`, `.calibration.progress.mode:
"fixed_progress_baseline"`) and the sparse signal is binary. The pairwise-ranking and
average-precision rows are threshold-free and therefore calibration-fair; the false-positive row
compares a calibrated judge operating point against uncalibrated baseline defaults.

JSON paths, all under `.overall` of the summary above: judge `judge_failure_hit_rate`,
`judge_false_positive_rate`, `judge_pairwise_accuracy` / `judge_auroc` (aliases of one
statistic), `judge_average_precision`; sparse baseline `baseline_sparse_absence_*`; progress
baseline `baseline_progress_*` (pairwise `0.55625`, AP `0.485243`).

Slices: per-family judge FPR is `0.2` for `expert` and `0.0` for `misleading`
(`.families.<family>.judge_false_positive_rate`); per-task judge FPR is `0.0` for `reach-v3` and
`push-v3` and `0.25` for `pick-place-v3` — the single false positive
(`.tasks.<task>.judge_false_positive_rate`); judge hit rate is `1.0` on every task.

The hybrid replay-time variant (`summary.json`, threshold `0.311141`; judge rows
`judge-hybrid.jsonl`, `"judge_mode": "hybrid_prefix_latent_judge"`, CLI `--mode hybrid_surprise`)
reproduces the identical table, per-family and per-task slices included, on the same split. Its
latent-mismatch feature is computed against post-cutoff observations by construction, so it is a
replay/triage signal, not a cutoff-time verdict; on this slice it moved `failure_score` by at
most a few hundredths (max +0.036, mean +0.011) and changed no reported number. See
[method.md](method.md#cutoff-time-vs-replay-time-judging).

**Mandatory caveats — read these with the table:**

1. The evaluation slice is 18 prefixes from 12 episodes. This is small.
2. Both baselines also reach hit rate 1.0. The judge's win is entirely in false positive rate and
   ranking quality, not in raw detection.
3. The rollouts in this run are the same capture as the earlier 2026-04-23 real smoke runs
   (byte-identical file). "Held-out" refers to the family split and threshold provenance, not to
   new episodes.
4. The 1.0 pairwise / AUROC / AP values are perfect-separation results on a tiny slice. This
   artifact proves the wiring and the provenance story, not broad generalization.
5. The push-v3 label rules were revised in the same commit (`5e0b872`, 2026-04-28) that produced
   this artifact, against this same byte-identical capture, after observing an earlier judge miss
   on it (see History below). The held-out property applies to the threshold, not to the label
   rules: label rules and headline artifact are not independent. Independent validation requires
   a fresh capture collected with the rules frozen ([roadmap.md](roadmap.md)).
6. The labels and the judge consume the same evidence family, so judge-vs-label agreement is
   partly circular; a single shared-family feature (distance regret) ranks nearly as well as the
   judge on these slices. See [method.md](method.md#label-circularity).

### Fresh capture at 5×: held-out family split on new episodes

Source: `artifacts/hard-family-real-fresh-capture-2026-07-13/summary-composite.json` — composite
judge (`judge_mode: composite_prefix_judge`). This is the first capture collected **after** the
label rules were frozen at 0.2.0, and it directly addresses headline caveats 3 and 5: new
episodes (base seed `1013`, not the 2026-04-23 capture) collected with rules that were not
revised against them. Collected on Hugging Face Jobs (`cpu-upgrade`, ~25 s of stage wall time;
[RFC-011](rfcs/RFC-011-hf-jobs-pipeline.md)): `metaworld==3.0.0`, `--max-steps 75`, 5 episodes
per (task, family) — 60 episodes, 180 prefixes. Calibrated **only** on `weak` + `doomed`
(90 prefixes), evaluated **only** on `expert` + `misleading` (90 prefixes, 39 failure / 51
non-failure), mode `held_out_family_split`, `family_overlap: false`.

| Metric (evaluation slice, n=90) | Judge (composite) | Sparse-reward baseline | Progress baseline |
|---|---:|---:|---:|
| Failure hit rate | **0.949** | 1.00 | 1.00 |
| False positive rate | **0.137** | 1.00 | 1.00 |
| Pairwise ranking accuracy (= AUROC) | **0.978** | 0.50 | 0.554 |
| Average precision | **0.971** | 0.433 | 0.463 |

Two results worth stating plainly:

- **The operating point transfers.** Calibrating on this capture's `weak`+`doomed` cohort
  yields threshold `0.29768`, versus `0.298006` on the 2026-04-28 capture — the same operating
  point to three decimals, recovered from disjoint episodes.
- **Perfect separation does not survive 5× more prefixes.** The n=18 slice's 1.00/1.00/1.00
  rows relax to 0.949 hit rate (misses concentrate in `push-v3` `misleading`, per-task hit
  0.833 there), 0.137 FPR (false positives concentrate in `pick-place-v3`, per-task FPR 0.261),
  and 0.978 ranking accuracy. This is the honest degradation the roadmap's "larger real held-out
  slices" item existed to expose; the ranking-quality gap over both baselines is intact.

The hybrid replay-time variant (`summary.json`, threshold `0.313437`, `judge_mode:
hybrid_prefix_latent_judge`) scores hit rate `0.923`, FPR `0.137`, pairwise `0.976`, AP `0.970`
on the same split — replay/triage signal only, as ever (see
[method.md](method.md#cutoff-time-vs-replay-time-judging)).

Caveats: the cohort is still modest (90 evaluation prefixes from 60 episodes); headline caveats
1, 2, 4 and 6 apply unchanged (baselines also hit 1.0 — the win is FPR and ranking; the labeler
and judge still share evidence). Meta-World collection is not byte-reproducible across simulator
builds, so `rollouts.jsonl` is checked in as the input of record. Full cloud provenance —
pinned git sha `666670e`, package `0.2.0`, per-stage commands/timings, sha256 of every file —
ships in the artifact's `provenance.json`, and the published copy (with the `ml-intern`
operator/review transcripts) lives at
[`runs/metaworld-benchmark-20260713-080743-g666670e`](https://huggingface.co/datasets/abdelstark/leworldmodel-judge-runs/tree/main/runs/metaworld-benchmark-20260713-080743-g666670e).

### Secondary: synthetic hard-family benchmark (in-slice)

Source: `artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/summary.json`. Judge rows
(`judge.jsonl`) carry `"judge_mode": "composite_prefix_judge"` (CLI `--mode heuristic_surprise`).
Threshold `0.360053`, mode `in_slice_balanced_accuracy` (`.calibration.judge`), n=72 prefixes.
The rollouts are the original 2026-04-23 seed-7 capture (byte-identical under the current
collector). The checked-in labels were stale — they predated the push-v3 label hardening shipped
in 0.1.0 — so labels, judge/baseline rows, and this summary were regenerated 2026-07-10 under
rules unchanged since then and the current evaluator schema, which adds the AUROC /
average-precision fields and `.calibration.provenance`. Headline metrics are unchanged from the
original run.

| Metric (n=72, in-slice threshold) | Judge (composite) | Sparse-reward baseline | Progress baseline |
|---|---:|---:|---:|
| Failure hit rate | **1.00** | 1.00 | 0.00 |
| False positive rate | **0.029** | 1.00 | 0.176 |
| Pairwise ranking accuracy (= AUROC) | **0.985** | 0.50 | 0.147 |
| Average precision | **0.667** | 0.056 | 0.061 |

The calibration asymmetry noted for the headline table applies here too: the judge threshold is
tuned (in-slice on this run), the progress baseline keeps its fixed default
(`progress_failure_threshold` `0.2`), and the sparse signal is binary — the ranking and AP rows
are threshold-free, the FPR row is not.

JSON paths under `.overall`: `judge_failure_hit_rate`, `judge_false_positive_rate` (`0.029412`),
`judge_pairwise_accuracy` / `judge_auroc` (`0.985294`), `judge_average_precision` (`0.666667`),
`baseline_sparse_absence_pairwise_accuracy` / `baseline_sparse_absence_auroc` (`0.5`),
`baseline_sparse_absence_average_precision` (`0.055556`),
`baseline_progress_pairwise_accuracy` / `baseline_progress_auroc` (`0.147059`),
`baseline_progress_average_precision` (`0.060606`), `baseline_progress_hit_rate` (`0.0`),
`baseline_progress_false_positive_rate` (`0.176471`). The in-slice split is recorded at
`.calibration.provenance` (`calibration_families` = `evaluation_families` = `["all"]`,
`family_overlap: true`).

Caveats: failure-label coverage is only `0.055556` (`.overall.failure_label_coverage` — 4 labeled
failures out of 72 prefixes, so AP rests on 4 positives), the threshold is in-slice (debugging
calibration, not an operating point; `family_overlap: true` says so in the file), and synthetic
separation does not automatically prove real embodied value.

### Negative control: random-policy multitask smoke

The first real end-to-end pass (2026-04-23) collected 3 random-action episodes per task over the
locked trio (675 rollout steps, 27 prefixes, 11 failure-labeled). Judge
(`judge_mode: composite_prefix_judge`), sparse reward, and progress proxy each hit 11/11 labeled
failures — a three-way tie with no discriminative content. That run is why the hard policy
families exist: on random rollouts there is nothing to separate. Its artifact directory was
removed in the 0.2.0 curation (the run report and data are in git history, pre-0.2.0 tree); the
tie is the only result it produced.

### History: fixed and in-slice operating points (superseded)

Kept for the calibration story; these artifacts were removed in the 0.2.0 curation and their
numbers live in the pre-0.2.0 run reports in git history. All used
`judge_mode: composite_prefix_judge`.

- Synthetic, fixed threshold 0.5: pairwise `0.897059`, FPR `0.294118`. In-slice calibration (the
  v2 run above) moved that to `0.985294` / `0.029412` on identical rollouts.
- Real capture, fixed threshold 0.5: pairwise `0.872428`, FPR `0.888889`.
- Real capture, in-slice threshold `0.384724`: pairwise `0.959866`, FPR `0.043478`, hit rate
  `0.923077` — it missed one labeled `push-v3` failure, later fixed by the push-v3 label
  hardening; `pick-place-v3` failure-label coverage improved from `0.0` to `0.333333` in the same
  pass. This in-slice run should not be read as the repo's best threshold story; the held-out
  2026-04-28 artifact supersedes it.

## Reproduction

Both kept runs regenerate end-to-end with the CLI. Deterministic stages (prefixes onward)
reproduce the checked-in files byte-for-byte on the pinned CPython 3.11 (`.python-version`); on
3.12+ `prefixes.jsonl` can differ in the final float digit of a few `reward_density` values
(CPython's compensated `sum()`), while all rounded, claim-bearing outputs (judge, baselines,
summary) are unchanged. PNG plots require matplotlib (installed with the dev group); without it
the renderers fall back to SVG filenames, which will not match the checked-in PNGs.

### Synthetic benchmark (`artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/`)

The synthetic generator is seeded (default seed 7), so collection itself is reproducible:

```bash
uv run lewm-judge collect \
  --source synthetic \
  --task all \
  --episodes 2 \
  --policy-family expert,weak,doomed,misleading \
  --output artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/rollouts.jsonl

uv run lewm-judge prefixes \
  --input artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/rollouts.jsonl \
  --output artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/prefixes.jsonl

uv run lewm-judge baselines \
  --input artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/prefixes.jsonl \
  --output artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/baselines.jsonl

uv run lewm-judge judge \
  --input artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/prefixes.jsonl \
  --mode heuristic_surprise \
  --output artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/judge.jsonl

uv run lewm-judge evaluate \
  --prefixes artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/prefixes.jsonl \
  --baselines artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/baselines.jsonl \
  --judge artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/judge.jsonl \
  --output artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/summary.json

uv run lewm-judge report \
  --summary artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/summary.json \
  --output-dir artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/report
```

### Held-out real run (`artifacts/hard-family-real-held-out-2026-04-28/`)

Real collection requires Meta-World — repo users: `uv sync --group benchmark`; the `metaworld`
extra exists for pip/wheel installs. Collection is not byte-reproducible across simulator
versions; the checked-in `rollouts.jsonl` is the capture of record. The command that produced
it, for the record:

```bash
uv run lewm-judge collect \
  --source metaworld \
  --task all \
  --episodes 1 \
  --max-steps 75 \
  --policy-family expert,weak,doomed,misleading \
  --output artifacts/hard-family-real-held-out-2026-04-28/rollouts.jsonl
```

Everything downstream is deterministic from the checked-in capture. The composite (cutoff-time,
headline) judge and the hybrid (replay-time) judge are both scored on the same prefixes:

```bash
uv run lewm-judge prefixes \
  --input artifacts/hard-family-real-held-out-2026-04-28/rollouts.jsonl \
  --output artifacts/hard-family-real-held-out-2026-04-28/prefixes.jsonl

uv run lewm-judge latents \
  --rollouts artifacts/hard-family-real-held-out-2026-04-28/rollouts.jsonl \
  --prefixes artifacts/hard-family-real-held-out-2026-04-28/prefixes.jsonl \
  --output artifacts/hard-family-real-held-out-2026-04-28/latent-cache.jsonl

uv run lewm-judge baselines \
  --input artifacts/hard-family-real-held-out-2026-04-28/prefixes.jsonl \
  --output artifacts/hard-family-real-held-out-2026-04-28/baselines.jsonl

# Composite judge (cutoff-time) -> the headline summary-composite.json
uv run lewm-judge judge \
  --input artifacts/hard-family-real-held-out-2026-04-28/prefixes.jsonl \
  --mode heuristic_surprise \
  --output artifacts/hard-family-real-held-out-2026-04-28/judge-composite.jsonl

uv run lewm-judge evaluate \
  --prefixes artifacts/hard-family-real-held-out-2026-04-28/prefixes.jsonl \
  --baselines artifacts/hard-family-real-held-out-2026-04-28/baselines.jsonl \
  --judge artifacts/hard-family-real-held-out-2026-04-28/judge-composite.jsonl \
  --calibration-families weak,doomed \
  --evaluation-families expert,misleading \
  --output artifacts/hard-family-real-held-out-2026-04-28/summary-composite.json

# Hybrid judge (replay-time) -> summary.json
uv run lewm-judge judge \
  --input artifacts/hard-family-real-held-out-2026-04-28/prefixes.jsonl \
  --mode hybrid_surprise \
  --latent-cache artifacts/hard-family-real-held-out-2026-04-28/latent-cache.jsonl \
  --output artifacts/hard-family-real-held-out-2026-04-28/judge-hybrid.jsonl

uv run lewm-judge evaluate \
  --prefixes artifacts/hard-family-real-held-out-2026-04-28/prefixes.jsonl \
  --baselines artifacts/hard-family-real-held-out-2026-04-28/baselines.jsonl \
  --judge artifacts/hard-family-real-held-out-2026-04-28/judge-hybrid.jsonl \
  --calibration-families weak,doomed \
  --evaluation-families expert,misleading \
  --output artifacts/hard-family-real-held-out-2026-04-28/summary.json

uv run lewm-judge report \
  --summary artifacts/hard-family-real-held-out-2026-04-28/summary.json \
  --output-dir artifacts/hard-family-real-held-out-2026-04-28/report

uv run lewm-judge demo \
  --prefixes artifacts/hard-family-real-held-out-2026-04-28/prefixes.jsonl \
  --baselines artifacts/hard-family-real-held-out-2026-04-28/baselines.jsonl \
  --judge artifacts/hard-family-real-held-out-2026-04-28/judge-hybrid.jsonl \
  --families expert,misleading \
  --output artifacts/hard-family-real-held-out-2026-04-28/demo-artifact.md
```

See `artifacts/README.md` for the per-run manifest.

### Fresh capture (`artifacts/hard-family-real-fresh-capture-2026-07-13/`)

This run was collected and evaluated on Hugging Face Jobs
([RFC-011](rfcs/RFC-011-hf-jobs-pipeline.md)); the launcher pins the job's install to a pushed
commit and gates the published folder against the run contract:

```bash
uv run jobs/launch.py launch --preset metaworld-benchmark
# equivalent local pipeline: lewm-judge collect --source metaworld --task all --episodes 5 \
#   --max-steps 75 --seed 1013 --policy-family expert,weak,doomed,misleading, then the same
#   prefixes/latents/baselines/judge×2/evaluate×2 (--calibration-families weak,doomed
#   --evaluation-families expert,misleading)/report/demo chain as above.
```

Meta-World capture is not byte-reproducible across simulator builds, so a rerun reproduces the
protocol, not the bytes; the checked-in `rollouts.jsonl` is the input of record, everything
downstream regenerates deterministically from it, and `provenance.json` carries the exact stage
argv, timings, and per-file sha256 of the published run. The published copy is verified by
`uv run jobs/launch.py verify --run-id metaworld-benchmark-20260713-080743-g666670e`.

## Benchmark questions

1. Can the judge detect doomed trajectories earlier than sparse reward alone?
2. Can the judge rank partial trajectories better than sparse reward alone?
3. Does the judge provide useful uncertainty or implausibility information on ambiguous prefixes?

On the current slices: (1) is a tie on detection but a clear win on false positives — with the
caveat that "earlier" is not yet measured as detection lead time (no per-cutoff slicing exists;
see the scoping notes under Primary metrics), (2) is a win on both slices, (3) is emitted and
serialized but not yet independently validated.

## Related docs

- [method.md](method.md) — how the scores and labels are computed
- [contracts.md](contracts.md) — schemas for every file named above
- [roadmap.md](roadmap.md) — what it takes to reach publishable-grade
- [vision.md](vision.md) — success and failure criteria
