# HARD-FAMILY-BENCHMARK-2026-04-23

## Why this exists
The earlier random-action smoke pass was structurally useful but strategically weak:
- sparse-success baseline, simple progress proxy, and composite judge all moved together too much
- there was no clean regime where the benchmark actually stressed the difference between them

This pass adds deliberately harder trajectory families to force the separation problem.

## Trajectory families
Implemented in `scripts/collect_rollouts.py` for both synthetic and Meta-World modes:
- `expert`
- `weak`
- `doomed`
- `misleading`
- `random`

The synthetic families are the first place where we can *guarantee* controlled failure modes:
- `weak`: slower but ultimately successful
- `doomed`: strong early progress, then regression / unrecoverability
- `misleading`: shaping-heavy partial progress without success

## New evaluation surfaces
`src/leworldmodel_judge/evaluate.py` now reports more than hit counts:
- failure hit rate
- false positive rate
- failure-label coverage
- pairwise ranking accuracy for:
  - judge failure score
  - sparse-success absence baseline
  - simple progress baseline

## Synthetic hard-family benchmark
Artifact folder:
- `artifacts/hard-family-synthetic-benchmark-2026-04-23/`

Command shape:
```bash
python scripts/collect_rollouts.py \
  --source synthetic \
  --task all \
  --episodes 2 \
  --policy-family expert,weak,doomed,misleading \
  --output artifacts/hard-family-synthetic-benchmark-2026-04-23/rollouts.jsonl
```

Summary highlights:
- total prefixes: `72`
- failure-labeled prefixes: `4`
- judge pairwise accuracy: `0.897059`
- sparse-success-absence pairwise accuracy: `0.5`
- simple progress pairwise accuracy: `0.147059`
- judge false positive rate: `0.294118`
- sparse-success-absence false positive rate: `1.0`
- simple progress false positive rate: `0.176471`

## Honest read on synthetic benchmark
This is the first benchmark slice where the three signals *actually diverge*:
- the sparse-success baseline is maximally blunt
- the simple progress baseline ranks these hard families badly
- the composite judge separates doomed-vs-slow-success much better

But:
- judge false positives are still too high
- failure coverage is still small because the current labeler only marks a narrow subset as doomed

So the synthetic benchmark now proves the *shape* of the claim, not the final quality bar.

## Real hard-family smoke
Artifact folder:
- `artifacts/hard-family-real-smoke-2026-04-23/`

Command shape:
```bash
python scripts/collect_rollouts.py \
  --source metaworld \
  --task all \
  --episodes 1 \
  --max-steps 75 \
  --policy-family expert,weak,doomed,misleading \
  --output artifacts/hard-family-real-smoke-2026-04-23/rollouts.jsonl
```

Summary highlights:
- total prefixes: `36`
- failure-labeled prefixes: `9`
- judge pairwise accuracy: `0.872428`
- sparse-success-absence pairwise accuracy: `0.5`
- simple progress pairwise accuracy: `0.68107`
- judge false positive rate: `0.888889`

## Honest read on real hard-family smoke
Real harder-than-random trajectories now exist.
That matters.

But this run is still not clean enough to claim victory because:
- false positives are extremely high
- task coverage is uneven (`pick-place-v3` still has no failure-labeled prefixes)
- the judge is better at ranking than the simple baselines, but not yet usable as a calibrated detector

## Next moves
1. reduce judge false positives on non-failure prefixes
2. improve task-aware label coverage, especially for `pick-place-v3`
3. add family-aware plots / tables instead of one aggregate JSON only
4. search for a better thresholded operating point instead of fixing everything at `0.5`
