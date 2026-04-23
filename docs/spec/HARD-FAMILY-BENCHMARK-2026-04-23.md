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
Artifact folders:
- `artifacts/hard-family-synthetic-benchmark-2026-04-23/`
- `artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/`

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
- v1 judge pairwise accuracy: `0.897059`
- v1 judge false positive rate: `0.294118`
- v2 calibrated judge threshold: `0.360053`
- v2 judge pairwise accuracy: `0.985294`
- v2 judge false positive rate: `0.029412`
- sparse-success-absence pairwise accuracy: `0.5`
- simple progress pairwise accuracy: `0.147059`
- family-aware report: `report/family-report.md` + `report/family-report.png`

## Honest read on synthetic benchmark
The synthetic benchmark is now a much cleaner proof surface:
- the sparse-success baseline is still maximally blunt
- the simple progress baseline still ranks these hard families badly
- the composite judge now keeps the separation advantage *and* has a much better operating point after in-slice calibration

But:
- coverage is still narrow because the current labeler only marks a small subset as doomed
- the calibration story is still in-slice, so it is useful for debugging but not yet a publishable held-out operating point

## Real hard-family smoke
Artifact folders:
- `artifacts/hard-family-real-smoke-2026-04-23/`
- `artifacts/hard-family-real-smoke-2026-04-23-v2/`

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
- v1 judge pairwise accuracy: `0.872428`
- v1 judge false positive rate: `0.888889`
- v2 calibrated judge threshold: `0.384724`
- v2 judge pairwise accuracy: `0.959866`
- v2 judge false positive rate: `0.043478`
- v2 judge failure hit rate: `0.923077`
- pick-place-v3 failure-label coverage: `0.333333`
- family-aware report: `report/family-report.md` + `report/family-report.png`

## Honest read on real hard-family smoke
Real harder-than-random trajectories still matter, but the benchmark surface is now materially better:
- false positives dropped sharply once the judge became more patient on early engaged prefixes and the summary switched to a calibrated operating point
- `pick-place-v3` is no longer a blind spot; late grasp-without-transport prefixes now become labeled failures instead of disappearing into `at_risk`
- family-aware markdown + plots now make it obvious where the judge wins and where it still misses

But this is still not final because:
- the calibration is still in-slice rather than held-out
- `push-v3` remains the weak task and currently loses one labeled failure at the chosen threshold
- the failure labeler is still narrow and mostly catches late doomed cases rather than a richer recoverability spectrum

## Next moves
1. harden push-v3 so the calibrated judge does not miss the one labeled failure in the current real smoke
2. split in-slice calibration from held-out calibration so the threshold story is scientifically cleaner
3. widen failure labeling beyond narrow late-prefix doomed cases
4. add score-over-time replay plots on top of the family summary figure
