# LeWorldModel Judge demo artifact

## What this artifact is for
- show prefix-level score movement, not just endpoint success
- show where judge and baseline disagree
- show the evidence fields driving the current verdict surface

## Output inventory
- comparison table: `demo-artifact-comparison.csv`
- score timeline plot: `demo-artifact-timeline.png`

## Coverage snapshot
- prefix rows: `12`
- tasks: `push-v3`
- families: `doomed, weak`
- cutoffs: `0.25, 0.5, 0.75`

## Biggest baseline-vs-judge disagreements
- `push-v3-weak-ep-0` @ cutoff `0.25` (push-v3/weak) → judge `0.213876` vs baseline `0.862069`; gap `-0.648193`; label=`False` recoverability=`at_risk`
- `push-v3-weak-ep-1` @ cutoff `0.25` (push-v3/weak) → judge `0.213876` vs baseline `0.862069`; gap `-0.648193`; label=`False` recoverability=`at_risk`
- `push-v3-doomed-ep-0` @ cutoff `0.25` (push-v3/doomed) → judge `0.046997` vs baseline `0.6`; gap `-0.553003`; label=`False` recoverability=`at_risk`
- `push-v3-doomed-ep-1` @ cutoff `0.25` (push-v3/doomed) → judge `0.046997` vs baseline `0.6`; gap `-0.553003`; label=`False` recoverability=`at_risk`
- `push-v3-weak-ep-0` @ cutoff `0.5` (push-v3/weak) → judge `0.209989` vs baseline `0.689655`; gap `-0.479666`; label=`False` recoverability=`at_risk`

## Evidence decomposition example
Example prefix: `push-v3-weak-ep-0` / `push-v3` / cutoff `0.25`
- progress_proxy: `0.137931`
- distance_progress: `0.137931`
- target_distance_last: `0.833333`
- target_distance_best: `0.833333`
- in_place_score: `0.166667`
- grasp_signal_peak: `0.0`
- success_signal_peak: `0.0`
- reward_density: `0.15`
- stall_score: `0.862069`
- judge_on_track_score: `0.106006`
- judge_implausibility_score: `0.10681`
- judge_uncertainty_score: `0.34868`

## Read it honestly
- this is still a score-surface showcase, not a faithful JEPA world model
- disagreement rows are useful because they show where sparse reward misses prefix state
- if the timeline looks clean but disagreement rows are nonsense, the artifact is still weak

## Provenance rule
- every claim in this demo should be traceable back to the comparison CSV plus the underlying prefixes/baselines/judge JSONL files
