# LeWorldModel Judge demo artifact

Judge mode: `hybrid_prefix_latent_judge`

## What this artifact is for
- show prefix-level score movement, not just endpoint success
- show where judge and baseline disagree
- show the evidence fields driving the current verdict surface

## Output inventory
- comparison table: `demo-artifact-comparison.csv`
- score timeline plot: `demo-artifact-timeline.png`
- push-v3 hard-family disagreement pack: `demo-artifact-push-v3-hard-disagreement-pack.csv`
- episode score replay table: `demo-artifact-score-replay.csv`

## Coverage snapshot
- prefix rows: `90`
- tasks: `pick-place-v3, push-v3, reach-v3`
- families: `expert, misleading`
- cutoffs: `0.25, 0.5, 0.75`

## Biggest baseline-vs-judge disagreements
- `reach-v3-expert-ep-3` @ cutoff `0.25` (reach-v3/expert) → judge `0.194212` vs baseline `1.0`; gap `-0.805788`; label=`False` recoverability=`at_risk`
- `reach-v3-expert-ep-1` @ cutoff `0.25` (reach-v3/expert) → judge `0.196216` vs baseline `1.0`; gap `-0.803784`; label=`False` recoverability=`at_risk`
- `reach-v3-expert-ep-2` @ cutoff `0.25` (reach-v3/expert) → judge `0.202341` vs baseline `1.0`; gap `-0.797659`; label=`False` recoverability=`at_risk`
- `push-v3-expert-ep-3` @ cutoff `0.5` (push-v3/expert) → judge `0.208465` vs baseline `0.999998`; gap `-0.791533`; label=`False` recoverability=`at_risk`
- `push-v3-expert-ep-2` @ cutoff `0.5` (push-v3/expert) → judge `0.216647` vs baseline `0.999998`; gap `-0.783351`; label=`False` recoverability=`at_risk`

## Push-v3 hard-family disagreement pack
- this pack is intentionally family-diverse so the demo is not just synthetic weak/doomed cherry-picking
- `push-v3-expert-ep-3` (expert) @ cutoff `0.5` → judge `0.208465` vs baseline `0.999998`; gap `-0.791533`; uncertainty `0.642034`; recoverability=`at_risk`
- `push-v3-misleading-ep-4` (misleading) @ cutoff `0.25` → judge `0.24533` vs baseline `0.999999`; gap `-0.754669`; uncertainty `0.396632`; recoverability=`at_risk`
- `push-v3-expert-ep-2` (expert) @ cutoff `0.5` → judge `0.216647` vs baseline `0.999998`; gap `-0.783351`; uncertainty `0.644907`; recoverability=`at_risk`
- `push-v3-expert-ep-0` (expert) @ cutoff `0.5` → judge `0.22137` vs baseline `0.999999`; gap `-0.778629`; uncertainty `0.653931`; recoverability=`at_risk`
- `push-v3-expert-ep-1` (expert) @ cutoff `0.5` → judge `0.223752` vs baseline `0.999999`; gap `-0.776247`; uncertainty `0.649142`; recoverability=`at_risk`
- `push-v3-expert-ep-4` (expert) @ cutoff `0.5` → judge `0.224923` vs baseline `0.999999`; gap `-0.775076`; uncertainty `0.639903`; recoverability=`at_risk`
- `push-v3-misleading-ep-0` (misleading) @ cutoff `0.25` → judge `0.246025` vs baseline `0.999999`; gap `-0.753974`; uncertainty `0.399088`; recoverability=`at_risk`
- `push-v3-misleading-ep-1` (misleading) @ cutoff `0.25` → judge `0.252884` vs baseline `0.999999`; gap `-0.747115`; uncertainty `0.399675`; recoverability=`at_risk`

## Score-over-time replays
- the replay CSV is one row per episode with per-cutoff judge/baseline/sparse signals and first-difference deltas so reviewers can audit drift, not just means
- `reach-v3-expert-ep-3` (reach-v3/expert) success=`False` → judge `0.194 -> 0.371 -> 0.518` | baseline `1.000 -> 1.000 -> 1.000` | sparse `0.000 -> 0.000 -> 0.000` | latest label=`doomed`
- `reach-v3-expert-ep-1` (reach-v3/expert) success=`False` → judge `0.196 -> 0.370 -> 0.517` | baseline `1.000 -> 1.000 -> 1.000` | sparse `0.000 -> 0.000 -> 0.000` | latest label=`doomed`
- `reach-v3-expert-ep-2` (reach-v3/expert) success=`False` → judge `0.202 -> 0.382 -> 0.529` | baseline `1.000 -> 1.000 -> 1.000` | sparse `0.000 -> 0.000 -> 0.000` | latest label=`doomed`
- `push-v3-expert-ep-3` (push-v3/expert) success=`False` → judge `0.263 -> 0.208 -> 0.522` | baseline `1.000 -> 1.000 -> 1.000` | sparse `0.000 -> 0.000 -> 0.000` | latest label=`doomed`
- `push-v3-expert-ep-2` (push-v3/expert) success=`False` → judge `0.266 -> 0.217 -> 0.509` | baseline `1.000 -> 1.000 -> 1.000` | sparse `0.000 -> 0.000 -> 0.000` | latest label=`doomed`

## Evidence decomposition example
Example prefix: `reach-v3-expert-ep-3` / `reach-v3` / cutoff `0.25`
- progress_proxy: `0.0`
- distance_progress: `0.0`
- target_distance_last: `0.431899`
- target_distance_best: `0.301097`
- in_place_score: `0.151018`
- grasp_signal_peak: `1.0`
- success_signal_peak: `0.0`
- reward_density: `1.098271`
- stall_score: `1.0`
- latent_mismatch_score: `0.028388`
- latent_alignment_score: `0.971612`
- judge_on_track_score: `0.206702`
- judge_implausibility_score: `0.281473`
- judge_uncertainty_score: `0.647911`

## Read it honestly
- this is still a score-surface showcase, not a faithful JEPA world model
- disagreement rows are useful because they show where sparse reward misses prefix state
- if the timeline looks clean but disagreement rows are nonsense, the artifact is still weak

## Provenance rule
- every claim in this demo should be traceable back to the comparison CSV plus the underlying prefixes/baselines/judge JSONL files
