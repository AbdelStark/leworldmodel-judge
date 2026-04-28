# LeWorldModel Judge demo artifact

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
- prefix rows: `18`
- tasks: `pick-place-v3, push-v3, reach-v3`
- families: `expert, misleading`
- cutoffs: `0.25, 0.5, 0.75`

## Biggest baseline-vs-judge disagreements
- `pick-place-v3-expert-ep-0` @ cutoff `0.25` (pick-place-v3/expert) → judge `0.17825` vs baseline `1.0`; gap `-0.82175`; label=`False` recoverability=`at_risk`
- `push-v3-expert-ep-0` @ cutoff `0.5` (push-v3/expert) → judge `0.221233` vs baseline `0.999886`; gap `-0.778653`; label=`False` recoverability=`at_risk`
- `reach-v3-expert-ep-0` @ cutoff `0.25` (reach-v3/expert) → judge `0.234667` vs baseline `1.0`; gap `-0.765333`; label=`False` recoverability=`at_risk`
- `reach-v3-misleading-ep-0` @ cutoff `0.25` (reach-v3/misleading) → judge `0.247049` vs baseline `1.0`; gap `-0.752951`; label=`False` recoverability=`at_risk`
- `push-v3-misleading-ep-0` @ cutoff `0.25` (push-v3/misleading) → judge `0.248628` vs baseline `0.999888`; gap `-0.75126`; label=`False` recoverability=`at_risk`

## Push-v3 hard-family disagreement pack
- this pack is intentionally family-diverse so the demo is not just synthetic weak/doomed cherry-picking
- `push-v3-expert-ep-0` (expert) @ cutoff `0.5` → judge `0.221233` vs baseline `0.999886`; gap `-0.778653`; uncertainty `0.643248`; recoverability=`at_risk`
- `push-v3-misleading-ep-0` (misleading) @ cutoff `0.25` → judge `0.248628` vs baseline `0.999888`; gap `-0.75126`; uncertainty `0.399276`; recoverability=`at_risk`
- `push-v3-expert-ep-0` (expert) @ cutoff `0.25` → judge `0.264463` vs baseline `0.999891`; gap `-0.735428`; uncertainty `0.411259`; recoverability=`at_risk`
- `push-v3-misleading-ep-0` (misleading) @ cutoff `0.5` → judge `0.282383` vs baseline `0.999888`; gap `-0.717505`; uncertainty `0.367817`; recoverability=`at_risk`
- `push-v3-misleading-ep-0` (misleading) @ cutoff `0.75` → judge `0.335691` vs baseline `0.999888`; gap `-0.664197`; uncertainty `0.338191`; recoverability=`doomed`
- `push-v3-expert-ep-0` (expert) @ cutoff `0.75` → judge `0.489028` vs baseline `0.999886`; gap `-0.510858`; uncertainty `0.561776`; recoverability=`doomed`

## Score-over-time replays
- the replay CSV is one row per episode with per-cutoff judge/baseline/sparse signals and first-difference deltas so reviewers can audit drift, not just means
- `pick-place-v3-expert-ep-0` (pick-place-v3/expert) success=`False` → judge `0.178 -> 0.322 -> 0.546` | baseline `1.000 -> 1.000 -> 1.000` | sparse `0.000 -> 0.000 -> 0.000` | latest label=`doomed`
- `push-v3-expert-ep-0` (push-v3/expert) success=`False` → judge `0.264 -> 0.221 -> 0.489` | baseline `1.000 -> 1.000 -> 1.000` | sparse `0.000 -> 0.000 -> 0.000` | latest label=`doomed`
- `reach-v3-expert-ep-0` (reach-v3/expert) success=`False` → judge `0.235 -> 0.415 -> 0.563` | baseline `1.000 -> 1.000 -> 1.000` | sparse `0.000 -> 0.000 -> 0.000` | latest label=`doomed`
- `reach-v3-misleading-ep-0` (reach-v3/misleading) success=`False` → judge `0.247 -> 0.427 -> 0.575` | baseline `1.000 -> 1.000 -> 1.000` | sparse `0.000 -> 0.000 -> 0.000` | latest label=`doomed`
- `push-v3-misleading-ep-0` (push-v3/misleading) success=`False` → judge `0.249 -> 0.282 -> 0.336` | baseline `1.000 -> 1.000 -> 1.000` | sparse `0.000 -> 0.000 -> 0.000` | latest label=`doomed`

## Evidence decomposition example
Example prefix: `pick-place-v3-expert-ep-0` / `pick-place-v3` / cutoff `0.25`
- progress_proxy: `0.0`
- distance_progress: `0.0`
- target_distance_last: `0.315663`
- target_distance_best: `0.315293`
- in_place_score: `0.135577`
- grasp_signal_peak: `0.393521`
- success_signal_peak: `0.0`
- reward_density: `0.033745`
- stall_score: `1.0`
- latent_mismatch_score: `0.056973`
- latent_alignment_score: `0.943027`
- judge_on_track_score: `0.149989`
- judge_implausibility_score: `0.329628`
- judge_uncertainty_score: `0.70595`

## Read it honestly
- this is still a score-surface showcase, not a faithful JEPA world model
- disagreement rows are useful because they show where sparse reward misses prefix state
- if the timeline looks clean but disagreement rows are nonsense, the artifact is still weak

## Provenance rule
- every claim in this demo should be traceable back to the comparison CSV plus the underlying prefixes/baselines/judge JSONL files
