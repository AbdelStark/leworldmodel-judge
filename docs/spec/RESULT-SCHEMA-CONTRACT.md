# RESULT-SCHEMA-CONTRACT

## Required result files
- benchmark summary JSON
- benchmark comparison CSV
- score timeline plot
- one replay/demo-aligned markdown artifact
- one family-aware report artifact

## Comparison CSV columns
At minimum:
- `task_id`
- `episode_id`
- `policy_family`
- `prefix_cutoff`
- `prefix_index`
- `baseline_metric`
- `judge_metric`
- `sparse_reward_signal`
- `judge_signal`
- `success_label`
- `prefix_failure_label`
- `prefix_recoverability_label`
- `baseline_vs_judge_gap`

## Evidence columns
The comparison CSV should also preserve the evidence fields that explain *why* the current prefix scored the way it did:
- `progress_proxy`
- `distance_progress`
- `target_distance_last`
- `target_distance_best`
- `in_place_score`
- `grasp_signal_peak`
- `success_signal_peak`
- `reward_density`
- `stall_score`
- `judge_on_track_score`
- `judge_implausibility_score`
- `judge_uncertainty_score`

## Required benchmark aggregates
At minimum:
- early failure detection score
- ranking quality score
- false positive cost visibility
- baseline-vs-judge delta
- family-slice breakdown
- threshold provenance

## Logging rule
Every result file must make it possible to reproduce the benchmark claim later.

## Honesty rule
If thresholds were chosen on the same benchmark slice, say that plainly in the result artifact. Do not present in-slice tuning as held-out calibration.
