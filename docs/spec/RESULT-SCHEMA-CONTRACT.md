# RESULT-SCHEMA-CONTRACT

## Required result files
- benchmark summary CSV
- benchmark summary JSON
- score timeline plot
- one replay/demo-aligned artifact

## Summary table columns
At minimum:
- task_id
- prefix_cutoff
- baseline_metric
- judge_metric
- sparse_reward_signal
- judge_signal
- success_label

## Required benchmark aggregates
- early failure detection score
- ranking quality score
- baseline-vs-judge delta

## Logging rule
Every result file must make it possible to reproduce the benchmark claim later.
