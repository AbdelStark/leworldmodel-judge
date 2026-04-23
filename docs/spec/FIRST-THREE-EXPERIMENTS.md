# FIRST-THREE-EXPERIMENTS

## Experiment 1 — baseline-only trajectory readout
Goal:
- collect rollouts
- compute sparse reward and terminal-success views
- prove what those signals miss on partial prefixes

## Experiment 2 — first judge signal
Goal:
- add one world-model-derived score
- measure whether it improves early failure detection or ranking

## Experiment 3 — uncertainty / implausibility extension
Goal:
- add uncertainty or implausibility score
- test whether it explains cases where success prediction alone is insufficient
