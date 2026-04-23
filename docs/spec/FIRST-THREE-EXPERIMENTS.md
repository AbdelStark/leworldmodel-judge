# FIRST-THREE-EXPERIMENTS

## Experiment 1 — baseline-only benchmark pass
Goal:
- collect or synthesize rollouts for `reach-v3`, `push-v3`, `pick-place-v3`
- build prefixes at 25/50/75%
- compute sparse reward and terminal-success views
- prove what those signals miss on partial prefixes

## Experiment 2 — first judge signal
Goal:
- add one world-model-derived score
- first acceptable v1: a lightweight surprise / residual-inspired signal
- measure whether it improves early failure detection or ranking

## Experiment 3 — uncertainty / implausibility extension
Goal:
- add uncertainty or implausibility score
- test whether it explains cases where success prediction alone is insufficient
