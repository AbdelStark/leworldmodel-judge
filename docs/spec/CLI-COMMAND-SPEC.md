# CLI-COMMAND-SPEC

V1 may start as scripts rather than a polished CLI, but the command surface should already be stable.

## Required commands or script equivalents

### collect-rollouts
Purpose: collect raw trajectories.

### build-prefixes
Purpose: slice trajectories into benchmarkable prefixes.

### run-baselines
Purpose: compute sparse-reward and heuristic baselines.

### run-judge
Purpose: compute world-model-derived judge scores.

### evaluate
Purpose: compare baselines and judge signals.

### render-demo
Purpose: create replay/demo artifacts.

## Rule
If the project starts as scripts, keep names and arguments close enough to this contract that a later CLI wrapper is trivial.
