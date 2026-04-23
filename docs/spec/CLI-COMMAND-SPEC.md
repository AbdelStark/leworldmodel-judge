# CLI-COMMAND-SPEC

V1 may start as scripts rather than a polished CLI, but the command surface should already be stable.

## Required commands or script equivalents

### collect-rollouts
Purpose: collect raw trajectories.

Expected args:
- `--source {synthetic,metaworld}`
- `--task TASK_ID`
- `--episodes N`
- `--output PATH`

### build-prefixes
Purpose: slice trajectories into benchmarkable prefixes.

Expected args:
- `--input PATH`
- `--output PATH`
- `--fractions 0.25,0.50,0.75`

### run-baselines
Purpose: compute sparse-reward and heuristic baselines.

Expected args:
- `--input PATH`
- `--output PATH`

### run-judge
Purpose: compute world-model-derived judge scores.

Expected args:
- `--input PATH`
- `--output PATH`
- `--mode {heuristic_surprise,dummy}`

### evaluate
Purpose: compare baselines and judge signals.

Expected args:
- `--prefixes PATH`
- `--baselines PATH`
- `--judge PATH`
- `--output PATH`

### render-demo
Purpose: create replay/demo artifacts.

Expected args:
- `--prefixes PATH`
- `--judge PATH`
- `--output PATH`

## Rule
If the project starts as scripts, keep names and arguments close enough to this contract that a later CLI wrapper is trivial.
