# IMPLEMENTATION-PLAN

## Phase 1 — environment and contract lock
Deliverables:
- exact environment choice = Meta-World
- exact task IDs = `reach-v3`, `push-v3`, `pick-place-v3`
- exact prefix cutoffs = 25/50/75%
- exact labels = final success, prefix failure, recoverability
- exact metrics = early failure detection, ranking quality, optional calibration

## Phase 2 — data layer
Deliverables:
- rollout collection
- canonical schema
- prefix builder
- synthetic fixture mode for script testing

## Phase 3 — baselines
Deliverables:
- sparse reward baseline
- terminal success baseline
- heuristic progress baseline

## Phase 4 — judge v1
Deliverables:
- one world-model-derived score
- saved score outputs
- uncertainty score if feasible

## Phase 5 — evaluation surface
Deliverables:
- benchmark table
- comparison JSON/CSV
- first analysis note

## Phase 6 — demo surface
Deliverables:
- replay artifact
- score-over-time plot
- baseline vs judge comparison artifact

## Verification rule
Every phase must end with a file artifact that a later agent can inspect without re-running everything.
