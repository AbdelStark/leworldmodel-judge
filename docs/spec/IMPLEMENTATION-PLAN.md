# IMPLEMENTATION-PLAN

## Phase 1 — environment and contract lock
Deliverables:
- exact environment choice
- exact task IDs
- exact prefix cutoffs
- exact labels
- exact metrics

## Phase 2 — data layer
Deliverables:
- rollout collection
- canonical schema
- prefix builder

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
