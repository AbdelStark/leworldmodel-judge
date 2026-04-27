# SCHEDULE

## M0 — full spec package
Write and deepen:
- README
- product spec
- system spec
- research note
- contracts
- RFCs
- implementation plan

## M1 — benchmark contract lock
Choose and freeze:
- environment family
- tasks
- prefix cutoffs
- labels
- baselines
- metrics
- family-based stress regimes

## M2 — rollout capture + schema
Implement:
- rollout capture
- normalized storage
- prefix builder
- task-aware derived signals
- policy-family tagging

## M3 — baseline scorers
Implement:
- sparse reward baseline
- terminal success baseline
- heuristic progress baseline

## M4 — first judge signal
Implement one world-model-derived judging path.

Current reality:
- a lighter composite prefix judge already exists
- the next step is not “invent another score” at random
- the next step is to improve faithfulness and benchmark pressure without breaking the contract

## M5 — evaluation surface
Generate:
- benchmark tables
- comparison JSON/CSV
- threshold provenance
- family-aware summaries
- main plot

## M6 — replay/demo surface
Generate:
- replay artifact
- score timeline
- side-by-side baseline vs judge view
- raw evidence decomposition view

## M7 — benchmark hardening
Before any victory-lap narrative:
- split held-out calibration from in-slice tuning
- widen failure-label coverage
- harden weak tasks, especially `push-v3`
- make the disagreement cases legible in replay form

## M8 — narrative hardening
After stronger results exist:
- tighten README
- cut unsupported claims
- record limitations
- state clearly what is heuristic today versus what becomes more JEPA-faithful next
