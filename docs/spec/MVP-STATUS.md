# MVP-STATUS

## Current state
- Repo created
- Docs/spec package seeded
- Core specs expanded
- RFCs expanded
- Baseline scaffold created
- Synthetic-mode pipeline runs end-to-end
- Meta-World environment installed in local venv
- Real `reach-v3` smoke pass completed end-to-end
- Multi-task real rollout pass now works across the full locked v1 slice
- Prefix records now carry Meta-World-derived signals (`obj_to_target`, `in_place_reward`, `grasp`, `success`, `unscaled_reward`)
- Judge v1 upgraded from a pure placeholder heuristic to a label-free composite prefix judge with explicit raw sub-scores
- Evaluator now supports held-out family calibration provenance and average-precision reporting

## Current phase
**Held-out calibration hardening + task-aware failure coverage + family-aware reporting**

## Current milestone target
Ship an honest benchmark surface with both artifact types:
- real multi-task smoke over the locked Meta-World slice
- hard-family synthetic benchmark where judge/progress/sparse actually diverge
- harder-than-random real trajectory families for follow-up debugging

## Latest verified runs
### Real multi-task smoke
Artifact folder:
- `artifacts/multitask-real-smoke-2026-04-23/`

Run shape:
- source: real Meta-World
- tasks: all locked v1 tasks
- episodes per task: 3
- max steps per episode: 75
- total rollout steps: 675
- total prefixes: 27

Current summary (`summary.json`):
- overall failure-labeled prefixes: 11
- judge failure hits: 11/11
- sparse reward failure hits: 11/11
- progress-proxy failure hits: 11/11

### Hard-family synthetic benchmark
Artifact folders:
- `artifacts/hard-family-synthetic-benchmark-2026-04-23/`
- `artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/`

Run shape:
- source: synthetic hard families
- policy families: `expert`, `weak`, `doomed`, `misleading`
- tasks: all locked v1 tasks
- episodes per task/family: 2
- total prefixes: 72

Current summary (`summary.json`):
- v1 judge pairwise accuracy: `0.897059`
- v1 judge false positive rate: `0.294118`
- v2 calibrated judge threshold: `0.360053`
- v2 judge pairwise accuracy: `0.985294`
- v2 judge false positive rate: `0.029412`
- sparse-success-absence pairwise accuracy: `0.5`
- simple progress pairwise accuracy: `0.147059`
- family-aware report saved under `report/family-report.md` and `report/family-report.png`

### Hard-family real smoke
Artifact folders:
- `artifacts/hard-family-real-smoke-2026-04-23/`
- `artifacts/hard-family-real-smoke-2026-04-23-v2/`

Run shape:
- source: real Meta-World hard families
- policy families: `expert`, `weak`, `doomed`, `misleading`
- tasks: all locked v1 tasks
- episodes per task/family: 1
- max steps per episode: 75
- total prefixes: 36

Current summary (`summary.json`):
- v1 judge pairwise accuracy: `0.872428`
- v1 judge false positive rate: `0.888889`
- v2 calibrated judge threshold: `0.384724`
- v2 judge pairwise accuracy: `0.959866`
- v2 judge false positive rate: `0.043478`
- v2 judge failure hit rate: `0.923077`
- pick-place-v3 failure-label coverage improved from `0.0` to `0.333333`
- family-aware report saved under `report/family-report.md` and `report/family-report.png`

Interpretation:
- the pipeline is now using a true sparse-success baseline rather than the dense shaping reward sum
- the synthetic hard-family benchmark is now a much cleaner proof surface: calibrated judge false positives dropped from `0.294118` to `0.029412` while preserving full hit rate on the narrow labeled failures
- the real hard-family smoke now has family-aware reports, in-slice calibration, and far better task-aware coverage; `pick-place-v3` moved from zero failure labels to late-prefix doomed labels with `0.333333` coverage
- the code path now supports held-out family calibration provenance and average-precision reporting, but the checked-in benchmark artifacts still need a true held-out rerun before the threshold story is publishable
- the real benchmark is still not solved: push-v3 remains brittle and the calibrated judge gives up one hit (`12/13`) to buy a large false-positive reduction

## Next milestone
1. run a true held-out family calibration pass and save those artifacts separately from the in-slice debug runs
2. harden push-v3 so the calibrated judge does not miss the single labeled failure
3. widen failure-label coverage beyond the narrow late-prefix doomed cases
4. add score-over-time replay visuals on top of the family-aware summary plots

## V1 exit criteria
- rollout capture works
- sparse reward baseline exists
- heuristic baseline exists
- first judge score exists
- one benchmark table exists
- one plot exists
- one replay/demo surface exists
- README remains honest after results
