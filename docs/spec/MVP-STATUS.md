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
- Push-v3 prefix labeling now catches late contact-without-transport failures much more honestly
- Demo artifacts now emit episode-level score replay tables in addition to mean timeline plots

## Current phase
**Held-out artifact execution + push-v3 hardening + replay reporting**

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
- the code path now supports held-out family calibration provenance and average-precision reporting; the newer `2026-04-28` artifact turns that path into a real checked-in benchmark surface
- the old `2026-04-23-v2` real smoke artifact is still useful as an in-slice debugging reference, but it should no longer be read as the repo's best threshold story

### Held-out hard-family real artifact
Artifact folder:
- `artifacts/hard-family-real-held-out-2026-04-28/`

Run shape:
- source: real Meta-World hard families
- calibration families: `weak`, `doomed`
- evaluation families: `expert`, `misleading`
- judge mode: `hybrid_surprise`
- tasks: all locked v1 tasks
- episodes per task/family: 1
- total evaluation prefixes: 18

Current summary (`summary.json`):
- threshold mode: `held_out_family_split`
- calibrated judge threshold: `0.311141`
- overall judge pairwise accuracy: `1.0`
- overall judge average precision: `1.0`
- overall judge false positive rate: `0.1`
- overall judge failure hit rate: `1.0`
- push-v3 evaluation hit rate: `1.0` with `0.0` false positives
- replay/demo outputs now include `demo-artifact-score-replay.csv`

Interpretation:
- this is the first checked-in artifact set where the threshold is actually coming from a disjoint family split instead of in-slice tuning
- push-v3 no longer drops the held-out failures in this hard-family real slice after the task-aware late-prefix label hardening
- the artifact is still small; it proves the wiring and the provenance story, not broad generalization

## Next milestone
1. widen failure-label coverage beyond the current narrow late-prefix doomed cases
2. run larger real held-out slices so the current held-out artifact is not the only publishable threshold story
3. enrich replay reporting with rendered-state snapshots or frame strips, not just scalar score drift
4. keep shrinking the remaining expert-family false positives in `pick-place-v3`

## V1 exit criteria
- rollout capture works
- sparse reward baseline exists
- heuristic baseline exists
- first judge score exists
- one benchmark table exists
- one plot exists
- one replay/demo surface exists
- README remains honest after results
