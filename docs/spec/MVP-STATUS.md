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

## Current phase
**Hard-family benchmark shaping + metric divergence work**

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
Artifact folder:
- `artifacts/hard-family-synthetic-benchmark-2026-04-23/`

Run shape:
- source: synthetic hard families
- policy families: `expert`, `weak`, `doomed`, `misleading`
- tasks: all locked v1 tasks
- episodes per task/family: 2
- total prefixes: 72

Current summary (`summary.json`):
- judge pairwise accuracy: `0.897059`
- sparse-success-absence pairwise accuracy: `0.5`
- simple progress pairwise accuracy: `0.147059`
- judge false positive rate: `0.294118`

### Hard-family real smoke
Artifact folder:
- `artifacts/hard-family-real-smoke-2026-04-23/`

Run shape:
- source: real Meta-World hard families
- policy families: `expert`, `weak`, `doomed`, `misleading`
- tasks: all locked v1 tasks
- episodes per task/family: 1
- max steps per episode: 75
- total prefixes: 36

Current summary (`summary.json`):
- judge pairwise accuracy: `0.872428`
- sparse-success-absence pairwise accuracy: `0.5`
- simple progress pairwise accuracy: `0.68107`
- judge false positive rate: `0.888889`

Interpretation:
- the pipeline is now using a true sparse-success baseline rather than the dense shaping reward sum
- the synthetic hard-family benchmark is the first artifact where judge / sparse / progress genuinely diverge
- the real hard-family smoke proves harder-than-random trajectories exist in Meta-World collection now
- but real false positives are still far too high, so this is a promising benchmark shape, not a solved detector

## Next milestone
1. reduce judge false positives on hard-family synthetic and real runs
2. add family-aware tables / plots instead of aggregate JSON only
3. improve task-aware label coverage, especially for `pick-place-v3`
4. search for better thresholding / calibration instead of locking everything to `0.5`

## V1 exit criteria
- rollout capture works
- sparse reward baseline exists
- heuristic baseline exists
- first judge score exists
- one benchmark table exists
- one plot exists
- one replay/demo surface exists
- README remains honest after results
