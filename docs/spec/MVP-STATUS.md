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
**Real multi-task smoke + label/judge tightening**

## Current milestone target
Turn the current smoke path into the first honest benchmark surface:
- `reach-v3`
- `push-v3`
- `pick-place-v3`
- judge vs sparse reward vs progress proxy
- saved artifacts under `artifacts/multitask-real-smoke-2026-04-23/`

## Latest verified run
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

Interpretation:
- the pipeline is now using a true sparse-success baseline rather than the dense shaping reward sum
- the new judge is alive and behaves coherently
- but this tiny random-action smoke pass still shows **no separation** between judge, sparse reward, and the simple progress proxy
- next value comes from stronger rollouts, richer ranking metrics, and harder prefixes rather than pretending this is already a paper result

## Next milestone
1. add ranking metrics and score-over-time artifacts
2. run a larger pass with stronger policies / more diverse trajectories
3. force a setting where judge and heuristic progress actually separate
4. keep the README and research framing honest

## V1 exit criteria
- rollout capture works
- sparse reward baseline exists
- heuristic baseline exists
- first judge score exists
- one benchmark table exists
- one plot exists
- one replay/demo surface exists
- README remains honest after results
