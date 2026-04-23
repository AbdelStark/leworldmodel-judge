# DATASET-CONTRACT

## V1 storage stance
V1 prefers simple file-based storage:
- JSONL for rollouts and prefixes
- JSON for small summaries
- CSV for benchmark outputs

No heavy database layer in v1.

## Rollout schema
Each stored step should include:
- `episode_id`
- `task_id`
- `timestep`
- `episode_horizon`
- `observation`
- `action`
- `reward`
- `done`
- `success_label`
- optional `info`
- optional metadata

## Prefix schema
Each prefix should include:
- `episode_id`
- `task_id`
- `prefix_index`
- `prefix_fraction`
- `final_success_label`
- `prefix_failure_label`
- `prefix_recoverability_label`
- `sparse_reward_prefix`
- optional heuristic progress fields

## Observation contract
For v1, observations may be stored as:
- a flat vector list
- a dict of small numeric fields

Do not block on image storage in v1.
The benchmark can begin with vector-style observations.

## Task stance
The exact v1 benchmark slice is:
- `reach-v2`
- `push-v2`
- `pick-place-v2`

## Synthetic fallback
Because Meta-World may not be installed in every environment, the data layer must support a **synthetic fixture mode** that emits schema-valid toy rollouts for testing scripts and CI.

This is a testing convenience only.
It is not the real benchmark.
