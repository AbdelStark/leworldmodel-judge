# DATASET-CONTRACT

## Rollout schema
Each stored step should include:
- `episode_id`
- `task_id`
- `timestep`
- `observation`
- `action`
- `reward`
- `done`
- `success_label`
- optional metadata

## Prefix schema
Each prefix should include:
- episode reference
- prefix cutoff index
- prefix fraction or ratio
- final success label
- any derived failure/recoverability labels

## Storage stance
V1 prefers simple file-based storage:
- JSONL
- NPZ
- Parquet if needed later

## v1 environment stance
Start with one manipulation family only.
Do not build general ingestion.
