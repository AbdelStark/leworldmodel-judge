# MULTITASK-REAL-SMOKE-2026-04-23

## Purpose
Document the first real end-to-end pass over the full locked Meta-World v1 slice.

## Command shape
```bash
source .venv/bin/activate
python scripts/collect_rollouts.py \
  --source metaworld \
  --task all \
  --episodes 3 \
  --max-steps 75 \
  --output artifacts/multitask-real-smoke-2026-04-23/rollouts.jsonl

python scripts/build_prefixes.py \
  --input artifacts/multitask-real-smoke-2026-04-23/rollouts.jsonl \
  --output artifacts/multitask-real-smoke-2026-04-23/prefixes.jsonl

python scripts/run_baselines.py \
  --input artifacts/multitask-real-smoke-2026-04-23/prefixes.jsonl \
  --output artifacts/multitask-real-smoke-2026-04-23/baselines.jsonl

python scripts/run_judge.py \
  --input artifacts/multitask-real-smoke-2026-04-23/prefixes.jsonl \
  --output artifacts/multitask-real-smoke-2026-04-23/judge.jsonl

python scripts/evaluate.py \
  --prefixes artifacts/multitask-real-smoke-2026-04-23/prefixes.jsonl \
  --baselines artifacts/multitask-real-smoke-2026-04-23/baselines.jsonl \
  --judge artifacts/multitask-real-smoke-2026-04-23/judge.jsonl \
  --output artifacts/multitask-real-smoke-2026-04-23/summary.json
```

## Artifact folder
- `artifacts/multitask-real-smoke-2026-04-23/rollouts.jsonl`
- `artifacts/multitask-real-smoke-2026-04-23/prefixes.jsonl`
- `artifacts/multitask-real-smoke-2026-04-23/baselines.jsonl`
- `artifacts/multitask-real-smoke-2026-04-23/judge.jsonl`
- `artifacts/multitask-real-smoke-2026-04-23/summary.json`

## Verified counts
- tasks: `reach-v3`, `push-v3`, `pick-place-v3`
- episodes per task: 3
- max steps: 75
- rollout steps: 675
- prefix records: 27

## Summary snapshot
```json
{
  "overall": {
    "count": 27,
    "failure_labels": 11,
    "judge_failure_hits": 11,
    "baseline_reward_hits": 11,
    "baseline_progress_hits": 11,
    "judge_failure_hit_rate": 1.0,
    "baseline_reward_hit_rate": 1.0,
    "baseline_progress_hit_rate": 1.0
  },
  "tasks": {
    "reach-v3": {
      "count": 9,
      "failure_labels": 3,
      "judge_failure_hits": 3,
      "baseline_reward_hits": 3,
      "baseline_progress_hits": 3,
      "judge_failure_hit_rate": 1.0,
      "baseline_reward_hit_rate": 1.0,
      "baseline_progress_hit_rate": 1.0
    },
    "push-v3": {
      "count": 9,
      "failure_labels": 2,
      "judge_failure_hits": 2,
      "baseline_reward_hits": 2,
      "baseline_progress_hits": 2,
      "judge_failure_hit_rate": 1.0,
      "baseline_reward_hit_rate": 1.0,
      "baseline_progress_hit_rate": 1.0
    },
    "pick-place-v3": {
      "count": 9,
      "failure_labels": 6,
      "judge_failure_hits": 6,
      "baseline_reward_hits": 6,
      "baseline_progress_hits": 6,
      "judge_failure_hit_rate": 1.0,
      "baseline_reward_hit_rate": 1.0,
      "baseline_progress_hit_rate": 1.0
    }
  }
}
```

## Honest read
- This pass proves the *full locked trio* now runs end-to-end on real environments.
- The reward baseline is now a true sparse-success signal instead of the dense shaping reward sum.
- The composite judge is alive and tracks the heuristic failure labels without reading them as inputs.
- On this tiny random-action pass, the judge still ties both baselines instead of beating them.
- That means the next work is not cosmetic. It is about producing harder trajectories and metrics where separation is even possible.

## Next actions
1. add ranking / ordering metrics, not just hit counts
2. collect stronger trajectory families than pure random actions
3. render score-over-time examples for good vs doomed prefixes
4. keep every claim bounded to what the artifact actually shows
