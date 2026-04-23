from __future__ import annotations

from collections import defaultdict


def summarize(prefixes: list[dict], baselines: list[dict], judge: list[dict]) -> dict:
    baseline_map = {(r['episode_id'], r['prefix_fraction']): r for r in baselines}
    judge_map = {(r['episode_id'], r['prefix_fraction']): r for r in judge}
    per_task: dict[str, dict[str, float]] = defaultdict(lambda: {
        'count': 0,
        'judge_failure_hits': 0,
        'baseline_reward_hits': 0,
    })
    for prefix in prefixes:
        key = (prefix['episode_id'], prefix['prefix_fraction'])
        b = baseline_map.get(key, {})
        j = judge_map.get(key, {})
        task = prefix['task_id']
        per_task[task]['count'] += 1
        if prefix['prefix_failure_label'] and float(j.get('failure_score', 0.0)) > 0:
            per_task[task]['judge_failure_hits'] += 1
        if prefix['prefix_failure_label'] and float(b.get('sparse_reward_score', 0.0)) < 0:
            per_task[task]['baseline_reward_hits'] += 1
    return {'tasks': per_task}
