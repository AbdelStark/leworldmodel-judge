from __future__ import annotations

from collections import defaultdict


THRESHOLDS = {
    'judge_failure_threshold': 0.5,
    'progress_failure_threshold': 0.2,
}


def _blank_bucket() -> dict[str, float]:
    return {
        'count': 0,
        'failure_labels': 0,
        'non_failure_labels': 0,
        'judge_failure_hits': 0,
        'judge_false_positives': 0,
        'baseline_sparse_absence_hits': 0,
        'baseline_sparse_absence_false_positives': 0,
        'baseline_progress_hits': 0,
        'baseline_progress_false_positives': 0,
    }


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _finalize_bucket(bucket: dict[str, float]) -> dict[str, float]:
    failures = bucket['failure_labels']
    non_failures = bucket['non_failure_labels']
    bucket['judge_failure_hit_rate'] = _rate(bucket['judge_failure_hits'], failures)
    bucket['judge_false_positive_rate'] = _rate(bucket['judge_false_positives'], non_failures)
    bucket['baseline_sparse_absence_hit_rate'] = _rate(bucket['baseline_sparse_absence_hits'], failures)
    bucket['baseline_sparse_absence_false_positive_rate'] = _rate(bucket['baseline_sparse_absence_false_positives'], non_failures)
    bucket['baseline_progress_hit_rate'] = _rate(bucket['baseline_progress_hits'], failures)
    bucket['baseline_progress_false_positive_rate'] = _rate(bucket['baseline_progress_false_positives'], non_failures)
    bucket['failure_label_coverage'] = _rate(failures, bucket['count'])
    return bucket


def summarize(prefixes: list[dict], baselines: list[dict], judge: list[dict]) -> dict:
    baseline_map = {(r['task_id'], r['episode_id'], r['prefix_fraction']): r for r in baselines}
    judge_map = {(r['task_id'], r['episode_id'], r['prefix_fraction']): r for r in judge}
    per_task: dict[str, dict[str, float]] = defaultdict(_blank_bucket)
    overall = _blank_bucket()

    for prefix in prefixes:
        key = (prefix['task_id'], prefix['episode_id'], prefix['prefix_fraction'])
        b = baseline_map.get(key, {})
        j = judge_map.get(key, {})
        task = prefix['task_id']
        task_bucket = per_task[task]
        task_bucket['count'] += 1
        overall['count'] += 1

        judge_predicts_failure = float(j.get('failure_score', 0.0)) >= THRESHOLDS['judge_failure_threshold']
        progress_predicts_failure = float(b.get('progress_proxy_score', 0.0)) < THRESHOLDS['progress_failure_threshold']
        sparse_signal_absent = float(b.get('sparse_reward_score', 0.0)) <= 0.0

        if prefix.get('prefix_failure_label'):
            task_bucket['failure_labels'] += 1
            overall['failure_labels'] += 1
            if judge_predicts_failure:
                task_bucket['judge_failure_hits'] += 1
                overall['judge_failure_hits'] += 1
            if sparse_signal_absent:
                task_bucket['baseline_sparse_absence_hits'] += 1
                overall['baseline_sparse_absence_hits'] += 1
            if progress_predicts_failure:
                task_bucket['baseline_progress_hits'] += 1
                overall['baseline_progress_hits'] += 1
        else:
            task_bucket['non_failure_labels'] += 1
            overall['non_failure_labels'] += 1
            if judge_predicts_failure:
                task_bucket['judge_false_positives'] += 1
                overall['judge_false_positives'] += 1
            if sparse_signal_absent:
                task_bucket['baseline_sparse_absence_false_positives'] += 1
                overall['baseline_sparse_absence_false_positives'] += 1
            if progress_predicts_failure:
                task_bucket['baseline_progress_false_positives'] += 1
                overall['baseline_progress_false_positives'] += 1

    finalized_tasks = {task: _finalize_bucket(bucket) for task, bucket in per_task.items()}
    return {
        'thresholds': THRESHOLDS,
        'overall': _finalize_bucket(overall),
        'tasks': finalized_tasks,
    }
