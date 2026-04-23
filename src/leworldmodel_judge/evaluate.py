from __future__ import annotations

from collections import defaultdict


THRESHOLDS = {
    'judge_failure_threshold': 0.5,
    'progress_failure_threshold': 0.2,
}


def _blank_bucket() -> dict[str, float | list[tuple[bool, float]]]:
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
        '_judge_pairs': [],
        '_baseline_sparse_absence_pairs': [],
        '_baseline_progress_pairs': [],
    }


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _pairwise_accuracy(pairs: list[tuple[bool, float]]) -> float | None:
    failures = [score for label, score in pairs if label]
    non_failures = [score for label, score in pairs if not label]
    if not failures or not non_failures:
        return None

    wins = 0.0
    total = 0
    for fail_score in failures:
        for non_fail_score in non_failures:
            total += 1
            if fail_score > non_fail_score:
                wins += 1.0
            elif fail_score == non_fail_score:
                wins += 0.5
    return round(wins / total, 6)


def _finalize_bucket(bucket: dict[str, float | list[tuple[bool, float]]]) -> dict[str, float]:
    failures = float(bucket['failure_labels'])
    non_failures = float(bucket['non_failure_labels'])
    finalized = {k: v for k, v in bucket.items() if not k.startswith('_')}
    finalized['judge_failure_hit_rate'] = _rate(float(bucket['judge_failure_hits']), failures)
    finalized['judge_false_positive_rate'] = _rate(float(bucket['judge_false_positives']), non_failures)
    finalized['baseline_sparse_absence_hit_rate'] = _rate(float(bucket['baseline_sparse_absence_hits']), failures)
    finalized['baseline_sparse_absence_false_positive_rate'] = _rate(float(bucket['baseline_sparse_absence_false_positives']), non_failures)
    finalized['baseline_progress_hit_rate'] = _rate(float(bucket['baseline_progress_hits']), failures)
    finalized['baseline_progress_false_positive_rate'] = _rate(float(bucket['baseline_progress_false_positives']), non_failures)
    finalized['failure_label_coverage'] = _rate(failures, float(bucket['count']))
    finalized['judge_pairwise_accuracy'] = _pairwise_accuracy(bucket['_judge_pairs'])
    finalized['baseline_sparse_absence_pairwise_accuracy'] = _pairwise_accuracy(bucket['_baseline_sparse_absence_pairs'])
    finalized['baseline_progress_pairwise_accuracy'] = _pairwise_accuracy(bucket['_baseline_progress_pairs'])
    return finalized


def summarize(prefixes: list[dict], baselines: list[dict], judge: list[dict]) -> dict:
    baseline_map = {(r['task_id'], r['episode_id'], r['prefix_fraction']): r for r in baselines}
    judge_map = {(r['task_id'], r['episode_id'], r['prefix_fraction']): r for r in judge}
    per_task: dict[str, dict[str, float | list[tuple[bool, float]]]] = defaultdict(_blank_bucket)
    overall = _blank_bucket()

    for prefix in prefixes:
        key = (prefix['task_id'], prefix['episode_id'], prefix['prefix_fraction'])
        b = baseline_map.get(key, {})
        j = judge_map.get(key, {})
        task = prefix['task_id']
        task_bucket = per_task[task]
        task_bucket['count'] += 1
        overall['count'] += 1

        label = bool(prefix.get('prefix_failure_label'))
        judge_score = float(j.get('failure_score', 0.0))
        progress_failure_score = 1.0 - float(b.get('progress_proxy_score', 0.0))
        sparse_absence_score = 1.0 if float(b.get('sparse_reward_score', 0.0)) <= 0.0 else 0.0

        task_bucket['_judge_pairs'].append((label, judge_score))
        task_bucket['_baseline_progress_pairs'].append((label, progress_failure_score))
        task_bucket['_baseline_sparse_absence_pairs'].append((label, sparse_absence_score))
        overall['_judge_pairs'].append((label, judge_score))
        overall['_baseline_progress_pairs'].append((label, progress_failure_score))
        overall['_baseline_sparse_absence_pairs'].append((label, sparse_absence_score))

        judge_predicts_failure = judge_score >= THRESHOLDS['judge_failure_threshold']
        progress_predicts_failure = float(b.get('progress_proxy_score', 0.0)) < THRESHOLDS['progress_failure_threshold']
        sparse_signal_absent = sparse_absence_score >= 1.0

        if label:
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
