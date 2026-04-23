from __future__ import annotations

from collections import defaultdict
from math import floor
from typing import Iterable

from .schema import PrefixRecord


DEFAULT_FRACTIONS = (0.25, 0.50, 0.75)


def group_by_episode(steps: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for step in steps:
        grouped[step['episode_id']].append(step)
    for episode_steps in grouped.values():
        episode_steps.sort(key=lambda row: row['timestep'])
    return dict(grouped)


def build_prefixes(steps: Iterable[dict], fractions: tuple[float, ...] = DEFAULT_FRACTIONS) -> list[PrefixRecord]:
    grouped = group_by_episode(steps)
    prefixes: list[PrefixRecord] = []
    for episode_id, episode_steps in grouped.items():
        total = len(episode_steps)
        if total == 0:
            continue
        final_success = bool(episode_steps[-1]['success_label'])
        task_id = episode_steps[-1]['task_id']
        policy_family = (episode_steps[0].get('info') or {}).get('policy_family')
        for fraction in fractions:
            prefix_index = max(1, min(total, floor(total * fraction)))
            prefix_steps = episode_steps[:prefix_index]
            sparse_reward_prefix = _sparse_reward_prefix(prefix_steps)
            metrics = _prefix_metrics(task_id, prefix_steps)
            progress_proxy = _progress_proxy(task_id, prefix_steps, metrics)
            prefix_failure_label, recoverability = _prefix_labels(
                task_id=task_id,
                prefix_steps=prefix_steps,
                final_success=final_success,
                prefix_fraction=fraction,
                progress_proxy=progress_proxy,
                metrics=metrics,
            )
            prefixes.append(PrefixRecord(
                episode_id=episode_id,
                task_id=task_id,
                prefix_index=prefix_index,
                prefix_fraction=fraction,
                final_success_label=final_success,
                prefix_failure_label=prefix_failure_label,
                prefix_recoverability_label=recoverability,
                sparse_reward_prefix=sparse_reward_prefix,
                policy_family=policy_family,
                progress_proxy=progress_proxy,
                target_distance_start=metrics['target_distance_start'],
                target_distance_last=metrics['target_distance_last'],
                target_distance_best=metrics['target_distance_best'],
                distance_progress=metrics['distance_progress'],
                in_place_score=metrics['in_place_score'],
                near_object_score=metrics['near_object_score'],
                grasp_signal_peak=metrics['grasp_signal_peak'],
                success_signal_peak=metrics['success_signal_peak'],
                reward_density=metrics['reward_density'],
                stall_score=metrics['stall_score'],
            ))
    return prefixes


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _info_float(step: dict, key: str) -> float | None:
    info = step.get('info') or {}
    value = info.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _series(prefix_steps: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for step in prefix_steps:
        value = _info_float(step, key)
        if value is not None:
            values.append(value)
    return values


def _sparse_reward_prefix(prefix_steps: list[dict]) -> float:
    success_events = _series(prefix_steps, 'success')
    if success_events:
        return float(sum(1.0 for value in success_events if value >= 1.0))
    return 0.0


def _prefix_metrics(task_id: str, prefix_steps: list[dict]) -> dict[str, float | None]:
    del task_id  # task-aware heuristics can branch here later without changing the record schema

    obj_to_target = _series(prefix_steps, 'obj_to_target')
    in_place = [_clip01(v) for v in _series(prefix_steps, 'in_place_reward')]
    near_object = [_clip01(v) for v in _series(prefix_steps, 'near_object')]
    grasp_success = [_clip01(v) for v in _series(prefix_steps, 'grasp_success')]
    grasp_reward = [_clip01(v) for v in _series(prefix_steps, 'grasp_reward')]
    success_signal = [_clip01(v) for v in _series(prefix_steps, 'success')]
    unscaled_reward = [max(0.0, v) for v in _series(prefix_steps, 'unscaled_reward')]

    target_distance_start = obj_to_target[0] if obj_to_target else None
    target_distance_last = obj_to_target[-1] if obj_to_target else None
    target_distance_best = min(obj_to_target) if obj_to_target else None
    if target_distance_start is not None and target_distance_best is not None:
        denom = max(abs(target_distance_start), 1e-6)
        distance_progress = _clip01((target_distance_start - target_distance_best) / denom)
    else:
        distance_progress = None

    in_place_score = max(in_place) if in_place else None
    near_object_score = max(near_object) if near_object else None
    grasp_signal_peak = max(grasp_success + grasp_reward) if (grasp_success or grasp_reward) else None
    success_signal_peak = max(success_signal) if success_signal else None
    reward_density = _mean(unscaled_reward)

    if distance_progress is not None:
        stall_score = _clip01(1.0 - distance_progress)
    elif reward_density is not None:
        stall_score = _clip01(1.0 - reward_density)
    else:
        stall_score = None

    return {
        'target_distance_start': target_distance_start,
        'target_distance_last': target_distance_last,
        'target_distance_best': target_distance_best,
        'distance_progress': distance_progress,
        'in_place_score': in_place_score,
        'near_object_score': near_object_score,
        'grasp_signal_peak': grasp_signal_peak,
        'success_signal_peak': success_signal_peak,
        'reward_density': reward_density,
        'stall_score': stall_score,
    }


def _progress_proxy(task_id: str, prefix_steps: list[dict], metrics: dict[str, float | None]) -> float:
    del task_id

    if metrics['distance_progress'] is not None:
        return round(metrics['distance_progress'], 6)

    if not prefix_steps:
        return 0.0
    first = prefix_steps[0]['observation'][0] if prefix_steps[0]['observation'] else 0.0
    last = prefix_steps[-1]['observation'][0] if prefix_steps[-1]['observation'] else 0.0
    return float(first - last)


def _metric_value(metrics: dict[str, float | None], key: str, fallback: float) -> float:
    value = metrics.get(key)
    if value is None:
        return float(fallback)
    return float(value)


def _prefix_labels(
    task_id: str,
    prefix_steps: list[dict],
    final_success: bool,
    prefix_fraction: float,
    progress_proxy: float,
    metrics: dict[str, float | None],
) -> tuple[bool, str]:
    del prefix_steps

    progress = _clip01(progress_proxy)
    distance_progress_raw = metrics.get('distance_progress')
    in_place_raw = metrics.get('in_place_score')
    near_object_raw = metrics.get('near_object_score')
    grasp_raw = metrics.get('grasp_signal_peak')
    success_raw = metrics.get('success_signal_peak')
    reward_density_raw = metrics.get('reward_density')
    last_distance_raw = metrics.get('target_distance_last')

    distance_progress = _clip01(distance_progress_raw if distance_progress_raw is not None else 0.0)
    in_place = _clip01(in_place_raw if in_place_raw is not None else 0.0)
    near_object = _clip01(near_object_raw if near_object_raw is not None else 0.0)
    grasp = _clip01(grasp_raw if grasp_raw is not None else 0.0)
    success = _clip01(success_raw if success_raw is not None else 0.0)
    reward_density = _clip01(reward_density_raw if reward_density_raw is not None else 0.0)
    last_distance = float(last_distance_raw if last_distance_raw is not None else 1.0)

    has_observed_state_signal = (distance_progress_raw is not None) or (last_distance_raw is not None)

    stalled = distance_progress < 0.10 and in_place < 0.20
    far_from_target = last_distance > 0.20
    no_contact = grasp < 0.15 and success == 0.0
    engaged = max(grasp, near_object)
    transport = max(in_place, success, distance_progress)
    weak_prefix = progress < 0.20 and reward_density < 0.10
    distance_regret = max(0.0, last_distance - _metric_value(metrics, 'target_distance_best', last_distance))

    if final_success:
        if progress >= 0.35 or success > 0.0 or in_place >= 0.35:
            return False, 'recoverable'
        return False, 'at_risk'

    if not has_observed_state_signal:
        return False, 'at_risk'

    if task_id == 'reach-v3':
        if prefix_fraction >= 0.50 and distance_progress < 0.10 and last_distance > 0.18:
            return True, 'doomed'
        if progress >= 0.45 and last_distance < 0.12:
            return False, 'recoverable'
        return False, 'at_risk'

    if task_id == 'pick-place-v3':
        if prefix_fraction >= 0.75 and engaged >= 0.60 and transport < 0.20 and last_distance > 0.28:
            return True, 'doomed'
        if prefix_fraction >= 0.75 and engaged >= 0.50 and distance_regret > 0.08 and last_distance > 0.28:
            return True, 'doomed'
        if prefix_fraction >= 0.50 and engaged < 0.20 and far_from_target and weak_prefix:
            return True, 'doomed'
        if transport >= 0.35:
            return False, 'recoverable'
        return False, 'at_risk'

    if prefix_fraction >= 0.75 and distance_regret > 0.18 and last_distance > 0.30:
        return True, 'doomed'
    if prefix_fraction >= 0.75 and stalled and far_from_target and no_contact:
        return True, 'doomed'
    if prefix_fraction >= 0.50 and stalled and far_from_target and weak_prefix and no_contact:
        return True, 'doomed'
    if progress >= 0.55 or in_place >= 0.50:
        return False, 'recoverable'
    return False, 'at_risk'
