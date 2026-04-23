from __future__ import annotations

from collections import defaultdict
from math import floor
from typing import Iterable

from .schema import RolloutStep, PrefixRecord


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
        rewards = [float(s['reward']) for s in episode_steps]
        for fraction in fractions:
            prefix_index = max(1, min(total, floor(total * fraction)))
            prefix_steps = episode_steps[:prefix_index]
            sparse_reward_prefix = sum(float(s['reward']) for s in prefix_steps)
            progress_proxy = _progress_proxy(task_id, prefix_steps)
            prefix_failure_label, recoverability = _prefix_labels(task_id, prefix_steps, final_success)
            prefixes.append(PrefixRecord(
                episode_id=episode_id,
                task_id=task_id,
                prefix_index=prefix_index,
                prefix_fraction=fraction,
                final_success_label=final_success,
                prefix_failure_label=prefix_failure_label,
                prefix_recoverability_label=recoverability,
                sparse_reward_prefix=sparse_reward_prefix,
                progress_proxy=progress_proxy,
            ))
    return prefixes


def _progress_proxy(task_id: str, prefix_steps: list[dict]) -> float:
    # Simple v1 heuristic: use first observation dim as distance-like signal if present.
    if not prefix_steps:
        return 0.0
    first = prefix_steps[0]['observation'][0] if prefix_steps[0]['observation'] else 0.0
    last = prefix_steps[-1]['observation'][0] if prefix_steps[-1]['observation'] else 0.0
    return float(first - last)


def _prefix_labels(task_id: str, prefix_steps: list[dict], final_success: bool) -> tuple[bool, str]:
    # Conservative v1 heuristic. Real environment-specific logic can replace this later.
    progress = _progress_proxy(task_id, prefix_steps)
    if final_success:
        if progress > 0:
            return False, 'recoverable'
        return False, 'at_risk'
    if progress <= 0:
        return True, 'doomed'
    return False, 'at_risk'
