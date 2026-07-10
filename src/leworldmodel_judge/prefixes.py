"""Prefix construction: slice episodes at fractional cutoffs and derive evidence.

A prefix is the first ``floor(len(episode) * fraction)`` steps of an episode
(clamped to at least one step). For each prefix this module computes the
evidence metrics consumed by the judge and the baselines, then delegates
labeling to :mod:`leworldmodel_judge.labels`.

Evidence discipline: missing Meta-World info signals stay ``None`` — they are
never silently defaulted to zero. Zero is a legitimate measured value; absence
is not.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import floor
from typing import Any

from ._math import clip01 as _clip01
from .labels import label_prefix
from .schema import PrefixRecord

DEFAULT_FRACTIONS = (0.25, 0.50, 0.75)


def group_by_episode(steps: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group step dicts by ``episode_id``, each episode sorted by ``timestep``."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        grouped[step["episode_id"]].append(step)
    for episode_steps in grouped.values():
        episode_steps.sort(key=lambda row: row["timestep"])
    return dict(grouped)


def build_prefixes(
    steps: Iterable[dict[str, Any]], fractions: tuple[float, ...] = DEFAULT_FRACTIONS
) -> list[PrefixRecord]:
    """Build one labeled :class:`PrefixRecord` per episode per cutoff fraction.

    ``steps`` are rollout step dicts (the JSONL row shape produced by
    ``collect``). The prefix index is ``floor(total * fraction)`` clamped to
    ``[1, total]``, so even a 0.25 cutoff of a short episode keeps at least
    one step. Labels come from :func:`leworldmodel_judge.labels.label_prefix`
    and are evaluation-only: the judge never reads them.
    """
    grouped = group_by_episode(steps)
    prefixes: list[PrefixRecord] = []
    for episode_id, episode_steps in grouped.items():
        total = len(episode_steps)
        if total == 0:
            continue
        final_success = bool(episode_steps[-1]["success_label"])
        task_id = episode_steps[-1]["task_id"]
        policy_family = (episode_steps[0].get("info") or {}).get("policy_family")
        for fraction in fractions:
            prefix_index = max(1, min(total, floor(total * fraction)))
            prefix_steps = episode_steps[:prefix_index]
            sparse_reward_prefix = _sparse_reward_prefix(prefix_steps)
            metrics = _prefix_metrics(task_id, prefix_steps)
            progress_proxy = _progress_proxy(task_id, prefix_steps, metrics)
            prefix_failure_label, recoverability = label_prefix(
                task_id,
                final_success=final_success,
                prefix_fraction=fraction,
                progress_proxy=progress_proxy,
                metrics=metrics,
            )
            prefixes.append(
                PrefixRecord(
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
                    target_distance_start=metrics["target_distance_start"],
                    target_distance_last=metrics["target_distance_last"],
                    target_distance_best=metrics["target_distance_best"],
                    distance_progress=metrics["distance_progress"],
                    in_place_score=metrics["in_place_score"],
                    near_object_score=metrics["near_object_score"],
                    grasp_signal_peak=metrics["grasp_signal_peak"],
                    success_signal_peak=metrics["success_signal_peak"],
                    reward_density=metrics["reward_density"],
                    stall_score=metrics["stall_score"],
                )
            )
    return prefixes


def _mean(values: list[float]) -> float | None:
    """Mean of ``values``; ``None`` on empty input — absence is not zero."""
    if not values:
        return None
    return sum(values) / len(values)


def _info_float(step: dict[str, Any], key: str) -> float | None:
    info = step.get("info") or {}
    value = info.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _series(prefix_steps: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for step in prefix_steps:
        value = _info_float(step, key)
        if value is not None:
            values.append(value)
    return values


def _sparse_reward_prefix(prefix_steps: list[dict[str, Any]]) -> float:
    """Count sparse success events (``info.success >= 1.0``) inside the prefix.

    Derived from the environment's sparse success signal, not the dense
    reward: the sparse-reward baseline must stay blind to shaping terms.
    """
    success_events = _series(prefix_steps, "success")
    if success_events:
        return float(sum(1.0 for value in success_events if value >= 1.0))
    return 0.0


def _prefix_metrics(task_id: str, prefix_steps: list[dict[str, Any]]) -> dict[str, float | None]:
    """Derive the per-prefix evidence metrics from Meta-World info signals.

    Every metric is ``None`` when its underlying signal was never observed in
    the prefix; downstream consumers must handle absence explicitly.
    """
    del task_id  # task-aware heuristics can branch here later without changing the record schema

    obj_to_target = _series(prefix_steps, "obj_to_target")
    in_place = [_clip01(v) for v in _series(prefix_steps, "in_place_reward")]
    near_object = [_clip01(v) for v in _series(prefix_steps, "near_object")]
    grasp_success = [_clip01(v) for v in _series(prefix_steps, "grasp_success")]
    grasp_reward = [_clip01(v) for v in _series(prefix_steps, "grasp_reward")]
    success_signal = [_clip01(v) for v in _series(prefix_steps, "success")]
    unscaled_reward = [max(0.0, v) for v in _series(prefix_steps, "unscaled_reward")]

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
    grasp_signal_peak = (
        max(grasp_success + grasp_reward) if (grasp_success or grasp_reward) else None
    )
    success_signal_peak = max(success_signal) if success_signal else None
    reward_density = _mean(unscaled_reward)

    if distance_progress is not None:
        stall_score = _clip01(1.0 - distance_progress)
    elif reward_density is not None:
        stall_score = _clip01(1.0 - reward_density)
    else:
        stall_score = None

    return {
        "target_distance_start": target_distance_start,
        "target_distance_last": target_distance_last,
        "target_distance_best": target_distance_best,
        "distance_progress": distance_progress,
        "in_place_score": in_place_score,
        "near_object_score": near_object_score,
        "grasp_signal_peak": grasp_signal_peak,
        "success_signal_peak": success_signal_peak,
        "reward_density": reward_density,
        "stall_score": stall_score,
    }


def _progress_proxy(
    task_id: str, prefix_steps: list[dict[str, Any]], metrics: dict[str, float | None]
) -> float:
    """Progress proxy: distance progress when observed, else an observation fallback.

    The fallback (first minus last value of observation dim 0) exists so
    rollouts without distance info still produce a usable baseline signal; it
    is intentionally crude.
    """
    del task_id

    distance_progress = metrics["distance_progress"]
    if distance_progress is not None:
        return round(distance_progress, 6)

    if not prefix_steps:
        return 0.0
    first = prefix_steps[0]["observation"][0] if prefix_steps[0]["observation"] else 0.0
    last = prefix_steps[-1]["observation"][0] if prefix_steps[-1]["observation"] else 0.0
    return float(first - last)
