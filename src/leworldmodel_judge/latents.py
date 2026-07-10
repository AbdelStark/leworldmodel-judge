"""Observation-space latent cache: a placeholder for a learned JEPA encoder.

Honesty framing, stated plainly: nothing here is learned. The "latents" are
mean-pooled raw observation windows, and the "predictor" is a linear
extrapolation — the context mean plus the mean per-step delta scaled by the
remaining horizon. This module exists so the hybrid judge and its cache
contract (row schema, ``latent_cache_version``) are exercised end-to-end
before a real JEPA encoder replaces the representation (docs/roadmap.md).
Calling this a world model would be laundering; it is an observation-space
proxy.

Per prefix, the cache row records the context latent, the predicted and
actual future latents, their norms, and two derived scalars:
``latent_mismatch_score`` / ``latent_alignment_score`` — the mean absolute
predicted-vs-actual gap clipped to [0, 1], and its complement.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import sqrt
from typing import Any

from ._math import clip01 as _clip01

LATENT_CACHE_VERSION = "v0.1"


def _mean_or_zero(values: list[float]) -> float:
    """Mean of ``values``; ``0.0`` on empty input.

    Named to disambiguate from ``prefixes._mean``, which returns ``None`` on
    empty input — the two absence contracts must not share a name.
    """
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _observation_window(steps: list[dict[str, Any]]) -> list[list[float]]:
    return [
        list(map(float, step.get("observation", []) or []))
        for step in steps
        if step.get("observation")
    ]


def _vector_mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = min(len(vector) for vector in vectors)
    if width == 0:
        return []
    return [round(sum(vector[idx] for vector in vectors) / len(vectors), 6) for idx in range(width)]


def _vector_sub(left: list[float], right: list[float]) -> list[float]:
    width = min(len(left), len(right))
    return [round(left[idx] - right[idx], 6) for idx in range(width)]


def _vector_add(left: list[float], right: list[float]) -> list[float]:
    width = min(len(left), len(right))
    return [round(left[idx] + right[idx], 6) for idx in range(width)]


def _vector_scale(vector: list[float], scale: float) -> list[float]:
    return [round(value * scale, 6) for value in vector]


def _vector_norm(vector: list[float]) -> float:
    return round(sqrt(sum(value * value for value in vector)), 6)


def _mean_step_delta(vectors: list[list[float]]) -> list[float]:
    if len(vectors) < 2:
        return [0.0 for _ in (vectors[0] if vectors else [])]
    deltas = [_vector_sub(vectors[idx], vectors[idx - 1]) for idx in range(1, len(vectors))]
    return _vector_mean(deltas)


def _predict_future_latent(context_vectors: list[list[float]], horizon: int) -> list[float]:
    """Linear extrapolation: context mean + mean step delta scaled by horizon."""
    if not context_vectors:
        return []
    context_latent = _vector_mean(context_vectors)
    mean_delta = _mean_step_delta(context_vectors)
    scale = max(1.0, float(horizon))
    return _vector_add(context_latent, _vector_scale(mean_delta, scale))


def _alignment_score(predicted: list[float], actual: list[float]) -> float:
    width = min(len(predicted), len(actual))
    if width == 0:
        return 0.0
    gap = sum(abs(predicted[idx] - actual[idx]) for idx in range(width)) / width
    return round(_clip01(1.0 - gap), 6)


def _mismatch_score(predicted: list[float], actual: list[float]) -> float:
    width = min(len(predicted), len(actual))
    if width == 0:
        return 0.0
    gap = sum(abs(predicted[idx] - actual[idx]) for idx in range(width)) / width
    return round(_clip01(gap), 6)


def _group_by_episode(steps: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        grouped[str(step["episode_id"])].append(step)
    for episode_steps in grouped.values():
        episode_steps.sort(key=lambda row: int(row["timestep"]))
    return dict(grouped)


def build_latent_cache(
    prefixes: list[dict[str, Any]], steps: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build one latent-cache row per prefix from the full rollout steps.

    The context window is the prefix's steps; the target window is everything
    after the cutoff. When no post-cutoff observations exist, the actual
    future latent falls back to the context latent (zero mismatch by
    construction). Prefixes with no usable index or episode are skipped
    rather than emitted with fabricated latents. Rows carry
    ``latent_cache_version`` so downstream consumers can reject stale caches.
    """
    grouped_steps = _group_by_episode(steps)
    cache_rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        episode_steps = grouped_steps.get(str(prefix["episode_id"]), [])
        prefix_index = int(prefix.get("prefix_index", 0))
        if prefix_index <= 0 or not episode_steps:
            continue
        context_steps = episode_steps[:prefix_index]
        target_steps = episode_steps[prefix_index:]
        context_vectors = _observation_window(context_steps)
        target_vectors = _observation_window(target_steps)
        context_latent = _vector_mean(context_vectors)
        predicted_future_latent = _predict_future_latent(
            context_vectors, horizon=len(target_vectors) or 1
        )
        actual_future_latent = _vector_mean(target_vectors) if target_vectors else context_latent
        latent_mismatch = _mismatch_score(predicted_future_latent, actual_future_latent)
        latent_alignment = _alignment_score(predicted_future_latent, actual_future_latent)
        cache_rows.append(
            {
                "episode_id": prefix["episode_id"],
                "task_id": prefix["task_id"],
                "policy_family": prefix.get("policy_family"),
                "prefix_fraction": float(prefix["prefix_fraction"]),
                "prefix_index": prefix_index,
                "latent_cache_version": LATENT_CACHE_VERSION,
                "context_latent": context_latent,
                "predicted_future_latent": predicted_future_latent,
                "actual_future_latent": actual_future_latent,
                "context_latent_norm": _vector_norm(context_latent),
                "predicted_future_latent_norm": _vector_norm(predicted_future_latent),
                "actual_future_latent_norm": _vector_norm(actual_future_latent),
                "latent_alignment_score": latent_alignment,
                "latent_mismatch_score": latent_mismatch,
            }
        )
    return cache_rows
