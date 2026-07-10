"""Canonical record schemas for the rollout pipeline.

Field names and order are contract: they define the JSONL column layout of
``rollouts.jsonl`` and ``prefixes.jsonl`` (docs/contracts.md), and checked-in
artifacts must remain valid outputs of this code. Do not rename or reorder
fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RolloutStep:
    """One environment step of one episode, as captured by ``collect``.

    ``info`` carries the raw Meta-World (or synthetic) diagnostic signals
    (``obj_to_target``, ``in_place_reward``, ``near_object``, ``grasp_*``,
    ``success``, ``unscaled_reward``, ``source``, ``policy_family``).
    ``success_label`` is episode-level: it is backfilled onto every step once
    the episode outcome is known.
    """

    episode_id: str
    task_id: str
    timestep: int
    episode_horizon: int
    observation: list[float]
    action: list[float]
    reward: float
    done: bool
    success_label: bool
    info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSONL row shape (field order preserved)."""
        return asdict(self)


@dataclass
class PrefixRecord:
    """One labeled prefix of one episode at one fractional cutoff.

    Labels (``prefix_failure_label``, ``prefix_recoverability_label``) are
    evaluation-only ground truth; the judge never reads them. Evidence fields
    are ``None`` when the underlying signal was never observed — absence is
    not zero.
    """

    episode_id: str
    task_id: str
    prefix_index: int
    prefix_fraction: float
    final_success_label: bool
    prefix_failure_label: bool
    prefix_recoverability_label: str
    sparse_reward_prefix: float
    policy_family: str | None = None
    progress_proxy: float | None = None
    target_distance_start: float | None = None
    target_distance_last: float | None = None
    target_distance_best: float | None = None
    distance_progress: float | None = None
    in_place_score: float | None = None
    near_object_score: float | None = None
    grasp_signal_peak: float | None = None
    success_signal_peak: float | None = None
    reward_density: float | None = None
    stall_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSONL row shape (field order preserved)."""
        return asdict(self)


def prefix_key(row: Mapping[str, Any]) -> tuple[str, str, float]:
    """Join key identifying one prefix across pipeline files.

    Prefix, baseline, judge and latent-cache rows for the same prefix share
    ``(task_id, episode_id, prefix_fraction)``; ``prefix_fraction`` is cast to
    float so JSON round-trips and in-memory rows key identically.
    """
    return (row["task_id"], row["episode_id"], float(row["prefix_fraction"]))
