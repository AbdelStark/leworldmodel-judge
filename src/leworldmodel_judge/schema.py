from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class RolloutStep:
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
        return asdict(self)


@dataclass
class PrefixRecord:
    episode_id: str
    task_id: str
    prefix_index: int
    prefix_fraction: float
    final_success_label: bool
    prefix_failure_label: bool
    prefix_recoverability_label: str
    sparse_reward_prefix: float
    progress_proxy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
