"""Baseline prefix scorers the judge is benchmarked against.

Three signals per prefix, kept explicitly separate from the judge's signals
(benchmark invariant):

- ``sparse_reward_score`` — count of sparse success events inside the prefix;
  what a sparse-reward monitor would know at the cutoff.
- ``terminal_success_score`` — the episode's final outcome; an oracle upper
  bound, not something available at the cutoff.
- ``progress_proxy_score`` — the prefix's progress proxy; the strongest
  honest non-judge signal and the headline comparison.
"""

from __future__ import annotations

from typing import Any


def score_prefix(prefix: dict[str, Any]) -> dict[str, Any]:
    """Score one prefix record with all three baseline signals.

    A missing/None ``progress_proxy`` scores 0.0 here (baselines must always
    emit a number to be rankable); the prefix record itself preserves the
    None-vs-zero distinction.
    """
    sparse_reward = float(prefix["sparse_reward_prefix"])
    final_success = 1.0 if prefix["final_success_label"] else 0.0
    progress_proxy = float(prefix.get("progress_proxy") or 0.0)
    return {
        "episode_id": prefix["episode_id"],
        "task_id": prefix["task_id"],
        "policy_family": prefix.get("policy_family"),
        "prefix_fraction": prefix["prefix_fraction"],
        "sparse_reward_score": sparse_reward,
        "terminal_success_score": final_success,
        "progress_proxy_score": progress_proxy,
    }


def score_prefixes(prefixes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every prefix record; row-for-row :func:`score_prefix`."""
    return [score_prefix(prefix) for prefix in prefixes]
