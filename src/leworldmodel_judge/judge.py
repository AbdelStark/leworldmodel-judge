"""The judge: score rollout prefixes without ever reading the labels.

Three judge modes share the core score/identity fields; the per-mode field
lists live in docs/contracts.md (``dummy`` rows omit ``policy_family`` and the
``*_evidence`` fields, hybrid rows add the latent fields):

- ``heuristic_surprise`` — a hand-weighted composite over the prefix evidence
  signals (``judge_mode: composite_prefix_judge``). It is a heuristic and is
  named as one; the weights live in :class:`CompositeWeights`.
- ``hybrid_surprise`` — the composite plus mismatch pressure from the
  observation-space latent cache (``judge_mode: hybrid_prefix_latent_judge``).
- ``dummy`` — a null judge (all scores 0.0, uncertainty 1.0) so the evaluation
  harness can be sanity-checked against a signal-free scorer.

Invariant pinned by tests: the judge is a pure function of prefix features.
``prefix_failure_label`` and ``prefix_recoverability_label`` are never inputs;
flipping them must leave every score unchanged. Every verdict decomposes: each
row carries the evidence values (``*_evidence``) the scores were computed
from, plus a ``judge_mode`` provenance tag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ._math import clip01 as _clip01
from .schema import prefix_key

logger = logging.getLogger(__name__)

JUDGE_MODES = ("heuristic_surprise", "hybrid_surprise", "dummy")


@dataclass(frozen=True)
class CompositeWeights:
    """Named weights of the composite heuristic judge.

    The defaults are the published benchmark weights; checked-in artifacts
    were produced with them, so changing a default is a benchmark-contract
    change. Groups:

    - ``on_track_*`` — linear mix of positive evidence (progress, proximity,
      contact, reward) forming ``on_track_score``.
    - ``failure_*`` — pressure terms raising ``failure_score``: stalling
      without transport, late distance gap, being off track, engagement that
      never converts to transport, late prefixes not in place, reward drought,
      and distance regret (drift back from the best distance reached).
    - ``credit_*`` — subtracted from ``failure_score``: early engaged prefixes
      are forgiven (patience), sparse success events buy credit.
    - ``implausibility_*`` — disagreement between evidence channels that
      should agree if the prefix is on-manifold.
    - ``uncertainty_*`` — self-distrust: score conflict, spread between the
      weakest and strongest evidence, and unconverted engagement.
    """

    on_track_progress: float = 0.22
    on_track_distance_progress: float = 0.18
    on_track_target_proximity: float = 0.15
    on_track_in_place: float = 0.14
    on_track_near_object: float = 0.12
    on_track_grasp: float = 0.09
    on_track_reward_density: float = 0.05
    on_track_success: float = 0.05

    failure_stalled_without_transport: float = 0.10
    failure_late_distance_gap: float = 0.12
    failure_off_track: float = 0.12
    failure_engaged_not_transporting: float = 0.16
    failure_late_not_in_place: float = 0.08
    failure_reward_drought: float = 0.06
    failure_distance_regret: float = 0.36
    credit_early_patience: float = 0.18
    credit_success: float = 0.08

    implausibility_progress_disagreement: float = 0.30
    implausibility_distance_regret: float = 0.35
    implausibility_unconverted_engagement: float = 0.20
    implausibility_stall_reward_gap: float = 0.15

    uncertainty_score_conflict: float = 0.50
    uncertainty_evidence_spread: float = 0.30
    uncertainty_engaged_not_transporting: float = 0.20


@dataclass(frozen=True)
class HybridWeights:
    """Named weights of the hybrid latent judge's mismatch adjustments.

    ``mismatch_pressure_*`` blend the latent-cache signals (predicted-vs-actual
    mismatch, norm gap, misalignment) into one pressure value; the remaining
    weights say how strongly that pressure moves each composite score. The
    defaults are the published benchmark weights.
    """

    mismatch_pressure_latent_mismatch: float = 0.65
    mismatch_pressure_norm_gap: float = 0.20
    mismatch_pressure_misalignment: float = 0.15

    failure_mismatch_pressure: float = 0.22
    failure_unrewarded_mismatch: float = 0.08
    on_track_mismatch_pressure: float = 0.18
    implausibility_latent_mismatch: float = 0.20
    uncertainty_mismatch_pressure: float = 0.15


DEFAULT_COMPOSITE_WEIGHTS = CompositeWeights()
DEFAULT_HYBRID_WEIGHTS = HybridWeights()


def _value(prefix: dict[str, Any], key: str, fallback: float) -> float:
    value = prefix.get(key)
    if value is None:
        return float(fallback)
    return float(value)


def heuristic_surprise_score(
    prefix: dict[str, Any], weights: CompositeWeights = DEFAULT_COMPOSITE_WEIGHTS
) -> dict[str, Any]:
    """Score one prefix with the composite heuristic judge.

    Derives intermediate features (target proximity, distance regret,
    engagement, transport, early patience, stalling without transport,
    engagement without transport) from the prefix evidence, combines them
    linearly per :class:`CompositeWeights`, and clips every score to [0, 1].
    Missing evidence falls back conservatively (e.g. ``distance_progress``
    falls back to ``progress_proxy``); labels are never read.
    """
    w = weights
    progress = _clip01(_value(prefix, "progress_proxy", 0.0))
    sparse_reward = _clip01(max(0.0, _value(prefix, "sparse_reward_prefix", 0.0)))
    distance_progress = _clip01(_value(prefix, "distance_progress", progress))
    in_place = _clip01(_value(prefix, "in_place_score", 0.0))
    near_object = _clip01(_value(prefix, "near_object_score", in_place))
    grasp = _clip01(_value(prefix, "grasp_signal_peak", 0.0))
    success = _clip01(_value(prefix, "success_signal_peak", 0.0))
    reward_density = _clip01(_value(prefix, "reward_density", sparse_reward))
    stall = _clip01(_value(prefix, "stall_score", (1.0 - distance_progress)))
    prefix_fraction = _clip01(_value(prefix, "prefix_fraction", 0.0))
    target_distance_last = _value(prefix, "target_distance_last", 1.0)
    target_distance_best = _value(prefix, "target_distance_best", target_distance_last)
    distance_gap = (
        _clip01(target_distance_last / max(target_distance_last, 0.30))
        if target_distance_last > 0
        else 0.0
    )
    target_proximity = _clip01(1.0 - distance_gap)
    distance_regret = _clip01(
        max(0.0, target_distance_last - target_distance_best) / max(target_distance_last, 0.30)
    )
    engagement = _clip01(max(near_object, grasp))
    transport = _clip01(max(in_place, success, progress, distance_progress))
    early_patience = _clip01((1.0 - prefix_fraction) * engagement)
    stalled_without_transport = _clip01(
        stall * max(0.0, 1.0 - transport) * (0.25 + 0.75 * prefix_fraction)
    )
    engaged_but_not_transporting = _clip01(max(0.0, engagement - transport) * prefix_fraction)

    on_track = _clip01(
        w.on_track_progress * progress
        + w.on_track_distance_progress * distance_progress
        + w.on_track_target_proximity * target_proximity
        + w.on_track_in_place * in_place
        + w.on_track_near_object * near_object
        + w.on_track_grasp * grasp
        + w.on_track_reward_density * reward_density
        + w.on_track_success * success
    )
    failure = _clip01(
        w.failure_stalled_without_transport * stalled_without_transport
        + w.failure_late_distance_gap * distance_gap * prefix_fraction
        + w.failure_off_track * (1.0 - on_track)
        + w.failure_engaged_not_transporting * engaged_but_not_transporting
        + w.failure_late_not_in_place * prefix_fraction * (1.0 - in_place)
        + w.failure_reward_drought * (1.0 - reward_density)
        + w.failure_distance_regret * distance_regret
        - w.credit_early_patience * early_patience
        - w.credit_success * success
    )
    implausibility = _clip01(
        w.implausibility_progress_disagreement * abs(progress - distance_progress)
        + w.implausibility_distance_regret * distance_regret
        + w.implausibility_unconverted_engagement * max(0.0, engagement - in_place)
        + w.implausibility_stall_reward_gap * max(0.0, stall - reward_density)
    )

    evidence_low = min(progress, in_place, reward_density)
    evidence_high = max(progress, in_place, reward_density, success, engagement)
    uncertainty = _clip01(
        w.uncertainty_score_conflict * abs(on_track - (1.0 - failure))
        + w.uncertainty_evidence_spread * (evidence_high - evidence_low)
        + w.uncertainty_engaged_not_transporting * engaged_but_not_transporting
    )

    return {
        "episode_id": prefix["episode_id"],
        "task_id": prefix["task_id"],
        "policy_family": prefix.get("policy_family"),
        "prefix_fraction": prefix["prefix_fraction"],
        "on_track_score": round(on_track, 6),
        "failure_score": round(failure, 6),
        "implausibility_score": round(implausibility, 6),
        "uncertainty_score": round(uncertainty, 6),
        "progress_evidence": round(progress, 6),
        "distance_progress_evidence": round(distance_progress, 6),
        "in_place_evidence": round(in_place, 6),
        "near_object_evidence": round(near_object, 6),
        "grasp_evidence": round(grasp, 6),
        "reward_evidence": round(reward_density, 6),
        "stall_evidence": round(stall, 6),
        "judge_mode": "composite_prefix_judge",
    }


def hybrid_surprise_score(
    prefix: dict[str, Any],
    latent_row: dict[str, Any] | None = None,
    weights: HybridWeights = DEFAULT_HYBRID_WEIGHTS,
) -> dict[str, Any]:
    """Score one prefix with the hybrid latent judge.

    Takes the composite row and adds mismatch pressure from the latent-cache
    row for the same prefix: high predicted-vs-actual latent mismatch raises
    ``failure_score``, ``implausibility_score`` and ``uncertainty_score`` and
    lowers ``on_track_score``. With ``latent_row=None`` all latent signals
    default to zero mismatch, so the scores degrade to the composite judge
    (the row still reports ``judge_mode: hybrid_prefix_latent_judge`` and the
    pass-through latent fields).
    """
    w = weights
    row = heuristic_surprise_score(prefix)
    latent_row = latent_row or {}
    latent_mismatch = _clip01(_value(latent_row, "latent_mismatch_score", 0.0))
    latent_alignment = _clip01(_value(latent_row, "latent_alignment_score", 1.0 - latent_mismatch))
    context_norm = max(0.0, _value(latent_row, "context_latent_norm", 0.0))
    predicted_norm = max(0.0, _value(latent_row, "predicted_future_latent_norm", 0.0))
    actual_norm = max(0.0, _value(latent_row, "actual_future_latent_norm", 0.0))

    mismatch_pressure = _clip01(
        w.mismatch_pressure_latent_mismatch * latent_mismatch
        + w.mismatch_pressure_norm_gap * abs(predicted_norm - actual_norm)
        + w.mismatch_pressure_misalignment * max(0.0, 1.0 - latent_alignment)
    )
    failure = _clip01(
        float(row["failure_score"])
        + w.failure_mismatch_pressure * mismatch_pressure
        + w.failure_unrewarded_mismatch * latent_mismatch * (1.0 - float(row["reward_evidence"]))
    )
    on_track = _clip01(
        float(row["on_track_score"]) - w.on_track_mismatch_pressure * mismatch_pressure
    )
    implausibility = _clip01(
        float(row["implausibility_score"]) + w.implausibility_latent_mismatch * latent_mismatch
    )
    uncertainty = _clip01(
        float(row["uncertainty_score"]) + w.uncertainty_mismatch_pressure * mismatch_pressure
    )

    row.update(
        {
            "on_track_score": round(on_track, 6),
            "failure_score": round(failure, 6),
            "implausibility_score": round(implausibility, 6),
            "uncertainty_score": round(uncertainty, 6),
            "latent_mismatch_score": round(latent_mismatch, 6),
            "latent_alignment_score": round(latent_alignment, 6),
            "context_latent_norm": round(context_norm, 6),
            "predicted_future_latent_norm": round(predicted_norm, 6),
            "actual_future_latent_norm": round(actual_norm, 6),
            "judge_mode": "hybrid_prefix_latent_judge",
        }
    )
    return row


def dummy_score(prefix: dict[str, Any]) -> dict[str, Any]:
    """Null judge row: all scores 0.0, uncertainty 1.0, ``judge_mode: dummy``.

    Kept in the package (not inlined in a CLI) so the null judge's record
    shape cannot drift from the real judges'.
    """
    return {
        "episode_id": prefix["episode_id"],
        "task_id": prefix["task_id"],
        "prefix_fraction": prefix["prefix_fraction"],
        "on_track_score": 0.0,
        "failure_score": 0.0,
        "implausibility_score": 0.0,
        "uncertainty_score": 1.0,
        "judge_mode": "dummy",
    }


def run_judge(
    prefixes: list[dict[str, Any]],
    mode: str = "heuristic_surprise",
    latent_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Score every prefix in ``mode``, owning the latent-cache join.

    ``latent_rows`` are joined to prefixes on
    :func:`leworldmodel_judge.schema.prefix_key`; they are only consumed by
    ``hybrid_surprise`` mode. Running ``hybrid_surprise`` without a cache is
    legal but logs a warning, because every score silently degrades to the
    composite judge; a non-empty cache that fails to join one or more prefixes
    (stale cache, wrong run) logs a warning with the joined/total counts, since
    every unjoined prefix degrades the same way while still carrying the hybrid
    provenance tag and default latent fields. Unknown modes raise
    ``ValueError``.
    """
    if mode not in JUDGE_MODES:
        raise ValueError(f"mode must be one of {JUDGE_MODES}; got {mode!r}")
    if mode == "dummy":
        return [dummy_score(prefix) for prefix in prefixes]
    if mode == "hybrid_surprise":
        latent_cache_map = {prefix_key(row): row for row in (latent_rows or [])}
        if not latent_cache_map:
            logger.warning(
                "hybrid_surprise mode has no latent cache; "
                "all scores degrade to the composite judge"
            )
        else:
            joined = sum(1 for prefix in prefixes if prefix_key(prefix) in latent_cache_map)
            if joined < len(prefixes):
                logger.warning(
                    "hybrid_surprise latent cache joined %d of %d prefixes; "
                    "unjoined prefixes degrade to the composite judge with "
                    "default latent fields under hybrid provenance",
                    joined,
                    len(prefixes),
                )
        return [
            hybrid_surprise_score(prefix, latent_cache_map.get(prefix_key(prefix)))
            for prefix in prefixes
        ]
    return [heuristic_surprise_score(prefix) for prefix in prefixes]
