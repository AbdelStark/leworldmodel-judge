"""Task-aware failure and recoverability labeling for rollout prefixes.

This is the most contested logic in the benchmark: every metric downstream is
measured against the labels produced here, so the rules live in one auditable
module. Each prefix receives two labels:

- ``prefix_failure_label`` (bool) — the trajectory is already effectively
  doomed at this cutoff.
- ``prefix_recoverability_label`` (str) — one of ``"recoverable"``,
  ``"at_risk"``, ``"doomed"``.

Rule constraints (see docs/method.md):

- Episodes that end in success are never failure-labeled; their prefixes are
  ``recoverable`` when progress or in-place evidence is decent, else
  ``at_risk``.
- No observed state signal means ``at_risk``, never ``doomed`` — the labeler
  refuses to condemn a prefix it cannot see.
- Task-specific gates exist for ``reach-v3``, ``pick-place-v3`` and
  ``push-v3`` (hardened 2026-04-28); every other task falls through to a
  generic stall/regret rule.

The numeric gates below are benchmark contract: changing them relabels the
dataset and shifts every reported metric, so changes must ship with
regenerated artifacts and a provenance note.
"""

from __future__ import annotations

from collections.abc import Mapping

from ._math import clip01 as _clip01

RECOVERABLE = "recoverable"
AT_RISK = "at_risk"
DOOMED = "doomed"


def _metric_value(metrics: Mapping[str, float | None], key: str, fallback: float) -> float:
    value = metrics.get(key)
    if value is None:
        return float(fallback)
    return float(value)


def label_prefix(
    task_id: str,
    *,
    final_success: bool,
    prefix_fraction: float,
    progress_proxy: float,
    metrics: Mapping[str, float | None],
) -> tuple[bool, str]:
    """Label one prefix as (prefix_failure_label, prefix_recoverability_label).

    ``metrics`` is the evidence dict produced by the prefix builder
    (``distance_progress``, ``in_place_score``, ``near_object_score``,
    ``grasp_signal_peak``, ``success_signal_peak``, ``reward_density``,
    ``target_distance_last``, ``target_distance_best``). Missing evidence is
    treated conservatively: ``None`` values default toward "no signal", and a
    prefix with no observed state signal at all is never labeled doomed.

    The judge never sees these labels; they exist only for evaluation.
    """
    progress = _clip01(progress_proxy)
    distance_progress_raw = metrics.get("distance_progress")
    in_place_raw = metrics.get("in_place_score")
    near_object_raw = metrics.get("near_object_score")
    grasp_raw = metrics.get("grasp_signal_peak")
    success_raw = metrics.get("success_signal_peak")
    reward_density_raw = metrics.get("reward_density")
    last_distance_raw = metrics.get("target_distance_last")

    distance_progress = _clip01(distance_progress_raw if distance_progress_raw is not None else 0.0)
    in_place = _clip01(in_place_raw if in_place_raw is not None else 0.0)
    near_object = _clip01(near_object_raw if near_object_raw is not None else 0.0)
    grasp = _clip01(grasp_raw if grasp_raw is not None else 0.0)
    success = _clip01(success_raw if success_raw is not None else 0.0)
    reward_density = _clip01(reward_density_raw if reward_density_raw is not None else 0.0)
    last_distance = float(last_distance_raw if last_distance_raw is not None else 1.0)

    has_observed_state_signal = (distance_progress_raw is not None) or (
        last_distance_raw is not None
    )

    stalled = distance_progress < 0.10 and in_place < 0.20
    far_from_target = last_distance > 0.20
    no_contact = grasp < 0.15 and success == 0.0
    engaged = max(grasp, near_object)
    transport = max(in_place, success, distance_progress)
    weak_prefix = progress < 0.20 and reward_density < 0.10
    distance_regret = max(
        0.0, last_distance - _metric_value(metrics, "target_distance_best", last_distance)
    )

    if final_success:
        if progress >= 0.35 or success > 0.0 or in_place >= 0.35:
            return False, RECOVERABLE
        return False, AT_RISK

    if not has_observed_state_signal:
        return False, AT_RISK

    if task_id == "reach-v3":
        if prefix_fraction >= 0.50 and distance_progress < 0.10 and last_distance > 0.18:
            return True, DOOMED
        if progress >= 0.45 and last_distance < 0.12:
            return False, RECOVERABLE
        return False, AT_RISK

    if task_id == "pick-place-v3":
        if (
            prefix_fraction >= 0.75
            and engaged >= 0.60
            and transport < 0.20
            and last_distance > 0.28
        ):
            return True, DOOMED
        if (
            prefix_fraction >= 0.75
            and engaged >= 0.50
            and distance_regret > 0.08
            and last_distance > 0.28
        ):
            return True, DOOMED
        if prefix_fraction >= 0.50 and engaged < 0.20 and far_from_target and weak_prefix:
            return True, DOOMED
        if transport >= 0.35:
            return False, RECOVERABLE
        return False, AT_RISK

    if task_id == "push-v3":
        if prefix_fraction >= 0.75 and distance_regret > 0.18 and last_distance > 0.30:
            return True, DOOMED
        if (
            prefix_fraction >= 0.75
            and distance_progress < 0.05
            and last_distance > 0.18
            and in_place < 0.25
        ):
            return True, DOOMED
        if (
            prefix_fraction >= 0.75
            and engaged >= 0.45
            and transport < 0.25
            and distance_regret > 0.08
            and last_distance > 0.18
        ):
            return True, DOOMED
        if prefix_fraction >= 0.50 and engaged < 0.20 and far_from_target and weak_prefix:
            return True, DOOMED
        if (
            prefix_fraction >= 0.50
            and engaged < 0.20
            and last_distance > 0.20
            and reward_density < 0.12
        ):
            return True, DOOMED
        if transport >= 0.35:
            return False, RECOVERABLE
        return False, AT_RISK

    if prefix_fraction >= 0.75 and distance_regret > 0.18 and last_distance > 0.30:
        return True, DOOMED
    if prefix_fraction >= 0.75 and stalled and far_from_target and no_contact:
        return True, DOOMED
    if prefix_fraction >= 0.50 and stalled and far_from_target and weak_prefix and no_contact:
        return True, DOOMED
    if progress >= 0.55 or in_place >= 0.50:
        return False, RECOVERABLE
    return False, AT_RISK
