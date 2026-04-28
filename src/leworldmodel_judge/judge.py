from __future__ import annotations


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _value(prefix: dict, key: str, fallback: float) -> float:
    value = prefix.get(key)
    if value is None:
        return float(fallback)
    return float(value)


def heuristic_surprise_score(prefix: dict) -> dict:
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
        0.22 * progress
        + 0.18 * distance_progress
        + 0.15 * target_proximity
        + 0.14 * in_place
        + 0.12 * near_object
        + 0.09 * grasp
        + 0.05 * reward_density
        + 0.05 * success
    )
    failure = _clip01(
        0.10 * stalled_without_transport
        + 0.12 * distance_gap * prefix_fraction
        + 0.12 * (1.0 - on_track)
        + 0.16 * engaged_but_not_transporting
        + 0.08 * prefix_fraction * (1.0 - in_place)
        + 0.06 * (1.0 - reward_density)
        + 0.36 * distance_regret
        - 0.18 * early_patience
        - 0.08 * success
    )
    implausibility = _clip01(
        0.30 * abs(progress - distance_progress)
        + 0.35 * distance_regret
        + 0.20 * max(0.0, engagement - in_place)
        + 0.15 * max(0.0, stall - reward_density)
    )

    evidence_low = min(progress, in_place, reward_density)
    evidence_high = max(progress, in_place, reward_density, success, engagement)
    uncertainty = _clip01(
        0.50 * abs(on_track - (1.0 - failure))
        + 0.30 * (evidence_high - evidence_low)
        + 0.20 * engaged_but_not_transporting
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


def hybrid_surprise_score(prefix: dict, latent_row: dict | None = None) -> dict:
    row = heuristic_surprise_score(prefix)
    latent_row = latent_row or {}
    latent_mismatch = _clip01(_value(latent_row, "latent_mismatch_score", 0.0))
    latent_alignment = _clip01(_value(latent_row, "latent_alignment_score", 1.0 - latent_mismatch))
    context_norm = max(0.0, _value(latent_row, "context_latent_norm", 0.0))
    predicted_norm = max(0.0, _value(latent_row, "predicted_future_latent_norm", 0.0))
    actual_norm = max(0.0, _value(latent_row, "actual_future_latent_norm", 0.0))

    mismatch_pressure = _clip01(
        0.65 * latent_mismatch
        + 0.20 * abs(predicted_norm - actual_norm)
        + 0.15 * max(0.0, 1.0 - latent_alignment)
    )
    failure = _clip01(
        float(row["failure_score"])
        + 0.22 * mismatch_pressure
        + 0.08 * latent_mismatch * (1.0 - float(row["reward_evidence"]))
    )
    on_track = _clip01(float(row["on_track_score"]) - 0.18 * mismatch_pressure)
    implausibility = _clip01(float(row["implausibility_score"]) + 0.20 * latent_mismatch)
    uncertainty = _clip01(float(row["uncertainty_score"]) + 0.15 * mismatch_pressure)

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
