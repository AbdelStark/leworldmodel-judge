from __future__ import annotations


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _value(prefix: dict, key: str, fallback: float) -> float:
    value = prefix.get(key)
    if value is None:
        return float(fallback)
    return float(value)


def heuristic_surprise_score(prefix: dict) -> dict:
    progress = _clip01(_value(prefix, 'progress_proxy', 0.0))
    sparse_reward = _clip01(max(0.0, _value(prefix, 'sparse_reward_prefix', 0.0)))
    distance_progress = _clip01(_value(prefix, 'distance_progress', progress))
    in_place = _clip01(_value(prefix, 'in_place_score', 0.0))
    grasp = _clip01(_value(prefix, 'grasp_signal_peak', 0.0))
    success = _clip01(_value(prefix, 'success_signal_peak', 0.0))
    reward_density = _clip01(_value(prefix, 'reward_density', sparse_reward))
    stall = _clip01(_value(prefix, 'stall_score', (1.0 - distance_progress)))
    prefix_fraction = _clip01(_value(prefix, 'prefix_fraction', 0.0))
    target_distance_last = _value(prefix, 'target_distance_last', 1.0)
    target_distance_best = _value(prefix, 'target_distance_best', target_distance_last)
    distance_gap = _clip01(target_distance_last / max(target_distance_last, 0.30)) if target_distance_last > 0 else 0.0
    target_proximity = _clip01(1.0 - distance_gap)
    distance_regret = _clip01(max(0.0, target_distance_last - target_distance_best) / max(target_distance_last, 0.30))

    on_track = _clip01(
        0.25 * progress
        + 0.20 * distance_progress
        + 0.20 * target_proximity
        + 0.15 * in_place
        + 0.08 * grasp
        + 0.07 * reward_density
        + 0.05 * success
    )
    failure = _clip01(
        0.12 * stall
        + 0.18 * distance_gap
        + 0.16 * (1.0 - on_track)
        + 0.10 * prefix_fraction * (1.0 - in_place)
        + 0.06 * (1.0 - reward_density)
        + 0.05 * (1.0 - target_proximity)
        + 0.33 * distance_regret
    )
    implausibility = _clip01(
        0.35 * abs(progress - distance_progress)
        + 0.40 * distance_regret
        + 0.25 * max(0.0, stall - reward_density)
    )

    evidence_low = min(progress, in_place, reward_density)
    evidence_high = max(progress, in_place, reward_density, success)
    uncertainty = _clip01(
        0.55 * abs(on_track - (1.0 - failure))
        + 0.45 * (evidence_high - evidence_low)
    )

    return {
        'episode_id': prefix['episode_id'],
        'task_id': prefix['task_id'],
        'prefix_fraction': prefix['prefix_fraction'],
        'on_track_score': round(on_track, 6),
        'failure_score': round(failure, 6),
        'implausibility_score': round(implausibility, 6),
        'uncertainty_score': round(uncertainty, 6),
        'progress_evidence': round(progress, 6),
        'distance_progress_evidence': round(distance_progress, 6),
        'in_place_evidence': round(in_place, 6),
        'grasp_evidence': round(grasp, 6),
        'reward_evidence': round(reward_density, 6),
        'stall_evidence': round(stall, 6),
        'judge_mode': 'composite_prefix_judge',
    }
