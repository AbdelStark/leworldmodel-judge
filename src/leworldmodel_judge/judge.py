from __future__ import annotations


def heuristic_surprise_score(prefix: dict) -> dict:
    progress = float(prefix.get('progress_proxy') or 0.0)
    sparse_reward = float(prefix.get('sparse_reward_prefix') or 0.0)
    failure_hint = 1.0 if prefix.get('prefix_failure_label') else 0.0
    implausibility = max(0.0, failure_hint - progress)
    on_track = max(0.0, progress + sparse_reward)
    failure = max(0.0, failure_hint + (-progress if progress < 0 else 0.0))
    uncertainty = 1.0 if prefix.get('prefix_recoverability_label') == 'at_risk' else 0.25
    return {
        'episode_id': prefix['episode_id'],
        'task_id': prefix['task_id'],
        'prefix_fraction': prefix['prefix_fraction'],
        'on_track_score': round(on_track, 6),
        'failure_score': round(failure, 6),
        'implausibility_score': round(implausibility, 6),
        'uncertainty_score': round(uncertainty, 6),
        'judge_mode': 'heuristic_surprise',
    }
