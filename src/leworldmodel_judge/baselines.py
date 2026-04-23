from __future__ import annotations


def score_prefix(prefix: dict) -> dict:
    sparse_reward = float(prefix['sparse_reward_prefix'])
    final_success = 1.0 if prefix['final_success_label'] else 0.0
    progress_proxy = float(prefix.get('progress_proxy') or 0.0)
    return {
        'episode_id': prefix['episode_id'],
        'task_id': prefix['task_id'],
        'policy_family': prefix.get('policy_family'),
        'prefix_fraction': prefix['prefix_fraction'],
        'sparse_reward_score': sparse_reward,
        'terminal_success_score': final_success,
        'progress_proxy_score': progress_proxy,
    }
