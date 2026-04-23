from leworldmodel_judge.data import build_prefixes
from leworldmodel_judge.evaluate import summarize
from leworldmodel_judge.judge import heuristic_surprise_score
from scripts.collect_rollouts import collect_synthetic


def test_synthetic_doomed_family_creates_failure_with_partial_progress():
    rows = collect_synthetic('push-v3', 1, policy_family='doomed')
    prefix = build_prefixes(rows, (0.75,))[0].to_dict()

    assert prefix['prefix_failure_label'] is True
    assert prefix['progress_proxy'] > 0.3
    assert prefix['target_distance_last'] > prefix['target_distance_best']


def test_synthetic_expert_family_stays_recoverable():
    rows = collect_synthetic('push-v3', 1, policy_family='expert')
    prefix = build_prefixes(rows, (0.75,))[0].to_dict()

    assert prefix['final_success_label'] is True
    assert prefix['prefix_failure_label'] is False
    assert prefix['progress_proxy'] > 0.5


def test_synthetic_weak_family_emits_terminal_success_signal():
    rows = collect_synthetic('push-v3', 1, policy_family='weak')

    assert rows[-1]['success_label'] is True
    assert rows[-1]['info']['success'] == 1.0


def test_hard_synthetic_families_create_judge_vs_progress_divergence():
    weak_rows = collect_synthetic('push-v3', 1, policy_family='weak')
    doomed_rows = collect_synthetic('push-v3', 1, policy_family='doomed')
    prefixes = [p.to_dict() for p in build_prefixes(weak_rows + doomed_rows, (0.75,))]
    baselines = []
    judge = []
    for prefix in prefixes:
        baselines.append({
            'episode_id': prefix['episode_id'],
            'task_id': prefix['task_id'],
            'prefix_fraction': prefix['prefix_fraction'],
            'sparse_reward_score': prefix['sparse_reward_prefix'],
            'progress_proxy_score': prefix['progress_proxy'],
            'terminal_success_score': 1.0 if prefix['final_success_label'] else 0.0,
        })
        judge.append(heuristic_surprise_score(prefix))

    summary = summarize(prefixes, baselines, judge)

    assert summary['overall']['judge_pairwise_accuracy'] > summary['overall']['baseline_progress_pairwise_accuracy']
