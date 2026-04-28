from leworldmodel_judge.evaluate import summarize


def test_summary_reports_pairwise_ranking_accuracy_for_each_signal():
    prefixes = [
        {
            "episode_id": "fail-1",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "prefix_failure_label": True,
        },
        {
            "episode_id": "ok-1",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "prefix_failure_label": False,
        },
    ]
    baselines = [
        {
            "episode_id": "fail-1",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.6,
            "terminal_success_score": 0.0,
        },
        {
            "episode_id": "ok-1",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.2,
            "terminal_success_score": 1.0,
        },
    ]
    judge = [
        {
            "episode_id": "fail-1",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "failure_score": 0.9,
        },
        {
            "episode_id": "ok-1",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "failure_score": 0.1,
        },
    ]

    summary = summarize(prefixes, baselines, judge)

    assert summary["overall"]["judge_pairwise_accuracy"] == 1.0
    assert summary["overall"]["baseline_progress_pairwise_accuracy"] == 0.0
    assert summary["overall"]["baseline_sparse_absence_pairwise_accuracy"] == 0.5


def test_pairwise_accuracy_is_none_when_only_one_label_class_exists():
    prefixes = [
        {
            "episode_id": "ok-1",
            "task_id": "reach-v3",
            "prefix_fraction": 0.75,
            "prefix_failure_label": False,
        }
    ]
    baselines = [
        {
            "episode_id": "ok-1",
            "task_id": "reach-v3",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.4,
            "terminal_success_score": 1.0,
        }
    ]
    judge = [
        {
            "episode_id": "ok-1",
            "task_id": "reach-v3",
            "prefix_fraction": 0.75,
            "failure_score": 0.1,
        }
    ]

    summary = summarize(prefixes, baselines, judge)

    assert summary["overall"]["judge_pairwise_accuracy"] is None
    assert summary["overall"]["baseline_progress_pairwise_accuracy"] is None
    assert summary["overall"]["baseline_sparse_absence_pairwise_accuracy"] is None
