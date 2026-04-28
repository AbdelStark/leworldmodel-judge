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
    assert summary["overall"]["judge_auroc"] == 1.0
    assert summary["overall"]["judge_average_precision"] == 1.0
    assert summary["overall"]["baseline_progress_pairwise_accuracy"] == 0.0
    assert summary["overall"]["baseline_progress_average_precision"] == 0.5
    assert summary["overall"]["baseline_sparse_absence_pairwise_accuracy"] == 0.5
    assert summary["overall"]["baseline_sparse_absence_average_precision"] == 0.5


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
    assert summary["overall"]["judge_average_precision"] is None
    assert summary["overall"]["baseline_progress_pairwise_accuracy"] is None
    assert summary["overall"]["baseline_progress_average_precision"] is None
    assert summary["overall"]["baseline_sparse_absence_pairwise_accuracy"] is None
    assert summary["overall"]["baseline_sparse_absence_average_precision"] is None


def test_summary_supports_disjoint_held_out_family_threshold_provenance():
    prefixes = [
        {
            "episode_id": "reach-v3-weak-ep-0",
            "task_id": "reach-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "prefix_failure_label": False,
        },
        {
            "episode_id": "reach-v3-doomed-ep-0",
            "task_id": "reach-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "prefix_failure_label": True,
        },
        {
            "episode_id": "push-v3-misleading-ep-0",
            "task_id": "push-v3",
            "policy_family": "misleading",
            "prefix_fraction": 0.75,
            "prefix_failure_label": False,
        },
        {
            "episode_id": "pick-place-v3-adversarial-ep-0",
            "task_id": "pick-place-v3",
            "policy_family": "adversarial",
            "prefix_fraction": 0.75,
            "prefix_failure_label": True,
        },
    ]
    baselines = [
        {
            "episode_id": "reach-v3-weak-ep-0",
            "task_id": "reach-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.55,
            "terminal_success_score": 1.0,
        },
        {
            "episode_id": "reach-v3-doomed-ep-0",
            "task_id": "reach-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.35,
            "terminal_success_score": 0.0,
        },
        {
            "episode_id": "push-v3-misleading-ep-0",
            "task_id": "push-v3",
            "policy_family": "misleading",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.6,
            "terminal_success_score": 1.0,
        },
        {
            "episode_id": "pick-place-v3-adversarial-ep-0",
            "task_id": "pick-place-v3",
            "policy_family": "adversarial",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.15,
            "terminal_success_score": 0.0,
        },
    ]
    judge = [
        {
            "episode_id": "reach-v3-weak-ep-0",
            "task_id": "reach-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "failure_score": 0.22,
        },
        {
            "episode_id": "reach-v3-doomed-ep-0",
            "task_id": "reach-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "failure_score": 0.81,
        },
        {
            "episode_id": "push-v3-misleading-ep-0",
            "task_id": "push-v3",
            "policy_family": "misleading",
            "prefix_fraction": 0.75,
            "failure_score": 0.43,
        },
        {
            "episode_id": "pick-place-v3-adversarial-ep-0",
            "task_id": "pick-place-v3",
            "policy_family": "adversarial",
            "prefix_fraction": 0.75,
            "failure_score": 0.91,
        },
    ]

    summary = summarize(
        prefixes,
        baselines,
        judge,
        calibration_families=["weak", "doomed"],
        evaluation_families=["misleading", "adversarial"],
    )

    assert summary["calibration"]["judge"]["mode"] == "held_out_family_split"
    assert summary["calibration"]["provenance"]["family_overlap"] is False
    assert summary["calibration"]["provenance"]["calibration_families"] == ["doomed", "weak"]
    assert summary["calibration"]["provenance"]["evaluation_families"] == ["adversarial", "misleading"]
    assert summary["calibration"]["judge"]["cohort_stats"]["count"] == 2
    assert summary["calibration"]["judge"]["evaluation_cohort"]["count"] == 2
    assert summary["calibration"]["judge"]["recommended_threshold"] == 0.81


def test_overlapping_family_splits_fall_back_to_in_slice_semantics():
    prefixes = [
        {
            "episode_id": "reach-v3-weak-ep-0",
            "task_id": "reach-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "prefix_failure_label": False,
        },
        {
            "episode_id": "reach-v3-doomed-ep-0",
            "task_id": "reach-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "prefix_failure_label": True,
        },
        {
            "episode_id": "push-v3-misleading-ep-0",
            "task_id": "push-v3",
            "policy_family": "misleading",
            "prefix_fraction": 0.75,
            "prefix_failure_label": False,
        },
    ]
    baselines = [
        {
            "episode_id": "reach-v3-weak-ep-0",
            "task_id": "reach-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.55,
            "terminal_success_score": 1.0,
        },
        {
            "episode_id": "reach-v3-doomed-ep-0",
            "task_id": "reach-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.35,
            "terminal_success_score": 0.0,
        },
        {
            "episode_id": "push-v3-misleading-ep-0",
            "task_id": "push-v3",
            "policy_family": "misleading",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.6,
            "terminal_success_score": 1.0,
        },
    ]
    judge = [
        {
            "episode_id": "reach-v3-weak-ep-0",
            "task_id": "reach-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "failure_score": 0.22,
        },
        {
            "episode_id": "reach-v3-doomed-ep-0",
            "task_id": "reach-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "failure_score": 0.81,
        },
        {
            "episode_id": "push-v3-misleading-ep-0",
            "task_id": "push-v3",
            "policy_family": "misleading",
            "prefix_fraction": 0.75,
            "failure_score": 0.43,
        },
    ]

    summary = summarize(
        prefixes,
        baselines,
        judge,
        calibration_families=["weak", "doomed"],
        evaluation_families=["weak", "doomed", "misleading"],
    )

    assert summary["calibration"]["judge"]["mode"] == "in_slice_balanced_accuracy"
    assert summary["calibration"]["provenance"]["family_overlap"] is True


def test_empty_evaluation_slice_does_not_claim_held_out_calibration():
    prefixes = [
        {
            "episode_id": "reach-v3-weak-ep-0",
            "task_id": "reach-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "prefix_failure_label": False,
        },
        {
            "episode_id": "reach-v3-doomed-ep-0",
            "task_id": "reach-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "prefix_failure_label": True,
        },
    ]
    baselines = [
        {
            "episode_id": "reach-v3-weak-ep-0",
            "task_id": "reach-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.55,
            "terminal_success_score": 1.0,
        },
        {
            "episode_id": "reach-v3-doomed-ep-0",
            "task_id": "reach-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.35,
            "terminal_success_score": 0.0,
        },
    ]
    judge = [
        {
            "episode_id": "reach-v3-weak-ep-0",
            "task_id": "reach-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "failure_score": 0.22,
        },
        {
            "episode_id": "reach-v3-doomed-ep-0",
            "task_id": "reach-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "failure_score": 0.81,
        },
    ]

    summary = summarize(
        prefixes,
        baselines,
        judge,
        calibration_families=["weak", "doomed"],
        evaluation_families=["missing"],
    )

    assert summary["calibration"]["judge"]["mode"] == "in_slice_balanced_accuracy"
    assert summary["calibration"]["provenance"]["evaluation_count"] == 0
    assert summary["overall"]["count"] == 0
    assert summary["overall"]["judge_failure_hit_rate"] is None
