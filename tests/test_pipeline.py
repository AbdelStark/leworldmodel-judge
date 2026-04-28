import json
import os
import subprocess
import sys
from pathlib import Path

from leworldmodel_judge.baselines import score_prefix
from leworldmodel_judge.data import build_prefixes
from leworldmodel_judge.evaluate import summarize
from leworldmodel_judge.judge import heuristic_surprise_score
from leworldmodel_judge.tasks import LOCKED_TASKS, resolve_tasks
from scripts.collect_rollouts import collect_synthetic

ROOT = Path(__file__).resolve().parents[1]


def test_prefix_pipeline_smoke():
    steps = []
    for t in range(4):
        steps.append(
            {
                "episode_id": "ep1",
                "task_id": "reach-v3",
                "timestep": t,
                "episode_horizon": 4,
                "observation": [1.0 - t * 0.2, 0.0],
                "action": [0.0],
                "reward": 0.0,
                "done": t == 3,
                "success_label": True,
            }
        )
    prefixes = build_prefixes(steps, (0.5,))
    assert len(prefixes) == 1
    baseline = score_prefix(prefixes[0].to_dict())
    judge = heuristic_surprise_score(prefixes[0].to_dict())
    assert baseline["task_id"] == "reach-v3"
    assert "failure_score" in judge


def test_resolve_tasks_all_uses_locked_benchmark_slice():
    assert resolve_tasks("all") == list(LOCKED_TASKS)


def test_collect_rollouts_script_supports_all_task_bundle(tmp_path):
    output_path = tmp_path / "rollouts.jsonl"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "collect_rollouts.py"),
            "--source",
            "synthetic",
            "--task",
            "all",
            "--episodes",
            "1",
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert {row["task_id"] for row in rows} == set(LOCKED_TASKS)


def test_prefix_labels_detect_doomed_metaworld_style_prefix():
    steps = []
    for t in range(4):
        steps.append(
            {
                "episode_id": "push-ep-1",
                "task_id": "push-v3",
                "timestep": t,
                "episode_horizon": 4,
                "observation": [0.0, 0.0],
                "action": [0.0],
                "reward": 0.0,
                "done": t == 3,
                "success_label": False,
                "info": {
                    "obj_to_target": 0.33,
                    "in_place_reward": 0.05,
                    "grasp_success": 0.0,
                    "grasp_reward": 0.0,
                    "success": 0.0,
                    "unscaled_reward": 0.02,
                },
            }
        )
    prefixes = build_prefixes(steps, (0.75,))
    assert len(prefixes) == 1
    prefix = prefixes[0].to_dict()
    assert prefix["prefix_failure_label"] is True
    assert prefix["prefix_recoverability_label"] == "doomed"


def test_push_v3_late_contact_without_transport_is_now_labeled_doomed():
    steps = []
    for t in range(4):
        steps.append(
            {
                "episode_id": "push-stall-ep-1",
                "task_id": "push-v3",
                "timestep": t,
                "episode_horizon": 4,
                "observation": [0.0, 0.0],
                "action": [0.0],
                "reward": 1.2 if t >= 2 else 0.2,
                "done": t == 3,
                "success_label": False,
                "info": {
                    "obj_to_target": 0.29 if t < 3 else 0.28,
                    "near_object": 1.0,
                    "in_place_reward": 0.18,
                    "grasp_success": 0.8,
                    "grasp_reward": 0.8,
                    "success": 0.0,
                    "unscaled_reward": 1.2 if t >= 2 else 0.2,
                },
            }
        )
    prefix = build_prefixes(steps, (0.75,))[0].to_dict()
    assert prefix["final_success_label"] is False
    assert prefix["prefix_failure_label"] is True
    assert prefix["prefix_recoverability_label"] == "doomed"


def test_reach_failures_do_not_get_masked_by_grasp_or_dense_reward():
    steps = []
    for t in range(4):
        steps.append(
            {
                "episode_id": "reach-ep-1",
                "task_id": "reach-v3",
                "timestep": t,
                "episode_horizon": 4,
                "observation": [0.0, 0.0],
                "action": [0.0],
                "reward": 1.2,
                "done": t == 3,
                "success_label": False,
                "info": {
                    "obj_to_target": 0.28,
                    "in_place_reward": 0.14,
                    "grasp_success": 1.0,
                    "grasp_reward": 1.0,
                    "success": 0.0,
                    "unscaled_reward": 1.2,
                },
            }
        )
    prefix = build_prefixes(steps, (0.75,))[0].to_dict()
    assert prefix["prefix_failure_label"] is True
    assert prefix["prefix_recoverability_label"] == "doomed"


def test_missing_metaworld_info_stays_missing_not_fake_zero():
    steps = [
        {
            "episode_id": "ep-missing",
            "task_id": "reach-v3",
            "timestep": 0,
            "episode_horizon": 1,
            "observation": [0.1, 0.2],
            "action": [0.0],
            "reward": 0.0,
            "done": True,
            "success_label": False,
            "info": {"source": "metaworld"},
        }
    ]

    prefix = build_prefixes(steps, (1.0,))[0].to_dict()

    assert prefix["target_distance_start"] is None
    assert prefix["target_distance_last"] is None
    assert prefix["prefix_failure_label"] is False
    assert prefix["prefix_recoverability_label"] == "at_risk"


def test_sparse_reward_prefix_uses_success_signal_not_dense_reward():
    steps = []
    for t in range(3):
        steps.append(
            {
                "episode_id": "ep-dense",
                "task_id": "push-v3",
                "timestep": t,
                "episode_horizon": 3,
                "observation": [0.0, 0.0],
                "action": [0.0],
                "reward": 0.7,
                "done": t == 2,
                "success_label": False,
                "info": {"source": "metaworld", "success": 0.0},
            }
        )

    prefix = build_prefixes(steps, (1.0,))[0].to_dict()

    assert prefix["sparse_reward_prefix"] == 0.0


def test_successful_synthetic_prefix_has_positive_progress_signal():
    rows = collect_synthetic("reach-v3", 1, policy_family="expert")
    prefix = build_prefixes(rows, (0.5,))[0].to_dict()

    assert prefix["progress_proxy"] > 0.0
    assert prefix["distance_progress"] > 0.0
    assert prefix["stall_score"] < 1.0


def test_zero_distance_metrics_are_not_replaced_by_fallback_defaults():
    prefix = {
        "episode_id": "at-target",
        "task_id": "reach-v3",
        "prefix_fraction": 0.75,
        "progress_proxy": 1.0,
        "sparse_reward_prefix": 0.0,
        "distance_progress": 0.0,
        "target_distance_last": 0.0,
        "target_distance_best": 0.0,
        "in_place_score": 1.0,
        "success_signal_peak": 0.0,
        "grasp_signal_peak": 0.0,
        "reward_density": 0.0,
        "stall_score": 0.0,
    }

    score = heuristic_surprise_score(prefix)

    assert score["failure_score"] < 0.3
    assert score["on_track_score"] > 0.5


def test_composite_judge_prefers_progressing_prefix_over_stalled_prefix():
    healthy = {
        "episode_id": "reach-good",
        "task_id": "reach-v3",
        "prefix_fraction": 0.5,
        "progress_proxy": 0.6,
        "sparse_reward_prefix": 0.1,
        "distance_progress": 0.7,
        "target_distance_last": 0.05,
        "target_distance_best": 0.05,
        "in_place_score": 0.85,
        "success_signal_peak": 0.0,
        "grasp_signal_peak": 1.0,
        "reward_density": 0.2,
        "stall_score": 0.0,
    }
    stalled = {
        "episode_id": "reach-bad",
        "task_id": "reach-v3",
        "prefix_fraction": 0.75,
        "progress_proxy": 0.02,
        "sparse_reward_prefix": 0.0,
        "distance_progress": 0.01,
        "target_distance_last": 0.31,
        "target_distance_best": 0.31,
        "in_place_score": 0.05,
        "success_signal_peak": 0.0,
        "grasp_signal_peak": 0.0,
        "reward_density": 0.0,
        "stall_score": 0.95,
    }

    healthy_score = heuristic_surprise_score(healthy)
    stalled_score = heuristic_surprise_score(stalled)

    assert healthy_score["on_track_score"] > stalled_score["on_track_score"]
    assert stalled_score["failure_score"] > healthy_score["failure_score"]


def test_judge_does_not_change_when_failure_labels_change():
    prefix = {
        "episode_id": "ep1",
        "task_id": "push-v3",
        "prefix_fraction": 0.75,
        "progress_proxy": 0.18,
        "sparse_reward_prefix": 0.0,
        "distance_progress": 0.05,
        "target_distance_last": 0.28,
        "target_distance_best": 0.27,
        "in_place_score": 0.1,
        "success_signal_peak": 0.0,
        "grasp_signal_peak": 0.0,
        "reward_density": 0.01,
        "stall_score": 0.95,
        "prefix_failure_label": False,
        "prefix_recoverability_label": "recoverable",
    }
    flipped = dict(prefix)
    flipped["prefix_failure_label"] = True
    flipped["prefix_recoverability_label"] = "doomed"

    assert heuristic_surprise_score(prefix) == heuristic_surprise_score(flipped)


def test_summary_reports_hits_false_positives_and_coverage():
    prefixes = [
        {
            "episode_id": "ep1",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "prefix_failure_label": True,
        },
        {
            "episode_id": "ep2",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "prefix_failure_label": False,
        },
    ]
    baselines = [
        {
            "episode_id": "ep1",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.1,
            "terminal_success_score": 0.0,
        },
        {
            "episode_id": "ep2",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.1,
            "terminal_success_score": 1.0,
        },
    ]
    judge = [
        {
            "episode_id": "ep1",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "failure_score": 0.8,
        },
        {
            "episode_id": "ep2",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "failure_score": 0.2,
        },
    ]

    summary = summarize(prefixes, baselines, judge)

    assert summary["overall"]["judge_failure_hits"] == 1
    assert summary["overall"]["judge_false_positives"] == 0
    assert summary["overall"]["baseline_sparse_absence_hits"] == 1
    assert summary["overall"]["baseline_sparse_absence_false_positives"] == 1
    assert summary["overall"]["baseline_progress_hits"] == 1
    assert summary["overall"]["baseline_progress_false_positives"] == 1
    assert summary["overall"]["failure_label_coverage"] == 0.5
