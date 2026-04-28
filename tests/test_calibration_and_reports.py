import csv
from pathlib import Path

from leworldmodel_judge.data import build_prefixes
from leworldmodel_judge.evaluate import summarize
from leworldmodel_judge.judge import heuristic_surprise_score


def test_pick_place_late_grasp_without_transport_is_labeled_doomed():
    steps = []
    distances = [0.34, 0.31, 0.30, 0.30]
    for t, distance in enumerate(distances):
        steps.append(
            {
                "episode_id": "pick-place-doomed-1",
                "task_id": "pick-place-v3",
                "timestep": t,
                "episode_horizon": len(distances),
                "observation": [0.0, 0.0],
                "action": [0.0],
                "reward": 0.0,
                "done": t == len(distances) - 1,
                "success_label": False,
                "info": {
                    "near_object": 1.0 if t >= 1 else 0.0,
                    "obj_to_target": distance,
                    "in_place_reward": 0.12,
                    "grasp_success": 0.0,
                    "grasp_reward": 1.0 if t >= 1 else 0.0,
                    "success": 0.0,
                    "unscaled_reward": 0.08,
                },
            }
        )

    prefix = build_prefixes(steps, (0.75,))[0].to_dict()

    assert prefix["prefix_failure_label"] is True
    assert prefix["prefix_recoverability_label"] == "doomed"


def test_judge_is_more_patient_on_early_engaged_prefixes_than_late_stalled_ones():
    early = {
        "episode_id": "ep-early",
        "task_id": "pick-place-v3",
        "prefix_fraction": 0.25,
        "progress_proxy": 0.0,
        "sparse_reward_prefix": 0.0,
        "distance_progress": 0.0,
        "target_distance_last": 0.30,
        "target_distance_best": 0.30,
        "in_place_score": 0.14,
        "near_object_score": 1.0,
        "success_signal_peak": 0.0,
        "grasp_signal_peak": 1.0,
        "reward_density": 0.08,
        "stall_score": 1.0,
    }
    late = dict(early)
    late["episode_id"] = "ep-late"
    late["prefix_fraction"] = 0.75
    late["target_distance_last"] = 0.42
    late["target_distance_best"] = 0.28

    early_score = heuristic_surprise_score(early)["failure_score"]
    late_score = heuristic_surprise_score(late)["failure_score"]

    assert early_score < 0.5
    assert late_score > early_score


def test_summary_reports_calibrated_thresholds_and_family_slices(tmp_path):
    prefixes = [
        {
            "episode_id": "push-v3-weak-ep-0",
            "task_id": "push-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "prefix_failure_label": False,
        },
        {
            "episode_id": "push-v3-doomed-ep-0",
            "task_id": "push-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "prefix_failure_label": True,
        },
    ]
    baselines = [
        {
            "episode_id": "push-v3-weak-ep-0",
            "task_id": "push-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.45,
            "terminal_success_score": 1.0,
        },
        {
            "episode_id": "push-v3-doomed-ep-0",
            "task_id": "push-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.35,
            "terminal_success_score": 0.0,
        },
    ]
    judge = [
        {
            "episode_id": "push-v3-weak-ep-0",
            "task_id": "push-v3",
            "policy_family": "weak",
            "prefix_fraction": 0.75,
            "failure_score": 0.31,
        },
        {
            "episode_id": "push-v3-doomed-ep-0",
            "task_id": "push-v3",
            "policy_family": "doomed",
            "prefix_fraction": 0.75,
            "failure_score": 0.72,
        },
    ]

    summary = summarize(prefixes, baselines, judge)

    assert summary["calibration"]["judge"]["recommended_threshold"] == 0.72
    assert summary["families"]["doomed"]["failure_labels"] == 1
    assert summary["families"]["weak"]["non_failure_labels"] == 1

    report_dir = tmp_path / "report"
    report_dir.mkdir()
    from scripts.render_family_report import render_family_report

    outputs = render_family_report(summary, report_dir)
    assert Path(outputs["markdown"]).exists()
    assert Path(outputs["plot"]).exists()
    report_text = Path(outputs["markdown"]).read_text()
    assert "Threshold provenance" in report_text
    assert "in-slice tuning" in report_text
    assert "family overlap" in report_text
    assert "non-failure labels" in report_text


def test_render_demo_emits_markdown_csv_and_timeline_plot(tmp_path):
    prefixes = [
        {
            "episode_id": "push-v3-doomed-ep-0",
            "task_id": "push-v3",
            "policy_family": "doomed",
            "prefix_index": 15,
            "prefix_fraction": 0.75,
            "final_success_label": False,
            "prefix_failure_label": True,
            "prefix_recoverability_label": "doomed",
            "progress_proxy": 0.18,
            "distance_progress": 0.12,
            "target_distance_last": 0.33,
            "target_distance_best": 0.22,
            "in_place_score": 0.14,
            "grasp_signal_peak": 1.0,
            "success_signal_peak": 0.0,
            "reward_density": 0.09,
            "stall_score": 0.88,
        },
        {
            "episode_id": "push-v3-weak-ep-0",
            "task_id": "push-v3",
            "policy_family": "weak",
            "prefix_index": 10,
            "prefix_fraction": 0.5,
            "final_success_label": True,
            "prefix_failure_label": False,
            "prefix_recoverability_label": "recoverable",
            "progress_proxy": 0.61,
            "distance_progress": 0.61,
            "target_distance_last": 0.08,
            "target_distance_best": 0.08,
            "in_place_score": 0.72,
            "grasp_signal_peak": 1.0,
            "success_signal_peak": 0.0,
            "reward_density": 0.42,
            "stall_score": 0.19,
        },
    ]
    baselines = [
        {
            "episode_id": "push-v3-doomed-ep-0",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.55,
            "terminal_success_score": 0.0,
        },
        {
            "episode_id": "push-v3-weak-ep-0",
            "task_id": "push-v3",
            "prefix_fraction": 0.5,
            "sparse_reward_score": 0.0,
            "progress_proxy_score": 0.65,
            "terminal_success_score": 1.0,
        },
    ]
    judge = [
        {
            "episode_id": "push-v3-doomed-ep-0",
            "task_id": "push-v3",
            "prefix_fraction": 0.75,
            "failure_score": 0.81,
            "on_track_score": 0.16,
            "implausibility_score": 0.77,
            "uncertainty_score": 0.21,
        },
        {
            "episode_id": "push-v3-weak-ep-0",
            "task_id": "push-v3",
            "prefix_fraction": 0.5,
            "failure_score": 0.18,
            "on_track_score": 0.82,
            "implausibility_score": 0.12,
            "uncertainty_score": 0.11,
        },
    ]

    from leworldmodel_judge.io import write_jsonl
    from scripts.render_demo import _comparison_rows
    from scripts.render_demo import main as render_demo_main

    prefixes_path = tmp_path / "prefixes.jsonl"
    baselines_path = tmp_path / "baselines.jsonl"
    judge_path = tmp_path / "judge.jsonl"
    output_path = tmp_path / "demo.md"
    write_jsonl(prefixes_path, prefixes)
    write_jsonl(baselines_path, baselines)
    write_jsonl(judge_path, judge)

    import sys

    old_argv = sys.argv
    sys.argv = [
        "render_demo.py",
        "--prefixes",
        str(prefixes_path),
        "--baselines",
        str(baselines_path),
        "--judge",
        str(judge_path),
        "--output",
        str(output_path),
    ]
    try:
        render_demo_main()
    finally:
        sys.argv = old_argv

    csv_path = tmp_path / "demo-comparison.csv"
    plot_png_path = tmp_path / "demo-timeline.png"
    plot_svg_path = tmp_path / "demo-timeline.svg"
    disagreement_pack_path = tmp_path / "demo-push-v3-hard-disagreement-pack.csv"
    assert output_path.exists()
    assert csv_path.exists()
    assert plot_png_path.exists() or plot_svg_path.exists()
    assert disagreement_pack_path.exists()

    markdown = output_path.read_text()
    assert "Biggest baseline-vs-judge disagreements" in markdown
    assert "Push-v3 hard-family disagreement pack" in markdown
    assert "Evidence decomposition example" in markdown

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert rows[0]["task_id"] == "push-v3"
    assert "baseline_vs_judge_gap" in rows[0]
    assert _comparison_rows(prefixes, baselines, judge)[0]["judge_metric"] == 0.81
