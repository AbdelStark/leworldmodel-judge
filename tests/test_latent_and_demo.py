from __future__ import annotations

from leworldmodel_judge.judge import heuristic_surprise_score, hybrid_surprise_score
from leworldmodel_judge.latents import build_latent_cache
from scripts import render_demo


def _step(
    episode_id: str,
    timestep: int,
    observation: list[float],
    *,
    task_id: str = "push-v3",
    success_label: bool = False,
    family: str = "misleading",
) -> dict:
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "timestep": timestep,
        "episode_horizon": 6,
        "observation": observation,
        "action": [0.05, -0.02],
        "reward": 0.0,
        "done": timestep == 5,
        "success_label": success_label,
        "info": {
            "policy_family": family,
            "obj_to_target": max(0.05, 0.40 - 0.03 * timestep),
            "in_place_reward": 0.10 + 0.02 * timestep,
            "near_object": 0.15 + 0.05 * timestep,
            "grasp_success": 0.0,
            "grasp_reward": 0.05 * timestep,
            "success": 0.0,
            "unscaled_reward": 0.02 * timestep,
        },
    }


def test_build_latent_cache_emits_predicted_and_actual_future_latents() -> None:
    prefixes = [
        {
            "episode_id": "push-v3-misleading-ep-0",
            "task_id": "push-v3",
            "policy_family": "misleading",
            "prefix_fraction": 0.5,
            "prefix_index": 3,
        }
    ]
    steps = [
        _step("push-v3-misleading-ep-0", 0, [0.00, 0.10, 0.20]),
        _step("push-v3-misleading-ep-0", 1, [0.10, 0.15, 0.25]),
        _step("push-v3-misleading-ep-0", 2, [0.20, 0.20, 0.30]),
        _step("push-v3-misleading-ep-0", 3, [0.45, 0.55, 0.70]),
        _step("push-v3-misleading-ep-0", 4, [0.60, 0.75, 0.90]),
        _step("push-v3-misleading-ep-0", 5, [0.80, 0.95, 1.10]),
    ]

    cache = build_latent_cache(prefixes, steps)

    assert len(cache) == 1
    row = cache[0]
    assert row["episode_id"] == "push-v3-misleading-ep-0"
    assert row["task_id"] == "push-v3"
    assert row["latent_cache_version"].startswith("v")
    assert len(row["context_latent"]) > 0
    assert len(row["predicted_future_latent"]) == len(row["actual_future_latent"])
    assert row["latent_mismatch_score"] > 0.0


def test_hybrid_surprise_score_appends_latent_fields_and_changes_failure_score() -> None:
    prefix = {
        "episode_id": "push-v3-misleading-ep-0",
        "task_id": "push-v3",
        "policy_family": "misleading",
        "prefix_fraction": 0.75,
        "progress_proxy": 0.08,
        "sparse_reward_prefix": 0.0,
        "distance_progress": 0.04,
        "target_distance_last": 0.34,
        "target_distance_best": 0.28,
        "in_place_score": 0.12,
        "near_object_score": 0.65,
        "grasp_signal_peak": 0.05,
        "success_signal_peak": 0.0,
        "reward_density": 0.02,
        "stall_score": 0.94,
    }
    latent_row = {
        "episode_id": prefix["episode_id"],
        "task_id": prefix["task_id"],
        "prefix_fraction": prefix["prefix_fraction"],
        "latent_mismatch_score": 0.82,
        "context_latent_norm": 0.61,
        "predicted_future_latent_norm": 0.18,
        "actual_future_latent_norm": 1.07,
        "latent_alignment_score": 0.12,
    }

    heuristic = heuristic_surprise_score(prefix)
    hybrid = hybrid_surprise_score(prefix, latent_row)

    assert hybrid["judge_mode"] == "hybrid_prefix_latent_judge"
    assert hybrid["latent_mismatch_score"] == 0.82
    assert hybrid["latent_alignment_score"] == 0.12
    assert hybrid["failure_score"] > heuristic["failure_score"]


def test_timeline_plot_uses_real_svg_fallback_without_matplotlib(tmp_path, monkeypatch) -> None:
    rows = [
        {
            "prefix_cutoff": 0.25,
            "judge_metric": 0.7,
            "baseline_metric": 0.3,
            "sparse_reward_signal": 0.0,
        },
        {
            "prefix_cutoff": 0.75,
            "judge_metric": 0.9,
            "baseline_metric": 0.4,
            "sparse_reward_signal": 0.1,
        },
    ]
    target = tmp_path / "timeline.svg"

    monkeypatch.setattr(render_demo, "MATPLOTLIB_AVAILABLE", False)
    monkeypatch.setattr(render_demo, "plt", None)
    render_demo._write_timeline_plot(rows, target)

    payload = target.read_text(encoding="utf-8")
    assert payload.lstrip().startswith("<svg")
    assert "judge failure score" in payload
    assert "<polyline" in payload


def test_push_v3_hard_disagreement_pack_prefers_family_diversity() -> None:
    rows = [
        {
            "task_id": "push-v3",
            "episode_id": "push-v3-expert-ep-0",
            "policy_family": "expert",
            "prefix_cutoff": 0.75,
            "baseline_vs_judge_gap": 0.55,
            "judge_metric": 0.70,
            "baseline_metric": 0.15,
            "judge_uncertainty_score": 0.40,
            "prefix_failure_label": False,
            "prefix_recoverability_label": "recoverable",
        },
        {
            "task_id": "push-v3",
            "episode_id": "push-v3-weak-ep-0",
            "policy_family": "weak",
            "prefix_cutoff": 0.75,
            "baseline_vs_judge_gap": 0.65,
            "judge_metric": 0.82,
            "baseline_metric": 0.17,
            "judge_uncertainty_score": 0.35,
            "prefix_failure_label": True,
            "prefix_recoverability_label": "doomed",
        },
        {
            "task_id": "push-v3",
            "episode_id": "push-v3-doomed-ep-0",
            "policy_family": "doomed",
            "prefix_cutoff": 0.50,
            "baseline_vs_judge_gap": 0.44,
            "judge_metric": 0.73,
            "baseline_metric": 0.29,
            "judge_uncertainty_score": 0.52,
            "prefix_failure_label": True,
            "prefix_recoverability_label": "doomed",
        },
        {
            "task_id": "push-v3",
            "episode_id": "push-v3-misleading-ep-0",
            "policy_family": "misleading",
            "prefix_cutoff": 0.50,
            "baseline_vs_judge_gap": 0.61,
            "judge_metric": 0.79,
            "baseline_metric": 0.18,
            "judge_uncertainty_score": 0.62,
            "prefix_failure_label": False,
            "prefix_recoverability_label": "at_risk",
        },
        {
            "task_id": "reach-v3",
            "episode_id": "reach-v3-weak-ep-0",
            "policy_family": "weak",
            "prefix_cutoff": 0.75,
            "baseline_vs_judge_gap": 0.99,
            "judge_metric": 0.99,
            "baseline_metric": 0.0,
            "judge_uncertainty_score": 0.99,
            "prefix_failure_label": True,
            "prefix_recoverability_label": "doomed",
        },
    ]

    pack = render_demo._build_push_v3_hard_disagreement_pack(rows, limit=4)

    assert len(pack) == 4
    assert {row["task_id"] for row in pack} == {"push-v3"}
    assert {row["policy_family"] for row in pack} == {"expert", "weak", "doomed", "misleading"}
