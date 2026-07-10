"""End-to-end tests of the ``lewm-judge`` CLI surface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from leworldmodel_judge.cli import main

ROOT = Path(__file__).resolve().parents[1]

SUBCOMMANDS = (
    "collect",
    "prefixes",
    "latents",
    "baselines",
    "judge",
    "evaluate",
    "report",
    "demo",
)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``python -m leworldmodel_judge`` in a subprocess (real stderr/exit)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "leworldmodel_judge", *args],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def _single_error_line(stderr: str) -> str:
    """Assert stderr is exactly one clean error line (no traceback), return it."""
    assert "Traceback" not in stderr
    lines = [line for line in stderr.splitlines() if line.strip()]
    assert len(lines) == 1, stderr
    return lines[0]


def test_full_pipeline_runs_end_to_end_via_cli(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    prefixes = tmp_path / "prefixes.jsonl"
    latents = tmp_path / "latent-cache.jsonl"
    baselines = tmp_path / "baselines.jsonl"
    judge = tmp_path / "judge.jsonl"
    summary = tmp_path / "summary.json"
    report_dir = tmp_path / "report"
    demo_md = tmp_path / "demo.md"

    assert (
        main(
            [
                "collect",
                "--source",
                "synthetic",
                "--task",
                "all",
                "--episodes",
                "2",
                "--policy-family",
                "expert,weak,doomed,misleading",
                "--output",
                str(rollouts),
            ]
        )
        == 0
    )
    assert main(["prefixes", "--input", str(rollouts), "--output", str(prefixes)]) == 0
    assert (
        main(
            [
                "latents",
                "--rollouts",
                str(rollouts),
                "--prefixes",
                str(prefixes),
                "--output",
                str(latents),
            ]
        )
        == 0
    )
    assert main(["baselines", "--input", str(prefixes), "--output", str(baselines)]) == 0
    assert (
        main(
            [
                "judge",
                "--input",
                str(prefixes),
                "--latent-cache",
                str(latents),
                "--mode",
                "hybrid_surprise",
                "--output",
                str(judge),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "evaluate",
                "--prefixes",
                str(prefixes),
                "--baselines",
                str(baselines),
                "--judge",
                str(judge),
                "--calibration-families",
                "weak,doomed",
                "--evaluation-families",
                "expert,misleading",
                "--output",
                str(summary),
            ]
        )
        == 0
    )
    assert main(["report", "--summary", str(summary), "--output-dir", str(report_dir)]) == 0
    assert (
        main(
            [
                "demo",
                "--prefixes",
                str(prefixes),
                "--baselines",
                str(baselines),
                "--judge",
                str(judge),
                "--output",
                str(demo_md),
            ]
        )
        == 0
    )

    for path in (rollouts, prefixes, latents, baselines, judge, summary, demo_md):
        assert path.exists(), path
    assert (report_dir / "family-report.md").exists()
    assert (report_dir / "family-report.png").exists() or (
        report_dir / "family-report.svg"
    ).exists()
    assert (tmp_path / "demo-comparison.csv").exists()
    assert (tmp_path / "demo-push-v3-hard-disagreement-pack.csv").exists()
    assert (tmp_path / "demo-score-replay.csv").exists()
    assert (tmp_path / "demo-timeline.png").exists() or (tmp_path / "demo-timeline.svg").exists()

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert set(payload) == {"thresholds", "calibration", "overall", "tasks", "families"}
    assert payload["calibration"]["judge"]["mode"] == "held_out_family_split"

    demo_markdown = demo_md.read_text(encoding="utf-8")
    assert "Judge mode: `hybrid_prefix_latent_judge`" in demo_markdown


def test_python_dash_m_help_exits_zero():
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "lewm-judge" in result.stdout


def test_missing_input_file_exits_with_one_clean_error_line(tmp_path):
    result = _run_cli(
        [
            "prefixes",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(tmp_path / "prefixes.jsonl"),
        ]
    )
    assert result.returncode != 0
    line = _single_error_line(result.stderr)
    assert line.startswith("lewm-judge prefixes: error:")
    assert "missing.jsonl" in line


def test_malformed_jsonl_error_names_file_and_line(tmp_path):
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text('{"episode_id": "ok"}\nnot json\n', encoding="utf-8")
    result = _run_cli(
        [
            "prefixes",
            "--input",
            str(rollouts),
            "--output",
            str(tmp_path / "prefixes.jsonl"),
        ]
    )
    assert result.returncode != 0
    line = _single_error_line(result.stderr)
    assert line.startswith("lewm-judge prefixes: error:")
    assert "rollouts.jsonl" in line
    assert "line 2" in line


def test_invalid_task_exits_with_one_clean_error_line(tmp_path):
    result = _run_cli(
        [
            "collect",
            "--task",
            "bogus-task",
            "--output",
            str(tmp_path / "rollouts.jsonl"),
        ]
    )
    assert result.returncode != 0
    line = _single_error_line(result.stderr)
    assert line.startswith("lewm-judge collect: error:")
    assert "bogus-task" in line


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_every_subcommand_help_exits_zero(subcommand):
    with pytest.raises(SystemExit) as excinfo:
        main([subcommand, "--help"])
    assert excinfo.value.code == 0


def test_collect_is_deterministic_for_a_fixed_seed(tmp_path):
    outputs = []
    for name in ("first.jsonl", "second.jsonl"):
        output = tmp_path / name
        assert (
            main(
                [
                    "collect",
                    "--source",
                    "synthetic",
                    "--task",
                    "all",
                    "--episodes",
                    "2",
                    "--seed",
                    "123",
                    "--policy-family",
                    "expert,doomed",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]


def test_collect_rejects_unknown_policy_family(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "collect",
                "--task",
                "all",
                "--policy-family",
                "heroic",
                "--output",
                str(tmp_path / "rollouts.jsonl"),
            ]
        )
