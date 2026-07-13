# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.0,<2"]
# ///
"""Remote job payload for RFC-011 cloud benchmark runs.

Executed on Hugging Face Jobs via ``hf jobs uv run`` (or ``jobs/launch.py``). The
``leworldmodel-judge`` package itself is *not* a dependency of this script: the launcher
pins it to the pushed HEAD commit with ``--with "leworldmodel-judge[...] @ git+...@<sha>"``
so the run installs exactly the code under review.

Runs the full eight-stage pipeline (both judge modes, held-out family split), writes
``provenance.json``, and uploads the run folder to the runs dataset repository.

Run identity arrives exclusively through environment variables set by the launcher:
``LEWM_RUN_ID``, ``LEWM_PRESET``, ``LEWM_GIT_SHA``, ``LEWM_FLAVOR`` and optionally
``LEWM_DATASET_REPO`` / ``LEWM_SKIP_UPLOAD``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path

DEFAULT_DATASET_REPO = "abdelstark/leworldmodel-judge-runs"
GIT_REPO_URL = "https://github.com/AbdelStark/leworldmodel-judge"
PROVENANCE_SCHEMA_VERSION = "1.0"
CALIBRATION_FAMILIES = "weak,doomed"
EVALUATION_FAMILIES = "expert,misleading"
POLICY_FAMILIES = "expert,weak,doomed,misleading"
DEMO_FAMILIES = "expert,misleading"
JOB_ENV_PREFIXES = ("JOB_", "SPACE_")
SECRET_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD")


@dataclass(frozen=True)
class Preset:
    source: str
    episodes: int
    seed: int
    max_steps: int | None


PRESETS: dict[str, Preset] = {
    "smoke": Preset(source="synthetic", episodes=2, seed=7, max_steps=None),
    "synthetic-benchmark": Preset(source="synthetic", episodes=50, seed=7, max_steps=None),
    "metaworld-benchmark": Preset(source="metaworld", episodes=5, seed=1013, max_steps=75),
}


def stage_commands(preset: Preset, run_dir: Path) -> list[tuple[str, list[str]]]:
    """The exact pipeline stage argv sequence, mirroring the README quickstart plus the
    held-out convention of the 2026-04-28 artifact (both judges on the same prefixes)."""
    d = run_dir
    collect = [
        "collect",
        "--source",
        preset.source,
        "--task",
        "all",
        "--episodes",
        str(preset.episodes),
        "--policy-family",
        POLICY_FAMILIES,
        "--seed",
        str(preset.seed),
        "--output",
        str(d / "rollouts.jsonl"),
    ]
    if preset.max_steps is not None:
        collect += ["--max-steps", str(preset.max_steps)]
    held_out = [
        "--calibration-families",
        CALIBRATION_FAMILIES,
        "--evaluation-families",
        EVALUATION_FAMILIES,
    ]
    return [
        ("collect", collect),
        (
            "prefixes",
            [
                "prefixes",
                "--input",
                str(d / "rollouts.jsonl"),
                "--output",
                str(d / "prefixes.jsonl"),
            ],
        ),
        (
            "latents",
            [
                "latents",
                "--rollouts",
                str(d / "rollouts.jsonl"),
                "--prefixes",
                str(d / "prefixes.jsonl"),
                "--output",
                str(d / "latent-cache.jsonl"),
            ],
        ),
        (
            "baselines",
            [
                "baselines",
                "--input",
                str(d / "prefixes.jsonl"),
                "--output",
                str(d / "baselines.jsonl"),
            ],
        ),
        (
            "judge-composite",
            [
                "judge",
                "--input",
                str(d / "prefixes.jsonl"),
                "--mode",
                "heuristic_surprise",
                "--output",
                str(d / "judge-composite.jsonl"),
            ],
        ),
        (
            "judge-hybrid",
            [
                "judge",
                "--input",
                str(d / "prefixes.jsonl"),
                "--mode",
                "hybrid_surprise",
                "--latent-cache",
                str(d / "latent-cache.jsonl"),
                "--output",
                str(d / "judge-hybrid.jsonl"),
            ],
        ),
        (
            "evaluate-composite",
            [
                "evaluate",
                "--prefixes",
                str(d / "prefixes.jsonl"),
                "--baselines",
                str(d / "baselines.jsonl"),
                "--judge",
                str(d / "judge-composite.jsonl"),
                "--output",
                str(d / "summary-composite.json"),
                *held_out,
            ],
        ),
        (
            "evaluate-hybrid",
            [
                "evaluate",
                "--prefixes",
                str(d / "prefixes.jsonl"),
                "--baselines",
                str(d / "baselines.jsonl"),
                "--judge",
                str(d / "judge-hybrid.jsonl"),
                "--output",
                str(d / "summary.json"),
                *held_out,
            ],
        ),
        (
            "report",
            ["report", "--summary", str(d / "summary.json"), "--output-dir", str(d / "report")],
        ),
        (
            "demo",
            [
                "demo",
                "--prefixes",
                str(d / "prefixes.jsonl"),
                "--baselines",
                str(d / "baselines.jsonl"),
                "--judge",
                str(d / "judge-hybrid.jsonl"),
                "--families",
                DEMO_FAMILIES,
                "--output",
                str(d / "demo-artifact.md"),
            ],
        ),
    ]


def run_stages(preset: Preset, run_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for name, argv in stage_commands(preset, run_dir):
        cmd = [sys.executable, "-m", "leworldmodel_judge", *argv]
        print(f"[stage:{name}] {' '.join(argv)}", flush=True)
        started = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = round(time.monotonic() - started, 3)
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.returncode != 0:
            print(result.stderr, end="", file=sys.stderr, flush=True)
            raise RuntimeError(f"stage {name!r} failed with exit code {result.returncode}")
        records.append({"stage": name, "argv": argv, "wall_seconds": elapsed})
        print(f"[stage:{name}] done in {elapsed}s", flush=True)
    return records


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(run_dir: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "provenance.json":
            continue
        rel = path.relative_to(run_dir).as_posix()
        entry: dict[str, object] = {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                entry["rows"] = sum(1 for _ in handle)
        inventory[rel] = entry
    return inventory


def job_env_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in sorted(os.environ):
        if not key.startswith(JOB_ENV_PREFIXES):
            continue
        if any(marker in key.upper() for marker in SECRET_MARKERS):
            continue
        snapshot[key] = os.environ[key]
    return snapshot


def echo_headline(run_dir: Path) -> None:
    for summary_name in ("summary-composite.json", "summary.json"):
        summary = json.loads((run_dir / summary_name).read_text(encoding="utf-8"))
        judge = summary["calibration"]["judge"]
        prov = summary["calibration"]["provenance"]
        print(
            f"[result:{summary_name}] mode={judge['mode']} "
            f"threshold={judge['recommended_threshold']} "
            f"eval_stats={judge['evaluation_stats']} "
            f"eval_count={prov['evaluation_count']}",
            flush=True,
        )


def main() -> int:
    run_id = os.environ["LEWM_RUN_ID"]
    preset_name = os.environ["LEWM_PRESET"]
    git_sha = os.environ["LEWM_GIT_SHA"]
    flavor = os.environ.get("LEWM_FLAVOR", "unknown")
    dataset_repo = os.environ.get("LEWM_DATASET_REPO", DEFAULT_DATASET_REPO)
    skip_upload = os.environ.get("LEWM_SKIP_UPLOAD") == "1"
    preset = PRESETS[preset_name]

    # A fresh absolute directory, never a cwd-relative path: the container's cwd is /, where
    # a bare "run" would resolve to the OS /run directory and rglob would sweep foreign files
    # (or a stale local run/) into the public upload.
    run_dir = Path(tempfile.mkdtemp(prefix=f"lewm-{run_id}-"))
    created_utc = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[run] id={run_id} preset={preset_name} sha={git_sha[:12]} flavor={flavor}", flush=True)

    if not skip_upload:
        # Fail fast on missing write scope before paying for the pipeline stages.
        from huggingface_hub import HfApi

        HfApi().create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True)
        print(f"[preflight] write access to {dataset_repo} confirmed", flush=True)

    stage_records = run_stages(preset, run_dir)
    echo_headline(run_dir)

    provenance = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "run_id": run_id,
        "preset": preset_name,
        "created_utc": created_utc,
        "git": {"repo": GIT_REPO_URL, "sha": git_sha},
        "package": {"name": "leworldmodel-judge", "version": package_version("leworldmodel-judge")},
        "python": sys.version,
        "platform": platform.platform(),
        "hardware_flavor": flavor,
        "job_env": job_env_snapshot(),
        "stages": stage_records,
        "files": file_inventory(run_dir),
        "upload": {"repo_id": dataset_repo, "path_in_repo": f"runs/{run_id}"},
    }
    provenance_path = run_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"[provenance] {len(provenance['files'])} files inventoried", flush=True)

    if skip_upload:
        print("[upload] skipped (LEWM_SKIP_UPLOAD=1)", flush=True)
        print(f"[upload] run folder left at {run_dir}", flush=True)
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=str(run_dir),
        path_in_repo=f"runs/{run_id}",
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message=f"Publish run {run_id} (preset={preset_name}, sha={git_sha[:12]})",
    )
    print(
        f"[upload] https://huggingface.co/datasets/{dataset_repo}/tree/main/runs/{run_id}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
