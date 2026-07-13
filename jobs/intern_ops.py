# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.0,<2"]
# ///
"""Scoped ml-intern (github.com/huggingface/ml-intern) steps for RFC-011 runs.

Two supervised agent roles, each driven by a versioned prompt from ``jobs/prompts/``:

- ``operate-smoke`` — the agent runs the repo launcher for the ``smoke`` preset, then
  independently confirms the job state with read-only ``hf_jobs`` operations and writes an
  operations report.
- ``review`` — the agent writes an independent review of already-downloaded run folders
  against the checked-in reference artifact.

Headless ml-intern auto-approves everything and exits 0 even on errors (its exit code is not
a signal), so this wrapper (a) keeps prompts single-purpose with enumerated permissions and
(b) treats the expected output file's existence as the success criterion. Transcripts
(rendered prompt, stdout, session log) upload to ``runs/<run_id>/agent/`` in the dataset repo.

Usage:
    uv run jobs/intern_ops.py operate-smoke
    uv run jobs/intern_ops.py review --run-id <metaworld-run-id> --compare-run-id <synthetic-run-id>
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from string import Template

DEFAULT_DATASET_REPO = "abdelstark/leworldmodel-judge-runs"
REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "jobs" / "prompts"
REFERENCE_ARTIFACT = REPO_ROOT / "artifacts" / "hard-family-real-held-out-2026-04-28"
INTERN_MODEL = "huggingface/moonshotai/Kimi-K2-Instruct"
MAX_ITERATIONS = 40
INTERN_TIMEOUT_SECONDS = 2400
# ml-intern needs HF_TOKEN (model routing + read-only hf_jobs); everything else that smells
# like a credential stays out of its environment. HF_SESSION_UPLOAD_TOKEN in particular
# enables ml-intern's built-in trajectory upload to a third-party dataset repo
# (akseljoonas/hf-agent-sessions) — transcripts must only ship through this wrapper.
BLOCKED_CHILD_ENV = (
    "HF_SESSION_UPLOAD_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "INFERENCE_TOKEN",
)
# Token shapes that must never reach the public dataset repo, whatever their origin.
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


def render_prompt(name: str, mapping: dict[str, str]) -> str:
    template = Template((PROMPTS_DIR / name).read_text(encoding="utf-8"))
    return template.substitute(mapping)


def child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in BLOCKED_CHILD_ENV}


def run_intern(prompt: str, workdir: Path, model: str, expected_output: Path) -> bool:
    """Run one headless ml-intern turn; success == the expected output file exists after.

    Returns False instead of raising so callers can still collect and ship whatever
    transcript exists — a paid run without its transcript is the worse failure mode.
    """
    if shutil.which("ml-intern") is None:
        raise SystemExit(
            "intern_ops: `ml-intern` not on PATH; install github.com/huggingface/ml-intern"
        )
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "prompt.md").write_text(prompt, encoding="utf-8")
    print(f"[intern] model={model} workdir={workdir}")
    timed_out = False
    with (
        (workdir / "stdout.log").open("w", encoding="utf-8") as out,
        (workdir / "stderr.log").open("w", encoding="utf-8") as err,
    ):
        try:
            subprocess.run(
                [
                    "ml-intern",
                    "--model",
                    model,
                    "--max-iterations",
                    str(MAX_ITERATIONS),
                    "--no-stream",
                    prompt,
                ],
                cwd=workdir,
                env=child_env(),
                stdout=out,
                stderr=err,
                timeout=INTERN_TIMEOUT_SECONDS,
                check=False,  # ml-intern exits 0 even on errors; the file check is the gate
            )
        except subprocess.TimeoutExpired:
            timed_out = True
    if timed_out:
        print(f"[intern] timed out after {INTERN_TIMEOUT_SECONDS}s; keeping partial transcript")
    if not expected_output.is_file():
        tail = (workdir / "stdout.log").read_text(encoding="utf-8", errors="replace")[-2000:]
        print(f"[intern] expected output {expected_output} missing; stdout tail:\n{tail}")
        return False
    print(f"[intern] wrote {expected_output}")
    return True


def redact_secrets(agent_dir: Path) -> int:
    """Scrub token-shaped strings from every transcript file before any public upload."""
    redactions = 0
    for path in sorted(agent_dir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            text, count = pattern.subn("[REDACTED]", text)
            redactions += count
        path.write_text(text, encoding="utf-8")
    if redactions:
        print(f"[intern] redacted {redactions} token-shaped string(s) from the transcript")
    return redactions


def collect_transcript(workdir: Path) -> Path:
    """Gather prompt, stdout, and the ml-intern session log into an ``agent/`` folder."""
    agent_dir = workdir / "agent"
    agent_dir.mkdir(exist_ok=True)
    for name in ("prompt.md", "stdout.log"):
        source = workdir / name
        if source.is_file():
            shutil.copy2(source, agent_dir / name)
    session_logs = sorted((workdir / "session_logs").glob("*.json"))
    if session_logs:
        shutil.copy2(session_logs[-1], agent_dir / "session-log.json")
    redact_secrets(agent_dir)
    return agent_dir


def upload_agent_dir(agent_dir: Path, run_id: str, dataset_repo: str) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        folder_path=str(agent_dir),
        path_in_repo=f"runs/{run_id}/agent",
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message=f"Attach ml-intern transcript to {run_id}",
    )
    print(f"[upload] https://huggingface.co/datasets/{dataset_repo}/tree/main/runs/{run_id}/agent")


def cmd_operate_smoke(args: argparse.Namespace) -> int:
    stamp = f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    workdir = REPO_ROOT / "out" / "intern" / f"operate-smoke-{stamp}"
    report_path = workdir / "operator-report.md"
    prompt = render_prompt(
        "intern-operator.md",
        {"repo_root": str(REPO_ROOT), "report_path": str(report_path)},
    )
    succeeded = run_intern(prompt, workdir, args.model, expected_output=report_path)

    sources = []
    if report_path.is_file():
        sources.append(report_path.read_text(encoding="utf-8"))
    stdout_log = workdir / "stdout.log"
    if stdout_log.is_file():
        sources.append(stdout_log.read_text(encoding="utf-8", errors="replace"))
    match = next(
        (m for s in sources for m in [re.search(r"\b(smoke-\d{8}-\d{6}-g[0-9a-f]{7})\b", s)] if m),
        None,
    )

    agent_dir = collect_transcript(workdir)
    if report_path.is_file():
        shutil.copy2(report_path, agent_dir / "operator-report.md")
        redact_secrets(agent_dir)
    if match is None:
        raise SystemExit(
            f"intern_ops: no smoke run id found; transcript kept locally at {agent_dir}"
        )
    run_id = match.group(1)
    print(f"[intern] operated run {run_id}")
    if not args.skip_upload:
        upload_agent_dir(agent_dir, run_id, args.dataset_repo)
    if not succeeded:
        raise SystemExit("intern_ops: operator turn failed (see transcript); run recorded above")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            allow_patterns=[f"runs/{args.run_id}/*", f"runs/{args.compare_run_id}/*"],
        )
    )
    primary = root / "runs" / args.run_id
    compare = root / "runs" / args.compare_run_id
    for run_dir in (primary, compare):
        if not (run_dir / "provenance.json").is_file():
            raise SystemExit(f"intern_ops: {run_dir} is not a published run")

    stamp = f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    workdir = REPO_ROOT / "out" / "intern" / f"review-{stamp}"
    review_path = workdir / "intern-review.md"
    prompt = render_prompt(
        "intern-reviewer.md",
        {
            "primary_run_dir": str(primary),
            "compare_run_dir": str(compare),
            "reference_artifact_dir": str(REFERENCE_ARTIFACT),
            "review_path": str(review_path),
        },
    )
    succeeded = run_intern(prompt, workdir, args.model, expected_output=review_path)

    agent_dir = collect_transcript(workdir)
    if review_path.is_file():
        shutil.copy2(review_path, agent_dir / "intern-review.md")
        redact_secrets(agent_dir)
    if not args.skip_upload:
        upload_agent_dir(agent_dir, args.run_id, args.dataset_repo)
    if not succeeded:
        raise SystemExit("intern_ops: review turn failed; transcript shipped for the record")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    operate = sub.add_parser("operate-smoke", help="agent launches + confirms the smoke preset")
    operate.add_argument("--model", default=INTERN_MODEL)
    operate.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    operate.add_argument("--skip-upload", action="store_true")
    operate.set_defaults(func=cmd_operate_smoke)

    review = sub.add_parser("review", help="agent reviews published runs against the reference")
    review.add_argument("--run-id", required=True, help="primary (metaworld) run id")
    review.add_argument("--compare-run-id", required=True, help="synthetic run id")
    review.add_argument("--model", default=INTERN_MODEL)
    review.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    review.add_argument("--skip-upload", action="store_true")
    review.set_defaults(func=cmd_review)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
