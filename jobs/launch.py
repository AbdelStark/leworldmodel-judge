# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.0,<2"]
# ///
"""Local launcher for RFC-011 cloud benchmark runs.

Subcommands:

- ``launch``  — preflight (clean tree, HEAD pushed), start a Hugging Face Job running
  ``jobs/run_benchmark.py`` with the package pinned to the pushed HEAD commit, stream it to a
  terminal verdict, then verify the published run.
- ``verify``  — download a published run folder and gate it against the RFC-011 §3 contract.
  The exit code is the verdict.
- ``card``    — regenerate the runs dataset repository's README card from every published run.

Usage: ``uv run jobs/launch.py launch --preset smoke``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DATASET_REPO = "abdelstark/leworldmodel-judge-runs"
GIT_REPO_URL = "https://github.com/AbdelStark/leworldmodel-judge"
REPO_ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = REPO_ROOT / "jobs" / "run_benchmark.py"
# CPython 3.11 is the byte-exact reproduction interpreter (docs/contracts.md).
JOB_PYTHON = "3.11"
POLL_SECONDS = 15
# Transient poll failures tolerated before giving up on monitoring (the job itself is
# unaffected; the launcher prints how to verify manually).
MAX_POLL_FAILURES = 8
# Client-side deadline margin over the job's server-side runtime timeout, to absorb queue time.
QUEUE_MARGIN_SECONDS = 30 * 60
TERMINAL_STAGES = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}

REQUIRED_FILES = (
    "rollouts.jsonl",
    "prefixes.jsonl",
    "latent-cache.jsonl",
    "baselines.jsonl",
    "judge-composite.jsonl",
    "judge-hybrid.jsonl",
    "summary-composite.json",
    "summary.json",
    "report/family-report.md",
    "report/family-report.png",
    "demo-artifact.md",
    "demo-artifact-comparison.csv",
    "demo-artifact-timeline.png",
    "demo-artifact-push-v3-hard-disagreement-pack.csv",
    "demo-artifact-score-replay.csv",
    "provenance.json",
)
EXPECTED_JUDGE_MODES = {
    "judge-composite.jsonl": "composite_prefix_judge",
    "judge-hybrid.jsonl": "hybrid_prefix_latent_judge",
}


@dataclass(frozen=True)
class LaunchSpec:
    flavor: str
    timeout: str
    extras: str


LAUNCH_SPECS: dict[str, LaunchSpec] = {
    "smoke": LaunchSpec(flavor="cpu-basic", timeout="15m", extras="viz"),
    "synthetic-benchmark": LaunchSpec(flavor="cpu-basic", timeout="20m", extras="viz"),
    "metaworld-benchmark": LaunchSpec(flavor="cpu-upgrade", timeout="45m", extras="viz,metaworld"),
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def preflight() -> str:
    """The --with pin must name a commit anyone can install: refuse dirty or unpushed HEADs."""
    dirty = git("status", "--porcelain")
    if dirty:
        raise SystemExit(f"preflight: working tree is dirty; commit or stash first:\n{dirty}")
    sha = git("rev-parse", "HEAD")
    subprocess.run(["git", "fetch", "--quiet", "origin"], cwd=REPO_ROOT, check=True)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=REPO_ROOT
    )
    if ancestor.returncode != 0:
        raise SystemExit(f"preflight: HEAD {sha[:12]} is not on origin/main; push first")
    return sha


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timeout_to_seconds(timeout: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if timeout[-1] in units:
        return int(float(timeout[:-1]) * units[timeout[-1]])
    return int(float(timeout))


def print_log_tail(api: Any, job_id: str, namespace: str) -> None:
    """Best-effort: a log-fetch hiccup must never fail a completed run."""
    try:
        print(f"[job:{job_id}] log tail:")
        for line in api.fetch_job_logs(job_id=job_id, namespace=namespace, tail=30):
            print(f"  {line}")
    except Exception as exc:
        print(f"[job:{job_id}] log fetch failed (non-fatal): {exc}")


def cmd_launch(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi, get_token

    spec = LAUNCH_SPECS[args.preset]
    sha = preflight()
    run_id = f"{args.preset}-{datetime.now(UTC):%Y%m%d-%H%M%S}-g{sha[:7]}"
    pin = f"leworldmodel-judge[{spec.extras}] @ git+{GIT_REPO_URL}@{sha}"
    env = {
        "LEWM_RUN_ID": run_id,
        "LEWM_PRESET": args.preset,
        "LEWM_GIT_SHA": sha,
        "LEWM_FLAVOR": spec.flavor,
        "LEWM_DATASET_REPO": args.dataset_repo,
    }
    print(f"[launch] run_id={run_id}")
    print(f"[launch] pin={pin}")
    print(f"[launch] flavor={spec.flavor} timeout={spec.timeout} python={JOB_PYTHON}")
    if args.dry_run:
        print("[launch] dry run: no job started")
        return 0

    token = get_token()
    if not token:
        raise SystemExit("launch: no Hugging Face token found; run `hf auth login`")
    api = HfApi()
    # Probe write scope before paying for a run that could not publish (also creates the
    # dataset repo on first use). Resolve the namespace once so polls make one call each.
    api.create_repo(repo_id=args.dataset_repo, repo_type="dataset", exist_ok=True)
    namespace = args.namespace or api.whoami()["name"]
    job = api.run_uv_job(
        script=str(PAYLOAD),
        dependencies=[pin],
        python=JOB_PYTHON,
        env=env,
        secrets={"HF_TOKEN": token},
        flavor=spec.flavor,
        timeout=spec.timeout,
        labels={"project": "leworldmodel-judge", "preset": args.preset, "run_id": run_id},
        namespace=namespace,
    )
    print(f"[launch] job_id={job.id}")
    print(f"[launch] url={job.url}")

    deadline = time.monotonic() + timeout_to_seconds(spec.timeout) + QUEUE_MARGIN_SECONDS
    stage = ""
    poll_failures = 0
    while stage not in TERMINAL_STAGES:
        if time.monotonic() > deadline:
            print(f"[launch] client deadline exceeded (stage={stage or 'unknown'}); cancelling")
            try:
                api.cancel_job(job_id=job.id, namespace=namespace)
            except Exception as exc:
                print(f"[launch] cancel failed: {exc}")
            return 2
        time.sleep(POLL_SECONDS)
        try:
            info = api.inspect_job(job_id=job.id, namespace=namespace)
            poll_failures = 0
        except Exception as exc:
            poll_failures += 1
            print(f"[launch] poll failed ({poll_failures}/{MAX_POLL_FAILURES}): {exc}")
            if poll_failures >= MAX_POLL_FAILURES:
                print(
                    f"[launch] giving up on monitoring; the job may still be running.\n"
                    f"  inspect: hf jobs inspect {job.id}\n"
                    f"  verify:  uv run jobs/launch.py verify --run-id {run_id}"
                )
                return 3
            continue
        if info.status.stage != stage:
            stage = info.status.stage
            message = info.status.message or ""
            print(f"[job:{job.id}] stage={stage} {message}".rstrip())

    print_log_tail(api, job.id, namespace)
    if stage != "COMPLETED":
        print(f"[launch] job ended in {stage}; not verifying")
        return 1
    print(f"[launch] job completed; verifying run {run_id}")
    return verify_run(run_id=run_id, dataset_repo=args.dataset_repo, local_dir=None)


def verify_run(run_id: str, dataset_repo: str, local_dir: str | None) -> int:
    from huggingface_hub import snapshot_download

    if local_dir is None:
        root = snapshot_download(
            repo_id=dataset_repo, repo_type="dataset", allow_patterns=[f"runs/{run_id}/*"]
        )
        run_dir = Path(root) / "runs" / run_id
    else:
        run_dir = Path(local_dir)

    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        print(f"  [{'ok' if condition else 'FAIL'}] {label}")
        if not condition:
            failures.append(label)

    print(f"[verify] {run_dir}")
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    check(
        not missing, f"required files present{': missing ' + ', '.join(missing) if missing else ''}"
    )
    if missing:
        print(f"[verify] FAILED ({len(failures)} checks)")
        return 1

    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    check(provenance.get("run_id") == run_id, "provenance run_id matches")
    check(provenance.get("preset") in LAUNCH_SPECS, "provenance preset is known")
    check(bool(provenance.get("git", {}).get("sha")), "provenance pins a git sha")

    inventory = provenance.get("files", {})
    covered = set(inventory)
    check(
        {name for name in REQUIRED_FILES if name != "provenance.json"} <= covered,
        "provenance inventory covers required files",
    )
    mismatched = [
        rel
        for rel, entry in inventory.items()
        if not (run_dir / rel).is_file()
        or sha256_of(run_dir / rel) != entry["sha256"]
        or (run_dir / rel).stat().st_size != entry["bytes"]
    ]
    check(
        not mismatched,
        f"checksums match provenance{': ' + ', '.join(mismatched) if mismatched else ''}",
    )
    # Anything in the run folder that provenance does not vouch for is contamination,
    # except the post-run agent/ transcripts (RFC-011 §4).
    unexpected = [
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in inventory
        and path.relative_to(run_dir).as_posix() != "provenance.json"
        and path.relative_to(run_dir).parts[0] != "agent"
    ]
    check(
        not unexpected,
        f"no files outside provenance inventory{': ' + ', '.join(unexpected) if unexpected else ''}",
    )

    def rows(name: str) -> int:
        with (run_dir / name).open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    n_prefixes = rows("prefixes.jsonl")
    check(n_prefixes > 0, f"prefixes non-empty ({n_prefixes} rows)")
    check(rows("baselines.jsonl") == n_prefixes, "baselines rows == prefixes rows")
    check(rows("judge-composite.jsonl") == n_prefixes, "judge-composite rows == prefixes rows")
    check(rows("judge-hybrid.jsonl") == n_prefixes, "judge-hybrid rows == prefixes rows")
    n_latents = rows("latent-cache.jsonl")
    check(0 < n_latents <= n_prefixes, f"latent cache rows in (0, prefixes] ({n_latents})")

    for judge_file, expected_mode in EXPECTED_JUDGE_MODES.items():
        with (run_dir / judge_file).open("r", encoding="utf-8") as handle:
            modes = {json.loads(line)["judge_mode"] for line in handle}
        check(modes == {expected_mode}, f"{judge_file} judge_mode == {expected_mode}")

    for summary_name in ("summary-composite.json", "summary.json"):
        summary = json.loads((run_dir / summary_name).read_text(encoding="utf-8"))
        calibration = summary.get("calibration", {})
        judge = calibration.get("judge", {})
        prov = calibration.get("provenance", {})
        check(
            set(summary) >= {"thresholds", "calibration", "overall", "tasks", "families"},
            f"{summary_name} has the contract sections",
        )
        check(
            judge.get("mode") == "held_out_family_split",
            f"{summary_name} calibration mode is held_out_family_split",
        )
        check(prov.get("family_overlap") is False, f"{summary_name} families are disjoint")
        check(
            prov.get("calibration_count", 0) > 0 and prov.get("evaluation_count", 0) > 0,
            f"{summary_name} cohorts are non-empty",
        )

    if failures:
        print(f"[verify] FAILED ({len(failures)} of the checks above)")
        return 1
    print("[verify] PASSED")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    return verify_run(run_id=args.run_id, dataset_repo=args.dataset_repo, local_dir=args.local_dir)


def load_runs(dataset_repo: str) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo_id=dataset_repo, repo_type="dataset")
    runs = []
    for path in sorted(files):
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "provenance.json":
            provenance = json.loads(
                Path(
                    hf_hub_download(repo_id=dataset_repo, repo_type="dataset", filename=path)
                ).read_text(encoding="utf-8")
            )
            summary_path = hf_hub_download(
                repo_id=dataset_repo,
                repo_type="dataset",
                filename=f"runs/{parts[1]}/summary-composite.json",
            )
            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            runs.append({"provenance": provenance, "summary_composite": summary})
    return runs


def format_metric(value: object) -> str:
    return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def cmd_card(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    runs = load_runs(args.dataset_repo)
    rows = []
    for run in sorted(runs, key=lambda r: str(r["provenance"]["created_utc"])):
        prov = run["provenance"]
        cal = run["summary_composite"]["calibration"]
        stats = cal["judge"]["evaluation_stats"]
        rows.append(
            f"| [{prov['run_id']}](runs/{prov['run_id']}) | {prov['preset']} "
            f"| {prov['created_utc']} | `{prov['git']['sha'][:12]}` | {prov['package']['version']} "
            f"| {prov['hardware_flavor']} | {cal['provenance']['evaluation_count']} "
            f"| {format_metric(stats['hit_rate'])} | {format_metric(stats['false_positive_rate'])} |"
        )
    table = "\n".join(rows) if rows else "| _no runs published yet_ | | | | | | | | |"
    card = f"""---
license: mit
pretty_name: LeWorldModel Judge — cloud benchmark runs
tags:
  - robotics
  - world-model
  - trajectory-evaluation
  - metaworld
---

# LeWorldModel Judge — cloud benchmark runs

Complete, self-describing benchmark runs of
[leworldmodel-judge]({GIT_REPO_URL}) executed on Hugging Face Jobs
per [RFC-011]({GIT_REPO_URL}/blob/main/docs/rfcs/RFC-011-hf-jobs-pipeline.md).
Each `runs/<run_id>/` folder carries the full RFC-008 file surface — the rollout capture of
record, every derived table, both judge outputs, rendered reports — plus `provenance.json`
(pinned git sha, installed package version, per-stage commands and timings, sha256 of every
file). Runs are verified against the contract by `jobs/launch.py verify` before being cited.

| run | preset | created (UTC) | git sha | pkg | flavor | eval n | judge hit rate | judge FPR |
|---|---|---|---|---|---|---|---|---|
{table}

Metrics above are the **composite cutoff-time judge** (`judge_mode: composite_prefix_judge`,
`summary-composite.json`) on the held-out evaluation cohort (calibrated on `weak,doomed`,
evaluated on `expert,misleading`). Each run folder also ships the hybrid latent variant
(`summary.json`, `judge_mode: hybrid_prefix_latent_judge`) — a **replay-time** signal that reads
post-cutoff observations and is not a cutoff-time judge (RFC-007 caveat). The shipped judge is a
heuristic composite plus an observation-space latent proxy, not a trained JEPA world model, and
the benchmark slice is narrow: three Meta-World tasks, scripted policy families, small cohorts.
See the repository's [benchmark doc]({GIT_REPO_URL}/blob/main/docs/benchmark.md) for the caveats
that accompany every number.

_This card is generated by `jobs/launch.py card`; edits land in the generator, not here._
"""
    if args.dry_run:
        print(card)
        return 0
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.dataset_repo,
        repo_type="dataset",
        commit_message=f"Regenerate card from {len(runs)} published run(s)",
    )
    print(f"[card] updated https://huggingface.co/datasets/{args.dataset_repo} ({len(runs)} runs)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="preflight, start a job, stream it, verify the run")
    launch.add_argument("--preset", choices=sorted(LAUNCH_SPECS), required=True)
    launch.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    launch.add_argument("--namespace", default=None)
    launch.add_argument("--dry-run", action="store_true")
    launch.set_defaults(func=cmd_launch)

    verify = sub.add_parser("verify", help="gate a published run against the RFC-011 contract")
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    verify.add_argument("--local-dir", default=None, help="verify an already-downloaded folder")
    verify.set_defaults(func=cmd_verify)

    card = sub.add_parser("card", help="regenerate the runs dataset README from published runs")
    card.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    card.add_argument("--dry-run", action="store_true")
    card.set_defaults(func=cmd_card)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
