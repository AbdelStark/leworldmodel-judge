"""``lewm-judge``: one console entry point for the whole benchmark pipeline.

Eight subcommands, one per pipeline stage, each a thin argparse wrapper over
the library:

    collect → prefixes → latents → baselines → judge → evaluate → report → demo

All paths come from flags; nothing is hardcoded. Comma lists (``--task``,
``--policy-family``, ``--fractions``, ``--*-families``, ``--families``) are
plain CSV strings. ``python -m leworldmodel_judge`` is equivalent to
``lewm-judge``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .baselines import score_prefixes
from .collect import POLICY_FAMILIES, collect_metaworld, collect_synthetic, derive_task_seed
from .io import read_jsonl, write_json, write_jsonl
from .judge import JUDGE_MODES, run_judge
from .latents import build_latent_cache
from .metrics import summarize
from .prefixes import build_prefixes
from .tasks import resolve_tasks


def csv_list(value: str) -> list[str]:
    """Argparse type for comma-separated lists: split, strip, drop empties."""
    return [part.strip() for part in value.split(",") if part.strip()]


def csv_floats(value: str) -> tuple[float, ...]:
    """Argparse type for comma-separated float lists (e.g. ``--fractions``)."""
    return tuple(float(part) for part in csv_list(value))


def _cmd_collect(args: argparse.Namespace) -> int:
    tasks = resolve_tasks(args.task)
    policy_families: list[str] = args.policy_family
    rows: list[dict[str, Any]] = []
    for family_index, policy_family in enumerate(policy_families):
        if policy_family not in POLICY_FAMILIES:
            raise ValueError(f"policy-family must be one of {POLICY_FAMILIES}; got {policy_family}")
        for offset, task in enumerate(tasks):
            if args.source == "metaworld":
                task_seed = derive_task_seed(args.seed, family_index, offset)
                rows.extend(
                    collect_metaworld(
                        task, args.episodes, args.max_steps, task_seed, policy_family=policy_family
                    )
                )
            else:
                rows.extend(
                    collect_synthetic(
                        task, args.episodes, policy_family=policy_family, seed=args.seed
                    )
                )
    write_jsonl(args.output, rows)
    print(
        f"wrote {len(rows)} rollout steps for tasks={tasks} policy_families={policy_families} to {Path(args.output)}"
    )
    return 0


def _cmd_prefixes(args: argparse.Namespace) -> int:
    steps = read_jsonl(args.input)
    prefixes = build_prefixes(steps, args.fractions)
    write_jsonl(args.output, [p.to_dict() for p in prefixes])
    print(f"wrote {len(prefixes)} prefixes to {args.output}")
    return 0


def _cmd_latents(args: argparse.Namespace) -> int:
    rollouts = read_jsonl(args.rollouts)
    prefixes = read_jsonl(args.prefixes)
    cache_rows = build_latent_cache(prefixes, rollouts)
    write_jsonl(args.output, cache_rows)
    print(f"wrote {len(cache_rows)} latent cache rows to {args.output}")
    return 0


def _cmd_baselines(args: argparse.Namespace) -> int:
    prefixes = read_jsonl(args.input)
    rows = score_prefixes(prefixes)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} baseline rows to {args.output}")
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    prefixes = read_jsonl(args.input)
    latent_rows = read_jsonl(args.latent_cache) if args.latent_cache else None
    rows = run_judge(prefixes, mode=args.mode, latent_rows=latent_rows)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} judge rows to {args.output}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    payload = summarize(
        read_jsonl(args.prefixes),
        read_jsonl(args.baselines),
        read_jsonl(args.judge),
        calibration_families=args.calibration_families or None,
        evaluation_families=args.evaluation_families or None,
    )
    write_json(args.output, payload)
    print(f"wrote summary to {args.output}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .report import render_family_report

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    outputs = render_family_report(summary, args.output_dir)
    print(f"wrote report markdown to {outputs['markdown']}")
    print(f"wrote report plot to {outputs['plot']}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import render_demo

    outputs = render_demo(
        read_jsonl(args.prefixes),
        read_jsonl(args.baselines),
        read_jsonl(args.judge),
        args.output,
        families=args.families,
    )
    print(f"wrote demo artifact to {outputs['markdown']}")
    print(f"wrote comparison table to {outputs['comparison_csv']}")
    print(f"wrote push-v3 disagreement pack to {outputs['disagreement_pack_csv']}")
    print(f"wrote episode replay table to {outputs['replay_csv']}")
    print(f"wrote timeline plot to {outputs['plot']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``lewm-judge`` argument parser with all eight subcommands."""
    parser = argparse.ArgumentParser(
        prog="lewm-judge",
        description=(
            "A world model as a judge: collect rollouts, slice prefixes, score them, "
            "and evaluate the judge against sparse-reward baselines."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect",
        help="collect rollout steps (synthetic generator or Meta-World capture)",
        description=(
            "Collect rollout step JSONL. Synthetic mode is stdlib-only and fully "
            "deterministic per seed; metaworld mode needs the benchmark dependency "
            "group and derives one seed per (family, task) pair."
        ),
    )
    collect.add_argument(
        "--source",
        choices=["synthetic", "metaworld"],
        default="synthetic",
        help="rollout source (default: synthetic)",
    )
    collect.add_argument(
        "--task", required=True, help="single task, comma-separated subset, or all"
    )
    collect.add_argument("--episodes", type=int, default=5, help="episodes per (task, family)")
    collect.add_argument("--output", required=True, help="rollouts JSONL output path")
    collect.add_argument(
        "--max-steps", type=int, default=None, help="episode step cap (metaworld only)"
    )
    collect.add_argument("--seed", type=int, default=7, help="base RNG seed (default: 7)")
    collect.add_argument(
        "--policy-family",
        type=csv_list,
        default=["random"],
        help="random, expert, weak, doomed, misleading, or comma-separated list",
    )
    collect.set_defaults(handler=_cmd_collect)

    prefixes = subparsers.add_parser(
        "prefixes",
        help="slice rollouts into labeled prefix records at fractional cutoffs",
        description="Slice rollout steps into labeled prefix records at fractional cutoffs.",
    )
    prefixes.add_argument("--input", required=True, help="rollouts JSONL path")
    prefixes.add_argument("--output", required=True, help="prefixes JSONL output path")
    prefixes.add_argument(
        "--fractions",
        type=csv_floats,
        default=(0.25, 0.5, 0.75),
        help="comma-separated prefix cutoffs (default: 0.25,0.50,0.75)",
    )
    prefixes.set_defaults(handler=_cmd_prefixes)

    latents = subparsers.add_parser(
        "latents",
        help="build the observation-space latent prediction cache",
        description=(
            "Build latent-cache rows (mean-pooled observation windows plus a linear "
            "extrapolation predictor) for the hybrid judge."
        ),
    )
    latents.add_argument("--rollouts", required=True, help="rollouts JSONL path")
    latents.add_argument("--prefixes", required=True, help="prefixes JSONL path")
    latents.add_argument("--output", required=True, help="latent-cache JSONL output path")
    latents.set_defaults(handler=_cmd_latents)

    baselines = subparsers.add_parser(
        "baselines",
        help="score every prefix with the baseline scorers",
        description=(
            "Score every prefix with the three baseline signals "
            "(sparse reward, terminal success, progress proxy)."
        ),
    )
    baselines.add_argument("--input", required=True, help="prefixes JSONL path")
    baselines.add_argument("--output", required=True, help="baselines JSONL output path")
    baselines.set_defaults(handler=_cmd_baselines)

    judge = subparsers.add_parser(
        "judge",
        help="score every prefix with the judge in one of three modes",
        description=(
            "Score prefixes with the judge. hybrid_surprise joins --latent-cache rows "
            "on (task_id, episode_id, prefix_fraction); without a cache it degrades to "
            "the composite judge and logs a warning. dummy is the null judge."
        ),
    )
    judge.add_argument("--input", required=True, help="prefixes JSONL path")
    judge.add_argument("--output", required=True, help="judge JSONL output path")
    judge.add_argument("--latent-cache", help="latent-cache JSONL path (hybrid_surprise mode)")
    judge.add_argument(
        "--mode",
        choices=list(JUDGE_MODES),
        default="heuristic_surprise",
        help="judge mode (default: heuristic_surprise)",
    )
    judge.set_defaults(handler=_cmd_judge)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="join prefixes/baselines/judge rows into the summary JSON",
        description=(
            "Join prefix, baseline and judge rows into the benchmark summary, with "
            "optional held-out family-split calibration."
        ),
    )
    evaluate.add_argument("--prefixes", required=True, help="prefixes JSONL path")
    evaluate.add_argument("--baselines", required=True, help="baselines JSONL path")
    evaluate.add_argument("--judge", required=True, help="judge JSONL path")
    evaluate.add_argument("--output", required=True, help="summary JSON output path")
    evaluate.add_argument(
        "--calibration-families",
        type=csv_list,
        default=None,
        help=(
            "Comma-separated policy families to use for threshold calibration. "
            "Defaults to the same evaluation slice."
        ),
    )
    evaluate.add_argument(
        "--evaluation-families",
        type=csv_list,
        default=None,
        help=(
            "Comma-separated policy families to include in the reported evaluation "
            "slice. Defaults to all families."
        ),
    )
    evaluate.set_defaults(handler=_cmd_evaluate)

    report = subparsers.add_parser(
        "report",
        help="render the family report bundle from a summary JSON",
        description=(
            "Render family-report.md plus a per-family plot (PNG with matplotlib, "
            "SVG without) from an evaluate summary."
        ),
    )
    report.add_argument("--summary", required=True, help="summary JSON path")
    report.add_argument("--output-dir", required=True, help="report output directory")
    report.set_defaults(handler=_cmd_report)

    demo = subparsers.add_parser(
        "demo",
        help="render the demo artifact bundle (markdown + CSVs + timeline plot)",
        description=(
            "Render the demo artifact: markdown report plus sibling files derived from "
            "the --output stem (<stem>-comparison.csv, <stem>-timeline.png|.svg, "
            "<stem>-push-v3-hard-disagreement-pack.csv, <stem>-score-replay.csv)."
        ),
    )
    demo.add_argument("--prefixes", required=True, help="prefixes JSONL path")
    demo.add_argument("--baselines", required=True, help="baselines JSONL path")
    demo.add_argument("--judge", required=True, help="judge JSONL path")
    demo.add_argument(
        "--output",
        required=True,
        help="markdown artifact path; sibling CSV/plot files are emitted automatically",
    )
    demo.add_argument(
        "--families",
        type=csv_list,
        default=None,
        help="optional comma-separated policy families to include in the artifact",
    )
    demo.set_defaults(handler=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code.

    Expected handler failures — ``ValueError`` (which covers
    ``json.JSONDecodeError``, re-raised with file/line context by
    :func:`leworldmodel_judge.io.read_jsonl`) and ``OSError`` (missing or
    unreadable input files) — exit via ``SystemExit`` with a one-line
    ``lewm-judge <cmd>: error: ...`` message on stderr instead of a traceback.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"lewm-judge {args.command}: error: {exc}") from exc
