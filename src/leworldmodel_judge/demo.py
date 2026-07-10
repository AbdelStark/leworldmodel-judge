"""Demo artifact bundle: the judge's verdict surface made auditable.

Joins prefix, baseline and judge rows into flat comparison rows, then renders
one markdown artifact plus sibling files derived from the ``--output`` stem.
The sibling filenames and the markdown section headings are contract
(docs/contracts.md); tests pin them:

- ``<stem>-comparison.csv`` — one row per prefix with every evidence field.
- ``<stem>-timeline.png|.svg`` — mean score movement across cutoffs.
- ``<stem>-push-v3-hard-disagreement-pack.csv`` — family-diverse push-v3
  disagreements (the anti-cherry-picking exhibit).
- ``<stem>-score-replay.csv`` — one row per episode, per-cutoff signals plus
  first-difference deltas, so reviewers can audit drift rather than means.

``baseline_metric`` is defined here as ``1 - progress_proxy_score``: the
progress baseline expressed as a failure score, so it is directly comparable
with the judge's ``failure_score``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import plotting
from .metrics import family_name
from .schema import prefix_key

EVIDENCE_KEYS = (
    "progress_proxy",
    "distance_progress",
    "target_distance_last",
    "target_distance_best",
    "in_place_score",
    "grasp_signal_peak",
    "success_signal_peak",
    "reward_density",
    "stall_score",
)

# The locked disagreement-pack slice: push-v3 across every policy family.
HARD_PACK_TASK_ID = "push-v3"
HARD_POLICY_FAMILIES = ("expert", "weak", "doomed", "misleading", "random")

# Empty-input fallback header for the comparison CSV. Populated rows append
# two latent columns (latent_mismatch_score, latent_alignment_score) not
# listed here — kept as-is for artifact byte-stability.
COMPARISON_HEADERS = (
    "task_id",
    "episode_id",
    "policy_family",
    "prefix_cutoff",
    "prefix_index",
    "baseline_metric",
    "judge_metric",
    "sparse_reward_signal",
    "judge_signal",
    "success_label",
    "prefix_failure_label",
    "prefix_recoverability_label",
    "judge_on_track_score",
    "judge_implausibility_score",
    "judge_uncertainty_score",
    "baseline_vs_judge_gap",
    *EVIDENCE_KEYS,
)

# Canonical static replay columns; per-cutoff columns are appended dynamically.
REPLAY_STATIC_HEADERS = (
    "task_id",
    "episode_id",
    "policy_family",
    "success_label",
    "final_prefix_failure_label",
    "final_prefix_recoverability_label",
    "max_abs_gap",
    "max_judge_metric",
    "max_baseline_metric",
)


def _normalise_family_filter(families: Iterable[str] | str | None) -> set[str] | None:
    if families is None:
        return None
    if isinstance(families, str):
        parts = {part.strip() for part in families.split(",") if part.strip()}
    else:
        parts = {str(part).strip() for part in families if str(part).strip()}
    return parts or None


def _filter_inputs_by_family(
    prefixes: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    judge: list[dict[str, Any]],
    families: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not families:
        return prefixes, baselines, judge
    filtered_prefixes = [prefix for prefix in prefixes if family_name(prefix) in families]
    keep_keys = {prefix_key(prefix) for prefix in filtered_prefixes}
    filtered_baselines = [row for row in baselines if prefix_key(row) in keep_keys]
    filtered_judge = [row for row in judge if prefix_key(row) in keep_keys]
    return filtered_prefixes, filtered_baselines, filtered_judge


def _comparison_rows(
    prefixes: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    judge: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join the three row streams on :func:`~leworldmodel_judge.schema.prefix_key`."""
    baseline_map = {prefix_key(row): row for row in baselines}
    judge_map = {prefix_key(row): row for row in judge}
    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        key = prefix_key(prefix)
        baseline = baseline_map.get(key, {})
        judge_row = judge_map.get(key, {})
        progress_score = float(baseline.get("progress_proxy_score", 0.0))
        progress_failure_score = 1.0 - progress_score
        sparse_reward_signal = float(baseline.get("sparse_reward_score", 0.0))
        judge_signal = float(judge_row.get("failure_score", 0.0))
        row = {
            "task_id": prefix["task_id"],
            "episode_id": prefix["episode_id"],
            "policy_family": family_name(prefix),
            "prefix_cutoff": float(prefix["prefix_fraction"]),
            "prefix_index": int(prefix["prefix_index"]),
            "baseline_metric": round(progress_failure_score, 6),
            "judge_metric": round(judge_signal, 6),
            "sparse_reward_signal": round(sparse_reward_signal, 6),
            "judge_signal": round(judge_signal, 6),
            "success_label": bool(prefix.get("final_success_label", False)),
            "prefix_failure_label": bool(prefix.get("prefix_failure_label", False)),
            "prefix_recoverability_label": prefix.get("prefix_recoverability_label", "unknown"),
            "judge_on_track_score": round(float(judge_row.get("on_track_score", 0.0)), 6),
            "judge_implausibility_score": round(
                float(judge_row.get("implausibility_score", 0.0)), 6
            ),
            "judge_uncertainty_score": round(float(judge_row.get("uncertainty_score", 0.0)), 6),
            "baseline_vs_judge_gap": round(judge_signal - progress_failure_score, 6),
            "latent_mismatch_score": round(float(judge_row.get("latent_mismatch_score", 0.0)), 6)
            if "latent_mismatch_score" in judge_row
            else "",
            "latent_alignment_score": round(float(judge_row.get("latent_alignment_score", 0.0)), 6)
            if "latent_alignment_score" in judge_row
            else "",
        }
        for evidence_key in EVIDENCE_KEYS:
            value = prefix.get(evidence_key)
            row[evidence_key] = "" if value is None else round(float(value), 6)
        rows.append(row)
    return rows


def _cutoff_token(cutoff: float) -> str:
    return str(cutoff).replace(".", "p")


def _episode_replay_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task_id"]), str(row["episode_id"]))].append(row)

    replay_rows: list[dict[str, Any]] = []
    for (task_id, episode_id), episode_rows in sorted(grouped.items()):
        ordered = sorted(episode_rows, key=lambda row: float(row["prefix_cutoff"]))
        head = ordered[0]
        replay_row: dict[str, Any] = {
            "task_id": task_id,
            "episode_id": episode_id,
            "policy_family": head["policy_family"],
            "success_label": bool(head["success_label"]),
            "final_prefix_failure_label": bool(ordered[-1]["prefix_failure_label"]),
            "final_prefix_recoverability_label": ordered[-1]["prefix_recoverability_label"],
            "max_abs_gap": round(
                max(abs(float(row["baseline_vs_judge_gap"])) for row in ordered),
                6,
            ),
            "max_judge_metric": round(max(float(row["judge_metric"]) for row in ordered), 6),
            "max_baseline_metric": round(max(float(row["baseline_metric"]) for row in ordered), 6),
        }
        previous_judge: float | None = None
        previous_baseline: float | None = None
        for row in ordered:
            token = _cutoff_token(float(row["prefix_cutoff"]))
            judge_metric = float(row["judge_metric"])
            baseline_metric = float(row["baseline_metric"])
            sparse_signal = float(row["sparse_reward_signal"])
            replay_row[f"cutoff_{token}_judge_metric"] = round(judge_metric, 6)
            replay_row[f"cutoff_{token}_baseline_metric"] = round(baseline_metric, 6)
            replay_row[f"cutoff_{token}_sparse_reward_signal"] = round(sparse_signal, 6)
            replay_row[f"cutoff_{token}_gap"] = round(float(row["baseline_vs_judge_gap"]), 6)
            replay_row[f"cutoff_{token}_judge_delta"] = (
                "" if previous_judge is None else round(judge_metric - previous_judge, 6)
            )
            replay_row[f"cutoff_{token}_baseline_delta"] = (
                "" if previous_baseline is None else round(baseline_metric - previous_baseline, 6)
            )
            previous_judge = judge_metric
            previous_baseline = baseline_metric
        replay_rows.append(replay_row)
    return replay_rows


def _write_replay_csv(rows: list[dict[str, Any]], path: Path) -> None:
    replay_rows = _episode_replay_rows(rows)
    headers = list(REPLAY_STATIC_HEADERS)
    if replay_rows:
        headers += sorted(
            {
                header
                for replay_row in replay_rows
                for header in replay_row
                if header not in REPLAY_STATIC_HEADERS
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(replay_rows)


def _format_series(rows: list[dict[str, Any]], key: str) -> str:
    ordered = sorted(rows, key=lambda row: float(row["prefix_cutoff"]))
    return " -> ".join(f"{float(row[key]):.3f}" for row in ordered)


def _top_replay_slices(rows: list[dict[str, Any]], limit: int = 5) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task_id"]), str(row["episode_id"]))].append(row)
    ranked = sorted(
        grouped.values(),
        key=lambda episode_rows: (
            max(abs(float(row["baseline_vs_judge_gap"])) for row in episode_rows),
            max(float(row["judge_uncertainty_score"]) for row in episode_rows),
            max(float(row["judge_metric"]) for row in episode_rows),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    headers = list(rows[0].keys()) if rows else list(COMPARISON_HEADERS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _timeline_series(
    rows: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Mean judge/baseline/sparse signal per cutoff, cutoffs sorted ascending."""
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["prefix_cutoff"])].append(row)
    cutoffs = sorted(grouped)
    judge_values = [
        sum(float(row["judge_metric"]) for row in grouped[cutoff]) / len(grouped[cutoff])
        for cutoff in cutoffs
    ]
    baseline_values = [
        sum(float(row["baseline_metric"]) for row in grouped[cutoff]) / len(grouped[cutoff])
        for cutoff in cutoffs
    ]
    sparse_values = [
        sum(float(row["sparse_reward_signal"]) for row in grouped[cutoff]) / len(grouped[cutoff])
        for cutoff in cutoffs
    ]
    return cutoffs, judge_values, baseline_values, sparse_values


def _write_svg_timeline_plot(rows: list[dict[str, Any]], path: Path) -> None:
    _cutoffs, judge_values, baseline_values, sparse_values = _timeline_series(rows)

    width = 760
    height = 420
    padding = 48
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\" />
  <text x=\"{padding}\" y=\"28\" font-size=\"20\" font-family=\"Arial, sans-serif\">Prefix-level score timeline</text>
  <line x1=\"{padding}\" y1=\"{height - padding}\" x2=\"{width - padding}\" y2=\"{height - padding}\" stroke=\"#444\" />
  <line x1=\"{padding}\" y1=\"{padding}\" x2=\"{padding}\" y2=\"{height - padding}\" stroke=\"#444\" />
  {plotting.svg_polyline(plotting.line_points(judge_values, width, height, padding), "#2f7ed8")}
  {plotting.svg_polyline(plotting.line_points(baseline_values, width, height, padding), "#8bbc21")}
  {plotting.svg_polyline(plotting.line_points(sparse_values, width, height, padding), "#c42525")}
  <text x=\"{padding}\" y=\"{height - 12}\" font-size=\"12\">judge failure score</text>
  <text x=\"{padding + 180}\" y=\"{height - 12}\" font-size=\"12\">baseline failure score</text>
  <text x=\"{padding + 390}\" y=\"{height - 12}\" font-size=\"12\">sparse reward signal</text>
</svg>
"""
    path.write_text(payload, encoding="utf-8")


def _write_timeline_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not plotting.MATPLOTLIB_AVAILABLE:
        _write_svg_timeline_plot(rows, path)
        return

    cutoffs, judge_values, baseline_values, sparse_values = _timeline_series(rows)

    fig, ax = plotting.plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        cutoffs,
        judge_values,
        marker="o",
        linewidth=2.0,
        color="#2f7ed8",
        label="judge failure score",
    )
    ax.plot(
        cutoffs,
        baseline_values,
        marker="s",
        linewidth=2.0,
        color="#8bbc21",
        label="baseline failure score",
    )
    ax.plot(
        cutoffs,
        sparse_values,
        marker="^",
        linewidth=1.5,
        color="#c42525",
        label="sparse reward signal",
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(cutoffs)
    ax.set_xlabel("prefix cutoff")
    ax.set_ylabel("mean score")
    ax.set_title("Prefix-level score timeline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plotting.plt.close(fig)


def _top_disagreements(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: abs(float(row["baseline_vs_judge_gap"])), reverse=True)[
        :limit
    ]


def _build_push_v3_hard_disagreement_pack(
    rows: list[dict[str, Any]],
    limit: int = 8,
    *,
    task_id: str = HARD_PACK_TASK_ID,
    families: tuple[str, ...] = HARD_POLICY_FAMILIES,
) -> list[dict[str, Any]]:
    """Pick the pack rows: family diversity first, then raw gap magnitude.

    One row per policy family is selected before any family repeats, so the
    pack cannot degenerate into a single-family cherry-pick.
    """
    filtered = [
        row for row in rows if row["task_id"] == task_id and row["policy_family"] in families
    ]
    ranked = sorted(
        filtered,
        key=lambda row: (
            abs(float(row["baseline_vs_judge_gap"])),
            float(row.get("judge_uncertainty_score", 0.0)),
            float(row.get("judge_metric", 0.0)),
        ),
        reverse=True,
    )
    pack: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for row in ranked:
        family = str(row["policy_family"])
        if family in seen_families:
            continue
        pack.append(row)
        seen_families.add(family)
        if len(pack) >= limit:
            return pack
    for row in ranked:
        if row in pack:
            continue
        pack.append(row)
        if len(pack) >= limit:
            break
    return pack


def _judge_mode_provenance(judge: list[dict[str, Any]]) -> str:
    """Distinct ``judge_mode`` values of the scored rows, or ``unknown``."""
    modes = sorted({str(row["judge_mode"]) for row in judge if row.get("judge_mode")})
    return ", ".join(modes) if modes else "unknown"


def _markdown_artifact(
    rows: list[dict[str, Any]],
    csv_path: Path,
    plot_path: Path,
    disagreement_pack: list[dict[str, Any]],
    disagreement_pack_path: Path,
    replay_path: Path,
    judge_mode: str,
) -> str:
    lines = ["# LeWorldModel Judge demo artifact", ""]
    lines.append(f"Judge mode: `{judge_mode}`")
    lines.append("")
    lines.append("## What this artifact is for")
    lines.append("- show prefix-level score movement, not just endpoint success")
    lines.append("- show where judge and baseline disagree")
    lines.append("- show the evidence fields driving the current verdict surface")
    lines.append("")
    lines.append("## Output inventory")
    lines.append(f"- comparison table: `{csv_path.name}`")
    lines.append(f"- score timeline plot: `{plot_path.name}`")
    lines.append(f"- push-v3 hard-family disagreement pack: `{disagreement_pack_path.name}`")
    lines.append(f"- episode score replay table: `{replay_path.name}`")
    lines.append("")

    if rows:
        lines.append("## Coverage snapshot")
        lines.append(f"- prefix rows: `{len(rows)}`")
        lines.append(f"- tasks: `{', '.join(sorted({row['task_id'] for row in rows}))}`")
        lines.append(f"- families: `{', '.join(sorted({row['policy_family'] for row in rows}))}`")
        lines.append(
            f"- cutoffs: `{', '.join(str(row) for row in sorted({r['prefix_cutoff'] for r in rows}))}`"
        )
        lines.append("")

        lines.append("## Biggest baseline-vs-judge disagreements")
        for row in _top_disagreements(rows):
            lines.append(
                f"- `{row['episode_id']}` @ cutoff `{row['prefix_cutoff']}` ({row['task_id']}/{row['policy_family']}) → "
                f"judge `{row['judge_metric']}` vs baseline `{row['baseline_metric']}`; gap `{row['baseline_vs_judge_gap']}`; "
                f"label=`{row['prefix_failure_label']}` recoverability=`{row['prefix_recoverability_label']}`"
            )
        lines.append("")

        if disagreement_pack:
            lines.append("## Push-v3 hard-family disagreement pack")
            lines.append(
                "- this pack is intentionally family-diverse so the demo is not just synthetic weak/doomed cherry-picking"
            )
            for row in disagreement_pack:
                lines.append(
                    f"- `{row['episode_id']}` ({row['policy_family']}) @ cutoff `{row['prefix_cutoff']}` → "
                    f"judge `{row['judge_metric']}` vs baseline `{row['baseline_metric']}`; gap `{row['baseline_vs_judge_gap']}`; "
                    f"uncertainty `{row['judge_uncertainty_score']}`; recoverability=`{row['prefix_recoverability_label']}`"
                )
            lines.append("")

        lines.append("## Score-over-time replays")
        lines.append(
            "- the replay CSV is one row per episode with per-cutoff judge/baseline/sparse signals and first-difference deltas so reviewers can audit drift, not just means"
        )
        for episode_rows in _top_replay_slices(rows):
            ordered = sorted(episode_rows, key=lambda row: float(row["prefix_cutoff"]))
            head = ordered[0]
            lines.append(
                f"- `{head['episode_id']}` ({head['task_id']}/{head['policy_family']}) success=`{head['success_label']}` → "
                f"judge `{_format_series(ordered, 'judge_metric')}` | "
                f"baseline `{_format_series(ordered, 'baseline_metric')}` | "
                f"sparse `{_format_series(ordered, 'sparse_reward_signal')}` | "
                f"latest label=`{ordered[-1]['prefix_recoverability_label']}`"
            )
        lines.append("")

        exemplar = _top_disagreements(rows, limit=1)[0]
        lines.append("## Evidence decomposition example")
        lines.append(
            f"Example prefix: `{exemplar['episode_id']}` / `{exemplar['task_id']}` / cutoff `{exemplar['prefix_cutoff']}`"
        )
        for key in EVIDENCE_KEYS:
            lines.append(f"- {key}: `{exemplar[key]}`")
        if "latent_mismatch_score" in exemplar:
            lines.append(f"- latent_mismatch_score: `{exemplar['latent_mismatch_score']}`")
        if "latent_alignment_score" in exemplar:
            lines.append(f"- latent_alignment_score: `{exemplar['latent_alignment_score']}`")
        lines.append(f"- judge_on_track_score: `{exemplar['judge_on_track_score']}`")
        lines.append(f"- judge_implausibility_score: `{exemplar['judge_implausibility_score']}`")
        lines.append(f"- judge_uncertainty_score: `{exemplar['judge_uncertainty_score']}`")
        lines.append("")

        lines.append("## Read it honestly")
        lines.append("- this is still a score-surface showcase, not a faithful JEPA world model")
        lines.append(
            "- disagreement rows are useful because they show where sparse reward misses prefix state"
        )
        lines.append(
            "- if the timeline looks clean but disagreement rows are nonsense, the artifact is still weak"
        )
    else:
        lines.append(
            "No joined rows were available. The artifact generation path ran, but the inputs were empty."
        )

    lines.append("")
    lines.append("## Provenance rule")
    lines.append(
        "- every claim in this demo should be traceable back to the comparison CSV plus the underlying prefixes/baselines/judge JSONL files"
    )
    return "\n".join(lines) + "\n"


def render_demo(
    prefixes: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    judge: list[dict[str, Any]],
    output: str | Path,
    families: Iterable[str] | str | None = None,
) -> dict[str, str]:
    """Render the full demo bundle; ``output`` is the markdown artifact path.

    Sibling CSV/plot files are derived from the output stem (contract — see
    module docstring). ``families`` optionally restricts every output to the
    named policy families. Returns the written paths keyed by role
    (``markdown``, ``comparison_csv``, ``disagreement_pack_csv``,
    ``replay_csv``, ``plot``).
    """
    prefixes, baselines, judge = _filter_inputs_by_family(
        prefixes,
        baselines,
        judge,
        _normalise_family_filter(families),
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _comparison_rows(prefixes, baselines, judge)
    csv_path = output_path.with_name(output_path.stem + "-comparison.csv")
    plot_suffix = ".png" if plotting.MATPLOTLIB_AVAILABLE else ".svg"
    plot_path = output_path.with_name(output_path.stem + "-timeline" + plot_suffix)
    disagreement_pack = _build_push_v3_hard_disagreement_pack(rows)
    disagreement_pack_path = output_path.with_name(
        output_path.stem + "-push-v3-hard-disagreement-pack.csv"
    )
    replay_path = output_path.with_name(output_path.stem + "-score-replay.csv")

    _write_comparison_csv(rows, csv_path)
    _write_comparison_csv(disagreement_pack, disagreement_pack_path)
    _write_replay_csv(rows, replay_path)
    _write_timeline_plot(rows, plot_path)
    output_path.write_text(
        _markdown_artifact(
            rows,
            csv_path,
            plot_path,
            disagreement_pack,
            disagreement_pack_path,
            replay_path,
            _judge_mode_provenance(judge),
        ),
        encoding="utf-8",
    )
    return {
        "markdown": str(output_path),
        "comparison_csv": str(csv_path),
        "disagreement_pack_csv": str(disagreement_pack_path),
        "replay_csv": str(replay_path),
        "plot": str(plot_path),
    }
