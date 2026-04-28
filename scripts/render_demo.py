from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised in CI/runtime environments without matplotlib
    plt = None
    MATPLOTLIB_AVAILABLE = False

from leworldmodel_judge.io import read_jsonl

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
HARD_POLICY_FAMILIES = ("expert", "weak", "doomed", "misleading", "random")


def _svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    serialised = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{serialised}" />'


def _line_points(
    values: list[float], width: int = 760, height: int = 420, padding: int = 48
) -> list[tuple[float, float]]:
    if not values:
        return []
    usable_width = width - (2 * padding)
    usable_height = height - (2 * padding)
    if len(values) == 1:
        x_positions = [padding + usable_width / 2]
    else:
        x_positions = [
            padding + usable_width * idx / (len(values) - 1) for idx in range(len(values))
        ]
    return [
        (x_positions[idx], padding + usable_height * (1.0 - max(0.0, min(1.0, values[idx]))))
        for idx in range(len(values))
    ]


def _row_key(row: dict) -> tuple[str, str, float]:
    return (row["task_id"], row["episode_id"], float(row["prefix_fraction"]))


def _family_name(prefix: dict) -> str:
    family = prefix.get("policy_family")
    if family:
        return str(family)
    episode_id = str(prefix.get("episode_id", ""))
    task_id = str(prefix.get("task_id", ""))
    prefix_marker = f"{task_id}-"
    if episode_id.startswith(prefix_marker) and "-ep-" in episode_id:
        return episode_id[len(prefix_marker) :].split("-ep-")[0]
    return "unknown"


def _comparison_rows(prefixes: list[dict], baselines: list[dict], judge: list[dict]) -> list[dict]:
    baseline_map = {_row_key(row): row for row in baselines}
    judge_map = {_row_key(row): row for row in judge}
    rows: list[dict] = []
    for prefix in prefixes:
        key = _row_key(prefix)
        baseline = baseline_map.get(key, {})
        judge_row = judge_map.get(key, {})
        progress_score = float(baseline.get("progress_proxy_score", 0.0))
        progress_failure_score = 1.0 - progress_score
        sparse_reward_signal = float(baseline.get("sparse_reward_score", 0.0))
        judge_signal = float(judge_row.get("failure_score", 0.0))
        row = {
            "task_id": prefix["task_id"],
            "episode_id": prefix["episode_id"],
            "policy_family": _family_name(prefix),
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


def _write_comparison_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        headers = [
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
        ]
    else:
        headers = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_svg_timeline_plot(rows: list[dict], path: Path) -> None:
    grouped: dict[float, list[dict]] = defaultdict(list)
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

    width = 760
    height = 420
    padding = 48
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\" />
  <text x=\"{padding}\" y=\"28\" font-size=\"20\" font-family=\"Arial, sans-serif\">Prefix-level score timeline</text>
  <line x1=\"{padding}\" y1=\"{height - padding}\" x2=\"{width - padding}\" y2=\"{height - padding}\" stroke=\"#444\" />
  <line x1=\"{padding}\" y1=\"{padding}\" x2=\"{padding}\" y2=\"{height - padding}\" stroke=\"#444\" />
  {_svg_polyline(_line_points(judge_values, width, height, padding), "#2f7ed8")}
  {_svg_polyline(_line_points(baseline_values, width, height, padding), "#8bbc21")}
  {_svg_polyline(_line_points(sparse_values, width, height, padding), "#c42525")}
  <text x=\"{padding}\" y=\"{height - 12}\" font-size=\"12\">judge failure score</text>
  <text x=\"{padding + 180}\" y=\"{height - 12}\" font-size=\"12\">baseline failure score</text>
  <text x=\"{padding + 390}\" y=\"{height - 12}\" font-size=\"12\">sparse reward signal</text>
</svg>
"""
    path.write_text(payload, encoding="utf-8")


def _write_timeline_plot(rows: list[dict], path: Path) -> None:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[float(row["prefix_cutoff"])].append(row)

    if not MATPLOTLIB_AVAILABLE:
        _write_svg_timeline_plot(rows, path)
        return

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

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
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
    plt.close(fig)


def _top_disagreements(rows: list[dict], limit: int = 5) -> list[dict]:
    return sorted(rows, key=lambda row: abs(float(row["baseline_vs_judge_gap"])), reverse=True)[
        :limit
    ]


def _build_push_v3_hard_disagreement_pack(rows: list[dict], limit: int = 8) -> list[dict]:
    filtered = [
        row
        for row in rows
        if row["task_id"] == "push-v3" and row["policy_family"] in HARD_POLICY_FAMILIES
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
    pack: list[dict] = []
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


def _markdown_artifact(
    rows: list[dict],
    csv_path: Path,
    plot_path: Path,
    disagreement_pack: list[dict],
    disagreement_pack_path: Path,
) -> str:
    lines = ["# LeWorldModel Judge demo artifact", ""]
    lines.append("## What this artifact is for")
    lines.append("- show prefix-level score movement, not just endpoint success")
    lines.append("- show where judge and baseline disagree")
    lines.append("- show the evidence fields driving the current verdict surface")
    lines.append("")
    lines.append("## Output inventory")
    lines.append(f"- comparison table: `{csv_path.name}`")
    lines.append(f"- score timeline plot: `{plot_path.name}`")
    lines.append(f"- push-v3 hard-family disagreement pack: `{disagreement_pack_path.name}`")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="markdown artifact path; sibling CSV/plot files are emitted automatically",
    )
    args = parser.parse_args()

    prefixes = read_jsonl(args.prefixes)
    baselines = read_jsonl(args.baselines)
    judge = read_jsonl(args.judge)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _comparison_rows(prefixes, baselines, judge)
    csv_path = output_path.with_name(output_path.stem + "-comparison.csv")
    plot_suffix = ".png" if MATPLOTLIB_AVAILABLE else ".svg"
    plot_path = output_path.with_name(output_path.stem + "-timeline" + plot_suffix)
    disagreement_pack = _build_push_v3_hard_disagreement_pack(rows)
    disagreement_pack_path = output_path.with_name(
        output_path.stem + "-push-v3-hard-disagreement-pack.csv"
    )

    _write_comparison_csv(rows, csv_path)
    _write_comparison_csv(disagreement_pack, disagreement_pack_path)
    _write_timeline_plot(rows, plot_path)
    output_path.write_text(
        _markdown_artifact(rows, csv_path, plot_path, disagreement_pack, disagreement_pack_path),
        encoding="utf-8",
    )
    print(f"wrote demo artifact to {output_path}")
    print(f"wrote comparison table to {csv_path}")
    print(f"wrote push-v3 disagreement pack to {disagreement_pack_path}")
    print(f"wrote timeline plot to {plot_path}")


if __name__ == "__main__":
    main()
