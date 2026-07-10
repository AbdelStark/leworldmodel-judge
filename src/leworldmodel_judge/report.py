"""Family-aware benchmark report renderer.

Turns the ``summarize`` summary dict into ``family-report.md`` (threshold
provenance, honesty notes, per-family metric table) plus a bar plot —
``family-report.png`` with matplotlib, ``family-report.svg`` without. The
markdown deliberately leads with threshold provenance: a number without its
calibration story is not a result (docs/method.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import plotting


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _metric_row(name: str, payload: dict[str, Any]) -> str:
    return (
        f"| {name} | {payload['count']} | {payload['failure_labels']} | "
        f"{_fmt(payload.get('judge_failure_hit_rate'))} | {_fmt(payload.get('judge_false_positive_rate'))} | "
        f"{_fmt(payload.get('judge_pairwise_accuracy'))} | {_fmt(payload.get('judge_average_precision'))} | "
        f"{_fmt(payload.get('failure_label_coverage'))} |"
    )


def _write_svg_family_plot(families: dict[str, Any], path: Path) -> None:
    """SVG fallback for the per-family bar plot (no matplotlib needed).

    ``None`` metrics (degenerate label distributions) plot as 0.0, matching
    the matplotlib path.
    """
    family_names = sorted(families)
    hit_rates = [float(families[name]["judge_failure_hit_rate"] or 0.0) for name in family_names]
    false_positive_rates = [
        float(families[name]["judge_false_positive_rate"] or 0.0) for name in family_names
    ]
    pairwise = [float(families[name]["judge_pairwise_accuracy"] or 0.0) for name in family_names]
    padding = 40
    bar_width = 28
    gap = 16
    x = padding
    bars = []
    labels = []
    for idx, family in enumerate(family_names):
        hit_height = 220 * hit_rates[idx]
        fpr_height = 220 * false_positive_rates[idx]
        pair_height = 220 * pairwise[idx]
        bars.append(
            f'<rect x="{x}" y="{300 - hit_height:.1f}" width="{bar_width}" height="{hit_height:.1f}" fill="#2f7ed8" />'
        )
        bars.append(
            f'<rect x="{x + bar_width + 4}" y="{300 - fpr_height:.1f}" width="{bar_width}" height="{fpr_height:.1f}" fill="#c42525" />'
        )
        bars.append(
            f'<rect x="{x + (2 * (bar_width + 4))}" y="{300 - pair_height:.1f}" width="{bar_width}" height="{pair_height:.1f}" fill="#8bbc21" />'
        )
        labels.append(f'<text x="{x}" y="330" font-size="12">{family}</text>')
        x += (3 * (bar_width + 4)) + gap
    payload = """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"420\" viewBox=\"0 0 960 420\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\" />
  <text x=\"40\" y=\"28\" font-size=\"20\" font-family=\"Arial, sans-serif\">Judge family report</text>
  <line x1=\"40\" y1=\"300\" x2=\"920\" y2=\"300\" stroke=\"#444\" />
  {bars}
  {labels}
  <text x=\"40\" y=\"380\" font-size=\"12\">blue=hit rate red=false positive rate green=pairwise accuracy</text>
</svg>
""".replace("{bars}", "\n  ".join(bars)).replace("{labels}", "\n  ".join(labels))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def render_family_report(summary: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Render the family report bundle into ``output_dir``.

    Returns ``{"markdown": <path>, "plot": <path>}``. The plot filename
    extension records how it was produced (``.png`` matplotlib, ``.svg``
    fallback).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    families = summary.get("families", {})
    markdown_path = output_dir / "family-report.md"
    plot_path = output_dir / (
        "family-report.png" if plotting.MATPLOTLIB_AVAILABLE else "family-report.svg"
    )

    calibration = summary.get("calibration", {})
    judge_calibration = calibration.get("judge", {})
    progress_calibration = calibration.get("progress", {})
    provenance = calibration.get("provenance", {})
    calibration_cohort = judge_calibration.get("cohort_stats", {})
    evaluation_stats = judge_calibration.get("evaluation_stats", {})
    evaluation_cohort = judge_calibration.get("evaluation_cohort", {})

    lines = [
        "# Family-aware benchmark report",
        "",
        "## Threshold provenance",
        f"- chosen judge threshold: `{summary['thresholds']['judge_failure_threshold']}`",
        f"- fixed progress failure threshold: `{summary['thresholds']['progress_failure_threshold']}`",
        f"- judge threshold selection mode: `{judge_calibration.get('mode', 'unknown')}`",
        f"- calibration families: `{', '.join(provenance.get('calibration_families', ['all']))}`",
        f"- evaluation families: `{', '.join(provenance.get('evaluation_families', ['all']))}`",
        f"- calibration/evaluation family overlap: `{provenance.get('family_overlap', 'n/a')}`",
        f"- calibration prefixes: `{provenance.get('calibration_count', 'n/a')}` with `{provenance.get('calibration_failure_labels', 'n/a')}` failure labels and `{provenance.get('calibration_non_failure_labels', 'n/a')}` non-failure labels",
        f"- evaluation prefixes: `{provenance.get('evaluation_count', 'n/a')}` with `{provenance.get('evaluation_failure_labels', 'n/a')}` failure labels and `{provenance.get('evaluation_non_failure_labels', 'n/a')}` non-failure labels",
        f"- judge calibration balanced accuracy: `{judge_calibration.get('balanced_accuracy', 'n/a')}`",
        f"- judge calibration hit rate: `{judge_calibration.get('hit_rate', 'n/a')}`",
        f"- judge calibration false positive rate: `{judge_calibration.get('false_positive_rate', 'n/a')}`",
        f"- judge calibration average precision: `{calibration_cohort.get('average_precision', 'n/a')}`",
        f"- judge evaluation balanced accuracy: `{evaluation_stats.get('balanced_accuracy', 'n/a')}`",
        f"- judge evaluation hit rate: `{evaluation_stats.get('hit_rate', 'n/a')}`",
        f"- judge evaluation false positive rate: `{evaluation_stats.get('false_positive_rate', 'n/a')}`",
        f"- judge evaluation average precision: `{evaluation_cohort.get('average_precision', 'n/a')}`",
        f"- progress baseline mode: `{progress_calibration.get('mode', 'unknown')}`",
        "",
        "## Honesty note",
        "- if the threshold was chosen on the same benchmark slice, present it as in-slice tuning, not held-out calibration.",
        "- if the threshold comes from a held-out family split, say exactly which families calibrated it and which families were scored with it.",
        "- if a family looks good only because coverage is tiny, the artifact should say that out loud.",
        "",
        "## Per-family table",
        "",
        "| family | count | failure labels | judge hit rate | judge false positive rate | judge pairwise accuracy | judge average precision | failure coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, payload in sorted(families.items()):
        lines.append(_metric_row(family, payload))
    markdown_path.write_text("\n".join(lines) + "\n")

    if not plotting.MATPLOTLIB_AVAILABLE:
        _write_svg_family_plot(families, plot_path)
        return {"markdown": str(markdown_path), "plot": str(plot_path)}

    family_names = sorted(families)
    hit_rates = [families[name]["judge_failure_hit_rate"] or 0.0 for name in family_names]
    false_positive_rates = [
        families[name]["judge_false_positive_rate"] or 0.0 for name in family_names
    ]
    pairwise = [families[name]["judge_pairwise_accuracy"] or 0.0 for name in family_names]

    fig, axes = plotting.plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(family_names, hit_rates, label="hit rate", color="#2f7ed8")
    axes[0].bar(
        family_names, false_positive_rates, label="false positive rate", color="#c42525", alpha=0.75
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Judge hit/FPR by family")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()

    axes[1].bar(family_names, pairwise, color="#8bbc21")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Judge pairwise accuracy by family")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plotting.plt.close(fig)

    return {"markdown": str(markdown_path), "plot": str(plot_path)}
