# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=2,<3"]
# ///
"""Generate every figure in the companion paper from the checked-in artifacts.

All figures are deterministic functions of files under artifacts/; nothing is
re-simulated. Run from anywhere:

    uv run docs/paper/figures/make_figures.py

Outputs vector PDFs next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
FRESH = REPO_ROOT / "artifacts" / "hard-family-real-fresh-capture-2026-07-13"
REFERENCE = REPO_ROOT / "artifacts" / "hard-family-real-held-out-2026-04-28"

FAMILIES = ("expert", "weak", "doomed", "misleading")
# Okabe-Ito palette, colorblind-safe.
FAMILY_COLORS = {
    "expert": "#0072B2",
    "weak": "#E69F00",
    "doomed": "#D55E00",
    "misleading": "#CC79A7",
}
SIGNAL_COLORS = {"judge": "#0072B2", "sparse": "#999999", "progress": "#E69F00"}
CUTOFFS = (0.25, 0.50, 0.75)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
    }
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def judge_threshold(artifact: Path) -> float:
    summary = json.loads((artifact / "summary-composite.json").read_text(encoding="utf-8"))
    return summary["thresholds"]["judge_failure_threshold"]


def savefig(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def fig_early_separation() -> None:
    """Mean composite failure score by cutoff, split by the heuristic prefix failure label
    (fresh capture). The labeler never emits a failure label before cutoff 0.5, so the
    labeled line starts there."""
    prefixes = {
        (r["task_id"], r["episode_id"], r["prefix_fraction"]): r
        for r in read_jsonl(FRESH / "prefixes.jsonl")
    }
    rows = read_jsonl(FRESH / "judge-composite.jsonl")
    threshold = judge_threshold(FRESH)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for labeled, color, label in (
        (True, "#D55E00", "failure-labeled prefixes"),
        (False, "#0072B2", "unlabeled prefixes"),
    ):
        xs, ys, ns = [], [], []
        for cutoff in CUTOFFS:
            scores = [
                r["failure_score"]
                for r in rows
                if r["prefix_fraction"] == cutoff
                and bool(
                    prefixes[(r["task_id"], r["episode_id"], r["prefix_fraction"])][
                        "prefix_failure_label"
                    ]
                )
                == labeled
            ]
            if scores:
                xs.append(cutoff)
                ys.append(float(np.mean(scores)))
                ns.append(len(scores))
        ax.plot(xs, ys, marker="o", markersize=4, label=label, color=color)
        for x, y, n in zip(xs, ys, ns, strict=False):
            ax.annotate(
                f"n={n}", xy=(x, y), xytext=(3, -9), textcoords="offset points", fontsize=6.5
            )
    ax.axhline(threshold, color="#333333", linewidth=0.8, linestyle="--")
    ax.annotate(
        f"held-out threshold {threshold:.3f}",
        xy=(0.26, threshold + 0.015),
        fontsize=7.5,
        color="#333333",
    )
    ax.set_xlabel("prefix cutoff (fraction of horizon)")
    ax.set_ylabel("mean failure score")
    ax.set_xticks(CUTOFFS)
    ax.set_ylim(0, 0.7)
    ax.legend(frameon=False, loc="upper left")
    savefig(fig, "fig-early-separation.pdf")


def fig_baseline_comparison() -> None:
    """Judge vs baselines on the held-out evaluation slice of the fresh capture."""
    summary = json.loads((FRESH / "summary-composite.json").read_text(encoding="utf-8"))
    overall = summary["overall"]
    metrics = [
        (
            "Hit rate",
            "judge_failure_hit_rate",
            "baseline_sparse_absence_hit_rate",
            "baseline_progress_hit_rate",
        ),
        (
            "FPR",
            "judge_false_positive_rate",
            "baseline_sparse_absence_false_positive_rate",
            "baseline_progress_false_positive_rate",
        ),
        (
            "Pairwise acc.",
            "judge_pairwise_accuracy",
            "baseline_sparse_absence_pairwise_accuracy",
            "baseline_progress_pairwise_accuracy",
        ),
        (
            "Avg. precision",
            "judge_average_precision",
            "baseline_sparse_absence_average_precision",
            "baseline_progress_average_precision",
        ),
    ]
    signals = ("judge", "sparse", "progress")
    labels = ("Judge (composite)", "Sparse reward", "Progress proxy")
    x = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for i, (signal, label) in enumerate(zip(signals, labels, strict=False)):
        values = [m[1 + i] for m in metrics]
        heights = [overall[v] for v in values]
        bars = ax.bar(x + (i - 1) * width, heights, width, label=label, color=SIGNAL_COLORS[signal])
        for bar, height in zip(bars, heights, strict=False):
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("value")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    savefig(fig, "fig-baseline-comparison.pdf")


def balanced_accuracy_curve(artifact: Path) -> tuple[np.ndarray, np.ndarray]:
    """Balanced accuracy over failure-score thresholds on the calibration cohort."""
    prefixes = {
        (r["task_id"], r["episode_id"], r["prefix_fraction"]): r
        for r in read_jsonl(artifact / "prefixes.jsonl")
    }
    scores, labels = [], []
    for row in read_jsonl(artifact / "judge-composite.jsonl"):
        prefix = prefixes[(row["task_id"], row["episode_id"], row["prefix_fraction"])]
        if prefix["policy_family"] in ("weak", "doomed"):
            scores.append(row["failure_score"])
            labels.append(bool(prefix["prefix_failure_label"]))
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    grid = np.linspace(0.0, 1.0, 401)
    accs = []
    positives = labels_arr.sum()
    negatives = (~labels_arr).sum()
    for threshold in grid:
        flagged = scores_arr >= threshold
        tpr = (flagged & labels_arr).sum() / positives if positives else 0.0
        tnr = (~flagged & ~labels_arr).sum() / negatives if negatives else 0.0
        accs.append((tpr + tnr) / 2)
    return grid, np.array(accs)


def fig_threshold_transfer() -> None:
    """Calibration curves on disjoint captures; the chosen operating points nearly coincide."""
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    for artifact, label, color in (
        (REFERENCE, "2026-04-28 capture (18 cal. prefixes)", "#999999"),
        (FRESH, "fresh capture (90 cal. prefixes)", "#0072B2"),
    ):
        grid, accs = balanced_accuracy_curve(artifact)
        ax.plot(grid, accs, color=color, label=label)
        threshold = judge_threshold(artifact)
        ax.axvline(threshold, color=color, linewidth=0.8, linestyle="--")
    ax.set_xlabel("failure-score threshold")
    ax.set_ylabel("balanced accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.35, 1.05)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    savefig(fig, "fig-threshold-transfer.pdf")


def fig_score_distribution() -> None:
    """Every composite failure score in the fresh capture, by family, with the threshold."""
    rows = read_jsonl(FRESH / "judge-composite.jsonl")
    threshold = judge_threshold(FRESH)
    rng = np.random.default_rng(1013)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, family in enumerate(FAMILIES):
        scores = [r["failure_score"] for r in rows if r["policy_family"] == family]
        jitter = rng.uniform(-0.16, 0.16, size=len(scores))
        ax.scatter(
            np.full(len(scores), i) + jitter,
            scores,
            s=9,
            alpha=0.65,
            color=FAMILY_COLORS[family],
            linewidths=0,
        )
    ax.axhline(threshold, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels(FAMILIES)
    ax.set_ylabel("composite failure score")
    ax.set_ylim(-0.02, 1.02)
    savefig(fig, "fig-score-distribution.pdf")


def main() -> None:
    fig_early_separation()
    fig_baseline_comparison()
    fig_threshold_transfer()
    fig_score_distribution()


if __name__ == "__main__":
    main()
