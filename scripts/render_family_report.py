from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised in CI/runtime environments without matplotlib
    plt = None
    MATPLOTLIB_AVAILABLE = False


def _fmt(value: float | None) -> str:
    if value is None:
        return 'n/a'
    return f'{float(value):.3f}'



def _metric_row(name: str, payload: dict) -> str:
    return (
        f"| {name} | {payload['count']} | {payload['failure_labels']} | "
        f"{_fmt(payload.get('judge_failure_hit_rate'))} | {_fmt(payload.get('judge_false_positive_rate'))} | "
        f"{_fmt(payload.get('judge_pairwise_accuracy'))} | {_fmt(payload.get('failure_label_coverage'))} |"
    )


def _write_placeholder_png(path: Path) -> None:
    pixel = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6pQxQAAAAASUVORK5CYII='
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pixel)



def render_family_report(summary: dict, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    families = summary.get('families', {})
    markdown_path = output_dir / 'family-report.md'
    plot_path = output_dir / 'family-report.png'

    calibration = summary.get('calibration', {})
    judge_calibration = calibration.get('judge', {})
    progress_calibration = calibration.get('progress', {})

    lines = [
        '# Family-aware benchmark report',
        '',
        '## Threshold provenance',
        f"- chosen judge threshold: `{summary['thresholds']['judge_failure_threshold']}`",
        f"- fixed progress failure threshold: `{summary['thresholds']['progress_failure_threshold']}`",
        f"- judge threshold selection mode: `{judge_calibration.get('mode', 'unknown')}`",
        f"- judge calibration balanced accuracy (same slice): `{judge_calibration.get('balanced_accuracy', 'n/a')}`",
        f"- judge calibration hit rate (same slice): `{judge_calibration.get('hit_rate', 'n/a')}`",
        f"- judge calibration false positive rate (same slice): `{judge_calibration.get('false_positive_rate', 'n/a')}`",
        f"- progress baseline mode: `{progress_calibration.get('mode', 'unknown')}`",
        '',
        '## Honesty note',
        '- if the threshold was chosen on the same benchmark slice, present it as in-slice tuning, not held-out calibration.',
        '- if a family looks good only because coverage is tiny, the artifact should say that out loud.',
        '',
        '## Per-family table',
        '',
        '| family | count | failure labels | judge hit rate | judge false positive rate | judge pairwise accuracy | failure coverage |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for family, payload in sorted(families.items()):
        lines.append(_metric_row(family, payload))
    markdown_path.write_text('\n'.join(lines) + '\n')

    family_names = sorted(families)
    hit_rates = [families[name]['judge_failure_hit_rate'] for name in family_names]
    false_positive_rates = [families[name]['judge_false_positive_rate'] for name in family_names]
    pairwise = [families[name]['judge_pairwise_accuracy'] or 0.0 for name in family_names]

    if not MATPLOTLIB_AVAILABLE:
        _write_placeholder_png(plot_path)
        return {'markdown': str(markdown_path), 'plot': str(plot_path)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(family_names, hit_rates, label='hit rate', color='#2f7ed8')
    axes[0].bar(family_names, false_positive_rates, label='false positive rate', color='#c42525', alpha=0.75)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title('Judge hit/FPR by family')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend()

    axes[1].bar(family_names, pairwise, color='#8bbc21')
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title('Judge pairwise accuracy by family')
    axes[1].tick_params(axis='x', rotation=30)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return {'markdown': str(markdown_path), 'plot': str(plot_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    with open(args.summary) as fh:
        summary = json.load(fh)
    outputs = render_family_report(summary, args.output_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    main()
