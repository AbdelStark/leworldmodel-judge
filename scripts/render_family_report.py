from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _metric_row(name: str, payload: dict) -> str:
    return (
        f"| {name} | {payload['count']} | {payload['failure_labels']} | "
        f"{payload['judge_failure_hit_rate']:.3f} | {payload['judge_false_positive_rate']:.3f} | "
        f"{payload['judge_pairwise_accuracy'] if payload['judge_pairwise_accuracy'] is not None else 'n/a'} |"
    )


def render_family_report(summary: dict, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    families = summary.get('families', {})
    markdown_path = output_dir / 'family-report.md'
    plot_path = output_dir / 'family-report.png'

    lines = [
        '# Family-aware benchmark report',
        '',
        '## Calibration',
        f"- judge threshold: `{summary['thresholds']['judge_failure_threshold']}`",
        f"- progress threshold: `{summary['thresholds']['progress_failure_threshold']}`",
        '',
        '## Per-family table',
        '',
        '| family | count | failure labels | judge hit rate | judge false positive rate | judge pairwise accuracy |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for family, payload in sorted(families.items()):
        lines.append(_metric_row(family, payload))
    markdown_path.write_text('\n'.join(lines) + '\n')

    family_names = sorted(families)
    hit_rates = [families[name]['judge_failure_hit_rate'] for name in family_names]
    false_positive_rates = [families[name]['judge_false_positive_rate'] for name in family_names]
    pairwise = [families[name]['judge_pairwise_accuracy'] or 0.0 for name in family_names]

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
