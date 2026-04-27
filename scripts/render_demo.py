from __future__ import annotations

import argparse
import base64
import csv
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised in CI/runtime environments without matplotlib
    plt = None
    MATPLOTLIB_AVAILABLE = False

from leworldmodel_judge.io import read_jsonl

EVIDENCE_KEYS = (
    'progress_proxy',
    'distance_progress',
    'target_distance_last',
    'target_distance_best',
    'in_place_score',
    'grasp_signal_peak',
    'success_signal_peak',
    'reward_density',
    'stall_score',
)


def _row_key(row: dict) -> tuple[str, str, float]:
    return (row['task_id'], row['episode_id'], float(row['prefix_fraction']))


def _family_name(prefix: dict) -> str:
    family = prefix.get('policy_family')
    if family:
        return str(family)
    episode_id = str(prefix.get('episode_id', ''))
    task_id = str(prefix.get('task_id', ''))
    prefix_marker = f'{task_id}-'
    if episode_id.startswith(prefix_marker) and '-ep-' in episode_id:
        return episode_id[len(prefix_marker):].split('-ep-')[0]
    return 'unknown'


def _comparison_rows(prefixes: list[dict], baselines: list[dict], judge: list[dict]) -> list[dict]:
    baseline_map = {_row_key(row): row for row in baselines}
    judge_map = {_row_key(row): row for row in judge}
    rows: list[dict] = []
    for prefix in prefixes:
        key = _row_key(prefix)
        baseline = baseline_map.get(key, {})
        judge_row = judge_map.get(key, {})
        progress_score = float(baseline.get('progress_proxy_score', 0.0))
        progress_failure_score = 1.0 - progress_score
        sparse_reward_signal = float(baseline.get('sparse_reward_score', 0.0))
        judge_signal = float(judge_row.get('failure_score', 0.0))
        row = {
            'task_id': prefix['task_id'],
            'episode_id': prefix['episode_id'],
            'policy_family': _family_name(prefix),
            'prefix_cutoff': float(prefix['prefix_fraction']),
            'prefix_index': int(prefix['prefix_index']),
            'baseline_metric': round(progress_failure_score, 6),
            'judge_metric': round(judge_signal, 6),
            'sparse_reward_signal': round(sparse_reward_signal, 6),
            'judge_signal': round(judge_signal, 6),
            'success_label': bool(prefix.get('final_success_label', False)),
            'prefix_failure_label': bool(prefix.get('prefix_failure_label', False)),
            'prefix_recoverability_label': prefix.get('prefix_recoverability_label', 'unknown'),
            'judge_on_track_score': round(float(judge_row.get('on_track_score', 0.0)), 6),
            'judge_implausibility_score': round(float(judge_row.get('implausibility_score', 0.0)), 6),
            'judge_uncertainty_score': round(float(judge_row.get('uncertainty_score', 0.0)), 6),
            'baseline_vs_judge_gap': round(judge_signal - progress_failure_score, 6),
        }
        for evidence_key in EVIDENCE_KEYS:
            value = prefix.get(evidence_key)
            row[evidence_key] = '' if value is None else round(float(value), 6)
        rows.append(row)
    return rows


def _write_comparison_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        headers = [
            'task_id', 'episode_id', 'policy_family', 'prefix_cutoff', 'prefix_index', 'baseline_metric', 'judge_metric',
            'sparse_reward_signal', 'judge_signal', 'success_label', 'prefix_failure_label',
            'prefix_recoverability_label', 'judge_on_track_score', 'judge_implausibility_score',
            'judge_uncertainty_score', 'baseline_vs_judge_gap', *EVIDENCE_KEYS,
        ]
    else:
        headers = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_placeholder_png(path: Path) -> None:
    pixel = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6pQxQAAAAASUVORK5CYII='
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pixel)



def _write_timeline_plot(rows: list[dict], path: Path) -> None:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[float(row['prefix_cutoff'])].append(row)

    if not MATPLOTLIB_AVAILABLE:
        _write_placeholder_png(path)
        return

    cutoffs = sorted(grouped)
    judge_values = [sum(float(row['judge_metric']) for row in grouped[cutoff]) / len(grouped[cutoff]) for cutoff in cutoffs]
    baseline_values = [sum(float(row['baseline_metric']) for row in grouped[cutoff]) / len(grouped[cutoff]) for cutoff in cutoffs]
    sparse_values = [sum(float(row['sparse_reward_signal']) for row in grouped[cutoff]) / len(grouped[cutoff]) for cutoff in cutoffs]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(cutoffs, judge_values, marker='o', linewidth=2.0, color='#2f7ed8', label='judge failure score')
    ax.plot(cutoffs, baseline_values, marker='s', linewidth=2.0, color='#8bbc21', label='baseline failure score')
    ax.plot(cutoffs, sparse_values, marker='^', linewidth=1.5, color='#c42525', label='sparse reward signal')
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(cutoffs)
    ax.set_xlabel('prefix cutoff')
    ax.set_ylabel('mean score')
    ax.set_title('Prefix-level score timeline')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _top_disagreements(rows: list[dict], limit: int = 5) -> list[dict]:
    return sorted(rows, key=lambda row: abs(float(row['baseline_vs_judge_gap'])), reverse=True)[:limit]


def _markdown_artifact(rows: list[dict], csv_path: Path, plot_path: Path) -> str:
    lines = ['# LeWorldModel Judge demo artifact', '']
    lines.append('## What this artifact is for')
    lines.append('- show prefix-level score movement, not just endpoint success')
    lines.append('- show where judge and baseline disagree')
    lines.append('- show the evidence fields driving the current verdict surface')
    lines.append('')
    lines.append('## Output inventory')
    lines.append(f'- comparison table: `{csv_path.name}`')
    lines.append(f'- score timeline plot: `{plot_path.name}`')
    lines.append('')

    if rows:
        lines.append('## Coverage snapshot')
        lines.append(f"- prefix rows: `{len(rows)}`")
        lines.append(f"- tasks: `{', '.join(sorted({row['task_id'] for row in rows}))}`")
        lines.append(f"- families: `{', '.join(sorted({row['policy_family'] for row in rows}))}`")
        lines.append(f"- cutoffs: `{', '.join(str(row) for row in sorted({r['prefix_cutoff'] for r in rows}))}`")
        lines.append('')

        lines.append('## Biggest baseline-vs-judge disagreements')
        for row in _top_disagreements(rows):
            lines.append(
                f"- `{row['episode_id']}` @ cutoff `{row['prefix_cutoff']}` ({row['task_id']}/{row['policy_family']}) → "
                f"judge `{row['judge_metric']}` vs baseline `{row['baseline_metric']}`; gap `{row['baseline_vs_judge_gap']}`; "
                f"label=`{row['prefix_failure_label']}` recoverability=`{row['prefix_recoverability_label']}`"
            )
        lines.append('')

        exemplar = _top_disagreements(rows, limit=1)[0]
        lines.append('## Evidence decomposition example')
        lines.append(
            f"Example prefix: `{exemplar['episode_id']}` / `{exemplar['task_id']}` / cutoff `{exemplar['prefix_cutoff']}`"
        )
        for key in EVIDENCE_KEYS:
            lines.append(f"- {key}: `{exemplar[key]}`")
        lines.append(f"- judge_on_track_score: `{exemplar['judge_on_track_score']}`")
        lines.append(f"- judge_implausibility_score: `{exemplar['judge_implausibility_score']}`")
        lines.append(f"- judge_uncertainty_score: `{exemplar['judge_uncertainty_score']}`")
        lines.append('')

        lines.append('## Read it honestly')
        lines.append('- this is still a score-surface showcase, not a faithful JEPA world model')
        lines.append('- disagreement rows are useful because they show where sparse reward misses prefix state')
        lines.append('- if the timeline looks clean but disagreement rows are nonsense, the artifact is still weak')
    else:
        lines.append('No joined rows were available. The artifact generation path ran, but the inputs were empty.')

    lines.append('')
    lines.append('## Provenance rule')
    lines.append('- every claim in this demo should be traceable back to the comparison CSV plus the underlying prefixes/baselines/judge JSONL files')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefixes', required=True)
    parser.add_argument('--judge', required=True)
    parser.add_argument('--baselines', required=True)
    parser.add_argument('--output', required=True, help='markdown artifact path; sibling CSV/PNG files are emitted automatically')
    args = parser.parse_args()

    prefixes = read_jsonl(args.prefixes)
    baselines = read_jsonl(args.baselines)
    judge = read_jsonl(args.judge)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _comparison_rows(prefixes, baselines, judge)
    csv_path = output_path.with_name(output_path.stem + '-comparison.csv')
    plot_path = output_path.with_name(output_path.stem + '-timeline.png')

    _write_comparison_csv(rows, csv_path)
    _write_timeline_plot(rows, plot_path)
    output_path.write_text(_markdown_artifact(rows, csv_path, plot_path), encoding='utf-8')
    print(f'wrote demo artifact to {output_path}')
    print(f'wrote comparison table to {csv_path}')
    print(f'wrote timeline plot to {plot_path}')


if __name__ == '__main__':
    main()
