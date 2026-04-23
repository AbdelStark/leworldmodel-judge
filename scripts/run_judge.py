from __future__ import annotations

import argparse

from leworldmodel_judge.io import read_jsonl, write_jsonl
from leworldmodel_judge.judge import heuristic_surprise_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['heuristic_surprise', 'dummy'], default='heuristic_surprise')
    args = parser.parse_args()

    prefixes = read_jsonl(args.input)
    if args.mode == 'dummy':
        rows = []
        for prefix in prefixes:
            rows.append({
                'episode_id': prefix['episode_id'],
                'task_id': prefix['task_id'],
                'prefix_fraction': prefix['prefix_fraction'],
                'on_track_score': 0.0,
                'failure_score': 0.0,
                'implausibility_score': 0.0,
                'uncertainty_score': 1.0,
                'judge_mode': 'dummy',
            })
    else:
        rows = [heuristic_surprise_score(prefix) for prefix in prefixes]
    write_jsonl(args.output, rows)
    print(f'wrote {len(rows)} judge rows to {args.output}')


if __name__ == '__main__':
    main()
