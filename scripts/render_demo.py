from __future__ import annotations

import argparse
from pathlib import Path

from leworldmodel_judge.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefixes', required=True)
    parser.add_argument('--judge', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    prefixes = read_jsonl(args.prefixes)
    judge = read_jsonl(args.judge)
    lines = ['# LeWorldModel Judge demo artifact', '']
    lines.append(f'prefix count: {len(prefixes)}')
    lines.append(f'judge rows: {len(judge)}')
    if judge:
        sample = judge[0]
        lines.append('')
        lines.append('sample judge row:')
        for key in sorted(sample):
            lines.append(f'- {key}: {sample[key]}')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f'wrote demo artifact to {args.output}')


if __name__ == '__main__':
    main()
