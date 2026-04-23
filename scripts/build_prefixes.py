from __future__ import annotations

import argparse

from leworldmodel_judge.data import build_prefixes
from leworldmodel_judge.io import read_jsonl, write_jsonl


def parse_fractions(value: str) -> tuple[float, ...]:
    return tuple(float(x) for x in value.split(','))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fractions', default='0.25,0.50,0.75')
    args = parser.parse_args()

    steps = read_jsonl(args.input)
    prefixes = build_prefixes(steps, parse_fractions(args.fractions))
    write_jsonl(args.output, [p.to_dict() for p in prefixes])
    print(f'wrote {len(prefixes)} prefixes to {args.output}')


if __name__ == '__main__':
    main()
