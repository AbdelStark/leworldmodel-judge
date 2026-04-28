from __future__ import annotations

import argparse

from leworldmodel_judge.baselines import score_prefix
from leworldmodel_judge.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prefixes = read_jsonl(args.input)
    rows = [score_prefix(prefix) for prefix in prefixes]
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} baseline rows to {args.output}")


if __name__ == "__main__":
    main()
