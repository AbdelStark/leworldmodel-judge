from __future__ import annotations

import argparse

from leworldmodel_judge.evaluate import summarize
from leworldmodel_judge.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = summarize(
        read_jsonl(args.prefixes),
        read_jsonl(args.baselines),
        read_jsonl(args.judge),
    )
    write_json(args.output, payload)
    print(f"wrote summary to {args.output}")


if __name__ == "__main__":
    main()
