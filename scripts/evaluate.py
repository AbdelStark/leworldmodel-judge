from __future__ import annotations

import argparse

from leworldmodel_judge.evaluate import summarize
from leworldmodel_judge.io import read_jsonl, write_json


def _csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--calibration-families",
        default=None,
        help="Comma-separated policy families to use for threshold calibration. Defaults to the same evaluation slice.",
    )
    parser.add_argument(
        "--evaluation-families",
        default=None,
        help="Comma-separated policy families to include in the reported evaluation slice. Defaults to all families.",
    )
    args = parser.parse_args()

    payload = summarize(
        read_jsonl(args.prefixes),
        read_jsonl(args.baselines),
        read_jsonl(args.judge),
        calibration_families=_csv_arg(args.calibration_families),
        evaluation_families=_csv_arg(args.evaluation_families),
    )
    write_json(args.output, payload)
    print(f"wrote summary to {args.output}")


if __name__ == "__main__":
    main()
