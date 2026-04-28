from __future__ import annotations

import argparse

from leworldmodel_judge.io import read_jsonl, write_jsonl
from leworldmodel_judge.judge import heuristic_surprise_score, hybrid_surprise_score


def _key(row: dict) -> tuple[str, str, float]:
    return (row["task_id"], row["episode_id"], float(row["prefix_fraction"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-cache")
    parser.add_argument(
        "--mode",
        choices=["heuristic_surprise", "hybrid_surprise", "dummy"],
        default="heuristic_surprise",
    )
    args = parser.parse_args()

    prefixes = read_jsonl(args.input)
    latent_cache_map = {}
    if args.latent_cache:
        latent_cache_map = {_key(row): row for row in read_jsonl(args.latent_cache)}
    if args.mode == "dummy":
        rows = []
        for prefix in prefixes:
            rows.append(
                {
                    "episode_id": prefix["episode_id"],
                    "task_id": prefix["task_id"],
                    "prefix_fraction": prefix["prefix_fraction"],
                    "on_track_score": 0.0,
                    "failure_score": 0.0,
                    "implausibility_score": 0.0,
                    "uncertainty_score": 1.0,
                    "judge_mode": "dummy",
                }
            )
    elif args.mode == "hybrid_surprise":
        rows = [
            hybrid_surprise_score(prefix, latent_cache_map.get(_key(prefix))) for prefix in prefixes
        ]
    else:
        rows = [heuristic_surprise_score(prefix) for prefix in prefixes]
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} judge rows to {args.output}")


if __name__ == "__main__":
    main()
