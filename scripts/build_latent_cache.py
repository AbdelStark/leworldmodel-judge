from __future__ import annotations

import argparse

from leworldmodel_judge.io import read_jsonl, write_jsonl
from leworldmodel_judge.latents import build_latent_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--prefixes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rollouts = read_jsonl(args.rollouts)
    prefixes = read_jsonl(args.prefixes)
    cache_rows = build_latent_cache(prefixes, rollouts)
    write_jsonl(args.output, cache_rows)
    print(f"wrote {len(cache_rows)} latent cache rows to {args.output}")


if __name__ == "__main__":
    main()
