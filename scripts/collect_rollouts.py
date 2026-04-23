from __future__ import annotations

import argparse
import random
from pathlib import Path

from leworldmodel_judge.io import write_jsonl
from leworldmodel_judge.schema import RolloutStep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['synthetic', 'metaworld'], default='synthetic')
    parser.add_argument('--task', required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if args.source != 'synthetic':
        raise SystemExit('Meta-World collection is not implemented yet. Use --source synthetic for now.')

    rows = []
    rng = random.Random(7)
    for ep in range(args.episodes):
        horizon = 20
        success = ep % 2 == 0
        for t in range(horizon):
            progress = (horizon - t) / horizon if success else (-t / horizon)
            reward = 1.0 if success and t == horizon - 1 else (-0.1 if not success and t > horizon // 2 else 0.0)
            step = RolloutStep(
                episode_id=f'{args.task}-ep-{ep}',
                task_id=args.task,
                timestep=t,
                episode_horizon=horizon,
                observation=[progress, rng.random(), rng.random()],
                action=[rng.uniform(-1, 1) for _ in range(4)],
                reward=reward,
                done=(t == horizon - 1),
                success_label=success,
                info={'source': 'synthetic'},
            )
            rows.append(step.to_dict())
    write_jsonl(args.output, rows)
    print(f'wrote {len(rows)} rollout steps to {Path(args.output)}')


if __name__ == '__main__':
    main()
