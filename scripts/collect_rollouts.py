from __future__ import annotations

import argparse
import random
from pathlib import Path

from leworldmodel_judge.io import write_jsonl
from leworldmodel_judge.schema import RolloutStep


LOCKED_TASKS = {'reach-v3', 'push-v3', 'pick-place-v3'}


def collect_synthetic(task: str, episodes: int) -> list[dict]:
    rows = []
    rng = random.Random(7)
    for ep in range(episodes):
        horizon = 20
        success = ep % 2 == 0
        for t in range(horizon):
            progress = (horizon - t) / horizon if success else (-t / horizon)
            reward = 1.0 if success and t == horizon - 1 else (-0.1 if not success and t > horizon // 2 else 0.0)
            step = RolloutStep(
                episode_id=f'{task}-ep-{ep}',
                task_id=task,
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
    return rows


def collect_metaworld(task: str, episodes: int, max_steps: int | None, seed: int) -> list[dict]:
    if task not in LOCKED_TASKS:
        raise SystemExit(f'task must be one of {sorted(LOCKED_TASKS)}')

    import metaworld  # local import so synthetic mode works without dependency

    ml1 = metaworld.ML1(task, seed=seed)
    env_cls = ml1.train_classes[task]
    env = env_cls(render_mode=None)
    tasks = list(ml1.train_tasks)

    rows: list[dict] = []
    for ep in range(episodes):
        task_spec = tasks[ep % len(tasks)]
        env.set_task(task_spec)
        obs, info = env.reset()
        horizon = min(int(getattr(env, 'max_path_length', 500)), max_steps or 10**9)
        episode_steps: list[dict] = []
        success = False
        for t in range(horizon):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            success = success or bool(step_info.get('success', 0.0) >= 1.0)
            step = RolloutStep(
                episode_id=f'{task}-ep-{ep}',
                task_id=task,
                timestep=t,
                episode_horizon=horizon,
                observation=obs.astype(float).tolist(),
                action=action.astype(float).tolist(),
                reward=float(reward),
                done=bool(terminated or truncated),
                success_label=False,  # filled after the episode finishes
                info={'source': 'metaworld', 'raw_success': float(step_info.get('success', 0.0))},
            ).to_dict()
            episode_steps.append(step)
            obs = next_obs
            if terminated or truncated:
                break
        for step in episode_steps:
            step['success_label'] = success
        rows.extend(episode_steps)
    env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['synthetic', 'metaworld'], default='synthetic')
    parser.add_argument('--task', required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    if args.source == 'metaworld':
        rows = collect_metaworld(args.task, args.episodes, args.max_steps, args.seed)
    else:
        rows = collect_synthetic(args.task, args.episodes)

    write_jsonl(args.output, rows)
    print(f'wrote {len(rows)} rollout steps to {Path(args.output)}')


if __name__ == '__main__':
    main()
