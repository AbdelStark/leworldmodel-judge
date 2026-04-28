from __future__ import annotations

import argparse
import random
from pathlib import Path

from leworldmodel_judge.io import write_jsonl
from leworldmodel_judge.schema import RolloutStep
from leworldmodel_judge.tasks import LOCKED_TASKS, resolve_tasks

SYNTHETIC_FAMILIES = ("random", "expert", "weak", "doomed", "misleading")


def _maybe_float(payload: dict[str, float | str], source: dict, key: str) -> None:
    if key in source and source[key] is not None:
        payload[key] = float(source[key])


def _synthetic_family_state(
    policy_family: str, t: int, horizon: int
) -> tuple[float, float, float, float, float, float]:
    frac = (t + 1) / horizon
    if policy_family == "expert":
        progress = min(1.0, frac / 0.8)
        in_place = progress
        grasp = 1.0 if frac >= 0.3 else 0.0
        success = 1.0 if t == horizon - 1 else 0.0
        unscaled_reward = 0.2 + 0.8 * progress
        return progress, in_place, grasp, success, unscaled_reward, progress
    if policy_family == "weak":
        progress = min(0.45, frac / 1.5)
        in_place = progress
        grasp = 1.0 if frac >= 0.4 else 0.0
        success = 1.0 if t == horizon - 1 else 0.0
        unscaled_reward = 0.1 + 0.5 * progress
        return progress, in_place, grasp, success, unscaled_reward, progress
    if policy_family == "doomed":
        if frac <= 0.45:
            progress = min(0.85, frac / 0.55)
        else:
            regress = min(1.0, (frac - 0.45) / 0.55)
            progress = max(0.3, 0.85 - 0.5 * regress)
        in_place = max(0.0, progress - 0.1)
        grasp = 1.0 if frac >= 0.25 else 0.0
        success = 0.0
        unscaled_reward = 0.15 + 0.75 * max(progress, 0.6 if frac < 0.55 else progress)
        return progress, in_place, grasp, success, unscaled_reward, progress
    if policy_family == "misleading":
        progress = min(0.35, frac / 1.6)
        in_place = 0.75 if frac >= 0.25 else progress
        grasp = 1.0 if frac >= 0.2 else 0.0
        success = 0.0
        unscaled_reward = 0.3 + 0.4 * in_place
        return progress, in_place, grasp, success, unscaled_reward, progress

    # random family fallback
    progress = max(0.0, 1.0 - frac)
    in_place = max(0.0, 0.25 - 0.2 * frac)
    grasp = 0.0
    success = 0.0
    unscaled_reward = 0.05 * (1.0 - frac)
    return progress, in_place, grasp, success, unscaled_reward, progress


def collect_synthetic(task: str, episodes: int, policy_family: str = "random") -> list[dict]:
    rows = []
    rng = random.Random(7)
    for ep in range(episodes):
        horizon = 20
        success_episode = policy_family in ("expert", "weak")
        for t in range(horizon):
            progress, in_place, grasp, success_signal, unscaled_reward, obs_progress = (
                _synthetic_family_state(policy_family, t, horizon)
            )
            reward = (
                1.0
                if success_episode and t == horizon - 1
                else (-0.1 if policy_family == "random" and t > horizon // 2 else 0.0)
            )
            step = RolloutStep(
                episode_id=f"{task}-{policy_family}-ep-{ep}",
                task_id=task,
                timestep=t,
                episode_horizon=horizon,
                observation=[obs_progress, rng.random(), rng.random()],
                action=[rng.uniform(-1, 1) for _ in range(4)],
                reward=reward,
                done=(t == horizon - 1),
                success_label=success_episode,
                info={
                    "source": "synthetic",
                    "policy_family": policy_family,
                    "obj_to_target": max(0.0, 1.0 - progress),
                    "in_place_reward": in_place,
                    "grasp_success": grasp,
                    "grasp_reward": grasp,
                    "success": success_signal,
                    "unscaled_reward": max(0.0, unscaled_reward),
                },
            )
            rows.append(step.to_dict())
    return rows


def collect_metaworld(
    task: str, episodes: int, max_steps: int | None, seed: int, policy_family: str = "random"
) -> list[dict]:
    if task not in LOCKED_TASKS:
        raise SystemExit(f"task must be one of {sorted(LOCKED_TASKS)}")

    import metaworld  # local import so synthetic mode works without dependency
    from metaworld.policies import SawyerPickPlaceV3Policy, SawyerPushV3Policy, SawyerReachV3Policy

    policy_map = {
        "reach-v3": SawyerReachV3Policy,
        "push-v3": SawyerPushV3Policy,
        "pick-place-v3": SawyerPickPlaceV3Policy,
    }

    ml1 = metaworld.ML1(task, seed=seed)
    env_cls = ml1.train_classes[task]
    env = env_cls(render_mode=None)
    tasks = list(ml1.train_tasks)
    expert_policy = policy_map[task]()
    rng = random.Random(seed)

    rows: list[dict] = []
    for ep in range(episodes):
        task_spec = tasks[ep % len(tasks)]
        env.set_task(task_spec)
        obs, info = env.reset()
        horizon = min(int(getattr(env, "max_path_length", 500)), max_steps or 10**9)
        episode_steps: list[dict] = []
        success = False
        for t in range(horizon):
            expert_action = expert_policy.get_action(obs)
            if policy_family == "expert":
                action = expert_action
            elif policy_family == "weak":
                action = expert_action * 0.55
                action[:3] += 0.15 * (rng.random() - 0.5)
            elif policy_family == "doomed":
                if t < int(0.4 * horizon):
                    action = expert_action
                else:
                    action = -0.6 * expert_action
            elif policy_family == "misleading":
                if t < int(0.3 * horizon):
                    action = expert_action
                else:
                    action = 0.0 * expert_action
            else:
                action = env.action_space.sample()
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            success = success or bool(step_info.get("success", 0.0) >= 1.0)
            step_payload: dict[str, float | str] = {
                "source": "metaworld",
                "policy_family": policy_family,
            }
            for key in (
                "success",
                "near_object",
                "grasp_success",
                "grasp_reward",
                "in_place_reward",
                "obj_to_target",
            ):
                _maybe_float(step_payload, step_info, key)
            step_payload["unscaled_reward"] = float(step_info.get("unscaled_reward", reward))
            step = RolloutStep(
                episode_id=f"{task}-{policy_family}-ep-{ep}",
                task_id=task,
                timestep=t,
                episode_horizon=horizon,
                observation=obs.astype(float).tolist(),
                action=action.astype(float).tolist(),
                reward=float(reward),
                done=bool(terminated or truncated),
                success_label=False,
                info=step_payload,
            ).to_dict()
            episode_steps.append(step)
            obs = next_obs
            if terminated or truncated:
                break
        for step in episode_steps:
            step["success_label"] = success
        rows.extend(episode_steps)
    env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "metaworld"], default="synthetic")
    parser.add_argument("--task", required=True, help="single task, comma-separated subset, or all")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--policy-family",
        default="random",
        help="random, expert, weak, doomed, misleading, or comma-separated list",
    )
    args = parser.parse_args()

    tasks = resolve_tasks(args.task)
    policy_families = [part.strip() for part in args.policy_family.split(",") if part.strip()]
    rows: list[dict] = []
    for family_index, policy_family in enumerate(policy_families):
        if policy_family not in SYNTHETIC_FAMILIES:
            raise SystemExit(
                f"policy-family must be one of {SYNTHETIC_FAMILIES}; got {policy_family}"
            )
        for offset, task in enumerate(tasks):
            task_seed = args.seed + family_index * 100 + offset
            if args.source == "metaworld":
                rows.extend(
                    collect_metaworld(
                        task, args.episodes, args.max_steps, task_seed, policy_family=policy_family
                    )
                )
            else:
                rows.extend(collect_synthetic(task, args.episodes, policy_family=policy_family))

    write_jsonl(args.output, rows)
    print(
        f"wrote {len(rows)} rollout steps for tasks={tasks} policy_families={policy_families} to {Path(args.output)}"
    )


if __name__ == "__main__":
    main()
