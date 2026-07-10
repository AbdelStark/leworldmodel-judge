"""Rollout collection: synthetic policy-family generator and Meta-World capture.

Both sources emit the same :class:`~leworldmodel_judge.schema.RolloutStep` row
shape, differing only in ``info.source``. The policy-family registry
(:data:`POLICY_FAMILIES`) is canonical: collection, the demo's hard-family
pack and the docs all refer to these five names.

Determinism: synthetic collection is fully determined by ``seed`` (default 7,
which reproduces the checked-in synthetic artifacts byte-for-byte).
Meta-World capture derives one seed per (family, task) pair via
:func:`derive_task_seed`. The ``metaworld`` import is lazy so synthetic mode
works without the benchmark dependency installed.
"""

from __future__ import annotations

import random
from typing import Any

from .schema import RolloutStep
from .tasks import LOCKED_TASKS

POLICY_FAMILIES = ("random", "expert", "weak", "doomed", "misleading")

# Synthetic episodes always run exactly this many steps.
SYNTHETIC_HORIZON = 20


def derive_task_seed(seed: int, family_index: int, task_offset: int) -> int:
    """Derive the per-(family, task) seed used for Meta-World capture.

    Scheme: ``seed + family_index * 100 + task_offset``. The 100-wide stride
    keeps family cohorts disjoint for any run with fewer than 100 tasks (the
    locked benchmark has 3); it would collide beyond that, which is accepted
    and documented rather than silently rehashed — the scheme is part of run
    provenance.
    """
    return seed + family_index * 100 + task_offset


def _maybe_float(payload: dict[str, float | str], source: dict[str, Any], key: str) -> None:
    if key in source and source[key] is not None:
        payload[key] = float(source[key])


def _synthetic_family_state(
    policy_family: str, t: int, horizon: int
) -> tuple[float, float, float, float, float, float]:
    """Deterministic per-family evidence curves at step ``t`` of ``horizon``.

    Returns ``(progress, in_place, grasp, success, unscaled_reward,
    obs_progress)``. The families encode the benchmark's failure taxonomy:

    - ``expert`` — clean progress to success.
    - ``weak`` — slow progress, still reaches terminal success.
    - ``doomed`` — strong early progress, then regression; never succeeds.
    - ``misleading`` — flat progress masked by high in-place/grasp signals.
    - ``random`` (fallback) — decaying progress, no contact, no success.
    """
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


def collect_synthetic(
    task: str, episodes: int, policy_family: str = "random", seed: int = 7
) -> list[dict[str, Any]]:
    """Generate synthetic rollout step rows for one task and policy family.

    ``task`` is accepted as any string and only stamped onto the emitted rows:
    synthetic dynamics are task-independent, so unlike :func:`collect_metaworld`
    this function does not validate against ``LOCKED_TASKS``. Episode dynamics
    come from :func:`_synthetic_family_state`; the RNG only jitters observation
    dims 1-2 and the action vector, so labels and evidence are
    seed-independent. ``seed`` defaults to 7, which reproduces the checked-in
    synthetic benchmark artifacts byte-for-byte (historical runs hardcoded 7
    regardless of the CLI seed).
    """
    rows = []
    rng = random.Random(seed)
    for ep in range(episodes):
        horizon = SYNTHETIC_HORIZON
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
) -> list[dict[str, Any]]:
    """Capture rollouts from a Meta-World ML1 env with a family-degraded policy.

    Requires the ``metaworld`` extra (pinned ``metaworld==3.0.0`` for artifact
    reproducibility); the import is local so synthetic mode never needs it.
    Family degradations of the scripted expert policy:

    - ``expert`` — the scripted expert action unchanged.
    - ``weak`` — expert scaled by 0.55, plus noise on the first three action
      dims. Note (kept as-is for artifact stability): the noise adds the SAME
      scalar to all three dims per step, not independent per-dim noise.
    - ``doomed`` — expert until 40% of the horizon, then the negated (-0.6x)
      expert action.
    - ``misleading`` — expert until 30% of the horizon, then zero actions.
    - anything else — uniform random actions from the action space.

    ``success_label`` is backfilled retroactively onto every step of an
    episode once any step reported ``info.success >= 1.0``. Tasks outside
    ``LOCKED_TASKS`` raise ``ValueError`` (the CLI translates it to a clean
    one-line exit).
    """
    if task not in LOCKED_TASKS:
        raise ValueError(f"task must be one of {sorted(LOCKED_TASKS)}; got {task!r}")

    import metaworld  # local import so synthetic mode works without the dependency
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

    rows: list[dict[str, Any]] = []
    for ep in range(episodes):
        task_spec = tasks[ep % len(tasks)]
        env.set_task(task_spec)
        obs, info = env.reset()
        horizon = min(int(getattr(env, "max_path_length", 500)), max_steps or 10**9)
        episode_steps: list[dict[str, Any]] = []
        success = False
        for t in range(horizon):
            expert_action = expert_policy.get_action(obs)
            if policy_family == "expert":
                action = expert_action
            elif policy_family == "weak":
                action = expert_action * 0.55
                # Deliberately kept: one shared scalar offset across dims 0-2.
                action[:3] += 0.15 * (rng.random() - 0.5)
            elif policy_family == "doomed":
                action = expert_action if t < int(0.4 * horizon) else -0.6 * expert_action
            elif policy_family == "misleading":
                action = expert_action if t < int(0.3 * horizon) else 0.0 * expert_action
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
