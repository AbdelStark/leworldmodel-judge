# LeWorldModel Judge

A JEPA-style world-model-derived trajectory judge for embodied reinforcement learning.

## Core thesis

This project studies whether a world-model-derived judge can score whether **partial manipulation rollouts** are still on track, already failed, or physically implausible before sparse reward or terminal success fully reveals the answer.

The project is intentionally narrow.
It is **not** trying to build a universal reward model for all RL.
It is trying to answer one practical question:

> can a world-model-derived judging signal add useful information over sparse reward alone in manipulation rollouts?

## Why this exists

Sparse reward is often too late and too weak to say much about partial trajectories.
For many embodied tasks, the interesting information appears earlier:
- the grasp is already doomed
- the object is moving off the intended path
- the scene transition is physically implausible
- the agent is still technically alive but no longer recoverable

A judge that can detect those states earlier is interesting for:
- early failure detection
- trajectory ranking
- offline RL data filtering
- process reward modeling for embodied agents
- replay triage and policy debugging

## Strategic anchor: why LeWorldModel

This repo is anchored on **LeWorldModel** as the conceptual JEPA family because it gives the cleanest positioning for:
- JEPA-native taste
- world-model-first framing
- physical regularity / surprise language
- AMI-style world-model / physical reasoning alignment

Important honesty rule:
- v1 does **not** require a heavyweight faithful reproduction of LeWorldModel
- v1 only needs a judging artifact whose signals are honestly **world-model-derived** and consistent with the LeWorldModel thesis
- stronger architectural faithfulness belongs in later phases if the first artifact proves the benchmark story is real

## v1 product shape

V1 is intentionally small:
- one manipulation environment family
- one rollout-prefix judging task
- one benchmark table
- one replay/demo surface
- one world-model-derived judge implementation
- one sparse-reward baseline and one heuristic baseline

## Recommended environment

Default recommendation:
- **Meta-World** first

Fallback / alternative:
- **robosuite** if the setup cost remains manageable and the visual surface is materially better

## Main claim this repo should try to earn

> A world-model-derived trajectory judge can detect doomed manipulation trajectories earlier, or rank partial rollouts better, than sparse reward alone.

This is the exact claim the benchmark should live or die on.

## Non-goals

- universal reward modeling
- giant world-model pretraining from scratch
- broad claims across all RL domains
- pretending a plausibility score is automatically a valid reward function
- claiming that LeWorldModel-style surprise alone is a sufficient control signal

## Repo map

- `docs/spec/PRODUCT-SPEC.md` — product scope, users, success criteria, and non-goals
- `docs/spec/SYSTEM-SPEC.md` — architecture, data flow, judge pipeline, and output surfaces
- `docs/spec/RESEARCH.md` — research rationale and literature-aligned framing
- `docs/spec/MVP-STATUS.md` — current state and next milestone
- `docs/spec/SCHEDULE.md` — phased build order
- `docs/spec/EVAL-CONTRACT.md` — exact judging task and metrics
- `docs/spec/JUDGE-SIGNAL-CONTRACT.md` — required signals, semantics, and allowed v1 implementations
- `docs/spec/DATASET-CONTRACT.md` — rollout storage contract
- `docs/spec/DEMO-CONTRACT.md` — what the demo must show
- `docs/spec/RESULT-SCHEMA-CONTRACT.md` — benchmark output contract
- `docs/spec/CLI-COMMAND-SPEC.md` — script/CLI surface to keep implementation honest
- `docs/spec/FIRST-THREE-EXPERIMENTS.md` — pre-coding experiment pack
- `docs/spec/DEEPENING-PASS-1.md` — first tightening pass
- `docs/spec/DEEPENING-PASS-2.md` — second tightening pass
- `docs/spec/DEEPENING-PASS-3.md` — third tightening pass
- `docs/spec/IMPLEMENTATION-PLAN.md` — exact build order, file plan, and verification path
- `docs/rfcs/` — design decisions and v1/v2 boundaries

## Build order

1. lock the problem and evaluation contracts
2. lock environment and task boundaries
3. implement rollout capture and baselines
4. implement one judge signal
5. generate benchmark outputs
6. generate replay/demo outputs
7. only then consider broader LeWorldModel faithfulness or extension work

## Current status

Private incubation repo. Docs-first. No implementation claims yet.
