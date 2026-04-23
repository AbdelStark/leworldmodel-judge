# LeWorldModel Judge

A JEPA-style world-model-derived trajectory judge for embodied RL.

## Thesis
This project studies whether a world-model-derived judge can score whether partial manipulation rollouts are still on track, already failed, or physically implausible before sparse reward or terminal success fully reveals the answer.

## Why this exists
Sparse reward is often too late and too weak to say much about partial trajectories.
This project explores whether a world-model-style latent judge can provide richer signals for:
- early failure detection
- trajectory ranking
- progress scoring
- recoverability estimation

## Strategic choice
The project is anchored on **LeWorldModel** as the conceptual JEPA family because it gives the cleanest AMI-facing story:
- JEPA-native
- world-model-first
- physical regularity / surprise framing
- fresh enough to feel current

Important reality:
- v1 does **not** require a full heavyweight LeWorldModel reproduction if that slows the artifact too much
- v1 only needs a world-model-derived judging surface that preserves the thesis honestly

## Current scope
v1 is intentionally narrow:
- one manipulation environment family
- one rollout-prefix judging task
- one benchmark table
- one demo surface

## Non-goals
- universal reward modeling
- giant world-model pretraining from scratch
- broad claims across all RL domains
- pretending a plausibility score is automatically a valid reward function

## Planned benchmark story
The main claim this repo should try to earn is:

> a world-model-derived trajectory judge can detect doomed manipulation trajectories earlier, or rank partial rollouts better, than sparse reward alone.

## Build order
1. docs/spec package
2. benchmark contract lock
3. baseline trajectory scorer
4. first judge signal
5. benchmark surface
6. demo surface

## Status
Private incubation repo. Docs-first.
