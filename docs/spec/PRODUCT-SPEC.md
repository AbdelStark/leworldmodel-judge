# PRODUCT-SPEC

## Product name
LeWorldModel Judge

## Product category
Research-shaped embodied RL artifact and credibility project.

## Product goal
Build a narrow embodied-RL system where a world-model-derived judge scores partial manipulation rollouts as:
- on-track
- failed / doomed
- implausible
- uncertain / low-confidence

The product should make it possible to evaluate whether this judge adds useful signal beyond sparse reward alone.

## Primary users

### User A — research evaluator
Someone who wants to know whether world-model-derived judging signals are useful for embodied RL trajectory evaluation.

### User B — technical reviewer / hiring manager
Someone trying to assess whether the builder has:
- real world-model taste
- embodied RL judgment
- benchmark discipline
- the ability to turn a fuzzy idea into a sharp artifact

### User C — future self / collaborator
Someone who may later extend the prototype into:
- offline RL data filtering
- process reward modeling
- trajectory triage for embodied agents

## User problem
Sparse reward and terminal success labels often fail to say enough about partial trajectories.
Users want earlier answers to questions like:
- is the rollout still likely to succeed?
- has it already become unrecoverable?
- does the state evolution look physically off-manifold?
- how uncertain should we be about any of those judgments?

## Product promise
Given a partial rollout prefix, the system will output a structured judgment signal and benchmark it against sparse reward and simple baselines.

## V1 scope
### Included
- one manipulation environment family
- one rollout-prefix judging task
- one dataset / rollout capture path
- one or more baseline signals
- one world-model-derived judging path
- one benchmark output table
- one plot / replay demo artifact

### Explicitly excluded
- multi-environment generalization claims
- broad robot policy training platform
- giant LeWorldModel reproduction effort
- replacing environment reward universally
- broad sim-to-real claims
- agent training loop integration as a required part of v1

## Core v1 success criteria
The repo is successful if it produces a benchmark result showing at least one of the following:
1. earlier failure detection than sparse reward alone
2. better ranking of partial trajectories than sparse reward alone
3. a useful implausibility / uncertainty signal that exposes cases sparse reward misses

## Acceptable weaker success mode
Even if the judge does not beat sparse reward cleanly, the repo is still valuable if it honestly reveals:
- where the judge fails
- what kinds of trajectories confuse it
- why plausibility and success diverge

## Failure criteria
The project fails if:
- it becomes mostly speculative prose
- it produces a score without a benchmark contract
- it cannot outperform or meaningfully complement trivial baselines
- it overclaims LeWorldModel faithfulness without evidence
- the demo looks polished but the evaluation story is weak

## Why this product matters
This product is interesting because it connects three active threads:
- world models and JEPAs
- process-level judging / verifier logic
- embodied RL trajectory evaluation

The point is not novelty theater.
The point is to create an artifact that is narrow enough to be benchmarked and sharp enough to communicate technical taste.
