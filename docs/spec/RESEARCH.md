# RESEARCH

## Research thesis
The strongest framing is not “world model as universal judge.”
The strongest framing is **trajectory verifier / process reward model for embodied RL**.

## Why LeWorldModel
LeWorldModel is the strategic anchor because it naturally supports:
- JEPA framing
- latent predictive world modeling
- surprise / physical regularity stories
- a credible world-model-first narrative for AMI-facing positioning

## Why not default to Dreamer or TD-MPC2 as the headline
Dreamer and TD-MPC2 are safer engineering references, but they are weaker as the top-line narrative for this specific project because they read more like classic model-based RL baselines than a sharper JEPA/world-model taste signal.

## Honest positioning rule
The repo must clearly distinguish:
- the **conceptual anchor** (LeWorldModel / JEPA / surprise-driven world modeling)
- the **v1 implementation path** (which may be lighter)

That distinction is not a weakness. It is a correctness constraint.

## What the artifact must prove
At least one of:
- earlier failure detection than sparse reward
- better ranking of partial rollouts than sparse reward
- useful uncertainty or implausibility detection absent from sparse reward

## What the artifact must *not* pretend
- that plausibility is equivalent to success
- that a judge score is automatically a valid RL reward
- that the system is a faithful general reproduction of LeWorldModel
- that one benchmark proves universal value

## Most relevant inspiration buckets
- LeWorldModel and JEPA-style world models
- process reward models / verifier logic
- reward-model benchmarking discipline
- embodied trajectory evaluation and replay analysis
