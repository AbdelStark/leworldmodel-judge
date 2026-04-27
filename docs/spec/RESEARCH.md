# RESEARCH

## Research thesis
The strongest framing is not “world model as universal judge.”
The strongest framing is:
- **trajectory verifier / process reward model for embodied RL**
- **prefix-level failure detector and ranker**
- **audit-friendly reward decomposition surface**

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
- the **current implementation path** (a lighter composite prefix judge with explicit raw evidence)
- the **next model path** (more faithful latent/world-model-derived judging)

That distinction is not a weakness. It is a correctness constraint.

## What the artifact must prove
At least one of:
- earlier failure detection than sparse reward
- better ranking of partial rollouts than sparse reward
- useful uncertainty or implausibility detection absent from sparse reward
- more inspectable reward/judge evidence than a raw scalar baseline

## What the artifact must *not* pretend
- that plausibility is equivalent to success
- that a judge score is automatically a valid RL reward
- that the current system is a faithful general reproduction of LeWorldModel
- that one benchmark proves universal value
- that in-slice threshold tuning is already a publishable operating point

## Why the verifiable rewards angle is still legitimate
The useful version of “verifiable rewards” here is not cryptographic proof.
It is that the reward-like judging signal is:
- decomposed
- serialized to files
- replay-inspectable
- challengeable at the per-prefix level
- paired with explicit threshold provenance and benchmark slices

That makes the signal materially more auditable than a black-box scalar, even before any stronger formal verification layer exists.

## Most relevant inspiration buckets
- LeWorldModel and JEPA-style world models
- process reward models / verifier logic
- reward-model benchmarking discipline
- embodied trajectory evaluation and replay analysis
- calibration and selective prediction for decision thresholds

## Current strategic read
The repo is already beyond speculative idea stage because it has:
- a real benchmark scaffold
- hard-family stress cases
- calibration logic
- family-aware reporting
- real Meta-World smoke artifacts

The immediate research bottleneck is no longer “can this run at all?”
It is now:
- how honest and strong the benchmark slice is
- how well failure labels match recoverability reality
- how clearly the replay surface explains score movement
- when to upgrade from the current proxy judge to a more faithful JEPA-native path
