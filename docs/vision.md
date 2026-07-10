# Vision

A world model as a judge: early, auditable verdicts on partial robot-manipulation rollouts.

> LeWorldModel Judge is a JEPA-anchored embodied RL project for prefix-level trajectory
> verification. It studies whether a world-model-derived, audit-friendly judge can produce
> earlier and more useful process-reward-style signals than sparse success alone.

The top-line framing is a **JEPA-anchored trajectory verifier with verifiable-reward-style
outputs** — not generic reward modeling, and not "world model does everything." Today's shipped
signal is a hand-weighted composite heuristic plus a linear observation-space proxy;
"world-model-derived" is the thesis and the roadmap, not the shipped mechanism
(see [method.md](method.md), [roadmap.md](roadmap.md)).

## The problem

Sparse reward and terminal success labels say almost nothing about a partial trajectory.
Practitioners want earlier answers to:

- is the rollout still likely to succeed?
- has it already become unrecoverable?
- does the state evolution look physically off-manifold?
- how uncertain should we be about any of those judgments?
- can the reward-like signal be inspected rather than blindly trusted?

Given a partial rollout prefix, this system outputs a structured judgment across four
categories — **on-track**, **failed / doomed**, **implausible**, **uncertain / low-confidence** —
preserves the raw evidence fields behind that judgment, and benchmarks it against sparse reward
and simple baselines.

## The research framing

The strongest framing is not "world model as universal judge." It is:

- **trajectory verifier / process reward model for embodied RL**
- **prefix-level failure detector and ranker**
- **audit-friendly reward decomposition surface**

## Why this is worth building

The project connects four active threads:

1. world models and JEPAs
2. process-level judging / verifier logic
3. embodied RL trajectory evaluation
4. verifiable-reward-style auditability for reward signals

The "verifiable rewards" angle claims no cryptographic proof. It claims that the judging signal is:

- decomposed into named sub-scores
- serialized to files
- replay-inspectable
- challengeable at the per-prefix level
- paired with explicit threshold provenance and benchmark slices

That makes the signal materially more auditable than a black-box scalar, even before any stronger
formal verification layer exists.

## Honest positioning rule

The repo must always distinguish three things:

- the **conceptual anchor**: LeWorldModel / JEPA / surprise-driven world modeling
- the **current implementation path**: a lighter composite prefix judge with explicit raw
  evidence, plus an observation-space latent proxy (see [method.md](method.md))
- the **next model path**: more faithful latent, world-model-derived judging
  (see [roadmap.md](roadmap.md))

That distinction is not a weakness. It is a correctness constraint. The shipped judge is
**JEPA-anchored in thesis, not JEPA-faithful in implementation**, and every scored row carries a
`judge_mode` field so no number can hide which judge produced it.

## Success criteria

The project succeeds if it produces a benchmark result showing at least one of:

1. earlier failure detection than sparse reward alone
2. better ranking of partial trajectories than sparse reward alone
3. a useful implausibility / uncertainty signal that exposes cases sparse reward misses
4. a replay-inspectable, decomposed judging surface that makes the score easier to challenge and
   audit than a raw scalar baseline

Acceptable weaker success mode: even if the judge does not beat sparse reward cleanly, the repo is
still valuable if it honestly reveals where the judge fails, what kinds of trajectories confuse
it, why plausibility and success diverge, and why a more faithful JEPA-native model pass is still
needed.

## Failure criteria

The project fails if:

- it becomes mostly speculative prose
- it produces a score without a benchmark contract
- it cannot outperform or meaningfully complement trivial baselines
- it overclaims LeWorldModel faithfulness without evidence
- the demo looks polished but the evaluation story is weak
- the reward/judge story is not inspectable enough to debug record-by-record

## Scope: what this is not

Explicitly out of scope for v1:

- multi-environment generalization claims
- a broad robot policy training platform
- a full LeWorldModel reproduction effort
- replacing environment reward universally
- broad sim-to-real claims
- agent training loop integration as a required deliverable
- cryptographic proof that the reward is "correct"

## Claim discipline

The benchmark claim always outranks the hype claim
([RFC-009](rfcs/RFC-009-public-positioning-and-claim-discipline.md)). The repo must not imply any
of the following unless directly supported by checked-in artifacts:

- that the current judge is a faithful LeWorldModel implementation
- that the current score is a valid RL reward for online training
- that synthetic separation automatically proves real embodied value
- that calibration chosen on the same slice is deployment-ready
- that plausibility is equivalent to success
- that one benchmark proves universal value

## The two-minute test

A skeptical reviewer should be able to grasp, in under two minutes:

1. what the judge is
2. what the benchmark slice is
3. why sparse reward is insufficient
4. where the current judge already beats trivial baselines
5. what remains heuristic / non-faithful today
6. what the next JEPA-native upgrade would replace

The win condition, in the reviewer's words: "I can see the benchmark, I can see the disagreement
with sparse reward, I can inspect why the judge made its call, and I can tell exactly what part is
JEPA-inspired versus still heuristic."

The point is not novelty theater. The point is an artifact that is narrow enough to be
benchmarked, sharp enough to communicate technical taste, and honest enough to show exactly what
is real today versus deferred to the next model pass.

## Related docs

- [method.md](method.md) — pipeline architecture, judge design, labeling, calibration
- [benchmark.md](benchmark.md) — the benchmark contract, results, reproduction commands
- [contracts.md](contracts.md) — every record schema and the CLI surface
- [roadmap.md](roadmap.md) — the JEPA-faithfulness upgrade path
- [rfcs/](rfcs/README.md) — the decision log
