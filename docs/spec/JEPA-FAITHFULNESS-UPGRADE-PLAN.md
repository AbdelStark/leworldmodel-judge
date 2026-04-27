# JEPA-FAITHFULNESS-UPGRADE-PLAN

## Why this doc exists
Right now the repo tells a tighter and more honest story: this is a prefix-level trajectory verifier / verifiable-reward-style judging artifact inspired by JEPA-lineage thinking. That is good. But if we want to keep the JEPA anchor without hand-waving, we need an explicit path from today's heuristic score surface to a more faithful latent-prediction system.

This doc is that path.

## Current honest state
What exists today:
- prefix slicing over rollout trajectories
- judge-side failure / on-track / implausibility / uncertainty scores
- family-aware benchmark summaries
- replay/demo artifact generation
- evidence decomposition over hand-engineered prefix fields

What does **not** exist yet:
- learned latent state encoder
- target latent prediction objective
- context/target masking in the JEPA sense
- latent dynamics predictor trained on trajectory continuation
- representation-level planning or latent imagination rollouts

So the current repo is **JEPA-anchored in thesis**, not JEPA-faithful in implementation.

## Upgrade goal
Move from:
- heuristic verifier over prefix features

to:
- latent verifier over predicted future trajectory state

without losing:
- benchmark honesty
- auditability
- clean baseline comparisons
- short demo legibility

## Phase 1 — Representation bootstrapping
Goal: replace the most brittle hand-coded prefix evidence with learned latent summaries while keeping the rest of the pipeline stable.

Deliverables:
- observation encoder that maps raw observation windows to latent states
- cached latent datasets alongside existing JSONL prefixes
- evaluation showing whether latent summaries preserve family separability

Rules:
- do not remove existing heuristic evidence columns yet
- log latent extraction provenance next to the benchmark outputs
- compare latent-only, heuristic-only, and hybrid judge variants

Success condition:
A latent summary can separate at least some doomed vs recoverable prefixes without relying entirely on manually engineered fields.

## Phase 2 — JEPA-style prediction target
Goal: introduce an actual predictive task over latent futures rather than a pure static classifier surface.

Deliverables:
- context window / target window split for each prefix
- latent predictor that estimates future target embeddings from current context
- prediction error or incompatibility score usable as a judge feature

Key question:
Does mismatch between predicted latent future and actual latent future surface failure earlier than sparse reward and earlier than simple progress heuristics?

Rules:
- preserve the existing benchmark contract
- report whether predictive latent mismatch improves ranking quality or just adds noise
- keep the current heuristic judge as a baseline, not as hidden scaffolding

Success condition:
The predictive latent signal adds measurable benchmark value on at least one hard family slice.

## Phase 3 — World-model-style judging surface
Goal: make the judge read like a lightweight world-model verifier rather than a decorated metric combiner.

Deliverables:
- per-prefix predicted future latent state
- actual future latent state
- divergence / compatibility score
- decomposition showing which latent transition region triggered the warning
- disagreement examples where latent forecast moves before sparse reward

Rules:
- do not claim planning unless a planning loop actually exists
- do not claim imagination unless future latent rollouts are truly generated and evaluated
- preserve a demo artifact that still makes sense in under 30 seconds

Success condition:
A viewer can see a prefix, a predicted future, an actual future, and a clean reason why the judge became pessimistic.

## Phase 4 — Optional planning bridge
This is optional and should happen only if the verifier path is already solid.

Goal:
Bridge the verifier into a planning-adjacent artifact.

Possible directions:
- use latent verifier score as a process reward candidate for policy improvement
- rank alternative action continuations from the same prefix
- show that verifier-guided continuation ranking beats sparse reward timing on a held-out slice

Anti-bullshit rule:
Do not call this planning just because a score exists. It only becomes planning when the score changes action selection or continuation choice.

## Benchmark implications
Every phase above must preserve these checks:
- held-out vs in-slice threshold distinction
- family-aware reporting
- false positive cost visibility
- prefix-level ranking quality
- disagreement views against sparse reward and progress heuristics

If a more JEPA-like model becomes harder to explain, the artifact gets weaker, not stronger.

## What to build next
The best immediate next implementation wedge is:
1. latent cache generation for current prefix records
2. hybrid judge rows that append latent mismatch features
3. comparison artifacts showing heuristic-only vs hybrid disagreement rows
4. only then a fuller JEPA-style predictor

That sequence keeps the repo honest while still moving toward the real thesis.
