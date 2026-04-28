# LeWorldModel Judge

A JEPA-anchored embodied RL showcase for **prefix-level trajectory verification** and **verifiable-reward-style judging**.

## Core thesis

This project studies whether a world-model-derived judge can score whether **partial manipulation rollouts** are still on track, already failed, or physically implausible before sparse reward or terminal success fully reveals the answer.

The project is intentionally narrow.
It is **not** trying to build a universal reward model for all RL.
It is trying to answer one practical question:

> can a world-model-derived, audit-friendly judging signal add useful information over sparse reward alone in manipulation rollouts?

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
- v1 does **not** claim a heavyweight faithful reproduction of LeWorldModel
- v1 currently ships a lighter **composite prefix judge** with explicit raw sub-scores and calibration artifacts
- stronger architectural faithfulness belongs in later phases if the benchmark and replay surface prove the story is real

## Verifiable rewards angle

This repo uses **verifiable rewards** in the practical showcase sense, not the cryptographic-marketing sense.

What that means here:
- the judge output is file-based and replay-inspectable
- each scored prefix carries decomposed evidence fields, not just one magic scalar
- benchmark summaries preserve threshold provenance and family slices
- another reviewer should be able to challenge a score by looking at the rollout, the prefix record, and the raw sub-scores

What it does **not** mean yet:
- cryptographic proof of reward correctness
- a production-safe online RL reward function
- a universal verifier for all embodied tasks

## Main claim this repo is trying to earn

> A world-model-derived trajectory judge can detect doomed manipulation trajectories earlier, or rank partial rollouts better, than sparse reward alone — while producing more inspectable process-reward-style evidence than a raw scalar baseline.

That is the exact claim the benchmark should live or die on.

## V1 benchmark slice

### Environment family
- **Meta-World**

### Locked tasks
- `reach-v3`
- `push-v3`
- `pick-place-v3`

### Prefix cutoffs
- `0.25`
- `0.50`
- `0.75`

### Policy families used to stress the benchmark
- `expert`
- `weak`
- `doomed`
- `misleading`
- `random`

## Current system shape

The current repo already contains a working end-to-end benchmark path:
- rollout capture and normalization
- prefix building with Meta-World-derived signals
- latent-cache generation for prefix/future comparisons
- baseline scorers
- a composite prefix judge plus a hybrid latent-augmented judge
- summary metrics with threshold recommendation
- held-out-family calibration provenance support in the evaluator
- family-aware markdown and plot reports
- synthetic hard-family benchmark artifacts
- real Meta-World smoke artifacts

## Modern local workflow

This repo now works as a modern `uv` project.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run python scripts/build_prefixes.py --input artifacts/.../rollouts.jsonl --output artifacts/.../prefixes.jsonl
uv run python scripts/build_latent_cache.py --rollouts artifacts/.../rollouts.jsonl --prefixes artifacts/.../prefixes.jsonl --output artifacts/.../latent-cache.jsonl
uv run python scripts/run_judge.py --input artifacts/.../prefixes.jsonl --latent-cache artifacts/.../latent-cache.jsonl --mode hybrid_surprise --output artifacts/.../judge-hybrid.jsonl
uv run python scripts/evaluate.py --prefixes artifacts/.../prefixes.jsonl --baselines artifacts/.../baselines.jsonl --judge artifacts/.../judge-hybrid.jsonl --output artifacts/.../summary.json --calibration-families weak,doomed --evaluation-families expert,misleading
uv run python scripts/render_demo.py --prefixes artifacts/.../prefixes.jsonl --baselines artifacts/.../baselines.jsonl --judge artifacts/.../judge-hybrid.jsonl --families expert,misleading --output artifacts/.../demo-artifact.md
```

That keeps dependency state explicit, gives us a lockfile, and makes the benchmark path reproducible.

Current judge output includes:
- `on_track_score`
- `failure_score`
- `implausibility_score`
- `uncertainty_score`
- decomposed evidence fields like progress, distance progress, in-place, near-object, grasp, reward, and stall evidence

## Current honest read

What is already real:
- the pipeline runs end-to-end
- the hard-family synthetic slice gives clean separation between judge and weak baselines
- the real hard-family smoke now has materially better false-positive behavior after calibration
- family-aware reports make it obvious where the judge wins and where it still misses
- the evaluator now records calibration provenance, held-out family splits, and average-precision views instead of pretending every threshold is the same kind of evidence

What is still not solved:
- the current judge is still a lighter heuristic/composite proxy, not a faithful JEPA-native latent verifier
- the repo now has one checked-in **held-out family split** artifact set, but it is still a small smoke slice rather than broad held-out coverage
- `push-v3` late-prefix failure labeling is materially less slippery than before, but broader recoverability labeling is still too narrow
- the current judge is still strongest on the narrow hard-family smoke slice; broader held-out slices still need more coverage before the story is robust

## Non-goals

- universal reward modeling
- giant world-model pretraining from scratch
- broad claims across all RL domains
- pretending a plausibility score is automatically a valid reward function
- claiming that LeWorldModel-style surprise alone is a sufficient control signal
- hiding heuristic components behind JEPA branding

## Immediate next moves

1. widen failure and recoverability labeling beyond the current late-prefix doomed cases
2. run larger held-out real slices so the threshold story survives beyond one smoke artifact
3. make the replay surface richer with frame-level or rendered-state snapshots, not just scalar score drift
4. only then push toward a more faithful JEPA-native judge implementation

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
- `docs/spec/DEEPENING-PASS-1.md` — thesis tightening pass
- `docs/spec/DEEPENING-PASS-2.md` — benchmark tightening pass
- `docs/spec/DEEPENING-PASS-3.md` — showcase tightening pass
- `docs/spec/DEEPENING-PASS-4.md` — evaluation-honesty pass for held-out calibration provenance and benchmark reporting
- `docs/spec/DEEPENING-PASS-5.md` — held-out artifact execution pass, push-v3 hardening, and score-replay reporting
- `docs/spec/IMPLEMENTATION-PLAN.md` — exact build order, file plan, and verification path
- `docs/rfcs/` — design decisions and v1/v2 boundaries

## Build order

1. keep the problem and evaluation contracts stable
2. harden the benchmark slice and labels
3. strengthen held-out calibration and task-specific failure coverage
4. improve replay/demo legibility
5. only then invest in broader LeWorldModel faithfulness or extension work
