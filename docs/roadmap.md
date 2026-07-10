# Roadmap: from heuristic proxy to JEPA-faithful judging

The repo tells a deliberately narrow story today: a prefix-level trajectory verifier with
verifiable-reward-style outputs, inspired by JEPA-lineage thinking. Keeping the JEPA anchor
without hand-waving requires an explicit path from the current heuristic score surface to a
faithful latent-prediction system. This is that path.

## Current honest state (updated 2026-07-10)

What exists:

- prefix slicing over rollout trajectories
- judge-side failure / on-track / implausibility / uncertainty scores
- family-aware benchmark summaries with calibration provenance
- replay/demo artifact generation
- evidence decomposition over hand-engineered prefix fields
- a latent cache (`lewm-judge latents`) and a hybrid judge mode
  (`judge_mode: hybrid_prefix_latent_judge`) that consumes `latent_mismatch_score` — but the
  "latents" are mean-pooled raw observation windows with a linear-extrapolation predictor, an
  observation-space proxy, not a learned representation; and the mismatch is scored against the
  realized post-cutoff observations, so the hybrid score is replay-time only
  ([method.md](method.md#cutoff-time-vs-replay-time-judging))

What does **not** exist yet:

- learned latent state encoder
- target latent prediction objective
- context/target masking in the JEPA sense
- latent dynamics predictor trained on trajectory continuation
- representation-level planning or latent imagination rollouts

So the repo remains **JEPA-anchored in thesis, not JEPA-faithful in implementation**. The hybrid
mode narrows the gap in wiring, not in representation.

## Upgrade goal

Move from a heuristic verifier over prefix features to a latent verifier over predicted future
trajectory state — without losing benchmark honesty, auditability, clean baseline comparisons, or
short demo legibility.

## Phase 1 — Representation bootstrapping (partially shipped)

Goal: replace the most brittle hand-coded prefix evidence with learned latent summaries while
keeping the rest of the pipeline stable.

Status of the original wedge items:

1. latent cache generation for current prefix records — **shipped** (`latents.py`,
   `lewm-judge latents`; the held-out artifact includes `latent-cache.jsonl`)
2. hybrid judge rows that append latent mismatch features — **shipped** (`hybrid_surprise` mode;
   the held-out headline run used it end-to-end)
3. comparison artifacts comparing the heuristic-only and hybrid judges on the same prefixes —
   **shipped** (2026-07-10: the held-out artifact carries `judge-composite.jsonl` /
   `summary-composite.json` next to the hybrid files; identical metrics, max `failure_score`
   delta +0.036, so there are no disagreement rows to show yet — the latent feature adds nothing
   measurable on this slice)
4. a fuller JEPA-style predictor — **open**

Remaining Phase 1 deliverables: an observation encoder that maps raw observation windows to
learned latent states; cached learned-latent datasets alongside the existing JSONL prefixes; an
evaluation showing whether latent summaries preserve family separability.

Rules: do not remove existing heuristic evidence columns yet; log latent extraction provenance
next to the benchmark outputs; compare latent-only, heuristic-only, and hybrid judge variants.

Success condition: a latent summary can separate at least some doomed vs recoverable prefixes
without relying entirely on manually engineered fields.

## Phase 2 — JEPA-style prediction target

Goal: introduce an actual predictive task over latent futures rather than a pure static
classifier surface.

Deliverables: a context/target window split for each prefix; a latent predictor that estimates
future target embeddings from current context; a prediction error or incompatibility score usable
as a judge feature.

Key question: does mismatch between predicted latent future and actual latent future surface
failure earlier than sparse reward and earlier than simple progress heuristics?

The upgrade that matters here is a predictor that does **not** peek. The current cache reads the
actual future by construction (`build_latent_cache` scores mismatch against the observations
after the cutoff) — fine for replay analysis, useless at the cutoff. The Phase 2 predictor is
trained offline against realized futures, but the judge-time feature must be computable from the
prefix alone, so that the hybrid mode stops being replay-only
([method.md](method.md#cutoff-time-vs-replay-time-judging)).

Rules: preserve the existing benchmark contract; report whether predictive latent mismatch
improves ranking quality or just adds noise; keep the current heuristic judge as a baseline, not
as hidden scaffolding.

Success condition: the predictive latent signal adds measurable benchmark value on at least one
hard family slice.

## Phase 3 — World-model-style judging surface

Goal: make the judge read like a lightweight world-model verifier rather than a decorated metric
combiner.

Deliverables: per-prefix predicted future latent state; actual future latent state; a divergence
/ compatibility score; a decomposition showing which latent transition region triggered the
warning; disagreement examples where the latent forecast moves before sparse reward.

Rules: do not claim planning unless a planning loop actually exists; do not claim imagination
unless future latent rollouts are truly generated and evaluated; preserve a demo artifact that
still makes sense in under 30 seconds.

Success condition: a viewer can see a prefix, a predicted future, an actual future, and a clean
reason why the judge became pessimistic.

## Phase 4 — Optional planning bridge

Only if the verifier path is already solid. Possible directions: use the latent verifier score
as a process reward candidate for policy improvement; rank alternative action continuations from
the same prefix; show that verifier-guided continuation ranking beats sparse reward timing on a
held-out slice.

Rule: do not call this planning just because a score exists. It only becomes planning when the
score changes action selection or continuation choice.

## Benchmark implications

Every phase must preserve these checks (see [benchmark.md](benchmark.md)):

- held-out vs in-slice threshold distinction
- family-aware reporting
- false positive cost visibility
- prefix-level ranking quality
- disagreement views against sparse reward and progress heuristics

If a more JEPA-like model becomes harder to explain, the artifact gets weaker, not stronger.

## Near-term work items (pre-Phase-2)

Independent of the model upgrade, the benchmark itself needs:

1. wider failure/recoverability label coverage beyond the current narrow late-prefix doomed cases
   (held-out run coverage is `0.444444`; synthetic run coverage is `0.055556`)
2. larger real held-out slices, so the 18-prefix held-out artifact is not the only held-out
   threshold story
3. rendered-state replay snapshots or frame strips in the demo bundle, not just scalar score
   drift — the open gap recorded in
   [RFC-006](rfcs/RFC-006-demo-surface.md)
4. shrinking the remaining expert-family false positives in `pick-place-v3` (the single false
   positive in the held-out run)
5. disagreement-based (ensemble) uncertainty, which was planned but never implemented — the open
   gap recorded in [RFC-004](rfcs/RFC-004-judge-signal-design.md)
6. a fresh capture collected after freezing the label rules: the push-v3 gates were last revised
   (2026-04-28) against the same capture the held-out artifact evaluates on, so the current label
   rules and the headline artifact are not independent (see the
   [benchmark caveats](benchmark.md#headline-held-out-family-split-real-meta-world))
7. failure/recoverability labels that are independent of the judge's evidence family (human
   annotation or environment-defined failure states), so judge-vs-label agreement stops being
   partly circular ([method.md](method.md#label-circularity))
8. a per-cutoff metric slice (and eventually a detection lead-time measurement), so "early" is
   measured rather than structural (see the scoping notes in
   [benchmark.md](benchmark.md#primary-metrics))

## Related docs

- [vision.md](vision.md) — why the honesty constraints exist
- [method.md](method.md) — what the current judge actually computes
- [benchmark.md](benchmark.md) — the contract every phase must keep
- [rfcs/RFC-007-v1-vs-v2-boundary.md](rfcs/RFC-007-v1-vs-v2-boundary.md) — the decision that made v1 ship with a lighter judge
