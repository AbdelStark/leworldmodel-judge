# RFC-004 — Judge signal design

- **Status:** Accepted (amended 2026-07-10, see addendum)
- **Date:** 2026-04-23

## Decision
The first judge signal must be simple, benchmarkable, and explicitly compared against sparse reward.

## Preferred v1 signals
- latent surprise / residual
- rollout consistency
- disagreement-based uncertainty

## Rejected v1 pattern
Do not start with a giant opaque scalar called “judge score” with unclear semantics.

## Consequence
All judge outputs should be decomposed into named sub-signals where possible.

## Addendum (2026-07-10)

What v1 actually shipped, recorded so this RFC reflects reality rather than intent:

- The primary v1 judge is a **hand-weighted heuristic composite**
  (`judge_mode: composite_prefix_judge`), which this RFC did not anticipate as such, plus an
  optional latent-mismatch hybrid (`judge_mode: hybrid_prefix_latent_judge`).
- "Latent surprise / residual" materialized as mean-delta extrapolation mismatch over raw
  observation windows — an observation-space proxy, not a learned JEPA latent.
- "Rollout consistency" is at best approximated by the heuristic `implausibility_score`.
- **Disagreement-based uncertainty was never implemented.** The shipped `uncertainty_score` is a
  fixed formula over evidence spread, not ensemble disagreement. This is an open gap, tracked in
  [../roadmap.md](../roadmap.md).
- The decomposition requirement was fully honored: every judge row emits named sub-scores,
  evidence fields, and a `judge_mode` tag.
