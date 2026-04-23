# RFC-004 — Judge signal design

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
