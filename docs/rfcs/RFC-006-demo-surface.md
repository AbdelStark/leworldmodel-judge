# RFC-006 — Demo surface

- **Status:** Accepted (amended 2026-07-10, see addendum)
- **Date:** 2026-04-23

## Decision
The demo must show rollout evidence and score evolution together.

## Why
A trajectory judge only becomes legible when the viewer can connect frames, score changes, and eventual outcome.

## Consequence
The replay/demo artifact is not cosmetic. It is part of the evidence surface.

## Addendum (2026-07-10)

The shipped demo surface is comparison CSVs, score-timeline plots, per-episode score-replay
tables, and markdown disagreement packs. Score evolution and evidence decomposition are met.
**No rollout frames or video are rendered**, so the "connect frames, score changes, and eventual
outcome" bar is not yet met. This is an open gap, deferred rather than descoped: rendered-state
replay snapshots are a near-term work item in [../roadmap.md](../roadmap.md).
