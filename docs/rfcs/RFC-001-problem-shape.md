# RFC-001 — Problem shape

- **Status:** Accepted
- **Date:** 2026-04-23

## Decision
The project is about **trajectory judging in embodied RL**, not universal reward replacement.

## Why
That narrower shape is the only one likely to remain benchmarkable, honest, and portfolio-legible.

## Consequence
All contracts, baselines, and evaluation surfaces should be built around rollout prefixes and downstream task outcomes.
