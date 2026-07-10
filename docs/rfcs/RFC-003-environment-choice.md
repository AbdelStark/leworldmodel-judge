# RFC-003 — Environment choice

- **Status:** Accepted
- **Date:** 2026-04-23

## Decision
Default to **Meta-World** first.
Use **robosuite** only if its extra visual value clearly outweighs setup cost.

## Why
Meta-World is the better v1 tradeoff for:
- benchmark discipline
- engineering tractability
- multi-task manipulation framing without giant simulator overhead

## Consequence
All early contracts should be written so they can support Meta-World first.
