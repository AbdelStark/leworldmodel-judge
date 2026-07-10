# RFC-007 — V1 vs V2 boundary

- **Status:** Accepted
- **Date:** 2026-04-23

## Decision
V1 may use a lighter latent judge if exact LeWorldModel adaptation is too heavy.

## Why
Shipping a real, benchmarked artifact is strategically more valuable than blocking on architectural purity.

## V2 candidate extensions
- stronger LeWorldModel faithfulness
- more visual JEPA grounding
- more tasks or environments
- training-loop integration

## Consequence
The repo must be explicit about what is and is not faithful to LeWorldModel in v1.
