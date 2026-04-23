# JUDGE-SIGNAL-CONTRACT

## Required v1 outputs
- `on_track_score`
- `failure_score`
- `implausibility_score`
- optional `uncertainty_score`

## Semantics
### on_track_score
Estimated likelihood that the rollout prefix is still consistent with successful completion.

### failure_score
Estimated likelihood that the trajectory is already effectively doomed or unrecoverable.

### implausibility_score
Estimated degree to which the observed prefix or predicted continuation appears off-manifold, physically inconsistent, or surprising under the world-model-derived representation.

### uncertainty_score
How much the system trusts its own judgment.

## Acceptable first implementations
- latent surprise score
- rollout consistency score
- predictive residual score
- ensemble disagreement score

## Guardrails
1. A visually plausible score is not enough.
2. The score must be evaluated against task outcomes.
3. The score must be defined precisely enough to benchmark.
4. If uncertainty is emitted, it must not be confused with failure.

## Output format
Each scored prefix should emit a structured record with all raw sub-scores before any final aggregation.
