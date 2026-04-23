# EVAL-CONTRACT

## Main task
Given a partial manipulation rollout, predict one or more of:
- success likelihood
- failure likelihood
- recoverability
- implausibility

## Primary metrics
- early failure detection accuracy
- partial-trajectory ranking quality
- calibration / uncertainty notes if available

## Baselines
- terminal success only
- cumulative sparse reward
- simple heuristic progress proxy if available

## v1 rule
The judge must be compared against sparse reward. No judge-only victory lap.
