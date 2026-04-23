# EVAL-CONTRACT

## Main task
Given a partial manipulation rollout, predict one or more of:
- success likelihood
- failure likelihood
- recoverability
- implausibility
- optional uncertainty

## Primary benchmark questions
1. Can the judge detect doomed trajectories earlier than sparse reward alone?
2. Can the judge rank partial trajectories better than sparse reward alone?
3. Does the judge provide useful uncertainty or implausibility information on ambiguous prefixes?

## Ground-truth references
Ground truth should come from environment-truth signals where possible:
- episode success flag
- final task completion outcome
- manually defined unrecoverability heuristics if explicitly documented

## Required baselines
- terminal success only
- cumulative sparse reward
- heuristic progress proxy

## Primary metrics
- early failure detection accuracy
- AUROC / AUPRC for doomed vs salvageable prefixes
- partial-trajectory ranking metric (Spearman / Kendall / pairwise accuracy)
- calibration notes if uncertainty is emitted

## Prefix slicing rule
The evaluation must support multiple prefix cutoffs, such as:
- 25%
- 50%
- 75%

This is important because the whole point is judging before the episode is over.

## v1 rule
The judge must be compared against sparse reward. No judge-only victory lap.
