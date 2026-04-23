# EVAL-CONTRACT

## V1 benchmark slice

### Environment family
**Meta-World**

### Locked v1 tasks
- `reach-v3`
- `push-v3`
- `pick-place-v3`

## Main task
Given a **partial manipulation rollout prefix**, predict one or more of:
- success likelihood
- failure likelihood
- recoverability
- implausibility
- optional uncertainty

## Why these tasks

### reach-v3
- simplest contact-light manipulation task
- good for debugging prefix scoring and progress proxies

### push-v3
- introduces longer-horizon object interaction and recoverability failure
- useful for showing that sparse reward often reveals failure too late

### pick-place-v3
- richest of the v1 trio
- strongest visual demo surface
- most likely to expose partial progress vs eventual failure mismatch

This trio is small enough to ship and diverse enough to prevent the artifact from looking like a one-task toy.

## Prefix cutoffs
The v1 benchmark will evaluate prefixes at:
- `0.25`
- `0.50`
- `0.75`

These are fractions of the episode horizon.

## Ground-truth references
Ground truth should come from environment-truth signals where possible:
- episode success flag
- final task completion outcome
- documented heuristic unrecoverability labels for some tasks

## Derived labels
Each prefix should carry:
- `final_success_label`
- `prefix_failure_label`
- `prefix_recoverability_label`

### Prefix failure label
A binary label indicating whether the trajectory is already effectively doomed at that prefix.

### Prefix recoverability label
A categorical label:
- `recoverable`
- `at_risk`
- `doomed`

For the first pass, this may be derived using simple task heuristics documented alongside the dataset.

## Required baselines
- terminal success only
- cumulative sparse reward
- heuristic progress proxy

## Primary metrics
- early failure detection accuracy
- AUROC / AUPRC for doomed vs recoverable prefixes
- partial-trajectory ranking quality (Spearman / Kendall / pairwise accuracy)
- calibration notes if uncertainty is emitted

## Benchmark questions
1. Can the judge detect doomed trajectories earlier than sparse reward alone?
2. Can the judge rank partial trajectories better than sparse reward alone?
3. Does the judge provide useful uncertainty or implausibility information on ambiguous prefixes?

## V1 rule
The judge must be compared against sparse reward. No judge-only victory lap.
