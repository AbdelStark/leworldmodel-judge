# PRODUCT-SPEC

## Goal
Build a narrow embodied-RL artifact where a world-model-derived judge scores partial manipulation rollouts as on-track, failed, implausible, or uncertain.

## Primary user
A researcher / engineer evaluating whether a world-model-derived scoring function adds useful signal beyond sparse reward alone.

## v1 product surface
- one manipulation environment family
- rollout capture pipeline
- sparse-reward and heuristic baselines
- one judge score
- one benchmark table
- one replay-style demo artifact

## Core success condition
The repo produces a benchmark result showing that the judge adds useful information earlier than sparse reward alone.

## Failure condition
The project devolves into vague architecture talk with no benchmarkable claim.
