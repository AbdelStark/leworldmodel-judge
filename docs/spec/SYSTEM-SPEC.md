# SYSTEM-SPEC

## System objective
Given stored or live rollout prefixes from a manipulation environment, produce structured judging outputs and benchmark them against sparse reward baselines.

## High-level architecture

```text
environment / dataset
        │
        ▼
rollout capture + normalization
        │
        ▼
prefix builder
        │
        ├──────────────► baseline scorers
        │                 - sparse reward
        │                 - terminal-success proxy
        │                 - heuristic progress proxy
        │
        └──────────────► world-model-derived judge
                          - latent / surprise score
                          - progress score
                          - failure score
                          - uncertainty score
        │
        ▼
evaluation layer
        │
        ├──────────────► benchmark tables / JSON / CSV
        └──────────────► replay/demo artifacts
```

## Core modules

### 1. Rollout capture layer
Responsibilities:
- collect or ingest trajectories
- normalize records into one canonical schema
- persist task ID, timestep, success label, rewards, actions, and observations

### 2. Prefix builder
Responsibilities:
- slice partial trajectories at configured prefix lengths
- attach labels for downstream evaluation
- support benchmark generation at multiple cutoff points

### 3. Baseline scoring layer
Responsibilities:
- compute trivial and semi-trivial signals
- prevent the judge from being evaluated in a vacuum

Required v1 baselines:
- sparse reward aggregate
- terminal success only
- simple heuristic progress proxy

### 4. World-model-derived judge layer
Responsibilities:
- map a rollout prefix to one or more structured judgment scores
- expose uncertainty where possible
- remain swappable across different implementations

Allowed v1 implementations:
- latent prediction residual score
- surprise / off-manifold score
- ensemble disagreement score
- compact latent dynamics consistency score

### 5. Evaluation layer
Responsibilities:
- compare judge signals against baselines
- compute ranking and early-detection metrics
- emit reproducible result files

### 6. Demo / replay layer
Responsibilities:
- display rollout prefix frames or replay snapshots
- show score trajectories over time
- make the claim legible in one glance

## Data flow details

### Input record
Each rollout step should contain at minimum:
- `episode_id`
- `task_id`
- `timestep`
- `observation`
- `action`
- `reward`
- `done`
- `success_label` (episode-level ground truth)

### Prefix record
Each prefix should contain:
- full episode reference
- prefix cutoff index
- derived label(s)
- baseline signal values
- judge signal values

## Output surfaces

### Evaluation outputs
- `results/*.csv`
- `results/*.json`
- one benchmark summary table
- one main plot

### Demo outputs
- replay snippets
- score timeline plots
- optional static HTML or notebook-style artifact

## Key invariants
1. The judging task must be defined before implementation drift begins.
2. All judge outputs must be benchmarked against sparse reward.
3. Baseline signals and judge signals must remain explicitly separated.
4. V1 claims must stay narrower than the architecture’s future potential.
5. The benchmark contract must survive implementation swaps.

## Main architectural risk
The largest architectural risk is confusing the conceptual anchor with the v1 implementation.

To avoid that:
- LeWorldModel remains the strategic thesis anchor
- the judge interface remains implementation-agnostic
- v1 can ship with a lighter world-model-derived judge if needed
- the benchmark contract stays stable regardless of implementation swap
