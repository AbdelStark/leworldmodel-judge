# DEEPENING-PASS-5

## What this pass is about

This pass closes the loop on the last honest gap from pass 4:
- actually checking in a **true held-out family artifact set**
- hardening `push-v3` so late failed prefixes are labeled more honestly
- making the demo surface replayable over time instead of only showing mean score curves

## Why this pass was necessary

After pass 4, the code could talk about held-out calibration, but the repo still did not contain a checked-in artifact where that story was true.

That left three problems:
1. the publishable threshold story was still hypothetical
2. `push-v3` was under-labeling real late failures, especially contact-without-transport cases
3. the demo still made reviewers trust aggregate timelines more than replayable per-episode score drift

## What changed in this pass

### 1. First true held-out artifact set
A new artifact folder now exists:
- `artifacts/hard-family-real-held-out-2026-04-28/`

It uses:
- calibration families: `weak`, `doomed`
- evaluation families: `expert`, `misleading`
- judge mode: `hybrid_surprise`

That means the threshold in this artifact is not an in-slice debug threshold.
It is a real family-held-out threshold with explicit provenance in `summary.json` and `report/family-report.md`.

### 2. Push-v3 late-prefix hardening
`push-v3` labeling was tightened so the benchmark stops pretending some obviously bad late prefixes are merely “at risk”.

The important new doomed cases are:
- late prefixes with almost no distance progress, still far from target, and weak transport
- late prefixes with strong contact signals but clear distance regret and no useful transport
- medium/late prefixes with weak contact, low reward evidence, and persistent distance failure

This especially matters for real hard-family smoke runs, where contact-like signals can be noisy enough to mask a dead trajectory if the label rules are too timid.

### 3. Replay reporting got less shallow
`render_demo.py` now emits:
- the existing comparison CSV
- the existing mean timeline plot
- the existing push-v3 disagreement pack
- a new per-episode replay table: `*-score-replay.csv`

That replay table is one row per episode with per-cutoff fields for:
- judge score
- baseline score
- sparse reward signal
- baseline-vs-judge gap
- first-difference deltas across cutoffs

The markdown artifact also now includes a `Score-over-time replays` section so a reviewer can inspect trajectory score drift directly.

## What this pass does **not** prove

It does **not** prove broad generalization.
It proves something narrower and still important:
- the held-out calibration path is real, not fictional
- `push-v3` is less benchmark-slippery than before
- replay artifacts are more audit-friendly than a single aggregate timeline

## What still remains after this pass
- widen failure/recoverability labeling beyond narrow late doomed states
- add richer replay surfaces with rendered-state strips or video-linked snapshots
- validate the held-out story on larger real slices, not just this smoke artifact
- replace more of the heuristic stack with a more faithful JEPA-native latent judging path
