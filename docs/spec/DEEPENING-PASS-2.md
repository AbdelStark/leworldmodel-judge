# DEEPENING-PASS-2

## Pass focus
Tighten the benchmark so the result could eventually survive skeptical technical review, not just look good in a demo.

## What got tightened in this pass
- Made the **benchmark contract** the real product surface.
  - The main unit is not the episode.
  - The main unit is the **prefix record** with attached labels, baselines, judge outputs, and raw evidence fields.
- Elevated the **family-based stress regime** from implementation detail to first-class benchmark design.
  - `expert`
  - `weak`
  - `doomed`
  - `misleading`
- Clarified what the current synthetic and real hard-family runs already proved:
  - the sparse-success baseline is too blunt
  - the simple progress baseline can rank hard families badly
  - the calibrated judge can separate families much better
- Named the current scientific weakness explicitly:
  - threshold calibration is still **in-slice**
  - that is acceptable for debugging
  - it is **not** the publishable end-state
- Tightened the next benchmark bar:
  - held-out calibration split
  - richer failure-label coverage
  - per-task failure taxonomy, especially for `push-v3`
  - score-over-time replay traces that explain why the judge flips when it flips

## What is now non-negotiable
A serious result from this repo must have all of the following:
1. **prefix-level labels** that are inspectable and task-aware
2. **baseline vs judge comparison** on the same records
3. **family-aware slices** instead of a single aggregate headline
4. **calibration provenance** stating whether thresholds are fixed, in-slice, or held-out
5. **artifact files** that let someone inspect rollouts, prefixes, judge outputs, and summary metrics without rerunning the full stack

## Benchmark interpretation rule
The project does **not** earn credibility from one large topline number.
It earns credibility from showing all three at once:
- where the judge wins
- where it loses
- why the decomposition of evidence makes those outcomes believable

## New benchmark boundary
The repo should now treat these as different maturity levels:

### Debug-grade
- synthetic hard-family benchmark
- in-slice threshold recommendation
- useful for shaping the judge and labeler

### Showcase-grade
- real hard-family smoke with family report
- honest writeup of weak tasks and misses
- replay-visible score trajectories

### Publishable-grade
- held-out calibration
- wider failure taxonomy
- task-level ablations
- stronger claim that the operating point generalizes beyond the slice used to choose it

## Main conclusion
The benchmark is now strong enough to support a serious showcase, but not yet strong enough to pretend it is a finished scientific result.

That distinction needs to stay explicit everywhere.