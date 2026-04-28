# DEEPENING-PASS-4

## Pass focus
Tighten the evaluation honesty layer so the repo stops treating every threshold as equal evidence.

This pass is about one thing:
- separating **debug-friendly in-slice tuning** from **publishable held-out calibration provenance**

## Why this pass was necessary
The repo already had a decent benchmark loop and better family-aware reporting, but there was still one soft spot:
- a threshold chosen on the same slice could look cleaner than it really is
- the report language warned about that, but the evaluator itself did not make the distinction first-class

That is not enough for a world-model judge artifact that wants to be taken seriously.

If the project is claiming "verifiable-reward-style" evidence, threshold provenance is part of the evidence.

## What got tightened in this pass
- Added explicit evaluator support for **held-out family calibration**.
- Added calibration provenance fields so reports can say:
  - which families calibrated the threshold
  - which families were evaluated with it
  - how many labeled failure / non-failure prefixes each cohort had
- Added **average precision** reporting alongside pairwise/AUROC-style ranking views.
- Tightened the family-aware report so it can no longer talk vaguely about threshold quality.
- Updated the CLI contract so the evaluation surface admits split-aware calibration instead of only one undifferentiated path.

## New honesty rule
From this point on, benchmark artifacts should clearly fall into one of three buckets:
1. **fixed** threshold
2. **in-slice** threshold
3. **held-out** threshold

Anything else is muddy and should be treated as a documentation bug.

## What this changes strategically
This does **not** magically make the current judge more JEPA-faithful.
It does make the benchmark story more serious.

That matters because the repo's next credibility jump is not just better raw scores.
It is proving that:
- the score survives a cleaner evaluation boundary
- the threshold is not just a same-slice convenience hack
- the report says exactly what kind of evidence the reader is looking at

## What still remains after this pass
- checked-in artifacts still need at least one true held-out rerun
- `push-v3` still needs hardening
- failure/recoverability labels are still too narrow
- score-over-time replay visuals still need another pass to become great
- the model path is still proxy-first, not faithful LeWorldModel / JEPA-native judging

## Main conclusion
The repo is stronger now because the evaluation layer got less slippery.

The next serious milestone is not "more charts."
It is:

> one benchmark artifact where the threshold comes from a clearly named held-out family split, the report says that plainly, and the judge still looks worth caring about.
