# DEEPENING-PASS-3

## Pass focus
Tighten the showcase narrative so the demo, the benchmark, and the JEPA/verifiable-reward story all point at the same claim.

## What got tightened in this pass
- Clarified that the final artifact should read as a **research-shaped showcase**, not a random RL notebook and not a pure paper-reproduction repo.
- Made the demo obligation much stricter:
  - show prefix progression over time
  - show judge score movement over time
  - show baseline disagreement
  - show the raw evidence decomposition that explains the final failure score
- Tightened the **verifiable rewards** showcase story:
  - the value is not just an extra scalar
  - the value is a reward/judge signal that can be replayed, decomposed, and challenged record-by-record
- Tightened the JEPA honesty rule again:
  - current v1 is a benchmarked proxy judge
  - stronger latent faithfulness belongs in the next model pass
  - the showcase should explicitly present that roadmap rather than hiding it
- Made the next milestone sequence concrete:
  1. fix `push-v3` weakness
  2. split held-out calibration from in-slice tuning
  3. widen recoverability/failure labeling
  4. add score-over-time replay visuals
  5. only then push harder on JEPA-native model faithfulness

## Required end-state for the repo front door
A strong README/demo flow should let a reviewer understand this in under two minutes:
1. what the judge is
2. what the benchmark slice is
3. why sparse reward is insufficient
4. where the current judge already beats trivial baselines
5. what remains heuristic / non-faithful today
6. what the next JEPA-native upgrade would replace

## Anti-bullshit rule
The repo must not imply any of the following unless directly supported by artifacts:
- that the current judge is a faithful LeWorldModel implementation
- that the current score is a valid RL reward for online training
- that synthetic separation automatically proves real embodied value
- that calibration chosen on the same slice is deployment-ready

## Showcase win condition
The project lands if a skeptical reader says:

> I can see the benchmark, I can see the disagreement with sparse reward, I can inspect why the judge made its call, and I can tell exactly what part is JEPA-inspired versus still heuristic.

## Main conclusion
The project only becomes a real banger if the benchmark, replay surface, and honesty about the model path all reinforce the same narrow story:

> **early, inspectable, world-model-derived process reward signals for embodied RL prefixes**.
