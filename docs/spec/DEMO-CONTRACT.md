# DEMO-CONTRACT

## The demo must show
- partial rollout / prefix state, even if rendered as a compact markdown replay rather than video
- judge score over time across prefix cutoffs
- final success vs failure label
- baseline vs judge timing difference
- one disagreement example where sparse reward is late or blind
- one evidence decomposition example showing what the judge actually looked at

## Demo success condition
A viewer should understand the claim in under 30 seconds and should be able to answer:
- what the judge is scoring
- when the judge moves before sparse reward
- why a highlighted disagreement is interesting
- which evidence fields drove the current score

## Demo failure condition
The viewer sees a score but cannot tell:
- what it means
- when it changes
- why it matters compared to sparse reward
- whether the disagreement rows are real signal or narrative cherry-picking

## Honesty condition
The demo must explicitly say that the current artifact is a verifier-style judging surface, not yet a faithful JEPA world model implementation.
