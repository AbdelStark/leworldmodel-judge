# RFC index

Decision log for the project. Despite the "RFC" name these function as architecture decision
records: each file is a single recorded decision with rationale and consequences. Numbers are
stable and never reused; new decisions append as RFC-011+. Amendments are recorded as addenda,
not rewrites.

| RFC | Decision | Status |
|---|---|---|
| [RFC-001](RFC-001-problem-shape.md) — Problem shape | Trajectory judging in embodied RL, built around rollout prefixes and task outcomes; not universal reward replacement | Accepted (2026-04-23) |
| [RFC-002](RFC-002-why-leworldmodel.md) — Why LeWorldModel | LeWorldModel is the conceptual anchor; the judge interface stays implementation-agnostic | Accepted (2026-04-23) |
| [RFC-003](RFC-003-environment-choice.md) — Environment choice | Meta-World first; robosuite only if its visual value clearly outweighs setup cost | Accepted (2026-04-23) |
| [RFC-004](RFC-004-judge-signal-design.md) — Judge signal design | First judge signal must be simple, benchmarkable, decomposed into named sub-signals | Accepted, amended 2026-07-10: v1 shipped a heuristic composite; disagreement-based uncertainty is an open gap |
| [RFC-005](RFC-005-benchmark-and-baselines.md) — Benchmark and baselines | The benchmark is part of the product; no result without baseline comparison | Accepted (2026-04-23) |
| [RFC-006](RFC-006-demo-surface.md) — Demo surface | The demo must show rollout evidence and score evolution together | Accepted, amended 2026-07-10: rendered frames are an open gap |
| [RFC-007](RFC-007-v1-vs-v2-boundary.md) — V1 vs V2 boundary | V1 may ship a lighter judge; the repo must be explicit about what is not faithful | Accepted (2026-04-23) |
| [RFC-008](RFC-008-result-and-artifact-contracts.md) — Result and artifact contracts | All benchmark and demo outputs are file-based and reproducible | Accepted (2026-04-23) |
| [RFC-009](RFC-009-public-positioning-and-claim-discipline.md) — Public positioning and claim discipline | Public language stays narrower than max ambition; the benchmark claim outranks the hype claim | Accepted (2026-04-23) |
| [RFC-010](RFC-010-repo-restructure.md) — Repository restructure | Single `lewm-judge` CLI, five curated docs, curated artifacts; behavior and schemas preserved | Accepted (2026-07-10) |
| [RFC-011](RFC-011-hf-jobs-pipeline.md) — Cloud benchmark runs | The pipeline runs unchanged on Hugging Face Jobs; every run publishes a verified, self-describing artifact folder; `ml-intern` operates under scoped prompts | Accepted (2026-07-13) |

Curated documentation built on these decisions: [vision](../vision.md), [method](../method.md),
[benchmark](../benchmark.md), [contracts](../contracts.md), [roadmap](../roadmap.md).
