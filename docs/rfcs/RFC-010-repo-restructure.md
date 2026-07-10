# RFC-010 — Repository restructure: single CLI, curated docs, curated artifacts

- **Status:** Accepted
- **Date:** 2026-07-10

## Decision

Restructure the repository in one pass (version 0.2.0): one installable CLI replaces the
standalone pipeline step files, five curated docs replace the accumulated spec pile, and the
artifact tree is curated down to the runs the claims actually rest on. Behavior is preserved
throughout.

### 1. Single CLI

Replace the eight standalone pipeline step files with one console entry point, `lewm-judge`
(equivalently `python -m leworldmodel_judge`), with a 1:1 subcommand mapping:

| Old step file | New subcommand |
|---|---|
| `collect_rollouts.py` | `lewm-judge collect` |
| `build_prefixes.py` | `lewm-judge prefixes` |
| `build_latent_cache.py` | `lewm-judge latents` |
| `run_baselines.py` | `lewm-judge baselines` |
| `run_judge.py` | `lewm-judge judge` |
| `evaluate.py` | `lewm-judge evaluate` |
| `render_family_report.py` | `lewm-judge report` |
| `render_demo.py` | `lewm-judge demo` |

Flags carry over unchanged; the full surface is documented in
[../contracts.md](../contracts.md#cli-surface). The old CLI command spec anticipated this move
("V1 may start as scripts rather than a polished CLI, but the command surface should already be
stable") and is superseded by the implemented CLI. No compatibility shims are kept: the step
files' logic moved into the `leworldmodel_judge` package, and the files were removed.

### 2. Doc consolidation

The 22-file spec directory that lived beside these RFCs mixed durable contracts with dated run
reports and process diaries, and had drifted from the code (stale judge-mode lists, "planned"
stubs for shipped features, a nonexistent `adversarial` family). It is replaced by five curated
docs. Superseded-by map (original filenames are preserved in git history, pre-0.2.0 tree):

| Superseded source | Carried into |
|---|---|
| product spec, research thesis notes | [../vision.md](../vision.md) |
| system spec, judge-signal contract | [../method.md](../method.md) |
| eval contract, hard-family and multitask run reports, rolling status log | [../benchmark.md](../benchmark.md) |
| dataset / result-schema / demo contracts, CLI command spec (corrected) | [../contracts.md](../contracts.md) |
| JEPA-faithfulness upgrade plan | [../roadmap.md](../roadmap.md) |
| iteration pass notes 1–5 | split: canonical thesis wording → vision.md; push-v3 label rules → method.md; benchmark non-negotiables and maturity ladder → benchmark.md; remaining-work lists → roadmap.md |
| milestone plan, implementation plan, first-experiments plan | dropped (process logs; git history preserves them) |
| the placeholder architecture note | [../method.md](../method.md) |

No claim was dropped silently: every headline number from the superseded run reports appears in
[../benchmark.md](../benchmark.md) with provenance, including numbers whose artifact directories
were removed (see 4).

### 3. Claims carry-over

RFC-001 and RFC-009 remain binding constraints on the rewritten README and docs: the restructure
must not inflate claims. The RFC-002 / RFC-007 caveat is restated wherever results appear: the
shipped judge is a heuristic composite plus a simple observation-space latent proxy, not a
LeWorldModel/JEPA reproduction, and `judge_mode` provenance accompanies every number. Known open
gaps are recorded as addenda instead of being hidden: RFC-004 (disagreement-based uncertainty
never implemented) and RFC-006 (no rendered frames in the demo).

### 4. Artifact and schema stability guarantee

The restructure changes packaging and documentation, not behavior. Scoring math, labeling rules,
metric values, and JSONL/JSON/CSV schemas are unchanged. The checked-in artifacts remain valid
outputs of the new code, and the deterministic pipeline stages regenerate them byte-identically
from the checked-in rollout captures.

Artifact curation kept the two runs every current claim rests on:

- `artifacts/hard-family-real-held-out-2026-04-28/` — the held-out threshold story
  (`judge_mode: hybrid_prefix_latent_judge`)
- `artifacts/hard-family-synthetic-benchmark-2026-04-23-v2/` — the canonical synthetic benchmark
  (`judge_mode: composite_prefix_judge`)

Five superseded or orphaned run directories were removed: their rollouts were byte-identical
duplicates of kept captures, their summaries carried no discriminative content, or nothing
referenced them. Their headline numbers survive with provenance notes in
[../benchmark.md](../benchmark.md); the raw data remains in git history.

## Why

- Eight entry points with no installable command made the pipeline awkward to run and document.
- The spec pile contradicted the code in several places and buried the durable contracts under
  process narration; a reviewer could not tell which document was authoritative.
- Three byte-identical copies of a ~1 MB rollout capture (and two of the synthetic capture) were
  checked in across superseded run directories.
- The claim surface was scattered across a status log, run reports, and pass notes, which made
  claim discipline (RFC-009) harder to enforce than it should be.

## Consequences

- One command surface, documented in one place, exercised by CI.
- Five docs own the durable content; the RFC directory stays the decision log. New decisions
  append as RFC-011+; numbers are never reused or renumbered.
- The artifact tree contains only runs that current claims cite, each covered by the manifest in
  `artifacts/README.md`.
- Changes to record schemas or labeling rules are benchmark-contract changes: they require
  regenerated artifacts and an updated [../benchmark.md](../benchmark.md).
