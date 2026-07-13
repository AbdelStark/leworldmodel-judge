# RFC-011 — Cloud benchmark runs: Hugging Face Jobs pipeline with published run artifacts

- **Status:** Accepted
- **Date:** 2026-07-13

## Decision

Add a cloud execution path for the locked benchmark: the existing `lewm-judge` pipeline runs
unchanged inside [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs),
every run publishes a complete, self-describing artifact folder to a public Hugging Face dataset
repository, and a local launcher owns preflight, monitoring, download, and contract verification.
Hugging Face's `ml-intern` agent CLI participates as a supervised operator — launching the smoke
run and writing an independent review of the results — with its transcripts shipped as artifacts.

The library itself does not change. `src/leworldmodel_judge` stays stdlib-only, offline, and
path-agnostic (RFC-010); everything cloud-specific lives in a new top-level `jobs/` directory as
self-contained PEP 723 scripts, outside the installable package and outside the sdist.

### 1. Job payload: `jobs/run_benchmark.py`

A single PEP 723 uv script executed remotely via `hf jobs uv run`. It runs the same eight CLI
stages as the README quickstart — collect, prefixes, latents, baselines, judge (both modes),
evaluate (both judges), report, demo — as subprocesses of the installed `lewm-judge` console
script, then writes `provenance.json` and uploads the run folder to the dataset repository.

The pinned package is not a dependency of the script itself: the launcher passes
`--with "leworldmodel-judge[viz(,metaworld)] @ git+https://github.com/AbdelStark/leworldmodel-judge@<sha>"`
at launch time, where `<sha>` is the pushed HEAD commit. This keeps the pin exact per run without
templating the script. The `viz` extra is always installed so plots ship as PNG, matching the
checked-in artifact convention; `metaworld` is added only for the Meta-World preset.

Presets (selected by `LEWM_PRESET`; all run the full stage chain, both judge modes, and the
held-out family calibration `--calibration-families weak,doomed --evaluation-families
expert,misleading` from the 2026-04-28 artifact):

| Preset | Source | Episodes per (task, family) | Seed | Flavor | Timeout |
|---|---|---|---|---|---|
| `smoke` | synthetic | 2 | 7 | `cpu-basic` | 15m |
| `synthetic-benchmark` | synthetic | 50 | 7 | `cpu-basic` | 20m |
| `metaworld-benchmark` | metaworld (`--max-steps 75`) | 5 | 1013 | `cpu-upgrade` | 45m |

Rationale for the two benchmark presets:

- `synthetic-benchmark` scales the synthetic slice from 5 to 50 episodes per (task, family) —
  1,800 prefixes instead of 180 — under the same locked generator and seed 7.
- `metaworld-benchmark` is a **fresh capture**: a new base seed (1013) with the label rules
  frozen at 0.2.0. This directly serves two standing roadmap items — "larger real held-out
  slices" (5 episodes per (task, family) puts ~90 prefixes in the held-out evaluation cohort
  versus 18 in the 2026-04-28 artifact) and "a fresh capture collected with the rules frozen"
  (the 2026-04-28 labels were revised against the same capture they evaluate; this one is
  collected after the rules were locked).

The job needs no display and no GPU: Meta-World collection runs `render_mode=None` (CPU MuJoCo
physics only), and everything downstream is pure Python. At current pricing the full three-run
matrix costs under one cent; the binding budget is wall-clock, capped by the per-preset timeout.

### 2. Launcher: `jobs/launch.py`

A local PEP 723 script (dependency: `huggingface_hub`) with subcommands:

- `launch` — preflight, then start a job and stream it to a terminal verdict.
  Preflight refuses to launch when the working tree is dirty or HEAD is not on the pushed
  remote: the `--with` pin must name a commit anyone can install.
- `verify` — download a run folder from the dataset repo and gate it against the run contract
  (see 3). Exit code is the verdict; used after every run and usable standalone.
- `card` — regenerate the dataset repository's README card from all published runs'
  `provenance.json` + `summary*.json` (run index with headline metrics and provenance).

The launcher passes run identity to the job exclusively through environment variables
(`LEWM_RUN_ID`, `LEWM_PRESET`, `LEWM_GIT_SHA`, `LEWM_FLAVOR`) and `HF_TOKEN` as a secret. Run id
format: `<preset>-<UTC yyyymmdd-hhmmss>-g<short sha>`.

### 3. Run artifact contract (extends RFC-008)

Each run publishes `runs/<run_id>/` to the dataset repository
**`abdelstark/leworldmodel-judge-runs`** containing exactly the RFC-008 file surface plus cloud
provenance:

```
runs/<run_id>/
  rollouts.jsonl            # input of record for the run
  prefixes.jsonl
  latent-cache.jsonl
  baselines.jsonl
  judge-composite.jsonl     # heuristic_surprise
  judge-hybrid.jsonl        # hybrid_surprise (replay-time; RFC-007 caveat applies)
  summary-composite.json    # evaluate on judge-composite, held-out split
  summary.json              # evaluate on judge-hybrid, held-out split
  report/family-report.md   # rendered from summary.json
  report/family-report.png
  demo-artifact.md          # + stem-derived CSV/PNG siblings (RFC-008)
  provenance.json           # cloud run provenance (below)
```

`provenance.json` fields: `provenance_schema_version`, `run_id`, `preset`, `created_utc`,
`git` (repo URL, pinned sha), `package` (name, installed version), `python`, `platform`,
`hardware_flavor`, `job_env` (whitelisted `JOB_*`/`SPACE_*` keys, secret-marker names dropped),
`stages` (per stage: exact argv and wall-clock seconds), `files` (per artifact: byte size,
sha256, row count for JSONL), and `upload` (repo id, path prefix).

The `verify` gate checks: required file set for the preset; `summary*.json` parse with
`thresholds`, `calibration.provenance` (families disjoint, mode `held_out_family_split`),
non-degenerate evaluation cohort; row-count joins (prefixes = baselines = each judge file);
`judge_mode` values match the file's judge; and every published file's sha256 matches
`provenance.json`. A run that fails verification is deleted from the dataset repo or superseded
by a new run id — never patched in place.

Curation back into git follows RFC-010 unchanged: `artifacts/` only gains a run directory when a
claim in the docs cites it, copied verbatim from the published run with a reproduction note
(commands, seed, rationale) in the commit.

### 4. Agent operation: `ml-intern`

The pipeline demonstrates supervised agent operation with Hugging Face's `ml-intern` CLI
(github.com/huggingface/ml-intern), pinned to an HF-router model (`moonshotai/Kimi-K2-Instruct`)
so it runs on `HF_TOKEN` alone:

- **Operator role.** `ml-intern` launches the `smoke` run by driving the repo launcher (so
  preflight and run identity stay deterministic), then independently confirms the job's state
  through read-only `hf_jobs` operations (ps, inspect, logs) and writes an operations report.
- **Reviewer role.** After the benchmark runs publish, `ml-intern` receives the published
  summaries plus the checked-in 2026-04-28 artifact and writes an independent run review
  (`intern-review.md`): metric deltas, provenance audit, anomalies.

Both transcripts (stdout and the session log) ship with the run artifacts under
`runs/<run_id>/agent/`. Guardrails, stated because headless `ml-intern` auto-approves
everything: prompts are single-purpose and enumerate permitted operations; only CPU flavors; the
review is advisory — the merge gate remains `jobs/launch.py verify`, which is deterministic. The
agent's exit code is not a signal (it exits 0 on errors); the wrapper checks for the expected
output files instead, and still collects and ships whatever transcript exists when a turn fails
or times out — a paid run without its transcript is the worse failure mode.

Two hazards of the agent itself are handled explicitly. First, `ml-intern` records its full
trajectory to `session_logs/` and, when `HF_SESSION_UPLOAD_TOKEN` is set, auto-uploads it to a
third-party dataset repo (`akseljoonas/hf-agent-sessions`); the wrapper strips that variable —
and every other credential except `HF_TOKEN` — from the agent's environment, so transcripts
leave the machine only through this pipeline. Second, transcripts are published verbatim to a
public repo, so the wrapper redacts token-shaped strings (`hf_…`, `gh*_…`, `sk-…`, `AKIA…`)
from every collected file before upload.

### 5. CI dispatch (optional path)

`.github/workflows/hf-jobs-benchmark.yml` adds a `workflow_dispatch`-only workflow that runs
`jobs/launch.py launch --preset <input>` on a GitHub runner. It requires an `HF_TOKEN` repository
secret and is not part of required CI; the required CI workflow is untouched. This exists so a
maintainer can fire a benchmark run from the GitHub UI without a local checkout.

## Why

- The benchmark's determinism story (RFC-008) has only ever been exercised on one laptop. A
  containerized cloud run with pinned installs is the cheapest honest test of "anyone can rerun
  this" — and the roadmap already calls for a bigger fresh capture, which is exactly the kind of
  batch work Jobs exist for.
- Publishing complete run folders (inputs of record included) to a dataset repo makes results
  inspectable without cloning the repo or trusting the README — the RFC-008 principle, extended
  to where the runs happen.
- Claim discipline (RFC-009) binds the dataset card and the intern review exactly as it binds the
  README: `judge_mode` provenance on every number, no held-out language for in-slice thresholds,
  and the RFC-007 replay-time caveat wherever hybrid numbers appear.
- An agent CLI that can launch paid compute and push to the Hub needs its blast radius written
  down before it is wired in, not after. Scoped prompts, CPU-only flavors, deterministic
  verification, and shipped transcripts are that write-down.

## Consequences

- `jobs/` joins the repo as the cloud-ops surface: two PEP 723 scripts and a README, excluded
  from the package and sdist; the library keeps zero runtime dependencies.
- `abdelstark/leworldmodel-judge-runs` becomes the public home of cloud runs; git `artifacts/`
  stays curated to claim-bearing runs only.
- The benchmark gains its first post-freeze Meta-World capture and a 10× synthetic slice; the
  18-prefix held-out story stops being the only held-out story. New numbers land in
  [../benchmark.md](../benchmark.md) with the same caveats as the existing table.
- Schema changes remain benchmark-contract changes (RFC-010): `provenance.json` is versioned
  independently and additively.
