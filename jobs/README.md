# Cloud benchmark runs (RFC-011)

Self-contained PEP 723 scripts that run the locked benchmark on
[Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) and publish
every run as a verified artifact folder under
[`abdelstark/leworldmodel-judge-runs`](https://huggingface.co/datasets/abdelstark/leworldmodel-judge-runs).
Spec: [RFC-011](../docs/rfcs/RFC-011-hf-jobs-pipeline.md). Nothing here is part of the
installable package; the library stays stdlib-only.

## Files

| File | Role |
|---|---|
| [`run_benchmark.py`](run_benchmark.py) | Remote payload: eight pipeline stages, both judge modes, held-out family split, `provenance.json`, upload |
| [`launch.py`](launch.py) | Local launcher: preflight → launch → stream → contract verification; also `verify` and `card` |
| [`intern_ops.py`](intern_ops.py) | Scoped [ml-intern](https://github.com/huggingface/ml-intern) steps: smoke-run operator, independent run reviewer |
| [`prompts/`](prompts/) | The versioned prompts the agent runs under — prompts are code |

## Run it

Requires a Hugging Face token with write access (`hf auth login`) and a clean, pushed HEAD —
the launcher pins the job's package install to your exact commit and refuses dirty trees.

```bash
uv run jobs/launch.py launch --preset smoke                 # ~2 min end-to-end, cpu-basic
uv run jobs/launch.py launch --preset synthetic-benchmark   # 50 eps/(task,family), seed 7
uv run jobs/launch.py launch --preset metaworld-benchmark   # fresh capture, seed 1013, cpu-upgrade

uv run jobs/launch.py verify --run-id <run_id>              # re-gate any published run
uv run jobs/launch.py card                                  # regenerate the dataset card
```

Every preset runs the full stage chain with the held-out family calibration
(`weak,doomed → expert,misleading`) and publishes to `runs/<run_id>/`. A run that fails the
verify gate is deleted or superseded, never patched in place. Full pricing is CPU-tier:
the whole three-preset matrix costs under one cent at current rates.

## Agent operation

```bash
uv run jobs/intern_ops.py operate-smoke
uv run jobs/intern_ops.py review --run-id <metaworld-run> --compare-run-id <synthetic-run>
```

Headless `ml-intern` auto-approves everything it does, so these steps only ever hand it
single-purpose prompts with enumerated permissions (read-only `hf_jobs`, one output file,
CPU only) and judge success by the expected output file, not the exit code (it exits 0 even
on errors). The wrapper additionally strips every credential except `HF_TOKEN` from the
agent's environment (including `HF_SESSION_UPLOAD_TOKEN`, which would otherwise let
`ml-intern` auto-upload its trajectory to a third-party dataset) and redacts token-shaped
strings from transcripts before they ship to `runs/<run_id>/agent/`. The agent's review is
advisory — the merge gate is `launch.py verify`, which is deterministic.

## From GitHub Actions

The `hf-jobs-benchmark` workflow (manual `workflow_dispatch` only) launches any preset from
the GitHub UI. It needs an `HF_TOKEN` repository secret with write access; required CI does
not depend on it.
