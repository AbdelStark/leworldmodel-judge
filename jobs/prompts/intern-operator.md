You are operating the RFC-011 cloud benchmark pipeline for the leworldmodel-judge
repository, from its checkout at $repo_root.

Do exactly this, nothing else:

1. With your bash tool (set the tool's timeout parameter to 1800 seconds — the command takes
   several minutes), run:

   cd $repo_root && uv run jobs/launch.py launch --preset smoke

   Wait for it to finish and read its full output.
2. From the output, note the run id (`[launch] run_id=...`), the job id (`[launch] job_id=...`),
   and whether the verify gate printed `[verify] PASSED`.
3. Independently confirm the job's terminal state with your hf_jobs tool, using read-only
   operations only (ps, inspect, logs).
4. Write a short operations report (markdown) to $report_path with exactly these sections:
   `# Smoke run operations report`, `## Identity` (run id, job id, hardware flavor),
   `## Outcome` (terminal stage from hf_jobs inspect, launcher verify verdict),
   `## Log tail` (the last 5 lines of the job logs).

Hard constraints — violating any of these makes the run unusable:

- Do NOT launch, schedule, cancel, or re-run any job with hf_jobs; run/uv operations are
  forbidden. The launcher command in step 1 owns launching. hf_jobs is read-only here.
- Do NOT upload, delete, or modify anything on the Hugging Face Hub or on GitHub.
- Do NOT edit any file in the repository. The only file you write is $report_path.
- CPU only; never request GPU flavors; no scheduled jobs; no sandboxes.
