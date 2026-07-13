You are writing an independent review of published cloud benchmark runs of the
leworldmodel-judge project (RFC-011). Everything you need is already on local disk; do not
fetch anything from the network.

Materials:

- Primary run (fresh Meta-World capture): $primary_run_dir
- Comparison run (synthetic at 50 episodes): $compare_run_dir
- Checked-in reference artifact (2026-04-28 held-out, n=18): $reference_artifact_dir
- Each run folder contains summary-composite.json (composite cutoff-time judge,
  judge_mode composite_prefix_judge), summary.json (hybrid latent judge, judge_mode
  hybrid_prefix_latent_judge — a REPLAY-TIME signal that reads post-cutoff observations),
  provenance.json (git sha, package version, per-stage commands/timings, file checksums),
  and report/family-report.md.

Write a markdown review to $review_path with exactly these sections:

1. `# Intern review` — one-paragraph verdict: are the published numbers consistent,
   provenance-complete, and stated within the project's claim discipline?
2. `## Headline metrics` — a table of the held-out evaluation-cohort metrics
   (hit rate, false positive rate, pairwise accuracy, average precision, evaluation count)
   for the composite judge of both runs and of the reference artifact
   (its summary-composite.json). Label every number with its judge_mode and note that
   hybrid numbers, where you mention them, are replay-time.
3. `## Scale-up vs the reference` — how the fresh capture's evaluation cohort size compares
   to the reference's 18 prefixes, and what did or did not change in the metrics.
4. `## Provenance audit` — for each run: git sha, package version, hardware flavor, total
   pipeline wall-time from the stage records, and whether every required file is listed with
   a checksum in provenance.json.
5. `## Anomalies` — anything inconsistent, degenerate (empty cohorts, None metrics),
   or surprising. Say "none found" only if you actually checked.
6. `## Claim-discipline check` — verify the run folders' reports stay within: "trajectory
   judge", "world-model-derived verifier signal", "early failure detection / trajectory
   ranking"; no "universal reward model" or "general judge for RL" language; hybrid clearly
   marked replay-time. Quote any violating sentence you find.

Hard constraints:

- Read-only: the ONLY file you write is $review_path. Do not edit the repository.
- Do not use hf_jobs, do not upload/delete anything on the Hub or GitHub, no sandboxes.
- Base every number on the local files; if a value is missing or null, print it as n/a —
  never substitute or estimate.
