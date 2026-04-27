# Family-aware benchmark report

## Threshold provenance
- chosen judge threshold: `0.360053`
- fixed progress failure threshold: `0.2`
- judge threshold selection mode: `in_slice_balanced_accuracy`
- judge calibration balanced accuracy (same slice): `1.0`
- judge calibration hit rate (same slice): `1.0`
- judge calibration false positive rate (same slice): `0.0`
- progress baseline mode: `fixed_progress_baseline`

## Honesty note
- if the threshold was chosen on the same benchmark slice, present it as in-slice tuning, not held-out calibration.
- if a family looks good only because coverage is tiny, the artifact should say that out loud.

## Per-family table

| family | count | failure labels | judge hit rate | judge false positive rate | judge pairwise accuracy | failure coverage |
|---|---:|---:|---:|---:|---:|---:|
| doomed | 6 | 2 | 1.000 | 0.000 | 1.000 | 0.333 |
| weak | 6 | 0 | 0.000 | 0.000 | n/a | 0.000 |
