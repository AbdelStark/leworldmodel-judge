# Family-aware benchmark report

## Threshold provenance
- chosen judge threshold: `0.360053`
- fixed progress failure threshold: `0.2`
- judge threshold selection mode: `in_slice_balanced_accuracy`
- calibration families: `all`
- evaluation families: `all`
- calibration/evaluation family overlap: `True`
- calibration prefixes: `72` with `4` failure labels and `68` non-failure labels
- evaluation prefixes: `72` with `4` failure labels and `68` non-failure labels
- judge calibration balanced accuracy: `0.985294`
- judge calibration hit rate: `1.0`
- judge calibration false positive rate: `0.029412`
- judge calibration average precision: `0.666667`
- judge evaluation balanced accuracy: `0.985294`
- judge evaluation hit rate: `1.0`
- judge evaluation false positive rate: `0.029412`
- judge evaluation average precision: `0.666667`
- progress baseline mode: `fixed_progress_baseline`

## Honesty note
- if the threshold was chosen on the same benchmark slice, present it as in-slice tuning, not held-out calibration.
- if the threshold comes from a held-out family split, say exactly which families calibrated it and which families were scored with it.
- if a family looks good only because coverage is tiny, the artifact should say that out loud.

## Per-family table

| family | count | failure labels | judge hit rate | judge false positive rate | judge pairwise accuracy | judge average precision | failure coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| doomed | 18 | 4 | 1.000 | 0.143 | 0.929 | 0.667 | 0.222 |
| expert | 18 | 0 | n/a | 0.000 | n/a | n/a | 0.000 |
| misleading | 18 | 0 | n/a | 0.000 | n/a | n/a | 0.000 |
| weak | 18 | 0 | n/a | 0.000 | n/a | n/a | 0.000 |
