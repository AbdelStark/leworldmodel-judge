# Family-aware benchmark report

## Threshold provenance
- chosen judge threshold: `0.311141`
- fixed progress failure threshold: `0.2`
- judge threshold selection mode: `held_out_family_split`
- calibration families: `doomed, weak`
- evaluation families: `expert, misleading`
- calibration/evaluation family overlap: `False`
- calibration prefixes: `18` with `9` failure labels and `9` non-failure labels
- evaluation prefixes: `18` with `8` failure labels and `10` non-failure labels
- judge calibration balanced accuracy: `1.0`
- judge calibration hit rate: `1.0`
- judge calibration false positive rate: `0.0`
- judge calibration average precision: `1.0`
- judge evaluation balanced accuracy: `0.95`
- judge evaluation hit rate: `1.0`
- judge evaluation false positive rate: `0.1`
- judge evaluation average precision: `1.0`
- progress baseline mode: `fixed_progress_baseline`

## Honesty note
- if the threshold was chosen on the same benchmark slice, present it as in-slice tuning, not held-out calibration.
- if the threshold comes from a held-out family split, say exactly which families calibrated it and which families were scored with it.
- if a family looks good only because coverage is tiny, the artifact should say that out loud.

## Per-family table

| family | count | failure labels | judge hit rate | judge false positive rate | judge pairwise accuracy | judge average precision | failure coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| expert | 9 | 4 | 1.000 | 0.200 | 1.000 | 1.000 | 0.444 |
| misleading | 9 | 4 | 1.000 | 0.000 | 1.000 | 1.000 | 0.444 |
