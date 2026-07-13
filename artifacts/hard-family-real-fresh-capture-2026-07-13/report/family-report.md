# Family-aware benchmark report

## Threshold provenance
- chosen judge threshold: `0.313437`
- fixed progress failure threshold: `0.2`
- judge threshold selection mode: `held_out_family_split`
- calibration families: `doomed, weak`
- evaluation families: `expert, misleading`
- calibration/evaluation family overlap: `False`
- calibration prefixes: `90` with `32` failure labels and `58` non-failure labels
- evaluation prefixes: `90` with `39` failure labels and `51` non-failure labels
- judge calibration balanced accuracy: `0.913793`
- judge calibration hit rate: `1.0`
- judge calibration false positive rate: `0.172414`
- judge calibration average precision: `0.879112`
- judge evaluation balanced accuracy: `0.892911`
- judge evaluation hit rate: `0.923077`
- judge evaluation false positive rate: `0.137255`
- judge evaluation average precision: `0.970098`
- progress baseline mode: `fixed_progress_baseline`

## Honesty note
- if the threshold was chosen on the same benchmark slice, present it as in-slice tuning, not held-out calibration.
- if the threshold comes from a held-out family split, say exactly which families calibrated it and which families were scored with it.
- if a family looks good only because coverage is tiny, the artifact should say that out loud.

## Per-family table

| family | count | failure labels | judge hit rate | judge false positive rate | judge pairwise accuracy | judge average precision | failure coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| expert | 45 | 20 | 1.000 | 0.120 | 1.000 | 1.000 | 0.444 |
| misleading | 45 | 19 | 0.842 | 0.154 | 0.947 | 0.926 | 0.422 |
