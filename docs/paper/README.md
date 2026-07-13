# Companion paper

LaTeX sources for the preprint "A World Model as a Judge: Early, Auditable Verdicts on
Partial Robot-Manipulation Rollouts". The built PDF is checked in as
[`leworldmodel-judge.pdf`](leworldmodel-judge.pdf).

Every number in the paper is read from a checked-in file under [`../../artifacts/`](../../artifacts/),
and every figure regenerates deterministically from those artifacts.

## Build

```bash
make -C docs/paper            # figures + PDF (tectonic if installed, else pdflatex+bibtex)
make -C docs/paper figures    # regenerate figures/*.pdf from artifacts/ (needs uv)
make -C docs/paper pdf        # PDF only
```

## Layout

| Path | Contents |
|---|---|
| `main.tex` | The paper, single file |
| `references.bib` | Bibliography; every entry fetched from arXiv/CrossRef metadata, never written from memory |
| `figures/make_figures.py` | Generates every figure from the checked-in artifacts (PEP 723 script) |
| `figures/*.pdf` | Generated vector figures, checked in so the PDF builds without a Python environment |
| `Makefile` | Build entry points |

## Provenance discipline

The paper follows the repository's claim rules ([RFC-009](../rfcs/RFC-009-public-positioning-and-claim-discipline.md)):
every metric carries its `judge_mode` and threshold provenance, hybrid-judge numbers are
marked replay-time, and the limitations section states what the benchmark cannot support.
