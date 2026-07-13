# Build the companion preprint.
#
#   make            # figures (from checked-in artifacts) + PDF
#   make pdf        # PDF only
#   make figures    # regenerate figures/*.pdf from artifacts/
#   make clean      # remove LaTeX build residue (keeps the PDF)
#
# Engines: tectonic if installed (fetches packages on demand), else pdflatex + bibtex.

PDF      := leworldmodel-judge.pdf
TECTONIC := $(shell command -v tectonic 2>/dev/null)

.PHONY: all pdf figures clean distclean

all: figures pdf

pdf:
ifneq ($(TECTONIC),)
	tectonic main.tex
else
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
endif
	mv main.pdf $(PDF)
	@echo "wrote docs/paper/$(PDF)"

figures:
	uv run figures/make_figures.py

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.toc

distclean: clean
	rm -f $(PDF)
