root = Thesis
dist = xelatex
latexmkopts = -$(dist) -shell-escape

all: $(root).pdf
.PHONY: $(root).pdf

$(root).pdf: $(root).tex
	latexmk $(latexmkopts) $(root)

clean:
	rm -rf `biber --cache`
	rm -rf _minted-$(root)
	latexmk -bibtex -CA
	find . -name \*.aux -type f -delete
	rm $(root).run.xml || true
	rm $(root).nlg || true
	rm $(root).nlo || true
	rm $(root).nls || true
	rm indent.log || true

watch: clean
	latexmk $(latexmkopts) -pvc $(root)

quick: clean
	$(dist) -shell-escape $(root)

view:
	open $(root).pdf
