HAS_LATEXMK := $(shell command -v latexmk 2> /dev/null)

TEX = max_print_line=1048576 pdflatex -interaction=nonstopmode
MAIN = main
BIB = biber

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: *.tex
ifdef HAS_LATEXMK
	latexmk $(MAIN)
else
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)
endif

clean:
	rm -f *~ *.fls *.fdb_latexmk *.pdf *.aux *.log *.glg *.ist *.out *.tdo *.toc *.bbl *.bcf *.blg *.run.xml *.backup *.acn *.acr *.alg

