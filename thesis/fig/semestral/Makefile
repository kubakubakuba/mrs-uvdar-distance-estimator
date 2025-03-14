SHELL := /bin/bash

# Find all .pgf files in the directory
PGF_FILES := $(wildcard *.pgf)

# Generate corresponding .pdf file names
PDF_FILES := $(PGF_FILES:.pgf=.pdf)

# Main target to build all PDFs
all: $(PDF_FILES)

# Rule to build .pdf from .pgf using a LaTeX template
%.pdf: %.pgf
	sed "s|<PLOT_FILE>|$<|g" template.tex > temp_$*.tex
	pdflatex -interaction=nonstopmode -halt-on-error temp_$*.tex
	rm -f temp_$*.tex
	mv temp_$*.pdf $@
	rm -f temp_$*.aux temp_$*.log

# Clean target to remove intermediate and generated files
clean:
	rm -f *.aux *.log *.pdf temp_*.tex

collect-garbage:
	rm -f *.aux *.log temp_*.tex

.PHONY: all clean
